"""Deep learning rungs of the ladder, in PyTorch, small enough to train on a laptop CPU.

Two kinds per question:

  * plain deep learning - a network learns the answer straight from the data
    (an MLP on the features, a 1-D CNN on the raw daily sequence);
  * physics-informed deep learning - the physics is part of the network: it
    predicts only the quantity the physics leaves unknown (a fouling rate, the
    weights of four fouling mechanisms, the bend of a cost curve), and a fixed
    physical law turns that into the answer. The law cannot be unlearned.

Training: Adam, weight decay, early stopping on a validation split BY GROUP.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn

torch.set_num_threads(2)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


class Standardiser:
    """Median-impute with missing flags, then standardise - fitted on training rows only."""

    def fit(self, X: np.ndarray):
        X = np.asarray(X, float)
        self.med = np.nanmedian(X, axis=0)
        self.med = np.where(np.isfinite(self.med), self.med, 0.0)
        self.miss = np.isnan(X).any(axis=0)
        Z = self._fill(X)
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-6
        return self

    def _fill(self, X):
        X = np.asarray(X, float)
        filled = np.where(np.isnan(X), self.med, X)
        flags = np.isnan(X[:, self.miss]).astype(float)
        return np.hstack([filled, flags])

    def transform(self, X):
        return ((self._fill(X) - self.mu) / self.sd).astype(np.float32)


def sequence_channels(seq: np.ndarray) -> np.ndarray:
    """(n, 2, L) value/censored with NaN gaps -> (n, 3, L): value carried forward and
    centred on 1.0, observed mask, censored flag."""
    val, cen = seq[:, 0, :].copy(), np.nan_to_num(seq[:, 1, :], nan=0.0)
    obs = np.isfinite(val).astype(np.float32)
    for i in range(val.shape[0]):
        v = val[i]
        last = np.nan
        for j in range(v.shape[0]):
            if np.isfinite(v[j]):
                last = v[j]
            elif np.isfinite(last):
                v[j] = last
        first = np.flatnonzero(np.isfinite(v))
        v[~np.isfinite(v)] = v[first[0]] if len(first) else 1.0
    return np.stack([(val - 1.0) * 10.0, obs, cen], axis=1).astype(np.float32)


class MLP(nn.Module):
    def __init__(self, n_in: int, hidden: int = 64, n_out: int = 1, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, n_out))

    def forward(self, x, *_):
        return self.net(x).squeeze(-1)


class SeqEncoder(nn.Module):
    """1-D CNN over a daily (or per-minute) sequence, joined with a few scalars."""

    def __init__(self, n_ch: int, n_scalar: int, hidden: int = 32):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(n_ch, 16, 5, padding=2), nn.ReLU(),
                                  nn.Conv1d(16, 16, 5, padding=2), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(16 * 2 + n_scalar, hidden), nn.ReLU())

    def forward(self, s, x):
        h = self.conv(s)
        h = torch.cat([h.mean(-1), h[..., -7:].mean(-1), x], dim=1)
        return self.head(h)


class SeqCNN(nn.Module):
    """Plain DL: the sequence and the scalars straight to the logit of 'crosses within 28 days'."""

    def __init__(self, n_ch: int, n_scalar: int, hidden: int = 32):
        super().__init__()
        self.enc = SeqEncoder(n_ch, n_scalar, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, s, *_):
        return self.out(self.enc(s, x)).squeeze(-1)


class GapLawNet(nn.Module):
    """Physics-informed DL for the cleaning question.

    The network predicts only how fast the feed channel is closing: a log-normal
    distribution of r = -dz/dt, where z = DP_norm^(-1/n) is the quantity the gap
    law (dP ∝ gap^-n) makes straight in time. The law then gives the answer:

        days to the level = (z_now - z_level) / r
        P(cross within H) = P(r >= (z_now - z_level) / H) = Φ((μ - ln(Δz / H)) / σ)

    It outputs a probability, not a logit.
    """

    def __init__(self, n_ch: int, n_scalar: int, n_gap: float = 2.2, horizon: float = 28.0, hidden: int = 32,
                 upsets: bool = True):
        super().__init__()
        self.enc = SeqEncoder(n_ch, n_scalar, hidden)
        self.mu = nn.Linear(hidden, 1)
        self.log_sigma = nn.Linear(hidden, 1)
        self.jump = nn.Linear(hidden, 1) if upsets else None
        self.n, self.h = n_gap, horizon

    def forward(self, x, s, phys):
        level, now = phys[:, 0], phys[:, 1]
        dz = torch.clamp(now ** (-1.0 / self.n) - level ** (-1.0 / self.n), min=1e-5)
        h = self.enc(s, x)
        mu = self.mu(h).squeeze(-1) - 7.5                    # rates are ~5e-4 per day in z units
        sigma = torch.nn.functional.softplus(self.log_sigma(h).squeeze(-1)) + 0.05
        z = (mu - torch.log(dz / self.h)) / sigma
        p = 0.5 * (1 + torch.erf(z / math.sqrt(2.0)))           # crossing by steady fouling
        if self.jump is not None:                               # or by an upset the law cannot see
            q = torch.sigmoid(self.jump(h).squeeze(-1) - 3.0)
            p = q + (1 - q) * p
        return p                                                # a probability, not a logit


def train(model: nn.Module, inputs_tr: list, y_tr: np.ndarray, inputs_va: list, y_va: np.ndarray,
          loss: str = "bce_logit", epochs: int = 60, lr: float = 2e-3, wd: float = 1e-4,
          batch: int = 256, patience: int = 8, seed: int = 0, weight: float | None = None) -> nn.Module:
    """Adam with early stopping on the validation loss; returns the best weights."""
    set_seed(seed)
    tt = [torch.as_tensor(a) for a in inputs_tr]
    tv = [torch.as_tensor(a) for a in inputs_va]
    ytr, yva = torch.as_tensor(y_tr, dtype=torch.float32), torch.as_tensor(y_va, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    pos_w = torch.tensor(weight if weight else 1.0)

    def crit(pred, y):
        if loss == "bce_logit":
            return nn.functional.binary_cross_entropy_with_logits(pred, y, pos_weight=pos_w)
        if loss == "bce_prob":
            p = pred.clamp(1e-5, 1 - 1e-5)
            return -(pos_w * y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()
        if loss == "mae":
            return (pred - y).abs().mean()
        return ((pred - y) ** 2).mean()

    best, best_state, wait = np.inf, copy.deepcopy(model.state_dict()), 0
    n = len(ytr)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            out = model(*[a[b] for a in tt])
            crit(out, ytr[b]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(crit(model(*tv), yva))
        if v < best - 1e-5:
            best, best_state, wait = v, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def predict(model: nn.Module, inputs: list, prob: bool = False) -> np.ndarray:
    with torch.no_grad():
        out = model(*[torch.as_tensor(a) for a in inputs])
        if prob:
            out = torch.sigmoid(out)
    return out.numpy().astype(float)
