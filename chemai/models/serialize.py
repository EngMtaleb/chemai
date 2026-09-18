"""Saving and loading a fitted model - with the versions it was fitted under.

A pickled scikit-learn estimator is only valid for the library version that
wrote it. Unpickling it under another version still "works": it warns on stderr
and may compute something subtly different. A warning in a log nobody reads is
not a safeguard, so the payload records the versions and the service reports the
mismatch through /health, where an operator can see it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import logging
import pickle

import sklearn

from chemai import __version__

log = logging.getLogger(__name__)

FORMAT = 2          # 1 = a bare pickled model, 2 = payload with metadata


def save_model(model, path: Path | str) -> dict:
    """Write the model together with the versions that produced it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": FORMAT, "model": model,
               "sklearn_version": sklearn.__version__,
               "chemai_version": __version__,
               "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)
    log.info("saved model to %s (scikit-learn %s)", path, payload["sklearn_version"])
    return {k: v for k, v in payload.items() if k != "model"}


def load_model(path: Path | str) -> tuple[object, dict]:
    """(model, metadata). Metadata carries `version_match`: False means the model
    was fitted under a different scikit-learn and its numbers are not guaranteed."""
    with open(Path(path), "rb") as fh:
        payload = pickle.load(fh)

    if not isinstance(payload, dict) or "model" not in payload:
        # A model saved before this format existed: usable, but its provenance
        # is unknown, which is itself worth reporting.
        meta = {"format": 1, "sklearn_version": None, "chemai_version": None,
                "created": None, "version_match": False,
                "note": "model saved without version metadata - re-run training to record it"}
        log.warning("model at %s has no version metadata", path)
        return payload, meta

    meta = {k: v for k, v in payload.items() if k != "model"}
    meta["version_match"] = meta.get("sklearn_version") == sklearn.__version__
    if not meta["version_match"]:
        log.warning("model was fitted with scikit-learn %s, this environment has %s - "
                    "results are not guaranteed to match",
                    meta.get("sklearn_version"), sklearn.__version__)
    return payload["model"], meta
