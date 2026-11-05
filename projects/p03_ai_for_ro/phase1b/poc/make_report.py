"""Phase 1B proof of concept - the report (REPORT_POC.md), assembled from the run's own files: out/report_tables.md
(make_poc_data.py), out/poc_data.js, the control's investigation (ctrl_check2/3), and the hashes. Prose that states a number
reads it from those files.

    python3 -B make_report.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
src = (OUT / "poc_data.js").read_text()
POC = json.loads(src[src.index("{"): src.rindex("}") + 1])
assert POC["final"], "poc_data.js is not the finished run"
R1, R2 = "record 1", "record 2"
S7 = POC["scores"]["rolling"]["dp_pct"]["7"]
LEARN = [m for m in POC["methods"] if m not in ("no change", "fouling law")]
P = lambda rec, m: POC["passes"].get(f"{rec}|{m}", 0)                       # noqa: E731
N = POC["cells"][f"{R2}|XGBoost"]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()           # noqa: E731


def skill_table(csv, variants):
    d = pd.read_csv(csv)
    rows = []
    for (v, s), g in d.groupby(["variant", "seeds"]):
        if v in variants:
            rows.append({"variant": v, "seeds": s, "skill": 100 * (1 - np.abs(g.y - g.XGBoost).mean() / np.abs(g.y - g.now).mean())})
    return pd.DataFrame(rows).pivot(index="seeds", columns="variant", values="skill").reindex(["0-4", "5-9", "10-14"])


c2 = skill_table(OUT / "ctrl_check2_flux_pct_14.csv", ("without", "shifted"))
c3 = skill_table(OUT / "ctrl_check3_flux_pct_14.csv", ("shifted, no yearly cycle",))
ctl = c2.join(c3)
ctl["shifted − without"] = ctl["shifted"] - ctl["without"]
ctl["shifted, no yearly cycle − without"] = ctl["shifted, no yearly cycle"] - ctl["without"]
gap = POC["control"]["skill"]["flux_pct|14"]["shifted - without"]
rep = abs(ctl.loc["0-4", "shifted − without"] - gap) < 0.01                    # step 2 reproduces the run's own number
# the yearly cycle's share of SDI's and turbidity's variance (log scale), as ctrl_check1.py measures it
pw = pd.read_csv(HERE / "data" / "plantwide.csv")
shares = []
for _, g in pw.groupby("plant"):
    g = g.sort_values("day"); x = np.arange(len(g))
    X = np.column_stack([np.ones(len(g)), np.sin(2 * np.pi * x / 365), np.cos(2 * np.pi * x / 365)])
    for col in ("sdi", "turbidity_ntu"):
        y = np.log(g[col].to_numpy()); beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        shares.append(1 - np.var(y - X @ beta) / np.var(y))
I = POC["control"]["investigation"]
if POC["control"]["ok"]:
    NOTE = {"summary": "The negative control passed.", "cycle_share": f"{100 * min(shares):.0f}–{100 * max(shares):.0f} %",
            "finding": "The control passed; no investigation was needed."}
else:
    g = [v["gap"] for v in I["seeds"].values()]; nc = [v["gap_no_cycle"] for v in I["seeds"].values()]
    NOTE = {
        "summary": f"The declared negative control missed its bar in one cell of four (flux 14 days ahead, {gap:+.2f} points; bar: "
                   "within 2). As declared, nothing was reported until the cause was found: there is no leak — SDI and turbidity "
                   "carry a yearly cycle, which a circular shift moves but keeps; with the cycle taken out before the same shift, "
                   f"the gap is {min(nc):+.2f} to {max(nc):+.2f} points over three sets of seeds (post hoc).",
        "cycle_share": f"{100 * min(shares):.0f}–{100 * max(shares):.0f} %",
        "finding": (f"**Finding.** The code shows no leak (step 1). The gap is not the seeds' doing: it stays at {min(g):+.2f} to "
                    f"{max(g):+.2f} points over three sets of five seeds (step 2). It is the yearly cycle: with the cycle taken out "
                    f"before the same shift, the gap is {min(nc):+.2f} to {max(nc):+.2f} — within the bar for every set (step 3). The "
                    "shifted columns still told the model the time of year, which helps flux two weeks ahead; the declared control "
                    "should have taken the cycle out before shifting. The declared verdict stands as written (missed, one cell); steps "
                    "2 and 3 were designed after the miss and are reported as such. For ΔP, the cells the deck shows, the shift took the "
                    "gain away as it should. Nothing in the results changes: the shifted columns enter no reported forecast.")}

lines = [
    "# Phase 1B, proof of concept — report",
    "",
    f"Run 3–4 November 2026 as declared in `declaration_poc.md` (approved 3 November 2026, 22:40 UTC+3; SHA-256 "
    f"`{(HERE / 'declaration_poc.sha256').read_text().split()[0]}`). **SIMULATION · PROOF OF CONCEPT:** every number below "
    "describes this simulator under the declared assumptions, not any real plant.",
    "",
    "## In one paragraph",
    "",
    f"Three simulated plants (one per fouling law; {POC['rows']:,} train-days, {len(POC['columns'])} logged columns with noise) ran "
    f"phase 1's steps twice: on record 1 (the four indicators, as the real record has them) and on record 2 (every logged column). "
    f"ΔP seven days ahead, rolling test, three plants pooled: *no change* {S7['no change'][R1]['mae']:.2f} points; XGBoost "
    f"{S7['XGBoost'][R1]['mae']:.2f} on record 1 and {S7['XGBoost'][R2]['mae']:.2f} on record 2. Over all {N} cells (ΔP and flux × "
    "1, 3, 7, 14 days × both tests) the methods passed phase 1's bar this many times, record 1 → record 2: "
    + "; ".join(f"{m} {P(R1, m)} → {P(R2, m)}" for m in LEARN) + ". "
    + ("XGBoost passed even on record 1 here — on the real plant no method did for ΔP: the simulator is easier than the plant. "
       if S7["XGBoost"][R1]["pass"] else "")
    + "Permutation importance: XGBoost leaned mostly on the results group; columns with no link moved its error about as much as "
    "the causes. " + NOTE["summary"],
    "",
    "## What ran",
    "",
    "- The table: `make_table.py` → `data/table.csv` (SHA-256 `" + sha(HERE / "data" / "table.csv") + "`) and `data/cleanings.csv` "
    "(`" + sha(HERE / "data" / "cleanings.csv") + "`), made once before any model ran; hidden quantities never enter a model "
    "(`poc_lib.load` checks the column list).",
    "- Trees: `poc_trees.py` — no change, the fouling law, XGBoost and the hybrid (phase 1's `p3lib`, unchanged), both records, "
    "both tests; the negative control; permutation importance.",
    "- Networks: `poc_nets.py` — addition 1's LSTM and RNN (`p3add`, unchanged; five seeds), both records, every fold. Only the "
    "input sizes follow the record (no count of 10-min samples; record 2 carries every logged column in the 17-day sequence).",
    "- Scores: `make_poc_data.py` → `out/poc_data.js` (the deck's numbers), `out/report_tables.md`, `out/scores_*.csv`.",
    "",
    "## Results",
    "",
    __import__("re").sub(r"(?m)^(#+) ", lambda m: "#" * (len(m.group(1)) + 1) + " ",
                         (OUT / "report_tables.md").read_text().split("## The negative control")[0].strip()),
    "",
    "## The negative control",
    "",
    "Declared: record 2 with SDI and turbidity shifted in time (circular, 60–300 days per plant); XGBoost's skill 7 and 14 days "
    "ahead must fall back to within 2 points of its skill without those two columns; if not, nothing is reported until the cause "
    "is found.",
    "",
    "Skill against *no change*, %, rolling test, three plants pooled (shifts: "
    + ", ".join(f"{p} {k} days" for p, k in POC["control"]["shift_days"].items()) + "):",
    "",
    "| target · h | as run | without SDI and turbidity | shifted | shifted − without | as run − without |",
    "|---|---|---|---|---|---|",
]
for key, v in POC["control"]["skill"].items():
    tg, h = key.split("|")
    lines.append(f"| {tg} · {h} d | {v['record 2']:+.2f} | {v['record 2 without SDI and turbidity']:+.2f} | "
                 f"{v['record 2, SDI and turbidity shifted']:+.2f} | **{v['shifted - without']:+.2f}** | {v['as run - without']:+.2f} |")
lines += [
    "",
    f"**Verdict as declared: {'passed' if POC['control']['ok'] else 'missed'}**"
    + ("." if POC["control"]["ok"] else
       f" — in one cell of four: flux 14 days ahead, {gap:+.2f} points (bar: within 2). For ΔP the shift took the columns' gain "
       "away, as it should."),
    "",
    "### The investigation",
    "",
    "1. **The code** (`ctrl_check1.py`): the shifted build is identical to the as-run build in every column except the twelve "
    "SDI and turbidity inputs, in all three plants; the shift only moves values in time. No target, no hidden quantity and no "
    "future value of the plant's own record enters through it.",
    f"2. **The seeds** (`ctrl_check2.py`): the missed cell re-run with the declared seeds (0–4; it reproduces the run's "
    f"{gap:+.2f}: {'yes' if rep else 'NO'}) and with two other sets of five seeds. Nothing else changes.",
    f"3. **The yearly cycle** (`ctrl_check3.py`): SDI and turbidity carry a yearly cycle (on the log scale, "
    f"{NOTE['cycle_share']} of their variance), and a circular shift moves that cycle but keeps it. The same cell with the "
    "cycle taken out before the same shift.",
    "",
    "Flux 14 days ahead, XGBoost's skill, %:",
    "",
    ctl.round(2).to_markdown(),
    "",
    NOTE["finding"],
    "",
    "## Which columns XGBoost used",
    "",
    (OUT / "report_tables.md").read_text().split("## Which columns XGBoost used")[1].strip().split("\n", 1)[1].strip()
    if "## Which columns XGBoost used" in (OUT / "report_tables.md").read_text() else "",
    "",
    "Read with care: the no-link columns play no part in this simulator, yet permuting them moves the error about as much as the "
    "causes — slow signals and set-point steps can line up with fouling over a short record by chance. The results group holds "
    "the indicators' own past (record 1's inputs) and the measured pressures and ΔP.",
    "",
    "## Notes",
    "",
    "- The no-link columns come from `default_rng([seed, 4])` with the plant's seed (4101–4103), as declared; the plant's own "
    "draws use their streams, so the plant is draw for draw what version 2 makes without these columns.",
    "- The deck refuses a partial run: `poc_data.js` carries `final`, true only with all three plants and the networks.",
    "- Before anything goes to the repo: the declaration names suppliers and two plants in its sources (a product sheet's maker, "
    "a supplier's guide, plants in a published comparison). The deck names only the membrane makers' manuals; the owner decides "
    "on the declaration's text.",
    "",
    "## Files",
    "",
]
for f in ["declaration_poc.md", "make_table.py", "poc_lib.py", "poc_trees.py", "poc_nets.py", "make_poc_data.py", "ctrl_check1.py",
          "ctrl_check2.py", "ctrl_check3.py", "out/pred_trees.csv", "out/pred_nets.csv", "out/pred_control.csv", "out/poc_data.js"]:
    lines.append(f"- `{f}` · SHA-256 `{sha(HERE / f)}`")
(OUT / "REPORT_POC.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines[:40]))
