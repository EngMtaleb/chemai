"""Where Project 5 (when to stop) comes from.

Project 5 is no longer on main: it is kept in this repository's history at commit 0b12b39. Phase 3 needs one of its files,
projects/p05_when_to_stop/step4_ro_cip_economics.py (the cycle inputs and the cleaning-economics constants). That file is read
from git history into a temporary folder, checked against the SHA-256 of the copy the results were made with, and the folder is
put on sys.path. To use a copy you already have instead, set P05_DIR to the folder that holds step4_ro_cip_economics.py.
"""
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

COMMIT = "0b12b392e4efdbe7227d3dff201882b1a9926cd8"
FILE = "projects/p05_when_to_stop/step4_ro_cip_economics.py"
SHA256 = "fa872ead288e235083e5fddc24dbee935c8e16e1cc0e4c404bd414faa2a60a2f"


def add_to_path() -> Path:
    """Put a folder holding Project 5's step4 file (exactly as at commit 0b12b39) on sys.path; return the folder."""
    given = os.environ.get("P05_DIR")
    if given:
        folder = Path(given)
        code = (folder / Path(FILE).name).read_bytes()
    else:
        root = Path(__file__).resolve().parents[3]
        try:
            code = subprocess.run(["git", "show", f"{COMMIT}:{FILE}"], cwd=root, capture_output=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError) as e:
            raise SystemExit(f"Project 5 is kept at commit {COMMIT[:7]}, and this clone cannot read it ({e}). Clone the "
                             "repository with its full history, or set P05_DIR to a folder holding "
                             f"{Path(FILE).name} from that commit.") from None
        folder = Path(tempfile.mkdtemp(prefix="p05_0b12b39_"))
        (folder / Path(FILE).name).write_bytes(code)
    got = hashlib.sha256(code).hexdigest()
    if got != SHA256:
        raise SystemExit(f"{Path(FILE).name} is not the file of commit {COMMIT[:7]} (SHA-256 {got[:12]}..., expected {SHA256[:12]}...)")
    sys.path.insert(0, str(folder))
    return folder
