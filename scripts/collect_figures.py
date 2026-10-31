"""Copy the group-level result figures into docs/img/results (committed, outputs/ is not)."""

import shutil
from pathlib import Path

src, dst = Path("outputs/group/figures"), Path("docs/img/results")
dst.mkdir(parents=True, exist_ok=True)
for f in src.glob("*.png"):
    shutil.copy(f, dst / f.name)
    print("copied", f.name)
