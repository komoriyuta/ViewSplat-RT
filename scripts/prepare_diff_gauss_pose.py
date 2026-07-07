from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / ".uv-local" / "diff-gaussian-rasterization"
REPO = "https://github.com/slothfulxtx/diff-gaussian-rasterization.git"
REV = "b1cedf5cb565c676a1df3dc823da4d1d2cec3806"

PYPROJECT = """\
[build-system]
requires = ["setuptools", "wheel", "torch"]
build-backend = "setuptools.build_meta"
"""


def run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"Expected text not found in {path}: {old!r}")
    path.write_text(text.replace(old, new, 1))


def main() -> None:
    if TARGET.exists():
        current = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=TARGET,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
        if current != REV:
            shutil.rmtree(TARGET)

    if not TARGET.exists():
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", REPO, str(TARGET))
        run("git", "checkout", REV, cwd=TARGET)

    (TARGET / "pyproject.toml").write_text(PYPROJECT)

    replace_once(
        TARGET / "cuda_rasterizer" / "rasterizer_impl.h",
        "#include <iostream>\n",
        "#include <cstdint>\n#include <iostream>\n",
    )
    print(f"Prepared diff_gauss_pose from {REPO}@{REV}")
    print(f"Path: {TARGET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
