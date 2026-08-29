
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(ROOT / "app.py"),
            ],
            cwd=str(ROOT),
            check=False,
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
