"""Entry point for ``python -m eval``."""

from __future__ import annotations

import sys

from eval.cli import main

if __name__ == "__main__":
    sys.exit(main())
