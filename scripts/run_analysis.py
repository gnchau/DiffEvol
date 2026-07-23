#!/usr/bin/env python3
"""Thin wrapper so the pipeline can be run without installing the package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffevol.cli import main

if __name__ == "__main__":
    sys.exit(main())
