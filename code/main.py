"""Convenience launcher: `python main.py ...` is equivalent to `python -m triage.cli ...`."""
from triage.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
