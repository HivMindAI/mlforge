"""Allow MLForge to run with python -m mlforge."""

from mlforge.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
