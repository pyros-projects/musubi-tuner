"""Northstar training CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import toml

from northstar.config.loader import load_run_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Northstar training configuration.")
    parser.add_argument("run_config", type=str, help="Path to the Northstar TOML run configuration.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_config_path = Path(args.run_config)
    try:
        load_run_config(run_config_path)
    except FileNotFoundError:
        print(f"Config file not found: {run_config_path}", file=sys.stderr)
        return 1
    except toml.TomlDecodeError as exc:
        print(f"TOML parse error in {run_config_path}: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Config validation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
