"""Validation hyperparameter tuning script (implemented in Step 3)."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation tuning for GSHR (implemented in step 3).")
    parser.add_argument("--config", type=str, required=False, default=None, help="Path to config YAML")
    return parser.parse_args()


def main() -> None:
    _ = parse_args()
    print("implemented in step 3")


if __name__ == "__main__":
    main()
