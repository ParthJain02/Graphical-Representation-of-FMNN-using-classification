"""Final evaluation script with confidence intervals (implemented in Step 4)."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final evaluation for GSHR (implemented in step 4).")
    parser.add_argument("--config", type=str, required=False, default=None, help="Path to config YAML")
    return parser.parse_args()


def main() -> None:
    _ = parse_args()
    print("implemented in step 4")


if __name__ == "__main__":
    main()
