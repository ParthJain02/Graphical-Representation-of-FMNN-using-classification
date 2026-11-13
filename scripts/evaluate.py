"""Results aggregation, figure generation, and report writer (implemented in Step 6)."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate results and generate report (implemented in step 6).")
    parser.add_argument("--config", type=str, required=False, default=None, help="Path to config YAML")
    return parser.parse_args()


def main() -> None:
    _ = parse_args()
    print("implemented in step 6")


if __name__ == "__main__":
    main()
