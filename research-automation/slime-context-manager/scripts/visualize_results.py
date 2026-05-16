"""Generate visual analysis artifacts from metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_project_root

ROOT = add_project_root(__file__)

from visualization.plots import write_analysis_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default=str(ROOT / "runs" / "metrics.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "analysis"))
    args = parser.parse_args()

    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    write_analysis_report(metrics, args.output_dir)
    print(f"Wrote analysis artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
