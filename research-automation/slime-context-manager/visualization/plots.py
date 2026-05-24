"""Generate lightweight visual reports for evaluation metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_analysis_report(metrics: dict[str, Any], output_dir: str | Path) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_svg_bars(metrics, target / "metrics_bar.svg")
    report = [
        "# Context Policy Evaluation Analysis",
        "",
        "## Summary",
        "",
        f"- Samples: `{metrics.get('num_samples', 0)}`",
        f"- Action accuracy: `{metrics.get('action_accuracy', 0.0):.4f}`",
        f"- Macro F1: `{metrics.get('macro_f1', 0.0):.4f}`",
        f"- Evidence recall: `{metrics.get('evidence_recall', 0.0):.4f}`",
        f"- Invalid output rate: `{metrics.get('invalid_output_rate', 0.0):.4f}`",
        f"- Forbidden action rate: `{metrics.get('forbidden_action_rate', 0.0):.4f}`",
        "",
        "## Files",
        "",
        "- `metrics.json`",
        "- `metrics_bar.svg`",
    ]
    (target / "analysis_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def _write_svg_bars(metrics: dict[str, Any], path: Path) -> None:
    values = [
        ("accuracy", float(metrics.get("action_accuracy", 0.0))),
        ("macro_f1", float(metrics.get("macro_f1", 0.0))),
        ("evidence", float(metrics.get("evidence_recall", 0.0))),
        ("invalid", float(metrics.get("invalid_output_rate", 0.0))),
        ("forbidden", float(metrics.get("forbidden_action_rate", 0.0))),
    ]
    width = 720
    height = 260
    bar_width = 90
    gap = 35
    base_y = 210
    max_h = 160
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="30" font-family="Arial" font-size="18" font-weight="700">Context Policy Metrics</text>',
        f'<line x1="40" y1="{base_y}" x2="{width - 30}" y2="{base_y}" stroke="#333"/>',
    ]
    x = 50
    for label, value in values:
        bar_h = max(0, min(max_h, int(value * max_h)))
        y = base_y - bar_h
        color = "#2563eb" if label not in {"invalid", "forbidden"} else "#dc2626"
        svg.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_h}" fill="{color}" rx="4"/>')
        svg.append(f'<text x="{x + bar_width / 2}" y="{y - 8}" text-anchor="middle" font-family="Arial" font-size="12">{value:.2f}</text>')
        svg.append(f'<text x="{x + bar_width / 2}" y="{base_y + 22}" text-anchor="middle" font-family="Arial" font-size="12">{label}</text>')
        x += bar_width + gap
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")
