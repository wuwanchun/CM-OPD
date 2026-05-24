import argparse
import datetime as dt
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "templates"
RUNS = ROOT / "runs"


def slugify(text: str) -> str:
    keep = []
    for ch in text.lower().strip():
        if ch.isalnum():
            keep.append(ch)
        elif ch in {" ", "-", "_"}:
            keep.append("-")
    slug = "".join(keep).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "research-run"


def copy_template(name: str, target: Path, topic: str) -> None:
    source = TEMPLATES / name
    text = source.read_text(encoding="utf-8")
    header = f"<!-- Topic: {topic} -->\n<!-- Created: {dt.date.today().isoformat()} -->\n\n"
    target.write_text(header + text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a new research iteration folder.")
    parser.add_argument("--topic", required=True, help="Short name for the research run.")
    args = parser.parse_args()

    today = dt.date.today().isoformat()
    run_dir = RUNS / f"{today}-{slugify(args.topic)}"
    run_dir.mkdir(parents=True, exist_ok=False)

    copy_template("paper-card.md", run_dir / "paper-card.md", args.topic)
    copy_template("experiment-card.md", run_dir / "experiment-card.md", args.topic)
    copy_template("eval-report.md", run_dir / "eval-report.md", args.topic)

    notes = run_dir / "notes.md"
    notes.write_text(
        f"# Notes\n\nTopic: {args.topic}\nDate: {today}\n\n## Daily Log\n\n- \n",
        encoding="utf-8",
    )

    print(run_dir)


if __name__ == "__main__":
    main()

