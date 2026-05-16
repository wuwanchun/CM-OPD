"""Serve rule or HF context policy over HTTP."""

from __future__ import annotations

import argparse

from _bootstrap import add_project_root

ROOT = add_project_root(__file__)

from context_manager.hf_policy import HFModelPolicy
from context_manager.rule_policy import RulePolicy
from context_manager.schemas import ContextActionSample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["rule", "model"], default="rule")
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    args = parser.parse_args()

    try:
        from fastapi import FastAPI
        import uvicorn
    except ImportError as exc:
        raise SystemExit("fastapi and uvicorn are required. Install requirements.txt first.") from exc

    policy = RulePolicy() if args.policy == "rule" else HFModelPolicy(args.model_path)
    app = FastAPI(title="Context Policy Server")

    @app.get("/healthz")
    async def healthz() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/predict")
    async def predict(payload: dict) -> dict:
        sample = ContextActionSample.from_dict(payload)
        return policy.predict(sample).to_dict()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
