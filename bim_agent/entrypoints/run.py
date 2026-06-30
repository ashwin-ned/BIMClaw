"""Batch evaluation runner for BIMClaw.

Usage::
    python -m bim_agent.entrypoints.run \
        --questions questions.json \
        --out-dir results/

    questions.json format:
    [
        {"ifc": "val/model_0.ifc", "question": "How many doors?", "id": "q001"},
        ...
    ]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import List


async def run_batch(args) -> None:
    from bim_agent.config import BIMAgentConfig
    from bim_agent.workflow import BIMAgentWorkflow

    questions_path = Path(args.questions)
    questions = json.loads(questions_path.read_text())

    cfg = BIMAgentConfig(
        llm_model=args.model,
        llm_base_url=args.base_url,
        max_steps=args.max_steps,
        use_rendered_images=not args.no_images,
    )
    workflow = BIMAgentWorkflow(config=cfg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, q in enumerate(questions):
        qid = q.get("id", f"q{i:04d}")
        ifc = str(Path(q["ifc"]).resolve())
        question = q["question"]
        modalities_dir = str(out_dir / "modalities" / Path(ifc).stem)

        logging.info("[run] %s: %s", qid, question)
        try:
            result = await workflow.arun(
                ifc_path=ifc,
                question=question,
                modalities_out_dir=modalities_dir,
                session_id=qid,
            )
            results.append({
                "id": qid,
                "ifc": ifc,
                "question": question,
                "answer": result["answer"],
                "steps": result["steps"],
                "termination_reason": result["termination_reason"],
                "error": None,
            })
            logging.info("[run] %s → %s", qid, result["answer"])
        except Exception as exc:
            logging.error("[run] %s FAILED: %s", qid, exc)
            results.append({
                "id": qid,
                "ifc": ifc,
                "question": question,
                "answer": None,
                "error": str(exc),
            })

        # Save incrementally
        (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\nDone. {len([r for r in results if r.get('answer')])} / {len(results)} succeeded.")
    print(f"Results saved to: {out_dir / 'results.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="BIMClaw batch evaluation")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(run_batch(args))


if __name__ == "__main__":
    main()
