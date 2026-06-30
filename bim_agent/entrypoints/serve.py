"""Interactive CLI for BIMClaw agent.

Usage::
    python -m bim_agent.entrypoints.serve \
        --ifc BIM-GNI-Dataset/val/model_0.ifc \
        --question "How many doors are on the ground floor?"

    # Or interactive REPL (one IFC, multiple questions):
    python -m bim_agent.entrypoints.serve \
        --ifc BIM-GNI-Dataset/val/model_0.ifc
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="BIMClaw — answer spatial questions about IFC files"
    )
    p.add_argument("--ifc", required=True, help="Path to IFC file")
    p.add_argument("--question", "-q", default=None, help="Single question (REPL if omitted)")
    p.add_argument("--out", default=None, help="Modalities cache directory")
    p.add_argument("--model", default="claude-sonnet-4-6", help="LLM model ID")
    p.add_argument("--base-url", default="", help="OpenAI-compat base URL (leave empty for Anthropic)")
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--no-images", action="store_true", help="Disable rendered images (IFC only)")
    p.add_argument("--render-frames", type=int, default=64, help="BIM-Walker frame budget")
    p.add_argument("--num-key-frames", type=int, default=16)
    p.add_argument("--blender", default="/usr/bin/blender")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


async def _run_once(workflow, ifc_path: str, question: str, out: str) -> None:
    print(f"\nQuestion: {question}")
    print("-" * 60)
    result = await workflow.arun(
        ifc_path=ifc_path,
        question=question,
        modalities_out_dir=out,
    )
    print(f"\nAnswer: {result['answer']}")
    print(f"Steps: {result['steps']} | Termination: {result['termination_reason']}")


async def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from bim_agent.config import BIMAgentConfig
    from bim_agent.workflow import BIMAgentWorkflow

    cfg = BIMAgentConfig(
        llm_model=args.model,
        llm_base_url=args.base_url,
        max_steps=args.max_steps,
        use_rendered_images=not args.no_images,
        render_frames=args.render_frames,
        num_key_frames=args.num_key_frames,
        blender=args.blender,
    )
    workflow = BIMAgentWorkflow(config=cfg)

    ifc_path = str(Path(args.ifc).resolve())
    out = args.out

    if args.question:
        await _run_once(workflow, ifc_path, args.question, out)
    else:
        print(f"BIMClaw interactive mode. IFC: {ifc_path}")
        print("Type your question and press Enter. Ctrl+C or 'quit' to exit.\n")
        while True:
            try:
                question = input("Question> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye.")
                break
            if question.lower() in ("quit", "exit", "q"):
                break
            if not question:
                continue
            await _run_once(workflow, ifc_path, question, out)


if __name__ == "__main__":
    asyncio.run(main())
