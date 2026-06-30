"""BIMAgentWorkflow: main entry point for the BIM spatial agent.

Orchestrates:
1. IFC loading + BIMMetadata extraction
2. (Optional) BIM-Walker rendering → RGBD + PLY
3. Kernel initialization and object injection
4. LangGraph agentic loop (plan → code → execute → feedback → reflect → route)

Usage::
    workflow = BIMAgentWorkflow(config)
    result = await workflow.arun(
        ifc_path="model_0.ifc",
        question="How many doors are on the ground floor?",
    )
    print(result["answer"])  # "5"
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from langgraph.graph import StateGraph, END

from bim_agent.config import BIMAgentConfig
from bim_agent.state import AgentState
from bim_agent.kernel.manager import JupyterKernelManager

logger = logging.getLogger(__name__)


class BIMAgentWorkflow:
    """Wires together all BIM agent components into a runnable workflow."""

    def __init__(self, config: Optional[BIMAgentConfig] = None):
        self.config = config or BIMAgentConfig()
        self._graph = None
        self._km: Optional[JupyterKernelManager] = None

    async def arun(
        self,
        ifc_path: str,
        question: str,
        modalities_out_dir: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Answer a natural-language question about an IFC file.

        Parameters
        ----------
        ifc_path:
            Path to the IFC file.
        question:
            Natural-language spatial question.
        modalities_out_dir:
            Directory to cache BIM-Walker render outputs.
            If None, uses a temp dir derived from ifc_path.
        session_id:
            Session identifier for logging. Auto-generated if not given.

        Returns
        -------
        dict with keys: answer, session_id, ifc_path, steps, termination_reason
        """
        session_id = session_id or str(uuid.uuid4())[:8]
        ifc_path = str(Path(ifc_path).resolve())
        cfg = self.config

        # Step 1: Load IFC + extract metadata
        logger.info("[BIMAgent] Loading IFC: %s", ifc_path)
        from bim_agent.kernel_types.ifc_model import IFCModel
        from bim_agent.kernel_types.bim_metadata import BIMMetadata
        ifc_model = IFCModel(ifc_path)
        bim_metadata = BIMMetadata.from_ifc(ifc_path)
        logger.info("[BIMAgent] BIMMetadata: %s", bim_metadata)

        # Step 2: (Optional) BIM-Walker rendering
        modalities_output = None
        key_frames_pil = []
        reconstruction = None

        if cfg.use_rendered_images:
            if modalities_out_dir is None:
                modalities_out_dir = str(
                    Path(ifc_path).parent / "bimclaw_modalities" / Path(ifc_path).stem
                )
            modalities_output = await asyncio.get_event_loop().run_in_executor(
                None, self._run_modalities, ifc_path, modalities_out_dir
            )
            if modalities_output is not None:
                key_frames_pil = [modalities_output.floor_plan_image] + modalities_output.key_rgb_images
                reconstruction = modalities_output.reconstruction

        # Step 3: Build kernel objects
        from bim_agent.kernel_types.input_images import InputImages
        from bim_agent.kernel_types.frame_image import FrameImage
        from bim_agent.tools import BIMToolsModule

        if key_frames_pil:
            input_images = InputImages(
                key_frames_pil,
                max_edge=cfg.image_max_edge,
            )
        else:
            input_images = InputImages([])

        tools_module = BIMToolsModule(
            ifc_model=ifc_model,
            reconstruction=reconstruction,
        )

        # Step 4: Build feedback stub
        feedback_module = _build_feedback_module()

        # Step 5: Build LLM client
        from bim_agent.llm.client import LLMClient
        llm_client = LLMClient(
            model=cfg.llm_model,
            api_key=cfg.llm_api_key,
            base_url=cfg.llm_base_url,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
        )

        # Step 6: Build system prompt
        from bim_agent.llm.system_prompt import build_bim_system_prompt
        tools_prompt = tools_module.get_prompt_descriptions()
        system_prompt = build_bim_system_prompt(
            tools_prompt=tools_prompt,
            has_reconstruction=reconstruction is not None,
            num_images=len(input_images),
        )

        # Step 7: Start kernel
        km = JupyterKernelManager(timeout_sec=cfg.timeout_sec)

        # Step 8: Build and run the LangGraph
        graph = self._build_graph()
        initial_state: AgentState = {
            "session_id": session_id,
            "ifc_path": ifc_path,
            "messages": [],
            "step_count": 0,
            "failure_count": 0,
            "max_steps": cfg.max_steps,
            "max_failures": cfg.max_failures,
            "current_llm_response": None,
            "current_step_result": None,
            "last_error_type": None,
            "kernel_id": None,
            "variable_registry": {},
            "final_answer": None,
            "termination_reason": None,
            "plan": None,
            "instruction": question,
            "total_show_images": 0,
            "total_tool_calls": 0,
            "max_tool_calls": -1,
            "sample_id": None,
            "input_metadata": {},
            "checklist": [],
            "answer_block_count": 0,
            "total_answer_attempts": 0,
            "last_submitted_answer": None,
            "consecutive_conn_failures": 0,
        }

        run_config = {
            "configurable": {
                "kernel_manager": km,
                "llm_client": llm_client,
                "tools_module": tools_module,
                "feedback_module": feedback_module,
                "system_prompt": system_prompt,
                "input_images": input_images,
                "key_frames": key_frames_pil[:cfg.num_key_frames],
                "bim_metadata": bim_metadata,
                "reconstruction": reconstruction,
                "agent_config": cfg,
                "logger": _SessionLogger(logger),
            }
        }

        try:
            final_state = await graph.ainvoke(initial_state, config=run_config)
        finally:
            await km.shutdown()

        answer = final_state.get("final_answer") or {}
        return {
            "answer": answer.get("text", answer.get("raw_value", "")),
            "raw_answer": answer,
            "session_id": session_id,
            "ifc_path": ifc_path,
            "steps": final_state.get("step_count", 0),
            "termination_reason": final_state.get("termination_reason", "unknown"),
        }

    def _run_modalities(self, ifc_path: str, out_dir: str):
        """Run BIM-Walker synchronously (called via run_in_executor)."""
        from bim_modalities.pipeline import BIMModalitiesPipeline, BIMModalitiesConfig
        cfg = self.config
        pipeline_config = BIMModalitiesConfig(
            render_frames=cfg.render_frames,
            num_key_frames=cfg.num_key_frames,
            blender=cfg.blender,
            conda_env=cfg.conda_env,
            walker_timeout_min=cfg.walker_timeout_min,
        )
        pipeline = BIMModalitiesPipeline(config=pipeline_config)
        try:
            return pipeline.extract(Path(ifc_path), Path(out_dir))
        except Exception as exc:
            logger.warning("[BIMAgent] Modalities extraction failed (non-fatal): %s", exc)
            return None

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph agentic loop."""
        from bim_agent.nodes.init_node import init_node
        from bim_agent.nodes.plan_node import plan_node
        from bim_agent.nodes.llm_step_node import llm_step_node
        from bim_agent.nodes.execute_node import execute_node
        from bim_agent.nodes.feedback_node import feedback_node
        from bim_agent.nodes.reflection_node import reflection_node
        from bim_agent.nodes.router import should_continue, force_terminate

        builder = StateGraph(AgentState)
        builder.add_node("init_node", init_node)
        builder.add_node("plan_node", plan_node)
        builder.add_node("llm_step_node", llm_step_node)
        builder.add_node("execute_node", execute_node)
        builder.add_node("feedback_node", feedback_node)
        builder.add_node("reflection_node", reflection_node)
        builder.add_node("force_terminate", force_terminate)

        builder.set_entry_point("init_node")
        builder.add_edge("init_node", "plan_node")
        builder.add_edge("plan_node", "llm_step_node")
        builder.add_edge("llm_step_node", "execute_node")
        builder.add_edge("execute_node", "feedback_node")
        builder.add_edge("feedback_node", "reflection_node")
        builder.add_conditional_edges(
            "reflection_node",
            should_continue,
            {
                "llm_step_node": "llm_step_node",
                "force_terminate": "force_terminate",
                END: END,
            },
        )
        builder.add_edge("force_terminate", END)

        return builder.compile()


# ---------------------------------------------------------------------------
# Minimal stub objects for the kernel namespace
# ---------------------------------------------------------------------------

def _build_feedback_module():
    """Return a minimal show() handler that captures images."""
    class _FeedbackModule:
        def __init__(self):
            self.images = []
        def show(self, img):
            self.images.append(img)
    return _FeedbackModule()


class _SessionLogger:
    """Thin wrapper that routes log_step() calls to the Python logger."""

    def __init__(self, py_logger: logging.Logger):
        self._log = py_logger

    def log_step(self, session_id: str, event: dict) -> None:
        import json
        event_type = event.get("event_type", "unknown")
        step = event.get("step_index", "?")
        # Log concise version; full event at DEBUG
        brief = {k: v for k, v in event.items()
                 if k not in ("raw_response", "code", "stdout", "stderr",
                              "vlm_queries", "parsed_response")}
        self._log.info("[session:%s step:%s] %s %s", session_id, step, event_type,
                       json.dumps(brief, default=str)[:300])
        if self._log.isEnabledFor(logging.DEBUG):
            self._log.debug("[session:%s step:%s] full_event=%s", session_id, step,
                            json.dumps(event, default=str)[:2000])


