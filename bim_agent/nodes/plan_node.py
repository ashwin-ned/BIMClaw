"""plan_node: blind planner for BIM agent.

Receives BIMMetadata text summary + question, produces a step-by-step plan
of which IFC queries and geometric operations to perform.

Non-fatal: if planning fails, the agent proceeds directly to coding.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from bim_agent.llm.planning_prompt import (
    build_bim_planning_system_prompt,
    build_bim_planning_user_message,
)
from bim_agent.state import AgentState

_logger = logging.getLogger(__name__)

TRANSITION_PROMPT = (
    "Good plan. Now execute it step by step. Each response MUST use the required markdown format "
    "(**Purpose**, **Reasoning**, **Next Goal**, **Code** with a ```python fenced block). "
    "Start with step 1 of your plan."
)


async def plan_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Generate a BIM-aware execution plan in an isolated LLM session."""
    cfg = config["configurable"]
    agent_config = cfg["agent_config"]

    if not getattr(agent_config, "enable_planning", True):
        return {}
    if getattr(agent_config, "max_steps", 20) <= 1:
        return {}

    llm_client = cfg["llm_client"]
    bim_metadata = cfg.get("bim_metadata")
    key_frames = cfg.get("key_frames", [])

    system_prompt = build_bim_planning_system_prompt()
    user_message = build_bim_planning_user_message(
        question=state["instruction"],
        bim_metadata=bim_metadata,
        has_reconstruction=cfg.get("reconstruction") is not None,
        num_key_frames=len(key_frames),
    )

    try:
        planning_params = getattr(agent_config, "planning_params", None)
        response = await llm_client.agenerate(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            params=planning_params,
        )
        plan_text = response.content.strip()
        _logger.info("[plan_node] Plan generated (%d chars)", len(plan_text))
    except Exception as exc:
        _logger.warning("[plan_node] Planning failed (non-fatal): %s", exc)
        return {}

    if not plan_text:
        return {}

    return {
        "plan": plan_text,
        "messages": [
            HumanMessage(content=f"**Execution Plan:**\n\n{plan_text}\n\n{TRANSITION_PROMPT}"),
        ],
    }
