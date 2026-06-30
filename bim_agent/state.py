"""LangGraph state for the BIM agent."""

from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class ChecklistItem(TypedDict):
    """A single verification task tracked across the agent loop."""

    item_id: str
    description: str
    priority: Literal["HIGH", "MEDIUM", "LOW"]
    status: Literal["PENDING", "VERIFIED", "FLAGGED"]
    added_at_step: int
    resolved_at_step: Optional[int]
    resolution_note: Optional[str]


class AgentState(TypedDict):
    """LangGraph state for the BIM Spatial Understanding Agent."""

    # Identity
    session_id: str
    ifc_path: str

    # Conversation
    messages: Annotated[List[AnyMessage], add_messages]

    # Loop counters
    step_count: int
    failure_count: int
    max_steps: int
    max_failures: int

    # Current step
    current_llm_response: Optional[Dict[str, str]]
    current_step_result: Optional[Dict]
    last_error_type: Optional[str]

    # Kernel
    kernel_id: Optional[str]
    variable_registry: Dict[str, Dict[str, Any]]

    # Final result
    final_answer: Optional[Dict[str, Any]]
    termination_reason: Optional[str]
    plan: Optional[str]

    # Input
    instruction: str
    total_show_images: int
    total_tool_calls: int
    max_tool_calls: int
    sample_id: Optional[str]
    input_metadata: Dict[str, Any]

    # Reflection checklist
    checklist: List[ChecklistItem]
    answer_block_count: int
    total_answer_attempts: int
    last_submitted_answer: Optional[Dict[str, Any]]

    # Connection failure tracking (resets to 0 on any successful LLM call)
    consecutive_conn_failures: int
