"""BIMAgentConfig: all configuration for the BIM agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class BIMAgentConfig:
    """Configuration for the BIM spatial understanding agent."""

    # LLM connection
    llm_model: str = "claude-sonnet-4-6"
    llm_base_url: str = ""           # empty = Anthropic API; non-empty = OpenAI-compat
    llm_api_key: str = ""            # loaded from ANTHROPIC_API_KEY if empty

    # LLM parameters
    max_tokens: int = 4096
    temperature: float = 0.2
    planning_temperature: float = 0.4
    planning_max_tokens: int = 2048

    # Agent loop
    max_steps: int = 20
    max_failures: int = 15
    timeout_sec: int = 600          # per Jupyter cell execution

    # Planning
    enable_planning: bool = True

    # Reflection
    enable_reflection: bool = False

    # Modalities
    use_rendered_images: bool = True
    num_key_frames: int = 16
    render_frames: int = 64         # BIM-Walker frame budget
    image_max_edge: int = 1024      # resize key frames before sending to LLM

    # Vision: set False for text-only local models (Ollama without VL model)
    vision_enabled: bool = True

    # BIM-Walker
    blender: str = "/usr/bin/blender"
    conda_env: str = "databim"
    walker_timeout_min: int = 20

    # Logging
    work_dir: Optional[str] = None
    enable_logging: bool = True

    # Concurrency
    concurrency: int = 4

    # SpatialClaw-compatible fields (used by copied nodes)
    executor_type: str = "code"        # always "code" in BIMClaw
    enable_sighted_feedback: bool = True
    max_show_images_per_step: int = -1
    max_show_images_per_session: int = 250
    show_image_labels: bool = True
    condense_errors: bool = True
    max_variable_size_mb: int = 500
    reflect_every_n_steps: int = 5
    max_answer_blocks: int = -1
    max_total_answer_attempts: int = 5
    min_budget_for_answer_reject: int = 3
    prompt_section_ablations: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Allow env vars to override dataclass defaults
        if not self.llm_base_url:
            self.llm_base_url = os.environ.get("LLM_BASE_URL", "")
        if self.llm_model == "claude-sonnet-4-6":
            env_model = os.environ.get("LLM_MODEL", "")
            if env_model:
                self.llm_model = env_model
        if not self.llm_api_key:
            self.llm_api_key = (
                os.environ.get("LLM_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or (os.environ.get("ANTHROPIC_API_KEY") if not self.llm_base_url else None)
                or "ollama"
            )
        # Auto-disable vision for known text-only models
        _vision_models = ("claude", "gpt-4o", "gemini", "vl", "vision", "llava", "minicpm-v")
        if not any(v in self.llm_model.lower() for v in _vision_models):
            self.vision_enabled = False

    @property
    def main_params(self):
        """Compatibility shim: returns a dict-like object for the main LLM role."""
        return _SimpleParams(self.max_tokens, self.temperature)

    @property
    def planning_params(self):
        return _SimpleParams(self.planning_max_tokens, self.planning_temperature)

    @property
    def reflection_params(self):
        return _SimpleParams(2048, 0.6)

    @property
    def general_params(self):
        return _SimpleParams(self.max_tokens, 0.6)


class _SimpleParams:
    """Minimal stand-in for SpatialClaw's LLMRoleParams."""

    def __init__(self, max_tokens: int = 16384, temperature: float = 0.6):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = None
        self.top_k = None
        self.enable_thinking = None

    def to_dict(self):
        d = {"max_tokens": self.max_tokens, "temperature": self.temperature}
        return {k: v for k, v in d.items() if v is not None}
