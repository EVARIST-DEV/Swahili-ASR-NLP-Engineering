"""Reference local-serving composition for the modular pipeline.

This file intentionally avoids downloading models at import time. The actual
model paths/checkpoints are supplied by the caller or environment.
"""

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    asr_checkpoint: str
    reasoning_base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    reasoning_adapter: str | None = None
    sampling_rate: int = 16000


def describe_local_pipeline(config: PipelineConfig) -> dict:
    """Return a deployment manifest suitable for logging/configuration."""
    return {
        "architecture": "modular_multimodal",
        "asr_checkpoint": config.asr_checkpoint,
        "reasoning_base_model": config.reasoning_base_model,
        "reasoning_adapter": config.reasoning_adapter,
        "sampling_rate": config.sampling_rate,
        "local_serving": True,
    }
