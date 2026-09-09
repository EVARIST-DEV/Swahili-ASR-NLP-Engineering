"""Quantization helpers for constrained local inference.

The project uses 4-bit NF4 quantization for the Qwen reasoning model during
QLoRA training/inference. Whisper Small is kept in FP16 for the reported ASR
benchmark because the measured T4 footprint is already modest and no
post-quantization accuracy benchmark was completed during the assessment.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantizationPlan:
    reasoning_bits: int = 4
    reasoning_scheme: str = "NF4"
    double_quant: bool = True
    compute_dtype: str = "float16"
    asr_dtype: str = "float16"

    def as_dict(self):
        return {
            "reasoning_bits": self.reasoning_bits,
            "reasoning_scheme": self.reasoning_scheme,
            "double_quant": self.double_quant,
            "compute_dtype": self.compute_dtype,
            "asr_dtype": self.asr_dtype,
        }


def make_bnb_4bit_config():
    """Return a BitsAndBytesConfig for the tested QLoRA setup."""
    from transformers import BitsAndBytesConfig
    import torch

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
