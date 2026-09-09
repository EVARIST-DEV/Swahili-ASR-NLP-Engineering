import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def load_whisper(model_path_or_id, device=None, language="swahili"):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = WhisperProcessor.from_pretrained(
        str(model_path_or_id), language=language, task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        str(model_path_or_id)
    ).to(device)
    model.eval()
    return processor, model

def transcribe_audio(processor, model, waveform, sample_rate=16000):
    device = next(model.parameters()).device
    inputs = processor(
        waveform, sampling_rate=sample_rate, return_tensors="pt"
    )
    input_features = inputs.input_features.to(device)
    with torch.no_grad():
        ids = model.generate(
            input_features, language="sw", task="transcribe"
        )
    return processor.batch_decode(
        ids, skip_special_tokens=True
    )[0].strip()
