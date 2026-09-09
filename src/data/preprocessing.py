import numpy as np

def decode_audio_sample(sample):
    """Decode a Datasets AudioDecoder sample on demand."""
    audio = sample["audio"].get_all_samples()
    waveform = audio.data.squeeze().numpy()
    return waveform.astype(np.float32), int(audio.sample_rate)

def audio_duration_seconds(sample):
    return float(sample["num_samples"]) / 16000.0
