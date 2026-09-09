"""Small, reproducible latency and GPU-memory benchmarking utilities."""

import time
import torch


def benchmark_callable(fn, inputs, warmup=1):
    inputs = list(inputs)
    if not inputs:
        return {"samples": 0, "total_seconds": 0.0, "average_seconds": 0.0, "outputs": []}

    for x in inputs[:warmup]:
        fn(x)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    outputs = [fn(x) for x in inputs]

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    peak_gb = None
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    return {
        "samples": len(inputs),
        "total_seconds": elapsed,
        "average_seconds": elapsed / len(inputs),
        "throughput_samples_per_second": len(inputs) / elapsed if elapsed else 0.0,
        "peak_gpu_memory_gb": peak_gb,
        "outputs": outputs,
    }
