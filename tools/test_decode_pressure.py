#!/usr/bin/env python3
"""Test: does high GPU memory pressure cause miopen-conv-fix page fault?

Server had 12.2GB used during decode. This test simulates that exact pressure.
"""

import os
import sys
import time
import threading
import queue

import torch

os.environ.setdefault("HSA_ENABLE_SDMA", "0")
os.environ.setdefault("GPU_MAX_HW_QUEUES", "1")
os.environ.setdefault("HSA_USE_SVM", "0")

vram_fraction = float(os.environ.get("VRAM_FRACTION", "0"))
if 0 < vram_fraction <= 1 and torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(vram_fraction)

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from fish_speech.models.dac.inference import load_model as load_decoder
from fish_speech.utils.gpu import auto_detect_rocm_gfx

auto_detect_rocm_gfx()


def gpu_worker(filler_size_gb, init_event, work_queue, keep_allocated):
    """Background thread that allocates GPU memory + does matmuls."""
    # Simulate LLM model weights + KV cache
    n_elements = int(filler_size_gb * 1024**3 / 2)  # bf16 = 2 bytes
    print(f"[worker] Allocating {filler_size_gb}GB...")
    weights = torch.randn(n_elements, dtype=torch.bfloat16, device="cuda")

    # Do heavy matmuls
    print("[worker] Running matmuls...")
    with torch.no_grad():
        a = torch.randn(1, 2560, 9728, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(1, 9728, 2560, dtype=torch.bfloat16, device="cuda")
        for _ in range(10):
            c = torch.bmm(a, b)
        torch.cuda.synchronize()
    del a, b, c
    torch.cuda.empty_cache()

    if not keep_allocated:
        del weights
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"[worker] Freed filler. GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    else:
        print(f"[worker] Keeping filler. GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    init_event.set()
    item = work_queue.get()
    if keep_allocated:
        del weights
    print("[worker] Shutting down")


def test_decode(dec_path, label):
    print(f"\n[main] Running VQ decode ({label})...")
    dec_model = load_decoder("modded_dac_vq", dec_path, device="cuda", precision=torch.bfloat16)
    print(f"[main] Decoder loaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    codes = torch.zeros(1, 10, 512, dtype=torch.long, device="cuda")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            result = dec_model.from_indices(codes)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"[main] Decode SUCCESS in {dt:.3f}s ({label})")
        del dec_model, codes, result
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"[main] Decode FAILED ({label}): {e}")
        return False


def main():
    dec_path = os.environ.get("DECODER_CHECKPOINT_PATH", "checkpoints/fish-speech-s2-pro-int8/codec.pth")

    # Test with worker holding 8GB on GPU (simulates LLM still loaded)
    print("=" * 60)
    print("TEST: Worker holds 8GB on GPU during decode")
    print("=" * 60)
    init_event = threading.Event()
    work_queue = queue.Queue()
    worker = threading.Thread(
        target=gpu_worker,
        args=(8.0, init_event, work_queue, True),  # keep_allocated=True
        daemon=True,
    )
    worker.start()
    init_event.wait()
    print(f"[main] Total GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    ok = test_decode(dec_path, "8GB pressure + thread")
    work_queue.put(None)
    worker.join(timeout=5)
    torch.cuda.empty_cache()

    if not ok:
        sys.exit(1)
    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
