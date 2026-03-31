#!/usr/bin/env python3
"""Test: does running decode with a GPU-active background thread cause the crash?

The server has a worker thread that loaded/ran the LLM and is now idle.
This test replicates that exact pattern.
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


def gpu_worker(init_event, work_queue):
    """Background thread that does GPU work then waits (mimics LLM worker)."""
    # Allocate GPU tensors (like LLM model weights)
    print("[worker] Allocating GPU tensors...")
    weights = torch.randn(2560, 9728, dtype=torch.bfloat16, device="cuda")

    # Do heavy matmuls (like LLM generation)
    print("[worker] Running matmuls...")
    with torch.no_grad():
        x = torch.randn(1, 32, 2560, dtype=torch.bfloat16, device="cuda")
        for _ in range(10):
            out = torch.matmul(x, weights)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    del x, out

    print(f"[worker] Done. GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    init_event.set()

    # Now idle — wait for shutdown (like LLM worker waiting on queue)
    item = work_queue.get()
    # Clean up
    del weights
    print("[worker] Shutting down")


def main():
    device = "cuda"
    dec_path = os.environ.get("DECODER_CHECKPOINT_PATH", "checkpoints/fish-speech-s2-pro-int8/codec.pth")

    # --- Step 1: Start background thread with GPU work ---
    init_event = threading.Event()
    work_queue = queue.Queue()
    worker = threading.Thread(target=gpu_worker, args=(init_event, work_queue), daemon=True)
    worker.start()
    init_event.wait()
    print(f"[main] Worker done, GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB\n")

    # --- Step 2: Load decoder on main thread ---
    print(f"[main] Loading decoder from {dec_path}...")
    dec_model = load_decoder("modded_dac_vq", dec_path, device=device, precision=torch.bfloat16)
    torch.cuda.synchronize()
    print(f"[main] Decoder loaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # --- Step 3: Run VQ decode on main thread (worker idle in background) ---
    codes = torch.zeros(1, 10, 512, dtype=torch.long, device=device)
    print(f"\n[main] Running VQ decode (512 tokens)...")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            result = dec_model.from_indices(codes)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"\n[main] Decode SUCCESS in {dt:.3f}s")
        print(f"  Output shape: {result.shape}")
    except Exception as e:
        print(f"\n[main] Decode FAILED: {e}")
        work_queue.put(None)
        sys.exit(1)

    # Shutdown worker
    work_queue.put(None)
    worker.join(timeout=5)
    print("[main] All done!")


if __name__ == "__main__":
    main()
