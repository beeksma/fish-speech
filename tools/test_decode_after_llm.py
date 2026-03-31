#!/usr/bin/env python3
"""Test: does running LLM before decode cause the page fault?

Hypothesis: the LLM's hipBLAS operations leave GPU state that
interferes with MIOpen's Immediate Mode API during decode.
"""

import os
import sys
import time

import torch
import numpy as np

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


def main():
    device = "cuda"
    precision = torch.bfloat16
    dec_path = os.environ.get("DECODER_CHECKPOINT_PATH", "checkpoints/fish-speech-s2-pro-int8/codec.pth")

    # --- Step 1: Simulate LLM memory footprint + hipBLAS usage ---
    print("Allocating ~8GB to simulate LLM memory footprint...")
    filler = torch.empty(8 * 1024**3 // 2, dtype=torch.bfloat16, device=device)  # 8GB
    print(f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # --- Step 2: Run heavy matmuls (simulates LLM generation via hipBLAS) ---
    print("Running heavy matmuls (simulates hipBLAS from LLM generation)...")
    with torch.no_grad():
        a = torch.randn(1, 2560, 9728, dtype=torch.bfloat16, device=device)
        b = torch.randn(1, 9728, 2560, dtype=torch.bfloat16, device=device)
        for _ in range(10):
            c = torch.bmm(a, b)
        torch.cuda.synchronize()
    del a, b, c
    torch.cuda.empty_cache()
    print(f"  Matmul done. GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # --- Step 2b: Free filler (simulates LLM offload) ---
    del filler
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"  Filler freed. GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # --- Step 3: Load decoder ---
    print(f"\nLoading decoder from {dec_path}...")
    dec_model = load_decoder("modded_dac_vq", dec_path, device=device, precision=precision)
    torch.cuda.synchronize()
    print(f"  Decoder loaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # --- Step 4: Run VQ decode ---
    codes = torch.zeros(1, 10, 512, dtype=torch.long, device=device)
    print(f"\nRunning VQ decode (512 tokens)...")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            result = dec_model.from_indices(codes)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"\nDecode SUCCESS in {dt:.3f}s")
        print(f"  Output shape: {result.shape}")
    except Exception as e:
        print(f"\nDecode FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
