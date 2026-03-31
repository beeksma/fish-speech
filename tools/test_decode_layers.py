#!/usr/bin/env python3
"""Minimal test: run DAC decode with miopen-conv-fix in a single process.

Tests all conv layers sequentially (like the server does) to identify
which layer causes the GPU page fault.

Adds hipDeviceSynchronize + error checking after each conv layer by
wrapping the model forward with hooks.
"""

import os
import sys
import time

import torch
import numpy as np

os.environ.setdefault("HSA_ENABLE_SDMA", "0")
os.environ.setdefault("GPU_MAX_HW_QUEUES", "1")
os.environ.setdefault("HSA_USE_SVM", "0")

# Apply VRAM fraction early
vram_fraction = float(os.environ.get("VRAM_FRACTION", "0"))
if 0 < vram_fraction <= 1 and torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(vram_fraction)

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from fish_speech.models.dac.inference import load_model


def add_sync_hooks(model):
    """Add forward hooks that synchronize GPU after each conv layer."""
    conv_count = [0]
    crash_layer = [None]

    def make_hook(name):
        def hook(module, input, output):
            try:
                torch.cuda.synchronize()
                conv_count[0] += 1
            except RuntimeError as e:
                crash_layer[0] = name
                print(f"\n*** GPU ERROR after layer: {name} ***", file=sys.stderr)
                print(f"    Module: {module}", file=sys.stderr)
                print(f"    Error: {e}", file=sys.stderr)
                if hasattr(input[0], 'shape'):
                    print(f"    Input shape: {input[0].shape}", file=sys.stderr)
                if hasattr(output, 'shape'):
                    print(f"    Output shape: {output.shape}", file=sys.stderr)
                raise
        return hook

    hooks = []
    for name, mod in model.named_modules():
        if isinstance(mod, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            hooks.append(mod.register_forward_hook(make_hook(name)))

    return hooks, conv_count, crash_layer


def main():
    checkpoint = os.environ.get(
        "DECODER_CHECKPOINT_PATH", "checkpoints/fish-speech-s2-pro-int8/codec.pth"
    )
    config = os.environ.get("DECODER_CONFIG_NAME", "modded_dac_vq")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading decoder from {checkpoint}...")
    model = load_model(config, checkpoint, device=device, precision=torch.bfloat16)
    print(f"Model loaded. GPU mem: {torch.cuda.memory_allocated()/1e6:.0f}MB")

    # Add sync hooks to catch the exact failing layer
    hooks, conv_count, crash_layer = add_sync_hooks(model)

    # Generate dummy VQ codes (10 codebooks, 512 tokens — same as DECODE_PAD_TO)
    codes = torch.zeros(1, 10, 512, dtype=torch.long, device=device)

    print(f"\nRunning decode (512 tokens, {len(hooks)} conv hooks)...")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            result = model.from_indices(codes)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"\nDecode SUCCESS in {dt:.3f}s")
        print(f"  Output shape: {result.shape}")
        print(f"  Conv layers executed: {conv_count[0]}")
        print(f"  GPU mem: {torch.cuda.memory_allocated()/1e6:.0f}MB")
    except Exception as e:
        print(f"\nDecode FAILED after {conv_count[0]} conv layers")
        if crash_layer[0]:
            print(f"  Crash layer: {crash_layer[0]}")
        print(f"  Error: {e}")
        sys.exit(1)

    # Clean up hooks
    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()
