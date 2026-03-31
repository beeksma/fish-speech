#!/usr/bin/env python3
"""Test: exact server pipeline without Uvicorn.

Replicates the server warmup: LLM worker thread → generate → offload → decode.
"""

import os
import sys
import time
import queue

import torch

os.environ.setdefault("HSA_ENABLE_SDMA", "0")
os.environ.setdefault("GPU_MAX_HW_QUEUES", "1")
os.environ.setdefault("HSA_USE_SVM", "0")

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from fish_speech.models.dac.inference import load_model as load_decoder
from fish_speech.models.text2semantic.inference import (
    launch_thread_safe_queue,
    GenerateRequest,
    GenerateResponse,
    WrappedGenerateResponse,
)
from fish_speech.utils.gpu import auto_detect_rocm_gfx

auto_detect_rocm_gfx()


def main():
    device = "cuda"
    precision = torch.bfloat16
    llm_path = os.environ.get("LLAMA_CHECKPOINT_PATH", "checkpoints/fish-speech-s2-pro-int8")
    dec_path = os.environ.get("DECODER_CHECKPOINT_PATH", "checkpoints/fish-speech-s2-pro-int8/codec.pth")

    # --- Step 1: Launch LLM worker thread (exactly like server) ---
    print(f"Launching LLM worker from {llm_path}...")
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=llm_path,
        device=device,
        precision=precision,
        compile=False,
    )
    print(f"LLM loaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # --- Step 2: Offload LLM (like model_manager init) ---
    print("Offloading LLM (init-time offload)...")
    llama_queue.offload_to_cpu()
    print(f"LLM offloaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # --- Step 3: Load decoder ---
    print(f"Loading decoder from {dec_path}...")
    dec_model = load_decoder("modded_dac_vq", dec_path, device=device, precision=precision)
    print(f"Decoder loaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # --- Step 4: Reload LLM (like start of inference) ---
    print("Reloading LLM for generation...")
    llama_queue.reload_to_gpu()
    print(f"LLM on GPU. GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # --- Step 5: Send generation request ---
    print("\nSending generation request...")
    response_queue = queue.Queue()
    request = GenerateRequest(
        request=dict(
            device=torch.device(device),
            max_new_tokens=1024,
            text="Hello world.",
            prompt_tokens=None,
            prompt_text=None,
        ),
        response_queue=response_queue,
    )
    llama_queue.put(request)

    # Collect results
    results = []
    while True:
        wrapped = response_queue.get()
        if wrapped.status == "error":
            print(f"LLM error: {wrapped.response}")
            sys.exit(1)
        result = wrapped.response
        if result.action != "next":
            results.append(result)
        else:
            break
    print(f"LLM generated {len(results)} segments. GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # --- Step 6: Offload LLM before decode (like inference engine) ---
    print("Offloading LLM for decode...")
    llama_queue.offload_to_cpu()
    print(f"LLM offloaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # --- Step 7: VQ decode ---
    if results:
        codes = results[0].codes
        print(f"\nRunning VQ decode ({codes.shape})...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Pad to 512
        seq_len = codes.shape[-1]
        pad_to = max(seq_len, 512)
        pad_to = ((pad_to + 512 - 1) // 512) * 512
        if seq_len < pad_to:
            padded = torch.zeros((*codes.shape[:-1], pad_to), dtype=codes.dtype, device=codes.device)
            padded[..., :seq_len] = codes
        else:
            padded = codes

        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                audio = dec_model.from_indices(padded[None])
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            print(f"\n*** VQ DECODE SUCCESS in {dt:.3f}s ***")
            print(f"  Audio shape: {audio.shape}")
        except Exception as e:
            print(f"\n*** VQ DECODE FAILED: {e} ***")
            sys.exit(1)
    else:
        print("No results from LLM, skipping decode")

    # Reload for cleanup
    llama_queue.reload_to_gpu()
    print("\nAll done!")


if __name__ == "__main__":
    main()
