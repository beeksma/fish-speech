#!/usr/bin/env python3
"""Test: does running the pipeline inside asyncio cause the crash?

The server runs through Uvicorn (async). This test adds asyncio +
multiprocessing.set_start_method("spawn") to the pipeline test.
"""

import asyncio
import multiprocessing
import os
import sys
import time
import queue

# Match exactly what api_server.py does
multiprocessing.set_start_method("spawn", force=True)

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


async def run_pipeline():
    """Run the exact server pipeline inside an async context."""
    device = "cuda"
    precision = torch.bfloat16
    llm_path = os.environ.get("LLAMA_CHECKPOINT_PATH", "checkpoints/fish-speech-s2-pro-int8")
    dec_path = os.environ.get("DECODER_CHECKPOINT_PATH", "checkpoints/fish-speech-s2-pro-int8/codec.pth")

    print(f"[async] Launching LLM worker from {llm_path}...")
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=llm_path, device=device, precision=precision, compile=False,
    )
    print(f"[async] LLM loaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    llama_queue.offload_to_cpu()
    print(f"[async] LLM offloaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    dec_model = load_decoder("modded_dac_vq", dec_path, device=device, precision=precision)
    print(f"[async] Decoder loaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    llama_queue.reload_to_gpu()
    print(f"[async] LLM reloaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # Generate
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

    results = []
    while True:
        # Use run_in_executor to not block the event loop (like server would)
        wrapped = await asyncio.get_event_loop().run_in_executor(None, response_queue.get)
        if wrapped.status == "error":
            print(f"LLM error: {wrapped.response}")
            sys.exit(1)
        result = wrapped.response
        if result.action != "next":
            results.append(result)
        else:
            break
    print(f"[async] Generated {len(results)} segments. GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    llama_queue.offload_to_cpu()
    print(f"[async] LLM offloaded for decode. GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    if results:
        codes = results[0].codes
        seq_len = codes.shape[-1]
        pad_to = max(seq_len, 512)
        pad_to = ((pad_to + 512 - 1) // 512) * 512
        if seq_len < pad_to:
            padded = torch.zeros((*codes.shape[:-1], pad_to), dtype=codes.dtype, device=codes.device)
            padded[..., :seq_len] = codes
        else:
            padded = codes

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                audio = dec_model.from_indices(padded[None])
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            print(f"\n*** VQ DECODE SUCCESS in {dt:.3f}s ***")
        except Exception as e:
            print(f"\n*** VQ DECODE FAILED: {e} ***")
            sys.exit(1)

    llama_queue.reload_to_gpu()
    print("[async] All done!")


if __name__ == "__main__":
    asyncio.run(run_pipeline())
