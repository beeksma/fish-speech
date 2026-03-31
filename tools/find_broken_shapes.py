"""Find MIOpen conv shapes that crash on gfx1201.

Iterates through every Conv1d/ConvTranspose1d in the DAC model, extracts
its shape parameters, and tests each via miopen_conv_fix in a subprocess.
If the subprocess segfaults (GPU page fault from a broken MIOpen kernel),
we log the shape. If it succeeds, we log the solver ID and timing.

Usage (inside Docker with GPU):
    /app/.venv/bin/python tools/find_broken_shapes.py
"""

import json
import multiprocessing as mp
import os
import sys
import time

os.environ["EINX_FILTER_TRACEBACK"] = "false"


def extract_conv_shapes():
    """Extract all unique conv configs from the DAC model."""
    import torch.nn as nn
    from hydra import compose, initialize
    from hydra.utils import instantiate
    import hydra

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../fish_speech/configs"):
        cfg = compose(config_name="modded_dac_vq")
    model = instantiate(cfg)

    shapes = []
    seen = set()

    for name, m in model.named_modules():
        if isinstance(m, nn.ConvTranspose1d):
            key = (
                "transpose",
                m.in_channels,
                m.out_channels,
                m.kernel_size[0],
                m.stride[0],
                m.padding[0],
                m.output_padding[0] if hasattr(m, "output_padding") else 0,
                m.dilation[0],
                m.groups,
            )
            if key not in seen:
                seen.add(key)
                shapes.append({"name": name, "type": "transpose", **_conv_params(m)})
        elif isinstance(m, nn.Conv1d):
            key = (
                "conv",
                m.in_channels,
                m.out_channels,
                m.kernel_size[0],
                m.stride[0],
                m.padding[0],
                m.dilation[0],
                m.groups,
            )
            if key not in seen:
                seen.add(key)
                shapes.append({"name": name, "type": "conv", **_conv_params(m)})

    return shapes


def _conv_params(m):
    d = {
        "in_channels": m.in_channels,
        "out_channels": m.out_channels,
        "kernel_size": m.kernel_size[0],
        "stride": m.stride[0],
        "padding": m.padding[0],
        "dilation": m.dilation[0],
        "groups": m.groups,
    }
    if hasattr(m, "output_padding"):
        d["output_padding"] = m.output_padding[0]
    return d


def test_shape(shape, input_length, dtype_str, result_queue):
    """Run a single conv shape through miopen_conv_fix. Runs in subprocess."""
    import torch
    import miopen_conv_fix

    device = "cuda"
    dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float32

    try:
        x = torch.randn(1, shape["in_channels"], input_length, device=device, dtype=dtype)
        w = torch.randn(
            shape["out_channels"] if shape["type"] == "conv" else shape["in_channels"],
            (shape["in_channels"] if shape["type"] == "conv" else shape["out_channels"]) // shape["groups"],
            shape["kernel_size"],
            device=device,
            dtype=dtype,
        )

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        if shape["type"] == "conv":
            out = miopen_conv_fix.conv1d(
                x, w, stride=shape["stride"], padding=shape["padding"],
                dilation=shape["dilation"], groups=shape["groups"],
            )
        else:
            out = miopen_conv_fix.conv_transpose1d(
                x, w, stride=shape["stride"], padding=shape["padding"],
                output_padding=shape.get("output_padding", 0),
                groups=shape["groups"], dilation=shape["dilation"],
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        result_queue.put({
            "status": "ok",
            "shape": shape["name"],
            "time_ms": elapsed * 1000,
            "output_shape": list(out.shape),
        })

    except Exception as e:
        result_queue.put({
            "status": "error",
            "shape": shape["name"],
            "error": str(e),
        })


def main():
    print("Extracting conv shapes from DAC model...")
    shapes = extract_conv_shapes()
    print(f"Found {len(shapes)} unique conv shapes\n")

    # Test input lengths matching the decode pipeline:
    # quantizer: short (128-2048), decoder: grows (2048 -> 1M)
    test_lengths = [64, 512, 2048]

    crashed = []
    passed = []

    for dtype_str in ["bf16"]:
        for length in test_lengths:
            print(f"\n{'='*60}")
            print(f"Testing {len(shapes)} shapes at L={length}, dtype={dtype_str}")
            print(f"{'='*60}")

            for shape in shapes:
                label = f"{shape['type']}({shape['in_channels']}->{shape['out_channels']}, k={shape['kernel_size']}, s={shape['stride']}, g={shape['groups']}) L={length}"

                result_queue = mp.Queue()
                p = mp.Process(target=test_shape, args=(shape, length, dtype_str, result_queue))
                p.start()
                p.join(timeout=30)

                if p.exitcode is None:
                    # Timeout
                    p.kill()
                    p.join()
                    print(f"  TIMEOUT  {label}")
                    crashed.append({"label": label, "reason": "timeout", **shape})
                elif p.exitcode != 0:
                    # Crash (segfault = -11, other signals)
                    print(f"  CRASHED  {label}  (exit={p.exitcode})")
                    crashed.append({"label": label, "reason": f"exit={p.exitcode}", **shape})
                else:
                    try:
                        result = result_queue.get_nowait()
                        if result["status"] == "ok":
                            print(f"  OK       {label}  ({result['time_ms']:.1f}ms)")
                            passed.append(label)
                        else:
                            print(f"  ERROR    {label}  ({result['error']})")
                            crashed.append({"label": label, "reason": result["error"], **shape})
                    except Exception:
                        print(f"  OK       {label}  (no result)")
                        passed.append(label)

    print(f"\n{'='*60}")
    print(f"RESULTS: {len(passed)} passed, {len(crashed)} crashed/failed")
    print(f"{'='*60}")

    if crashed:
        print("\nCrashed shapes:")
        for c in crashed:
            print(f"  {c['label']}  reason={c['reason']}")

        # Save to file for blacklist generation
        with open("/tmp/crashed_shapes.json", "w") as f:
            json.dump(crashed, f, indent=2, default=str)
        print(f"\nSaved to /tmp/crashed_shapes.json")
    else:
        print("\nAll shapes passed!")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
