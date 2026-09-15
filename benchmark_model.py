"""Benchmark RITnet compute across candidate eye-crop resolutions.

This is a PyTorch development benchmark, not an iPhone/Core ML claim. Use it to choose which
resolutions deserve export and physical-device testing.

Usage: ``python benchmark_model.py --repeats 20``
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch

from ritnet_segment import DEVICE, model


def synchronize():
    if DEVICE == "mps": torch.mps.synchronize()
    elif DEVICE.startswith("cuda"): torch.cuda.synchronize()


def convolution_macs(net, sample):
    total = 0
    handles = []
    def count(module, _inputs, output):
        nonlocal total
        kh, kw = module.kernel_size
        total += output.numel() * (module.in_channels // module.groups) * kh * kw
    for module in net.modules():
        if isinstance(module, torch.nn.Conv2d): handles.append(module.register_forward_hook(count))
    with torch.no_grad(): net(sample)
    for handle in handles: handle.remove()
    return total


def benchmark(width, height, repeats):
    sample = torch.zeros(1, 1, height, width, device=DEVICE)
    macs = convolution_macs(model, sample)
    with torch.no_grad():
        for _ in range(3): model(sample)
        synchronize(); durations = []
        for _ in range(repeats):
            start = time.perf_counter(); model(sample); synchronize()
            durations.append((time.perf_counter() - start) * 1000)
    median_ms = statistics.median(durations)
    return macs / 1e9, median_ms, 1000 / median_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    print(f"device={DEVICE} params={sum(p.numel() for p in model.parameters()):,}")
    print(f"{'input':>10} {'conv GMAC':>10} {'median ms':>11} {'fps':>8}")
    for width, height in ((400, 640), (320, 512), (256, 416)):
        macs, milliseconds, fps = benchmark(width, height, args.repeats)
        print(f"{width}x{height: <5} {macs:10.3f} {milliseconds:11.2f} {fps:8.2f}")


if __name__ == "__main__":
    main()
