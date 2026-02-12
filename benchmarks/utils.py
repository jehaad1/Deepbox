"""
Shared benchmark utilities for the Python benchmark suite.
"""

import json
import os
import platform
import time


def create_suite(name, library):
    return {
        "benchmark": name,
        "platform": library,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
        "system": {
            "runtime": f"Python {platform.python_implementation()}",
            "version": platform.python_version(),
        },
        "results": [],
    }


def run(suite, operation, size, fn, iterations=20, warmup=5):
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    ops = 1000 / mean if mean > 0 else float("inf")

    suite["results"].append({
        "operation": operation,
        "size": size,
        "iterations": iterations,
        "mean_ms": round(mean, 4),
        "std_ms": round(std, 4),
        "min_ms": round(min(times), 4),
        "max_ms": round(max(times), 4),
        "ops_per_sec": round(ops, 2),
    })

    op = operation.ljust(44)
    sz = size.ljust(12)
    m = f"{mean:.3f} ms".rjust(10)
    s = f"{std:.3f} ms".rjust(10)
    o = f"{int(ops)}".rjust(10)
    print(f"  {op} {sz} {m}  ± {s}  ({o} ops/s)")


def header(title, library):
    v = platform.python_version()
    print("=" * 100)
    print(f"  {title}")
    print(f"  Platform: {library} | Python {v}")
    print("=" * 100)
    h_op = "Operation".ljust(44)
    h_sz = "Size".ljust(12)
    h_m = "Mean".rjust(10)
    h_s = "Std Dev".rjust(10)
    h_t = "Throughput".rjust(12)
    print(f"  {h_op} {h_sz} {h_m}     {h_s}  {h_t}")
    print("-" * 100)


def footer(suite, output_file):
    print("-" * 100)
    print(f"  Total: {len(suite['results'])} benchmarks")
    print("=" * 100)

    out_path = os.path.join("benchmarks", "results", output_file)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(suite, f, indent=2)
    print(f"  ✓ Saved → {out_path}\n")
