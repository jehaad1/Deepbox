"""
Shared benchmark utilities for the Python benchmark suite.

Schema v2 mirrors the TypeScript runner:
- adaptive batching for very fast operations
- richer statistics (median, p95, relative std)
- consistent environment + methodology metadata
- comparable/deepbox-only case tracking
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Callable


BENCHMARK_SCHEMA_VERSION = 2
DEFAULT_WARMUP = 5
DEFAULT_MIN_SAMPLES = 20
DEFAULT_MAX_SAMPLES = 60
DEFAULT_TARGET_SAMPLE_MS = 5.0
DEFAULT_TARGET_TOTAL_MS = 250.0
MAX_BATCH_SIZE = 10_000

_blackhole = 0


def _consume(value: Any) -> None:
    global _blackhole
    if isinstance(value, bool):
        _blackhole ^= int(value)
        return
    if isinstance(value, int):
        _blackhole ^= value & 0xFFFF
        return
    if isinstance(value, float):
        if value == value and value not in (float("inf"), float("-inf")):
            _blackhole ^= int(value * 1_000_000) & 0xFFFF
        return
    if isinstance(value, str):
        _blackhole ^= len(value)
        return
    if isinstance(value, (list, tuple, set, dict)):
        _blackhole ^= len(value)
        return


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return sorted_values[lower]
    weight = pos - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * weight


def _round(value: float, digits: int = 4) -> float:
    return round(value, digits)


def _stats(samples: list[float]) -> dict[str, float]:
    sorted_values = sorted(samples)
    mean = sum(samples) / len(samples)
    std = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    return {
        "mean": mean,
        "median": _percentile(sorted_values, 0.5),
        "std": std,
        "min": sorted_values[0],
        "p95": _percentile(sorted_values, 0.95),
        "max": sorted_values[-1],
    }


def _scope_of(comparable: bool) -> str:
    return "match" if comparable else "local"


def _log_result(
    operation: str,
    size: str,
    stats: dict[str, float],
    relative_std: float,
    sample_count: int,
    batch_size: int,
    comparable: bool,
) -> None:
    op = operation.ljust(34)
    sz = size.ljust(14)
    median = f"{stats['median']:.3f} ms".rjust(12)
    p95 = f"{stats['p95']:.3f} ms".rjust(12)
    cv = f"{relative_std:.1f}%".rjust(8)
    sample_info = f"{sample_count}x{batch_size}".rjust(9)
    print(f"  {op} {sz} {median} {p95} {cv} {sample_info}  {_scope_of(comparable)}")


def _validate_suite(suite: dict[str, Any]) -> None:
    seen_ids: set[str] = set()
    seen_cases: set[str] = set()
    for result in suite["results"]:
        result_id = str(result["id"])
        if result_id in seen_ids:
            raise ValueError(f"Duplicate benchmark id detected: {result_id}")
        seen_ids.add(result_id)

        case_key = f"{result['operation']}::{result['size']}"
        if case_key in seen_cases:
            raise ValueError(
                f"Duplicate benchmark operation/size detected in suite '{suite['benchmark']}': {case_key}"
            )
        seen_cases.add(case_key)


def _make_id(benchmark: str, operation: str, size: str) -> str:
    raw = f"{benchmark}-{operation}-{size}".lower()
    slug = "".join(ch if ch.isalnum() else "-" for ch in raw)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or f"{benchmark}-case"


def create_suite(name: str, library: str) -> dict[str, Any]:
    cpu = platform.processor() or platform.machine() or "unknown"
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark": name,
        "platform": library,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
        "system": {
            "runtime": f"Python {platform.python_implementation()}",
            "version": platform.python_version(),
            "os": platform.system().lower(),
            "release": platform.release(),
            "arch": platform.machine(),
            "cpu": cpu,
        },
        "methodology": {
            "timer": "time.perf_counter_ns",
            "compare_metric": "median_ms",
            "default_warmup": DEFAULT_WARMUP,
            "default_min_samples": DEFAULT_MIN_SAMPLES,
            "default_max_samples": DEFAULT_MAX_SAMPLES,
            "target_sample_ms": DEFAULT_TARGET_SAMPLE_MS,
            "target_total_ms": DEFAULT_TARGET_TOTAL_MS,
            "adaptive_batching": True,
            "gc_between_samples": False,
        },
        "results": [],
    }


def run(
    suite: dict[str, Any],
    operation: str,
    size: str,
    fn: Callable[[], Any],
    iterations: int | None = None,
    warmup: int = DEFAULT_WARMUP,
    comparable: bool = True,
    tags: list[str] | None = None,
    case_id: str | None = None,
) -> None:
    for _ in range(warmup):
        _consume(fn())

    calibration_start = time.perf_counter_ns()
    _consume(fn())
    calibration_ms = max((time.perf_counter_ns() - calibration_start) / 1_000_000, 0.0001)
    batch_size = max(1, min(MAX_BATCH_SIZE, int((DEFAULT_TARGET_SAMPLE_MS / calibration_ms) + 0.9999)))

    fixed_samples = iterations is not None
    min_samples = iterations or DEFAULT_MIN_SAMPLES
    max_samples = iterations or DEFAULT_MAX_SAMPLES

    samples: list[float] = []
    total_elapsed_ms = 0.0

    while len(samples) < max_samples:
        start = time.perf_counter_ns()
        for _ in range(batch_size):
            _consume(fn())
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000 / batch_size
        samples.append(elapsed_ms)
        total_elapsed_ms += elapsed_ms * batch_size

        if fixed_samples and len(samples) >= (iterations or 0):
            break
        if not fixed_samples and len(samples) >= min_samples and total_elapsed_ms >= DEFAULT_TARGET_TOTAL_MS:
            break

    stats = _stats(samples)
    ops = 1000 / stats["median"] if stats["median"] > 0 else float("inf")
    relative_std = (stats["std"] / stats["mean"] * 100) if stats["mean"] > 0 else 0.0

    suite["results"].append(
        {
            "id": case_id or _make_id(suite["benchmark"], operation, size),
            "operation": operation,
            "size": size,
            "comparable": comparable,
            "scope": _scope_of(comparable),
            "tags": tags or [],
            "warmup": warmup,
            "samples": len(samples),
            "batch_size": batch_size,
            "total_invocations": batch_size * (warmup + len(samples)),
            "mean_ms": _round(stats["mean"]),
            "median_ms": _round(stats["median"]),
            "std_ms": _round(stats["std"]),
            "min_ms": _round(stats["min"]),
            "p95_ms": _round(stats["p95"]),
            "max_ms": _round(stats["max"]),
            "ops_per_sec": _round(ops, 2),
            "relative_std_pct": _round(relative_std, 2),
        }
    )

    _log_result(operation, size, stats, relative_std, len(samples), batch_size, comparable)


def header(title: str, library: str) -> None:
    print("=" * 116)
    print(f"  {title}")
    print(f"  Platform: {library} | Python {platform.python_version()} | Compare metric: median_ms")
    print(
        f"  Sampling: adaptive batching, {DEFAULT_MIN_SAMPLES}-{DEFAULT_MAX_SAMPLES} samples, target {int(DEFAULT_TARGET_TOTAL_MS)} ms/case"
    )
    print("=" * 116)
    print(
        f"  {'Operation'.ljust(34)} {'Size'.ljust(14)} {'Median'.rjust(12)} {'P95'.rjust(12)} {'CV'.rjust(8)} {'Samples'.rjust(9)}  Scope"
    )
    print("-" * 116)


def footer(suite: dict[str, Any], output_file: str) -> None:
    _validate_suite(suite)
    print("-" * 116)
    comparable_count = sum(1 for result in suite["results"] if result["comparable"])
    local_count = len(suite["results"]) - comparable_count
    print(f"  Total: {len(suite['results'])} benchmarks ({comparable_count} comparable, {local_count} local)")
    print(f"  Blackhole: {_blackhole}")
    print("=" * 116)

    out_path = Path("benchmarks") / "results" / output_file
    os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(suite, handle, indent=2)
    print(f"  Saved: {out_path}\n")
