"""
Benchmark 06 — NDArray / Tensor Operations
NumPy
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from utils import run, create_suite, header, footer

suite = create_suite("ndarray", "NumPy")
header("Benchmark 06 — NDArray / Tensor Operations", "NumPy")

rng = np.random.RandomState(42)

# ── Creation ─────────────────────────────────────────────

run(suite, "zeros", "1K", lambda: np.zeros(1000))
run(suite, "zeros", "100K", lambda: np.zeros(100000))
run(suite, "zeros", "1M", lambda: np.zeros(1000000))
run(suite, "ones", "1K", lambda: np.ones(1000))
run(suite, "ones", "100K", lambda: np.ones(100000))
run(suite, "full(42)", "1K", lambda: np.full(1000, 42.0))
run(suite, "full(42)", "100K", lambda: np.full(100000, 42.0))
run(suite, "empty", "1K", lambda: np.empty(1000))
run(suite, "empty", "100K", lambda: np.empty(100000))
run(suite, "arange", "1K", lambda: np.arange(0, 1000, dtype=np.float64))
run(suite, "arange", "100K", lambda: np.arange(0, 100000, dtype=np.float64))
run(suite, "linspace", "1K", lambda: np.linspace(0, 1, 1000))
run(suite, "linspace", "100K", lambda: np.linspace(0, 1, 100000))
run(suite, "eye", "100x100", lambda: np.eye(100))
run(suite, "eye", "500x500", lambda: np.eye(500))
run(suite, "randn", "1K", lambda: rng.randn(1000))
run(suite, "randn", "100K", lambda: rng.randn(100000))

# ── Element-wise Arithmetic ─────────────────────────────

a1k = rng.randn(1000).astype(np.float32)
b1k = rng.randn(1000).astype(np.float32)
a100k = rng.randn(100000).astype(np.float32)
b100k = rng.randn(100000).astype(np.float32)
a1m = rng.randn(1000000).astype(np.float32)
b1m = rng.randn(1000000).astype(np.float32)

run(suite, "add", "1K", lambda: np.add(a1k, b1k))
run(suite, "add", "100K", lambda: np.add(a100k, b100k))
run(suite, "add", "1M", lambda: np.add(a1m, b1m))
run(suite, "sub", "1K", lambda: np.subtract(a1k, b1k))
run(suite, "sub", "100K", lambda: np.subtract(a100k, b100k))
run(suite, "mul", "1K", lambda: np.multiply(a1k, b1k))
run(suite, "mul", "100K", lambda: np.multiply(a100k, b100k))
run(suite, "div", "1K", lambda: np.divide(a1k, b1k))
run(suite, "div", "100K", lambda: np.divide(a100k, b100k))
run(suite, "neg", "1K", lambda: np.negative(a1k))
run(suite, "neg", "100K", lambda: np.negative(a100k))
run(suite, "pow (x²)", "1K", lambda: np.power(a1k, 2))
run(suite, "pow (x²)", "100K", lambda: np.power(a100k, 2))

# ── Math Functions ──────────────────────────────────────

pos1k = np.abs(a1k)
pos100k = np.abs(a100k)

run(suite, "sqrt", "1K", lambda: np.sqrt(pos1k))
run(suite, "sqrt", "100K", lambda: np.sqrt(pos100k))
run(suite, "exp", "1K", lambda: np.exp(a1k))
run(suite, "exp", "100K", lambda: np.exp(a100k))
run(suite, "log", "1K", lambda: np.log(pos1k))
run(suite, "log", "100K", lambda: np.log(pos100k))
run(suite, "abs", "1K", lambda: np.abs(a1k))
run(suite, "abs", "100K", lambda: np.abs(a100k))
run(suite, "sin", "1K", lambda: np.sin(a1k))
run(suite, "sin", "100K", lambda: np.sin(a100k))
run(suite, "cos", "1K", lambda: np.cos(a1k))
run(suite, "cos", "100K", lambda: np.cos(a100k))
run(suite, "clip", "1K", lambda: np.clip(a1k, -1, 1))
run(suite, "clip", "100K", lambda: np.clip(a100k, -1, 1))
run(suite, "sign", "1K", lambda: np.sign(a1k))
run(suite, "sign", "100K", lambda: np.sign(a100k))

# ── Reductions ──────────────────────────────────────────

run(suite, "sum", "1K", lambda: np.sum(a1k))
run(suite, "sum", "100K", lambda: np.sum(a100k))
run(suite, "mean", "1K", lambda: np.mean(a1k))
run(suite, "mean", "100K", lambda: np.mean(a100k))
run(suite, "max", "1K", lambda: np.max(a1k))
run(suite, "max", "100K", lambda: np.max(a100k))
run(suite, "min", "1K", lambda: np.min(a1k))
run(suite, "min", "100K", lambda: np.min(a100k))
run(suite, "variance", "1K", lambda: np.var(a1k))
run(suite, "variance", "100K", lambda: np.var(a100k))
run(suite, "std", "1K", lambda: np.std(a1k))
run(suite, "std", "100K", lambda: np.std(a100k))
run(suite, "prod", "1K", lambda: np.prod(a1k))
run(suite, "median", "1K", lambda: np.median(a1k))
run(suite, "cumsum", "1K", lambda: np.cumsum(a1k))
run(suite, "cumsum", "100K", lambda: np.cumsum(a100k))
run(suite, "cumprod", "1K", lambda: np.cumprod(a1k))

# ── Sorting ─────────────────────────────────────────────

run(suite, "sort", "1K", lambda: np.sort(a1k))
run(suite, "sort", "100K", lambda: np.sort(a100k))
run(suite, "argsort", "1K", lambda: np.argsort(a1k))
run(suite, "argsort", "100K", lambda: np.argsort(a100k))

# ── Shape Operations ────────────────────────────────────

mat100 = rng.randn(100, 100).astype(np.float32)
mat500 = rng.randn(500, 500).astype(np.float32)
flat10k = rng.randn(10000).astype(np.float32)

run(suite, "reshape", "10K→100x100", lambda: flat10k.reshape(100, 100))
run(suite, "flatten", "100x100", lambda: mat100.flatten())
run(suite, "transpose", "100x100", lambda: mat100.T.copy())
run(suite, "transpose", "500x500", lambda: mat500.T.copy())
run(suite, "squeeze", "[1,100,1]", lambda: np.squeeze(rng.randn(1, 100, 1)))
run(suite, "unsqueeze", "1K→1x1K", lambda: np.expand_dims(a1k, 0))

# ── Manipulation ────────────────────────────────────────

run(suite, "concatenate", "2×1K", lambda: np.concatenate([a1k, b1k]))
run(suite, "concatenate", "2×100K", lambda: np.concatenate([a100k, b100k]))
run(suite, "stack", "2×1K", lambda: np.stack([a1k, b1k]))
run(suite, "stack", "2×100K", lambda: np.stack([a100k, b100k]))
run(suite, "slice", "[0:500] of 1K", lambda: a1k[0:500].copy())

# ── Comparison / Logical ────────────────────────────────

run(suite, "equal", "1K", lambda: np.equal(a1k, b1k))
run(suite, "greater", "1K", lambda: np.greater(a1k, b1k))
run(suite, "less", "1K", lambda: np.less(a1k, b1k))
mask1k = a1k > 0
mask1k2 = b1k < 0
run(suite, "logicalAnd", "1K", lambda: np.logical_and(mask1k, mask1k2))
run(suite, "logicalOr", "1K", lambda: np.logical_or(mask1k, mask1k2))
run(suite, "logicalNot", "1K", lambda: np.logical_not(mask1k))

# ── Activations ─────────────────────────────────────────

run(suite, "relu", "1K", lambda: np.maximum(a1k, 0))
run(suite, "relu", "100K", lambda: np.maximum(a100k, 0))
run(suite, "sigmoid", "1K", lambda: 1 / (1 + np.exp(-a1k)))
run(suite, "sigmoid", "100K", lambda: 1 / (1 + np.exp(-a100k)))
run(suite, "tanh", "1K", lambda: np.tanh(a1k))
run(suite, "tanh", "100K", lambda: np.tanh(a100k))

def np_softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

run(suite, "softmax", "1K", lambda: np_softmax(a1k))

# ── Matmul ──────────────────────────────────────────────

m50 = rng.randn(50, 50).astype(np.float64)
m100d = rng.randn(100, 100).astype(np.float64)
m200 = rng.randn(200, 200).astype(np.float64)

run(suite, "matmul", "50x50", lambda: m50 @ m50)
run(suite, "matmul", "100x100", lambda: m100d @ m100d)
run(suite, "matmul", "200x200", lambda: m200 @ m200, iterations=10)

footer(suite, "numpy-ndarray.json")
