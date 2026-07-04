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
sorted1k = np.sort(a1k)
select_idx = np.array([0, 10, 20, 30, 40])
set_a = np.array([1, 2, 3, 4, 5, 8])
set_b = np.array([3, 4, 5, 6, 7])
interp_x = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)
interp_xp = np.array([0, 1, 2, 3, 4], dtype=np.float64)
interp_fp = np.array([0, 10, 20, 30, 40], dtype=np.float64)
grad_signal = np.array([np.sin(i / 20) for i in range(4096)], dtype=np.float64)

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

# ── Additional v1.0.0 ndarray coverage ──────────────────

run(suite, "fft", "4K", lambda: np.fft.fft(grad_signal))
run(suite, "einsum", "100x100", lambda: np.einsum("ij,jk->ik", mat100, mat100))
run(suite, "meshgrid", "100x100", lambda: np.meshgrid(np.arange(100), np.arange(100)))
run(suite, "index_select", "100x100", lambda: np.take(mat100, select_idx, axis=0))
run(suite, "insert", "1K", lambda: np.insert(a1k, 10, 99))
run(suite, "delete_", "1K", lambda: np.delete(a1k, [0, 10, 20]))
run(suite, "searchsorted", "1K", lambda: np.searchsorted(sorted1k, np.array([-1, 0, 1])))
run(suite, "digitize", "1K", lambda: np.digitize(a1k, np.array([-1, 0, 1])))
run(suite, "interp", "4 pts", lambda: np.interp(interp_x, interp_xp, interp_fp))
run(suite, "gradient", "4K", lambda: np.gradient(grad_signal))
run(suite, "isin", "1K", lambda: np.isin(a1k, np.array([-1, 0, 1])))
run(suite, "union1d", "6+5", lambda: np.union1d(set_a, set_b))
run(suite, "intersect1d", "6+5", lambda: np.intersect1d(set_a, set_b))
run(suite, "setdiff1d", "6-5", lambda: np.setdiff1d(set_a, set_b))

# ── Extended coverage (v1.1 benchmark expansion) ────────

a10k = rng.randn(10000).astype(np.float32)
b10k = rng.randn(10000).astype(np.float32)
pos10k = np.abs(a10k)
posb1k = np.abs(b1k) + 1
posb100k = np.abs(b100k) + 1
unit1k = np.clip(a1k, -0.99, 0.99)
unit100k = np.clip(a100k, -0.99, 0.99)
ge1_1k = pos1k + 1
ge1_100k = pos100k + 1

run(suite, "add", "10K", lambda: a10k + b10k)
run(suite, "sub", "10K", lambda: a10k - b10k)
run(suite, "sub", "1M", lambda: a1m - b1m)
run(suite, "mul", "10K", lambda: a10k * b10k)
run(suite, "mul", "1M", lambda: a1m * b1m)
run(suite, "div", "10K", lambda: a10k / b10k)
run(suite, "div", "1M", lambda: a1m / b1m)
run(suite, "neg", "10K", lambda: -a10k)

run(suite, "maximum", "1K", lambda: np.maximum(a1k, b1k))
run(suite, "maximum", "100K", lambda: np.maximum(a100k, b100k))
run(suite, "minimum", "1K", lambda: np.minimum(a1k, b1k))
run(suite, "minimum", "100K", lambda: np.minimum(a100k, b100k))
run(suite, "mod", "1K", lambda: np.mod(a1k, posb1k))
run(suite, "mod", "100K", lambda: np.mod(a100k, posb100k))
run(suite, "floorDiv", "1K", lambda: np.floor_divide(a1k, posb1k))
run(suite, "atan2", "1K", lambda: np.arctan2(a1k, b1k))
run(suite, "atan2", "100K", lambda: np.arctan2(a100k, b100k))
run(suite, "logicalXor", "1K", lambda: np.logical_xor(a1k > 0, b1k < 0))

run(suite, "floor", "1K", lambda: np.floor(a1k))
run(suite, "floor", "100K", lambda: np.floor(a100k))
run(suite, "ceil", "1K", lambda: np.ceil(a1k))
run(suite, "ceil", "100K", lambda: np.ceil(a100k))
run(suite, "round", "1K", lambda: np.round(a1k))
run(suite, "round", "100K", lambda: np.round(a100k))
run(suite, "trunc", "1K", lambda: np.trunc(a1k))
run(suite, "trunc", "100K", lambda: np.trunc(a100k))
run(suite, "square", "1K", lambda: np.square(a1k))
run(suite, "square", "100K", lambda: np.square(a100k))
run(suite, "reciprocal", "1K", lambda: np.reciprocal(ge1_1k))
run(suite, "reciprocal", "100K", lambda: np.reciprocal(ge1_100k))
run(suite, "cbrt", "1K", lambda: np.cbrt(a1k))
run(suite, "cbrt", "100K", lambda: np.cbrt(a100k))
run(suite, "exp2", "1K", lambda: np.exp2(a1k))
run(suite, "exp2", "100K", lambda: np.exp2(a100k))
run(suite, "expm1", "1K", lambda: np.expm1(a1k))
run(suite, "expm1", "100K", lambda: np.expm1(a100k))
run(suite, "log1p", "1K", lambda: np.log1p(pos1k))
run(suite, "log1p", "100K", lambda: np.log1p(pos100k))
run(suite, "log2", "1K", lambda: np.log2(pos1k))
run(suite, "log2", "100K", lambda: np.log2(pos100k))
run(suite, "log10", "1K", lambda: np.log10(pos1k))
run(suite, "log10", "100K", lambda: np.log10(pos100k))
run(suite, "rsqrt", "1K", lambda: 1.0 / np.sqrt(pos1k))
run(suite, "rsqrt", "100K", lambda: 1.0 / np.sqrt(pos100k))
run(suite, "sqrt", "10K", lambda: np.sqrt(pos10k))
run(suite, "exp", "10K", lambda: np.exp(a10k))
run(suite, "log", "10K", lambda: np.log(pos10k))

run(suite, "sinh", "1K", lambda: np.sinh(a1k))
run(suite, "sinh", "100K", lambda: np.sinh(a100k))
run(suite, "cosh", "1K", lambda: np.cosh(a1k))
run(suite, "cosh", "100K", lambda: np.cosh(a100k))
run(suite, "asin", "1K", lambda: np.arcsin(unit1k))
run(suite, "asin", "100K", lambda: np.arcsin(unit100k))
run(suite, "acos", "1K", lambda: np.arccos(unit1k))
run(suite, "atan", "1K", lambda: np.arctan(a1k))
run(suite, "atan", "100K", lambda: np.arctan(a100k))
run(suite, "asinh", "1K", lambda: np.arcsinh(a1k))
run(suite, "acosh", "1K", lambda: np.arccosh(ge1_1k))
run(suite, "atanh", "1K", lambda: np.arctanh(unit1k))

run(suite, "nanmean", "1K", lambda: np.nanmean(a1k))
run(suite, "nanmean", "100K", lambda: np.nanmean(a100k))
run(suite, "nansum", "1K", lambda: np.nansum(a1k))
run(suite, "nansum", "100K", lambda: np.nansum(a100k))
run(suite, "nanmax", "100K", lambda: np.nanmax(a100k))
run(suite, "nanmin", "100K", lambda: np.nanmin(a100k))
run(suite, "nanstd", "1K", lambda: np.nanstd(a1k))
run(suite, "all", "100K", lambda: np.all(a100k > 0))
run(suite, "any", "100K", lambda: np.any(a100k > 0))
run(suite, "sum", "10K", lambda: np.sum(a10k))
run(suite, "mean", "10K", lambda: np.mean(a10k))
run(suite, "max", "10K", lambda: np.max(a10k))
run(suite, "min", "10K", lambda: np.min(a10k))
run(suite, "std", "10K", lambda: np.std(a10k))
run(suite, "variance", "10K", lambda: np.var(a10k))
run(suite, "cumsum", "10K", lambda: np.cumsum(a10k))

run(suite, "roll", "100K", lambda: np.roll(a100k, 100))
run(suite, "flip", "100K", lambda: np.flip(a100k))
run(suite, "fliplr", "100x100", lambda: np.fliplr(mat100))
run(suite, "flipud", "100x100", lambda: np.flipud(mat100))
run(suite, "rot90", "100x100", lambda: np.rot90(mat100))
run(suite, "repeat", "1K→2x", lambda: np.repeat(a1k, 2))
run(suite, "tile", "1K→2x", lambda: np.tile(a1k, 2))
run(suite, "diff", "100K", lambda: np.diff(a100k))
run(suite, "diag", "100x100", lambda: np.diag(mat100))
run(suite, "diagonal", "100x100", lambda: np.diagonal(mat100))
run(suite, "tril", "100x100", lambda: np.tril(mat100))
run(suite, "triu", "100x100", lambda: np.triu(mat100))
run(suite, "moveaxis", "100x100", lambda: np.moveaxis(mat100, 0, 1))
run(suite, "swapaxes", "100x100", lambda: np.swapaxes(mat100, 0, 1))
run(suite, "hstack", "2×100K", lambda: np.hstack([a100k, b100k]))
run(suite, "vstack", "2×1K", lambda: np.vstack([a1k, b1k]))
run(suite, "column_stack", "2×1K", lambda: np.column_stack([a1k, b1k]))
run(suite, "broadcast_to", "1K→100x1K", lambda: np.broadcast_to(a1k, (100, 1000)))
run(suite, "where", "100K", lambda: np.where(a100k > 0, a100k, b100k))
run(suite, "unique", "1K", lambda: np.unique(a1k))
run(suite, "trapz", "100K", lambda: np.trapezoid(a100k))
run(suite, "cross", "3x3", lambda: np.cross(np.array([1.0, 0, 0]), np.array([0, 1.0, 0])))
run(suite, "pad", "1K→pad2", lambda: np.pad(a1k, (2, 2)))

run(suite, "isnan", "100K", lambda: np.isnan(a100k))
run(suite, "isinf", "100K", lambda: np.isinf(a100k))
run(suite, "isfinite", "100K", lambda: np.isfinite(a100k))

int_pool = np.arange(0, 1000, dtype=np.int32)
run(suite, "bincount", "1K", lambda: np.bincount(int_pool))

footer(suite, "numpy-ndarray.json")
