/**
 * Benchmark 06: NDArray / Tensor Operations
 * Deepbox vs NumPy
 */

import {
  abs,
  acos,
  acosh,
  add,
  all,
  any,
  arange,
  argsort,
  asin,
  asinh,
  atan,
  atan2,
  atanh,
  bincount,
  broadcast_to,
  cbrt,
  ceil,
  clip,
  column_stack,
  concatenate,
  copy,
  cos,
  cosh,
  cross,
  cumprod,
  cumsum,
  delete_,
  diag,
  diagonal,
  diff,
  digitize,
  div,
  dot,
  einsum,
  empty,
  equal,
  exp,
  exp2,
  expm1,
  eye,
  fft,
  flatten,
  flip,
  fliplr,
  flipud,
  floor,
  floorDiv,
  full,
  gradient,
  greater,
  hstack,
  index_select,
  insert,
  interp,
  intersect1d,
  isfinite,
  isin,
  isinf,
  isnan,
  less,
  linspace,
  log,
  log1p,
  log2,
  log10,
  logicalAnd,
  logicalNot,
  logicalOr,
  logicalXor,
  max,
  maximum,
  mean,
  median,
  meshgrid,
  min,
  minimum,
  mod,
  moveaxis,
  mul,
  nanmax,
  nanmean,
  nanmin,
  nanstd,
  nansum,
  neg,
  ones,
  pad,
  pow,
  prod,
  randn,
  reciprocal,
  relu,
  repeat,
  reshape,
  roll,
  rot90,
  round,
  rsqrt,
  searchsorted,
  setdiff1d,
  sigmoid,
  sign,
  sin,
  sinh,
  slice,
  softmax,
  sort,
  sqrt,
  square,
  squeeze,
  stack,
  std,
  sub,
  sum,
  swapaxes,
  tanh,
  tensor,
  tile,
  transpose,
  trapz,
  tril,
  triu,
  trunc,
  union1d,
  unique,
  unsqueeze,
  variance,
  vstack,
  where,
  zeros,
} from "deepbox/ndarray";
import { createSuite, footer, header, run } from "../utils";

const suite = createSuite("ndarray");
header("Benchmark 06: NDArray / Tensor Operations");

// ── Creation ─────────────────────────────────────────────

run(suite, "zeros", "1K", () => zeros([1000]));
run(suite, "zeros", "100K", () => zeros([100000]));
run(suite, "zeros", "1M", () => zeros([1000000]));
run(suite, "ones", "1K", () => ones([1000]));
run(suite, "ones", "100K", () => ones([100000]));
run(suite, "full(42)", "1K", () => full([1000], 42));
run(suite, "full(42)", "100K", () => full([100000], 42));
run(suite, "empty", "1K", () => empty([1000]));
run(suite, "empty", "100K", () => empty([100000]));
run(suite, "arange", "1K", () => arange(0, 1000));
run(suite, "arange", "100K", () => arange(0, 100000));
run(suite, "linspace", "1K", () => linspace(0, 1, 1000));
run(suite, "linspace", "100K", () => linspace(0, 1, 100000));
run(suite, "eye", "100x100", () => eye(100));
run(suite, "eye", "500x500", () => eye(500));
run(suite, "randn", "1K", () => randn([1000]));
run(suite, "randn", "100K", () => randn([100000]));

// ── Element-wise Arithmetic ─────────────────────────────

const a1k = randn([1000]);
const b1k = randn([1000]);
const a100k = randn([100000]);
const b100k = randn([100000]);
const a1m = randn([1000000]);
const b1m = randn([1000000]);

run(suite, "add", "1K", () => add(a1k, b1k));
run(suite, "add", "100K", () => add(a100k, b100k));
run(suite, "add", "1M", () => add(a1m, b1m));
run(suite, "sub", "1K", () => sub(a1k, b1k));
run(suite, "sub", "100K", () => sub(a100k, b100k));
run(suite, "mul", "1K", () => mul(a1k, b1k));
run(suite, "mul", "100K", () => mul(a100k, b100k));
run(suite, "div", "1K", () => div(a1k, b1k));
run(suite, "div", "100K", () => div(a100k, b100k));
run(suite, "neg", "1K", () => neg(a1k));
run(suite, "neg", "100K", () => neg(a100k));
run(suite, "pow (x²)", "1K", () => pow(a1k, tensor(2)));
run(suite, "pow (x²)", "100K", () => pow(a100k, tensor(2)));

// ── Math Functions ──────────────────────────────────────

const pos1k = abs(a1k);
const pos100k = abs(a100k);

run(suite, "sqrt", "1K", () => sqrt(pos1k));
run(suite, "sqrt", "100K", () => sqrt(pos100k));
run(suite, "exp", "1K", () => exp(a1k));
run(suite, "exp", "100K", () => exp(a100k));
run(suite, "log", "1K", () => log(pos1k));
run(suite, "log", "100K", () => log(pos100k));
run(suite, "abs", "1K", () => abs(a1k));
run(suite, "abs", "100K", () => abs(a100k));
run(suite, "sin", "1K", () => sin(a1k));
run(suite, "sin", "100K", () => sin(a100k));
run(suite, "cos", "1K", () => cos(a1k));
run(suite, "cos", "100K", () => cos(a100k));
run(suite, "clip", "1K", () => clip(a1k, -1, 1));
run(suite, "clip", "100K", () => clip(a100k, -1, 1));
run(suite, "sign", "1K", () => sign(a1k));
run(suite, "sign", "100K", () => sign(a100k));

// ── Reductions ──────────────────────────────────────────

run(suite, "sum", "1K", () => sum(a1k));
run(suite, "sum", "100K", () => sum(a100k));
run(suite, "mean", "1K", () => mean(a1k));
run(suite, "mean", "100K", () => mean(a100k));
run(suite, "max", "1K", () => max(a1k));
run(suite, "max", "100K", () => max(a100k));
run(suite, "min", "1K", () => min(a1k));
run(suite, "min", "100K", () => min(a100k));
run(suite, "variance", "1K", () => variance(a1k));
run(suite, "variance", "100K", () => variance(a100k));
run(suite, "std", "1K", () => std(a1k));
run(suite, "std", "100K", () => std(a100k));
run(suite, "prod", "1K", () => prod(a1k));
run(suite, "median", "1K", () => median(a1k));
run(suite, "cumsum", "1K", () => cumsum(a1k));
run(suite, "cumsum", "100K", () => cumsum(a100k));
run(suite, "cumprod", "1K", () => cumprod(a1k));

// ── Sorting ─────────────────────────────────────────────

run(suite, "sort", "1K", () => sort(a1k));
run(suite, "sort", "100K", () => sort(a100k));
run(suite, "argsort", "1K", () => argsort(a1k));
run(suite, "argsort", "100K", () => argsort(a100k));

// ── Shape Operations ────────────────────────────────────

const mat100 = randn([100, 100]);
const mat500 = randn([500, 500]);
const flat10k = randn([10000]);
const sorted1k = sort(a1k);
const selectIdx = tensor([0, 10, 20, 30, 40]);
const setA = tensor([1, 2, 3, 4, 5, 8]);
const setB = tensor([3, 4, 5, 6, 7]);
const interpX = tensor([0.5, 1.5, 2.5, 3.5]);
const interpXp = tensor([0, 1, 2, 3, 4]);
const interpFp = tensor([0, 10, 20, 30, 40]);
const gradSignal = tensor(Array.from({ length: 4096 }, (_, i) => Math.sin(i / 20)));

// `reshape`/`squeeze`/`unsqueeze` are lazy strided views on BOTH sides (NumPy
// `.reshape`/`np.squeeze`/`np.expand_dims` also return views), so the comparison
// is symmetric and left as-is. `transpose`/`flatten`/`slice`, however, force a
// contiguous copy on the NumPy side (`.T.copy()`, `.flatten()`, `[…].copy()`).
// Deepbox's `transpose`/`flatten`/`slice` return lazy strided views, so timing
// them bare would compare a materialized copy against a pointer/stride tweak.
// We wrap those three with `copy(...)` so BOTH sides materialize identical
// contiguous buffers, an honest apples-to-apples measurement.
run(suite, "reshape", "10K→100x100", () => reshape(flat10k, [100, 100]));
run(suite, "flatten", "100x100", () => copy(flatten(mat100)));
run(suite, "transpose", "100x100", () => copy(transpose(mat100)));
run(suite, "transpose", "500x500", () => copy(transpose(mat500)));
run(suite, "squeeze", "[1,100,1]", () => squeeze(randn([1, 100, 1])));
run(suite, "unsqueeze", "1K→1x1K", () => unsqueeze(a1k, 0));

// ── Manipulation ────────────────────────────────────────

run(suite, "concatenate", "2×1K", () => concatenate([a1k, b1k]));
run(suite, "concatenate", "2×100K", () => concatenate([a100k, b100k]));
run(suite, "stack", "2×1K", () => stack([a1k, b1k]));
run(suite, "stack", "2×100K", () => stack([a100k, b100k]));
// NumPy benchmarks `a1k[0:500].copy()` (forced materialization); match it by
// copying the Deepbox strided-view slice so both allocate the 500-element buffer.
run(suite, "slice", "[0:500] of 1K", () => copy(slice(a1k, { start: 0, end: 500 })));

// ── Comparison / Logical ────────────────────────────────

run(suite, "equal", "1K", () => equal(a1k, b1k));
run(suite, "greater", "1K", () => greater(a1k, b1k));
run(suite, "less", "1K", () => less(a1k, b1k));
const mask1k = greater(a1k, zeros([1000]));
const mask1k2 = less(b1k, zeros([1000]));
run(suite, "logicalAnd", "1K", () => logicalAnd(mask1k, mask1k2));
run(suite, "logicalOr", "1K", () => logicalOr(mask1k, mask1k2));
run(suite, "logicalNot", "1K", () => logicalNot(mask1k));

// ── Activations ─────────────────────────────────────────

run(suite, "relu", "1K", () => relu(a1k));
run(suite, "relu", "100K", () => relu(a100k));
run(suite, "sigmoid", "1K", () => sigmoid(a1k));
run(suite, "sigmoid", "100K", () => sigmoid(a100k));
run(suite, "tanh", "1K", () => tanh(a1k));
run(suite, "tanh", "100K", () => tanh(a100k));
run(suite, "softmax", "1K", () => softmax(a1k));

// ── Matmul ──────────────────────────────────────────────

run(suite, "matmul", "50x50", () => dot(randn([50, 50]), randn([50, 50])));
run(suite, "matmul", "100x100", () => dot(mat100, mat100));
run(suite, "matmul", "200x200", () => dot(randn([200, 200]), randn([200, 200])), {
  iterations: 10,
});

// ── Additional v1.0.0 ndarray coverage ────────────────

run(suite, "fft", "4K", () => fft(gradSignal));
run(suite, "einsum", "100x100", () => einsum("ij,jk->ik", mat100, mat100));
run(suite, "meshgrid", "100x100", () => meshgrid(arange(0, 100), arange(0, 100)));
run(suite, "index_select", "100x100", () => index_select(mat100, 0, selectIdx));
run(suite, "insert", "1K", () => insert(a1k, 10, 99));
run(suite, "delete_", "1K", () => delete_(a1k, [0, 10, 20]));
run(suite, "searchsorted", "1K", () => searchsorted(sorted1k, tensor([-1, 0, 1])));
run(suite, "digitize", "1K", () => digitize(a1k, tensor([-1, 0, 1])));
run(suite, "interp", "4 pts", () => interp(interpX, interpXp, interpFp));
run(suite, "gradient", "4K", () => gradient(gradSignal));
run(suite, "isin", "1K", () => isin(a1k, [-1, 0, 1]));
run(suite, "union1d", "6+5", () => union1d(setA, setB));
run(suite, "intersect1d", "6+5", () => intersect1d(setA, setB));
run(suite, "setdiff1d", "6-5", () => setdiff1d(setA, setB));

// ── Extended coverage (v1.1 benchmark expansion) ────────

// More sizes for core elementwise ops.
const a10k = randn([10000]);
const b10k = randn([10000]);
const pos10k = abs(a10k);
const posb1k = add(abs(b1k), ones([1000]));
const posb100k = add(abs(b100k), ones([100000]));
const unit1k = clip(a1k, -0.99, 0.99);
const unit100k = clip(a100k, -0.99, 0.99);
const ge1_1k = add(pos1k, ones([1000]));
const ge1_100k = add(pos100k, ones([100000]));

run(suite, "add", "10K", () => add(a10k, b10k));
run(suite, "sub", "10K", () => sub(a10k, b10k));
run(suite, "sub", "1M", () => sub(a1m, b1m));
run(suite, "mul", "10K", () => mul(a10k, b10k));
run(suite, "mul", "1M", () => mul(a1m, b1m));
run(suite, "div", "10K", () => div(a10k, b10k));
run(suite, "div", "1M", () => div(a1m, b1m));
run(suite, "neg", "10K", () => neg(a10k));

// New binary ops.
run(suite, "maximum", "1K", () => maximum(a1k, b1k));
run(suite, "maximum", "100K", () => maximum(a100k, b100k));
run(suite, "minimum", "1K", () => minimum(a1k, b1k));
run(suite, "minimum", "100K", () => minimum(a100k, b100k));
run(suite, "mod", "1K", () => mod(a1k, posb1k));
run(suite, "mod", "100K", () => mod(a100k, posb100k));
run(suite, "floorDiv", "1K", () => floorDiv(a1k, posb1k));
run(suite, "atan2", "1K", () => atan2(a1k, b1k));
run(suite, "atan2", "100K", () => atan2(a100k, b100k));
run(suite, "logicalXor", "1K", () =>
  logicalXor(greater(a1k, zeros([1000])), less(b1k, zeros([1000])))
);

// New unary math.
run(suite, "floor", "1K", () => floor(a1k));
run(suite, "floor", "100K", () => floor(a100k));
run(suite, "ceil", "1K", () => ceil(a1k));
run(suite, "ceil", "100K", () => ceil(a100k));
run(suite, "round", "1K", () => round(a1k));
run(suite, "round", "100K", () => round(a100k));
run(suite, "trunc", "1K", () => trunc(a1k));
run(suite, "trunc", "100K", () => trunc(a100k));
run(suite, "square", "1K", () => square(a1k));
run(suite, "square", "100K", () => square(a100k));
run(suite, "reciprocal", "1K", () => reciprocal(ge1_1k));
run(suite, "reciprocal", "100K", () => reciprocal(ge1_100k));
run(suite, "cbrt", "1K", () => cbrt(a1k));
run(suite, "cbrt", "100K", () => cbrt(a100k));
run(suite, "exp2", "1K", () => exp2(a1k));
run(suite, "exp2", "100K", () => exp2(a100k));
run(suite, "expm1", "1K", () => expm1(a1k));
run(suite, "expm1", "100K", () => expm1(a100k));
run(suite, "log1p", "1K", () => log1p(pos1k));
run(suite, "log1p", "100K", () => log1p(pos100k));
run(suite, "log2", "1K", () => log2(pos1k));
run(suite, "log2", "100K", () => log2(pos100k));
run(suite, "log10", "1K", () => log10(pos1k));
run(suite, "log10", "100K", () => log10(pos100k));
run(suite, "rsqrt", "1K", () => rsqrt(pos1k));
run(suite, "rsqrt", "100K", () => rsqrt(pos100k));
run(suite, "sqrt", "10K", () => sqrt(pos10k));
run(suite, "exp", "10K", () => exp(a10k));
run(suite, "log", "10K", () => log(pos10k));

// Hyperbolic / inverse trig.
run(suite, "sinh", "1K", () => sinh(a1k));
run(suite, "sinh", "100K", () => sinh(a100k));
run(suite, "cosh", "1K", () => cosh(a1k));
run(suite, "cosh", "100K", () => cosh(a100k));
run(suite, "asin", "1K", () => asin(unit1k));
run(suite, "asin", "100K", () => asin(unit100k));
run(suite, "acos", "1K", () => acos(unit1k));
run(suite, "atan", "1K", () => atan(a1k));
run(suite, "atan", "100K", () => atan(a100k));
run(suite, "asinh", "1K", () => asinh(a1k));
run(suite, "acosh", "1K", () => acosh(ge1_1k));
run(suite, "atanh", "1K", () => atanh(unit1k));

// Reductions (nan-aware, boolean).
run(suite, "nanmean", "1K", () => nanmean(a1k));
run(suite, "nanmean", "100K", () => nanmean(a100k));
run(suite, "nansum", "1K", () => nansum(a1k));
run(suite, "nansum", "100K", () => nansum(a100k));
run(suite, "nanmax", "100K", () => nanmax(a100k));
run(suite, "nanmin", "100K", () => nanmin(a100k));
run(suite, "nanstd", "1K", () => nanstd(a1k));
run(suite, "all", "100K", () => all(greater(a100k, zeros([100000]))));
run(suite, "any", "100K", () => any(greater(a100k, zeros([100000]))));
run(suite, "sum", "10K", () => sum(a10k));
run(suite, "mean", "10K", () => mean(a10k));
run(suite, "max", "10K", () => max(a10k));
run(suite, "min", "10K", () => min(a10k));
run(suite, "std", "10K", () => std(a10k));
run(suite, "variance", "10K", () => variance(a10k));
run(suite, "cumsum", "10K", () => cumsum(a10k));

// Manipulation.
run(suite, "roll", "100K", () => roll(a100k, 100));
run(suite, "flip", "100K", () => flip(a100k));
run(suite, "fliplr", "100x100", () => fliplr(mat100));
run(suite, "flipud", "100x100", () => flipud(mat100));
run(suite, "rot90", "100x100", () => rot90(mat100));
run(suite, "repeat", "1K→2x", () => repeat(a1k, 2));
run(suite, "tile", "1K→2x", () => tile(a1k, [2]));
run(suite, "diff", "100K", () => diff(a100k));
run(suite, "diag", "100x100", () => diag(mat100));
run(suite, "diagonal", "100x100", () => diagonal(mat100));
run(suite, "tril", "100x100", () => tril(mat100));
run(suite, "triu", "100x100", () => triu(mat100));
run(suite, "moveaxis", "100x100", () => moveaxis(mat100, 0, 1));
run(suite, "swapaxes", "100x100", () => swapaxes(mat100, 0, 1));
run(suite, "hstack", "2×100K", () => hstack([a100k, b100k]));
run(suite, "vstack", "2×1K", () => vstack([a1k, b1k]));
run(suite, "column_stack", "2×1K", () => column_stack([a1k, b1k]));
run(suite, "broadcast_to", "1K→100x1K", () => broadcast_to(a1k, [100, 1000]));
run(suite, "where", "100K", () => where(greater(a100k, zeros([100000])), a100k, b100k));
run(suite, "unique", "1K", () => unique(a1k));
run(suite, "trapz", "100K", () => trapz(a100k));
run(suite, "cross", "3x3", () => cross(tensor([1, 0, 0]), tensor([0, 1, 0])));
run(suite, "pad", "1K→pad2", () => pad(a1k, [[2, 2]]));

// Predicates.
run(suite, "isnan", "100K", () => isnan(a100k));
run(suite, "isinf", "100K", () => isinf(a100k));
run(suite, "isfinite", "100K", () => isfinite(a100k));

const intPool = arange(0, 1000, 1, { dtype: "int32" });
run(suite, "bincount", "1K", () => bincount(intPool));

footer(suite, "deepbox-ndarray.json");
