/**
 * Benchmark 06 — NDArray / Tensor Operations
 * Deepbox vs NumPy
 */

import {
  abs,
  add,
  arange,
  argsort,
  clip,
  concatenate,
  cos,
  cumprod,
  cumsum,
  div,
  dot,
  empty,
  equal,
  exp,
  eye,
  flatten,
  full,
  greater,
  less,
  linspace,
  log,
  logicalAnd,
  logicalNot,
  logicalOr,
  max,
  mean,
  median,
  min,
  mul,
  neg,
  ones,
  pow,
  prod,
  randn,
  relu,
  reshape,
  sigmoid,
  sign,
  sin,
  slice,
  softmax,
  sort,
  sqrt,
  squeeze,
  stack,
  std,
  sub,
  sum,
  tanh,
  tensor,
  transpose,
  unsqueeze,
  variance,
  zeros,
} from "deepbox/ndarray";
import { createSuite, footer, header, run } from "../utils";

const suite = createSuite("ndarray");
header("Benchmark 06 — NDArray / Tensor Operations");

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

run(suite, "reshape", "10K→100x100", () => reshape(flat10k, [100, 100]));
run(suite, "flatten", "100x100", () => flatten(mat100));
run(suite, "transpose", "100x100", () => transpose(mat100));
run(suite, "transpose", "500x500", () => transpose(mat500));
run(suite, "squeeze", "[1,100,1]", () => squeeze(randn([1, 100, 1])));
run(suite, "unsqueeze", "1K→1x1K", () => unsqueeze(a1k, 0));

// ── Manipulation ────────────────────────────────────────

run(suite, "concatenate", "2×1K", () => concatenate([a1k, b1k]));
run(suite, "concatenate", "2×100K", () => concatenate([a100k, b100k]));
run(suite, "stack", "2×1K", () => stack([a1k, b1k]));
run(suite, "stack", "2×100K", () => stack([a100k, b100k]));
run(suite, "slice", "[0:500] of 1K", () => slice(a1k, { start: 0, end: 500 }));

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

footer(suite, "deepbox-ndarray.json");
