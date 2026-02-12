/**
 * Benchmark 01 — Tensor Operations
 * Deepbox vs NumPy
 */

import {
  abs,
  add,
  arange,
  clip,
  concatenate,
  cos,
  div,
  dot,
  empty,
  exp,
  eye,
  flatten,
  full,
  linspace,
  log,
  max,
  mean,
  min,
  mul,
  neg,
  ones,
  pow,
  randn,
  reshape,
  sign,
  sin,
  sort,
  sqrt,
  squeeze,
  stack,
  sub,
  sum,
  tensor,
  transpose,
  unsqueeze,
  variance,
  zeros,
} from "deepbox/ndarray";
import { createSuite, footer, header, run } from "../utils";

const suite = createSuite("tensor");
header("Benchmark 01 — Tensor Operations");

// ── Creation ──────────────────────────────────────────────

run(suite, "zeros", "1K", () => zeros([1000]));
run(suite, "zeros", "100K", () => zeros([100000]));
run(suite, "zeros", "1M", () => zeros([1000000]));
run(suite, "ones", "100K", () => ones([100000]));
run(suite, "ones", "1M", () => ones([1000000]));
run(suite, "full", "100K", () => full([100000], 3.14));
run(suite, "empty", "100K", () => empty([100000]));
run(suite, "arange", "10K", () => arange(0, 10000));
run(suite, "linspace", "10K", () => linspace(0, 1, 10000));
run(suite, "eye", "100x100", () => eye(100));
run(suite, "eye", "500x500", () => eye(500));
run(suite, "randn", "10K", () => randn([10000]));
run(suite, "randn", "100K", () => randn([100000]));
run(suite, "tensor (nested array)", "100x100", () => {
  tensor(Array.from({ length: 100 }, () => Array.from({ length: 100 }, () => Math.random())));
});
run(
  suite,
  "tensor (nested array)",
  "500x500",
  () => {
    tensor(Array.from({ length: 500 }, () => Array.from({ length: 500 }, () => Math.random())));
  },
  { iterations: 10 }
);

// ── Element-wise Arithmetic ───────────────────────────────

const a10k = ones([10000]);
const b10k = ones([10000]);
const a100k = ones([100000]);
const b100k = ones([100000]);

run(suite, "add", "10K", () => add(a10k, b10k));
run(suite, "add", "100K", () => add(a100k, b100k));
run(suite, "sub", "10K", () => sub(a10k, b10k));
run(suite, "sub", "100K", () => sub(a100k, b100k));
run(suite, "mul", "10K", () => mul(a10k, b10k));
run(suite, "mul", "100K", () => mul(a100k, b100k));
run(suite, "div", "10K", () => div(a10k, b10k));
run(suite, "div", "100K", () => div(a100k, b100k));
run(suite, "neg", "100K", () => neg(a100k));
run(suite, "pow", "10K", () => pow(a10k, tensor(Array.from({ length: 10000 }, () => 2))));
run(suite, "clip", "100K", () => clip(a100k, 0.2, 0.8));
run(suite, "sign", "100K", () => sign(a100k));

// ── Math Functions ────────────────────────────────────────

run(suite, "sqrt", "10K", () => sqrt(a10k));
run(suite, "sqrt", "100K", () => sqrt(a100k));
run(suite, "exp", "10K", () => exp(a10k));
run(suite, "exp", "100K", () => exp(a100k));
run(suite, "log", "10K", () => log(a10k));
run(suite, "log", "100K", () => log(a100k));
run(suite, "abs", "100K", () => abs(a100k));
run(suite, "sin", "10K", () => sin(a10k));
run(suite, "sin", "100K", () => sin(a100k));
run(suite, "cos", "10K", () => cos(a10k));
run(suite, "cos", "100K", () => cos(a100k));

// ── Reductions ────────────────────────────────────────────

run(suite, "sum", "10K", () => sum(a10k));
run(suite, "sum", "100K", () => sum(a100k));
run(suite, "mean", "10K", () => mean(a10k));
run(suite, "mean", "100K", () => mean(a100k));
run(suite, "max", "10K", () => max(a10k));
run(suite, "max", "100K", () => max(a100k));
run(suite, "min", "10K", () => min(a10k));
run(suite, "min", "100K", () => min(a100k));
run(suite, "variance", "10K", () => variance(a10k));
run(suite, "variance", "100K", () => variance(a100k));

// ── Sorting ──────────────────────────────────────────────

const rand10k = randn([10000]);
const rand100k = randn([100000]);

run(suite, "sort", "10K", () => sort(rand10k));
run(suite, "sort", "100K", () => sort(rand100k));

// ── Matrix Operations ─────────────────────────────────────

const m100 = ones([100, 100]);
const m200 = ones([200, 200]);
const m500 = ones([500, 500]);

run(suite, "matmul", "100x100", () => dot(m100, m100));
run(suite, "matmul", "200x200", () => dot(m200, m200));
run(suite, "matmul", "500x500", () => dot(m500, m500), { iterations: 5 });
run(suite, "transpose", "100x100", () => transpose(m100));
run(suite, "transpose", "500x500", () => transpose(m500));

// ── Shape Operations ──────────────────────────────────────

const flat100k = ones([100000]);
const mat100x1k = ones([100, 1000]);

run(suite, "reshape", "100K → 100x1000", () => reshape(flat100k, [100, 1000]));
run(suite, "reshape", "100K → 1000x100", () => reshape(flat100k, [1000, 100]));
run(suite, "flatten", "100x1000", () => flatten(mat100x1k));
run(suite, "squeeze", "1x100Kx1", () => squeeze(ones([1, 100000, 1])));
run(suite, "unsqueeze", "100K → 1x100K", () => unsqueeze(flat100k, 0));

// ── Concatenation & Stacking ─────────────────────────────

const c1 = ones([50000]);
const c2 = ones([50000]);

run(suite, "concatenate", "2x50K", () => concatenate([c1, c2]));
run(suite, "stack", "2x10K", () => stack([a10k, b10k]));

footer(suite, "deepbox-tensor.json");
