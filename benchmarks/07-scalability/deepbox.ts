/**
 * Benchmark 07: Scalability — Deepbox vs NumPy
 *
 * Tests how performance scales with increasing data sizes.
 * Measures throughput (elements/sec) at multiple scales to
 * reveal algorithmic complexity and memory behavior.
 */

import { add, dot, mean, mul, ones, sqrt, sum, transpose, zeros } from "deepbox/ndarray";
import { createSuite, footer, header, run } from "../utils";

const suite = createSuite("scalability");
header("Benchmark 07: Scalability");

// ── Tensor Creation Scaling ───────────────────────────────

const sizes = [1000, 10000, 100000, 500000, 1000000];

for (const n of sizes) {
  const label = n >= 1000000 ? `${n / 1000000}M` : `${n / 1000}K`;
  run(suite, "zeros creation", label, () => {
    zeros([n]);
  });
}

for (const n of sizes) {
  const label = n >= 1000000 ? `${n / 1000000}M` : `${n / 1000}K`;
  run(suite, "ones creation", label, () => {
    ones([n]);
  });
}

// ── Element-wise Addition Scaling ─────────────────────────

for (const n of sizes) {
  const label = n >= 1000000 ? `${n / 1000000}M` : `${n / 1000}K`;
  const a = ones([n]);
  const b = ones([n]);
  run(suite, "add (element-wise)", label, () => {
    add(a, b);
  });
}

// ── Element-wise Multiply Scaling ─────────────────────────

for (const n of sizes) {
  const label = n >= 1000000 ? `${n / 1000000}M` : `${n / 1000}K`;
  const a = ones([n]);
  const b = ones([n]);
  run(suite, "mul (element-wise)", label, () => {
    mul(a, b);
  });
}

// ── Reduction Scaling ─────────────────────────────────────

for (const n of sizes) {
  const label = n >= 1000000 ? `${n / 1000000}M` : `${n / 1000}K`;
  const a = ones([n]);
  run(suite, "sum reduction", label, () => {
    sum(a);
  });
}

for (const n of sizes) {
  const label = n >= 1000000 ? `${n / 1000000}M` : `${n / 1000}K`;
  const a = ones([n]);
  run(suite, "mean reduction", label, () => {
    mean(a);
  });
}

// ── Math Function Scaling ─────────────────────────────────

for (const n of sizes) {
  const label = n >= 1000000 ? `${n / 1000000}M` : `${n / 1000}K`;
  const a = ones([n]);
  run(suite, "sqrt", label, () => {
    sqrt(a);
  });
}

// ── Matrix Multiply Scaling ───────────────────────────────

const matSizes = [50, 100, 200, 300, 500];
for (const n of matSizes) {
  const m = ones([n, n]);
  const iters = n >= 300 ? 5 : n >= 200 ? 10 : 20;
  run(
    suite,
    "matmul (NxN @ NxN)",
    `${n}x${n}`,
    () => {
      dot(m, m);
    },
    { iterations: iters }
  );
}

// ── Transpose Scaling ─────────────────────────────────────

for (const n of matSizes) {
  const m = ones([n, n]);
  run(suite, "transpose", `${n}x${n}`, () => {
    transpose(m);
  });
}

footer(suite, "deepbox-scalability.json");
