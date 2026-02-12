/**
 * Example 14: Automatic Differentiation (Autograd)
 *
 * Deepbox's autograd system tracks operations on GradTensors to build a
 * computation graph, then computes gradients via reverse-mode differentiation.
 */

import { GradTensor, noGrad, parameter, tensor } from "deepbox/ndarray";

console.log("=== Automatic Differentiation ===\n");

// ---------------------------------------------------------------------------
// Part 1: Basic gradient computation
// ---------------------------------------------------------------------------
console.log("--- Part 1: Basic Gradients ---");

// f(x) = x^2  =>  df/dx = 2x
const x = parameter([2, 3, 4]);
const y = x.mul(x).sum();
y.backward();

console.log("x       :", x.tensor.toString());
console.log("f(x)=x²  sum:", y.tensor.toString());
console.log("grad    :", x.grad?.toString() ?? "null");
// Expected gradients: [4, 6, 8]

// ---------------------------------------------------------------------------
// Part 2: Multi-variable gradients
// ---------------------------------------------------------------------------
console.log("\n--- Part 2: Multi-Variable Gradients ---");

const a = parameter([
  [1, 2],
  [3, 4],
]);
const w = parameter([[0.5], [0.5]]);

// y = sum(a @ w)
const z = a.matmul(w).sum();
z.backward();

console.log("a =", a.tensor.toString());
console.log("w =", w.tensor.toString());
console.log("z = sum(a @ w) =", z.tensor.toString());
console.log("dz/da =", a.grad?.toString() ?? "null");
console.log("dz/dw =", w.grad?.toString() ?? "null");

// ---------------------------------------------------------------------------
// Part 3: Chained operations
// ---------------------------------------------------------------------------
console.log("\n--- Part 3: Chained Operations ---");

const p = parameter([1, 2, 3, 4]);

// f(p) = sum(relu(p * 2 - 3))
const scaled = p.mul(GradTensor.fromTensor(tensor([2, 2, 2, 2]), { requiresGrad: false }));
const shifted = scaled.sub(GradTensor.fromTensor(tensor([3, 3, 3, 3]), { requiresGrad: false }));
const activated = shifted.relu();
const loss = activated.sum();
loss.backward();

console.log("p       :", p.tensor.toString());
console.log("2p - 3  :", shifted.tensor.toString());
console.log("relu    :", activated.tensor.toString());
console.log("grad    :", p.grad?.toString() ?? "null");

// ---------------------------------------------------------------------------
// Part 4: noGrad for inference
// ---------------------------------------------------------------------------
console.log("\n--- Part 4: noGrad for Inference ---");

const q = parameter([1, 2, 3]);
noGrad(() => {
  // Operations inside noGrad do not track gradients
  const result = q.mul(q);
  console.log("noGrad result:", result.tensor.toString());
  console.log("requiresGrad:", result.requiresGrad);
});

// ---------------------------------------------------------------------------
// Part 5: Zero gradients and re-compute
// ---------------------------------------------------------------------------
console.log("\n--- Part 5: Gradient Accumulation ---");

const v = parameter([1, 2, 3]);

// First backward
const loss1 = v.mul(v).sum();
loss1.backward();
console.log("After first backward, grad:", v.grad?.toString() ?? "null");

// Zero gradients before second pass
v.zeroGrad();
console.log("After zeroGrad, grad:", v.grad?.toString() ?? "null");

// Second backward with different computation
const loss2 = v.mul(GradTensor.fromTensor(tensor([3, 3, 3]), { requiresGrad: false })).sum();
loss2.backward();
console.log("After second backward, grad:", v.grad?.toString() ?? "null");

console.log("\n=== Autograd Complete ===");
