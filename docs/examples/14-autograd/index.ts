/**
 * Example 14: Automatic Differentiation (Autograd)
 *
 * Deepbox records the operations done on tracked tensors (GradTensor) as a
 * computation graph. Calling backward() walks the graph in reverse and fills
 * in the gradient of every tracked input.
 *
 * Use parameter(...) only for tensors you want gradients for, such as weights
 * or a value you differentiate with respect to. Training data stays a plain
 * tensor.
 */

import { noGrad, parameter, tensor } from "deepbox/ndarray";
import { Linear } from "deepbox/nn";

console.log("=== Automatic Differentiation ===\n");

// ---------------------------------------------------------------------------
// Part 1: Basic gradient computation
// ---------------------------------------------------------------------------
console.log("--- Part 1: Basic Gradients ---");

// f(x) = sum(x^2)  =>  df/dx = 2x
const x = parameter([2, 3, 4]);
const y = x.mul(x).sum();
y.backward();

console.log("x       :", x.tensor.toString());
console.log("f(x)    :", y.item());
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

// z = sum(a @ w)
const z = a.matmul(w).sum();
z.backward();

console.log("a =", a.tensor.toString());
console.log("w =", w.tensor.toString());
console.log("z = sum(a @ w) =", z.item());
console.log("dz/da =", a.grad?.toString() ?? "null");
console.log("dz/dw =", w.grad?.toString() ?? "null");

// ---------------------------------------------------------------------------
// Part 3: Chained operations
// ---------------------------------------------------------------------------
console.log("\n--- Part 3: Chained Operations ---");

const p = parameter([1, 2, 3, 4]);

// f(p) = sum(relu(2p - 3)). Plain numbers can be used directly in the chain.
const shifted = p.mul(2).sub(3);
const activated = shifted.relu();
const loss = activated.sum();
loss.backward();

console.log("p       :", p.tensor.toString());
console.log("2p - 3  :", shifted.tensor.toString());
console.log("relu    :", activated.tensor.toString());
console.log("grad    :", p.grad?.toString() ?? "null");
// relu passes the gradient only where 2p - 3 is positive, so p = 1 gets 0.

// ---------------------------------------------------------------------------
// Part 4: noGrad for inference
// ---------------------------------------------------------------------------
console.log("\n--- Part 4: noGrad for Inference ---");

const q = parameter([1, 2, 3]);
noGrad(() => {
  // Operations inside noGrad are not recorded, so the result has no graph.
  const result = q.mul(q);
  console.log("noGrad result:", result.tensor.toString());
  console.log("requiresGrad:", result.requiresGrad);
});

// ---------------------------------------------------------------------------
// Part 5: Gradient accumulation and zeroGrad
// ---------------------------------------------------------------------------
console.log("\n--- Part 5: Gradient Accumulation ---");

const v = parameter([1, 2, 3]);

// First backward
v.mul(v).sum().backward();
console.log("After first backward, grad:", v.grad?.toString() ?? "null");

// Without zeroGrad, a second backward adds to the stored gradient.
v.mul(3).sum().backward();
console.log("After second backward without zeroGrad:", v.grad?.toString() ?? "null");

// Reset the gradient before computing a new one.
v.zeroGrad();
v.mul(3).sum().backward();
console.log("After zeroGrad and a new backward:", v.grad?.toString() ?? "null");

// ---------------------------------------------------------------------------
// Part 6: Gradients of a layer with plain-tensor input
// ---------------------------------------------------------------------------
console.log("\n--- Part 6: Layer Gradients from a Plain Tensor ---");

// The input is a plain tensor. The layer's weight and bias are the tracked
// parameters, so forward() returns a tracked result.
const layer = new Linear(2, 1);
const input = tensor([
  [1, 2],
  [3, 4],
]);
const out = layer.forward(input);
console.log("Output tracks gradients:", out.requiresGrad);

out.sum().backward();
// namedParameters() yields [name, GradTensor] pairs, here "weight" and "bias".
for (const [name, param] of layer.namedParameters()) {
  console.log(`${name}.grad:`, param.grad?.toString() ?? "null");
}
// d(sum(out))/d(weight) is the column sums of the input, [1 + 3, 2 + 4].
// d(sum(out))/d(bias) is the number of input rows, 2.

console.log("\n=== Autograd Complete ===");
