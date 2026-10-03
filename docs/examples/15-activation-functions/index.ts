/**
 * Example 15: Activation Functions
 *
 * Apply common activation functions to a range of inputs, compare their values
 * at a few points, and draw them. Each function is available as a function
 * (relu(x)) and as a tensor method (x.relu()).
 */

import { mkdirSync, writeFileSync } from "node:fs";
import {
  celu,
  elu,
  gelu,
  hardswish,
  leakyRelu,
  linspace,
  mish,
  relu,
  relu6,
  selu,
  sigmoid,
  softmax,
  softplus,
  softsign,
  swish,
  tensor,
} from "deepbox/ndarray";
import { Figure } from "deepbox/plot";

console.log("=== Activation Functions ===\n");

mkdirSync("docs/examples/15-activation-functions/output", { recursive: true });

// Generate the input range for the plots.
const x = linspace(-5, 5, 100);

// Each entry: name, formula, typical use, and the function itself.
const activations = [
  {
    name: "ReLU",
    formula: "f(x) = max(0, x)",
    use: "Default for hidden layers, cheap to compute",
    fn: relu,
  },
  {
    name: "Sigmoid",
    formula: "f(x) = 1 / (1 + e^(-x))",
    use: "Binary classification output, range (0, 1)",
    fn: sigmoid,
  },
  {
    name: "GELU",
    formula: "f(x) = x * Phi(x), Phi is the standard normal CDF",
    use: "Transformers. Deepbox uses the tanh approximation by default",
    fn: (t: typeof x) => gelu(t),
  },
  {
    name: "Leaky ReLU",
    formula: "f(x) = x if x > 0, else 0.01 * x",
    use: "Keeps a small gradient for negative inputs",
    fn: (t: typeof x) => leakyRelu(t, 0.01),
  },
  {
    name: "ELU",
    formula: "f(x) = x if x > 0, else alpha * (e^x - 1)",
    use: "Smooth, negative outputs pull the mean toward zero",
    fn: (t: typeof x) => elu(t, 1.0),
  },
  {
    name: "Mish",
    formula: "f(x) = x * tanh(softplus(x))",
    use: "Smooth and non-monotonic",
    fn: mish,
  },
  {
    name: "Swish (SiLU)",
    formula: "f(x) = x * sigmoid(x)",
    use: "Smooth and non-monotonic",
    fn: swish,
  },
  {
    name: "Softplus",
    formula: "f(x) = log(1 + e^x)",
    use: "Smooth version of ReLU, always positive",
    fn: softplus,
  },
  {
    name: "ReLU6",
    formula: "f(x) = min(max(0, x), 6)",
    use: "Bounded ReLU, common in mobile networks",
    fn: relu6,
  },
  {
    name: "SELU",
    formula: "f(x) = scale * (x if x > 0, else alpha * (e^x - 1))",
    use: "Self-normalizing networks",
    fn: selu,
  },
  {
    name: "CELU",
    formula: "f(x) = max(0, x) + min(0, alpha * (e^(x / alpha) - 1))",
    use: "ELU with a continuous derivative",
    fn: (t: typeof x) => celu(t, 1.0),
  },
  {
    name: "Softsign",
    formula: "f(x) = x / (1 + |x|)",
    use: "Like tanh, with slower saturation",
    fn: softsign,
  },
  {
    name: "Hardswish",
    formula: "f(x) = x * relu6(x + 3) / 6",
    use: "Cheap piecewise approximation of Swish",
    fn: hardswish,
  },
];

// Print the function, its use, and its value at five inputs.
const probe = tensor([-2, -1, 0, 1, 2]);
console.log(`Values at x = ${probe.toArray()}\n`);
for (const [index, { name, formula, use, fn }] of activations.entries()) {
  console.log(`${index + 1}. ${name}`);
  console.log(`   ${formula}`);
  console.log(`   Use: ${use}`);
  const values = Array.from(fn(probe).toArray() as number[], (v) => v.toFixed(4));
  console.log(`   Output: ${values.join(", ")}\n`);
}

// Softmax works on a whole vector, so it is shown on its own.
console.log("Softmax");
const sample = tensor([1.0, 2.0, 3.0, 4.0]);
const probabilities = softmax(sample);
console.log("   f(x_i) = e^(x_i) / sum_j e^(x_j)");
console.log("   Use: Multi-class classification output");
console.log("   Input: ", sample.toString());
console.log("   Output:", probabilities.toString());
console.log(`   Sum of outputs: ${Number(probabilities.sum().item()).toFixed(4)}\n`);

// GELU: tanh approximation (default) vs the exact form used by PyTorch's default.
const geluTanh = gelu(x);
const geluExact = gelu(x, { approximate: "none" });
const maxGap = Number(geluTanh.sub(geluExact).abs().max().item());
console.log("GELU variants");
console.log(`   Largest gap between tanh and exact GELU on [-5, 5]: ${maxGap.toExponential(2)}`);
console.log('   Pass { approximate: "none" } to match PyTorch\'s default gelu.\n');

// Draw a selection of the curves.
console.log("Creating visualization...");
const fig = new Figure({ width: 800, height: 600 });
const ax = fig.addAxes();

ax.plot(x, relu(x), { color: "#1f77b4", linewidth: 2, label: "ReLU" });
ax.plot(x, sigmoid(x), { color: "#ff7f0e", linewidth: 2, label: "Sigmoid" });
ax.plot(x, geluTanh, { color: "#2ca02c", linewidth: 2, label: "GELU" });
ax.plot(x, swish(x), { color: "#d62728", linewidth: 2, label: "Swish" });
ax.plot(x, elu(x, 1.0), { color: "#9467bd", linewidth: 2, label: "ELU" });
ax.setTitle("Activation Functions Comparison");
ax.setXLabel("Input");
ax.setYLabel("Output");
ax.legend();

const svg = fig.renderSVG();
writeFileSync("docs/examples/15-activation-functions/output/activations.svg", svg.svg);
console.log("Saved: output/activations.svg\n");

console.log("Selection guide:");
console.log("  ReLU: default choice, fast");
console.log("  Sigmoid: binary classification output layer");
console.log("  Softmax: multi-class classification output layer");
console.log("  GELU, Swish, Mish: smooth alternatives, common in transformers and recent models");
console.log("  Leaky ReLU, ELU: when units stop firing because of the zero gradient of ReLU");
