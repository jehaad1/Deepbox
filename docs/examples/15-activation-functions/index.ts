/**
 * Example 15: Activation Functions
 *
 * Explore different activation functions used in neural networks.
 * Learn when to use each activation function.
 */

import { mkdirSync, writeFileSync } from "node:fs";
import {
  elu,
  gelu,
  leakyRelu,
  linspace,
  mish,
  relu,
  sigmoid,
  softmax,
  softplus,
  swish,
  tensor,
} from "deepbox/ndarray";
import { Figure } from "deepbox/plot";

console.log("=== Activation Functions ===\n");

mkdirSync("docs/examples/15-activation-functions/output", { recursive: true });

// Generate input range
const x = linspace(-5, 5, 100);

console.log("1. ReLU (Rectified Linear Unit):");
console.log("-".repeat(50));
const relu_out = relu(x);
console.log("f(x) = max(0, x)");
console.log("Use: Hidden layers, fast computation");
console.log("Range: [0, ∞)\n");

console.log("2. Sigmoid:");
console.log("-".repeat(50));
const sigmoid_out = sigmoid(x);
console.log("f(x) = 1 / (1 + e^(-x))");
console.log("Use: Binary classification output");
console.log("Range: (0, 1)\n");

console.log("3. Softmax:");
console.log("-".repeat(50));
const sample = tensor([1.0, 2.0, 3.0, 4.0]);
const softmax_out = softmax(sample);
console.log("Input:", sample.toString());
console.log("Output:", softmax_out.toString());
console.log("Use: Multi-class classification output");
console.log("Properties: Outputs sum to 1.0\n");

console.log("4. GELU (Gaussian Error Linear Unit):");
console.log("-".repeat(50));
const gelu_out = gelu(x);
console.log("f(x) = x * Φ(x), where Φ is CDF of normal distribution");
console.log("Use: Transformers, modern architectures");
console.log("Smoother than ReLU\n");

console.log("5. Leaky ReLU:");
console.log("-".repeat(50));
leakyRelu(x, 0.01);
console.log("f(x) = max(αx, x), α = 0.01");
console.log("Use: Prevents dying ReLU problem");
console.log("Allows small negative values\n");

console.log("6. ELU (Exponential Linear Unit):");
console.log("-".repeat(50));
elu(x, 1.0);
console.log("f(x) = x if x > 0, else α(e^x - 1)");
console.log("Use: Can produce negative outputs");
console.log("Smoother than ReLU\n");

console.log("7. Mish:");
console.log("-".repeat(50));
mish(x);
console.log("f(x) = x * tanh(softplus(x))");
console.log("Use: State-of-the-art in some tasks");
console.log("Self-regularizing, smooth\n");

console.log("8. Swish (SiLU):");
console.log("-".repeat(50));
swish(x);
console.log("f(x) = x * sigmoid(x)");
console.log("Use: Discovered by neural architecture search");
console.log("Non-monotonic, smooth\n");

console.log("9. Softplus:");
console.log("-".repeat(50));
softplus(x);
console.log("f(x) = log(1 + e^x)");
console.log("Use: Smooth approximation of ReLU");
console.log("Always positive\n");

// Visualize activations
console.log("Creating visualization...");
const fig = new Figure({ width: 800, height: 600 });
const ax = fig.addAxes();

ax.plot(x, relu_out, { color: "#1f77b4", linewidth: 2 });
ax.plot(x, sigmoid_out, { color: "#ff7f0e", linewidth: 2 });
ax.plot(x, gelu_out, { color: "#2ca02c", linewidth: 2 });
ax.setTitle("Activation Functions Comparison");
ax.setXLabel("Input");
ax.setYLabel("Output");

const svg = fig.renderSVG();
writeFileSync("docs/examples/15-activation-functions/output/activations.svg", svg.svg);
console.log("✓ Saved: output/activations.svg\n");

console.log("Selection Guide:");
console.log("• ReLU: Default choice, fast and effective");
console.log("• Sigmoid: Binary classification output layer");
console.log("• Softmax: Multi-class classification output layer");
console.log("• GELU/Mish/Swish: Modern alternatives, often better performance");
console.log("• Leaky ReLU/ELU: When dying ReLU is a problem");

console.log("\n✓ Activation functions complete!");
