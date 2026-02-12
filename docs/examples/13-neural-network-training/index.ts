/**
 * Example 13: Neural Network Training
 *
 * Build and train neural networks using the nn, optim, and autograd modules.
 * Covers Sequential models, custom modules, loss functions, and optimizers.
 */

import { isNumericTypedArray, isTypedArray } from "deepbox/core";
import { GradTensor, parameter, type Tensor, tensor } from "deepbox/ndarray";
import { Linear, Module, mseLoss, ReLU, Sequential } from "deepbox/nn";
import { Adam, SGD } from "deepbox/optim";

console.log("=== Neural Network Training ===\n");

// Helper to read a scalar value from tensor data
const scalarValue = (t: Tensor): number => {
  const d = t.data;
  if (!isTypedArray(d) || !isNumericTypedArray(d)) return NaN;
  return Number(d[t.offset] ?? 0);
};

// ---------------------------------------------------------------------------
// Part 1: Sequential model with autograd training
// ---------------------------------------------------------------------------
console.log("--- Part 1: Sequential Model with Autograd ---");

const model = new Sequential(new Linear(2, 16), new ReLU(), new Linear(16, 1));

const paramCount = Array.from(model.parameters()).length;
console.log("Model parameters:", paramCount);

// Training data: y = x0 + 2*x1
const X = parameter([
  [1, 0],
  [0, 1],
  [1, 1],
  [2, 1],
  [1, 2],
  [3, 1],
  [2, 2],
  [0, 3],
]);
const yTargets = parameter([[1], [2], [3], [4], [5], [5], [6], [6]]);

const optimizer = new Adam(model.parameters(), { lr: 0.01 });

console.log("Training for 200 epochs...");
for (let epoch = 0; epoch < 200; epoch++) {
  // Forward pass with GradTensor builds the computation graph
  const pred = model.forward(X);

  // Compute MSE loss using GradTensor ops (tracks gradients)
  if (!(pred instanceof GradTensor)) throw new Error("Expected GradTensor from forward");
  const diff = pred.sub(yTargets);
  const loss = diff.mul(diff).mean();

  // Backward pass and optimize
  optimizer.zeroGrad();
  loss.backward();
  optimizer.step();

  if (epoch % 50 === 0) {
    console.log(`  Epoch ${epoch}: loss = ${scalarValue(loss.tensor).toFixed(6)}`);
  }
}

// Evaluate using plain Tensor forward pass (no gradient tracking)
const finalPred = model.forward(X.tensor);
console.log("Predictions:", finalPred.toString());
console.log("Targets:    ", yTargets.tensor.toString());

// ---------------------------------------------------------------------------
// Part 2: Custom Module
// ---------------------------------------------------------------------------
console.log("\n--- Part 2: Custom Module ---");

class TwoLayerNet extends Module {
  fc1: Linear;
  relu: ReLU;
  fc2: Linear;

  constructor(inputDim: number, hiddenDim: number, outputDim: number) {
    super();
    this.fc1 = new Linear(inputDim, hiddenDim);
    this.relu = new ReLU();
    this.fc2 = new Linear(hiddenDim, outputDim);
    this.registerModule("fc1", this.fc1);
    this.registerModule("relu", this.relu);
    this.registerModule("fc2", this.fc2);
  }

  override forward(x: GradTensor): GradTensor;
  override forward(x: Tensor): Tensor;
  override forward(x: Tensor | GradTensor): Tensor | GradTensor {
    if (x instanceof GradTensor) {
      let out: GradTensor = this.fc1.forward(x);
      out = this.relu.forward(out);
      return this.fc2.forward(out);
    }
    let out: Tensor = this.fc1.forward(x);
    out = this.relu.forward(out);
    return this.fc2.forward(out);
  }
}

const net = new TwoLayerNet(2, 8, 1);
const netParamCount = Array.from(net.parameters()).length;
console.log("Custom module parameters:", netParamCount);

// Train/eval mode
net.train();
console.log("Training mode:", net.training);
net.eval();
console.log("Eval mode:", net.training);

// State dict for serialization
const state = net.stateDict();
console.log("State dict keys:", Object.keys(state.parameters).join(", "));

// ---------------------------------------------------------------------------
// Part 3: Plain Tensor forward pass with mseLoss
// ---------------------------------------------------------------------------
console.log("\n--- Part 3: Plain Tensor Forward + mseLoss ---");

const inputTensor = tensor([
  [1, 0],
  [0, 1],
  [1, 1],
  [2, 1],
]);
const targetTensor = tensor([[1], [2], [3], [4]]);

// Forward pass with plain Tensors — no autograd, just inference
const rawOutput = model.forward(inputTensor);
const output = rawOutput instanceof GradTensor ? rawOutput.tensor : rawOutput;
const evalLoss = mseLoss(output, targetTensor);
console.log("Eval loss (plain Tensor):", scalarValue(evalLoss).toFixed(6));

// ---------------------------------------------------------------------------
// Part 4: SGD with momentum
// ---------------------------------------------------------------------------
console.log("\n--- Part 4: SGD with Momentum ---");

const sgdModel = new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
const sgdOptimizer = new SGD(sgdModel.parameters(), {
  lr: 0.01,
  momentum: 0.9,
});

for (let epoch = 0; epoch < 100; epoch++) {
  const pred = sgdModel.forward(X);
  if (!(pred instanceof GradTensor)) throw new Error("Expected GradTensor from forward");
  const diff = pred.sub(yTargets);
  const loss = diff.mul(diff).mean();
  sgdOptimizer.zeroGrad();
  loss.backward();
  sgdOptimizer.step();
}

const rawSgdPred = sgdModel.forward(X.tensor);
const sgdPred = rawSgdPred instanceof GradTensor ? rawSgdPred.tensor : rawSgdPred;
const sgdLoss = mseLoss(sgdPred, yTargets.tensor);
console.log("SGD final loss:", scalarValue(sgdLoss).toFixed(6));

console.log("\n=== Neural Network Training Complete ===");
