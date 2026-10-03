/**
 * Example 13: Neural Network Training
 *
 * Train small networks with the nn, optim and ndarray modules.
 * Covers Sequential models, custom modules, loss functions and optimizers.
 *
 * Training uses plain tensors for the data. When gradient mode is on (the
 * default) and a module has trainable parameters, model.forward(x) returns a
 * GradTensor that tracks the weights, so loss.backward() fills their gradients.
 * Wrap evaluation in noGrad() to turn tracking off.
 */

import { type AnyTensor, noGrad, tensor } from "deepbox/ndarray";
import { Linear, Module, mseLoss, ReLU, Sequential } from "deepbox/nn";
import { Adam, SGD } from "deepbox/optim";

console.log("=== Neural Network Training ===\n");

// Training data: y = x0 + 2*x1
const X = tensor([
  [1, 0],
  [0, 1],
  [1, 1],
  [2, 1],
  [1, 2],
  [3, 1],
  [2, 2],
  [0, 3],
]);
const y = tensor([[1], [2], [3], [4], [5], [5], [6], [6]]);

// ---------------------------------------------------------------------------
// Part 1: Sequential model trained with Adam
// ---------------------------------------------------------------------------
console.log("--- Part 1: Sequential Model ---");

const model = new Sequential(new Linear(2, 16), new ReLU(), new Linear(16, 1));

const paramCount = Array.from(model.parameters()).length;
console.log("Model parameters:", paramCount);

const optimizer = new Adam(model.parameters(), { lr: 0.01 });

console.log("Training for 200 epochs...");
for (let epoch = 0; epoch < 200; epoch++) {
  optimizer.zeroGrad();

  // The forward pass records the computation graph. The loss is a scalar.
  const loss = mseLoss(model.forward(X), y);

  // Backward pass, then one optimizer step.
  loss.backward();
  optimizer.step();

  if (epoch % 50 === 0) {
    console.log(`  Epoch ${epoch}: loss = ${Number(loss.item()).toFixed(6)}`);
  }
}

// Evaluate without gradient tracking. The result is a plain Tensor.
const finalPred = noGrad(() => model.forward(X));
console.log("Predictions:", finalPred.toString());
console.log("Targets:    ", y.toString());

// ---------------------------------------------------------------------------
// Part 2: Custom module
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

  override forward(x: AnyTensor): AnyTensor {
    return this.fc2.forward(this.relu.forward(this.fc1.forward(x)));
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
// Part 3: Evaluation with noGrad and mseLoss
// ---------------------------------------------------------------------------
console.log("\n--- Part 3: Evaluation with noGrad ---");

const testInput = tensor([
  [1, 0],
  [0, 1],
  [1, 1],
  [2, 1],
]);
const testTarget = tensor([[1], [2], [3], [4]]);

// Inside noGrad nothing is recorded, so forward() and the loss are plain tensors.
const evalLoss = noGrad(() => mseLoss(model.forward(testInput), testTarget));
console.log("Eval loss (no gradient tracking):", Number(evalLoss.item()).toFixed(6));

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
  sgdOptimizer.zeroGrad();
  const loss = mseLoss(sgdModel.forward(X), y);
  loss.backward();
  sgdOptimizer.step();
}

const sgdLoss = noGrad(() => mseLoss(sgdModel.forward(X), y));
console.log("SGD final loss:", Number(sgdLoss.item()).toFixed(6));

console.log("\n=== Neural Network Training Complete ===");
