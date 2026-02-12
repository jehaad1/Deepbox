/**
 * Example 32: Neural Network Module System
 *
 * Demonstrates the Module base class features: parameter registration, state
 * serialization, train/eval modes, freeze/unfreeze, and forward hooks.
 */

import { GradTensor, parameter, type Tensor, tensor } from "deepbox/ndarray";
import { Linear, Module, ReLU, Sequential } from "deepbox/nn";

console.log("=== Neural Network Module System ===\n");

// ---------------------------------------------------------------------------
// Part 1: Custom Module with parameter registration
// ---------------------------------------------------------------------------
console.log("--- Part 1: Custom Module ---");

class MyNet extends Module {
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

const net = new MyNet(4, 8, 2);
console.log("MyNet(4 -> 8 -> 2)");

// ---------------------------------------------------------------------------
// Part 2: Parameter enumeration
// ---------------------------------------------------------------------------
console.log("\n--- Part 2: Parameters ---");

const params = Array.from(net.parameters());
console.log(`Total parameter tensors: ${params.length}`);
for (const p of params) {
  const t = p instanceof GradTensor ? p.tensor : p;
  console.log(`  Shape: [${t.shape.join(", ")}]`);
}

// ---------------------------------------------------------------------------
// Part 3: State dict — serialization & loading
// ---------------------------------------------------------------------------
console.log("\n--- Part 3: State Dict ---");

const stateDict = net.stateDict();
console.log("State dict parameter keys:");
for (const key of Object.keys(stateDict.parameters)) {
  console.log(`  ${key}`);
}

// Load state dict back (e.g., from a saved checkpoint)
net.loadStateDict(stateDict);
console.log("State dict loaded successfully");

// ---------------------------------------------------------------------------
// Part 4: Train/Eval mode
// ---------------------------------------------------------------------------
console.log("\n--- Part 4: Train/Eval Mode ---");

net.train();
console.log(`Training mode: ${net.training}`);

net.eval();
console.log(`Eval mode: ${net.training}`);
console.log("  Eval mode disables dropout and uses running stats for batchnorm");

// ---------------------------------------------------------------------------
// Part 5: Freeze/Unfreeze parameters
// ---------------------------------------------------------------------------
console.log("\n--- Part 5: Freeze/Unfreeze ---");

net.freezeParameters();
console.log("After freezeParameters:");
const frozenParams = Array.from(net.parameters());
const frozenGrads = frozenParams.filter((p) => p instanceof GradTensor && p.requiresGrad);
console.log(`  Parameters requiring grad: ${frozenGrads.length}`);

net.unfreezeParameters();
console.log("After unfreezeParameters:");
const unfrozenParams = Array.from(net.parameters());
const unfrozenGrads = unfrozenParams.filter((p) => p instanceof GradTensor && p.requiresGrad);
console.log(`  Parameters requiring grad: ${unfrozenGrads.length}`);

// ---------------------------------------------------------------------------
// Part 6: Sequential container
// ---------------------------------------------------------------------------
console.log("\n--- Part 6: Sequential Container ---");

const seqModel = new Sequential(new Linear(4, 8), new ReLU(), new Linear(8, 2));

console.log("Sequential(Linear(4,8), ReLU, Linear(8,2))");
const seqParams = Array.from(seqModel.parameters()).length;
console.log(`Parameters: ${seqParams}`);

// Forward pass with plain Tensor (inference)
const input = tensor([[1, 2, 3, 4]]);
const output = seqModel.forward(input);
const outTensor = output instanceof GradTensor ? output.tensor : output;
console.log(`Input shape:  [${input.shape.join(", ")}]`);
console.log(`Output shape: [${outTensor.shape.join(", ")}]`);

// Forward pass with GradTensor (training)
const gradInput = parameter([[1, 2, 3, 4]]);
const gradOutput = seqModel.forward(gradInput);
console.log(
  `GradTensor output requiresGrad: ${gradOutput instanceof GradTensor ? gradOutput.requiresGrad : false}`
);

console.log("\n=== Module System Complete ===");
