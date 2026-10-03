/**
 * Example 32: Neural Network Module System
 *
 * The Module base class: parameter registration, state dicts, train/eval
 * modes, freeze/unfreeze, forward hooks, and the Sequential container.
 */

import { type AnyTensor, GradTensor, noGrad, tensor } from "deepbox/ndarray";
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

  override forward(x: AnyTensor): AnyTensor {
    return this.fc2.forward(this.relu.forward(this.fc1.forward(x)));
  }
}

const net = new MyNet(4, 8, 2);
console.log("MyNet(4 -> 8 -> 2)");

// ---------------------------------------------------------------------------
// Part 2: Parameter enumeration
// ---------------------------------------------------------------------------
console.log("\n--- Part 2: Parameters ---");

// parameters() and namedParameters() yield GradTensors, the type optimizers take
const named = Array.from(net.namedParameters());
console.log(`Total parameter tensors: ${named.length}`);
for (const [name, p] of named) {
  console.log(`  ${name.padEnd(11)} shape [${p.shape.join(", ")}]`);
}

// ---------------------------------------------------------------------------
// Part 3: State dict (serialization & loading)
// ---------------------------------------------------------------------------
console.log("\n--- Part 3: State Dict ---");

const stateDict = net.stateDict();
console.log("State dict parameter keys:");
for (const key of Object.keys(stateDict.parameters)) {
  console.log(`  ${key}`);
}

// Load the state dict back, for example from a saved checkpoint
net.loadStateDict(stateDict);
console.log("State dict loaded");

// ---------------------------------------------------------------------------
// Part 4: Train/Eval mode
// ---------------------------------------------------------------------------
console.log("\n--- Part 4: Train/Eval Mode ---");

net.train();
console.log(`After train(): training = ${net.training}`);

net.eval();
console.log(`After eval():  training = ${net.training}`);
console.log("  eval() turns dropout off and makes batch norm use its running statistics");
console.log("  It does not stop gradient tracking. Use noGrad() for that.");

// ---------------------------------------------------------------------------
// Part 5: Freeze/Unfreeze parameters
// ---------------------------------------------------------------------------
console.log("\n--- Part 5: Freeze/Unfreeze ---");

const countTrainable = (module: Module): number =>
  Array.from(module.parameters()).filter((p) => p.requiresGrad).length;

console.log(`Trainable parameter tensors: ${countTrainable(net)}`);

net.freezeParameters();
console.log(`After freezeParameters(): ${countTrainable(net)}`);

// Freeze or unfreeze selected parameters by name
net.unfreezeParameters(["fc2.weight", "fc2.bias"]);
console.log(`After unfreezeParameters(["fc2.weight", "fc2.bias"]): ${countTrainable(net)}`);

net.unfreezeParameters();
console.log(`After unfreezeParameters(): ${countTrainable(net)}`);

// ---------------------------------------------------------------------------
// Part 6: Sequential container
// ---------------------------------------------------------------------------
console.log("\n--- Part 6: Sequential Container ---");

const seqModel = new Sequential(new Linear(4, 8), new ReLU(), new Linear(8, 2));

console.log("Sequential(Linear(4,8), ReLU, Linear(8,2))");
console.log(`Parameter tensors: ${Array.from(seqModel.parameters()).length}`);

// A plain tensor goes in, no parameter(...) wrapping needed
const input = tensor([[1, 2, 3, 4]]);

// Training: the output is a GradTensor that tracks the weights
const trainOutput = seqModel.forward(input);
console.log(`Input shape:  [${input.shape.join(", ")}]`);
console.log(`Output shape: [${trainOutput.shape.join(", ")}]`);
console.log(`Output is a GradTensor: ${GradTensor.isGradTensor(trainOutput)}`);
console.log(
  `Output requiresGrad: ${GradTensor.isGradTensor(trainOutput) && trainOutput.requiresGrad}`
);

// Inference: inside noGrad() no graph is built and the output is a plain tensor
const inferOutput = noGrad(() => seqModel.forward(input));
console.log(`Inside noGrad(), output is a GradTensor: ${GradTensor.isGradTensor(inferOutput)}`);

// A model with every parameter frozen also returns a plain tensor
seqModel.freezeParameters();
console.log(
  `With frozen parameters, output is a GradTensor: ${GradTensor.isGradTensor(seqModel.forward(input))}`
);
seqModel.unfreezeParameters();

// ---------------------------------------------------------------------------
// Part 7: Forward hooks
// ---------------------------------------------------------------------------
console.log("\n--- Part 7: Forward Hooks ---");

// call() runs the hooks around forward(). Calling forward() directly skips them.
const removeHook = seqModel.registerForwardHook((_module, _inputs, output) => {
  console.log(`  hook saw output shape [${output.shape.join(", ")}]`);
  return undefined; // keep the output unchanged
});
noGrad(() => seqModel.call(input));
removeHook();
console.log("Hook removed: the next call prints nothing");
noGrad(() => seqModel.call(input));

console.log("\n=== Module System Complete ===");
