/**
 * Example 39: Advanced Neural Networks
 *
 * Weight initialization, GELU and PReLU, LayerNorm and GroupNorm, Embedding,
 * ModuleList and ModuleDict, and the Trainer with EarlyStopping and ModelCheckpoint.
 */

import { type AnyTensor, noGrad, type Tensor, tensor } from "deepbox/ndarray";
import {
  Dropout,
  EarlyStopping,
  Embedding,
  GELU,
  GroupNorm,
  kaimingNormal_,
  LayerNorm,
  Linear,
  ModelCheckpoint,
  ModuleDict,
  ModuleList,
  mseLoss,
  PReLU,
  ReLU,
  Sequential,
  Trainer,
  xavierUniform_,
  zeros_,
} from "deepbox/nn";
import { Adam } from "deepbox/optim";
import { setSeed } from "deepbox/random";

console.log("=".repeat(60));
console.log("Example 39: Advanced Neural Networks");
console.log("=".repeat(60));

// A seed makes the initial weights, and so the training results, repeat between runs
setSeed(42);

// ============================================================================
// Part 1: Weight Initialization
// ============================================================================
console.log("\nPart 1: Weight Initialization");
console.log("-".repeat(60));

// The starting weights affect how fast a network trains. These functions fill a tensor in place.
const layer = new Linear(64, 32);

// Xavier (Glorot) uniform keeps the variance of activations steady, a common choice for sigmoid and tanh
xavierUniform_(layer.getWeight());
if (layer.getBias()) zeros_(layer.getBias()!);
console.log("xavierUniform_ applied to Linear(64, 32):");
console.log(
  `  Weight shape: [${layer.getWeight().shape.join(", ")}], Bias shape: [${layer.getBias()?.shape.join(", ")}]`
);

// Kaiming (He) normal accounts for ReLU zeroing half of its inputs
const reluLayer = new Linear(128, 64);
kaimingNormal_(reluLayer.getWeight(), 0, "fan_in", "relu");
if (reluLayer.getBias()) zeros_(reluLayer.getBias()!);
console.log("kaimingNormal_ applied to Linear(128, 64):");
console.log(`  Weight shape: [${reluLayer.getWeight().shape.join(", ")}]`);

// ============================================================================
// Part 2: Advanced Activation Functions
// ============================================================================
console.log("\nPart 2: Advanced Activations (GELU, PReLU)");
console.log("-".repeat(60));

// GELU (Gaussian Error Linear Unit) is common in Transformers. Deepbox uses the tanh
// approximation by default. PyTorch's default is the exact form, available as
// new GELU({ approximate: "none" }).
const gelu = new GELU();
const geluInput = tensor([-2, -1, 0, 1, 2]);
const geluOutput = gelu.forward(geluInput);
console.log("GELU activation (tanh approximation):");
console.log(`  Input:  ${geluInput.toString()}`);
console.log(`  Output: ${geluOutput.toString()}`);

const geluExact = new GELU({ approximate: "none" }).forward(geluInput);
console.log(`  Exact:  ${geluExact.toString()}`);

// PReLU is a ReLU whose negative slope is a learnable parameter
const prelu = new PReLU();
const preluInput = tensor([-2, -1, 0, 1, 2], { dtype: "float64" });
const preluOutput = prelu.forward(preluInput);
console.log("\nPReLU activation (learnable slope):");
console.log(`  Input:  ${preluInput.toString()}`);
console.log(`  Output: ${preluOutput.toString()}`);

// ============================================================================
// Part 3: Normalization Layers
// ============================================================================
console.log("\nPart 3: Normalization Layers");
console.log("-".repeat(60));

// LayerNorm normalizes each sample across its features (used in Transformers)
const ln = new LayerNorm([4]);
const lnInput = tensor([
  [1, 2, 3, 4],
  [5, 6, 7, 8],
]);
const lnOutput = ln.forward(lnInput);
console.log("LayerNorm([4]):");
console.log(`  Input:  ${lnInput.toString()}`);
console.log(`  Output: ${lnOutput.toString()}`);

// GroupNorm normalizes within groups of channels, so it does not depend on the batch size
const gn = new GroupNorm(2, 4); // 2 groups, 4 channels
const gnInput = tensor([
  [
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
  ],
]);
const gnOutput = gn.forward(gnInput);
console.log(`\nGroupNorm(2 groups, 4 channels):`);
console.log(`  Input shape: [${gnInput.shape.join(", ")}]`);
console.log(`  Output shape: [${gnOutput.shape.join(", ")}]`);

// ============================================================================
// Part 4: Embedding Layer
// ============================================================================
console.log("\nPart 4: Embedding Layer");
console.log("-".repeat(60));

// Embedding maps integer indices (words, tokens) to learned dense vectors
const vocabSize = 10;
const embeddingDim = 4;
const emb = new Embedding(vocabSize, embeddingDim);

// Look up embeddings for token indices
const tokenIds = tensor([0, 3, 7, 1]);
const embeddings = emb.forward(tokenIds);
console.log(`Embedding(vocab=${vocabSize}, dim=${embeddingDim}):`);
console.log(`  Token IDs: ${tokenIds.toString()}`);
console.log(`  Embeddings shape: [${embeddings.shape.join(", ")}]`);
console.log(`  Each token becomes a vector of ${embeddingDim} numbers`);

// ============================================================================
// Part 5: Module Containers (ModuleList, ModuleDict)
// ============================================================================
console.log("\nPart 5: Module Containers");
console.log("-".repeat(60));

// ModuleList holds modules in order and registers their parameters
const layers = new ModuleList([
  new Linear(32, 16),
  new ReLU(),
  new Linear(16, 8),
  new ReLU(),
  new Linear(8, 1),
]);

console.log("ModuleList with 5 layers:");
let totalParams = 0;
for (const [name, param] of layers.namedParameters()) {
  totalParams += param.size;
  console.log(`  ${name}: [${param.shape.join(", ")}]`);
}
console.log(`  Total parameters: ${totalParams}`);

// ModuleDict holds modules by name and registers their parameters
const branches = new ModuleDict({
  encoder: new Sequential(new Linear(10, 8), new ReLU()),
  decoder: new Sequential(new Linear(8, 10), new ReLU()),
});

console.log("\nModuleDict with encoder/decoder branches:");
for (const [name, param] of branches.namedParameters()) {
  console.log(`  ${name}: [${param.shape.join(", ")}]`);
}

// ============================================================================
// Part 6: Sequential with Dropout
// ============================================================================
console.log("\nPart 6: Sequential Model with Dropout");
console.log("-".repeat(60));

// A small model with dropout between the layers
const model = new Sequential(
  new Linear(4, 16),
  new ReLU(),
  new Dropout(0.2),
  new Linear(16, 8),
  new ReLU(),
  new Dropout(0.1),
  new Linear(8, 1)
);

console.log("Sequential model architecture:");
let paramCount = 0;
for (const [name, param] of model.namedParameters()) {
  paramCount += param.size;
  console.log(`  ${name}: [${param.shape.join(", ")}]`);
}
console.log(`  Total parameters: ${paramCount}`);

// Forward pass
const xDemo = tensor([
  [1, 2, 3, 4],
  [5, 6, 7, 8],
]);
model.eval(); // dropout off
const yDemo = noGrad(() => model.forward(xDemo)); // noGrad: no graph, plain tensor out
console.log(`\n  Input shape:  [${xDemo.shape.join(", ")}]`);
console.log(`  Output shape: [${yDemo.shape.join(", ")}]`);

// ============================================================================
// Part 7: Trainer with EarlyStopping
// ============================================================================
console.log("\nPart 7: Trainer with EarlyStopping");
console.log("-".repeat(60));

// Create a simple regression model
const trainerModel = new Sequential(new Linear(4, 16), new ReLU(), new Linear(16, 1));

const optimizer = new Adam(trainerModel.parameters(), { lr: 0.01 });
const lossFn = (pred: AnyTensor, target: Tensor) => mseLoss(pred, target);

// The Trainer runs the loop: zero gradients, forward, loss, backward, step.
// - earlyStopping stops when the monitored loss has not improved for `patience` epochs
// - restoreBestWeights loads the weights of the best epoch back into the model
// - accumulationSteps sums the gradients of several batches before each step
const trainer = new Trainer(trainerModel, optimizer, lossFn, {
  epochs: 50,
  earlyStopping: { patience: 5, minDelta: 0.001 },
  restoreBestWeights: true,
  accumulationSteps: 2,
  verbose: false,
});

// Training data: 10 batches of 2 samples, targets near 10
const trainBatches: [Tensor, Tensor][] = [];
for (let i = 0; i < 10; i++) {
  const xBatch = tensor([
    [1 + i * 0.1, 2 + i * 0.2, 3 - i * 0.1, 4 + i * 0.05],
    [2 + i * 0.1, 1 + i * 0.1, 4 - i * 0.2, 3 + i * 0.1],
  ]);
  const yBatch = tensor([[10 + i * 0.25], [10 + i * 0.0]]);
  trainBatches.push([xBatch, yBatch]);
}

const result = trainer.fit(trainBatches);

console.log("Trainer results:");
console.log(`  Total epochs run: ${result.history.length}`);
console.log(
  `  Train loss at the last epoch: ${result.history[result.history.length - 1]?.trainLoss.toFixed(6)}`
);
console.log(`  Best epoch (weights restored): ${result.bestEpoch}`);
console.log(result.stoppedEarly ? "  Stopped early" : "  Ran all epochs");

// ============================================================================
// Part 8: EarlyStopping & ModelCheckpoint (standalone)
// ============================================================================
console.log("\nPart 8: EarlyStopping & ModelCheckpoint");
console.log("-".repeat(60));

// EarlyStopping.step(value) returns true once the metric has not improved for `patience` calls
const earlyStop = new EarlyStopping({
  patience: 3,
  minDelta: 0.01,
  mode: "min",
});

const losses = [1.0, 0.8, 0.6, 0.55, 0.54, 0.54, 0.54, 0.53];
console.log("EarlyStopping (patience=3, minDelta=0.01, mode=min):");
for (let i = 0; i < losses.length; i++) {
  const shouldStop = earlyStop.step(losses[i]!);
  console.log(`  Epoch ${i + 1}: loss=${losses[i]!.toFixed(2)}, stop=${shouldStop}`);
  if (shouldStop) break;
}

// ModelCheckpoint remembers the model state of the best step
const checkpointModel = new Sequential(new Linear(4, 2));
const checkpoint = new ModelCheckpoint({ mode: "min" });

console.log("\nModelCheckpoint (keeps the best state):");
const checkLosses = [1.0, 0.8, 0.9, 0.7, 0.75];
for (let i = 0; i < checkLosses.length; i++) {
  const improved = checkpoint.step(checkpointModel, checkLosses[i]!);
  console.log(`  Epoch ${i + 1}: loss=${checkLosses[i]!.toFixed(2)}, saved=${improved}`);
}

// Load the best weights back into the model
checkpoint.restore(checkpointModel);
console.log("  Best weights restored");

// ============================================================================
// Summary
// ============================================================================
console.log("\nKey Takeaways");
console.log("-".repeat(60));
console.log("• xavierUniform_, kaimingNormal_: in-place weight initializers");
console.log('• GELU: tanh approximation by default, exact with { approximate: "none" }');
console.log("• PReLU: ReLU with a learnable negative slope");
console.log("• LayerNorm, GroupNorm: normalization that does not depend on the batch size");
console.log("• Embedding: integer indices to learned vectors");
console.log("• ModuleList, ModuleDict: containers that register their children's parameters");
console.log("• Trainer: epochs, early stopping, gradient accumulation and best-weight restore");
console.log("• EarlyStopping: stops when a monitored loss stops improving");
console.log("• ModelCheckpoint: keeps and restores the best model state");

console.log("\nAdvanced Neural Networks Example Complete!");
console.log("=".repeat(60));
