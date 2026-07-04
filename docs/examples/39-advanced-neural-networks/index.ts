/**
 * Example 39: Advanced Neural Networks
 *
 * New in v1.0.0: Trainer with EarlyStopping, weight initialization functions,
 * Embedding layers, normalization layers, containers (ModuleList, ModuleDict),
 * and advanced activation functions (GELU, PReLU).
 */

import { type AnyTensor, type Tensor, tensor } from "deepbox/ndarray";
import {
  Dropout,
  EarlyStopping,
  Embedding,
  GELU,
  GroupNorm,
  kaiming_normal_,
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
  xavier_uniform_,
  zeros_,
} from "deepbox/nn";
import { Adam } from "deepbox/optim";

console.log("=".repeat(60));
console.log("Example 39: Advanced Neural Networks");
console.log("=".repeat(60));

// ============================================================================
// Part 1: Weight Initialization
// ============================================================================
console.log("\n🎲 Part 1: Weight Initialization");
console.log("-".repeat(60));

// Weight initialization is crucial for training stability
const layer = new Linear(64, 32);

// Xavier/Glorot uniform — best for sigmoid/tanh activations
xavier_uniform_(layer.getWeight());
if (layer.getBias()) zeros_(layer.getBias()!);
console.log("xavier_uniform_ applied to Linear(64, 32):");
console.log(`  Weight shape: ${layer.getWeight().shape}, Bias shape: ${layer.getBias()?.shape}`);

// Kaiming/He normal — best for ReLU activations
const reluLayer = new Linear(128, 64);
kaiming_normal_(reluLayer.getWeight(), 0, "fan_in", "relu");
if (reluLayer.getBias()) zeros_(reluLayer.getBias()!);
console.log("kaiming_normal_ applied to Linear(128, 64):");
console.log(`  Weight shape: ${reluLayer.getWeight().shape}`);

// ============================================================================
// Part 2: Advanced Activation Functions
// ============================================================================
console.log("\n⚡ Part 2: Advanced Activations (GELU, PReLU)");
console.log("-".repeat(60));

// GELU — Gaussian Error Linear Unit, used in Transformers
const gelu = new GELU();
const geluInput = tensor([-2, -1, 0, 1, 2]);
const geluOutput = gelu.forward(geluInput);
console.log("GELU activation:");
console.log(`  Input:  ${geluInput.toString()}`);
console.log(`  Output: ${geluOutput.toString()}`);

// PReLU — Parametric ReLU with learnable negative slope
const prelu = new PReLU();
const preluInput = tensor([-2, -1, 0, 1, 2], { dtype: "float64" });
const preluOutput = prelu.forward(preluInput);
console.log("\nPReLU activation (learnable slope):");
console.log(`  Input:  ${preluInput.toString()}`);
console.log(`  Output: ${preluOutput.toString()}`);

// ============================================================================
// Part 3: Normalization Layers
// ============================================================================
console.log("\n📏 Part 3: Normalization Layers");
console.log("-".repeat(60));

// LayerNorm — normalizes across features (used in Transformers)
const ln = new LayerNorm([4]);
const lnInput = tensor([
  [1, 2, 3, 4],
  [5, 6, 7, 8],
]);
const lnOutput = ln.forward(lnInput);
console.log("LayerNorm([4]):");
console.log(`  Input:  ${lnInput.toString()}`);
console.log(`  Output: ${lnOutput.toString()}`);

// GroupNorm — normalizes within groups of channels
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
console.log(`  Input shape: ${gnInput.shape}`);
console.log(`  Output shape: ${gnOutput.shape}`);

// ============================================================================
// Part 4: Embedding Layer
// ============================================================================
console.log("\n📖 Part 4: Embedding Layer");
console.log("-".repeat(60));

// Embedding maps integer indices to dense vectors (used for words, tokens, etc.)
const vocabSize = 10;
const embeddingDim = 4;
const emb = new Embedding(vocabSize, embeddingDim);

// Look up embeddings for token indices
const tokenIds = tensor([0, 3, 7, 1]);
const embeddings = emb.forward(tokenIds);
console.log(`Embedding(vocab=${vocabSize}, dim=${embeddingDim}):`);
console.log(`  Token IDs: ${tokenIds.toString()}`);
console.log(`  Embeddings shape: ${embeddings.shape}`);
console.log(`  Each token → ${embeddingDim}D vector`);

// ============================================================================
// Part 5: Module Containers (ModuleList, ModuleDict)
// ============================================================================
console.log("\n📦 Part 5: Module Containers");
console.log("-".repeat(60));

// ModuleList — ordered list of modules (proper parameter tracking)
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
  console.log(`  ${name}: ${param.shape}`);
}
console.log(`  Total parameters: ${totalParams}`);

// ModuleDict — dictionary of named modules
const branches = new ModuleDict({
  encoder: new Sequential(new Linear(10, 8), new ReLU()),
  decoder: new Sequential(new Linear(8, 10), new ReLU()),
});

console.log("\nModuleDict with encoder/decoder branches:");
for (const [name, param] of branches.namedParameters()) {
  console.log(`  ${name}: ${param.shape}`);
}

// ============================================================================
// Part 6: Sequential with Dropout
// ============================================================================
console.log("\n🏗️  Part 6: Sequential Model with Dropout");
console.log("-".repeat(60));

// Build a multi-layer model with dropout regularization
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
  console.log(`  ${name}: ${param.shape}`);
}
console.log(`  Total parameters: ${paramCount}`);

// Forward pass
const xDemo = tensor([
  [1, 2, 3, 4],
  [5, 6, 7, 8],
]);
model.eval(); // disable dropout for inference
const yDemo = model.forward(xDemo);
console.log(`\n  Input shape:  ${xDemo.shape}`);
console.log(`  Output shape: ${yDemo.shape}`);

// ============================================================================
// Part 7: Trainer with EarlyStopping
// ============================================================================
console.log("\n🏋️  Part 7: Trainer with EarlyStopping");
console.log("-".repeat(60));

// Create a simple regression model
const trainerModel = new Sequential(new Linear(4, 16), new ReLU(), new Linear(16, 1));

const optimizer = new Adam(trainerModel.parameters(), { lr: 0.01 });
const lossFn = (pred: AnyTensor, target: Tensor) => mseLoss(pred, target);

// The Trainer manages the training loop with callbacks
const trainer = new Trainer(trainerModel, optimizer, lossFn, {
  epochs: 50,
  earlyStopping: { patience: 5, minDelta: 0.001 },
  verbose: false,
});

// Generate simple training data: y = sum(x) + noise
const trainBatches: [ReturnType<typeof tensor>, ReturnType<typeof tensor>][] = [];
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
  `  Final train loss: ${result.history[result.history.length - 1]?.trainLoss.toFixed(6)}`
);
if (result.stoppedEarly) {
  console.log(`  Early stopping triggered at epoch ${result.bestEpoch}`);
} else {
  console.log("  Completed all epochs");
}

// ============================================================================
// Part 8: EarlyStopping & ModelCheckpoint (standalone)
// ============================================================================
console.log("\n💾 Part 8: EarlyStopping & ModelCheckpoint");
console.log("-".repeat(60));

// EarlyStopping monitors a metric and stops when it stops improving
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

// ModelCheckpoint saves and restores the best model state
const checkpointModel = new Sequential(new Linear(4, 2));
const checkpoint = new ModelCheckpoint({ mode: "min" });

console.log("\nModelCheckpoint (saves best model state):");
const checkLosses = [1.0, 0.8, 0.9, 0.7, 0.75];
for (let i = 0; i < checkLosses.length; i++) {
  const improved = checkpoint.step(checkpointModel, checkLosses[i]!);
  console.log(`  Epoch ${i + 1}: loss=${checkLosses[i]!.toFixed(2)}, saved=${improved}`);
}

// Restore best model weights
checkpoint.restore(checkpointModel);
console.log("  Best model restored!");

// ============================================================================
// Summary
// ============================================================================
console.log("\n💡 Key Takeaways");
console.log("-".repeat(60));
console.log("• xavier_uniform_/kaiming_normal_: proper initialization for stable training");
console.log("• GELU: smooth activation used in modern Transformers");
console.log("• PReLU: learnable negative slope for adaptive activation");
console.log("• LayerNorm/GroupNorm: normalization layers for different architectures");
console.log("• Embedding: maps discrete indices to dense learned vectors");
console.log("• ModuleList/ModuleDict: containers with proper parameter tracking");
console.log("• Trainer: managed training loop with epochs, callbacks, logging");
console.log("• EarlyStopping: prevents overfitting by monitoring validation metrics");
console.log("• ModelCheckpoint: saves and restores the best model weights");

console.log("\n✅ Advanced Neural Networks Example Complete!");
console.log("=".repeat(60));
