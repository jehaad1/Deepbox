/**
 * Example 30: Normalization & Dropout Layers
 *
 * Demonstrates BatchNorm1d, LayerNorm, and Dropout for training stability
 * and regularization. These layers are essential for deep network training.
 */

import { GradTensor, tensor } from "deepbox/ndarray";
import { BatchNorm1d, Dropout, LayerNorm } from "deepbox/nn";

console.log("=== Normalization & Dropout Layers ===\n");

// ---------------------------------------------------------------------------
// Part 1: BatchNorm1d — Normalize over the batch dimension
// ---------------------------------------------------------------------------
console.log("--- Part 1: BatchNorm1d ---");

// BatchNorm1d(numFeatures) — normalizes each feature across the batch
const bn = new BatchNorm1d(3);
console.log("BatchNorm1d(numFeatures=3)");
console.log("  Formula: y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta\n");

// Input shape: (batch, features)
const bnInput = tensor([
  [10, 20, 30],
  [11, 22, 28],
  [9, 18, 32],
  [12, 21, 29],
]);
console.log(`Input shape: [${bnInput.shape.join(", ")}]`);
console.log(`Input:\n${bnInput.toString()}`);

// Training mode: uses batch statistics
bn.train();
const bnOut = bn.forward(bnInput);
const bnTensor = bnOut instanceof GradTensor ? bnOut.tensor : bnOut;
console.log(`\nOutput (training mode): shape [${bnTensor.shape.join(", ")}]`);
console.log("  Uses batch mean/variance, updates running statistics\n");

// Eval mode: uses running statistics
bn.eval();
const bnEvalOut = bn.forward(bnInput);
const bnEvalTensor = bnEvalOut instanceof GradTensor ? bnEvalOut.tensor : bnEvalOut;
console.log(`Output (eval mode): shape [${bnEvalTensor.shape.join(", ")}]`);
console.log("  Uses accumulated running mean/variance\n");

// ---------------------------------------------------------------------------
// Part 2: LayerNorm — Normalize over the feature dimension
// ---------------------------------------------------------------------------
console.log("--- Part 2: LayerNorm ---");

// LayerNorm normalizes over the last dimension(s)
const ln = new LayerNorm(3);
console.log("LayerNorm(normalizedShape=3)");
console.log("  Normalizes each sample independently across features\n");

const lnOut = ln.forward(bnInput);
const lnTensor = lnOut instanceof GradTensor ? lnOut.tensor : lnOut;
console.log(`Input shape:  [${bnInput.shape.join(", ")}]`);
console.log(`Output shape: [${lnTensor.shape.join(", ")}]`);
console.log("  LayerNorm is batch-size independent (used in Transformers)\n");

// ---------------------------------------------------------------------------
// Part 3: Dropout — Regularization by random zeroing
// ---------------------------------------------------------------------------
console.log("--- Part 3: Dropout ---");

const dropout = new Dropout(0.5);
console.log("Dropout(p=0.5) — drops 50% of elements during training");

const dropInput = tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]);

// Training mode: randomly zeros elements
dropout.train();
console.log("\nTraining mode:");
const dropOut1 = dropout.forward(dropInput);
const drop1 = dropOut1 instanceof GradTensor ? dropOut1.tensor : dropOut1;
console.log(`  Output: ${drop1.toString()}`);
console.log("  Surviving elements are scaled by 1/(1-p) = 2.0");

// Eval mode: passes input unchanged
dropout.eval();
const dropOut2 = dropout.forward(dropInput);
const drop2 = dropOut2 instanceof GradTensor ? dropOut2.tensor : dropOut2;
console.log("\nEval mode:");
console.log(`  Output: ${drop2.toString()}`);
console.log("  Input passed through unchanged\n");

// ---------------------------------------------------------------------------
// Part 4: Parameter counts
// ---------------------------------------------------------------------------
console.log("--- Part 4: Parameter Counts ---");
const bnParams = Array.from(bn.parameters()).length;
const lnParams = Array.from(ln.parameters()).length;
const dropParams = Array.from(dropout.parameters()).length;
console.log(`BatchNorm1d(3) params: ${bnParams} (gamma + beta)`);
console.log(`LayerNorm(3)   params: ${lnParams} (weight + bias)`);
console.log(`Dropout(0.5)   params: ${dropParams} (no learnable params)`);

console.log("\n=== Normalization & Dropout Complete ===");
