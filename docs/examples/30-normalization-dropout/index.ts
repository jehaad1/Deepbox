/**
 * Example 30: Normalization & Dropout Layers
 *
 * BatchNorm1d, LayerNorm and Dropout: two normalization layers and one
 * regularizer, each shown in train and eval mode.
 *
 * eval() changes the layer's behavior but does not turn gradient tracking off.
 * For inference, run the forward pass inside noGrad() to get a plain tensor.
 * The training-mode outputs below are GradTensors, which share .shape,
 * .mean() and .toString() with Tensor.
 */

import { type AnyTensor, noGrad, tensor } from "deepbox/ndarray";
import { BatchNorm1d, Dropout, LayerNorm } from "deepbox/nn";

console.log("=== Normalization & Dropout Layers ===\n");

// Values of a small tensor as numbers rounded to 4 decimals
const rounded = (t: AnyTensor): number[] => {
  const out: number[] = [];
  for (const v of t.data) out.push(Math.round(Number(v) * 1e4) / 1e4);
  return out;
};

// ---------------------------------------------------------------------------
// Part 1: BatchNorm1d (normalize over the batch dimension)
// ---------------------------------------------------------------------------
console.log("--- Part 1: BatchNorm1d ---");

// BatchNorm1d(numFeatures): normalizes each feature across the batch
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
console.log(`\nOutput (training mode): shape [${bnOut.shape.join(", ")}]`);
console.log(`  Mean of each feature: [${rounded(bnOut.mean(0)).join(", ")}]  (0 up to rounding)`);
console.log("  Uses batch mean/variance, updates running statistics\n");

// Eval mode: uses running statistics
bn.eval();
const bnEvalOut = noGrad(() => bn.forward(bnInput));
console.log(`Output (eval mode): shape [${bnEvalOut.shape.join(", ")}]`);
console.log("  Uses the running mean/variance, which one training step has only started to fill\n");

// ---------------------------------------------------------------------------
// Part 2: LayerNorm (normalize over the feature dimension)
// ---------------------------------------------------------------------------
console.log("--- Part 2: LayerNorm ---");

// LayerNorm normalizes over the last dimension(s)
const ln = new LayerNorm(3);
console.log("LayerNorm(normalizedShape=3)");
console.log("  Normalizes each sample independently across features\n");

const lnOut = noGrad(() => ln.forward(bnInput));
console.log(`Input shape:  [${bnInput.shape.join(", ")}]`);
console.log(`Output shape: [${lnOut.shape.join(", ")}]`);
console.log(`Mean of each sample: [${rounded(lnOut.mean(1)).join(", ")}]  (0 up to rounding)`);
console.log("  The result for one sample does not depend on the rest of the batch\n");

// ---------------------------------------------------------------------------
// Part 3: Dropout (regularization by random zeroing)
// ---------------------------------------------------------------------------
console.log("--- Part 3: Dropout ---");

const dropout = new Dropout(0.5);
console.log("Dropout(p=0.5) drops 50% of elements during training");

const dropInput = tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]);

// Training mode: randomly zeros elements. Dropout has no parameters, so a plain
// tensor in gives a plain tensor out.
dropout.train();
console.log("\nTraining mode:");
const drop1 = dropout.forward(dropInput);
console.log(`  Output: ${drop1.toString()}`);
console.log("  Surviving elements are scaled by 1/(1-p) = 2.0");

// Eval mode: passes input unchanged
dropout.eval();
const drop2 = dropout.forward(dropInput);
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
console.log(`BatchNorm1d(3) parameter tensors: ${bnParams} (gamma and beta)`);
console.log(`LayerNorm(3)   parameter tensors: ${lnParams} (weight and bias)`);
console.log(`Dropout(0.5)   parameter tensors: ${dropParams} (nothing to learn)`);

console.log("\n=== Normalization & Dropout Complete ===");
