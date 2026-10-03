/**
 * Example 29: Attention & Transformer Layers
 *
 * MultiheadAttention and TransformerEncoderLayer on a short sequence.
 * Attention lets each position in a sequence weigh every other position.
 *
 * The forward passes are inference only, so they run inside noGrad() and
 * return plain tensors.
 */

import { noGrad, tensor } from "deepbox/ndarray";
import { causalMask, MultiheadAttention, TransformerEncoderLayer } from "deepbox/nn";

console.log("=== Attention & Transformer Layers ===\n");

// ---------------------------------------------------------------------------
// Part 1: Multi-Head Attention
// ---------------------------------------------------------------------------
console.log("--- Part 1: Multi-Head Attention ---");

// MultiheadAttention(embedDim, numHeads)
// embedDim must be divisible by numHeads
const mha = new MultiheadAttention(8, 2);
console.log("MultiheadAttention(embedDim=8, numHeads=2)");
console.log("  Each head has dimension 8/2 = 4\n");

// Input: (batch, seqLen, embedDim)
// Self-attention: query = key = value = same input
const seqData = tensor([
  [
    [1, 0, 1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 1, 0, 1],
    [1, 1, 0, 0, 1, 1, 0, 0],
  ],
]);
console.log(`Input shape: [${seqData.shape.join(", ")}]  (batch=1, seq=3, embed=8)`);

// Self-attention: Q=K=V=input
const attnOut = noGrad(() => mha.forward(seqData, seqData, seqData));
console.log(`Output shape: [${attnOut.shape.join(", ")}]`);
console.log("  Each position attends to all other positions\n");

// ---------------------------------------------------------------------------
// Part 2: Attention weights, padding mask and causal mask
// ---------------------------------------------------------------------------
console.log("--- Part 2: Attention Weights and Masks ---");

// needWeights: true returns [output, weights]. The weights are averaged over heads
// and have shape (batch, seqLenQuery, seqLenKey). Each row sums to 1.
const [, weights] = noGrad(() =>
  mha.forward(seqData, seqData, seqData, undefined, { needWeights: true })
);
console.log(`Weights shape: [${weights.shape.join(", ")}]`);
console.log(`Row sums: ${weights.sum(-1).toString()}`);

// keyPaddingMask marks key positions to ignore: true means padded.
// Here the last position is padding, so no query attends to it.
const keyPaddingMask = tensor([[false, false, true]]);
const [, maskedWeights] = noGrad(() =>
  mha.forward(seqData, seqData, seqData, undefined, { needWeights: true, keyPaddingMask })
);
console.log(`Weight on the padded position: ${maskedWeights.slice({}, {}, 2).max().item()}`);

// causalMask(n) lets position i attend only to positions up to i
const [, causalWeights] = noGrad(() =>
  mha.forward(seqData, seqData, seqData, causalMask(3), { needWeights: true })
);
console.log(`Causal weights, first query row: ${causalWeights.slice({}, 0).toString()}\n`);

// ---------------------------------------------------------------------------
// Part 3: TransformerEncoderLayer
// ---------------------------------------------------------------------------
console.log("--- Part 3: TransformerEncoderLayer ---");

// TransformerEncoderLayer combines:
//   MultiheadAttention + FeedForward + LayerNorm + Dropout
const encoderLayer = new TransformerEncoderLayer(8, 2, 16);
console.log("TransformerEncoderLayer(dModel=8, nHead=2, dimFeedforward=16)");
console.log(`Input shape: [${seqData.shape.join(", ")}]`);

const encoderOut = noGrad(() => encoderLayer.forward(seqData));
console.log(`Output shape: [${encoderOut.shape.join(", ")}]`);
console.log(
  "  Attention and feed-forward sub-layers, each with a residual connection and LayerNorm"
);

// activation: "gelu" and normFirst: true select a GELU feed-forward network and the pre-norm layout
const preNormLayer = new TransformerEncoderLayer(8, 2, 16, { activation: "gelu", normFirst: true });
console.log(
  `Pre-norm layer output shape: [${noGrad(() => preNormLayer.forward(seqData)).shape.join(", ")}]\n`
);

// ---------------------------------------------------------------------------
// Part 4: Parameter counts
// ---------------------------------------------------------------------------
console.log("--- Part 4: Parameter Counts ---");
const mhaParams = Array.from(mha.parameters()).length;
const encParams = Array.from(encoderLayer.parameters()).length;
console.log(`MultiheadAttention parameter tensors: ${mhaParams}`);
console.log(`TransformerEncoderLayer parameter tensors: ${encParams}`);
console.log("  The encoder layer adds the feed-forward network and two LayerNorms");

console.log("\n=== Attention & Transformer Complete ===");
