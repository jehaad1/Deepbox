/**
 * Example 29: Attention & Transformer Layers
 *
 * Demonstrates MultiheadAttention and TransformerEncoderLayer.
 * Attention mechanisms allow models to focus on relevant parts of the input sequence.
 */

import { GradTensor, tensor } from "deepbox/ndarray";
import { MultiheadAttention, TransformerEncoderLayer } from "deepbox/nn";

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
const attnOut = mha.forward(seqData, seqData, seqData);
const attnShape = attnOut instanceof GradTensor ? attnOut.tensor.shape : attnOut.shape;
console.log(`Output shape: [${attnShape.join(", ")}]`);
console.log("  Each position attends to all other positions\n");

// ---------------------------------------------------------------------------
// Part 2: TransformerEncoderLayer
// ---------------------------------------------------------------------------
console.log("--- Part 2: TransformerEncoderLayer ---");

// TransformerEncoderLayer combines:
//   MultiheadAttention + FeedForward + LayerNorm + Dropout
const encoderLayer = new TransformerEncoderLayer(8, 2, 16);
console.log("TransformerEncoderLayer(dModel=8, nHead=2, dimFeedforward=16)");
console.log(`Input shape: [${seqData.shape.join(", ")}]`);

const encoderOut = encoderLayer.forward(seqData);
const encShape = encoderOut instanceof GradTensor ? encoderOut.tensor.shape : encoderOut.shape;
console.log(`Output shape: [${encShape.join(", ")}]`);
console.log("  Full transformer encoder block with residual connections\n");

// ---------------------------------------------------------------------------
// Part 3: Parameter inspection
// ---------------------------------------------------------------------------
console.log("--- Part 3: Parameter Counts ---");
const mhaParams = Array.from(mha.parameters()).length;
const encParams = Array.from(encoderLayer.parameters()).length;
console.log(`MultiheadAttention params: ${mhaParams}`);
console.log(`TransformerEncoderLayer params: ${encParams}`);
console.log("  Encoder layer includes attention + feedforward + normalization");

console.log("\n=== Attention & Transformer Complete ===");
