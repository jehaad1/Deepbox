/**
 * Example 28: Recurrent Neural Network Layers
 *
 * RNN, LSTM and GRU layers on a small batch of sequences, with output shapes
 * and parameter counts. Recurrent layers carry a hidden state across time steps.
 *
 * The forward passes are inference only, so they run inside noGrad() and
 * return plain tensors. Without it, the layers return a GradTensor that
 * tracks the weights. Both kinds of tensor have .shape.
 */

import { noGrad, tensor } from "deepbox/ndarray";
import { GRU, LSTM, RNN } from "deepbox/nn";

console.log("=== Recurrent Neural Network Layers ===\n");

// ---------------------------------------------------------------------------
// Part 1: Simple RNN
// ---------------------------------------------------------------------------
console.log("--- Part 1: Simple RNN ---");

// RNN(inputSize, hiddenSize, options)
// Input shape (batchFirst=true): (batch, seqLen, inputSize)
const rnn = new RNN(4, 8, { batchFirst: true });
console.log("RNN(inputSize=4, hiddenSize=8, batchFirst=true)");

// Batch of 2 sequences, each with 3 time steps and 4 features
const rnnInput = tensor([
  [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
  ],
  [
    [13, 14, 15, 16],
    [17, 18, 19, 20],
    [21, 22, 23, 24],
  ],
]);
console.log(`Input shape:  [${rnnInput.shape.join(", ")}]`);

const rnnOut = noGrad(() => rnn.forward(rnnInput));
console.log(`Output shape: [${rnnOut.shape.join(", ")}]`);
console.log("  The output holds the hidden state at every time step\n");

// ---------------------------------------------------------------------------
// Part 2: LSTM (Long Short-Term Memory)
// ---------------------------------------------------------------------------
console.log("--- Part 2: LSTM ---");

// LSTM adds cell state for better long-range dependencies
const lstm = new LSTM(4, 8, { batchFirst: true });
console.log("LSTM(inputSize=4, hiddenSize=8, batchFirst=true)");
console.log(`Input shape:  [${rnnInput.shape.join(", ")}]`);

const lstmOut = noGrad(() => lstm.forward(rnnInput));
console.log(`Output shape: [${lstmOut.shape.join(", ")}]`);
console.log("  LSTM uses forget/input/output gates for selective memory\n");

// ---------------------------------------------------------------------------
// Part 3: GRU (Gated Recurrent Unit)
// ---------------------------------------------------------------------------
console.log("--- Part 3: GRU ---");

// GRU is a simplified version of LSTM with fewer parameters
const gru = new GRU(4, 8, { batchFirst: true });
console.log("GRU(inputSize=4, hiddenSize=8, batchFirst=true)");
console.log(`Input shape:  [${rnnInput.shape.join(", ")}]`);

const gruOut = noGrad(() => gru.forward(rnnInput));
console.log(`Output shape: [${gruOut.shape.join(", ")}]`);
console.log("  GRU uses reset/update gates, fewer parameters than LSTM\n");

// ---------------------------------------------------------------------------
// Part 4: Multi-layer RNN
// ---------------------------------------------------------------------------
console.log("--- Part 4: Multi-Layer Stacking ---");

const deepRnn = new RNN(4, 16, { numLayers: 2, batchFirst: true });
console.log("RNN(inputSize=4, hiddenSize=16, numLayers=2)");
console.log(`Input shape:  [${rnnInput.shape.join(", ")}]`);

const deepOut = noGrad(() => deepRnn.forward(rnnInput));
console.log(`Output shape: [${deepOut.shape.join(", ")}]`);
console.log("  2-layer RNN feeds the first layer's hidden states into a second layer\n");

// ---------------------------------------------------------------------------
// Part 5: Unbatched (single sequence) input
// ---------------------------------------------------------------------------
console.log("--- Part 5: Unbatched Input ---");

const singleSeq = tensor([
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9, 10, 11, 12],
]);
console.log("Single sequence (no batch dim):");
console.log(`Input shape:  [${singleSeq.shape.join(", ")}]`);

const singleOut = noGrad(() => rnn.forward(singleSeq));
console.log(`Output shape: [${singleOut.shape.join(", ")}]`);
console.log("  2D input is treated as unbatched sequence\n");

// ---------------------------------------------------------------------------
// Part 6: Parameter counts
// ---------------------------------------------------------------------------
console.log("--- Part 6: Weight Counts ---");
// Count scalar weights, not parameter tensors: every layer has 4 tensors
// (input weights, hidden weights and two biases), but the tensors differ in size.
const countWeights = (layer: { parameters(): Iterable<{ size: number }> }): number =>
  Array.from(layer.parameters()).reduce((total, p) => total + p.size, 0);

const rnnWeights = countWeights(rnn);
const lstmWeights = countWeights(lstm);
const gruWeights = countWeights(gru);
console.log(`RNN  weights: ${rnnWeights}  (1 transform per step)`);
console.log(`LSTM weights: ${lstmWeights}  (4 gates, ${lstmWeights / rnnWeights}x the RNN)`);
console.log(`GRU  weights: ${gruWeights}  (3 gates, ${gruWeights / rnnWeights}x the RNN)`);

console.log("\n=== Recurrent Layers Complete ===");
