/**
 * Example 28: Recurrent Neural Network Layers
 *
 * Demonstrates RNN, LSTM, and GRU layers for sequence modeling.
 * Recurrent layers process sequential data by maintaining hidden state across time steps.
 */

import { GradTensor, tensor } from "deepbox/ndarray";
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

const rnnResult = rnn.forward(rnnInput);
const rnnOut = rnnResult instanceof GradTensor ? rnnResult.tensor : rnnResult;
console.log(`Output shape: [${rnnOut.shape.join(", ")}]`);
console.log("  Output contains hidden states for all time steps\n");

// ---------------------------------------------------------------------------
// Part 2: LSTM (Long Short-Term Memory)
// ---------------------------------------------------------------------------
console.log("--- Part 2: LSTM ---");

// LSTM adds cell state for better long-range dependencies
const lstm = new LSTM(4, 8, { batchFirst: true });
console.log("LSTM(inputSize=4, hiddenSize=8, batchFirst=true)");
console.log(`Input shape:  [${rnnInput.shape.join(", ")}]`);

const lstmResult = lstm.forward(rnnInput);
const lstmOut = lstmResult instanceof GradTensor ? lstmResult.tensor : lstmResult;
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

const gruResult = gru.forward(rnnInput);
const gruOut = gruResult instanceof GradTensor ? gruResult.tensor : gruResult;
console.log(`Output shape: [${gruOut.shape.join(", ")}]`);
console.log("  GRU uses reset/update gates — fewer params than LSTM\n");

// ---------------------------------------------------------------------------
// Part 4: Multi-layer RNN
// ---------------------------------------------------------------------------
console.log("--- Part 4: Multi-Layer Stacking ---");

const deepRnn = new RNN(4, 16, { numLayers: 2, batchFirst: true });
console.log("RNN(inputSize=4, hiddenSize=16, numLayers=2)");
console.log(`Input shape:  [${rnnInput.shape.join(", ")}]`);

const deepResult = deepRnn.forward(rnnInput);
const deepOut = deepResult instanceof GradTensor ? deepResult.tensor : deepResult;
console.log(`Output shape: [${deepOut.shape.join(", ")}]`);
console.log("  2-layer RNN extracts higher-level sequential patterns\n");

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

const singleResult = rnn.forward(singleSeq);
const singleOut = singleResult instanceof GradTensor ? singleResult.tensor : singleResult;
console.log(`Output shape: [${singleOut.shape.join(", ")}]`);
console.log("  2D input is treated as unbatched sequence\n");

// ---------------------------------------------------------------------------
// Part 6: Parameter counts
// ---------------------------------------------------------------------------
console.log("--- Part 6: Parameter Comparison ---");
const rnnParams = Array.from(rnn.parameters()).length;
const lstmParams = Array.from(lstm.parameters()).length;
const gruParams = Array.from(gru.parameters()).length;
console.log(`RNN  parameters: ${rnnParams}`);
console.log(`LSTM parameters: ${lstmParams} (4x gates)`);
console.log(`GRU  parameters: ${gruParams} (3x gates)`);

console.log("\n=== Recurrent Layers Complete ===");
