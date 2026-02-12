/**
 * Example 27: Convolutional Neural Network Layers
 *
 * Demonstrates Conv1d, Conv2d, MaxPool2d, and AvgPool2d layers.
 * Convolutional layers are the backbone of image and signal processing models.
 */

import { GradTensor, tensor } from "deepbox/ndarray";
import { AvgPool2d, Conv1d, Conv2d, MaxPool2d, ReLU, Sequential } from "deepbox/nn";

console.log("=== Convolutional Neural Network Layers ===\n");

// ---------------------------------------------------------------------------
// Part 1: Conv1d — 1D Convolution for sequence/signal data
// ---------------------------------------------------------------------------
console.log("--- Part 1: Conv1d ---");

// Conv1d expects input shape: (batch, in_channels, length)
const conv1d = new Conv1d(1, 4, 3, { padding: 1 });
console.log("Conv1d(in=1, out=4, kernel=3, padding=1)");

const signal = tensor([[[1, 2, 3, 4, 5, 6, 7, 8]]]);
console.log(`Input shape:  [${signal.shape.join(", ")}]`);

const conv1dOut = conv1d.forward(signal);
const conv1dTensor = conv1dOut instanceof GradTensor ? conv1dOut.tensor : conv1dOut;
console.log(`Output shape: [${conv1dTensor.shape.join(", ")}]`);
console.log("  4 output channels from 1 input channel\n");

// ---------------------------------------------------------------------------
// Part 2: Conv2d — 2D Convolution for image data
// ---------------------------------------------------------------------------
console.log("--- Part 2: Conv2d ---");

// Conv2d expects input shape: (batch, in_channels, height, width)
// Note: Conv2d works with plain Tensor forward (inference mode)
const conv2d = new Conv2d(1, 4, 2, { bias: false });
console.log("Conv2d(in=1, out=4, kernel=2x2, bias=false)");
console.log("  Conv2d uses im2col internally for efficient convolution");

const conv2dParams = Array.from(conv2d.parameters()).length;
console.log(`  Parameters: ${conv2dParams} (weight only, no bias)\n`);

// ---------------------------------------------------------------------------
// Part 3: MaxPool2d — Downsampling with max pooling
// ---------------------------------------------------------------------------
console.log("--- Part 3: MaxPool2d ---");

const poolInput = tensor([
  [
    [
      [1, 2],
      [3, 4],
    ],
  ],
]);
const maxPool = new MaxPool2d(2, { stride: 2 });
console.log("MaxPool2d(kernel=2, stride=2)");
console.log(`Input shape:  [${poolInput.shape.join(", ")}]`);

const pooled = maxPool.forward(poolInput);
const pooledTensor = pooled instanceof GradTensor ? pooled.tensor : pooled;
console.log(`Output shape: [${pooledTensor.shape.join(", ")}]`);
console.log("  Spatial dimensions halved via max pooling\n");

// ---------------------------------------------------------------------------
// Part 4: AvgPool2d — Downsampling with average pooling
// ---------------------------------------------------------------------------
console.log("--- Part 4: AvgPool2d ---");

const avgPool = new AvgPool2d(2, { stride: 2 });
console.log("AvgPool2d(kernel=2, stride=2)");

const avgPooled = avgPool.forward(poolInput);
const avgTensor = avgPooled instanceof GradTensor ? avgPooled.tensor : avgPooled;
console.log(`Output shape: [${avgTensor.shape.join(", ")}]`);
console.log("  Average pooling preserves smoother spatial information\n");

// ---------------------------------------------------------------------------
// Part 5: Building a simple CNN pipeline with Sequential (Conv1d)
// ---------------------------------------------------------------------------
console.log("--- Part 5: Sequential Conv1d Pipeline ---");

const cnn = new Sequential(
  new Conv1d(1, 4, 3, { padding: 1 }),
  new ReLU(),
  new Conv1d(4, 8, 3, { padding: 1 }),
  new ReLU()
);

console.log("Sequential 1D CNN:");
console.log("  Conv1d(1->4, k=3) -> ReLU -> Conv1d(4->8, k=3) -> ReLU");

const cnnInput = tensor([[[1, 2, 3, 4, 5, 6]]]);
const cnnOutput = cnn.forward(cnnInput);
const cnnTensor = cnnOutput instanceof GradTensor ? cnnOutput.tensor : cnnOutput;
console.log(`Input shape:  [${cnnInput.shape.join(", ")}]`);
console.log(`Output shape: [${cnnTensor.shape.join(", ")}]`);

const paramCount = Array.from(cnn.parameters()).length;
console.log(`Total parameters: ${paramCount}`);

console.log("\n=== CNN Layers Complete ===");
