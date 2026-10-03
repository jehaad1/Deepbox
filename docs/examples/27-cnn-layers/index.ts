/**
 * Example 27: Convolutional Neural Network Layers
 *
 * Conv1d, Conv2d, MaxPool2d and AvgPool2d on small inputs, with the output
 * shapes printed. Convolutions are the building block of image and signal models.
 *
 * The forward passes below are inference only, so they run inside noGrad().
 * Without it, a layer with trainable weights returns a GradTensor that tracks
 * the weights. Both kinds of tensor have .shape and .toString().
 */

import { noGrad, tensor } from "deepbox/ndarray";
import { AvgPool2d, Conv1d, Conv2d, MaxPool2d, ReLU, Sequential } from "deepbox/nn";

console.log("=== Convolutional Neural Network Layers ===\n");

// ---------------------------------------------------------------------------
// Part 1: Conv1d (1D convolution for sequence and signal data)
// ---------------------------------------------------------------------------
console.log("--- Part 1: Conv1d ---");

// Conv1d expects input of shape (batch, inChannels, length)
const conv1d = new Conv1d(1, 4, 3, { padding: 1 });
console.log("Conv1d(in=1, out=4, kernel=3, padding=1)");

const signal = tensor([[[1, 2, 3, 4, 5, 6, 7, 8]]]);
console.log(`Input shape:  [${signal.shape.join(", ")}]`);

const conv1dOut = noGrad(() => conv1d.forward(signal));
console.log(`Output shape: [${conv1dOut.shape.join(", ")}]`);
console.log("  4 output channels from 1 input channel");

// padding: "same" keeps the length for any kernel size, "valid" adds no padding
const sameConv = new Conv1d(1, 4, 5, { padding: "same" });
const validConv = new Conv1d(1, 4, 5, { padding: "valid" });
console.log(
  `kernel=5, padding="same":  [${noGrad(() => sameConv.forward(signal)).shape.join(", ")}]`
);
console.log(
  `kernel=5, padding="valid": [${noGrad(() => validConv.forward(signal)).shape.join(", ")}]\n`
);

// ---------------------------------------------------------------------------
// Part 2: Conv2d (2D convolution for image data)
// ---------------------------------------------------------------------------
console.log("--- Part 2: Conv2d ---");

// Conv2d expects input of shape (batch, inChannels, height, width)
const conv2d = new Conv2d(1, 4, 2, { bias: false });
console.log("Conv2d(in=1, out=4, kernel=2x2, bias=false)");

const image = tensor([
  [
    [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
      [13, 14, 15, 16],
    ],
  ],
]);
console.log(`Input shape:  [${image.shape.join(", ")}]`);
const conv2dOut = noGrad(() => conv2d.forward(image));
console.log(`Output shape: [${conv2dOut.shape.join(", ")}]  (4 - 2 + 1 = 3 along each side)`);

const conv2dParams = Array.from(conv2d.parameters()).length;
console.log(`Parameters: ${conv2dParams} (weight only, no bias)\n`);

// ---------------------------------------------------------------------------
// Part 3: MaxPool2d (downsampling with max pooling)
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

// Pooling layers have no parameters, so the result is a plain tensor
const pooled = maxPool.forward(poolInput);
console.log(`Output shape: [${pooled.shape.join(", ")}]`);
console.log(`Output value: ${pooled.toString()}`);
console.log("  Each 2x2 window is replaced by its maximum\n");

// ---------------------------------------------------------------------------
// Part 4: AvgPool2d (downsampling with average pooling)
// ---------------------------------------------------------------------------
console.log("--- Part 4: AvgPool2d ---");

const avgPool = new AvgPool2d(2, { stride: 2 });
console.log("AvgPool2d(kernel=2, stride=2)");

const avgPooled = avgPool.forward(poolInput);
console.log(`Output shape: [${avgPooled.shape.join(", ")}]`);
console.log(`Output value: ${avgPooled.toString()}`);
console.log("  Each 2x2 window is replaced by its mean\n");

// ---------------------------------------------------------------------------
// Part 5: A small Conv1d pipeline with Sequential
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
const cnnOutput = noGrad(() => cnn.forward(cnnInput));
console.log(`Input shape:  [${cnnInput.shape.join(", ")}]`);
console.log(`Output shape: [${cnnOutput.shape.join(", ")}]`);

const paramCount = Array.from(cnn.parameters()).length;
console.log(`Parameter tensors: ${paramCount} (weight and bias for each Conv1d)`);

console.log("\n=== CNN Layers Complete ===");
