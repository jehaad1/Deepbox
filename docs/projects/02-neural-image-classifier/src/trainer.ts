/**
 * Neural Network Trainer Module
 *
 * Implements training loops and utilities for neural network training.
 * Demonstrates deepbox/optim and deepbox/nn usage with proper type safety.
 */

import { isNumericTypedArray, isTypedArray } from "deepbox/core";
import { GradTensor, parameter, tensor } from "deepbox/ndarray";
import type { Sequential } from "deepbox/nn";
import { crossEntropyLoss } from "deepbox/nn";

/**
 * Training configuration
 */
export interface TrainingConfig {
  epochs: number;
  learningRate: number;
  batchSize: number;
  optimizer: "adam" | "adamw" | "sgd" | "rmsprop" | "adagrad";
  weightDecay?: number;
  momentum?: number;
  verbose?: boolean;
}

/**
 * Training history for tracking metrics
 */
export interface TrainingHistory {
  trainLoss: number[];
  trainAccuracy: number[];
  valLoss: number[];
  valAccuracy: number[];
  epochs: number[];
}

const expectNumericTypedArray = (
  value: unknown
): Float32Array | Float64Array | Int32Array | Uint8Array => {
  if (!isTypedArray(value) || !isNumericTypedArray(value)) {
    throw new Error("Expected numeric typed array");
  }
  return value;
};

/**
 * Calculate accuracy from predictions and labels
 */
export function calculateAccuracy(
  predData: Float32Array | Float64Array,
  labelData: Float32Array | Float64Array,
  numSamples: number,
  numClasses: number
): number {
  let correct = 0;

  for (let i = 0; i < numSamples; i++) {
    let maxVal = -Infinity;
    let predClass = 0;
    for (let j = 0; j < numClasses; j++) {
      const val = predData[i * numClasses + j];
      if (val > maxVal) {
        maxVal = val;
        predClass = j;
      }
    }

    const trueClass = Math.round(labelData[i]);
    if (predClass === trueClass) {
      correct++;
    }
  }

  return correct / numSamples;
}

/**
 * Extract batch from data
 */
export function extractBatch(
  X: Float32Array,
  y: Float32Array,
  startIdx: number,
  batchSize: number,
  numFeatures: number,
  numSamples: number
): { XBatch: number[][]; yBatch: number[] } {
  const endIdx = Math.min(startIdx + batchSize, numSamples);

  const XBatch: number[][] = [];
  const yBatch: number[] = [];

  for (let i = startIdx; i < endIdx; i++) {
    const row: number[] = [];
    for (let f = 0; f < numFeatures; f++) {
      row.push(X[i * numFeatures + f]);
    }
    XBatch.push(row);
    yBatch.push(y[i]);
  }

  return { XBatch, yBatch };
}

/**
 * Create one-hot encoded targets
 */
export function createOneHot(labels: number[], numClasses: number): number[][] {
  return labels.map((label) => {
    const oneHot = Array(numClasses).fill(0);
    oneHot[Math.round(label)] = 1;
    return oneHot;
  });
}

/**
 * Simple training step for demonstration
 */
export function trainStep(
  model: Sequential,
  XBatch: number[][],
  yBatch: number[],
  numClasses: number
): { loss: number; predictions: number[] } {
  // Create input tensor with gradient tracking
  const input = parameter(tensor(XBatch, { dtype: "float32" }));

  // Forward pass
  const output = model.forward(input.tensor);
  const outputTensor = output instanceof GradTensor ? output.tensor : output;

  // Create targets - crossEntropyLoss expects 1D class labels, not one-hot
  const targetTensor = tensor(yBatch, { dtype: "float32" });

  // Compute loss - returns number for evaluation
  const lossValue = crossEntropyLoss(outputTensor, targetTensor);

  // Get predictions
  const outputData = expectNumericTypedArray(outputTensor.data);
  const predictions: number[] = [];
  for (let i = 0; i < yBatch.length; i++) {
    let maxVal = -Infinity;
    let predClass = 0;
    for (let j = 0; j < numClasses; j++) {
      const val = outputData[i * numClasses + j];
      if (val > maxVal) {
        maxVal = val;
        predClass = j;
      }
    }
    predictions.push(predClass);
  }

  return {
    loss: lossValue,
    predictions,
  };
}

/**
 * Evaluate model on test data
 */
export function evaluateModel(
  model: Sequential,
  X: Float32Array,
  y: Float32Array,
  numClasses: number,
  numFeatures: number,
  numSamples: number
): { loss: number; accuracy: number; predictions: number[] } {
  model.train(false);

  // Convert to array format
  const XArray: number[][] = [];
  const yArray: number[] = [];

  for (let i = 0; i < numSamples; i++) {
    const row: number[] = [];
    for (let f = 0; f < numFeatures; f++) {
      row.push(X[i * numFeatures + f]);
    }
    XArray.push(row);
    yArray.push(y[i]);
  }

  // Forward pass
  const input = GradTensor.fromTensor(tensor(XArray, { dtype: "float32" }), {
    requiresGrad: false,
  });
  const output = model.forward(input.tensor);
  const outputTensor = output instanceof GradTensor ? output.tensor : output;

  // Create targets - crossEntropyLoss expects 1D class labels, not one-hot
  const targetTensor = tensor(yArray, { dtype: "float32" });

  // Compute loss
  const lossValue = crossEntropyLoss(outputTensor, targetTensor);

  // Calculate predictions and accuracy
  const outputData = expectNumericTypedArray(outputTensor.data);
  const predictions: number[] = [];
  let correct = 0;

  for (let i = 0; i < numSamples; i++) {
    let maxVal = -Infinity;
    let predClass = 0;
    for (let j = 0; j < numClasses; j++) {
      const val = outputData[i * numClasses + j];
      if (val > maxVal) {
        maxVal = val;
        predClass = j;
      }
    }
    predictions.push(predClass);
    if (predClass === Math.round(yArray[i])) {
      correct++;
    }
  }

  return {
    loss: lossValue,
    accuracy: correct / numSamples,
    predictions,
  };
}

/**
 * Early stopping callback
 */
export class EarlyStopping {
  private patience: number;
  private minDelta: number;
  private counter: number;
  private bestLoss: number;
  private shouldStop: boolean;

  constructor(patience = 5, minDelta = 0.001) {
    this.patience = patience;
    this.minDelta = minDelta;
    this.counter = 0;
    this.bestLoss = Infinity;
    this.shouldStop = false;
  }

  check(valLoss: number): boolean {
    if (valLoss < this.bestLoss - this.minDelta) {
      this.bestLoss = valLoss;
      this.counter = 0;
    } else {
      this.counter++;
      if (this.counter >= this.patience) {
        this.shouldStop = true;
      }
    }
    return this.shouldStop;
  }

  reset(): void {
    this.counter = 0;
    this.bestLoss = Infinity;
    this.shouldStop = false;
  }
}

/**
 * Learning rate schedulers
 */
export function stepLRScheduler(
  baseLR: number,
  epoch: number,
  stepSize: number,
  gamma = 0.1
): number {
  return baseLR * gamma ** Math.floor(epoch / stepSize);
}

export function cosineLRScheduler(
  baseLR: number,
  epoch: number,
  maxEpochs: number,
  minLR = 0
): number {
  return minLR + ((baseLR - minLR) * (1 + Math.cos((Math.PI * epoch) / maxEpochs))) / 2;
}
