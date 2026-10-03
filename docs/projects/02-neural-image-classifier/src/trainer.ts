/**
 * Neural Network Trainer Module
 *
 * Helpers for training a classifier on plain tensors: batch extraction, one-hot
 * targets, a single loss step, evaluation under noGrad(), early stopping and
 * learning-rate schedules. index.ts has its own training loop and does not import
 * this file.
 */

import { type AnyTensor, GradTensor, noGrad, tensor } from "deepbox/ndarray";
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
 * Predicted class for each row of a logits tensor.
 */
function predictClasses(logits: AnyTensor): number[] {
  return logits.argmax(1).toArray() as number[];
}

/**
 * Compute the loss and predictions for one mini-batch.
 *
 * The input is a plain tensor. Because the model has trainable parameters and grad
 * mode is on, the output tracks the weights, so a caller can run `loss.backward()`
 * on the loss. This helper only reads the loss value.
 */
export function trainStep(
  model: Sequential,
  XBatch: number[][],
  yBatch: number[],
  _numClasses: number
): { loss: number; predictions: number[] } {
  const output = model.forward(tensor(XBatch, { dtype: "float32" }));
  if (!(output instanceof GradTensor)) {
    throw new Error("Expected a GradTensor: is grad mode off?");
  }

  // crossEntropyLoss expects 1D class labels, not one-hot rows
  const loss = crossEntropyLoss(output, tensor(yBatch, { dtype: "int32" }));

  return {
    loss: Number(loss.item()),
    predictions: predictClasses(output),
  };
}

/**
 * Evaluate the model on test data without tracking gradients.
 */
export function evaluateModel(
  model: Sequential,
  X: Float32Array,
  y: Float32Array,
  _numClasses: number,
  numFeatures: number,
  numSamples: number
): { loss: number; accuracy: number; predictions: number[] } {
  model.train(false);

  const XArray: number[][] = [];
  const yArray: number[] = [];

  for (let i = 0; i < numSamples; i++) {
    XArray.push(Array.from(X.subarray(i * numFeatures, (i + 1) * numFeatures)));
    yArray.push(y[i]);
  }

  const output = noGrad(() => model.forward(tensor(XArray, { dtype: "float32" })));
  if (output instanceof GradTensor) {
    throw new Error("Expected a plain Tensor inside noGrad()");
  }

  // With a plain Tensor input, crossEntropyLoss returns the mean loss as a number
  const loss = crossEntropyLoss(output, tensor(yArray, { dtype: "int32" }));

  const predictions = predictClasses(output);
  const correct = predictions.filter((p, i) => p === Math.round(yArray[i])).length;

  return { loss, accuracy: correct / numSamples, predictions };
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
