/**
 * Multi-layer Perceptron (MLP) estimators.
 *
 * Provides sklearn-compatible MLPClassifier and MLPRegressor that use
 * backpropagation for training a feedforward neural network.
 *
 * @see {@link https://deepbox.dev/docs/ml-advanced | Deepbox Neural Network Estimators}
 */

import { DataValidationError, InvalidParameterError, NotFittedError, ShapeError } from "../core";
import { type Tensor, tensor } from "../ndarray";
import { __random } from "../random/random";
import { assertContiguous, validateFitInputs, validatePredictInputs } from "./_validation";
import type { Classifier, Regressor } from "./base";

type ActivationFn = "relu" | "tanh" | "logistic" | "identity";

function activate(x: number, fn: ActivationFn): number {
  switch (fn) {
    case "relu":
      return x > 0 ? x : 0;
    case "tanh":
      return Math.tanh(x);
    case "logistic":
      return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
    case "identity":
      return x;
  }
}

function activateDerivative(output: number, fn: ActivationFn): number {
  switch (fn) {
    case "relu":
      return output > 0 ? 1 : 0;
    case "tanh":
      return 1 - output * output;
    case "logistic":
      return output * (1 - output);
    case "identity":
      return 1;
  }
}

function softmax(logits: Float64Array): Float64Array {
  const n = logits.length;
  const result = new Float64Array(n);
  let maxVal = -Infinity;
  for (let i = 0; i < n; i++) {
    if ((logits[i] ?? 0) > maxVal) maxVal = logits[i] ?? 0;
  }
  let sum = 0;
  for (let i = 0; i < n; i++) {
    result[i] = Math.exp((logits[i] ?? 0) - maxVal);
    sum += result[i] ?? 0;
  }
  for (let i = 0; i < n; i++) {
    result[i] = (result[i] ?? 0) / sum;
  }
  return result;
}

// Xavier initialization
function xavierInit(fanIn: number, fanOut: number): number {
  const limit = Math.sqrt(6 / (fanIn + fanOut));
  return (__random() * 2 - 1) * limit;
}

interface MLPLayer {
  weights: Float64Array; // (inputSize x outputSize) row-major
  biases: Float64Array; // (outputSize)
  inputSize: number;
  outputSize: number;
}

function createLayer(inputSize: number, outputSize: number): MLPLayer {
  const weights = new Float64Array(inputSize * outputSize);
  for (let i = 0; i < weights.length; i++) {
    weights[i] = xavierInit(inputSize, outputSize);
  }
  const biases = new Float64Array(outputSize);
  return { weights, biases, inputSize, outputSize };
}

function forwardLayer(
  input: Float64Array,
  layer: MLPLayer,
  activation: ActivationFn
): { preActivation: Float64Array; output: Float64Array } {
  const out = new Float64Array(layer.outputSize);
  const pre = new Float64Array(layer.outputSize);
  for (let j = 0; j < layer.outputSize; j++) {
    let sum = layer.biases[j] ?? 0;
    for (let i = 0; i < layer.inputSize; i++) {
      sum += (input[i] ?? 0) * (layer.weights[i * layer.outputSize + j] ?? 0);
    }
    pre[j] = sum;
    out[j] = activate(sum, activation);
  }
  return { preActivation: pre, output: out };
}

/**
 * Multi-layer Perceptron Classifier.
 *
 * A feedforward neural network classifier trained with backpropagation.
 * Supports configurable hidden layer sizes, activation functions, and
 * learning rate schedules.
 *
 * @example
 * ```ts
 * import { MLPClassifier } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[0, 0], [0, 1], [1, 0], [1, 1]]);
 * const y = tensor([0, 1, 1, 0]); // XOR
 *
 * const clf = new MLPClassifier({ hiddenLayerSizes: [4], maxIter: 500 });
 * clf.fit(X, y);
 * const predictions = clf.predict(X);
 * ```
 */
export class MLPClassifier implements Classifier {
  private hiddenLayerSizes: readonly number[];
  private activation: ActivationFn;
  private learningRate: number;
  private maxIter: number;
  private tol: number;
  private alpha: number; // L2 regularization

  private layers_?: MLPLayer[];
  private classes_?: number[];
  private nFeaturesIn_?: number;
  private nOutputs_ = 0;
  private fitted = false;

  /**
   * @param options.hiddenLayerSizes - Array of hidden layer sizes (default: [100])
   * @param options.activation - Activation function: "relu", "tanh", "logistic", "identity" (default: "relu")
   * @param options.learningRate - Initial learning rate (default: 0.001)
   * @param options.maxIter - Maximum number of epochs (default: 200)
   * @param options.tol - Tolerance for convergence (default: 1e-4)
   * @param options.alpha - L2 regularization strength (default: 1e-4)
   */
  constructor(
    options: {
      readonly hiddenLayerSizes?: readonly number[];
      readonly activation?: ActivationFn;
      readonly learningRate?: number;
      readonly maxIter?: number;
      readonly tol?: number;
      readonly alpha?: number;
    } = {}
  ) {
    this.hiddenLayerSizes = options.hiddenLayerSizes ?? [100];
    this.activation = options.activation ?? "relu";
    this.learningRate = options.learningRate ?? 0.001;
    this.maxIter = options.maxIter ?? 200;
    this.tol = options.tol ?? 1e-4;
    this.alpha = options.alpha ?? 1e-4;

    if (this.hiddenLayerSizes.length === 0) {
      throw new InvalidParameterError(
        "hiddenLayerSizes must have at least one layer",
        "hiddenLayerSizes",
        this.hiddenLayerSizes
      );
    }
    for (const s of this.hiddenLayerSizes) {
      if (!Number.isInteger(s) || s < 1) {
        throw new InvalidParameterError(
          "Each hidden layer size must be a positive integer",
          "hiddenLayerSizes",
          this.hiddenLayerSizes
        );
      }
    }
    if (!Number.isFinite(this.learningRate) || this.learningRate <= 0) {
      throw new InvalidParameterError(
        "learningRate must be positive",
        "learningRate",
        this.learningRate
      );
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new InvalidParameterError(
        "maxIter must be a positive integer",
        "maxIter",
        this.maxIter
      );
    }
  }

  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    this.nFeaturesIn_ = nFeatures;

    // Extract data
    const XData: Float64Array[] = [];
    const yData: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      const row = new Float64Array(nFeatures);
      for (let j = 0; j < nFeatures; j++) {
        row[j] = Number(X.data[X.offset + i * nFeatures + j] ?? 0);
      }
      XData.push(row);
      yData.push(Number(y.data[y.offset + i] ?? 0));
    }

    // Discover classes
    this.classes_ = [...new Set(yData)].sort((a, b) => a - b);
    const nClasses = this.classes_.length;
    const classIndex = new Map<number, number>();
    for (let c = 0; c < nClasses; c++) {
      classIndex.set(this.classes_[c] ?? 0, c);
    }

    // Output size: 1 for binary (sigmoid), nClasses for multiclass (softmax)
    this.nOutputs_ = nClasses <= 2 ? 1 : nClasses;

    // Build network layers
    const layerSizes = [nFeatures, ...this.hiddenLayerSizes, this.nOutputs_];
    this.layers_ = [];
    for (let l = 0; l < layerSizes.length - 1; l++) {
      this.layers_.push(createLayer(layerSizes[l] ?? 0, layerSizes[l + 1] ?? 0));
    }

    // One-hot encode targets
    const targets: Float64Array[] = [];
    for (let i = 0; i < nSamples; i++) {
      const ci = classIndex.get(yData[i] ?? 0) ?? 0;
      if (this.nOutputs_ === 1) {
        targets.push(new Float64Array([ci]));
      } else {
        const t = new Float64Array(nClasses);
        t[ci] = 1;
        targets.push(t);
      }
    }

    // Training loop (mini-batch SGD)
    let prevLoss = Infinity;
    for (let epoch = 0; epoch < this.maxIter; epoch++) {
      let totalLoss = 0;

      for (let i = 0; i < nSamples; i++) {
        const input = XData[i]!;
        const target = targets[i]!;

        // Forward pass
        const activations: Float64Array[] = [input];
        const preActivations: Float64Array[] = [];

        let current = input;
        for (let l = 0; l < this.layers_.length; l++) {
          const isOutput = l === this.layers_.length - 1;
          const actFn = isOutput ? "identity" : this.activation;
          const { preActivation, output } = forwardLayer(current, this.layers_[l]!, actFn);
          preActivations.push(preActivation);

          if (isOutput) {
            if (this.nOutputs_ === 1) {
              // Sigmoid for binary
              output[0] = activate(preActivation[0] ?? 0, "logistic");
            } else {
              // Softmax for multiclass
              const sm = softmax(preActivation);
              for (let k = 0; k < output.length; k++) {
                output[k] = sm[k] ?? 0;
              }
            }
          }

          activations.push(output);
          current = output;
        }

        const output = activations[activations.length - 1]!;

        // Compute loss (cross-entropy)
        for (let k = 0; k < this.nOutputs_; k++) {
          const p = Math.max(1e-15, Math.min(1 - 1e-15, output[k] ?? 0));
          const t = target[k] ?? 0;
          totalLoss -= t * Math.log(p) + (1 - t) * Math.log(1 - p);
        }

        // Backward pass
        // Output layer delta
        let delta = new Float64Array(this.nOutputs_);
        for (let k = 0; k < this.nOutputs_; k++) {
          delta[k] = (output[k] ?? 0) - (target[k] ?? 0);
        }

        // Backpropagate through layers
        for (let l = this.layers_.length - 1; l >= 0; l--) {
          const layer = this.layers_[l]!;
          const layerInput = activations[l]!;

          // Compute weight gradients and update
          for (let j = 0; j < layer.outputSize; j++) {
            for (let i2 = 0; i2 < layer.inputSize; i2++) {
              const grad =
                (delta[j] ?? 0) * (layerInput[i2] ?? 0) +
                this.alpha * (layer.weights[i2 * layer.outputSize + j] ?? 0);
              layer.weights[i2 * layer.outputSize + j] =
                (layer.weights[i2 * layer.outputSize + j] ?? 0) - this.learningRate * grad;
            }
            layer.biases[j] = (layer.biases[j] ?? 0) - this.learningRate * (delta[j] ?? 0);
          }

          // Compute delta for previous layer
          if (l > 0) {
            const prevDelta = new Float64Array(layer.inputSize);
            const prevOutput = activations[l]!;
            for (let i2 = 0; i2 < layer.inputSize; i2++) {
              let sum = 0;
              for (let j = 0; j < layer.outputSize; j++) {
                sum += (delta[j] ?? 0) * (layer.weights[i2 * layer.outputSize + j] ?? 0);
              }
              prevDelta[i2] = sum * activateDerivative(prevOutput[i2] ?? 0, this.activation);
            }
            delta = prevDelta;
          }
        }
      }

      totalLoss /= nSamples;

      // Check convergence
      if (Math.abs(prevLoss - totalLoss) < this.tol) break;
      prevLoss = totalLoss;
    }

    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.layers_ || !this.classes_) {
      throw new NotFittedError("MLPClassifier must be fitted before prediction");
    }
    const outputs = this.forwardAll(X);
    const nSamples = outputs.length;
    const predictions: number[] = [];

    for (let i = 0; i < nSamples; i++) {
      const out = outputs[i]!;
      if (this.nOutputs_ === 1) {
        const p = out[0] ?? 0;
        predictions.push(p >= 0.5 ? (this.classes_[1] ?? 1) : (this.classes_[0] ?? 0));
      } else {
        let maxVal = -Infinity;
        let maxIdx = 0;
        for (let k = 0; k < out.length; k++) {
          if ((out[k] ?? 0) > maxVal) {
            maxVal = out[k] ?? 0;
            maxIdx = k;
          }
        }
        predictions.push(this.classes_[maxIdx] ?? 0);
      }
    }

    return tensor(predictions, { dtype: "int32" });
  }

  predictProba(X: Tensor): Tensor {
    if (!this.fitted || !this.layers_ || !this.classes_) {
      throw new NotFittedError("MLPClassifier must be fitted before prediction");
    }
    const outputs = this.forwardAll(X);
    const nClasses = this.classes_.length;
    const probabilities: number[][] = [];

    for (const out of outputs) {
      if (this.nOutputs_ === 1) {
        const p = Math.max(1e-15, Math.min(1 - 1e-15, out[0] ?? 0));
        probabilities.push([1 - p, p]);
      } else {
        const row: number[] = [];
        for (let k = 0; k < nClasses; k++) {
          row.push(Math.max(0, out[k] ?? 0));
        }
        const sum = row.reduce((a, b) => a + b, 0) || 1;
        probabilities.push(row.map((v) => v / sum));
      }
    }

    return tensor(probabilities);
  }

  score(X: Tensor, y: Tensor): number {
    if (y.ndim !== 1) {
      throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    }
    assertContiguous(y, "y");
    const yPred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.size; i++) {
      if (Number(y.data[y.offset + i] ?? 0) === Number(yPred.data[yPred.offset + i] ?? 0)) {
        correct++;
      }
    }
    return correct / y.size;
  }

  get classes(): Tensor | undefined {
    if (!this.fitted || !this.classes_) return undefined;
    return tensor(this.classes_, { dtype: "int32" });
  }

  getParams(): Record<string, unknown> {
    return {
      hiddenLayerSizes: this.hiddenLayerSizes,
      activation: this.activation,
      learningRate: this.learningRate,
      maxIter: this.maxIter,
      tol: this.tol,
      alpha: this.alpha,
    };
  }

  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "hiddenLayerSizes":
          if (!Array.isArray(value) || value.length === 0) {
            throw new InvalidParameterError(
              "hiddenLayerSizes must be a non-empty array",
              "hiddenLayerSizes",
              value
            );
          }
          this.hiddenLayerSizes = value as number[];
          break;
        case "activation":
          if (
            value !== "relu" &&
            value !== "tanh" &&
            value !== "logistic" &&
            value !== "identity"
          ) {
            throw new InvalidParameterError("invalid activation", "activation", value);
          }
          this.activation = value;
          break;
        case "learningRate":
          if (typeof value !== "number" || value <= 0) {
            throw new InvalidParameterError("learningRate must be > 0", "learningRate", value);
          }
          this.learningRate = value;
          break;
        case "maxIter":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError("maxIter must be integer >= 1", "maxIter", value);
          }
          this.maxIter = value;
          break;
        case "alpha":
          if (typeof value !== "number" || value < 0) {
            throw new InvalidParameterError("alpha must be >= 0", "alpha", value);
          }
          this.alpha = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }

  private forwardAll(X: Tensor): Float64Array[] {
    validatePredictInputs(X, this.nFeaturesIn_ ?? 0, "MLPClassifier");
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const results: Float64Array[] = [];

    for (let i = 0; i < nSamples; i++) {
      let current: Float64Array = new Float64Array(nFeatures);
      for (let j = 0; j < nFeatures; j++) {
        current[j] = Number(X.data[X.offset + i * nFeatures + j] ?? 0);
      }

      for (let l = 0; l < this.layers_!.length; l++) {
        const isOutput = l === this.layers_!.length - 1;
        const actFn = isOutput ? "identity" : this.activation;
        const { preActivation, output } = forwardLayer(current, this.layers_![l]!, actFn);

        if (isOutput) {
          if (this.nOutputs_ === 1) {
            output[0] = activate(preActivation[0] ?? 0, "logistic");
          } else {
            const sm = softmax(preActivation);
            for (let k = 0; k < output.length; k++) {
              output[k] = sm[k] ?? 0;
            }
          }
        }

        current = output;
      }

      results.push(current);
    }

    return results;
  }
}

/**
 * Multi-layer Perceptron Regressor.
 *
 * A feedforward neural network regressor trained with backpropagation.
 * Uses identity activation on the output layer and optimizes squared error.
 *
 * @example
 * ```ts
 * import { MLPRegressor } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1], [2], [3], [4], [5]]);
 * const y = tensor([1.0, 4.0, 9.0, 16.0, 25.0]);
 *
 * const reg = new MLPRegressor({ hiddenLayerSizes: [10], maxIter: 500 });
 * reg.fit(X, y);
 * const predictions = reg.predict(X);
 * ```
 */
export class MLPRegressor implements Regressor {
  private hiddenLayerSizes: readonly number[];
  private activation: ActivationFn;
  private learningRate: number;
  private maxIter: number;
  private tol: number;
  private alpha: number;

  private layers_?: MLPLayer[];
  private nFeaturesIn_?: number;
  private yMean_ = 0;
  private yStd_ = 1;
  private fitted = false;

  /**
   * @param options.hiddenLayerSizes - Array of hidden layer sizes (default: [100])
   * @param options.activation - Activation function (default: "relu")
   * @param options.learningRate - Learning rate (default: 0.001)
   * @param options.maxIter - Maximum epochs (default: 200)
   * @param options.tol - Convergence tolerance (default: 1e-4)
   * @param options.alpha - L2 regularization (default: 1e-4)
   */
  constructor(
    options: {
      readonly hiddenLayerSizes?: readonly number[];
      readonly activation?: ActivationFn;
      readonly learningRate?: number;
      readonly maxIter?: number;
      readonly tol?: number;
      readonly alpha?: number;
    } = {}
  ) {
    this.hiddenLayerSizes = options.hiddenLayerSizes ?? [100];
    this.activation = options.activation ?? "relu";
    this.learningRate = options.learningRate ?? 0.001;
    this.maxIter = options.maxIter ?? 200;
    this.tol = options.tol ?? 1e-4;
    this.alpha = options.alpha ?? 1e-4;

    if (this.hiddenLayerSizes.length === 0) {
      throw new InvalidParameterError(
        "hiddenLayerSizes must have at least one layer",
        "hiddenLayerSizes",
        this.hiddenLayerSizes
      );
    }
    for (const s of this.hiddenLayerSizes) {
      if (!Number.isInteger(s) || s < 1) {
        throw new InvalidParameterError(
          "Each hidden layer size must be a positive integer",
          "hiddenLayerSizes",
          this.hiddenLayerSizes
        );
      }
    }
  }

  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    this.nFeaturesIn_ = nFeatures;

    // Extract data
    const XData: Float64Array[] = [];
    const yData: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      const row = new Float64Array(nFeatures);
      for (let j = 0; j < nFeatures; j++) {
        row[j] = Number(X.data[X.offset + i * nFeatures + j] ?? 0);
      }
      XData.push(row);
      yData.push(Number(y.data[y.offset + i] ?? 0));
    }

    // Normalize target for stable training
    this.yMean_ = 0;
    for (const v of yData) this.yMean_ += v;
    this.yMean_ /= nSamples;

    this.yStd_ = 0;
    for (const v of yData) this.yStd_ += (v - this.yMean_) ** 2;
    this.yStd_ = Math.sqrt(this.yStd_ / nSamples);
    if (this.yStd_ < 1e-10) this.yStd_ = 1;

    const yNorm = yData.map((v) => (v - this.yMean_) / this.yStd_);

    // Build network (output size = 1)
    const layerSizes = [nFeatures, ...this.hiddenLayerSizes, 1];
    this.layers_ = [];
    for (let l = 0; l < layerSizes.length - 1; l++) {
      this.layers_.push(createLayer(layerSizes[l] ?? 0, layerSizes[l + 1] ?? 0));
    }

    // Training loop
    let prevLoss = Infinity;
    for (let epoch = 0; epoch < this.maxIter; epoch++) {
      let totalLoss = 0;

      for (let i = 0; i < nSamples; i++) {
        const input = XData[i]!;
        const target = yNorm[i] ?? 0;

        // Forward pass
        const activations: Float64Array[] = [input];

        let current = input;
        for (let l = 0; l < this.layers_.length; l++) {
          const isOutput = l === this.layers_.length - 1;
          const actFn = isOutput ? "identity" : this.activation;
          const { output } = forwardLayer(current, this.layers_[l]!, actFn);
          activations.push(output);
          current = output;
        }

        const output = activations[activations.length - 1]!;
        const error = (output[0] ?? 0) - target;
        totalLoss += error * error;

        // Backward pass
        let delta = new Float64Array([error]);

        for (let l = this.layers_.length - 1; l >= 0; l--) {
          const layer = this.layers_[l]!;
          const layerInput = activations[l]!;

          for (let j = 0; j < layer.outputSize; j++) {
            for (let i2 = 0; i2 < layer.inputSize; i2++) {
              const grad =
                (delta[j] ?? 0) * (layerInput[i2] ?? 0) +
                this.alpha * (layer.weights[i2 * layer.outputSize + j] ?? 0);
              layer.weights[i2 * layer.outputSize + j] =
                (layer.weights[i2 * layer.outputSize + j] ?? 0) - this.learningRate * grad;
            }
            layer.biases[j] = (layer.biases[j] ?? 0) - this.learningRate * (delta[j] ?? 0);
          }

          if (l > 0) {
            const prevDelta = new Float64Array(layer.inputSize);
            const prevOutput = activations[l]!;
            for (let i2 = 0; i2 < layer.inputSize; i2++) {
              let sum = 0;
              for (let j = 0; j < layer.outputSize; j++) {
                sum += (delta[j] ?? 0) * (layer.weights[i2 * layer.outputSize + j] ?? 0);
              }
              prevDelta[i2] = sum * activateDerivative(prevOutput[i2] ?? 0, this.activation);
            }
            delta = prevDelta;
          }
        }
      }

      totalLoss /= nSamples;
      if (Math.abs(prevLoss - totalLoss) < this.tol) break;
      prevLoss = totalLoss;
    }

    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.layers_) {
      throw new NotFittedError("MLPRegressor must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeaturesIn_ ?? 0, "MLPRegressor");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const predictions: number[] = [];

    for (let i = 0; i < nSamples; i++) {
      let current: Float64Array = new Float64Array(nFeatures);
      for (let j = 0; j < nFeatures; j++) {
        current[j] = Number(X.data[X.offset + i * nFeatures + j] ?? 0);
      }

      for (let l = 0; l < this.layers_.length; l++) {
        const isOutput = l === this.layers_.length - 1;
        const actFn = isOutput ? "identity" : this.activation;
        const { output } = forwardLayer(current, this.layers_[l]!, actFn);
        current = output;
      }

      // Denormalize
      predictions.push((current[0] ?? 0) * this.yStd_ + this.yMean_);
    }

    return tensor(predictions);
  }

  score(X: Tensor, y: Tensor): number {
    if (y.ndim !== 1) {
      throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    }
    assertContiguous(y, "y");
    for (let i = 0; i < y.size; i++) {
      if (!Number.isFinite(y.data[y.offset + i] ?? 0)) {
        throw new DataValidationError("y contains non-finite values (NaN or Inf)");
      }
    }
    const predictions = this.predict(X);
    let ssRes = 0;
    let ssTot = 0;
    let yMean = 0;
    for (let i = 0; i < y.size; i++) {
      yMean += Number(y.data[y.offset + i] ?? 0);
    }
    yMean /= y.size;
    for (let i = 0; i < y.size; i++) {
      const yVal = Number(y.data[y.offset + i] ?? 0);
      const pVal = Number(predictions.data[predictions.offset + i] ?? 0);
      ssRes += (yVal - pVal) ** 2;
      ssTot += (yVal - yMean) ** 2;
    }
    return ssTot === 0 ? (ssRes === 0 ? 1.0 : 0.0) : 1 - ssRes / ssTot;
  }

  getParams(): Record<string, unknown> {
    return {
      hiddenLayerSizes: this.hiddenLayerSizes,
      activation: this.activation,
      learningRate: this.learningRate,
      maxIter: this.maxIter,
      tol: this.tol,
      alpha: this.alpha,
    };
  }

  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "hiddenLayerSizes":
          if (!Array.isArray(value) || value.length === 0) {
            throw new InvalidParameterError(
              "hiddenLayerSizes must be a non-empty array",
              "hiddenLayerSizes",
              value
            );
          }
          this.hiddenLayerSizes = value as number[];
          break;
        case "activation":
          if (
            value !== "relu" &&
            value !== "tanh" &&
            value !== "logistic" &&
            value !== "identity"
          ) {
            throw new InvalidParameterError("invalid activation", "activation", value);
          }
          this.activation = value;
          break;
        case "learningRate":
          if (typeof value !== "number" || value <= 0) {
            throw new InvalidParameterError("learningRate must be > 0", "learningRate", value);
          }
          this.learningRate = value;
          break;
        case "maxIter":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError("maxIter must be integer >= 1", "maxIter", value);
          }
          this.maxIter = value;
          break;
        case "alpha":
          if (typeof value !== "number" || value < 0) {
            throw new InvalidParameterError("alpha must be >= 0", "alpha", value);
          }
          this.alpha = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}
