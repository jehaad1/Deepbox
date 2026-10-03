/**
 * Multi-layer Perceptron (MLP) estimators.
 *
 * Provides sklearn-compatible MLPClassifier and MLPRegressor that use
 * backpropagation for training a feedforward neural network.
 *
 * @see {@link https://deepbox.dev/docs/ml-advanced | Deepbox Neural Network Estimators}
 */

import {
  ConvergenceError,
  DataValidationError,
  InvalidParameterError,
  NotFittedError,
  ShapeError,
} from "../core";
import { type Tensor, tensor } from "../ndarray";
import { __random, __randomBelow, __SeededRandom, __seedToUint64 } from "../random/random";
import { r2Score } from "./_internal";
import {
  assertContiguous,
  toFloat64View,
  validateFitInputs,
  validatePredictInputs,
} from "./_validation";
import type { Classifier, Regressor } from "./base";

type ActivationFn = "relu" | "tanh" | "logistic" | "identity";

/** How the output layer is interpreted and which loss is minimized. */
type OutputKind = "binary" | "softmax" | "squared";

const ACTIVATIONS: readonly ActivationFn[] = ["relu", "tanh", "logistic", "identity"];

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

/** Derivative of the activation expressed through its own output. */
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

/** Numerically stable softmax, in place. */
function softmaxInPlace(v: Float64Array): void {
  let maxVal = -Infinity;
  for (let i = 0; i < v.length; i++) {
    if ((v[i] as number) > maxVal) maxVal = v[i] as number;
  }
  let sum = 0;
  for (let i = 0; i < v.length; i++) {
    const e = Math.exp((v[i] as number) - maxVal);
    v[i] = e;
    sum += e;
  }
  for (let i = 0; i < v.length; i++) v[i] = (v[i] as number) / sum;
}

/** Uniform draw from (-limit, limit). */
function glorotUniform(limit: number, rng: () => number): number {
  return (rng() * 2 - 1) * limit;
}

/**
 * Build a uniform [0, 1) generator. A seed gives a private deterministic stream;
 * without one the global Deepbox generator is used, so `setSeed` still applies.
 */
function createRng(seed: number | undefined): () => number {
  if (seed === undefined) return __random;
  const gen = new __SeededRandom(__seedToUint64(seed));
  return () => gen.next();
}

interface MLPLayer {
  weights: Float64Array; // (inputSize x outputSize) row-major
  biases: Float64Array; // (outputSize)
  inputSize: number;
  outputSize: number;
}

/**
 * Glorot/Xavier uniform initialization of weights and biases. The bound is
 * `sqrt(6 / (fanIn + fanOut))`, or `sqrt(2 / (fanIn + fanOut))` for the logistic
 * activation (the same rule as scikit-learn).
 */
function createLayer(
  inputSize: number,
  outputSize: number,
  activation: ActivationFn,
  rng: () => number
): MLPLayer {
  const factor = activation === "logistic" ? 2 : 6;
  const limit = Math.sqrt(factor / (inputSize + outputSize));
  const weights = new Float64Array(inputSize * outputSize);
  for (let i = 0; i < weights.length; i++) weights[i] = glorotUniform(limit, rng);
  const biases = new Float64Array(outputSize);
  for (let j = 0; j < outputSize; j++) biases[j] = glorotUniform(limit, rng);
  return { weights, biases, inputSize, outputSize };
}

type Network = {
  readonly layers: MLPLayer[];
  readonly activation: ActivationFn;
  readonly outputKind: OutputKind;
};

/** Per-layer activation buffers: `acts[0]` is the input, `acts[l + 1]` the output of layer `l`. */
function allocateActivations(layers: readonly MLPLayer[]): Float64Array[] {
  const acts: Float64Array[] = [new Float64Array(layers[0]?.inputSize ?? 0)];
  for (const layer of layers) acts.push(new Float64Array(layer.outputSize));
  return acts;
}

/** Forward pass for one sample whose features are already in `acts[0]`. */
function forward(net: Network, acts: Float64Array[]): void {
  const last = net.layers.length - 1;
  for (let l = 0; l <= last; l++) {
    const layer = net.layers[l] as MLPLayer;
    const input = acts[l] as Float64Array;
    const out = acts[l + 1] as Float64Array;
    const { weights, inputSize, outputSize } = layer;
    out.set(layer.biases);
    for (let i = 0; i < inputSize; i++) {
      const xi = input[i] as number;
      const row = i * outputSize;
      for (let j = 0; j < outputSize; j++) {
        out[j] = (out[j] as number) + xi * (weights[row + j] as number);
      }
    }
    if (l < last) {
      for (let j = 0; j < outputSize; j++) out[j] = activate(out[j] as number, net.activation);
    } else if (net.outputKind === "binary") {
      out[0] = activate(out[0] as number, "logistic");
    } else if (net.outputKind === "softmax") {
      softmaxInPlace(out);
    }
  }
}

/** Forward pass over all rows of `X` (row-major `m x nFeatures`); returns `m x nOutputs`. */
function forwardAll(net: Network, X: Float64Array, m: number, nFeatures: number): Float64Array {
  const acts = allocateActivations(net.layers);
  const nOut = (net.layers[net.layers.length - 1] as MLPLayer).outputSize;
  const result = new Float64Array(m * nOut);
  const input = acts[0] as Float64Array;
  const output = acts[acts.length - 1] as Float64Array;
  for (let i = 0; i < m; i++) {
    for (let f = 0; f < nFeatures; f++) input[f] = X[i * nFeatures + f] as number;
    forward(net, acts);
    result.set(output, i * nOut);
  }
  return result;
}

type TrainConfig = {
  readonly learningRate: number;
  readonly maxIter: number;
  readonly tol: number;
  readonly alpha: number;
  readonly shuffle: boolean;
  readonly nIterNoChange: number;
  readonly rng: () => number;
};

const LOSS_EPS = 1e-15;

/**
 * Online (one sample per update) gradient descent with L2 weight decay.
 *
 * Training stops when the epoch loss has not improved by more than `tol` for
 * more than `nIterNoChange` consecutive epochs, or after `maxIter` epochs.
 *
 * @param X - Row-major `nSamples x nFeatures` inputs
 * @param T - Row-major `nSamples x nOutputs` targets
 * @returns Number of epochs run and the mean loss of every epoch
 * @throws {ConvergenceError} If the loss becomes non-finite
 */
function train(
  net: Network,
  X: Float64Array,
  T: Float64Array,
  nSamples: number,
  nFeatures: number,
  cfg: TrainConfig
): { nIter: number; lossCurve: number[] } {
  const { layers, activation, outputKind } = net;
  const L = layers.length;
  const nOut = (layers[L - 1] as MLPLayer).outputSize;
  const acts = allocateActivations(layers);
  const deltas = layers.map((layer) => new Float64Array(layer.outputSize));
  const order = new Int32Array(nSamples);
  for (let i = 0; i < nSamples; i++) order[i] = i;
  const input = acts[0] as Float64Array;
  const output = acts[L] as Float64Array;
  const outDelta = deltas[L - 1] as Float64Array;
  const lr = cfg.learningRate;
  const alpha = cfg.alpha;

  const lossCurve: number[] = [];
  let bestLoss = Infinity;
  let noImprovement = 0;

  for (let epoch = 0; epoch < cfg.maxIter; epoch++) {
    if (cfg.shuffle) {
      for (let i = nSamples - 1; i > 0; i--) {
        const j = Math.min(__randomBelow(cfg.rng, i + 1), i);
        const tmp = order[i] as number;
        order[i] = order[j] as number;
        order[j] = tmp;
      }
    }

    let epochLoss = 0;
    for (let s = 0; s < nSamples; s++) {
      const idx = order[s] as number;
      const xBase = idx * nFeatures;
      for (let f = 0; f < nFeatures; f++) input[f] = X[xBase + f] as number;
      forward(net, acts);

      const tBase = idx * nOut;
      if (outputKind === "binary") {
        const p = Math.max(LOSS_EPS, Math.min(1 - LOSS_EPS, output[0] as number));
        const t = T[tBase] as number;
        epochLoss -= t * Math.log(p) + (1 - t) * Math.log(1 - p);
      } else if (outputKind === "softmax") {
        for (let k = 0; k < nOut; k++) {
          const t = T[tBase + k] as number;
          if (t !== 0) epochLoss -= t * Math.log(Math.max(LOSS_EPS, output[k] as number));
        }
      }
      for (let k = 0; k < nOut; k++) {
        const err = (output[k] as number) - (T[tBase + k] as number);
        outDelta[k] = err;
        if (outputKind === "squared") epochLoss += 0.5 * err * err;
      }

      // Backward pass. The delta of the layer below is computed from the weights
      // before they are updated.
      for (let l = L - 1; l >= 0; l--) {
        const layer = layers[l] as MLPLayer;
        const layerInput = acts[l] as Float64Array;
        const delta = deltas[l] as Float64Array;
        const prevDelta = l > 0 ? (deltas[l - 1] as Float64Array) : null;
        const { weights, biases, inputSize, outputSize } = layer;
        for (let i = 0; i < inputSize; i++) {
          const xi = layerInput[i] as number;
          const row = i * outputSize;
          let back = 0;
          for (let j = 0; j < outputSize; j++) {
            const w = weights[row + j] as number;
            const dj = delta[j] as number;
            back += dj * w;
            weights[row + j] = w - lr * (dj * xi + alpha * w);
          }
          if (prevDelta) prevDelta[i] = back * activateDerivative(xi, activation);
        }
        for (let j = 0; j < outputSize; j++) {
          biases[j] = (biases[j] as number) - lr * (delta[j] as number);
        }
      }
    }

    epochLoss /= nSamples;
    if (!Number.isFinite(epochLoss)) {
      throw new ConvergenceError(
        `Training diverged at epoch ${epoch + 1}: the loss is not finite. ` +
          "Lower learningRate or standardize the input features.",
        { iterations: epoch + 1 }
      );
    }
    lossCurve.push(epochLoss);

    if (epochLoss > bestLoss - cfg.tol) noImprovement++;
    else noImprovement = 0;
    if (epochLoss < bestLoss) bestLoss = epochLoss;
    if (noImprovement > cfg.nIterNoChange) break;
  }

  return { nIter: lossCurve.length, lossCurve };
}

/** Constructor options shared by {@link MLPClassifier} and {@link MLPRegressor}. */
type MLPOptions = {
  readonly hiddenLayerSizes?: readonly number[];
  readonly activation?: ActivationFn;
  readonly learningRate?: number;
  readonly maxIter?: number;
  readonly tol?: number;
  readonly alpha?: number;
  readonly shuffle?: boolean;
  readonly randomState?: number;
  readonly nIterNoChange?: number;
};

type ResolvedMLPOptions = {
  hiddenLayerSizes: readonly number[];
  activation: ActivationFn;
  learningRate: number;
  maxIter: number;
  tol: number;
  alpha: number;
  shuffle: boolean;
  randomState: number | undefined;
  nIterNoChange: number;
};

function resolveOptions(options: MLPOptions): ResolvedMLPOptions {
  const hidden = options.hiddenLayerSizes ?? [100];
  return {
    hiddenLayerSizes: Array.isArray(hidden) ? [...hidden] : hidden,
    activation: options.activation ?? "relu",
    learningRate: options.learningRate ?? 0.001,
    maxIter: options.maxIter ?? 200,
    tol: options.tol ?? 1e-4,
    alpha: options.alpha ?? 1e-4,
    shuffle: options.shuffle ?? true,
    randomState: options.randomState,
    nIterNoChange: options.nIterNoChange ?? 10,
  };
}

function validateOptions(o: ResolvedMLPOptions): void {
  if (!Array.isArray(o.hiddenLayerSizes) || o.hiddenLayerSizes.length === 0) {
    throw new InvalidParameterError(
      "hiddenLayerSizes must have at least one layer",
      "hiddenLayerSizes",
      o.hiddenLayerSizes
    );
  }
  for (const s of o.hiddenLayerSizes) {
    if (!Number.isInteger(s) || s < 1) {
      throw new InvalidParameterError(
        "Each hidden layer size must be a positive integer",
        "hiddenLayerSizes",
        o.hiddenLayerSizes
      );
    }
  }
  if (!ACTIVATIONS.includes(o.activation)) {
    throw new InvalidParameterError(
      `activation must be one of ${ACTIVATIONS.join(", ")}; received ${String(o.activation)}`,
      "activation",
      o.activation
    );
  }
  if (
    typeof o.learningRate !== "number" ||
    !Number.isFinite(o.learningRate) ||
    o.learningRate <= 0
  ) {
    throw new InvalidParameterError(
      "learningRate must be positive",
      "learningRate",
      o.learningRate
    );
  }
  if (!Number.isInteger(o.maxIter) || o.maxIter < 1) {
    throw new InvalidParameterError("maxIter must be a positive integer", "maxIter", o.maxIter);
  }
  if (typeof o.tol !== "number" || !Number.isFinite(o.tol) || o.tol < 0) {
    throw new InvalidParameterError("tol must be >= 0", "tol", o.tol);
  }
  if (typeof o.alpha !== "number" || !Number.isFinite(o.alpha) || o.alpha < 0) {
    throw new InvalidParameterError("alpha must be >= 0", "alpha", o.alpha);
  }
  if (typeof o.shuffle !== "boolean") {
    throw new InvalidParameterError("shuffle must be a boolean", "shuffle", o.shuffle);
  }
  if (!Number.isInteger(o.nIterNoChange) || o.nIterNoChange < 1) {
    throw new InvalidParameterError(
      "nIterNoChange must be a positive integer",
      "nIterNoChange",
      o.nIterNoChange
    );
  }
  if (o.randomState !== undefined && !Number.isFinite(o.randomState)) {
    throw new InvalidParameterError(
      "randomState must be a finite number",
      "randomState",
      o.randomState
    );
  }
}

const PARAM_KEYS = [
  "hiddenLayerSizes",
  "activation",
  "learningRate",
  "maxIter",
  "tol",
  "alpha",
  "shuffle",
  "randomState",
  "nIterNoChange",
] as const;

/** Apply `params` to a copy of `current`, validate, and return the new options. */
function mergeParams(
  current: ResolvedMLPOptions,
  params: Record<string, unknown>
): ResolvedMLPOptions {
  const next: Record<string, unknown> = { ...current };
  for (const [key, value] of Object.entries(params)) {
    if (!(PARAM_KEYS as readonly string[]).includes(key)) {
      throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
    }
    next[key] = key === "hiddenLayerSizes" && Array.isArray(value) ? [...value] : value;
  }
  const merged = next as ResolvedMLPOptions;
  validateOptions(merged);
  return merged;
}

function optionsToParams(o: ResolvedMLPOptions): Record<string, unknown> {
  return {
    hiddenLayerSizes: [...o.hiddenLayerSizes],
    activation: o.activation,
    learningRate: o.learningRate,
    maxIter: o.maxIter,
    tol: o.tol,
    alpha: o.alpha,
    shuffle: o.shuffle,
    randomState: o.randomState,
    nIterNoChange: o.nIterNoChange,
  };
}

function buildLayers(
  nFeatures: number,
  hidden: readonly number[],
  nOutputs: number,
  activation: ActivationFn,
  rng: () => number
): MLPLayer[] {
  const sizes = [nFeatures, ...hidden, nOutputs];
  const layers: MLPLayer[] = [];
  for (let l = 0; l < sizes.length - 1; l++) {
    layers.push(createLayer(sizes[l] as number, sizes[l + 1] as number, activation, rng));
  }
  return layers;
}

/**
 * Multi-layer Perceptron Classifier.
 *
 * A feedforward neural network classifier trained with backpropagation
 * (stochastic gradient descent, one sample per update). Binary problems use a
 * single logistic output, problems with three or more classes use a softmax
 * output; both minimize cross-entropy with L2 weight decay. Class labels must be
 * integers.
 *
 * Training stops when the epoch loss has not improved by more than `tol` for
 * more than `nIterNoChange` consecutive epochs, or after `maxIter` epochs. Inputs
 * should be standardized: unscaled features make plain SGD slow or unstable.
 *
 * @example
 * ```ts
 * import { MLPClassifier } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[0, 0], [0, 1], [1, 0], [1, 1]]);
 * const y = tensor([0, 1, 1, 0]); // XOR
 *
 * const clf = new MLPClassifier({ hiddenLayerSizes: [8], activation: 'tanh', learningRate: 0.1, maxIter: 1000, randomState: 0 });
 * clf.fit(X, y);
 * const predictions = clf.predict(X);
 * ```
 */
export class MLPClassifier implements Classifier {
  private opts: ResolvedMLPOptions;

  private net_?: Network;
  private classes_?: number[];
  private nFeaturesIn_?: number;
  private nOutputs_ = 0;
  private nIter_ = 0;
  private lossCurve_: number[] = [];
  private fitted = false;

  /**
   * @param options.hiddenLayerSizes - Array of hidden layer sizes (default: [100])
   * @param options.activation - Hidden activation: "relu", "tanh", "logistic", "identity" (default: "relu")
   * @param options.learningRate - Constant step size of each per-sample SGD update (default: 0.001)
   * @param options.maxIter - Maximum number of epochs (default: 200)
   * @param options.tol - Minimum loss improvement that counts as progress (default: 1e-4)
   * @param options.alpha - L2 regularization strength (default: 1e-4)
   * @param options.shuffle - Shuffle the samples every epoch (default: true)
   * @param options.randomState - Seed for weight initialization and shuffling; without it the global Deepbox generator is used
   * @param options.nIterNoChange - Epochs without improvement tolerated before stopping (default: 10)
   */
  constructor(options: MLPOptions = {}) {
    this.opts = resolveOptions(options);
    validateOptions(this.opts);
  }

  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;

    const XData = toFloat64View(X);
    const yData = toFloat64View(y);
    for (let i = 0; i < nSamples; i++) {
      if (!Number.isInteger(yData[i])) {
        throw new DataValidationError(
          `MLPClassifier requires integer class labels; y[${i}] = ${yData[i]}`
        );
      }
    }

    const classes = [...new Set(yData)].sort((a, b) => a - b);
    const nClasses = classes.length;
    if (nClasses < 2) {
      throw new DataValidationError(
        `MLPClassifier requires at least 2 classes in y; found ${nClasses}`
      );
    }
    const classIndex = new Map<number, number>();
    for (let c = 0; c < nClasses; c++) classIndex.set(classes[c] as number, c);

    // One logistic output for binary problems, one softmax output per class otherwise.
    const nOutputs = nClasses === 2 ? 1 : nClasses;
    const outputKind: OutputKind = nOutputs === 1 ? "binary" : "softmax";

    const T = new Float64Array(nSamples * nOutputs);
    for (let i = 0; i < nSamples; i++) {
      const ci = classIndex.get(yData[i] as number) as number;
      if (nOutputs === 1) T[i] = ci;
      else T[i * nOutputs + ci] = 1;
    }

    const rng = createRng(this.opts.randomState);
    const net: Network = {
      layers: buildLayers(
        nFeatures,
        this.opts.hiddenLayerSizes,
        nOutputs,
        this.opts.activation,
        rng
      ),
      activation: this.opts.activation,
      outputKind,
    };
    const { nIter, lossCurve } = train(net, XData, T, nSamples, nFeatures, {
      learningRate: this.opts.learningRate,
      maxIter: this.opts.maxIter,
      tol: this.opts.tol,
      alpha: this.opts.alpha,
      shuffle: this.opts.shuffle,
      nIterNoChange: this.opts.nIterNoChange,
      rng,
    });

    this.net_ = net;
    this.classes_ = classes;
    this.nFeaturesIn_ = nFeatures;
    this.nOutputs_ = nOutputs;
    this.nIter_ = nIter;
    this.lossCurve_ = lossCurve;
    this.fitted = true;
    return this;
  }

  /**
   * Predict class labels.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Labels of shape (n_samples,), dtype int32
   * @throws {NotFittedError} If the model has not been fitted
   */
  predict(X: Tensor): Tensor {
    const { out, classes, m } = this.forwardRows(X);
    const nOut = this.nOutputs_;
    const predictions = new Int32Array(m);
    for (let i = 0; i < m; i++) {
      if (nOut === 1) {
        predictions[i] =
          (out[i] as number) >= 0.5 ? (classes[1] as number) : (classes[0] as number);
      } else {
        let maxVal = -Infinity;
        let maxIdx = 0;
        for (let k = 0; k < nOut; k++) {
          const v = out[i * nOut + k] as number;
          if (v > maxVal) {
            maxVal = v;
            maxIdx = k;
          }
        }
        predictions[i] = classes[maxIdx] as number;
      }
    }
    return tensor(Array.from(predictions), { dtype: "int32" });
  }

  /**
   * Predict class probabilities.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Probabilities of shape (n_samples, n_classes); columns follow `classes`
   * @throws {NotFittedError} If the model has not been fitted
   */
  predictProba(X: Tensor): Tensor {
    const { out, classes, m } = this.forwardRows(X);
    const nClasses = classes.length;
    const probabilities: number[][] = [];
    for (let i = 0; i < m; i++) {
      if (this.nOutputs_ === 1) {
        const p = out[i] as number;
        probabilities.push([1 - p, p]);
      } else {
        const row: number[] = [];
        for (let k = 0; k < nClasses; k++) row.push(out[i * nClasses + k] as number);
        probabilities.push(row);
      }
    }
    return tensor(probabilities, { dtype: "float64" });
  }

  /**
   * Mean accuracy on the given data.
   *
   * @throws {ShapeError} If `y` is not 1-D or does not match the number of rows of `X`
   * @throws {NotFittedError} If the model has not been fitted
   */
  score(X: Tensor, y: Tensor): number {
    if (y.ndim !== 1) {
      throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    }
    assertContiguous(y, "y");
    const yPred = this.predict(X);
    if (y.size !== yPred.size) {
      throw new ShapeError(
        `X and y must have the same number of samples; got X.shape[0]=${yPred.size}, y.shape[0]=${y.size}`
      );
    }
    if (y.size === 0) {
      throw new DataValidationError("score requires at least one sample");
    }
    let correct = 0;
    for (let i = 0; i < y.size; i++) {
      if (Number(y.data[y.offset + i] ?? 0) === Number(yPred.data[yPred.offset + i] ?? 0)) {
        correct++;
      }
    }
    return correct / y.size;
  }

  /** Sorted class labels seen during fit, or `undefined` before fitting. */
  get classes(): Tensor | undefined {
    if (!this.fitted || !this.classes_) return undefined;
    return tensor(this.classes_, { dtype: "int32" });
  }

  /**
   * Number of features seen during `fit`.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get nFeaturesIn(): number {
    if (!this.fitted) {
      throw new NotFittedError("MLPClassifier must be fitted before accessing nFeaturesIn");
    }
    return this.nFeaturesIn_ ?? 0;
  }

  /**
   * Number of epochs run by the last `fit`.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get nIter(): number {
    if (!this.fitted) {
      throw new NotFittedError("MLPClassifier must be fitted before accessing nIter");
    }
    return this.nIter_;
  }

  /**
   * Mean cross-entropy of every epoch of the last `fit`.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get lossCurve(): number[] {
    if (!this.fitted) {
      throw new NotFittedError("MLPClassifier must be fitted before accessing lossCurve");
    }
    return [...this.lossCurve_];
  }

  /**
   * Cross-entropy of the final epoch of the last `fit`.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get loss(): number {
    if (!this.fitted) {
      throw new NotFittedError("MLPClassifier must be fitted before accessing loss");
    }
    return this.lossCurve_[this.lossCurve_.length - 1] ?? Number.NaN;
  }

  getParams(): Record<string, unknown> {
    return optionsToParams(this.opts);
  }

  /**
   * Update hyperparameters. Applies to the next `fit`; an already fitted network is unchanged.
   *
   * @throws {InvalidParameterError} On an unknown or invalid parameter
   */
  setParams(params: Record<string, unknown>): this {
    this.opts = mergeParams(this.opts, params);
    return this;
  }

  private forwardRows(X: Tensor): { out: Float64Array; classes: number[]; m: number } {
    if (!this.fitted || !this.net_ || !this.classes_) {
      throw new NotFittedError("MLPClassifier must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeaturesIn_ ?? 0, "MLPClassifier");
    const m = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    return {
      out: forwardAll(this.net_, toFloat64View(X), m, nFeatures),
      classes: this.classes_,
      m,
    };
  }
}

/**
 * Multi-layer Perceptron Regressor.
 *
 * A feedforward neural network regressor trained with backpropagation
 * (stochastic gradient descent, one sample per update). Uses an identity output
 * and minimizes half the squared error with L2 weight decay. The target is
 * standardized internally and predictions are mapped back to its original scale.
 *
 * Training stops when the epoch loss (on the standardized target) has not
 * improved by more than `tol` for more than `nIterNoChange` consecutive epochs,
 * or after `maxIter` epochs. Inputs should be standardized.
 *
 * @example
 * ```ts
 * import { MLPRegressor } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1], [2], [3], [4], [5]]);
 * const y = tensor([1.0, 4.0, 9.0, 16.0, 25.0]);
 *
 * const reg = new MLPRegressor({ hiddenLayerSizes: [10], learningRate: 0.01, maxIter: 500, randomState: 0 });
 * reg.fit(X, y);
 * const predictions = reg.predict(X);
 * ```
 */
export class MLPRegressor implements Regressor {
  private opts: ResolvedMLPOptions;

  private net_?: Network;
  private nFeaturesIn_?: number;
  private yMean_ = 0;
  private yStd_ = 1;
  private nIter_ = 0;
  private lossCurve_: number[] = [];
  private fitted = false;

  /**
   * @param options.hiddenLayerSizes - Array of hidden layer sizes (default: [100])
   * @param options.activation - Hidden activation (default: "relu")
   * @param options.learningRate - Constant step size of each per-sample SGD update (default: 0.001)
   * @param options.maxIter - Maximum epochs (default: 200)
   * @param options.tol - Minimum loss improvement that counts as progress (default: 1e-4)
   * @param options.alpha - L2 regularization (default: 1e-4)
   * @param options.shuffle - Shuffle the samples every epoch (default: true)
   * @param options.randomState - Seed for weight initialization and shuffling; without it the global Deepbox generator is used
   * @param options.nIterNoChange - Epochs without improvement tolerated before stopping (default: 10)
   */
  constructor(options: MLPOptions = {}) {
    this.opts = resolveOptions(options);
    validateOptions(this.opts);
  }

  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;

    const XData = toFloat64View(X);
    const yData = toFloat64View(y);

    // Standardize the target for stable training.
    let yMean = 0;
    for (let i = 0; i < nSamples; i++) yMean += yData[i] as number;
    yMean /= nSamples;
    let yVar = 0;
    for (let i = 0; i < nSamples; i++) yVar += ((yData[i] as number) - yMean) ** 2;
    let yStd = Math.sqrt(yVar / nSamples);
    if (yStd < 1e-10) yStd = 1;
    const T = new Float64Array(nSamples);
    for (let i = 0; i < nSamples; i++) T[i] = ((yData[i] as number) - yMean) / yStd;

    const rng = createRng(this.opts.randomState);
    const net: Network = {
      layers: buildLayers(nFeatures, this.opts.hiddenLayerSizes, 1, this.opts.activation, rng),
      activation: this.opts.activation,
      outputKind: "squared",
    };
    const { nIter, lossCurve } = train(net, XData, T, nSamples, nFeatures, {
      learningRate: this.opts.learningRate,
      maxIter: this.opts.maxIter,
      tol: this.opts.tol,
      alpha: this.opts.alpha,
      shuffle: this.opts.shuffle,
      nIterNoChange: this.opts.nIterNoChange,
      rng,
    });

    this.net_ = net;
    this.nFeaturesIn_ = nFeatures;
    this.yMean_ = yMean;
    this.yStd_ = yStd;
    this.nIter_ = nIter;
    this.lossCurve_ = lossCurve;
    this.fitted = true;
    return this;
  }

  /**
   * Predict target values.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predictions of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.net_) {
      throw new NotFittedError("MLPRegressor must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeaturesIn_ ?? 0, "MLPRegressor");
    const m = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const out = forwardAll(this.net_, toFloat64View(X), m, nFeatures);
    const predictions = new Array<number>(m);
    for (let i = 0; i < m; i++) predictions[i] = (out[i] as number) * this.yStd_ + this.yMean_;
    return tensor(predictions, { dtype: "float64" });
  }

  /**
   * Coefficient of determination R^2 on the given data.
   *
   * @throws {ShapeError} If `y` is not 1-D or does not match the number of rows of `X`
   * @throws {DataValidationError} If `y` contains NaN or Infinity
   * @throws {NotFittedError} If the model has not been fitted
   */
  score(X: Tensor, y: Tensor): number {
    if (y.ndim !== 1) {
      throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    }
    assertContiguous(y, "y");
    // Read through toFloat64View: int64 targets hold BigInt values, which Number.isFinite rejects.
    const yv = toFloat64View(y);
    for (let i = 0; i < yv.length; i++) {
      if (!Number.isFinite(yv[i])) {
        throw new DataValidationError("y contains non-finite values (NaN or Inf)");
      }
    }
    const predictions = this.predict(X);
    if (y.size !== predictions.size) {
      throw new ShapeError(
        `X and y must have the same number of samples; got X.shape[0]=${predictions.size}, y.shape[0]=${y.size}`
      );
    }
    if (y.size === 0) {
      throw new DataValidationError("score requires at least one sample");
    }
    return r2Score(yv, toFloat64View(predictions));
  }

  /**
   * Number of features seen during `fit`.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get nFeaturesIn(): number {
    if (!this.fitted) {
      throw new NotFittedError("MLPRegressor must be fitted before accessing nFeaturesIn");
    }
    return this.nFeaturesIn_ ?? 0;
  }

  /**
   * Number of epochs run by the last `fit`.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get nIter(): number {
    if (!this.fitted) {
      throw new NotFittedError("MLPRegressor must be fitted before accessing nIter");
    }
    return this.nIter_;
  }

  /**
   * Mean half squared error (on the standardized target) of every epoch of the last `fit`.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get lossCurve(): number[] {
    if (!this.fitted) {
      throw new NotFittedError("MLPRegressor must be fitted before accessing lossCurve");
    }
    return [...this.lossCurve_];
  }

  /**
   * Loss of the final epoch of the last `fit`.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get loss(): number {
    if (!this.fitted) {
      throw new NotFittedError("MLPRegressor must be fitted before accessing loss");
    }
    return this.lossCurve_[this.lossCurve_.length - 1] ?? Number.NaN;
  }

  getParams(): Record<string, unknown> {
    return optionsToParams(this.opts);
  }

  /**
   * Update hyperparameters. Applies to the next `fit`; an already fitted network is unchanged.
   *
   * @throws {InvalidParameterError} On an unknown or invalid parameter
   */
  setParams(params: Record<string, unknown>): this {
    this.opts = mergeParams(this.opts, params);
    return this;
  }
}
