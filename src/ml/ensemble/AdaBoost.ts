/**
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { DataValidationError, InvalidParameterError, NotFittedError, ShapeError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { __random } from "../../random/random";
import { assertContiguous, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Classifier, Regressor } from "../base";
import { DecisionTreeClassifier, DecisionTreeRegressor } from "../tree/DecisionTree";

/**
 * AdaBoost Classifier (Adaptive Boosting).
 *
 * Fits a sequence of weak classifiers (decision stumps by default) on
 * re-weighted versions of the data, then combines them via weighted majority vote.
 *
 * **Algorithm**: SAMME (Stagewise Additive Modeling using a Multi-class Exponential loss)
 *
 * @example
 * ```ts
 * import { AdaBoostClassifier } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const clf = new AdaBoostClassifier({ nEstimators: 50 });
 * clf.fit(X_train, y_train);
 * const predictions = clf.predict(X_test);
 * ```
 */
export class AdaBoostClassifier implements Classifier {
  private nEstimators: number;
  private learningRate: number;
  private maxDepth: number;

  private estimators: DecisionTreeClassifier[] = [];
  private estimatorWeights: number[] = [];
  private classLabels: number[] = [];
  private nFeatures = 0;
  private fitted = false;

  constructor(
    options: {
      nEstimators?: number;
      learningRate?: number;
      maxDepth?: number;
    } = {}
  ) {
    this.nEstimators = options.nEstimators ?? 50;
    this.learningRate = options.learningRate ?? 1.0;
    this.maxDepth = options.maxDepth ?? 1; // decision stumps by default

    if (!Number.isInteger(this.nEstimators) || this.nEstimators <= 0) {
      throw new InvalidParameterError(
        "nEstimators must be a positive integer",
        "nEstimators",
        this.nEstimators
      );
    }
    if (!Number.isFinite(this.learningRate) || this.learningRate <= 0) {
      throw new InvalidParameterError(
        "learningRate must be positive",
        "learningRate",
        this.learningRate
      );
    }
  }

  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    this.nFeatures = nFeatures;

    // Extract y data and class labels
    const yData: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      yData.push(Number(y.data[y.offset + i]));
    }
    this.classLabels = [...new Set(yData)].sort((a, b) => a - b);
    const nClasses = this.classLabels.length;

    if (nClasses < 2) {
      throw new InvalidParameterError(
        "AdaBoostClassifier requires at least 2 classes",
        "y",
        nClasses
      );
    }

    // Initialize sample weights uniformly
    const sampleWeights = new Float64Array(nSamples).fill(1 / nSamples);

    this.estimators = [];
    this.estimatorWeights = [];

    for (let m = 0; m < this.nEstimators; m++) {
      // Train a weak learner with weighted sampling (bootstrap by weight)
      const { sampledX, sampledY } = this.weightedBootstrap(
        X,
        y,
        sampleWeights,
        nSamples,
        nFeatures
      );

      const tree = new DecisionTreeClassifier({
        maxDepth: this.maxDepth,
        minSamplesSplit: 2,
        minSamplesLeaf: 1,
      });
      tree.fit(sampledX, sampledY);

      // Predict on full training set
      const predictions = tree.predict(X);

      // Compute weighted error
      let weightedError = 0;
      for (let i = 0; i < nSamples; i++) {
        if (Number(predictions.data[predictions.offset + i]) !== yData[i]) {
          weightedError += sampleWeights[i] ?? 0;
        }
      }

      // Clip error to avoid log(0)
      weightedError = Math.max(weightedError, 1e-10);
      weightedError = Math.min(weightedError, 1 - 1e-10);

      // If error >= 0.5 for binary or >= 1 - 1/nClasses for multiclass, stop
      if (weightedError >= 1 - 1 / nClasses) {
        if (this.estimators.length === 0) {
          // Keep at least one estimator
          this.estimators.push(tree);
          this.estimatorWeights.push(1.0);
        }
        break;
      }

      // SAMME estimator weight
      const alpha =
        this.learningRate *
        (Math.log((1 - weightedError) / weightedError) + Math.log(nClasses - 1));

      this.estimators.push(tree);
      this.estimatorWeights.push(alpha);

      // Update sample weights
      for (let i = 0; i < nSamples; i++) {
        if (Number(predictions.data[predictions.offset + i]) !== yData[i]) {
          sampleWeights[i] = (sampleWeights[i] ?? 0) * Math.exp(alpha);
        }
      }

      // Normalize weights
      let wSum = 0;
      for (let i = 0; i < nSamples; i++) {
        wSum += sampleWeights[i] ?? 0;
      }
      if (wSum > 0) {
        for (let i = 0; i < nSamples; i++) {
          sampleWeights[i] = (sampleWeights[i] ?? 0) / wSum;
        }
      }
    }

    this.fitted = true;
    return this;
  }

  private weightedBootstrap(
    X: Tensor,
    y: Tensor,
    weights: Float64Array,
    nSamples: number,
    nFeatures: number
  ): { sampledX: Tensor; sampledY: Tensor } {
    // Sample indices proportional to weights using cumulative distribution
    const cumWeights = new Float64Array(nSamples);
    cumWeights[0] = weights[0] ?? 0;
    for (let i = 1; i < nSamples; i++) {
      cumWeights[i] = (cumWeights[i - 1] ?? 0) + (weights[i] ?? 0);
    }

    const xData: number[][] = [];
    const yData: number[] = [];

    for (let s = 0; s < nSamples; s++) {
      const r = __random();
      // Binary search for index
      let lo = 0;
      let hi = nSamples - 1;
      while (lo < hi) {
        const mid = (lo + hi) >>> 1;
        if ((cumWeights[mid] ?? 0) < r) {
          lo = mid + 1;
        } else {
          hi = mid;
        }
      }
      const idx = lo;
      const row: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        row.push(Number(X.data[X.offset + idx * nFeatures + j] ?? 0));
      }
      xData.push(row);
      yData.push(Number(y.data[y.offset + idx] ?? 0));
    }

    return { sampledX: tensor(xData), sampledY: tensor(yData) };
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("AdaBoostClassifier must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeatures, "AdaBoostClassifier");

    const nSamples = X.shape[0] ?? 0;
    const nClasses = this.classLabels.length;
    const predictions: number[] = [];

    for (let i = 0; i < nSamples; i++) {
      // Weighted vote per class
      const classScores = new Array<number>(nClasses).fill(0);
      for (let m = 0; m < this.estimators.length; m++) {
        const tree = this.estimators[m];
        if (!tree) continue;
        const sample = tensor([
          Array.from({ length: X.shape[1] ?? 0 }, (_, j) =>
            Number(X.data[X.offset + i * (X.shape[1] ?? 0) + j] ?? 0)
          ),
        ]);
        const pred = Number(tree.predict(sample).data[0]);
        const classIdx = this.classLabels.indexOf(pred);
        if (classIdx >= 0) {
          classScores[classIdx] = (classScores[classIdx] ?? 0) + (this.estimatorWeights[m] ?? 0);
        }
      }

      let bestClass = 0;
      let bestScore = -Infinity;
      for (let c = 0; c < nClasses; c++) {
        if ((classScores[c] ?? 0) > bestScore) {
          bestScore = classScores[c] ?? 0;
          bestClass = c;
        }
      }
      predictions.push(this.classLabels[bestClass] ?? 0);
    }

    return tensor(predictions, { dtype: "int32" });
  }

  predictProba(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("AdaBoostClassifier must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeatures, "AdaBoostClassifier");

    const nSamples = X.shape[0] ?? 0;
    const nClasses = this.classLabels.length;
    const probabilities: number[][] = [];

    for (let i = 0; i < nSamples; i++) {
      const classScores = new Array<number>(nClasses).fill(0);
      for (let m = 0; m < this.estimators.length; m++) {
        const tree = this.estimators[m];
        if (!tree) continue;
        const sample = tensor([
          Array.from({ length: X.shape[1] ?? 0 }, (_, j) =>
            Number(X.data[X.offset + i * (X.shape[1] ?? 0) + j] ?? 0)
          ),
        ]);
        const pred = Number(tree.predict(sample).data[0]);
        const classIdx = this.classLabels.indexOf(pred);
        if (classIdx >= 0) {
          classScores[classIdx] = (classScores[classIdx] ?? 0) + (this.estimatorWeights[m] ?? 0);
        }
      }

      // Softmax-like normalization of weighted scores
      const maxScore = Math.max(...classScores);
      const exps = classScores.map((s) => Math.exp(s - maxScore));
      const sumExps = exps.reduce((a, b) => a + b, 0);
      probabilities.push(exps.map((e) => e / sumExps));
    }

    return tensor(probabilities);
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
    let correct = 0;
    for (let i = 0; i < y.size; i++) {
      if (Number(predictions.data[predictions.offset + i]) === Number(y.data[y.offset + i])) {
        correct++;
      }
    }
    return correct / y.size;
  }

  get classes(): Tensor | undefined {
    if (!this.fitted || this.classLabels.length === 0) return undefined;
    return tensor(this.classLabels, { dtype: "int32" });
  }

  get featureImportances(): Tensor {
    if (!this.fitted || this.estimators.length === 0 || this.nFeatures === 0) {
      throw new NotFittedError("AdaBoostClassifier must be fitted to access feature_importances_");
    }
    const nF = this.nFeatures;
    const avg = new Array<number>(nF).fill(0);
    let totalWeight = 0;
    for (let m = 0; m < this.estimators.length; m++) {
      const w = this.estimatorWeights[m] ?? 0;
      totalWeight += w;
      const tree = this.estimators[m];
      if (!tree) continue;
      const treeImp = tree.featureImportances;
      for (let j = 0; j < nF; j++) {
        avg[j] = (avg[j] ?? 0) + w * Number(treeImp.data[treeImp.offset + j] ?? 0);
      }
    }
    if (totalWeight > 0) {
      let total = 0;
      for (let j = 0; j < nF; j++) {
        avg[j] = (avg[j] ?? 0) / totalWeight;
        total += avg[j] ?? 0;
      }
      if (total > 0) {
        for (let j = 0; j < nF; j++) {
          avg[j] = (avg[j] ?? 0) / total;
        }
      }
    }
    return tensor(avg);
  }

  getParams(): Record<string, unknown> {
    return {
      nEstimators: this.nEstimators,
      learningRate: this.learningRate,
      maxDepth: this.maxDepth,
    };
  }

  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "nEstimators":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError(
              "nEstimators must be an integer >= 1",
              "nEstimators",
              value
            );
          }
          this.nEstimators = value;
          break;
        case "learningRate":
          if (typeof value !== "number" || value <= 0) {
            throw new InvalidParameterError("learningRate must be > 0", "learningRate", value);
          }
          this.learningRate = value;
          break;
        case "maxDepth":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError("maxDepth must be an integer >= 1", "maxDepth", value);
          }
          this.maxDepth = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }

  clone(): AdaBoostClassifier {
    return new AdaBoostClassifier(
      this.getParams() as {
        nEstimators?: number;
        learningRate?: number;
        maxDepth?: number;
      }
    );
  }
}

/**
 * AdaBoost Regressor (Adaptive Boosting for Regression).
 *
 * Uses the AdaBoost.R2 algorithm with decision tree regressors as base estimators.
 *
 * @example
 * ```ts
 * import { AdaBoostRegressor } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const reg = new AdaBoostRegressor({ nEstimators: 50 });
 * reg.fit(X_train, y_train);
 * const predictions = reg.predict(X_test);
 * ```
 */
export class AdaBoostRegressor implements Regressor {
  private nEstimators: number;
  private learningRate: number;
  private maxDepth: number;
  private loss: "linear" | "square" | "exponential";

  private estimators: DecisionTreeRegressor[] = [];
  private estimatorWeights: number[] = [];
  private nFeatures = 0;
  private fitted = false;

  constructor(
    options: {
      readonly nEstimators?: number;
      readonly learningRate?: number;
      readonly maxDepth?: number;
      readonly loss?: "linear" | "square" | "exponential";
    } = {}
  ) {
    this.nEstimators = options.nEstimators ?? 50;
    this.learningRate = options.learningRate ?? 1.0;
    this.maxDepth = options.maxDepth ?? 3;
    this.loss = options.loss ?? "linear";

    if (!Number.isInteger(this.nEstimators) || this.nEstimators <= 0) {
      throw new InvalidParameterError(
        "nEstimators must be a positive integer",
        "nEstimators",
        this.nEstimators
      );
    }
    if (!Number.isFinite(this.learningRate) || this.learningRate <= 0) {
      throw new InvalidParameterError(
        "learningRate must be positive",
        "learningRate",
        this.learningRate
      );
    }
  }

  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    this.nFeatures = nFeatures;

    const yData: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      yData.push(Number(y.data[y.offset + i]));
    }

    // Initialize sample weights uniformly
    const sampleWeights = new Float64Array(nSamples).fill(1 / nSamples);

    this.estimators = [];
    this.estimatorWeights = [];

    for (let m = 0; m < this.nEstimators; m++) {
      // Weighted bootstrap
      const { sampledX, sampledY } = this.weightedBootstrap(
        X,
        y,
        sampleWeights,
        nSamples,
        nFeatures
      );

      const tree = new DecisionTreeRegressor({
        maxDepth: this.maxDepth,
        minSamplesSplit: 2,
        minSamplesLeaf: 1,
      });
      tree.fit(sampledX, sampledY);

      const predictions = tree.predict(X);

      // Compute max absolute error for normalization
      let maxError = 0;
      for (let i = 0; i < nSamples; i++) {
        const err = Math.abs(yData[i]! - Number(predictions.data[predictions.offset + i]));
        if (err > maxError) maxError = err;
      }

      if (maxError === 0) {
        // Perfect fit
        this.estimators.push(tree);
        this.estimatorWeights.push(1.0);
        break;
      }

      // Compute normalized losses
      const losses = new Float64Array(nSamples);
      for (let i = 0; i < nSamples; i++) {
        const normalizedErr =
          Math.abs(yData[i]! - Number(predictions.data[predictions.offset + i])) / maxError;
        if (this.loss === "linear") {
          losses[i] = normalizedErr;
        } else if (this.loss === "square") {
          losses[i] = normalizedErr * normalizedErr;
        } else {
          losses[i] = 1 - Math.exp(-normalizedErr);
        }
      }

      // Weighted average loss
      let avgLoss = 0;
      for (let i = 0; i < nSamples; i++) {
        avgLoss += (sampleWeights[i] ?? 0) * (losses[i] ?? 0);
      }

      if (avgLoss >= 0.5) {
        if (this.estimators.length === 0) {
          this.estimators.push(tree);
          this.estimatorWeights.push(1.0);
        }
        break;
      }

      const beta = avgLoss / (1 - avgLoss);
      const alpha = this.learningRate * Math.log(1 / Math.max(beta, 1e-10));

      this.estimators.push(tree);
      this.estimatorWeights.push(alpha);

      // Update sample weights
      for (let i = 0; i < nSamples; i++) {
        sampleWeights[i] = (sampleWeights[i] ?? 0) * beta ** (1 - (losses[i] ?? 0));
      }

      // Normalize weights
      let wSum = 0;
      for (let i = 0; i < nSamples; i++) {
        wSum += sampleWeights[i] ?? 0;
      }
      if (wSum > 0) {
        for (let i = 0; i < nSamples; i++) {
          sampleWeights[i] = (sampleWeights[i] ?? 0) / wSum;
        }
      }
    }

    this.fitted = true;
    return this;
  }

  private weightedBootstrap(
    X: Tensor,
    y: Tensor,
    weights: Float64Array,
    nSamples: number,
    nFeatures: number
  ): { sampledX: Tensor; sampledY: Tensor } {
    const cumWeights = new Float64Array(nSamples);
    cumWeights[0] = weights[0] ?? 0;
    for (let i = 1; i < nSamples; i++) {
      cumWeights[i] = (cumWeights[i - 1] ?? 0) + (weights[i] ?? 0);
    }

    const xData: number[][] = [];
    const yData: number[] = [];

    for (let s = 0; s < nSamples; s++) {
      const r = __random();
      let lo = 0;
      let hi = nSamples - 1;
      while (lo < hi) {
        const mid = (lo + hi) >>> 1;
        if ((cumWeights[mid] ?? 0) < r) {
          lo = mid + 1;
        } else {
          hi = mid;
        }
      }
      const idx = lo;
      const row: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        row.push(Number(X.data[X.offset + idx * nFeatures + j] ?? 0));
      }
      xData.push(row);
      yData.push(Number(y.data[y.offset + idx] ?? 0));
    }

    return { sampledX: tensor(xData), sampledY: tensor(yData) };
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("AdaBoostRegressor must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeatures, "AdaBoostRegressor");

    const nSamples = X.shape[0] ?? 0;
    const predictions: number[] = [];

    // Weighted median prediction
    for (let i = 0; i < nSamples; i++) {
      const sample = tensor([
        Array.from({ length: X.shape[1] ?? 0 }, (_, j) =>
          Number(X.data[X.offset + i * (X.shape[1] ?? 0) + j] ?? 0)
        ),
      ]);

      // Collect (prediction, weight) pairs
      const predWeightPairs: Array<{ pred: number; weight: number }> = [];
      for (let m = 0; m < this.estimators.length; m++) {
        const tree = this.estimators[m];
        if (!tree) continue;
        const p = Number(tree.predict(sample).data[0]);
        predWeightPairs.push({
          pred: p,
          weight: this.estimatorWeights[m] ?? 0,
        });
      }

      // Weighted median
      predWeightPairs.sort((a, b) => a.pred - b.pred);
      let totalW = 0;
      for (const pw of predWeightPairs) totalW += pw.weight;
      let cumW = 0;
      let median = predWeightPairs[0]?.pred ?? 0;
      for (const pw of predWeightPairs) {
        cumW += pw.weight;
        if (cumW >= totalW / 2) {
          median = pw.pred;
          break;
        }
      }
      predictions.push(median);
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
    if (predictions.size !== y.size) {
      throw new ShapeError(
        `X and y must have the same number of samples; got X=${predictions.size}, y=${y.size}`
      );
    }
    let ssRes = 0;
    let ssTot = 0;
    let yMean = 0;
    for (let i = 0; i < y.size; i++) {
      yMean += Number(y.data[y.offset + i]);
    }
    yMean /= y.size;
    for (let i = 0; i < y.size; i++) {
      const yVal = Number(y.data[y.offset + i]);
      const pVal = Number(predictions.data[predictions.offset + i]);
      ssRes += (yVal - pVal) ** 2;
      ssTot += (yVal - yMean) ** 2;
    }
    return ssTot === 0 ? (ssRes === 0 ? 1.0 : 0.0) : 1 - ssRes / ssTot;
  }

  get featureImportances(): Tensor {
    if (!this.fitted || this.estimators.length === 0 || this.nFeatures === 0) {
      throw new NotFittedError("AdaBoostRegressor must be fitted to access feature_importances_");
    }
    const nF = this.nFeatures;
    const avg = new Array<number>(nF).fill(0);
    let totalWeight = 0;
    for (let m = 0; m < this.estimators.length; m++) {
      const w = this.estimatorWeights[m] ?? 0;
      totalWeight += w;
      const tree = this.estimators[m];
      if (!tree) continue;
      const treeImp = tree.featureImportances;
      for (let j = 0; j < nF; j++) {
        avg[j] = (avg[j] ?? 0) + w * Number(treeImp.data[treeImp.offset + j] ?? 0);
      }
    }
    if (totalWeight > 0) {
      let total = 0;
      for (let j = 0; j < nF; j++) {
        avg[j] = (avg[j] ?? 0) / totalWeight;
        total += avg[j] ?? 0;
      }
      if (total > 0) {
        for (let j = 0; j < nF; j++) {
          avg[j] = (avg[j] ?? 0) / total;
        }
      }
    }
    return tensor(avg);
  }

  getParams(): Record<string, unknown> {
    return {
      nEstimators: this.nEstimators,
      learningRate: this.learningRate,
      maxDepth: this.maxDepth,
      loss: this.loss,
    };
  }

  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "nEstimators":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError(
              "nEstimators must be an integer >= 1",
              "nEstimators",
              value
            );
          }
          this.nEstimators = value;
          break;
        case "learningRate":
          if (typeof value !== "number" || value <= 0) {
            throw new InvalidParameterError("learningRate must be > 0", "learningRate", value);
          }
          this.learningRate = value;
          break;
        case "maxDepth":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError("maxDepth must be an integer >= 1", "maxDepth", value);
          }
          this.maxDepth = value;
          break;
        case "loss":
          if (value !== "linear" && value !== "square" && value !== "exponential") {
            throw new InvalidParameterError(
              `loss must be "linear", "square", or "exponential"`,
              "loss",
              value
            );
          }
          this.loss = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }

  clone(): AdaBoostRegressor {
    return new AdaBoostRegressor(
      this.getParams() as {
        nEstimators?: number;
        learningRate?: number;
        maxDepth?: number;
        loss?: "linear" | "square" | "exponential";
      }
    );
  }
}
