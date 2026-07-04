import { DataValidationError, InvalidParameterError, NotFittedError, ShapeError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { __random } from "../../random/random";
import { assertContiguous, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Classifier, Regressor } from "../base";
import { DecisionTreeRegressor } from "../tree/DecisionTree";

/**
 * Average feature importances across an array of DecisionTreeRegressor instances.
 * Returns a normalized tensor of shape (nFeatures,) summing to 1.
 */
function averageTreeImportances(trees: DecisionTreeRegressor[], nFeatures: number): Tensor {
  const avg = new Array<number>(nFeatures).fill(0);
  for (const tree of trees) {
    const imp = tree.featureImportances;
    for (let j = 0; j < nFeatures; j++) {
      avg[j] = (avg[j] ?? 0) + Number(imp.data[imp.offset + j] ?? 0);
    }
  }
  const nTrees = trees.length;
  let total = 0;
  for (let j = 0; j < nFeatures; j++) {
    avg[j] = (avg[j] ?? 0) / nTrees;
    total += avg[j] ?? 0;
  }
  if (total > 0) {
    for (let j = 0; j < nFeatures; j++) {
      avg[j] = (avg[j] ?? 0) / total;
    }
  }
  return tensor(avg);
}

/**
 * Gradient Boosting Regressor.
 *
 * Builds an additive model in a forward stage-wise fashion using
 * regression trees as weak learners. Optimizes squared error loss.
 *
 * **Algorithm**: Gradient Boosting with regression trees
 * - Stage-wise additive modeling
 * - Uses gradient of squared loss (residuals)
 *
 * @example
 * ```ts
 * import { GradientBoostingRegressor } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1], [2], [3], [4], [5]]);
 * const y = tensor([1.2, 2.1, 2.9, 4.0, 5.1]);
 *
 * const gbr = new GradientBoostingRegressor({ nEstimators: 100 });
 * gbr.fit(X, y);
 * const predictions = gbr.predict(X);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-ensemble | Deepbox Ensemble Methods}
 */
export class GradientBoostingRegressor implements Regressor {
  /** Number of boosting stages (trees) */
  private nEstimators: number;

  /** Learning rate shrinks the contribution of each tree */
  private learningRate: number;

  /** Maximum depth of individual regression trees */
  private maxDepth: number;

  /** Minimum samples required to split */
  private minSamplesSplit: number;

  /** Whether to reuse previously fitted trees and add more */
  private warmStart: boolean;

  /** Fraction of samples to use per boosting stage (stochastic GB) */
  private subsample: number;

  /** Number of features to consider for best split at each tree */
  private maxFeatures: "sqrt" | "log2" | number | undefined;

  /** Fraction of training data to set aside for early stopping validation */
  private validationFraction: number;

  /** Number of iterations with no improvement before stopping */
  private nIterNoChange: number | undefined;

  /** Loss function */
  private loss: "ls" | "lad" | "huber" | "quantile";

  /** Quantile for huber/quantile loss */
  private alpha: number;

  /** Array of weak learners (regression trees) */
  private estimators: DecisionTreeRegressor[] = [];

  /** Initial prediction (mean of targets) */
  private initPrediction = 0;

  /** Number of features */
  private nFeatures = 0;

  /** Whether the model has been fitted */
  private fitted = false;

  /** Number of estimators actually fitted (may be < nEstimators if early stopped) */
  private nEstimatorsFitted_ = 0;

  constructor(
    options: {
      readonly nEstimators?: number;
      readonly learningRate?: number;
      readonly maxDepth?: number;
      readonly minSamplesSplit?: number;
      readonly warmStart?: boolean;
      readonly subsample?: number;
      readonly maxFeatures?: "sqrt" | "log2" | number;
      readonly validationFraction?: number;
      readonly nIterNoChange?: number;
      readonly loss?: "ls" | "lad" | "huber" | "quantile";
      readonly alpha?: number;
    } = {}
  ) {
    this.nEstimators = options.nEstimators ?? 100;
    this.learningRate = options.learningRate ?? 0.1;
    this.maxDepth = options.maxDepth ?? 3;
    this.minSamplesSplit = options.minSamplesSplit ?? 2;
    this.warmStart = options.warmStart ?? false;
    this.subsample = options.subsample ?? 1.0;
    if (options.maxFeatures !== undefined) {
      this.maxFeatures = options.maxFeatures;
    }
    this.validationFraction = options.validationFraction ?? 0.1;
    if (options.nIterNoChange !== undefined) {
      this.nIterNoChange = options.nIterNoChange;
    }
    this.loss = options.loss ?? "ls";
    this.alpha = options.alpha ?? 0.9;

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
    if (!Number.isInteger(this.maxDepth) || this.maxDepth < 1) {
      throw new InvalidParameterError(
        "maxDepth must be an integer >= 1",
        "maxDepth",
        this.maxDepth
      );
    }
    if (!Number.isInteger(this.minSamplesSplit) || this.minSamplesSplit < 2) {
      throw new InvalidParameterError(
        "minSamplesSplit must be an integer >= 2",
        "minSamplesSplit",
        this.minSamplesSplit
      );
    }
    if (!Number.isFinite(this.subsample) || this.subsample <= 0 || this.subsample > 1) {
      throw new InvalidParameterError("subsample must be in (0, 1]", "subsample", this.subsample);
    }
    if (
      this.loss !== "ls" &&
      this.loss !== "lad" &&
      this.loss !== "huber" &&
      this.loss !== "quantile"
    ) {
      throw new InvalidParameterError(
        "loss must be 'ls', 'lad', 'huber', or 'quantile'",
        "loss",
        this.loss
      );
    }
    if (!Number.isFinite(this.alpha) || this.alpha <= 0 || this.alpha >= 1) {
      throw new InvalidParameterError("alpha must be in (0, 1)", "alpha", this.alpha);
    }
  }

  /**
   * Compute the negative gradient (pseudo-residuals) for the chosen loss.
   */
  private computeNegGradient(yData: number[], predictions: number[], nSamples: number): number[] {
    const residuals: number[] = [];
    switch (this.loss) {
      case "ls":
        // Squared loss: negative gradient = y - f(x)
        for (let i = 0; i < nSamples; i++) {
          residuals.push((yData[i] ?? 0) - (predictions[i] ?? 0));
        }
        break;
      case "lad":
        // Absolute loss: negative gradient = sign(y - f(x))
        for (let i = 0; i < nSamples; i++) {
          const diff = (yData[i] ?? 0) - (predictions[i] ?? 0);
          residuals.push(diff > 0 ? 1 : diff < 0 ? -1 : 0);
        }
        break;
      case "huber": {
        // Huber loss: uses alpha-quantile of absolute residuals as threshold
        const absResiduals: number[] = [];
        for (let i = 0; i < nSamples; i++) {
          absResiduals.push(Math.abs((yData[i] ?? 0) - (predictions[i] ?? 0)));
        }
        const sorted = [...absResiduals].sort((a, b) => a - b);
        const idx = Math.min(Math.floor(this.alpha * sorted.length), sorted.length - 1);
        const delta = sorted[idx] ?? 0;
        for (let i = 0; i < nSamples; i++) {
          const diff = (yData[i] ?? 0) - (predictions[i] ?? 0);
          if (Math.abs(diff) <= delta) {
            residuals.push(diff);
          } else {
            residuals.push(delta * (diff > 0 ? 1 : -1));
          }
        }
        break;
      }
      case "quantile":
        // Quantile loss: negative gradient
        for (let i = 0; i < nSamples; i++) {
          const diff = (yData[i] ?? 0) - (predictions[i] ?? 0);
          residuals.push(diff >= 0 ? this.alpha : this.alpha - 1);
        }
        break;
    }
    return residuals;
  }

  /**
   * Compute loss for early stopping validation.
   */
  private computeLoss(yData: number[], predictions: number[]): number {
    let loss = 0;
    const n = yData.length;
    if (n === 0) return 0;
    switch (this.loss) {
      case "ls":
        for (let i = 0; i < n; i++) {
          loss += ((yData[i] ?? 0) - (predictions[i] ?? 0)) ** 2;
        }
        return loss / n;
      case "lad":
        for (let i = 0; i < n; i++) {
          loss += Math.abs((yData[i] ?? 0) - (predictions[i] ?? 0));
        }
        return loss / n;
      case "huber": {
        const absRes: number[] = [];
        for (let i = 0; i < n; i++) {
          absRes.push(Math.abs((yData[i] ?? 0) - (predictions[i] ?? 0)));
        }
        const sorted = [...absRes].sort((a, b) => a - b);
        const idx = Math.min(Math.floor(this.alpha * sorted.length), sorted.length - 1);
        const delta = sorted[idx] ?? 0;
        for (let i = 0; i < n; i++) {
          const a = absRes[i] ?? 0;
          loss += a <= delta ? 0.5 * a * a : delta * (a - 0.5 * delta);
        }
        return loss / n;
      }
      case "quantile":
        for (let i = 0; i < n; i++) {
          const diff = (yData[i] ?? 0) - (predictions[i] ?? 0);
          loss += diff >= 0 ? this.alpha * diff : (this.alpha - 1) * diff;
        }
        return loss / n;
    }
  }

  /**
   * Resolve maxFeatures into an actual integer count.
   */
  private resolveMaxFeatures(nFeatures: number): number | undefined {
    if (this.maxFeatures === undefined) return undefined;
    if (typeof this.maxFeatures === "number") {
      return Math.max(1, Math.min(this.maxFeatures, nFeatures));
    }
    if (this.maxFeatures === "sqrt") {
      return Math.max(1, Math.floor(Math.sqrt(nFeatures)));
    }
    // log2
    return Math.max(1, Math.floor(Math.log2(nFeatures)));
  }

  /**
   * Fit the gradient boosting regressor on training data.
   *
   * Builds an additive model by sequentially fitting regression trees
   * to the negative gradient (residuals) of the loss function.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Target values of shape (n_samples,)
   * @returns this - The fitted estimator
   * @throws {ShapeError} If X is not 2D or y is not 1D
   * @throws {ShapeError} If X and y have different number of samples
   * @throws {DataValidationError} If X or y contain NaN/Inf values
   */
  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;

    this.nFeatures = nFeatures;

    const yData: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      yData.push(Number(y.data[y.offset + i]));
    }

    // Split off validation data for early stopping
    let trainX = X;
    let trainYData = yData;
    let trainN = nSamples;
    let valYData: number[] = [];
    let valX: Tensor | undefined;
    if (this.nIterNoChange !== undefined) {
      const valSize = Math.max(1, Math.floor(this.validationFraction * nSamples));
      trainN = nSamples - valSize;
      if (trainN < 1) {
        trainN = nSamples;
      } else {
        trainYData = yData.slice(0, trainN);
        valYData = yData.slice(trainN);
        // Build train/val tensors
        const trainRows: number[] = [];
        const valRows: number[] = [];
        for (let i = 0; i < trainN; i++) {
          for (let j = 0; j < nFeatures; j++) {
            trainRows.push(Number(X.data[X.offset + i * nFeatures + j]));
          }
        }
        for (let i = trainN; i < nSamples; i++) {
          for (let j = 0; j < nFeatures; j++) {
            valRows.push(Number(X.data[X.offset + i * nFeatures + j]));
          }
        }
        trainX = tensor(trainRows).reshape([trainN, nFeatures]);
        valX = tensor(valRows).reshape([nSamples - trainN, nFeatures]);
      }
    }

    // Warm start: continue from existing ensemble
    let startIdx = 0;
    if (this.warmStart && this.fitted && this.estimators.length > 0) {
      startIdx = this.estimators.length;
      if (startIdx >= this.nEstimators) {
        return this; // Already have enough estimators
      }
    } else {
      this.initPrediction = trainYData.reduce((sum, val) => sum + val, 0) / trainN;
      this.estimators = [];
    }

    // Current predictions on training data
    const predictions = new Array<number>(trainN).fill(this.initPrediction);
    // Replay existing trees
    for (const tree of this.estimators) {
      const treePred = tree.predict(trainX);
      for (let i = 0; i < trainN; i++) {
        predictions[i] =
          (predictions[i] ?? 0) + this.learningRate * Number(treePred.data[treePred.offset + i]);
      }
    }

    // Early stopping state
    let bestValLoss = Infinity;
    let noImprovementCount = 0;

    // Subsample draw count
    const drawSize = this.subsample < 1 ? Math.max(1, Math.floor(this.subsample * trainN)) : trainN;
    const useSubsample = drawSize < trainN;

    // Resolve maxFeatures for trees
    const treeMaxFeatures = this.resolveMaxFeatures(nFeatures);

    for (let m = startIdx; m < this.nEstimators; m++) {
      // Compute negative gradient (pseudo-residuals)
      const residuals = this.computeNegGradient(trainYData, predictions, trainN);

      // Subsample
      let fitX = trainX;
      let fitResiduals = residuals;
      if (useSubsample) {
        const indices: number[] = [];
        for (let i = 0; i < drawSize; i++) {
          indices.push(Math.floor(__random() * trainN));
        }
        const subRows: number[] = [];
        const subRes: number[] = [];
        for (const idx of indices) {
          for (let j = 0; j < nFeatures; j++) {
            subRows.push(Number(trainX.data[trainX.offset + idx * nFeatures + j]));
          }
          subRes.push(residuals[idx] ?? 0);
        }
        fitX = tensor(subRows).reshape([drawSize, nFeatures]);
        fitResiduals = subRes;
      }

      // Fit a regression tree to residuals
      const treeOpts: {
        maxDepth: number;
        minSamplesSplit: number;
        minSamplesLeaf: number;
        maxFeatures?: number;
      } = {
        maxDepth: this.maxDepth,
        minSamplesSplit: this.minSamplesSplit,
        minSamplesLeaf: 1,
      };
      if (treeMaxFeatures !== undefined) {
        treeOpts.maxFeatures = treeMaxFeatures;
      }
      const tree = new DecisionTreeRegressor(treeOpts);
      tree.fit(fitX, tensor(fitResiduals));
      this.estimators.push(tree);

      // Update training predictions
      const treePred = tree.predict(trainX);
      for (let i = 0; i < trainN; i++) {
        predictions[i] =
          (predictions[i] ?? 0) + this.learningRate * Number(treePred.data[treePred.offset + i]);
      }

      this.nEstimatorsFitted_ = m + 1;

      // Early stopping check
      if (this.nIterNoChange !== undefined && valX !== undefined && valYData.length > 0) {
        // Compute val predictions
        const valN = valYData.length;
        const valPreds = new Array<number>(valN).fill(this.initPrediction);
        for (const t of this.estimators) {
          const vp = t.predict(valX);
          for (let i = 0; i < valN; i++) {
            valPreds[i] = (valPreds[i] ?? 0) + this.learningRate * Number(vp.data[vp.offset + i]);
          }
        }
        const valLoss = this.computeLoss(valYData, valPreds);
        if (valLoss < bestValLoss - 1e-7) {
          bestValLoss = valLoss;
          noImprovementCount = 0;
        } else {
          noImprovementCount++;
        }
        if (noImprovementCount >= this.nIterNoChange) {
          break;
        }
      }
    }

    this.fitted = true;
    return this;
  }

  /**
   * Predict target values for samples in X.
   *
   * Aggregates the initial prediction and the scaled contributions of all trees.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predicted values of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("GradientBoostingRegressor must be fitted before prediction");
    }

    validatePredictInputs(X, this.nFeatures ?? 0, "GradientBoostingRegressor");

    const nSamples = X.shape[0] ?? 0;
    const predictions = new Array<number>(nSamples).fill(this.initPrediction);

    for (const tree of this.estimators) {
      const treePred = tree.predict(X);
      for (let i = 0; i < nSamples; i++) {
        predictions[i] =
          (predictions[i] ?? 0) + this.learningRate * Number(treePred.data[treePred.offset + i]);
      }
    }

    return tensor(predictions);
  }

  /**
   * Return the R² score on the given test data and target values.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True target values of shape (n_samples,)
   * @returns R² score (best possible is 1.0, can be negative)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1-dimensional or sample counts mismatch
   * @throws {DataValidationError} If y contains NaN/Inf values
   */
  score(X: Tensor, y: Tensor): number {
    if (y.ndim !== 1) {
      throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    }
    assertContiguous(y, "y");
    for (let i = 0; i < y.size; i++) {
      const val = y.data[y.offset + i] ?? 0;
      if (!Number.isFinite(val)) {
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
      const yTrue = Number(y.data[y.offset + i]);
      const yPred = Number(predictions.data[predictions.offset + i]);
      ssRes += (yTrue - yPred) ** 2;
      ssTot += (yTrue - yMean) ** 2;
    }

    return ssTot === 0 ? (ssRes === 0 ? 1.0 : 0.0) : 1 - ssRes / ssTot;
  }

  /**
   * Get feature importances averaged across all boosting stages.
   *
   * @returns Tensor of shape (n_features,) with importance values summing to 1
   * @throws {NotFittedError} If the model has not been fitted
   */
  get featureImportances(): Tensor {
    if (!this.fitted || this.estimators.length === 0 || this.nFeatures === 0) {
      throw new NotFittedError(
        "GradientBoostingRegressor must be fitted to access feature_importances_"
      );
    }
    return averageTreeImportances(this.estimators, this.nFeatures);
  }

  /**
   * Get hyperparameters for this estimator.
   *
   * @returns Object containing all hyperparameters
   */
  /**
   * Get the number of estimators actually fitted (may be < nEstimators if early stopped).
   */
  get nEstimatorsFitted(): number {
    return this.nEstimatorsFitted_;
  }

  getParams(): Record<string, unknown> {
    return {
      nEstimators: this.nEstimators,
      learningRate: this.learningRate,
      maxDepth: this.maxDepth,
      minSamplesSplit: this.minSamplesSplit,
      warmStart: this.warmStart,
      subsample: this.subsample,
      maxFeatures: this.maxFeatures,
      validationFraction: this.validationFraction,
      nIterNoChange: this.nIterNoChange,
      loss: this.loss,
      alpha: this.alpha,
    };
  }

  /**
   * Set the parameters of this estimator.
   *
   * @param params - Parameters to set
   * @throws {InvalidParameterError} If a parameter value is invalid or unknown
   */
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
        case "minSamplesSplit":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 2) {
            throw new InvalidParameterError(
              "minSamplesSplit must be an integer >= 2",
              "minSamplesSplit",
              value
            );
          }
          this.minSamplesSplit = value;
          break;
        case "warmStart":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError("warmStart must be a boolean", "warmStart", value);
          }
          this.warmStart = value;
          break;
        case "subsample":
          if (typeof value !== "number" || value <= 0 || value > 1) {
            throw new InvalidParameterError("subsample must be in (0, 1]", "subsample", value);
          }
          this.subsample = value;
          break;
        case "maxFeatures":
          if (
            value !== undefined &&
            value !== "sqrt" &&
            value !== "log2" &&
            (typeof value !== "number" || value < 1)
          ) {
            throw new InvalidParameterError(
              'maxFeatures must be "sqrt", "log2", a number >= 1, or undefined',
              "maxFeatures",
              value
            );
          }
          this.maxFeatures = value;
          break;
        case "validationFraction":
          if (typeof value !== "number" || value <= 0 || value >= 1) {
            throw new InvalidParameterError(
              "validationFraction must be in (0, 1)",
              "validationFraction",
              value
            );
          }
          this.validationFraction = value;
          break;
        case "nIterNoChange":
          if (
            value !== undefined &&
            (typeof value !== "number" || !Number.isInteger(value) || value < 1)
          ) {
            throw new InvalidParameterError(
              "nIterNoChange must be an integer >= 1 or undefined",
              "nIterNoChange",
              value
            );
          }
          this.nIterNoChange = value;
          break;
        case "loss":
          if (value !== "ls" && value !== "lad" && value !== "huber" && value !== "quantile") {
            throw new InvalidParameterError(
              `loss must be "ls", "lad", "huber", or "quantile"`,
              "loss",
              value
            );
          }
          this.loss = value;
          break;
        case "alpha":
          if (typeof value !== "number" || value <= 0 || value >= 1) {
            throw new InvalidParameterError("alpha must be in (0, 1)", "alpha", value);
          }
          this.alpha = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }

  clone(): GradientBoostingRegressor {
    return new GradientBoostingRegressor(
      this.getParams() as {
        nEstimators?: number;
        learningRate?: number;
        maxDepth?: number;
        minSamplesSplit?: number;
        warmStart?: boolean;
        subsample?: number;
        maxFeatures?: "sqrt" | "log2" | number;
        validationFraction?: number;
        nIterNoChange?: number;
        loss?: "ls" | "lad" | "huber" | "quantile";
        alpha?: number;
      }
    );
  }
}

/**
 * Gradient Boosting Classifier.
 *
 * Uses gradient boosting with shallow regression trees for classification.
 * Supports both binary and multiclass classification.
 * - Binary: optimizes log loss using sigmoid function.
 * - Multiclass: uses One-vs-Rest (OvR) strategy, training one binary model per class.
 *
 * @example
 * ```ts
 * import { GradientBoostingClassifier } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 2], [2, 3], [3, 1], [4, 2]]);
 * const y = tensor([0, 0, 1, 1]);
 *
 * const gbc = new GradientBoostingClassifier({ nEstimators: 100 });
 * gbc.fit(X, y);
 * const predictions = gbc.predict(X);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-ensemble | Deepbox Ensemble Methods}
 */
export class GradientBoostingClassifier implements Classifier {
  /** Number of boosting stages */
  private nEstimators: number;

  /** Learning rate */
  private learningRate: number;

  /** Maximum depth */
  private maxDepth: number;

  /** Minimum samples to split */
  private minSamplesSplit: number;

  /** Whether to reuse previously fitted trees and add more */
  private warmStart: boolean;

  /** Fraction of samples to use per boosting stage (stochastic GB) */
  private subsample: number;

  /** Number of features to consider for best split at each tree */
  private maxFeatures: "sqrt" | "log2" | number | undefined;

  /** Per-class arrays of weak learners (OvR for multiclass, single for binary) */
  private estimatorsPerClass: DecisionTreeRegressor[][] = [];

  /** Per-class initial log-odds predictions */
  private initPredictions: number[] = [];

  /** Number of features */
  private nFeatures = 0;

  /** Unique class labels */
  private classLabels: number[] = [];

  /** Whether fitted */
  private fitted = false;

  constructor(
    options: {
      readonly nEstimators?: number;
      readonly learningRate?: number;
      readonly maxDepth?: number;
      readonly minSamplesSplit?: number;
      readonly warmStart?: boolean;
      readonly subsample?: number;
      readonly maxFeatures?: "sqrt" | "log2" | number;
    } = {}
  ) {
    this.nEstimators = options.nEstimators ?? 100;
    this.learningRate = options.learningRate ?? 0.1;
    this.maxDepth = options.maxDepth ?? 3;
    this.minSamplesSplit = options.minSamplesSplit ?? 2;
    this.warmStart = options.warmStart ?? false;
    this.subsample = options.subsample ?? 1.0;
    if (options.maxFeatures !== undefined) {
      this.maxFeatures = options.maxFeatures;
    }

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
    if (!Number.isInteger(this.maxDepth) || this.maxDepth < 1) {
      throw new InvalidParameterError(
        "maxDepth must be an integer >= 1",
        "maxDepth",
        this.maxDepth
      );
    }
    if (!Number.isInteger(this.minSamplesSplit) || this.minSamplesSplit < 2) {
      throw new InvalidParameterError(
        "minSamplesSplit must be an integer >= 2",
        "minSamplesSplit",
        this.minSamplesSplit
      );
    }
    if (!Number.isFinite(this.subsample) || this.subsample <= 0 || this.subsample > 1) {
      throw new InvalidParameterError("subsample must be in (0, 1]", "subsample", this.subsample);
    }
  }

  /**
   * Resolve maxFeatures into an actual integer count.
   */
  private resolveMaxFeatures(nFeatures: number): number | undefined {
    if (this.maxFeatures === undefined) return undefined;
    if (typeof this.maxFeatures === "number") {
      return Math.max(1, Math.min(this.maxFeatures, nFeatures));
    }
    if (this.maxFeatures === "sqrt") {
      return Math.max(1, Math.floor(Math.sqrt(nFeatures)));
    }
    return Math.max(1, Math.floor(Math.log2(nFeatures)));
  }

  /**
   * Fit a single binary boosting ensemble.
   * Trains nEstimators regression trees to optimize log loss for a binary target.
   */
  private fitBinary(
    X: Tensor,
    yBinary: number[],
    nSamples: number,
    nFeatures: number,
    existingEstimators?: DecisionTreeRegressor[],
    existingInitPred?: number
  ): { estimators: DecisionTreeRegressor[]; initPred: number } {
    let initPred: number;
    let estimators: DecisionTreeRegressor[];
    let startIdx: number;

    if (existingEstimators && existingInitPred !== undefined) {
      initPred = existingInitPred;
      estimators = [...existingEstimators];
      startIdx = estimators.length;
    } else {
      const posCount = yBinary.filter((v) => v === 1).length;
      const negCount = nSamples - posCount;
      initPred = Math.log((posCount + 1) / (negCount + 1));
      estimators = [];
      startIdx = 0;
    }

    const rawScores = new Array<number>(nSamples).fill(initPred);
    // Replay existing trees
    for (const tree of estimators) {
      const treePred = tree.predict(X);
      for (let i = 0; i < nSamples; i++) {
        rawScores[i] =
          (rawScores[i] ?? 0) + this.learningRate * Number(treePred.data[treePred.offset + i]);
      }
    }

    // Subsample setup
    const drawSize =
      this.subsample < 1 ? Math.max(1, Math.floor(this.subsample * nSamples)) : nSamples;
    const useSubsample = drawSize < nSamples;

    // Resolve maxFeatures for trees
    const treeMaxFeatures = this.resolveMaxFeatures(nFeatures);

    for (let m = startIdx; m < this.nEstimators; m++) {
      const residuals: number[] = [];
      for (let i = 0; i < nSamples; i++) {
        const prob = 1 / (1 + Math.exp(-(rawScores[i] ?? 0)));
        residuals.push((yBinary[i] ?? 0) - prob);
      }

      // Subsample
      let fitX = X;
      let fitResiduals = residuals;
      if (useSubsample) {
        const indices: number[] = [];
        for (let i = 0; i < drawSize; i++) {
          indices.push(Math.floor(__random() * nSamples));
        }
        const subRows: number[] = [];
        const subRes: number[] = [];
        for (const idx of indices) {
          for (let j = 0; j < nFeatures; j++) {
            subRows.push(Number(X.data[X.offset + idx * nFeatures + j]));
          }
          subRes.push(residuals[idx] ?? 0);
        }
        fitX = tensor(subRows).reshape([drawSize, nFeatures]);
        fitResiduals = subRes;
      }

      const treeOpts: {
        maxDepth: number;
        minSamplesSplit: number;
        minSamplesLeaf: number;
        maxFeatures?: number;
      } = {
        maxDepth: this.maxDepth,
        minSamplesSplit: this.minSamplesSplit,
        minSamplesLeaf: 1,
      };
      if (treeMaxFeatures !== undefined) {
        treeOpts.maxFeatures = treeMaxFeatures;
      }
      const tree = new DecisionTreeRegressor(treeOpts);
      tree.fit(fitX, tensor(fitResiduals));
      estimators.push(tree);

      // TreeBoost Newton leaf update (Friedman 2001): replace each terminal
      // region's mean-residual value with Σr / Σ p(1-p), the Newton step for
      // the binomial deviance. Without it the regression tree's raw
      // mean-residual output yields systematically under-confident
      // probabilities. Samples sharing a tree prediction share a leaf, so we
      // group by predicted value and store the corrected value on the tree.
      const treePred = tree.predict(X);
      const groups = new Map<string, { residSum: number; hessSum: number }>();
      for (let i = 0; i < nSamples; i++) {
        const key = Number(treePred.data[treePred.offset + i]).toExponential(10);
        const p = 1 / (1 + Math.exp(-(rawScores[i] ?? 0)));
        const g = groups.get(key) ?? { residSum: 0, hessSum: 0 };
        g.residSum += residuals[i] ?? 0;
        g.hessSum += p * (1 - p);
        groups.set(key, g);
      }
      const leafValues = new Map<string, number>();
      for (const [key, g] of groups) {
        leafValues.set(key, g.hessSum > 1e-12 ? g.residSum / g.hessSum : 0);
      }
      // Persist corrected leaf values so predict() uses them too.
      tree.remapLeaves((v) => leafValues.get(v.toExponential(10)) ?? v);
      const corrected = tree.predict(X);
      for (let i = 0; i < nSamples; i++) {
        rawScores[i] =
          (rawScores[i] ?? 0) + this.learningRate * Number(corrected.data[corrected.offset + i]);
      }
    }

    return { estimators, initPred };
  }

  /**
   * Compute raw scores for a single binary ensemble.
   */
  private predictRawBinary(X: Tensor, classIdx: number): number[] {
    const nSamples = X.shape[0] ?? 0;
    const rawScores = new Array<number>(nSamples).fill(this.initPredictions[classIdx] ?? 0);
    const estimators = this.estimatorsPerClass[classIdx] ?? [];
    for (const tree of estimators) {
      const treePred = tree.predict(X);
      for (let i = 0; i < nSamples; i++) {
        rawScores[i] =
          (rawScores[i] ?? 0) + this.learningRate * Number(treePred.data[treePred.offset + i]);
      }
    }
    return rawScores;
  }

  /**
   * Fit the gradient boosting classifier on training data.
   *
   * Builds an additive model by sequentially fitting regression trees
   * to the pseudo-residuals (gradient of log loss).
   * Supports binary (2 classes) and multiclass (>2 classes via OvR).
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Target class labels of shape (n_samples,). Must contain at least 2 classes.
   * @returns this - The fitted estimator
   * @throws {ShapeError} If X is not 2D or y is not 1D
   * @throws {ShapeError} If X and y have different number of samples
   * @throws {DataValidationError} If X or y contain NaN/Inf values
   * @throws {InvalidParameterError} If y does not contain at least 2 classes
   */
  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;

    this.nFeatures = nFeatures;

    const yData: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      yData.push(Number(y.data[y.offset + i]));
    }

    this.classLabels = [...new Set(yData)].sort((a, b) => a - b);
    if (this.classLabels.length < 2) {
      throw new InvalidParameterError(
        "GradientBoostingClassifier requires at least 2 classes",
        "y",
        this.classLabels.length
      );
    }

    // Warm start: reuse existing estimators
    const useWarm = this.warmStart && this.fitted && this.estimatorsPerClass.length > 0;
    const oldEstPerClass = useWarm ? [...this.estimatorsPerClass] : [];
    const oldInitPreds = useWarm ? [...this.initPredictions] : [];

    this.estimatorsPerClass = [];
    this.initPredictions = [];

    if (this.classLabels.length === 2) {
      // Binary: single sigmoid model
      const yBinary = yData.map((label) => (label === this.classLabels[0] ? 0 : 1));
      const { estimators, initPred } = this.fitBinary(
        X,
        yBinary,
        nSamples,
        nFeatures,
        useWarm ? oldEstPerClass[0] : undefined,
        useWarm ? oldInitPreds[0] : undefined
      );
      this.estimatorsPerClass.push(estimators);
      this.initPredictions.push(initPred);
    } else {
      // Multiclass: One-vs-Rest — one binary model per class
      for (let c = 0; c < this.classLabels.length; c++) {
        const classLabel = this.classLabels[c];
        const yBinary = yData.map((label) => (label === classLabel ? 1 : 0));
        const { estimators, initPred } = this.fitBinary(
          X,
          yBinary,
          nSamples,
          nFeatures,
          useWarm ? oldEstPerClass[c] : undefined,
          useWarm ? oldInitPreds[c] : undefined
        );
        this.estimatorsPerClass.push(estimators);
        this.initPredictions.push(initPred);
      }
    }

    this.fitted = true;
    return this;
  }

  /**
   * Predict class labels for samples in X.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predicted class labels of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("GradientBoostingClassifier must be fitted before prediction");
    }

    validatePredictInputs(X, this.nFeatures ?? 0, "GradientBoostingClassifier");

    const nSamples = X.shape[0] ?? 0;
    const predictions: number[] = [];

    if (this.classLabels.length === 2) {
      // Binary
      const rawScores = this.predictRawBinary(X, 0);
      for (let i = 0; i < nSamples; i++) {
        const prob = 1 / (1 + Math.exp(-(rawScores[i] ?? 0)));
        predictions.push(prob >= 0.5 ? (this.classLabels[1] ?? 0) : (this.classLabels[0] ?? 0));
      }
    } else {
      // Multiclass OvR: pick class with highest raw score
      const allScores: number[][] = [];
      for (let c = 0; c < this.classLabels.length; c++) {
        allScores.push(this.predictRawBinary(X, c));
      }
      for (let i = 0; i < nSamples; i++) {
        let bestClass = 0;
        let bestScore = -Infinity;
        for (let c = 0; c < this.classLabels.length; c++) {
          const score = allScores[c]?.[i] ?? 0;
          if (score > bestScore) {
            bestScore = score;
            bestClass = c;
          }
        }
        predictions.push(this.classLabels[bestClass] ?? 0);
      }
    }

    return tensor(predictions, { dtype: "int32" });
  }

  /**
   * Predict class probabilities for samples in X.
   *
   * Returns a matrix of shape (n_samples, n_classes).
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Class probability matrix of shape (n_samples, n_classes)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predictProba(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("GradientBoostingClassifier must be fitted before prediction");
    }

    validatePredictInputs(X, this.nFeatures ?? 0, "GradientBoostingClassifier");

    const nSamples = X.shape[0] ?? 0;
    const nClasses = this.classLabels.length;
    const proba: number[][] = [];

    if (nClasses === 2) {
      // Binary
      const rawScores = this.predictRawBinary(X, 0);
      for (let i = 0; i < nSamples; i++) {
        const prob = 1 / (1 + Math.exp(-(rawScores[i] ?? 0)));
        proba.push([1 - prob, prob]);
      }
    } else {
      // Multiclass OvR: softmax over per-class sigmoid scores
      const allScores: number[][] = [];
      for (let c = 0; c < nClasses; c++) {
        allScores.push(this.predictRawBinary(X, c));
      }
      for (let i = 0; i < nSamples; i++) {
        const sigScores: number[] = [];
        for (let c = 0; c < nClasses; c++) {
          sigScores.push(1 / (1 + Math.exp(-(allScores[c]?.[i] ?? 0))));
        }
        const total = sigScores.reduce((s, v) => s + v, 0) || 1;
        proba.push(sigScores.map((v) => v / total));
      }
    }

    return tensor(proba);
  }

  /**
   * Return the mean accuracy on the given test data and labels.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True labels of shape (n_samples,)
   * @returns Accuracy score in range [0, 1]
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1-dimensional or sample counts mismatch
   * @throws {DataValidationError} If y contains NaN/Inf values
   */
  score(X: Tensor, y: Tensor): number {
    if (y.ndim !== 1) {
      throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    }
    assertContiguous(y, "y");
    for (let i = 0; i < y.size; i++) {
      const val = y.data[y.offset + i] ?? 0;
      if (!Number.isFinite(val)) {
        throw new DataValidationError("y contains non-finite values (NaN or Inf)");
      }
    }
    const predictions = this.predict(X);
    if (predictions.size !== y.size) {
      throw new ShapeError(
        `X and y must have the same number of samples; got X=${predictions.size}, y=${y.size}`
      );
    }
    let correct = 0;
    for (let i = 0; i < y.size; i++) {
      if (Number(predictions.data[predictions.offset + i]) === Number(y.data[y.offset + i])) {
        correct++;
      }
    }
    return correct / y.size;
  }

  /**
   * Get feature importances averaged across all boosting stages and classes.
   *
   * @returns Tensor of shape (n_features,) with importance values summing to 1
   * @throws {NotFittedError} If the model has not been fitted
   */
  get featureImportances(): Tensor {
    if (!this.fitted || this.estimatorsPerClass.length === 0 || this.nFeatures === 0) {
      throw new NotFittedError(
        "GradientBoostingClassifier must be fitted to access feature_importances_"
      );
    }
    // Flatten all trees across all classes
    const allTrees: DecisionTreeRegressor[] = [];
    for (const trees of this.estimatorsPerClass) {
      for (const tree of trees) {
        allTrees.push(tree);
      }
    }
    return averageTreeImportances(allTrees, this.nFeatures);
  }

  /**
   * Get hyperparameters for this estimator.
   *
   * @returns Object containing all hyperparameters
   */
  getParams(): Record<string, unknown> {
    return {
      nEstimators: this.nEstimators,
      learningRate: this.learningRate,
      maxDepth: this.maxDepth,
      minSamplesSplit: this.minSamplesSplit,
      warmStart: this.warmStart,
      subsample: this.subsample,
      maxFeatures: this.maxFeatures,
    };
  }

  /**
   * Set the parameters of this estimator.
   *
   * @param params - Parameters to set
   * @throws {InvalidParameterError} If a parameter value is invalid or unknown
   */
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
        case "minSamplesSplit":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 2) {
            throw new InvalidParameterError(
              "minSamplesSplit must be an integer >= 2",
              "minSamplesSplit",
              value
            );
          }
          this.minSamplesSplit = value;
          break;
        case "warmStart":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError("warmStart must be a boolean", "warmStart", value);
          }
          this.warmStart = value;
          break;
        case "subsample":
          if (typeof value !== "number" || value <= 0 || value > 1) {
            throw new InvalidParameterError("subsample must be in (0, 1]", "subsample", value);
          }
          this.subsample = value;
          break;
        case "maxFeatures":
          if (
            value !== undefined &&
            value !== "sqrt" &&
            value !== "log2" &&
            (typeof value !== "number" || value < 1)
          ) {
            throw new InvalidParameterError(
              'maxFeatures must be "sqrt", "log2", a number >= 1, or undefined',
              "maxFeatures",
              value
            );
          }
          this.maxFeatures = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }

  clone(): GradientBoostingClassifier {
    return new GradientBoostingClassifier(
      this.getParams() as {
        nEstimators?: number;
        learningRate?: number;
        maxDepth?: number;
        minSamplesSplit?: number;
        warmStart?: boolean;
        subsample?: number;
        maxFeatures?: "sqrt" | "log2" | number;
      }
    );
  }
}
