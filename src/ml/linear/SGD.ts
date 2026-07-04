/**
 * Stochastic Gradient Descent (SGD) linear models.
 *
 * SGDClassifier and SGDRegressor implement regularized linear models
 * with SGD optimization. Processes one sample at a time, making them
 * suitable for large-scale datasets.
 *
 * @module ml/linear/SGD
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { DataValidationError, InvalidParameterError, NotFittedError, ShapeError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { __random } from "../../random/random";
import { assertContiguous, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Classifier, Regressor } from "../base";

type Loss = "hinge" | "log_loss" | "modified_huber" | "squared_hinge";
type RegressionLoss = "squared_error" | "huber" | "epsilon_insensitive";
type Penalty = "l1" | "l2" | "elasticnet" | "none";
type LearningRateSchedule = "constant" | "optimal" | "invscaling" | "adaptive";

interface SGDBaseOptions {
  readonly penalty?: Penalty;
  readonly alpha?: number;
  readonly l1Ratio?: number;
  readonly fitIntercept?: boolean;
  readonly maxIter?: number;
  readonly tol?: number;
  readonly shuffle?: boolean;
  readonly randomState?: number;
  readonly learningRate?: LearningRateSchedule;
  readonly eta0?: number;
  readonly powerT?: number;
}

function createRng(seed?: number): () => number {
  if (seed === undefined) return __random;
  let s = seed;
  return () => {
    s = (s * 1103515245 + 12345) % 2147483648;
    return s / 2147483648;
  };
}

function shuffleIndices(indices: number[], rng: () => number): void {
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = indices[i]!;
    indices[i] = indices[j]!;
    indices[j] = tmp;
  }
}

function applyPenaltyGradient(
  w: Float64Array,
  j: number,
  alpha: number,
  penalty: Penalty,
  l1Ratio: number
): number {
  const wj = w[j] ?? 0;
  switch (penalty) {
    case "l2":
      return alpha * wj;
    case "l1":
      return alpha * Math.sign(wj);
    case "elasticnet":
      return alpha * (l1Ratio * Math.sign(wj) + (1 - l1Ratio) * wj);
    default:
      return 0;
  }
}

/**
 * SGD Classifier — Linear classifiers with SGD training.
 *
 * Supports multiple loss functions:
 * - `"hinge"` — Linear SVM (default)
 * - `"log_loss"` — Logistic regression
 * - `"modified_huber"` — Smoothed hinge loss with probability estimates
 * - `"squared_hinge"` — Squared hinge loss
 *
 * @example
 * ```ts
 * import { SGDClassifier } from 'deepbox/ml';
 *
 * const clf = new SGDClassifier({ loss: 'log_loss', alpha: 0.0001 });
 * clf.fit(X_train, y_train);
 * const preds = clf.predict(X_test);
 * ```
 */
export class SGDClassifier implements Classifier {
  private readonly loss: Loss;
  private readonly penalty: Penalty;
  private readonly alpha: number;
  private readonly l1Ratio: number;
  private readonly fitIntercept: boolean;
  private readonly maxIter: number;
  private readonly tol: number;
  private readonly shuffle_: boolean;
  private readonly randomState?: number;
  private readonly learningRate: LearningRateSchedule;
  private readonly eta0: number;
  private readonly powerT: number;

  private coef_?: Float64Array;
  private intercept_ = 0;
  private classes_?: Tensor;
  private nFeaturesIn_ = 0;
  private fitted = false;
  private multiclass_ = false;
  private allCoefs_?: Float64Array[]; // one per OvR binary classifier
  private allIntercepts_?: number[];

  constructor(
    options: SGDBaseOptions & {
      readonly loss?: Loss;
    } = {}
  ) {
    this.loss = options.loss ?? "hinge";
    this.penalty = options.penalty ?? "l2";
    this.alpha = options.alpha ?? 0.0001;
    this.l1Ratio = options.l1Ratio ?? 0.15;
    this.fitIntercept = options.fitIntercept ?? true;
    this.maxIter = options.maxIter ?? 1000;
    this.tol = options.tol ?? 1e-3;
    this.shuffle_ = options.shuffle ?? true;
    if (options.randomState !== undefined) this.randomState = options.randomState;
    this.learningRate = options.learningRate ?? "optimal";
    this.eta0 = options.eta0 ?? 0.01;
    this.powerT = options.powerT ?? 0.5;

    if (this.alpha < 0) {
      throw new InvalidParameterError("alpha must be >= 0", "alpha", this.alpha);
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

    // Extract y data and find unique classes
    const yData = new Float64Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      yData[i] = Number(y.data[y.offset + i]);
    }
    const uniqueClasses = [...new Set(yData)].sort((a, b) => a - b);
    this.classes_ = tensor(uniqueClasses);

    if (uniqueClasses.length <= 2) {
      this.multiclass_ = false;
      // Map to +1/-1 for hinge losses, 0/1 for log_loss
      const positive = uniqueClasses[1] ?? uniqueClasses[0] ?? 0;
      const yBinary = new Float64Array(nSamples);
      for (let i = 0; i < nSamples; i++) {
        if (this.loss === "log_loss") {
          yBinary[i] = (yData[i] ?? 0) === positive ? 1 : 0;
        } else {
          yBinary[i] = (yData[i] ?? 0) === positive ? 1 : -1;
        }
      }

      const { w, b } = this.fitBinary(X, yBinary, nSamples, nFeatures);
      this.coef_ = w;
      this.intercept_ = b;
    } else {
      // One-vs-Rest multiclass
      this.multiclass_ = true;
      this.allCoefs_ = [];
      this.allIntercepts_ = [];

      for (const cls of uniqueClasses) {
        const yBinary = new Float64Array(nSamples);
        for (let i = 0; i < nSamples; i++) {
          if (this.loss === "log_loss") {
            yBinary[i] = (yData[i] ?? 0) === cls ? 1 : 0;
          } else {
            yBinary[i] = (yData[i] ?? 0) === cls ? 1 : -1;
          }
        }
        const { w, b } = this.fitBinary(X, yBinary, nSamples, nFeatures);
        this.allCoefs_.push(w);
        this.allIntercepts_.push(b);
      }
    }

    this.fitted = true;
    return this;
  }

  private fitBinary(
    X: Tensor,
    yBinary: Float64Array,
    nSamples: number,
    nFeatures: number
  ): { w: Float64Array; b: number } {
    const w = new Float64Array(nFeatures);
    let b = 0;
    const rng = createRng(this.randomState);
    const indices = Array.from({ length: nSamples }, (_, i) => i);
    let t = 0;

    for (let epoch = 0; epoch < this.maxIter; epoch++) {
      if (this.shuffle_) shuffleIndices(indices, rng);

      let totalLoss = 0;

      for (const idx of indices) {
        t++;
        const eta = this.getLR(t);
        const rowBase = X.offset + idx * nFeatures;
        const yi = yBinary[idx] ?? 0;

        // Compute decision value: w · x + b
        let decision = this.fitIntercept ? b : 0;
        for (let j = 0; j < nFeatures; j++) {
          decision += (w[j] ?? 0) * Number(X.data[rowBase + j] ?? 0);
        }

        // Compute loss gradient
        const dloss = this.lossGradient(decision, yi);
        totalLoss += Math.abs(dloss);

        // Update weights
        for (let j = 0; j < nFeatures; j++) {
          const xj = Number(X.data[rowBase + j] ?? 0);
          const reg = applyPenaltyGradient(w, j, this.alpha, this.penalty, this.l1Ratio);
          w[j] = (w[j] ?? 0) - eta * (dloss * xj + reg);
        }
        if (this.fitIntercept) {
          b -= eta * dloss;
        }
      }

      // Check convergence
      if (nSamples > 0 && totalLoss / nSamples < this.tol) break;
    }

    return { w, b };
  }

  private lossGradient(decision: number, y: number): number {
    switch (this.loss) {
      case "hinge":
        return y * decision < 1 ? -y : 0;
      case "squared_hinge": {
        const margin = y * decision;
        return margin < 1 ? -2 * y * (1 - margin) : 0;
      }
      case "log_loss": {
        // y in {0, 1}, use logistic loss
        const p = 1 / (1 + Math.exp(-decision));
        return p - y;
      }
      case "modified_huber": {
        const margin = y * decision;
        if (margin >= 1) return 0;
        if (margin >= -1) return -2 * y * (1 - margin);
        return -4 * y;
      }
      default:
        return 0;
    }
  }

  private getLR(t: number): number {
    switch (this.learningRate) {
      case "constant":
        return this.eta0;
      case "optimal":
        return 1.0 / (this.alpha * t);
      case "invscaling":
        return this.eta0 / t ** this.powerT;
      case "adaptive":
        return this.eta0;
      default:
        return this.eta0;
    }
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted) throw new NotFittedError("SGDClassifier must be fitted before predict");
    validatePredictInputs(X, this.nFeaturesIn_, "SGDClassifier");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const classes = this.classes_!;
    const result = new Float64Array(nSamples);

    if (this.multiclass_) {
      const nClasses = classes.size;
      for (let i = 0; i < nSamples; i++) {
        let bestScore = -Infinity;
        let bestClass = 0;
        for (let k = 0; k < nClasses; k++) {
          let score = this.allIntercepts_![k] ?? 0;
          const w = this.allCoefs_![k]!;
          const rowBase = X.offset + i * nFeatures;
          for (let j = 0; j < nFeatures; j++) {
            score += (w[j] ?? 0) * Number(X.data[rowBase + j] ?? 0);
          }
          if (score > bestScore) {
            bestScore = score;
            bestClass = k;
          }
        }
        result[i] = Number(classes.data[classes.offset + bestClass]);
      }
    } else {
      const w = this.coef_!;
      for (let i = 0; i < nSamples; i++) {
        let decision = this.fitIntercept ? this.intercept_ : 0;
        const rowBase = X.offset + i * nFeatures;
        for (let j = 0; j < nFeatures; j++) {
          decision += (w[j] ?? 0) * Number(X.data[rowBase + j] ?? 0);
        }
        if (this.loss === "log_loss") {
          const p = 1 / (1 + Math.exp(-decision));
          result[i] =
            p >= 0.5
              ? Number(classes.data[classes.offset + (classes.size > 1 ? 1 : 0)])
              : Number(classes.data[classes.offset]);
        } else {
          result[i] =
            decision >= 0
              ? Number(classes.data[classes.offset + (classes.size > 1 ? 1 : 0)])
              : Number(classes.data[classes.offset]);
        }
      }
    }

    return tensor(Array.from(result));
  }

  predictProba(X: Tensor): Tensor {
    if (!this.fitted) throw new NotFittedError("SGDClassifier must be fitted before predictProba");
    if (this.loss !== "log_loss" && this.loss !== "modified_huber") {
      throw new DataValidationError(
        "predictProba is only available for loss='log_loss' or 'modified_huber'"
      );
    }
    validatePredictInputs(X, this.nFeaturesIn_, "SGDClassifier");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const classes = this.classes_!;

    if (this.multiclass_) {
      const nClasses = classes.size;
      const proba = new Float64Array(nSamples * nClasses);

      for (let i = 0; i < nSamples; i++) {
        let sumP = 0;
        for (let k = 0; k < nClasses; k++) {
          let score = this.allIntercepts_![k] ?? 0;
          const w = this.allCoefs_![k]!;
          const rowBase = X.offset + i * nFeatures;
          for (let j = 0; j < nFeatures; j++) {
            score += (w[j] ?? 0) * Number(X.data[rowBase + j] ?? 0);
          }
          const p = 1 / (1 + Math.exp(-score));
          proba[i * nClasses + k] = p;
          sumP += p;
        }
        // Normalize
        if (sumP > 0) {
          for (let k = 0; k < nClasses; k++) {
            proba[i * nClasses + k] = (proba[i * nClasses + k] ?? 0) / sumP;
          }
        }
      }
      return tensor(Array.from(proba)).reshape([nSamples, nClasses]);
    } else {
      const nClasses = classes.size > 1 ? 2 : 1;
      const proba = new Float64Array(nSamples * nClasses);
      const w = this.coef_!;

      for (let i = 0; i < nSamples; i++) {
        let decision = this.fitIntercept ? this.intercept_ : 0;
        const rowBase = X.offset + i * nFeatures;
        for (let j = 0; j < nFeatures; j++) {
          decision += (w[j] ?? 0) * Number(X.data[rowBase + j] ?? 0);
        }
        const p1 = 1 / (1 + Math.exp(-decision));
        if (nClasses === 2) {
          proba[i * 2] = 1 - p1;
          proba[i * 2 + 1] = p1;
        } else {
          proba[i] = p1;
        }
      }
      return tensor(Array.from(proba)).reshape([nSamples, nClasses]);
    }
  }

  score(X: Tensor, y: Tensor): number {
    if (y.ndim !== 1) throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    assertContiguous(y, "y");
    const pred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.size; i++) {
      if (Number(pred.data[pred.offset + i]) === Number(y.data[y.offset + i])) correct++;
    }
    return correct / y.size;
  }

  get classes(): Tensor | undefined {
    return this.classes_;
  }

  get coef(): Float64Array {
    if (!this.fitted) throw new NotFittedError("SGDClassifier must be fitted to access coef");
    return this.coef_!;
  }

  get intercept(): number {
    if (!this.fitted) throw new NotFittedError("SGDClassifier must be fitted to access intercept");
    return this.intercept_;
  }

  getParams(): Record<string, unknown> {
    return {
      loss: this.loss,
      penalty: this.penalty,
      alpha: this.alpha,
      l1Ratio: this.l1Ratio,
      fitIntercept: this.fitIntercept,
      maxIter: this.maxIter,
      tol: this.tol,
      shuffle: this.shuffle_,
      randomState: this.randomState,
      learningRate: this.learningRate,
      eta0: this.eta0,
      powerT: this.powerT,
    };
  }

  setParams(_params: Record<string, unknown>): this {
    return this;
  }
}

/**
 * SGD Regressor — Linear regressor with SGD training.
 *
 * Supports multiple loss functions:
 * - `"squared_error"` — Ordinary least squares (default)
 * - `"huber"` — Huber loss (robust to outliers)
 * - `"epsilon_insensitive"` — Support vector regression loss
 *
 * @example
 * ```ts
 * import { SGDRegressor } from 'deepbox/ml';
 *
 * const reg = new SGDRegressor({ loss: 'huber', alpha: 0.0001 });
 * reg.fit(X_train, y_train);
 * const preds = reg.predict(X_test);
 * ```
 */
export class SGDRegressor implements Regressor {
  private readonly loss: RegressionLoss;
  private readonly penalty: Penalty;
  private readonly alpha: number;
  private readonly l1Ratio: number;
  private readonly fitIntercept: boolean;
  private readonly maxIter: number;
  private readonly tol: number;
  private readonly shuffle_: boolean;
  private readonly randomState?: number;
  private readonly learningRate: LearningRateSchedule;
  private readonly eta0: number;
  private readonly powerT: number;
  private readonly epsilon: number;

  private coef_?: Float64Array;
  private intercept_ = 0;
  private nFeaturesIn_ = 0;
  private fitted = false;

  constructor(
    options: SGDBaseOptions & {
      readonly loss?: RegressionLoss;
      readonly epsilon?: number;
    } = {}
  ) {
    this.loss = options.loss ?? "squared_error";
    this.penalty = options.penalty ?? "l2";
    this.alpha = options.alpha ?? 0.0001;
    this.l1Ratio = options.l1Ratio ?? 0.15;
    this.fitIntercept = options.fitIntercept ?? true;
    this.maxIter = options.maxIter ?? 1000;
    this.tol = options.tol ?? 1e-3;
    this.shuffle_ = options.shuffle ?? true;
    if (options.randomState !== undefined) this.randomState = options.randomState;
    this.learningRate = options.learningRate ?? "invscaling";
    this.eta0 = options.eta0 ?? 0.01;
    this.powerT = options.powerT ?? 0.25;
    this.epsilon = options.epsilon ?? 0.1;

    if (this.alpha < 0) {
      throw new InvalidParameterError("alpha must be >= 0", "alpha", this.alpha);
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

    const yData = new Float64Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      yData[i] = Number(y.data[y.offset + i]);
    }

    const w = new Float64Array(nFeatures);
    let b = 0;
    const rng = createRng(this.randomState);
    const indices = Array.from({ length: nSamples }, (_, i) => i);
    let t = 0;

    for (let epoch = 0; epoch < this.maxIter; epoch++) {
      if (this.shuffle_) shuffleIndices(indices, rng);

      let totalLoss = 0;

      for (const idx of indices) {
        t++;
        const eta = this.getLR(t);
        const rowBase = X.offset + idx * nFeatures;
        const yi = yData[idx] ?? 0;

        // Compute prediction
        let pred = this.fitIntercept ? b : 0;
        for (let j = 0; j < nFeatures; j++) {
          pred += (w[j] ?? 0) * Number(X.data[rowBase + j] ?? 0);
        }

        // Compute loss gradient
        const dloss = this.lossGradient(pred, yi);
        totalLoss += Math.abs(dloss);

        // Update weights
        for (let j = 0; j < nFeatures; j++) {
          const xj = Number(X.data[rowBase + j] ?? 0);
          const reg = applyPenaltyGradient(w, j, this.alpha, this.penalty, this.l1Ratio);
          w[j] = (w[j] ?? 0) - eta * (dloss * xj + reg);
        }
        if (this.fitIntercept) {
          b -= eta * dloss;
        }
      }

      if (nSamples > 0 && totalLoss / nSamples < this.tol) break;
    }

    this.coef_ = w;
    this.intercept_ = b;
    this.fitted = true;
    return this;
  }

  private lossGradient(pred: number, y: number): number {
    const residual = pred - y;
    switch (this.loss) {
      case "squared_error":
        return residual;
      case "huber":
        if (Math.abs(residual) <= this.epsilon) return residual;
        return this.epsilon * Math.sign(residual);
      case "epsilon_insensitive":
        if (Math.abs(residual) <= this.epsilon) return 0;
        return Math.sign(residual);
      default:
        return residual;
    }
  }

  private getLR(t: number): number {
    switch (this.learningRate) {
      case "constant":
        return this.eta0;
      case "optimal":
        return 1.0 / (this.alpha * t);
      case "invscaling":
        return this.eta0 / t ** this.powerT;
      case "adaptive":
        return this.eta0;
      default:
        return this.eta0;
    }
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted) throw new NotFittedError("SGDRegressor must be fitted before predict");
    validatePredictInputs(X, this.nFeaturesIn_, "SGDRegressor");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const w = this.coef_!;
    const result = new Float64Array(nSamples);

    for (let i = 0; i < nSamples; i++) {
      let pred = this.fitIntercept ? this.intercept_ : 0;
      const rowBase = X.offset + i * nFeatures;
      for (let j = 0; j < nFeatures; j++) {
        pred += (w[j] ?? 0) * Number(X.data[rowBase + j] ?? 0);
      }
      result[i] = pred;
    }

    return tensor(Array.from(result));
  }

  score(X: Tensor, y: Tensor): number {
    if (y.ndim !== 1) throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    assertContiguous(y, "y");
    const pred = this.predict(X);
    const nSamples = y.size;

    // R² score
    let yMean = 0;
    for (let i = 0; i < nSamples; i++) yMean += Number(y.data[y.offset + i]);
    yMean /= nSamples;

    let ssRes = 0;
    let ssTot = 0;
    for (let i = 0; i < nSamples; i++) {
      const yi = Number(y.data[y.offset + i]);
      const pi = Number(pred.data[pred.offset + i]);
      ssRes += (yi - pi) ** 2;
      ssTot += (yi - yMean) ** 2;
    }
    return ssTot === 0 ? 0 : 1 - ssRes / ssTot;
  }

  get coef(): Float64Array {
    if (!this.fitted) throw new NotFittedError("SGDRegressor must be fitted to access coef");
    return this.coef_!;
  }

  get intercept(): number {
    if (!this.fitted) throw new NotFittedError("SGDRegressor must be fitted to access intercept");
    return this.intercept_;
  }

  getParams(): Record<string, unknown> {
    return {
      loss: this.loss,
      penalty: this.penalty,
      alpha: this.alpha,
      l1Ratio: this.l1Ratio,
      fitIntercept: this.fitIntercept,
      maxIter: this.maxIter,
      tol: this.tol,
      shuffle: this.shuffle_,
      randomState: this.randomState,
      learningRate: this.learningRate,
      eta0: this.eta0,
      powerT: this.powerT,
      epsilon: this.epsilon,
    };
  }

  setParams(_params: Record<string, unknown>): this {
    return this;
  }
}
