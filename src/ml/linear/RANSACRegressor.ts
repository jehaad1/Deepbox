/**
 * RANSAC (Random Sample Consensus) Regressor.
 *
 * Outlier-tolerant regression that repeatedly fits a base regressor on a random
 * minimal subset, keeps the model with the largest consensus set (inliers), and refits
 * on that set. Follows scikit-learn's `RANSACRegressor`.
 *
 * @module ml/linear/RANSACRegressor
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { ConvergenceError, InvalidParameterError, NotFittedError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { __randomBelow } from "../../random/random";
import { toFloat64View, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Regressor } from "../base";
import { createTreeRng } from "../tree/DecisionTree";
import { LinearRegression, r2ScoreOf } from "./LinearRegression";

/** Loss used to decide whether a sample is an inlier. */
type RansacLoss = "absolute_error" | "squared_error";

/** Options accepted by {@link RANSACRegressor}. */
interface RANSACOptions {
  readonly estimator?: Regressor;
  readonly minSamples?: number;
  readonly residualThreshold?: number;
  readonly maxTrials?: number;
  readonly stopNInliers?: number;
  readonly stopScore?: number;
  readonly stopProbability?: number;
  readonly loss?: RansacLoss;
  readonly randomState?: number;
}

const EPSILON = Number.EPSILON;

function median(values: Float64Array): number {
  const sorted = Float64Array.from(values).sort();
  const n = sorted.length;
  const mid = n >> 1;
  return n % 2 === 1
    ? (sorted[mid] as number)
    : ((sorted[mid - 1] as number) + (sorted[mid] as number)) / 2;
}

/**
 * Number of trials needed to draw at least one all-inlier subset with the
 * given probability (scikit-learn's `_dynamic_max_trials`).
 */
function dynamicMaxTrials(
  nInliers: number,
  nSamples: number,
  minSamples: number,
  probability: number
): number {
  const inlierRatio = nInliers / nSamples;
  const nom = Math.max(EPSILON, 1 - probability);
  const denom = Math.max(EPSILON, 1 - inlierRatio ** minSamples);
  if (nom === 1) return 0;
  if (denom === 1) return Infinity;
  return Math.abs(Math.ceil(Math.log(nom) / Math.log(denom)));
}

/**
 * RANSAC (Random Sample Consensus) Regressor.
 *
 * Each trial draws `minSamples` rows at random, fits a clone of the base
 * estimator, and counts the samples whose loss is at most `residualThreshold`.
 * The trial with the most inliers wins (ties go to the higher R² on the
 * inliers). The final model is the base estimator refitted on the winning
 * inliers.
 *
 * Fitting stops early once `stopNInliers` or `stopScore` is reached, or once
 * enough trials have run to have drawn an outlier-free subset with probability
 * `stopProbability`.
 *
 * @example
 * ```ts
 * import { RANSACRegressor } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1], [2], [3], [4], [5], [6]]);
 * const y = tensor([2, 4, 6, 8, 10, 90]); // last point is an outlier
 * const model = new RANSACRegressor({ minSamples: 2, randomState: 0 }).fit(X, y);
 * console.log(model.inlierMask); // Uint8Array [1, 1, 1, 1, 1, 0]
 * ```
 *
 * @category Linear Models
 * @implements {Regressor}
 */
export class RANSACRegressor implements Regressor {
  private estimator_: Regressor | undefined;
  private minSamples: number | undefined;
  private residualThreshold: number | undefined;
  private maxTrials: number;
  private stopNInliers: number;
  private stopScore: number;
  private stopProbability: number;
  private loss: RansacLoss;
  private randomState: number | undefined;

  private bestEstimator_?: Regressor;
  private inlierMask_?: Uint8Array;
  private nTrials_ = 0;
  private nFeaturesIn_ = 0;
  private fitted = false;

  /**
   * Create a new RANSAC regressor.
   *
   * @param options - Configuration options
   * @param options.estimator - Base regressor; must implement `clone()` (default: `LinearRegression`)
   * @param options.minSamples - Samples drawn per trial. An integer >= 1, or a fraction in (0, 1) of the training samples (default: n_features + 1)
   * @param options.residualThreshold - Maximum loss for a sample to count as an inlier (default: median absolute deviation of y)
   * @param options.maxTrials - Maximum number of random trials (default: 100)
   * @param options.stopNInliers - Stop once a model has this many inliers (default: Infinity)
   * @param options.stopScore - Stop once a model reaches this R² on its inliers (default: Infinity)
   * @param options.stopProbability - Probability, in [0, 1), of having drawn an outlier-free subset that ends the search early (default: 0.99)
   * @param options.loss - 'absolute_error' or 'squared_error' (default: 'absolute_error'). `residualThreshold` is in the units of this loss.
   * @param options.randomState - Seed for the random subsets; omit for a random run
   * @throws {InvalidParameterError} If any option is out of range
   */
  constructor(options: RANSACOptions = {}) {
    this.estimator_ = options.estimator;
    this.minSamples = options.minSamples;
    this.residualThreshold = options.residualThreshold;
    this.maxTrials = options.maxTrials ?? 100;
    this.stopNInliers = options.stopNInliers ?? Infinity;
    this.stopScore = options.stopScore ?? Infinity;
    this.stopProbability = options.stopProbability ?? 0.99;
    this.loss = options.loss ?? "absolute_error";
    this.randomState = options.randomState;
    this.validateParams();
  }

  private validateParams(): void {
    const ms = this.minSamples;
    if (ms !== undefined) {
      const isInt = Number.isInteger(ms) && ms >= 1;
      const isFraction = typeof ms === "number" && ms > 0 && ms < 1;
      if (!isInt && !isFraction) {
        throw new InvalidParameterError(
          `minSamples must be an integer >= 1 or a fraction in (0, 1); received ${String(ms)}`,
          "minSamples",
          ms
        );
      }
    }
    if (!Number.isInteger(this.maxTrials) || this.maxTrials < 1) {
      throw new InvalidParameterError(
        `maxTrials must be an integer >= 1; received ${String(this.maxTrials)}`,
        "maxTrials",
        this.maxTrials
      );
    }
    const rt = this.residualThreshold;
    if (rt !== undefined && (typeof rt !== "number" || !Number.isFinite(rt) || rt < 0)) {
      throw new InvalidParameterError(
        `residualThreshold must be a finite number >= 0; received ${String(rt)}`,
        "residualThreshold",
        rt
      );
    }
    if (typeof this.stopNInliers !== "number" || Number.isNaN(this.stopNInliers)) {
      throw new InvalidParameterError(
        `stopNInliers must be a number; received ${String(this.stopNInliers)}`,
        "stopNInliers",
        this.stopNInliers
      );
    }
    if (typeof this.stopScore !== "number" || Number.isNaN(this.stopScore)) {
      throw new InvalidParameterError(
        `stopScore must be a number; received ${String(this.stopScore)}`,
        "stopScore",
        this.stopScore
      );
    }
    if (
      typeof this.stopProbability !== "number" ||
      !(this.stopProbability >= 0 && this.stopProbability <= 1)
    ) {
      throw new InvalidParameterError(
        `stopProbability must be in [0, 1]; received ${String(this.stopProbability)}`,
        "stopProbability",
        this.stopProbability
      );
    }
    if (this.loss !== "absolute_error" && this.loss !== "squared_error") {
      throw new InvalidParameterError(
        `loss must be 'absolute_error' or 'squared_error'; received ${String(this.loss)}`,
        "loss",
        this.loss
      );
    }
    if (this.randomState !== undefined && !Number.isFinite(this.randomState)) {
      throw new InvalidParameterError(
        `randomState must be a finite number; received ${String(this.randomState)}`,
        "randomState",
        this.randomState
      );
    }
    const est = this.estimator_;
    if (
      est !== undefined &&
      (typeof est !== "object" ||
        est === null ||
        typeof est.fit !== "function" ||
        typeof est.predict !== "function" ||
        typeof est.clone !== "function")
    ) {
      throw new InvalidParameterError(
        "estimator must be a regressor with fit, predict and clone methods",
        "estimator",
        est
      );
    }
  }

  /** Fit a fresh copy of the base estimator and return the fitted model (`fit` may return a new object). */
  private fitEstimator(X: Tensor, y: Tensor): Regressor {
    const base = this.estimator_;
    const fresh: Regressor = base?.clone ? (base.clone() as Regressor) : new LinearRegression();
    return fresh.fit(X, y) as Regressor;
  }

  /**
   * Fit the model with RANSAC.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Target values of shape (n_samples,)
   * @returns this - The fitted estimator
   * @throws {ShapeError} If X is not 2D, y is not 1D, or their sample counts differ
   * @throws {DataValidationError} If X or y are empty or contain NaN/Inf
   * @throws {InvalidParameterError} If `minSamples` exceeds the number of samples
   * @throws {ConvergenceError} If no trial produced a single inlier
   */
  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const n = X.shape[0] ?? 0;
    const nF = X.shape[1] ?? 0;

    const xData = toFloat64View(X);
    const yData = toFloat64View(y);

    let minSamples: number;
    const ms = this.minSamples;
    if (ms === undefined) {
      minSamples = nF + 1;
    } else if (ms < 1) {
      minSamples = Math.ceil(ms * n);
    } else {
      minSamples = ms;
    }
    if (minSamples > n) {
      throw new InvalidParameterError(
        `minSamples (${minSamples}) may not be larger than the number of samples (${n})`,
        "minSamples",
        minSamples
      );
    }

    // Default threshold: median absolute deviation of y.
    let threshold = this.residualThreshold;
    if (threshold === undefined) {
      const yMedian = median(yData);
      const dev = new Float64Array(n);
      for (let i = 0; i < n; i++) dev[i] = Math.abs((yData[i] as number) - yMedian);
      threshold = median(dev);
      if (threshold === 0) {
        // Constant-majority targets: allow rounding error so exact fits still count.
        threshold = 1e-10 * (1 + Math.abs(yMedian));
      }
    }

    const X64 = tensor(xData, { dtype: "float64" }).reshape([n, nF]);
    const rng = createTreeRng(this.randomState);

    const perm = new Int32Array(n);
    for (let i = 0; i < n; i++) perm[i] = i;
    const subX = new Float64Array(minSamples * nF);
    const subY = new Float64Array(minSamples);

    let bestNInliers = 0;
    let bestScore = -Infinity;
    let bestMask: Uint8Array | undefined;
    let maxTrials = this.maxTrials;
    let trial = 0;

    while (trial < maxTrials) {
      trial++;

      // Partial Fisher-Yates: the first minSamples entries of perm are a uniform sample.
      for (let i = 0; i < minSamples; i++) {
        const j = i + __randomBelow(rng, n - i);
        const tmp = perm[i] as number;
        perm[i] = perm[j] as number;
        perm[j] = tmp;
        const row = perm[i] as number;
        for (let f = 0; f < nF; f++) subX[i * nF + f] = xData[row * nF + f] as number;
        subY[i] = yData[row] as number;
      }

      const reg = this.fitEstimator(
        tensor(Float64Array.from(subX), { dtype: "float64" }).reshape([minSamples, nF]),
        tensor(Float64Array.from(subY), { dtype: "float64" })
      );

      const pred = toFloat64View(reg.predict(X64));
      const mask = new Uint8Array(n);
      let nInliers = 0;
      for (let i = 0; i < n; i++) {
        const diff = (yData[i] as number) - (pred[i] as number);
        const r = this.loss === "absolute_error" ? Math.abs(diff) : diff * diff;
        if (r <= threshold) {
          mask[i] = 1;
          nInliers++;
        }
      }

      if (nInliers === 0 || nInliers < bestNInliers) continue;

      // Ties on the inlier count are broken by R² on the inliers.
      const inlierScore = (): number => {
        const idx: number[] = [];
        for (let i = 0; i < n; i++) if (mask[i]) idx.push(i);
        const ix = new Float64Array(idx.length * nF);
        const iy = new Float64Array(idx.length);
        for (let k = 0; k < idx.length; k++) {
          const row = idx[k] as number;
          for (let f = 0; f < nF; f++) ix[k * nF + f] = xData[row * nF + f] as number;
          iy[k] = yData[row] as number;
        }
        return reg.score(
          tensor(ix, { dtype: "float64" }).reshape([idx.length, nF]),
          tensor(iy, { dtype: "float64" })
        );
      };
      const score = inlierScore();
      if (nInliers === bestNInliers && score < bestScore) continue;

      bestNInliers = nInliers;
      bestScore = score;
      bestMask = mask;

      maxTrials = Math.min(
        maxTrials,
        dynamicMaxTrials(bestNInliers, n, minSamples, this.stopProbability)
      );
      if (bestNInliers >= this.stopNInliers || bestScore >= this.stopScore) break;
    }

    if (bestMask === undefined || bestNInliers === 0) {
      throw new ConvergenceError(
        "RANSAC could not find a valid consensus set; increase maxTrials or residualThreshold, or lower minSamples",
        { iterations: trial }
      );
    }

    // Refit on the consensus set.
    const inlierX = new Float64Array(bestNInliers * nF);
    const inlierY = new Float64Array(bestNInliers);
    let k = 0;
    for (let i = 0; i < n; i++) {
      if (bestMask[i]) {
        for (let f = 0; f < nF; f++) inlierX[k * nF + f] = xData[i * nF + f] as number;
        inlierY[k] = yData[i] as number;
        k++;
      }
    }
    const finalReg = this.fitEstimator(
      tensor(inlierX, { dtype: "float64" }).reshape([bestNInliers, nF]),
      tensor(inlierY, { dtype: "float64" })
    );

    this.nFeaturesIn_ = nF;
    this.bestEstimator_ = finalReg;
    this.inlierMask_ = bestMask;
    this.nTrials_ = trial;
    this.fitted = true;
    return this;
  }

  /**
   * Predict with the base estimator refitted on the inliers.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predictions of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has the wrong shape or feature count
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.bestEstimator_) {
      throw new NotFittedError("RANSACRegressor must be fitted before predict");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "RANSACRegressor");
    return this.bestEstimator_.predict(X);
  }

  /**
   * Coefficient of determination R² of the prediction.
   *
   * A constant y gives 1 when the predictions are exact and 0 otherwise.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True values of shape (n_samples,)
   * @returns R² score (can be negative)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1D or its length differs from the number of samples
   * @throws {DataValidationError} If y is empty or contains NaN/Inf
   */
  score(X: Tensor, y: Tensor): number {
    if (!this.fitted) {
      throw new NotFittedError("RANSACRegressor must be fitted before scoring");
    }
    return r2ScoreOf(y, () => this.predict(X));
  }

  /**
   * Mask over the training samples: 1 for inliers of the best model, 0 for outliers.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get inlierMask(): Uint8Array {
    if (!this.fitted || !this.inlierMask_) {
      throw new NotFittedError("RANSACRegressor must be fitted to access inlierMask");
    }
    return this.inlierMask_;
  }

  /**
   * The base estimator refitted on the inliers.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get estimator(): Regressor {
    if (!this.fitted || !this.bestEstimator_) {
      throw new NotFittedError("RANSACRegressor must be fitted to access estimator");
    }
    return this.bestEstimator_;
  }

  /**
   * Number of random trials run by the last fit.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get nTrials(): number {
    if (!this.fitted) {
      throw new NotFittedError("RANSACRegressor must be fitted to access nTrials");
    }
    return this.nTrials_;
  }

  /**
   * Get the hyperparameters of this estimator.
   */
  getParams(): Record<string, unknown> {
    return {
      estimator: this.estimator_,
      minSamples: this.minSamples,
      residualThreshold: this.residualThreshold,
      maxTrials: this.maxTrials,
      stopNInliers: this.stopNInliers,
      stopScore: this.stopScore,
      stopProbability: this.stopProbability,
      loss: this.loss,
      randomState: this.randomState,
    };
  }

  /**
   * Set hyperparameters. The fitted model is unchanged until `fit` is called again.
   * Pass `undefined` for `minSamples`, `residualThreshold`, `estimator` or
   * `randomState` to restore the default.
   *
   * @param params - Parameters to set
   * @returns this
   * @throws {InvalidParameterError} If a name is unknown or a value is invalid
   */
  setParams(params: Record<string, unknown>): this {
    const previous = {
      estimator_: this.estimator_,
      minSamples: this.minSamples,
      residualThreshold: this.residualThreshold,
      maxTrials: this.maxTrials,
      stopNInliers: this.stopNInliers,
      stopScore: this.stopScore,
      stopProbability: this.stopProbability,
      loss: this.loss,
      randomState: this.randomState,
    };
    try {
      for (const [key, value] of Object.entries(params)) {
        switch (key) {
          case "estimator":
            this.estimator_ = value as Regressor | undefined;
            break;
          case "minSamples":
            this.minSamples = value as number | undefined;
            break;
          case "residualThreshold":
            this.residualThreshold = value as number | undefined;
            break;
          case "maxTrials":
            this.maxTrials = value as number;
            break;
          case "stopNInliers":
            this.stopNInliers = value as number;
            break;
          case "stopScore":
            this.stopScore = value as number;
            break;
          case "stopProbability":
            this.stopProbability = value as number;
            break;
          case "loss":
            this.loss = value as RansacLoss;
            break;
          case "randomState":
            this.randomState = value as number | undefined;
            break;
          default:
            throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
        }
      }
      this.validateParams();
    } catch (e) {
      this.estimator_ = previous.estimator_;
      this.minSamples = previous.minSamples;
      this.residualThreshold = previous.residualThreshold;
      this.maxTrials = previous.maxTrials;
      this.stopNInliers = previous.stopNInliers;
      this.stopScore = previous.stopScore;
      this.stopProbability = previous.stopProbability;
      this.loss = previous.loss;
      this.randomState = previous.randomState;
      throw e;
    }
    return this;
  }

  /**
   * Create an unfitted copy of this estimator with the same parameters.
   * A custom base estimator is cloned as well.
   */
  clone(): RANSACRegressor {
    const options: {
      -readonly [K in keyof RANSACOptions]: RANSACOptions[K];
    } = {
      maxTrials: this.maxTrials,
      stopNInliers: this.stopNInliers,
      stopScore: this.stopScore,
      stopProbability: this.stopProbability,
      loss: this.loss,
    };
    if (this.estimator_?.clone) options.estimator = this.estimator_.clone() as Regressor;
    if (this.minSamples !== undefined) options.minSamples = this.minSamples;
    if (this.residualThreshold !== undefined) options.residualThreshold = this.residualThreshold;
    if (this.randomState !== undefined) options.randomState = this.randomState;
    return new RANSACRegressor(options);
  }
}
