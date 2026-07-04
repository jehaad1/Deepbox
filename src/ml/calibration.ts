/**
 * Probability calibration for classifiers.
 *
 * Provides `CalibratedClassifierCV` which wraps a classifier and calibrates
 * its probability estimates using Platt scaling (sigmoid) or isotonic regression.
 * Also provides `calibrationCurve` for evaluating calibration quality.
 *
 * @module ml/calibration
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError } from "../core";
import { type Tensor, tensor } from "../ndarray";
import { assertContiguous, validateFitInputs, validatePredictInputs } from "./_validation";
import type { Classifier } from "./base";

/**
 * Calibrated Classifier with Cross-Validation.
 *
 * Wraps a base classifier and calibrates its probability predictions
 * using either Platt scaling (sigmoid) or isotonic regression.
 *
 * @example
 * ```ts
 * import { CalibratedClassifierCV, LogisticRegression } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const base = new LogisticRegression();
 * const cal = new CalibratedClassifierCV({ estimator: base, method: 'sigmoid' });
 * cal.fit(X_train, y_train);
 * const proba = cal.predictProba(X_test);
 * ```
 */
export class CalibratedClassifierCV implements Classifier {
  private readonly estimator: Classifier;
  private method: "sigmoid" | "isotonic";
  private cv: number;

  // Calibration parameters (per class for multiclass)
  private calibA_?: Float64Array; // sigmoid: a parameter per class
  private calibB_?: Float64Array; // sigmoid: b parameter per class
  private isotonicX_?: number[][]; // isotonic: sorted scores per class
  private isotonicY_?: number[][]; // isotonic: calibrated values per class
  private classes_?: number[];
  private nFeaturesIn_ = 0;
  private fitted = false;

  constructor(options: {
    readonly estimator: Classifier;
    readonly method?: "sigmoid" | "isotonic";
    readonly cv?: number;
  }) {
    this.estimator = options.estimator;
    this.method = options.method ?? "sigmoid";
    this.cv = options.cv ?? 5;

    if (!Number.isInteger(this.cv) || this.cv < 2) {
      throw new InvalidParameterError("cv must be >= 2", "cv", this.cv);
    }
  }

  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    this.nFeaturesIn_ = nFeatures;

    // Extract classes
    const classSet = new Set<number>();
    for (let i = 0; i < nSamples; i++) {
      classSet.add(Number(y.data[y.offset + i]));
    }
    this.classes_ = [...classSet].sort((a, b) => a - b);
    const nClasses = this.classes_.length;

    // Compute out-of-fold (cross-validated) probability predictions for
    // calibration. Using held-out predictions — rather than predictions on the
    // training data the base estimator was fitted on — avoids the optimistic
    // bias that makes the calibrator overfit (matching scikit-learn's
    // CalibratedClassifierCV with cv="prefit"=False).
    const oofProba = this.crossValPredictProba(X, y, nSamples, nFeatures, nClasses);

    // Fit the final base estimator on all data; it is what predict/predictProba
    // applies the learned calibration map to.
    this.estimator.fit(X, y);

    if (this.method === "sigmoid") {
      // Platt scaling: fit sigmoid a*f(x) + b for each class
      this.calibA_ = new Float64Array(nClasses);
      this.calibB_ = new Float64Array(nClasses);

      for (let c = 0; c < nClasses; c++) {
        const classVal = this.classes_[c]!;
        const scores: number[] = [];
        const targets: number[] = [];

        for (let i = 0; i < nSamples; i++) {
          scores.push(oofProba[i * nClasses + c] ?? 0);
          targets.push(Number(y.data[y.offset + i]) === classVal ? 1 : 0);
        }

        // Fit sigmoid using gradient descent on log-loss
        const [a, b] = this.fitSigmoid(scores, targets);
        this.calibA_[c] = a;
        this.calibB_[c] = b;
      }
    } else {
      // Isotonic regression
      this.isotonicX_ = [];
      this.isotonicY_ = [];

      for (let c = 0; c < nClasses; c++) {
        const classVal = this.classes_[c]!;
        const pairs: [number, number][] = [];

        for (let i = 0; i < nSamples; i++) {
          const p = oofProba[i * nClasses + c] ?? 0;
          pairs.push([p, Number(y.data[y.offset + i]) === classVal ? 1 : 0]);
        }

        // Sort by score
        pairs.sort((a, b) => a[0] - b[0]);

        // Pool Adjacent Violators (isotonic regression)
        const xs = pairs.map((p) => p[0]);
        const ys = pairs.map((p) => p[1]);
        const isoY = this.isotonicRegression(ys);

        this.isotonicX_.push(xs);
        this.isotonicY_.push(isoY);
      }
    }

    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    const proba = this.predictProba(X);
    const nSamples = X.shape[0] ?? 0;
    const nClasses = this.classes_!.length;
    const labels = new Float64Array(nSamples);

    for (let i = 0; i < nSamples; i++) {
      let bestC = 0;
      let bestP = -Infinity;
      for (let c = 0; c < nClasses; c++) {
        const p = Number(proba.data[proba.offset + i * nClasses + c]);
        if (p > bestP) {
          bestP = p;
          bestC = c;
        }
      }
      labels[i] = this.classes_![bestC]!;
    }

    return tensor(Array.from(labels));
  }

  predictProba(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("CalibratedClassifierCV must be fitted before predictProba");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "CalibratedClassifierCV");
    const nSamples = X.shape[0] ?? 0;
    const nClasses = this.classes_!.length;

    // Get base probabilities
    const baseProba = this.estimator.predictProba(X);
    const result = new Float64Array(nSamples * nClasses);

    if (this.method === "sigmoid") {
      for (let i = 0; i < nSamples; i++) {
        let sum = 0;
        for (let c = 0; c < nClasses; c++) {
          const p = Number(baseProba.data[baseProba.offset + i * nClasses + c]);
          const a = this.calibA_![c] ?? 1;
          const b = this.calibB_![c] ?? 0;
          const calibrated = 1 / (1 + Math.exp(-(a * p + b)));
          result[i * nClasses + c] = calibrated;
          sum += calibrated;
        }
        // Normalize
        if (sum > 0) {
          for (let c = 0; c < nClasses; c++) {
            result[i * nClasses + c] = (result[i * nClasses + c] ?? 0) / sum;
          }
        }
      }
    } else {
      for (let i = 0; i < nSamples; i++) {
        let sum = 0;
        for (let c = 0; c < nClasses; c++) {
          const p = Number(baseProba.data[baseProba.offset + i * nClasses + c]);
          const calibrated = this.isotonicInterp(this.isotonicX_![c]!, this.isotonicY_![c]!, p);
          result[i * nClasses + c] = calibrated;
          sum += calibrated;
        }
        if (sum > 0) {
          for (let c = 0; c < nClasses; c++) {
            result[i * nClasses + c] = (result[i * nClasses + c] ?? 0) / sum;
          }
        }
      }
    }

    return tensor(Array.from(result)).reshape([nSamples, nClasses]);
  }

  score(X: Tensor, y: Tensor): number {
    assertContiguous(y, "y");
    const pred = this.predict(X);
    const nSamples = y.size;
    let correct = 0;
    for (let i = 0; i < nSamples; i++) {
      if (Number(pred.data[pred.offset + i]) === Number(y.data[y.offset + i])) {
        correct++;
      }
    }
    return correct / nSamples;
  }

  getParams(): Record<string, unknown> {
    return {
      estimator: this.estimator,
      method: this.method,
      cv: this.cv,
    };
  }

  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "method":
          if (value !== "sigmoid" && value !== "isotonic") {
            throw new InvalidParameterError(
              `method must be "sigmoid" or "isotonic"; got ${String(value)}`,
              "method",
              value
            );
          }
          this.method = value;
          break;
        case "cv":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 2) {
            throw new InvalidParameterError(
              `cv must be an integer >= 2; got ${String(value)}`,
              "cv",
              value
            );
          }
          this.cv = value;
          break;
        case "estimator":
          // The wrapped estimator is fixed at construction time.
          throw new InvalidParameterError(
            "estimator cannot be changed via setParams; construct a new CalibratedClassifierCV",
            "estimator",
            value
          );
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }

  /**
   * Clone the wrapped base estimator into a fresh, unfitted instance so each
   * cross-validation fold trains independently of the others.
   */
  private cloneEstimator(): Classifier {
    if (typeof this.estimator.clone === "function") {
      return this.estimator.clone() as unknown as Classifier;
    }
    const params = this.estimator.getParams();
    const EstimatorClass = this.estimator.constructor as new (...args: unknown[]) => Classifier;
    try {
      return new EstimatorClass(params);
    } catch {
      throw new InvalidParameterError(
        "Cannot clone base estimator for CalibratedClassifierCV cross-validation. " +
          "Implement clone() on the estimator or ensure its constructor accepts a params object.",
        "estimator",
        this.estimator
      );
    }
  }

  /**
   * Produce out-of-fold (held-out) probability predictions via K-fold
   * cross-validation. Row `i` of the returned flat array holds the
   * `nClasses` probabilities predicted for sample `i` by a base estimator
   * that was NOT trained on sample `i`.
   */
  private crossValPredictProba(
    X: Tensor,
    y: Tensor,
    nSamples: number,
    nFeatures: number,
    nClasses: number
  ): Float64Array {
    const oof = new Float64Array(nSamples * nClasses);
    // Number of folds is capped at the sample count so each fold is non-empty.
    const nFolds = Math.max(2, Math.min(this.cv, nSamples));
    const classToCol = new Map<number, number>();
    for (let c = 0; c < nClasses; c++) classToCol.set(this.classes_![c]!, c);

    for (let fold = 0; fold < nFolds; fold++) {
      const trainRows: number[] = [];
      const testRows: number[] = [];
      for (let i = 0; i < nSamples; i++) {
        if (i % nFolds === fold) testRows.push(i);
        else trainRows.push(i);
      }
      if (trainRows.length === 0 || testRows.length === 0) continue;

      // Skip folds whose training split is missing a class — predictProba
      // column alignment would otherwise be ambiguous.
      const trainClasses = new Set<number>();
      for (const r of trainRows) trainClasses.add(Number(y.data[y.offset + r]));
      if (trainClasses.size < nClasses) {
        // Fall back: leave these rows to be filled by the global estimator below.
        continue;
      }

      const XTrain = new Float64Array(trainRows.length * nFeatures);
      const yTrain = new Float64Array(trainRows.length);
      for (let r = 0; r < trainRows.length; r++) {
        const src = X.offset + trainRows[r]! * nFeatures;
        for (let j = 0; j < nFeatures; j++) XTrain[r * nFeatures + j] = Number(X.data[src + j]);
        yTrain[r] = Number(y.data[y.offset + trainRows[r]!]);
      }
      const XTest = new Float64Array(testRows.length * nFeatures);
      for (let r = 0; r < testRows.length; r++) {
        const src = X.offset + testRows[r]! * nFeatures;
        for (let j = 0; j < nFeatures; j++) XTest[r * nFeatures + j] = Number(X.data[src + j]);
      }

      const fold_est = this.cloneEstimator();
      fold_est.fit(
        tensor(Array.from(XTrain)).reshape([trainRows.length, nFeatures]),
        tensor(Array.from(yTrain))
      );
      const proba = fold_est.predictProba(
        tensor(Array.from(XTest)).reshape([testRows.length, nFeatures])
      );
      // Map the fold estimator's class columns onto the global class order.
      const foldClasses = this.estimatorClasses(fold_est);
      for (let r = 0; r < testRows.length; r++) {
        const dest = testRows[r]! * nClasses;
        for (let k = 0; k < foldClasses.length; k++) {
          const col = classToCol.get(foldClasses[k]!);
          if (col !== undefined) {
            oof[dest + col] = Number(proba.data[proba.offset + r * foldClasses.length + k]);
          }
        }
      }
    }

    // Any rows that were skipped (e.g. degenerate folds) fall back to a global
    // fit so the calibrator still has a value for every sample.
    const filled = new Uint8Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      let any = false;
      for (let c = 0; c < nClasses; c++) {
        if (oof[i * nClasses + c] !== 0) {
          any = true;
          break;
        }
      }
      filled[i] = any ? 1 : 0;
    }
    if (filled.some((f) => f === 0)) {
      const global = this.cloneEstimator();
      global.fit(X, y);
      const proba = global.predictProba(X);
      const gClasses = this.estimatorClasses(global);
      for (let i = 0; i < nSamples; i++) {
        if (filled[i]) continue;
        const dest = i * nClasses;
        for (let k = 0; k < gClasses.length; k++) {
          const col = classToCol.get(gClasses[k]!);
          if (col !== undefined) {
            oof[dest + col] = Number(proba.data[proba.offset + i * gClasses.length + k]);
          }
        }
      }
    }

    return oof;
  }

  /**
   * Read the fitted class order from an estimator, falling back to the
   * globally observed classes when the estimator does not expose them.
   */
  private estimatorClasses(est: Classifier): number[] {
    const classesTensor = est.classes;
    if (classesTensor) {
      const out: number[] = [];
      for (let i = 0; i < classesTensor.size; i++) {
        out.push(Number(classesTensor.data[classesTensor.offset + i]));
      }
      return out;
    }
    return this.classes_!;
  }

  private fitSigmoid(scores: number[], targets: number[]): [number, number] {
    // Platt scaling: minimize NLL with sigmoid(a*s + b)
    let a = 1;
    let b = 0;
    const lr = 0.01;
    const n = scores.length;

    for (let iter = 0; iter < 100; iter++) {
      let gradA = 0;
      let gradB = 0;
      for (let i = 0; i < n; i++) {
        const z = a * (scores[i] ?? 0) + b;
        const p = 1 / (1 + Math.exp(-z));
        const err = p - (targets[i] ?? 0);
        gradA += err * (scores[i] ?? 0);
        gradB += err;
      }
      a -= (lr * gradA) / n;
      b -= (lr * gradB) / n;
    }

    return [a, b];
  }

  private isotonicRegression(y: number[]): number[] {
    // Pool Adjacent Violators Algorithm
    const n = y.length;
    const result = [...y];
    const weight = new Array<number>(n).fill(1);

    let changed = true;
    while (changed) {
      changed = false;
      for (let i = 0; i < n - 1; i++) {
        if ((result[i] ?? 0) > (result[i + 1] ?? 0)) {
          // Pool
          const totalW = (weight[i] ?? 0) + (weight[i + 1] ?? 0);
          const pooled =
            ((result[i] ?? 0) * (weight[i] ?? 0) + (result[i + 1] ?? 0) * (weight[i + 1] ?? 0)) /
            totalW;
          result[i] = pooled;
          result[i + 1] = pooled;
          weight[i] = totalW;
          weight[i + 1] = totalW;
          changed = true;
        }
      }
    }

    return result;
  }

  private isotonicInterp(xs: number[], ys: number[], x: number): number {
    if (xs.length === 0) return 0.5;
    if (x <= (xs[0] ?? 0)) return ys[0] ?? 0;
    if (x >= (xs[xs.length - 1] ?? 0)) return ys[ys.length - 1] ?? 0;

    // Binary search for position
    let lo = 0;
    let hi = xs.length - 1;
    while (lo < hi - 1) {
      const mid = (lo + hi) >> 1;
      if ((xs[mid] ?? 0) <= x) lo = mid;
      else hi = mid;
    }

    // Linear interpolation
    const x0 = xs[lo] ?? 0;
    const x1 = xs[hi] ?? 0;
    const y0 = ys[lo] ?? 0;
    const y1 = ys[hi] ?? 0;
    if (Math.abs(x1 - x0) < 1e-15) return y0;
    const t = (x - x0) / (x1 - x0);
    return y0 + t * (y1 - y0);
  }
}

/**
 * Compute calibration curve (reliability diagram data).
 *
 * Returns mean predicted probabilities and fraction of positives
 * for each bin of predicted probability.
 *
 * @param yTrue - True binary labels (0 or 1)
 * @param yProb - Predicted probabilities for the positive class
 * @param options - Configuration options
 * @returns Object with `meanPredicted` and `fractionPositives` arrays
 *
 * @example
 * ```ts
 * import { calibrationCurve } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const yTrue = tensor([0, 0, 1, 1, 1]);
 * const yProb = tensor([0.1, 0.3, 0.6, 0.8, 0.9]);
 * const { meanPredicted, fractionPositives } = calibrationCurve(yTrue, yProb);
 * ```
 */
export function calibrationCurve(
  yTrue: Tensor,
  yProb: Tensor,
  options: {
    readonly nBins?: number;
    readonly strategy?: "uniform" | "quantile";
  } = {}
): { meanPredicted: number[]; fractionPositives: number[] } {
  const nBins = options.nBins ?? 10;
  const strategy = options.strategy ?? "uniform";
  const n = yTrue.size;

  if (n !== yProb.size) {
    throw new InvalidParameterError(
      `yTrue and yProb must have same size; got ${n} vs ${yProb.size}`,
      "size",
      yProb.size
    );
  }

  // Extract data
  const trueVals: number[] = [];
  const probVals: number[] = [];
  for (let i = 0; i < n; i++) {
    trueVals.push(Number(yTrue.data[yTrue.offset + i]));
    probVals.push(Number(yProb.data[yProb.offset + i]));
  }

  let binEdges: number[];

  if (strategy === "uniform") {
    binEdges = [];
    for (let i = 0; i <= nBins; i++) {
      binEdges.push(i / nBins);
    }
  } else {
    // Quantile strategy
    const sorted = [...probVals].sort((a, b) => a - b);
    binEdges = [0];
    for (let i = 1; i < nBins; i++) {
      const idx = Math.floor((i / nBins) * n);
      binEdges.push(sorted[Math.min(idx, n - 1)] ?? 0);
    }
    binEdges.push(1.0001); // slightly above 1 to include 1.0
  }

  const meanPredicted: number[] = [];
  const fractionPositives: number[] = [];

  for (let b = 0; b < nBins; b++) {
    const lo = binEdges[b] ?? 0;
    const hi = binEdges[b + 1] ?? 1;

    let sumProb = 0;
    let sumTrue = 0;
    let count = 0;

    for (let i = 0; i < n; i++) {
      const p = probVals[i] ?? 0;
      if (p >= lo && (b === nBins - 1 ? p <= hi : p < hi)) {
        sumProb += p;
        sumTrue += trueVals[i] ?? 0;
        count++;
      }
    }

    if (count > 0) {
      meanPredicted.push(sumProb / count);
      fractionPositives.push(sumTrue / count);
    }
  }

  return { meanPredicted, fractionPositives };
}
