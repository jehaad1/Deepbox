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

import { DataValidationError, InvalidParameterError, NotFittedError, ShapeError } from "../core";
import { type Tensor, tensor } from "../ndarray";
import { cloneEstimator } from "./_internal";
import {
  assertContiguous,
  percentileSorted,
  toFloat64View,
  validateFitInputs,
  validatePredictInputs,
} from "./_validation";
import type { Classifier } from "./base";

/** Numerically stable logistic function. */
function sigmoid(z: number): number {
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const e = Math.exp(z);
  return e / (1 + e);
}

/** Numerically stable log(1 + exp(z)). */
function softplus(z: number): number {
  return z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z));
}

/**
 * Platt scaling: fit `P(y = 1 | f) = sigmoid(a * f + b)` by minimizing the
 * cross-entropy against the prior-smoothed targets of Platt (1999)
 * (`(n1 + 1) / (n1 + 2)` for positives, `1 / (n0 + 2)` for negatives), which
 * is also what scikit-learn does. The problem is convex in `(a, b)`; it is
 * solved with a damped Newton method and a backtracking line search.
 */
function fitPlatt(scores: Float64Array, positive: Uint8Array): [number, number] {
  const n = scores.length;
  let n1 = 0;
  for (let i = 0; i < n; i++) n1 += positive[i] as number;
  const n0 = n - n1;
  const hi = (n1 + 1) / (n1 + 2);
  const lo = 1 / (n0 + 2);
  const targets = new Float64Array(n);
  for (let i = 0; i < n; i++) targets[i] = positive[i] ? hi : lo;

  const loss = (a: number, b: number): number => {
    let total = 0;
    for (let i = 0; i < n; i++) {
      const z = a * (scores[i] as number) + b;
      total += softplus(z) - (targets[i] as number) * z;
    }
    return total;
  };

  let a = 0;
  let b = Math.log((n1 + 1) / (n0 + 1));
  let current = loss(a, b);

  for (let iter = 0; iter < 100; iter++) {
    let gA = 0;
    let gB = 0;
    let hAA = 0;
    let hAB = 0;
    let hBB = 0;
    for (let i = 0; i < n; i++) {
      const f = scores[i] as number;
      const p = sigmoid(a * f + b);
      const err = p - (targets[i] as number);
      const w = p * (1 - p);
      gA += err * f;
      gB += err;
      hAA += w * f * f;
      hAB += w * f;
      hBB += w;
    }
    if (Math.abs(gA) < 1e-10 && Math.abs(gB) < 1e-10) break;

    // Solve (H + ridge * I) d = -g; the tiny ridge keeps H invertible when
    // all scores coincide or the classes are perfectly separated.
    const ridge = 1e-12 * (1 + hAA + hBB);
    const m00 = hAA + ridge;
    const m11 = hBB + ridge;
    const det = m00 * m11 - hAB * hAB;
    const dA = -(m11 * gA - hAB * gB) / det;
    const dB = -(m00 * gB - hAB * gA) / det;
    const slope = gA * dA + gB * dB;
    if (!(slope < 0)) break;

    let step = 1;
    let accepted = false;
    for (let ls = 0; ls < 40; ls++) {
      const next = loss(a + step * dA, b + step * dB);
      if (next <= current + 1e-4 * step * slope) {
        a += step * dA;
        b += step * dB;
        current = next;
        accepted = true;
        break;
      }
      step /= 2;
    }
    if (!accepted) break;
    if (-slope < 1e-14 * Math.max(1, Math.abs(current))) break;
  }

  return [a, b];
}

/**
 * Isotonic regression of 0/1 labels on scores (pool adjacent violators).
 *
 * Scores with equal values are merged first (mean label). The result is the
 * list of breakpoints of the non-decreasing step function, where flat runs keep
 * only their two end points, so linear interpolation between the returned
 * points reproduces `sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")`.
 */
function fitIsotonic(
  scores: Float64Array,
  positive: Uint8Array
): { x: Float64Array; y: Float64Array } {
  const n = scores.length;
  const order = new Int32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  order.sort((p, q) => (scores[p] as number) - (scores[q] as number));

  // Unique scores with the sum of labels and the number of samples at each.
  const ux: number[] = [];
  const uSum: number[] = [];
  const uCnt: number[] = [];
  for (let k = 0; k < n; k++) {
    const i = order[k] as number;
    const s = scores[i] as number;
    if (ux.length > 0 && ux[ux.length - 1] === s) {
      uSum[uSum.length - 1] = (uSum[uSum.length - 1] as number) + (positive[i] as number);
      uCnt[uCnt.length - 1] = (uCnt[uCnt.length - 1] as number) + 1;
    } else {
      ux.push(s);
      uSum.push(positive[i] as number);
      uCnt.push(1);
    }
  }

  // Weighted pool-adjacent-violators over blocks of consecutive unique scores.
  const bSum: number[] = [];
  const bCnt: number[] = [];
  const bStart: number[] = [];
  const bEnd: number[] = [];
  for (let u = 0; u < ux.length; u++) {
    bSum.push(uSum[u] as number);
    bCnt.push(uCnt[u] as number);
    bStart.push(u);
    bEnd.push(u);
    while (bSum.length > 1) {
      const m = bSum.length - 1;
      const meanPrev = (bSum[m - 1] as number) / (bCnt[m - 1] as number);
      const meanCur = (bSum[m] as number) / (bCnt[m] as number);
      if (meanPrev < meanCur) break;
      bSum[m - 1] = (bSum[m - 1] as number) + (bSum[m] as number);
      bCnt[m - 1] = (bCnt[m - 1] as number) + (bCnt[m] as number);
      bEnd[m - 1] = bEnd[m] as number;
      bSum.pop();
      bCnt.pop();
      bStart.pop();
      bEnd.pop();
    }
  }

  const xs: number[] = [];
  const ys: number[] = [];
  for (let b = 0; b < bSum.length; b++) {
    const mean = (bSum[b] as number) / (bCnt[b] as number);
    xs.push(ux[bStart[b] as number] as number);
    ys.push(mean);
    if (bEnd[b] !== bStart[b]) {
      xs.push(ux[bEnd[b] as number] as number);
      ys.push(mean);
    }
  }
  return { x: Float64Array.from(xs), y: Float64Array.from(ys) };
}

/** Piecewise-linear interpolation through `(xs, ys)`, clipped outside the range. */
function interpolate(xs: Float64Array, ys: Float64Array, x: number): number {
  const n = xs.length;
  if (n === 0) return 0.5;
  if (x <= (xs[0] as number)) return ys[0] as number;
  if (x >= (xs[n - 1] as number)) return ys[n - 1] as number;

  let lo = 0;
  let hi = n - 1;
  while (lo < hi - 1) {
    const mid = (lo + hi) >> 1;
    if ((xs[mid] as number) <= x) lo = mid;
    else hi = mid;
  }
  const x0 = xs[lo] as number;
  const x1 = xs[hi] as number;
  const y0 = ys[lo] as number;
  const y1 = ys[hi] as number;
  return y0 + ((x - x0) / (x1 - x0)) * (y1 - y0);
}

/**
 * Calibrated Classifier with Cross-Validation.
 *
 * Wraps a base classifier and calibrates its probability predictions
 * using either Platt scaling (sigmoid) or isotonic regression.
 *
 * `fit` computes out-of-fold probabilities with stratified K-fold cross-validation
 * (each fold uses a fresh clone of the base estimator), fits one calibrator per
 * class on those probabilities, and finally fits the base estimator itself on all
 * of the data. This matches scikit-learn's `CalibratedClassifierCV(ensemble=False)`.
 * Binary problems calibrate only the positive class and use `1 - p` for the
 * other; multiclass problems calibrate each class one-vs-rest and renormalize.
 *
 * Note that the wrapped estimator instance is fitted in place, and that its
 * constructor or `clone()` must be able to rebuild it from `getParams()`.
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

  // One calibrator per entry of calibCols_ (all classes, or only the positive
  // class for binary problems).
  private calibCols_: number[] = [];
  private calibA_?: Float64Array; // sigmoid: slope a
  private calibB_?: Float64Array; // sigmoid: intercept b
  private isotonicX_?: Float64Array[]; // isotonic: breakpoints (scores)
  private isotonicY_?: Float64Array[]; // isotonic: calibrated values
  private classes_?: number[];
  private nFeaturesIn_ = 0;
  private fitted = false;

  /**
   * @param options.estimator - Base classifier exposing `predictProba`
   * @param options.method - `"sigmoid"` (Platt scaling, default) or `"isotonic"`
   * @param options.cv - Number of cross-validation folds, an integer >= 2 (default: 5)
   * @throws {InvalidParameterError} If `method` or `cv` is invalid
   */
  constructor(options: {
    readonly estimator: Classifier;
    readonly method?: "sigmoid" | "isotonic";
    readonly cv?: number;
  }) {
    const method = options.method ?? "sigmoid";
    const cv = options.cv ?? 5;
    if (method !== "sigmoid" && method !== "isotonic") {
      throw new InvalidParameterError(
        `method must be "sigmoid" or "isotonic"; got ${String(method)}`,
        "method",
        method
      );
    }
    if (!Number.isInteger(cv) || cv < 2) {
      throw new InvalidParameterError(`cv must be an integer >= 2; got ${String(cv)}`, "cv", cv);
    }
    this.estimator = options.estimator;
    this.method = method;
    this.cv = cv;
  }

  /**
   * Fit the calibrated classifier.
   *
   * @param X - Training features of shape (n_samples, n_features)
   * @param y - Class labels of shape (n_samples,), at least two distinct classes
   * @returns this
   * @throws {ShapeError} If X is not 2D, y is not 1D, or their lengths differ
   * @throws {DataValidationError} If the data is empty, non-finite or has fewer than two classes
   */
  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const yv = toFloat64View(y);

    const classes = [...new Set(yv)].sort((a, b) => a - b);
    const nClasses = classes.length;
    if (nClasses < 2) {
      throw new DataValidationError(
        "y must contain at least two distinct classes to calibrate probabilities"
      );
    }
    const classToCol = new Map<number, number>();
    for (let c = 0; c < nClasses; c++) classToCol.set(classes[c] as number, c);

    // Invalidate the previous fit so a failure below cannot leave a model that
    // mixes old calibrators with a newly fitted estimator.
    this.fitted = false;

    // Out-of-fold probabilities: row i was predicted by an estimator that did
    // not see sample i, which avoids the optimistic bias of in-sample scores.
    const oof = this.crossValPredictProba(X, yv, classes, classToCol, nSamples, nFeatures);

    // Binary problems calibrate the positive class only.
    const calibCols = nClasses === 2 ? [1] : classes.map((_, c) => c);
    const calibA = new Float64Array(calibCols.length);
    const calibB = new Float64Array(calibCols.length);
    const isoX: Float64Array[] = [];
    const isoY: Float64Array[] = [];

    for (let k = 0; k < calibCols.length; k++) {
      const col = calibCols[k] as number;
      const classVal = classes[col] as number;
      const scores = new Float64Array(nSamples);
      const positive = new Uint8Array(nSamples);
      for (let i = 0; i < nSamples; i++) {
        scores[i] = oof[i * nClasses + col] as number;
        positive[i] = yv[i] === classVal ? 1 : 0;
      }
      if (this.method === "sigmoid") {
        const [a, b] = fitPlatt(scores, positive);
        calibA[k] = a;
        calibB[k] = b;
      } else {
        const iso = fitIsotonic(scores, positive);
        isoX.push(iso.x);
        isoY.push(iso.y);
      }
    }

    // Fit the final estimator on all data; predict/predictProba apply the
    // learned calibration map to its probabilities.
    this.estimator.fit(X, y);

    this.classes_ = classes;
    this.nFeaturesIn_ = nFeatures;
    this.calibCols_ = calibCols;
    if (this.method === "sigmoid") {
      this.calibA_ = calibA;
      this.calibB_ = calibB;
      delete this.isotonicX_;
      delete this.isotonicY_;
    } else {
      this.isotonicX_ = isoX;
      this.isotonicY_ = isoY;
      delete this.calibA_;
      delete this.calibB_;
    }
    this.fitted = true;
    return this;
  }

  /**
   * Predict class labels (the class with the highest calibrated probability).
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns float64 labels of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   */
  predict(X: Tensor): Tensor {
    const proba = toFloat64View(this.predictProba(X));
    const classes = this.classes_ as number[];
    const nClasses = classes.length;
    const nSamples = X.shape[0] ?? 0;
    const labels = new Float64Array(nSamples);

    for (let i = 0; i < nSamples; i++) {
      let bestC = 0;
      let bestP = -Infinity;
      for (let c = 0; c < nClasses; c++) {
        const p = proba[i * nClasses + c] as number;
        if (p > bestP) {
          bestP = p;
          bestC = c;
        }
      }
      labels[i] = classes[bestC] as number;
    }

    return tensor(labels, { dtype: "float64" });
  }

  /**
   * Calibrated class probabilities. Columns follow the sorted class labels
   * (see {@link CalibratedClassifierCV.classes}) and each row sums to 1.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns float64 probabilities of shape (n_samples, n_classes)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has the wrong number of features
   */
  predictProba(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("CalibratedClassifierCV must be fitted before predictProba");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "CalibratedClassifierCV");
    const classes = this.classes_ as number[];
    const nSamples = X.shape[0] ?? 0;
    const nClasses = classes.length;

    const classToCol = new Map<number, number>();
    for (let c = 0; c < nClasses; c++) classToCol.set(classes[c] as number, c);
    const base = this.alignProba(this.estimator, X, nSamples, classes, classToCol);
    const result = new Float64Array(nSamples * nClasses);
    const cols = this.calibCols_;
    const isotonic = this.method === "isotonic";

    for (let i = 0; i < nSamples; i++) {
      const row = i * nClasses;
      const calibrate = (k: number): number => {
        const p = base[row + (cols[k] as number)] as number;
        if (isotonic) {
          return interpolate(
            (this.isotonicX_ as Float64Array[])[k] as Float64Array,
            (this.isotonicY_ as Float64Array[])[k] as Float64Array,
            p
          );
        }
        const a = (this.calibA_ as Float64Array)[k] as number;
        const b = (this.calibB_ as Float64Array)[k] as number;
        return sigmoid(a * p + b);
      };

      if (nClasses === 2) {
        // Only the positive class (column 1) has a calibrator.
        const p1 = Math.min(1, Math.max(0, calibrate(0)));
        result[row] = 1 - p1;
        result[row + 1] = p1;
        continue;
      }

      let sum = 0;
      for (let k = 0; k < cols.length; k++) {
        const v = calibrate(k);
        result[row + (cols[k] as number)] = v;
        sum += v;
      }
      if (sum > 0) {
        for (let c = 0; c < nClasses; c++) {
          result[row + c] = (result[row + c] as number) / sum;
        }
      } else {
        for (let c = 0; c < nClasses; c++) result[row + c] = 1 / nClasses;
      }
    }

    return tensor(result, { dtype: "float64" }).reshape([nSamples, nClasses]);
  }

  /**
   * Mean accuracy of `predict(X)` against `y`.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True labels of shape (n_samples,)
   * @returns Fraction of correctly classified samples
   * @throws {ShapeError} If y is not 1D with one entry per row of X
   */
  score(X: Tensor, y: Tensor): number {
    assertContiguous(y, "y");
    if (y.ndim !== 1 || y.shape[0] !== X.shape[0]) {
      throw new ShapeError(
        `y must be 1-dimensional with one label per row of X; got y.shape=[${y.shape.join(", ")}], X.shape=[${X.shape.join(", ")}]`
      );
    }
    const pred = toFloat64View(this.predict(X));
    const truth = toFloat64View(y);
    const nSamples = y.size;
    if (nSamples === 0) return Number.NaN;
    let correct = 0;
    for (let i = 0; i < nSamples; i++) {
      if (pred[i] === truth[i]) correct++;
    }
    return correct / nSamples;
  }

  /**
   * Class labels seen during fit in ascending order, or `undefined` before fit.
   */
  get classes(): Tensor | undefined {
    return this.classes_
      ? tensor(Float64Array.from(this.classes_), { dtype: "float64" })
      : undefined;
  }

  getParams(): Record<string, unknown> {
    return {
      estimator: this.estimator,
      method: this.method,
      cv: this.cv,
    };
  }

  /**
   * Set `method` or `cv`. All values are validated before any is applied.
   *
   * @throws {InvalidParameterError} If a name is unknown, a value is invalid, or `estimator` is passed
   */
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
          break;
        case "cv":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 2) {
            throw new InvalidParameterError(
              `cv must be an integer >= 2; got ${String(value)}`,
              "cv",
              value
            );
          }
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
    for (const [key, value] of Object.entries(params)) {
      if (key === "method") this.method = value as "sigmoid" | "isotonic";
      else if (key === "cv") this.cv = value as number;
    }
    return this;
  }

  /**
   * Create an unfitted copy with the same parameters and a clone of the base estimator.
   */
  clone(): CalibratedClassifierCV {
    return new CalibratedClassifierCV({
      estimator: this.cloneEstimator(),
      method: this.method,
      cv: this.cv,
    });
  }

  /**
   * Clone the wrapped base estimator into a fresh, unfitted instance so each
   * cross-validation fold trains independently of the others.
   */
  private cloneEstimator(): Classifier {
    return cloneEstimator(this.estimator, "CalibratedClassifierCV");
  }

  /**
   * Predict probabilities with `est` and reorder its columns to the global
   * class order. Classes unknown to `est` get probability 0. Estimators that do
   * not expose `classes` are assumed to use the global class order.
   */
  private alignProba(
    est: Classifier,
    X: Tensor,
    nRows: number,
    classes: readonly number[],
    classToCol: Map<number, number>
  ): Float64Array {
    const nClasses = classes.length;
    const proba = est.predictProba(X);
    const estClasses = est.classes ? Array.from(toFloat64View(est.classes)) : classes;
    if (proba.ndim !== 2 || proba.shape[0] !== nRows || proba.shape[1] !== estClasses.length) {
      throw new ShapeError(
        `base estimator predictProba returned shape [${proba.shape.join(", ")}]; expected [${nRows}, ${estClasses.length}]`
      );
    }
    const values = toFloat64View(proba);
    const out = new Float64Array(nRows * nClasses);
    const width = estClasses.length;
    for (let k = 0; k < width; k++) {
      const col = classToCol.get(estClasses[k] as number);
      if (col === undefined) continue;
      for (let r = 0; r < nRows; r++) {
        out[r * nClasses + col] = values[r * width + k] as number;
      }
    }
    return out;
  }

  /**
   * Produce out-of-fold (held-out) probability predictions via stratified
   * K-fold cross-validation. Row `i` of the returned flat array holds the
   * `nClasses` probabilities predicted for sample `i` by a base estimator
   * that was NOT trained on sample `i`.
   */
  private crossValPredictProba(
    X: Tensor,
    y: Float64Array,
    classes: readonly number[],
    classToCol: Map<number, number>,
    nSamples: number,
    nFeatures: number
  ): Float64Array {
    const nClasses = classes.length;
    const Xv = toFloat64View(X);
    const oof = new Float64Array(nSamples * nClasses);
    const filled = new Uint8Array(nSamples);
    // Number of folds is capped at the sample count so each fold is non-empty.
    const nFolds = Math.max(2, Math.min(this.cv, nSamples));

    // Stratified assignment: walk the samples ordered by class and deal them
    // to the folds in turn, so every class is spread evenly over the folds.
    const order = Array.from({ length: nSamples }, (_, i) => i);
    order.sort(
      (p, q) =>
        (classToCol.get(y[p] as number) as number) - (classToCol.get(y[q] as number) as number) ||
        p - q
    );
    const foldOf = new Int32Array(nSamples);
    for (let k = 0; k < nSamples; k++) foldOf[order[k] as number] = k % nFolds;

    for (let fold = 0; fold < nFolds; fold++) {
      const trainRows: number[] = [];
      const testRows: number[] = [];
      for (let i = 0; i < nSamples; i++) {
        if (foldOf[i] === fold) testRows.push(i);
        else trainRows.push(i);
      }
      if (trainRows.length === 0 || testRows.length === 0) continue;

      // Skip folds whose training split is missing a class; their rows are
      // filled by the global estimator below.
      const trainClasses = new Set<number>();
      for (const r of trainRows) trainClasses.add(y[r] as number);
      if (trainClasses.size < nClasses) continue;

      const XTrain = new Float64Array(trainRows.length * nFeatures);
      const yTrain = new Float64Array(trainRows.length);
      for (let r = 0; r < trainRows.length; r++) {
        const src = (trainRows[r] as number) * nFeatures;
        XTrain.set(Xv.subarray(src, src + nFeatures), r * nFeatures);
        yTrain[r] = y[trainRows[r] as number] as number;
      }
      const XTest = new Float64Array(testRows.length * nFeatures);
      for (let r = 0; r < testRows.length; r++) {
        const src = (testRows[r] as number) * nFeatures;
        XTest.set(Xv.subarray(src, src + nFeatures), r * nFeatures);
      }

      const foldEstimator = this.cloneEstimator();
      foldEstimator.fit(
        tensor(XTrain, { dtype: "float64" }).reshape([trainRows.length, nFeatures]),
        tensor(yTrain, { dtype: "float64" })
      );
      const proba = this.alignProba(
        foldEstimator,
        tensor(XTest, { dtype: "float64" }).reshape([testRows.length, nFeatures]),
        testRows.length,
        classes,
        classToCol
      );
      for (let r = 0; r < testRows.length; r++) {
        const row = testRows[r] as number;
        oof.set(proba.subarray(r * nClasses, (r + 1) * nClasses), row * nClasses);
        filled[row] = 1;
      }
    }

    // Rows left over from skipped folds fall back to a global fit so the
    // calibrator still has a value for every sample.
    if (filled.some((f) => f === 0)) {
      const global = this.cloneEstimator();
      global.fit(X, tensor(y, { dtype: "float64" }));
      const proba = this.alignProba(global, X, nSamples, classes, classToCol);
      for (let i = 0; i < nSamples; i++) {
        if (filled[i]) continue;
        oof.set(proba.subarray(i * nClasses, (i + 1) * nClasses), i * nClasses);
      }
    }

    return oof;
  }
}

/**
 * Compute calibration curve (reliability diagram data).
 *
 * Returns mean predicted probabilities and fraction of positives
 * for each bin of predicted probability. Bins that contain no samples are
 * omitted, so the returned arrays can be shorter than `nBins`.
 *
 * Bins follow `sklearn.calibration.calibration_curve`: the first bin is
 * `[edge0, edge1]` and the others are `(lo, hi]`. The default of 10 bins differs
 * from scikit-learn's default of 5.
 *
 * @param yTrue - True binary labels. Without `posLabel` the labels must be 0/1 (or -1/1, where 1 is positive)
 * @param yProb - Predicted probabilities for the positive class, each in [0, 1]
 * @param options.nBins - Number of bins, an integer >= 1 (default: 10)
 * @param options.strategy - `"uniform"` (equal-width bins, default) or `"quantile"` (equal-count bins)
 * @param options.posLabel - Label treated as the positive class
 * @returns Object with `meanPredicted` and `fractionPositives` arrays
 * @throws {InvalidParameterError} If the sizes differ or an option is invalid
 * @throws {DataValidationError} If inputs are empty, probabilities are outside [0, 1] or labels are not binary
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
    readonly posLabel?: number;
  } = {}
): { meanPredicted: number[]; fractionPositives: number[] } {
  const nBins = options.nBins ?? 10;
  const strategy = options.strategy ?? "uniform";
  const n = yTrue.size;

  if (!Number.isInteger(nBins) || nBins < 1) {
    throw new InvalidParameterError(
      `nBins must be an integer >= 1; got ${String(nBins)}`,
      "nBins",
      nBins
    );
  }
  if (strategy !== "uniform" && strategy !== "quantile") {
    throw new InvalidParameterError(
      `strategy must be "uniform" or "quantile"; got ${String(strategy)}`,
      "strategy",
      strategy
    );
  }
  if (n !== yProb.size) {
    throw new InvalidParameterError(
      `yTrue and yProb must have same size; got ${n} vs ${yProb.size}`,
      "size",
      yProb.size
    );
  }
  if (n === 0) {
    throw new DataValidationError("yTrue and yProb must not be empty");
  }
  assertContiguous(yTrue, "yTrue");
  assertContiguous(yProb, "yProb");

  const trueVals = toFloat64View(yTrue);
  const probVals = toFloat64View(yProb);

  for (let i = 0; i < n; i++) {
    const p = probVals[i] as number;
    if (!(p >= 0 && p <= 1)) {
      throw new DataValidationError(`yProb must be finite and within [0, 1]; found ${p}`);
    }
    if (!Number.isFinite(trueVals[i] as number)) {
      throw new DataValidationError("yTrue contains non-finite values (NaN or Inf)");
    }
  }

  const labels = [...new Set(trueVals)];
  if (labels.length > 2) {
    throw new DataValidationError(
      `Only binary classification is supported; yTrue has ${labels.length} distinct labels`
    );
  }
  let posLabel = options.posLabel;
  if (posLabel === undefined) {
    const subsetOf01 = labels.every((v) => v === 0 || v === 1);
    const subsetOfPm1 = labels.every((v) => v === -1 || v === 1);
    if (!subsetOf01 && !subsetOfPm1) {
      throw new DataValidationError(
        `yTrue has labels {${labels.join(", ")}}; pass options.posLabel to choose the positive class`
      );
    }
    posLabel = 1;
  }

  // Bin edges as numpy.linspace / numpy.percentile compute them.
  let binEdges: Float64Array;
  if (strategy === "uniform") {
    binEdges = new Float64Array(nBins + 1);
    const step = 1 / nBins;
    for (let i = 0; i < nBins; i++) binEdges[i] = i * step;
    binEdges[nBins] = 1;
  } else {
    const sorted = Float64Array.from(probVals).sort();
    binEdges = new Float64Array(nBins + 1);
    const step = 1 / nBins;
    for (let i = 0; i <= nBins; i++) {
      const q = i === nBins ? 1 : i * step;
      binEdges[i] = percentileSorted(sorted, q * 100);
    }
  }

  // A sample falls in bin j when exactly j of the inner edges are < p.
  const sumProb = new Float64Array(nBins);
  const sumTrue = new Float64Array(nBins);
  const count = new Float64Array(nBins);
  for (let i = 0; i < n; i++) {
    const p = probVals[i] as number;
    let lo = 0;
    let hi = nBins - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if ((binEdges[mid + 1] as number) < p) lo = mid + 1;
      else hi = mid;
    }
    sumProb[lo] = (sumProb[lo] as number) + p;
    sumTrue[lo] = (sumTrue[lo] as number) + (trueVals[i] === posLabel ? 1 : 0);
    count[lo] = (count[lo] as number) + 1;
  }

  const meanPredicted: number[] = [];
  const fractionPositives: number[] = [];
  for (let b = 0; b < nBins; b++) {
    const c = count[b] as number;
    if (c > 0) {
      meanPredicted.push((sumProb[b] as number) / c);
      fractionPositives.push((sumTrue[b] as number) / c);
    }
  }

  return { meanPredicted, fractionPositives };
}
