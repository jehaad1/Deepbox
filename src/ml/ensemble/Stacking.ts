/**
 * @see {@link https://deepbox.dev/docs/ml-ensemble | Deepbox Ensemble Methods}
 */

import {
  DataValidationError,
  InvalidParameterError,
  NotFittedError,
  NotImplementedError,
  ShapeError,
} from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import {
  assertContiguous,
  toFloat64View,
  validateFitInputs,
  validatePredictInputs,
} from "../_validation";
import type { Classifier, Estimator, Regressor } from "../base";
import { LinearRegression } from "../linear/LinearRegression";
import { LogisticRegression } from "../linear/LogisticRegression";

/** Default number of cross-validation folds used to build the meta-features. */
const DEFAULT_CV = 5;

/**
 * Which base-estimator output {@link StackingClassifier} feeds to the final estimator.
 *
 * - `"predictProba"`: class probabilities (one column for two classes, `K` columns otherwise)
 * - `"predict"`: the predicted label as a single column
 * - `"auto"`: `"predictProba"` when the estimator provides it, `"predict"` otherwise
 */
export type StackingMethod = "auto" | "predict" | "predictProba";

function floatMatrix(data: Float64Array, rows: number, cols: number): Tensor {
  return tensor(data, { dtype: "float64" }).reshape([rows, cols]);
}

function checkPassthrough(value: unknown): boolean {
  if (typeof value !== "boolean") {
    throw new InvalidParameterError("passthrough must be a boolean", "passthrough", value);
  }
  return value;
}

function checkCv(value: unknown): number | undefined {
  if (value !== undefined && (typeof value !== "number" || !Number.isInteger(value) || value < 1)) {
    throw new InvalidParameterError(
      "cv must be an integer >= 1 (1 disables cross-validation)",
      "cv",
      value
    );
  }
  return value;
}

function checkStackMethod(value: unknown): StackingMethod {
  if (value !== "auto" && value !== "predict" && value !== "predictProba") {
    throw new InvalidParameterError(
      `stackMethod must be "auto", "predict" or "predictProba"`,
      "stackMethod",
      value
    );
  }
  return value;
}

/** Throw unless every label is an integer that fits in int32 (the dtype of `predict`). */
function assertIntegerLabels(y: Float64Array, who: string): void {
  for (let i = 0; i < y.length; i++) {
    const v = y[i] as number;
    if (!Number.isInteger(v) || v < -2147483648 || v > 2147483647) {
      throw new DataValidationError(
        `${who} requires integer class labels in the int32 range; y[${i}] = ${v}`
      );
    }
  }
}

/** Shared `score` input checks; returns the targets as a flat array. */
function checkScoreTarget(y: Tensor, nPredicted: number): Float64Array {
  if (y.ndim !== 1) {
    throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
  }
  assertContiguous(y, "y");
  const yv = toFloat64View(y);
  for (let i = 0; i < yv.length; i++) {
    if (!Number.isFinite(yv[i])) {
      throw new DataValidationError("y contains non-finite values (NaN or Inf)");
    }
  }
  if (yv.length !== nPredicted) {
    throw new ShapeError(
      `X and y must have the same number of samples; got X=${nPredicted}, y=${yv.length}`
    );
  }
  if (yv.length === 0) {
    throw new DataValidationError("y must contain at least one sample");
  }
  return yv;
}

/**
 * Fresh unfitted copy of an estimator: its `clone()` when it has one, otherwise a new
 * instance built from `getParams()`.
 */
function cloneEstimator<T extends Estimator>(estimator: T, who: string): T {
  if (typeof estimator.clone === "function") {
    return estimator.clone() as T;
  }
  const EstimatorClass = estimator.constructor as new (params: Record<string, unknown>) => T;
  try {
    return new EstimatorClass(estimator.getParams());
  } catch {
    throw new InvalidParameterError(
      `Cannot clone a base estimator of ${who} for cross-validation. Implement clone() on the ` +
        "estimator, make its constructor accept the object returned by getParams(), or pass cv: 1.",
      "estimators",
      estimator
    );
  }
}

/** Copy the rows `rows` of the row-major matrix `x` (with `d` columns). */
function gatherRows(x: Float64Array, d: number, rows: Int32Array): Float64Array {
  const out = new Float64Array(rows.length * d);
  for (let r = 0; r < rows.length; r++) {
    const src = (rows[r] as number) * d;
    out.set(x.subarray(src, src + d), r * d);
  }
  return out;
}

function gatherValues(v: Float64Array, rows: Int32Array): Float64Array {
  const out = new Float64Array(rows.length);
  for (let r = 0; r < rows.length; r++) out[r] = v[rows[r] as number] as number;
  return out;
}

/**
 * Assign every row to a cross-validation fold. Rows are dealt out in turn, which spreads
 * sorted data evenly over the folds. Given `labels` (one per row) the rows are dealt out
 * class by class instead, so each class is spread over the folds (stratification).
 */
function assignFolds(n: number, nFolds: number, labels: Float64Array | undefined): Int32Array {
  const foldOf = new Int32Array(n);
  if (labels === undefined) {
    for (let i = 0; i < n; i++) foldOf[i] = i % nFolds;
    return foldOf;
  }
  const order = Array.from({ length: n }, (_, i) => i);
  order.sort((p, q) => (labels[p] as number) - (labels[q] as number) || p - q);
  for (let k = 0; k < n; k++) foldOf[order[k] as number] = k % nFolds;
  return foldOf;
}

/** Rows of fold `fold` (test) and all other rows (train), in ascending order. */
function splitFold(foldOf: Int32Array, fold: number): { train: Int32Array; test: Int32Array } {
  const train: number[] = [];
  const test: number[] = [];
  for (let i = 0; i < foldOf.length; i++) {
    (foldOf[i] === fold ? test : train).push(i);
  }
  return { train: Int32Array.from(train), test: Int32Array.from(test) };
}

function definedOptions<T>(params: Record<string, unknown>): T {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined) out[key] = value;
  }
  return out as T;
}

/**
 * Assemble the meta-feature matrix from the per-estimator blocks (and the original
 * features when `passthrough` is set).
 */
function assembleMeta(
  blocks: Float64Array[],
  widths: number[],
  n: number,
  xv: Float64Array | undefined,
  d: number
): { data: Float64Array; width: number } {
  let width = widths.reduce((a, b) => a + b, 0);
  if (xv !== undefined) width += d;
  const data = new Float64Array(n * width);
  let offset = 0;
  for (let b = 0; b < blocks.length; b++) {
    const w = widths[b] as number;
    const block = blocks[b] as Float64Array;
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < w; c++) data[i * width + offset + c] = block[i * w + c] as number;
    }
    offset += w;
  }
  if (xv !== undefined) {
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < d; c++) data[i * width + offset + c] = xv[i * d + c] as number;
    }
  }
  return { data, width };
}

/** Options of {@link StackingClassifier}. */
export type StackingClassifierOptions = {
  readonly estimators: Classifier[];
  readonly finalEstimator?: Classifier;
  readonly passthrough?: boolean;
  readonly cv?: number;
  readonly stackMethod?: StackingMethod;
};

/** Options of {@link StackingRegressor}. */
export type StackingRegressorOptions = {
  readonly estimators: Regressor[];
  readonly finalEstimator?: Regressor;
  readonly passthrough?: boolean;
  readonly cv?: number;
};

/**
 * Stacking Classifier.
 *
 * Combines multiple classifiers using a meta-learner (final estimator) that learns to
 * combine the base estimators' outputs.
 *
 * The meta-learner is trained on out-of-fold predictions: with `cv` folds (default 5, rows
 * assigned to folds in turn, stratified by class), each base estimator is cloned and fitted
 * on `cv - 1` folds and predicts the remaining one. The base estimators passed to the
 * constructor are then fitted on all the data (in place) and used at prediction time.
 * Training the meta-learner on in-sample predictions instead would let it trust base
 * estimators that memorized the training set. Set `cv: 1` to do that anyway.
 *
 * Two cases fall back to in-sample predictions for the affected rows: a fold whose
 * training part has fewer than two classes, and datasets with fewer than two rows.
 *
 * By default (`stackMethod: "auto"`) the final estimator receives the base estimators'
 * class probabilities: one column per estimator for two classes (the probability of the
 * larger label), `K` columns per estimator otherwise.
 *
 * Class labels must be integers.
 *
 * @example
 * ```ts
 * import { StackingClassifier, DecisionTreeClassifier, LogisticRegression } from 'deepbox/ml';
 *
 * const clf = new StackingClassifier({
 *   estimators: [
 *     new DecisionTreeClassifier({ maxDepth: 3 }),
 *     new DecisionTreeClassifier({ maxDepth: 5 }),
 *   ],
 *   finalEstimator: new LogisticRegression(),
 * });
 * clf.fit(X_train, y_train);
 * const predictions = clf.predict(X_test);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-ensemble | Deepbox Ensemble Methods}
 * @category Ensemble
 */
export class StackingClassifier implements Classifier {
  private readonly estimators: Classifier[];
  private readonly finalEstimator: Classifier;
  private passthrough: boolean;
  private cv: number | undefined;
  private stackMethod: StackingMethod;

  private classLabels: number[] = [];
  private nFeatures = 0;
  private methods: Array<"predict" | "predictProba"> = [];
  private fitted = false;

  /**
   * @param options.estimators - Base classifiers (at least one). They are fitted in place.
   * @param options.finalEstimator - Meta-learner (default: `LogisticRegression`)
   * @param options.passthrough - Also give the original features to the final estimator (default: false)
   * @param options.cv - Number of folds for the out-of-fold meta-features, an integer >= 1;
   *   1 trains the final estimator on in-sample predictions (default: 5)
   * @param options.stackMethod - Which base output to use, see {@link StackingMethod} (default: "auto")
   * @throws {InvalidParameterError} If there are no base estimators or an option is invalid
   */
  constructor(options: StackingClassifierOptions) {
    if (!options.estimators || options.estimators.length === 0) {
      throw new InvalidParameterError(
        "StackingClassifier requires at least one base estimator",
        "estimators",
        options.estimators
      );
    }
    this.estimators = [...options.estimators];
    this.finalEstimator = (options.finalEstimator ?? new LogisticRegression()) as Classifier;
    this.passthrough = checkPassthrough(options.passthrough ?? false);
    this.cv = checkCv(options.cv);
    this.stackMethod = checkStackMethod(options.stackMethod ?? "auto");
  }

  /**
   * Output of `est` on `X` as an (n, width) block, with probability columns aligned to
   * the global class order.
   */
  private block(
    est: Classifier,
    method: "predict" | "predictProba",
    X: Tensor,
    n: number
  ): { data: Float64Array; width: number } {
    if (method === "predict") {
      const pred = toFloat64View(est.predict(X));
      if (pred.length !== n) {
        throw new ShapeError(
          `base estimator predict returned ${pred.length} values for ${n} samples`
        );
      }
      return { data: Float64Array.from(pred), width: 1 };
    }
    const proba = est.predictProba(X);
    const labels = est.classes ? Array.from(toFloat64View(est.classes)) : this.classLabels;
    if (proba.ndim !== 2 || proba.shape[0] !== n || proba.shape[1] !== labels.length) {
      throw new ShapeError(
        `base estimator predictProba returned shape [${proba.shape.join(", ")}]; expected [${n}, ${labels.length}]`
      );
    }
    const pv = toFloat64View(proba);
    const k = this.classLabels.length;
    const width = k === 2 ? 1 : k;
    const data = new Float64Array(n * width);
    for (let j = 0; j < labels.length; j++) {
      const global = this.classLabels.indexOf(labels[j] as number);
      if (global < 0) continue;
      // Two classes: keep only the column of the larger label (the other is 1 - p).
      const col = k === 2 ? (global === 1 ? 0 : -1) : global;
      if (col < 0) continue;
      for (let i = 0; i < n; i++) data[i * width + col] = pv[i * labels.length + j] as number;
    }
    return { data, width };
  }

  /** Decide, per base estimator, whether its probabilities or its labels are stacked. */
  private resolveMethods(X: Tensor, n: number): void {
    this.methods = this.estimators.map((est) => {
      if (this.stackMethod === "predict") return "predict";
      if (typeof est.predictProba !== "function") {
        if (this.stackMethod === "predictProba") {
          throw new InvalidParameterError(
            'stackMethod "predictProba" needs every base estimator to implement predictProba',
            "stackMethod",
            this.stackMethod
          );
        }
        return "predict";
      }
      if (this.stackMethod === "predictProba") return "predictProba";
      try {
        this.block(est, "predictProba", X, n);
        return "predictProba";
      } catch {
        // The estimator declares predictProba but cannot produce it (for example an SVM
        // without probability estimates); stack its labels instead.
        return "predict";
      }
    });
  }

  private metaFeatures(X: Tensor): Tensor {
    const n = X.shape[0] ?? 0;
    const d = X.shape[1] ?? 0;
    const blocks: Float64Array[] = [];
    const widths: number[] = [];
    this.estimators.forEach((est, m) => {
      const b = this.block(est, this.methods[m] as "predict" | "predictProba", X, n);
      blocks.push(b.data);
      widths.push(b.width);
    });
    const { data, width } = assembleMeta(
      blocks,
      widths,
      n,
      this.passthrough ? toFloat64View(X) : undefined,
      d
    );
    return floatMatrix(data, n, width);
  }

  /**
   * Fit the base estimators and the final estimator.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Integer class labels of shape (n_samples,)
   * @returns this
   * @throws {ShapeError} If X is not 2D, y is not 1D or the sample counts differ
   * @throws {DataValidationError} If X or y contain NaN/Inf or y has non-integer labels
   * @throws {InvalidParameterError} If `cv > 1` and a base estimator cannot be cloned
   */
  fit(X: Tensor, y: Tensor): this {
    this.fitted = false;
    validateFitInputs(X, y);
    const n = X.shape[0] ?? 0;
    const d = X.shape[1] ?? 0;
    const xv = toFloat64View(X);
    const yv = toFloat64View(y);
    assertIntegerLabels(yv, "StackingClassifier");
    this.classLabels = [...new Set(yv)].sort((a, b) => a - b);
    this.nFeatures = d;

    // Base estimators used at prediction time are fitted on all the data.
    for (const est of this.estimators) est.fit(X, y);
    this.resolveMethods(X, n);

    const nFolds = Math.min(this.cv ?? DEFAULT_CV, n);
    const blocks: Float64Array[] = [];
    const widths: number[] = [];
    const nClasses = this.classLabels.length;
    // In-sample outputs of the full fits: the result for cv = 1 and the fallback rows.
    const inSample = this.estimators.map((est, m) =>
      this.block(est, this.methods[m] as "predict" | "predictProba", X, n)
    );
    for (const b of inSample) {
      blocks.push(b.data.slice());
      widths.push(b.width);
    }

    if (nFolds >= 2) {
      const foldOf = assignFolds(n, nFolds, yv);
      for (let fold = 0; fold < nFolds; fold++) {
        const { train, test } = splitFold(foldOf, fold);
        if (train.length === 0 || test.length === 0) continue;
        const trainY = gatherValues(yv, train);
        if (new Set(trainY).size < Math.min(2, nClasses)) continue;
        const trainX = floatMatrix(gatherRows(xv, d, train), train.length, d);
        const trainYTensor = tensor(trainY, { dtype: "float64" });
        const testX = floatMatrix(gatherRows(xv, d, test), test.length, d);
        this.estimators.forEach((est, m) => {
          const foldEstimator = cloneEstimator(est, "StackingClassifier");
          foldEstimator.fit(trainX, trainYTensor);
          const b = this.block(
            foldEstimator,
            this.methods[m] as "predict" | "predictProba",
            testX,
            test.length
          );
          const out = blocks[m] as Float64Array;
          for (let r = 0; r < test.length; r++) {
            const row = test[r] as number;
            for (let c = 0; c < b.width; c++) {
              out[row * b.width + c] = b.data[r * b.width + c] as number;
            }
          }
        });
      }
    }

    const meta = assembleMeta(blocks, widths, n, this.passthrough ? xv : undefined, d);
    this.finalEstimator.fit(floatMatrix(meta.data, n, meta.width), y);

    this.fitted = true;
    return this;
  }

  /**
   * Predict class labels with the final estimator.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predicted labels of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has the wrong rank or feature count
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("StackingClassifier must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeatures, "StackingClassifier");
    return this.finalEstimator.predict(this.metaFeatures(X));
  }

  /**
   * Predict class probabilities with the final estimator.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Probabilities of shape (n_samples, n_classes)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has the wrong rank or feature count
   */
  predictProba(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("StackingClassifier must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeatures, "StackingClassifier");
    if (typeof this.finalEstimator.predictProba !== "function") {
      throw new NotImplementedError(
        "The final estimator of StackingClassifier has no predictProba"
      );
    }
    return this.finalEstimator.predictProba(this.metaFeatures(X));
  }

  /**
   * Mean accuracy on the given data.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True labels of shape (n_samples,)
   * @returns Accuracy in [0, 1]
   * @throws {ShapeError} If y is not 1D or its length differs from the number of rows in X
   * @throws {DataValidationError} If y is empty or contains NaN/Inf
   */
  score(X: Tensor, y: Tensor): number {
    if (y.ndim !== 1) {
      throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    }
    const predictions = this.predict(X);
    const yv = checkScoreTarget(y, predictions.size);
    const pv = toFloat64View(predictions);
    let correct = 0;
    for (let i = 0; i < yv.length; i++) {
      if (pv[i] === yv[i]) correct++;
    }
    return correct / yv.length;
  }

  /** Sorted class labels seen during fit, or `undefined` before fitting. */
  get classes(): Tensor | undefined {
    if (!this.fitted) return undefined;
    return tensor(this.classLabels, { dtype: "int32" });
  }

  getParams(): Record<string, unknown> {
    return {
      nEstimators: this.estimators.length,
      passthrough: this.passthrough,
      cv: this.cv,
      stackMethod: this.stackMethod,
    };
  }

  /**
   * Set `passthrough`, `cv` or `stackMethod`. The base and final estimators are fixed at
   * construction.
   *
   * @throws {InvalidParameterError} If a value is invalid or the key is unknown
   */
  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "passthrough":
          this.passthrough = checkPassthrough(value);
          break;
        case "cv":
          this.cv = checkCv(value);
          break;
        case "stackMethod":
          this.stackMethod = checkStackMethod(value);
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }

  /** Create an unfitted copy with cloned base and final estimators and the same options. */
  clone(): StackingClassifier {
    return new StackingClassifier({
      estimators: this.estimators.map((e) => cloneEstimator(e, "StackingClassifier")),
      finalEstimator: cloneEstimator(this.finalEstimator, "StackingClassifier"),
      ...definedOptions<{ passthrough: boolean; cv?: number; stackMethod: StackingMethod }>({
        passthrough: this.passthrough,
        cv: this.cv,
        stackMethod: this.stackMethod,
      }),
    });
  }
}

/**
 * Stacking Regressor.
 *
 * Combines multiple regressors using a meta-learner (final estimator) that learns to
 * combine the base estimators' predictions.
 *
 * The meta-learner is trained on out-of-fold predictions: with `cv` folds (default 5, rows
 * assigned to folds in turn), each base estimator is cloned and fitted on `cv - 1` folds and
 * predicts the remaining one. The base estimators passed to the constructor are then fitted
 * on all the data (in place) and used at prediction time. Set `cv: 1` to train the final
 * estimator on in-sample predictions instead, which lets it trust base estimators that
 * memorized the training set.
 *
 * @example
 * ```ts
 * import { StackingRegressor, DecisionTreeRegressor, LinearRegression } from 'deepbox/ml';
 *
 * const reg = new StackingRegressor({
 *   estimators: [
 *     new DecisionTreeRegressor({ maxDepth: 3 }),
 *     new DecisionTreeRegressor({ maxDepth: 5 }),
 *   ],
 *   finalEstimator: new LinearRegression(),
 * });
 * reg.fit(X_train, y_train);
 * const predictions = reg.predict(X_test);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-ensemble | Deepbox Ensemble Methods}
 * @category Ensemble
 */
export class StackingRegressor implements Regressor {
  private readonly estimators: Regressor[];
  private readonly finalEstimator: Regressor;
  private passthrough: boolean;
  private cv: number | undefined;

  private nFeatures = 0;
  private fitted = false;

  /**
   * @param options.estimators - Base regressors (at least one). They are fitted in place.
   * @param options.finalEstimator - Meta-learner (default: `LinearRegression`)
   * @param options.passthrough - Also give the original features to the final estimator (default: false)
   * @param options.cv - Number of folds for the out-of-fold meta-features, an integer >= 1;
   *   1 trains the final estimator on in-sample predictions (default: 5)
   * @throws {InvalidParameterError} If there are no base estimators or an option is invalid
   */
  constructor(options: StackingRegressorOptions) {
    if (!options.estimators || options.estimators.length === 0) {
      throw new InvalidParameterError(
        "StackingRegressor requires at least one base estimator",
        "estimators",
        options.estimators
      );
    }
    this.estimators = [...options.estimators];
    this.finalEstimator = (options.finalEstimator ?? new LinearRegression()) as Regressor;
    this.passthrough = checkPassthrough(options.passthrough ?? false);
    this.cv = checkCv(options.cv);
  }

  /** Predictions of `est` on `X` as a flat array of length `n`. */
  private column(est: Regressor, X: Tensor, n: number): Float64Array {
    const pred = toFloat64View(est.predict(X));
    if (pred.length !== n) {
      throw new ShapeError(
        `base estimator predict returned ${pred.length} values for ${n} samples`
      );
    }
    return Float64Array.from(pred);
  }

  private metaFeatures(X: Tensor): Tensor {
    const n = X.shape[0] ?? 0;
    const d = X.shape[1] ?? 0;
    const blocks = this.estimators.map((est) => this.column(est, X, n));
    const widths = blocks.map(() => 1);
    const { data, width } = assembleMeta(
      blocks,
      widths,
      n,
      this.passthrough ? toFloat64View(X) : undefined,
      d
    );
    return floatMatrix(data, n, width);
  }

  /**
   * Fit the base estimators and the final estimator.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Targets of shape (n_samples,)
   * @returns this
   * @throws {ShapeError} If X is not 2D, y is not 1D or the sample counts differ
   * @throws {DataValidationError} If X or y contain NaN/Inf
   * @throws {InvalidParameterError} If `cv > 1` and a base estimator cannot be cloned
   */
  fit(X: Tensor, y: Tensor): this {
    this.fitted = false;
    validateFitInputs(X, y);
    const n = X.shape[0] ?? 0;
    const d = X.shape[1] ?? 0;
    const xv = toFloat64View(X);
    const yv = toFloat64View(y);
    this.nFeatures = d;

    // Base estimators used at prediction time are fitted on all the data.
    for (const est of this.estimators) est.fit(X, y);
    const blocks = this.estimators.map((est) => this.column(est, X, n));
    const widths = blocks.map(() => 1);

    const nFolds = Math.min(this.cv ?? DEFAULT_CV, n);
    if (nFolds >= 2) {
      const foldOf = assignFolds(n, nFolds, undefined);
      for (let fold = 0; fold < nFolds; fold++) {
        const { train, test } = splitFold(foldOf, fold);
        if (train.length === 0 || test.length === 0) continue;
        const trainX = floatMatrix(gatherRows(xv, d, train), train.length, d);
        const trainY = tensor(gatherValues(yv, train), { dtype: "float64" });
        const testX = floatMatrix(gatherRows(xv, d, test), test.length, d);
        this.estimators.forEach((est, m) => {
          const foldEstimator = cloneEstimator(est, "StackingRegressor");
          foldEstimator.fit(trainX, trainY);
          const pred = this.column(foldEstimator, testX, test.length);
          const out = blocks[m] as Float64Array;
          for (let r = 0; r < test.length; r++) out[test[r] as number] = pred[r] as number;
        });
      }
    }

    const meta = assembleMeta(blocks, widths, n, this.passthrough ? xv : undefined, d);
    this.finalEstimator.fit(floatMatrix(meta.data, n, meta.width), y);

    this.fitted = true;
    return this;
  }

  /**
   * Predict targets with the final estimator.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predictions of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has the wrong rank or feature count
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("StackingRegressor must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeatures, "StackingRegressor");
    return this.finalEstimator.predict(this.metaFeatures(X));
  }

  /**
   * Coefficient of determination R^2 on the given data.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True targets of shape (n_samples,)
   * @returns R^2 (1.0 is perfect, it can be negative)
   * @throws {ShapeError} If y is not 1D or its length differs from the number of rows in X
   * @throws {DataValidationError} If y is empty or contains NaN/Inf
   */
  score(X: Tensor, y: Tensor): number {
    if (y.ndim !== 1) {
      throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    }
    const predictions = this.predict(X);
    const yv = checkScoreTarget(y, predictions.size);
    const pv = toFloat64View(predictions);
    let mean = 0;
    for (let i = 0; i < yv.length; i++) mean += yv[i] as number;
    mean /= yv.length;
    let ssRes = 0;
    let ssTot = 0;
    for (let i = 0; i < yv.length; i++) {
      const yi = yv[i] as number;
      ssRes += (yi - (pv[i] as number)) ** 2;
      ssTot += (yi - mean) ** 2;
    }
    return ssTot === 0 ? (ssRes === 0 ? 1.0 : 0.0) : 1 - ssRes / ssTot;
  }

  getParams(): Record<string, unknown> {
    return {
      nEstimators: this.estimators.length,
      passthrough: this.passthrough,
      cv: this.cv,
    };
  }

  /**
   * Set `passthrough` or `cv`. The base and final estimators are fixed at construction.
   *
   * @throws {InvalidParameterError} If a value is invalid or the key is unknown
   */
  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "passthrough":
          this.passthrough = checkPassthrough(value);
          break;
        case "cv":
          this.cv = checkCv(value);
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }

  /** Create an unfitted copy with cloned base and final estimators and the same options. */
  clone(): StackingRegressor {
    return new StackingRegressor({
      estimators: this.estimators.map((e) => cloneEstimator(e, "StackingRegressor")),
      finalEstimator: cloneEstimator(this.finalEstimator, "StackingRegressor"),
      ...definedOptions<{ passthrough: boolean; cv?: number }>({
        passthrough: this.passthrough,
        cv: this.cv,
      }),
    });
  }
}
