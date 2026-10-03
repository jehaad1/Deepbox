/**
 * Kernel ridge regression.
 *
 * @module ml/linear/KernelRidge
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { DataValidationError, InvalidParameterError, NotFittedError, warn } from "../../core";
import { lstsq } from "../../linalg";
import { solve } from "../../linalg/solvers/solve";
import { type Tensor, tensor } from "../../ndarray";
import { toFloat64View, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Regressor } from "../base";
import { r2ScoreOf } from "./LinearRegression";

/** Kernel functions supported by {@link KernelRidge}. `"poly"` is an alias of `"polynomial"`. */
export type KernelRidgeKernel = "linear" | "rbf" | "polynomial" | "poly" | "sigmoid" | "laplacian";

/** Constructor options of {@link KernelRidge}. */
export type KernelRidgeOptions = {
  /** Strength of the L2 penalty, must be >= 0 (default: 1.0). */
  readonly alpha?: number;
  /** Kernel function (default: "rbf"). */
  readonly kernel?: KernelRidgeKernel;
  /**
   * Kernel coefficient of `rbf`, `laplacian`, `polynomial` and `sigmoid`, must be >= 0
   * (default: 1.0). Unlike scikit-learn, which uses `1 / n_features`, this is a fixed number.
   */
  readonly gamma?: number;
  /** Degree of the polynomial kernel, must be >= 0 (default: 3). */
  readonly degree?: number;
  /** Independent term of the polynomial and sigmoid kernels (default: 1). */
  readonly coef0?: number;
};

type Options = {
  alpha: number;
  kernel: KernelRidgeKernel;
  gamma: number;
  degree: number;
  coef0: number;
};

const KERNELS: readonly string[] = ["linear", "rbf", "polynomial", "poly", "sigmoid", "laplacian"];
const OPTION_KEYS: readonly string[] = ["alpha", "kernel", "gamma", "degree", "coef0"];

function resolveOptions(options: KernelRidgeOptions): Options {
  const resolved: Options = {
    alpha: options.alpha ?? 1.0,
    kernel: options.kernel ?? "rbf",
    gamma: options.gamma ?? 1.0,
    degree: options.degree ?? 3,
    coef0: options.coef0 ?? 1,
  };
  if (!(resolved.alpha >= 0) || !Number.isFinite(resolved.alpha)) {
    throw new InvalidParameterError("alpha must be >= 0", "alpha", resolved.alpha);
  }
  if (!KERNELS.includes(resolved.kernel)) {
    throw new InvalidParameterError(
      `kernel must be one of [${KERNELS.join(", ")}]; received ${String(resolved.kernel)}`,
      "kernel",
      resolved.kernel
    );
  }
  if (!(resolved.gamma >= 0) || !Number.isFinite(resolved.gamma)) {
    throw new InvalidParameterError("gamma must be >= 0", "gamma", resolved.gamma);
  }
  if (!(resolved.degree >= 0) || !Number.isFinite(resolved.degree)) {
    throw new InvalidParameterError("degree must be >= 0", "degree", resolved.degree);
  }
  if (!Number.isFinite(resolved.coef0)) {
    throw new InvalidParameterError("coef0 must be a finite number", "coef0", resolved.coef0);
  }
  return resolved;
}

type KernelFn = (a: Float64Array, aOff: number, b: Float64Array, bOff: number, d: number) => number;

function makeKernel(o: Options): KernelFn {
  const { gamma, degree, coef0 } = o;
  const dot = (a: Float64Array, aOff: number, b: Float64Array, bOff: number, d: number): number => {
    let s = 0;
    for (let f = 0; f < d; f++) s += (a[aOff + f] as number) * (b[bOff + f] as number);
    return s;
  };
  switch (o.kernel) {
    case "linear":
      return dot;
    case "rbf":
      return (a, aOff, b, bOff, d) => {
        let sq = 0;
        for (let f = 0; f < d; f++) {
          const diff = (a[aOff + f] as number) - (b[bOff + f] as number);
          sq += diff * diff;
        }
        return Math.exp(-gamma * sq);
      };
    case "laplacian":
      return (a, aOff, b, bOff, d) => {
        let l1 = 0;
        for (let f = 0; f < d; f++)
          l1 += Math.abs((a[aOff + f] as number) - (b[bOff + f] as number));
        return Math.exp(-gamma * l1);
      };
    case "sigmoid":
      return (a, aOff, b, bOff, d) => Math.tanh(gamma * dot(a, aOff, b, bOff, d) + coef0);
    default:
      return (a, aOff, b, bOff, d) => (gamma * dot(a, aOff, b, bOff, d) + coef0) ** degree;
  }
}

/**
 * Kernel Ridge Regression.
 *
 * Combines Ridge Regression (L2 penalty) with the kernel trick,
 * learning a non-linear function in the feature space induced by the kernel.
 * There is no intercept term; center `y` beforehand if you need one.
 *
 * Solves: (K + alpha * I) * dual_coef = y
 *
 * and predicts `K(X_test, X_train) @ dual_coef`. The defaults differ from scikit-learn's
 * (`kernel: "rbf"` and `gamma: 1.0` here, `"linear"` and `1 / n_features` there), so pass them
 * explicitly when comparing the two.
 *
 * @example
 * ```ts
 * import { KernelRidge } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X_train = tensor([[0], [1], [2], [3]]);
 * const y_train = tensor([0, 1, 4, 9]);
 * const model = new KernelRidge({ alpha: 1.0, kernel: 'rbf', gamma: 0.1 });
 * model.fit(X_train, y_train);
 * const predictions = model.predict(tensor([[1.5]]));
 * ```
 *
 * @category Linear Models
 * @implements {Regressor}
 */
export class KernelRidge implements Regressor {
  private options: Options;

  private dualCoef_?: Float64Array;
  /** Hyper-parameters in effect when the model was fitted; `predict` always uses these. */
  private fitOptions_?: Options;
  private XFit_?: Float64Array;
  private nSamplesFit_?: number;
  private nFeaturesFit_?: number;
  private fitted = false;

  /**
   * Create a new kernel ridge model.
   *
   * @param options - Configuration options, see {@link KernelRidgeOptions}
   * @throws {InvalidParameterError} If an option is invalid
   */
  constructor(options: KernelRidgeOptions = {}) {
    this.options = resolveOptions(options);
  }

  /**
   * Fit the model by solving `(K + alpha I) c = y`.
   *
   * When the system is singular (for example `alpha = 0` with a rank-deficient kernel matrix) the
   * minimum-norm least squares solution is used and a warning is issued.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Target values of shape (n_samples,)
   * @returns this - The fitted estimator
   * @throws {ShapeError} If X is not 2D, y is not 1D, or their sample counts differ
   * @throws {DataValidationError} If X or y are empty or contain NaN/Inf
   */
  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const n = X.shape[0] ?? 0;
    const d = X.shape[1] ?? 0;
    const xFit = Float64Array.from(toFloat64View(X));
    const yv = toFloat64View(y);

    const kernel = makeKernel(this.options);
    const K = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        const v = kernel(xFit, i * d, xFit, j * d, d);
        K[i * n + j] = v;
        K[j * n + i] = v;
      }
      K[i * n + i] = (K[i * n + i] as number) + this.options.alpha;
    }

    const KTensor = tensor(K, { dtype: "float64" }).reshape([n, n]);
    const yTensor = tensor(yv.slice(), { dtype: "float64" });
    let solution: Tensor;
    try {
      solution = solve(KTensor, yTensor);
    } catch (err) {
      if (!(err instanceof DataValidationError)) throw err;
      warn(
        "KernelRidge: the kernel system is singular; using the minimum-norm least squares solution. Increase alpha to regularize.",
        "UserWarning",
        "KernelRidge"
      );
      solution = lstsq(KTensor, yTensor).x;
    }

    this.fitOptions_ = { ...this.options };
    this.dualCoef_ = Float64Array.from(toFloat64View(solution));
    this.XFit_ = xFit;
    this.nSamplesFit_ = n;
    this.nFeaturesFit_ = d;
    this.fitted = true;
    return this;
  }

  /**
   * Predict target values.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predictions of shape (n_samples,) as a float64 tensor
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X is not 2D or has the wrong number of features
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.dualCoef_ || !this.XFit_ || !this.fitOptions_) {
      throw new NotFittedError("KernelRidge must be fitted before predict");
    }
    const nTrain = this.nSamplesFit_ ?? 0;
    const d = this.nFeaturesFit_ ?? 0;
    validatePredictInputs(X, d, "KernelRidge");

    const n = X.shape[0] ?? 0;
    const xv = toFloat64View(X);
    const kernel = makeKernel(this.fitOptions_);
    const dual = this.dualCoef_;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < nTrain; j++) {
        s += kernel(xv, i * d, this.XFit_, j * d, d) * (dual[j] as number);
      }
      out[i] = s;
    }
    return tensor(out, { dtype: "float64" });
  }

  /**
   * Coefficient of determination R² of the predictions.
   *
   * A constant `y` scores 1 when predicted exactly and 0 otherwise.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True targets of shape (n_samples,)
   * @returns R² score (1 is perfect, can be negative)
   * @throws {NotFittedError} If the model has not been fitted
   */
  score(X: Tensor, y: Tensor): number {
    if (!this.fitted) {
      throw new NotFittedError("KernelRidge must be fitted before scoring");
    }
    return r2ScoreOf(y, () => this.predict(X));
  }

  /** Dual coefficients (one per training sample). */
  get dualCoef(): Float64Array {
    if (!this.fitted || !this.dualCoef_) {
      throw new NotFittedError("KernelRidge must be fitted to access dualCoef");
    }
    return this.dualCoef_;
  }

  /** Number of features seen during `fit`. */
  get nFeaturesIn(): number {
    if (!this.fitted) {
      throw new NotFittedError("KernelRidge must be fitted to access nFeaturesIn");
    }
    return this.nFeaturesFit_ ?? 0;
  }

  /** Hyper-parameters of this estimator. */
  getParams(): Record<string, unknown> {
    return { ...this.options };
  }

  /**
   * Set hyper-parameters. All values are validated before any is applied; `undefined` values are
   * ignored. A fitted model keeps predicting with the parameters it was fitted with until `fit`
   * is called again.
   *
   * @param params - Parameters to change (`alpha`, `kernel`, `gamma`, `degree`, `coef0`)
   * @returns this
   * @throws {InvalidParameterError} If a name is unknown or a value is invalid
   */
  setParams(params: Record<string, unknown>): this {
    const next: Record<string, unknown> = { ...this.options };
    for (const [key, value] of Object.entries(params)) {
      if (!OPTION_KEYS.includes(key)) {
        throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
      // `undefined` leaves the current value in place.
      if (value === undefined) continue;
      if (key !== "kernel" && typeof value !== "number") {
        throw new InvalidParameterError(`${key} must be a number`, key, value);
      }
      next[key] = value;
    }
    this.options = resolveOptions(next as KernelRidgeOptions);
    return this;
  }

  /** Create an unfitted copy with the same hyper-parameters. */
  clone(): KernelRidge {
    return new KernelRidge(this.getParams() as KernelRidgeOptions);
  }
}
