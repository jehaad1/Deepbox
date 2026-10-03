/**
 * Nu-parameterized SVM variants and One-Class SVM.
 *
 * - NuSVC: classification SVM using nu instead of C
 * - NuSVR: regression SVM using nu instead of epsilon
 * - OneClassSVM: unsupervised outlier detection
 *
 * All three use the LIBSVM working-set SMO solver from `KernelSVM.ts`, so they agree with
 * scikit-learn's `NuSVC`, `NuSVR` and `OneClassSVM` up to the solver tolerance.
 *
 * @module ml/svm/NuSVM
 * @see {@link https://deepbox.dev/docs/ml-svm | Deepbox SVM}
 */

import { DataValidationError, InvalidParameterError, NotFittedError, ShapeError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import {
  toFloat64View,
  validateFitInputs,
  validatePredictInputs,
  validateUnsupervisedFitInputs,
} from "../_validation";
import type { Classifier, OutlierDetector, Regressor } from "../base";
import {
  accuracyOf,
  ClassificationQ,
  encodeLabels,
  expansionPredict,
  expansionTensors,
  type GammaOption,
  KernelExpansion,
  type KernelType,
  labelsToTensor,
  mergeParams,
  OvoModel,
  optionOr,
  ovoDecisionFunction,
  ovoPredict,
  ovoPredictProba,
  type PairSolution,
  parseC,
  parseCacheSize,
  parseCoef0,
  parseDegree,
  parseGamma,
  parseKernel,
  parseMaxIter,
  parseNu,
  parseTol,
  RegressionQ,
  r2Of,
  regressionCoefficients,
  resolveGamma,
  type SvmKernelParams,
  solveSmo,
  warnNotConverged,
} from "./KernelSVM";

type NuSvcConfig = {
  nu: number;
  kernel: KernelType;
  gamma: GammaOption;
  coef0: number;
  degree: number;
  maxIter: number;
  tol: number;
  cacheSize: number;
};

/** Constructor options of {@link NuSVC}. */
export type NuSVCOptions = {
  /**
   * Upper bound on the fraction of margin errors and lower bound on the fraction of support
   * vectors, in (0, 1] (default: 0.5). Some values are infeasible for unbalanced classes.
   */
  readonly nu?: number;
  /** Kernel function (default: "rbf"). */
  readonly kernel?: KernelType;
  /** Kernel coefficient of `rbf`, `poly` and `sigmoid` (default: "scale"). */
  readonly gamma?: GammaOption;
  /** Independent term of the `poly` and `sigmoid` kernels (default: 0). */
  readonly coef0?: number;
  /** Degree of the `poly` kernel, an integer >= 1 (default: 3). */
  readonly degree?: number;
  /**
   * Budget of SMO updates, expressed as passes over the data: the solver stops after at
   * most `maxIter * n` working-set updates per binary problem (default: 1000).
   */
  readonly maxIter?: number;
  /** Stopping tolerance on the maximal KKT violation (default: 1e-3). */
  readonly tol?: number;
  /** Size of the kernel row cache in megabytes (default: 200). */
  readonly cacheSize?: number;
};

const NU_SVC_KEYS = [
  "nu",
  "kernel",
  "gamma",
  "coef0",
  "degree",
  "maxIter",
  "tol",
  "cacheSize",
] as const;

function normalizeNuSvcConfig(o: Record<string, unknown>): NuSvcConfig {
  return {
    nu: optionOr(o, "nu", 0.5, parseNu),
    kernel: optionOr(o, "kernel", "rbf", parseKernel),
    gamma: optionOr<GammaOption>(o, "gamma", "scale", parseGamma),
    coef0: optionOr(o, "coef0", 0, parseCoef0),
    degree: optionOr(o, "degree", 3, parseDegree),
    maxIter: optionOr(o, "maxIter", 1000, parseMaxIter),
    tol: optionOr(o, "tol", 1e-3, parseTol),
    cacheSize: optionOr(o, "cacheSize", 200, parseCacheSize),
  };
}

/**
 * Nu-Support Vector Classification.
 *
 * Like {@link SVC} but `nu` in (0, 1] replaces `C`: it is an upper bound on the fraction
 * of margin errors and a lower bound on the fraction of support vectors. More than two
 * classes are handled with one-vs-one voting.
 *
 * `predictProba` returns a monotone squashing of the decision values, not calibrated
 * probabilities.
 *
 * @example
 * ```ts
 * import { NuSVC } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[0, 0], [1, 1], [1, 0], [0, 1]]);
 * const y = tensor([0, 0, 1, 1]);
 *
 * const clf = new NuSVC({ nu: 0.5 });
 * clf.fit(X, y);
 * const predictions = clf.predict(X);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-svm | Deepbox SVM}
 */
export class NuSVC implements Classifier {
  private cfg: NuSvcConfig;
  private model_: OvoModel | undefined;

  /**
   * @param options - Hyperparameters, see {@link NuSVCOptions}
   * @throws {InvalidParameterError} If an option is out of range
   */
  constructor(options: NuSVCOptions = {}) {
    this.cfg = normalizeNuSvcConfig(options as Record<string, unknown>);
  }

  private get fitted(): OvoModel {
    if (this.model_ === undefined) {
      throw new NotFittedError("NuSVC must be fitted before prediction");
    }
    return this.model_;
  }

  /**
   * Fit the classifier.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Class labels of shape (n_samples,), at least two distinct values
   * @returns this
   * @throws {ShapeError} If X is not 2D, y is not 1D or the sample counts differ
   * @throws {DataValidationError} If X or y contain NaN/Inf or the kernel overflows
   * @throws {InvalidParameterError} If y has fewer than 2 classes or `nu` is infeasible for
   *   a pair of classes
   */
  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const n = X.shape[0] ?? 0;
    const d = X.shape[1] ?? 0;
    const Xf = toFloat64View(X);
    const { labels, index } = encodeLabels(toFloat64View(y));
    if (labels.length < 2) {
      throw new InvalidParameterError("NuSVC requires at least 2 classes", "y", labels.length);
    }
    const { nu, tol, maxIter, cacheSize } = this.cfg;
    const kp: SvmKernelParams = {
      kernel: this.cfg.kernel,
      gamma: resolveGamma(this.cfg.gamma, Xf, d),
      coef0: this.cfg.coef0,
      degree: this.cfg.degree,
    };

    // nu must be feasible for every pair of classes before any solver runs.
    const counts = new Float64Array(labels.length);
    for (let i = 0; i < n; i++) counts[index[i] as number]!++;
    for (let a = 0; a < labels.length; a++) {
      for (let b = a + 1; b < labels.length; b++) {
        const na = counts[a] as number;
        const nb = counts[b] as number;
        if ((nu * (na + nb)) / 2 > Math.min(na, nb)) {
          throw new InvalidParameterError(
            `nu=${nu} is infeasible for classes ${labels[a]} and ${labels[b]} ` +
              `(${na} and ${nb} samples); nu * (n_a + n_b) / 2 must not exceed min(n_a, n_b)`,
            "nu",
            nu
          );
        }
      }
    }

    const { model, converged } = OvoModel.fit(
      Xf,
      n,
      d,
      labels,
      index,
      kp,
      (Xsub, m, ySub, _idx, a, b): PairSolution => {
        const Q = new ClassificationQ(Xsub, m, d, ySub, kp, cacheSize);
        const alpha = new Float64Array(m);
        let sumPos = (nu * m) / 2;
        let sumNeg = (nu * m) / 2;
        for (let t = 0; t < m; t++) {
          if (ySub[t] === 1) {
            alpha[t] = Math.min(1, sumPos);
            sumPos -= alpha[t] as number;
          } else {
            alpha[t] = Math.min(1, sumNeg);
            sumNeg -= alpha[t] as number;
          }
        }
        const res = solveSmo(
          Q,
          new Float64Array(m),
          ySub,
          alpha,
          new Float64Array(m).fill(1),
          tol,
          maxIter * m,
          true
        );
        // r is the margin scale of the solution. At the rounding level (relative to the size
        // of the kernel expansion) the classes cannot be separated at this nu, and dividing
        // by r would produce coefficients of order 1e16.
        const scale = res.r;
        let maxDiag = 0;
        for (let t = 0; t < m; t++) maxDiag = Math.max(maxDiag, Q.QD[t] as number);
        if (!(scale > 1e-10 * maxDiag * ((nu * m) / 2)) || !Number.isFinite(scale)) {
          throw new DataValidationError(
            `NuSVC found no margin between classes ${labels[a]} and ${labels[b]} at nu=${nu}; ` +
              "the classes overlap too much for this nu (increase nu) or the data contains " +
              "identical samples with different labels"
          );
        }
        const coef = new Float64Array(m);
        for (let t = 0; t < m; t++) {
          coef[t] = ((alpha[t] as number) * (ySub[t] as number)) / scale;
        }
        return {
          coef,
          rho: res.rho / scale,
          iterations: res.iterations,
          converged: res.converged,
        };
      }
    );
    if (!converged) warnNotConverged("NuSVC", maxIter);

    this.model_ = model;
    return this;
  }

  /**
   * Predict class labels by one-vs-one voting.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Labels of shape (n_samples,): int32 for integer classes, float64 otherwise
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predict(X: Tensor): Tensor {
    return ovoPredict(this.fitted, X, "NuSVC");
  }

  /**
   * Signed decision values.
   *
   * Two classes give shape (n_samples,) and a positive value means `classes[1]`. More classes
   * give shape (n_samples, n_classes) of one-vs-rest scores.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @throws {NotFittedError} If the model has not been fitted
   */
  decisionFunction(X: Tensor): Tensor {
    return ovoDecisionFunction(this.fitted, X, "NuSVC");
  }

  /**
   * Class scores squashed into rows that sum to one. These are not calibrated probabilities.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Shape (n_samples, n_classes)
   * @throws {NotFittedError} If the model has not been fitted
   */
  predictProba(X: Tensor): Tensor {
    return ovoPredictProba(this.fitted, X, "NuSVC");
  }

  /**
   * Mean accuracy on the given data.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True labels of shape (n_samples,)
   * @returns Accuracy in [0, 1]
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1-dimensional or the sample counts differ
   * @throws {DataValidationError} If y is empty or contains NaN/Inf
   */
  score(X: Tensor, y: Tensor): number {
    if (y.ndim !== 1) {
      throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    }
    return accuracyOf(this.predict(X), y);
  }

  /** Sorted class labels seen during `fit`, or `undefined` before fitting. */
  get classes(): Tensor | undefined {
    return this.model_ === undefined ? undefined : labelsToTensor(this.model_.labels);
  }

  /** Support vectors, shape (n_SV, n_features), grouped by class. */
  get supportVectors(): Tensor {
    return this.fitted.supportVectorsTensor();
  }

  /** Indices of the support vectors in the training data, shape (n_SV,), dtype int32. */
  get supportIndices(): Tensor {
    return tensor(Int32Array.from(this.fitted.supportIndices), { dtype: "int32" });
  }

  /** Number of support vectors of each class, shape (n_classes,), dtype int32. */
  get nSupport(): Tensor {
    return tensor(Int32Array.from(this.fitted.nSupportPerClass), { dtype: "int32" });
  }

  /** Dual coefficients `alpha_i * y_i`, shape (n_classes - 1, n_SV), scikit-learn layout. */
  get dualCoef(): Tensor {
    return this.fitted.dualCoefTensor();
  }

  /** Decision-function offsets, one per class pair, scikit-learn sign convention. */
  get intercept(): Tensor {
    return this.fitted.interceptTensor();
  }

  /**
   * Get hyperparameters.
   *
   * @returns Object that can be passed back to the constructor
   */
  getParams(): Record<string, unknown> {
    return { ...this.cfg };
  }

  /**
   * Set hyperparameters. The call is atomic: if any value is invalid nothing changes.
   *
   * @param params - Parameters to set
   * @throws {InvalidParameterError} If a parameter is unknown or its value is invalid
   */
  setParams(params: Record<string, unknown>): this {
    this.cfg = normalizeNuSvcConfig(mergeParams(this.getParams(), params, NU_SVC_KEYS));
    return this;
  }

  /** Create an unfitted copy with the same hyperparameters. */
  clone(): NuSVC {
    return new NuSVC(this.getParams() as NuSVCOptions);
  }
}

// ---------------------------------------------------------------------------
// NuSVR
// ---------------------------------------------------------------------------

/** Constructor options of {@link NuSVR}. */
export type NuSVROptions = {
  /**
   * Upper bound on the fraction of training errors and lower bound on the fraction of
   * support vectors, in (0, 1] (default: 0.5).
   */
  readonly nu?: number;
  /** Penalty of errors, must be positive (default: 1.0). */
  readonly C?: number;
  /** Kernel function (default: "rbf"). */
  readonly kernel?: KernelType;
  /** Kernel coefficient of `rbf`, `poly` and `sigmoid` (default: "scale"). */
  readonly gamma?: GammaOption;
  /** Independent term of the `poly` and `sigmoid` kernels (default: 0). */
  readonly coef0?: number;
  /** Degree of the `poly` kernel, an integer >= 1 (default: 3). */
  readonly degree?: number;
  /** Budget of SMO updates in passes over the 2 * n dual variables (default: 1000). */
  readonly maxIter?: number;
  /** Stopping tolerance on the maximal KKT violation (default: 1e-3). */
  readonly tol?: number;
  /** Size of the kernel row cache in megabytes (default: 200). */
  readonly cacheSize?: number;
};

type NuSvrConfig = NuSvcConfig & { C: number };

const NU_SVR_KEYS = [...NU_SVC_KEYS, "C"] as const;

function normalizeNuSvrConfig(o: Record<string, unknown>): NuSvrConfig {
  return { ...normalizeNuSvcConfig(o), C: optionOr(o, "C", 1.0, parseC) };
}

/**
 * Nu-Support Vector Regression.
 *
 * Like {@link SVR} but `nu` in (0, 1] controls the fraction of support vectors and the
 * width of the insensitive tube is found by the solver instead of being given.
 *
 * @example
 * ```ts
 * import { NuSVR } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1], [2], [3], [4], [5]]);
 * const y = tensor([1.2, 2.1, 2.9, 4.0, 5.1]);
 *
 * const svr = new NuSVR({ nu: 0.5 });
 * svr.fit(X, y);
 * const predictions = svr.predict(X);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-svm | Deepbox SVM}
 */
export class NuSVR implements Regressor {
  private cfg: NuSvrConfig;
  private model_: KernelExpansion | undefined;

  /**
   * @param options - Hyperparameters, see {@link NuSVROptions}
   * @throws {InvalidParameterError} If an option is out of range
   */
  constructor(options: NuSVROptions = {}) {
    this.cfg = normalizeNuSvrConfig(options as Record<string, unknown>);
  }

  private get fitted(): KernelExpansion {
    if (this.model_ === undefined) {
      throw new NotFittedError("NuSVR must be fitted before prediction");
    }
    return this.model_;
  }

  /**
   * Fit the regressor.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Targets of shape (n_samples,)
   * @returns this
   * @throws {ShapeError} If X is not 2D, y is not 1D or the sample counts differ
   * @throws {DataValidationError} If X or y contain NaN/Inf or the kernel overflows
   */
  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const n = X.shape[0] ?? 0;
    const d = X.shape[1] ?? 0;
    const Xf = toFloat64View(X);
    const yv = toFloat64View(y);
    const { nu, C: penalty } = this.cfg;
    const kp: SvmKernelParams = {
      kernel: this.cfg.kernel,
      gamma: resolveGamma(this.cfg.gamma, Xf, d),
      coef0: this.cfg.coef0,
      degree: this.cfg.degree,
    };

    const Q = new RegressionQ(Xf, n, d, kp, this.cfg.cacheSize);
    const alpha = new Float64Array(2 * n);
    const p = new Float64Array(2 * n);
    const sign = new Int8Array(2 * n);
    let sum = (penalty * nu * n) / 2;
    for (let i = 0; i < n; i++) {
      const a = Math.min(sum, penalty);
      alpha[i] = a;
      alpha[i + n] = a;
      sum -= a;
      p[i] = -(yv[i] as number);
      p[i + n] = yv[i] as number;
      sign[i] = 1;
      sign[i + n] = -1;
    }
    const res = solveSmo(
      Q,
      p,
      sign,
      alpha,
      new Float64Array(2 * n).fill(penalty),
      this.cfg.tol,
      this.cfg.maxIter * 2 * n,
      true
    );
    if (!res.converged) warnNotConverged("NuSVR", this.cfg.maxIter);

    this.model_ = KernelExpansion.fromCoefficients(
      Xf,
      n,
      d,
      regressionCoefficients(res.alpha, n),
      res.rho,
      kp
    );
    return this;
  }

  /**
   * Predict target values.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predictions of shape (n_samples,), dtype float64
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predict(X: Tensor): Tensor {
    return expansionPredict(this.fitted, X, "NuSVR");
  }

  /**
   * Coefficient of determination R^2 on the given data.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True targets of shape (n_samples,)
   * @returns R^2 (1 is perfect, can be negative); a constant y scores 1 or 0
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1-dimensional or the sample counts differ
   * @throws {DataValidationError} If y is empty or contains NaN/Inf
   */
  score(X: Tensor, y: Tensor): number {
    if (y.ndim !== 1) {
      throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    }
    return r2Of(this.predict(X), y);
  }

  /** Support vectors, shape (n_SV, n_features). */
  get supportVectors(): Tensor {
    return expansionTensors(this.fitted).supportVectors;
  }

  /** Indices of the support vectors in the training data, shape (n_SV,), dtype int32. */
  get supportIndices(): Tensor {
    return expansionTensors(this.fitted).supportIndices;
  }

  /** Dual coefficients `alpha_i - alpha_i*`, shape (1, n_SV). */
  get dualCoef(): Tensor {
    return expansionTensors(this.fitted).dualCoef;
  }

  /** Intercept of the decision function, shape (1,). */
  get intercept(): Tensor {
    return expansionTensors(this.fitted).intercept;
  }

  /**
   * Get hyperparameters.
   *
   * @returns Object that can be passed back to the constructor
   */
  getParams(): Record<string, unknown> {
    return { ...this.cfg };
  }

  /**
   * Set hyperparameters. The call is atomic: if any value is invalid nothing changes.
   *
   * @param params - Parameters to set
   * @throws {InvalidParameterError} If a parameter is unknown or its value is invalid
   */
  setParams(params: Record<string, unknown>): this {
    this.cfg = normalizeNuSvrConfig(mergeParams(this.getParams(), params, NU_SVR_KEYS));
    return this;
  }

  /** Create an unfitted copy with the same hyperparameters. */
  clone(): NuSVR {
    return new NuSVR(this.getParams() as NuSVROptions);
  }
}

// ---------------------------------------------------------------------------
// OneClassSVM
// ---------------------------------------------------------------------------

/** Constructor options of {@link OneClassSVM}. */
export type OneClassSVMOptions = NuSVCOptions;

/**
 * One-Class SVM for unsupervised outlier detection.
 *
 * Learns a boundary that separates the data from the origin in kernel feature space
 * (Scholkopf et al.). `nu` in (0, 1] bounds the fraction of training errors from above and
 * the fraction of support vectors from below.
 *
 * `scoreSamples` returns the raw kernel expansion `sum_i alpha_i K(sv_i, x)`,
 * `decisionFunction` returns `scoreSamples(X) - offset`, and `predict` returns +1 where the
 * decision function is `>= 0` and -1 elsewhere.
 *
 * @example
 * ```ts
 * import { OneClassSVM } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 1], [1, 2], [2, 1], [2, 2], [10, 10]]);
 *
 * const ocsvm = new OneClassSVM({ nu: 0.2 });
 * ocsvm.fit(X);
 * const labels = ocsvm.predict(X); // +1 inlier, -1 outlier
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-svm | Deepbox SVM}
 */
export class OneClassSVM implements OutlierDetector {
  private cfg: NuSvcConfig;
  private model_: KernelExpansion | undefined;

  /**
   * @param options - Hyperparameters, see {@link OneClassSVMOptions}
   * @throws {InvalidParameterError} If an option is out of range
   */
  constructor(options: OneClassSVMOptions = {}) {
    this.cfg = normalizeNuSvcConfig(options as Record<string, unknown>);
  }

  private get fitted(): KernelExpansion {
    if (this.model_ === undefined) {
      throw new NotFittedError("OneClassSVM must be fitted before prediction");
    }
    return this.model_;
  }

  /**
   * Fit the detector on inlier-dominated data.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param _y - Ignored, present for API consistency
   * @returns this
   * @throws {ShapeError} If X is not 2D
   * @throws {DataValidationError} If X is empty, contains NaN/Inf, or the kernel overflows
   */
  fit(X: Tensor, _y?: Tensor): this {
    validateUnsupervisedFitInputs(X);
    const n = X.shape[0] ?? 0;
    const d = X.shape[1] ?? 0;
    const Xf = toFloat64View(X);
    const { nu } = this.cfg;
    const kp: SvmKernelParams = {
      kernel: this.cfg.kernel,
      gamma: resolveGamma(this.cfg.gamma, Xf, d),
      coef0: this.cfg.coef0,
      degree: this.cfg.degree,
    };

    // LIBSVM scaling: 0 <= alpha_i <= 1 and sum(alpha) = nu * n.
    const alpha = new Float64Array(n);
    const full = Math.floor(nu * n);
    for (let i = 0; i < full; i++) alpha[i] = 1;
    if (full < n) alpha[full] = nu * n - full;

    const Q = new ClassificationQ(Xf, n, d, new Int8Array(n).fill(1), kp, this.cfg.cacheSize);
    const res = solveSmo(
      Q,
      new Float64Array(n),
      new Int8Array(n).fill(1),
      alpha,
      new Float64Array(n).fill(1),
      this.cfg.tol,
      this.cfg.maxIter * n,
      false
    );
    if (!res.converged) warnNotConverged("OneClassSVM", this.cfg.maxIter);

    this.model_ = KernelExpansion.fromCoefficients(Xf, n, d, res.alpha, res.rho, kp);
    return this;
  }

  private rawScores(X: Tensor): Float64Array {
    const model = this.fitted;
    validatePredictInputs(X, model.nFeatures, "OneClassSVM");
    return model.raw(toFloat64View(X), X.shape[0] ?? 0);
  }

  /**
   * Predict whether samples are inliers.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns +1 for inliers and -1 for outliers, shape (n_samples,), dtype int32
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predict(X: Tensor): Tensor {
    const scores = this.rawScores(X);
    const rho = this.fitted.rho;
    return tensor(
      Int32Array.from(scores, (s) => (s - rho >= 0 ? 1 : -1)),
      { dtype: "int32" }
    );
  }

  /**
   * Fit on `X`, then predict labels for the same samples.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param _y - Ignored, present for API consistency
   * @returns +1 for inliers and -1 for outliers
   */
  fitPredict(X: Tensor, _y?: Tensor): Tensor {
    return this.fit(X).predict(X);
  }

  /**
   * Raw scores `sum_i alpha_i K(sv_i, x)`: lower values are more abnormal.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Scores of shape (n_samples,), dtype float64
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  scoreSamples(X: Tensor): Tensor {
    return tensor(this.rawScores(X), { dtype: "float64" });
  }

  /**
   * Signed distance to the decision boundary, `scoreSamples(X) - offset`: negative values are
   * outliers.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Decision values of shape (n_samples,), dtype float64
   * @throws {NotFittedError} If the model has not been fitted
   */
  decisionFunction(X: Tensor): Tensor {
    const scores = this.rawScores(X);
    const rho = this.fitted.rho;
    for (let i = 0; i < scores.length; i++) scores[i] = (scores[i] as number) - rho;
    return tensor(scores, { dtype: "float64" });
  }

  /**
   * Decision threshold in `scoreSamples` units: samples scoring below it are outliers.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get offset(): number {
    return this.fitted.rho;
  }

  /** Support vectors, shape (n_SV, n_features). */
  get supportVectors(): Tensor {
    return expansionTensors(this.fitted).supportVectors;
  }

  /** Indices of the support vectors in the training data, shape (n_SV,), dtype int32. */
  get supportIndices(): Tensor {
    return expansionTensors(this.fitted).supportIndices;
  }

  /** Dual coefficients `alpha_i` (sum to `nu * n_samples`), shape (1, n_SV). */
  get dualCoef(): Tensor {
    return expansionTensors(this.fitted).dualCoef;
  }

  /**
   * Get hyperparameters.
   *
   * @returns Object that can be passed back to the constructor
   */
  getParams(): Record<string, unknown> {
    return { ...this.cfg };
  }

  /**
   * Set hyperparameters. The call is atomic: if any value is invalid nothing changes.
   *
   * @param params - Parameters to set
   * @throws {InvalidParameterError} If a parameter is unknown or its value is invalid
   */
  setParams(params: Record<string, unknown>): this {
    this.cfg = normalizeNuSvcConfig(mergeParams(this.getParams(), params, NU_SVC_KEYS));
    return this;
  }

  /** Create an unfitted copy with the same hyperparameters. */
  clone(): OneClassSVM {
    return new OneClassSVM(this.getParams() as OneClassSVMOptions);
  }
}
