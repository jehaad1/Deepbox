/**
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { DataValidationError, InvalidParameterError, NotFittedError, warn } from "../../core";
import { svd } from "../../linalg/decomposition/svd";
import { type Tensor, tensor } from "../../ndarray";
import { toFloat64View, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Regressor } from "../base";
import { r2ScoreOf } from "./LinearRegression";

type RidgeSolver = "auto" | "svd" | "cholesky" | "lsqr" | "sag";

const RIDGE_SOLVERS: readonly RidgeSolver[] = ["auto", "svd", "cholesky", "lsqr", "sag"];

const SINGULAR_MESSAGE = "Matrix is singular or ill-conditioned";

/** Relative pivot size below which `X^T X` (alpha = 0) is treated as singular. */
const UNREGULARIZED_PIVOT_TOL = 1e-12;

/**
 * Solve the symmetric positive definite system `A x = b` with a Cholesky
 * factorization. `A` is read, not modified.
 *
 * @param relativeTol - Smallest accepted pivot relative to `max(diag(A))`; at least `n * eps`.
 * @returns The solution, or `undefined` when `A` is not numerically positive
 * definite (a pivot falls below `max(n * eps, relativeTol) * max(diag(A))`).
 */
function choleskySolve(
  A: Float64Array,
  b: Float64Array,
  n: number,
  relativeTol = 0
): Float64Array | undefined {
  let maxDiag = 0;
  for (let i = 0; i < n; i++) {
    const d = A[i * n + i] as number;
    if (d > maxDiag) maxDiag = d;
  }
  if (!(maxDiag > 0) || !Number.isFinite(maxDiag)) return undefined;
  const pivotTol = Math.max(n * Number.EPSILON, relativeTol) * maxDiag;

  const L = new Float64Array(n * n);
  for (let j = 0; j < n; j++) {
    const jRow = j * n;
    let diag = A[jRow + j] as number;
    for (let k = 0; k < j; k++) {
      const v = L[jRow + k] as number;
      diag -= v * v;
    }
    if (!(diag > pivotTol)) return undefined;
    const ljj = Math.sqrt(diag);
    L[jRow + j] = ljj;
    for (let i = j + 1; i < n; i++) {
      const iRow = i * n;
      let sum = A[iRow + j] as number;
      for (let k = 0; k < j; k++) {
        sum -= (L[iRow + k] as number) * (L[jRow + k] as number);
      }
      L[iRow + j] = sum / ljj;
    }
  }

  // Forward substitution: L z = b
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let sum = b[i] as number;
    for (let k = 0; k < i; k++) sum -= (L[i * n + k] as number) * (x[k] as number);
    x[i] = sum / (L[i * n + i] as number);
  }
  // Back substitution: L^T x = z
  for (let i = n - 1; i >= 0; i--) {
    let sum = x[i] as number;
    for (let k = i + 1; k < n; k++) sum -= (L[k * n + i] as number) * (x[k] as number);
    x[i] = sum / (L[i * n + i] as number);
  }
  return x;
}

/**
 * Solve `A x = b` by Gaussian elimination with partial pivoting.
 *
 * @throws {DataValidationError} If `A` is singular to working precision
 */
function gaussianSolve(A: Float64Array, b: Float64Array, n: number): Float64Array {
  const w = n + 1;
  const aug = new Float64Array(n * w);
  let maxAbs = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const v = A[i * n + j] as number;
      aug[i * w + j] = v;
      const a = Math.abs(v);
      if (a > maxAbs) maxAbs = a;
    }
    aug[i * w + n] = b[i] as number;
  }
  if (maxAbs === 0 || !Number.isFinite(maxAbs)) {
    throw new DataValidationError(SINGULAR_MESSAGE);
  }
  const tol = Number.EPSILON * n * maxAbs;

  for (let col = 0; col < n; col++) {
    let maxRow = col;
    let maxVal = Math.abs(aug[col * w + col] as number);
    for (let r = col + 1; r < n; r++) {
      const v = Math.abs(aug[r * w + col] as number);
      if (v > maxVal) {
        maxVal = v;
        maxRow = r;
      }
    }
    if (maxRow !== col) {
      for (let j = 0; j < w; j++) {
        const tmp = aug[col * w + j] as number;
        aug[col * w + j] = aug[maxRow * w + j] as number;
        aug[maxRow * w + j] = tmp;
      }
    }
    const pivot = aug[col * w + col] as number;
    if (!Number.isFinite(pivot) || Math.abs(pivot) <= tol) {
      throw new DataValidationError(SINGULAR_MESSAGE);
    }
    for (let r = col + 1; r < n; r++) {
      const c = (aug[r * w + col] as number) / pivot;
      if (c === 0) continue;
      for (let j = col; j < w; j++) {
        aug[r * w + j] = (aug[r * w + j] as number) - c * (aug[col * w + j] as number);
      }
    }
  }

  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = aug[i * w + n] as number;
    for (let j = i + 1; j < n; j++) sum -= (aug[i * w + j] as number) * (x[j] as number);
    x[i] = sum / (aug[i * w + i] as number);
  }
  return x;
}

/**
 * Conjugate gradient for the symmetric positive definite system `A x = b`.
 *
 * Stops when the residual norm drops below `tol * ||b||`.
 */
function conjugateGradient(
  A: Float64Array,
  b: Float64Array,
  n: number,
  maxIter: number,
  tol: number
): { x: Float64Array; nIter: number; converged: boolean } {
  const x = new Float64Array(n);
  const r = Float64Array.from(b);
  let rsOld = 0;
  for (let i = 0; i < n; i++) rsOld += (r[i] as number) * (r[i] as number);
  if (rsOld === 0) return { x, nIter: 0, converged: true };

  const target = tol * tol * rsOld;
  const p = Float64Array.from(r);
  const Ap = new Float64Array(n);
  let nIter = 0;
  let converged = false;

  for (let iter = 0; iter < maxIter; iter++) {
    let denom = 0;
    for (let i = 0; i < n; i++) {
      let sum = 0;
      const row = i * n;
      for (let j = 0; j < n; j++) sum += (A[row + j] as number) * (p[j] as number);
      Ap[i] = sum;
      denom += (p[i] as number) * sum;
    }
    if (!Number.isFinite(denom) || denom <= 0) {
      throw new DataValidationError(
        "Conjugate gradient failed: the system is not positive definite or is non-finite"
      );
    }
    const step = rsOld / denom;
    let rsNew = 0;
    for (let i = 0; i < n; i++) {
      x[i] = (x[i] as number) + step * (p[i] as number);
      const ri = (r[i] as number) - step * (Ap[i] as number);
      r[i] = ri;
      rsNew += ri * ri;
    }
    nIter = iter + 1;
    if (rsNew <= target || rsNew === 0) {
      converged = true;
      break;
    }
    const beta = rsNew / rsOld;
    for (let i = 0; i < n; i++) p[i] = (r[i] as number) + beta * (p[i] as number);
    rsOld = rsNew;
  }
  return { x, nIter, converged };
}

/**
 * Accelerated full-batch gradient descent on the averaged ridge objective
 * `(1/2m) ||Xw - y||^2 + (alpha/2m) ||w||^2`.
 *
 * The minimizer is the same as the closed-form ridge solution. The step size
 * is `1 / L` with `L = max_i ||x_i||^2 + alpha / m`, which bounds the largest
 * eigenvalue of the averaged Hessian, so the iteration cannot diverge.
 * Iteration stops when the largest gradient entry falls below `tol` times the
 * largest entry of the initial gradient.
 */
function accelerated(
  Xc: Float64Array,
  yc: Float64Array,
  m: number,
  n: number,
  alpha: number,
  maxIter: number,
  tol: number
): { x: Float64Array; nIter: number; converged: boolean } {
  const w = new Float64Array(n);
  let maxNormSq = 0;
  for (let i = 0; i < m; i++) {
    let normSq = 0;
    const base = i * n;
    for (let j = 0; j < n; j++) normSq += (Xc[base + j] as number) ** 2;
    if (normSq > maxNormSq) maxNormSq = normSq;
  }
  const L = maxNormSq + alpha / m;
  if (!(L > 0)) return { x: w, nIter: 0, converged: true };
  const step = 1 / L;

  const z = new Float64Array(n); // look-ahead point
  const grad = new Float64Array(n);
  let momentum = 1;
  let nIter = 0;
  let gradScale = 0;
  let converged = false;

  for (let iter = 0; iter < maxIter; iter++) {
    grad.fill(0);
    for (let i = 0; i < m; i++) {
      const base = i * n;
      let dot = 0;
      for (let j = 0; j < n; j++) dot += (z[j] as number) * (Xc[base + j] as number);
      const residual = dot - (yc[i] as number);
      for (let j = 0; j < n; j++) {
        grad[j] = (grad[j] as number) + residual * (Xc[base + j] as number);
      }
    }
    let maxGrad = 0;
    for (let j = 0; j < n; j++) {
      const g = ((grad[j] as number) + alpha * (z[j] as number)) / m;
      grad[j] = g;
      if (Math.abs(g) > maxGrad) maxGrad = Math.abs(g);
    }
    if (iter === 0) gradScale = maxGrad;
    nIter = iter + 1;
    if (maxGrad <= tol * gradScale) {
      // z is the current best iterate once the gradient there is small enough.
      w.set(z);
      converged = true;
      break;
    }

    const nextMomentum = (1 + Math.sqrt(1 + 4 * momentum * momentum)) / 2;
    const beta = (momentum - 1) / nextMomentum;
    for (let j = 0; j < n; j++) {
      const wNew = (z[j] as number) - step * (grad[j] as number);
      z[j] = wNew + beta * (wNew - (w[j] as number));
      w[j] = wNew;
    }
    momentum = nextMomentum;
  }

  for (let j = 0; j < n; j++) {
    if (!Number.isFinite(w[j])) {
      throw new DataValidationError(
        "Ridge sag solver diverged to non-finite values; try scaling features or a different solver"
      );
    }
  }
  return { x: w, nIter, converged };
}

/**
 * Ridge Regression (L2 Regularized Linear Regression).
 *
 * Minimizes `||y - Xw||^2 + alpha * ||w||^2`. The intercept is never
 * penalized: when `fitIntercept` is true, X and y are centered before solving
 * and the intercept is recovered from the column means.
 *
 * The model mirrors scikit-learn's `Ridge`, with one exception: `normalize`
 * (removed from scikit-learn 1.2) divides every centered column by its L2 norm
 * before the penalty is applied, and the coefficients are mapped back to the
 * original feature scale afterwards.
 *
 * @example
 * ```ts
 * import { Ridge } from 'deepbox/ml';
 *
 * const model = new Ridge({ alpha: 0.5 });
 * model.fit(X_train, y_train);
 * const predictions = model.predict(X_test);
 * ```
 *
 * @category Linear Models
 * @implements {Regressor}
 */
export class Ridge implements Regressor {
  /** Configuration options for the Ridge regression model */
  private options: {
    alpha?: number;
    fitIntercept?: boolean;
    normalize?: boolean;
    solver?: RidgeSolver;
    maxIter?: number;
    tol?: number;
  };

  /** Model coefficients (weights) after fitting - shape (n_features,) */
  private coef_?: Tensor;

  /** Coefficients as a plain array, used by predict */
  private coefArray_?: Float64Array;

  /** Intercept (bias) term after fitting */
  private intercept_?: number;

  /** Number of features seen during fit - used for validation */
  private nFeaturesIn_?: number;

  /** Number of iterations run by the solver (for iterative solvers) */
  private nIter_: number | undefined;

  /** Whether the model has been fitted to data */
  private fitted = false;

  /**
   * Create a new Ridge Regression model.
   *
   * Option values are validated when `fit` is called.
   *
   * @param options - Configuration options
   * @param options.alpha - Regularization strength (default: 1.0). Must be a finite number >= 0.
   * @param options.fitIntercept - Whether to calculate the intercept (default: true).
   * @param options.normalize - Whether to scale each centered feature column to unit L2 norm before regression (default: false).
   * @param options.solver - Solver to use (default: 'auto'). Options: 'auto', 'svd', 'cholesky', 'lsqr', 'sag'.
   * @param options.maxIter - Maximum number of iterations for 'lsqr' and 'sag' (default: 1000)
   * @param options.tol - Tolerance for 'lsqr' and 'sag' (default: 1e-4). 'lsqr' stops at a relative residual of `tol`; 'sag' stops when the gradient has shrunk by a factor of `tol`.
   */
  constructor(
    options: {
      readonly alpha?: number;
      readonly fitIntercept?: boolean;
      readonly normalize?: boolean;
      readonly solver?: RidgeSolver;
      readonly maxIter?: number;
      readonly tol?: number;
    } = {}
  ) {
    this.options = { ...options };
  }

  /**
   * Fit Ridge regression model.
   *
   * Solves the regularized least squares problem
   * `minimize ||y - Xw||^2 + alpha * ||w||^2`, i.e. the normal equations
   * `(X^T X + alpha I) w = X^T y` on the centered data.
   *
   * - `'auto'` and `'cholesky'` factor `X^T X + alpha I`. With `alpha > 0`, `'auto'` falls back
   *   to Gaussian elimination when the matrix is not numerically positive definite. When the
   *   system is singular or nearly so (`alpha = 0` with collinear features) both fall back to
   *   the SVD solution, which is the minimum-norm least-squares solution, as in scikit-learn.
   * - `'svd'` uses the singular value decomposition of the (centered) X and
   *   returns the minimum-norm solution for rank-deficient problems.
   * - `'lsqr'` runs conjugate gradient on the normal equations.
   * - `'sag'` runs accelerated gradient descent that never forms `X^T X`.
   *
   * **Time Complexity**: O(m n^2 + n^3) for the direct solvers, where m = samples, n = features
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Target values of shape (n_samples,)
   * @returns this - The fitted estimator for method chaining
   * @throws {ShapeError} If X is not 2D or y is not 1D
   * @throws {ShapeError} If X and y have different number of samples
   * @throws {DataValidationError} If X or y contain NaN/Inf values
   * @throws {DataValidationError} If X or y are empty
   * @throws {InvalidParameterError} If alpha < 0 or another option is invalid
   */
  fit(X: Tensor, y: Tensor): this {
    // Validate inputs (dimensions, empty data, NaN/Inf)
    validateFitInputs(X, y);

    const alpha = this.options.alpha ?? 1.0;
    if (typeof alpha !== "number" || !Number.isFinite(alpha) || alpha < 0) {
      throw new InvalidParameterError(
        `alpha must be >= 0 and finite; received ${String(alpha)}`,
        "alpha",
        alpha
      );
    }
    const solver = this.options.solver ?? "auto";
    if (!RIDGE_SOLVERS.includes(solver)) {
      throw new InvalidParameterError(
        `solver must be one of ${RIDGE_SOLVERS.map((s) => `'${s}'`).join(", ")}; received ${String(solver)}`,
        "solver",
        solver
      );
    }
    const maxIter = this.options.maxIter ?? 1000;
    if (!Number.isInteger(maxIter) || maxIter < 1) {
      throw new InvalidParameterError(
        `maxIter must be a positive integer; received ${String(maxIter)}`,
        "maxIter",
        maxIter
      );
    }
    const tol = this.options.tol ?? 1e-4;
    if (!Number.isFinite(tol) || tol < 0) {
      throw new InvalidParameterError(
        `tol must be a finite number >= 0; received ${String(tol)}`,
        "tol",
        tol
      );
    }
    const fitIntercept = this.options.fitIntercept ?? true;
    const normalize = this.options.normalize ?? false;

    const m = X.shape[0] ?? 0;
    const n = X.shape[1] ?? 0;

    // Fitted state is replaced only after the solve succeeds.
    const xRaw = toFloat64View(X);
    const yRaw = toFloat64View(y);

    // Center X and y. The intercept is not penalized, so centering reduces the
    // problem to a penalized fit through the origin.
    const xMean = new Float64Array(n);
    let yMean = 0;
    if (fitIntercept) {
      for (let i = 0; i < m; i++) {
        const base = i * n;
        for (let j = 0; j < n; j++) xMean[j] = (xMean[j] as number) + (xRaw[base + j] as number);
        yMean += yRaw[i] as number;
      }
      for (let j = 0; j < n; j++) xMean[j] = (xMean[j] as number) / m;
      yMean /= m;
    }

    const Xc = new Float64Array(m * n);
    const yc = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      const base = i * n;
      for (let j = 0; j < n; j++) {
        Xc[base + j] = (xRaw[base + j] as number) - (xMean[j] as number);
      }
      yc[i] = (yRaw[i] as number) - yMean;
    }

    let xScale: Float64Array | undefined;
    if (normalize) {
      xScale = new Float64Array(n);
      for (let i = 0; i < m; i++) {
        const base = i * n;
        for (let j = 0; j < n; j++) {
          xScale[j] = (xScale[j] as number) + (Xc[base + j] as number) ** 2;
        }
      }
      for (let j = 0; j < n; j++) xScale[j] = Math.sqrt(xScale[j] as number);
      for (let i = 0; i < m; i++) {
        const base = i * n;
        for (let j = 0; j < n; j++) {
          const s = xScale[j] as number;
          Xc[base + j] = s === 0 ? 0 : (Xc[base + j] as number) / s;
        }
      }
    }

    let coef: Float64Array;
    let nIter: number | undefined;
    let converged = true;

    if (solver === "sag") {
      const res = accelerated(Xc, yc, m, n, alpha, maxIter, tol);
      coef = res.x;
      nIter = res.nIter;
      converged = res.converged;
    } else if (solver === "svd") {
      coef = this.solveSvd(Xc, yc, m, n, alpha);
    } else {
      // Normal equations: gram = X^T X + alpha I, rhs = X^T y.
      const gram = new Float64Array(n * n);
      const rhs = new Float64Array(n);
      for (let r = 0; r < m; r++) {
        const base = r * n;
        const yr = yc[r] as number;
        for (let i = 0; i < n; i++) {
          const xi = Xc[base + i] as number;
          if (xi === 0) continue;
          rhs[i] = (rhs[i] as number) + xi * yr;
          const gRow = i * n;
          for (let j = i; j < n; j++) {
            gram[gRow + j] = (gram[gRow + j] as number) + xi * (Xc[base + j] as number);
          }
        }
      }
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < i; j++) gram[i * n + j] = gram[j * n + i] as number;
        gram[i * n + i] = (gram[i * n + i] as number) + alpha;
      }

      if (solver === "lsqr") {
        const res = conjugateGradient(gram, rhs, n, maxIter, tol);
        coef = res.x;
        nIter = res.nIter;
        converged = res.converged;
      } else {
        // A singular system (alpha = 0 with collinear columns) falls back to the minimum-norm
        // least-squares solution, as scikit-learn does. Without regularization X^T X is only
        // positive semi-definite, so a pivot that is tiny next to the largest diagonal entry
        // (rounding noise of a rank-deficient matrix, condition number above 1e12) also sends
        // the fit to the SVD, which stays accurate where the normal equations do not.
        let direct: Float64Array | undefined = choleskySolve(
          gram,
          rhs,
          n,
          alpha === 0 ? UNREGULARIZED_PIVOT_TOL : 0
        );
        if (direct === undefined && solver === "auto" && alpha > 0) {
          try {
            direct = gaussianSolve(gram, rhs, n);
          } catch (error) {
            if (!(error instanceof DataValidationError)) throw error;
          }
        }
        coef = direct ?? this.solveSvd(Xc, yc, m, n, alpha);
      }
    }

    if (!converged) {
      warn(
        `Solver '${solver}' did not converge within maxIter=${maxIter} iterations; ` +
          "increase maxIter, loosen tol, or scale the features",
        "ConvergenceWarning",
        "Ridge"
      );
    }

    if (xScale) {
      for (let j = 0; j < n; j++) {
        const s = xScale[j] as number;
        coef[j] = s === 0 ? 0 : (coef[j] as number) / s;
      }
    }

    let intercept = 0;
    if (fitIntercept) {
      let xMeanDotW = 0;
      for (let j = 0; j < n; j++) xMeanDotW += (xMean[j] as number) * (coef[j] as number);
      intercept = yMean - xMeanDotW;
    }

    this.nFeaturesIn_ = n;
    this.coefArray_ = coef;
    this.coef_ = tensor(coef, { dtype: "float64" });
    this.intercept_ = intercept;
    this.nIter_ = nIter;
    this.fitted = true;
    return this;
  }

  /**
   * Ridge solution from the SVD of the centered design matrix:
   * `w = V diag(s / (s^2 + alpha)) U^T y`.
   */
  private solveSvd(Xc: Float64Array, yc: Float64Array, m: number, n: number, alpha: number) {
    const design = tensor(Xc, { dtype: "float64" }).reshape([m, n]);
    const [U, s, Vt] = svd(design, false);
    const uData = toFloat64View(U);
    const sData = toFloat64View(s);
    const vtData = toFloat64View(Vt);
    const k = sData.length;
    const sMax = k > 0 ? (sData[0] as number) : 0;
    // Without regularization, drop directions below numerical rank so the
    // result is the minimum-norm least squares solution.
    const cutoff = alpha === 0 ? Number.EPSILON * Math.max(m, n) * sMax : 0;

    const coef = new Float64Array(n);
    for (let c = 0; c < k; c++) {
      const sigma = sData[c] as number;
      if (!(sigma > cutoff)) continue;
      let uty = 0;
      for (let i = 0; i < m; i++) uty += (uData[i * k + c] as number) * (yc[i] as number);
      const factor = (sigma / (sigma * sigma + alpha)) * uty;
      for (let j = 0; j < n; j++) {
        coef[j] = (coef[j] as number) + factor * (vtData[c * n + j] as number);
      }
    }
    return coef;
  }

  /**
   * Predict using the Ridge regression model.
   *
   * Computes predictions as: ŷ = X @ coef + intercept
   *
   * **Time Complexity**: O(nm) where n = samples, m = features
   * **Space Complexity**: O(n)
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predicted values of shape (n_samples,), dtype float64
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predict(X: Tensor): Tensor {
    const coef = this.coefArray_;
    if (!this.fitted || !coef) {
      throw new NotFittedError("Ridge must be fitted before prediction");
    }

    validatePredictInputs(X, this.nFeaturesIn_ ?? 0, "Ridge");

    const m = X.shape[0] ?? 0;
    const n = X.shape[1] ?? 0;
    const xData = toFloat64View(X);
    const intercept = this.intercept_ ?? 0;
    const pred = new Float64Array(m);

    for (let i = 0; i < m; i++) {
      let sum = intercept;
      const base = i * n;
      for (let j = 0; j < n; j++) sum += (xData[base + j] as number) * (coef[j] as number);
      pred[i] = sum;
    }

    return tensor(pred, { dtype: "float64" });
  }

  /**
   * Return the coefficient of determination R² of the prediction.
   *
   * R² = 1 - SS_res / SS_tot, where SS_res = Σ(y - ŷ)² and SS_tot = Σ(y - mean(y))².
   * A constant y gives 1 when the predictions are exact and 0 otherwise.
   *
   * **Time Complexity**: O(n) where n = number of samples
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True values of shape (n_samples,)
   * @returns R² score (best possible score is 1.0, can be negative)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1-dimensional or its length differs from the number of samples in X
   * @throws {DataValidationError} If y is empty or contains NaN/Inf
   */
  score(X: Tensor, y: Tensor): number {
    if (!this.fitted) {
      throw new NotFittedError("Ridge must be fitted before scoring");
    }

    return r2ScoreOf(y, () => this.predict(X));
  }

  /**
   * Get parameters for this estimator.
   *
   * Returns every hyperparameter with its effective value, so the result can
   * be passed back to the constructor.
   *
   * @returns Object containing all parameters with their current values
   */
  getParams(): Record<string, unknown> {
    return {
      alpha: this.options.alpha ?? 1.0,
      fitIntercept: this.options.fitIntercept ?? true,
      normalize: this.options.normalize ?? false,
      solver: this.options.solver ?? "auto",
      maxIter: this.options.maxIter ?? 1000,
      tol: this.options.tol ?? 1e-4,
    };
  }

  /**
   * Set the parameters of this estimator.
   *
   * Changing parameters does not alter a fitted model; call `fit` again.
   * Range checks (for example `alpha >= 0`) run at `fit`.
   *
   * @param params - Dictionary of parameters to set
   * @returns this - The estimator for method chaining
   * @throws {InvalidParameterError} If a parameter name is unknown or its value has the wrong type
   */
  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "alpha":
          if (typeof value !== "number" || !Number.isFinite(value)) {
            throw new InvalidParameterError(
              `alpha must be a finite number; received ${String(value)}`,
              "alpha",
              value
            );
          }
          this.options.alpha = value;
          break;

        case "maxIter":
          if (typeof value !== "number" || !Number.isFinite(value)) {
            throw new InvalidParameterError(
              `maxIter must be a finite number; received ${String(value)}`,
              "maxIter",
              value
            );
          }
          this.options.maxIter = value;
          break;

        case "tol":
          if (typeof value !== "number" || !Number.isFinite(value)) {
            throw new InvalidParameterError(
              `tol must be a finite number; received ${String(value)}`,
              "tol",
              value
            );
          }
          this.options.tol = value;
          break;

        case "fitIntercept":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError(
              `fitIntercept must be a boolean; received ${String(value)}`,
              "fitIntercept",
              value
            );
          }
          this.options.fitIntercept = value;
          break;

        case "normalize":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError(
              `normalize must be a boolean; received ${String(value)}`,
              "normalize",
              value
            );
          }
          this.options.normalize = value;
          break;

        case "solver":
          if (typeof value !== "string" || !RIDGE_SOLVERS.includes(value as RidgeSolver)) {
            throw new InvalidParameterError(`Invalid solver: ${String(value)}`, "solver", value);
          }
          this.options.solver = value as RidgeSolver;
          break;

        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }

  /**
   * Create an unfitted copy of this estimator with the same parameters.
   *
   * @returns A new Ridge instance
   */
  clone(): Ridge {
    return new Ridge(
      this.getParams() as {
        alpha?: number;
        fitIntercept?: boolean;
        normalize?: boolean;
        solver?: RidgeSolver;
        maxIter?: number;
        tol?: number;
      }
    );
  }

  /**
   * Get the model coefficients (weights).
   *
   * @returns Coefficient tensor of shape (n_features,), dtype float64
   * @throws {NotFittedError} If the model has not been fitted
   */
  get coef(): Tensor {
    if (!this.fitted || !this.coef_) {
      throw new NotFittedError("Ridge must be fitted to access coefficients");
    }
    return this.coef_;
  }

  /**
   * Get the intercept (bias term).
   *
   * @returns Intercept value (0 when `fitIntercept` is false)
   * @throws {NotFittedError} If the model has not been fitted
   */
  get intercept(): number {
    if (!this.fitted) {
      throw new NotFittedError("Ridge must be fitted to access intercept");
    }
    return this.intercept_ ?? 0;
  }

  /**
   * Get the number of iterations run by the solver.
   *
   * @returns Number of iterations (undefined for direct solvers)
   * @throws {NotFittedError} If the model has not been fitted
   */
  get nIter(): number | undefined {
    if (!this.fitted) {
      throw new NotFittedError("Ridge must be fitted to access nIter");
    }
    return this.nIter_;
  }
}
