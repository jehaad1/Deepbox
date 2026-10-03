/**
 * Quantile Regressor.
 *
 * Linear model for conditional quantiles. Minimizes the mean pinball (quantile)
 * loss with an optional L1 penalty on the coefficients, which is the same
 * objective as scikit-learn's `QuantileRegressor`. The problem is a linear
 * program; it is solved with a primal-dual interior point method.
 *
 * @module ml/linear/QuantileRegressor
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { DataValidationError, InvalidParameterError, NotFittedError, warn } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { toFloat64View, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Regressor } from "../base";
import { r2ScoreOf } from "./LinearRegression";

/**
 * Cholesky factor (lower triangular, row-major) of the symmetric positive
 * definite matrix `M`, or `undefined` when a pivot is not positive.
 */
function choleskyFactor(M: Float64Array, p: number): Float64Array | undefined {
  const L = new Float64Array(p * p);
  for (let j = 0; j < p; j++) {
    let diag = M[j * p + j] as number;
    for (let k = 0; k < j; k++) {
      const v = L[j * p + k] as number;
      diag -= v * v;
    }
    if (!(diag > 0) || !Number.isFinite(diag)) return undefined;
    const ljj = Math.sqrt(diag);
    L[j * p + j] = ljj;
    for (let i = j + 1; i < p; i++) {
      let sum = M[i * p + j] as number;
      for (let k = 0; k < j; k++) sum -= (L[i * p + k] as number) * (L[j * p + k] as number);
      L[i * p + j] = sum / ljj;
    }
  }
  return L;
}

/** Solve `L L^T x = b` given the Cholesky factor `L`. */
function choleskyBackSolve(L: Float64Array, b: Float64Array, p: number): Float64Array {
  const x = new Float64Array(p);
  for (let i = 0; i < p; i++) {
    let sum = b[i] as number;
    for (let k = 0; k < i; k++) sum -= (L[i * p + k] as number) * (x[k] as number);
    x[i] = sum / (L[i * p + i] as number);
  }
  for (let i = p - 1; i >= 0; i--) {
    let sum = x[i] as number;
    for (let k = i + 1; k < p; k++) sum -= (L[k * p + i] as number) * (x[k] as number);
    x[i] = sum / (L[i * p + i] as number);
  }
  return x;
}

/**
 * Solve a weighted normal-equation system `(A^T diag(weights) A) x = rhs`.
 *
 * A tiny ridge term makes the factorization succeed for collinear or all-zero
 * columns (it grows until the factorization works); iterative refinement
 * against the unregularized matrix removes the bias the ridge would add.
 */
function solveNormalEquations(
  A: Float64Array,
  weights: Float64Array,
  rhs: Float64Array,
  rows: number,
  p: number
): Float64Array {
  const M = new Float64Array(p * p);
  for (let k = 0; k < rows; k++) {
    const wk = weights[k] as number;
    const base = k * p;
    for (let i = 0; i < p; i++) {
      const ai = A[base + i] as number;
      if (ai === 0) continue;
      const wai = wk * ai;
      const mRow = i * p;
      for (let j = 0; j <= i; j++) {
        M[mRow + j] = (M[mRow + j] as number) + wai * (A[base + j] as number);
      }
    }
  }
  let maxDiag = 0;
  for (let i = 0; i < p; i++) {
    for (let j = 0; j < i; j++) M[j * p + i] = M[i * p + j] as number;
    maxDiag = Math.max(maxDiag, M[i * p + i] as number);
  }
  if (!(maxDiag > 0)) maxDiag = 1;

  let ridge = 1e-13 * maxDiag;
  for (let attempt = 0; attempt < 12; attempt++) {
    const Mr = Float64Array.from(M);
    for (let i = 0; i < p; i++) Mr[i * p + i] = (Mr[i * p + i] as number) + ridge;
    const L = choleskyFactor(Mr, p);
    if (L) {
      const x = choleskyBackSolve(L, rhs, p);
      for (let refine = 0; refine < 3; refine++) {
        const r = new Float64Array(p);
        let rNorm = 0;
        for (let i = 0; i < p; i++) {
          let mx = 0;
          for (let j = 0; j < p; j++) mx += (M[i * p + j] as number) * (x[j] as number);
          const ri = (rhs[i] as number) - mx;
          r[i] = ri;
          rNorm = Math.max(rNorm, Math.abs(ri));
        }
        if (rNorm === 0) break;
        const corr = choleskyBackSolve(L, r, p);
        for (let i = 0; i < p; i++) x[i] = (x[i] as number) + (corr[i] as number);
      }
      return x;
    }
    ridge *= 100;
  }
  throw new DataValidationError(
    "QuantileRegressor could not factor the interior point system; check X for non-finite or extreme values"
  );
}

/**
 * Largest step `t <= 1` that keeps `x + t * dx` strictly positive, damped by 0.99.
 */
function maxStep(x: Float64Array, dx: Float64Array): number {
  let t = Infinity;
  for (let i = 0; i < x.length; i++) {
    const d = dx[i] as number;
    if (d < 0) {
      const r = -(x[i] as number) / d;
      if (r < t) t = r;
    }
  }
  return Math.min(1, 0.99 * t);
}

/**
 * Minimize `sum_k rho_q(y_k - A_k beta)` with Mehrotra's primal-dual interior
 * point method applied to the bounded dual
 * `max y^T d  s.t.  A^T d = (1 - q) A^T 1,  0 <= d <= 1`.
 * The multipliers of the equality constraints are the regression coefficients
 * (up to sign).
 */
function solvePinballLP(
  A: Float64Array,
  y: Float64Array,
  rows: number,
  p: number,
  q: number,
  maxIter: number,
  tol: number
): { beta: Float64Array; nIter: number; converged: boolean } {
  // Right-hand side b = (1 - q) A^T 1; d = (1 - q) 1 is a feasible start.
  const b = new Float64Array(p);
  for (let k = 0; k < rows; k++) {
    for (let j = 0; j < p; j++) b[j] = (b[j] as number) + (A[k * p + j] as number);
  }
  for (let j = 0; j < p; j++) b[j] = (b[j] as number) * (1 - q);

  const d = new Float64Array(rows).fill(1 - q);
  const s = new Float64Array(rows).fill(q);
  const z = new Float64Array(rows);
  const w = new Float64Array(rows);

  // Dual start from a least squares fit: beta_dual = -beta_ols, w - z = residual.
  const ones = new Float64Array(rows).fill(1);
  const aty = new Float64Array(p);
  let yScale = 0;
  for (let k = 0; k < rows; k++) {
    const yk = y[k] as number;
    yScale = Math.max(yScale, Math.abs(yk));
    for (let j = 0; j < p; j++) aty[j] = (aty[j] as number) + (A[k * p + j] as number) * yk;
  }
  let betaDual = solveNormalEquations(A, ones, aty, rows, p);
  let meanAbsRes = 0;
  const res0 = new Float64Array(rows);
  for (let k = 0; k < rows; k++) {
    let fit = 0;
    for (let j = 0; j < p; j++) fit += (A[k * p + j] as number) * (betaDual[j] as number);
    const r = (y[k] as number) - fit;
    res0[k] = r;
    meanAbsRes += Math.abs(r);
  }
  meanAbsRes /= rows;
  const delta = Math.max(0.1 * meanAbsRes, 1e-8 * Math.max(yScale, 1));
  for (let k = 0; k < rows; k++) {
    const r = res0[k] as number;
    z[k] = Math.max(-r, 0) + delta;
    w[k] = Math.max(r, 0) + delta;
  }
  betaDual = betaDual.map((v) => -v);

  const rd = new Float64Array(rows);
  const rp = new Float64Array(p);
  const rs = new Float64Array(rows);
  const D = new Float64Array(rows);
  const t = new Float64Array(rows);
  const dd = new Float64Array(rows);
  const ds = new Float64Array(rows);
  const dz = new Float64Array(rows);
  const dw = new Float64Array(rows);
  const dda = new Float64Array(rows);
  const dsa = new Float64Array(rows);
  const dza = new Float64Array(rows);
  const dwa = new Float64Array(rows);
  const rhs = new Float64Array(p);

  /** Newton direction for the given complementarity targets. */
  const direction = (
    sigmaMu: number,
    corrD: Float64Array | undefined,
    corrS: Float64Array | undefined,
    outD: Float64Array,
    outS: Float64Array,
    outZ: Float64Array,
    outW: Float64Array
  ): Float64Array => {
    for (let k = 0; k < rows; k++) {
      const dk = d[k] as number;
      const sk = s[k] as number;
      const zk = z[k] as number;
      const wk = w[k] as number;
      const cd = corrD ? (corrD[k] as number) : 0;
      const cs = corrS ? (corrS[k] as number) : 0;
      // Targets: z d -> sigmaMu - cd, w s -> sigmaMu - cs.
      const tz = sigmaMu - zk * dk - cd;
      const tw = sigmaMu - wk * sk - cs;
      t[k] = -(rd[k] as number) + tz / dk - tw / sk + (wk / sk) * (rs[k] as number);
    }
    // rhs = rp - A^T (D t)
    for (let j = 0; j < p; j++) rhs[j] = rp[j] as number;
    for (let k = 0; k < rows; k++) {
      const v = (D[k] as number) * (t[k] as number);
      if (v === 0) continue;
      for (let j = 0; j < p; j++) {
        rhs[j] = (rhs[j] as number) - (A[k * p + j] as number) * v;
      }
    }
    const dBeta = solveNormalEquations(A, D, rhs, rows, p);
    for (let k = 0; k < rows; k++) {
      let adb = 0;
      for (let j = 0; j < p; j++) adb += (A[k * p + j] as number) * (dBeta[j] as number);
      const dk = d[k] as number;
      const sk = s[k] as number;
      const ddk = (D[k] as number) * (adb + (t[k] as number));
      const dsk = (rs[k] as number) - ddk;
      outD[k] = ddk;
      outS[k] = dsk;
      const cd = corrD ? (corrD[k] as number) : 0;
      const cs = corrS ? (corrS[k] as number) : 0;
      outZ[k] = (sigmaMu - (z[k] as number) * dk - cd - (z[k] as number) * ddk) / dk;
      outW[k] = (sigmaMu - (w[k] as number) * sk - cs - (w[k] as number) * dsk) / sk;
    }
    return dBeta;
  };

  let nIter = 0;
  let converged = false;
  let bScale = 0;
  for (let j = 0; j < p; j++) bScale = Math.max(bScale, Math.abs(b[j] as number));

  for (let iter = 0; iter < maxIter; iter++) {
    // Residuals of the KKT system: rd = -y - A beta - z + w, rp = b - A^T d, rs = 1 - d - s.
    let gap = 0;
    let dualObj = 0;
    let maxRd = 0;
    let maxRs = 0;
    for (let k = 0; k < rows; k++) {
      let ab = 0;
      for (let j = 0; j < p; j++) ab += (A[k * p + j] as number) * (betaDual[j] as number);
      const r = -(y[k] as number) - ab - (z[k] as number) + (w[k] as number);
      rd[k] = r;
      maxRd = Math.max(maxRd, Math.abs(r));
      const rsk = 1 - (d[k] as number) - (s[k] as number);
      rs[k] = rsk;
      maxRs = Math.max(maxRs, Math.abs(rsk));
      gap += (z[k] as number) * (d[k] as number) + (w[k] as number) * (s[k] as number);
      dualObj += (y[k] as number) * (d[k] as number);
    }
    for (let j = 0; j < p; j++) rp[j] = b[j] as number;
    for (let k = 0; k < rows; k++) {
      const dk = d[k] as number;
      for (let j = 0; j < p; j++) rp[j] = (rp[j] as number) - (A[k * p + j] as number) * dk;
    }
    let maxRp = 0;
    for (let j = 0; j < p; j++) maxRp = Math.max(maxRp, Math.abs(rp[j] as number));

    nIter = iter;
    if (
      gap <= tol * (1 + Math.abs(dualObj)) &&
      maxRp <= tol * (1 + bScale) &&
      maxRd <= tol * (1 + yScale) &&
      maxRs <= tol
    ) {
      converged = true;
      break;
    }
    if (!Number.isFinite(gap)) break;

    const mu = gap / (2 * rows);
    for (let k = 0; k < rows; k++) {
      D[k] = 1 / ((z[k] as number) / (d[k] as number) + (w[k] as number) / (s[k] as number));
    }

    // Predictor (affine scaling) step.
    direction(0, undefined, undefined, dda, dsa, dza, dwa);
    const aPrimalAff = Math.min(maxStep(d, dda), maxStep(s, dsa));
    const aDualAff = Math.min(maxStep(z, dza), maxStep(w, dwa));
    let gapAff = 0;
    for (let k = 0; k < rows; k++) {
      gapAff +=
        ((z[k] as number) + aDualAff * (dza[k] as number)) *
          ((d[k] as number) + aPrimalAff * (dda[k] as number)) +
        ((w[k] as number) + aDualAff * (dwa[k] as number)) *
          ((s[k] as number) + aPrimalAff * (dsa[k] as number));
    }
    const muAff = gapAff / (2 * rows);
    const sigma = mu > 0 ? Math.min(1, (muAff / mu) ** 3) : 0;

    // Corrector step with second-order terms.
    const corrD = new Float64Array(rows);
    const corrS = new Float64Array(rows);
    for (let k = 0; k < rows; k++) {
      corrD[k] = (dza[k] as number) * (dda[k] as number);
      corrS[k] = (dwa[k] as number) * (dsa[k] as number);
    }
    const dBeta = direction(sigma * mu, corrD, corrS, dd, ds, dz, dw);
    const aPrimal = Math.min(maxStep(d, dd), maxStep(s, ds));
    const aDual = Math.min(maxStep(z, dz), maxStep(w, dw));
    if (!(aPrimal > 0) || !(aDual > 0)) break;
    for (let k = 0; k < rows; k++) {
      d[k] = (d[k] as number) + aPrimal * (dd[k] as number);
      s[k] = (s[k] as number) + aPrimal * (ds[k] as number);
      z[k] = (z[k] as number) + aDual * (dz[k] as number);
      w[k] = (w[k] as number) + aDual * (dw[k] as number);
    }
    for (let j = 0; j < p; j++) {
      betaDual[j] = (betaDual[j] as number) + aDual * (dBeta[j] as number);
    }
  }

  const beta = new Float64Array(p);
  for (let j = 0; j < p; j++) beta[j] = -(betaDual[j] as number);
  return { beta, nIter, converged };
}

/**
 * Quantile Regressor.
 *
 * Fits a linear model of the conditional `quantile` of y by minimizing
 *
 *   (1 / n) * sum_i rho_q(y_i - x_i . w - b) + alpha * ||w||_1
 *
 * where `rho_q(r) = q * max(r, 0) + (1 - q) * max(-r, 0)` is the pinball loss.
 * The intercept `b` is never penalized. This is the objective used by
 * scikit-learn's `QuantileRegressor`; note that its default `alpha = 1.0` is a
 * strong L1 penalty, so pass `alpha: 0` for an unpenalized fit.
 *
 * @example
 * ```ts
 * import { QuantileRegressor } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1], [2], [3], [4], [5]]);
 * const y = tensor([1.2, 1.9, 3.4, 3.9, 5.8]);
 * const median = new QuantileRegressor({ quantile: 0.5, alpha: 0 }).fit(X, y);
 * const upper = new QuantileRegressor({ quantile: 0.9, alpha: 0 }).fit(X, y);
 * ```
 *
 * @category Linear Models
 * @implements {Regressor}
 */
export class QuantileRegressor implements Regressor {
  private quantile: number;
  private alpha: number;
  private maxIter: number;
  private tol: number;
  private fitIntercept: boolean;

  private coef_?: Float64Array;
  private coefTensor_?: Tensor;
  private intercept_ = 0;
  private nIter_ = 0;
  private nFeaturesIn_ = 0;
  private fitted = false;

  /**
   * Create a new Quantile Regressor.
   *
   * @param options - Configuration options
   * @param options.quantile - Quantile to predict, strictly between 0 and 1 (default: 0.5)
   * @param options.alpha - L1 regularization strength, >= 0 (default: 1.0)
   * @param options.fitIntercept - Whether to fit an intercept (default: true)
   * @param options.maxIter - Maximum interior point iterations (default: 100)
   * @param options.tol - Tolerance on the relative duality gap and the KKT residuals (default: 1e-8)
   * @throws {InvalidParameterError} If any option is out of range
   */
  constructor(
    options: {
      readonly quantile?: number;
      readonly alpha?: number;
      readonly fitIntercept?: boolean;
      readonly maxIter?: number;
      readonly tol?: number;
    } = {}
  ) {
    this.quantile = options.quantile ?? 0.5;
    this.alpha = options.alpha ?? 1.0;
    this.fitIntercept = options.fitIntercept ?? true;
    this.maxIter = options.maxIter ?? 100;
    this.tol = options.tol ?? 1e-8;
    this.validateParams();
  }

  private validateParams(): void {
    if (typeof this.quantile !== "number" || !(this.quantile > 0 && this.quantile < 1)) {
      throw new InvalidParameterError(
        `quantile must be in (0, 1); received ${String(this.quantile)}`,
        "quantile",
        this.quantile
      );
    }
    if (typeof this.alpha !== "number" || !Number.isFinite(this.alpha) || this.alpha < 0) {
      throw new InvalidParameterError(
        `alpha must be a finite number >= 0; received ${String(this.alpha)}`,
        "alpha",
        this.alpha
      );
    }
    if (typeof this.fitIntercept !== "boolean") {
      throw new InvalidParameterError(
        `fitIntercept must be a boolean; received ${String(this.fitIntercept)}`,
        "fitIntercept",
        this.fitIntercept
      );
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new InvalidParameterError(
        `maxIter must be a positive integer; received ${String(this.maxIter)}`,
        "maxIter",
        this.maxIter
      );
    }
    if (typeof this.tol !== "number" || !Number.isFinite(this.tol) || this.tol <= 0) {
      throw new InvalidParameterError(
        `tol must be a finite number > 0; received ${String(this.tol)}`,
        "tol",
        this.tol
      );
    }
  }

  /**
   * Fit the model.
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
    const nF = X.shape[1] ?? 0;
    const xRaw = toFloat64View(X);
    const yRaw = toFloat64View(y);
    const q = this.quantile;
    const hasIntercept = this.fitIntercept;

    // Column scales (max |x|) keep the interior point system well conditioned.
    const scale = new Float64Array(nF).fill(0);
    for (let i = 0; i < n; i++) {
      for (let f = 0; f < nF; f++) {
        const v = Math.abs(xRaw[i * nF + f] as number);
        if (v > (scale[f] as number)) scale[f] = v;
      }
    }
    for (let f = 0; f < nF; f++) if (scale[f] === 0) scale[f] = 1;

    // Augmented problem: data rows, then two pseudo rows (+c e_j and -c e_j,
    // target 0) per feature. Their pinball losses add up to c * |w_j|, which
    // encodes n * alpha * |w_j| (the objective multiplied by n).
    const p = nF + (hasIntercept ? 1 : 0);
    const penalized = this.alpha > 0;
    const rows = n + (penalized ? 2 * nF : 0);
    const A = new Float64Array(rows * p);
    const target = new Float64Array(rows);
    for (let i = 0; i < n; i++) {
      for (let f = 0; f < nF; f++) {
        A[i * p + f] = (xRaw[i * nF + f] as number) / (scale[f] as number);
      }
      if (hasIntercept) A[i * p + nF] = 1;
      target[i] = yRaw[i] as number;
    }
    if (penalized) {
      for (let f = 0; f < nF; f++) {
        const c = (n * this.alpha) / (scale[f] as number);
        A[(n + 2 * f) * p + f] = c;
        A[(n + 2 * f + 1) * p + f] = -c;
      }
    }

    const sol = solvePinballLP(A, target, rows, p, q, this.maxIter, this.tol);
    if (!sol.converged) {
      warn(
        `Interior point solver did not reach tol=${this.tol} in ${this.maxIter} iterations; ` +
          "increase maxIter or loosen tol",
        "ConvergenceWarning",
        "QuantileRegressor"
      );
    }

    // The interior point solution is strictly interior, so coefficients the L1
    // penalty drives to zero come out as tiny nonzero values; snap them to zero.
    let yMax = 0;
    for (let i = 0; i < n; i++) yMax = Math.max(yMax, Math.abs(yRaw[i] as number));
    const snap = penalized ? this.tol * (1 + yMax) : 0;

    const coef = new Float64Array(nF);
    for (let f = 0; f < nF; f++) {
      const scaled = sol.beta[f] as number;
      const v = Math.abs(scaled) <= snap ? 0 : scaled / (scale[f] as number);
      if (!Number.isFinite(v)) {
        throw new DataValidationError("QuantileRegressor produced non-finite coefficients");
      }
      coef[f] = v;
    }
    const intercept = hasIntercept ? (sol.beta[nF] as number) : 0;
    if (!Number.isFinite(intercept)) {
      throw new DataValidationError("QuantileRegressor produced a non-finite intercept");
    }

    this.nFeaturesIn_ = nF;
    this.coef_ = coef;
    this.coefTensor_ = tensor(coef, { dtype: "float64" });
    this.intercept_ = intercept;
    this.nIter_ = sol.nIter;
    this.fitted = true;
    return this;
  }

  /**
   * Predict the conditional quantile for each sample.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predictions of shape (n_samples,), dtype float64
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has the wrong shape or feature count
   */
  predict(X: Tensor): Tensor {
    const coef = this.coef_;
    if (!this.fitted || !coef) {
      throw new NotFittedError("QuantileRegressor must be fitted before predict");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "QuantileRegressor");
    const nTest = X.shape[0] ?? 0;
    const nF = this.nFeaturesIn_;
    const xData = toFloat64View(X);
    const result = new Float64Array(nTest);

    for (let i = 0; i < nTest; i++) {
      let pred = this.intercept_;
      for (let f = 0; f < nF; f++) {
        pred += (coef[f] as number) * (xData[i * nF + f] as number);
      }
      result[i] = pred;
    }
    return tensor(result, { dtype: "float64" });
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
      throw new NotFittedError("QuantileRegressor must be fitted before scoring");
    }
    return r2ScoreOf(y, () => this.predict(X));
  }

  /**
   * Fitted coefficients, one per feature.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get coef(): Float64Array {
    if (!this.fitted || !this.coef_) {
      throw new NotFittedError("QuantileRegressor must be fitted before accessing coef");
    }
    return this.coef_;
  }

  /**
   * Fitted coefficients as a float64 tensor of shape (n_features,).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get coefTensor(): Tensor {
    if (!this.fitted || !this.coefTensor_) {
      throw new NotFittedError("QuantileRegressor must be fitted before accessing coefTensor");
    }
    return this.coefTensor_;
  }

  /**
   * Fitted intercept (0 when `fitIntercept` is false).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get intercept(): number {
    if (!this.fitted) {
      throw new NotFittedError("QuantileRegressor must be fitted before accessing intercept");
    }
    return this.intercept_;
  }

  /**
   * Number of interior point iterations used by the last fit.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get nIter(): number {
    if (!this.fitted) {
      throw new NotFittedError("QuantileRegressor must be fitted before accessing nIter");
    }
    return this.nIter_;
  }

  /**
   * Get the hyperparameters of this estimator.
   */
  getParams(): Record<string, unknown> {
    return {
      quantile: this.quantile,
      alpha: this.alpha,
      fitIntercept: this.fitIntercept,
      maxIter: this.maxIter,
      tol: this.tol,
    };
  }

  /**
   * Set hyperparameters. The fitted model is unchanged until `fit` is called again.
   *
   * @param params - Parameters to set (quantile, alpha, fitIntercept, maxIter, tol)
   * @returns this
   * @throws {InvalidParameterError} If a name is unknown or a value is invalid
   */
  setParams(params: Record<string, unknown>): this {
    const next = {
      quantile: this.quantile,
      alpha: this.alpha,
      fitIntercept: this.fitIntercept,
      maxIter: this.maxIter,
      tol: this.tol,
    };
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "quantile":
        case "alpha":
        case "maxIter":
        case "tol":
          if (typeof value !== "number") {
            throw new InvalidParameterError(
              `${key} must be a number; received ${String(value)}`,
              key,
              value
            );
          }
          next[key] = value;
          break;
        case "fitIntercept":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError(
              `fitIntercept must be a boolean; received ${String(value)}`,
              "fitIntercept",
              value
            );
          }
          next.fitIntercept = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    const previous = this.getParams();
    this.quantile = next.quantile;
    this.alpha = next.alpha;
    this.fitIntercept = next.fitIntercept;
    this.maxIter = next.maxIter;
    this.tol = next.tol;
    try {
      this.validateParams();
    } catch (e) {
      this.quantile = previous["quantile"] as number;
      this.alpha = previous["alpha"] as number;
      this.fitIntercept = previous["fitIntercept"] as boolean;
      this.maxIter = previous["maxIter"] as number;
      this.tol = previous["tol"] as number;
      throw e;
    }
    return this;
  }

  /**
   * Create an unfitted copy of this estimator with the same parameters.
   */
  clone(): QuantileRegressor {
    return new QuantileRegressor({
      quantile: this.quantile,
      alpha: this.alpha,
      fitIntercept: this.fitIntercept,
      maxIter: this.maxIter,
      tol: this.tol,
    });
  }
}
