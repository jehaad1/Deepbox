/**
 * Quantile Regressor.
 *
 * Linear regression model that predicts conditional quantiles instead
 * of the mean. Uses iteratively reweighted least squares (IRLS) to
 * minimize the pinball (quantile) loss.
 *
 * @module ml/linear/QuantileRegressor
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { validateFitInputs, validatePredictInputs } from "../_validation";
import type { Regressor } from "../base";

export class QuantileRegressor implements Regressor {
  private readonly quantile: number;
  private readonly alpha: number;
  private readonly maxIter: number;
  private readonly tol: number;

  private coef_?: Float64Array;
  private intercept_ = 0;
  private nFeaturesIn_ = 0;
  private fitted = false;

  constructor(
    options: {
      readonly quantile?: number;
      readonly alpha?: number;
      readonly maxIter?: number;
      readonly tol?: number;
    } = {}
  ) {
    this.quantile = options.quantile ?? 0.5;
    this.alpha = options.alpha ?? 1.0;
    this.maxIter = options.maxIter ?? 100;
    this.tol = options.tol ?? 1e-4;

    if (this.quantile <= 0 || this.quantile >= 1) {
      throw new InvalidParameterError("quantile must be in (0, 1)", "quantile", this.quantile);
    }
    if (this.alpha < 0) {
      throw new InvalidParameterError("alpha must be >= 0", "alpha", this.alpha);
    }
  }

  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const n = X.shape[0] ?? 0;
    const nF = X.shape[1] ?? 0;
    this.nFeaturesIn_ = nF;

    const xData = new Float64Array(n * nF);
    const yData = new Float64Array(n);
    for (let i = 0; i < n * nF; i++) xData[i] = Number(X.data[X.offset + i]);
    for (let i = 0; i < n; i++) yData[i] = Number(y.data[y.offset + i]);

    // IRLS for quantile regression
    // Initialize with OLS-like solution
    const coef = new Float64Array(nF);
    let intercept = 0;

    // Compute mean of y as initial intercept
    for (let i = 0; i < n; i++) intercept += yData[i] ?? 0;
    intercept /= n;

    const q = this.quantile;

    for (let iter = 0; iter < this.maxIter; iter++) {
      // Compute residuals
      const residuals = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let pred = intercept;
        for (let f = 0; f < nF; f++) {
          pred += (coef[f] ?? 0) * (xData[i * nF + f] ?? 0);
        }
        residuals[i] = (yData[i] ?? 0) - pred;
      }

      // Compute weights for IRLS
      const weights = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const r = residuals[i] ?? 0;
        const absR = Math.abs(r);
        if (absR < 1e-10) {
          weights[i] = 1 / 1e-10;
        } else {
          // Weight = quantile_loss_derivative / residual
          weights[i] = r > 0 ? q / absR : (1 - q) / absR;
        }
      }

      // Weighted least squares step
      // Solve (X^T W X + alpha I) beta = X^T W y (with intercept)
      // Augment with intercept column
      const dim = nF + 1;
      const XtWX = new Float64Array(dim * dim);
      const XtWy = new Float64Array(dim);

      for (let i = 0; i < n; i++) {
        const w = weights[i] ?? 0;
        const yi = yData[i] ?? 0;

        // Intercept column (index nF)
        XtWX[nF * dim + nF] = (XtWX[nF * dim + nF] ?? 0) + w;
        XtWy[nF] = (XtWy[nF] ?? 0) + w * yi;

        for (let f = 0; f < nF; f++) {
          const xif = xData[i * nF + f] ?? 0;
          XtWy[f] = (XtWy[f] ?? 0) + w * xif * yi;
          XtWX[f * dim + nF] = (XtWX[f * dim + nF] ?? 0) + w * xif;
          XtWX[nF * dim + f] = (XtWX[nF * dim + f] ?? 0) + w * xif;

          for (let g = 0; g < nF; g++) {
            XtWX[f * dim + g] = (XtWX[f * dim + g] ?? 0) + w * xif * (xData[i * nF + g] ?? 0);
          }
        }
      }

      // Add L2 regularization (not on intercept)
      for (let f = 0; f < nF; f++) {
        XtWX[f * dim + f] = (XtWX[f * dim + f] ?? 0) + this.alpha;
      }

      // Solve via Gaussian elimination
      const solution = this.solveLinear(XtWX, XtWy, dim);

      // Check convergence
      let maxChange = 0;
      for (let f = 0; f < nF; f++) {
        const change = Math.abs((solution[f] ?? 0) - (coef[f] ?? 0));
        if (change > maxChange) maxChange = change;
        coef[f] = solution[f] ?? 0;
      }
      const intChange = Math.abs((solution[nF] ?? 0) - intercept);
      if (intChange > maxChange) maxChange = intChange;
      intercept = solution[nF] ?? 0;

      if (maxChange < this.tol) break;
    }

    this.coef_ = coef;
    this.intercept_ = intercept;
    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.coef_) {
      throw new NotFittedError("QuantileRegressor must be fitted before predict");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "QuantileRegressor");
    const nTest = X.shape[0] ?? 0;
    const nF = this.nFeaturesIn_;
    const result = new Float64Array(nTest);

    for (let i = 0; i < nTest; i++) {
      let pred = this.intercept_;
      for (let f = 0; f < nF; f++) {
        pred += (this.coef_![f] ?? 0) * Number(X.data[X.offset + i * nF + f]);
      }
      result[i] = pred;
    }
    return tensor(Array.from(result));
  }

  score(X: Tensor, y: Tensor): number {
    const pred = this.predict(X);
    const n = y.size;
    let ssRes = 0;
    let ssTot = 0;
    let yMean = 0;
    for (let i = 0; i < n; i++) yMean += Number(y.data[y.offset + i]);
    yMean /= n;
    for (let i = 0; i < n; i++) {
      const yi = Number(y.data[y.offset + i]);
      const pi = Number(pred.data[pred.offset + i]);
      ssRes += (yi - pi) ** 2;
      ssTot += (yi - yMean) ** 2;
    }
    return ssTot > 0 ? 1 - ssRes / ssTot : 0;
  }

  get coef(): Float64Array {
    if (!this.fitted || !this.coef_) throw new NotFittedError("QuantileRegressor must be fitted");
    return this.coef_;
  }

  get intercept(): number {
    if (!this.fitted) throw new NotFittedError("QuantileRegressor must be fitted");
    return this.intercept_;
  }

  getParams(): Record<string, unknown> {
    return {
      quantile: this.quantile,
      alpha: this.alpha,
      maxIter: this.maxIter,
      tol: this.tol,
    };
  }

  setParams(_p: Record<string, unknown>): this {
    return this;
  }

  private solveLinear(A: Float64Array, b: Float64Array, n: number): Float64Array {
    // Gaussian elimination with partial pivoting
    const aug = new Float64Array(n * (n + 1));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) aug[i * (n + 1) + j] = A[i * n + j] ?? 0;
      aug[i * (n + 1) + n] = b[i] ?? 0;
    }

    for (let col = 0; col < n; col++) {
      let maxRow = col;
      let maxVal = Math.abs(aug[col * (n + 1) + col] ?? 0);
      for (let row = col + 1; row < n; row++) {
        const val = Math.abs(aug[row * (n + 1) + col] ?? 0);
        if (val > maxVal) {
          maxVal = val;
          maxRow = row;
        }
      }
      if (maxRow !== col) {
        for (let j = 0; j <= n; j++) {
          const tmp = aug[col * (n + 1) + j] ?? 0;
          aug[col * (n + 1) + j] = aug[maxRow * (n + 1) + j] ?? 0;
          aug[maxRow * (n + 1) + j] = tmp;
        }
      }
      const pivot = aug[col * (n + 1) + col] ?? 1;
      if (Math.abs(pivot) < 1e-20) continue;
      for (let row = col + 1; row < n; row++) {
        const factor = (aug[row * (n + 1) + col] ?? 0) / pivot;
        for (let j = col; j <= n; j++) {
          aug[row * (n + 1) + j] =
            (aug[row * (n + 1) + j] ?? 0) - factor * (aug[col * (n + 1) + j] ?? 0);
        }
      }
    }

    const x = new Float64Array(n);
    for (let i = n - 1; i >= 0; i--) {
      let sum = aug[i * (n + 1) + n] ?? 0;
      for (let j = i + 1; j < n; j++) {
        sum -= (aug[i * (n + 1) + j] ?? 0) * (x[j] ?? 0);
      }
      const diag = aug[i * (n + 1) + i] ?? 1;
      x[i] = Math.abs(diag) > 1e-20 ? sum / diag : 0;
    }
    return x;
  }
}
