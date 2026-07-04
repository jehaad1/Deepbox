/**
 * Bayesian Ridge Regression.
 *
 * Fits a linear model using Bayesian inference with automatic
 * regularization parameter tuning via evidence maximization (type-II ML).
 * The model assumes Gaussian priors on the weights and noise.
 *
 * @module ml/linear/BayesianRidge
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError, ShapeError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { assertContiguous, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Regressor } from "../base";

/**
 * Bayesian Ridge Regression with automatic regularization.
 *
 * Iteratively updates the precision parameters alpha (noise) and
 * lambda (weights) using evidence maximization. Provides uncertainty
 * estimates via posterior covariance.
 *
 * @example
 * ```ts
 * import { BayesianRidge } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1], [2], [3], [4], [5]]);
 * const y = tensor([2.1, 3.9, 6.1, 7.9, 10.2]);
 * const reg = new BayesianRidge();
 * reg.fit(X, y);
 * const pred = reg.predict(X);
 * ```
 */
export class BayesianRidge implements Regressor {
  private readonly maxIter: number;
  private readonly tol: number;
  private readonly fitInterceptOpt: boolean;
  private alphaInit: number;
  private lambdaInit: number;

  private coef_?: Float64Array;
  private intercept_ = 0;
  private alpha_ = 0; // noise precision
  private lambda_ = 0; // weight precision
  private nFeaturesIn_ = 0;
  private nIter_ = 0;
  private fitted = false;

  constructor(
    options: {
      readonly maxIter?: number;
      readonly tol?: number;
      readonly fitIntercept?: boolean;
      readonly alphaInit?: number;
      readonly lambdaInit?: number;
    } = {}
  ) {
    this.maxIter = options.maxIter ?? 300;
    this.tol = options.tol ?? 1e-3;
    this.fitInterceptOpt = options.fitIntercept ?? true;
    this.alphaInit = options.alphaInit ?? 1e-6;
    this.lambdaInit = options.lambdaInit ?? 1e-6;

    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new InvalidParameterError(
        "maxIter must be a positive integer",
        "maxIter",
        this.maxIter
      );
    }
    if (this.alphaInit <= 0) {
      throw new InvalidParameterError("alphaInit must be > 0", "alphaInit", this.alphaInit);
    }
    if (this.lambdaInit <= 0) {
      throw new InvalidParameterError("lambdaInit must be > 0", "lambdaInit", this.lambdaInit);
    }
  }

  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    this.nFeaturesIn_ = nFeatures;

    // Extract data
    const xData: Float64Array[] = [];
    for (let i = 0; i < nSamples; i++) {
      const row = new Float64Array(nFeatures);
      for (let j = 0; j < nFeatures; j++) {
        row[j] = Number(X.data[X.offset + i * nFeatures + j]);
      }
      xData.push(row);
    }
    const yData = new Float64Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      yData[i] = Number(y.data[y.offset + i]);
    }

    // Center data if fitting intercept
    const xMean = new Float64Array(nFeatures);
    let yMean = 0;
    if (this.fitInterceptOpt) {
      for (let j = 0; j < nFeatures; j++) {
        let s = 0;
        for (let i = 0; i < nSamples; i++) s += xData[i]![j] ?? 0;
        xMean[j] = s / nSamples;
      }
      for (let i = 0; i < nSamples; i++) {
        yMean += yData[i] ?? 0;
      }
      yMean /= nSamples;

      for (let i = 0; i < nSamples; i++) {
        for (let j = 0; j < nFeatures; j++) {
          xData[i]![j] = (xData[i]![j] ?? 0) - (xMean[j] ?? 0);
        }
        yData[i] = (yData[i] ?? 0) - yMean;
      }
    }

    // Compute X^T X
    const XtX = new Float64Array(nFeatures * nFeatures);
    for (let j = 0; j < nFeatures; j++) {
      for (let k = j; k < nFeatures; k++) {
        let s = 0;
        for (let i = 0; i < nSamples; i++) {
          s += (xData[i]![j] ?? 0) * (xData[i]![k] ?? 0);
        }
        XtX[j * nFeatures + k] = s;
        XtX[k * nFeatures + j] = s;
      }
    }

    // Compute X^T y
    const Xty = new Float64Array(nFeatures);
    for (let j = 0; j < nFeatures; j++) {
      let s = 0;
      for (let i = 0; i < nSamples; i++) {
        s += (xData[i]![j] ?? 0) * (yData[i] ?? 0);
      }
      Xty[j] = s;
    }

    // Compute eigenvalues of X^T X for efficient updates
    // Use power iteration to get eigenvalues (simplified)
    const eigenvalues = this.computeEigenvaluesXtX(XtX, nFeatures);

    let alpha = this.alphaInit;
    let lambda = this.lambdaInit;

    for (let iter = 0; iter < this.maxIter; iter++) {
      const prevAlpha = alpha;
      const prevLambda = lambda;

      // Compute posterior: Sigma = (alpha * X^T X + lambda * I)^{-1}
      // w = alpha * Sigma * X^T y
      const A = new Float64Array(nFeatures * nFeatures);
      for (let j = 0; j < nFeatures; j++) {
        for (let k = 0; k < nFeatures; k++) {
          A[j * nFeatures + k] = alpha * (XtX[j * nFeatures + k] ?? 0);
        }
        A[j * nFeatures + j] = (A[j * nFeatures + j] ?? 0) + lambda;
      }

      // Solve A * w = alpha * X^T y using Cholesky-like approach
      const Sigma = this.invertSymmetric(A, nFeatures);
      const w = new Float64Array(nFeatures);
      for (let j = 0; j < nFeatures; j++) {
        let s = 0;
        for (let k = 0; k < nFeatures; k++) {
          s += (Sigma[j * nFeatures + k] ?? 0) * (Xty[k] ?? 0);
        }
        w[j] = alpha * s;
      }

      // Compute gamma = sum_i (alpha * eigenvalue_i) / (alpha * eigenvalue_i + lambda)
      let gamma = 0;
      for (let i = 0; i < nFeatures; i++) {
        const aei = alpha * (eigenvalues[i] ?? 0);
        gamma += aei / (aei + lambda);
      }

      // Update alpha: alpha = n / (||y - Xw||^2 + trace(Sigma * X^T X) * alpha_old / alpha_old)
      // Simplified: alpha = (n - gamma) / ||y - Xw||^2
      let residualSS = 0;
      for (let i = 0; i < nSamples; i++) {
        let pred = 0;
        for (let j = 0; j < nFeatures; j++) {
          pred += (xData[i]![j] ?? 0) * (w[j] ?? 0);
        }
        const r = (yData[i] ?? 0) - pred;
        residualSS += r * r;
      }

      const newAlpha = residualSS > 1e-20 ? (nSamples - gamma) / residualSS : 1e10; // near-perfect fit => very high precision
      alpha = Math.max(1e-10, Number.isFinite(newAlpha) ? newAlpha : 1e10);

      // Update lambda: lambda = gamma / ||w||^2
      let wNormSq = 0;
      for (let j = 0; j < nFeatures; j++) {
        wNormSq += (w[j] ?? 0) * (w[j] ?? 0);
      }
      const newLambda = gamma / Math.max(wNormSq, 1e-20);
      lambda = Math.max(1e-10, Number.isFinite(newLambda) ? newLambda : 1e-10);

      this.nIter_ = iter + 1;

      // Check convergence
      if (
        Math.abs(alpha - prevAlpha) < this.tol * Math.max(1, alpha) &&
        Math.abs(lambda - prevLambda) < this.tol * Math.max(1, lambda)
      ) {
        break;
      }
    }

    // Final weights with converged alpha, lambda
    const Afinal = new Float64Array(nFeatures * nFeatures);
    for (let j = 0; j < nFeatures; j++) {
      for (let k = 0; k < nFeatures; k++) {
        Afinal[j * nFeatures + k] = alpha * (XtX[j * nFeatures + k] ?? 0);
      }
      Afinal[j * nFeatures + j] = (Afinal[j * nFeatures + j] ?? 0) + lambda;
    }
    const Sigma = this.invertSymmetric(Afinal, nFeatures);
    const w = new Float64Array(nFeatures);
    for (let j = 0; j < nFeatures; j++) {
      let s = 0;
      for (let k = 0; k < nFeatures; k++) {
        s += (Sigma[j * nFeatures + k] ?? 0) * (Xty[k] ?? 0);
      }
      w[j] = alpha * s;
    }

    this.coef_ = w;
    this.alpha_ = alpha;
    this.lambda_ = lambda;

    // Compute intercept
    if (this.fitInterceptOpt) {
      let dot = 0;
      for (let j = 0; j < nFeatures; j++) {
        dot += (xMean[j] ?? 0) * (w[j] ?? 0);
      }
      this.intercept_ = yMean - dot;
    } else {
      this.intercept_ = 0;
    }

    this.fitted = true;
    return this;
  }

  private computeEigenvaluesXtX(XtX: Float64Array, n: number): Float64Array {
    // Simple eigenvalue computation using Jacobi iteration (for small n)
    // For large n this is slow but sufficient for typical ML use cases
    const A = new Float64Array(XtX);
    const eigenvalues = new Float64Array(n);

    for (let sweep = 0; sweep < 100; sweep++) {
      let offDiagSum = 0;
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          offDiagSum += Math.abs(A[i * n + j] ?? 0);
        }
      }
      if (offDiagSum < 1e-12) break;

      for (let p = 0; p < n; p++) {
        for (let q = p + 1; q < n; q++) {
          const apq = A[p * n + q] ?? 0;
          if (Math.abs(apq) < 1e-15) continue;

          const app = A[p * n + p] ?? 0;
          const aqq = A[q * n + q] ?? 0;
          const theta = 0.5 * Math.atan2(2 * apq, app - aqq);
          const c = Math.cos(theta);
          const s = Math.sin(theta);

          // Apply Givens rotation
          for (let i = 0; i < n; i++) {
            const aip = A[i * n + p] ?? 0;
            const aiq = A[i * n + q] ?? 0;
            A[i * n + p] = c * aip + s * aiq;
            A[i * n + q] = -s * aip + c * aiq;
          }
          for (let j = 0; j < n; j++) {
            const apj = A[p * n + j] ?? 0;
            const aqj = A[q * n + j] ?? 0;
            A[p * n + j] = c * apj + s * aqj;
            A[q * n + j] = -s * apj + c * aqj;
          }
        }
      }
    }

    for (let i = 0; i < n; i++) {
      eigenvalues[i] = Math.max(0, A[i * n + i] ?? 0);
    }
    return eigenvalues;
  }

  private invertSymmetric(A: Float64Array, n: number): Float64Array {
    // Invert via Gauss-Jordan elimination
    const aug = new Float64Array(n * 2 * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        aug[i * 2 * n + j] = A[i * n + j] ?? 0;
      }
      aug[i * 2 * n + n + i] = 1;
    }

    for (let col = 0; col < n; col++) {
      // Partial pivoting
      let maxVal = Math.abs(aug[col * 2 * n + col] ?? 0);
      let maxRow = col;
      for (let row = col + 1; row < n; row++) {
        const val = Math.abs(aug[row * 2 * n + col] ?? 0);
        if (val > maxVal) {
          maxVal = val;
          maxRow = row;
        }
      }
      if (maxRow !== col) {
        for (let j = 0; j < 2 * n; j++) {
          const tmp = aug[col * 2 * n + j] ?? 0;
          aug[col * 2 * n + j] = aug[maxRow * 2 * n + j] ?? 0;
          aug[maxRow * 2 * n + j] = tmp;
        }
      }

      const pivot = aug[col * 2 * n + col] ?? 1;
      if (Math.abs(pivot) < 1e-20) continue;

      for (let j = 0; j < 2 * n; j++) {
        aug[col * 2 * n + j] = (aug[col * 2 * n + j] ?? 0) / pivot;
      }

      for (let row = 0; row < n; row++) {
        if (row === col) continue;
        const factor = aug[row * 2 * n + col] ?? 0;
        for (let j = 0; j < 2 * n; j++) {
          aug[row * 2 * n + j] = (aug[row * 2 * n + j] ?? 0) - factor * (aug[col * 2 * n + j] ?? 0);
        }
      }
    }

    const inv = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        inv[i * n + j] = aug[i * 2 * n + n + j] ?? 0;
      }
    }
    return inv;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted) throw new NotFittedError("BayesianRidge must be fitted before predict");
    validatePredictInputs(X, this.nFeaturesIn_, "BayesianRidge");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const w = this.coef_!;
    const result = new Float64Array(nSamples);

    for (let i = 0; i < nSamples; i++) {
      let pred = this.fitInterceptOpt ? this.intercept_ : 0;
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
    if (!this.fitted) throw new NotFittedError("BayesianRidge must be fitted to access coef");
    return this.coef_!;
  }

  get intercept(): number {
    if (!this.fitted) throw new NotFittedError("BayesianRidge must be fitted to access intercept");
    return this.intercept_;
  }

  get alpha(): number {
    if (!this.fitted) throw new NotFittedError("BayesianRidge must be fitted to access alpha");
    return this.alpha_;
  }

  get lambda(): number {
    if (!this.fitted) throw new NotFittedError("BayesianRidge must be fitted to access lambda");
    return this.lambda_;
  }

  get nIter(): number {
    if (!this.fitted) throw new NotFittedError("BayesianRidge must be fitted to access nIter");
    return this.nIter_;
  }

  getParams(): Record<string, unknown> {
    return {
      maxIter: this.maxIter,
      tol: this.tol,
      fitIntercept: this.fitInterceptOpt,
      alphaInit: this.alphaInit,
      lambdaInit: this.lambdaInit,
    };
  }

  setParams(_params: Record<string, unknown>): this {
    return this;
  }
}
