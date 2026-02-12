import { DataValidationError, InvalidParameterError, NotFittedError, ShapeError } from "../../core";
import { cholesky } from "../../linalg/decomposition/cholesky";
import { svd } from "../../linalg/decomposition/svd";
import { solveTriangular } from "../../linalg/solvers/solve";
import { dot, type Tensor, tensor, transpose } from "../../ndarray";
import { assertContiguous, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Regressor } from "../base";

/**
 * Ridge Regression (L2 Regularized Linear Regression).
 *
 * Ridge regression addresses multicollinearity by adding a penalty term
 * (L2 regularization) to the loss function.
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
    solver?: "auto" | "svd" | "cholesky" | "lsqr" | "sag";
    maxIter?: number;
    tol?: number;
  };

  /** Model coefficients (weights) after fitting - shape (n_features,) */
  private coef_?: Tensor;

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
   * @param options - Configuration options
   * @param options.alpha - Regularization strength (default: 1.0). Must be >= 0.
   * @param options.fitIntercept - Whether to calculate the intercept (default: true).
   * @param options.normalize - Whether to normalize features before regression (default: false).
   * @param options.solver - Solver to use (default: 'auto'). Options: 'auto', 'svd', 'cholesky', 'lsqr', 'sag'.
   * @param options.maxIter - Maximum number of iterations for iterative solvers (default: 1000)
   * @param options.tol - Tolerance for stopping criterion (default: 1e-4)
   */
  constructor(
    options: {
      readonly alpha?: number;
      readonly fitIntercept?: boolean;
      readonly normalize?: boolean;
      readonly solver?: "auto" | "svd" | "cholesky" | "lsqr" | "sag";
      readonly maxIter?: number;
      readonly tol?: number;
    } = {}
  ) {
    this.options = { ...options };
  }

  /**
   * Fit Ridge regression model.
   *
   * Solves the regularized least squares problem:
   * minimize ||y - Xw||² + α||w||²
   *
   * Uses the closed-form solution:
   * w = (X^T X + αI)^(-1) X^T y
   *
   * **Time Complexity**: O(n²p + p³) where n = samples, p = features
   * **Space Complexity**: O(p²)
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Target values of shape (n_samples,)
   * @returns this - The fitted estimator for method chaining
   * @throws {ShapeError} If X is not 2D or y is not 1D
   * @throws {ShapeError} If X and y have different number of samples
   * @throws {DataValidationError} If X or y contain NaN/Inf values
   * @throws {DataValidationError} If X or y are empty
   * @throws {InvalidParameterError} If alpha < 0
   */
  fit(X: Tensor, y: Tensor): this {
    // Validate inputs (dimensions, empty data, NaN/Inf)
    validateFitInputs(X, y);
    this.nIter_ = undefined;

    // Extract and validate regularization parameter
    const alpha = this.options.alpha ?? 1.0;
    if (!(alpha >= 0)) {
      throw new InvalidParameterError(`alpha must be >= 0; received ${alpha}`, "alpha", alpha);
    }

    // Determine whether to fit intercept
    const fitIntercept = this.options.fitIntercept ?? true;

    // Extract dimensions: m = number of samples, n = number of features
    const m = X.shape[0] ?? 0;
    const n = X.shape[1] ?? 0;

    // Store number of features for prediction validation
    this.nFeaturesIn_ = n;

    // Compute means for centering (if fitIntercept is true)
    // Centering improves numerical stability and allows intercept calculation
    // Note: By centering X and y, we ensure the intercept is not regularized.
    // The regularization penalty α||w||² only applies to the coefficients,
    // not the intercept term. This is the standard Ridge regression behavior.
    let yMean = 0;
    const xMean = new Array<number>(n).fill(0);

    if (fitIntercept) {
      // Compute sum of y values
      for (let i = 0; i < m; i++) {
        yMean += Number(y.data[y.offset + i] ?? 0);
      }

      // Compute sum of each feature column
      for (let i = 0; i < m; i++) {
        const rowBase = X.offset + i * n;
        for (let j = 0; j < n; j++) {
          xMean[j] = (xMean[j] ?? 0) + Number(X.data[rowBase + j] ?? 0);
        }
      }

      // Convert sums to means by dividing by number of samples
      const invM = m === 0 ? 0 : 1 / m;
      yMean *= invM;
      for (let j = 0; j < n; j++) {
        xMean[j] = (xMean[j] ?? 0) * invM;
      }
    }

    const normalize = this.options.normalize ?? false;
    const maxIter = this.options.maxIter ?? 1000;
    const tol = this.options.tol ?? 1e-4;

    let xScale: number[] | undefined;
    if (normalize) {
      xScale = new Array<number>(n).fill(0);
      for (let i = 0; i < m; i++) {
        const rowBase = X.offset + i * n;
        for (let j = 0; j < n; j++) {
          const centered = Number(X.data[rowBase + j] ?? 0) - (fitIntercept ? (xMean[j] ?? 0) : 0);
          xScale[j] = (xScale[j] ?? 0) + centered * centered;
        }
      }
      for (let j = 0; j < n; j++) {
        xScale[j] = Math.sqrt(xScale[j] ?? 0);
      }
    }

    const getX = (sampleIndex: number, featureIndex: number): number => {
      const raw = Number(X.data[X.offset + sampleIndex * n + featureIndex] ?? 0);
      const centered = raw - (fitIntercept ? (xMean[featureIndex] ?? 0) : 0);
      if (normalize && xScale) {
        const s = xScale[featureIndex] ?? 0;
        return s === 0 ? 0 : centered / s;
      }
      return centered;
    };

    const getY = (sampleIndex: number): number => {
      const raw = Number(y.data[y.offset + sampleIndex] ?? 0);
      return fitIntercept ? raw - yMean : raw;
    };

    // Solve the linear system (X^T X + αI) w = X^T y
    // This gives us the optimal coefficients w
    let coefTensor: Tensor;
    const solver = this.options.solver ?? "auto";

    if (solver === "sag") {
      const res = this.solveSag(getX, getY, m, n, alpha, maxIter, tol);
      coefTensor = tensor(res.x);
      this.nIter_ = res.nIter;
    } else {
      // Compute X^T X + αI (Gram matrix with regularization)
      // This is the core of the Ridge regression solution
      // Time complexity: O(n²m) for computing X^T X
      const XTX = Array(n)
        .fill(0)
        .map(() => Array(n).fill(0));

      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          let sum = 0;

          // Compute (X^T X)[i,j] = Σ_k X[k,i] * X[k,j]
          for (let k = 0; k < m; k++) {
            const xi = getX(k, i);
            const xj = getX(k, j);
            sum += xi * xj;
          }

          // Add regularization term αI to diagonal
          // This ensures the matrix is positive definite and invertible
          const xtxRow = XTX[i];
          if (xtxRow) xtxRow[j] = sum + (i === j ? alpha : 0);
        }
      }

      // Compute X^T y (feature-target correlation vector)
      // Time complexity: O(nm)
      const XTy = new Array<number>(n).fill(0);

      for (let i = 0; i < n; i++) {
        let sum = 0;

        // Compute (X^T y)[i] = Σ_j X[j,i] * y[j]
        for (let j = 0; j < m; j++) {
          const yVal = getY(j);
          const xVal = getX(j, i);
          sum += xVal * yVal;
        }
        XTy[i] = sum;
      }

      if (solver === "lsqr") {
        const res = this.solveConjugateGradient(XTX, XTy, maxIter, tol);
        coefTensor = tensor(res.x);
        this.nIter_ = res.nIter;
      } else if (solver === "cholesky" || solver === "auto") {
        try {
          const xtxTensor = tensor(XTX);
          const xtyTensor = tensor(XTy);
          const L = cholesky(xtxTensor);
          const y_ = solveTriangular(L, xtyTensor, true);
          coefTensor = solveTriangular(transpose(L), y_, false);
        } catch (e) {
          if (solver === "auto") {
            // Fallback to Gaussian elimination
            const res = this.solveLinearSystem(XTX, XTy);
            coefTensor = tensor(res);
          } else {
            throw e;
          }
        }
      } else if (solver === "svd") {
        const xtxTensor = tensor(XTX);
        const xtyTensor = tensor(XTy);
        const [U, s, Vt] = svd(xtxTensor);

        // w = V * S^-1 * U^T * y
        const Ut = transpose(U);
        const Uty = dot(Ut, xtyTensor);

        const sData = s.data;
        if (!(sData instanceof Float64Array)) {
          throw new DataValidationError("svd returned non-float64 singular values");
        }
        const scaledData = new Float64Array(Uty.size);
        for (let i = 0; i < Uty.size; i++) {
          const val = Number(Uty.data[Uty.offset + i]);
          const sigma = sData[i] ?? 0;
          scaledData[i] = Math.abs(sigma) > 1e-15 ? val / sigma : 0;
        }
        const scaled = tensor(scaledData);

        const V = transpose(Vt);
        coefTensor = dot(V, scaled);
      } else {
        const res = this.solveLinearSystem(XTX, XTy);
        coefTensor = tensor(res);
      }
    }

    if (normalize && xScale) {
      coefTensor = this.rescaleCoefs(coefTensor, xScale);
    }

    this.coef_ = coefTensor;

    // Compute intercept if needed
    // intercept = mean(y) - mean(X) @ coef
    // This accounts for the centering we did earlier
    if (fitIntercept) {
      let xMeanDotW = 0;
      for (let j = 0; j < n; j++) {
        const wj = Number(coefTensor.data[coefTensor.offset + j] ?? 0);
        xMeanDotW += (xMean[j] ?? 0) * wj;
      }
      this.intercept_ = yMean - xMeanDotW;
    } else {
      this.intercept_ = 0;
    }

    // Mark model as fitted
    this.fitted = true;
    return this;
  }

  /**
   * Solve linear system Ax = b using Gaussian elimination with partial pivoting.
   *
   * This is a numerically stable method for solving dense linear systems.
   * For Ridge regression, A = X^T X + αI is symmetric positive definite,
   * so Cholesky decomposition would be more efficient, but Gaussian elimination
   * is more general and still provides good numerical stability.
   *
   * **Algorithm**:
   * 1. Forward elimination: Convert A to upper triangular form
   * 2. Partial pivoting: Swap rows to avoid division by small numbers
   * 3. Back substitution: Solve for x from bottom to top
   *
   * **Time Complexity**: O(n³)
   * **Space Complexity**: O(n²)
   *
   * @param A - Coefficient matrix (n × n)
   * @param b - Right-hand side vector (n × 1)
   * @returns Solution vector x such that Ax = b
   */
  private solveLinearSystem(A: number[][], b: number[]): number[] {
    const n = A.length;

    // Create augmented matrix [A | b]
    // This allows us to perform row operations on both A and b simultaneously
    const aug = A.map((row, i) => [...row, b[i] ?? 0]);
    let maxAbs = 0;
    for (let i = 0; i < n; i++) {
      const row = aug[i];
      if (!row) continue;
      for (let j = 0; j < n; j++) {
        const v = Math.abs(row[j] ?? 0);
        if (v > maxAbs) maxAbs = v;
      }
    }
    if (maxAbs === 0 || !Number.isFinite(maxAbs)) {
      throw new DataValidationError("Matrix is singular or ill-conditioned");
    }
    const tol = Number.EPSILON * n * maxAbs;

    // Forward elimination with partial pivoting
    for (let i = 0; i < n; i++) {
      // Find pivot: row with largest absolute value in column i
      // This improves numerical stability by avoiding division by small numbers
      let maxRow = i;
      for (let k = i + 1; k < n; k++) {
        if (Math.abs(aug[k]?.[i] ?? 0) > Math.abs(aug[maxRow]?.[i] ?? 0)) {
          maxRow = k;
        }
      }

      // Swap rows i and maxRow
      const augI = aug[i] ?? [];
      const augMax = aug[maxRow] ?? [];
      aug[i] = augMax;
      aug[maxRow] = augI;

      const pivot = aug[i]?.[i] ?? 0;
      if (!Number.isFinite(pivot) || Math.abs(pivot) <= tol) {
        throw new DataValidationError("Matrix is singular or ill-conditioned");
      }

      // Eliminate column i in rows below i
      for (let k = i + 1; k < n; k++) {
        // Compute multiplier: c = A[k,i] / A[i,i]
        const c = (aug[k]?.[i] ?? 0) / pivot;
        const augK = aug[k];

        if (augK) {
          // Subtract c * row_i from row_k
          for (let j = i; j <= n; j++) {
            augK[j] = (augK[j] ?? 0) - c * (aug[i]?.[j] ?? 0);
          }
        }
      }
    }

    // Back substitution: solve upper triangular system
    const x = Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
      // Start with b[i]
      x[i] = aug[i]?.[n] ?? 0;

      // Subtract contributions from already-solved variables
      for (let j = i + 1; j < n; j++) {
        x[i] = (x[i] ?? 0) - (aug[i]?.[j] ?? 0) * (x[j] ?? 0);
      }

      // Divide by diagonal element
      const diag = aug[i]?.[i] ?? 0;
      if (!Number.isFinite(diag) || Math.abs(diag) <= tol) {
        throw new DataValidationError("Matrix is singular or ill-conditioned");
      }
      x[i] = (x[i] ?? 0) / diag;
    }

    return x;
  }

  private solveConjugateGradient(
    A: number[][],
    b: number[],
    maxIter: number,
    tol: number
  ): { x: number[]; nIter: number } {
    const n = A.length;
    const x = new Array<number>(n).fill(0);
    const r = new Array<number>(n).fill(0);

    let rsOld = 0;
    for (let i = 0; i < n; i++) {
      const bi = b[i] ?? 0;
      r[i] = bi;
      rsOld += bi * bi;
    }

    if (rsOld === 0) {
      return { x, nIter: 0 };
    }

    const p = r.slice();
    const tolSq = tol * tol;
    let nIter = 0;

    for (let iter = 0; iter < maxIter; iter++) {
      const Ap = new Array<number>(n).fill(0);
      for (let i = 0; i < n; i++) {
        let sum = 0;
        const row = A[i];
        if (!row) continue;
        for (let j = 0; j < n; j++) {
          sum += (row[j] ?? 0) * (p[j] ?? 0);
        }
        Ap[i] = sum;
      }

      let denom = 0;
      for (let i = 0; i < n; i++) {
        denom += (p[i] ?? 0) * (Ap[i] ?? 0);
      }
      if (!Number.isFinite(denom) || denom === 0) {
        throw new DataValidationError(
          "Conjugate gradient failed: denominator is zero or non-finite"
        );
      }

      const alpha = rsOld / denom;
      for (let i = 0; i < n; i++) {
        x[i] = (x[i] ?? 0) + alpha * (p[i] ?? 0);
        r[i] = (r[i] ?? 0) - alpha * (Ap[i] ?? 0);
      }

      let rsNew = 0;
      for (let i = 0; i < n; i++) {
        const ri = r[i] ?? 0;
        rsNew += ri * ri;
      }
      nIter = iter + 1;
      if (rsNew < tolSq) {
        break;
      }

      const beta = rsNew / rsOld;
      for (let i = 0; i < n; i++) {
        p[i] = (r[i] ?? 0) + beta * (p[i] ?? 0);
      }
      rsOld = rsNew;
    }

    return { x, nIter };
  }

  private solveSag(
    getX: (sampleIndex: number, featureIndex: number) => number,
    getY: (sampleIndex: number) => number,
    nSamples: number,
    nFeatures: number,
    alpha: number,
    maxIter: number,
    tol: number
  ): { x: number[]; nIter: number } {
    const w = new Array<number>(nFeatures).fill(0);
    const avgGrad = new Array<number>(nFeatures).fill(0);
    const residuals = new Array<number>(nSamples).fill(0);

    let maxNormSq = 0;
    for (let i = 0; i < nSamples; i++) {
      let normSq = 0;
      for (let j = 0; j < nFeatures; j++) {
        const xij = getX(i, j);
        normSq += xij * xij;
      }
      if (normSq > maxNormSq) {
        maxNormSq = normSq;
      }
    }

    const scale = nSamples === 0 ? 1 : nSamples;
    const L = maxNormSq * scale + alpha;
    const step = L > 0 ? 1 / L : 1;

    let nIter = 0;
    for (let iter = 0; iter < maxIter; iter++) {
      let maxUpdate = 0;

      for (let i = 0; i < nSamples; i++) {
        let dotProd = 0;
        for (let j = 0; j < nFeatures; j++) {
          dotProd += (w[j] ?? 0) * getX(i, j);
        }

        const yi = getY(i);
        const newResidual = dotProd - yi;
        const delta = newResidual - (residuals[i] ?? 0);
        residuals[i] = newResidual;

        if (delta !== 0) {
          for (let j = 0; j < nFeatures; j++) {
            avgGrad[j] = (avgGrad[j] ?? 0) + delta * getX(i, j);
          }
        }

        for (let j = 0; j < nFeatures; j++) {
          const grad = (avgGrad[j] ?? 0) + alpha * (w[j] ?? 0);
          const update = step * grad;
          w[j] = (w[j] ?? 0) - update;
          if (Math.abs(update) > maxUpdate) {
            maxUpdate = Math.abs(update);
          }
        }
      }

      nIter = iter + 1;
      if (maxUpdate < tol) {
        break;
      }
    }

    return { x: w, nIter };
  }

  private rescaleCoefs(coef: Tensor, scale: number[]): Tensor {
    const nFeatures = coef.shape[0] ?? 0;
    const result: number[] = [];
    for (let j = 0; j < nFeatures; j++) {
      const c = Number(coef.data[coef.offset + j] ?? 0);
      const s = scale[j] ?? 1;
      result.push(s === 0 ? 0 : c / s);
    }
    return tensor(result);
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
   * @returns Predicted values of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predict(X: Tensor): Tensor {
    // Check if model has been fitted
    if (!this.fitted || !this.coef_) {
      throw new NotFittedError("Ridge must be fitted before prediction");
    }

    // Validate input
    validatePredictInputs(X, this.nFeaturesIn_ ?? 0, "Ridge");

    const m = X.shape[0] ?? 0; // Number of samples to predict
    const n = X.shape[1] ?? 0; // Number of features
    const pred = Array(m).fill(0);

    // Compute predictions: ŷ[i] = Σ_j X[i,j] * coef[j] + intercept
    for (let i = 0; i < m; i++) {
      let sum = this.intercept_ ?? 0; // Start with intercept

      // Add weighted sum of features
      for (let j = 0; j < n; j++) {
        sum +=
          Number(X.data[X.offset + i * n + j] ?? 0) *
          Number(this.coef_.data[this.coef_.offset + j] ?? 0);
      }
      pred[i] = sum;
    }

    return tensor(pred);
  }

  /**
   * Return the coefficient of determination R² of the prediction.
   *
   * R² (R-squared) measures the proportion of variance in y explained by the model.
   * Formula: R² = 1 - (SS_res / SS_tot)
   *
   * Where:
   * - SS_res = Σ(y_true - y_pred)² (residual sum of squares)
   * - SS_tot = Σ(y_true - y_mean)² (total sum of squares)
   *
   * **Interpretation**:
   * - R² = 1: Perfect predictions
   * - R² = 0: Model performs as well as predicting the mean
   * - R² < 0: Model performs worse than predicting the mean
   *
   * **Time Complexity**: O(n) where n = number of samples
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True values of shape (n_samples,)
   * @returns R² score (best possible score is 1.0, can be negative)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1-dimensional
   */
  score(X: Tensor, y: Tensor): number {
    // Check if model has been fitted
    if (!this.fitted) {
      throw new NotFittedError("Ridge must be fitted before scoring");
    }

    // Validate y dimensions
    if (y.ndim !== 1) {
      throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    }
    assertContiguous(y, "y");
    for (let i = 0; i < y.size; i++) {
      const val = y.data[y.offset + i] ?? 0;
      if (!Number.isFinite(val)) {
        throw new DataValidationError("y contains non-finite values (NaN or Inf)");
      }
    }

    // Get predictions
    const pred = this.predict(X);
    if (pred.size !== y.size) {
      throw new ShapeError(
        `X and y must have the same number of samples; got X=${pred.size}, y=${y.size}`
      );
    }

    let ssRes = 0; // Residual sum of squares
    let ssTot = 0; // Total sum of squares
    let yMean = 0; // Mean of y

    // Compute mean of y
    for (let i = 0; i < y.size; i++) {
      yMean += Number(y.data[y.offset + i] ?? 0);
    }
    yMean /= y.size;

    // Compute SS_res and SS_tot
    for (let i = 0; i < y.size; i++) {
      const yVal = Number(y.data[y.offset + i] ?? 0);
      const predVal = Number(pred.data[pred.offset + i] ?? 0);

      // Residual sum of squares: measures prediction error
      ssRes += (yVal - predVal) ** 2;

      // Total sum of squares: measures total variance in y
      ssTot += (yVal - yMean) ** 2;
    }

    // Handle edge case: constant y (zero variance)
    // If y is constant and predictions match, R² = 1
    // If y is constant but predictions don't match, R² = 0
    if (ssTot === 0) {
      return ssRes === 0 ? 1.0 : 0.0;
    }

    // Compute R² = 1 - (SS_res / SS_tot)
    return 1 - ssRes / ssTot;
  }

  /**
   * Get parameters for this estimator.
   *
   * Returns a copy of all hyperparameters set during construction or via setParams.
   *
   * @returns Object containing all parameters with their current values
   */
  getParams(): Record<string, unknown> {
    return { ...this.options };
  }

  /**
   * Set the parameters of this estimator.
   *
   * Allows modifying hyperparameters after construction.
   * Note: Changing parameters requires refitting the model.
   *
   * @param params - Dictionary of parameters to set
   * @returns this - The estimator for method chaining
   * @throws {TypeError} If parameter value has wrong type
   * @throws {Error} If parameter name is unknown or value is invalid
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
          if (
            value !== "auto" &&
            value !== "svd" &&
            value !== "cholesky" &&
            value !== "lsqr" &&
            value !== "sag"
          ) {
            throw new InvalidParameterError(`Invalid solver: ${String(value)}`, "solver", value);
          }
          this.options.solver = value;
          break;

        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }

  /**
   * Get the model coefficients (weights).
   *
   * @returns Coefficient tensor of shape (n_features,)
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
   * @returns Intercept value
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
