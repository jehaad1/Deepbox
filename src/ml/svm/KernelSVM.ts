/**
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { DataValidationError, InvalidParameterError, NotFittedError, ShapeError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { __random } from "../../random/random";
import { assertContiguous, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Classifier, Regressor } from "../base";

type KernelType = "rbf" | "linear" | "poly" | "sigmoid";

/**
 * Compute kernel value between two vectors.
 */
function kernelValue(
  xi: number[],
  xj: number[],
  kernel: KernelType,
  gamma: number,
  coef0: number,
  degree: number
): number {
  let dot = 0;
  for (let k = 0; k < xi.length; k++) {
    dot += (xi[k] ?? 0) * (xj[k] ?? 0);
  }

  switch (kernel) {
    case "linear":
      return dot;
    case "rbf": {
      let sqDist = 0;
      for (let k = 0; k < xi.length; k++) {
        const d = (xi[k] ?? 0) - (xj[k] ?? 0);
        sqDist += d * d;
      }
      return Math.exp(-gamma * sqDist);
    }
    case "poly":
      return (gamma * dot + coef0) ** degree;
    case "sigmoid":
      return Math.tanh(gamma * dot + coef0);
    default:
      return dot;
  }
}

/**
 * Precompute kernel matrix for training data.
 */
function computeKernelMatrix(
  XData: number[][],
  nSamples: number,
  kernel: KernelType,
  gamma: number,
  coef0: number,
  degree: number
): Float64Array {
  const K = new Float64Array(nSamples * nSamples);
  for (let i = 0; i < nSamples; i++) {
    for (let j = i; j < nSamples; j++) {
      const val = kernelValue(XData[i]!, XData[j]!, kernel, gamma, coef0, degree);
      K[i * nSamples + j] = val;
      K[j * nSamples + i] = val;
    }
  }
  return K;
}

/**
 * Support Vector Classification with kernel trick.
 *
 * Uses Simplified SMO (Sequential Minimal Optimization) to solve the dual
 * quadratic programming problem. Supports RBF, polynomial, sigmoid, and linear kernels.
 *
 * @example
 * ```ts
 * import { SVC } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[0, 0], [1, 1], [1, 0], [0, 1]]);
 * const y = tensor([0, 0, 1, 1]);
 *
 * const svc = new SVC({ kernel: 'rbf', C: 1.0 });
 * svc.fit(X, y);
 * const predictions = svc.predict(X);
 * ```
 */
export class SVC implements Classifier {
  private C: number;
  private kernel: KernelType;
  private gamma: number | "scale" | "auto";
  private coef0: number;
  private degree: number;
  private maxIter: number;
  private tol: number;

  private gamma_: number = 1;
  private supportVectors_?: number[][];
  private supportAlphas_?: number[];
  private supportLabels_?: number[];
  private bias_ = 0;
  private classLabels: number[] = [];
  // For multiclass OvR
  private models: Array<{
    sv: number[][];
    alphas: number[];
    labels: number[];
    bias: number;
    posClass: number;
  }> = [];
  private nFeatures = 0;
  private fitted = false;

  constructor(
    options: {
      readonly C?: number;
      readonly kernel?: KernelType;
      readonly gamma?: number | "scale" | "auto";
      readonly coef0?: number;
      readonly degree?: number;
      readonly maxIter?: number;
      readonly tol?: number;
    } = {}
  ) {
    this.C = options.C ?? 1.0;
    this.kernel = options.kernel ?? "rbf";
    this.gamma = options.gamma ?? "scale";
    this.coef0 = options.coef0 ?? 0;
    this.degree = options.degree ?? 3;
    this.maxIter = options.maxIter ?? 1000;
    this.tol = options.tol ?? 1e-3;

    if (!Number.isFinite(this.C) || this.C <= 0) {
      throw new InvalidParameterError("C must be positive", "C", this.C);
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter <= 0) {
      throw new InvalidParameterError(
        "maxIter must be a positive integer",
        "maxIter",
        this.maxIter
      );
    }
  }

  private resolveGamma(nFeatures: number, XData: number[][], nSamples: number): number {
    if (typeof this.gamma === "number") return this.gamma;
    if (this.gamma === "auto") return 1 / nFeatures;
    // "scale": 1 / (nFeatures * var(X))
    let mean = 0;
    const total = nSamples * nFeatures;
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        mean += XData[i]![j] ?? 0;
      }
    }
    mean /= total;
    let variance = 0;
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        const d = (XData[i]![j] ?? 0) - mean;
        variance += d * d;
      }
    }
    variance /= total;
    return variance > 0 ? 1 / (nFeatures * variance) : 1;
  }

  /**
   * Simplified SMO for binary SVM.
   * yMapped should be {-1, +1}.
   */
  private solveBinarySMO(
    XData: number[][],
    yMapped: number[],
    nSamples: number
  ): { sv: number[][]; alphas: number[]; labels: number[]; bias: number } {
    const K = computeKernelMatrix(
      XData,
      nSamples,
      this.kernel,
      this.gamma_,
      this.coef0,
      this.degree
    );
    const alphas = new Float64Array(nSamples);
    let b = 0;

    for (let iter = 0; iter < this.maxIter; iter++) {
      let numChanged = 0;

      for (let i = 0; i < nSamples; i++) {
        // Compute f(x_i)
        let fi = -b;
        for (let j = 0; j < nSamples; j++) {
          fi += (alphas[j] ?? 0) * (yMapped[j] ?? 0) * (K[j * nSamples + i] ?? 0);
        }
        const yi = yMapped[i] ?? 0;
        const Ei = fi - yi;

        if (
          (yi * Ei < -this.tol && (alphas[i] ?? 0) < this.C) ||
          (yi * Ei > this.tol && (alphas[i] ?? 0) > 0)
        ) {
          // Select j randomly, j != i
          let j = Math.floor(__random() * (nSamples - 1));
          if (j >= i) j++;

          let fj = -b;
          for (let k = 0; k < nSamples; k++) {
            fj += (alphas[k] ?? 0) * (yMapped[k] ?? 0) * (K[k * nSamples + j] ?? 0);
          }
          const yj = yMapped[j] ?? 0;
          const Ej = fj - yj;

          const alphaIOld = alphas[i] ?? 0;
          const alphaJOld = alphas[j] ?? 0;

          // Compute bounds
          let L: number;
          let H: number;
          if (yi !== yj) {
            L = Math.max(0, alphaJOld - alphaIOld);
            H = Math.min(this.C, this.C + alphaJOld - alphaIOld);
          } else {
            L = Math.max(0, alphaIOld + alphaJOld - this.C);
            H = Math.min(this.C, alphaIOld + alphaJOld);
          }

          if (Math.abs(L - H) < 1e-12) continue;

          const eta =
            2 * (K[i * nSamples + j] ?? 0) -
            (K[i * nSamples + i] ?? 0) -
            (K[j * nSamples + j] ?? 0);
          if (eta >= 0) continue;

          // Update alpha_j
          let newAlphaJ = alphaJOld - (yj * (Ei - Ej)) / eta;
          newAlphaJ = Math.min(H, Math.max(L, newAlphaJ));

          if (Math.abs(newAlphaJ - alphaJOld) < 1e-5) continue;

          alphas[j] = newAlphaJ;
          alphas[i] = alphaIOld + yi * yj * (alphaJOld - newAlphaJ);

          // Update bias
          const b1 =
            b +
            Ei +
            yi * ((alphas[i] ?? 0) - alphaIOld) * (K[i * nSamples + i] ?? 0) +
            yj * (newAlphaJ - alphaJOld) * (K[i * nSamples + j] ?? 0);
          const b2 =
            b +
            Ej +
            yi * ((alphas[i] ?? 0) - alphaIOld) * (K[i * nSamples + j] ?? 0) +
            yj * (newAlphaJ - alphaJOld) * (K[j * nSamples + j] ?? 0);

          if ((alphas[i] ?? 0) > 0 && (alphas[i] ?? 0) < this.C) {
            b = b1;
          } else if (newAlphaJ > 0 && newAlphaJ < this.C) {
            b = b2;
          } else {
            b = (b1 + b2) / 2;
          }

          numChanged++;
        }
      }

      if (numChanged === 0) break;
    }

    // Extract support vectors
    const sv: number[][] = [];
    const svAlphas: number[] = [];
    const svLabels: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      if ((alphas[i] ?? 0) > 1e-8) {
        sv.push(XData[i]!);
        svAlphas.push(alphas[i] ?? 0);
        svLabels.push(yMapped[i] ?? 0);
      }
    }

    return { sv, alphas: svAlphas, labels: svLabels, bias: b };
  }

  private decisionFunction(
    x: number[],
    model: { sv: number[][]; alphas: number[]; labels: number[]; bias: number }
  ): number {
    let f = -model.bias;
    for (let i = 0; i < model.sv.length; i++) {
      f +=
        (model.alphas[i] ?? 0) *
        (model.labels[i] ?? 0) *
        kernelValue(model.sv[i]!, x, this.kernel, this.gamma_, this.coef0, this.degree);
    }
    return f;
  }

  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    this.nFeatures = nFeatures;

    const XData: number[][] = [];
    const yData: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      const row: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        row.push(Number(X.data[X.offset + i * nFeatures + j]));
      }
      XData.push(row);
      yData.push(Number(y.data[y.offset + i]));
    }

    this.classLabels = [...new Set(yData)].sort((a, b) => a - b);
    if (this.classLabels.length < 2) {
      throw new InvalidParameterError(
        "SVC requires at least 2 classes",
        "y",
        this.classLabels.length
      );
    }

    this.gamma_ = this.resolveGamma(nFeatures, XData, nSamples);

    if (this.classLabels.length === 2) {
      const yMapped = yData.map((l) => (l === this.classLabels[0] ? -1 : 1));
      const result = this.solveBinarySMO(XData, yMapped, nSamples);
      this.supportVectors_ = result.sv;
      this.supportAlphas_ = result.alphas;
      this.supportLabels_ = result.labels;
      this.bias_ = result.bias;
      this.models = [];
    } else {
      // OvR multiclass
      this.models = [];
      for (const cls of this.classLabels) {
        const yMapped = yData.map((l) => (l === cls ? 1 : -1));
        const result = this.solveBinarySMO(XData, yMapped, nSamples);
        this.models.push({ ...result, posClass: cls });
      }
    }

    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("SVC must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeatures, "SVC");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const predictions: number[] = [];

    for (let i = 0; i < nSamples; i++) {
      const xi: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        xi.push(Number(X.data[X.offset + i * nFeatures + j]));
      }

      if (this.classLabels.length === 2) {
        const d = this.decisionFunction(xi, {
          sv: this.supportVectors_!,
          alphas: this.supportAlphas_!,
          labels: this.supportLabels_!,
          bias: this.bias_,
        });
        predictions.push(d >= 0 ? (this.classLabels[1] ?? 0) : (this.classLabels[0] ?? 0));
      } else {
        let bestC = 0;
        let bestScore = -Infinity;
        for (let c = 0; c < this.models.length; c++) {
          const score = this.decisionFunction(xi, this.models[c]!);
          if (score > bestScore) {
            bestScore = score;
            bestC = c;
          }
        }
        predictions.push(this.models[bestC]?.posClass ?? 0);
      }
    }

    return tensor(predictions, { dtype: "int32" });
  }

  predictProba(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("SVC must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeatures, "SVC");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const nClasses = this.classLabels.length;
    const proba: number[][] = [];

    for (let i = 0; i < nSamples; i++) {
      const xi: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        xi.push(Number(X.data[X.offset + i * nFeatures + j]));
      }

      if (nClasses === 2) {
        const d = this.decisionFunction(xi, {
          sv: this.supportVectors_!,
          alphas: this.supportAlphas_!,
          labels: this.supportLabels_!,
          bias: this.bias_,
        });
        const p1 = 1 / (1 + Math.exp(-d));
        proba.push([1 - p1, p1]);
      } else {
        const scores: number[] = [];
        for (const model of this.models) {
          scores.push(1 / (1 + Math.exp(-this.decisionFunction(xi, model))));
        }
        const total = scores.reduce((a, b) => a + b, 0) || 1;
        proba.push(scores.map((s) => s / total));
      }
    }

    return tensor(proba);
  }

  score(X: Tensor, y: Tensor): number {
    if (y.ndim !== 1) {
      throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    }
    assertContiguous(y, "y");
    for (let i = 0; i < y.size; i++) {
      if (!Number.isFinite(y.data[y.offset + i] ?? 0)) {
        throw new DataValidationError("y contains non-finite values (NaN or Inf)");
      }
    }
    const predictions = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.size; i++) {
      if (Number(predictions.data[predictions.offset + i]) === Number(y.data[y.offset + i])) {
        correct++;
      }
    }
    return correct / y.size;
  }

  get classes(): Tensor | undefined {
    if (!this.fitted) return undefined;
    return tensor(this.classLabels, { dtype: "int32" });
  }

  getParams(): Record<string, unknown> {
    return {
      C: this.C,
      kernel: this.kernel,
      gamma: this.gamma,
      coef0: this.coef0,
      degree: this.degree,
      maxIter: this.maxIter,
      tol: this.tol,
    };
  }

  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "C":
          if (typeof value !== "number" || value <= 0) {
            throw new InvalidParameterError("C must be > 0", "C", value);
          }
          this.C = value;
          break;
        case "kernel":
          if (value !== "rbf" && value !== "linear" && value !== "poly" && value !== "sigmoid") {
            throw new InvalidParameterError(
              `kernel must be "rbf", "linear", "poly", or "sigmoid"`,
              "kernel",
              value
            );
          }
          this.kernel = value;
          break;
        case "gamma":
          if (value !== "scale" && value !== "auto" && (typeof value !== "number" || value <= 0)) {
            throw new InvalidParameterError(
              'gamma must be "scale", "auto", or a positive number',
              "gamma",
              value
            );
          }
          this.gamma = value;
          break;
        case "coef0":
          if (typeof value !== "number") {
            throw new InvalidParameterError("coef0 must be a number", "coef0", value);
          }
          this.coef0 = value;
          break;
        case "degree":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError("degree must be an integer >= 1", "degree", value);
          }
          this.degree = value;
          break;
        case "maxIter":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError("maxIter must be an integer >= 1", "maxIter", value);
          }
          this.maxIter = value;
          break;
        case "tol":
          if (typeof value !== "number" || value < 0) {
            throw new InvalidParameterError("tol must be >= 0", "tol", value);
          }
          this.tol = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}

/**
 * Support Vector Regression with kernel trick.
 *
 * Uses Simplified SMO on the dual epsilon-SVR formulation.
 * Supports RBF, polynomial, sigmoid, and linear kernels.
 *
 * @example
 * ```ts
 * import { SVR } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1], [2], [3], [4], [5]]);
 * const y = tensor([1.2, 2.1, 2.9, 4.0, 5.1]);
 *
 * const svr = new SVR({ kernel: 'rbf', C: 10 });
 * svr.fit(X, y);
 * const predictions = svr.predict(X);
 * ```
 */
export class SVR implements Regressor {
  private C: number;
  private kernel: KernelType;
  private gamma: number | "scale" | "auto";
  private coef0: number;
  private degree: number;
  private epsilon: number;
  private maxIter: number;
  private tol: number;

  private gamma_: number = 1;
  private supportVectors_?: number[][];
  private supportAlphasDiff_?: number[]; // alpha_i - alpha_i*
  private bias_ = 0;
  private nFeatures = 0;
  private fitted = false;

  constructor(
    options: {
      readonly C?: number;
      readonly kernel?: KernelType;
      readonly gamma?: number | "scale" | "auto";
      readonly coef0?: number;
      readonly degree?: number;
      readonly epsilon?: number;
      readonly maxIter?: number;
      readonly tol?: number;
    } = {}
  ) {
    this.C = options.C ?? 1.0;
    this.kernel = options.kernel ?? "rbf";
    this.gamma = options.gamma ?? "scale";
    this.coef0 = options.coef0 ?? 0;
    this.degree = options.degree ?? 3;
    this.epsilon = options.epsilon ?? 0.1;
    this.maxIter = options.maxIter ?? 1000;
    this.tol = options.tol ?? 1e-3;

    if (!Number.isFinite(this.C) || this.C <= 0) {
      throw new InvalidParameterError("C must be positive", "C", this.C);
    }
    if (!Number.isFinite(this.epsilon) || this.epsilon < 0) {
      throw new InvalidParameterError("epsilon must be >= 0", "epsilon", this.epsilon);
    }
  }

  private resolveGamma(nFeatures: number, XData: number[][], nSamples: number): number {
    if (typeof this.gamma === "number") return this.gamma;
    if (this.gamma === "auto") return 1 / nFeatures;
    let mean = 0;
    const total = nSamples * nFeatures;
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        mean += XData[i]![j] ?? 0;
      }
    }
    mean /= total;
    let variance = 0;
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        const d = (XData[i]![j] ?? 0) - mean;
        variance += d * d;
      }
    }
    variance /= total;
    return variance > 0 ? 1 / (nFeatures * variance) : 1;
  }

  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    this.nFeatures = nFeatures;

    const XData: number[][] = [];
    const yData: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      const row: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        row.push(Number(X.data[X.offset + i * nFeatures + j]));
      }
      XData.push(row);
      yData.push(Number(y.data[y.offset + i]));
    }

    this.gamma_ = this.resolveGamma(nFeatures, XData, nSamples);

    const K = computeKernelMatrix(
      XData,
      nSamples,
      this.kernel,
      this.gamma_,
      this.coef0,
      this.degree
    );

    // epsilon-SVR using simplified gradient-based approach
    // alpha[i] = alpha_i - alpha_i* (can be negative or positive)
    const alpha = new Float64Array(nSamples);
    let b = 0;
    const lr = 0.01;

    for (let iter = 0; iter < this.maxIter; iter++) {
      let maxChange = 0;
      for (let i = 0; i < nSamples; i++) {
        // Compute prediction f(x_i)
        let fi = b;
        for (let j = 0; j < nSamples; j++) {
          fi += (alpha[j] ?? 0) * (K[j * nSamples + i] ?? 0);
        }

        const error = fi - (yData[i] ?? 0);
        const absError = Math.abs(error);

        if (absError <= this.epsilon) continue;

        // Gradient step
        const grad = error > 0 ? 1 : -1;
        const oldAlpha = alpha[i] ?? 0;
        alpha[i] = oldAlpha - lr * (grad + oldAlpha / (this.C * nSamples));

        // Clip to [-C, C]
        alpha[i] = Math.min(this.C, Math.max(-this.C, alpha[i] ?? 0));

        const change = Math.abs((alpha[i] ?? 0) - oldAlpha);
        if (change > maxChange) maxChange = change;
      }

      // Update bias
      let bSum = 0;
      let bCount = 0;
      for (let i = 0; i < nSamples; i++) {
        const ai = alpha[i] ?? 0;
        if (Math.abs(ai) > 1e-8 && Math.abs(ai) < this.C - 1e-8) {
          let fi = 0;
          for (let j = 0; j < nSamples; j++) {
            fi += (alpha[j] ?? 0) * (K[j * nSamples + i] ?? 0);
          }
          bSum += (yData[i] ?? 0) - fi;
          bCount++;
        }
      }
      if (bCount > 0) {
        b = bSum / bCount;
      }

      if (maxChange < this.tol) break;
    }

    // Extract support vectors
    const sv: number[][] = [];
    const svAlphas: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      if (Math.abs(alpha[i] ?? 0) > 1e-8) {
        sv.push(XData[i]!);
        svAlphas.push(alpha[i] ?? 0);
      }
    }

    // If no support vectors found, use all points with small alphas
    if (sv.length === 0) {
      for (let i = 0; i < nSamples; i++) {
        sv.push(XData[i]!);
        svAlphas.push(alpha[i] ?? 0);
      }
    }

    this.supportVectors_ = sv;
    this.supportAlphasDiff_ = svAlphas;
    this.bias_ = b;
    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("SVR must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeatures, "SVR");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const predictions: number[] = [];

    for (let i = 0; i < nSamples; i++) {
      const xi: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        xi.push(Number(X.data[X.offset + i * nFeatures + j]));
      }

      let f = this.bias_;
      for (let s = 0; s < this.supportVectors_!.length; s++) {
        f +=
          (this.supportAlphasDiff_![s] ?? 0) *
          kernelValue(
            this.supportVectors_![s]!,
            xi,
            this.kernel,
            this.gamma_,
            this.coef0,
            this.degree
          );
      }
      predictions.push(f);
    }

    return tensor(predictions);
  }

  score(X: Tensor, y: Tensor): number {
    if (y.ndim !== 1) {
      throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    }
    assertContiguous(y, "y");
    for (let i = 0; i < y.size; i++) {
      if (!Number.isFinite(y.data[y.offset + i] ?? 0)) {
        throw new DataValidationError("y contains non-finite values (NaN or Inf)");
      }
    }
    const predictions = this.predict(X);
    if (predictions.size !== y.size) {
      throw new ShapeError(
        `X and y must have the same number of samples; got X=${predictions.size}, y=${y.size}`
      );
    }
    let ssRes = 0;
    let ssTot = 0;
    let yMean = 0;
    for (let i = 0; i < y.size; i++) {
      yMean += Number(y.data[y.offset + i]);
    }
    yMean /= y.size;
    for (let i = 0; i < y.size; i++) {
      const yVal = Number(y.data[y.offset + i]);
      const pVal = Number(predictions.data[predictions.offset + i]);
      ssRes += (yVal - pVal) ** 2;
      ssTot += (yVal - yMean) ** 2;
    }
    return ssTot === 0 ? (ssRes === 0 ? 1.0 : 0.0) : 1 - ssRes / ssTot;
  }

  getParams(): Record<string, unknown> {
    return {
      C: this.C,
      kernel: this.kernel,
      gamma: this.gamma,
      coef0: this.coef0,
      degree: this.degree,
      epsilon: this.epsilon,
      maxIter: this.maxIter,
      tol: this.tol,
    };
  }

  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "C":
          if (typeof value !== "number" || value <= 0) {
            throw new InvalidParameterError("C must be > 0", "C", value);
          }
          this.C = value;
          break;
        case "kernel":
          if (value !== "rbf" && value !== "linear" && value !== "poly" && value !== "sigmoid") {
            throw new InvalidParameterError(
              `kernel must be "rbf", "linear", "poly", or "sigmoid"`,
              "kernel",
              value
            );
          }
          this.kernel = value;
          break;
        case "gamma":
          if (value !== "scale" && value !== "auto" && (typeof value !== "number" || value <= 0)) {
            throw new InvalidParameterError(
              'gamma must be "scale", "auto", or a positive number',
              "gamma",
              value
            );
          }
          this.gamma = value;
          break;
        case "coef0":
          if (typeof value !== "number") {
            throw new InvalidParameterError("coef0 must be a number", "coef0", value);
          }
          this.coef0 = value;
          break;
        case "degree":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError("degree must be an integer >= 1", "degree", value);
          }
          this.degree = value;
          break;
        case "epsilon":
          if (typeof value !== "number" || value < 0) {
            throw new InvalidParameterError("epsilon must be >= 0", "epsilon", value);
          }
          this.epsilon = value;
          break;
        case "maxIter":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError("maxIter must be an integer >= 1", "maxIter", value);
          }
          this.maxIter = value;
          break;
        case "tol":
          if (typeof value !== "number" || value < 0) {
            throw new InvalidParameterError("tol must be >= 0", "tol", value);
          }
          this.tol = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}
