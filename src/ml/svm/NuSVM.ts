/**
 * Nu-parameterized SVM variants and One-Class SVM.
 *
 * - NuSVC: Classification SVM using nu parameter instead of C
 * - NuSVR: Regression SVM using nu parameter instead of epsilon
 * - OneClassSVM: Unsupervised outlier detection using one-class SVM
 *
 * @see {@link https://deepbox.dev/docs/ml-svm | Deepbox SVM}
 */

import { InvalidParameterError, NotFittedError, ShapeError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { __random } from "../../random/random";
import {
  assertContiguous,
  validateFitInputs,
  validatePredictInputs,
  validateUnsupervisedFitInputs,
} from "../_validation";
import type { Classifier, OutlierDetector, Regressor } from "../base";

type KernelType = "rbf" | "linear" | "poly" | "sigmoid";

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

function resolveGamma(
  gammaOpt: number | "scale" | "auto",
  nFeatures: number,
  XData: number[][],
  nSamples: number
): number {
  if (typeof gammaOpt === "number") return gammaOpt;
  if (gammaOpt === "auto") return 1 / nFeatures;
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

function extractRows(X: Tensor, nSamples: number, nFeatures: number): number[][] {
  const XData: number[][] = [];
  for (let i = 0; i < nSamples; i++) {
    const row: number[] = [];
    for (let j = 0; j < nFeatures; j++) {
      row.push(Number(X.data[X.offset + i * nFeatures + j] ?? 0));
    }
    XData.push(row);
  }
  return XData;
}

/**
 * Nu-Support Vector Classification.
 *
 * Similar to SVC but uses the `nu` parameter (in (0, 1]) to control the number
 * of support vectors and margin errors, instead of `C`.
 *
 * `nu` is an upper bound on the fraction of margin errors and a lower bound
 * on the fraction of support vectors.
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
 */
export class NuSVC implements Classifier {
  private nu: number;
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
  private models: Array<{
    sv: number[][];
    alphas: number[];
    labels: number[];
    bias: number;
    posClass: number;
  }> = [];
  private nFeatures = 0;
  private fitted = false;

  /**
   * @param options.nu - Upper bound on fraction of margin errors and lower bound on fraction of support vectors (default: 0.5). Must be in (0, 1].
   * @param options.kernel - Kernel type (default: "rbf")
   * @param options.gamma - Kernel coefficient (default: "scale")
   * @param options.coef0 - Independent term in kernel (default: 0)
   * @param options.degree - Degree for poly kernel (default: 3)
   * @param options.maxIter - Maximum number of iterations (default: 1000)
   * @param options.tol - Tolerance for stopping criterion (default: 1e-3)
   */
  constructor(
    options: {
      readonly nu?: number;
      readonly kernel?: KernelType;
      readonly gamma?: number | "scale" | "auto";
      readonly coef0?: number;
      readonly degree?: number;
      readonly maxIter?: number;
      readonly tol?: number;
    } = {}
  ) {
    this.nu = options.nu ?? 0.5;
    this.kernel = options.kernel ?? "rbf";
    this.gamma = options.gamma ?? "scale";
    this.coef0 = options.coef0 ?? 0;
    this.degree = options.degree ?? 3;
    this.maxIter = options.maxIter ?? 1000;
    this.tol = options.tol ?? 1e-3;

    if (!Number.isFinite(this.nu) || this.nu <= 0 || this.nu > 1) {
      throw new InvalidParameterError("nu must be in (0, 1]", "nu", this.nu);
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter <= 0) {
      throw new InvalidParameterError(
        "maxIter must be a positive integer",
        "maxIter",
        this.maxIter
      );
    }
  }

  private solveBinaryNuSMO(
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
    // For nu-SVM, C_eff = 1/(nu * nSamples) gives an equivalent bound
    const Ceff = 1 / (this.nu * nSamples);
    const alphas = new Float64Array(nSamples);

    // Initialize alphas to satisfy sum(alpha_i * y_i) = 0 and sum(alpha_i) = nu * n
    const nPos = yMapped.filter((v) => v > 0).length;
    const nNeg = nSamples - nPos;
    const targetSum = this.nu * nSamples;
    // Distribute alphas equally within each class
    const alphaPos = Math.min(Ceff, targetSum / (2 * nPos));
    const alphaNeg = Math.min(Ceff, targetSum / (2 * nNeg));
    for (let i = 0; i < nSamples; i++) {
      alphas[i] = (yMapped[i] ?? 0) > 0 ? alphaPos : alphaNeg;
    }

    let b = 0;

    for (let iter = 0; iter < this.maxIter; iter++) {
      let numChanged = 0;

      for (let i = 0; i < nSamples; i++) {
        let fi = -b;
        for (let j = 0; j < nSamples; j++) {
          fi += (alphas[j] ?? 0) * (yMapped[j] ?? 0) * (K[j * nSamples + i] ?? 0);
        }
        const yi = yMapped[i] ?? 0;
        const Ei = fi - yi;

        if (
          (yi * Ei < -this.tol && (alphas[i] ?? 0) < Ceff) ||
          (yi * Ei > this.tol && (alphas[i] ?? 0) > 0)
        ) {
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

          let L: number;
          let H: number;
          if (yi !== yj) {
            L = Math.max(0, alphaJOld - alphaIOld);
            H = Math.min(Ceff, Ceff + alphaJOld - alphaIOld);
          } else {
            L = Math.max(0, alphaIOld + alphaJOld - Ceff);
            H = Math.min(Ceff, alphaIOld + alphaJOld);
          }

          if (Math.abs(L - H) < 1e-12) continue;

          const eta =
            2 * (K[i * nSamples + j] ?? 0) -
            (K[i * nSamples + i] ?? 0) -
            (K[j * nSamples + j] ?? 0);
          if (eta >= 0) continue;

          let newAlphaJ = alphaJOld - (yj * (Ei - Ej)) / eta;
          newAlphaJ = Math.min(H, Math.max(L, newAlphaJ));

          if (Math.abs(newAlphaJ - alphaJOld) < 1e-5) continue;

          alphas[j] = newAlphaJ;
          alphas[i] = alphaIOld + yi * yj * (alphaJOld - newAlphaJ);

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

          if ((alphas[i] ?? 0) > 0 && (alphas[i] ?? 0) < Ceff) {
            b = b1;
          } else if (newAlphaJ > 0 && newAlphaJ < Ceff) {
            b = b2;
          } else {
            b = (b1 + b2) / 2;
          }

          numChanged++;
        }
      }

      if (numChanged === 0) break;
    }

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

  private decisionFn(
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

    const XData = extractRows(X, nSamples, nFeatures);
    const yData: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      yData.push(Number(y.data[y.offset + i] ?? 0));
    }

    this.classLabels = [...new Set(yData)].sort((a, b) => a - b);
    if (this.classLabels.length < 2) {
      throw new InvalidParameterError(
        "NuSVC requires at least 2 classes",
        "y",
        this.classLabels.length
      );
    }

    this.gamma_ = resolveGamma(this.gamma, nFeatures, XData, nSamples);

    if (this.classLabels.length === 2) {
      const yMapped = yData.map((l) => (l === this.classLabels[0] ? -1 : 1));
      const result = this.solveBinaryNuSMO(XData, yMapped, nSamples);
      this.supportVectors_ = result.sv;
      this.supportAlphas_ = result.alphas;
      this.supportLabels_ = result.labels;
      this.bias_ = result.bias;
      this.models = [];
    } else {
      this.models = [];
      for (const cls of this.classLabels) {
        const yMapped = yData.map((l) => (l === cls ? 1 : -1));
        const result = this.solveBinaryNuSMO(XData, yMapped, nSamples);
        this.models.push({ ...result, posClass: cls });
      }
    }

    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("NuSVC must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeatures, "NuSVC");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const predictions: number[] = [];

    for (let i = 0; i < nSamples; i++) {
      const xi: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        xi.push(Number(X.data[X.offset + i * nFeatures + j] ?? 0));
      }

      if (this.classLabels.length === 2) {
        const d = this.decisionFn(xi, {
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
          const score = this.decisionFn(xi, this.models[c]!);
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
      throw new NotFittedError("NuSVC must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeatures, "NuSVC");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const nClasses = this.classLabels.length;
    const proba: number[][] = [];

    for (let i = 0; i < nSamples; i++) {
      const xi: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        xi.push(Number(X.data[X.offset + i * nFeatures + j] ?? 0));
      }

      if (nClasses === 2) {
        const d = this.decisionFn(xi, {
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
          scores.push(1 / (1 + Math.exp(-this.decisionFn(xi, model))));
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
    const predictions = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.size; i++) {
      if (
        Number(predictions.data[predictions.offset + i] ?? 0) === Number(y.data[y.offset + i] ?? 0)
      ) {
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
      nu: this.nu,
      kernel: this.kernel,
      gamma: this.gamma,
      coef0: this.coef0,
      degree: this.degree,
    };
  }

  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "nu":
          if (typeof value !== "number" || value <= 0 || value > 1) {
            throw new InvalidParameterError("nu must be in (0, 1]", "nu", value);
          }
          this.nu = value;
          break;
        case "kernel":
          if (value !== "rbf" && value !== "linear" && value !== "poly" && value !== "sigmoid") {
            throw new InvalidParameterError("invalid kernel", "kernel", value);
          }
          this.kernel = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}

/**
 * Nu-Support Vector Regression.
 *
 * Similar to SVR but uses `nu` to control the fraction of support vectors
 * instead of specifying epsilon directly.
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
 */
export class NuSVR implements Regressor {
  private nu: number;
  private C: number;
  private kernel: KernelType;
  private gamma: number | "scale" | "auto";
  private coef0: number;
  private degree: number;
  private maxIter: number;
  private tol: number;

  private gamma_: number = 1;
  private supportVectors_?: number[][];
  private supportAlphasDiff_?: number[];
  private bias_ = 0;
  private nFeatures = 0;
  private fitted = false;

  /**
   * @param options.nu - Fraction of support vectors (default: 0.5). Must be in (0, 1].
   * @param options.C - Penalty parameter (default: 1.0)
   * @param options.kernel - Kernel type (default: "rbf")
   * @param options.gamma - Kernel coefficient (default: "scale")
   */
  constructor(
    options: {
      readonly nu?: number;
      readonly C?: number;
      readonly kernel?: KernelType;
      readonly gamma?: number | "scale" | "auto";
      readonly coef0?: number;
      readonly degree?: number;
      readonly maxIter?: number;
      readonly tol?: number;
    } = {}
  ) {
    this.nu = options.nu ?? 0.5;
    this.C = options.C ?? 1.0;
    this.kernel = options.kernel ?? "rbf";
    this.gamma = options.gamma ?? "scale";
    this.coef0 = options.coef0 ?? 0;
    this.degree = options.degree ?? 3;
    this.maxIter = options.maxIter ?? 1000;
    this.tol = options.tol ?? 1e-3;

    if (!Number.isFinite(this.nu) || this.nu <= 0 || this.nu > 1) {
      throw new InvalidParameterError("nu must be in (0, 1]", "nu", this.nu);
    }
    if (!Number.isFinite(this.C) || this.C <= 0) {
      throw new InvalidParameterError("C must be positive", "C", this.C);
    }
  }

  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    this.nFeatures = nFeatures;

    const XData = extractRows(X, nSamples, nFeatures);
    const yData: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      yData.push(Number(y.data[y.offset + i] ?? 0));
    }

    this.gamma_ = resolveGamma(this.gamma, nFeatures, XData, nSamples);

    const K = computeKernelMatrix(
      XData,
      nSamples,
      this.kernel,
      this.gamma_,
      this.coef0,
      this.degree
    );

    // Nu-SVR: epsilon is automatically determined from nu
    // Approximate: epsilon ~ nu * range(y) / n
    let yMin = Infinity;
    let yMax = -Infinity;
    for (const v of yData) {
      if (v < yMin) yMin = v;
      if (v > yMax) yMax = v;
    }
    const epsilon = (this.nu * (yMax - yMin)) / nSamples;
    const Ceff = this.C / nSamples;

    const alpha = new Float64Array(nSamples);
    let b = 0;
    const lr = 0.01;

    for (let iter = 0; iter < this.maxIter; iter++) {
      let maxChange = 0;
      for (let i = 0; i < nSamples; i++) {
        let fi = b;
        for (let j = 0; j < nSamples; j++) {
          fi += (alpha[j] ?? 0) * (K[j * nSamples + i] ?? 0);
        }

        const error = fi - (yData[i] ?? 0);
        const absError = Math.abs(error);

        if (absError <= epsilon) continue;

        const grad = error > 0 ? 1 : -1;
        const oldAlpha = alpha[i] ?? 0;
        alpha[i] = oldAlpha - lr * (grad + oldAlpha / (this.C * nSamples));
        alpha[i] = Math.min(Ceff, Math.max(-Ceff, alpha[i] ?? 0));

        const change = Math.abs((alpha[i] ?? 0) - oldAlpha);
        if (change > maxChange) maxChange = change;
      }

      let bSum = 0;
      let bCount = 0;
      for (let i = 0; i < nSamples; i++) {
        const ai = alpha[i] ?? 0;
        if (Math.abs(ai) > 1e-8 && Math.abs(ai) < Ceff - 1e-8) {
          let fi = 0;
          for (let j = 0; j < nSamples; j++) {
            fi += (alpha[j] ?? 0) * (K[j * nSamples + i] ?? 0);
          }
          bSum += (yData[i] ?? 0) - fi;
          bCount++;
        }
      }
      if (bCount > 0) b = bSum / bCount;

      if (maxChange < this.tol) break;
    }

    const sv: number[][] = [];
    const svAlphas: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      if (Math.abs(alpha[i] ?? 0) > 1e-8) {
        sv.push(XData[i]!);
        svAlphas.push(alpha[i] ?? 0);
      }
    }
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
      throw new NotFittedError("NuSVR must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeatures, "NuSVR");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const predictions: number[] = [];

    for (let i = 0; i < nSamples; i++) {
      const xi: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        xi.push(Number(X.data[X.offset + i * nFeatures + j] ?? 0));
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
    const predictions = this.predict(X);
    let ssRes = 0;
    let ssTot = 0;
    let yMean = 0;
    for (let i = 0; i < y.size; i++) {
      yMean += Number(y.data[y.offset + i] ?? 0);
    }
    yMean /= y.size;
    for (let i = 0; i < y.size; i++) {
      const yVal = Number(y.data[y.offset + i] ?? 0);
      const pVal = Number(predictions.data[predictions.offset + i] ?? 0);
      ssRes += (yVal - pVal) ** 2;
      ssTot += (yVal - yMean) ** 2;
    }
    return ssTot === 0 ? (ssRes === 0 ? 1.0 : 0.0) : 1 - ssRes / ssTot;
  }

  getParams(): Record<string, unknown> {
    return { nu: this.nu, C: this.C, kernel: this.kernel, gamma: this.gamma };
  }

  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "nu":
          if (typeof value !== "number" || value <= 0 || value > 1) {
            throw new InvalidParameterError("nu must be in (0, 1]", "nu", value);
          }
          this.nu = value;
          break;
        case "C":
          if (typeof value !== "number" || value <= 0) {
            throw new InvalidParameterError("C must be > 0", "C", value);
          }
          this.C = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}

/**
 * One-Class SVM for unsupervised outlier detection.
 *
 * Estimates the support of a high-dimensional distribution and classifies
 * new points as inliers (+1) or outliers (-1).
 *
 * Uses a simplified SMO approach on the one-class formulation where
 * the decision boundary separates the data from the origin in feature space.
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
 */
export class OneClassSVM implements OutlierDetector {
  private nu: number;
  private kernel: KernelType;
  private gamma: number | "scale" | "auto";
  private coef0: number;
  private degree: number;
  private maxIter: number;
  private tol: number;

  private gamma_: number = 1;
  private supportVectors_?: number[][];
  private supportAlphas_?: number[];
  private rho_ = 0;
  private nFeatures = 0;
  private fitted = false;

  /**
   * @param options.nu - Upper bound on fraction of outliers and lower bound on fraction of support vectors (default: 0.5)
   * @param options.kernel - Kernel type (default: "rbf")
   * @param options.gamma - Kernel coefficient (default: "scale")
   */
  constructor(
    options: {
      readonly nu?: number;
      readonly kernel?: KernelType;
      readonly gamma?: number | "scale" | "auto";
      readonly coef0?: number;
      readonly degree?: number;
      readonly maxIter?: number;
      readonly tol?: number;
    } = {}
  ) {
    this.nu = options.nu ?? 0.5;
    this.kernel = options.kernel ?? "rbf";
    this.gamma = options.gamma ?? "scale";
    this.coef0 = options.coef0 ?? 0;
    this.degree = options.degree ?? 3;
    this.maxIter = options.maxIter ?? 1000;
    this.tol = options.tol ?? 1e-3;

    if (!Number.isFinite(this.nu) || this.nu <= 0 || this.nu > 1) {
      throw new InvalidParameterError("nu must be in (0, 1]", "nu", this.nu);
    }
  }

  fit(X: Tensor, _y?: Tensor): this {
    validateUnsupervisedFitInputs(X);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    this.nFeatures = nFeatures;

    const XData = extractRows(X, nSamples, nFeatures);
    this.gamma_ = resolveGamma(this.gamma, nFeatures, XData, nSamples);

    const K = computeKernelMatrix(
      XData,
      nSamples,
      this.kernel,
      this.gamma_,
      this.coef0,
      this.degree
    );

    // One-class SVM: all labels are +1, constraint sum(alpha) = 1
    // Upper bound: alpha_i <= 1/(nu * n)
    const upperBound = 1 / (this.nu * nSamples);
    const alphas = new Float64Array(nSamples);

    // Initialize: distribute alphas uniformly to sum = 1
    const initAlpha = Math.min(upperBound, 1 / nSamples);
    for (let i = 0; i < nSamples; i++) {
      alphas[i] = initAlpha;
    }

    // Simplified SMO for one-class SVM
    for (let iter = 0; iter < this.maxIter; iter++) {
      let numChanged = 0;

      for (let i = 0; i < nSamples; i++) {
        // Compute f(x_i) = sum_j alpha_j K(x_j, x_i)
        let fi = 0;
        for (let j = 0; j < nSamples; j++) {
          fi += (alphas[j] ?? 0) * (K[j * nSamples + i] ?? 0);
        }

        // Check KKT violations
        const ai = alphas[i] ?? 0;
        const kktViolation =
          (ai < upperBound - 1e-8 && fi < this.rho_ - this.tol) ||
          (ai > 1e-8 && fi > this.rho_ + this.tol);

        if (!kktViolation) continue;

        // Select j randomly
        let j = Math.floor(__random() * (nSamples - 1));
        if (j >= i) j++;

        let fj = 0;
        for (let k = 0; k < nSamples; k++) {
          fj += (alphas[k] ?? 0) * (K[k * nSamples + j] ?? 0);
        }

        const alphaIOld = alphas[i] ?? 0;
        const alphaJOld = alphas[j] ?? 0;

        // Bounds: ensure sum stays constant
        const L = Math.max(0, alphaIOld + alphaJOld - upperBound);
        const H = Math.min(upperBound, alphaIOld + alphaJOld);

        if (Math.abs(L - H) < 1e-12) continue;

        const eta =
          2 * (K[i * nSamples + j] ?? 0) - (K[i * nSamples + i] ?? 0) - (K[j * nSamples + j] ?? 0);
        if (eta >= 0) continue;

        let newAlphaJ = alphaJOld + (fi - fj) / eta;
        newAlphaJ = Math.min(H, Math.max(L, newAlphaJ));

        if (Math.abs(newAlphaJ - alphaJOld) < 1e-5) continue;

        alphas[j] = newAlphaJ;
        alphas[i] = alphaIOld + (alphaJOld - newAlphaJ);

        numChanged++;
      }

      // Update rho: average f(x_i) for support vectors with 0 < alpha < upperBound
      let rhoSum = 0;
      let rhoCount = 0;
      for (let i = 0; i < nSamples; i++) {
        const ai = alphas[i] ?? 0;
        if (ai > 1e-8 && ai < upperBound - 1e-8) {
          let fi = 0;
          for (let j = 0; j < nSamples; j++) {
            fi += (alphas[j] ?? 0) * (K[j * nSamples + i] ?? 0);
          }
          rhoSum += fi;
          rhoCount++;
        }
      }
      if (rhoCount > 0) {
        this.rho_ = rhoSum / rhoCount;
      }

      if (numChanged === 0) break;
    }

    // Extract support vectors
    const sv: number[][] = [];
    const svAlphas: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      if ((alphas[i] ?? 0) > 1e-8) {
        sv.push(XData[i]!);
        svAlphas.push(alphas[i] ?? 0);
      }
    }
    if (sv.length === 0) {
      for (let i = 0; i < nSamples; i++) {
        sv.push(XData[i]!);
        svAlphas.push(alphas[i] ?? 0);
      }
    }

    this.supportVectors_ = sv;
    this.supportAlphas_ = svAlphas;
    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("OneClassSVM must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeatures, "OneClassSVM");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const predictions: number[] = [];

    for (let i = 0; i < nSamples; i++) {
      const xi: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        xi.push(Number(X.data[X.offset + i * nFeatures + j] ?? 0));
      }
      let f = 0;
      for (let s = 0; s < this.supportVectors_!.length; s++) {
        f +=
          (this.supportAlphas_![s] ?? 0) *
          kernelValue(
            this.supportVectors_![s]!,
            xi,
            this.kernel,
            this.gamma_,
            this.coef0,
            this.degree
          );
      }
      predictions.push(f >= this.rho_ ? 1 : -1);
    }

    return tensor(predictions, { dtype: "int32" });
  }

  fitPredict(X: Tensor, _y?: Tensor): Tensor {
    this.fit(X);
    return this.predict(X);
  }

  scoreSamples(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("OneClassSVM must be fitted before scoring");
    }
    validatePredictInputs(X, this.nFeatures, "OneClassSVM");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const scores: number[] = [];

    for (let i = 0; i < nSamples; i++) {
      const xi: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        xi.push(Number(X.data[X.offset + i * nFeatures + j] ?? 0));
      }
      let f = 0;
      for (let s = 0; s < this.supportVectors_!.length; s++) {
        f +=
          (this.supportAlphas_![s] ?? 0) *
          kernelValue(
            this.supportVectors_![s]!,
            xi,
            this.kernel,
            this.gamma_,
            this.coef0,
            this.degree
          );
      }
      scores.push(f - this.rho_);
    }

    return tensor(scores);
  }

  getParams(): Record<string, unknown> {
    return { nu: this.nu, kernel: this.kernel, gamma: this.gamma };
  }

  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "nu":
          if (typeof value !== "number" || value <= 0 || value > 1) {
            throw new InvalidParameterError("nu must be in (0, 1]", "nu", value);
          }
          this.nu = value;
          break;
        case "kernel":
          if (value !== "rbf" && value !== "linear" && value !== "poly" && value !== "sigmoid") {
            throw new InvalidParameterError("invalid kernel", "kernel", value);
          }
          this.kernel = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}
