/**
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError } from "../../core";
import { solve } from "../../linalg/solvers/solve";
import { type Tensor, tensor } from "../../ndarray";
import { validateFitInputs, validatePredictInputs } from "../_validation";
import type { Regressor } from "../base";

/**
 * Kernel Ridge Regression.
 *
 * Combines Ridge Regression (L2 penalty) with the kernel trick,
 * learning a non-linear function in the feature space induced by the kernel.
 *
 * Solves: (K + alpha * I) * dual_coef = y
 *
 * @example
 * ```ts
 * import { KernelRidge } from 'deepbox/ml';
 *
 * const model = new KernelRidge({ alpha: 1.0, kernel: 'rbf', gamma: 0.1 });
 * model.fit(X_train, y_train);
 * const predictions = model.predict(X_test);
 * ```
 *
 * @category Linear Models
 * @implements {Regressor}
 */
export class KernelRidge implements Regressor {
  private options: {
    alpha: number;
    kernel: "linear" | "rbf" | "polynomial";
    gamma: number;
    degree: number;
    coef0: number;
  };

  private dualCoef_?: Float64Array;
  private XFit_?: Float64Array;
  private nSamplesFit_?: number;
  private nFeaturesFit_?: number;
  private fitted = false;

  constructor(
    options: {
      readonly alpha?: number;
      readonly kernel?: "linear" | "rbf" | "polynomial";
      readonly gamma?: number;
      readonly degree?: number;
      readonly coef0?: number;
    } = {}
  ) {
    this.options = {
      alpha: options.alpha ?? 1.0,
      kernel: options.kernel ?? "rbf",
      gamma: options.gamma ?? 1.0,
      degree: options.degree ?? 3,
      coef0: options.coef0 ?? 1,
    };
    if (this.options.alpha < 0) {
      throw new InvalidParameterError("alpha must be >= 0", "alpha", this.options.alpha);
    }
  }

  fit(X: Tensor, y: Tensor): KernelRidge {
    validateFitInputs(X, y);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;

    // Store training data for kernel computation at predict time
    const xFit = new Float64Array(nSamples * nFeatures);
    for (let i = 0; i < nSamples * nFeatures; i++) {
      xFit[i] = Number(X.data[(X.offset ?? 0) + i] ?? 0);
    }
    this.XFit_ = xFit;
    this.nSamplesFit_ = nSamples;
    this.nFeaturesFit_ = nFeatures;

    // Compute kernel matrix K
    const K = this.computeKernel(this.XFit_, nSamples, nFeatures, this.XFit_, nSamples, nFeatures);

    // Add alpha * I to diagonal
    for (let i = 0; i < nSamples; i++) {
      K[i * nSamples + i] = (K[i * nSamples + i] ?? 0) + this.options.alpha;
    }

    // Solve (K + alpha*I) * dual_coef = y
    const yArr = new Float64Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      yArr[i] = Number(y.data[(y.offset ?? 0) + i] ?? 0);
    }

    const KTensor = tensor(Array.from(K), { dtype: "float64" }).reshape([nSamples, nSamples]);
    const yTensor = tensor(Array.from(yArr), { dtype: "float64" }).reshape([nSamples, 1]);
    const solution = solve(KTensor, yTensor);
    this.dualCoef_ = new Float64Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      this.dualCoef_[i] = Number(solution.data[i] ?? 0);
    }

    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.dualCoef_ || !this.XFit_) {
      throw new NotFittedError("KernelRidge");
    }
    validatePredictInputs(X, this.nFeaturesFit_ ?? 0, "KernelRidge");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;

    // Extract X data
    const xData = new Float64Array(nSamples * nFeatures);
    for (let i = 0; i < nSamples * nFeatures; i++) {
      xData[i] = Number(X.data[(X.offset ?? 0) + i] ?? 0);
    }

    // Compute kernel between X and XFit
    const K = this.computeKernel(
      xData,
      nSamples,
      nFeatures,
      this.XFit_,
      this.nSamplesFit_ ?? 0,
      this.nFeaturesFit_ ?? 0
    );

    // predictions = K * dual_coef
    const predictions = new Float64Array(nSamples);
    const nTrain = this.nSamplesFit_ ?? 0;
    for (let i = 0; i < nSamples; i++) {
      let s = 0;
      for (let j = 0; j < nTrain; j++) {
        s += (K[i * nTrain + j] ?? 0) * (this.dualCoef_[j] ?? 0);
      }
      predictions[i] = s;
    }

    return tensor(Array.from(predictions), { dtype: "float64" });
  }

  score(X: Tensor, y: Tensor): number {
    const pred = this.predict(X);
    const n = y.size;
    let ssRes = 0;
    let ssTot = 0;
    let yMean = 0;
    for (let i = 0; i < n; i++) {
      yMean += Number(y.data[(y.offset ?? 0) + i] ?? 0);
    }
    yMean /= n;
    for (let i = 0; i < n; i++) {
      const yi = Number(y.data[(y.offset ?? 0) + i] ?? 0);
      const pi = Number(pred.data[i] ?? 0);
      ssRes += (yi - pi) ** 2;
      ssTot += (yi - yMean) ** 2;
    }
    return ssTot === 0 ? 0 : 1 - ssRes / ssTot;
  }

  getParams(): Record<string, unknown> {
    return { ...this.options };
  }

  setParams(params: Record<string, unknown>): KernelRidge {
    if (params["alpha"] !== undefined) this.options.alpha = params["alpha"] as number;
    if (params["kernel"] !== undefined)
      this.options.kernel = params["kernel"] as "linear" | "rbf" | "polynomial";
    if (params["gamma"] !== undefined) this.options.gamma = params["gamma"] as number;
    if (params["degree"] !== undefined) this.options.degree = params["degree"] as number;
    if (params["coef0"] !== undefined) this.options.coef0 = params["coef0"] as number;
    return this;
  }

  private computeKernel(
    A: Float64Array,
    nA: number,
    dA: number,
    B: Float64Array,
    nB: number,
    _dB: number
  ): Float64Array {
    const K = new Float64Array(nA * nB);
    switch (this.options.kernel) {
      case "linear":
        for (let i = 0; i < nA; i++) {
          for (let j = 0; j < nB; j++) {
            let dot = 0;
            for (let f = 0; f < dA; f++) {
              dot += (A[i * dA + f] ?? 0) * (B[j * dA + f] ?? 0);
            }
            K[i * nB + j] = dot;
          }
        }
        break;
      case "rbf": {
        const gamma = this.options.gamma;
        for (let i = 0; i < nA; i++) {
          for (let j = 0; j < nB; j++) {
            let sqDist = 0;
            for (let f = 0; f < dA; f++) {
              const diff = (A[i * dA + f] ?? 0) - (B[j * dA + f] ?? 0);
              sqDist += diff * diff;
            }
            K[i * nB + j] = Math.exp(-gamma * sqDist);
          }
        }
        break;
      }
      case "polynomial": {
        const { gamma, degree, coef0 } = this.options;
        for (let i = 0; i < nA; i++) {
          for (let j = 0; j < nB; j++) {
            let dot = 0;
            for (let f = 0; f < dA; f++) {
              dot += (A[i * dA + f] ?? 0) * (B[j * dA + f] ?? 0);
            }
            K[i * nB + j] = (gamma * dot + coef0) ** degree;
          }
        }
        break;
      }
    }
    return K;
  }
}
