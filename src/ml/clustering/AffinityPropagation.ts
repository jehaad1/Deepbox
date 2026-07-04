/**
 * Affinity Propagation clustering.
 *
 * Clusters data by sending messages between pairs of samples until
 * convergence. Automatically determines the number of clusters from
 * the data using a preference parameter.
 *
 * @module ml/clustering/AffinityPropagation
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { validatePredictInputs, validateUnsupervisedFitInputs } from "../_validation";
import type { Clusterer } from "../base";

/**
 * Affinity Propagation clustering algorithm.
 *
 * @example
 * ```ts
 * import { AffinityPropagation } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[0, 0], [0.1, 0], [10, 10], [10.1, 10]]);
 * const ap = new AffinityPropagation({ damping: 0.5 });
 * ap.fit(X);
 * console.log(ap.labels);
 * ```
 */
export class AffinityPropagation implements Clusterer {
  private damping: number;
  private maxIter: number;
  private convergenceIter: number;
  private preference: number | undefined;

  private labels_?: Tensor;
  private clusterCentersIndices_?: Int32Array;
  private clusterCenters_?: Float64Array;
  private nFeaturesIn_ = 0;
  private nIter_ = 0;
  private fitted = false;

  constructor(
    options: {
      readonly damping?: number;
      readonly maxIter?: number;
      readonly convergenceIter?: number;
      readonly preference?: number;
    } = {}
  ) {
    this.damping = options.damping ?? 0.5;
    this.maxIter = options.maxIter ?? 200;
    this.convergenceIter = options.convergenceIter ?? 15;
    if (options.preference !== undefined) this.preference = options.preference;

    if (this.damping < 0.5 || this.damping >= 1) {
      throw new InvalidParameterError("damping must be in [0.5, 1)", "damping", this.damping);
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new InvalidParameterError(
        "maxIter must be a positive integer",
        "maxIter",
        this.maxIter
      );
    }
  }

  fit(X: Tensor, _y?: Tensor): this {
    validateUnsupervisedFitInputs(X);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    this.nFeaturesIn_ = nFeatures;

    // Compute similarity matrix (negative squared Euclidean distance)
    const S = new Float64Array(nSamples * nSamples);
    for (let i = 0; i < nSamples; i++) {
      for (let j = i + 1; j < nSamples; j++) {
        let d = 0;
        for (let f = 0; f < nFeatures; f++) {
          const diff =
            Number(X.data[X.offset + i * nFeatures + f]) -
            Number(X.data[X.offset + j * nFeatures + f]);
          d += diff * diff;
        }
        S[i * nSamples + j] = -d;
        S[j * nSamples + i] = -d;
      }
    }

    // Set preference (diagonal of S)
    let pref: number;
    if (this.preference !== undefined) {
      pref = this.preference;
    } else {
      // Default: median of similarities
      const offDiag: number[] = [];
      for (let i = 0; i < nSamples; i++) {
        for (let j = 0; j < nSamples; j++) {
          if (i !== j) offDiag.push(S[i * nSamples + j] ?? 0);
        }
      }
      offDiag.sort((a, b) => a - b);
      pref = offDiag[Math.floor(offDiag.length / 2)] ?? 0;
    }
    for (let i = 0; i < nSamples; i++) {
      S[i * nSamples + i] = pref;
    }

    // Initialize responsibility and availability matrices
    const R = new Float64Array(nSamples * nSamples);
    const A = new Float64Array(nSamples * nSamples);

    let convergedCount = 0;
    let prevExemplars = new Int32Array(nSamples);

    for (let iter = 0; iter < this.maxIter; iter++) {
      // Update responsibilities
      for (let i = 0; i < nSamples; i++) {
        for (let k = 0; k < nSamples; k++) {
          // r(i,k) = s(i,k) - max_{k' != k}(a(i,k') + s(i,k'))
          let maxAS = -Infinity;
          for (let kp = 0; kp < nSamples; kp++) {
            if (kp === k) continue;
            const val = (A[i * nSamples + kp] ?? 0) + (S[i * nSamples + kp] ?? 0);
            if (val > maxAS) maxAS = val;
          }
          const newR = (S[i * nSamples + k] ?? 0) - maxAS;
          R[i * nSamples + k] =
            this.damping * (R[i * nSamples + k] ?? 0) + (1 - this.damping) * newR;
        }
      }

      // Update availabilities
      for (let i = 0; i < nSamples; i++) {
        for (let k = 0; k < nSamples; k++) {
          if (i === k) {
            // a(k,k) = sum_{i' != k} max(0, r(i',k))
            let sum = 0;
            for (let ip = 0; ip < nSamples; ip++) {
              if (ip === k) continue;
              sum += Math.max(0, R[ip * nSamples + k] ?? 0);
            }
            const newA = sum;
            A[k * nSamples + k] =
              this.damping * (A[k * nSamples + k] ?? 0) + (1 - this.damping) * newA;
          } else {
            // a(i,k) = min(0, r(k,k) + sum_{i' != i,k} max(0, r(i',k)))
            let sum = 0;
            for (let ip = 0; ip < nSamples; ip++) {
              if (ip === i || ip === k) continue;
              sum += Math.max(0, R[ip * nSamples + k] ?? 0);
            }
            const newA = Math.min(0, (R[k * nSamples + k] ?? 0) + sum);
            A[i * nSamples + k] =
              this.damping * (A[i * nSamples + k] ?? 0) + (1 - this.damping) * newA;
          }
        }
      }

      // Check convergence: identify exemplars
      const exemplars = new Int32Array(nSamples);
      for (let i = 0; i < nSamples; i++) {
        let bestK = 0;
        let bestVal = -Infinity;
        for (let k = 0; k < nSamples; k++) {
          const val = (A[i * nSamples + k] ?? 0) + (R[i * nSamples + k] ?? 0);
          if (val > bestVal) {
            bestVal = val;
            bestK = k;
          }
        }
        exemplars[i] = bestK;
      }

      // Check if exemplars changed
      let same = true;
      for (let i = 0; i < nSamples; i++) {
        if (exemplars[i] !== prevExemplars[i]) {
          same = false;
          break;
        }
      }

      if (same) {
        convergedCount++;
        if (convergedCount >= this.convergenceIter) {
          this.nIter_ = iter + 1;
          break;
        }
      } else {
        convergedCount = 0;
      }

      prevExemplars = new Int32Array(exemplars);
      this.nIter_ = iter + 1;
    }

    // Extract final exemplars and labels
    const exemplarSet = new Set<number>();
    const finalExemplars = new Int32Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      let bestK = 0;
      let bestVal = -Infinity;
      for (let k = 0; k < nSamples; k++) {
        const val = (A[i * nSamples + k] ?? 0) + (R[i * nSamples + k] ?? 0);
        if (val > bestVal) {
          bestVal = val;
          bestK = k;
        }
      }
      finalExemplars[i] = bestK;
      exemplarSet.add(bestK);
    }

    // Map exemplar indices to cluster labels
    const exemplarArr = [...exemplarSet].sort((a, b) => a - b);
    const exemplarMap = new Map<number, number>();
    for (let idx = 0; idx < exemplarArr.length; idx++) {
      exemplarMap.set(exemplarArr[idx] ?? 0, idx);
    }

    const labels = new Int32Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      labels[i] = exemplarMap.get(finalExemplars[i] ?? 0) ?? 0;
    }

    this.labels_ = tensor(Array.from(labels), { dtype: "int32" });
    this.clusterCentersIndices_ = new Int32Array(exemplarArr);

    // Store cluster centers
    const nClusters = exemplarArr.length;
    this.clusterCenters_ = new Float64Array(nClusters * nFeatures);
    for (let c = 0; c < nClusters; c++) {
      const idx = exemplarArr[c] ?? 0;
      for (let j = 0; j < nFeatures; j++) {
        this.clusterCenters_[c * nFeatures + j] = Number(X.data[X.offset + idx * nFeatures + j]);
      }
    }

    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.clusterCenters_) {
      throw new NotFittedError("AffinityPropagation must be fitted before predict");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "AffinityPropagation");
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const nClusters = this.clusterCentersIndices_!.length;
    const centers = this.clusterCenters_;

    const labels = new Int32Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      let bestC = 0;
      let bestD = Infinity;
      for (let c = 0; c < nClusters; c++) {
        let d = 0;
        for (let j = 0; j < nFeatures; j++) {
          const diff =
            Number(X.data[X.offset + i * nFeatures + j]) - (centers[c * nFeatures + j] ?? 0);
          d += diff * diff;
        }
        if (d < bestD) {
          bestD = d;
          bestC = c;
        }
      }
      labels[i] = bestC;
    }

    return tensor(Array.from(labels), { dtype: "int32" });
  }

  fitPredict(X: Tensor, _y?: Tensor): Tensor {
    this.fit(X);
    return this.labels_!;
  }

  get labels(): Tensor {
    if (!this.fitted || !this.labels_) {
      throw new NotFittedError("AffinityPropagation must be fitted to access labels");
    }
    return this.labels_;
  }

  get clusterCenters(): Tensor {
    if (!this.fitted || !this.clusterCenters_) {
      throw new NotFittedError("AffinityPropagation must be fitted to access cluster centers");
    }
    const nClusters = this.clusterCentersIndices_!.length;
    const nF = this.nFeaturesIn_;
    return tensor(Array.from(this.clusterCenters_)).reshape([nClusters, nF]);
  }

  get clusterCentersIndices(): Int32Array {
    if (!this.fitted || !this.clusterCentersIndices_) {
      throw new NotFittedError(
        "AffinityPropagation must be fitted to access cluster center indices"
      );
    }
    return this.clusterCentersIndices_;
  }

  get nIter(): number {
    if (!this.fitted)
      throw new NotFittedError("AffinityPropagation must be fitted to access nIter");
    return this.nIter_;
  }

  getParams(): Record<string, unknown> {
    return {
      damping: this.damping,
      maxIter: this.maxIter,
      convergenceIter: this.convergenceIter,
      preference: this.preference,
    };
  }

  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "damping":
          if (typeof value !== "number" || value < 0.5 || value >= 1) {
            throw new InvalidParameterError("damping must be in [0.5, 1)", "damping", value);
          }
          this.damping = value;
          break;
        case "maxIter":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError("maxIter must be an integer >= 1", "maxIter", value);
          }
          this.maxIter = value;
          break;
        case "convergenceIter":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError(
              "convergenceIter must be an integer >= 1",
              "convergenceIter",
              value
            );
          }
          this.convergenceIter = value;
          break;
        case "preference":
          if (value !== undefined && typeof value !== "number") {
            throw new InvalidParameterError(
              "preference must be a number or undefined",
              "preference",
              value
            );
          }
          this.preference = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}
