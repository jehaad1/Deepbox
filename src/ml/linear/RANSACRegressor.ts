/**
 * RANSAC (Random Sample Consensus) Regressor.
 *
 * Robust regression that fits a model on random inlier subsets and
 * identifies outliers. Uses a base regressor (default: LinearRegression).
 *
 * @module ml/linear/RANSACRegressor
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { __random } from "../../random/random";
import { validateFitInputs, validatePredictInputs } from "../_validation";
import type { Regressor } from "../base";
import { LinearRegression } from "./LinearRegression";

export class RANSACRegressor implements Regressor {
  private readonly minSamples: number;
  private readonly residualThreshold: number | undefined;
  private readonly maxTrials: number;
  private readonly randomState?: number;

  private bestEstimator_?: Regressor;
  private inlierMask_?: Uint8Array;
  private nFeaturesIn_ = 0;
  private fitted = false;

  constructor(
    options: {
      readonly minSamples?: number;
      readonly residualThreshold?: number;
      readonly maxTrials?: number;
      readonly randomState?: number;
    } = {}
  ) {
    this.minSamples = options.minSamples ?? 5;
    this.maxTrials = options.maxTrials ?? 100;
    if (options.residualThreshold !== undefined) this.residualThreshold = options.residualThreshold;
    if (options.randomState !== undefined) this.randomState = options.randomState;

    if (!Number.isInteger(this.minSamples) || this.minSamples < 1) {
      throw new InvalidParameterError("minSamples must be >= 1", "minSamples", this.minSamples);
    }
    if (!Number.isInteger(this.maxTrials) || this.maxTrials < 1) {
      throw new InvalidParameterError("maxTrials must be >= 1", "maxTrials", this.maxTrials);
    }
  }

  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const n = X.shape[0] ?? 0;
    const nF = X.shape[1] ?? 0;
    this.nFeaturesIn_ = nF;

    // Extract data
    const xData = new Float64Array(n * nF);
    const yData = new Float64Array(n);
    for (let i = 0; i < n * nF; i++) xData[i] = Number(X.data[X.offset + i]);
    for (let i = 0; i < n; i++) yData[i] = Number(y.data[y.offset + i]);

    // Determine residual threshold (MAD-based if not provided)
    let threshold = this.residualThreshold;
    if (threshold === undefined) {
      // Fit on all data to get residuals, then use median absolute deviation
      const allReg = new LinearRegression();
      allReg.fit(X, y);
      const allPred = allReg.predict(X);
      const residuals: number[] = [];
      for (let i = 0; i < n; i++) {
        residuals.push(Math.abs(yData[i]! - Number(allPred.data[allPred.offset + i])));
      }
      residuals.sort((a, b) => a - b);
      const median = residuals[Math.floor(residuals.length / 2)] ?? 1;
      threshold = median * 2; // 2x median residual
      if (threshold < 1e-10) threshold = 1;
    }

    const rng = this.createRng();
    const minSamples = Math.min(this.minSamples, n);

    let bestNInliers = 0;
    let bestInlierMask = new Uint8Array(n);
    let bestEstimator: Regressor | undefined;

    for (let trial = 0; trial < this.maxTrials; trial++) {
      // Random subset
      const indices = this.randomSubset(n, minSamples, rng);

      // Fit on subset
      const subX = new Float64Array(minSamples * nF);
      const subY = new Float64Array(minSamples);
      for (let si = 0; si < minSamples; si++) {
        const idx = indices[si]!;
        for (let f = 0; f < nF; f++) {
          subX[si * nF + f] = xData[idx * nF + f] ?? 0;
        }
        subY[si] = yData[idx] ?? 0;
      }

      const reg = new LinearRegression();
      reg.fit(tensor(Array.from(subX)).reshape([minSamples, nF]), tensor(Array.from(subY)));

      // Compute residuals on all data
      const pred = reg.predict(X);
      const inlierMask = new Uint8Array(n);
      let nInliers = 0;
      for (let i = 0; i < n; i++) {
        const residual = Math.abs(yData[i]! - Number(pred.data[pred.offset + i]));
        if (residual <= threshold) {
          inlierMask[i] = 1;
          nInliers++;
        }
      }

      if (nInliers > bestNInliers) {
        bestNInliers = nInliers;
        bestInlierMask = inlierMask;
        bestEstimator = reg;
      }
    }

    // Refit on best inlier set
    if (bestNInliers > 0 && bestEstimator) {
      const inlierX = new Float64Array(bestNInliers * nF);
      const inlierY = new Float64Array(bestNInliers);
      let idx = 0;
      for (let i = 0; i < n; i++) {
        if (bestInlierMask[i]) {
          for (let f = 0; f < nF; f++) {
            inlierX[idx * nF + f] = xData[i * nF + f] ?? 0;
          }
          inlierY[idx] = yData[i] ?? 0;
          idx++;
        }
      }
      const finalReg = new LinearRegression();
      finalReg.fit(
        tensor(Array.from(inlierX)).reshape([bestNInliers, nF]),
        tensor(Array.from(inlierY))
      );
      this.bestEstimator_ = finalReg;
    } else {
      // Fallback: fit on all data
      this.bestEstimator_ = new LinearRegression();
      this.bestEstimator_.fit(X, y);
      bestInlierMask.fill(1);
    }

    this.inlierMask_ = bestInlierMask;
    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.bestEstimator_) {
      throw new NotFittedError("RANSACRegressor must be fitted before predict");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "RANSACRegressor");
    return this.bestEstimator_.predict(X);
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
      ssRes += (yi - pi) * (yi - pi);
      ssTot += (yi - yMean) * (yi - yMean);
    }
    return ssTot > 0 ? 1 - ssRes / ssTot : 0;
  }

  get inlierMask(): Uint8Array {
    if (!this.fitted || !this.inlierMask_) {
      throw new NotFittedError("RANSACRegressor must be fitted to access inlierMask");
    }
    return this.inlierMask_;
  }

  getParams(): Record<string, unknown> {
    return {
      minSamples: this.minSamples,
      residualThreshold: this.residualThreshold,
      maxTrials: this.maxTrials,
      randomState: this.randomState,
    };
  }

  setParams(_params: Record<string, unknown>): this {
    return this;
  }

  private randomSubset(n: number, k: number, rng: () => number): number[] {
    const indices: number[] = [];
    const used = new Set<number>();
    while (indices.length < k) {
      const idx = Math.floor(rng() * n);
      if (!used.has(idx)) {
        used.add(idx);
        indices.push(idx);
      }
    }
    return indices;
  }

  private createRng(): () => number {
    if (this.randomState === undefined) return __random;
    let s = this.randomState;
    return () => {
      s = (s * 9301 + 49297) % 233280;
      return s / 233280;
    };
  }
}
