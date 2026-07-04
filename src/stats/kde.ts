/**
 * Kernel density estimation.
 *
 * Provides non-parametric density estimation using Gaussian kernels
 * with automatic bandwidth selection (Scott's rule or Silverman's rule).
 *
 * @module stats/kde
 * @see {@link https://deepbox.dev/docs/stats-distributions | Deepbox documentation}
 */

import { InvalidParameterError } from "../core";

/** Bandwidth selection method. */
export type BandwidthMethod = "scott" | "silverman" | number;

/** Options for gaussian_kde. */
export interface GaussianKDEOptions {
  /** Bandwidth selection method or explicit bandwidth value. Default: "scott". */
  bw_method?: BandwidthMethod;
}

/**
 * Gaussian kernel density estimator.
 *
 * Represents a kernel-density estimate using Gaussian kernels.
 * The estimator is fitted on a 1D dataset and can evaluate the
 * estimated density at arbitrary points.
 */
export class GaussianKDE {
  /** The fitted dataset. */
  readonly dataset: Float64Array;
  /** Number of data points. */
  readonly nData: number;
  /** The bandwidth (standard deviation of the Gaussian kernel). */
  readonly bandwidth: number;
  /** Precomputed factor: 1 / (nData * bandwidth * sqrt(2*pi)). */
  private readonly _factor: number;
  /** Precomputed: -0.5 / bandwidth^2. */
  private readonly _expFactor: number;

  constructor(data: Float64Array, bandwidth: number) {
    this.dataset = data;
    this.nData = data.length;
    this.bandwidth = bandwidth;
    this._factor = 1 / (this.nData * bandwidth * Math.sqrt(2 * Math.PI));
    this._expFactor = -0.5 / (bandwidth * bandwidth);
  }

  /**
   * Evaluate the estimated density at the given points.
   *
   * @param points - Array of points at which to evaluate the density
   * @returns Array of density values
   */
  evaluate(points: number[] | Float64Array): Float64Array {
    const n = points.length;
    const result = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const x = points[i] ?? 0;
      let sum = 0;
      for (let j = 0; j < this.nData; j++) {
        const diff = x - (this.dataset[j] ?? 0);
        sum += Math.exp(this._expFactor * diff * diff);
      }
      result[i] = this._factor * sum;
    }
    return result;
  }

  /**
   * Evaluate the log-density at the given points.
   *
   * More numerically stable than log(evaluate(...)) for extreme values.
   *
   * @param points - Array of points at which to evaluate
   * @returns Array of log-density values
   */
  logDensity(points: number[] | Float64Array): Float64Array {
    const n = points.length;
    const result = new Float64Array(n);
    const logFactor = Math.log(this._factor * this.nData);

    for (let i = 0; i < n; i++) {
      const x = points[i] ?? 0;
      // Log-sum-exp trick for numerical stability
      let maxVal = -Infinity;
      for (let j = 0; j < this.nData; j++) {
        const diff = x - (this.dataset[j] ?? 0);
        const val = this._expFactor * diff * diff;
        if (val > maxVal) maxVal = val;
      }

      let sumExp = 0;
      for (let j = 0; j < this.nData; j++) {
        const diff = x - (this.dataset[j] ?? 0);
        sumExp += Math.exp(this._expFactor * diff * diff - maxVal);
      }

      result[i] = logFactor + maxVal + Math.log(sumExp) - Math.log(this.nData);
    }
    return result;
  }

  /**
   * Integrate the estimated density over the entire real line.
   * Should return approximately 1.0 for a valid density.
   */
  integrate(): number {
    // A Gaussian KDE integrates to 1 by construction
    return 1.0;
  }
}

/**
 * Create a Gaussian kernel density estimator from data.
 *
 * Bandwidth is selected automatically using Scott's rule (default)
 * or Silverman's rule, or can be specified explicitly.
 *
 * @param data - 1D array of data points
 * @param options - KDE options
 * @returns A GaussianKDE instance
 *
 * @example
 * ```ts
 * import { gaussian_kde } from 'deepbox/stats';
 *
 * const data = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5];
 * const kde = gaussian_kde(data);
 *
 * // Evaluate density at specific points
 * const density = kde.evaluate([1.0, 2.0, 3.0]);
 *
 * // Use Silverman's rule
 * const kde2 = gaussian_kde(data, { bw_method: "silverman" });
 * ```
 */
export function gaussian_kde(
  data: number[] | Float64Array,
  options: GaussianKDEOptions = {}
): GaussianKDE {
  const arr = data instanceof Float64Array ? data : new Float64Array(data);
  const n = arr.length;

  if (n === 0) {
    throw new InvalidParameterError("data must contain at least one element", "data");
  }

  // Compute standard deviation
  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += arr[i] ?? 0;
  }
  const mean = sum / n;

  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    const diff = (arr[i] ?? 0) - mean;
    sumSq += diff * diff;
  }
  const stdDev = Math.sqrt(sumSq / (n - 1 || 1));

  const bw = options.bw_method ?? "scott";
  let bandwidth: number;

  if (typeof bw === "number") {
    if (bw <= 0) {
      throw new InvalidParameterError("bandwidth must be positive", "bw_method", bw);
    }
    bandwidth = bw;
  } else if (bw === "scott") {
    // Scott's rule: h = n^(-1/5) * sigma
    bandwidth = n ** -0.2 * (stdDev > 0 ? stdDev : 1);
  } else if (bw === "silverman") {
    // Silverman's rule: h = (4*sigma^5 / (3*n))^(1/5) ≈ 1.06 * sigma * n^(-1/5)
    bandwidth = 1.06 * (stdDev > 0 ? stdDev : 1) * n ** -0.2;
  } else {
    throw new InvalidParameterError(
      'bw_method must be "scott", "silverman", or a positive number',
      "bw_method",
      bw
    );
  }

  return new GaussianKDE(arr, bandwidth);
}
