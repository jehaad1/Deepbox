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
import { normalCdf, normalSf } from "./_internal";

/**
 * Bandwidth selection method: `"scott"`, `"silverman"`, or an explicit bandwidth, which is
 * the standard deviation of the Gaussian kernel in the units of the data.
 *
 * Unlike SciPy, where a number is a factor that is multiplied by the standard deviation of the
 * data, a number here is the absolute bandwidth.
 */
export type BandwidthMethod = "scott" | "silverman" | number;

/** Options for {@link gaussian_kde}. */
export interface GaussianKDEOptions {
  /** Bandwidth selection method or explicit bandwidth value. Default: `"scott"`. */
  bwMethod?: BandwidthMethod;
  /**
   * Bandwidth selection method or explicit bandwidth value. Default: `"scott"`.
   *
   * @deprecated Prefer {@link GaussianKDEOptions.bwMethod}.
   */
  bw_method?: BandwidthMethod;
  /**
   * Non-negative weights of the data points (normalized to sum to one). With weights, the
   * bandwidth rules use the effective sample size `1 / sum(w^2)` and the weighted standard
   * deviation, as in SciPy. Default: equal weights.
   */
  weights?: number[] | Float64Array;
}

const SQRT_2PI = Math.sqrt(2 * Math.PI);

/**
 * Gaussian kernel density estimator.
 *
 * Represents a kernel-density estimate using Gaussian kernels.
 * The estimator is fitted on a 1D dataset and can evaluate the
 * estimated density at arbitrary points.
 */
export class GaussianKDE {
  /** The fitted dataset (a copy of the input). */
  readonly dataset: Float64Array;
  /** Number of data points. */
  readonly nData: number;
  /** The bandwidth (standard deviation of the Gaussian kernel). */
  readonly bandwidth: number;
  /** Normalized weights of the data points, or `undefined` for equal weights. */
  readonly weights: Float64Array | undefined;
  /** Effective number of data points, `1 / sum(w^2)` (equal to `nData` without weights). */
  readonly neff: number;
  /** Precomputed: -0.5 / bandwidth^2. */
  private readonly _expFactor: number;
  /** Precomputed: log of the weights (only with weights). */
  private readonly _logWeights: Float64Array | undefined;

  /**
   * @param data - Data points (must be finite and non-empty)
   * @param bandwidth - Standard deviation of the Gaussian kernel (positive and finite)
   * @param weights - Optional non-negative weights (normalized internally)
   * @throws {InvalidParameterError} If the data are empty or not finite, the bandwidth is not
   *   positive, or the weights are invalid
   */
  constructor(data: Float64Array, bandwidth: number, weights?: Float64Array | number[]) {
    if (data.length === 0) {
      throw new InvalidParameterError("data must contain at least one element", "data");
    }
    if (!Number.isFinite(bandwidth) || bandwidth <= 0) {
      throw new InvalidParameterError(
        "bandwidth must be positive and finite",
        "bandwidth",
        bandwidth
      );
    }
    this.dataset = Float64Array.from(data);
    this.nData = data.length;
    this.bandwidth = bandwidth;
    this._expFactor = -0.5 / (bandwidth * bandwidth);
    if (weights === undefined) {
      this.weights = undefined;
      this._logWeights = undefined;
      this.neff = this.nData;
    } else {
      const w = normalizeWeights(weights, this.nData);
      this.weights = w;
      const logW = new Float64Array(w.length);
      let sumSq = 0;
      for (let i = 0; i < w.length; i++) {
        const wi = w[i] ?? 0;
        logW[i] = Math.log(wi);
        sumSq += wi * wi;
      }
      this._logWeights = logW;
      this.neff = 1 / sumSq;
    }
  }

  /**
   * Evaluate the estimated density at the given points.
   *
   * @param points - Points at which to evaluate the density
   * @returns Density values (`NaN` for `NaN` points)
   */
  evaluate(points: number[] | Float64Array): Float64Array {
    const m = points.length;
    const n = this.nData;
    const result = new Float64Array(m);
    const data = this.dataset;
    const w = this.weights;
    const expFactor = this._expFactor;
    const norm = 1 / (this.bandwidth * SQRT_2PI);
    for (let i = 0; i < m; i++) {
      const x = points[i] ?? Number.NaN;
      let sum = 0;
      if (w === undefined) {
        for (let j = 0; j < n; j++) {
          const diff = x - (data[j] as number);
          sum += Math.exp(expFactor * diff * diff);
        }
        sum /= n;
      } else {
        for (let j = 0; j < n; j++) {
          const diff = x - (data[j] as number);
          sum += (w[j] as number) * Math.exp(expFactor * diff * diff);
        }
      }
      result[i] = norm * sum;
    }
    return result;
  }

  /**
   * Evaluate the estimated density at the given points. Same as {@link evaluate}, named after
   * SciPy's `gaussian_kde.pdf`.
   */
  pdf(points: number[] | Float64Array): Float64Array {
    return this.evaluate(points);
  }

  /**
   * Evaluate the log-density at the given points.
   *
   * More numerically stable than log(evaluate(...)) for extreme values: it stays finite far
   * from the data, where the density itself underflows to zero.
   *
   * @param points - Array of points at which to evaluate
   * @returns Array of log-density values (`NaN` for `NaN` points)
   */
  logDensity(points: number[] | Float64Array): Float64Array {
    const m = points.length;
    const n = this.nData;
    const result = new Float64Array(m);
    const data = this.dataset;
    const logW = this._logWeights;
    const expFactor = this._expFactor;
    const logNorm = -Math.log(this.bandwidth * SQRT_2PI);
    const logN = Math.log(n);
    const exps = new Float64Array(n);

    for (let i = 0; i < m; i++) {
      const x = points[i] ?? Number.NaN;
      if (Number.isNaN(x)) {
        result[i] = Number.NaN;
        continue;
      }
      // Log-sum-exp trick for numerical stability
      let maxVal = Number.NEGATIVE_INFINITY;
      for (let j = 0; j < n; j++) {
        const diff = x - (data[j] as number);
        const val = expFactor * diff * diff + (logW === undefined ? 0 : (logW[j] as number));
        exps[j] = val;
        if (val > maxVal) maxVal = val;
      }
      if (maxVal === Number.NEGATIVE_INFINITY) {
        result[i] = Number.NEGATIVE_INFINITY;
        continue;
      }
      let sumExp = 0;
      for (let j = 0; j < n; j++) {
        sumExp += Math.exp((exps[j] as number) - maxVal);
      }
      result[i] = logNorm + maxVal + Math.log(sumExp) - (logW === undefined ? logN : 0);
    }
    return result;
  }

  /**
   * Evaluate the log-density at the given points. Same as {@link logDensity}, named after
   * SciPy's `gaussian_kde.logpdf`.
   */
  logpdf(points: number[] | Float64Array): Float64Array {
    return this.logDensity(points);
  }

  /**
   * Integrate the estimated density over the interval `[low, high]`.
   *
   * @param low - Lower limit (may be `-Infinity`)
   * @param high - Upper limit (may be `Infinity`)
   * @returns The probability mass of the estimate in the interval (negative if `low > high`)
   */
  integrateBox1d(low: number, high: number): number {
    if (Number.isNaN(low) || Number.isNaN(high)) return Number.NaN;
    const n = this.nData;
    const h = this.bandwidth;
    const w = this.weights;
    let total = 0;
    for (let j = 0; j < n; j++) {
      const xj = this.dataset[j] as number;
      const a = (low - xj) / h;
      const b = (high - xj) / h;
      // Use the upper tail when both limits lie above the center, to avoid cancellation.
      const mass = a > 0 ? normalSf(a) - normalSf(b) : normalCdf(b) - normalCdf(a);
      total += (w === undefined ? 1 / n : (w[j] as number)) * mass;
    }
    return total;
  }

  /**
   * Integrate the estimated density over the entire real line.
   * A Gaussian KDE integrates to 1 by construction.
   */
  integrate(): number {
    return 1.0;
  }
}

function normalizeWeights(weights: ArrayLike<number>, n: number): Float64Array {
  if (weights.length !== n) {
    throw new InvalidParameterError("weights must have the same length as data", "weights", {
      data: n,
      weights: weights.length,
    });
  }
  const w = new Float64Array(n);
  let total = 0;
  for (let i = 0; i < n; i++) {
    const wi = weights[i] as number;
    if (!Number.isFinite(wi) || wi < 0) {
      throw new InvalidParameterError("weights must be finite and non-negative", "weights", wi);
    }
    w[i] = wi;
    total += wi;
  }
  if (!(total > 0) || !Number.isFinite(total)) {
    throw new InvalidParameterError("weights must have a positive finite sum", "weights", total);
  }
  for (let i = 0; i < n; i++) w[i] = (w[i] as number) / total;
  return w;
}

/**
 * Create a Gaussian kernel density estimator from data.
 *
 * Bandwidth is selected automatically using Scott's rule (default)
 * or Silverman's rule, or can be specified explicitly. For one-dimensional data the
 * bandwidth is `factor * s`, where `s` is the sample standard deviation (`n - 1`
 * denominator) and the factor is `n^(-1/5)` (Scott) or `(3 n / 4)^(-1/5)` (Silverman),
 * as in `scipy.stats.gaussian_kde`. If all values are equal, `s` is replaced by 1.
 *
 * @param data - 1D array of finite data points
 * @param options - KDE options (`bwMethod`, `weights`)
 * @returns A GaussianKDE instance
 * @throws {InvalidParameterError} If the data are empty or contain non-finite values, the
 *   bandwidth method is invalid, or the weights are invalid
 *
 * @example
 * ```ts
 * import { gaussianKde } from 'deepbox/stats';
 *
 * const data = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5];
 * const kde = gaussianKde(data);
 *
 * // Evaluate density at specific points
 * const density = kde.evaluate([1.0, 2.0, 3.0]);
 *
 * // Use Silverman's rule
 * const kde2 = gaussianKde(data, { bwMethod: "silverman" });
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-distributions | Deepbox documentation}
 * @deprecated Prefer {@link gaussianKde}.
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
  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(arr[i])) {
      throw new InvalidParameterError("data must contain only finite values", "data", arr[i]);
    }
  }

  const bw = options.bwMethod ?? options.bw_method ?? "scott";
  if (typeof bw === "number") {
    if (!Number.isFinite(bw) || bw <= 0) {
      throw new InvalidParameterError("bandwidth must be positive and finite", "bwMethod", bw);
    }
    return new GaussianKDE(arr, bw, options.weights);
  }
  if (bw !== "scott" && bw !== "silverman") {
    throw new InvalidParameterError(
      'bwMethod must be "scott", "silverman", or a positive number',
      "bwMethod",
      bw
    );
  }

  // Effective sample size and (weighted) variance with the unbiased correction
  const w = options.weights === undefined ? undefined : normalizeWeights(options.weights, n);
  let mean = 0;
  let neff = n;
  let sumSqW = 1 / n;
  if (w === undefined) {
    for (let i = 0; i < n; i++) mean += arr[i] ?? 0;
    mean /= n;
  } else {
    sumSqW = 0;
    for (let i = 0; i < n; i++) {
      mean += (w[i] as number) * (arr[i] as number);
      sumSqW += (w[i] as number) ** 2;
    }
    neff = 1 / sumSqW;
  }
  let weightedSq = 0;
  for (let i = 0; i < n; i++) {
    const diff = (arr[i] as number) - mean;
    weightedSq += (w === undefined ? 1 / n : (w[i] as number)) * diff * diff;
  }
  const variance = weightedSq / (1 - sumSqW);
  const stdDev = Number.isFinite(variance) && variance > 0 ? Math.sqrt(variance) : 1;

  const factor = bw === "scott" ? neff ** -0.2 : ((3 * neff) / 4) ** -0.2;
  return new GaussianKDE(arr, factor * stdDev, options.weights);
}

/**
 * Create a Gaussian kernel density estimator from data.
 *
 * Bandwidth is selected automatically using Scott's rule (default)
 * or Silverman's rule, or can be specified explicitly. For one-dimensional data the
 * bandwidth is `factor * s`, where `s` is the sample standard deviation (`n - 1`
 * denominator) and the factor is `n^(-1/5)` (Scott) or `(3 n / 4)^(-1/5)` (Silverman),
 * as in `scipy.stats.gaussian_kde`. If all values are equal, `s` is replaced by 1.
 *
 * @param data - 1D array of finite data points
 * @param options - KDE options (`bwMethod`, `weights`)
 * @returns A GaussianKDE instance
 * @throws {InvalidParameterError} If the data are empty or contain non-finite values, the
 *   bandwidth method is invalid, or the weights are invalid
 *
 * @example
 * ```ts
 * import { gaussianKde } from 'deepbox/stats';
 *
 * const data = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5];
 * const kde = gaussianKde(data);
 *
 * // Evaluate density at specific points
 * const density = kde.evaluate([1.0, 2.0, 3.0]);
 *
 * // Use Silverman's rule
 * const kde2 = gaussianKde(data, { bwMethod: "silverman" });
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-distributions | Deepbox documentation}
 */
export const gaussianKde = gaussian_kde;
