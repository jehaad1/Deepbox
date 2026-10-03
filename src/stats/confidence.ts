/**
 * Confidence interval computation utilities.
 *
 * @module stats/confidence
 * @see {@link https://deepbox.dev/docs/stats-distributions | Deepbox Confidence Intervals}
 */

import { InvalidParameterError } from "../core";
import { norm, t as tDist } from "./distributions";

/**
 * Result of a confidence interval computation.
 */
export interface ConfidenceInterval {
  /** Lower bound of the interval */
  readonly lower: number;
  /** Upper bound of the interval */
  readonly upper: number;
  /** Point estimate (center) */
  readonly mean: number;
  /** Margin of error */
  readonly marginOfError: number;
  /** Confidence level used */
  readonly confidenceLevel: number;
}

/** Method used to build a confidence interval for a proportion. */
export type ProportionIntervalMethod = "wald" | "wilson";

function assertConfidenceLevel(confidenceLevel: number): void {
  // Written as a negated conjunction so NaN is rejected too.
  if (!(confidenceLevel > 0 && confidenceLevel < 1)) {
    throw new InvalidParameterError(
      "confidenceLevel must be between 0 and 1 (exclusive)",
      "confidenceLevel",
      confidenceLevel
    );
  }
}

/** Sample mean and unbiased sample variance (two-pass), for `data.length >= 2`. */
function meanAndVariance(data: readonly number[]): { mean: number; variance: number } {
  const n = data.length;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += data[i] as number;
  const mean = sum / n;
  let ss = 0;
  for (let i = 0; i < n; i++) {
    const d = (data[i] as number) - mean;
    ss += d * d;
  }
  return { mean, variance: ss / (n - 1) };
}

/**
 * Two-sided critical value of Student's t with `df` degrees of freedom.
 * Evaluated in the lower tail (`-ppf(alpha/2)`), where `alpha/2` is exact, rather
 * than as `ppf(1 - alpha/2)`, which loses digits for confidence levels near 1.
 */
function tCritical(confidenceLevel: number, df: number): number {
  const alpha = 1 - confidenceLevel;
  return -tDist(df).ppf(alpha / 2);
}

/** Two-sided critical value of the standard normal distribution. */
function zCritical(confidenceLevel: number): number {
  const alpha = 1 - confidenceLevel;
  return -norm(0, 1).ppf(alpha / 2);
}

/**
 * Compute a confidence interval for the mean of a sample using the t-distribution.
 *
 * Use this when the population standard deviation is unknown (most common case).
 *
 * @param data - Array of sample values (at least 2)
 * @param confidenceLevel - Confidence level, strictly between 0 and 1 (default 0.95 for 95%)
 * @returns Confidence interval result
 * @throws {InvalidParameterError} If `data` has fewer than 2 values or `confidenceLevel`
 *   is not strictly between 0 and 1
 *
 * @example
 * ```ts
 * import { meanConfidenceInterval } from 'deepbox/stats';
 *
 * const data = [2.3, 2.5, 2.1, 2.4, 2.6, 2.2];
 * const ci = meanConfidenceInterval(data, 0.95);
 * console.log(`95% CI: [${ci.lower.toFixed(3)}, ${ci.upper.toFixed(3)}]`);
 * ```
 *
 * @remarks
 * If `data` contains NaN or an infinite value, the interval bounds are NaN.
 */
export function meanConfidenceInterval(
  data: readonly number[],
  confidenceLevel = 0.95
): ConfidenceInterval {
  if (data.length < 2) {
    throw new InvalidParameterError(
      "Need at least 2 data points for confidence interval",
      "data",
      data.length
    );
  }
  assertConfidenceLevel(confidenceLevel);

  const n = data.length;
  const { mean: sampleMean, variance: sampleVar } = meanAndVariance(data);
  const se = Math.sqrt(sampleVar) / Math.sqrt(n);

  const tCrit = tCritical(confidenceLevel, n - 1);

  const marginOfError = tCrit * se;
  return {
    lower: sampleMean - marginOfError,
    upper: sampleMean + marginOfError,
    mean: sampleMean,
    marginOfError,
    confidenceLevel,
  };
}

/**
 * Compute a confidence interval for the mean using a known population standard deviation
 * (z-interval).
 *
 * @param data - Array of sample values (at least 1)
 * @param popStd - Known population standard deviation (positive and finite)
 * @param confidenceLevel - Confidence level, strictly between 0 and 1 (default 0.95)
 * @returns Confidence interval result
 * @throws {InvalidParameterError} If `data` is empty, `popStd` is not a positive finite
 *   number, or `confidenceLevel` is not strictly between 0 and 1
 *
 * @example
 * ```ts
 * import { meanConfidenceIntervalZ } from 'deepbox/stats';
 *
 * const ci = meanConfidenceIntervalZ([10, 12, 11, 13, 9], 2.0, 0.99);
 * ```
 */
export function meanConfidenceIntervalZ(
  data: readonly number[],
  popStd: number,
  confidenceLevel = 0.95
): ConfidenceInterval {
  if (data.length < 1) {
    throw new InvalidParameterError(
      "Need at least 1 data point for z confidence interval",
      "data",
      data.length
    );
  }
  if (!(popStd > 0) || !Number.isFinite(popStd)) {
    throw new InvalidParameterError("popStd must be positive and finite", "popStd", popStd);
  }
  assertConfidenceLevel(confidenceLevel);

  const n = data.length;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += data[i] as number;
  const sampleMean = sum / n;
  const se = popStd / Math.sqrt(n);

  const zCrit = zCritical(confidenceLevel);

  const marginOfError = zCrit * se;
  return {
    lower: sampleMean - marginOfError,
    upper: sampleMean + marginOfError,
    mean: sampleMean,
    marginOfError,
    confidenceLevel,
  };
}

/**
 * Compute a confidence interval for a proportion.
 *
 * The default is the Wald (normal approximation) interval, `p ± z·sqrt(p(1-p)/n)`,
 * which has poor coverage for small samples or proportions near 0 or 1 (it
 * collapses to zero width when p is 0 or 1). Pass `"wilson"` for the Wilson
 * score interval, which does not have those problems.
 *
 * @param successes - Number of successes (between 0 and `total`)
 * @param total - Total number of trials (at least 1)
 * @param confidenceLevel - Confidence level, strictly between 0 and 1 (default 0.95)
 * @param method - `"wald"` (default) or `"wilson"`
 * @returns Confidence interval result (`mean` is the sample proportion; `marginOfError` is
 *   the half-width of the interval before clipping to [0, 1])
 * @throws {InvalidParameterError} If `total` is not at least 1, `successes` is outside
 *   `[0, total]`, `confidenceLevel` is not strictly between 0 and 1, or `method` is unknown
 *
 * @example
 * ```ts
 * import { proportionConfidenceInterval } from 'deepbox/stats';
 *
 * const ci = proportionConfidenceInterval(45, 100, 0.95);
 * console.log(`Proportion: ${ci.mean}, 95% CI: [${ci.lower}, ${ci.upper}]`);
 *
 * const wilson = proportionConfidenceInterval(0, 20, 0.95, "wilson");
 * ```
 */
export function proportionConfidenceInterval(
  successes: number,
  total: number,
  confidenceLevel = 0.95,
  method: ProportionIntervalMethod = "wald"
): ConfidenceInterval {
  if (!(total >= 1) || !Number.isFinite(total)) {
    throw new InvalidParameterError("total must be at least 1", "total", total);
  }
  if (!(successes >= 0 && successes <= total)) {
    throw new InvalidParameterError(
      "successes must be between 0 and total",
      "successes",
      successes
    );
  }
  assertConfidenceLevel(confidenceLevel);
  if (method !== "wald" && method !== "wilson") {
    throw new InvalidParameterError(
      `method must be "wald" or "wilson"; received ${String(method)}`,
      "method",
      method
    );
  }

  const p = successes / total;
  const zCrit = zCritical(confidenceLevel);

  if (method === "wilson") {
    const z2n = (zCrit * zCrit) / total;
    const denom = 1 + z2n;
    const center = (p + z2n / 2) / denom;
    const marginOfError = (zCrit * Math.sqrt((p * (1 - p)) / total + z2n / (4 * total))) / denom;
    // At p = 0 (p = 1) the Wilson lower (upper) bound is exactly 0 (1); the
    // rounded center - margin would otherwise leave a residue of about 1e-17.
    return {
      lower: successes === 0 ? 0 : Math.max(0, center - marginOfError),
      upper: successes === total ? 1 : Math.min(1, center + marginOfError),
      mean: p,
      marginOfError,
      confidenceLevel,
    };
  }

  const se = Math.sqrt((p * (1 - p)) / total);
  const marginOfError = zCrit * se;
  return {
    lower: Math.max(0, p - marginOfError),
    upper: Math.min(1, p + marginOfError),
    mean: p,
    marginOfError,
    confidenceLevel,
  };
}

/**
 * Compute a confidence interval for the difference of two means (independent samples).
 *
 * By default uses Welch's t interval for unequal variances, with the
 * (fractional) Welch-Satterthwaite degrees of freedom, as `scipy.stats.ttest_ind(...,
 * equal_var=False).confidence_interval()` does. With `equalVar = true` it uses
 * the pooled-variance interval with `n1 + n2 - 2` degrees of freedom.
 *
 * @param data1 - First sample (at least 2 values)
 * @param data2 - Second sample (at least 2 values)
 * @param confidenceLevel - Confidence level, strictly between 0 and 1 (default 0.95)
 * @param equalVar - Assume equal population variances and pool them (default false)
 * @returns Confidence interval for (mean1 - mean2)
 * @throws {InvalidParameterError} If either sample has fewer than 2 values or
 *   `confidenceLevel` is not strictly between 0 and 1
 *
 * @example
 * ```ts
 * import { meanDiffConfidenceInterval } from 'deepbox/stats';
 *
 * const ci = meanDiffConfidenceInterval([5.1, 4.9, 5.6, 5.8], [4.2, 4.8, 4.4, 5.0], 0.95);
 * ```
 */
export function meanDiffConfidenceInterval(
  data1: readonly number[],
  data2: readonly number[],
  confidenceLevel = 0.95,
  equalVar = false
): ConfidenceInterval {
  if (data1.length < 2 || data2.length < 2) {
    throw new InvalidParameterError("Need at least 2 data points in each sample", "data", [
      data1.length,
      data2.length,
    ]);
  }
  assertConfidenceLevel(confidenceLevel);

  const n1 = data1.length;
  const n2 = data2.length;
  const { mean: mean1, variance: var1 } = meanAndVariance(data1);
  const { mean: mean2, variance: var2 } = meanAndVariance(data2);

  let se: number;
  let df: number;
  if (equalVar) {
    const pooled = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2);
    se = Math.sqrt(pooled * (1 / n1 + 1 / n2));
    df = n1 + n2 - 2;
  } else {
    const v1 = var1 / n1;
    const v2 = var2 / n2;
    se = Math.sqrt(v1 + v2);
    // Welch-Satterthwaite degrees of freedom. With two constant samples the
    // interval has zero width, so any df works; use the pooled one.
    const denom = (v1 * v1) / (n1 - 1) + (v2 * v2) / (n2 - 1);
    df = denom === 0 ? n1 + n2 - 2 : ((v1 + v2) * (v1 + v2)) / denom;
  }

  const tCrit = tCritical(confidenceLevel, df);

  const diffMean = mean1 - mean2;
  const marginOfError = tCrit * se;

  return {
    lower: diffMean - marginOfError,
    upper: diffMean + marginOfError,
    mean: diffMean,
    marginOfError,
    confidenceLevel,
  };
}
