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

/**
 * Compute a confidence interval for the mean of a sample using the t-distribution.
 *
 * Use this when the population standard deviation is unknown (most common case).
 *
 * @param data - Array of sample values
 * @param confidenceLevel - Confidence level (default 0.95 for 95%)
 * @returns Confidence interval result
 *
 * @example
 * ```ts
 * import { meanConfidenceInterval } from 'deepbox/stats';
 *
 * const data = [2.3, 2.5, 2.1, 2.4, 2.6, 2.2];
 * const ci = meanConfidenceInterval(data, 0.95);
 * console.log(`95% CI: [${ci.lower.toFixed(3)}, ${ci.upper.toFixed(3)}]`);
 * ```
 */
export function meanConfidenceInterval(data: number[], confidenceLevel = 0.95): ConfidenceInterval {
  if (data.length < 2) {
    throw new InvalidParameterError(
      "Need at least 2 data points for confidence interval",
      "data",
      data.length
    );
  }
  if (confidenceLevel <= 0 || confidenceLevel >= 1) {
    throw new InvalidParameterError(
      "confidenceLevel must be between 0 and 1 (exclusive)",
      "confidenceLevel",
      confidenceLevel
    );
  }

  const n = data.length;
  const sampleMean = data.reduce((a, b) => a + b, 0) / n;
  const sampleVar = data.reduce((acc, v) => acc + (v - sampleMean) ** 2, 0) / (n - 1);
  const sampleStd = Math.sqrt(sampleVar);
  const se = sampleStd / Math.sqrt(n);

  const alpha = 1 - confidenceLevel;
  const df = n - 1;
  const dist = tDist(df);
  const tCrit = dist.ppf(1 - alpha / 2);

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
 * @param data - Array of sample values
 * @param popStd - Known population standard deviation
 * @param confidenceLevel - Confidence level (default 0.95)
 * @returns Confidence interval result
 *
 * @example
 * ```ts
 * import { meanConfidenceIntervalZ } from 'deepbox/stats';
 *
 * const ci = meanConfidenceIntervalZ([10, 12, 11, 13, 9], 2.0, 0.99);
 * ```
 */
export function meanConfidenceIntervalZ(
  data: number[],
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
  if (popStd <= 0) {
    throw new InvalidParameterError("popStd must be positive", "popStd", popStd);
  }
  if (confidenceLevel <= 0 || confidenceLevel >= 1) {
    throw new InvalidParameterError(
      "confidenceLevel must be between 0 and 1 (exclusive)",
      "confidenceLevel",
      confidenceLevel
    );
  }

  const n = data.length;
  const sampleMean = data.reduce((a, b) => a + b, 0) / n;
  const se = popStd / Math.sqrt(n);

  const alpha = 1 - confidenceLevel;
  const dist = norm(0, 1);
  const zCrit = dist.ppf(1 - alpha / 2);

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
 * Compute a confidence interval for a proportion (Wald interval).
 *
 * @param successes - Number of successes
 * @param total - Total number of trials
 * @param confidenceLevel - Confidence level (default 0.95)
 * @returns Confidence interval result (mean is the sample proportion)
 *
 * @example
 * ```ts
 * import { proportionConfidenceInterval } from 'deepbox/stats';
 *
 * const ci = proportionConfidenceInterval(45, 100, 0.95);
 * console.log(`Proportion: ${ci.mean}, 95% CI: [${ci.lower}, ${ci.upper}]`);
 * ```
 */
export function proportionConfidenceInterval(
  successes: number,
  total: number,
  confidenceLevel = 0.95
): ConfidenceInterval {
  if (total < 1) {
    throw new InvalidParameterError("total must be at least 1", "total", total);
  }
  if (successes < 0 || successes > total) {
    throw new InvalidParameterError(
      "successes must be between 0 and total",
      "successes",
      successes
    );
  }
  if (confidenceLevel <= 0 || confidenceLevel >= 1) {
    throw new InvalidParameterError(
      "confidenceLevel must be between 0 and 1 (exclusive)",
      "confidenceLevel",
      confidenceLevel
    );
  }

  const p = successes / total;
  const se = Math.sqrt((p * (1 - p)) / total);

  const alpha = 1 - confidenceLevel;
  const dist = norm(0, 1);
  const zCrit = dist.ppf(1 - alpha / 2);

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
 * Uses Welch's t-test approximation for unequal variances.
 *
 * @param data1 - First sample
 * @param data2 - Second sample
 * @param confidenceLevel - Confidence level (default 0.95)
 * @returns Confidence interval for (mean1 - mean2)
 */
export function meanDiffConfidenceInterval(
  data1: number[],
  data2: number[],
  confidenceLevel = 0.95
): ConfidenceInterval {
  if (data1.length < 2 || data2.length < 2) {
    throw new InvalidParameterError("Need at least 2 data points in each sample", "data", [
      data1.length,
      data2.length,
    ]);
  }
  if (confidenceLevel <= 0 || confidenceLevel >= 1) {
    throw new InvalidParameterError(
      "confidenceLevel must be between 0 and 1 (exclusive)",
      "confidenceLevel",
      confidenceLevel
    );
  }

  const n1 = data1.length;
  const n2 = data2.length;
  const mean1 = data1.reduce((a, b) => a + b, 0) / n1;
  const mean2 = data2.reduce((a, b) => a + b, 0) / n2;
  const var1 = data1.reduce((acc, v) => acc + (v - mean1) ** 2, 0) / (n1 - 1);
  const var2 = data2.reduce((acc, v) => acc + (v - mean2) ** 2, 0) / (n2 - 1);

  const se = Math.sqrt(var1 / n1 + var2 / n2);

  // Welch-Satterthwaite degrees of freedom
  const num = (var1 / n1 + var2 / n2) ** 2;
  const denom = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1);
  const df = Math.max(1, Math.floor(num / denom));

  const alpha = 1 - confidenceLevel;
  const dist = tDist(df);
  const tCrit = dist.ppf(1 - alpha / 2);

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
