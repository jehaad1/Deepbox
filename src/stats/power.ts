/**
 * Statistical power analysis.
 *
 * Compute required sample size, statistical power, minimum detectable
 * effect size, or significance level for common test designs.
 *
 * @module stats/power
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox documentation}
 */

import { InvalidParameterError } from "../core";
import { normalCdf, normalPpf } from "./_internal";

// ---- Public API ----

/** Options for power analysis of a two-sample t-test. */
export interface TTestPowerOptions {
  /** Cohen's d effect size */
  effectSize?: number;
  /** Sample size per group */
  nObs?: number;
  /** Significance level (Type I error rate) */
  alpha?: number;
  /** Statistical power (1 - Type II error rate) */
  power?: number;
  /** Number of tails: 1 or 2 (default: 2) */
  alternative?: 1 | 2;
}

/** Result of a power analysis. */
export interface PowerAnalysisResult {
  effectSize: number;
  nObs: number;
  alpha: number;
  power: number;
}

/**
 * Power analysis for independent two-sample t-test.
 *
 * Exactly **one** of `effectSize`, `nObs`, `alpha`, or `power` must be
 * omitted (undefined). The function solves for the missing parameter.
 *
 * Uses the normal approximation to the non-central t distribution.
 *
 * @param options - Parameters (leave exactly one undefined to solve for it)
 * @returns Object with all four parameters filled in
 *
 * @example
 * ```ts
 * import { tTestPower } from 'deepbox/stats';
 *
 * // Find required sample size for d=0.5, alpha=0.05, power=0.8
 * const result = tTestPower({ effectSize: 0.5, alpha: 0.05, power: 0.8 });
 * console.log(result.nObs); // ~64
 *
 * // Find power for given n and effect size
 * const result2 = tTestPower({ effectSize: 0.5, nObs: 50, alpha: 0.05 });
 * console.log(result2.power); // ~0.70
 * ```
 */
export function tTestPower(options: TTestPowerOptions): PowerAnalysisResult {
  const alt = options.alternative ?? 2;
  if (alt !== 1 && alt !== 2) {
    throw new InvalidParameterError("alternative must be 1 or 2", "alternative", alt);
  }

  const defined = [
    options.effectSize !== undefined,
    options.nObs !== undefined,
    options.alpha !== undefined,
    options.power !== undefined,
  ];
  const nDefined = defined.filter(Boolean).length;

  if (nDefined !== 3) {
    throw new InvalidParameterError(
      "Exactly one of effectSize, nObs, alpha, power must be undefined",
      "options",
      options
    );
  }

  // Validate provided values
  if (options.effectSize !== undefined && options.effectSize < 0) {
    throw new InvalidParameterError(
      "effectSize must be non-negative",
      "effectSize",
      options.effectSize
    );
  }
  if (options.nObs !== undefined && (options.nObs < 2 || !Number.isInteger(options.nObs))) {
    throw new InvalidParameterError("nObs must be an integer >= 2", "nObs", options.nObs);
  }
  if (options.alpha !== undefined && (options.alpha <= 0 || options.alpha >= 1)) {
    throw new InvalidParameterError("alpha must be in (0, 1)", "alpha", options.alpha);
  }
  if (options.power !== undefined && (options.power <= 0 || options.power >= 1)) {
    throw new InvalidParameterError("power must be in (0, 1)", "power", options.power);
  }

  // Helper: compute power given the other three
  const computePower = (d: number, n: number, alpha: number): number => {
    const zAlpha = alt === 2 ? normalPpf(1 - alpha / 2) : normalPpf(1 - alpha);
    const ncp = d * Math.sqrt(n / 2); // non-centrality parameter
    return 1 - normalCdf(zAlpha - ncp);
  };

  if (options.power === undefined) {
    // Solve for power
    const d = options.effectSize ?? 0;
    const n = options.nObs ?? 2;
    const alpha = options.alpha ?? 0.05;
    return { effectSize: d, nObs: n, alpha, power: computePower(d, n, alpha) };
  }

  if (options.nObs === undefined) {
    // Solve for n via bisection
    const d = options.effectSize ?? 0;
    const alpha = options.alpha ?? 0.05;
    const targetPower = options.power;

    if (d === 0) {
      throw new InvalidParameterError("Cannot solve for nObs with effectSize = 0", "effectSize", d);
    }

    let lo = 2;
    let hi = 10;
    // Find upper bound
    while (computePower(d, hi, alpha) < targetPower && hi < 1e8) {
      hi *= 2;
    }

    // Bisect
    while (hi - lo > 1) {
      const mid = Math.floor((lo + hi) / 2);
      if (computePower(d, mid, alpha) < targetPower) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    // Use ceiling
    const n = computePower(d, lo, alpha) >= targetPower ? lo : hi;
    return { effectSize: d, nObs: n, alpha, power: computePower(d, n, alpha) };
  }

  if (options.effectSize === undefined) {
    // Solve for effect size via bisection
    const n = options.nObs ?? 2;
    const alpha = options.alpha ?? 0.05;
    const targetPower = options.power;

    let lo = 0;
    let hi = 10;
    for (let iter = 0; iter < 100; iter++) {
      const mid = (lo + hi) / 2;
      if (computePower(mid, n, alpha) < targetPower) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    const d = (lo + hi) / 2;
    return { effectSize: d, nObs: n, alpha, power: computePower(d, n, alpha) };
  }

  // Solve for alpha via bisection
  const d = options.effectSize ?? 0;
  const n = options.nObs ?? 2;
  const targetPower = options.power;

  let lo = 1e-10;
  let hi = 1 - 1e-10;
  for (let iter = 0; iter < 100; iter++) {
    const mid = (lo + hi) / 2;
    if (computePower(d, n, mid) < targetPower) {
      lo = mid; // need more alpha to get more power
    } else {
      hi = mid;
    }
  }
  const alpha = (lo + hi) / 2;
  return { effectSize: d, nObs: n, alpha, power: computePower(d, n, alpha) };
}
