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
import { logGamma, normalCdf, regularizedIncompleteBeta } from "./_internal";
import { t as studentT } from "./distributions";

// ---- Public API ----

/** Options for power analysis of a two-sample t-test. */
export interface TTestPowerOptions {
  /** Cohen's d effect size */
  effectSize?: number;
  /** Sample size of the first group (per group when `ratio` is 1) */
  nObs?: number;
  /** Significance level (Type I error rate) */
  alpha?: number;
  /** Statistical power (1 - Type II error rate) */
  power?: number;
  /** Number of tails: 1 or 2 (default: 2) */
  alternative?: 1 | 2;
  /** Ratio of the size of the second group to the first, `nObs2 / nObs` (default: 1) */
  ratio?: number;
}

/** Result of a power analysis. */
export interface PowerAnalysisResult {
  effectSize: number;
  nObs: number;
  alpha: number;
  power: number;
}

/**
 * Cumulative distribution function of the non-central t distribution (Algorithm AS 243,
 * Lenth, 1989, in the form used by R's `pt` with `ncp`).
 *
 * @param t - Evaluation point
 * @param df - Degrees of freedom (positive)
 * @param ncp - Non-centrality parameter
 */
function noncentralTCdf(t: number, df: number, ncp: number): number {
  if (Number.isNaN(t) || Number.isNaN(df) || Number.isNaN(ncp) || df <= 0) return Number.NaN;
  if (!Number.isFinite(t)) return t < 0 ? 0 : 1;
  if (ncp === 0) return studentT(df).cdf(t);

  let negdel: boolean;
  let tt: number;
  let del: number;
  if (t >= 0) {
    negdel = false;
    tt = t;
    del = ncp;
  } else {
    // pt(q, df, ncp) <= pt(0, df, ncp) = Phi(-ncp), which is negligible for large ncp.
    if (ncp > 40) return 0;
    negdel = true;
    tt = -t;
    del = -ncp;
  }

  if (df > 4e5 || del * del > 2 * Math.LN2 * 1021) {
    // Abramowitz and Stegun 26.7.10
    const s = 1 / (4 * df);
    const z = (tt * (1 - s) - del) / Math.sqrt(1 + tt * tt * 2 * s);
    return negdel ? 1 - normalCdf(z) : normalCdf(z);
  }

  let tnc: number;
  const x0 = t * t;
  const x = x0 / (x0 + df);
  if (x > 0) {
    const lambda = del * del;
    let p = 0.5 * Math.exp(-0.5 * lambda);
    if (p === 0) return negdel ? 1 : 0;
    let q = Math.sqrt(2 / Math.PI) * p * del;
    let s = 0.5 - p;
    if (s < 1e-7) s = -0.5 * Math.expm1(-0.5 * lambda);
    let a = 0.5;
    const b = 0.5 * df;
    const rxb = (df / (t * t + df)) ** b;
    const albeta = 0.5 * Math.log(Math.PI) + logGamma(b) - logGamma(0.5 + b);
    let xodd = regularizedIncompleteBeta(a, b, x);
    let godd = 2 * rxb * Math.exp(a * Math.log(x) - albeta);
    tnc = b * x;
    let xeven = tnc < Number.EPSILON ? tnc : 1 - rxb;
    let geven = tnc * rxb;
    tnc = p * xodd + q * xeven;

    for (let it = 1; it <= 1000; it++) {
      a += 1;
      xodd -= godd;
      xeven -= geven;
      godd *= (x * (a + b - 1)) / a;
      geven *= (x * (a + b - 0.5)) / (a + 0.5);
      p *= lambda / (2 * it);
      q *= lambda / (2 * it + 1);
      tnc += p * xodd + q * xeven;
      s -= p;
      if (s < -1e-10) break; // rounding error: the series has converged as far as it can
      if (s >= 0 && Math.abs(2 * s * (xodd - godd)) < 1e-12) break;
    }
  } else {
    tnc = 0;
  }
  tnc += normalCdf(-del);
  const cdf = Math.min(tnc, 1);
  return negdel ? 1 - cdf : cdf;
}

/**
 * Power analysis for independent two-sample t-test.
 *
 * Exactly **one** of `effectSize`, `nObs`, `alpha`, or `power` must be
 * omitted (undefined). The function solves for the missing parameter.
 *
 * The power is computed from the non-central t distribution with
 * `df = nObs * (1 + ratio) - 2` degrees of freedom and non-centrality
 * `effectSize * sqrt(nObs * ratio / (1 + ratio))`, summing both rejection regions for the
 * two-sided test (the same model as `statsmodels.stats.power.TTestIndPower`). When solving
 * for `nObs` the result is the smallest integer sample size that reaches the requested power.
 *
 * @param options - Parameters (leave exactly one undefined to solve for it)
 * @returns Object with all four parameters filled in
 * @throws {InvalidParameterError} If the number of omitted parameters is not exactly one, a
 *   value is outside its valid range, or the requested value cannot be reached
 *
 * @example
 * ```ts
 * import { tTestPower } from 'deepbox/stats';
 *
 * // Find required sample size for d=0.5, alpha=0.05, power=0.8
 * const result = tTestPower({ effectSize: 0.5, alpha: 0.05, power: 0.8 });
 * console.log(result.nObs); // 64
 *
 * // Find power for given n and effect size
 * const result2 = tTestPower({ effectSize: 0.5, nObs: 50, alpha: 0.05 });
 * console.log(result2.power); // 0.6969...
 * ```
 *
 * @see {@link https://deepbox.dev/docs/stats-tests | Deepbox documentation}
 */
export function tTestPower(options: TTestPowerOptions): PowerAnalysisResult {
  const alt = options.alternative ?? 2;
  if (alt !== 1 && alt !== 2) {
    throw new InvalidParameterError("alternative must be 1 or 2", "alternative", alt);
  }
  const ratio = options.ratio ?? 1;
  if (!Number.isFinite(ratio) || ratio <= 0) {
    throw new InvalidParameterError("ratio must be positive and finite", "ratio", ratio);
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
  if (
    options.effectSize !== undefined &&
    (!Number.isFinite(options.effectSize) || options.effectSize < 0)
  ) {
    throw new InvalidParameterError(
      "effectSize must be finite and non-negative",
      "effectSize",
      options.effectSize
    );
  }
  if (
    options.nObs !== undefined &&
    (!Number.isInteger(options.nObs) || options.nObs < 2 || options.nObs > 2 ** 40)
  ) {
    throw new InvalidParameterError("nObs must be an integer >= 2", "nObs", options.nObs);
  }
  if (options.alpha !== undefined && !(options.alpha > 0 && options.alpha < 1)) {
    throw new InvalidParameterError("alpha must be in (0, 1)", "alpha", options.alpha);
  }
  if (options.power !== undefined && !(options.power > 0 && options.power < 1)) {
    throw new InvalidParameterError("power must be in (0, 1)", "power", options.power);
  }

  // Power of the t-test given the other three parameters.
  const computePower = (d: number, n: number, alpha: number): number => {
    const df = n * (1 + ratio) - 2;
    const ncp = d * Math.sqrt((n * ratio) / (1 + ratio));
    const dist = studentT(df);
    if (alt === 2) {
      const tcrit = dist.isf(alpha / 2);
      // P(T > tcrit) + P(T < -tcrit) under the non-central distribution
      return 1 - noncentralTCdf(tcrit, df, ncp) + noncentralTCdf(-tcrit, df, ncp);
    }
    return 1 - noncentralTCdf(dist.isf(alpha), df, ncp);
  };

  if (options.power === undefined) {
    // Solve for power
    const d = options.effectSize ?? 0;
    const n = options.nObs ?? 2;
    const alpha = options.alpha ?? 0.05;
    return { effectSize: d, nObs: n, alpha, power: clamp01(computePower(d, n, alpha)) };
  }

  const targetPower = options.power;

  if (options.nObs === undefined) {
    // Solve for the smallest n by bracketing and bisection (power increases with n)
    const d = options.effectSize ?? 0;
    const alpha = options.alpha ?? 0.05;

    if (d === 0) {
      throw new InvalidParameterError("Cannot solve for nObs with effectSize = 0", "effectSize", d);
    }

    let lo = 2;
    let hi = 4;
    while (computePower(d, hi, alpha) < targetPower) {
      lo = hi;
      hi *= 2;
      if (hi > 2 ** 40) {
        throw new InvalidParameterError(
          "The requested power cannot be reached with a feasible sample size",
          "power",
          targetPower
        );
      }
    }
    if (computePower(d, lo, alpha) >= targetPower) {
      hi = lo;
    } else {
      while (hi - lo > 1) {
        const mid = Math.floor((lo + hi) / 2);
        if (computePower(d, mid, alpha) < targetPower) {
          lo = mid;
        } else {
          hi = mid;
        }
      }
    }
    return { effectSize: d, nObs: hi, alpha, power: clamp01(computePower(d, hi, alpha)) };
  }

  if (options.effectSize === undefined) {
    // Solve for effect size via bisection (power increases with d)
    const n = options.nObs;
    const alpha = options.alpha ?? 0.05;

    const floorPower = computePower(0, n, alpha);
    if (targetPower < floorPower * (1 - 1e-9)) {
      throw new InvalidParameterError(
        "power must not be smaller than the power at effectSize = 0 (about alpha)",
        "power",
        targetPower
      );
    }
    let lo = 0;
    let hi = 1;
    while (computePower(hi, n, alpha) < targetPower) {
      lo = hi;
      hi *= 2;
      if (hi > 1e6) {
        throw new InvalidParameterError(
          "The requested power cannot be reached",
          "power",
          targetPower
        );
      }
    }
    for (let iter = 0; iter < 200 && hi - lo > 1e-15 * Math.max(1, hi); iter++) {
      const mid = (lo + hi) / 2;
      if (computePower(mid, n, alpha) < targetPower) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    const d = (lo + hi) / 2;
    return { effectSize: d, nObs: n, alpha, power: clamp01(computePower(d, n, alpha)) };
  }

  // Solve for alpha via bisection in log space (power increases with alpha)
  const d = options.effectSize;
  const n = options.nObs;
  let lo = 1e-12;
  let hi = 1 - 1e-12;
  if (targetPower < computePower(d, n, lo) || targetPower > computePower(d, n, hi)) {
    throw new InvalidParameterError(
      "The requested power cannot be reached with any alpha in (0, 1)",
      "power",
      targetPower
    );
  }
  for (let iter = 0; iter < 200; iter++) {
    const mid = iter < 100 ? Math.sqrt(lo * hi) : (lo + hi) / 2;
    if (computePower(d, n, mid) < targetPower) {
      lo = mid; // need more alpha to get more power
    } else {
      hi = mid;
    }
    if (hi - lo <= 1e-15 * hi) break;
  }
  const alpha = (lo + hi) / 2;
  return { effectSize: d, nObs: n, alpha, power: clamp01(computePower(d, n, alpha)) };
}

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}
