import { InvalidParameterError } from "../core";
import type { Tensor } from "../ndarray";
import {
  chiSquareCdf,
  fCdf,
  forEachIndexOffset,
  getNumberAt,
  normalCdf,
  rankData,
  studentTCdf,
} from "./_internal";

export type TestResult = {
  statistic: number;
  pvalue: number;
};

function toDenseSortedArray1D(x: Tensor): Float64Array {
  if (x.size < 1) {
    return new Float64Array(0);
  }
  const out = new Float64Array(x.size);
  let i = 0;
  forEachIndexOffset(x, (off) => {
    out[i] = getNumberAt(x, off);
    i++;
  });
  out.sort((a, b) => a - b);
  return out;
}

function toDenseArray1D(x: Tensor): Float64Array {
  const out = new Float64Array(x.size);
  let i = 0;
  forEachIndexOffset(x, (off) => {
    out[i] = getNumberAt(x, off);
    i++;
  });
  return out;
}

function meanAndM2(x: Float64Array): { mean: number; m2: number } {
  if (x.length === 0) {
    throw new InvalidParameterError("expected at least one element", "length", x.length);
  }
  let mean = 0;
  let m2 = 0;
  for (let i = 0; i < x.length; i++) {
    const v = x[i] ?? 0;
    const n = i + 1;
    const delta = v - mean;
    mean += delta / n;
    const delta2 = v - mean;
    m2 += delta * delta2;
  }
  return { mean, m2 };
}

function shapiroWilk(x: Float64Array): TestResult {
  // Ported from Algorithm AS R94 (Royston, 1995), following implementations
  // derived from the original FORTRAN and common ports.
  const n = x.length;
  if (n < 3 || n > 5000) {
    throw new InvalidParameterError("shapiro() sample size must be between 3 and 5000", "n", n);
  }

  const range = (x[n - 1] ?? 0) - (x[0] ?? 0);
  if (range === 0) {
    throw new InvalidParameterError("shapiro() all x values are identical", "range", range);
  }

  const small = 1e-19;
  if (range < small) {
    throw new InvalidParameterError("shapiro() range is too small", "range", range);
  }

  const nn2 = Math.floor(n / 2);
  const a = new Float64Array(nn2 + 1); // 1-based

  const g: readonly number[] = [-2.273, 0.459];
  const c1: readonly number[] = [0.0, 0.221157, -0.147981, -2.07119, 4.434685, -2.706056];
  const c2: readonly number[] = [0.0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633];
  const c3: readonly number[] = [0.544, -0.39978, 0.025054, -6.714e-4];
  const c4: readonly number[] = [1.3822, -0.77857, 0.062767, -0.0020322];
  const c5: readonly number[] = [-1.5861, -0.31082, -0.083751, 0.0038915];
  const c6: readonly number[] = [-0.4803, -0.082676, 0.0030302];

  const poly = (cc: readonly number[], x0: number): number => {
    let p = cc[cc.length - 1] ?? 0;
    for (let j = cc.length - 2; j >= 0; j--) {
      p = p * x0 + (cc[j] ?? 0);
    }
    return p;
  };

  const sign = (v: number): number => (v === 0 ? 0 : v > 0 ? 1 : -1);

  const an = n;
  if (n === 3) {
    a[1] = Math.SQRT1_2;
  } else {
    const an25 = an + 0.25;
    let summ2 = 0;
    for (let i = 1; i <= nn2; i++) {
      const p = (i - 0.375) / an25;
      // Inverse normal CDF via approximation using studentTCdf? We use a rational approximation:
      // For this package, approximate quantile using inverse error function via binary search on normalCdf.
      let lo = -10;
      let hi = 10;
      for (let it = 0; it < 80; it++) {
        const mid = (lo + hi) / 2;
        const cdf = normalCdf(mid);
        if (cdf < p) lo = mid;
        else hi = mid;
      }
      const z = (lo + hi) / 2;
      a[i] = z;
      summ2 += z * z;
    }
    summ2 *= 2;
    const ssumm2 = Math.sqrt(summ2);
    const rsn = 1 / Math.sqrt(an);

    const a1 = poly(c1, rsn) - (a[1] ?? 0) / ssumm2;
    let i1: number;
    let fac: number;
    if (n > 5) {
      i1 = 3;
      const a2 = -((a[2] ?? 0) / ssumm2) + poly(c2, rsn);
      fac = Math.sqrt(
        (summ2 - 2 * (a[1] ?? 0) * (a[1] ?? 0) - 2 * (a[2] ?? 0) * (a[2] ?? 0)) /
          (1 - 2 * a1 * a1 - 2 * a2 * a2)
      );
      a[2] = a2;
    } else {
      i1 = 2;
      fac = Math.sqrt((summ2 - 2 * (a[1] ?? 0) * (a[1] ?? 0)) / (1 - 2 * a1 * a1));
    }
    a[1] = a1;
    for (let i = i1; i <= nn2; i++) {
      a[i] = -((a[i] ?? 0) / fac);
    }
  }

  // Check sort order and compute scaled sums
  let xx = (x[0] ?? 0) / range;
  let sx = xx;
  let sa = -(a[1] ?? 0);
  for (let i = 1, j = n - 1; i < n; j--) {
    const xi = (x[i] ?? 0) / range;
    if (xx - xi > small) {
      throw new InvalidParameterError("shapiro() data is not sorted", "data", "unsorted");
    }
    sx += xi;
    i++;
    if (i !== j) {
      sa += sign(i - j) * (a[Math.min(i, j)] ?? 0);
    }
    xx = xi;
  }

  sa /= n;
  sx /= n;

  let ssa = 0;
  let ssx = 0;
  let sax = 0;
  for (let i = 0, j = n - 1; i < n; i++, j--) {
    const asa = i !== j ? sign(i - j) * (a[1 + Math.min(i, j)] ?? 0) - sa : -sa;
    const xsx = (x[i] ?? 0) / range - sx;
    ssa += asa * asa;
    ssx += xsx * xsx;
    sax += asa * xsx;
  }

  const ssassx = Math.sqrt(ssa * ssx);
  const w1 = ((ssassx - sax) * (ssassx + sax)) / (ssa * ssx);
  const w = 1 - w1;

  if (n === 3) {
    const pi6 = 1.90985931710274;
    const stqr = 1.0471975511966;
    let pw = pi6 * (Math.asin(Math.sqrt(w)) - stqr);
    if (pw < 0) pw = 0;
    if (pw > 1) pw = 1;
    return { statistic: w, pvalue: pw };
  }

  const y = Math.log(w1);
  const lnN = Math.log(an);
  let m: number;
  let s: number;
  if (n <= 11) {
    const gamma = poly(g, an);
    if (y >= gamma) {
      return { statistic: w, pvalue: 0 };
    }
    const yy = -Math.log(gamma - y);
    m = poly(c3, an);
    s = Math.exp(poly(c4, an));
    const z = (yy - m) / s;
    return { statistic: w, pvalue: normalCdf(z) };
  }
  m = poly(c5, lnN);
  s = Math.exp(poly(c6, lnN));
  const z = (y - m) / s;
  return { statistic: w, pvalue: normalCdf(z) };
}

/**
 * One-sample t-test.
 *
 * Tests whether mean of sample differs from population mean.
 *
 * @see {@link https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html | SciPy stats.ttest_1samp}
 */
export function ttest_1samp(a: Tensor, popmean: number): TestResult {
  const x = toDenseSortedArray1D(a);
  const n = x.length;
  if (n < 2) {
    throw new InvalidParameterError("ttest_1samp() requires at least 2 samples", "n", n);
  }

  // mean
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i] ?? 0;
  mean /= n;

  // sample variance
  let m2 = 0;
  for (let i = 0; i < n; i++) {
    const d = (x[i] ?? 0) - mean;
    m2 += d * d;
  }
  const variance = m2 / (n - 1);
  const std = Math.sqrt(variance);
  if (std === 0) {
    throw new InvalidParameterError("ttest_1samp() is undefined for constant input", "std", std);
  }
  const tstat = (mean - popmean) / (std / Math.sqrt(n));

  const df = n - 1;
  const pvalue = 2 * (1 - studentTCdf(Math.abs(tstat), df));
  return { statistic: tstat, pvalue };
}

/**
 * Independent two-sample t-test.
 *
 * Tests whether means of two independent samples are equal.
 *
 * @see {@link https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html | SciPy stats.ttest_ind}
 */
export function ttest_ind(a: Tensor, b: Tensor, equalVar = true): TestResult {
  const xa = toDenseSortedArray1D(a);
  const xb = toDenseSortedArray1D(b);
  const na = xa.length;
  const nb = xb.length;
  if (na < 2 || nb < 2) {
    throw new InvalidParameterError("ttest_ind() requires at least 2 samples in each group", "n", {
      na,
      nb,
    });
  }

  let meanA = 0;
  let meanB = 0;
  for (let i = 0; i < na; i++) meanA += xa[i] ?? 0;
  for (let i = 0; i < nb; i++) meanB += xb[i] ?? 0;
  meanA /= na;
  meanB /= nb;

  let ssa = 0;
  let ssb = 0;
  for (let i = 0; i < na; i++) {
    const d = (xa[i] ?? 0) - meanA;
    ssa += d * d;
  }
  for (let i = 0; i < nb; i++) {
    const d = (xb[i] ?? 0) - meanB;
    ssb += d * d;
  }
  const varA = ssa / (na - 1);
  const varB = ssb / (nb - 1);

  let tstat: number;
  let df: number;

  if (equalVar) {
    const pooledVar = ((na - 1) * varA + (nb - 1) * varB) / (na + nb - 2);
    const denom = Math.sqrt(pooledVar * (1 / na + 1 / nb));
    if (denom === 0)
      throw new InvalidParameterError(
        "ttest_ind() is undefined for constant input",
        "denom",
        denom
      );
    tstat = (meanA - meanB) / denom;
    df = na + nb - 2;
  } else {
    const denom = Math.sqrt(varA / na + varB / nb);
    if (denom === 0)
      throw new InvalidParameterError(
        "ttest_ind() is undefined for constant input",
        "denom",
        denom
      );
    tstat = (meanA - meanB) / denom;
    df = (varA / na + varB / nb) ** 2 / ((varA / na) ** 2 / (na - 1) + (varB / nb) ** 2 / (nb - 1));
  }

  const pvalue = 2 * (1 - studentTCdf(Math.abs(tstat), df));
  return { statistic: tstat, pvalue };
}

/**
 * Paired-sample t-test.
 *
 * Tests whether means of two related samples are equal.
 *
 * @see {@link https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html | SciPy stats.ttest_rel}
 */
export function ttest_rel(a: Tensor, b: Tensor): TestResult {
  if (a.size !== b.size) {
    throw new InvalidParameterError("ttest_rel() requires paired samples of equal length", "size", {
      a: a.size,
      b: b.size,
    });
  }
  const n = a.size;
  if (n < 2) {
    throw new InvalidParameterError("ttest_rel() requires at least 2 paired samples", "n", n);
  }

  // Differences
  const diffs = new Float64Array(n);
  let i = 0;
  forEachIndexOffset(a, (offA) => {
    // Map to corresponding flat element in b using the same iteration order.
    // We rely on size equality and iterate both tensors to dense arrays.
    diffs[i] = getNumberAt(a, offA);
    i++;
  });
  const bd = new Float64Array(n);
  i = 0;
  forEachIndexOffset(b, (offB) => {
    bd[i] = getNumberAt(b, offB);
    i++;
  });
  for (let k = 0; k < n; k++) {
    diffs[k] = (diffs[k] ?? 0) - (bd[k] ?? 0);
  }

  let mean = 0;
  for (let k = 0; k < n; k++) mean += diffs[k] ?? 0;
  mean /= n;

  let ss = 0;
  for (let k = 0; k < n; k++) {
    const d = (diffs[k] ?? 0) - mean;
    ss += d * d;
  }
  const varDiff = ss / (n - 1);
  const stdDiff = Math.sqrt(varDiff);
  if (stdDiff === 0) {
    throw new InvalidParameterError(
      "ttest_rel() is undefined for constant differences",
      "stdDiff",
      stdDiff
    );
  }
  const tstat = mean / (stdDiff / Math.sqrt(n));
  const df = n - 1;
  const pvalue = 2 * (1 - studentTCdf(Math.abs(tstat), df));
  return { statistic: tstat, pvalue };
}

/**
 * Chi-square goodness of fit test.
 *
 * Observed and expected frequencies must be non-negative and sum to the same total
 * (within floating-point tolerance).
 *
 * @see {@link https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html | SciPy stats.chisquare}
 */
export function chisquare(f_obs: Tensor, f_exp?: Tensor): TestResult {
  const obs = toDenseArray1D(f_obs);
  const n = obs.length;
  if (n < 1) {
    throw new InvalidParameterError("chisquare() requires at least one observed value", "n", n);
  }
  let chiSq = 0;
  let sumObs = 0;

  for (let i = 0; i < n; i++) {
    const v = obs[i] ?? 0;
    if (!Number.isFinite(v) || v < 0) {
      throw new InvalidParameterError(
        "chisquare() observed frequencies must be finite and >= 0",
        "f_obs",
        v
      );
    }
    sumObs += v;
  }

  if (f_exp && f_obs.size !== f_exp.size) {
    throw new InvalidParameterError(
      "Observed and expected frequency arrays must have the same length",
      "size",
      { f_obs: f_obs.size, f_exp: f_exp.size }
    );
  }

  if (!f_exp) {
    // Uniform expected frequencies
    const expected = sumObs / n;
    if (!Number.isFinite(expected) || expected <= 0) {
      throw new InvalidParameterError(
        "chisquare() expected frequencies must be finite and > 0",
        "expected",
        expected
      );
    }
    for (let i = 0; i < n; i++) {
      const v = obs[i] ?? 0;
      chiSq += (v - expected) ** 2 / expected;
    }
  } else {
    const exp = toDenseArray1D(f_exp);
    let sumExp = 0;
    for (let i = 0; i < n; i++) {
      const v = exp[i] ?? 0;
      if (!Number.isFinite(v) || v <= 0) {
        throw new InvalidParameterError(
          "chisquare() expected frequencies must be finite and > 0",
          "f_exp",
          v
        );
      }
      sumExp += v;
    }

    const rtol = Math.sqrt(Number.EPSILON);
    const denom = Math.max(Math.abs(sumObs), Math.abs(sumExp));
    if (Math.abs(sumObs - sumExp) > rtol * denom) {
      throw new InvalidParameterError(
        "chisquare() expected and observed frequencies must sum to the same value",
        "sum",
        { f_obs: sumObs, f_exp: sumExp }
      );
    }

    for (let i = 0; i < n; i++) {
      const vObs = obs[i] ?? 0;
      const vExp = exp[i] ?? 0;
      chiSq += (vObs - vExp) ** 2 / vExp;
    }
  }

  const df = n - 1;
  if (df < 1) {
    throw new InvalidParameterError(
      "chisquare() requires at least 2 categories (df must be >= 1)",
      "df",
      df
    );
  }
  const pvalue = 1 - chiSquareCdf(chiSq, df);

  return { statistic: chiSq, pvalue };
}

/**
 * Kolmogorov-Smirnov test for goodness of fit.
 *
 * @see {@link https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html | SciPy stats.kstest}
 */
export function kstest(data: Tensor, cdf: string | ((x: number) => number)): TestResult {
  const x = toDenseSortedArray1D(data);
  const n = x.length;
  if (n === 0) {
    throw new InvalidParameterError("kstest() requires at least one element", "n", n);
  }

  if (typeof cdf === "string" && cdf !== "norm") {
    throw new InvalidParameterError(
      `Unsupported distribution: '${cdf}'. Supported distributions: 'norm'`,
      "cdf",
      cdf
    );
  }

  const F = typeof cdf === "string" ? (v: number) => normalCdf(v) : cdf;

  let d = 0;
  for (let i = 0; i < n; i++) {
    const xi = x[i] ?? 0;
    const fi = F(xi);
    const dPlus = (i + 1) / n - fi;
    const dMinus = fi - i / n;
    d = Math.max(d, dPlus, dMinus);
  }

  // Asymptotic p-value (two-sided) using Kolmogorov distribution
  // p ≈ 2 * sum_{k=1..inf} (-1)^{k-1} exp(-2 k^2 d^2 n)
  let p = 0;
  for (let k = 1; k < 200; k++) {
    const term = Math.exp(-2 * k * k * d * d * n);
    p += (k % 2 === 1 ? 1 : -1) * term;
    if (term < 1e-12) break;
  }
  p = Math.max(0, Math.min(1, 2 * p));

  return { statistic: d, pvalue: p };
}

/**
 * Test for normality.
 *
 * Uses D'Agostino-Pearson's omnibus test combining skewness and kurtosis.
 *
 * @see {@link https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html | SciPy stats.normaltest}
 */
export function normaltest(a: Tensor): TestResult {
  const x = toDenseSortedArray1D(a);
  const n = x.length;
  if (n < 8) {
    throw new InvalidParameterError("normaltest() requires at least 8 samples", "n", n);
  }
  const { mean, m2 } = meanAndM2(x);
  const m2n = m2 / n;
  const std = Math.sqrt(m2n);
  if (!Number.isFinite(std) || std === 0) {
    throw new InvalidParameterError("normaltest() is undefined for constant input", "std", std);
  }

  let m3 = 0;
  let m4 = 0;
  for (let i = 0; i < n; i++) {
    const d = (x[i] ?? 0) - mean;
    m3 += d * d * d;
    m4 += d * d * d * d;
  }
  const skew = m3 / n / m2n ** 1.5;
  const kurt = m4 / n / (m2n * m2n);

  // Skewness component (D'Agostino)
  const y = skew * Math.sqrt(((n + 1) * (n + 3)) / (6 * (n - 2)));
  const beta2 =
    (3 * (n * n + 27 * n - 70) * (n + 1) * (n + 3)) / ((n - 2) * (n + 5) * (n + 7) * (n + 9));
  const w2 = -1 + Math.sqrt(2 * (beta2 - 1));
  const delta = 1 / Math.sqrt(0.5 * Math.log(w2));
  const alpha = Math.sqrt(2 / (w2 - 1));
  const yScaled = y / alpha;
  const z1 = delta * Math.log(yScaled + Math.sqrt(yScaled * yScaled + 1));

  // Kurtosis component (Anscombe-Glynn)
  const e = (3 * (n - 1)) / (n + 1);
  const varb2 = (24 * n * (n - 2) * (n - 3)) / ((n + 1) * (n + 1) * (n + 3) * (n + 5));
  const xval = (kurt - e) / Math.sqrt(varb2);
  const sqrtbeta1 =
    ((6 * (n * n - 5 * n + 2)) / ((n + 7) * (n + 9))) *
    Math.sqrt((6 * (n + 3) * (n + 5)) / (n * (n - 2) * (n - 3)));
  const aTerm = 6 + (8 / sqrtbeta1) * (2 / sqrtbeta1 + Math.sqrt(1 + 4 / (sqrtbeta1 * sqrtbeta1)));
  const term1 = 1 - 2 / (9 * aTerm);
  const denom = 1 + xval * Math.sqrt(2 / (aTerm - 4));
  const term2 =
    denom === 0 ? Number.NaN : Math.sign(denom) * ((1 - 2 / aTerm) / Math.abs(denom)) ** (1 / 3);
  const z2 = (term1 - term2) / Math.sqrt(2 / (9 * aTerm));

  const k2 = z1 * z1 + z2 * z2;
  const pvalue = 1 - chiSquareCdf(k2, 2);
  return { statistic: k2, pvalue };
}

/**
 * Shapiro-Wilk test for normality.
 *
 * @see {@link https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html | SciPy stats.shapiro}
 */
export function shapiro(x: Tensor): TestResult {
  const sorted = toDenseSortedArray1D(x);
  return shapiroWilk(sorted);
}

/**
 * Anderson-Darling test for normality.
 *
 * Uses sample standard deviation and size-adjusted critical values.
 * For very small samples (n < 10), uses an IQR-based scale estimate to stabilize
 * the statistic.
 *
 * @see {@link https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson.html | SciPy stats.anderson}
 */
export function anderson(x: Tensor): {
  statistic: number;
  critical_values: number[];
  significance_level: number[];
} {
  const sorted = toDenseSortedArray1D(x);
  const n = sorted.length;
  if (n < 1) {
    throw new InvalidParameterError("anderson() requires at least one element", "n", n);
  }
  const { mean, m2 } = meanAndM2(sorted);
  const variance = n > 1 ? m2 / (n - 1) : NaN;
  let std = Math.sqrt(variance);
  if (!Number.isFinite(std) || std === 0) {
    throw new InvalidParameterError("anderson() is undefined for constant input", "std", std);
  }
  if (n < 10) {
    const quantile = (q: number): number => {
      if (n === 1) return sorted[0] ?? 0;
      const pos = (n - 1) * q;
      const lo = Math.floor(pos);
      const hi = Math.ceil(pos);
      const v0 = sorted[lo] ?? 0;
      const v1 = sorted[hi] ?? v0;
      return v0 + (pos - lo) * (v1 - v0);
    };
    const q1 = quantile(0.25);
    const q3 = quantile(0.75);
    const iqr = q3 - q1;
    const robust = iqr / 1.349;
    if (Number.isFinite(robust) && robust > 0) {
      std = robust;
    }
  }

  // Anderson-Darling for normal: A^2 = -n - (1/n) sum_{i=1..n} (2i-1)[ln Phi(z_i) + ln(1-Phi(z_{n+1-i}))]
  let A2 = 0;
  for (let i = 0; i < n; i++) {
    const zi = ((sorted[i] ?? 0) - mean) / std;
    const zj = ((sorted[n - 1 - i] ?? 0) - mean) / std;
    const PhiI = Math.max(1e-300, Math.min(1 - 1e-16, normalCdf(zi)));
    const PhiJ = Math.max(1e-300, Math.min(1 - 1e-16, normalCdf(zj)));
    A2 += (2 * (i + 1) - 1) * (Math.log(PhiI) + Math.log(1 - PhiJ));
  }
  A2 = -n - A2 / n;

  const baseCritical = [0.576, 0.656, 0.787, 0.918, 1.092];
  const factor = 1 + 4 / n - 25 / (n * n);
  const critical_values = baseCritical.map((v) => Math.round((v / factor) * 1000) / 1000);

  return {
    statistic: A2,
    critical_values,
    significance_level: [0.15, 0.1, 0.05, 0.025, 0.01],
  };
}

/**
 * Mann-Whitney U test (non-parametric).
 *
 * Tests whether two independent samples come from same distribution.
 *
 * Note: Uses normal approximation for the p-value with tie correction and
 * continuity correction. No exact method selection is available.
 *
 * @see {@link https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html | SciPy stats.mannwhitneyu}
 */
export function mannwhitneyu(x: Tensor, y: Tensor): TestResult {
  const nx = x.size;
  const ny = y.size;

  if (nx < 1 || ny < 1) {
    throw new InvalidParameterError("Both samples must be non-empty", "size", {
      x: nx,
      y: ny,
    });
  }

  const xVals = toDenseArray1D(x);
  const yVals = toDenseArray1D(y);
  const n = nx + ny;
  const combined = new Float64Array(n);
  combined.set(xVals, 0);
  combined.set(yVals, nx);

  const { ranks, tieSum } = rankData(combined);

  // Rank sum for group 1 (first sample)
  let R1 = 0;
  for (let i = 0; i < nx; i++) {
    R1 += ranks[i] ?? 0;
  }

  const U1 = R1 - (nx * (nx + 1)) / 2;
  const U2 = nx * ny - U1;
  const U = Math.min(U1, U2);

  const meanU = (nx * ny) / 2;
  // Tie correction uses sum(t^3 - t) over tied groups.
  const tieAdj = n > 1 ? tieSum / (n * (n - 1)) : 0;
  const varU = (nx * ny * (n + 1 - tieAdj)) / 12;
  if (varU <= 0) {
    return { statistic: U, pvalue: NaN };
  }

  const stdU = Math.sqrt(varU);
  // Continuity correction for two-sided normal approximation.
  // For small samples, omit correction to better match exact distribution behavior.
  const useContinuity = nx + ny > 20;
  const continuity = useContinuity ? (U < meanU ? 0.5 : U > meanU ? -0.5 : 0) : 0;
  const z = (U - meanU + continuity) / stdU;
  const pvalue = 2 * (1 - normalCdf(Math.abs(z)));

  return { statistic: U, pvalue };
}

/**
 * Wilcoxon signed-rank test (non-parametric paired test).
 *
 * Note: Uses normal approximation for the p-value with tie correction and
 * continuity correction. No exact method selection is available.
 *
 * @see {@link https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html | SciPy stats.wilcoxon}
 */
export function wilcoxon(x: Tensor, y?: Tensor): TestResult {
  const n = x.size;
  const diffs: number[] = [];

  if (y) {
    if (x.size !== y.size) {
      throw new InvalidParameterError("Paired samples must have equal length", "size", {
        x: x.size,
        y: y.size,
      });
    }
    const xd = toDenseArray1D(x);
    const yd = toDenseArray1D(y);
    for (let i = 0; i < n; i++) {
      const diff = (xd[i] ?? 0) - (yd[i] ?? 0);
      if (diff !== 0) diffs.push(diff);
    }
  } else {
    const xd = toDenseArray1D(x);
    for (let i = 0; i < n; i++) {
      const val = xd[i] ?? 0;
      if (val !== 0) diffs.push(val);
    }
  }

  if (diffs.length === 0) {
    throw new InvalidParameterError(
      "wilcoxon() is undefined when all differences are zero",
      "diffs",
      diffs.length
    );
  }

  const absDiffs = new Float64Array(diffs.length);
  for (let i = 0; i < diffs.length; i++) {
    absDiffs[i] = Math.abs(diffs[i] ?? 0);
  }

  const { ranks, tieSum } = rankData(absDiffs);

  let Wplus = 0;
  for (let i = 0; i < diffs.length; i++) {
    if ((diffs[i] ?? 0) > 0) {
      Wplus += ranks[i] ?? 0;
    }
  }

  const nEff = diffs.length;
  const meanW = (nEff * (nEff + 1)) / 4;
  // Tie-corrected variance for signed-rank statistic.
  const varW = (nEff * (nEff + 1) * (2 * nEff + 1)) / 24 - tieSum / 48;
  if (varW <= 0) {
    return { statistic: Wplus, pvalue: NaN };
  }

  const stdW = Math.sqrt(varW);
  // Continuity correction for two-sided normal approximation.
  // For small samples, omit correction to better match exact distribution behavior.
  const useContinuity = nEff > 20;
  const continuity = useContinuity ? (Wplus < meanW ? 0.5 : Wplus > meanW ? -0.5 : 0) : 0;
  const z = (Wplus - meanW + continuity) / stdW;
  const pvalue = 2 * (1 - normalCdf(Math.abs(z)));

  return { statistic: Wplus, pvalue };
}

/**
 * Kruskal-Wallis H-test (non-parametric version of ANOVA).
 *
 * Note: Uses chi-square approximation for the p-value with tie correction.
 *
 * @see {@link https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html | SciPy stats.kruskal}
 */
export function kruskal(...samples: Tensor[]): TestResult {
  const k = samples.length;
  if (k < 2) {
    throw new InvalidParameterError("kruskal() requires at least 2 groups", "k", k);
  }

  let N = 0;
  const sizes = new Array<number | undefined>(k);
  const validatedSamples = new Array<Tensor | undefined>(k);
  for (let g = 0; g < k; g++) {
    const sample = samples[g];
    if (!sample || sample.size < 1) {
      throw new InvalidParameterError("kruskal() requires non-empty samples", "size", {
        group: g,
        size: sample?.size ?? 0,
      });
    }
    validatedSamples[g] = sample;
    sizes[g] = sample.size;
    N += sample.size;
  }

  const combined = new Float64Array(N);
  const groupIndex = new Int32Array(N);
  let idx = 0;
  for (let g = 0; g < k; g++) {
    const sample = validatedSamples[g];
    if (!sample) {
      throw new InvalidParameterError("kruskal() requires non-empty samples", "sample", g);
    }
    const vals = toDenseArray1D(sample);
    for (let i = 0; i < vals.length; i++) {
      combined[idx] = vals[i] ?? 0;
      groupIndex[idx] = g;
      idx++;
    }
  }

  const { ranks, tieSum } = rankData(combined);

  const rankSums = new Float64Array(k);
  for (let i = 0; i < N; i++) {
    const group = groupIndex[i] ?? 0;
    const rank = ranks[i] ?? 0;
    rankSums[group] = (rankSums[group] ?? 0) + rank;
  }

  let H = 0;
  for (let g = 0; g < k; g++) {
    const rs = rankSums[g] ?? 0;
    const sz = sizes[g] ?? 1;
    H += (rs * rs) / sz;
  }
  H = (12 / (N * (N + 1))) * H - 3 * (N + 1);

  // Tie correction factor for H statistic.
  const tieCorrection = N > 1 ? 1 - tieSum / (N * N * N - N) : 1;
  if (tieCorrection <= 0) {
    throw new InvalidParameterError(
      "kruskal() is undefined when all numbers are identical",
      "tieCorrection",
      tieCorrection
    );
  }
  H /= tieCorrection;

  const df = k - 1;
  const pvalue = 1 - chiSquareCdf(H, df);

  return { statistic: H, pvalue };
}

/**
 * Friedman test (non-parametric repeated measures ANOVA).
 *
 * Note: Uses chi-square approximation for the p-value with tie correction.
 *
 * @see {@link https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html | SciPy stats.friedmanchisquare}
 */
export function friedmanchisquare(...samples: Tensor[]): TestResult {
  const k = samples.length;
  if (k < 3) {
    throw new InvalidParameterError(
      "friedmanchisquare() requires at least 3 related samples",
      "k",
      k
    );
  }
  const n = samples[0]?.size ?? 0;
  if (n < 1) {
    throw new InvalidParameterError(
      "friedmanchisquare() requires all samples to be non-empty",
      "n",
      n
    );
  }

  for (let i = 1; i < k; i++) {
    const sample = samples[i];
    if (sample && sample.size !== n) {
      throw new InvalidParameterError(
        "All samples must have the same length for Friedman test",
        "size",
        { expected: n, got: sample.size, sampleIndex: i }
      );
    }
  }

  // Rank within each block (row)
  const denseSamples = samples.map((sample) =>
    sample ? toDenseArray1D(sample) : new Float64Array(0)
  );
  const rankSums = new Float64Array(k);
  let tieSum = 0;

  for (let i = 0; i < n; i++) {
    const block = new Float64Array(k);
    for (let j = 0; j < k; j++) {
      const arr = denseSamples[j];
      block[j] = arr?.[i] ?? 0;
    }
    const ranked = rankData(block);
    tieSum += ranked.tieSum;
    for (let j = 0; j < k; j++) {
      const rank = ranked.ranks[j] ?? 0;
      rankSums[j] = (rankSums[j] ?? 0) + rank;
    }
  }

  let chiSq = 0;
  for (let j = 0; j < k; j++) {
    const rs = rankSums[j] ?? 0;
    chiSq += rs * rs;
  }
  chiSq = (12 / (n * k * (k + 1))) * chiSq - 3 * n * (k + 1);

  // Tie correction factor for Friedman chi-square.
  const tieCorrection = n > 0 ? 1 - tieSum / (n * k * (k * k - 1)) : 1;
  if (tieCorrection <= 0) {
    throw new InvalidParameterError(
      "friedmanchisquare() is undefined when all numbers are identical within blocks",
      "tieCorrection",
      tieCorrection
    );
  }
  chiSq /= tieCorrection;

  const df = k - 1;
  const pvalue = 1 - chiSquareCdf(chiSq, df);

  return { statistic: chiSq, pvalue };
}

/**
 * Levene's test for equality of variances.
 *
 * Tests whether two or more groups have equal variances.
 * More robust than Bartlett's test for non-normal data.
 *
 * @param center - Method to use for centering: 'median' (default, most robust),
 *                 'mean' (traditional), or 'trimmed' (10% trimmed mean)
 * @param samples - Two or more sample tensors to compare
 * @returns Test result with statistic and p-value
 *
 * @example
 * ```ts
 * import { levene, tensor } from 'deepbox';
 *
 * const group1 = tensor([1, 2, 3, 4, 5]);
 * const group2 = tensor([2, 4, 6, 8, 10]);
 * const result = levene('median', group1, group2);
 * console.log(result.pvalue);  // p-value for equal variances
 * ```
 *
 * @see {@link https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levene.html | SciPy stats.levene}
 */
export function levene(center: "mean" | "median" | "trimmed", ...samples: Tensor[]): TestResult {
  const k = samples.length;
  if (k < 2) {
    throw new InvalidParameterError("levene() requires at least 2 groups", "k", k);
  }

  // Convert samples to arrays and compute centers
  const groups: Float64Array[] = [];
  const centers: number[] = [];

  for (let g = 0; g < k; g++) {
    const sample = samples[g];
    if (!sample || sample.size === 0) {
      throw new InvalidParameterError("levene() requires all groups to be non-empty", "groupSize", {
        group: g,
        size: sample?.size ?? 0,
      });
    }
    const arr = toDenseSortedArray1D(sample);
    if (arr.length < 2) {
      throw new InvalidParameterError(
        "levene() requires at least 2 samples per group",
        "groupSize",
        arr.length
      );
    }
    groups.push(arr);

    // Compute center based on method
    if (center === "mean") {
      let sum = 0;
      for (let i = 0; i < arr.length; i++) sum += arr[i] ?? 0;
      centers.push(sum / arr.length);
    } else if (center === "median") {
      const mid = Math.floor(arr.length / 2);
      if (arr.length % 2 === 0) {
        centers.push(((arr[mid - 1] ?? 0) + (arr[mid] ?? 0)) / 2);
      } else {
        centers.push(arr[mid] ?? 0);
      }
    } else {
      // Trimmed mean (10%)
      const trimCount = Math.floor(arr.length * 0.1);
      let sum = 0;
      const n = arr.length - 2 * trimCount;
      for (let i = trimCount; i < arr.length - trimCount; i++) {
        sum += arr[i] ?? 0;
      }
      centers.push(sum / n);
    }
  }

  // Compute absolute deviations from center (Z_ij = |Y_ij - center_i|)
  const Z: Float64Array[] = [];
  const groupMeansZ: number[] = [];
  let N = 0;
  let grandSumZ = 0;

  for (let g = 0; g < groups.length; g++) {
    const arr = groups[g];
    if (!arr) continue;
    const c = centers[g] ?? 0;
    const zArr = new Float64Array(arr.length);
    let sumZ = 0;

    for (let i = 0; i < arr.length; i++) {
      const absVal = Math.abs((arr[i] ?? 0) - c);
      zArr[i] = absVal;
      sumZ += absVal;
    }

    Z.push(zArr);
    groupMeansZ.push(sumZ / arr.length);
    N += arr.length;
    grandSumZ += sumZ;
  }

  const grandMeanZ = grandSumZ / N;

  // Compute Levene's W statistic (F-test on absolute deviations)
  let SSB = 0; // Between-group sum of squares
  let SSW = 0; // Within-group sum of squares

  for (let g = 0; g < Z.length; g++) {
    const zArr = Z[g];
    if (!zArr) continue;
    const n = zArr.length;
    SSB += n * ((groupMeansZ[g] ?? 0) - grandMeanZ) ** 2;

    for (let i = 0; i < n; i++) {
      SSW += ((zArr[i] ?? 0) - (groupMeansZ[g] ?? 0)) ** 2;
    }
  }

  const dfB = k - 1;
  const dfW = N - k;
  if (dfW <= 0) {
    throw new InvalidParameterError(
      "levene() requires more total observations than groups",
      "dfW",
      dfW
    );
  }
  if (SSW === 0) {
    return { statistic: Infinity, pvalue: 0 };
  }
  const W = SSB / dfB / (SSW / dfW);

  const pvalue = 1 - fCdf(W, dfB, dfW);

  return { statistic: W, pvalue };
}

/**
 * Bartlett's test for equality of variances.
 *
 * Tests whether two or more groups have equal variances.
 * Assumes data is normally distributed; use Levene's test for non-normal data.
 *
 * @param samples - Two or more sample tensors to compare
 * @returns Test result with statistic and p-value
 *
 * @example
 * ```ts
 * import { bartlett, tensor } from 'deepbox';
 *
 * const group1 = tensor([1, 2, 3, 4, 5]);
 * const group2 = tensor([2, 4, 6, 8, 10]);
 * const result = bartlett(group1, group2);
 * console.log(result.pvalue);  // p-value for equal variances
 * ```
 *
 * @see {@link https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bartlett.html | SciPy stats.bartlett}
 */
export function bartlett(...samples: Tensor[]): TestResult {
  const k = samples.length;
  if (k < 2) {
    throw new InvalidParameterError("bartlett() requires at least 2 groups", "k", k);
  }

  // Convert samples to arrays and compute variances
  const variances: number[] = [];
  const sizes: number[] = [];
  let N = 0;

  for (let g = 0; g < k; g++) {
    const sample = samples[g];
    if (!sample || sample.size === 0) {
      throw new InvalidParameterError(
        "bartlett() requires all groups to be non-empty",
        "groupSize",
        { group: g, size: sample?.size ?? 0 }
      );
    }
    const arr = toDenseSortedArray1D(sample);
    const n = arr.length;
    if (n < 2) {
      throw new InvalidParameterError(
        "bartlett() requires at least 2 samples per group",
        "groupSize",
        n
      );
    }

    // Compute mean
    let mean = 0;
    for (let i = 0; i < n; i++) mean += arr[i] ?? 0;
    mean /= n;

    // Compute sample variance (using n-1 denominator)
    let ss = 0;
    for (let i = 0; i < n; i++) {
      const d = (arr[i] ?? 0) - mean;
      ss += d * d;
    }
    const variance = ss / (n - 1);

    variances.push(variance);
    sizes.push(n);
    N += n;
  }

  // Check for zero variances
  for (let g = 0; g < k; g++) {
    if ((variances[g] ?? 0) === 0) {
      throw new InvalidParameterError(
        "bartlett() is undefined when a group has zero variance",
        "variance",
        variances[g]
      );
    }
  }

  // Compute pooled variance
  let pooledNumerator = 0;
  for (let g = 0; g < k; g++) {
    pooledNumerator += ((sizes[g] ?? 1) - 1) * (variances[g] ?? 1);
  }
  const pooledVariance = pooledNumerator / (N - k);

  // Compute Bartlett's statistic
  // T = (N-k) * ln(s_p^2) - sum((n_i - 1) * ln(s_i^2))
  let sumLogVar = 0;
  for (let g = 0; g < k; g++) {
    sumLogVar += ((sizes[g] ?? 1) - 1) * Math.log(variances[g] ?? 1);
  }
  const T = (N - k) * Math.log(pooledVariance) - sumLogVar;

  // Correction factor C
  let sumInvDf = 0;
  for (let g = 0; g < k; g++) {
    sumInvDf += 1 / ((sizes[g] ?? 1) - 1);
  }
  const C = 1 + (1 / (3 * (k - 1))) * (sumInvDf - 1 / (N - k));

  // Chi-square statistic
  const chiSq = T / C;

  const df = k - 1;
  const pvalue = 1 - chiSquareCdf(chiSq, df);

  return { statistic: chiSq, pvalue };
}

/**
 * One-way ANOVA.
 *
 * @see {@link https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html | SciPy stats.f_oneway}
 */
export function f_oneway(...samples: Tensor[]): TestResult {
  const k = samples.length;
  if (k < 2) {
    throw new InvalidParameterError("f_oneway() requires at least 2 groups", "groups", k);
  }

  for (let g = 0; g < k; g++) {
    const sample = samples[g];
    if (!sample || sample.size === 0) {
      throw new InvalidParameterError(
        "f_oneway() requires all groups to be non-empty",
        "groupSize",
        { group: g, size: sample?.size ?? 0 }
      );
    }
  }

  let N = 0;
  const means: number[] = [];
  const sizes: number[] = [];
  const groups: Float64Array[] = [];

  // Compute group means
  for (let g = 0; g < k; g++) {
    const sample = samples[g];
    if (!sample) {
      throw new InvalidParameterError(
        "f_oneway() requires all groups to be non-empty",
        "groupSize",
        { group: g, size: 0 }
      );
    }
    const arr = toDenseArray1D(sample);
    const n = arr.length;
    groups.push(arr);
    sizes.push(n);
    N += n;

    let sum = 0;
    for (let i = 0; i < n; i++) {
      sum += arr[i] ?? 0;
    }
    means.push(sum / n);
  }

  // Compute grand mean
  let grandSum = 0;
  for (let g = 0; g < k; g++) {
    grandSum += (means[g] ?? 0) * (sizes[g] ?? 0);
  }
  const grandMean = grandSum / N;

  // Compute between-group and within-group variance
  let SSB = 0; // Between-group sum of squares
  let SSW = 0; // Within-group sum of squares

  for (let g = 0; g < groups.length; g++) {
    const arr = groups[g];
    if (!arr) continue;
    const n = arr.length;
    SSB += n * ((means[g] ?? 0) - grandMean) ** 2;

    for (let i = 0; i < n; i++) {
      SSW += ((arr[i] ?? 0) - (means[g] ?? 0)) ** 2;
    }
  }

  const dfB = k - 1;
  const dfW = N - k;
  if (dfW <= 0) {
    throw new InvalidParameterError(
      "f_oneway() requires at least one group with more than one sample",
      "dfW",
      dfW
    );
  }
  const MSB = SSB / dfB;
  const MSW = SSW / dfW;
  if (MSW === 0) {
    // All within-group values are identical; F is infinite if groups differ, NaN otherwise
    const F = MSB === 0 ? NaN : Infinity;
    return { statistic: F, pvalue: MSB === 0 ? NaN : 0 };
  }
  const F = MSB / MSW;

  const pvalue = 1 - fCdf(F, dfB, dfW);

  return { statistic: F, pvalue };
}
