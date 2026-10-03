/**
 * Probability distribution objects.
 *
 * Every distribution provides `pdf` (continuous) or `pmf` (discrete), `cdf`, `ppf`, `sf`, `isf`,
 * `rvs`, `mean`, `variance`, `std` and `median`; continuous distributions also provide
 * `entropy`. `rvs` draws from the
 * shared random generator of `deepbox/random`, so `setSeed` makes samples reproducible.
 *
 * Parameterizations follow the textbook form named on each factory (rate for `expon` and
 * `gamma`, `mu`/`sigma` of the underlying normal for `lognorm`), which differs from SciPy's
 * `scale`/`loc` arguments in a few places; the factory docs spell out the mapping.
 *
 * @module stats/distributions
 * @see {@link https://deepbox.dev/docs/stats-distributions | Deepbox documentation}
 */

import { InvalidParameterError } from "../core";
import { __random } from "../random/random";
import {
  chiSquareCdf,
  chiSquareSf,
  digamma,
  fCdf,
  fSf,
  logBeta,
  logGamma,
  normalCdf,
  normalPpf,
  normalSf,
  regularizedIncompleteBeta,
  studentTCdf,
  studentTSf,
} from "./_internal";

// ---- Helpers ----

const SQRT_2PI = Math.sqrt(2 * Math.PI);

const LN_SQRT_2PI = 0.9189385332046728;
const LN_2PI = 2 * LN_SQRT_2PI;
const MIN_NORMAL = 2.2250738585072014e-308;

/** `stirlerr(k / 2)` for k = 0..30, from high-precision evaluation of the definition below. */
const STIRLERR_HALVES: readonly number[] = [
  0.0, 0.15342640972002736, 0.08106146679532726, 0.05481412105191765, 0.0413406959554093,
  0.03316287351993629, 0.02767792568499834, 0.023746163656297496, 0.020790672103765093,
  0.018488450532673187, 0.016644691189821193, 0.015134973221917378, 0.013876128823070748,
  0.012810465242920227, 0.01189670994589177, 0.011104559758206917, 0.010411265261972096,
  0.009799416126158804, 0.009255462182712733, 0.008768700134139386, 0.00833056343336287,
  0.00793411456431402, 0.007573675487951841, 0.007244554301320383, 0.00694284010720953,
  0.006665247032707682, 0.006408994188004207, 0.006171712263039458, 0.0059513701127588475,
  0.0057462165130101155, 0.005554733551962801,
];

/**
 * Stirling-series remainder `ln(n!) - ln(sqrt(2 pi n) (n / e)^n)`, the building block of
 * the saddle-point (Loader) evaluation of binomial and Poisson probabilities. It stays
 * accurate to about 1e-16 where a difference of two log-gamma values would lose
 * `log10(n)` digits.
 */
function stirlerr(n: number): number {
  if (n <= 15) {
    const twice = 2 * n;
    if (Number.isInteger(twice)) return STIRLERR_HALVES[twice] as number;
    return logGamma(n + 1) - (n + 0.5) * Math.log(n) + n - LN_SQRT_2PI;
  }
  const nn = n * n;
  const s0 = 1 / 12;
  const s1 = 1 / 360;
  const s2 = 1 / 1260;
  const s3 = 1 / 1680;
  const s4 = 1 / 1188;
  if (n > 500) return (s0 - s1 / nn) / n;
  if (n > 80) return (s0 - (s1 - s2 / nn) / nn) / n;
  if (n > 35) return (s0 - (s1 - (s2 - s3 / nn) / nn) / nn) / n;
  return (s0 - (s1 - (s2 - (s3 - s4 / nn) / nn) / nn) / nn) / n;
}

/** Deviance term `x ln(x / np) + np - x`, evaluated without cancellation when `x` is near `np`. */
function bd0(x: number, np: number): number {
  if (Math.abs(x - np) < 0.1 * (x + np)) {
    let v = (x - np) / (x + np);
    let sum = (x - np) * v;
    if (Math.abs(sum) < MIN_NORMAL) return sum;
    let ej = 2 * x * v;
    v *= v;
    for (let j = 1; j < 1000; j++) {
      ej *= v;
      const next = sum + ej / (2 * j + 1);
      if (next === sum) return next;
      sum = next;
    }
  }
  const ratio = x / np;
  // x / np can overflow or underflow for extreme arguments; the difference of logs is the fallback.
  const logRatio =
    Number.isFinite(ratio) && ratio > 0 ? Math.log(ratio) : Math.log(x) - Math.log(np);
  return x * logRatio + np - x;
}

/** Natural log of the Poisson probability `lambda^x e^-lambda / x!` for real `x >= 0`. */
function logPoissonRaw(x: number, lambda: number): number {
  if (lambda === 0) return x === 0 ? 0 : Number.NEGATIVE_INFINITY;
  if (x < 0) return Number.NEGATIVE_INFINITY;
  if (x <= lambda * MIN_NORMAL) return -lambda;
  if (lambda < x * MIN_NORMAL) return -lambda + x * Math.log(lambda) - logGamma(x + 1);
  return -stirlerr(x) - bd0(x, lambda) - LN_SQRT_2PI - 0.5 * Math.log(x);
}

/** Natural log of the binomial probability `C(n, x) p^x q^(n-x)` (`q = 1 - p`) for real `x`, `n`. */
function logBinomialRaw(x: number, n: number, p: number, q: number): number {
  if (p === 0) return x === 0 ? 0 : Number.NEGATIVE_INFINITY;
  if (q === 0) return x === n ? 0 : Number.NEGATIVE_INFINITY;
  if (x === 0) {
    if (n === 0) return 0;
    return p < 0.1 ? -bd0(n, n * q) - n * p : n * Math.log(q);
  }
  if (x === n) return q < 0.1 ? -bd0(n, n * p) - n * q : n * Math.log(p);
  if (x < 0 || x > n) return Number.NEGATIVE_INFINITY;
  const lc = stirlerr(n) - stirlerr(x) - stirlerr(n - x) - bd0(x, n * p) - bd0(n - x, n * q);
  const lf = LN_2PI + Math.log(x) + Math.log1p(-x / n);
  return lc - 0.5 * lf;
}

/** Throws unless `value` is a finite number strictly greater than zero. */
function requirePositive(name: string, value: number): void {
  if (!(value > 0) || !Number.isFinite(value)) {
    throw new InvalidParameterError(`${name} must be a finite number > 0`, name, value);
  }
}

/** Throws unless `value` is a finite number. */
function requireFinite(name: string, value: number): void {
  if (!Number.isFinite(value)) {
    throw new InvalidParameterError(`${name} must be a finite number`, name, value);
  }
}

/** Throws unless `p` is a probability in [0, 1] (NaN is rejected). */
function checkProbability(p: number): void {
  if (!(p >= 0 && p <= 1)) {
    throw new InvalidParameterError("p must be in [0,1]", "p", p);
  }
}

/** Throws unless `size` is a non-negative integer. */
function checkSize(size: number): void {
  if (!Number.isInteger(size) || size < 0) {
    throw new InvalidParameterError("size must be a non-negative integer", "size", size);
  }
}

/** Draws `size` values from `draw`. */
function sample(size: number, draw: () => number): number[] {
  checkSize(size);
  const out = new Array<number>(size);
  for (let i = 0; i < size; i++) out[i] = draw();
  return out;
}

/** Box-Muller transform for normal random variates */
function randn(): number {
  let u = 0,
    v = 0;
  while (u === 0) u = __random();
  while (v === 0) v = __random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Gamma random variate (scale 1) via the Marsaglia-Tsang method. */
function gammaSample(shape: number): number {
  if (shape < 1) {
    return gammaSample(shape + 1) * __random() ** (1 / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x: number, v: number;
    do {
      x = randn();
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = __random();
    if (u < 1 - 0.0331 * (x * x) * (x * x)) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

/**
 * Smallest x >= 0 with `reached(x)` true, for a monotone predicate that is false
 * below the target and true at and above it (a CDF threshold `cdf(x) >= p`, or a
 * survival threshold `sf(x) <= 1 - p`). `hi` is the upper end of the support
 * (`Infinity` for unbounded support) and `reached(hi)` is assumed true.
 *
 * The root is bracketed (doubling upwards from `guess` when the support is
 * unbounded, shrinking by 16x steps downwards) and then bisected, geometrically
 * while the bracket spans more than a factor of two. The final accuracy is therefore
 * relative to the quantile itself, so quantiles of 1e-30 are resolved as well as
 * quantiles of 10; a fixed absolute tolerance would return roughly 1e-12 for both.
 */
function invertOnNonNegative(reached: (x: number) => boolean, guess: number, hi: number): number {
  let top: number;
  if (Number.isFinite(hi)) {
    top = hi;
  } else {
    top = guess > 0 && Number.isFinite(guess) ? guess : 1;
    for (let i = 0; i < 1100 && !reached(top); i++) top *= 2;
    if (!Number.isFinite(top)) return Number.POSITIVE_INFINITY;
  }
  let bottom = 0;
  for (let i = 0; i < 1100; i++) {
    const cand = top / 16;
    if (cand < MIN_NORMAL) {
      // Below the smallest normal double: either the quantile is smaller still (0 is the
      // closest answer) or it lies in [MIN_NORMAL, top) and is bisected below.
      if (reached(MIN_NORMAL)) return 0;
      bottom = MIN_NORMAL;
      break;
    }
    if (reached(cand)) top = cand;
    else {
      bottom = cand;
      break;
    }
  }
  if (bottom === 0) return top;
  for (let i = 0; i < 300; i++) {
    const mid = top > 2 * bottom ? Math.sqrt(bottom) * Math.sqrt(top) : 0.5 * (bottom + top);
    if (!(mid > bottom && mid < top)) break;
    if (reached(mid)) top = mid;
    else bottom = mid;
    if (top - bottom <= 1e-15 * top) break;
  }
  return 0.5 * (bottom + top);
}

/**
 * Quantile on [0, hi] from a CDF and its survival function, given the lower-tail
 * probability `p` and its complement `q = 1 - p`. The smaller of the two is inverted
 * through the matching tail function, so neither loses precision to `1 - p` style
 * cancellation. Callers pass `(p, 1 - p)` for `ppf` and `(1 - q, q)` for `isf`, where
 * the smaller argument is always the one given by the user, hence exact.
 */
function continuousQuantile(
  cdf: (x: number) => number,
  sf: (x: number) => number,
  p: number,
  q: number,
  guess: number,
  hi: number
): number {
  const reached = p <= q ? (x: number) => cdf(x) >= p : (x: number) => sf(x) <= q;
  return invertOnNonNegative(reached, guess, hi);
}

/**
 * Smallest integer k in [lo, hi] with P(X <= k) >= p (equivalently P(X > k) <= q),
 * for a discrete distribution with CDF `cdf` and survival function `sf`, where
 * `q = 1 - p` is passed alongside `p` (see {@link continuousQuantile}). `hi` may be
 * `Infinity`. The tail probability used is whichever of `p` and `q` is the smaller
 * one, so a p close to 1 is resolved through `sf`.
 */
function discreteQuantile(
  cdf: (k: number) => number,
  sf: (k: number) => number,
  p: number,
  q: number,
  lo: number,
  hi: number,
  guess: number
): number {
  if (p === 0) return lo;
  if (q === 0) return hi;
  const reached = p <= q ? (k: number) => cdf(k) >= p : (k: number) => sf(k) <= q;
  let low = lo;
  let top = hi;
  if (!Number.isFinite(hi)) {
    top = Math.max(lo, Math.ceil(guess));
    while (!reached(top)) {
      low = top + 1;
      top = top * 2 + 1;
      if (top > 2 ** 52) return Number.POSITIVE_INFINITY;
    }
  }
  while (low < top) {
    const mid = Math.floor(low + (top - low) / 2);
    // Above 2^53 neighbouring doubles are more than 1 apart, so `mid` can equal `top`
    // or `mid + 1` can equal `mid`; stop there instead of looping forever.
    if (!(mid < top)) break;
    if (reached(mid)) top = mid;
    else if (mid + 1 > mid) low = mid + 1;
    else break;
  }
  return top;
}

/**
 * Binomial deviate by sequential inversion (BINV of Kachitvichyanukul and
 * Schmeiser); exact and cheap when `n * p < 10`, with `p <= 0.5`.
 */
function binomialInversion(n: number, p: number): number {
  const s = p / (1 - p);
  const a = (n + 1) * s;
  const f0 = Math.exp(n * Math.log1p(-p));
  for (;;) {
    let f = f0;
    let u = __random();
    let x = 0;
    while (u >= f) {
      u -= f;
      x++;
      if (x > n) break;
      f *= a / x - s;
    }
    if (x <= n) return x;
  }
}

/**
 * Binomial deviate by transformed rejection with squeeze (BTRS, Hörmann 1993);
 * exact, with constant expected cost, for `n * p >= 10` and `p <= 0.5`.
 */
function binomialBtrs(n: number, p: number): number {
  const q = 1 - p;
  const spq = Math.sqrt(n * p * q);
  const b = 1.15 + 2.53 * spq;
  const a = -0.0873 + 0.0248 * b + 0.01 * p;
  const c = n * p + 0.5;
  const vr = 0.92 - 4.2 / b;
  const alpha = (2.83 + 5.1 / b) * spq;
  const lpq = Math.log(p / q);
  const m = Math.floor((n + 1) * p);
  const h = logGamma(m + 1) + logGamma(n - m + 1);
  for (;;) {
    const u = __random() - 0.5;
    let v = __random();
    const us = 0.5 - Math.abs(u);
    const k = Math.floor(((2 * a) / us + b) * u + c);
    if (!(k >= 0 && k <= n)) continue;
    if (us >= 0.07 && v <= vr) return k;
    v = Math.log((v * alpha) / (a / (us * us) + b));
    if (v <= h - logGamma(k + 1) - logGamma(n - k + 1) + (k - m) * lpq) return k;
  }
}

/** Binomial deviate for any `n >= 0` and `p` in [0, 1]. */
function sampleBinomial(n: number, p: number): number {
  if (n === 0 || p === 0) return 0;
  if (p === 1) return n;
  const flip = p > 0.5;
  const pp = flip ? 1 - p : p;
  const k = n * pp < 10 ? binomialInversion(n, pp) : binomialBtrs(n, pp);
  return flip ? n - k : k;
}

/**
 * Sample a Poisson deviate. Knuth's product method is used for small lambda,
 * but exp(-lambda) underflows to 0 for lambda ≳ 745 (the loop then terminates
 * on the product underflowing, silently capping samples at ~700). For
 * lambda ≥ 30 a transformed-rejection method (Atkinson) is used, which
 * is exact across the full range.
 */
function samplePoisson(lambda: number): number {
  if (lambda < 30) {
    const L = Math.exp(-lambda);
    let k = 0;
    let p = 1;
    do {
      k++;
      p *= __random();
    } while (p > L);
    return k - 1;
  }
  const c = 0.767 - 3.36 / lambda;
  const beta = Math.PI / Math.sqrt(3 * lambda);
  const alpha = beta * lambda;
  const k = Math.log(c) - lambda - Math.log(beta);
  for (;;) {
    const u = __random();
    if (u === 0 || u === 1) continue;
    const x = (alpha - Math.log((1 - u) / u)) / beta;
    const n = Math.floor(x + 0.5);
    if (n < 0 || !Number.isFinite(n)) continue;
    const v = __random();
    const y = alpha - beta * x;
    const lhs = y + Math.log(v / (1 + Math.exp(y)) ** 2);
    const rhs = k + n * Math.log(lambda) - logGamma(n + 1);
    if (lhs <= rhs) return n;
  }
}

/**
 * Gamma density `rate^a x^(a-1) e^(-rate x) / Gamma(a)` for `x > 0`, through the Poisson
 * probability at `a - 1` (or `a` for `a < 1`) so that large shapes keep full precision.
 */
function gammaDensity(shape: number, rate: number, x: number): number {
  const z = rate * x;
  if (shape < 1) {
    return (Math.exp(logPoissonRaw(shape, z)) * shape) / x;
  }
  return Math.exp(logPoissonRaw(shape - 1, z)) * rate;
}

// ---- Distribution Interface ----

/**
 * A continuous probability distribution with a fixed set of parameters.
 *
 * Create one with a factory such as {@link norm} or {@link gamma}. `pdf`, `cdf` and `sf`
 * return NaN for a NaN argument. `ppf` throws for `p` outside [0, 1] (including NaN) and
 * returns the end of the support for `p` equal to 0 or 1.
 */
export interface ContinuousDistribution {
  /** Probability density at `x`. */
  pdf(x: number): number;
  /** Cumulative distribution function, P(X <= x). */
  cdf(x: number): number;
  /** Percent point function (quantile, the inverse of `cdf`) for `p` in [0, 1]. */
  ppf(p: number): number;
  /** Survival function, P(X > x), accurate in the upper tail where `1 - cdf(x)` is not. */
  sf(x: number): number;
  /**
   * Inverse survival function: the `x` with `sf(x) = q`, for `q` in [0, 1]. Equal to
   * `ppf(1 - q)` but accurate for a tiny `q`, where `1 - q` is not.
   */
  isf(q: number): number;
  /** Draws `size` (default 1) independent random variates. */
  rvs(size?: number): number[];
  /** Expected value (NaN or Infinity where it does not exist). */
  mean(): number;
  /** Variance (NaN or Infinity where it does not exist). */
  variance(): number;
  /** Standard deviation, the square root of `variance()`. */
  std(): number;
  /** Median, `ppf(0.5)`. */
  median(): number;
  /** Differential entropy in nats. */
  entropy(): number;
}

/**
 * A discrete probability distribution on the integers with a fixed set of parameters.
 *
 * Create one with a factory such as {@link binom} or {@link poisson}. `pmf` is 0 at
 * non-integer arguments. `cdf` and `sf` accept any real `x` (they use `floor(x)`).
 * `ppf` returns the smallest integer `k` with `cdf(k) >= p`, and throws for `p` outside
 * [0, 1] (including NaN).
 */
export interface DiscreteDistribution {
  /** Probability mass at `k`. */
  pmf(k: number): number;
  /** Cumulative distribution function, P(X <= x). */
  cdf(x: number): number;
  /** Percent point function: the smallest integer `k` with `cdf(k) >= p`, for `p` in [0, 1]. */
  ppf(p: number): number;
  /** Survival function, P(X > x), accurate in the upper tail where `1 - cdf(x)` is not. */
  sf(x: number): number;
  /**
   * Inverse survival function: the smallest integer `k` with `sf(k) <= q`, for `q` in
   * [0, 1]. Equal to `ppf(1 - q)` but accurate for a tiny `q`, where `1 - q` is not.
   */
  isf(q: number): number;
  /** Draws `size` (default 1) independent random variates. */
  rvs(size?: number): number[];
  /** Expected value. */
  mean(): number;
  /** Variance. */
  variance(): number;
  /** Standard deviation, the square root of `variance()`. */
  std(): number;
  /** Median, `ppf(0.5)`. */
  median(): number;
}

/** Shared `std`, `median` and a default `isf` for the continuous distributions. */
abstract class ContinuousBase implements ContinuousDistribution {
  abstract pdf(x: number): number;
  abstract cdf(x: number): number;
  abstract ppf(p: number): number;
  abstract sf(x: number): number;
  abstract rvs(size?: number): number[];
  abstract mean(): number;
  abstract variance(): number;
  abstract entropy(): number;

  isf(q: number): number {
    checkProbability(q);
    return this.ppf(1 - q);
  }

  std(): number {
    return Math.sqrt(this.variance());
  }

  median(): number {
    return this.ppf(0.5);
  }
}

/** Shared `std`, `median` and a default `isf` for the discrete distributions. */
abstract class DiscreteBase implements DiscreteDistribution {
  abstract pmf(k: number): number;
  abstract cdf(x: number): number;
  abstract ppf(p: number): number;
  abstract sf(x: number): number;
  abstract rvs(size?: number): number[];
  abstract mean(): number;
  abstract variance(): number;

  isf(q: number): number {
    checkProbability(q);
    return this.ppf(1 - q);
  }

  std(): number {
    return Math.sqrt(this.variance());
  }

  median(): number {
    return this.ppf(0.5);
  }
}

// ---- Normal Distribution ----

class NormalDist extends ContinuousBase {
  constructor(
    private loc: number = 0,
    private scale: number = 1
  ) {
    super();
    requireFinite("loc", loc);
    requirePositive("scale", scale);
  }

  pdf(x: number): number {
    const z = (x - this.loc) / this.scale;
    return Math.exp(-0.5 * z * z) / (this.scale * SQRT_2PI);
  }

  cdf(x: number): number {
    return normalCdf((x - this.loc) / this.scale);
  }

  ppf(p: number): number {
    checkProbability(p);
    return this.loc + this.scale * normalPpf(p);
  }

  sf(x: number): number {
    // Upper tail computed directly rather than as 1 - Φ(z), which cancels to
    // exactly 0 for x more than ~8σ out.
    return normalSf((x - this.loc) / this.scale);
  }

  override isf(q: number): number {
    checkProbability(q);
    return this.loc - this.scale * normalPpf(q);
  }

  rvs(size = 1): number[] {
    return sample(size, () => this.loc + this.scale * randn());
  }

  mean(): number {
    return this.loc;
  }
  variance(): number {
    return this.scale * this.scale;
  }
  entropy(): number {
    return 0.5 * Math.log(2 * Math.PI * Math.E) + Math.log(this.scale);
  }
}

// ---- Student's t Distribution ----

class TDist extends ContinuousBase {
  constructor(private df: number) {
    super();
    requirePositive("df", df);
  }

  pdf(x: number): number {
    const v = this.df;
    // Written with logBeta and log1p: the Γ((v+1)/2) / Γ(v/2) ratio computed from
    // two log-gammas loses all precision for very large df.
    return Math.exp(
      -logBeta(v / 2, 0.5) - 0.5 * Math.log(v) - ((v + 1) / 2) * Math.log1p((x * x) / v)
    );
  }

  cdf(x: number): number {
    return studentTCdf(x, this.df);
  }

  ppf(p: number): number {
    checkProbability(p);
    if (p === 0) return Number.NEGATIVE_INFINITY;
    if (p === 1) return Number.POSITIVE_INFINITY;
    if (p === 0.5) return 0;
    // Distribution is symmetric: solve the upper tail of the smaller probability,
    // 1 - p is exact for p >= 0.5 and the tail function stays accurate.
    const q = p < 0.5 ? p : 1 - p;
    const y = invertOnNonNegative(
      (x) => studentTSf(x, this.df) <= q,
      Math.max(1, normalPpf(1 - q)),
      Number.POSITIVE_INFINITY
    );
    return p < 0.5 ? -y : y;
  }

  sf(x: number): number {
    return studentTSf(x, this.df);
  }

  override isf(q: number): number {
    checkProbability(q);
    // Symmetric about 0.
    return -this.ppf(q) || 0;
  }

  rvs(size = 1): number[] {
    return sample(size, () => {
      const z = randn();
      const chi2 = 2 * gammaSample(this.df / 2);
      return z / Math.sqrt(chi2 / this.df);
    });
  }

  mean(): number {
    return this.df > 1 ? 0 : Number.NaN;
  }
  variance(): number {
    if (this.df > 2) return this.df / (this.df - 2);
    if (this.df > 1) return Number.POSITIVE_INFINITY;
    return Number.NaN;
  }
  entropy(): number {
    // Exact differential entropy of Student's t:
    //   h = ((v+1)/2)·[ψ((v+1)/2) − ψ(v/2)] + ½·ln(v) + ln B(v/2, ½)
    const v = this.df;
    const x = v / 2;
    // ψ(x + 1/2) − ψ(x): for large x the two digammas agree to ~log10(x) digits and their
    // difference (≈ 1/(2x)) is then multiplied by x, so use the asymptotic series
    // 1/(2x) + 1/(8x²) − 1/(64x⁴) + 1/(128x⁶) − 17/(2048x⁸) instead.
    const dpsi =
      x >= 50
        ? 1 / (2 * x) +
          1 / (8 * x * x) -
          1 / (64 * x ** 4) +
          1 / (128 * x ** 6) -
          17 / (2048 * x ** 8)
        : digamma(x + 0.5) - digamma(x);
    return ((v + 1) / 2) * dpsi + 0.5 * Math.log(v) + logBeta(x, 0.5);
  }
}

// ---- Chi-squared Distribution ----

class Chi2Dist extends ContinuousBase {
  constructor(private df: number) {
    super();
    requirePositive("df", df);
  }

  pdf(x: number): number {
    if (x < 0) return 0;
    const k = this.df;
    if (x === 0) return k < 2 ? Number.POSITIVE_INFINITY : k === 2 ? 0.5 : 0;
    if (x === Number.POSITIVE_INFINITY) return 0;
    return gammaDensity(k / 2, 0.5, x);
  }

  cdf(x: number): number {
    return chiSquareCdf(x, this.df);
  }

  ppf(p: number): number {
    checkProbability(p);
    return this.quantile(p, 1 - p);
  }

  override isf(q: number): number {
    checkProbability(q);
    return this.quantile(1 - q, q);
  }

  /** Quantile for the lower-tail probability `p` and its complement `q = 1 - p`. */
  private quantile(p: number, q: number): number {
    if (p === 0) return 0;
    if (q === 0) return Number.POSITIVE_INFINITY;
    // Wilson–Hilferty starting point for the bracket.
    const guess = Math.max(
      0.01,
      this.df * (1 - 2 / (9 * this.df) + normalPpf(p) * Math.sqrt(2 / (9 * this.df))) ** 3
    );
    return continuousQuantile(
      (x) => this.cdf(x),
      (x) => this.sf(x),
      p,
      q,
      guess,
      Number.POSITIVE_INFINITY
    );
  }

  sf(x: number): number {
    return chiSquareSf(x, this.df);
  }

  rvs(size = 1): number[] {
    return sample(size, () => 2 * gammaSample(this.df / 2));
  }

  mean(): number {
    return this.df;
  }
  variance(): number {
    return 2 * this.df;
  }
  entropy(): number {
    // h = k/2 + ln(2) + lnΓ(k/2) + (1 − k/2)·ψ(k/2)
    const k = this.df;
    return k / 2 + Math.log(2) + logGamma(k / 2) + (1 - k / 2) * digamma(k / 2);
  }
}

// ---- F Distribution ----

class FDist extends ContinuousBase {
  constructor(
    private dfn: number,
    private dfd: number
  ) {
    super();
    requirePositive("dfn", dfn);
    requirePositive("dfd", dfd);
  }

  pdf(x: number): number {
    if (x < 0) return 0;
    const d1 = this.dfn,
      d2 = this.dfd;
    if (x === 0) return d1 < 2 ? Number.POSITIVE_INFINITY : d1 === 2 ? 1 : 0;
    if (x === Number.POSITIVE_INFINITY) return 0;
    // Binomial-probability form (as in R's df): the direct (d1 x)^d1 d2^d2 products
    // overflow for df beyond ~150 and a log-gamma form loses digits for large df.
    const w = 1 / (d2 + x * d1);
    const q = d2 * w;
    const p = x * d1 * w;
    if (d1 >= 2) {
      return Math.exp(logBinomialRaw((d1 - 2) / 2, (d1 + d2 - 2) / 2, p, q)) * ((d1 * q) / 2);
    }
    if (p === 0) return Number.POSITIVE_INFINITY;
    const lb = logBinomialRaw(d1 / 2, (d1 + d2) / 2, p, q);
    const direct = Math.exp(lb) * ((d1 * d1 * q) / (2 * p * (d1 + d2)));
    if (Number.isFinite(direct)) return direct;
    // For subnormal x the factor 1 / p overflows before the probability underflows.
    return Math.exp(lb + Math.log((d1 * d1 * q) / (2 * (d1 + d2))) - Math.log(p));
  }

  cdf(x: number): number {
    return fCdf(x, this.dfn, this.dfd);
  }

  ppf(p: number): number {
    checkProbability(p);
    return this.quantile(p, 1 - p);
  }

  override isf(q: number): number {
    checkProbability(q);
    return this.quantile(1 - q, q);
  }

  /** Quantile for the lower-tail probability `p` and its complement `q = 1 - p`. */
  private quantile(p: number, q: number): number {
    if (p === 0) return 0;
    if (q === 0) return Number.POSITIVE_INFINITY;
    return continuousQuantile(
      (x) => this.cdf(x),
      (x) => this.sf(x),
      p,
      q,
      1,
      Number.POSITIVE_INFINITY
    );
  }

  sf(x: number): number {
    return fSf(x, this.dfn, this.dfd);
  }

  rvs(size = 1): number[] {
    return sample(size, () => {
      const x1 = 2 * gammaSample(this.dfn / 2);
      const x2 = 2 * gammaSample(this.dfd / 2);
      return x1 / this.dfn / (x2 / this.dfd);
    });
  }

  mean(): number {
    // The F variable is positive, so the mean diverges to +Infinity for dfd <= 2.
    return this.dfd > 2 ? this.dfd / (this.dfd - 2) : Number.POSITIVE_INFINITY;
  }

  variance(): number {
    const d1 = this.dfn,
      d2 = this.dfd;
    if (d2 <= 4) return Number.POSITIVE_INFINITY;
    return (2 * d2 * d2 * (d1 + d2 - 2)) / (d1 * (d2 - 2) * (d2 - 2) * (d2 - 4));
  }

  entropy(): number {
    // h = ln(d2/d1) + ln B(d1/2, d2/2) + (1 − d1/2)·ψ(d1/2)
    //     − (1 + d2/2)·ψ(d2/2) + ((d1+d2)/2)·ψ((d1+d2)/2)
    const d1 = this.dfn,
      d2 = this.dfd;
    return (
      Math.log(d2 / d1) +
      logBeta(d1 / 2, d2 / 2) +
      (1 - d1 / 2) * digamma(d1 / 2) -
      (1 + d2 / 2) * digamma(d2 / 2) +
      ((d1 + d2) / 2) * digamma((d1 + d2) / 2)
    );
  }
}

// ---- Uniform Distribution ----

class UniformDist extends ContinuousBase {
  constructor(
    private low: number = 0,
    private high: number = 1
  ) {
    super();
    requireFinite("low", low);
    requireFinite("high", high);
    if (low >= high) throw new InvalidParameterError("low must be < high", "low", low);
    if (!Number.isFinite(high - low)) {
      throw new InvalidParameterError("high - low must be finite", "high", high);
    }
  }

  pdf(x: number): number {
    if (Number.isNaN(x)) return Number.NaN;
    return x >= this.low && x <= this.high ? 1 / (this.high - this.low) : 0;
  }

  cdf(x: number): number {
    if (x < this.low) return 0;
    if (x > this.high) return 1;
    return (x - this.low) / (this.high - this.low);
  }

  ppf(p: number): number {
    checkProbability(p);
    return this.low + p * (this.high - this.low);
  }

  sf(x: number): number {
    if (x < this.low) return 1;
    if (x > this.high) return 0;
    return (this.high - x) / (this.high - this.low);
  }

  override isf(q: number): number {
    checkProbability(q);
    return this.high - q * (this.high - this.low);
  }

  rvs(size = 1): number[] {
    return sample(size, () => this.low + __random() * (this.high - this.low));
  }

  mean(): number {
    return (this.low + this.high) / 2;
  }
  variance(): number {
    const r = this.high - this.low;
    return (r * r) / 12;
  }
  entropy(): number {
    return Math.log(this.high - this.low);
  }
}

// ---- Exponential Distribution ----

class ExponentialDist extends ContinuousBase {
  constructor(private rate: number = 1) {
    super();
    requirePositive("rate", rate);
  }

  pdf(x: number): number {
    return x < 0 ? 0 : this.rate * Math.exp(-this.rate * x);
  }

  cdf(x: number): number {
    return x < 0 ? 0 : -Math.expm1(-this.rate * x);
  }

  ppf(p: number): number {
    checkProbability(p);
    if (p === 1) return Number.POSITIVE_INFINITY;
    return -Math.log1p(-p) / this.rate;
  }

  sf(x: number): number {
    return x < 0 ? 1 : Math.exp(-this.rate * x);
  }

  override isf(q: number): number {
    checkProbability(q);
    return -Math.log(q) / this.rate || 0;
  }

  rvs(size = 1): number[] {
    return sample(size, () => {
      let u = __random();
      while (u === 0) u = __random();
      return -Math.log(u) / this.rate;
    });
  }

  mean(): number {
    return 1 / this.rate;
  }
  variance(): number {
    return 1 / (this.rate * this.rate);
  }
  entropy(): number {
    return 1 - Math.log(this.rate);
  }
}

// ---- Beta Distribution ----

class BetaDist extends ContinuousBase {
  constructor(
    private a: number,
    private b: number
  ) {
    super();
    requirePositive("a", a);
    requirePositive("b", b);
  }

  pdf(x: number): number {
    if (x < 0 || x > 1) return 0;
    if (x === 0) return this.a < 1 ? Number.POSITIVE_INFINITY : this.a === 1 ? this.b : 0;
    if (x === 1) return this.b < 1 ? Number.POSITIVE_INFINITY : this.b === 1 ? this.a : 0;
    const { a, b } = this;
    if (a <= 2 || b <= 2) {
      return Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log1p(-x) - logBeta(a, b));
    }
    // Binomial-probability form (as in R's dbeta): accurate for large a and b.
    return (a + b - 1) * Math.exp(logBinomialRaw(a - 1, a + b - 2, x, 1 - x));
  }

  cdf(x: number): number {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    return regularizedIncompleteBeta(this.a, this.b, x);
  }

  ppf(p: number): number {
    checkProbability(p);
    return this.quantile(p, 1 - p);
  }

  override isf(q: number): number {
    checkProbability(q);
    return this.quantile(1 - q, q);
  }

  /** Quantile for the lower-tail probability `p` and its complement `q = 1 - p`. */
  private quantile(p: number, q: number): number {
    if (p === 0) return 0;
    if (q === 0) return 1;
    return continuousQuantile(
      (x) => this.cdf(x),
      (x) => this.sf(x),
      p,
      q,
      0.5,
      1
    );
  }

  sf(x: number): number {
    if (x <= 0) return 1;
    if (x >= 1) return 0;
    // Below the middle of the support 1 - x rounds away the low bits of x, so the lower
    // tail is used instead (its complement is not small there, nothing is lost).
    if (x < 0.5) {
      const c = regularizedIncompleteBeta(this.a, this.b, x);
      if (c < 0.9) return 1 - c;
    }
    // 1 - I_x(a, b) = I_{1-x}(b, a); computed directly so the upper tail keeps its precision.
    return regularizedIncompleteBeta(this.b, this.a, 1 - x);
  }

  rvs(size = 1): number[] {
    return sample(size, () => {
      const x = gammaSample(this.a);
      const y = gammaSample(this.b);
      return x / (x + y);
    });
  }

  mean(): number {
    return this.a / (this.a + this.b);
  }
  variance(): number {
    const ab = this.a + this.b;
    return (this.a * this.b) / (ab * ab * (ab + 1));
  }
  entropy(): number {
    // h = ln B(a,b) − (a−1)·ψ(a) − (b−1)·ψ(b) + (a+b−2)·ψ(a+b)
    const ab = this.a + this.b;
    return (
      logBeta(this.a, this.b) -
      (this.a - 1) * digamma(this.a) -
      (this.b - 1) * digamma(this.b) +
      (ab - 2) * digamma(ab)
    );
  }
}

// ---- Gamma Distribution ----

class GammaDist extends ContinuousBase {
  constructor(
    private shape: number,
    private rate: number = 1
  ) {
    super();
    requirePositive("shape", shape);
    requirePositive("rate", rate);
  }

  pdf(x: number): number {
    if (x < 0) return 0;
    const a = this.shape,
      b = this.rate;
    if (x === 0) return a < 1 ? Number.POSITIVE_INFINITY : a === 1 ? b : 0;
    if (x === Number.POSITIVE_INFINITY) return 0;
    return gammaDensity(a, b, x);
  }

  cdf(x: number): number {
    if (x <= 0) return 0;
    // chi2_cdf(x, k) = regularized_gamma(k/2, x/2), so gamma_cdf(x, a, b) = regularized_gamma(a, b*x)
    return chiSquareCdf(2 * this.rate * x, 2 * this.shape);
  }

  ppf(p: number): number {
    checkProbability(p);
    return this.quantile(p, 1 - p);
  }

  override isf(q: number): number {
    checkProbability(q);
    return this.quantile(1 - q, q);
  }

  /** Quantile for the lower-tail probability `p` and its complement `q = 1 - p`. */
  private quantile(p: number, q: number): number {
    if (p === 0) return 0;
    if (q === 0) return Number.POSITIVE_INFINITY;
    return continuousQuantile(
      (x) => this.cdf(x),
      (x) => this.sf(x),
      p,
      q,
      this.shape / this.rate,
      Number.POSITIVE_INFINITY
    );
  }

  sf(x: number): number {
    if (x <= 0) return 1;
    return chiSquareSf(2 * this.rate * x, 2 * this.shape);
  }

  rvs(size = 1): number[] {
    return sample(size, () => gammaSample(this.shape) / this.rate);
  }

  mean(): number {
    return this.shape / this.rate;
  }
  variance(): number {
    return this.shape / (this.rate * this.rate);
  }
  entropy(): number {
    // h = a − ln(b) + lnΓ(a) + (1 − a)·ψ(a)
    const a = this.shape,
      b = this.rate;
    return a - Math.log(b) + logGamma(a) + (1 - a) * digamma(a);
  }
}

// ---- Binomial Distribution ----

class BinomialDist extends DiscreteBase {
  constructor(
    private n: number,
    private p_param: number
  ) {
    super();
    if (!Number.isInteger(n) || n < 0)
      throw new InvalidParameterError("n must be a non-negative integer", "n", n);
    if (!(p_param >= 0 && p_param <= 1))
      throw new InvalidParameterError("p must be in [0,1]", "p", p_param);
  }

  pmf(k: number): number {
    if (Number.isNaN(k)) return Number.NaN;
    if (!Number.isInteger(k) || k < 0 || k > this.n) return 0;
    return Math.exp(logBinomialRaw(k, this.n, this.p_param, 1 - this.p_param));
  }

  cdf(x: number): number {
    if (Number.isNaN(x)) return Number.NaN;
    const k = Math.floor(x);
    if (k < 0) return 0;
    if (k >= this.n) return 1;
    if (this.p_param === 0) return 1;
    if (this.p_param === 1) return 0;
    // For small p the argument 1 - p of the incomplete beta below has rounded away the low
    // bits of p, so the upper tail (computed from p itself) is used when it is not near 1.
    if (this.p_param < 0.5) {
      const s = regularizedIncompleteBeta(k + 1, this.n - k, this.p_param);
      if (s < 0.9) return 1 - s;
    }
    // CDF_binom(k; n, p) = I_{1-p}(n-k, k+1)
    return regularizedIncompleteBeta(this.n - k, k + 1, 1 - this.p_param);
  }

  ppf(p: number): number {
    checkProbability(p);
    return this.quantile(p, 1 - p);
  }

  override isf(q: number): number {
    checkProbability(q);
    return this.quantile(1 - q, q);
  }

  /** Quantile for the lower-tail probability `p` and its complement `q = 1 - p`. */
  private quantile(p: number, q: number): number {
    return discreteQuantile(
      (k) => this.cdf(k),
      (k) => this.sf(k),
      p,
      q,
      0,
      this.n,
      this.mean()
    );
  }

  sf(x: number): number {
    if (Number.isNaN(x)) return Number.NaN;
    const k = Math.floor(x);
    if (k < 0) return 1;
    if (k >= this.n) return 0;
    if (this.p_param === 0) return 0;
    if (this.p_param === 1) return 1;
    // P(X > k) = I_p(k+1, n-k), computed directly for the upper tail.
    return regularizedIncompleteBeta(k + 1, this.n - k, this.p_param);
  }

  rvs(size = 1): number[] {
    return sample(size, () => sampleBinomial(this.n, this.p_param));
  }

  mean(): number {
    return this.n * this.p_param;
  }
  variance(): number {
    return this.n * this.p_param * (1 - this.p_param);
  }
}

// ---- Poisson Distribution ----

class PoissonDist extends DiscreteBase {
  constructor(private mu: number) {
    super();
    requirePositive("mu", mu);
  }

  pmf(k: number): number {
    if (Number.isNaN(k)) return Number.NaN;
    if (!Number.isInteger(k) || k < 0) return 0;
    return Math.exp(logPoissonRaw(k, this.mu));
  }

  cdf(x: number): number {
    if (Number.isNaN(x)) return Number.NaN;
    const k = Math.floor(x);
    if (k < 0) return 0;
    if (!Number.isFinite(2 * (k + 1))) return 1;
    // P(X <= k) = Q(k+1, mu), the regularized upper incomplete gamma function.
    return chiSquareSf(2 * this.mu, 2 * (k + 1));
  }

  ppf(p: number): number {
    checkProbability(p);
    return this.quantile(p, 1 - p);
  }

  override isf(q: number): number {
    checkProbability(q);
    return this.quantile(1 - q, q);
  }

  /** Quantile for the lower-tail probability `p` and its complement `q = 1 - p`. */
  private quantile(p: number, q: number): number {
    return discreteQuantile(
      (k) => this.cdf(k),
      (k) => this.sf(k),
      p,
      q,
      0,
      Number.POSITIVE_INFINITY,
      this.mu
    );
  }

  sf(x: number): number {
    if (Number.isNaN(x)) return Number.NaN;
    const k = Math.floor(x);
    if (k < 0) return 1;
    if (!Number.isFinite(2 * (k + 1))) return 0;
    // P(X > k) = P(k+1, mu), the regularized lower incomplete gamma function.
    return chiSquareCdf(2 * this.mu, 2 * (k + 1));
  }

  rvs(size = 1): number[] {
    return sample(size, () => samplePoisson(this.mu));
  }

  mean(): number {
    return this.mu;
  }
  variance(): number {
    return this.mu;
  }
}

// ---- Factory Functions ----

/**
 * Normal (Gaussian) distribution.
 *
 * Matches `scipy.stats.norm(loc, scale)`.
 *
 * @param loc - Mean (default 0)
 * @param scale - Standard deviation (default 1), must be a finite number > 0
 * @returns Distribution object
 * @throws {InvalidParameterError} If `loc` is not finite or `scale` is not a finite number > 0
 *
 * @example
 * ```ts
 * const d = norm(0, 1);
 * d.cdf(1.96); // 0.9750021048517795
 * d.ppf(0.975); // 1.959963984540054
 * ```
 */
export function norm(loc = 0, scale = 1): ContinuousDistribution {
  return new NormalDist(loc, scale);
}

/**
 * Student's t distribution.
 *
 * Matches `scipy.stats.t(df)`. The mean is NaN for `df <= 1` and the variance is
 * `Infinity` for `1 < df <= 2` and NaN for `df <= 1`.
 *
 * @param df - Degrees of freedom, must be a finite number > 0
 * @returns Distribution object
 * @throws {InvalidParameterError} If `df` is not a finite number > 0
 *
 * @example
 * ```ts
 * t(10).ppf(0.975); // 2.2281388519649385
 * ```
 */
export function t(df: number): ContinuousDistribution {
  return new TDist(df);
}

/**
 * Chi-squared distribution.
 *
 * Matches `scipy.stats.chi2(df)`.
 *
 * @param df - Degrees of freedom, must be a finite number > 0
 * @returns Distribution object
 * @throws {InvalidParameterError} If `df` is not a finite number > 0
 *
 * @example
 * ```ts
 * chi2(3).sf(100); // upper-tail p-value, about 1.1e-21, without cancellation
 * ```
 */
export function chi2(df: number): ContinuousDistribution {
  return new Chi2Dist(df);
}

/**
 * F distribution.
 *
 * Matches `scipy.stats.f(dfn, dfd)`. The mean is `Infinity` for `dfd <= 2` and the
 * variance is `Infinity` for `dfd <= 4` (the variable is positive, so the moments diverge
 * rather than being undefined).
 *
 * @param dfn - Numerator degrees of freedom, must be a finite number > 0
 * @param dfd - Denominator degrees of freedom, must be a finite number > 0
 * @returns Distribution object
 * @throws {InvalidParameterError} If `dfn` or `dfd` is not a finite number > 0
 *
 * @example
 * ```ts
 * f(5, 10).ppf(0.95); // 3.325834530413012
 * ```
 */
export function f(dfn: number, dfd: number): ContinuousDistribution {
  return new FDist(dfn, dfd);
}

/**
 * Continuous uniform distribution.
 *
 * Matches `scipy.stats.uniform(loc=low, scale=high - low)`.
 *
 * @param low - Lower bound (default 0)
 * @param high - Upper bound (default 1), must be greater than `low`
 * @returns Distribution object
 * @throws {InvalidParameterError} If the bounds are not finite or `low >= high`
 *
 * @example
 * ```ts
 * uniform(2, 6).cdf(3); // 0.25
 * ```
 */
export function uniform(low = 0, high = 1): ContinuousDistribution {
  return new UniformDist(low, high);
}

/**
 * Exponential distribution.
 *
 * Matches `scipy.stats.expon(scale=1 / rate)`.
 *
 * @param rate - Rate parameter λ (default 1), must be a finite number > 0. Mean = 1/λ
 * @returns Distribution object
 * @throws {InvalidParameterError} If `rate` is not a finite number > 0
 *
 * @example
 * ```ts
 * expon(2).mean(); // 0.5
 * ```
 */
export function expon(rate = 1): ContinuousDistribution {
  return new ExponentialDist(rate);
}

/**
 * Beta distribution.
 *
 * Matches `scipy.stats.beta(a, b)`.
 *
 * @param a - Alpha shape parameter, must be a finite number > 0
 * @param b - Beta shape parameter, must be a finite number > 0
 * @returns Distribution object
 * @throws {InvalidParameterError} If `a` or `b` is not a finite number > 0
 *
 * @example
 * ```ts
 * beta(2, 5).mean(); // 0.2857142857142857
 * ```
 */
export function beta(a: number, b: number): ContinuousDistribution {
  return new BetaDist(a, b);
}

/**
 * Gamma distribution.
 *
 * Parameterized by shape and **rate**; SciPy's `scipy.stats.gamma(a, scale=...)` uses
 * `scale = 1 / rate`.
 *
 * @param shape - Shape parameter (alpha), must be a finite number > 0
 * @param rate - Rate parameter (beta = 1/scale), must be a finite number > 0. Default 1.
 * @returns Distribution object
 * @throws {InvalidParameterError} If `shape` or `rate` is not a finite number > 0
 *
 * @example
 * ```ts
 * gamma(3, 2).mean(); // 1.5, same as scipy.stats.gamma(3, scale=0.5).mean()
 * ```
 */
export function gamma(shape: number, rate = 1): ContinuousDistribution {
  return new GammaDist(shape, rate);
}

/**
 * Binomial distribution.
 *
 * Matches `scipy.stats.binom(n, p)`.
 *
 * @param n - Number of trials (non-negative integer)
 * @param p - Probability of success per trial, in [0, 1]
 * @returns Distribution object
 * @throws {InvalidParameterError} If `n` is not a non-negative integer or `p` is not in [0, 1]
 *
 * @example
 * ```ts
 * binom(10, 0.5).pmf(5); // 0.24609375
 * ```
 */
export function binom(n: number, p: number): DiscreteDistribution {
  return new BinomialDist(n, p);
}

/**
 * Poisson distribution.
 *
 * Matches `scipy.stats.poisson(mu)`.
 *
 * @param mu - Expected number of events (rate parameter λ), must be a finite number > 0
 * @returns Distribution object
 * @throws {InvalidParameterError} If `mu` is not a finite number > 0
 *
 * @example
 * ```ts
 * poisson(3).cdf(2); // 0.42319008112684353
 * ```
 */
export function poisson(mu: number): DiscreteDistribution {
  return new PoissonDist(mu);
}

// ---- Lognormal Distribution ----

class LognormalDist extends ContinuousBase {
  constructor(
    private mu: number = 0,
    private sigma: number = 1
  ) {
    super();
    requireFinite("mu", mu);
    requirePositive("sigma", sigma);
  }

  pdf(x: number): number {
    if (x <= 0) return 0;
    const z = (Math.log(x) - this.mu) / this.sigma;
    return Math.exp(-0.5 * z * z) / (x * this.sigma * SQRT_2PI);
  }

  cdf(x: number): number {
    if (x <= 0) return 0;
    return normalCdf((Math.log(x) - this.mu) / this.sigma);
  }

  ppf(p: number): number {
    checkProbability(p);
    if (p === 0) return 0;
    if (p === 1) return Number.POSITIVE_INFINITY;
    return Math.exp(this.mu + this.sigma * normalPpf(p));
  }

  sf(x: number): number {
    if (x <= 0) return 1;
    return normalSf((Math.log(x) - this.mu) / this.sigma);
  }

  override isf(q: number): number {
    checkProbability(q);
    if (q === 0) return Number.POSITIVE_INFINITY;
    if (q === 1) return 0;
    return Math.exp(this.mu - this.sigma * normalPpf(q));
  }

  rvs(size = 1): number[] {
    return sample(size, () => Math.exp(this.mu + this.sigma * randn()));
  }

  mean(): number {
    return Math.exp(this.mu + (this.sigma * this.sigma) / 2);
  }
  variance(): number {
    const s2 = this.sigma * this.sigma;
    return Math.expm1(s2) * Math.exp(2 * this.mu + s2);
  }
  entropy(): number {
    return this.mu + 0.5 * Math.log(2 * Math.PI * Math.E) + Math.log(this.sigma);
  }
}

// ---- Weibull Distribution ----

class WeibullDist extends ContinuousBase {
  constructor(
    private k: number,
    private lam: number = 1
  ) {
    super();
    requirePositive("k", k);
    requirePositive("lambda", lam);
  }

  pdf(x: number): number {
    if (x < 0) return 0;
    if (x === 0) return this.k < 1 ? Number.POSITIVE_INFINITY : this.k === 1 ? 1 / this.lam : 0;
    if (x === Number.POSITIVE_INFINITY) return 0;
    const z = x / this.lam;
    // Log form: z^(k-1) · exp(-z^k) is Infinity · 0 = NaN once z^k overflows.
    return (this.k / this.lam) * Math.exp((this.k - 1) * Math.log(z) - z ** this.k);
  }

  cdf(x: number): number {
    if (x <= 0) return 0;
    return -Math.expm1(-((x / this.lam) ** this.k));
  }

  ppf(p: number): number {
    checkProbability(p);
    if (p === 0) return 0;
    if (p === 1) return Number.POSITIVE_INFINITY;
    return this.lam * (-Math.log1p(-p)) ** (1 / this.k);
  }

  sf(x: number): number {
    if (x <= 0) return 1;
    return Math.exp(-((x / this.lam) ** this.k));
  }

  override isf(q: number): number {
    checkProbability(q);
    if (q === 0) return Number.POSITIVE_INFINITY;
    return this.lam * (-Math.log(q)) ** (1 / this.k) || 0;
  }

  rvs(size = 1): number[] {
    return sample(size, () => this.lam * (-Math.log1p(-__random())) ** (1 / this.k));
  }

  mean(): number {
    return this.lam * Math.exp(logGamma(1 + 1 / this.k));
  }
  variance(): number {
    // Gamma(1 + 2/k) - Gamma(1 + 1/k)^2 = Gamma(1 + 1/k)^2 expm1(lnGamma(1 + 2/k) - 2 lnGamma(1 + 1/k));
    // the log form keeps the small difference exact for large shape k.
    const l1 = logGamma(1 + 1 / this.k);
    const l2 = logGamma(1 + 2 / this.k);
    return this.lam * this.lam * Math.exp(2 * l1) * Math.expm1(l2 - 2 * l1);
  }
  entropy(): number {
    const euler = 0.5772156649015329;
    return euler * (1 - 1 / this.k) + Math.log(this.lam / this.k) + 1;
  }
}

// ---- Pareto Distribution ----

class ParetoDist extends ContinuousBase {
  constructor(
    private alpha: number,
    private xm: number = 1
  ) {
    super();
    requirePositive("alpha", alpha);
    requirePositive("xm", xm);
  }

  pdf(x: number): number {
    if (x < this.xm) return 0;
    // Log form: xm^alpha overflows for large alpha even when the density is moderate.
    return Math.exp(
      Math.log(this.alpha) + this.alpha * Math.log(this.xm) - (this.alpha + 1) * Math.log(x)
    );
  }

  cdf(x: number): number {
    if (x < this.xm) return 0;
    if (x === Number.POSITIVE_INFINITY) return 1;
    // log(xm / x) = log1p((xm - x) / x) is exact near x = xm, where xm / x rounds. The
    // `0 -` (not a unary minus) makes the CDF at x = xm +0, never -0.
    return 0 - Math.expm1(this.alpha * Math.log1p((this.xm - x) / x));
  }

  ppf(p: number): number {
    checkProbability(p);
    if (p === 0) return this.xm;
    if (p === 1) return Number.POSITIVE_INFINITY;
    return this.xm * Math.exp(-Math.log1p(-p) / this.alpha);
  }

  sf(x: number): number {
    if (x < this.xm) return 1;
    return (this.xm / x) ** this.alpha;
  }

  override isf(q: number): number {
    checkProbability(q);
    if (q === 0) return Number.POSITIVE_INFINITY;
    return this.xm * Math.exp(-Math.log(q) / this.alpha);
  }

  rvs(size = 1): number[] {
    return sample(size, () => {
      let u = __random();
      while (u === 0) u = __random();
      return this.xm / u ** (1 / this.alpha);
    });
  }

  mean(): number {
    return this.alpha > 1 ? (this.alpha * this.xm) / (this.alpha - 1) : Number.POSITIVE_INFINITY;
  }
  variance(): number {
    if (this.alpha <= 2) return Number.POSITIVE_INFINITY;
    return (this.xm * this.xm * this.alpha) / ((this.alpha - 1) ** 2 * (this.alpha - 2));
  }
  entropy(): number {
    return Math.log(this.xm / this.alpha) + 1 + 1 / this.alpha;
  }
}

// ---- Cauchy Distribution ----

class CauchyDist extends ContinuousBase {
  constructor(
    private x0: number = 0,
    private gammaParam: number = 1
  ) {
    super();
    requireFinite("x0", x0);
    requirePositive("gamma", gammaParam);
  }

  pdf(x: number): number {
    const z = (x - this.x0) / this.gammaParam;
    return 1 / (Math.PI * this.gammaParam * (1 + z * z));
  }

  /** CDF at the standardized point z. For z < -1 the reflection atan(z) = -π/2 - atan(1/z) keeps the tiny lower tail exact. */
  private cdfZ(z: number): number {
    return z < -1 ? Math.atan(-1 / z) / Math.PI : 0.5 + Math.atan(z) / Math.PI;
  }

  cdf(x: number): number {
    return this.cdfZ((x - this.x0) / this.gammaParam);
  }

  ppf(p: number): number {
    checkProbability(p);
    if (p === 0) return Number.NEGATIVE_INFINITY;
    if (p === 1) return Number.POSITIVE_INFINITY;
    if (p === 0.5) return this.x0;
    // tan(π(p - 1/2)) loses all relative precision near the ends, and the cotangent form
    // of each tail loses it near the centre (the angle approaches the pole of tan), so
    // each form is used only where it is accurate. p - 1/2 is exact for p in [1/4, 1].
    if (p <= 0.25) return this.x0 - this.gammaParam / Math.tan(Math.PI * p);
    if (p >= 0.75) return this.x0 + this.gammaParam / Math.tan(Math.PI * (1 - p));
    return this.x0 + this.gammaParam * Math.tan(Math.PI * (p - 0.5));
  }

  sf(x: number): number {
    return this.cdfZ(-(x - this.x0) / this.gammaParam);
  }

  override isf(q: number): number {
    checkProbability(q);
    // Symmetric about x0.
    return 2 * this.x0 - this.ppf(q);
  }

  rvs(size = 1): number[] {
    return sample(size, () => this.x0 + this.gammaParam * Math.tan(Math.PI * (__random() - 0.5)));
  }

  mean(): number {
    return Number.NaN;
  } // Cauchy has no defined mean
  variance(): number {
    return Number.NaN;
  } // Cauchy has no defined variance
  entropy(): number {
    return Math.log(4 * Math.PI * this.gammaParam);
  }
}

// ---- Laplace Distribution ----

class LaplaceDist extends ContinuousBase {
  constructor(
    private loc: number = 0,
    private scale: number = 1
  ) {
    super();
    requireFinite("loc", loc);
    requirePositive("scale", scale);
  }

  pdf(x: number): number {
    return Math.exp(-Math.abs(x - this.loc) / this.scale) / (2 * this.scale);
  }

  cdf(x: number): number {
    const z = (x - this.loc) / this.scale;
    return z <= 0 ? 0.5 * Math.exp(z) : 1 - 0.5 * Math.exp(-z);
  }

  ppf(p: number): number {
    checkProbability(p);
    if (p === 0) return Number.NEGATIVE_INFINITY;
    if (p === 1) return Number.POSITIVE_INFINITY;
    return p <= 0.5
      ? this.loc + this.scale * Math.log(2 * p)
      : this.loc - this.scale * Math.log(2 * (1 - p));
  }

  sf(x: number): number {
    const z = (x - this.loc) / this.scale;
    return z >= 0 ? 0.5 * Math.exp(-z) : 1 - 0.5 * Math.exp(z);
  }

  override isf(q: number): number {
    checkProbability(q);
    // Symmetric about loc.
    return 2 * this.loc - this.ppf(q);
  }

  rvs(size = 1): number[] {
    return sample(size, () => {
      let u = __random();
      while (u === 0) u = __random();
      return this.ppf(u);
    });
  }

  mean(): number {
    return this.loc;
  }
  variance(): number {
    return 2 * this.scale * this.scale;
  }
  entropy(): number {
    return 1 + Math.log(2 * this.scale);
  }
}

// ---- Factory Functions (continued) ----

/**
 * Lognormal distribution.
 *
 * `mu` and `sigma` are the mean and standard deviation of the underlying normal
 * distribution, so `lognorm(mu, sigma)` equals `scipy.stats.lognorm(s=sigma, scale=exp(mu))`.
 *
 * @param mu - Mean of the underlying normal distribution (default 0)
 * @param sigma - Std dev of the underlying normal distribution (default 1), must be > 0
 * @returns Distribution object
 * @throws {InvalidParameterError} If `mu` is not finite or `sigma` is not a finite number > 0
 *
 * @example
 * ```ts
 * lognorm(0, 1).ppf(0.5); // 1
 * ```
 */
export function lognorm(mu = 0, sigma = 1): ContinuousDistribution {
  return new LognormalDist(mu, sigma);
}

/**
 * Weibull distribution.
 *
 * Matches `scipy.stats.weibull_min(c=k, scale=lam)`.
 *
 * @param k - Shape parameter, must be a finite number > 0
 * @param lam - Scale parameter (default 1), must be a finite number > 0
 * @returns Distribution object
 * @throws {InvalidParameterError} If `k` or `lam` is not a finite number > 0
 *
 * @example
 * ```ts
 * weibull(2, 1).cdf(1); // 0.6321205588285577
 * ```
 */
export function weibull(k: number, lam = 1): ContinuousDistribution {
  return new WeibullDist(k, lam);
}

/**
 * Pareto distribution.
 *
 * Matches `scipy.stats.pareto(b=alpha, scale=xm)`.
 *
 * @param alpha - Shape parameter, must be a finite number > 0
 * @param xm - Scale parameter (minimum value, default 1), must be a finite number > 0
 * @returns Distribution object
 * @throws {InvalidParameterError} If `alpha` or `xm` is not a finite number > 0
 *
 * @example
 * ```ts
 * pareto(3, 1).sf(2); // 0.125
 * ```
 */
export function pareto(alpha: number, xm = 1): ContinuousDistribution {
  return new ParetoDist(alpha, xm);
}

/**
 * Cauchy distribution.
 *
 * Matches `scipy.stats.cauchy(loc=x0, scale=gammaParam)`. Mean and variance are NaN.
 *
 * @param x0 - Location parameter (default 0)
 * @param gammaParam - Scale parameter (default 1), must be a finite number > 0
 * @returns Distribution object
 * @throws {InvalidParameterError} If `x0` is not finite or `gammaParam` is not a finite number > 0
 *
 * @example
 * ```ts
 * cauchy(0, 1).ppf(0.75); // 1.0000000000000002 (tan of π/4)
 * ```
 */
export function cauchy(x0 = 0, gammaParam = 1): ContinuousDistribution {
  return new CauchyDist(x0, gammaParam);
}

/**
 * Laplace distribution.
 *
 * Matches `scipy.stats.laplace(loc, scale)`.
 *
 * @param loc - Location parameter (default 0)
 * @param scale - Scale parameter (default 1), must be a finite number > 0
 * @returns Distribution object
 * @throws {InvalidParameterError} If `loc` is not finite or `scale` is not a finite number > 0
 *
 * @example
 * ```ts
 * laplace(0, 2).variance(); // 8
 * ```
 */
export function laplace(loc = 0, scale = 1): ContinuousDistribution {
  return new LaplaceDist(loc, scale);
}

// ---- Geometric Distribution ----

class GeometricDist extends DiscreteBase {
  constructor(private p_param: number) {
    super();
    if (!(p_param > 0 && p_param <= 1))
      throw new InvalidParameterError("p must be in (0,1]", "p", p_param);
  }

  pmf(k: number): number {
    if (Number.isNaN(k)) return Number.NaN;
    if (!Number.isInteger(k) || k < 1) return 0;
    if (this.p_param === 1) return k === 1 ? 1 : 0;
    return this.p_param * Math.exp((k - 1) * Math.log1p(-this.p_param));
  }

  cdf(x: number): number {
    if (Number.isNaN(x)) return Number.NaN;
    const k = Math.floor(x);
    if (k < 1) return 0;
    if (this.p_param === 1) return 1;
    return -Math.expm1(k * Math.log1p(-this.p_param));
  }

  ppf(p: number): number {
    checkProbability(p);
    if (this.p_param === 1) return 1;
    if (p === 0) return 1;
    if (p === 1) return Number.POSITIVE_INFINITY;
    // Closed form, then correct a possible off-by-one from rounding in the logarithms.
    let k = Math.max(1, Math.ceil(Math.log1p(-p) / Math.log1p(-this.p_param)));
    if (k > 1 && this.cdf(k - 1) >= p) k--;
    else if (this.cdf(k) < p) k++;
    return k;
  }

  sf(x: number): number {
    if (Number.isNaN(x)) return Number.NaN;
    const k = Math.floor(x);
    if (k < 1) return 1;
    if (this.p_param === 1) return 0;
    return Math.exp(k * Math.log1p(-this.p_param));
  }

  override isf(q: number): number {
    checkProbability(q);
    if (this.p_param === 1 || q === 1) return 1;
    if (q === 0) return Number.POSITIVE_INFINITY;
    // Closed form, then correct a possible off-by-one from rounding in the logarithms.
    let k = Math.max(1, Math.ceil(Math.log(q) / Math.log1p(-this.p_param)));
    if (k > 1 && this.sf(k - 1) <= q) k--;
    else if (this.sf(k) > q) k++;
    return k;
  }

  rvs(size = 1): number[] {
    return sample(size, () => {
      if (this.p_param === 1) return 1;
      return Math.max(1, Math.ceil(Math.log1p(-__random()) / Math.log1p(-this.p_param)));
    });
  }

  mean(): number {
    return 1 / this.p_param;
  }
  variance(): number {
    return (1 - this.p_param) / (this.p_param * this.p_param);
  }
}

// ---- Negative Binomial Distribution ----

class NegativeBinomialDist extends DiscreteBase {
  constructor(
    private r: number,
    private p_param: number
  ) {
    super();
    requirePositive("r", r);
    if (!(p_param > 0 && p_param <= 1))
      throw new InvalidParameterError("p must be in (0,1]", "p", p_param);
  }

  pmf(k: number): number {
    if (Number.isNaN(k)) return Number.NaN;
    if (!Number.isInteger(k) || k < 0) return 0;
    // p = 1: all mass at zero failures (k·log(0) would be 0·(-Inf) = NaN).
    if (this.p_param === 1) return k === 0 ? 1 : 0;
    // C(k + r - 1, k) p^r q^k = r / (k + r) * Binomial(r; k + r, p), in saddle-point form.
    const logProb = logBinomialRaw(this.r, k + this.r, this.p_param, 1 - this.p_param);
    return Math.exp(logProb) * (this.r / (k + this.r));
  }

  cdf(x: number): number {
    if (Number.isNaN(x)) return Number.NaN;
    const k = Math.floor(x);
    if (k < 0) return 0;
    // Beyond 1e290 the incomplete beta below overflows; the tail mass there is zero.
    if (k > 1e290) return 1;
    // P(X <= k) = I_p(r, k+1)
    return regularizedIncompleteBeta(this.r, k + 1, this.p_param);
  }

  ppf(p: number): number {
    checkProbability(p);
    return this.quantile(p, 1 - p);
  }

  override isf(q: number): number {
    checkProbability(q);
    return this.quantile(1 - q, q);
  }

  /** Quantile for the lower-tail probability `p` and its complement `q = 1 - p`. */
  private quantile(p: number, q: number): number {
    return discreteQuantile(
      (k) => this.cdf(k),
      (k) => this.sf(k),
      p,
      q,
      0,
      Number.POSITIVE_INFINITY,
      this.mean()
    );
  }

  sf(x: number): number {
    if (Number.isNaN(x)) return Number.NaN;
    const k = Math.floor(x);
    if (k < 0) return 1;
    if (k > 1e290) return 0;
    // P(X > k) = 1 - I_p(r, k+1). When that is not near 0 the complement of the direct
    // lower tail is exact enough, and it avoids the argument 1 - p (which has rounded away
    // the low bits of a small p) of the equivalent I_{1-p}(k+1, r).
    const lower = regularizedIncompleteBeta(this.r, k + 1, this.p_param);
    if (lower < 0.9) return 1 - lower;
    return regularizedIncompleteBeta(k + 1, this.r, 1 - this.p_param);
  }

  rvs(size = 1): number[] {
    // Gamma-Poisson mixture: X | λ ~ Poisson(λ) with λ ~ Gamma(r, scale = (1 - p) / p).
    // Exact, and O(1) per draw instead of one Bernoulli trial per success/failure.
    return sample(size, () => {
      if (this.p_param === 1) return 0;
      return samplePoisson((gammaSample(this.r) * (1 - this.p_param)) / this.p_param);
    });
  }

  mean(): number {
    return (this.r * (1 - this.p_param)) / this.p_param;
  }
  variance(): number {
    return (this.r * (1 - this.p_param)) / (this.p_param * this.p_param);
  }
}

// ---- Hypergeometric Distribution ----

class HypergeometricDist extends DiscreteBase {
  constructor(
    private N: number,
    private K: number,
    private nn: number
  ) {
    super();
    if (!Number.isInteger(N) || N < 0)
      throw new InvalidParameterError("N must be a non-negative integer", "N", N);
    if (!Number.isInteger(K) || K < 0 || K > N)
      throw new InvalidParameterError("K must be in [0, N]", "K", K);
    if (!Number.isInteger(nn) || nn < 0 || nn > N)
      throw new InvalidParameterError("n must be in [0, N]", "n", nn);
  }

  /** Smallest possible count of successes in the sample. */
  private get lowest(): number {
    return Math.max(0, this.nn + this.K - this.N);
  }

  /** Largest possible count of successes in the sample. */
  private get highest(): number {
    return Math.min(this.nn, this.K);
  }

  pmf(k: number): number {
    if (Number.isNaN(k)) return Number.NaN;
    if (!Number.isInteger(k)) return 0;
    if (k < this.lowest || k > this.highest) return 0;
    // Product of three binomial probabilities with success probability n / N (the
    // approach of R's dhyper): each factor is evaluated in saddle-point form.
    const { N, K, nn } = this;
    if (N === 0) return 1;
    const p = nn / N;
    const q = (N - nn) / N;
    const l1 = logBinomialRaw(k, K, p, q);
    const l2 = logBinomialRaw(nn - k, N - K, p, q);
    const l3 = logBinomialRaw(nn, N, p, q);
    return Math.exp(l1 + l2 - l3);
  }

  /**
   * Sum of pmf over the integers from `from` towards the nearest end of the support,
   * stepping by `step` (-1 downwards, +1 upwards) with the term ratio of consecutive
   * probabilities, until the terms are negligible. Valid on the side of the mode where
   * the terms decrease; the first term is evaluated directly, so tail sums keep their
   * relative precision.
   */
  private tailSum(from: number, step: 1 | -1): number {
    const { N, K, nn } = this;
    const stop = step === 1 ? this.highest : this.lowest;
    let term = this.pmf(from);
    let sum = term;
    for (let i = from; i !== stop && term > 0; i += step) {
      // pmf(i + 1) / pmf(i) = (K - i)(n - i) / ((i + 1)(N - K - n + i + 1)), and its
      // inverse taken at i - 1 gives the step from i down to i - 1.
      term *=
        step === 1
          ? ((K - i) * (nn - i)) / ((i + 1) * (N - K - nn + i + 1))
          : (i * (N - K - nn + i)) / ((K - i + 1) * (nn - i + 1));
      sum += term;
      if (term < sum * 1e-17) break;
    }
    return sum;
  }

  cdf(x: number): number {
    if (Number.isNaN(x)) return Number.NaN;
    const k = Math.floor(x);
    if (k < this.lowest) return 0;
    if (k >= this.highest) return 1;
    // Sum the tail on the side of the mode that k lies on, so the smaller of cdf and sf
    // is accumulated directly and the other is its complement.
    if (k < this.mean()) return Math.min(1, this.tailSum(k, -1));
    return Math.max(0, 1 - this.tailSum(k + 1, 1));
  }

  ppf(p: number): number {
    checkProbability(p);
    return this.quantile(p, 1 - p);
  }

  override isf(q: number): number {
    checkProbability(q);
    return this.quantile(1 - q, q);
  }

  /** Quantile for the lower-tail probability `p` and its complement `q = 1 - p`. */
  private quantile(p: number, q: number): number {
    return discreteQuantile(
      (k) => this.cdf(k),
      (k) => this.sf(k),
      p,
      q,
      this.lowest,
      this.highest,
      this.mean()
    );
  }

  sf(x: number): number {
    if (Number.isNaN(x)) return Number.NaN;
    const k = Math.floor(x);
    if (k < this.lowest) return 1;
    if (k >= this.highest) return 0;
    if (k < this.mean()) return Math.max(0, 1 - this.tailSum(k, -1));
    return Math.min(1, this.tailSum(k + 1, 1));
  }

  rvs(size = 1): number[] {
    return sample(size, () => {
      // Sequential sampling without replacement: each draw is a success with
      // probability (successes left) / (items left).
      let successes = 0;
      let remaining = this.N;
      let good = this.K;
      for (let i = 0; i < this.nn; i++) {
        if (__random() < good / remaining) {
          successes++;
          good--;
        }
        remaining--;
      }
      return successes;
    });
  }

  mean(): number {
    return this.N === 0 ? 0 : (this.nn * this.K) / this.N;
  }
  variance(): number {
    const { N, K, nn } = this;
    // A population of one (or none) has no spread; the formula would give 0/0.
    if (N <= 1) return 0;
    return (nn * K * (N - K) * (N - nn)) / (N * N * (N - 1));
  }
}

// ---- Factory Functions (continued) ----

/**
 * Geometric distribution (number of trials until first success).
 *
 * P(X = k) = p * (1-p)^(k-1) for k = 1, 2, ...
 *
 * Matches `scipy.stats.geom(p)`.
 *
 * @param p - Probability of success per trial, in (0, 1]
 * @returns Distribution object
 * @throws {InvalidParameterError} If `p` is not in (0, 1]
 *
 * @example
 * ```ts
 * geom(0.25).mean(); // 4
 * ```
 */
export function geom(p: number): DiscreteDistribution {
  return new GeometricDist(p);
}

/**
 * Negative binomial distribution (number of failures before r successes).
 *
 * Matches `scipy.stats.nbinom(r, p)`. Like SciPy, `r` may be any positive real number
 * (the gamma-Poisson mixture form); for an integer `r` it counts the failures before the
 * r-th success.
 *
 * @param r - Number of successes required, a finite number > 0
 * @param p - Probability of success per trial, in (0, 1]
 * @returns Distribution object
 * @throws {InvalidParameterError} If `r` is not a finite number > 0 or `p` is not in (0, 1]
 *
 * @example
 * ```ts
 * nbinom(3, 0.5).mean(); // 3
 * ```
 */
export function nbinom(r: number, p: number): DiscreteDistribution {
  return new NegativeBinomialDist(r, p);
}

/**
 * Hypergeometric distribution.
 *
 * Models drawing n items from a population of N containing K successes,
 * without replacement. Matches `scipy.stats.hypergeom(M=N, n=K, N=n)`: note that SciPy
 * names the population size `M`, the success count `n` and the number of draws `N`.
 *
 * @param N - Population size (non-negative integer)
 * @param K - Number of success states in population, in [0, N]
 * @param n - Number of draws, in [0, N]
 * @returns Distribution object
 * @throws {InvalidParameterError} If an argument is not an integer in its valid range
 *
 * @example
 * ```ts
 * hypergeom(52, 13, 5).pmf(2); // chance of two hearts in a five-card hand
 * ```
 */
export function hypergeom(N: number, K: number, n: number): DiscreteDistribution {
  return new HypergeometricDist(N, K, n);
}
