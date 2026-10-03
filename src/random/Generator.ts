/**
 * Independent random state object, similar to NumPy's Generator.
 *
 * Each Generator instance maintains its own internal PRNG state (xoshiro128++),
 * independent of the global seed. This allows for reproducible random
 * streams in parallel or isolated contexts.
 *
 * @module random/Generator
 * @see {@link https://deepbox.dev/docs/random-generation | Deepbox documentation}
 */

import { InvalidParameterError } from "../core";
import { __SeededRandom as Prng } from "./random";

const UINT32_RANGE = 4294967296;
const UINT53_RANGE = 2 ** 53;
const INT32_MIN = -2147483648;
const INT32_MAX = 2147483647;

// Largest bound for which Lemire's multiply is exact in float64: the product
// `uint32 * bound` must stay below 2^53, so bound < 2^21.
const LEMIRE_MAX_BOUND = 2 ** 21;

function assertFinite(value: number, name: string): void {
  if (!Number.isFinite(value)) {
    throw new InvalidParameterError(
      `${name} must be a finite number; received ${value}`,
      name,
      value
    );
  }
}

function assertCount(value: number, name: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new InvalidParameterError(
      `${name} must be a non-negative integer; received ${value}`,
      name,
      value
    );
  }
}

function assertWindow(out: Float64Array, off: number, count: number): void {
  assertCount(off, "off");
  assertCount(count, "count");
  if (off + count > out.length) {
    throw new InvalidParameterError(
      `off + count (${off + count}) exceeds the output length ${out.length}`,
      "count",
      count
    );
  }
}

/**
 * Independent random number generator with its own state.
 *
 * Unlike the module-level functions (which share a global seed),
 * each Generator instance is fully independent.
 *
 * @example
 * ```ts
 * import { Generator } from 'deepbox/random';
 *
 * const rng1 = new Generator(42);
 * const rng2 = new Generator(42);
 *
 * // Both produce identical sequences
 * console.log(rng1.random()); // same as rng2.random()
 *
 * // Independent from global seed
 * import { setSeed, rand } from 'deepbox/random';
 * setSeed(99);
 * // rng1 is unaffected by global seed changes
 * ```
 */
export class Generator {
  private rng: Prng;
  private readonly _seed: number;

  /**
   * Create a new Generator with a given seed.
   *
   * @param seed - Any finite number. The fractional part is discarded and the
   *   result is reduced modulo 2^64, so seeds 1.2 and 1.9 give the same stream.
   * @throws {InvalidParameterError} If `seed` is NaN or infinite.
   */
  constructor(seed: number) {
    if (!Number.isFinite(seed)) {
      throw new InvalidParameterError(
        `seed must be a finite number; received ${seed}`,
        "seed",
        seed
      );
    }
    this._seed = seed;
    const seedUint64 = BigInt.asUintN(64, BigInt(Math.trunc(seed)));
    this.rng = new Prng(seedUint64);
  }

  /**
   * Get the seed used to create this Generator.
   */
  get seed(): number {
    return this._seed;
  }

  /**
   * Generate a uniform random number in [0, 1).
   *
   * The value is a multiple of 2^-32.
   */
  random(): number {
    return this.rng.next();
  }

  /**
   * Generate an array of uniform random numbers in [0, 1).
   *
   * @param size - Number of samples (non-negative integer)
   */
  randomArray(size: number): Float64Array {
    assertCount(size, "size");
    const out = new Float64Array(size);
    // Bulk fill keeps the PRNG state in locals inside the RNG module so the
    // generator inlines (~2.5x over per-element `next()` module-boundary calls).
    this.rng.fillUniform01(out, size);
    return out;
  }

  /**
   * Fill `out[off .. off+count)` with standard-normal samples (Ziggurat),
   * drawing from this generator's own stream. Used by the dataset generators
   * so their Gaussian noise runs through the fast sampler instead of the
   * Box-Muller fallback.
   *
   * @throws {InvalidParameterError} If the window does not fit inside `out`.
   */
  fillNormalInto(out: Float64Array, off: number, count: number): void {
    assertWindow(out, off, count);
    this.rng.fillNormal(out.subarray(off, off + count), count);
  }

  /**
   * Fill `out[off .. off+count)` with uniform [0,1) samples from this stream.
   *
   * @throws {InvalidParameterError} If the window does not fit inside `out`.
   */
  fillUniformInto(out: Float64Array, off: number, count: number): void {
    assertWindow(out, off, count);
    this.rng.fillUniform01(out.subarray(off, off + count), count);
  }

  /**
   * Generate a sample from the normal distribution (default: mean 0, std 1).
   *
   * Uses the Ziggurat method.
   *
   * @param mean - Mean of the distribution (finite)
   * @param std - Standard deviation (finite, >= 0)
   * @throws {InvalidParameterError} If `mean` or `std` is not finite or `std < 0`.
   */
  normal(mean = 0, std = 1): number {
    assertNormalParams(mean, std);
    return mean + std * this.rng.nextNormal();
  }

  /**
   * Generate an array of normal random numbers.
   *
   * @param mean - Mean of the distribution (finite)
   * @param std - Standard deviation (finite, >= 0)
   * @param size - Number of samples (non-negative integer)
   * @throws {InvalidParameterError} If a parameter is invalid.
   */
  normalArray(mean = 0, std = 1, size = 1): Float64Array {
    assertNormalParams(mean, std);
    assertCount(size, "size");
    const out = new Float64Array(size);
    // Ziggurat bulk fill (standard normals), then affine-transform in place.
    this.rng.fillNormal(out, size);
    if (mean !== 0 || std !== 1) {
      for (let i = 0; i < size; i++) {
        out[i] = mean + std * (out[i] as number);
      }
    }
    return out;
  }

  /**
   * Generate a uniform random number in [low, high).
   *
   * @param low - Lower bound (inclusive, finite)
   * @param high - Upper bound (exclusive, finite, greater than `low`)
   * @throws {InvalidParameterError} If the bounds are not finite or `low >= high`.
   */
  uniform(low = 0, high = 1): number {
    assertUniformBounds(low, high);
    const u = this.rng.next();
    return uniformAffine(low, high, u);
  }

  /**
   * Generate an array of uniform random numbers in [low, high).
   *
   * @param low - Lower bound (inclusive, finite)
   * @param high - Upper bound (exclusive, finite, greater than `low`)
   * @param size - Number of samples (non-negative integer)
   * @throws {InvalidParameterError} If a parameter is invalid.
   */
  uniformArray(low = 0, high = 1, size = 1): Float64Array {
    assertUniformBounds(low, high);
    assertCount(size, "size");
    const out = new Float64Array(size);
    this.rng.fillUniform01(out, size);
    const range = high - low;
    if (Number.isFinite(range)) {
      for (let i = 0; i < size; i++) {
        out[i] = low + range * (out[i] as number);
      }
    } else {
      for (let i = 0; i < size; i++) {
        out[i] = uniformAffine(low, high, out[i] as number);
      }
    }
    return out;
  }

  /**
   * Generate an unbiased random integer in [low, high).
   *
   * Ranges up to 2^53 - 1 are supported; the result is exactly uniform.
   *
   * @param low - Lower bound (inclusive, safe integer)
   * @param high - Upper bound (exclusive, safe integer, greater than `low`)
   * @throws {InvalidParameterError} If the bounds are not safe integers, `low >= high`,
   *   or `high - low` exceeds 2^53 - 1.
   */
  randint(low: number, high: number): number {
    const range = assertIntegerBounds(low, high);
    return low + this.below(range);
  }

  /**
   * Generate an array of unbiased random integers in [low, high).
   *
   * The result is an Int32Array, so both bounds must fit in int32
   * (`low >= -2^31`, `high <= 2^31`).
   *
   * @param low - Lower bound (inclusive)
   * @param high - Upper bound (exclusive)
   * @param size - Number of samples (non-negative integer)
   * @throws {InvalidParameterError} If a parameter is invalid or the bounds do not fit in int32.
   */
  randintArray(low: number, high: number, size: number): Int32Array {
    const range = assertIntegerBounds(low, high);
    assertCount(size, "size");
    if (low < INT32_MIN || high > INT32_MAX + 1) {
      throw new InvalidParameterError(
        `randintArray returns int32 values; low must be >= ${INT32_MIN} and high <= ${INT32_MAX + 1}`,
        "low/high",
        low < INT32_MIN ? low : high
      );
    }
    const out = new Int32Array(size);
    if (range <= LEMIRE_MAX_BOUND) {
      // Draw all words in bulk into the output buffer itself, then map each
      // word to its bounded integer in place (index i is read before written).
      const words = new Uint32Array(out.buffer, out.byteOffset, size);
      this.rng.fillUint32(words, size);
      const threshold = UINT32_RANGE % range;
      for (let i = 0; i < size; i++) {
        let m = (words[i] as number) * range;
        let l = m >>> 0;
        while (l < threshold) {
          m = this.rng.nextUint32() * range;
          l = m >>> 0;
        }
        out[i] = low + Math.floor(m / UINT32_RANGE);
      }
    } else {
      for (let i = 0; i < size; i++) {
        out[i] = low + this.below(range);
      }
    }
    return out;
  }

  /**
   * Generate a sample from the exponential distribution.
   *
   * @param scale - Scale parameter (1/rate), finite and > 0
   * @throws {InvalidParameterError} If `scale` is not a finite number > 0.
   */
  exponential(scale = 1): number {
    if (!Number.isFinite(scale) || scale <= 0) {
      throw new InvalidParameterError("scale must be a finite number > 0", "scale", scale);
    }
    let u = this.rng.next();
    while (u === 0) u = this.rng.next();
    return -scale * Math.log(u);
  }

  /**
   * Generate a Bernoulli trial (0 or 1).
   *
   * @param p - Probability of 1, in [0, 1]
   * @throws {InvalidParameterError} If `p` is not in [0, 1].
   */
  bernoulli(p = 0.5): number {
    if (!(p >= 0 && p <= 1)) {
      throw new InvalidParameterError("p must be in [0, 1]", "p", p);
    }
    return this.rng.next() < p ? 1 : 0;
  }

  /**
   * Randomly select an index from weighted probabilities.
   *
   * Zero-weight entries are never selected.
   *
   * @param weights - Finite, non-negative weights (need not sum to 1)
   * @returns Selected index
   * @throws {InvalidParameterError} If `weights` is empty, contains a negative or
   *   non-finite entry, or sums to zero.
   */
  choice(weights: ArrayLike<number>): number {
    const n = weights.length;
    if (n === 0) {
      throw new InvalidParameterError("weights must be non-empty", "weights", 0);
    }

    let total = 0;
    let lastPositive = -1;
    for (let i = 0; i < n; i++) {
      const w = weights[i] as number;
      if (!Number.isFinite(w) || w < 0) {
        throw new InvalidParameterError(
          `weights must be finite and non-negative; got ${w} at index ${i}`,
          "weights",
          w
        );
      }
      if (w > 0) lastPositive = i;
      total += w;
    }
    if (!(total > 0) || !Number.isFinite(total)) {
      throw new InvalidParameterError(
        "weights must sum to a positive finite value",
        "weights",
        total
      );
    }

    let r = this.rng.next() * total;
    for (let i = 0; i < n; i++) {
      const w = weights[i] as number;
      if (r < w) return i;
      r -= w;
    }
    // Rounding drift left r just past the last bucket: take the last non-zero weight.
    return lastPositive;
  }

  /**
   * Shuffle an array in-place using the Fisher-Yates algorithm.
   *
   * Accepts plain arrays as well as typed arrays.
   *
   * @param array - Array to shuffle
   * @throws {InvalidParameterError} If the array has more than 2^31 elements.
   */
  shuffle<T>(array: { [index: number]: T; readonly length: number }): void {
    const n = array.length;
    if (n < 2) return;
    const js = this.swapTargets(n);
    for (let i = n - 1; i > 0; i--) {
      const j = js[i] as number;
      const tmp = array[i] as T;
      array[i] = array[j] as T;
      array[j] = tmp;
    }
  }

  /**
   * Return a random permutation of integers [0, n).
   *
   * @param n - Number of elements (integer in [0, 2^31])
   * @throws {InvalidParameterError} If `n` is not an integer in [0, 2^31].
   */
  permutation(n: number): Int32Array {
    if (!Number.isInteger(n) || n < 0 || n > INT32_MAX + 1) {
      throw new InvalidParameterError(
        `n must be an integer in [0, ${INT32_MAX + 1}]; received ${n}`,
        "n",
        n
      );
    }
    const out = new Int32Array(n);
    for (let i = 0; i < n; i++) out[i] = i;
    if (n < 2) return out;
    const js = this.swapTargets(n);
    for (let i = n - 1; i > 0; i--) {
      const j = js[i] as number;
      const tmp = out[i] as number;
      out[i] = out[j] as number;
      out[j] = tmp;
    }
    return out;
  }

  // ─── Internal helpers ───────────────────────────────────────────────

  /**
   * Draw an exactly uniform integer in [0, bound) for an integer bound in
   * [1, 2^53]. Uses Lemire's multiply for small bounds (identical to
   * `floor(u * bound)` except that biased low words are rejected), rejection
   * on one uint32 word up to 2^32, and rejection on 53 bits beyond that.
   */
  private below(bound: number): number {
    const rng = this.rng;
    if (bound <= LEMIRE_MAX_BOUND) {
      let m = rng.nextUint32() * bound;
      let l = m >>> 0;
      if (l < bound) {
        const t = UINT32_RANGE % bound;
        while (l < t) {
          m = rng.nextUint32() * bound;
          l = m >>> 0;
        }
      }
      return Math.floor(m / UINT32_RANGE);
    }
    if (bound <= UINT32_RANGE) {
      const limit = Math.floor(UINT32_RANGE / bound) * bound;
      let v = rng.nextUint32();
      while (v >= limit) v = rng.nextUint32();
      return v % bound;
    }
    const limit = Math.floor(UINT53_RANGE / bound) * bound;
    let v = this.uint53();
    while (v >= limit) v = this.uint53();
    return v % bound;
  }

  private uint53(): number {
    const hi = this.rng.nextUint32() >>> 5; // 27 bits
    const lo = this.rng.nextUint32() >>> 6; // 26 bits
    return hi * 2 ** 26 + lo;
  }

  /**
   * Fisher-Yates swap targets: `js[i]` is an exactly uniform integer in
   * `[0, i]` for i >= 1 (`js[0]` is 0). Words are drawn in one bulk fill when
   * the bounds are small enough for Lemire's multiply.
   */
  private swapTargets(n: number): Int32Array {
    if (n > INT32_MAX + 1) {
      throw new InvalidParameterError(
        `cannot shuffle more than ${INT32_MAX + 1} elements; received length ${n}`,
        "array",
        n
      );
    }
    const js = new Int32Array(n);
    if (n > LEMIRE_MAX_BOUND) {
      for (let i = n - 1; i > 0; i--) js[i] = this.below(i + 1);
      return js;
    }
    const words = new Uint32Array(n);
    this.rng.fillUint32(words, n);
    let p = 0;
    for (let i = n - 1; i > 0; i--) {
      const s = i + 1;
      let m = (words[p++] as number) * s;
      let l = m >>> 0;
      if (l < s) {
        const t = UINT32_RANGE % s;
        while (l < t) {
          m = this.rng.nextUint32() * s;
          l = m >>> 0;
        }
      }
      js[i] = Math.floor(m / UINT32_RANGE);
    }
    return js;
  }
}

function assertNormalParams(mean: number, std: number): void {
  assertFinite(mean, "mean");
  assertFinite(std, "std");
  if (std < 0) {
    throw new InvalidParameterError("std must be >= 0", "std", std);
  }
}

function assertUniformBounds(low: number, high: number): void {
  assertFinite(low, "low");
  assertFinite(high, "high");
  if (low >= high) {
    throw new InvalidParameterError(`low must be < high; got low=${low}, high=${high}`, "low", low);
  }
}

/** Map u in [0, 1) onto [low, high), staying finite when `high - low` overflows. */
function uniformAffine(low: number, high: number, u: number): number {
  const range = high - low;
  return Number.isFinite(range) ? low + range * u : low * (1 - u) + high * u;
}

/** Validate integer bounds and return the (safe-integer) range `high - low`. */
function assertIntegerBounds(low: number, high: number): number {
  if (!Number.isSafeInteger(low) || !Number.isSafeInteger(high)) {
    throw new InvalidParameterError(
      `low and high must be safe integers; got low=${low}, high=${high}`,
      "low/high",
      Number.isSafeInteger(low) ? high : low
    );
  }
  if (low >= high) {
    throw new InvalidParameterError(`low must be < high; got low=${low}, high=${high}`, "low", low);
  }
  const range = high - low;
  if (!Number.isSafeInteger(range)) {
    throw new InvalidParameterError(
      `high - low must not exceed 2^53 - 1; got low=${low}, high=${high}`,
      "high",
      high
    );
  }
  return range;
}
