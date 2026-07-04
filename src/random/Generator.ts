/**
 * Independent random state object, similar to NumPy's Generator.
 *
 * Each Generator instance maintains its own internal PRNG state (PCG32),
 * independent of the global seed. This allows for reproducible random
 * streams in parallel or isolated contexts.
 *
 * @module random/Generator
 * @see {@link https://deepbox.dev/docs/random-generation | Deepbox documentation}
 */

import { InvalidParameterError } from "../core";
// PCG32 backed by the shared 32-bit limb implementation (bit-identical to
// the previous BigInt version, ~50x faster per sample).
import { __SeededRandom as PCG32 } from "./random";

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
  private rng: PCG32;
  private readonly _seed: number;

  /**
   * Create a new Generator with a given seed.
   *
   * @param seed - Any finite number. Coerced to uint64 for PRNG state.
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
    this.rng = new PCG32(seedUint64);
  }

  /**
   * Get the seed used to create this Generator.
   */
  get seed(): number {
    return this._seed;
  }

  /**
   * Generate a uniform random number in [0, 1).
   */
  random(): number {
    return this.rng.next();
  }

  /**
   * Generate an array of uniform random numbers in [0, 1).
   *
   * @param size - Number of samples
   */
  randomArray(size: number): Float64Array {
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
   */
  fillNormalInto(out: Float64Array, off: number, count: number): void {
    this.rng.fillNormal(out.subarray(off, off + count), count);
  }

  /** Fill `out[off .. off+count)` with uniform [0,1) samples from this stream. */
  fillUniformInto(out: Float64Array, off: number, count: number): void {
    this.rng.fillUniform01(out.subarray(off, off + count), count);
  }

  /**
   * Generate a sample from the standard normal distribution (mean=0, std=1).
   * Uses Box-Muller transform.
   */
  normal(mean = 0, std = 1): number {
    if (std < 0) {
      throw new InvalidParameterError("std must be >= 0", "std", std);
    }
    return mean + std * this.standardNormal();
  }

  /**
   * Generate an array of normal random numbers.
   *
   * @param mean - Mean of the distribution
   * @param std - Standard deviation
   * @param size - Number of samples
   */
  normalArray(mean = 0, std = 1, size = 1): Float64Array {
    if (std < 0) {
      throw new InvalidParameterError("std must be >= 0", "std", std);
    }
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
   * @param low - Lower bound (inclusive)
   * @param high - Upper bound (exclusive)
   */
  uniform(low = 0, high = 1): number {
    if (low >= high) {
      throw new InvalidParameterError(
        `low must be < high; got low=${low}, high=${high}`,
        "low",
        low
      );
    }
    return low + (high - low) * this.rng.next();
  }

  /**
   * Generate an array of uniform random numbers in [low, high).
   */
  uniformArray(low = 0, high = 1, size = 1): Float64Array {
    if (low >= high) {
      throw new InvalidParameterError(
        `low must be < high; got low=${low}, high=${high}`,
        "low",
        low
      );
    }
    const range = high - low;
    const out = new Float64Array(size);
    this.rng.fillUniform01(out, size);
    for (let i = 0; i < size; i++) {
      out[i] = low + range * (out[i] as number);
    }
    return out;
  }

  /**
   * Generate a random integer in [low, high).
   *
   * @param low - Lower bound (inclusive)
   * @param high - Upper bound (exclusive)
   */
  randint(low: number, high: number): number {
    if (!Number.isInteger(low) || !Number.isInteger(high)) {
      throw new InvalidParameterError("low and high must be integers", "low/high", 0);
    }
    if (low >= high) {
      throw new InvalidParameterError(
        `low must be < high; got low=${low}, high=${high}`,
        "low",
        low
      );
    }
    const range = high - low;
    return low + Math.floor(this.rng.next() * range);
  }

  /**
   * Generate an array of random integers in [low, high).
   */
  randintArray(low: number, high: number, size: number): Int32Array {
    if (!Number.isInteger(low) || !Number.isInteger(high)) {
      throw new InvalidParameterError("low and high must be integers", "low/high", 0);
    }
    if (low >= high) {
      throw new InvalidParameterError(
        `low must be < high; got low=${low}, high=${high}`,
        "low",
        low
      );
    }
    const range = high - low;
    const out = new Int32Array(size);
    // Draw uniforms in bulk, then map into the integer range in a tight loop.
    const u = new Float64Array(size);
    this.rng.fillUniform01(u, size);
    for (let i = 0; i < size; i++) {
      out[i] = low + Math.floor((u[i] as number) * range);
    }
    return out;
  }

  /**
   * Generate a sample from the exponential distribution.
   *
   * @param scale - Scale parameter (1/rate), must be > 0
   */
  exponential(scale = 1): number {
    if (scale <= 0) {
      throw new InvalidParameterError("scale must be > 0", "scale", scale);
    }
    let u = this.rng.next();
    while (u === 0) u = this.rng.next();
    return -scale * Math.log(u);
  }

  /**
   * Generate a Bernoulli trial (0 or 1).
   *
   * @param p - Probability of 1
   */
  bernoulli(p = 0.5): number {
    if (p < 0 || p > 1) {
      throw new InvalidParameterError("p must be in [0, 1]", "p", p);
    }
    return this.rng.next() < p ? 1 : 0;
  }

  /**
   * Randomly select an index from weighted probabilities.
   *
   * @param weights - Array of non-negative weights (need not sum to 1)
   * @returns Selected index
   */
  choice(weights: ArrayLike<number>): number {
    const n = weights.length;
    if (n === 0) {
      throw new InvalidParameterError("weights must be non-empty", "weights", 0);
    }

    let total = 0;
    for (let i = 0; i < n; i++) {
      total += weights[i] ?? 0;
    }
    if (total <= 0) {
      throw new InvalidParameterError("weights must sum to > 0", "weights", total);
    }

    let r = this.rng.next() * total;
    for (let i = 0; i < n; i++) {
      r -= weights[i] ?? 0;
      if (r <= 0) return i;
    }
    return n - 1;
  }

  /**
   * Shuffle an array in-place using Fisher-Yates algorithm.
   *
   * @param array - Array to shuffle
   */
  shuffle<T>(array: T[]): void {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(this.rng.next() * (i + 1));
      const tmp = array[i];
      array[i] = array[j] as T;
      array[j] = tmp as T;
    }
  }

  /**
   * Return a random permutation of integers [0, n).
   *
   * @param n - Number of elements
   */
  permutation(n: number): Int32Array {
    if (!Number.isInteger(n) || n < 0) {
      throw new InvalidParameterError("n must be a non-negative integer", "n", n);
    }
    const out = new Int32Array(n);
    for (let i = 0; i < n; i++) out[i] = i;
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(this.rng.next() * (i + 1));
      const tmp = out[i] ?? 0;
      out[i] = out[j] ?? 0;
      out[j] = tmp;
    }
    return out;
  }

  // ─── Internal helpers ───────────────────────────────────────────────

  private standardNormal(): number {
    return this.rng.nextNormal();
  }
}
