import { InvalidParameterError, NotImplementedError } from "../core";

/**
 * Internal global seed.
 *
 * Note: we store the original user-provided number (after validation) so
 * `getSeed()` can return what the user set.
 */
let __globalSeed: number | undefined;

type CryptoLike = {
  getRandomValues<T extends ArrayBufferView>(array: T): T;
};

declare const crypto: CryptoLike | undefined;

const UINT64_MASK = (1n << 64n) - 1n;
const PCG_MULT = 6364136223846793005n;
const SPLITMIX_GAMMA = 0x9e3779b97f4a7c15n;
const SPLITMIX_MIX1 = 0xbf58476d1ce4e5b9n;
const SPLITMIX_MIX2 = 0x94d049bb133111ebn;

class __SplitMix64 {
  private state: bigint;

  constructor(seed: bigint) {
    this.state = seed & UINT64_MASK;
  }

  next(): bigint {
    this.state = (this.state + SPLITMIX_GAMMA) & UINT64_MASK;
    let z = this.state;
    z = (z ^ (z >> 30n)) * SPLITMIX_MIX1;
    z &= UINT64_MASK;
    z = (z ^ (z >> 27n)) * SPLITMIX_MIX2;
    z &= UINT64_MASK;
    return z ^ (z >> 31n);
  }
}

/**
 * PCG32 PRNG with 64-bit state.
 *
 * High statistical quality, fast, and deterministic across platforms.
 * Not cryptographically secure.
 */
class __SeededRandom {
  /** Current uint64 state. */
  private state: bigint;
  /** Stream/sequence selector (must be odd). */
  private inc: bigint;

  /**
   * Create a new PRNG from a seed.
   *
   * @param seedUint64 - Seed coerced to uint64.
   */
  constructor(seedUint64: bigint) {
    const sm = new __SplitMix64(seedUint64);
    this.state = sm.next();
    this.inc = (sm.next() << 1n) | 1n;
  }

  /**
   * Generate the next uint32 sample.
   */
  nextUint32(): number {
    const oldstate = this.state;
    this.state = (oldstate * PCG_MULT + this.inc) & UINT64_MASK;

    // Output function: xorshift high bits, then rotate.
    const xorshifted = Number(((oldstate >> 18n) ^ oldstate) >> 27n) >>> 0;
    const rot = Number(oldstate >> 59n) & 31;
    return ((xorshifted >>> rot) | (xorshifted << (-rot & 31))) >>> 0;
  }

  /**
   * Generate the next uniform sample in [0, 1).
   */
  next(): number {
    return this.nextUint32() / 2 ** 32;
  }
}

/** Internal PRNG instance when a seed is set. */
let __rng: __SeededRandom | null = null;

/**
 * Set the global seed for all random operations.
 *
 * @param seed - Any finite number. It will be coerced to uint64 for the PRNG.
 */
export function __setSeed(seed: number): void {
  // Validate input.
  if (!Number.isFinite(seed)) {
    throw new InvalidParameterError(`seed must be a finite number; received ${seed}`, "seed", seed);
  }

  // Store the (validated) user value.
  __globalSeed = seed;

  // Coerce to uint64 for deterministic PRNG state.
  const seedUint64 = BigInt.asUintN(64, BigInt(Math.trunc(seed)));
  __rng = new __SeededRandom(seedUint64);
}

/**
 * Get the current global seed.
 */
export function __getSeed(): number | undefined {
  return __globalSeed;
}

/**
 * Clear the global seed and revert to cryptographically secure randomness.
 */
export function __clearSeed(): void {
  __globalSeed = undefined;
  __rng = null;
}

/**
 * Generate a uniform random number in [0, 1).
 *
 * Uses the seeded PRNG when a seed is set; otherwise uses a cryptographically
 * secure RNG via `crypto.getRandomValues`.
 */
function getCrypto(): CryptoLike | undefined {
  if (typeof crypto === "undefined") {
    return undefined;
  }
  if (typeof crypto.getRandomValues !== "function") {
    return undefined;
  }
  return crypto;
}

function randomUint32FromCrypto(): number {
  const crypto = getCrypto();
  if (!crypto) {
    throw new NotImplementedError(
      "Cryptographically secure randomness is unavailable in this environment. " +
        "Provide a seed for deterministic randomness."
    );
  }
  const buf = new Uint32Array(1);
  crypto.getRandomValues(buf);
  const value = buf[0];
  if (value === undefined) {
    throw new InvalidParameterError("Failed to read cryptographic randomness", "crypto", value);
  }
  return value >>> 0;
}

export function __random(): number {
  // Use deterministic PRNG if available.
  if (__rng) {
    return __rng.next();
  }

  // Use cryptographically secure randomness when unseeded.
  return randomUint32FromCrypto() / 2 ** 32;
}

/**
 * Generate a uniform uint32 random number in [0, 2^32).
 *
 * Uses the seeded PRNG when a seed is set; otherwise uses a cryptographically
 * secure RNG via `crypto.getRandomValues`.
 */
export function __randomUint32(): number {
  if (__rng) {
    return __rng.nextUint32();
  }
  return randomUint32FromCrypto();
}

/**
 * Generate a uniform integer in [0, 2^53).
 *
 * Uses two uint32 draws to build 53 bits of randomness.
 */
export function __randomUint53(): number {
  const hi = __randomUint32() >>> 5; // 27 bits
  const lo = __randomUint32() >>> 6; // 26 bits
  return hi * 2 ** 26 + lo;
}

/**
 * Sample from the standard normal distribution (mean 0, std 1).
 *
 * Uses Box-Muller transform.
 *
 * Important: `log(0)` is `-Infinity`, so we avoid `u1 === 0`.
 */
export function __normalRandom(): number {
  // Ensure u1 is in (0, 1] to avoid log(0).
  let u1 = __random();
  while (u1 === 0) {
    u1 = __random();
  }

  // u2 can be 0 safely.
  const u2 = __random();

  // Box–Muller.
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/**
 * Sample from Gamma(shape, 1) using Marsaglia and Tsang's method (2000).
 *
 * Requires `shape > 1/3` (so that `d = shape - 1/3 > 0`).
 * For shape < 1, callers should use the transformation:
 * `Gamma(shape) = Gamma(shape+1) * U^(1/shape)` where U ~ Uniform(0,1).
 */
export function __gammaLarge(shape: number): number {
  // Validate input.
  // Marsaglia-Tsang requires d = shape - 1/3 > 0, i.e. shape > 1/3.
  // Callers handle shape < 1 via Gamma(shape+1) * U^(1/shape).
  if (!Number.isFinite(shape) || shape <= 1 / 3) {
    throw new InvalidParameterError(
      "shape must be a finite number > 1/3 for Marsaglia-Tsang method",
      "shape",
      shape
    );
  }

  // Marsaglia-Tsang constants.
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);

  // Rejection sampling loop.
  while (true) {
    let x: number;
    let v: number;

    // Generate a candidate using a standard normal.
    do {
      x = __normalRandom();
      v = 1 + c * x;
    } while (v <= 0);

    // Cube v.
    v = v * v * v;
    const u = __random();

    // Quick acceptance.
    if (u < 1 - 0.0331 * x * x * x * x) {
      return d * v;
    }

    // Squeeze test.
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) {
      return d * v;
    }
  }
}
