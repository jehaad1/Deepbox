/**
 * Internal random number plumbing: the xoshiro128++ generator, the global
 * seed state, the unseeded crypto fallback and the Ziggurat normal sampler.
 *
 * @see {@link https://deepbox.dev/docs/random-generation | Deepbox documentation}
 */

import { InvalidParameterError } from "../core/errors/invalid_parameter";
import { NotImplementedError } from "../core/errors/not_implemented";

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

/** Largest float32 strictly below 1 (1 - 2^-24). */
const FLOAT32_BELOW_ONE = 0.99999994039535522;

const UINT64_MASK = (1n << 64n) - 1n;
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
 * xoshiro128++ PRNG (Blackman & Vigna 2019) with 128-bit state.
 *
 * High statistical quality (passes BigCrush), deterministic across
 * platforms, and ~3x faster per draw than the limb-arithmetic PCG32 it
 * replaced: the whole state transition is six 32-bit integer ops, which
 * V8 compiles to straight-line machine code. Not cryptographically secure.
 *
 * The public contract is determinism per seed within a Deepbox version;
 * sequences are not pinned to any external reference implementation.
 */
export class __SeededRandom {
  /**
   * xoshiro128++ state as int32 bit patterns. An Int32Array sidesteps V8's
   * object-field representation transitions, which made the same integer
   * ops ~7x slower when the state lived in plain class fields.
   */
  private readonly s = new Int32Array(4);

  /**
   * Create a new PRNG from a seed.
   *
   * @param seedUint64 - Seed coerced to uint64; expanded to the 128-bit
   * state with SplitMix64 (never all-zero).
   */
  constructor(seedUint64: bigint) {
    const sm = new __SplitMix64(seedUint64);
    const a = sm.next();
    const b = sm.next();
    const s = this.s;
    s[0] = Number(a & 0xffffffffn) | 0;
    s[1] = Number(a >> 32n) | 0;
    s[2] = Number(b & 0xffffffffn) | 0;
    s[3] = Number(b >> 32n) | 0;
    if ((s[0]! | s[1]! | s[2]! | s[3]!) === 0) {
      s[3] = 1;
    }
  }

  /**
   * Generate the next uint32 sample.
   */
  nextUint32(): number {
    const s = this.s;
    const s0 = s[0] as number;
    const s1 = s[1] as number;
    const s3 = s[3] as number;
    const sum = (s0 + s3) | 0;
    const result = (((sum << 7) | (sum >>> 25)) + s0) >>> 0;
    const t = (s1 << 9) | 0;
    let n2 = ((s[2] as number) ^ s0) | 0;
    const n3 = (s3 ^ s1) | 0;
    s[1] = (s1 ^ n2) | 0;
    s[0] = (s0 ^ n3) | 0;
    n2 = (n2 ^ t) | 0;
    s[2] = n2;
    s[3] = (n3 << 11) | (n3 >>> 21) | 0;
    return result;
  }

  /**
   * Fill `target[0..count)` with uniform samples in [0, 1).
   *
   * State lives in locals for the whole loop (~2.5x over per-call draws).
   */
  fillUniform01(target: Float64Array | Float32Array, count: number): void {
    const s = this.s;
    let a = s[0] as number;
    let b = s[1] as number;
    let c = s[2] as number;
    let d = s[3] as number;
    if (target instanceof Float32Array) {
      // A uint32 draw above 2^32 - 128 rounds up to exactly 1.0 in float32, which
      // would break the half-open [0, 1) contract (and make log(1 - u) infinite).
      for (let i = 0; i < count; i++) {
        const sum = (a + d) | 0;
        const result = (((sum << 7) | (sum >>> 25)) + a) >>> 0;
        const t = (b << 9) | 0;
        c = (c ^ a) | 0;
        d = (d ^ b) | 0;
        b = (b ^ c) | 0;
        a = (a ^ d) | 0;
        c = (c ^ t) | 0;
        d = (d << 11) | (d >>> 21) | 0;
        const f = Math.fround(result / 4294967296);
        target[i] = f < 1 ? f : FLOAT32_BELOW_ONE;
      }
    } else {
      for (let i = 0; i < count; i++) {
        const sum = (a + d) | 0;
        const result = (((sum << 7) | (sum >>> 25)) + a) >>> 0;
        const t = (b << 9) | 0;
        c = (c ^ a) | 0;
        d = (d ^ b) | 0;
        b = (b ^ c) | 0;
        a = (a ^ d) | 0;
        c = (c ^ t) | 0;
        d = (d << 11) | (d >>> 21) | 0;
        target[i] = result / 4294967296;
      }
    }
    s[0] = a;
    s[1] = b;
    s[2] = c;
    s[3] = d;
  }

  /**
   * Fill `target[0..count)` with uint32 samples (state in locals).
   */
  fillUint32(target: Uint32Array, count: number): void {
    const s = this.s;
    let a = s[0] as number;
    let b = s[1] as number;
    let c = s[2] as number;
    let d = s[3] as number;
    for (let i = 0; i < count; i++) {
      const sum = (a + d) | 0;
      target[i] = (((sum << 7) | (sum >>> 25)) + a) >>> 0;
      const t = (b << 9) | 0;
      c = (c ^ a) | 0;
      d = (d ^ b) | 0;
      b = (b ^ c) | 0;
      a = (a ^ d) | 0;
      c = (c ^ t) | 0;
      d = (d << 11) | (d >>> 21) | 0;
    }
    s[0] = a;
    s[1] = b;
    s[2] = c;
    s[3] = d;
  }

  /**
   * Generate the next uniform sample in [0, 1).
   *
   * The value is `uint32 / 2^32`, so it has a resolution of 2^-32.
   */
  next(): number {
    return this.nextUint32() / 2 ** 32;
  }

  /** Uniform in (0, 1) from this instance's stream, never 0 or 1, safe for log(). */
  private nextOpen01(): number {
    return (this.nextUint32() + 0.5) / 4294967296;
  }

  /**
   * Ziggurat rejection path bound to this instance's own stream (so the
   * generated sequence stays independent of the global RNG).
   */
  private normalTail(hz: number, iz: number): number {
    for (;;) {
      if (iz === 0) {
        let x: number;
        let y: number;
        do {
          x = -Math.log(this.nextOpen01()) * ZIG_INV_R;
          y = -Math.log(this.nextOpen01());
        } while (y + y < x * x);
        return hz > 0 ? ZIG_R + x : -(ZIG_R + x);
      }
      const x = hz * (zigWn[iz] as number);
      const fi = zigFn[iz] as number;
      if (fi + this.next() * ((zigFn[iz - 1] as number) - fi) < Math.exp(-0.5 * x * x)) {
        return x;
      }
      hz = this.nextUint32() | 0;
      iz = hz & 127;
      if (Math.abs(hz) < ((zigKn as Uint32Array)[iz] as number)) {
        return hz * (zigWn[iz] as number);
      }
    }
  }

  /**
   * Sample one standard-normal deviate via the Ziggurat method, consuming
   * this instance's stream. ~2.5x cheaper than Box–Muller (one uint32 draw
   * and a multiply on the ~99% accept path).
   */
  nextNormal(): number {
    const kn = zigKn ?? zigInit();
    const hz = this.nextUint32() | 0;
    const iz = hz & 127;
    return Math.abs(hz) < (kn[iz] as number) ? hz * (zigWn[iz] as number) : this.normalTail(hz, iz);
  }

  /**
   * Fill `target[0..count)` with standard-normal samples (Ziggurat).
   *
   * State transition lives in this module so the hot accept path inlines,
   * matching the free-function {@link __fillNormal} throughput.
   */
  fillNormal(target: Float64Array | Float32Array, count: number): void {
    const kn = zigKn ?? zigInit();
    const wn = zigWn;
    for (let i = 0; i < count; i++) {
      const hz = this.nextUint32() | 0;
      const iz = hz & 127;
      target[i] =
        Math.abs(hz) < (kn[iz] as number) ? hz * (wn[iz] as number) : this.normalTail(hz, iz);
    }
  }
}

/**
 * Convert a model-level seed to the uint64 that seeds {@link __SeededRandom}.
 *
 * Integer seeds (including negative ones, reduced modulo 2^64) map to themselves, so
 * established sequences do not change. A fractional seed uses its IEEE-754 bits instead of
 * being truncated, so 0.5 and 0.7 give different streams.
 *
 * @internal
 */
export function __seedToUint64(seed: number): bigint {
  if (Number.isInteger(seed)) return BigInt.asUintN(64, BigInt(seed));
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, seed);
  return view.getBigUint64(0);
}

/**
 * Draw an integer in `[0, bound)` from a generator of uniform values in `[0, 1)`.
 *
 * `floor(u * bound)` is slightly biased when `u` has a resolution of 2^-32, as the draws of
 * {@link __random} and {@link __SeededRandom.next} do: some residues occur once more often than
 * others. For integer bounds up to 2^21 this applies Lemire's rejection test to the exact
 * 32-bit word behind `u`, so every value is exactly equally likely. A draw is rejected with
 * probability below `bound / 2^32`, so seeded streams give the same values as `floor(u * bound)`
 * almost always. For larger bounds the 32-bit word is reduced with plain rejection sampling.
 * A generator whose values are not multiples of 2^-32, or a non-integer bound, falls back to
 * `floor(u * bound)`.
 *
 * @internal
 */
export function __randomBelow(random: () => number, bound: number): number {
  let x = random() * 4294967296;
  if (!Number.isInteger(bound) || bound < 1 || !Number.isInteger(x)) {
    return Math.floor((x / 4294967296) * bound);
  }
  if (bound <= 2097152) {
    let m = x * bound;
    let low = m % 4294967296;
    if (low < bound) {
      const threshold = 4294967296 % bound;
      while (low < threshold) {
        x = random() * 4294967296;
        m = x * bound;
        low = m % 4294967296;
      }
    }
    return Math.floor(m / 4294967296);
  }
  if (bound <= 4294967296) {
    const limit = Math.floor(4294967296 / bound) * bound;
    while (x >= limit) x = random() * 4294967296;
    return x % bound;
  }
  return Math.floor((x / 4294967296) * bound);
}

/** Internal PRNG instance when a seed is set. */
let __rng: __SeededRandom | null = null;

/**
 * Set the global seed for all random operations.
 *
 * @param seed - Any finite number. The fractional part is discarded and the
 *   result is reduced modulo 2^64 to form the PRNG seed.
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

/** Resolve the platform `crypto` object, or `undefined` when it cannot supply random bytes. */
function getCrypto(): CryptoLike | undefined {
  if (typeof crypto === "undefined") {
    return undefined;
  }
  if (typeof crypto.getRandomValues !== "function") {
    return undefined;
  }
  return crypto;
}

// Batched crypto randomness: one getRandomValues syscall per 4096 words
// instead of per sample (the per-call overhead made every unseeded sample
// ~1us, dominating randn and the dataset generators).
const CRYPTO_BATCH = 4096;
let cryptoBuf: Uint32Array | null = null;
let cryptoPos = CRYPTO_BATCH;

function randomUint32FromCrypto(): number {
  if (cryptoPos >= CRYPTO_BATCH) {
    const crypto = getCrypto();
    if (!crypto) {
      throw new NotImplementedError(
        "Cryptographically secure randomness is unavailable in this environment. " +
          "Provide a seed for deterministic randomness."
      );
    }
    if (!cryptoBuf) cryptoBuf = new Uint32Array(CRYPTO_BATCH);
    crypto.getRandomValues(cryptoBuf);
    cryptoPos = 0;
  }
  return (cryptoBuf as Uint32Array)[cryptoPos++]! >>> 0;
}

/**
 * Generate a uniform random number in [0, 1).
 *
 * Uses the seeded PRNG when a seed is set; otherwise uses a cryptographically
 * secure RNG via `crypto.getRandomValues`.
 */
export function __random(): number {
  // Use deterministic PRNG if available.
  if (__rng) {
    return __rng.next();
  }

  // Use cryptographically secure randomness when unseeded.
  return randomUint32FromCrypto() / 2 ** 32;
}

/**
 * Fill `target[0..count)` with uniform uint32 samples.
 *
 * Bulk variant of {@link __randomUint32}: the sampling loop lives in the
 * same module as the RNG state so it inlines (per-call module-boundary
 * overhead made large fills ~10x slower), and the unseeded path writes
 * crypto randomness directly into the target. Consumes the seeded stream
 * in exactly the same order as repeated `__randomUint32()` calls.
 */
export function __fillUint32(target: Uint32Array, count: number): void {
  if (__rng) {
    __rng.fillUint32(target, count);
    return;
  }
  const crypto = getCrypto();
  if (!crypto) {
    throw new NotImplementedError(
      "Cryptographically secure randomness is unavailable in this environment. " +
        "Provide a seed for deterministic randomness."
    );
  }
  // Node caps getRandomValues at 65536 bytes per call.
  const MAX_WORDS = 16384;
  for (let start = 0; start < count; start += MAX_WORDS) {
    crypto.getRandomValues(target.subarray(start, Math.min(count, start + MAX_WORDS)));
  }
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
 * Fill `target[0..count)` with uniform samples in [0, 1).
 *
 * Bulk variant of {@link __random}: the sampling loop lives in the same
 * module as the RNG state so the generator inlines (a per-call module
 * boundary costs ~2x on large fills).
 */
export function __fillUniform(target: Float64Array | Float32Array, count: number): void {
  if (__rng) {
    __rng.fillUniform01(target, count);
    return;
  }
  if (target instanceof Float32Array) {
    for (let i = 0; i < count; i++) {
      const f = Math.fround(randomUint32FromCrypto() / 4294967296);
      target[i] = f < 1 ? f : FLOAT32_BELOW_ONE;
    }
    return;
  }
  for (let i = 0; i < count; i++) {
    target[i] = randomUint32FromCrypto() / 4294967296;
  }
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

// ─── Ziggurat standard-normal sampler (Marsaglia & Tsang 2000, 128 layers) ──
// ~99% of samples cost one uint32 draw and one multiply; the Box–Muller
// sampler this replaces paid log+sqrt+cos on two draws for every sample.

const ZIG_R = 3.442619855899;
const ZIG_INV_R = 1 / ZIG_R;
let zigKn: Uint32Array | null = null;
let zigWn: Float64Array = new Float64Array(0);
let zigFn: Float64Array = new Float64Array(0);

function zigInit(): Uint32Array {
  const m1 = 2147483648.0; // 2^31
  const vn = 9.91256303526217e-3;
  const kn = new Uint32Array(128);
  const wn = new Float64Array(128);
  const fn = new Float64Array(128);
  let dn = ZIG_R;
  let tn = dn;
  const q = vn / Math.exp(-0.5 * dn * dn);
  kn[0] = Math.floor((dn / q) * m1);
  kn[1] = 0;
  wn[0] = q / m1;
  wn[127] = dn / m1;
  fn[0] = 1;
  fn[127] = Math.exp(-0.5 * dn * dn);
  for (let i = 126; i >= 1; i--) {
    dn = Math.sqrt(-2 * Math.log(vn / dn + Math.exp(-0.5 * dn * dn)));
    kn[i + 1] = Math.floor((dn / tn) * m1);
    tn = dn;
    fn[i] = Math.exp(-0.5 * dn * dn);
    wn[i] = dn / m1;
  }
  zigKn = kn;
  zigWn = wn;
  zigFn = fn;
  return kn;
}

/** Uniform in (0, 1), never 0 or 1, safe for log(). */
function uniformOpen(): number {
  return (__randomUint32() + 0.5) / 4294967296;
}

/** Rejection path for samples outside a layer's guaranteed-accept region. */
function zigFix(hz: number, iz: number): number {
  for (;;) {
    if (iz === 0) {
      // Tail beyond ZIG_R (Marsaglia's exponential-wrap method); always finite.
      let x: number;
      let y: number;
      do {
        x = -Math.log(uniformOpen()) * ZIG_INV_R;
        y = -Math.log(uniformOpen());
      } while (y + y < x * x);
      return hz > 0 ? ZIG_R + x : -(ZIG_R + x);
    }
    const x = hz * (zigWn[iz] as number);
    const fi = zigFn[iz] as number;
    if (fi + __random() * ((zigFn[iz - 1] as number) - fi) < Math.exp(-0.5 * x * x)) {
      return x;
    }
    hz = __randomUint32() | 0;
    iz = hz & 127;
    if (Math.abs(hz) < ((zigKn as Uint32Array)[iz] as number)) {
      return hz * (zigWn[iz] as number);
    }
  }
}

/**
 * Sample from the standard normal distribution (mean 0, std 1).
 *
 * Uses the Ziggurat method (Marsaglia & Tsang 2000). All values are finite
 * and deterministic when a seed is set.
 */
export function __normalRandom(): number {
  const kn = zigKn ?? zigInit();
  const hz = __randomUint32() | 0;
  const iz = hz & 127;
  return Math.abs(hz) < (kn[iz] as number) ? hz * (zigWn[iz] as number) : zigFix(hz, iz);
}

/**
 * Fill `target[0..count)` with standard-normal samples.
 *
 * Bulk variant of {@link __normalRandom}: the sampling loop lives in the
 * same module as the RNG state so the common accept path inlines.
 */
export function __fillNormal(target: Float64Array | Float32Array, count: number): void {
  const kn = zigKn ?? zigInit();
  const wn = zigWn;
  if (__rng) {
    const rng = __rng;
    for (let i = 0; i < count; i++) {
      const hz = rng.nextUint32() | 0;
      const iz = hz & 127;
      target[i] = Math.abs(hz) < (kn[iz] as number) ? hz * (wn[iz] as number) : zigFix(hz, iz);
    }
    return;
  }
  for (let i = 0; i < count; i++) {
    const hz = randomUint32FromCrypto() | 0;
    const iz = hz & 127;
    target[i] = Math.abs(hz) < (kn[iz] as number) ? hz * (wn[iz] as number) : zigFix(hz, iz);
  }
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
