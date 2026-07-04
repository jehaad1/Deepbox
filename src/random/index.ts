export { Generator } from "./Generator";

import type { Device, DType, Shape } from "../core";
import {
  DeepboxError,
  DTypeError,
  getConfig,
  InvalidParameterError,
  isTypedArray,
  shapeToSize,
  validateDevice,
  validateShape,
} from "../core";
import { arange, type Tensor, Tensor as TensorClass, type TypedArray, tensor } from "../ndarray";
import {
  __clearSeed,
  __fillNormal,
  __fillUint32,
  __fillUniform,
  __gammaLarge,
  __getSeed,
  __normalRandom,
  __random,
  __randomUint32,
  __randomUint53,
  __setSeed,
} from "./random";

export type RandomOptions = {
  readonly dtype?: DType;
  readonly device?: Device;
};

type FloatDType = "float32" | "float64";
type IntegerDType = "int32" | "int64";
type FloatBuffer = Float32Array | Float64Array;
type IntegerBuffer = Int32Array | BigInt64Array;

const INT32_MIN = -2147483648;
const INT32_MAX = 2147483647;
const UINT32_RANGE = 2 ** 32;
const UINT53_RANGE = 2 ** 53;

function resolveDevice(device?: Device): Device {
  const resolved = device ?? getConfig().defaultDevice;
  return validateDevice(resolved, "device");
}

function resolveFloatDType(dtype: DType | undefined, functionName: string): FloatDType {
  const resolved = dtype ?? "float32";
  if (resolved !== "float32" && resolved !== "float64") {
    throw new DTypeError(`${functionName} only supports float32 or float64 dtype`);
  }
  return resolved;
}

function resolveIntegerDType(dtype: DType | undefined, functionName: string): IntegerDType {
  const resolved = dtype ?? "int32";
  if (resolved !== "int32" && resolved !== "int64") {
    throw new DTypeError(`${functionName} only supports int32 or int64 dtype`);
  }
  return resolved;
}

function assertSafeInteger(value: number, name: string): void {
  if (!Number.isFinite(value) || !Number.isInteger(value) || !Number.isSafeInteger(value)) {
    throw new InvalidParameterError(`${name} must be a safe integer`, name, value);
  }
}

function assertInt32Bounds(value: number, name: string): void {
  if (value < INT32_MIN || value > INT32_MAX) {
    throw new InvalidParameterError(
      `${name} must be within int32 range [${INT32_MIN}, ${INT32_MAX}]`,
      name,
      value
    );
  }
}

function randomIntBelow(maxExclusive: number): number {
  if (!Number.isSafeInteger(maxExclusive) || maxExclusive <= 0) {
    throw new InvalidParameterError("range must be a positive safe integer", "range", maxExclusive);
  }
  if (maxExclusive <= UINT32_RANGE) {
    const limit = Math.floor(UINT32_RANGE / maxExclusive) * maxExclusive;
    let value = __randomUint32();
    while (value >= limit) {
      value = __randomUint32();
    }
    return value % maxExclusive;
  }
  if (maxExclusive <= UINT53_RANGE) {
    const limit = Math.floor(UINT53_RANGE / maxExclusive) * maxExclusive;
    let value = __randomUint53();
    while (value >= limit) {
      value = __randomUint53();
    }
    return value % maxExclusive;
  }
  throw new InvalidParameterError(
    "range must be <= 2^53 for unbiased sampling",
    "range",
    maxExclusive
  );
}

// Largest bound for which Lemire's multiply is exact in float64: the product
// `uint32 * bound` must stay < 2^53, and uint32 < 2^32, so bound < 2^21.
const LEMIRE_MAX_BOUND = 1 << 21;

/**
 * Fill `js[from..0]` (descending) with Fisher–Yates swap targets: `js[i]` is an
 * unbiased integer in `[0, i]`. All random draws are produced in a single bulk
 * fill so the hot loop never crosses the RNG module boundary per element, and
 * Lemire's nearly-divisionless map replaces the per-draw modulo. `js[0]` is 0.
 *
 * Falls back to the general rejection sampler when `n` exceeds the range where
 * the 64-bit product would lose precision.
 */
function fisherYatesTargets(n: number): Int32Array {
  const js = new Int32Array(n);
  if (n < 2) return js;
  if (n > LEMIRE_MAX_BOUND) {
    for (let i = n - 1; i > 0; i--) js[i] = randomIntBelow(i + 1);
    return js;
  }
  const rnd = new Uint32Array(n);
  __fillUint32(rnd, n);
  let p = 0;
  for (let i = n - 1; i > 0; i--) {
    const s = i + 1;
    let m = (rnd[p++] as number) * s;
    let l = m >>> 0;
    if (l < s) {
      // Rare rejection region — compute the threshold once and resample.
      const t = UINT32_RANGE % s;
      while (l < t) {
        m = __randomUint32() * s;
        l = m >>> 0;
      }
    }
    js[i] = Math.floor(m / UINT32_RANGE);
  }
  return js;
}

/**
 * Fill `out[0..count)` with unbiased integers in `[0, bound)` using one bulk
 * RNG fill plus Lemire's map. Requires `bound < 2^21` (checked by the caller);
 * the reject threshold is constant across the whole fill.
 */
function fillBoundedInts(out: Int32Array, count: number, bound: number): void {
  const rnd = new Uint32Array(count);
  __fillUint32(rnd, count);
  const t = UINT32_RANGE % bound;
  for (let i = 0; i < count; i++) {
    let m = (rnd[i] as number) * bound;
    let l = m >>> 0;
    if (l < t) {
      do {
        m = __randomUint32() * bound;
        l = m >>> 0;
      } while (l < t);
    }
    out[i] = Math.floor(m / UINT32_RANGE);
  }
}

function allocateFloatBuffer(dtype: FloatDType, size: number): FloatBuffer {
  return dtype === "float32" ? new Float32Array(size) : new Float64Array(size);
}

function allocateIntegerBuffer(dtype: IntegerDType, size: number): IntegerBuffer {
  return dtype === "int64" ? new BigInt64Array(size) : new Int32Array(size);
}

function writeInteger(buffer: IntegerBuffer, index: number, value: number): void {
  if (buffer instanceof BigInt64Array) {
    buffer[index] = BigInt(value);
  } else {
    buffer[index] = value;
  }
}

function randomOpenUnit(): number {
  let u = __random();
  while (u === 0) {
    u = __random();
  }
  return u;
}

/**
 * Validate that a tensor is contiguous (no slicing/striding).
 * @param t - Tensor to validate
 * @param functionName - Name of the calling function for error messages
 */
function validateContiguous(t: Tensor, functionName: string): void {
  if (t.offset !== 0) {
    throw new InvalidParameterError(
      `${functionName} currently requires offset === 0`,
      "offset",
      t.offset
    );
  }
  for (let axis = 0; axis < t.ndim; axis++) {
    const expected = t.strides[axis];
    const tail = t.shape.slice(axis + 1).reduce((acc, v) => acc * v, 1);
    if (expected !== tail) {
      throw new InvalidParameterError(
        `${functionName} currently requires a contiguous tensor`,
        "strides",
        t.strides
      );
    }
  }
}

const LANCZOS_COEFFS = [
  676.5203681218851, -1259.1392167224028, 771.3234287776531, -176.6150291621406, 12.507343278686905,
  -0.13857109526572012, 0.000009984369578019572, 0.00000015056327351493116,
];

/**
 * Compute log Gamma(z) for z > 0 using Lanczos approximation.
 */
function logGamma(z: number): number {
  if (!Number.isFinite(z) || z <= 0) {
    throw new InvalidParameterError("logGamma requires a positive finite input", "z", z);
  }
  // Lanczos approximation with g=7, n=9 coefficients.
  let x = 0.99999999999980993;
  for (const [i, coeff] of LANCZOS_COEFFS.entries()) {
    x += coeff / (z + i);
  }
  const g = 7;
  const t = z + g - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z - 0.5) * Math.log(t) - t + Math.log(x);
}

/**
 * Compute log(n!) with high accuracy for all integer n >= 0.
 * For small n, use exact summation to avoid rounding error.
 */
function logFactorial(n: number): number {
  if (n <= 1) return 0;
  if (n <= 20) {
    // Exact computation for small n
    let result = 0;
    for (let i = 2; i <= n; i++) {
      result += Math.log(i);
    }
    return result;
  }
  // Use logGamma for stable, accurate results for large n.
  return logGamma(n + 1);
}

/**
 * Sample a single Poisson deviate. Knuth's product method underflows
 * (exp(-lambda) → 0) for lambda ≳ 745 and silently caps samples; the
 * transformed-rejection method (Ahrens & Dieter) is used for lambda ≥ 30.
 */
function samplePoissonScalar(lambda: number): number {
  if (!(lambda > 0)) return 0;
  if (lambda < 30) {
    const L = Math.exp(-lambda);
    let k = 0;
    let p = 1;
    for (;;) {
      p *= __random();
      if (p <= L) break;
      k++;
    }
    return k;
  }
  const c = 0.767 - 3.36 / lambda;
  const beta = Math.PI / Math.sqrt(3 * lambda);
  const alpha = beta * lambda;
  const kConst = Math.log(c) - lambda - Math.log(beta);
  for (;;) {
    const u = __random();
    if (u === 0 || u === 1) continue;
    const x = (alpha - Math.log((1 - u) / u)) / beta;
    const n = Math.floor(x + 0.5);
    if (n < 0 || !Number.isFinite(n)) continue;
    const v = __random();
    const y = alpha - beta * x;
    const lhs = y + Math.log(v / (1 + Math.exp(y)) ** 2);
    const rhs = kConst + n * Math.log(lambda) - logFactorial(n);
    if (lhs <= rhs) return n;
  }
}

function sampleGammaUnit(shape: number): number {
  if (shape < 1) {
    const u = randomOpenUnit();
    return __gammaLarge(shape + 1) * u ** (1 / shape);
  }
  return __gammaLarge(shape);
}

/**
 * Set global random seed.
 *
 * @param seed - Random seed value (any finite number). The seed is coerced to a uint64
 *               internally, so the same seed always produces the same sequence.
 *
 * @throws {InvalidParameterError} When seed is not finite (NaN or ±Infinity)
 *
 * @remarks
 * - Setting a seed makes all random operations deterministic and reproducible.
 * - The seed is truncated to uint64 range (0 to 2^64-1) for internal state.
 * - Use {@link getSeed} to retrieve the currently set seed.
 * - When no seed is set, random sampling uses a cryptographically secure RNG.
 *   Seeded mode is deterministic and **not** intended for cryptographic use.
 *
 * @example
 * ```js
 * import { setSeed, rand } from 'deepbox/random';
 *
 * setSeed(42);
 * const a = rand([5]);
 * setSeed(42);
 * const b = rand([5]);
 * // a and b contain identical values
 * ```
 */
export function setSeed(seed: number): void {
  __setSeed(seed);
}

/**
 * Get current random seed.
 *
 * @returns Current seed value or undefined if not set
 *
 * @example
 * ```js
 * import { setSeed, getSeed } from 'deepbox/random';
 *
 * setSeed(12345);
 * console.log(getSeed()); // 12345
 * ```
 */
export function getSeed(): number | undefined {
  return __getSeed();
}

/**
 * Clear the current random seed and revert to cryptographically secure randomness.
 *
 * @remarks
 * - After calling this, random sampling uses `crypto.getRandomValues`.
 * - Use this to leave deterministic mode after {@link setSeed}.
 *
 * @example
 * ```js
 * import { clearSeed, rand } from 'deepbox/random';
 *
 * clearSeed();
 * const x = rand([3]);  // cryptographically secure randomness
 * ```
 */
export function clearSeed(): void {
  __clearSeed();
}

/**
 * Random values in half-open interval [0, 1).
 *
 * @param shape - Output shape
 * @param opts - Options (dtype, device)
 *
 * @remarks
 * - Values are uniformly distributed in [0, 1) (inclusive lower, exclusive upper bound).
 * - Uses deterministic PRNG when seed is set via {@link setSeed}.
 * - Default dtype is float32; use float64 for higher precision.
 * - Only float32 and float64 dtypes are supported.
 *
 * @example
 * ```js
 * import { rand, setSeed } from 'deepbox/random';
 *
 * const x = rand([2, 3]);  // 2x3 matrix of random values
 *
 * // Deterministic generation
 * setSeed(42);
 * const a = rand([5]);
 * setSeed(42);
 * const b = rand([5]);
 * // a and b are identical
 * ```
 *
 * @see {@link https://deepbox.dev/docs/random-distributions | Deepbox Distributions}
 */
export function rand(shape: Shape, opts: RandomOptions = {}): Tensor {
  const size = shapeToSize(shape);
  const dtype = resolveFloatDType(opts.dtype, "rand");
  const data = allocateFloatBuffer(dtype, size);

  __fillUniform(data, size);

  return TensorClass.fromTypedArray({
    data,
    shape,
    dtype,
    device: resolveDevice(opts.device),
  });
}

/**
 * Random samples from standard normal distribution.
 *
 * @param shape - Output shape
 * @param opts - Options (dtype, device)
 *
 * @remarks
 * - Uses Box-Muller transform to generate normally distributed values.
 * - Mean = 0, standard deviation = 1.
 * - All values are finite (no infinities from tail behavior).
 * - Deterministic when seed is set via {@link setSeed}.
 * - Only float32 and float64 dtypes are supported.
 *
 * @example
 * ```js
 * import { randn } from 'deepbox/random';
 *
 * const x = randn([2, 3]);  // 2x3 matrix of normal random values
 * ```
 *
 * @see {@link https://deepbox.dev/docs/random-distributions | Deepbox Distributions}
 */
export function randn(shape: Shape, opts: RandomOptions = {}): Tensor {
  const size = shapeToSize(shape);
  const dtype = resolveFloatDType(opts.dtype, "randn");
  const data = allocateFloatBuffer(dtype, size);

  __fillNormal(data, size);

  return TensorClass.fromTypedArray({
    data,
    shape,
    dtype,
    device: resolveDevice(opts.device),
  });
}

/**
 * Random integers in half-open interval [low, high).
 *
 * @param low - Lowest integer (inclusive)
 * @param high - Highest integer (exclusive)
 * @param shape - Output shape
 * @param opts - Options (dtype, device)
 *
 * @throws {InvalidParameterError} When low or high is not finite
 * @throws {InvalidParameterError} When low or high is not an integer
 * @throws {InvalidParameterError} When high <= low
 *
 * @remarks
 * - Generates integers uniformly in [low, high) range.
 * - Both low and high must be safe integers (within ±2^53-1).
 * - dtype must be int32 or int64; int32 output requires bounds within int32 range.
 * - Deterministic when seed is set via {@link setSeed}.
 *
 * @example
 * ```js
 * import { randint } from 'deepbox/random';
 *
 * const x = randint(0, 10, [5]);  // 5 random integers from 0 to 9
 * ```
 *
 * @see {@link https://deepbox.dev/docs/random-distributions | Deepbox Distributions}
 */
const RANDINT_CHUNK = 4096;

function fillRandintNum(data: Int32Array, size: number, range: number, low: number): void {
  const limit = Math.floor(UINT32_RANGE / range) * range;
  const invRange = 1 / range;
  const scratch = new Uint32Array(Math.min(size, RANDINT_CHUNK));
  let produced = 0;
  // Draw in bulk; rejected samples (value >= limit) are simply skipped and
  // redrawn in the next chunk, consuming the seeded stream in exactly the
  // same order as per-sample rejection.
  while (produced < size) {
    const n = Math.min(size - produced, RANDINT_CHUNK);
    __fillUint32(scratch, n);
    for (let j = 0; j < n && produced < size; j++) {
      const value = scratch[j] as number;
      if (value < limit) {
        // value % range via reciprocal multiply — integer `%` compiles to
        // idiv (~20 cycles). The quotient can be off by one from float
        // rounding; the two fixups make the result exact (all intermediate
        // products are < 2^53, so f64 arithmetic on them is exact).
        let m = value - Math.floor(value * invRange) * range;
        if (m >= range) m -= range;
        else if (m < 0) m += range;
        data[produced++] = m + low;
      }
    }
  }
}

function fillRandintBig(data: BigInt64Array, size: number, range: number, low: number): void {
  const limit = Math.floor(UINT32_RANGE / range) * range;
  const hiMod = 2147483648 % range;
  const lowBig = BigInt(low);
  const scratch = new Uint32Array(Math.min(size, RANDINT_CHUNK));
  let produced = 0;
  while (produced < size) {
    const n = Math.min(size - produced, RANDINT_CHUNK);
    __fillUint32(scratch, n);
    for (let j = 0; j < n && produced < size; j++) {
      const value = scratch[j]! >>> 0;
      if (value < limit) {
        const m =
          value < 2147483648
            ? (value | 0) % range
            : ((((value - 2147483648) | 0) % range) + hiMod) % range;
        data[produced++] = BigInt(m) + lowBig;
      }
    }
  }
}

export function randint(low: number, high: number, shape: Shape, opts: RandomOptions = {}): Tensor {
  assertSafeInteger(low, "low");
  assertSafeInteger(high, "high");
  if (high <= low) {
    throw new InvalidParameterError("high must be > low", "high", high);
  }
  const dtype = resolveIntegerDType(opts.dtype, "randint");
  if (dtype === "int32") {
    assertInt32Bounds(low, "low");
    if (high > INT32_MAX + 1) {
      throw new InvalidParameterError(
        `high must be <= ${INT32_MAX + 1} for int32 output`,
        "high",
        high
      );
    }
  }
  const size = shapeToSize(shape);
  const data = allocateIntegerBuffer(dtype, size);
  const range = high - low;
  if (!Number.isSafeInteger(range) || range <= 0) {
    throw new InvalidParameterError("range must be a positive safe integer", "high", high);
  }

  // Hot loop, split into monomorphic helpers: mixing the BigInt branch into
  // this function prevents V8 from optimizing the int32 loop (~6x slower).
  if (range <= UINT32_RANGE) {
    if (data instanceof BigInt64Array) {
      fillRandintBig(data, size, range, low);
    } else {
      fillRandintNum(data, size, range, low);
    }
  } else {
    for (let i = 0; i < size; i++) {
      const sample = randomIntBelow(range) + low;
      writeInteger(data, i, sample);
    }
  }

  return TensorClass.fromTypedArray({
    data,
    shape,
    dtype,
    device: resolveDevice(opts.device),
  });
}

/**
 * Random samples from continuous uniform distribution.
 *
 * @param low - Lower boundary (default: 0)
 * @param high - Upper boundary (default: 1)
 * @param shape - Output shape
 * @param opts - Options
 *
 * @throws {InvalidParameterError} When low or high is not finite
 * @throws {InvalidParameterError} When high < low
 *
 * @remarks
 * - Values are uniformly distributed in [low, high).
 * - For very large ranges, floating-point precision may affect uniformity.
 * - Deterministic when seed is set via {@link setSeed}.
 * - Only float32 and float64 dtypes are supported.
 *
 * @example
 * ```js
 * import { uniform } from 'deepbox/random';
 *
 * const x = uniform(-1, 1, [3, 3]);  // Values between -1 and 1
 * ```
 *
 * @see {@link https://deepbox.dev/docs/random-distributions | Deepbox Distributions}
 */
export function uniform(
  low: number = 0,
  high: number = 1,
  shape: Shape = [],
  opts: RandomOptions = {}
): Tensor {
  if (!Number.isFinite(low) || !Number.isFinite(high)) {
    throw new InvalidParameterError("low and high must be finite", "low/high", {
      low,
      high,
    });
  }
  if (high < low) {
    throw new InvalidParameterError("high must be >= low", "high", high);
  }
  const size = shapeToSize(shape);
  const dtype = resolveFloatDType(opts.dtype, "uniform");
  const data = allocateFloatBuffer(dtype, size);
  const range = high - low;

  __fillUniform(data, size);
  for (let i = 0; i < size; i++) {
    data[i] = (data[i] as number) * range + low;
  }

  return TensorClass.fromTypedArray({
    data,
    shape,
    dtype,
    device: resolveDevice(opts.device),
  });
}

/**
 * Random samples from normal (Gaussian) distribution.
 *
 * @param mean - Mean of distribution (default: 0)
 * @param std - Standard deviation (default: 1)
 * @param shape - Output shape
 * @param opts - Options
 *
 * @throws {InvalidParameterError} When mean or std is not finite
 * @throws {InvalidParameterError} When std < 0
 *
 * @remarks
 * - Uses Box-Muller transform internally.
 * - All values are finite due to RNG resolution (no infinities from log(0)).
 * - std=0 produces constant values equal to mean.
 * - Deterministic when seed is set via {@link setSeed}.
 * - Only float32 and float64 dtypes are supported.
 *
 * @example
 * ```js
 * import { normal } from 'deepbox/random';
 *
 * const x = normal(0, 2, [100]);  // Mean 0, std 2
 * ```
 *
 * @see {@link https://deepbox.dev/docs/random-distributions | Deepbox Distributions}
 */
export function normal(
  mean: number = 0,
  std: number = 1,
  shape: Shape = [],
  opts: RandomOptions = {}
): Tensor {
  if (!Number.isFinite(mean) || !Number.isFinite(std)) {
    throw new InvalidParameterError("mean and std must be finite", "mean/std", {
      mean,
      std,
    });
  }
  if (std < 0) {
    throw new InvalidParameterError("std must be >= 0", "std", std);
  }
  const size = shapeToSize(shape);
  const dtype = resolveFloatDType(opts.dtype, "normal");
  const data = allocateFloatBuffer(dtype, size);

  __fillNormal(data, size);
  if (mean !== 0 || std !== 1) {
    for (let i = 0; i < size; i++) {
      data[i] = (data[i] as number) * std + mean;
    }
  }

  return TensorClass.fromTypedArray({
    data,
    shape,
    dtype,
    device: resolveDevice(opts.device),
  });
}

function binomialSmallMean(n: number, logQ: number): number {
  // Geometric waiting-time method: exact and efficient when mean is small.
  let successes = 0;
  let trials = 0;

  while (true) {
    const u = randomOpenUnit();
    const gap = Math.floor(Math.log(u) / logQ) + 1;
    if (gap > n - trials) {
      return successes;
    }
    trials += gap;
    successes++;
  }
}

function binomialChopDown(n: number, p: number, q: number, mode: number, pmfMode: number): number {
  const u = __random();
  let cumulative = pmfMode;
  if (u <= cumulative) {
    return mode;
  }

  let left = mode;
  let right = mode;
  let pmfLeft = pmfMode;
  let pmfRight = pmfMode;
  const ratioLeft = q / p;
  const ratioRight = p / q;

  while (left > 0 || right < n) {
    if (left > 0) {
      pmfLeft *= (left / (n - left + 1)) * ratioLeft;
      left -= 1;
      cumulative += pmfLeft;
      if (u <= cumulative) {
        return left;
      }
    }
    if (right < n) {
      pmfRight *= ((n - right) / (right + 1)) * ratioRight;
      right += 1;
      cumulative += pmfRight;
      if (u <= cumulative) {
        return right;
      }
    }
  }

  // Fallback: due to rounding, return the closest boundary.
  return u <= cumulative ? left : right;
}

/**
 * Random samples from binomial distribution.
 *
 * @param n - Number of trials (non-negative integer)
 * @param p - Probability of success (in [0, 1])
 * @param shape - Output shape
 * @param opts - Options
 *
 * @throws {InvalidParameterError} When n is not finite, not an integer, or < 0
 * @throws {InvalidParameterError} When p is not finite or not in [0, 1]
 *
 * @remarks
 * - Generates number of successes in n independent Bernoulli trials.
 * - Uses an exact geometric waiting-time method for small means and
 *   a mode-centered chop-down inversion for larger means.
 * - Results are in range [0, n].
 * - Deterministic when seed is set via {@link setSeed}.
 * - Only int32 and int64 dtypes are supported.
 *
 * @example
 * ```js
 * import { binomial } from 'deepbox/random';
 *
 * const x = binomial(10, 0.5, [100]);  // 10 coin flips, 100 times
 * ```
 *
 * @see {@link https://deepbox.dev/docs/random-distributions | Deepbox Distributions}
 */
export function binomial(
  n: number,
  p: number,
  shape: Shape = [],
  opts: RandomOptions = {}
): Tensor {
  assertSafeInteger(n, "n");
  if (n < 0) {
    throw new InvalidParameterError("n must be >= 0", "n", n);
  }
  if (!Number.isFinite(p) || p < 0 || p > 1) {
    throw new InvalidParameterError("p must be in [0, 1]", "p", p);
  }
  const size = shapeToSize(shape);
  const dtype = resolveIntegerDType(opts.dtype, "binomial");
  if (dtype === "int32") {
    if (n > INT32_MAX) {
      throw new InvalidParameterError(`n must be <= ${INT32_MAX} for int32 output`, "n", n);
    }
  }
  const data = allocateIntegerBuffer(dtype, size);
  if (n === 0) {
    return TensorClass.fromTypedArray({
      data,
      shape,
      dtype,
      device: resolveDevice(opts.device),
    });
  }
  if (p === 0) {
    return TensorClass.fromTypedArray({
      data,
      shape,
      dtype,
      device: resolveDevice(opts.device),
    });
  }
  if (p === 1) {
    for (let i = 0; i < size; i++) {
      writeInteger(data, i, n);
    }
    return TensorClass.fromTypedArray({
      data,
      shape,
      dtype,
      device: resolveDevice(opts.device),
    });
  }

  const flip = p > 0.5;
  const prob = flip ? 1 - p : p;
  const q = 1 - prob;
  if (q === 1) {
    const value = flip ? n : 0;
    for (let i = 0; i < size; i++) {
      writeInteger(data, i, value);
    }
    return TensorClass.fromTypedArray({
      data,
      shape,
      dtype,
      device: resolveDevice(opts.device),
    });
  }
  const mean = n * prob;
  const logQ = Math.log(q);

  if (mean < 10) {
    if (n <= 1024) {
      // Exact CDF inversion: build the cumulative table once, then draw all
      // uniforms in one bulk pass and invert. Replaces the geometric method's
      // ~mean module-boundary RNG calls and logs per sample with a single
      // uniform and a short linear scan (mode sits near 0 since prob <= 0.5).
      const cdf = new Float64Array(n + 1);
      const ratio = prob / q;
      let pmf = Math.exp(n * logQ); // q^n
      let cum = pmf;
      cdf[0] = cum;
      for (let k = 1; k <= n; k++) {
        pmf *= ((n - k + 1) / k) * ratio;
        cum += pmf;
        cdf[k] = cum;
      }
      cdf[n] = 1; // guard the tail against floating-point rounding
      const us = new Float64Array(size);
      __fillUniform(us, size);
      for (let i = 0; i < size; i++) {
        const u = us[i] as number;
        let k = 0;
        while (k < n && (cdf[k] as number) < u) k++;
        writeInteger(data, i, flip ? n - k : k);
      }
    } else {
      for (let i = 0; i < size; i++) {
        const sample = binomialSmallMean(n, logQ);
        writeInteger(data, i, flip ? n - sample : sample);
      }
    }
  } else {
    const mode = Math.floor((n + 1) * prob);
    const logP = Math.log(prob);
    const logPmfMode =
      logFactorial(n) -
      logFactorial(mode) -
      logFactorial(n - mode) +
      mode * logP +
      (n - mode) * logQ;
    const pmfMode = Math.exp(logPmfMode);
    if (!Number.isFinite(pmfMode) || pmfMode <= 0) {
      throw new InvalidParameterError("Failed to initialize binomial sampler", "p", p);
    }
    for (let i = 0; i < size; i++) {
      const sample = binomialChopDown(n, prob, q, mode, pmfMode);
      writeInteger(data, i, flip ? n - sample : sample);
    }
  }

  return TensorClass.fromTypedArray({
    data,
    shape,
    dtype,
    device: resolveDevice(opts.device),
  });
}

/**
 * Random samples from Poisson distribution.
 *
 * @param lambda - Expected number of events (rate, must be >= 0)
 * @param shape - Output shape
 * @param opts - Options
 *
 * @throws {InvalidParameterError} When lambda is not finite or < 0
 *
 * @remarks
 * - Uses Knuth's method for lambda < 30, transformed rejection for lambda >= 30.
 * - Stable and efficient for all lambda values (tested up to lambda=1000+).
 * - lambda=0 always produces 0.
 * - Deterministic when seed is set via {@link setSeed}.
 * - Only int32 and int64 dtypes are supported.
 *
 * @example
 * ```js
 * import { poisson } from 'deepbox/random';
 *
 * const x = poisson(5, [100]);  // Rate = 5 events
 * ```
 *
 * @see {@link https://deepbox.dev/docs/random-distributions | Deepbox Distributions}
 */
export function poisson(lambda: number, shape: Shape = [], opts: RandomOptions = {}): Tensor {
  if (!Number.isFinite(lambda) || lambda < 0) {
    throw new InvalidParameterError("lambda must be a finite number >= 0", "lambda", lambda);
  }
  const size = shapeToSize(shape);
  const dtype = resolveIntegerDType(opts.dtype, "poisson");
  if (dtype === "int32") {
    if (lambda > INT32_MAX) {
      throw new InvalidParameterError(
        `lambda must be <= ${INT32_MAX} for int32 output`,
        "lambda",
        lambda
      );
    }
  }
  const data = allocateIntegerBuffer(dtype, size);

  if (lambda < 30) {
    // Knuth's method for small lambda
    const L = Math.exp(-lambda);
    for (let i = 0; i < size; i++) {
      let k = 0;
      let p = 1;

      do {
        k++;
        p *= __random();
      } while (p > L);

      const sample = k - 1;
      if (!Number.isSafeInteger(sample)) {
        throw new InvalidParameterError(
          "poisson sample exceeds safe integer range",
          "lambda",
          lambda
        );
      }
      if (dtype === "int32" && sample > INT32_MAX) {
        throw new InvalidParameterError("poisson sample exceeds int32 range", "lambda", lambda);
      }
      writeInteger(data, i, sample);
    }
  } else {
    // Transformed rejection method for large lambda (Ahrens & Dieter)
    const c = 0.767 - 3.36 / lambda;
    const beta = Math.PI / Math.sqrt(3 * lambda);
    const alpha = beta * lambda;
    const k = Math.log(c) - lambda - Math.log(beta);

    for (let i = 0; i < size; i++) {
      while (true) {
        const u = __random();
        if (u === 0 || u === 1) continue;

        const x = (alpha - Math.log((1 - u) / u)) / beta;
        const n = Math.floor(x + 0.5);
        if (n < 0 || !Number.isFinite(n)) continue;

        const v = __random();
        const y = alpha - beta * x;
        const lhs = y + Math.log(v / (1 + Math.exp(y)) ** 2);
        const rhs = k + n * Math.log(lambda) - logFactorial(n);

        if (lhs <= rhs) {
          if (!Number.isSafeInteger(n)) {
            throw new InvalidParameterError(
              "poisson sample exceeds safe integer range",
              "lambda",
              lambda
            );
          }
          if (dtype === "int32" && n > INT32_MAX) {
            throw new InvalidParameterError("poisson sample exceeds int32 range", "lambda", lambda);
          }
          writeInteger(data, i, n);
          break;
        }
      }
    }
  }

  return TensorClass.fromTypedArray({
    data,
    shape,
    dtype,
    device: resolveDevice(opts.device),
  });
}

/**
 * Random samples from exponential distribution.
 *
 * @param scale - Scale parameter (1/lambda, default: 1, must be > 0)
 * @param shape - Output shape
 * @param opts - Options
 *
 * @throws {InvalidParameterError} When scale is not finite or <= 0
 *
 * @remarks
 * - Uses inverse transform sampling: -scale * log(U).
 * - All values are positive (u=0 is avoided to prevent infinities).
 * - Mean = scale, variance = scale^2.
 * - Deterministic when seed is set via {@link setSeed}.
 * - Only float32 and float64 dtypes are supported.
 *
 * @example
 * ```js
 * import { exponential } from 'deepbox/random';
 *
 * const x = exponential(2, [100]);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/random-distributions | Deepbox Distributions}
 */
export function exponential(
  scale: number = 1,
  shape: Shape = [],
  opts: RandomOptions = {}
): Tensor {
  if (!Number.isFinite(scale) || scale <= 0) {
    throw new InvalidParameterError("scale must be a finite number > 0", "scale", scale);
  }
  const size = shapeToSize(shape);
  const dtype = resolveFloatDType(opts.dtype, "exponential");
  const data = allocateFloatBuffer(dtype, size);

  // Bulk uniforms, then transform in place. 1-u maps [0,1) to (0,1], so the
  // log is always finite (same distribution as -log(u) on the open interval).
  __fillUniform(data, size);
  for (let i = 0; i < size; i++) {
    data[i] = -scale * Math.log(1 - (data[i] as number));
  }

  return TensorClass.fromTypedArray({
    data,
    shape,
    dtype,
    device: resolveDevice(opts.device),
  });
}

/**
 * Random samples from gamma distribution.
 *
 * @param shape_param - Shape parameter (k, must be > 0)
 * @param scale - Scale parameter (theta, default: 1, must be > 0)
 * @param shape - Output shape
 * @param opts - Options
 *
 * @throws {InvalidParameterError} When shape_param is not finite or <= 0
 * @throws {InvalidParameterError} When scale is not finite or <= 0
 *
 * @remarks
 * - Uses Marsaglia and Tsang's method (2000) for efficient sampling.
 * - All values are positive.
 * - Mean = shape_param * scale, variance = shape_param * scale^2.
 * - For shape_param < 1, uses a transformation to handle the case.
 * - Deterministic when seed is set via {@link setSeed}.
 * - Only float32 and float64 dtypes are supported.
 *
 * @example
 * ```js
 * import { gamma } from 'deepbox/random';
 *
 * const x = gamma(2, 2, [100]);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/random-distributions | Deepbox Distributions}
 */
export function gamma(
  shape_param: number,
  scale: number = 1,
  shape: Shape = [],
  opts: RandomOptions = {}
): Tensor {
  if (!Number.isFinite(shape_param) || shape_param <= 0) {
    throw new InvalidParameterError(
      "shape_param must be a finite number > 0",
      "shape_param",
      shape_param
    );
  }
  if (!Number.isFinite(scale) || scale <= 0) {
    throw new InvalidParameterError("scale must be a finite number > 0", "scale", scale);
  }
  const size = shapeToSize(shape);
  const dtype = resolveFloatDType(opts.dtype, "gamma");
  const data = allocateFloatBuffer(dtype, size);

  for (let i = 0; i < size; i++) {
    data[i] = sampleGammaUnit(shape_param) * scale;
  }

  return TensorClass.fromTypedArray({
    data,
    shape,
    dtype,
    device: resolveDevice(opts.device),
  });
}

/**
 * Random samples from beta distribution.
 *
 * @param alpha - Alpha parameter (must be > 0)
 * @param beta_param - Beta parameter (must be > 0)
 * @param shape - Output shape
 * @param opts - Options
 *
 * @throws {InvalidParameterError} When alpha is not finite or <= 0
 * @throws {InvalidParameterError} When beta_param is not finite or <= 0
 *
 * @remarks
 * - Uses ratio of two gamma distributions: X / (X + Y).
 * - All values are in the open interval (0, 1) up to floating-point rounding.
 * - Mean = alpha / (alpha + beta), useful for modeling proportions.
 * - Deterministic when seed is set via {@link setSeed}.
 * - Only float32 and float64 dtypes are supported.
 *
 * @example
 * ```js
 * import { beta } from 'deepbox/random';
 *
 * const x = beta(2, 5, [100]);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/random-distributions | Deepbox Distributions}
 */
export function beta(
  alpha: number,
  beta_param: number,
  shape: Shape = [],
  opts: RandomOptions = {}
): Tensor {
  if (!Number.isFinite(alpha) || alpha <= 0) {
    throw new InvalidParameterError("alpha must be a finite number > 0", "alpha", alpha);
  }
  if (!Number.isFinite(beta_param) || beta_param <= 0) {
    throw new InvalidParameterError("beta must be a finite number > 0", "beta_param", beta_param);
  }
  const size = shapeToSize(shape);
  const dtype = resolveFloatDType(opts.dtype, "beta");
  const data = allocateFloatBuffer(dtype, size);

  for (let i = 0; i < size; i++) {
    let sampled = false;
    for (let attempt = 0; attempt < 1024; attempt++) {
      const x = sampleGammaUnit(alpha);
      const y = sampleGammaUnit(beta_param);
      const sum = x + y;
      if (!Number.isFinite(sum) || sum <= 0) {
        continue;
      }
      const value = x / sum;
      if (Number.isFinite(value) && value >= 0 && value <= 1) {
        data[i] = value;
        sampled = true;
        break;
      }
    }
    if (!sampled) {
      throw new InvalidParameterError(
        "beta sampling failed to produce a finite sample",
        "alpha/beta_param",
        { alpha, beta_param }
      );
    }
  }

  return TensorClass.fromTypedArray({
    data,
    shape,
    dtype,
    device: resolveDevice(opts.device),
  });
}

function readNumericTensorValue(t: Tensor, index: number): number | bigint {
  if (t.dtype === "string") {
    throw new DTypeError("Expected numeric tensor");
  }
  const value = t.data[index];
  if (typeof value === "number" || typeof value === "bigint") {
    return value;
  }
  throw new InvalidParameterError("Internal error: tensor index out of bounds", "index", index);
}

function allocateNumericBuffer(dtype: DType, size: number): TypedArray {
  switch (dtype) {
    case "float16":
    case "bfloat16":
    case "float32":
    case "complex64":
      return new Float32Array(size);
    case "float64":
    case "complex128":
      return new Float64Array(size);
    case "int32":
      return new Int32Array(size);
    case "int64":
      return new BigInt64Array(size);
    case "uint8":
    case "bool":
      return new Uint8Array(size);
    case "string":
      throw new DTypeError("choice() does not support string tensors");
  }
}

function writeNumericValue(out: TypedArray, index: number, value: number | bigint): void {
  if (out instanceof BigInt64Array) {
    out[index] = typeof value === "bigint" ? value : BigInt(Math.trunc(value));
    return;
  }
  if (typeof value === "bigint") {
    out[index] = Number(value);
    return;
  }
  out[index] = value;
}

function buildNormalizedProbabilities(probabilities: Tensor, n: number): Float64Array {
  if (probabilities.dtype === "string") {
    throw new DTypeError("choice() probabilities must be numeric");
  }
  if (probabilities.ndim !== 1) {
    throw new InvalidParameterError("p must be a 1D tensor", "p", probabilities.shape);
  }
  if (probabilities.size !== n) {
    throw new InvalidParameterError(
      "p must have the same length as the population",
      "p",
      probabilities.size
    );
  }
  validateContiguous(probabilities, "choice(p)");

  const normalized = new Float64Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const value = Number(readNumericTensorValue(probabilities, i));
    if (!Number.isFinite(value) || value < 0) {
      throw new InvalidParameterError(
        "p must contain finite non-negative probabilities",
        "p",
        value
      );
    }
    normalized[i] = value;
    sum += value;
  }
  if (!Number.isFinite(sum) || sum <= 0) {
    throw new InvalidParameterError("sum(p) must be > 0 and finite", "p", sum);
  }
  for (let i = 0; i < n; i++) {
    normalized[i] = (normalized[i] ?? 0) / sum;
  }
  return normalized;
}

function sampleFromCdf(cdf: Float64Array): number {
  const u = randomOpenUnit();
  let left = 0;
  let right = cdf.length - 1;
  while (left < right) {
    const mid = Math.floor((left + right) / 2);
    const value = cdf[mid];
    if (value === undefined) {
      throw new InvalidParameterError("Internal error: invalid CDF index", "mid", mid);
    }
    if (u <= value) {
      right = mid;
    } else {
      left = mid + 1;
    }
  }
  return left;
}

/**
 * Random sample from array.
 *
 * @param a - Input array or integer (if integer, sample from arange(a))
 * @param size - Number of samples or output shape
 * @param replace - Whether to sample with replacement (default: true)
 * @param p - Optional probability weights for weighted sampling
 *
 * @throws {InvalidParameterError} When population size is invalid (not finite, not integer, or < 0)
 * @throws {InvalidParameterError} When size > population and replace is false
 * @throws {InvalidParameterError} When tensor is not contiguous (offset !== 0 or non-standard strides)
 * @throws {DTypeError} When input tensor has string dtype
 *
 * @remarks
 * - Input tensor must be contiguous (no slicing/striding).
 * - With replacement: can sample more elements than population size.
 * - Without replacement: size must be <= population size.
 * - Does NOT modify the input tensor (returns a new tensor).
 * - Deterministic when seed is set via {@link setSeed}.
 * - If `a` is a number, the population is `0..a-1` and output dtype is int32.
 * - Numeric populations are limited to `a <= 2^31` for int32 output.
 *
 * @example
 * ```js
 * import { choice, tensor } from 'deepbox/random';
 *
 * const x = tensor([1, 2, 3, 4, 5]);
 * const sample = choice(x, 3);  // Pick 3 elements with replacement
 *
 * // Without replacement
 * const unique = choice(x, 3, false);  // All different elements
 * ```
 *
 * @see {@link https://deepbox.dev/docs/random-distributions | Deepbox Distributions}
 */
export function choice(
  a: Tensor | number,
  size?: number | Shape,
  replace = true,
  p?: Tensor
): Tensor {
  if (typeof a === "number") {
    assertSafeInteger(a, "a");
    if (a < 0) {
      throw new InvalidParameterError("Invalid population size", "a", a);
    }
    if (a > INT32_MAX + 1) {
      throw new InvalidParameterError(
        `Population size must be <= ${INT32_MAX + 1} for choice()`,
        "a",
        a
      );
    }
  }

  const aa: Tensor = typeof a === "number" ? arange(0, a, 1, { dtype: "int32" }) : a;

  if (aa.dtype === "string") {
    throw new DTypeError("choice() does not support string tensors");
  }

  // Handle Tensor input: sample indices first, then gather values into a new tensor.
  // Note: we currently require contiguous storage, because `choice` is defined over
  // the flattened order. Using arbitrary strides would require computing a flat
  // index mapping.
  const n = aa.size;
  if (!Number.isInteger(n) || n < 0) {
    throw new InvalidParameterError("Invalid tensor size", "n", n);
  }
  if (n > INT32_MAX + 1) {
    throw new InvalidParameterError(`Population size must be <= ${INT32_MAX + 1}`, "n", n);
  }

  let outputSize: number;
  if (typeof size === "number") {
    outputSize = size;
  } else if (size) {
    validateShape(size);
    outputSize = shapeToSize(size);
  } else {
    outputSize = 1;
  }
  if (!Number.isSafeInteger(outputSize) || outputSize < 0) {
    throw new InvalidParameterError("size must be an integer >= 0", "size", size);
  }
  if (outputSize > INT32_MAX) {
    throw new InvalidParameterError(`size must be <= ${INT32_MAX}`, "size", outputSize);
  }
  if (n === 0 && outputSize > 0) {
    throw new InvalidParameterError("Cannot sample from an empty population", "a", a);
  }

  const indices = new Int32Array(outputSize);
  const weights = p ? buildNormalizedProbabilities(p, n) : undefined;

  if (weights) {
    if (replace) {
      const cdf = new Float64Array(weights.length);
      let cumulative = 0;
      for (let i = 0; i < weights.length; i++) {
        cumulative += weights[i] ?? 0;
        cdf[i] = cumulative;
      }
      cdf[cdf.length - 1] = 1;
      for (let i = 0; i < outputSize; i++) {
        indices[i] = sampleFromCdf(cdf);
      }
    } else {
      let nonZeroCount = 0;
      for (let i = 0; i < weights.length; i++) {
        if ((weights[i] ?? 0) > 0) nonZeroCount++;
      }
      if (outputSize > nonZeroCount) {
        throw new InvalidParameterError(
          "Cannot sample without replacement with zero-probability mass for requested size",
          "size",
          outputSize
        );
      }

      const remaining = new Float64Array(weights);
      let remainingMass = 1;
      for (let i = 0; i < outputSize; i++) {
        if (remainingMass <= 0) {
          throw new InvalidParameterError(
            "Insufficient probability mass to sample",
            "p",
            remainingMass
          );
        }
        const u = __random() * remainingMass;
        let cumulative = 0;
        let chosen = -1;
        for (let j = 0; j < remaining.length; j++) {
          const w = remaining[j] ?? 0;
          if (w <= 0) {
            continue;
          }
          cumulative += w;
          if (u <= cumulative) {
            chosen = j;
            break;
          }
        }
        if (chosen < 0) {
          for (let j = remaining.length - 1; j >= 0; j--) {
            if ((remaining[j] ?? 0) > 0) {
              chosen = j;
              break;
            }
          }
        }
        if (chosen < 0) {
          throw new InvalidParameterError("Failed to select weighted sample", "p", weights);
        }
        indices[i] = chosen;
        remainingMass -= remaining[chosen] ?? 0;
        remaining[chosen] = 0;
      }
    }
  } else if (replace) {
    // Uniform sampling with replacement: one batched RNG fill + Lemire map,
    // avoiding a per-draw module-boundary call and modulo.
    if (n <= LEMIRE_MAX_BOUND) {
      fillBoundedInts(indices, outputSize, n);
    } else {
      for (let i = 0; i < outputSize; i++) {
        indices[i] = randomIntBelow(n);
      }
    }
  } else {
    if (outputSize > n) {
      throw new InvalidParameterError(
        "Cannot sample without replacement when size > population",
        "size",
        outputSize
      );
    }
    // Partial Fisher–Yates over an index pool. Draw all bounds in bulk when the
    // population is small enough for Lemire's exact multiply.
    const pool = new Int32Array(n);
    for (let i = 0; i < n; i++) pool[i] = i;
    const useBatch = n <= LEMIRE_MAX_BOUND;
    const rnd = useBatch ? new Uint32Array(outputSize) : null;
    if (rnd) __fillUint32(rnd, outputSize);
    for (let i = 0; i < outputSize; i++) {
      let j: number;
      const bound = n - i;
      if (rnd) {
        let m = (rnd[i] as number) * bound;
        let l = m >>> 0;
        if (l < bound) {
          const t = UINT32_RANGE % bound;
          while (l < t) {
            m = __randomUint32() * bound;
            l = m >>> 0;
          }
        }
        j = Math.floor(m / UINT32_RANGE) + i;
      } else {
        j = randomIntBelow(bound) + i;
      }
      const poolJ = pool[j] as number;
      pool[j] = pool[i] as number;
      pool[i] = poolJ;
      indices[i] = poolJ;
    }
  }

  const outputShape: Shape = typeof size === "number" ? [size] : (size ?? [1]);

  // Require contiguous layout for correctness.
  validateContiguous(aa, "choice()");

  // Allocate output buffer in the same dtype/device.
  const out = allocateNumericBuffer(aa.dtype, outputSize);
  for (let i = 0; i < outputSize; i++) {
    const idx = indices[i];
    if (idx === undefined) {
      throw new InvalidParameterError("Internal error: undefined index", "indices", i);
    }
    const value = readNumericTensorValue(aa, idx);
    writeNumericValue(out, i, value);
  }

  return TensorClass.fromTypedArray({
    data: out,
    shape: outputShape,
    dtype: aa.dtype,
    device: aa.device,
  });
}

/**
 * Randomly shuffle array in-place.
 *
 * @param x - Input tensor (**MODIFIED IN-PLACE**)
 *
 * @throws {InvalidParameterError} When tensor is not contiguous (offset !== 0 or non-standard strides)
 * @throws {DTypeError} When input tensor has string dtype
 *
 * @remarks
 * - **WARNING: This function mutates the input tensor directly.**
 * - Uses Fisher-Yates shuffle algorithm (O(n) time, optimal).
 * - Input tensor must be contiguous (no slicing/striding).
 * - All elements are preserved, only their order changes.
 * - Deterministic when seed is set via {@link setSeed}.
 * - If you need a shuffled copy without mutation, use {@link permutation} instead.
 *
 * @example
 * ```js
 * import { shuffle, tensor } from 'deepbox/random';
 *
 * const x = tensor([1, 2, 3, 4, 5]);
 * shuffle(x);  // x is now shuffled IN-PLACE
 * console.log(x);  // e.g., [3, 1, 5, 2, 4]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/random-distributions | Deepbox Distributions}
 */
export function shuffle(x: Tensor): void {
  if (x.dtype === "string") {
    throw new DTypeError("shuffle() does not support string tensors");
  }
  // For correctness, only allow shuffling of a contiguous tensor with offset 0.
  // This ensures swapping elements maps to the logical flattened order.
  validateContiguous(x, "shuffle()");

  const data = x.data;
  if (!isTypedArray(data)) {
    throw new DTypeError("shuffle() does not support string tensors");
  }
  const n = data.length;

  // Fisher–Yates shuffle. All swap targets are drawn in one batched pass so the
  // per-element RNG cost is a table lookup rather than a module-boundary call.
  const js = fisherYatesTargets(n);

  // Split into two branches to maintain type safety without assertions.
  if (data instanceof BigInt64Array) {
    for (let i = n - 1; i > 0; i--) {
      const j = js[i] as number;
      const temp = data[i];
      const swap = data[j];
      if (temp === undefined || swap === undefined) {
        throw new DeepboxError("Internal error: shuffle index out of bounds");
      }
      data[i] = swap;
      data[j] = temp;
    }
  } else {
    for (let i = n - 1; i > 0; i--) {
      const j = js[i] as number;
      const temp = data[i];
      const swap = data[j];
      if (temp === undefined || swap === undefined) {
        throw new DeepboxError("Internal error: shuffle index out of bounds");
      }
      data[i] = swap;
      data[j] = temp;
    }
  }
}

/**
 * Return random permutation of array.
 *
 * @param x - Input tensor or integer
 *
 * @throws {DTypeError} When input tensor has string dtype
 *
 * @remarks
 * - Returns a NEW tensor (does NOT modify input).
 * - If x is an integer, returns permutation of arange(x).
 * - If x is a tensor, returns a shuffled copy with the same shape.
 * - Tensor inputs must be contiguous (no slicing/striding).
 * - Uses Fisher-Yates shuffle algorithm internally.
 * - Deterministic when seed is set via {@link setSeed}.
 * - Numeric input is limited to `x <= 2^31` for int32 output.
 *
 * @example
 * ```js
 * import { permutation, tensor } from 'deepbox/random';
 *
 * // Permutation of integers
 * const x = permutation(10);  // Random permutation of [0...9]
 *
 * // Permutation of tensor (does not modify original)
 * const original = tensor([1, 2, 3, 4, 5]);
 * const shuffled = permutation(original);
 * // original is unchanged
 * ```
 *
 * @see {@link https://deepbox.dev/docs/random-distributions | Deepbox Distributions}
 */
export function permutation(x: Tensor | number): Tensor {
  if (typeof x === "number") {
    assertSafeInteger(x, "x");
    const n = x;
    if (n < 0) {
      throw new InvalidParameterError("x must be a non-negative integer", "x", x);
    }
    if (n > INT32_MAX + 1) {
      throw new InvalidParameterError(`x must be <= ${INT32_MAX + 1} for int32 output`, "x", x);
    }
    const indices = new Int32Array(n);
    for (let i = 0; i < n; i++) indices[i] = i;

    // Batched Fisher–Yates: swap targets are precomputed in one RNG pass.
    const js = fisherYatesTargets(n);
    for (let i = n - 1; i > 0; i--) {
      const j = js[i] as number;
      const tmp = indices[i] as number;
      indices[i] = indices[j] as number;
      indices[j] = tmp;
    }

    return TensorClass.fromTypedArray({
      data: indices,
      shape: [n],
      dtype: "int32",
      device: resolveDevice(),
    });
  }

  if (x.dtype === "string") {
    throw new DTypeError("permutation() does not support string tensors");
  }

  validateContiguous(x, "permutation()");
  const data = x.data;
  if (!isTypedArray(data)) {
    throw new DTypeError("permutation() does not support string tensors");
  }
  const copy = TensorClass.fromTypedArray({
    data: data.slice(),
    shape: x.shape,
    dtype: x.dtype,
    device: x.device,
  });
  shuffle(copy);
  return copy;
}

/**
 * Draw samples from a multinomial distribution.
 *
 * @param n - Number of trials
 * @param pvals - Probabilities of each outcome (must sum to 1), 1D tensor of shape (k,)
 * @param size - Number of samples to draw (default: 1)
 * @returns Tensor of shape (size, k) with counts for each outcome
 *
 * @example
 * ```ts
 * import { multinomial } from 'deepbox/random';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const probs = tensor([0.2, 0.5, 0.3]);
 * const samples = multinomial(10, probs, 5); // shape [5, 3]
 * ```
 */
export function multinomial(n: number, pvals: Tensor, size = 1): Tensor {
  if (!Number.isInteger(n) || n < 0) {
    throw new InvalidParameterError("n must be a non-negative integer", "n", n);
  }
  if (pvals.ndim !== 1) {
    throw new InvalidParameterError(
      `pvals must be 1D; got ndim=${pvals.ndim}`,
      "pvals",
      pvals.shape
    );
  }
  const k = pvals.size;
  const probs: number[] = [];
  let pSum = 0;
  for (let i = 0; i < k; i++) {
    const p = Number(pvals.data[pvals.offset + i]);
    probs.push(p);
    pSum += p;
  }
  // Normalize
  if (Math.abs(pSum - 1) > 1e-6) {
    for (let i = 0; i < k; i++) {
      probs[i] = (probs[i] ?? 0) / pSum;
    }
  }

  const result: number[] = [];
  for (let s = 0; s < size; s++) {
    const counts = new Array<number>(k).fill(0);
    for (let trial = 0; trial < n; trial++) {
      const u = __random();
      let cumSum = 0;
      for (let i = 0; i < k; i++) {
        cumSum += probs[i] ?? 0;
        if (u < cumSum) {
          counts[i] = (counts[i] ?? 0) + 1;
          break;
        }
      }
      // Edge case: if rounding puts us past all probs, assign to last
      if (counts.reduce((a, b) => a + b, 0) < trial + 1) {
        counts[k - 1] = (counts[k - 1] ?? 0) + 1;
      }
    }
    result.push(...counts);
  }

  return tensor(result).reshape([size, k]);
}

/**
 * Draw samples from a multivariate normal distribution.
 *
 * Uses Cholesky decomposition of the covariance matrix.
 *
 * @param mean - Mean vector of shape (d,)
 * @param cov - Covariance matrix of shape (d, d), must be symmetric positive semi-definite
 * @param size - Number of samples (default: 1)
 * @returns Tensor of shape (size, d)
 *
 * @example
 * ```ts
 * import { multivariate_normal } from 'deepbox/random';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const mean = tensor([0, 0]);
 * const cov = tensor([[1, 0.5], [0.5, 1]]);
 * const samples = multivariate_normal(mean, cov, 100); // shape [100, 2]
 * ```
 */
export function multivariate_normal(mean: Tensor, cov: Tensor, size = 1): Tensor {
  if (mean.ndim !== 1) {
    throw new InvalidParameterError(`mean must be 1D; got ndim=${mean.ndim}`, "mean", mean.shape);
  }
  if (cov.ndim !== 2) {
    throw new InvalidParameterError(`cov must be 2D; got ndim=${cov.ndim}`, "cov", cov.shape);
  }
  const d = mean.size;
  if ((cov.shape[0] ?? 0) !== d || (cov.shape[1] ?? 0) !== d) {
    throw new InvalidParameterError(
      `cov must be (${d}, ${d}); got [${cov.shape.join(", ")}]`,
      "cov",
      cov.shape
    );
  }

  // Extract mean and covariance
  const mu: number[] = [];
  for (let i = 0; i < d; i++) {
    mu.push(Number(mean.data[mean.offset + i]));
  }
  const C: number[][] = [];
  for (let i = 0; i < d; i++) {
    const row: number[] = [];
    for (let j = 0; j < d; j++) {
      row.push(Number(cov.data[cov.offset + i * d + j]));
    }
    C.push(row);
  }

  // Cholesky decomposition: C = L * L^T
  const L: number[][] = Array.from({ length: d }, () => new Array<number>(d).fill(0));
  for (let i = 0; i < d; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0;
      for (let k = 0; k < j; k++) {
        sum += L[i]![k]! * L[j]![k]!;
      }
      if (i === j) {
        const val = C[i]![i]! - sum;
        L[i]![j] = val >= 0 ? Math.sqrt(val) : 0;
      } else {
        const diag = L[j]![j]!;
        L[i]![j] = diag > 0 ? (C[i]![j]! - sum) / diag : 0;
      }
    }
  }

  // Generate samples: x = mu + L * z where z ~ N(0, I)
  const result: number[] = [];
  for (let s = 0; s < size; s++) {
    // Generate standard normal vector
    const z: number[] = [];
    for (let i = 0; i < d; i++) {
      z.push(__normalRandom());
    }
    // x = mu + L * z
    for (let i = 0; i < d; i++) {
      let val = mu[i]!;
      for (let j = 0; j <= i; j++) {
        val += L[i]![j]! * z[j]!;
      }
      result.push(val);
    }
  }

  return tensor(result).reshape([size, d]);
}

/**
 * Draw samples from a Dirichlet distribution.
 *
 * @param alpha - Concentration parameters of shape (k,), all must be positive
 * @param size - Number of samples (default: 1)
 * @returns Tensor of shape (size, k) where each row sums to 1
 *
 * @example
 * ```ts
 * import { dirichlet } from 'deepbox/random';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const alpha = tensor([1, 1, 1]);
 * const samples = dirichlet(alpha, 5); // shape [5, 3], each row sums to 1
 * ```
 */
export function dirichlet(alpha: Tensor, size = 1): Tensor {
  if (alpha.ndim !== 1) {
    throw new InvalidParameterError(
      `alpha must be 1D; got ndim=${alpha.ndim}`,
      "alpha",
      alpha.shape
    );
  }
  const k = alpha.size;
  const alphaArr: number[] = [];
  for (let i = 0; i < k; i++) {
    const a = Number(alpha.data[alpha.offset + i]);
    if (a <= 0) {
      throw new InvalidParameterError(
        `All alpha values must be > 0; got ${a} at index ${i}`,
        "alpha",
        a
      );
    }
    alphaArr.push(a);
  }

  const result: number[] = [];
  for (let s = 0; s < size; s++) {
    // Sample from Gamma(alpha_i, 1) for each component
    const gammas: number[] = [];
    let gammaSum = 0;
    for (let i = 0; i < k; i++) {
      const g = sampleGamma(alphaArr[i]!, 1);
      gammas.push(g);
      gammaSum += g;
    }
    // Normalize
    for (let i = 0; i < k; i++) {
      result.push(gammaSum > 0 ? gammas[i]! / gammaSum : 1 / k);
    }
  }

  return tensor(result).reshape([size, k]);
}

/**
 * Sample from a categorical distribution.
 *
 * Draws samples from a categorical distribution defined by unnormalized
 * log-probabilities or probabilities.
 *
 * @param probs - 1D tensor of probabilities (will be normalized)
 * @param numSamples - Number of samples to draw (default: 1)
 * @param replacement - Whether to sample with replacement (default: true)
 * @returns 1D int32 tensor of sampled indices
 */
export function categorical(
  probs: Tensor,
  numSamples: number = 1,
  replacement: boolean = true
): Tensor {
  if (probs.ndim !== 1) {
    throw new InvalidParameterError(
      "categorical requires a 1D probability tensor",
      "probs",
      probs.ndim
    );
  }
  if (numSamples < 1 || !Number.isInteger(numSamples)) {
    throw new InvalidParameterError(
      "numSamples must be a positive integer",
      "numSamples",
      numSamples
    );
  }
  const k = probs.size;
  if (k === 0) {
    throw new InvalidParameterError("categorical requires at least one category", "probs", k);
  }

  // Normalize probabilities
  const p: number[] = [];
  let total = 0;
  for (let i = 0; i < k; i++) {
    const v = Number(probs.data[probs.offset + i]);
    if (v < 0 || !Number.isFinite(v)) {
      throw new InvalidParameterError("probs must contain non-negative finite values", "probs", v);
    }
    p.push(v);
    total += v;
  }
  if (total <= 0) {
    throw new InvalidParameterError("probs must sum to a positive value", "probs", total);
  }
  for (let i = 0; i < k; i++) p[i] = p[i]! / total;

  if (!replacement && numSamples > k) {
    throw new InvalidParameterError(
      `Cannot draw ${numSamples} samples without replacement from ${k} categories`,
      "numSamples",
      numSamples
    );
  }

  // Build CDF for sampling
  const cdf: number[] = [p[0]!];
  for (let i = 1; i < k; i++) cdf.push(cdf[i - 1]! + p[i]!);
  cdf[k - 1] = 1.0; // Ensure no floating-point gap

  const result = new Int32Array(numSamples);
  const used = new Set<number>();

  for (let s = 0; s < numSamples; s++) {
    let idx: number;
    do {
      const u = __random();
      idx = 0;
      while (idx < k - 1 && u > cdf[idx]!) idx++;
    } while (!replacement && used.has(idx));
    result[s] = idx;
    if (!replacement) used.add(idx);
  }

  return TensorClass.fromTypedArray({
    data: result,
    shape: [numSamples],
    dtype: "int32",
    device: resolveDevice(),
  });
}

/**
 * Sample from a categorical distribution using the Gumbel-Softmax trick.
 *
 * Produces differentiable approximate one-hot samples from categorical logits.
 *
 * @param logits - Unnormalized log-probabilities, shape (n_categories,) or (batch, n_categories)
 * @param tau - Temperature parameter (default: 1.0). Lower = more discrete.
 * @param hard - If true, returns hard one-hot vectors (default: false)
 * @returns Tensor of same shape as logits with softmax probabilities
 */
export function gumbel_softmax(logits: Tensor, tau: number = 1.0, hard: boolean = false): Tensor {
  if (logits.ndim < 1 || logits.ndim > 2) {
    throw new InvalidParameterError(
      "gumbel_softmax requires 1D or 2D logits",
      "logits",
      logits.ndim
    );
  }
  if (!Number.isFinite(tau) || tau <= 0) {
    throw new InvalidParameterError("tau must be a positive finite number", "tau", tau);
  }

  const is1D = logits.ndim === 1;
  const batchSize = is1D ? 1 : (logits.shape[0] ?? 1);
  const nCat = is1D ? logits.size : (logits.shape[1] ?? 1);
  const totalSize = batchSize * nCat;

  const result = new Float64Array(totalSize);

  for (let b = 0; b < batchSize; b++) {
    // Sample Gumbel noise and add to logits
    const vals: number[] = [];
    let maxVal = -Infinity;
    for (let j = 0; j < nCat; j++) {
      const logit = Number(logits.data[logits.offset + b * nCat + j]);
      // Gumbel(0,1) = -log(-log(U))
      const u = randomOpenUnit();
      const g = -Math.log(-Math.log(u));
      const v = (logit + g) / tau;
      vals.push(v);
      if (v > maxVal) maxVal = v;
    }

    // Softmax with numerical stability
    let sumExp = 0;
    for (let j = 0; j < nCat; j++) {
      vals[j] = Math.exp(vals[j]! - maxVal);
      sumExp += vals[j]!;
    }

    if (hard) {
      // Straight-through: argmax as one-hot
      let argmax = 0;
      let maxP = vals[0]!;
      for (let j = 1; j < nCat; j++) {
        if (vals[j]! > maxP) {
          maxP = vals[j]!;
          argmax = j;
        }
      }
      for (let j = 0; j < nCat; j++) {
        result[b * nCat + j] = j === argmax ? 1 : 0;
      }
    } else {
      for (let j = 0; j < nCat; j++) {
        result[b * nCat + j] = vals[j]! / sumExp;
      }
    }
  }

  const shape: Shape = is1D ? [nCat] : [batchSize, nCat];
  return TensorClass.fromTypedArray({
    data: result,
    shape,
    dtype: "float64",
    device: resolveDevice(),
  });
}

/**
 * Random samples from Bernoulli distribution.
 *
 * @param p - Probability of success (in [0, 1])
 * @param shape - Output shape
 * @param opts - Options
 * @returns Tensor of 0s and 1s
 */
export function bernoulli(p: number, shape: Shape = [], opts: RandomOptions = {}): Tensor {
  if (!Number.isFinite(p) || p < 0 || p > 1) {
    throw new InvalidParameterError("p must be in [0, 1]", "p", p);
  }
  const size = shapeToSize(shape);
  const dtype = resolveIntegerDType(opts.dtype, "bernoulli");
  const data = allocateIntegerBuffer(dtype, size);

  for (let i = 0; i < size; i++) {
    writeInteger(data, i, __random() < p ? 1 : 0);
  }

  return TensorClass.fromTypedArray({
    data,
    shape,
    dtype,
    device: resolveDevice(opts.device),
  });
}

/**
 * Random samples from geometric distribution.
 *
 * Number of Bernoulli trials needed to get one success.
 * P(X=k) = (1-p)^(k-1) * p, k=1,2,...
 *
 * @param p - Probability of success per trial (in (0, 1])
 * @param shape - Output shape
 * @param opts - Options
 * @returns Tensor of positive integers
 */
export function geometric(p: number, shape: Shape = [], opts: RandomOptions = {}): Tensor {
  if (!Number.isFinite(p) || p <= 0 || p > 1) {
    throw new InvalidParameterError("p must be in (0, 1]", "p", p);
  }
  const size = shapeToSize(shape);
  const dtype = resolveIntegerDType(opts.dtype, "geometric");
  const data = allocateIntegerBuffer(dtype, size);

  if (p === 1) {
    for (let i = 0; i < size; i++) {
      writeInteger(data, i, 1);
    }
  } else {
    const logQ = Math.log(1 - p);
    for (let i = 0; i < size; i++) {
      const u = randomOpenUnit();
      writeInteger(data, i, Math.floor(Math.log(u) / logQ) + 1);
    }
  }

  return TensorClass.fromTypedArray({
    data,
    shape,
    dtype,
    device: resolveDevice(opts.device),
  });
}

/**
 * Random samples from log-normal distribution.
 *
 * If X ~ Normal(mean, std), then exp(X) ~ LogNormal(mean, std).
 *
 * @param mean - Mean of underlying normal distribution (default: 0)
 * @param std - Std of underlying normal distribution (default: 1, must be >= 0)
 * @param shape - Output shape
 * @param opts - Options
 * @returns Tensor of positive floats
 */
export function lognormal(
  mean: number = 0,
  std: number = 1,
  shape: Shape = [],
  opts: RandomOptions = {}
): Tensor {
  if (!Number.isFinite(mean) || !Number.isFinite(std)) {
    throw new InvalidParameterError("mean and std must be finite", "mean/std", {
      mean,
      std,
    });
  }
  if (std < 0) {
    throw new InvalidParameterError("std must be >= 0", "std", std);
  }
  const size = shapeToSize(shape);
  const dtype = resolveFloatDType(opts.dtype, "lognormal");
  const data = allocateFloatBuffer(dtype, size);

  // Draw the underlying normals in one bulk pass (state stays in the RNG
  // module), then exponentiate in place — avoids a module-boundary call per
  // element the way randn already does.
  __fillNormal(data, size);
  for (let i = 0; i < size; i++) {
    data[i] = Math.exp((data[i] as number) * std + mean);
  }

  return TensorClass.fromTypedArray({
    data,
    shape,
    dtype,
    device: resolveDevice(opts.device),
  });
}

/**
 * Random samples from chi-squared distribution.
 *
 * Chi-squared with k degrees of freedom is Gamma(k/2, 2).
 *
 * @param df - Degrees of freedom (must be > 0)
 * @param shape - Output shape
 * @param opts - Options
 * @returns Tensor of positive floats
 *
 * @example
 * ```js
 * import { chi2 } from 'deepbox/random';
 * const x = chi2(5, [100]);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/random-distributions | Deepbox Distributions}
 */
export function chi2(df: number, shape: Shape = [], opts: RandomOptions = {}): Tensor {
  if (!Number.isFinite(df) || df <= 0) {
    throw new InvalidParameterError("df must be a finite number > 0", "df", df);
  }
  const size = shapeToSize(shape);
  const dtype = resolveFloatDType(opts.dtype, "chi2");
  const data = allocateFloatBuffer(dtype, size);

  for (let i = 0; i < size; i++) {
    data[i] = sampleGamma(df / 2, 2);
  }

  return TensorClass.fromTypedArray({
    data,
    shape,
    dtype,
    device: resolveDevice(opts.device),
  });
}

/**
 * Random samples from Student's t distribution.
 *
 * If Z ~ N(0,1) and V ~ Chi2(df), then Z / sqrt(V/df) ~ t(df).
 *
 * @param df - Degrees of freedom (must be > 0)
 * @param shape - Output shape
 * @param opts - Options
 * @returns Tensor of floats
 *
 * @example
 * ```js
 * import { student_t } from 'deepbox/random';
 * const x = student_t(10, [100]);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/random-distributions | Deepbox Distributions}
 */
export function student_t(df: number, shape: Shape = [], opts: RandomOptions = {}): Tensor {
  if (!Number.isFinite(df) || df <= 0) {
    throw new InvalidParameterError("df must be a finite number > 0", "df", df);
  }
  const size = shapeToSize(shape);
  const dtype = resolveFloatDType(opts.dtype, "student_t");
  const data = allocateFloatBuffer(dtype, size);

  for (let i = 0; i < size; i++) {
    const z = __normalRandom();
    const v = sampleGamma(df / 2, 2);
    data[i] = z / Math.sqrt(v / df);
  }

  return TensorClass.fromTypedArray({
    data,
    shape,
    dtype,
    device: resolveDevice(opts.device),
  });
}

/**
 * Random samples from F distribution.
 *
 * If X1 ~ Chi2(dfn) and X2 ~ Chi2(dfd), then (X1/dfn) / (X2/dfd) ~ F(dfn, dfd).
 *
 * @param dfn - Numerator degrees of freedom (must be > 0)
 * @param dfd - Denominator degrees of freedom (must be > 0)
 * @param shape - Output shape
 * @param opts - Options
 * @returns Tensor of positive floats
 *
 * @example
 * ```js
 * import { f_distribution } from 'deepbox/random';
 * const x = f_distribution(5, 10, [100]);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/random-distributions | Deepbox Distributions}
 */
export function f_distribution(
  dfn: number,
  dfd: number,
  shape: Shape = [],
  opts: RandomOptions = {}
): Tensor {
  if (!Number.isFinite(dfn) || dfn <= 0) {
    throw new InvalidParameterError("dfn must be a finite number > 0", "dfn", dfn);
  }
  if (!Number.isFinite(dfd) || dfd <= 0) {
    throw new InvalidParameterError("dfd must be a finite number > 0", "dfd", dfd);
  }
  const size = shapeToSize(shape);
  const dtype = resolveFloatDType(opts.dtype, "f_distribution");
  const data = allocateFloatBuffer(dtype, size);

  for (let i = 0; i < size; i++) {
    const x1 = sampleGamma(dfn / 2, 2);
    const x2 = sampleGamma(dfd / 2, 2);
    data[i] = x1 / dfn / (x2 / dfd);
  }

  return TensorClass.fromTypedArray({
    data,
    shape,
    dtype,
    device: resolveDevice(opts.device),
  });
}

/**
 * Random samples from Laplace distribution.
 *
 * Uses inverse CDF: loc - scale * sign(U - 0.5) * ln(1 - 2|U - 0.5|)
 *
 * @param loc - Location parameter (default: 0)
 * @param scale - Scale parameter (default: 1, must be > 0)
 * @param shape - Output shape
 * @param opts - Options
 * @returns Tensor of floats
 *
 * @example
 * ```js
 * import { laplace } from 'deepbox/random';
 * const x = laplace(0, 1, [100]);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/random-distributions | Deepbox Distributions}
 */
export function laplace(
  loc: number = 0,
  scale: number = 1,
  shape: Shape = [],
  opts: RandomOptions = {}
): Tensor {
  if (!Number.isFinite(loc)) {
    throw new InvalidParameterError("loc must be finite", "loc", loc);
  }
  if (!Number.isFinite(scale) || scale <= 0) {
    throw new InvalidParameterError("scale must be a finite number > 0", "scale", scale);
  }
  const size = shapeToSize(shape);
  const dtype = resolveFloatDType(opts.dtype, "laplace");
  const data = allocateFloatBuffer(dtype, size);

  for (let i = 0; i < size; i++) {
    const u = __random() - 0.5;
    data[i] = loc - scale * Math.sign(u) * Math.log(1 - 2 * Math.abs(u));
  }

  return TensorClass.fromTypedArray({
    data,
    shape,
    dtype,
    device: resolveDevice(opts.device),
  });
}

/**
 * Random samples from Cauchy distribution.
 *
 * Uses inverse CDF: loc + scale * tan(π * (U - 0.5))
 *
 * @param loc - Location parameter (default: 0)
 * @param scale - Scale parameter (default: 1, must be > 0)
 * @param shape - Output shape
 * @param opts - Options
 * @returns Tensor of floats
 *
 * @example
 * ```js
 * import { cauchy } from 'deepbox/random';
 * const x = cauchy(0, 1, [100]);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/random-distributions | Deepbox Distributions}
 */
export function cauchy(
  loc: number = 0,
  scale: number = 1,
  shape: Shape = [],
  opts: RandomOptions = {}
): Tensor {
  if (!Number.isFinite(loc)) {
    throw new InvalidParameterError("loc must be finite", "loc", loc);
  }
  if (!Number.isFinite(scale) || scale <= 0) {
    throw new InvalidParameterError("scale must be a finite number > 0", "scale", scale);
  }
  const size = shapeToSize(shape);
  const dtype = resolveFloatDType(opts.dtype, "cauchy");
  const data = allocateFloatBuffer(dtype, size);

  for (let i = 0; i < size; i++) {
    const u = randomOpenUnit();
    data[i] = loc + scale * Math.tan(Math.PI * (u - 0.5));
  }

  return TensorClass.fromTypedArray({
    data,
    shape,
    dtype,
    device: resolveDevice(opts.device),
  });
}

/**
 * Random samples from Weibull distribution.
 *
 * Uses inverse CDF: scale * (-ln(U))^(1/shape_param)
 *
 * @param shape_param - Shape parameter (k, must be > 0)
 * @param scale - Scale parameter (lambda, default: 1, must be > 0)
 * @param shape - Output shape
 * @param opts - Options
 * @returns Tensor of positive floats
 *
 * @example
 * ```js
 * import { weibull } from 'deepbox/random';
 * const x = weibull(1.5, 1, [100]);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/random-distributions | Deepbox Distributions}
 */
export function weibull(
  shape_param: number,
  scale: number = 1,
  shape: Shape = [],
  opts: RandomOptions = {}
): Tensor {
  if (!Number.isFinite(shape_param) || shape_param <= 0) {
    throw new InvalidParameterError(
      "shape_param must be a finite number > 0",
      "shape_param",
      shape_param
    );
  }
  if (!Number.isFinite(scale) || scale <= 0) {
    throw new InvalidParameterError("scale must be a finite number > 0", "scale", scale);
  }
  const size = shapeToSize(shape);
  const dtype = resolveFloatDType(opts.dtype, "weibull");
  const data = allocateFloatBuffer(dtype, size);

  // Bulk-fill uniforms, then apply the inverse CDF in place. The rare exact-0
  // draw is nudged into the open interval so log() stays finite.
  __fillUniform(data, size);
  const invShape = 1 / shape_param;
  for (let i = 0; i < size; i++) {
    let u = data[i] as number;
    if (u <= 0) u = randomOpenUnit();
    data[i] = scale * (-Math.log(u)) ** invShape;
  }

  return TensorClass.fromTypedArray({
    data,
    shape,
    dtype,
    device: resolveDevice(opts.device),
  });
}

/**
 * Random samples from triangular distribution.
 *
 * @param left - Lower limit
 * @param mode - Mode (peak)
 * @param right - Upper limit
 * @param shape - Output shape
 * @param opts - Options
 * @returns Tensor of floats in [left, right]
 *
 * @throws {InvalidParameterError} When left >= right or mode is out of [left, right]
 *
 * @example
 * ```js
 * import { triangular } from 'deepbox/random';
 * const x = triangular(0, 0.5, 1, [100]);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/random-distributions | Deepbox Distributions}
 */
export function triangular(
  left: number,
  mode: number,
  right: number,
  shape: Shape = [],
  opts: RandomOptions = {}
): Tensor {
  if (!Number.isFinite(left) || !Number.isFinite(mode) || !Number.isFinite(right)) {
    throw new InvalidParameterError("left, mode, and right must be finite", "left/mode/right", {
      left,
      mode,
      right,
    });
  }
  if (left >= right) {
    throw new InvalidParameterError("left must be < right", "left", left);
  }
  if (mode < left || mode > right) {
    throw new InvalidParameterError("mode must be in [left, right]", "mode", mode);
  }
  const size = shapeToSize(shape);
  const dtype = resolveFloatDType(opts.dtype, "triangular");
  const data = allocateFloatBuffer(dtype, size);
  const fc = (mode - left) / (right - left);

  for (let i = 0; i < size; i++) {
    const u = __random();
    if (u < fc) {
      data[i] = left + Math.sqrt(u * (right - left) * (mode - left));
    } else {
      data[i] = right - Math.sqrt((1 - u) * (right - left) * (right - mode));
    }
  }

  return TensorClass.fromTypedArray({
    data,
    shape,
    dtype,
    device: resolveDevice(opts.device),
  });
}

/**
 * Random samples from negative binomial distribution.
 *
 * Number of failures before achieving r successes.
 * Uses gamma-Poisson mixture: sample lambda ~ Gamma(r, (1-p)/p), then X ~ Poisson(lambda).
 *
 * @param r - Number of successes (must be > 0)
 * @param p - Probability of success per trial (in (0, 1])
 * @param shape - Output shape
 * @param opts - Options
 * @returns Tensor of non-negative integers
 *
 * @example
 * ```js
 * import { negative_binomial } from 'deepbox/random';
 * const x = negative_binomial(5, 0.5, [100]);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/random-distributions | Deepbox Distributions}
 */
export function negative_binomial(
  r: number,
  p: number,
  shape: Shape = [],
  opts: RandomOptions = {}
): Tensor {
  if (!Number.isFinite(r) || r <= 0) {
    throw new InvalidParameterError("r must be a finite number > 0", "r", r);
  }
  if (!Number.isFinite(p) || p <= 0 || p > 1) {
    throw new InvalidParameterError("p must be in (0, 1]", "p", p);
  }
  const size = shapeToSize(shape);
  const dtype = resolveIntegerDType(opts.dtype, "negative_binomial");
  const data = allocateIntegerBuffer(dtype, size);

  if (p === 1) {
    // All samples are 0 (0 failures before r successes with certainty)
    for (let i = 0; i < size; i++) {
      writeInteger(data, i, 0);
    }
  } else {
    const gammaScale = (1 - p) / p;
    for (let i = 0; i < size; i++) {
      // Gamma-Poisson mixture (lambda can be large for small p, so use the
      // underflow-safe Poisson sampler).
      const lambda = sampleGamma(r, gammaScale);
      writeInteger(data, i, samplePoissonScalar(lambda));
    }
  }

  return TensorClass.fromTypedArray({
    data,
    shape,
    dtype,
    device: resolveDevice(opts.device),
  });
}

/**
 * Random samples from hypergeometric distribution.
 *
 * Models drawing without replacement from a finite population.
 *
 * @param ngood - Number of good (success) items in the population
 * @param nbad - Number of bad (failure) items in the population
 * @param nsample - Number of items drawn (without replacement)
 * @param shape - Output shape
 * @param opts - Options
 * @returns Tensor of non-negative integers (number of good items drawn)
 *
 * @throws {InvalidParameterError} When parameters are invalid
 *
 * @example
 * ```js
 * import { hypergeometric } from 'deepbox/random';
 * const x = hypergeometric(10, 5, 7, [100]);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/random-distributions | Deepbox Distributions}
 */
export function hypergeometric(
  ngood: number,
  nbad: number,
  nsample: number,
  shape: Shape = [],
  opts: RandomOptions = {}
): Tensor {
  if (!Number.isInteger(ngood) || ngood < 0) {
    throw new InvalidParameterError("ngood must be a non-negative integer", "ngood", ngood);
  }
  if (!Number.isInteger(nbad) || nbad < 0) {
    throw new InvalidParameterError("nbad must be a non-negative integer", "nbad", nbad);
  }
  if (!Number.isInteger(nsample) || nsample < 0) {
    throw new InvalidParameterError("nsample must be a non-negative integer", "nsample", nsample);
  }
  const N = ngood + nbad;
  if (nsample > N) {
    throw new InvalidParameterError("nsample must be <= ngood + nbad", "nsample", nsample);
  }
  const size = shapeToSize(shape);
  const dtype = resolveIntegerDType(opts.dtype, "hypergeometric");
  const data = allocateIntegerBuffer(dtype, size);

  for (let i = 0; i < size; i++) {
    // Direct simulation via sequential draws
    let good = ngood;
    let total = N;
    let successes = 0;
    for (let d = 0; d < nsample; d++) {
      if (total === 0) break;
      if (__random() < good / total) {
        successes++;
        good--;
      }
      total--;
    }
    writeInteger(data, i, successes);
  }

  return TensorClass.fromTypedArray({
    data,
    shape,
    dtype,
    device: resolveDevice(opts.device),
  });
}

// Helper: sample from Gamma(shape, scale) using Marsaglia-Tsang method
function sampleGamma(shape: number, scale: number): number {
  if (shape < 1) {
    // Use Ahrens-Dieter for shape < 1
    const u = __random();
    return sampleGamma(shape + 1, scale) * u ** (1 / shape);
  }
  // Marsaglia and Tsang's method for shape >= 1
  const d = shape - 1.0 / 3.0;
  const c = 1.0 / Math.sqrt(9.0 * d);
  while (true) {
    let x: number;
    let v: number;
    do {
      x = __normalRandom();
      v = 1.0 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = __random();
    if (u < 1.0 - 0.0331 * (x * x) * (x * x)) {
      return d * v * scale;
    }
    if (Math.log(u) < 0.5 * x * x + d * (1.0 - v + Math.log(v))) {
      return d * v * scale;
    }
  }
}

/**
 * Random samples from the von Mises distribution (circular normal).
 *
 * @param mu - Mean direction in radians
 * @param kappa - Concentration parameter (must be >= 0)
 * @param shape - Output shape
 * @param opts - Options
 *
 * @remarks
 * - Uses Best & Fisher's algorithm for efficient sampling.
 * - When kappa=0, equivalent to uniform on [-pi, pi).
 * - Values are in [-pi, pi).
 * - Deterministic when seed is set via {@link setSeed}.
 */
export function vonmises(
  mu: number,
  kappa: number,
  shape: Shape = [],
  opts: RandomOptions = {}
): Tensor {
  if (!Number.isFinite(mu)) {
    throw new InvalidParameterError("mu must be finite", "mu", mu);
  }
  if (!Number.isFinite(kappa) || kappa < 0) {
    throw new InvalidParameterError("kappa must be a finite number >= 0", "kappa", kappa);
  }
  const size = shapeToSize(shape);
  const dtype = resolveFloatDType(opts.dtype, "vonmises");
  const data = allocateFloatBuffer(dtype, size);

  if (kappa < 1e-6) {
    // Effectively uniform on [-pi, pi)
    for (let i = 0; i < size; i++) {
      data[i] = __random() * 2 * Math.PI - Math.PI;
    }
  } else {
    // Best & Fisher algorithm
    const tau = 1 + Math.sqrt(1 + 4 * kappa * kappa);
    const rho = (tau - Math.sqrt(2 * tau)) / (2 * kappa);
    const r = (1 + rho * rho) / (2 * rho);

    for (let i = 0; i < size; i++) {
      let theta: number;
      while (true) {
        const u1 = __random();
        const z = Math.cos(Math.PI * u1);
        const f = (1 + r * z) / (r + z);
        const c = kappa * (r - f);

        const u2 = __random();
        if (c * (2 - c) > u2 || Math.log(c / u2) + 1 >= c) {
          const u3 = __random();
          theta = u3 > 0.5 ? Math.acos(f) : -Math.acos(f);
          break;
        }
      }
      // Shift by mu and wrap to [-pi, pi)
      let val = theta + mu;
      val = val - 2 * Math.PI * Math.floor((val + Math.PI) / (2 * Math.PI));
      data[i] = val;
    }
  }

  return TensorClass.fromTypedArray({
    data,
    shape,
    dtype,
    device: resolveDevice(opts.device),
  });
}

/**
 * Random samples from the Pareto (Type I) distribution.
 *
 * @param alpha - Shape parameter (must be > 0)
 * @param xm - Scale parameter (minimum value, default: 1, must be > 0)
 * @param shape - Output shape
 * @param opts - Options
 *
 * @remarks
 * - Uses inverse transform: xm / U^(1/alpha).
 * - All values are >= xm.
 * - Mean = alpha*xm/(alpha-1) for alpha > 1.
 * - Deterministic when seed is set via {@link setSeed}.
 */
export function pareto(
  alpha: number,
  xm: number = 1,
  shape: Shape = [],
  opts: RandomOptions = {}
): Tensor {
  if (!Number.isFinite(alpha) || alpha <= 0) {
    throw new InvalidParameterError("alpha must be a finite number > 0", "alpha", alpha);
  }
  if (!Number.isFinite(xm) || xm <= 0) {
    throw new InvalidParameterError("xm must be a finite number > 0", "xm", xm);
  }
  const size = shapeToSize(shape);
  const dtype = resolveFloatDType(opts.dtype, "pareto");
  const data = allocateFloatBuffer(dtype, size);

  const invAlpha = 1 / alpha;
  for (let i = 0; i < size; i++) {
    const u = randomOpenUnit();
    data[i] = xm / u ** invAlpha;
  }

  return TensorClass.fromTypedArray({
    data,
    shape,
    dtype,
    device: resolveDevice(opts.device),
  });
}

/**
 * Random samples from the Rayleigh distribution.
 *
 * @param sigma - Scale parameter (default: 1, must be > 0)
 * @param shape - Output shape
 * @param opts - Options
 *
 * @remarks
 * - Uses inverse transform: sigma * sqrt(-2 * log(U)).
 * - All values are positive.
 * - Mean = sigma * sqrt(pi/2).
 * - Deterministic when seed is set via {@link setSeed}.
 */
export function rayleigh(sigma: number = 1, shape: Shape = [], opts: RandomOptions = {}): Tensor {
  if (!Number.isFinite(sigma) || sigma <= 0) {
    throw new InvalidParameterError("sigma must be a finite number > 0", "sigma", sigma);
  }
  const size = shapeToSize(shape);
  const dtype = resolveFloatDType(opts.dtype, "rayleigh");
  const data = allocateFloatBuffer(dtype, size);

  // Bulk-fill uniforms, then apply the inverse CDF in place (see weibull).
  __fillUniform(data, size);
  for (let i = 0; i < size; i++) {
    let u = data[i] as number;
    if (u <= 0) u = randomOpenUnit();
    data[i] = sigma * Math.sqrt(-2 * Math.log(u));
  }

  return TensorClass.fromTypedArray({
    data,
    shape,
    dtype,
    device: resolveDevice(opts.device),
  });
}

/**
 * Random samples from the Zipf (zeta) distribution.
 *
 * @param s - Exponent parameter (must be > 1)
 * @param shape - Output shape
 * @param opts - Options
 *
 * @remarks
 * - Uses rejection sampling method.
 * - Values are positive integers >= 1.
 * - P(X=k) proportional to k^(-s).
 * - Deterministic when seed is set via {@link setSeed}.
 * - Only int32 and int64 dtypes are supported.
 */
export function zipf(s: number, shape: Shape = [], opts: RandomOptions = {}): Tensor {
  if (!Number.isFinite(s) || s <= 1) {
    throw new InvalidParameterError("s must be a finite number > 1", "s", s);
  }
  const size = shapeToSize(shape);
  const dtype = resolveIntegerDType(opts.dtype, "zipf");
  const data = allocateIntegerBuffer(dtype, size);

  // Rejection method based on Luc Devroye's algorithm
  const b = 2 ** (s - 1);

  for (let i = 0; i < size; i++) {
    while (true) {
      const u = randomOpenUnit();
      const v = __random();
      const x = Math.floor(u ** (-1 / (s - 1)));
      if (x < 1 || !Number.isFinite(x)) continue;
      const t = (1 + 1 / x) ** (s - 1);
      if ((v * x * (t - 1)) / (b - 1) <= t / b) {
        writeInteger(data, i, x);
        break;
      }
    }
  }

  return TensorClass.fromTypedArray({
    data,
    shape,
    dtype,
    device: resolveDevice(opts.device),
  });
}
