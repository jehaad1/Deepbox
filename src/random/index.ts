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
import { type Tensor, Tensor as TensorClass, type TypedArray } from "../ndarray";
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
      // Rare rejection region: compute the threshold once and resample.
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

// Largest integer n such that every integer in [0, n] is exactly representable in float32.
const FLOAT32_EXACT_INT_MAX = 2 ** 24;

// Largest number of draws served by the sparse (Map-backed) pool in choice().
const SPARSE_POOL_MAX_DRAWS = 4_000_000;

function allocateFloatBuffer(dtype: FloatDType, size: number): FloatBuffer {
  return dtype === "float32" ? new Float32Array(size) : new Float64Array(size);
}

function allocateIntegerBuffer(dtype: IntegerDType, size: number): IntegerBuffer {
  return dtype === "int64" ? new BigInt64Array(size) : new Int32Array(size);
}

const INT64_LIMIT = 2 ** 63;

/**
 * Store an integer sample, refusing values the buffer cannot represent
 * (typed arrays would otherwise wrap them silently).
 */
function writeInteger(buffer: IntegerBuffer, index: number, value: number): void {
  if (buffer instanceof BigInt64Array) {
    if (!Number.isFinite(value) || Math.abs(value) >= INT64_LIMIT) {
      throw new InvalidParameterError(
        `sampled value ${value} does not fit in int64`,
        "dtype",
        "int64"
      );
    }
    buffer[index] = BigInt(value);
  } else {
    if (!(value >= INT32_MIN && value <= INT32_MAX)) {
      throw new InvalidParameterError(
        `sampled value ${value} does not fit in int32; request dtype "int64"`,
        "dtype",
        "int32"
      );
    }
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
 * Check that a tensor is stored contiguously in row-major order and return its
 * storage window. Size-1 axes may carry any stride; the tensor may be a
 * contiguous view into a larger buffer (non-zero offset).
 *
 * @param t - Tensor to validate
 * @param functionName - Name of the calling function for error messages
 * @returns `start` (storage offset of the first element) and `length` (element count)
 */
function contiguousWindow(t: Tensor, functionName: string): { start: number; length: number } {
  const length = t.size;
  if (length > 0) {
    let expected = 1;
    for (let axis = t.ndim - 1; axis >= 0; axis--) {
      const dim = t.shape[axis] as number;
      if (dim !== 1 && t.strides[axis] !== expected) {
        throw new InvalidParameterError(
          `${functionName} currently requires a contiguous tensor`,
          "strides",
          t.strides
        );
      }
      expected *= dim;
    }
  }
  return { start: t.offset, length };
}

/**
 * Read the elements of a 1D or 2D numeric tensor as float64, honoring its
 * offset and strides (so sliced and transposed views work).
 */
function readNumbers(t: Tensor, functionName: string, name: string): Float64Array {
  if (t.dtype === "string") {
    throw new DTypeError(`${functionName} requires a numeric ${name} tensor`);
  }
  const out = new Float64Array(t.size);
  const data = t.data;
  if (t.ndim === 1) {
    const stride = t.strides[0] as number;
    for (let i = 0; i < out.length; i++) out[i] = Number(data[t.offset + i * stride]);
  } else if (t.ndim === 2) {
    const rows = t.shape[0] as number;
    const cols = t.shape[1] as number;
    const s0 = t.strides[0] as number;
    const s1 = t.strides[1] as number;
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        out[i * cols + j] = Number(data[t.offset + i * s0 + j * s1]);
      }
    }
  } else {
    throw new InvalidParameterError(`${name} must be 1D or 2D`, name, t.shape);
  }
  return out;
}

/** Machine epsilon of a floating-point storage dtype (0 for non-float dtypes). */
function dtypeEpsilon(dtype: DType): number {
  switch (dtype) {
    case "float64":
      return 2.220446049250313e-16;
    case "float32":
      return 1.1920928955078125e-7;
    case "float16":
      return 9.765625e-4;
    case "bfloat16":
      return 7.8125e-3;
    default:
      return 0;
  }
}

/** Resolve a `size` argument (count or shape) to the leading output shape. */
function resolveLeadingShape(size: number | Shape, name: string): number[] {
  if (typeof size === "number") {
    if (!Number.isSafeInteger(size) || size < 0) {
      throw new InvalidParameterError(`${name} must be a non-negative integer`, name, size);
    }
    return [size];
  }
  validateShape(size);
  return [...size];
}

/** Float dtype for sampling functions that infer it from the library default. */
function resolveSamplingFloatDType(dtype: DType | undefined, functionName: string): FloatDType {
  if (dtype !== undefined) return resolveFloatDType(dtype, functionName);
  const fallback = getConfig().defaultDtype;
  return fallback === "float64" ? "float64" : "float32";
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
 * (exp(-lambda) tends to 0) for lambda ≳ 745 and silently caps samples, so lambda >= 30
 * uses Atkinson's (1979) logistic-envelope rejection method instead.
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

/**
 * Sample from Gamma(shape, 1). Marsaglia-Tsang for shape >= 1; for shape < 1 the
 * boost `Gamma(shape + 1) * U^(1 / shape)` is applied.
 */
function sampleGammaUnit(shape: number): number {
  if (shape < 1) {
    const u = randomOpenUnit();
    return __gammaLarge(shape + 1) * u ** (1 / shape);
  }
  return __gammaLarge(shape);
}

/**
 * Natural log of a Gamma(shape, 1) sample, drawn from the same random numbers as
 * {@link sampleGammaUnit}. Working in log space avoids the underflow of
 * `U^(1 / shape)` for small shapes, which otherwise turns gamma ratios into 0/0.
 */
function sampleLogGammaUnit(shape: number): number {
  if (shape < 1) {
    const u = randomOpenUnit();
    return Math.log(__gammaLarge(shape + 1)) + Math.log(u) / shape;
  }
  return Math.log(__gammaLarge(shape));
}

/** Sample from Gamma(shape, scale). */
function sampleGamma(shape: number, scale: number): number {
  return sampleGammaUnit(shape) * scale;
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
 * - float64 values are multiples of 2^-32; float32 values are rounded to float32
 *   but never reach 1.
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
 * - Uses the Ziggurat method (Marsaglia and Tsang, 2000).
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
        // value % range via reciprocal multiply. Integer `%` compiles to
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

/**
 * Random integers in half-open interval [low, high).
 *
 * @param low - Lowest integer (inclusive)
 * @param high - Highest integer (exclusive)
 * @param shape - Output shape
 * @param opts - Options (dtype, device)
 *
 * @throws {InvalidParameterError} When low or high is not a safe integer
 * @throws {InvalidParameterError} When high <= low
 *
 * @remarks
 * - Generates integers uniformly in [low, high) range (exactly unbiased).
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
 * - Values are uniformly distributed in [low, high). When the range is narrow
 *   relative to the magnitude of `low`, rounding to the output dtype can make a
 *   value equal to `high`.
 * - For very large ranges, floating-point precision may affect uniformity.
 * - `high === low` is allowed and returns `low` for every element.
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
  if (Number.isFinite(range)) {
    for (let i = 0; i < size; i++) {
      data[i] = (data[i] as number) * range + low;
    }
  } else {
    // high - low overflows float64: blend the endpoints instead.
    for (let i = 0; i < size; i++) {
      const u = data[i] as number;
      data[i] = low * (1 - u) + high * u;
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
 * - Uses the Ziggurat method (Marsaglia and Tsang, 2000) for the standard normal draw.
 * - All values are finite (no infinities from tail behavior).
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

/**
 * Stirling-series error `log(n!) - log(sqrt(2 pi n) (n / e)^n)` for an integer
 * n >= 0 (Loader, 2000). It is small (about 1 / (12 n)), so subtracting it
 * instead of whole log-factorials avoids the cancellation of terms of size n log n.
 */
function stirlingError(n: number): number {
  if (n === 0) return 0;
  if (n <= 15) {
    let logFact = 0;
    for (let i = 2; i <= n; i++) logFact += Math.log(i);
    return logFact - (n + 0.5) * Math.log(n) + n - 0.5 * Math.log(2 * Math.PI);
  }
  const nn = n * n;
  if (n > 500) return (1 / 12 - 1 / 360 / nn) / n;
  if (n > 80) return (1 / 12 - (1 / 360 - 1 / 1260 / nn) / nn) / n;
  if (n > 35) return (1 / 12 - (1 / 360 - (1 / 1260 - 1 / 1680 / nn) / nn) / nn) / n;
  return (1 / 12 - (1 / 360 - (1 / 1260 - (1 / 1680 - 1 / 1188 / nn) / nn) / nn) / nn) / n;
}

/** Deviance term `x log(x / np) + np - x`, evaluated stably when x is close to np. */
function devianceTerm(x: number, np: number): number {
  if (Math.abs(x - np) < 0.1 * (x + np)) {
    const v = (x - np) / (x + np);
    let s = (x - np) * v;
    if (Math.abs(s) < Number.MIN_VALUE) return s;
    let ej = 2 * x * v;
    const v2 = v * v;
    for (let j = 1; j < 1000; j++) {
      ej *= v2;
      const next = s + ej / (2 * j + 1);
      if (next === s) return next;
      s = next;
    }
  }
  return x * Math.log(x / np) + np - x;
}

/**
 * Binomial pmf at `k` (0 < k < n) via Loader's saddle-point formula. Accurate to
 * near machine precision for any n, unlike `exp(logC(n, k) + ...)`, whose
 * absolute error grows with `n log n`.
 */
function binomialPmf(n: number, k: number, p: number, q: number): number {
  const logPmf =
    stirlingError(n) -
    stirlingError(k) -
    stirlingError(n - k) -
    devianceTerm(k, n * p) -
    devianceTerm(n - k, n * q);
  const logFactor = Math.log(2 * Math.PI) + Math.log(k) + Math.log1p(-k / n);
  return Math.exp(logPmf - 0.5 * logFactor);
}

/**
 * Inversion by "chop-down" search outward from the mode: the pmf is walked in
 * both directions with the recurrence `pmf(k+1) / pmf(k) = (n-k)/(k+1) * p/q`.
 */
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
  let leftActive = left > 0;
  let rightActive = right < n;

  while (leftActive || rightActive) {
    if (leftActive) {
      pmfLeft *= (left / (n - left + 1)) * ratioLeft;
      left -= 1;
      cumulative += pmfLeft;
      if (u <= cumulative) {
        return left;
      }
      // Stop once the tail underflows (nothing left to add on this side).
      leftActive = left > 0 && pmfLeft > 0;
    }
    if (rightActive) {
      pmfRight *= ((n - right) / (right + 1)) * ratioRight;
      right += 1;
      cumulative += pmfRight;
      if (u <= cumulative) {
        return right;
      }
      rightActive = right < n && pmfRight > 0;
    }
  }

  // u fell into the rounding gap above the summed pmf (about 1e-15): use the mode.
  return mode;
}

/** Stirling correction used in the BTPE squeeze, as in NumPy's random_binomial_btpe. */
function btpeStirling(x: number): number {
  const x2 = x * x;
  return (13680 - (462 - (132 - (99 - 140 / x2) / x2) / x2) / x2) / x / 166320;
}

/**
 * BTPE (Kachitvichyanukul and Schmeiser, 1988), the exact rejection sampler NumPy
 * uses for large means. Expected work is O(1) per draw for any n, where inversion
 * needs O(sqrt(n p q)) steps. Requires p <= 0.5 and n * p >= 30.
 */
function binomialBtpe(n: number, p: number): number {
  const q = 1 - p;
  const fm = n * p + p;
  const m = Math.floor(fm);
  const nrq = n * p * q;
  const p1 = Math.floor(2.195 * Math.sqrt(nrq) - 4.6 * q) + 0.5;
  const xm = m + 0.5;
  const xl = xm - p1;
  const xr = xm + p1;
  const c = 0.134 + 20.5 / (15.3 + m);
  let a = (fm - xl) / (fm - xl * p);
  const laml = a * (1 + a / 2);
  a = (xr - fm) / (xr * q);
  const lamr = a * (1 + a / 2);
  const p2 = p1 * (1 + 2 * c);
  const p3 = p2 + c / laml;
  const p4 = p3 + c / lamr;

  for (;;) {
    const u = __random() * p4;
    let v = __random();
    let y: number;
    if (u <= p1) {
      // Triangular region: always accepted.
      return Math.floor(xm - p1 * v + u);
    }
    if (u <= p2) {
      // Parallelograms.
      const x = xl + (u - p1) / c;
      v = v * c + 1 - Math.abs(m - x + 0.5) / p1;
      if (v > 1) continue;
      y = Math.floor(x);
    } else if (u <= p3) {
      // Left exponential tail.
      if (v === 0) continue;
      y = Math.floor(xl + Math.log(v) / laml);
      if (y < 0) continue;
      v = v * (u - p2) * laml;
    } else {
      // Right exponential tail.
      if (v === 0) continue;
      y = Math.floor(xr - Math.log(v) / lamr);
      if (y > n) continue;
      v = v * (u - p3) * lamr;
    }

    const k = Math.abs(y - m);
    if (k <= 20 || k >= nrq / 2 - 1) {
      // Explicit evaluation of f(y) / f(m) by the recurrence.
      const s = p / q;
      const aa = s * (n + 1);
      let f = 1;
      if (m < y) {
        for (let i = m + 1; i <= y; i++) f *= aa / i - s;
      } else if (m > y) {
        for (let i = y + 1; i <= m; i++) f /= aa / i - s;
      }
      if (v <= f) return y;
      continue;
    }

    // Squeeze using upper and lower bounds on log(f(y)), then the final test.
    const rho = (k / nrq) * ((k * (k / 3 + 0.625) + 0.16666666666666666) / nrq + 0.5);
    const t = (-k * k) / (2 * nrq);
    const logV = Math.log(v);
    if (logV < t - rho) return y;
    if (logV > t + rho) continue;
    const x1 = y + 1;
    const f1 = m + 1;
    const z = n + 1 - m;
    const w = n - y + 1;
    const bound =
      xm * Math.log(f1 / x1) +
      (n - m + 0.5) * Math.log(z / w) +
      (y - m) * Math.log((w * p) / (x1 * q)) +
      btpeStirling(f1) +
      btpeStirling(z) +
      btpeStirling(x1) +
      btpeStirling(w);
    if (logV <= bound) return y;
  }
}

/**
 * One Binomial(n, p) draw for a safe-integer n >= 0 and p in [0, 1]. All methods
 * are exact: the geometric waiting-time method when the mean is below 10,
 * chop-down inversion from the mode up to a mean of 30, and BTPE above that,
 * so the cost per draw stays bounded for n as large as 2^53 - 1.
 */
function sampleBinomialScalar(n: number, p: number): number {
  if (n <= 0 || !(p > 0)) return 0;
  if (p >= 1) return n;
  const flip = p > 0.5;
  const prob = flip ? 1 - p : p;
  const q = 1 - prob;
  let k: number;
  const mean = n * prob;
  if (mean < 10) {
    k = binomialSmallMean(n, Math.log1p(-prob));
  } else if (mean < 30) {
    const mode = Math.floor((n + 1) * prob);
    k = binomialChopDown(n, prob, q, mode, binomialPmf(n, mode, prob, q));
  } else {
    k = binomialBtpe(n, prob);
  }
  return flip ? n - k : k;
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
 * - Uses exact methods throughout: geometric waiting times or CDF inversion for
 *   means below 10, chop-down inversion from the mode below 30, and BTPE (as
 *   NumPy does) above that, so the cost per draw stays bounded for large n.
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
  const mean = n * prob;
  // log1p keeps log(q) accurate (and non-zero) for tiny success probabilities.
  const logQ = Math.log1p(-prob);

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
  } else if (mean < 30) {
    const mode = Math.floor((n + 1) * prob);
    const pmfMode = binomialPmf(n, mode, prob, q);
    if (!Number.isFinite(pmfMode) || pmfMode <= 0) {
      throw new InvalidParameterError("Failed to initialize binomial sampler", "p", p);
    }
    for (let i = 0; i < size; i++) {
      const sample = binomialChopDown(n, prob, q, mode, pmfMode);
      writeInteger(data, i, flip ? n - sample : sample);
    }
  } else {
    // BTPE: O(1) expected work per draw, where inversion needs O(sqrt(n p q)).
    for (let i = 0; i < size; i++) {
      const sample = binomialBtpe(n, prob);
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
 * - Uses Knuth's method for lambda < 30 and Atkinson's rejection method for lambda >= 30.
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

  for (let i = 0; i < size; i++) {
    writeInteger(data, i, samplePoissonScalar(lambda));
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
 * - Uses ratio of two gamma distributions: X / (X + Y). When either parameter is
 *   below 1 the ratio is evaluated from log-gamma draws, so tiny parameters
 *   give values at (or very near) 0 and 1 instead of failing.
 * - All values are in [0, 1]; they are inside (0, 1) up to floating-point rounding.
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

  if (alpha < 1 || beta_param < 1) {
    // A Gamma(a < 1) draw can underflow to 0, and 0 / (0 + 0) would then be
    // undefined. Work with log-gammas: x / (x + y) = 1 / (1 + exp(log y - log x)).
    for (let i = 0; i < size; i++) {
      const logX = sampleLogGammaUnit(alpha);
      const logY = sampleLogGammaUnit(beta_param);
      const diff = logY - logX;
      // Both logs at -Infinity (shapes near the smallest subnormal): the law
      // collapses to a Bernoulli draw on the endpoints with weights alpha : beta.
      data[i] = Number.isNaN(diff)
        ? __random() < alpha / (alpha + beta_param)
          ? 1
          : 0
        : 1 / (1 + Math.exp(diff));
    }
  } else {
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
  const value = t.data[t.offset + index];
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

  const normalized = readNumbers(probabilities, "choice()", "p");
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const value = normalized[i] as number;
    if (!Number.isFinite(value) || value < 0) {
      throw new InvalidParameterError(
        "p must contain finite non-negative probabilities",
        "p",
        value
      );
    }
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
 * @throws {InvalidParameterError} When the tensor is not contiguous (non-standard strides)
 * @throws {DTypeError} When input tensor has string dtype
 *
 * @remarks
 * - Input tensor must be contiguous (no striding); a contiguous view with an offset is fine.
 * - A multi-dimensional tensor is sampled over its flattened elements, and the result is 1D
 *   (or `size`-shaped); it is not sampled by rows.
 * - `p` may be any 1D tensor (strided views included); it is normalized by its sum.
 * - Without replacement and without `p`, memory use is proportional to `size`, not to a large
 *   integer population `a`.
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

  // A numeric `a` means the population 0..a-1, which is never materialized.
  const aa: Tensor | null = typeof a === "number" ? null : a;

  if (aa && aa.dtype === "string") {
    throw new DTypeError("choice() does not support string tensors");
  }

  // Handle Tensor input: sample indices first, then gather values into a new tensor.
  // Note: we currently require contiguous storage, because `choice` is defined over
  // the flattened order. Using arbitrary strides would require computing a flat
  // index mapping.
  const n = aa ? aa.size : (a as number);
  if (!Number.isInteger(n) || n < 0) {
    throw new InvalidParameterError("Invalid tensor size", "n", n);
  }
  if (n > INT32_MAX + 1) {
    throw new InvalidParameterError(`Population size must be <= ${INT32_MAX + 1}`, "n", n);
  }

  // Check the layout before any random numbers are drawn so a rejected call
  // does not advance the seeded stream.
  if (aa && n > 0) contiguousWindow(aa, "choice()");

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
      const cdf = buildCdf(weights);
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

      drawWeightedWithoutReplacement(weights, outputSize, indices);
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
    // Partial Fisher–Yates. Draw all bounds in bulk when the population is small
    // enough for Lemire's exact multiply.
    const useBatch = n <= LEMIRE_MAX_BOUND;
    const rnd = useBatch ? new Uint32Array(outputSize) : null;
    if (rnd) __fillUint32(rnd, outputSize);
    // A dense pool costs O(n) memory even for a handful of draws, so a large
    // population with few draws keeps only the displaced entries in a Map. Both
    // layouts apply exactly the same swaps, so the sample is identical. A Map
    // holds at most ~2^24 entries, so many draws always use the dense pool.
    const dense = n <= 4 * outputSize || n <= 65536 || outputSize > SPARSE_POOL_MAX_DRAWS;
    const pool = dense ? new Int32Array(n) : null;
    if (pool) for (let i = 0; i < n; i++) pool[i] = i;
    const displaced = new Map<number, number>();
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
      if (pool) {
        const poolJ = pool[j] as number;
        pool[j] = pool[i] as number;
        pool[i] = poolJ;
        indices[i] = poolJ;
      } else {
        const poolJ = displaced.get(j) ?? j;
        displaced.set(j, displaced.get(i) ?? i);
        indices[i] = poolJ;
      }
    }
  }

  const outputShape: Shape = typeof size === "number" ? [size] : size ? [...size] : [1];

  if (!aa) {
    return TensorClass.fromTypedArray({
      data: indices,
      shape: outputShape,
      dtype: "int32",
      device: resolveDevice(),
    });
  }

  if (aa.dtype === "string") {
    throw new DTypeError("choice() does not support string tensors");
  }

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
 * - Input tensor must be contiguous (no striding); a contiguous view into a larger
 *   buffer is shuffled without touching the elements outside the view.
 * - A multi-dimensional tensor is shuffled over its flattened elements, not its rows.
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
  // Only a contiguous tensor can be shuffled in place: swapping storage slots
  // must map onto the logical flattened order.
  const { start, length: n } = contiguousWindow(x, "shuffle()");

  const data = x.data;
  if (!isTypedArray(data)) {
    throw new DTypeError("shuffle() does not support string tensors");
  }

  // Fisher–Yates shuffle over exactly this tensor's elements (a view must not
  // disturb the rest of its underlying buffer). All swap targets are drawn in
  // one batched pass so the per-element RNG cost is a table lookup.
  const js = fisherYatesTargets(n);

  // Split into two branches to maintain type safety without assertions.
  if (data instanceof BigInt64Array) {
    for (let i = n - 1; i > 0; i--) {
      const j = start + (js[i] as number);
      const temp = data[start + i];
      const swap = data[j];
      if (temp === undefined || swap === undefined) {
        throw new DeepboxError("Internal error: shuffle index out of bounds");
      }
      data[start + i] = swap;
      data[j] = temp;
    }
  } else {
    for (let i = n - 1; i > 0; i--) {
      const j = start + (js[i] as number);
      const temp = data[start + i];
      const swap = data[j];
      if (temp === undefined || swap === undefined) {
        throw new DeepboxError("Internal error: shuffle index out of bounds");
      }
      data[start + i] = swap;
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
 * @throws {InvalidParameterError} When x is a negative or non-integer number, or a non-contiguous tensor
 *
 * @remarks
 * - Returns a NEW tensor (does NOT modify input).
 * - If x is an integer, returns permutation of arange(x).
 * - If x is a tensor, returns a shuffled copy with the same shape (shuffled over its flattened
 *   elements, not its rows).
 * - Tensor inputs must be contiguous (no striding); a contiguous view is copied on its own.
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

  const { start, length } = contiguousWindow(x, "permutation()");
  const data = x.data;
  if (!isTypedArray(data)) {
    throw new DTypeError("permutation() does not support string tensors");
  }
  // Copy only this tensor's elements (not the rest of a larger shared buffer).
  const copy = TensorClass.fromTypedArray({
    data: data.slice(start, start + length),
    shape: [...x.shape],
    dtype: x.dtype,
    device: x.device,
  });
  shuffle(copy);
  return copy;
}

/**
 * Draw `count` indices without replacement, each draw proportional to the weight
 * of the entries not yet taken. The caller guarantees that at least `count`
 * weights are positive. `weights` is not modified.
 */
function drawWeightedWithoutReplacement(
  weights: ArrayLike<number>,
  count: number,
  out: Int32Array
): void {
  const remaining = Float64Array.from(weights);
  for (let i = 0; i < count; i++) {
    let mass = 0;
    let lastPositive = -1;
    for (let j = 0; j < remaining.length; j++) {
      const w = remaining[j] as number;
      if (w > 0) {
        mass += w;
        lastPositive = j;
      }
    }
    if (lastPositive < 0) {
      throw new InvalidParameterError("Insufficient probability mass to sample", "p", mass);
    }
    const u = __random() * mass;
    let cumulative = 0;
    let chosen = lastPositive; // rounding drift can leave u just past the last bucket
    for (let j = 0; j < remaining.length; j++) {
      const w = remaining[j] as number;
      if (w <= 0) continue;
      cumulative += w;
      if (u < cumulative) {
        chosen = j;
        break;
      }
    }
    out[i] = chosen;
    remaining[chosen] = 0;
  }
}

/**
 * Cumulative distribution for normalized weights. The tail starting at the last
 * positive-probability entry is pinned to exactly 1, so rounding drift can never
 * select a zero-probability category at the end.
 */
function buildCdf(weights: ArrayLike<number>): Float64Array {
  const k = weights.length;
  const cdf = new Float64Array(k);
  let cumulative = 0;
  let lastPositive = 0;
  for (let i = 0; i < k; i++) {
    const w = weights[i] as number;
    cumulative += w;
    cdf[i] = cumulative;
    if (w > 0) lastPositive = i;
  }
  for (let i = lastPositive; i < k; i++) cdf[i] = 1;
  return cdf;
}

/**
 * Read a non-negative, finite weight vector (any strides) and its sum.
 * Throws when an entry is negative or non-finite, or when the sum is not positive.
 */
function readWeights(
  t: Tensor,
  functionName: string,
  name: string
): { values: Float64Array; total: number; positive: number } {
  const values = readNumbers(t, functionName, name);
  let total = 0;
  let positive = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i] as number;
    if (!Number.isFinite(v) || v < 0) {
      throw new InvalidParameterError(
        `${name} must contain non-negative finite values; got ${v} at index ${i}`,
        name,
        v
      );
    }
    if (v > 0) positive++;
    total += v;
  }
  if (!(total > 0) || !Number.isFinite(total)) {
    throw new InvalidParameterError(`${name} must sum to a positive finite value`, name, total);
  }
  return { values, total, positive };
}

/**
 * Draw samples from a multinomial distribution.
 *
 * Counts are generated with a chain of conditional binomial draws (one per
 * category), so the cost does not grow with the number of trials `n`.
 *
 * @param n - Number of trials (non-negative integer)
 * @param pvals - Probabilities of each outcome, 1D tensor of shape (k,). Entries must be
 *   finite and non-negative with a positive sum; they are normalized by their sum.
 * @param size - Number of samples to draw (default: 1), or a shape of independent draws
 * @param opts - Options. `dtype` is int32, int64, float32 or float64. The default is the
 *   configured float dtype (float32 unless changed), or float64 when `n` exceeds 2^24 so that
 *   every count is exact.
 * @returns Tensor of shape (size, k), or (...size, k) when `size` is a shape, holding the
 *   count of each outcome. Every row sums to `n`.
 *
 * @throws {InvalidParameterError} When `n`, `pvals` or `size` is invalid
 * @throws {DTypeError} When `pvals` is a string tensor or `dtype` is unsupported
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
export function multinomial(
  n: number,
  pvals: Tensor,
  size: number | Shape = 1,
  opts: RandomOptions = {}
): Tensor {
  assertSafeInteger(n, "n");
  if (n < 0) {
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
  if (k === 0) {
    throw new InvalidParameterError("pvals must have at least one entry", "pvals", k);
  }
  const lead = resolveLeadingShape(size, "size");
  const draws = lead.reduce((acc, v) => acc * v, 1);
  const { values: probs } = readWeights(pvals, "multinomial()", "pvals");
  // Without an explicit dtype the counts keep the library's float default (as they always
  // have), widened to float64 when n is too large for float32 to hold every count exactly.
  const dtype: DType =
    opts.dtype ??
    (n > FLOAT32_EXACT_INT_MAX ? "float64" : resolveSamplingFloatDType(undefined, "multinomial"));
  if (dtype !== "int32" && dtype !== "int64" && dtype !== "float32" && dtype !== "float64") {
    throw new DTypeError("multinomial only supports int32, int64, float32 or float64 dtype");
  }
  if (dtype === "int32" && n > INT32_MAX) {
    throw new InvalidParameterError(`n must be <= ${INT32_MAX} for int32 output`, "n", n);
  }

  // suffix[i] = sum of probs[i..k): conditioning on "not in an earlier category".
  const suffix = new Float64Array(k + 1);
  for (let i = k - 1; i >= 0; i--) suffix[i] = (suffix[i + 1] as number) + (probs[i] as number);

  const counts = new Float64Array(draws * k);
  for (let s = 0; s < draws; s++) {
    let remaining = n;
    const base = s * k;
    for (let i = 0; i < k && remaining > 0; i++) {
      const tail = suffix[i] as number;
      if (!(tail > 0)) break;
      const conditional = Math.min(1, (probs[i] as number) / tail);
      const c = sampleBinomialScalar(remaining, conditional);
      counts[base + i] = c;
      remaining -= c;
    }
  }

  const shape = [...lead, k];
  const device = resolveDevice(opts.device);
  if (dtype === "float64") {
    return TensorClass.fromTypedArray({ data: counts, shape, dtype, device });
  }
  if (dtype === "float32") {
    return TensorClass.fromTypedArray({ data: Float32Array.from(counts), shape, dtype, device });
  }
  const data = allocateIntegerBuffer(dtype, counts.length);
  for (let i = 0; i < counts.length; i++) writeInteger(data, i, counts[i] as number);
  return TensorClass.fromTypedArray({ data, shape, dtype, device });
}

/**
 * Draw samples from a multivariate normal distribution.
 *
 * Uses the Cholesky factor of the covariance matrix: `x = mean + L z` with
 * `z ~ N(0, I)`. Singular (rank-deficient) covariances are supported.
 *
 * @param mean - Mean vector of shape (d,)
 * @param cov - Covariance matrix of shape (d, d); must be symmetric positive semi-definite
 * @param size - Number of samples (default: 1), or a shape of independent draws
 * @param opts - Options. `dtype` is float32 or float64 (default: the configured default dtype).
 * @returns Tensor of shape (size, d), or (...size, d) when `size` is a shape
 *
 * @throws {InvalidParameterError} When shapes do not match, `size` is invalid, an entry is
 *   not finite, or `cov` is not symmetric positive semi-definite. The check uses a relative
 *   tolerance of `max(1e-8, 10 * d * eps)`, where `eps` is the machine epsilon of the dtype of
 *   `cov`, so covariances computed in float32 are not rejected for rounding noise.
 *
 * @example
 * ```ts
 * import { multivariateNormal } from 'deepbox/random';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const mean = tensor([0, 0]);
 * const cov = tensor([[1, 0.5], [0.5, 1]]);
 * const samples = multivariateNormal(mean, cov, 100); // shape [100, 2]
 * ```
 *
 * @deprecated Prefer {@link multivariateNormal}.
 */
export function multivariate_normal(
  mean: Tensor,
  cov: Tensor,
  size: number | Shape = 1,
  opts: RandomOptions = {}
): Tensor {
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
  const lead = resolveLeadingShape(size, "size");
  const draws = lead.reduce((acc, v) => acc * v, 1);
  const dtype = resolveSamplingFloatDType(opts.dtype, "multivariate_normal");

  const mu = readNumbers(mean, "multivariate_normal()", "mean");
  const C = readNumbers(cov, "multivariate_normal()", "cov");
  for (let i = 0; i < d; i++) {
    if (!Number.isFinite(mu[i] as number)) {
      throw new InvalidParameterError("mean must contain finite values", "mean", mu[i]);
    }
  }
  let scale = 0;
  for (let i = 0; i < C.length; i++) {
    const v = C[i] as number;
    if (!Number.isFinite(v)) {
      throw new InvalidParameterError("cov must contain finite values", "cov", v);
    }
    scale = Math.max(scale, Math.abs(v));
  }
  // Relative tolerance for symmetry and semi-definiteness. A covariance computed in
  // float32 (the default dtype) carries rounding of order d * 2^-23, so the check
  // widens with the storage precision of `cov` instead of rejecting such inputs.
  const tol = scale * Math.max(1e-8, 10 * d * dtypeEpsilon(cov.dtype));
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < i; j++) {
      if (Math.abs((C[i * d + j] as number) - (C[j * d + i] as number)) > tol) {
        throw new InvalidParameterError("cov must be symmetric", "cov", cov.shape);
      }
    }
  }

  // Cholesky factor C = L L^T (lower triangle). A pivot within tolerance of zero
  // marks a singular direction: its column is zero. Anything more negative, or a
  // non-zero column under a zero pivot, means C is not positive semi-definite.
  const L = new Float64Array(d * d);
  for (let i = 0; i < d; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0;
      for (let k = 0; k < j; k++) {
        sum += (L[i * d + k] as number) * (L[j * d + k] as number);
      }
      const residual = (C[i * d + j] as number) - sum;
      if (i === j) {
        if (residual < -tol) {
          throw new InvalidParameterError("cov must be positive semi-definite", "cov", cov.shape);
        }
        L[i * d + j] = residual > 0 ? Math.sqrt(residual) : 0;
      } else {
        const diag = L[j * d + j] as number;
        if (diag > 0) {
          L[i * d + j] = residual / diag;
        } else if (Math.abs(residual) > tol) {
          throw new InvalidParameterError("cov must be positive semi-definite", "cov", cov.shape);
        }
      }
    }
  }

  // x = mu + L z with z ~ N(0, I); all normals come from one bulk fill.
  const z = new Float64Array(draws * d);
  __fillNormal(z, z.length);
  const data = allocateFloatBuffer(dtype, draws * d);
  for (let s = 0; s < draws; s++) {
    const base = s * d;
    for (let i = 0; i < d; i++) {
      let val = mu[i] as number;
      for (let j = 0; j <= i; j++) {
        val += (L[i * d + j] as number) * (z[base + j] as number);
      }
      data[base + i] = val;
    }
  }

  return TensorClass.fromTypedArray({
    data,
    shape: [...lead, d],
    dtype,
    device: resolveDevice(opts.device),
  });
}

/**
 * Draw samples from a Dirichlet distribution.
 *
 * Each row is a vector of independent Gamma(alpha_i, 1) draws divided by their sum.
 * When any concentration is below 1 the draws are combined in log space, so
 * very small concentrations give near one-hot rows instead of failing.
 *
 * @param alpha - Concentration parameters of shape (k,), all must be positive and finite
 * @param size - Number of samples (default: 1), or a shape of independent draws
 * @param opts - Options. `dtype` is float32 or float64 (default: the configured default dtype).
 * @returns Tensor of shape (size, k), or (...size, k) when `size` is a shape; each row sums to 1
 *
 * @throws {InvalidParameterError} When `alpha` is not 1D, is empty, or has a non-positive or
 *   non-finite entry
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
export function dirichlet(
  alpha: Tensor,
  size: number | Shape = 1,
  opts: RandomOptions = {}
): Tensor {
  if (alpha.ndim !== 1) {
    throw new InvalidParameterError(
      `alpha must be 1D; got ndim=${alpha.ndim}`,
      "alpha",
      alpha.shape
    );
  }
  const k = alpha.size;
  if (k === 0) {
    throw new InvalidParameterError("alpha must have at least one entry", "alpha", k);
  }
  const lead = resolveLeadingShape(size, "size");
  const draws = lead.reduce((acc, v) => acc * v, 1);
  const dtype = resolveSamplingFloatDType(opts.dtype, "dirichlet");
  const alphaArr = readNumbers(alpha, "dirichlet()", "alpha");
  let minAlpha = Infinity;
  for (let i = 0; i < k; i++) {
    const a = alphaArr[i] as number;
    if (!(a > 0) || !Number.isFinite(a)) {
      throw new InvalidParameterError(
        `All alpha values must be finite and > 0; got ${a} at index ${i}`,
        "alpha",
        a
      );
    }
    minAlpha = Math.min(minAlpha, a);
  }

  const data = allocateFloatBuffer(dtype, draws * k);
  const row = new Float64Array(k);
  for (let s = 0; s < draws; s++) {
    let total = 0;
    if (minAlpha < 1) {
      // Gamma(a < 1) draws can underflow to 0: normalize in log space instead.
      let maxLog = -Infinity;
      for (let i = 0; i < k; i++) {
        const lg = sampleLogGammaUnit(alphaArr[i] as number);
        row[i] = lg;
        if (lg > maxLog) maxLog = lg;
      }
      if (maxLog === -Infinity) {
        // Every draw underflowed (concentrations near the smallest subnormal): the
        // law collapses to a one-hot row with category weights proportional to alpha.
        let alphaTotal = 0;
        for (let i = 0; i < k; i++) alphaTotal += alphaArr[i] as number;
        let pick = __random() * alphaTotal;
        let chosen = k - 1;
        for (let i = 0; i < k; i++) {
          pick -= alphaArr[i] as number;
          if (pick < 0) {
            chosen = i;
            break;
          }
        }
        for (let i = 0; i < k; i++) row[i] = i === chosen ? 1 : 0;
        total = 1;
      } else {
        for (let i = 0; i < k; i++) {
          const w = Math.exp((row[i] as number) - maxLog);
          row[i] = w;
          total += w;
        }
      }
    } else {
      for (let i = 0; i < k; i++) {
        const g = sampleGammaUnit(alphaArr[i] as number);
        row[i] = g;
        total += g;
      }
    }
    for (let i = 0; i < k; i++) data[s * k + i] = (row[i] as number) / total;
  }

  return TensorClass.fromTypedArray({
    data,
    shape: [...lead, k],
    dtype,
    device: resolveDevice(opts.device),
  });
}

/**
 * Sample from a categorical distribution.
 *
 * Draws category indices with probability proportional to `probs`.
 *
 * @param probs - 1D tensor of non-negative weights (normalized internally)
 * @param numSamples - Number of samples to draw (default: 1)
 * @param replacement - Whether to sample with replacement (default: true). Without
 *   replacement each draw is proportional to the weight of the categories not yet drawn.
 * @returns 1D int32 tensor of sampled indices
 *
 * @throws {InvalidParameterError} When `probs` is not 1D, is empty, has a negative or non-finite
 *   entry or a non-positive sum, when `numSamples` is not a positive integer, or when sampling
 *   without replacement needs more categories than have non-zero probability
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
  if (numSamples < 1 || !Number.isSafeInteger(numSamples)) {
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

  const { values: p, total, positive } = readWeights(probs, "categorical()", "probs");
  for (let i = 0; i < k; i++) p[i] = (p[i] as number) / total;

  if (!replacement && numSamples > k) {
    throw new InvalidParameterError(
      `Cannot draw ${numSamples} samples without replacement from ${k} categories`,
      "numSamples",
      numSamples
    );
  }
  if (!replacement && numSamples > positive) {
    throw new InvalidParameterError(
      `Cannot draw ${numSamples} samples without replacement: only ${positive} categories have non-zero probability`,
      "numSamples",
      numSamples
    );
  }

  const result = new Int32Array(numSamples);
  if (replacement) {
    const cdf = buildCdf(p);
    for (let s = 0; s < numSamples; s++) result[s] = sampleFromCdf(cdf);
  } else {
    drawWeightedWithoutReplacement(p, numSamples, result);
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
 * Adds Gumbel(0, 1) noise to the logits and applies a temperature softmax, giving
 * approximate one-hot samples. The result is a plain tensor: no gradient flows
 * through it.
 *
 * @param logits - Unnormalized log-probabilities, shape (n_categories,) or (batch, n_categories).
 *   `-Infinity` masks a category; NaN and `+Infinity` are rejected.
 * @param tau - Temperature parameter (default: 1.0). Lower = more discrete.
 * @param hard - If true, returns hard one-hot vectors (default: false)
 * @returns float64 tensor of the same shape as logits with softmax probabilities (or one-hot rows)
 *
 * @throws {InvalidParameterError} When `logits` is not 1D/2D or has no categories, `tau` is not
 *   a positive finite number, a logit is NaN or `+Infinity`, or a row has no finite logit
 *
 * @deprecated Prefer {@link gumbelSoftmax}.
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
  if (nCat === 0) {
    throw new InvalidParameterError("gumbel_softmax requires at least one category", "logits", 0);
  }
  const totalSize = batchSize * nCat;
  const values = readNumbers(logits, "gumbel_softmax()", "logits");

  const result = new Float64Array(totalSize);
  const vals = new Float64Array(nCat);

  for (let b = 0; b < batchSize; b++) {
    // Add Gumbel noise to the logits (Gumbel(0,1) = -log(-log(U))).
    let maxVal = -Infinity;
    for (let j = 0; j < nCat; j++) {
      const logit = values[b * nCat + j] as number;
      if (Number.isNaN(logit) || logit === Infinity) {
        throw new InvalidParameterError(
          `logits must not contain NaN or +Infinity; got ${logit}`,
          "logits",
          logit
        );
      }
      const u = randomOpenUnit();
      const g = -Math.log(-Math.log(u));
      const v = (logit + g) / tau;
      vals[j] = v;
      if (v > maxVal) maxVal = v;
    }
    if (maxVal === -Infinity) {
      throw new InvalidParameterError(
        "each row of logits needs at least one finite value",
        "logits",
        b
      );
    }

    // Softmax with numerical stability.
    let sumExp = 0;
    let argmax = 0;
    let maxP = -Infinity;
    for (let j = 0; j < nCat; j++) {
      const e = Math.exp((vals[j] as number) - maxVal);
      vals[j] = e;
      sumExp += e;
      if (e > maxP) {
        maxP = e;
        argmax = j;
      }
    }

    if (hard) {
      // One-hot of the argmax.
      for (let j = 0; j < nCat; j++) {
        result[b * nCat + j] = j === argmax ? 1 : 0;
      }
    } else {
      for (let j = 0; j < nCat; j++) {
        result[b * nCat + j] = (vals[j] as number) / sumExp;
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
 * @param opts - Options. `dtype` is int32 (default) or int64.
 * @returns Tensor of 0s and 1s
 * @throws {InvalidParameterError} When `p` is not a finite number in [0, 1]
 */
export function bernoulli(p: number, shape: Shape = [], opts: RandomOptions = {}): Tensor {
  if (!Number.isFinite(p) || p < 0 || p > 1) {
    throw new InvalidParameterError("p must be in [0, 1]", "p", p);
  }
  const size = shapeToSize(shape);
  const dtype = resolveIntegerDType(opts.dtype, "bernoulli");
  const data = allocateIntegerBuffer(dtype, size);

  const us = new Float64Array(size);
  __fillUniform(us, size);
  for (let i = 0; i < size; i++) {
    writeInteger(data, i, (us[i] as number) < p ? 1 : 0);
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
 * @param opts - Options. `dtype` is int32 (default) or int64.
 * @returns Tensor of positive integers
 * @throws {InvalidParameterError} When `p` is not in (0, 1], or a sample does not fit the
 *   requested dtype (small `p` with int32 can exceed 2^31 - 1; use `dtype: "int64"`)
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
    // log1p keeps log(1 - p) accurate (and non-zero) for tiny p.
    const logQ = Math.log1p(-p);
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
  // module), then exponentiate in place. This avoids a module-boundary call per
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
 * import { studentT } from 'deepbox/random';
 * const x = studentT(10, [100]);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/random-distributions | Deepbox Distributions}
 *
 * @deprecated Prefer {@link studentT}.
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
 * import { fDistribution } from 'deepbox/random';
 * const x = fDistribution(5, 10, [100]);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/random-distributions | Deepbox Distributions}
 *
 * @deprecated Prefer {@link fDistribution}.
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
 * Uses inverse CDF: loc + scale * ln(2U) for U < 1/2, loc - scale * ln(2(1 - U)) otherwise.
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
    // Open-interval u keeps both logs finite (u = 0 would give -Infinity).
    const u = randomOpenUnit();
    data[i] = u < 0.5 ? loc + scale * Math.log(2 * u) : loc - scale * Math.log(2 * (1 - u));
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
 * import { negativeBinomial } from 'deepbox/random';
 * const x = negativeBinomial(5, 0.5, [100]);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/random-distributions | Deepbox Distributions}
 *
 * @deprecated Prefer {@link negativeBinomial}.
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
  if (!Number.isSafeInteger(N)) {
    throw new InvalidParameterError("ngood + nbad must be a safe integer", "ngood", ngood);
  }
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

/**
 * Random samples from the von Mises distribution (circular normal).
 *
 * @param mu - Mean direction in radians
 * @param kappa - Concentration parameter (must be >= 0)
 * @param shape - Output shape
 * @param opts - Options
 *
 * @remarks
 * - Uses Best & Fisher's algorithm for efficient sampling; for kappa > 1e6 it uses a
 *   wrapped normal with standard deviation 1/sqrt(kappa).
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
  } else if (kappa > 1e6) {
    // The Best & Fisher envelope loses all precision here (r - f cancels); the
    // distribution is indistinguishable from a wrapped normal with variance 1 / kappa.
    const sd = Math.sqrt(1 / kappa);
    for (let i = 0; i < size; i++) {
      const val = mu + sd * __normalRandom();
      data[i] = val - 2 * Math.PI * Math.floor((val + Math.PI) / (2 * Math.PI));
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
 * - Values are positive integers >= 1, truncated at the largest value of the dtype.
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
  // Values the dtype cannot hold are rejected (as NumPy does for int64) rather
  // than wrapped. For s close to 1 this truncates the extreme tail.
  const maxValue = dtype === "int32" ? INT32_MAX : INT64_LIMIT - 1024;

  for (let i = 0; i < size; i++) {
    while (true) {
      const u = randomOpenUnit();
      const v = __random();
      const x = Math.floor(u ** (-1 / (s - 1)));
      if (x < 1 || x > maxValue) continue;
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
 * import { fDistribution } from 'deepbox/random';
 * const x = fDistribution(5, 10, [100]);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/random-distributions | Deepbox Distributions}
 */
export const fDistribution = f_distribution;

/**
 * Sample from a categorical distribution using the Gumbel-Softmax trick.
 *
 * Adds Gumbel(0, 1) noise to the logits and applies a temperature softmax, giving
 * approximate one-hot samples. The result is a plain tensor: no gradient flows
 * through it.
 *
 * @param logits - Unnormalized log-probabilities, shape (n_categories,) or (batch, n_categories).
 *   `-Infinity` masks a category; NaN and `+Infinity` are rejected.
 * @param tau - Temperature parameter (default: 1.0). Lower = more discrete.
 * @param hard - If true, returns hard one-hot vectors (default: false)
 * @returns float64 tensor of the same shape as logits with softmax probabilities (or one-hot rows)
 *
 * @throws {InvalidParameterError} When `logits` is not 1D/2D or has no categories, `tau` is not
 *   a positive finite number, a logit is NaN or `+Infinity`, or a row has no finite logit
 */
export const gumbelSoftmax = gumbel_softmax;

/**
 * Draw samples from a multivariate normal distribution.
 *
 * Uses the Cholesky factor of the covariance matrix: `x = mean + L z` with
 * `z ~ N(0, I)`. Singular (rank-deficient) covariances are supported.
 *
 * @param mean - Mean vector of shape (d,)
 * @param cov - Covariance matrix of shape (d, d); must be symmetric positive semi-definite
 * @param size - Number of samples (default: 1), or a shape of independent draws
 * @param opts - Options. `dtype` is float32 or float64 (default: the configured default dtype).
 * @returns Tensor of shape (size, d), or (...size, d) when `size` is a shape
 *
 * @throws {InvalidParameterError} When shapes do not match, `size` is invalid, an entry is
 *   not finite, or `cov` is not symmetric positive semi-definite. The check uses a relative
 *   tolerance of `max(1e-8, 10 * d * eps)`, where `eps` is the machine epsilon of the dtype of
 *   `cov`, so covariances computed in float32 are not rejected for rounding noise.
 *
 * @example
 * ```ts
 * import { multivariateNormal } from 'deepbox/random';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const mean = tensor([0, 0]);
 * const cov = tensor([[1, 0.5], [0.5, 1]]);
 * const samples = multivariateNormal(mean, cov, 100); // shape [100, 2]
 * ```
 */
export const multivariateNormal = multivariate_normal;

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
 * import { negativeBinomial } from 'deepbox/random';
 * const x = negativeBinomial(5, 0.5, [100]);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/random-distributions | Deepbox Distributions}
 */
export const negativeBinomial = negative_binomial;

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
 * import { studentT } from 'deepbox/random';
 * const x = studentT(10, [100]);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/random-distributions | Deepbox Distributions}
 */
export const studentT = student_t;
