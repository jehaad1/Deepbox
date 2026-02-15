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

  for (let i = 0; i < size; i++) {
    data[i] = __random();
  }

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

  for (let i = 0; i < size; i++) {
    data[i] = __normalRandom();
  }

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

  for (let i = 0; i < size; i++) {
    const sample = randomIntBelow(range) + low;
    writeInteger(data, i, sample);
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

  for (let i = 0; i < size; i++) {
    data[i] = __random() * range + low;
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

  for (let i = 0; i < size; i++) {
    data[i] = __normalRandom() * std + mean;
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
    for (let i = 0; i < size; i++) {
      const sample = binomialSmallMean(n, logQ);
      writeInteger(data, i, flip ? n - sample : sample);
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

  for (let i = 0; i < size; i++) {
    const u = randomOpenUnit();
    data[i] = -scale * Math.log(u);
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
    case "float32":
      return new Float32Array(size);
    case "float64":
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
    for (let i = 0; i < outputSize; i++) {
      indices[i] = randomIntBelow(n);
    }
  } else {
    if (outputSize > n) {
      throw new InvalidParameterError(
        "Cannot sample without replacement when size > population",
        "size",
        outputSize
      );
    }
    const pool = Array.from({ length: n }, (_, i) => i);
    for (let i = 0; i < outputSize; i++) {
      const j = randomIntBelow(n - i) + i;
      const poolI = pool[i];
      const poolJ = pool[j];
      if (poolI === undefined || poolJ === undefined) {
        throw new InvalidParameterError("Internal error: pool index out of bounds", "pool", {
          i,
          j,
          n,
        });
      }
      pool[i] = poolJ;
      pool[j] = poolI;
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

  // Fisher–Yates shuffle using the same RNG as the rest of the module.
  // Split into two branches to maintain type safety without assertions.
  if (data instanceof BigInt64Array) {
    for (let i = n - 1; i > 0; i--) {
      const j = randomIntBelow(i + 1);
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
      const j = randomIntBelow(i + 1);
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
    const indices = Array.from({ length: n }, (_, i) => i);

    for (let i = n - 1; i > 0; i--) {
      const j = randomIntBelow(i + 1);
      const indicesI = indices[i];
      const indicesJ = indices[j];
      if (indicesI === undefined || indicesJ === undefined) {
        throw new InvalidParameterError("Internal error: indices out of bounds", "indices", {
          i,
          j,
          n,
        });
      }
      indices[i] = indicesJ;
      indices[j] = indicesI;
    }

    return tensor(indices, { dtype: "int32" });
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
