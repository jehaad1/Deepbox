/**
 * Weight initialization functions for neural network layers.
 *
 * Provides the standard initialization strategies used in deep learning:
 * - Xavier/Glorot (for sigmoid/tanh activations)
 * - Kaiming/He (for ReLU activations)
 * - Uniform, Normal, Truncated normal, Constant, Zeros, Ones
 * - Orthogonal, Sparse, Identity
 *
 * All functions modify the tensor in-place and return it for chaining. They
 * accept a `Tensor` or a `GradTensor` (the underlying tensor is filled), work on
 * non-contiguous views, and draw from the global generator, so
 * `manualSeed(...)` from `deepbox/random` makes them reproducible.
 *
 * Random fills require a floating-point tensor (`float16`, `bfloat16`,
 * `float32` or `float64`); half-precision values are rounded to the nearest
 * representable number.
 *
 * @example
 * ```ts
 * import { Linear, xavierUniform_, zeros_ } from 'deepbox/nn';
 *
 * const layer = new Linear(128, 64);
 * xavierUniform_(layer.getWeight());
 * const bias = layer.getBias();
 * if (bias) zeros_(bias);
 * ```
 *
 * @module nn/init
 * @see {@link https://deepbox.dev/docs/nn-layers | Deepbox Weight Initialization}
 */

import { DTypeError, InvalidParameterError } from "../core";
import { type AnyTensor, GradTensor, type Tensor } from "../ndarray";
import { roundToBFloat16, roundToFloat16 } from "../ndarray/tensor/float16";
import { isContiguous, offsetFromFlatIndex } from "../ndarray/tensor/strides";
import { __random, __randomBelow } from "../random/random";

/**
 * Create a uniform random number generator on [0, 1).
 *
 * Without a seed this returns the global generator (which honors
 * `manualSeed`). With a seed it returns an independent, deterministic
 * mulberry32 stream; fractional seeds are truncated and negative seeds wrap
 * modulo 2^32.
 *
 * @param seed - Optional finite seed
 * @returns A function returning the next uniform sample
 */
function makeRng(seed?: number): () => number {
  if (seed !== undefined) {
    if (!Number.isFinite(seed)) {
      throw new InvalidParameterError(
        `seed must be a finite number; received ${seed}`,
        "seed",
        seed
      );
    }
    let s = Math.trunc(seed) >>> 0;
    return () => {
      s = (s + 0x6d2b79f5) >>> 0;
      let t = Math.imul(s ^ (s >>> 15), s | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  return __random;
}

// Box-Muller transform for normal distribution
function boxMuller(rng: () => number): number {
  let u1: number;
  do {
    u1 = rng();
  } while (u1 === 0);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// ---------------------------------------------------------------------------
// Validation and storage helpers
// ---------------------------------------------------------------------------

function unwrap(value: AnyTensor, fn: string): Tensor {
  if (typeof value !== "object" || value === null) {
    throw new InvalidParameterError(`${fn} expects a Tensor or GradTensor`, "tensor", value);
  }
  return GradTensor.isGradTensor(value) ? value.tensor : value;
}

function requireFinite(name: string, value: number): void {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new InvalidParameterError(
      `${name} must be a finite number; received ${String(value)}`,
      name,
      value
    );
  }
}

function requireNonNegative(name: string, value: number): void {
  requireFinite(name, value);
  if (value < 0) {
    throw new InvalidParameterError(`${name} must be >= 0; received ${value}`, name, value);
  }
}

function requireFloat(t: Tensor, fn: string): void {
  if (
    t.dtype !== "float32" &&
    t.dtype !== "float64" &&
    t.dtype !== "float16" &&
    t.dtype !== "bfloat16"
  ) {
    throw new DTypeError(`${fn} requires a floating-point tensor; received dtype ${t.dtype}`);
  }
}

/** Rounding applied to a value before it is stored, for half-precision dtypes only. */
function roundFor(t: Tensor): ((v: number) => number) | undefined {
  if (t.dtype === "float16") return roundToFloat16;
  if (t.dtype === "bfloat16") return roundToBFloat16;
  return undefined;
}

/** Storage offset of every logical element, or `null` when `offset + i` is correct. */
function storageOffsets(t: Tensor): Float64Array | null {
  if (isContiguous(t.shape, t.strides)) return null;
  const n = t.size;
  const logical = new Array<number>(t.ndim);
  let acc = 1;
  for (let axis = t.ndim - 1; axis >= 0; axis--) {
    logical[axis] = acc;
    acc *= t.shape[axis] ?? 1;
  }
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = offsetFromFlatIndex(i, logical, t.strides, t.offset);
  }
  return out;
}

/** Fill a floating-point tensor in logical (row-major) order with samples from `gen`. */
function fillWith(t: Tensor, fn: string, gen: () => number): void {
  requireFloat(t, fn);
  const data = t.data as Float32Array | Float64Array;
  const n = t.size;
  const round = roundFor(t);
  const offsets = storageOffsets(t);
  if (offsets === null) {
    const base = t.offset;
    if (round) {
      for (let i = 0; i < n; i++) data[base + i] = round(gen());
    } else {
      for (let i = 0; i < n; i++) data[base + i] = gen();
    }
  } else {
    for (let i = 0; i < n; i++) {
      const v = gen();
      data[offsets[i] as number] = round ? round(v) : v;
    }
  }
}

/** Write `values` (logical row-major order, length `t.size`) into a floating-point tensor. */
function scatter(t: Tensor, values: Float64Array): void {
  const data = t.data as Float32Array | Float64Array;
  const n = t.size;
  const round = roundFor(t);
  const offsets = storageOffsets(t);
  for (let i = 0; i < n; i++) {
    const v = values[i] as number;
    data[offsets === null ? t.offset + i : (offsets[i] as number)] = round ? round(v) : v;
  }
}

function calculateFanInOut(tensor: AnyTensor): { fanIn: number; fanOut: number } {
  const ndim = tensor.ndim;
  if (ndim < 1) {
    throw new InvalidParameterError(
      "Fan in/out cannot be computed for scalar tensors",
      "tensor",
      tensor.shape
    );
  }
  // A 1-D tensor (for example a bias) uses its length for both fans.
  if (ndim === 1) {
    return { fanIn: tensor.shape[0] ?? 1, fanOut: tensor.shape[0] ?? 1 };
  }
  if (ndim === 2) {
    return { fanIn: tensor.shape[1] ?? 1, fanOut: tensor.shape[0] ?? 1 };
  }
  // Conv: [outChannels, inChannels, ...kernelSize]
  const outChannels = tensor.shape[0] ?? 1;
  const inChannels = tensor.shape[1] ?? 1;
  let receptiveFieldSize = 1;
  for (let i = 2; i < ndim; i++) {
    receptiveFieldSize *= tensor.shape[i] ?? 1;
  }
  return {
    fanIn: inChannels * receptiveFieldSize,
    fanOut: outChannels * receptiveFieldSize,
  };
}

/**
 * Recommended gain for a nonlinearity (mirrors `torch.nn.init.calculate_gain`).
 *
 * @param nonlinearity - One of `linear`, `conv1d`, `conv2d`, `conv3d`,
 *   `conv_transpose1d`, `conv_transpose2d`, `conv_transpose3d`, `sigmoid`,
 *   `tanh`, `relu`, `leaky_relu`, `selu`
 * @param param - Negative slope for `leaky_relu` (default: 0.01)
 * @returns The gain factor
 * @throws {InvalidParameterError} For an unknown nonlinearity or a non-finite slope
 */
function calculateGain(nonlinearity: string, param?: number): number {
  switch (nonlinearity) {
    case "linear":
    case "conv1d":
    case "conv2d":
    case "conv3d":
    case "conv_transpose1d":
    case "conv_transpose2d":
    case "conv_transpose3d":
    case "sigmoid":
      return 1;
    case "tanh":
      return 5.0 / 3;
    case "relu":
      return Math.SQRT2;
    case "leaky_relu": {
      const slope = param ?? 0.01;
      requireFinite("negative slope", slope);
      return Math.sqrt(2.0 / (1 + slope * slope));
    }
    case "selu":
      return 3.0 / 4;
    default:
      throw new InvalidParameterError(
        `Unsupported nonlinearity: ${nonlinearity}`,
        "nonlinearity",
        nonlinearity
      );
  }
}

function resolveFan(tensor: Tensor, mode: "fan_in" | "fan_out"): number {
  if (mode !== "fan_in" && mode !== "fan_out") {
    throw new InvalidParameterError(
      `mode must be "fan_in" or "fan_out"; received "${String(mode)}"`,
      "mode",
      mode
    );
  }
  const { fanIn, fanOut } = calculateFanInOut(tensor);
  return mode === "fan_in" ? fanIn : fanOut;
}

/**
 * Fill tensor with values drawn from a uniform distribution U(low, high).
 *
 * @param tensor - Floating-point tensor to fill in-place
 * @param low - Lower bound (default: 0)
 * @param high - Upper bound (default: 1)
 * @returns The modified tensor
 * @throws {InvalidParameterError} If a bound is not finite or `low > high`
 * @throws {DTypeError} If the tensor is not floating-point
 */
export function uniform_<T extends AnyTensor>(tensor: T, low = 0, high = 1): T {
  requireFinite("low", low);
  requireFinite("high", high);
  if (low > high) {
    throw new InvalidParameterError(
      `low must be <= high; received low=${low}, high=${high}`,
      "low",
      low
    );
  }
  const t = unwrap(tensor, "uniform_");
  const span = high - low;
  fillWith(t, "uniform_", () => low + span * __random());
  return tensor;
}

/**
 * Fill tensor with values drawn from a normal distribution N(mean, std²).
 *
 * @param tensor - Floating-point tensor to fill in-place
 * @param mean - Mean of the distribution (default: 0)
 * @param std - Standard deviation, must be >= 0 (default: 1)
 * @returns The modified tensor
 * @throws {InvalidParameterError} If `mean` is not finite or `std` is negative or not finite
 * @throws {DTypeError} If the tensor is not floating-point
 */
export function normal_<T extends AnyTensor>(tensor: T, mean = 0, std = 1): T {
  requireFinite("mean", mean);
  requireNonNegative("std", std);
  const t = unwrap(tensor, "normal_");
  fillWith(t, "normal_", () => mean + std * boxMuller(__random));
  return tensor;
}

/**
 * Fill tensor with values drawn from a normal distribution N(mean, std²)
 * truncated to [a, b] (mirrors `torch.nn.init.trunc_normal_`).
 *
 * Uses rejection sampling, so the result is exactly the truncated normal.
 *
 * @param tensor - Floating-point tensor to fill in-place
 * @param mean - Mean of the underlying normal (default: 0)
 * @param std - Standard deviation of the underlying normal, must be >= 0 (default: 1)
 * @param a - Lower cut-off (default: -2)
 * @param b - Upper cut-off (default: 2)
 * @returns The modified tensor
 * @throws {InvalidParameterError} If a parameter is not finite, `std < 0`, `a > b`, or the
 *   interval lies so far in the tail that sampling would not finish
 * @throws {DTypeError} If the tensor is not floating-point
 */
export function trunc_normal_<T extends AnyTensor>(tensor: T, mean = 0, std = 1, a = -2, b = 2): T {
  requireFinite("mean", mean);
  requireNonNegative("std", std);
  requireFinite("a", a);
  requireFinite("b", b);
  if (a > b) {
    throw new InvalidParameterError(`a must be <= b; received a=${a}, b=${b}`, "a", a);
  }
  const t = unwrap(tensor, "trunc_normal_");
  const maxAttempts = 100000;
  fillWith(t, "trunc_normal_", () => {
    if (std === 0) return Math.min(b, Math.max(a, mean));
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      const v = mean + std * boxMuller(__random);
      if (v >= a && v <= b) return v;
    }
    throw new InvalidParameterError(
      `trunc_normal_ could not sample from [${a}, ${b}] with mean=${mean}, std=${std}; ` +
        "the interval is too far in the tail of the distribution",
      "a",
      a
    );
  });
  return tensor;
}

/**
 * Fill tensor with a constant value.
 *
 * Works for every numeric dtype. Integer and `bool` tensors store the
 * truncated value (`bool`: non-zero is true); `int64` converts through
 * `BigInt`. Half-precision tensors store the rounded value.
 *
 * @param tensor - Tensor to fill in-place
 * @param val - Value to fill with
 * @returns The modified tensor
 * @throws {DTypeError} For string tensors
 * @throws {InvalidParameterError} If `val` is not finite for an integer or `int64` tensor
 */
export function constant_<T extends AnyTensor>(tensor: T, val: number): T {
  const t = unwrap(tensor, "constant_");
  if (t.dtype === "string") {
    throw new DTypeError("constant_ does not support string tensors");
  }
  const n = t.size;
  const offsets = storageOffsets(t);
  const data = t.data as Float32Array | Float64Array | Int32Array | Uint8Array | BigInt64Array;
  if (data instanceof BigInt64Array) {
    if (!Number.isFinite(val)) {
      throw new InvalidParameterError(
        `val must be finite for dtype int64; received ${val}`,
        "val",
        val
      );
    }
    const big = BigInt(Math.trunc(val));
    for (let i = 0; i < n; i++)
      data[offsets === null ? t.offset + i : (offsets[i] as number)] = big;
    return tensor;
  }
  let value = val;
  if (t.dtype === "bool") {
    value = val !== 0 ? 1 : 0;
  } else if (t.dtype === "int32" || t.dtype === "uint8") {
    if (!Number.isFinite(val)) {
      throw new InvalidParameterError(
        `val must be finite for dtype ${t.dtype}; received ${val}`,
        "val",
        val
      );
    }
    value = Math.trunc(val);
  } else {
    const round = roundFor(t);
    if (round) value = round(val);
  }
  for (let i = 0; i < n; i++)
    data[offsets === null ? t.offset + i : (offsets[i] as number)] = value;
  return tensor;
}

/**
 * Fill tensor with zeros.
 *
 * @param tensor - Tensor to fill in-place
 * @returns The modified tensor
 */
export function zeros_<T extends AnyTensor>(tensor: T): T {
  return constant_(tensor, 0);
}

/**
 * Fill tensor with ones.
 *
 * @param tensor - Tensor to fill in-place
 * @returns The modified tensor
 */
export function ones_<T extends AnyTensor>(tensor: T): T {
  return constant_(tensor, 1);
}

/**
 * Fill a 2-D tensor with the identity matrix (ones on the main diagonal, zeros
 * elsewhere). The tensor may be rectangular.
 *
 * @param tensor - 2-D tensor to fill in-place
 * @returns The modified tensor
 * @throws {InvalidParameterError} If the tensor is not 2-D
 */
export function eye_<T extends AnyTensor>(tensor: T): T {
  const t = unwrap(tensor, "eye_");
  if (t.ndim !== 2) {
    throw new InvalidParameterError("eye_ requires a 2D tensor", "tensor", t.shape);
  }
  constant_(t, 0);
  const rows = t.shape[0] ?? 0;
  const cols = t.shape[1] ?? 0;
  const one = t.dtype === "int64" ? 1n : 1;
  const data = t.data as Float32Array | Float64Array | Int32Array | Uint8Array | BigInt64Array;
  const rs = t.strides[0] ?? 0;
  const cs = t.strides[1] ?? 0;
  for (let i = 0; i < Math.min(rows, cols); i++) {
    (data as unknown as Array<number | bigint>)[t.offset + i * rs + i * cs] = one;
  }
  return tensor;
}

/**
 * Fill tensor using Xavier uniform initialization.
 *
 * Draws from U(-a, a) where a = gain * sqrt(6 / (fan_in + fan_out)).
 * Recommended for sigmoid/tanh activations.
 *
 * @param tensor - Floating-point tensor to fill in-place (at least 1-D)
 * @param gain - Scaling factor, must be >= 0 (default: 1.0)
 * @returns The modified tensor
 * @throws {InvalidParameterError} For a scalar tensor or an invalid gain
 * @deprecated Prefer {@link xavierUniform_}.
 */
export function xavier_uniform_<T extends AnyTensor>(tensor: T, gain = 1.0): T {
  requireNonNegative("gain", gain);
  const t = unwrap(tensor, "xavier_uniform_");
  const { fanIn, fanOut } = calculateFanInOut(t);
  if (t.size === 0) return tensor;
  const a = gain * Math.sqrt(6.0 / (fanIn + fanOut));
  return uniform_(tensor, -a, a);
}

/**
 * Fill tensor using Xavier normal initialization.
 *
 * Draws from N(0, std²) where std = gain * sqrt(2 / (fan_in + fan_out)).
 * Recommended for sigmoid/tanh activations.
 *
 * @param tensor - Floating-point tensor to fill in-place (at least 1-D)
 * @param gain - Scaling factor, must be >= 0 (default: 1.0)
 * @returns The modified tensor
 * @throws {InvalidParameterError} For a scalar tensor or an invalid gain
 * @deprecated Prefer {@link xavierNormal_}.
 */
export function xavier_normal_<T extends AnyTensor>(tensor: T, gain = 1.0): T {
  requireNonNegative("gain", gain);
  const t = unwrap(tensor, "xavier_normal_");
  const { fanIn, fanOut } = calculateFanInOut(t);
  if (t.size === 0) return tensor;
  const std = gain * Math.sqrt(2.0 / (fanIn + fanOut));
  return normal_(tensor, 0, std);
}

/**
 * Fill tensor using Kaiming uniform initialization (He initialization).
 *
 * Draws from U(-bound, bound) where bound = gain * sqrt(3 / fan).
 * Recommended for ReLU activations.
 *
 * @param tensor - Floating-point tensor to fill in-place (at least 1-D)
 * @param a - Negative slope of rectifier (used when `nonlinearity` is 'leaky_relu', default: 0)
 * @param mode - 'fan_in' (default) or 'fan_out'
 * @param nonlinearity - Activation function name (default: 'leaky_relu')
 * @returns The modified tensor
 * @throws {InvalidParameterError} For a scalar tensor, an unknown mode or an unknown nonlinearity
 * @deprecated Prefer {@link kaimingUniform_}.
 */
export function kaiming_uniform_<T extends AnyTensor>(
  tensor: T,
  a = 0,
  mode: "fan_in" | "fan_out" = "fan_in",
  nonlinearity = "leaky_relu"
): T {
  const t = unwrap(tensor, "kaiming_uniform_");
  const fan = resolveFan(t, mode);
  const gain = calculateGain(nonlinearity, a);
  if (t.size === 0) return tensor;
  const std = gain / Math.sqrt(fan);
  const bound = Math.sqrt(3.0) * std;
  return uniform_(tensor, -bound, bound);
}

/**
 * Fill tensor using Kaiming normal initialization (He initialization).
 *
 * Draws from N(0, std²) where std = gain / sqrt(fan).
 * Recommended for ReLU activations.
 *
 * @param tensor - Floating-point tensor to fill in-place (at least 1-D)
 * @param a - Negative slope of rectifier (used when `nonlinearity` is 'leaky_relu', default: 0)
 * @param mode - 'fan_in' (default) or 'fan_out'
 * @param nonlinearity - Activation function name (default: 'leaky_relu')
 * @returns The modified tensor
 * @throws {InvalidParameterError} For a scalar tensor, an unknown mode or an unknown nonlinearity
 * @deprecated Prefer {@link kaimingNormal_}.
 */
export function kaiming_normal_<T extends AnyTensor>(
  tensor: T,
  a = 0,
  mode: "fan_in" | "fan_out" = "fan_in",
  nonlinearity = "leaky_relu"
): T {
  const t = unwrap(tensor, "kaiming_normal_");
  const fan = resolveFan(t, mode);
  const gain = calculateGain(nonlinearity, a);
  if (t.size === 0) return tensor;
  const std = gain / Math.sqrt(fan);
  return normal_(tensor, 0, std);
}

/**
 * Fill tensor with a (semi-)orthogonal matrix.
 *
 * A normal random matrix is orthonormalized with Gram-Schmidt (applied twice
 * for accuracy). With more columns than rows the rows are orthonormal; with
 * more rows than columns the columns are. Tensors with more than two
 * dimensions are treated as a `[shape[0], prod(shape[1:])]` matrix, like
 * PyTorch. The result is scaled by `gain`.
 *
 * @param tensor - Floating-point tensor of at least 2 dimensions, filled in-place
 * @param gain - Scaling factor (default: 1.0)
 * @returns The modified tensor
 * @throws {InvalidParameterError} If the tensor has fewer than 2 dimensions or `gain` is not finite
 */
export function orthogonal_<T extends AnyTensor>(tensor: T, gain = 1.0): T {
  const t = unwrap(tensor, "orthogonal_");
  if (t.ndim < 2) {
    throw new InvalidParameterError("orthogonal_ requires at least 2D tensor", "tensor", t.shape);
  }
  requireFinite("gain", gain);
  requireFloat(t, "orthogonal_");
  const rows = t.shape[0] ?? 1;
  let cols = 1;
  for (let i = 1; i < t.ndim; i++) {
    cols *= t.shape[i] ?? 1;
  }
  if (rows === 0 || cols === 0) return tensor;

  // `count` orthonormal vectors of length `len`. Wide matrices (rows <= cols)
  // keep them as rows; tall matrices keep them as columns.
  const wide = rows <= cols;
  const count = wide ? rows : cols;
  const len = wide ? cols : rows;
  const basis = new Float64Array(count * len);

  for (let k = 0; k < count; k++) {
    const v = basis.subarray(k * len, (k + 1) * len);
    let accepted = false;
    for (let attempt = 0; attempt < 10 && !accepted; attempt++) {
      for (let j = 0; j < len; j++) v[j] = boxMuller(__random);
      // Two Gram-Schmidt passes keep the vectors orthogonal to machine precision.
      for (let pass = 0; pass < 2; pass++) {
        for (let u = 0; u < k; u++) {
          const prev = basis.subarray(u * len, (u + 1) * len);
          let dot = 0;
          for (let j = 0; j < len; j++) dot += (v[j] as number) * (prev[j] as number);
          for (let j = 0; j < len; j++) v[j] = (v[j] as number) - dot * (prev[j] as number);
        }
      }
      let norm = 0;
      for (let j = 0; j < len; j++) norm += (v[j] as number) * (v[j] as number);
      norm = Math.sqrt(norm);
      if (norm > 1e-10) {
        for (let j = 0; j < len; j++) v[j] = (v[j] as number) / norm;
        accepted = true;
      }
    }
    if (!accepted) {
      throw new InvalidParameterError(
        "orthogonal_ failed to draw a linearly independent vector",
        "tensor",
        t.shape
      );
    }
  }

  const out = new Float64Array(rows * cols);
  if (wide) {
    for (let i = 0; i < rows * cols; i++) out[i] = gain * (basis[i] as number);
  } else {
    for (let j = 0; j < cols; j++) {
      for (let i = 0; i < rows; i++) {
        out[i * cols + j] = gain * (basis[j * len + i] as number);
      }
    }
  }
  scatter(t, out);
  return tensor;
}

/**
 * Fill a 2-D tensor as a sparse matrix with normally distributed non-zero entries.
 *
 * In every column, `ceil(sparsity * rows)` randomly chosen entries are set to
 * zero and the rest are drawn from N(0, std²), as in `torch.nn.init.sparse_`.
 *
 * @param tensor - Floating-point 2-D tensor to fill in-place
 * @param sparsity - Fraction of elements in each column to be zero, in [0, 1] (default: 0.1)
 * @param std - Standard deviation of the non-zero entries, must be >= 0 (default: 0.01)
 * @returns The modified tensor
 * @throws {InvalidParameterError} If the tensor is not 2-D or an argument is out of range
 */
export function sparse_<T extends AnyTensor>(tensor: T, sparsity = 0.1, std = 0.01): T {
  const t = unwrap(tensor, "sparse_");
  if (t.ndim !== 2) {
    throw new InvalidParameterError("sparse_ requires 2D tensor", "tensor", t.shape);
  }
  requireFinite("sparsity", sparsity);
  if (sparsity < 0 || sparsity > 1) {
    throw new InvalidParameterError(
      `sparsity must be in [0, 1]; received ${sparsity}`,
      "sparsity",
      sparsity
    );
  }
  requireNonNegative("std", std);
  requireFloat(t, "sparse_");
  const rows = t.shape[0] ?? 0;
  const cols = t.shape[1] ?? 0;
  if (rows === 0 || cols === 0) return tensor;

  const numZeros = Math.ceil(sparsity * rows);
  const out = new Float64Array(rows * cols);
  const indices = new Int32Array(rows);
  for (let j = 0; j < cols; j++) {
    for (let i = 0; i < rows; i++) indices[i] = i;
    // Partial Fisher-Yates: the first `numZeros` entries are the rows set to zero.
    for (let k = 0; k < numZeros; k++) {
      const pick = k + __randomBelow(__random, rows - k);
      const tmp = indices[k] as number;
      indices[k] = indices[pick] as number;
      indices[pick] = tmp;
    }
    for (let k = numZeros; k < rows; k++) {
      out[(indices[k] as number) * cols + j] = std * boxMuller(__random);
    }
  }
  scatter(t, out);
  return tensor;
}

export { calculateFanInOut, calculateGain, makeRng };

// ---------------------------------------------------------------------------
// Canonical camelCase aliases
//
// The snake_case spellings above mirror PyTorch and remain exported for
// backward compatibility. These camelCase aliases are the recommended names on
// Deepbox's public surface; each refers to the exact same in-place function.
// The trailing-underscore forms preserve PyTorch's in-place convention; the
// no-underscore forms are provided as a convenience.
// ---------------------------------------------------------------------------

/**
 * Fill tensor using Kaiming normal initialization (He initialization).
 *
 * Draws from N(0, std²) where std = gain / sqrt(fan).
 * Recommended for ReLU activations.
 *
 * @param tensor - Floating-point tensor to fill in-place (at least 1-D)
 * @param a - Negative slope of rectifier (used when `nonlinearity` is 'leaky_relu', default: 0)
 * @param mode - 'fan_in' (default) or 'fan_out'
 * @param nonlinearity - Activation function name (default: 'leaky_relu')
 * @returns The modified tensor
 * @throws {InvalidParameterError} For a scalar tensor, an unknown mode or an unknown nonlinearity
 */
export const kaimingNormal_ = kaiming_normal_;
/** Convenience alias of {@link kaiming_normal_} without the trailing underscore. */
export const kaimingNormal = kaiming_normal_;
/**
 * Fill tensor using Kaiming uniform initialization (He initialization).
 *
 * Draws from U(-bound, bound) where bound = gain * sqrt(3 / fan).
 * Recommended for ReLU activations.
 *
 * @param tensor - Floating-point tensor to fill in-place (at least 1-D)
 * @param a - Negative slope of rectifier (used when `nonlinearity` is 'leaky_relu', default: 0)
 * @param mode - 'fan_in' (default) or 'fan_out'
 * @param nonlinearity - Activation function name (default: 'leaky_relu')
 * @returns The modified tensor
 * @throws {InvalidParameterError} For a scalar tensor, an unknown mode or an unknown nonlinearity
 */
export const kaimingUniform_ = kaiming_uniform_;
/** Convenience alias of {@link kaiming_uniform_} without the trailing underscore. */
export const kaimingUniform = kaiming_uniform_;
/**
 * Fill tensor using Xavier normal initialization.
 *
 * Draws from N(0, std²) where std = gain * sqrt(2 / (fan_in + fan_out)).
 * Recommended for sigmoid/tanh activations.
 *
 * @param tensor - Floating-point tensor to fill in-place (at least 1-D)
 * @param gain - Scaling factor, must be >= 0 (default: 1.0)
 * @returns The modified tensor
 * @throws {InvalidParameterError} For a scalar tensor or an invalid gain
 */
export const xavierNormal_ = xavier_normal_;
/** Convenience alias of {@link xavier_normal_} without the trailing underscore. */
export const xavierNormal = xavier_normal_;
/**
 * Fill tensor using Xavier uniform initialization.
 *
 * Draws from U(-a, a) where a = gain * sqrt(6 / (fan_in + fan_out)).
 * Recommended for sigmoid/tanh activations.
 *
 * @param tensor - Floating-point tensor to fill in-place (at least 1-D)
 * @param gain - Scaling factor, must be >= 0 (default: 1.0)
 * @returns The modified tensor
 * @throws {InvalidParameterError} For a scalar tensor or an invalid gain
 */
export const xavierUniform_ = xavier_uniform_;
/** Convenience alias of {@link xavier_uniform_} without the trailing underscore. */
export const xavierUniform = xavier_uniform_;
/** Canonical camelCase alias of {@link trunc_normal_} (in-place). */
export const truncNormal_ = trunc_normal_;
/** Convenience alias of {@link orthogonal_} without the trailing underscore. */
export const orthogonal = orthogonal_;
/** Convenience alias of {@link zeros_} without the trailing underscore. */
export const zeros = zeros_;
/** Convenience alias of {@link ones_} without the trailing underscore. */
export const ones = ones_;
/** Convenience alias of {@link constant_} without the trailing underscore. */
export const constant = constant_;
