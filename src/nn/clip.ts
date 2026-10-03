/**
 * Gradient clipping utilities for neural network training.
 *
 * Prevents exploding gradients by capping gradient magnitudes,
 * essential for training RNNs and deep networks.
 *
 * @example
 * ```ts
 * import { clipGradNorm_, clipGradValue_ } from 'deepbox/nn';
 *
 * // Clip by global L2 norm (most common)
 * const totalNorm = clipGradNorm_(model.parameters(), 1.0);
 *
 * // Clip each gradient element to [-0.5, 0.5]
 * clipGradValue_(model.parameters(), 0.5);
 * ```
 *
 * @module nn/clip
 * @see {@link https://deepbox.dev/docs/nn-module | Deepbox Gradient Clipping}
 */

import { DataValidationError, DTypeError, InvalidParameterError } from "../core";
import { GradTensor, type Tensor } from "../ndarray";
import { roundToBFloat16, roundToFloat16 } from "../ndarray/tensor/float16";
import { isContiguous, offsetFromFlatIndex } from "../ndarray/tensor/strides";

/** A gradient buffer together with the storage offset of each logical element. */
type GradView = {
  readonly tensor: Tensor;
  readonly data: Float32Array | Float64Array;
  /** Storage offset per logical element, or `null` when `offset + i` applies. */
  readonly offsets: Float64Array | null;
  readonly round: ((v: number) => number) | undefined;
};

function toGradView(g: Tensor, fn: string): GradView {
  const dtype = g.dtype;
  if (dtype !== "float32" && dtype !== "float64" && dtype !== "float16" && dtype !== "bfloat16") {
    throw new DTypeError(`${fn} requires floating-point gradients; received dtype ${dtype}`);
  }
  let offsets: Float64Array | null = null;
  if (!isContiguous(g.shape, g.strides)) {
    const logical = new Array<number>(g.ndim);
    let acc = 1;
    for (let axis = g.ndim - 1; axis >= 0; axis--) {
      logical[axis] = acc;
      acc *= g.shape[axis] ?? 1;
    }
    offsets = new Float64Array(g.size);
    for (let i = 0; i < g.size; i++) {
      offsets[i] = offsetFromFlatIndex(i, logical, g.strides, g.offset);
    }
  }
  return {
    tensor: g,
    data: g.data as Float32Array | Float64Array,
    offsets,
    round:
      dtype === "float16" ? roundToFloat16 : dtype === "bfloat16" ? roundToBFloat16 : undefined,
  };
}

/** Collect the distinct gradients of `parameters`, skipping parameters without one. */
function collectGrads(parameters: GradTensor | Iterable<GradTensor>, fn: string): GradView[] {
  const list = GradTensor.isGradTensor(parameters) ? [parameters] : Array.from(parameters);
  const seen = new Set<Tensor>();
  const views: GradView[] = [];
  for (const p of list) {
    const g = p.grad;
    // A parameter listed twice (tied weights) must be clipped only once.
    if (g && !seen.has(g)) {
      seen.add(g);
      views.push(toGradView(g, fn));
    }
  }
  return views;
}

/**
 * Norm of all gradients viewed as one vector: `Infinity` is the maximum
 * absolute value and any finite `p > 0` the usual p-norm. A NaN anywhere gives
 * NaN. Finite p-norms are computed relative to the largest magnitude so that
 * very large or very small gradients neither overflow nor underflow.
 */
function globalNorm(views: readonly GradView[], p: number): number {
  let maxAbs = 0;
  let hasNaN = false;
  for (const v of views) {
    const n = v.tensor.size;
    const base = v.tensor.offset;
    for (let i = 0; i < n; i++) {
      const x = v.data[v.offsets === null ? base + i : (v.offsets[i] as number)] as number;
      if (Number.isNaN(x)) {
        hasNaN = true;
      } else {
        const a = Math.abs(x);
        if (a > maxAbs) maxAbs = a;
      }
    }
  }
  if (hasNaN) return Number.NaN;
  if (p === Infinity || maxAbs === 0 || maxAbs === Infinity) return maxAbs;

  // sum((|x| / m)^p)^(1/p) * m, with m the largest magnitude, so each term is <= 1.
  const scale = maxAbs;
  let sum = 0;
  for (const v of views) {
    const n = v.tensor.size;
    const base = v.tensor.offset;
    for (let i = 0; i < n; i++) {
      const a = Math.abs(
        v.data[v.offsets === null ? base + i : (v.offsets[i] as number)] as number
      );
      const r = a / scale;
      if (p === 2) sum += r * r;
      else if (p === 1) sum += r;
      else sum += r ** p;
    }
  }
  return scale * sum ** (1 / p);
}

/**
 * Clip the total norm of gradients of an iterable of parameters.
 *
 * The norm is computed over all gradients together, as if they were
 * concatenated into a single vector. Gradients are modified in-place, and
 * parameters without a gradient are skipped. A parameter that appears more
 * than once is counted and scaled once.
 *
 * Like PyTorch, gradients are multiplied by `maxNorm / (totalNorm + 1e-6)`
 * whenever that factor is below 1. If the norm is NaN the gradients are left
 * unchanged; pass `errorIfNonfinite` to fail loudly instead.
 *
 * Gradients are read on the host, in logical order, so strided views and a storage
 * offset are handled. Gradients stored on a kernel device (for example `webgpu`)
 * cannot be read synchronously: the call throws a `DeviceError` before any gradient
 * is changed. Move them back with `await grad.cpu()` and `setGrad` first.
 *
 * @param parameters - A GradTensor or an iterable of GradTensors whose gradients will be clipped
 * @param maxNorm - Maximum allowed norm value, must be >= 0
 * @param normType - Type of norm: `Infinity` for the max norm, otherwise the p > 0 of the p-norm
 *   (default: 2 for the L2 norm)
 * @param errorIfNonfinite - Throw if the total norm is NaN or infinite (default: false)
 * @returns The total norm of the gradients (before clipping)
 * @throws {InvalidParameterError} If `maxNorm` is negative or NaN, or `normType` is not positive
 * @throws {DataValidationError} If `errorIfNonfinite` is set and the total norm is not finite
 * @throws {DTypeError} If a gradient is not floating-point
 * @throws {DeviceError} If a gradient lives in device memory
 * @deprecated Prefer {@link clipGradNorm_}.
 */
export function clip_grad_norm_(
  parameters: GradTensor | Iterable<GradTensor>,
  maxNorm: number,
  normType = 2,
  errorIfNonfinite = false
): number {
  if (typeof maxNorm !== "number" || Number.isNaN(maxNorm) || maxNorm < 0) {
    throw new InvalidParameterError(
      `maxNorm must be >= 0; received ${maxNorm}`,
      "maxNorm",
      maxNorm
    );
  }
  if (typeof normType !== "number" || !(normType > 0)) {
    throw new InvalidParameterError(
      `normType must be a positive number or Infinity; received ${String(normType)}`,
      "normType",
      normType
    );
  }

  const views = collectGrads(parameters, "clip_grad_norm_");
  if (views.length === 0) {
    return 0;
  }

  const totalNorm = globalNorm(views, normType);

  if (errorIfNonfinite && !Number.isFinite(totalNorm)) {
    throw new DataValidationError(
      `The total norm of order ${normType} for gradients is non-finite (${totalNorm}), ` +
        "so it cannot be clipped"
    );
  }

  const clipCoef = maxNorm / (totalNorm + 1e-6);
  if (clipCoef < 1) {
    for (const v of views) {
      const n = v.tensor.size;
      const base = v.tensor.offset;
      for (let i = 0; i < n; i++) {
        const idx = v.offsets === null ? base + i : (v.offsets[i] as number);
        const scaled = (v.data[idx] as number) * clipCoef;
        v.data[idx] = v.round ? v.round(scaled) : scaled;
      }
    }
  }

  return totalNorm;
}

/**
 * Clip the gradients of an iterable of parameters at specified value.
 *
 * Each gradient element is clamped to [-clipValue, clipValue].
 * Gradients are modified in-place. NaN entries stay NaN. A parameter that
 * appears more than once is clipped once. Gradients stored on a kernel device
 * throw a `DeviceError`, as described for {@link clip_grad_norm_}.
 *
 * @param parameters - A GradTensor or an iterable of GradTensors whose gradients will be clipped
 * @param clipValue - Maximum absolute value for gradient elements, must be >= 0
 * @throws {InvalidParameterError} If `clipValue` is negative or NaN
 * @throws {DTypeError} If a gradient is not floating-point
 * @throws {DeviceError} If a gradient lives in device memory
 * @deprecated Prefer {@link clipGradValue_}.
 */
export function clip_grad_value_(
  parameters: GradTensor | Iterable<GradTensor>,
  clipValue: number
): void {
  if (typeof clipValue !== "number" || Number.isNaN(clipValue) || clipValue < 0) {
    throw new InvalidParameterError(
      `clipValue must be >= 0; received ${clipValue}`,
      "clipValue",
      clipValue
    );
  }

  for (const v of collectGrads(parameters, "clip_grad_value_")) {
    const n = v.tensor.size;
    const base = v.tensor.offset;
    for (let i = 0; i < n; i++) {
      const idx = v.offsets === null ? base + i : (v.offsets[i] as number);
      const val = v.data[idx] as number;
      // Math.min/max propagate NaN, so NaN gradients are preserved.
      const clamped = Math.max(-clipValue, Math.min(clipValue, val));
      v.data[idx] = v.round ? v.round(clamped) : clamped;
    }
  }
}

// ---------------------------------------------------------------------------
// Canonical camelCase aliases
//
// The snake_case spellings above mirror PyTorch and remain exported for
// backward compatibility. These camelCase aliases are the recommended names;
// each refers to the exact same in-place function.
// ---------------------------------------------------------------------------

/**
 * Clip the total norm of gradients of an iterable of parameters.
 *
 * The norm is computed over all gradients together, as if they were
 * concatenated into a single vector. Gradients are modified in-place, and
 * parameters without a gradient are skipped. A parameter that appears more
 * than once is counted and scaled once.
 *
 * Like PyTorch, gradients are multiplied by `maxNorm / (totalNorm + 1e-6)`
 * whenever that factor is below 1. If the norm is NaN the gradients are left
 * unchanged; pass `errorIfNonfinite` to fail loudly instead.
 *
 * Gradients are read on the host, in logical order, so strided views and a storage
 * offset are handled. Gradients stored on a kernel device (for example `webgpu`)
 * cannot be read synchronously: the call throws a `DeviceError` before any gradient
 * is changed. Move them back with `await grad.cpu()` and `setGrad` first.
 *
 * @param parameters - A GradTensor or an iterable of GradTensors whose gradients will be clipped
 * @param maxNorm - Maximum allowed norm value, must be >= 0
 * @param normType - Type of norm: `Infinity` for the max norm, otherwise the p > 0 of the p-norm
 *   (default: 2 for the L2 norm)
 * @param errorIfNonfinite - Throw if the total norm is NaN or infinite (default: false)
 * @returns The total norm of the gradients (before clipping)
 * @throws {InvalidParameterError} If `maxNorm` is negative or NaN, or `normType` is not positive
 * @throws {DataValidationError} If `errorIfNonfinite` is set and the total norm is not finite
 * @throws {DTypeError} If a gradient is not floating-point
 * @throws {DeviceError} If a gradient lives in device memory
 */
export const clipGradNorm_ = clip_grad_norm_;
/** Convenience alias of {@link clip_grad_norm_} without the trailing underscore. */
export const clipGradNorm = clip_grad_norm_;
/**
 * Clip the gradients of an iterable of parameters at specified value.
 *
 * Each gradient element is clamped to [-clipValue, clipValue].
 * Gradients are modified in-place. NaN entries stay NaN. A parameter that
 * appears more than once is clipped once. Gradients stored on a kernel device
 * throw a `DeviceError`, as described for {@link clip_grad_norm_}.
 *
 * @param parameters - A GradTensor or an iterable of GradTensors whose gradients will be clipped
 * @param clipValue - Maximum absolute value for gradient elements, must be >= 0
 * @throws {InvalidParameterError} If `clipValue` is negative or NaN
 * @throws {DTypeError} If a gradient is not floating-point
 * @throws {DeviceError} If a gradient lives in device memory
 */
export const clipGradValue_ = clip_grad_value_;
/** Convenience alias of {@link clip_grad_value_} without the trailing underscore. */
export const clipGradValue = clip_grad_value_;
