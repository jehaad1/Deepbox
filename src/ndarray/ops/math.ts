/**
 * Element-wise exponential, logarithmic, root and rounding functions.
 *
 * All functions accept any numeric dtype (int64 values must be exactly
 * representable as doubles) and reject string tensors with a `DTypeError`.
 * Float input keeps its dtype (`float16`, `bfloat16`, `float32`, `float64`):
 * values are computed in float64 and then rounded to that dtype. Integer and
 * bool input to a function with fractional results computes in `float32`.
 * `square`, `floor`, `ceil`, `trunc` and `round` return integer input as an
 * integer dtype. Strided views are read through their strides, never through
 * the raw buffer.
 *
 * On a kernel device (such as `webgpu`), `exp`, `log`, `sqrt`, `square`, `rsqrt`,
 * `expm1` and `log1p` run on the device. The other functions have no device kernel
 * and throw a `DeviceError` that asks for `await t.cpu()` first.
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */

import { DeviceError, DTypeError, InvalidParameterError } from "../../core";
import type { UnaryKernelOp } from "../../core/backend/kernels";
import { toFloatDType } from "../../core/utils/dtype_utils";
import { isContiguous } from "../tensor/strides";
import { computeStrides, Tensor } from "../tensor/Tensor";
import { allocFloat, flatOffset, floatResult, readNumbers } from "./_internal";
import { dispatchUnary } from "./device_dispatch";

/**
 * Shared prologue of every function in this module: reject string tensors and
 * route kernel-device tensors to their device kernel.
 *
 * Host-accelerator devices (such as `wasm`) keep their data in host memory, so
 * they fall through to the CPU loop. A tensor that lives in device memory but
 * has no kernel for `name` gets an explicit hint instead of the generic
 * "synchronous data access" error.
 *
 * @returns The device result, or `null` when the caller should compute on the host
 */
function routeToDevice(name: string, t: Tensor, op: UnaryKernelOp | undefined): Tensor | null {
  if (t.dtype === "string") {
    throw new DTypeError(`${name} is not defined for string dtype`);
  }
  if (t.device === "cpu") return null;
  if (op !== undefined) {
    const onDevice = dispatchUnary(op, t);
    if (onDevice) return onDevice;
    return null;
  }
  if (t.isDeviceTensor) {
    throw new DeviceError(
      `${name} has no kernel on device "${t.device}". ` +
        "Move the tensor to the CPU first with `await t.cpu()`."
    );
  }
  return null;
}

/** True for the integer dtypes that rounding functions leave unchanged. */
function isIntegerDType(dtype: Tensor["dtype"]): boolean {
  return dtype === "int32" || dtype === "uint8" || dtype === "int64";
}

/** Copy of an integer tensor (floor, ceil, trunc and round do not change integers). */
function copyInteger(t: Tensor, name: string): Tensor {
  if (t.data instanceof BigInt64Array) {
    const data = t.data;
    const out = new BigInt64Array(t.size);
    const logicalStrides = computeStrides(t.shape);
    const contiguous = isContiguous(t.shape, t.strides);
    for (let i = 0; i < t.size; i++) {
      out[i] = data[flatOffset(i, t.offset, contiguous, logicalStrides, t.strides)] as bigint;
    }
    return Tensor.fromTypedArray({ data: out, shape: t.shape, dtype: "int64", device: t.device });
  }
  const copy = (readNumbers(t, name) as Int32Array | Uint8Array).slice();
  const dtype = t.dtype === "int32" ? "int32" : "uint8";
  return Tensor.fromTypedArray({ data: copy, shape: t.shape, dtype, device: t.device });
}

/**
 * Round half to even (banker's rounding), matching NumPy's np.round.
 * JS Math.round rounds halves toward +Infinity (0.5 -> 1, 2.5 -> 3, -2.5 -> -2).
 * Results that round to zero keep the sign of the input, as in NumPy
 * (`round(-0.4)` is `-0`).
 */
function roundHalfToEven(x: number): number {
  if (!Number.isFinite(x)) return x;
  const fl = Math.floor(x);
  const diff = x - fl;
  let r: number;
  if (diff > 0.5) r = fl + 1;
  else if (diff < 0.5) r = fl;
  else r = fl % 2 === 0 ? fl : fl + 1;
  return r === 0 && (x < 0 || Object.is(x, -0)) ? -0 : r;
}

/**
 * Round to a number of decimals the way NumPy does: scale by a power of ten,
 * round half to even, scale back. `factor` is `10 ** |decimals|` and `scaleUp`
 * is true for non-negative decimals.
 */
function roundToDecimals(x: number, factor: number, scaleUp: boolean): number {
  if (!Number.isFinite(x)) return x;
  if (!Number.isFinite(factor)) return scaleUp ? x : x * 0;
  const y = scaleUp ? x * factor : x / factor;
  if (!Number.isFinite(y)) return x;
  const r = roundHalfToEven(y);
  return scaleUp ? r / factor : r * factor;
}

/**
 * Element-wise exponential.
 *
 * Output dtype: the input's float dtype; integer and bool input gives `float32`.
 * Tensors on a device with a kernel backend are computed on the device and keep
 * their device dtype.
 *
 * @param t - Input tensor
 * @returns Tensor with `exp(x)` for each element
 * @throws {DTypeError} If `t` has string dtype
 *
 * @example
 * ```ts
 * import { exp, tensor } from 'deepbox/ndarray';
 *
 * exp(tensor([0, 1]));  // [1, 2.718281828459045]
 * ```
 */
export function exp(t: Tensor): Tensor {
  const onDevice = routeToDevice("exp", t, "exp");
  if (onDevice) return onDevice;

  const src = readNumbers(t, "exp");
  const n = t.size;
  const dtype = toFloatDType(t.dtype);
  const out = allocFloat(dtype, n);
  for (let i = 0; i < n; i++) {
    const x = src[i] as number;
    out[i] = Math.exp(x);
  }
  return floatResult(out, t.shape, dtype, t.device);
}

/**
 * Element-wise natural logarithm.
 *
 * Output dtype: the input's float dtype; integer and bool input gives `float32`.
 * `log(0)` is `-Infinity` and the log of a negative number is `NaN`.
 *
 * @param t - Input tensor
 * @returns Tensor with log(x) for each element
 * @throws {DTypeError} If `t` has string dtype
 *
 * @example
 * ```ts
 * import { log, tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([1, 2.71828, 7.389]);
 * const result = log(x);  // [0, 1, 2]
 * ```
 */
export function log(t: Tensor): Tensor {
  const onDevice = routeToDevice("log", t, "log");
  if (onDevice) return onDevice;

  const src = readNumbers(t, "log");
  const n = t.size;
  const dtype = toFloatDType(t.dtype);
  const out = allocFloat(dtype, n);
  for (let i = 0; i < n; i++) out[i] = Math.log(src[i] as number);
  return floatResult(out, t.shape, dtype, t.device);
}

/**
 * Element-wise square root.
 *
 * Output dtype: the input's float dtype; integer and bool input gives `float32`.
 *
 * @param t - Input tensor
 * @returns Tensor with sqrt(x) for each element
 *
 * @example
 * ```ts
 * import { sqrt, tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([4, 9, 16]);
 * const result = sqrt(x);  // [2, 3, 4]
 * ```
 */
export function sqrt(t: Tensor): Tensor {
  const onDevice = routeToDevice("sqrt", t, "sqrt");
  if (onDevice) return onDevice;

  const src = readNumbers(t, "sqrt");
  const n = t.size;
  const dtype = toFloatDType(t.dtype);
  const out = allocFloat(dtype, n);
  for (let i = 0; i < n; i++) {
    const x = src[i] as number;
    out[i] = Math.sqrt(x);
  }
  return floatResult(out, t.shape, dtype, t.device);
}

/**
 * Element-wise square.
 *
 * Float input keeps its dtype. Integer input keeps its dtype and wraps on overflow
 * (`bool` gives `int32`), as in NumPy.
 *
 * @example
 * ```ts
 * import { square, tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([1, 2, 3]);
 * const result = square(x);  // [1, 4, 9]
 * ```
 */
export function square(t: Tensor): Tensor {
  const onDevice = routeToDevice("square", t, "square");
  if (onDevice) return onDevice;

  const n = t.size;
  if (t.data instanceof BigInt64Array) {
    const data = t.data;
    const out = new BigInt64Array(n);
    const logicalStrides = computeStrides(t.shape);
    const contiguous = isContiguous(t.shape, t.strides);
    for (let i = 0; i < n; i++) {
      const v = data[flatOffset(i, t.offset, contiguous, logicalStrides, t.strides)] as bigint;
      out[i] = BigInt.asIntN(64, v * v);
    }
    return Tensor.fromTypedArray({ data: out, shape: t.shape, dtype: "int64", device: t.device });
  }
  if (t.dtype === "int32" || t.dtype === "bool") {
    const src = readNumbers(t, "square");
    const out = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      const x = src[i] as number;
      out[i] = Math.imul(x, x);
    }
    return Tensor.fromTypedArray({ data: out, shape: t.shape, dtype: "int32", device: t.device });
  }
  if (t.dtype === "uint8") {
    const src = readNumbers(t, "square");
    const out = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const x = src[i] as number;
      out[i] = x * x;
    }
    return Tensor.fromTypedArray({ data: out, shape: t.shape, dtype: "uint8", device: t.device });
  }

  const src = readNumbers(t, "square");
  const dtype = toFloatDType(t.dtype);
  const out = allocFloat(dtype, n);
  for (let i = 0; i < n; i++) {
    const x = src[i] as number;
    out[i] = x * x;
  }
  return floatResult(out, t.shape, dtype, t.device);
}

/**
 * Element-wise reciprocal square root.
 *
 * Returns 1/sqrt(x) for each element.
 *
 * @example
 * ```ts
 * import { rsqrt, tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([4, 9, 16]);
 * const result = rsqrt(x);  // [0.5, 0.333..., 0.25]
 * ```
 */
export function rsqrt(t: Tensor): Tensor {
  const onDevice = routeToDevice("rsqrt", t, "rsqrt");
  if (onDevice) return onDevice;

  const src = readNumbers(t, "rsqrt");
  const n = t.size;
  const dtype = toFloatDType(t.dtype);
  const out = allocFloat(dtype, n);
  for (let i = 0; i < n; i++) {
    const x = src[i] as number;
    out[i] = 1 / Math.sqrt(x);
  }
  return floatResult(out, t.shape, dtype, t.device);
}

/**
 * Element-wise cube root.
 *
 * @example
 * ```ts
 * import { cbrt, tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([8, 27, 64]);
 * const result = cbrt(x);  // [2, 3, 4]
 * ```
 */
export function cbrt(t: Tensor): Tensor {
  const onDevice = routeToDevice("cbrt", t, undefined);
  if (onDevice) return onDevice;

  const src = readNumbers(t, "cbrt");
  const n = t.size;
  const dtype = toFloatDType(t.dtype);
  const out = allocFloat(dtype, n);
  for (let i = 0; i < n; i++) {
    const x = src[i] as number;
    out[i] = Math.cbrt(x);
  }
  return floatResult(out, t.shape, dtype, t.device);
}

/**
 * Element-wise exp(x) - 1.
 *
 * More accurate than exp(x) - 1 for small x.
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function expm1(t: Tensor): Tensor {
  const onDevice = routeToDevice("expm1", t, "expm1");
  if (onDevice) return onDevice;

  const src = readNumbers(t, "expm1");
  const n = t.size;
  const dtype = toFloatDType(t.dtype);
  const out = allocFloat(dtype, n);
  for (let i = 0; i < n; i++) {
    const x = src[i] as number;
    out[i] = Math.expm1(x);
  }
  return floatResult(out, t.shape, dtype, t.device);
}

/**
 * Element-wise base-2 exponential.
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function exp2(t: Tensor): Tensor {
  const onDevice = routeToDevice("exp2", t, undefined);
  if (onDevice) return onDevice;

  const src = readNumbers(t, "exp2");
  const n = t.size;
  const dtype = toFloatDType(t.dtype);
  const out = allocFloat(dtype, n);
  for (let i = 0; i < n; i++) {
    const x = src[i] as number;
    out[i] = 2 ** x;
  }
  return floatResult(out, t.shape, dtype, t.device);
}

/**
 * Element-wise log(1 + x).
 *
 * More accurate than log(1 + x) for small x.
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function log1p(t: Tensor): Tensor {
  const onDevice = routeToDevice("log1p", t, "log1p");
  if (onDevice) return onDevice;

  const src = readNumbers(t, "log1p");
  const n = t.size;
  const dtype = toFloatDType(t.dtype);
  const out = allocFloat(dtype, n);
  for (let i = 0; i < n; i++) {
    const x = src[i] as number;
    out[i] = Math.log1p(x);
  }
  return floatResult(out, t.shape, dtype, t.device);
}

/**
 * Element-wise base-2 logarithm.
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function log2(t: Tensor): Tensor {
  const onDevice = routeToDevice("log2", t, undefined);
  if (onDevice) return onDevice;

  const src = readNumbers(t, "log2");
  const n = t.size;
  const dtype = toFloatDType(t.dtype);
  const out = allocFloat(dtype, n);
  for (let i = 0; i < n; i++) {
    const x = src[i] as number;
    out[i] = Math.log2(x);
  }
  return floatResult(out, t.shape, dtype, t.device);
}

/**
 * Element-wise base-10 logarithm.
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function log10(t: Tensor): Tensor {
  const onDevice = routeToDevice("log10", t, undefined);
  if (onDevice) return onDevice;

  const src = readNumbers(t, "log10");
  const n = t.size;
  const dtype = toFloatDType(t.dtype);
  const out = allocFloat(dtype, n);
  for (let i = 0; i < n; i++) {
    const x = src[i] as number;
    out[i] = Math.log10(x);
  }
  return floatResult(out, t.shape, dtype, t.device);
}

/**
 * Element-wise floor (round down).
 *
 * Float input keeps its dtype, integer input is returned unchanged as the same integer
 * dtype, and `bool` input gives `float32`.
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function floor(t: Tensor): Tensor {
  const onDevice = routeToDevice("floor", t, undefined);
  if (onDevice) return onDevice;
  if (isIntegerDType(t.dtype)) return copyInteger(t, "floor");

  const src = readNumbers(t, "floor");
  const n = t.size;
  const dtype = toFloatDType(t.dtype);
  const out = allocFloat(dtype, n);
  for (let i = 0; i < n; i++) {
    const x = src[i] as number;
    out[i] = Math.floor(x);
  }
  return floatResult(out, t.shape, dtype, t.device);
}

/**
 * Element-wise ceil (round up).
 *
 * Float input keeps its dtype, integer input is returned unchanged as the same integer
 * dtype, and `bool` input gives `float32`.
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function ceil(t: Tensor): Tensor {
  const onDevice = routeToDevice("ceil", t, undefined);
  if (onDevice) return onDevice;
  if (isIntegerDType(t.dtype)) return copyInteger(t, "ceil");

  const src = readNumbers(t, "ceil");
  const n = t.size;
  const dtype = toFloatDType(t.dtype);
  const out = allocFloat(dtype, n);
  for (let i = 0; i < n; i++) {
    const x = src[i] as number;
    out[i] = Math.ceil(x);
  }
  return floatResult(out, t.shape, dtype, t.device);
}

/**
 * Element-wise rounding, halves to even (like `numpy.round`).
 *
 * `round(0.5)` is 0, `round(1.5)` and `round(2.5)` are 2. With `decimals` the
 * value is scaled by `10 ** decimals`, rounded, and scaled back, so a negative
 * `decimals` rounds to tens, hundreds, and so on. Binary floating point makes
 * some decimal halves inexact (`round(2.675, 2)` is 2.68 because `2.675 * 100`
 * evaluates to exactly 267.5).
 *
 * @param t - Input tensor
 * @param decimals - Number of decimal places, may be negative (default: 0)
 * @returns Tensor of rounded values: the input's float dtype, integer dtypes unchanged
 *   (rounding to tens with a negative `decimals` included), `float32` for bool input
 * @throws {DTypeError} If `t` has string dtype
 * @throws {InvalidParameterError} If `decimals` is not an integer
 *
 * @example
 * ```ts
 * import { round, tensor } from 'deepbox/ndarray';
 *
 * round(tensor([0.5, 1.5, 2.5]));              // [0, 2, 2]
 * round(tensor([1.234, 5.678]), 1);            // [1.2, 5.7]
 * round(tensor([1234.5678]), -2);              // [1200]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function round(t: Tensor, decimals = 0): Tensor {
  if (!Number.isInteger(decimals)) {
    throw new InvalidParameterError(
      `decimals must be an integer; received ${String(decimals)}`,
      "decimals",
      decimals
    );
  }
  const onDevice = routeToDevice("round", t, undefined);
  if (onDevice) return onDevice;

  if (isIntegerDType(t.dtype)) {
    // Integers are already whole; a negative `decimals` still rounds to tens, hundreds, ...
    if (decimals >= 0) return copyInteger(t, "round");
    const src = readNumbers(t, "round");
    const factor = 10 ** -decimals;
    if (t.dtype === "int64") {
      const wide = new BigInt64Array(t.size);
      for (let i = 0; i < t.size; i++) {
        const r = roundToDecimals(src[i] as number, factor, false);
        wide[i] = BigInt(Number.isFinite(r) ? r : 0);
      }
      return Tensor.fromTypedArray({
        data: wide,
        shape: t.shape,
        dtype: "int64",
        device: t.device,
      });
    }
    const isInt32 = t.dtype === "int32";
    const out = isInt32 ? new Int32Array(t.size) : new Uint8Array(t.size);
    for (let i = 0; i < t.size; i++) {
      out[i] = roundToDecimals(src[i] as number, factor, false);
    }
    return Tensor.fromTypedArray({
      data: out,
      shape: t.shape,
      dtype: isInt32 ? "int32" : "uint8",
      device: t.device,
    });
  }

  const src = readNumbers(t, "round");
  const n = t.size;
  const dtype = toFloatDType(t.dtype);
  const out = allocFloat(dtype, n);
  const scaleUp = decimals >= 0;
  const factor = 10 ** Math.abs(decimals);
  if (decimals === 0) {
    for (let i = 0; i < n; i++) out[i] = roundHalfToEven(src[i] as number);
  } else {
    for (let i = 0; i < n; i++) out[i] = roundToDecimals(src[i] as number, factor, scaleUp);
  }
  return floatResult(out, t.shape, dtype, t.device);
}

/**
 * Element-wise truncate (round toward zero).
 *
 * Float input keeps its dtype, integer input is returned unchanged as the same integer
 * dtype, and `bool` input gives `float32`.
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function trunc(t: Tensor): Tensor {
  const onDevice = routeToDevice("trunc", t, undefined);
  if (onDevice) return onDevice;
  if (isIntegerDType(t.dtype)) return copyInteger(t, "trunc");

  const src = readNumbers(t, "trunc");
  const n = t.size;
  const dtype = toFloatDType(t.dtype);
  const out = allocFloat(dtype, n);
  for (let i = 0; i < n; i++) {
    const x = src[i] as number;
    out[i] = Math.trunc(x);
  }
  return floatResult(out, t.shape, dtype, t.device);
}
