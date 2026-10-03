/**
 * Element-wise trigonometric and hyperbolic functions.
 *
 * Float input keeps its dtype (`float16`, `bfloat16`, `float32`, `float64`): values are
 * computed in float64 and rounded to that dtype. Integer and bool input computes in
 * `float32`. Strided views are read through their strides. int64 values beyond 2^53 - 1
 * throw instead of being rounded silently.
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */

import { DTypeError, type Shape } from "../../core";
import { promoteTypes, toFloatDType } from "../../core/utils/dtype_utils";
import { Tensor } from "../tensor/Tensor";
import {
  allocFloat,
  floatResult,
  readAsNumberSafe,
  readNumbers,
  roundHalfResult,
} from "./_internal";
import {
  broadcastApply,
  ensureBroadcastableScalar,
  getBroadcastShape,
  isScalar,
} from "./broadcast";
import { dispatchUnary } from "./device_dispatch";

/**
 * Element-wise sine of angles in radians.
 *
 * Defined for all finite inputs; `sin(±Infinity)` and `sin(NaN)` are NaN.
 *
 * The result has the input's float dtype; bool, uint8, int32 and int64 input gives `float32`.
 * int64 values beyond 2^53 - 1 throw instead of being rounded silently.
 *
 * @param t - Input tensor of any numeric dtype
 * @returns New float tensor with the same shape and device
 * @throws {DTypeError} If the tensor has string dtype
 * @throws {DataValidationError} If an int64 value exceeds 2^53 - 1
 *
 * @example
 * ```ts
 * import { tensor, sin } from "deepbox/ndarray";
 *
 * sin(tensor([0, Math.PI / 2], { dtype: "float64" })).toArray(); // [0, 1]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function sin(t: Tensor): Tensor {
  const src = readNumbers(t, "sin");
  const n = t.size;
  const dtype = toFloatDType(t.dtype);
  const out = allocFloat(dtype, n);
  for (let i = 0; i < n; i++) {
    out[i] = Math.sin(src[i] as number);
  }
  return floatResult(out, t.shape, dtype, t.device);
}

/**
 * Element-wise cosine of angles in radians.
 *
 * Defined for all finite inputs; `cos(±Infinity)` and `cos(NaN)` are NaN.
 *
 * The result has the input's float dtype; bool, uint8, int32 and int64 input gives `float32`.
 * int64 values beyond 2^53 - 1 throw instead of being rounded silently.
 *
 * @param t - Input tensor of any numeric dtype
 * @returns New float tensor with the same shape and device
 * @throws {DTypeError} If the tensor has string dtype
 * @throws {DataValidationError} If an int64 value exceeds 2^53 - 1
 *
 * @example
 * ```ts
 * import { tensor, cos } from "deepbox/ndarray";
 *
 * cos(tensor([0, Math.PI], { dtype: "float64" })).toArray(); // [1, -1]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function cos(t: Tensor): Tensor {
  const src = readNumbers(t, "cos");
  const n = t.size;
  const dtype = toFloatDType(t.dtype);
  const out = allocFloat(dtype, n);
  for (let i = 0; i < n; i++) {
    out[i] = Math.cos(src[i] as number);
  }
  return floatResult(out, t.shape, dtype, t.device);
}

/**
 * Element-wise tangent of angles in radians.
 *
 * Odd multiples of pi/2 give very large finite values, not Infinity, because pi/2 is not exactly representable.
 *
 * The result has the input's float dtype; bool, uint8, int32 and int64 input gives `float32`.
 * int64 values beyond 2^53 - 1 throw instead of being rounded silently.
 *
 * @param t - Input tensor of any numeric dtype
 * @returns New float tensor with the same shape and device
 * @throws {DTypeError} If the tensor has string dtype
 * @throws {DataValidationError} If an int64 value exceeds 2^53 - 1
 *
 * @example
 * ```ts
 * import { tensor, tan } from "deepbox/ndarray";
 *
 * tan(tensor([0, Math.PI / 4], { dtype: "float64" })).toArray(); // [0, 0.9999999999999999]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function tan(t: Tensor): Tensor {
  const src = readNumbers(t, "tan");
  const n = t.size;
  const dtype = toFloatDType(t.dtype);
  const out = allocFloat(dtype, n);
  for (let i = 0; i < n; i++) {
    out[i] = Math.tan(src[i] as number);
  }
  return floatResult(out, t.shape, dtype, t.device);
}

/**
 * Element-wise inverse sine, returning angles in `[-pi/2, pi/2]`.
 *
 * Inputs outside `[-1, 1]` give NaN.
 *
 * The result has the input's float dtype; bool, uint8, int32 and int64 input gives `float32`.
 * int64 values beyond 2^53 - 1 throw instead of being rounded silently.
 *
 * @param t - Input tensor of any numeric dtype
 * @returns New float tensor with the same shape and device
 * @throws {DTypeError} If the tensor has string dtype
 * @throws {DataValidationError} If an int64 value exceeds 2^53 - 1
 *
 * @example
 * ```ts
 * import { tensor, asin } from "deepbox/ndarray";
 *
 * asin(tensor([0, 1])).toArray(); // [0, 1.5707963267948966]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function asin(t: Tensor): Tensor {
  const src = readNumbers(t, "asin");
  const n = t.size;
  const dtype = toFloatDType(t.dtype);
  const out = allocFloat(dtype, n);
  for (let i = 0; i < n; i++) {
    out[i] = Math.asin(src[i] as number);
  }
  return floatResult(out, t.shape, dtype, t.device);
}

/**
 * Element-wise inverse cosine, returning angles in `[0, pi]`.
 *
 * Inputs outside `[-1, 1]` give NaN.
 *
 * The result has the input's float dtype; bool, uint8, int32 and int64 input gives `float32`.
 * int64 values beyond 2^53 - 1 throw instead of being rounded silently.
 *
 * @param t - Input tensor of any numeric dtype
 * @returns New float tensor with the same shape and device
 * @throws {DTypeError} If the tensor has string dtype
 * @throws {DataValidationError} If an int64 value exceeds 2^53 - 1
 *
 * @example
 * ```ts
 * import { tensor, acos } from "deepbox/ndarray";
 *
 * acos(tensor([1, 0])).toArray(); // [0, 1.5707963267948966]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function acos(t: Tensor): Tensor {
  const src = readNumbers(t, "acos");
  const n = t.size;
  const dtype = toFloatDType(t.dtype);
  const out = allocFloat(dtype, n);
  for (let i = 0; i < n; i++) {
    out[i] = Math.acos(src[i] as number);
  }
  return floatResult(out, t.shape, dtype, t.device);
}

/**
 * Element-wise inverse tangent, returning angles in `(-pi/2, pi/2)`.
 *
 * Defined for all real inputs; `atan(±Infinity)` is `±pi/2`.
 *
 * The result has the input's float dtype; bool, uint8, int32 and int64 input gives `float32`.
 * int64 values beyond 2^53 - 1 throw instead of being rounded silently.
 *
 * @param t - Input tensor of any numeric dtype
 * @returns New float tensor with the same shape and device
 * @throws {DTypeError} If the tensor has string dtype
 * @throws {DataValidationError} If an int64 value exceeds 2^53 - 1
 *
 * @example
 * ```ts
 * import { tensor, atan } from "deepbox/ndarray";
 *
 * atan(tensor([0, 1])).toArray(); // [0, 0.7853981633974483]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function atan(t: Tensor): Tensor {
  const src = readNumbers(t, "atan");
  const n = t.size;
  const dtype = toFloatDType(t.dtype);
  const out = allocFloat(dtype, n);
  for (let i = 0; i < n; i++) {
    out[i] = Math.atan(src[i] as number);
  }
  return floatResult(out, t.shape, dtype, t.device);
}

/**
 * Element-wise arctangent of `y / x`, using the signs of both arguments to pick the
 * quadrant. Results lie in `[-pi, pi]`; `atan2(0, 0)` is 0.
 *
 * `y` and `x` are broadcast against each other (NumPy rules) and may have
 * different dtypes. The result has the promoted float dtype of the two (integer
 * and bool operands give `float32`) and lives on the device of `y`.
 *
 * @param y - Numerator (ordinate) tensor
 * @param x - Denominator (abscissa) tensor
 * @returns New float tensor with the broadcast shape
 * @throws {DTypeError} If either tensor has string dtype
 * @throws {DataValidationError} If an int64 value exceeds 2^53 - 1
 * @throws {ShapeError} If the shapes are not broadcast-compatible
 *
 * @example
 * ```ts
 * import { tensor, atan2 } from "deepbox/ndarray";
 *
 * atan2(tensor([1, -1]), tensor([-1, -1])).toArray(); // [2.356194490192345, -2.356194490192345]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function atan2(y: Tensor, x: Tensor): Tensor {
  if (y.dtype === "string" || x.dtype === "string") {
    throw new DTypeError("atan2 is not defined for string dtype");
  }

  ensureBroadcastableScalar(y, x);

  const yIsScalar = isScalar(y);
  const xIsScalar = isScalar(x);
  const outShape: Shape = yIsScalar
    ? x.shape
    : xIsScalar
      ? y.shape
      : getBroadcastShape(y.shape, x.shape);
  const outSize = outShape.reduce((acc, dim) => acc * dim, 1);

  const dtype = toFloatDType(promoteTypes(y.dtype, x.dtype));
  const out = allocFloat(dtype, outSize);
  const result = Tensor.fromTypedArray({ data: out, shape: outShape, dtype, device: y.device });
  if (outSize === 0) return result;

  const sameShape = y.ndim === x.ndim && y.shape.every((dim, i) => dim === x.shape[i]);

  if (sameShape || xIsScalar || yIsScalar) {
    // Dense row-major reads; a 0-d operand is broadcast by index 0.
    const ys = readNumbers(y, "atan2");
    const xs = readNumbers(x, "atan2");
    if (sameShape) {
      for (let i = 0; i < outSize; i++) {
        out[i] = Math.atan2(ys[i] as number, xs[i] as number);
      }
    } else if (xIsScalar) {
      const xv = xs[0] as number;
      for (let i = 0; i < outSize; i++) out[i] = Math.atan2(ys[i] as number, xv);
    } else {
      const yv = ys[0] as number;
      for (let i = 0; i < outSize; i++) out[i] = Math.atan2(yv, xs[i] as number);
    }
    return roundHalfResult(result);
  }

  const yData = y.data;
  const xData = x.data;
  if (Array.isArray(yData) || Array.isArray(xData)) {
    throw new DTypeError("atan2 is not defined for string dtype");
  }
  broadcastApply(y, x, result, (offY, offX, offOut) => {
    out[offOut] = Math.atan2(readAsNumberSafe(yData, offY), readAsNumberSafe(xData, offX));
  });

  return roundHalfResult(result);
}

/**
 * Element-wise hyperbolic sine.
 *
 * Overflows to `±Infinity` for magnitudes above about 710.
 *
 * The result has the input's float dtype; bool, uint8, int32 and int64 input gives `float32`.
 * int64 values beyond 2^53 - 1 throw instead of being rounded silently.
 *
 * @param t - Input tensor of any numeric dtype
 * @returns New float tensor with the same shape and device
 * @throws {DTypeError} If the tensor has string dtype
 * @throws {DataValidationError} If an int64 value exceeds 2^53 - 1
 *
 * @example
 * ```ts
 * import { tensor, sinh } from "deepbox/ndarray";
 *
 * sinh(tensor([0, 1])).toArray(); // [0, 1.1752011936438014]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function sinh(t: Tensor): Tensor {
  const src = readNumbers(t, "sinh");
  const n = t.size;
  const dtype = toFloatDType(t.dtype);
  const out = allocFloat(dtype, n);
  for (let i = 0; i < n; i++) {
    out[i] = Math.sinh(src[i] as number);
  }
  return floatResult(out, t.shape, dtype, t.device);
}

/**
 * Element-wise hyperbolic cosine.
 *
 * Overflows to `Infinity` for magnitudes above about 710.
 *
 * The result has the input's float dtype; bool, uint8, int32 and int64 input gives `float32`.
 * int64 values beyond 2^53 - 1 throw instead of being rounded silently.
 *
 * @param t - Input tensor of any numeric dtype
 * @returns New float tensor with the same shape and device
 * @throws {DTypeError} If the tensor has string dtype
 * @throws {DataValidationError} If an int64 value exceeds 2^53 - 1
 *
 * @example
 * ```ts
 * import { tensor, cosh } from "deepbox/ndarray";
 *
 * cosh(tensor([0, 1])).toArray(); // [1, 1.5430806348152437]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function cosh(t: Tensor): Tensor {
  const src = readNumbers(t, "cosh");
  const n = t.size;
  const dtype = toFloatDType(t.dtype);
  const out = allocFloat(dtype, n);
  for (let i = 0; i < n; i++) {
    out[i] = Math.cosh(src[i] as number);
  }
  return floatResult(out, t.shape, dtype, t.device);
}

/**
 * Element-wise hyperbolic tangent.
 *
 * Saturates to `±1` for large magnitudes. The result has the input's float dtype
 * (bool, uint8, int32 and int64 input gives `float32`); a tensor on a kernel
 * backend stays on that device and keeps its floating dtype. int64 values beyond
 * 2^53 - 1 throw instead of being rounded silently.
 *
 * @param t - Input tensor of any numeric dtype
 * @returns New tensor with the same shape and device
 * @throws {DTypeError} If the tensor has string dtype
 * @throws {DataValidationError} If an int64 value exceeds 2^53 - 1
 *
 * @example
 * ```ts
 * import { tensor, tanh } from "deepbox/ndarray";
 *
 * tanh(tensor([0, 1])).toArray(); // [0, 0.7615941559557649]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function tanh(t: Tensor): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("tanh is not defined for string dtype");
  }

  if (t.device !== "cpu") {
    const onDevice = dispatchUnary("tanh", t);
    if (onDevice) return onDevice;
  }

  // Math.tanh stays accurate for tiny arguments (a 1 - exp(-2x) formula loses
  // most digits there) and is at least as fast in current V8.
  const src = readNumbers(t, "tanh");
  const n = t.size;
  const dtype = toFloatDType(t.dtype);
  const out = allocFloat(dtype, n);
  for (let i = 0; i < n; i++) {
    out[i] = Math.tanh(src[i] as number);
  }
  return floatResult(out, t.shape, dtype, t.device);
}

/**
 * Element-wise inverse hyperbolic sine.
 *
 * Defined for all real inputs.
 *
 * The result has the input's float dtype; bool, uint8, int32 and int64 input gives `float32`.
 * int64 values beyond 2^53 - 1 throw instead of being rounded silently.
 *
 * @param t - Input tensor of any numeric dtype
 * @returns New float tensor with the same shape and device
 * @throws {DTypeError} If the tensor has string dtype
 * @throws {DataValidationError} If an int64 value exceeds 2^53 - 1
 *
 * @example
 * ```ts
 * import { tensor, asinh } from "deepbox/ndarray";
 *
 * asinh(tensor([0, 1])).toArray(); // [0, 0.881373587019543]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function asinh(t: Tensor): Tensor {
  const src = readNumbers(t, "asinh");
  const n = t.size;
  const dtype = toFloatDType(t.dtype);
  const out = allocFloat(dtype, n);
  for (let i = 0; i < n; i++) {
    out[i] = Math.asinh(src[i] as number);
  }
  return floatResult(out, t.shape, dtype, t.device);
}

/**
 * Element-wise inverse hyperbolic cosine, returning values in `[0, Infinity)`.
 *
 * Inputs below 1 give NaN.
 *
 * The result has the input's float dtype; bool, uint8, int32 and int64 input gives `float32`.
 * int64 values beyond 2^53 - 1 throw instead of being rounded silently.
 *
 * @param t - Input tensor of any numeric dtype
 * @returns New float tensor with the same shape and device
 * @throws {DTypeError} If the tensor has string dtype
 * @throws {DataValidationError} If an int64 value exceeds 2^53 - 1
 *
 * @example
 * ```ts
 * import { tensor, acosh } from "deepbox/ndarray";
 *
 * acosh(tensor([1, 2])).toArray(); // [0, 1.3169578969248166]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function acosh(t: Tensor): Tensor {
  const src = readNumbers(t, "acosh");
  const n = t.size;
  const dtype = toFloatDType(t.dtype);
  const out = allocFloat(dtype, n);
  for (let i = 0; i < n; i++) {
    out[i] = Math.acosh(src[i] as number);
  }
  return floatResult(out, t.shape, dtype, t.device);
}

/**
 * Element-wise inverse hyperbolic tangent.
 *
 * `atanh(±1)` is `±Infinity`; inputs outside `[-1, 1]` give NaN.
 *
 * The result has the input's float dtype; bool, uint8, int32 and int64 input gives `float32`.
 * int64 values beyond 2^53 - 1 throw instead of being rounded silently.
 *
 * @param t - Input tensor of any numeric dtype
 * @returns New float tensor with the same shape and device
 * @throws {DTypeError} If the tensor has string dtype
 * @throws {DataValidationError} If an int64 value exceeds 2^53 - 1
 *
 * @example
 * ```ts
 * import { tensor, atanh } from "deepbox/ndarray";
 *
 * atanh(tensor([0, 0.5])).toArray(); // [0, 0.5493061443340548]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function atanh(t: Tensor): Tensor {
  const src = readNumbers(t, "atanh");
  const n = t.size;
  const dtype = toFloatDType(t.dtype);
  const out = allocFloat(dtype, n);
  for (let i = 0; i < n; i++) {
    out[i] = Math.atanh(src[i] as number);
  }
  return floatResult(out, t.shape, dtype, t.device);
}
