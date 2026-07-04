import { DTypeError, type Shape } from "../../core";
import { isContiguous } from "../tensor/strides";
import { computeStrides, isBigIntArray, Tensor } from "../tensor/Tensor";
import { flatOffset, readAsNumberSafe, readNumericContiguous } from "./_internal";
import {
  broadcastApply,
  ensureBroadcastableScalar,
  getBroadcastShape,
  isScalar,
} from "./broadcast";
import { dispatchUnary } from "./device_dispatch";

/**
 * Element-wise sine.
 *
 * Output dtype:
 * - Always `float64` for now.
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function sin(t: Tensor): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("sin is not defined for string dtype");
  }

  const out = new Float64Array(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("sin is not defined for string dtype");
  }

  if (isBigIntArray(data)) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      out[i] = Math.sin(readAsNumberSafe(data, srcOffset));
    }
  } else if (contiguous && t.offset === 0) {
    for (let i = 0; i < t.size; i++) {
      out[i] = Math.sin(data[i] as number);
    }
  } else {
    const src = readNumericContiguous(t);
    if (src === null) {
      throw new DTypeError("sin is not defined for string dtype");
    }
    for (let i = 0; i < t.size; i++) {
      out[i] = Math.sin(src[i] as number);
    }
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: t.shape,
    dtype: "float64",
    device: t.device,
  });
}

/**
 * Element-wise cosine.
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function cos(t: Tensor): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("cos is not defined for string dtype");
  }

  const out = new Float64Array(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("cos is not defined for string dtype");
  }

  if (isBigIntArray(data)) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      out[i] = Math.cos(readAsNumberSafe(data, srcOffset));
    }
  } else if (contiguous && t.offset === 0) {
    for (let i = 0; i < t.size; i++) {
      out[i] = Math.cos(data[i] as number);
    }
  } else {
    const src = readNumericContiguous(t);
    if (src === null) {
      throw new DTypeError("cos is not defined for string dtype");
    }
    for (let i = 0; i < t.size; i++) {
      out[i] = Math.cos(src[i] as number);
    }
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: t.shape,
    dtype: "float64",
    device: t.device,
  });
}

/**
 * Element-wise tangent.
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function tan(t: Tensor): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("tan is not defined for string dtype");
  }

  const out = new Float64Array(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("tan is not defined for string dtype");
  }

  if (isBigIntArray(data)) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      out[i] = Math.tan(readAsNumberSafe(data, srcOffset));
    }
  } else if (contiguous && t.offset === 0) {
    for (let i = 0; i < t.size; i++) {
      out[i] = Math.tan(data[i] as number);
    }
  } else {
    const src = readNumericContiguous(t);
    if (src === null) {
      throw new DTypeError("tan is not defined for string dtype");
    }
    for (let i = 0; i < t.size; i++) {
      out[i] = Math.tan(src[i] as number);
    }
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: t.shape,
    dtype: "float64",
    device: t.device,
  });
}

/**
 * Element-wise inverse sine.
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function asin(t: Tensor): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("asin is not defined for string dtype");
  }

  const out = new Float64Array(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("asin is not defined for string dtype");
  }

  if (isBigIntArray(data)) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      out[i] = Math.asin(readAsNumberSafe(data, srcOffset));
    }
  } else {
    const src = readNumericContiguous(t);
    if (src === null) {
      throw new DTypeError("asin is not defined for string dtype");
    }
    for (let i = 0; i < t.size; i++) {
      out[i] = Math.asin(src[i] as number);
    }
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: t.shape,
    dtype: "float64",
    device: t.device,
  });
}

/**
 * Element-wise inverse cosine.
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function acos(t: Tensor): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("acos is not defined for string dtype");
  }

  const out = new Float64Array(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("acos is not defined for string dtype");
  }

  if (isBigIntArray(data)) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      out[i] = Math.acos(readAsNumberSafe(data, srcOffset));
    }
  } else {
    const src = readNumericContiguous(t);
    if (src === null) {
      throw new DTypeError("acos is not defined for string dtype");
    }
    for (let i = 0; i < t.size; i++) {
      out[i] = Math.acos(src[i] as number);
    }
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: t.shape,
    dtype: "float64",
    device: t.device,
  });
}

/**
 * Element-wise inverse tangent.
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function atan(t: Tensor): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("atan is not defined for string dtype");
  }

  const out = new Float64Array(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("atan is not defined for string dtype");
  }

  if (isBigIntArray(data)) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      out[i] = Math.atan(readAsNumberSafe(data, srcOffset));
    }
  } else {
    const src = readNumericContiguous(t);
    if (src === null) {
      throw new DTypeError("atan is not defined for string dtype");
    }
    for (let i = 0; i < t.size; i++) {
      out[i] = Math.atan(src[i] as number);
    }
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: t.shape,
    dtype: "float64",
    device: t.device,
  });
}

/**
 * Element-wise arctangent of y/x with correct quadrant.
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

  const out = new Float64Array(outSize);
  const result = Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: "float64",
    device: y.device,
  });

  const yData = y.data;
  const xData = x.data;

  if (Array.isArray(yData) || Array.isArray(xData)) {
    throw new DTypeError("atan2 is not defined for string dtype");
  }

  broadcastApply(y, x, result, (offY, offX, offOut) => {
    // We need to read as number
    const yVal = readAsNumberSafe(yData, offY);
    const xVal = readAsNumberSafe(xData, offX);
    out[offOut] = Math.atan2(yVal, xVal);
  });

  return result;
}

/**
 * Element-wise hyperbolic sine.
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function sinh(t: Tensor): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("sinh is not defined for string dtype");
  }

  const out = new Float64Array(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("sinh is not defined for string dtype");
  }

  if (isBigIntArray(data)) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      out[i] = Math.sinh(readAsNumberSafe(data, srcOffset));
    }
  } else {
    const src = readNumericContiguous(t);
    if (src === null) {
      throw new DTypeError("sinh is not defined for string dtype");
    }
    for (let i = 0; i < t.size; i++) {
      out[i] = Math.sinh(src[i] as number);
    }
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: t.shape,
    dtype: "float64",
    device: t.device,
  });
}

/**
 * Element-wise hyperbolic cosine.
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function cosh(t: Tensor): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("cosh is not defined for string dtype");
  }

  const out = new Float64Array(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("cosh is not defined for string dtype");
  }

  if (isBigIntArray(data)) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      out[i] = Math.cosh(readAsNumberSafe(data, srcOffset));
    }
  } else {
    const src = readNumericContiguous(t);
    if (src === null) {
      throw new DTypeError("cosh is not defined for string dtype");
    }
    for (let i = 0; i < t.size; i++) {
      out[i] = Math.cosh(src[i] as number);
    }
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: t.shape,
    dtype: "float64",
    device: t.device,
  });
}

/**
 * Element-wise hyperbolic tangent.
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

  const out = new Float64Array(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("tanh is not defined for string dtype");
  }

  if (isBigIntArray(data)) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      out[i] = Math.tanh(readAsNumberSafe(data, srcOffset));
    }
  } else {
    const src = readNumericContiguous(t);
    if (src === null) {
      throw new DTypeError("tanh is not defined for string dtype");
    }
    // tanh(x) = sign(x) * (1 - e) / (1 + e) with e = exp(-2|x|). Computed from a
    // single Math.exp (V8's Math.tanh is a much slower libm call, ~1.4x here);
    // exp(-2|x|) stays in (0,1] so there is no overflow, and |x|>20 saturates to
    // ±1. Matches Math.tanh to ~1 ulp across the range.
    for (let i = 0; i < t.size; i++) {
      const v = src[i] as number;
      const ax = v < 0 ? -v : v;
      let r: number;
      if (ax > 20) {
        r = 1;
      } else {
        const e = Math.exp(-2 * ax);
        r = (1 - e) / (1 + e);
      }
      out[i] = v < 0 ? -r : r;
    }
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: t.shape,
    dtype: "float64",
    device: t.device,
  });
}

/**
 * Element-wise inverse hyperbolic sine.
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function asinh(t: Tensor): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("asinh is not defined for string dtype");
  }

  const out = new Float64Array(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("asinh is not defined for string dtype");
  }

  if (isBigIntArray(data)) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      out[i] = Math.asinh(readAsNumberSafe(data, srcOffset));
    }
  } else {
    const src = readNumericContiguous(t);
    if (src === null) {
      throw new DTypeError("asinh is not defined for string dtype");
    }
    for (let i = 0; i < t.size; i++) {
      out[i] = Math.asinh(src[i] as number);
    }
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: t.shape,
    dtype: "float64",
    device: t.device,
  });
}

/**
 * Element-wise inverse hyperbolic cosine.
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function acosh(t: Tensor): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("acosh is not defined for string dtype");
  }

  const out = new Float64Array(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("acosh is not defined for string dtype");
  }

  if (isBigIntArray(data)) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      out[i] = Math.acosh(readAsNumberSafe(data, srcOffset));
    }
  } else {
    const src = readNumericContiguous(t);
    if (src === null) {
      throw new DTypeError("acosh is not defined for string dtype");
    }
    for (let i = 0; i < t.size; i++) {
      out[i] = Math.acosh(src[i] as number);
    }
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: t.shape,
    dtype: "float64",
    device: t.device,
  });
}

/**
 * Element-wise inverse hyperbolic tangent.
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function atanh(t: Tensor): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("atanh is not defined for string dtype");
  }

  const out = new Float64Array(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("atanh is not defined for string dtype");
  }

  if (isBigIntArray(data)) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      out[i] = Math.atanh(readAsNumberSafe(data, srcOffset));
    }
  } else {
    const src = readNumericContiguous(t);
    if (src === null) {
      throw new DTypeError("atanh is not defined for string dtype");
    }
    for (let i = 0; i < t.size; i++) {
      out[i] = Math.atanh(src[i] as number);
    }
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: t.shape,
    dtype: "float64",
    device: t.device,
  });
}
