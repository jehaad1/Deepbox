import { DTypeError, getNumericElement, type Shape } from "../../core";
import { isContiguous } from "../tensor/strides";
import { computeStrides, isBigIntArray, Tensor } from "../tensor/Tensor";
import { flatOffset, readAsNumberSafe } from "./_internal";
import {
  broadcastApply,
  ensureBroadcastableScalar,
  getBroadcastShape,
  isScalar,
} from "./broadcast";

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
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      out[i] = Math.sin(getNumericElement(data, srcOffset));
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
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      out[i] = Math.cos(getNumericElement(data, srcOffset));
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
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      out[i] = Math.tan(getNumericElement(data, srcOffset));
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
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      out[i] = Math.asin(getNumericElement(data, srcOffset));
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
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      out[i] = Math.acos(getNumericElement(data, srcOffset));
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
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      out[i] = Math.atan(getNumericElement(data, srcOffset));
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
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      out[i] = Math.sinh(getNumericElement(data, srcOffset));
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
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      out[i] = Math.cosh(getNumericElement(data, srcOffset));
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
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      out[i] = Math.tanh(getNumericElement(data, srcOffset));
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
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      out[i] = Math.asinh(getNumericElement(data, srcOffset));
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
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      out[i] = Math.acosh(getNumericElement(data, srcOffset));
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
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      out[i] = Math.atanh(getNumericElement(data, srcOffset));
    }
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: t.shape,
    dtype: "float64",
    device: t.device,
  });
}
