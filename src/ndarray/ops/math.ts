import { DTypeError, getBigIntElement, getNumericElement } from "../../core";
import { isContiguous, offsetFromFlatIndex } from "../tensor/strides";
import { computeStrides, dtypeToTypedArrayCtor, Tensor } from "../tensor/Tensor";
import { bigintToNumberSafe } from "./_internal";

/**
 * Element-wise exponential.
 *
 * Output dtype:
 * - Always `float64` for now.
 */
export function exp(t: Tensor): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("exp is not defined for string dtype");
  }

  const out = new Float64Array(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("exp is not defined for string dtype");
  }

  if (data instanceof BigInt64Array) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = contiguous
        ? t.offset + i
        : offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
      out[i] = Math.exp(bigintToNumberSafe(getBigIntElement(data, srcOffset)));
    }
  } else if (contiguous && t.offset === 0) {
    for (let i = 0; i < t.size; i++) {
      out[i] = Math.exp(data[i] as number);
    }
  } else {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = contiguous
        ? t.offset + i
        : offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
      out[i] = Math.exp(getNumericElement(data, srcOffset));
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
 * Element-wise natural logarithm.
 *
 * @param t - Input tensor
 * @returns Tensor with log(x) for each element
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
  if (t.dtype === "string") {
    throw new DTypeError("log is not defined for string dtype");
  }

  const dtype = t.dtype === "float32" ? "float32" : "float64";
  const Ctor = dtypeToTypedArrayCtor(dtype);
  const out = new Ctor(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("log is not defined for string dtype");
  }

  if (data instanceof BigInt64Array) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = contiguous
        ? t.offset + i
        : offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
      out[i] = Math.log(bigintToNumberSafe(getBigIntElement(data, srcOffset)));
    }
  } else if (contiguous && t.offset === 0) {
    for (let i = 0; i < t.size; i++) {
      out[i] = Math.log(data[i] as number);
    }
  } else {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = contiguous
        ? t.offset + i
        : offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
      out[i] = Math.log(getNumericElement(data, srcOffset));
    }
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: t.shape,
    dtype,
    device: t.device,
  });
}

/**
 * Element-wise square root.
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
  if (t.dtype === "string") {
    throw new DTypeError("sqrt is not defined for string dtype");
  }

  const out = new Float64Array(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("sqrt is not defined for string dtype");
  }

  if (data instanceof BigInt64Array) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = contiguous
        ? t.offset + i
        : offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
      out[i] = Math.sqrt(bigintToNumberSafe(getBigIntElement(data, srcOffset)));
    }
  } else if (contiguous && t.offset === 0) {
    for (let i = 0; i < t.size; i++) {
      out[i] = Math.sqrt(data[i] as number);
    }
  } else {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = contiguous
        ? t.offset + i
        : offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
      out[i] = Math.sqrt(getNumericElement(data, srcOffset));
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
 * Element-wise square.
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
  if (t.dtype === "string") {
    throw new DTypeError("square is not defined for string dtype");
  }

  const out = new Float64Array(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("square is not defined for string dtype");
  }

  if (data instanceof BigInt64Array) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = contiguous
        ? t.offset + i
        : offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
      const val = bigintToNumberSafe(getBigIntElement(data, srcOffset));
      out[i] = val * val;
    }
  } else if (contiguous && t.offset === 0) {
    for (let i = 0; i < t.size; i++) {
      const val = data[i] as number;
      out[i] = val * val;
    }
  } else {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = contiguous
        ? t.offset + i
        : offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
      const val = getNumericElement(data, srcOffset);
      out[i] = val * val;
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
  if (t.dtype === "string") {
    throw new DTypeError("rsqrt is not defined for string dtype");
  }

  const out = new Float64Array(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("rsqrt is not defined for string dtype");
  }

  if (data instanceof BigInt64Array) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = contiguous
        ? t.offset + i
        : offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
      const val = bigintToNumberSafe(getBigIntElement(data, srcOffset));
      out[i] = 1 / Math.sqrt(val);
    }
  } else if (contiguous && t.offset === 0) {
    for (let i = 0; i < t.size; i++) {
      out[i] = 1 / Math.sqrt(data[i] as number);
    }
  } else {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = contiguous
        ? t.offset + i
        : offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
      const val = getNumericElement(data, srcOffset);
      out[i] = 1 / Math.sqrt(val);
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
  if (t.dtype === "string") {
    throw new DTypeError("cbrt is not defined for string dtype");
  }

  const out = new Float64Array(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("cbrt is not defined for string dtype");
  }

  if (data instanceof BigInt64Array) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = contiguous
        ? t.offset + i
        : offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
      out[i] = Math.cbrt(bigintToNumberSafe(getBigIntElement(data, srcOffset)));
    }
  } else if (contiguous && t.offset === 0) {
    for (let i = 0; i < t.size; i++) {
      out[i] = Math.cbrt(data[i] as number);
    }
  } else {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = contiguous
        ? t.offset + i
        : offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
      out[i] = Math.cbrt(getNumericElement(data, srcOffset));
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
 * Element-wise exp(x) - 1.
 *
 * More accurate than exp(x) - 1 for small x.
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function expm1(t: Tensor): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("expm1 is not defined for string dtype");
  }

  const out = new Float64Array(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("expm1 is not defined for string dtype");
  }

  if (data instanceof BigInt64Array) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = contiguous
        ? t.offset + i
        : offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
      out[i] = Math.expm1(bigintToNumberSafe(getBigIntElement(data, srcOffset)));
    }
  } else {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = contiguous
        ? t.offset + i
        : offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
      out[i] = Math.expm1(getNumericElement(data, srcOffset));
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
 * Element-wise base-2 exponential.
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function exp2(t: Tensor): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("exp2 is not defined for string dtype");
  }

  const out = new Float64Array(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("exp2 is not defined for string dtype");
  }

  if (data instanceof BigInt64Array) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = contiguous
        ? t.offset + i
        : offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
      const val = bigintToNumberSafe(getBigIntElement(data, srcOffset));
      out[i] = 2 ** val;
    }
  } else {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = contiguous
        ? t.offset + i
        : offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
      const val = getNumericElement(data, srcOffset);
      out[i] = 2 ** val;
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
 * Element-wise log(1 + x).
 *
 * More accurate than log(1 + x) for small x.
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function log1p(t: Tensor): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("log1p is not defined for string dtype");
  }

  const out = new Float64Array(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("log1p is not defined for string dtype");
  }

  if (data instanceof BigInt64Array) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = contiguous
        ? t.offset + i
        : offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
      out[i] = Math.log1p(bigintToNumberSafe(getBigIntElement(data, srcOffset)));
    }
  } else {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = contiguous
        ? t.offset + i
        : offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
      out[i] = Math.log1p(getNumericElement(data, srcOffset));
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
 * Element-wise base-2 logarithm.
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function log2(t: Tensor): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("log2 is not defined for string dtype");
  }

  const out = new Float64Array(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("log2 is not defined for string dtype");
  }

  if (data instanceof BigInt64Array) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = contiguous
        ? t.offset + i
        : offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
      out[i] = Math.log2(bigintToNumberSafe(getBigIntElement(data, srcOffset)));
    }
  } else {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = contiguous
        ? t.offset + i
        : offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
      out[i] = Math.log2(getNumericElement(data, srcOffset));
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
 * Element-wise base-10 logarithm.
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function log10(t: Tensor): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("log10 is not defined for string dtype");
  }

  const out = new Float64Array(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("log10 is not defined for string dtype");
  }

  if (data instanceof BigInt64Array) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = contiguous
        ? t.offset + i
        : offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
      out[i] = Math.log10(bigintToNumberSafe(getBigIntElement(data, srcOffset)));
    }
  } else {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = contiguous
        ? t.offset + i
        : offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
      out[i] = Math.log10(getNumericElement(data, srcOffset));
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
 * Element-wise floor (round down).
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function floor(t: Tensor): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("floor is not defined for string dtype");
  }

  const out = new Float64Array(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("floor is not defined for string dtype");
  }

  if (data instanceof BigInt64Array) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = contiguous
        ? t.offset + i
        : offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
      out[i] = Math.floor(bigintToNumberSafe(getBigIntElement(data, srcOffset)));
    }
  } else {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = contiguous
        ? t.offset + i
        : offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
      out[i] = Math.floor(getNumericElement(data, srcOffset));
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
 * Element-wise ceil (round up).
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function ceil(t: Tensor): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("ceil is not defined for string dtype");
  }

  const out = new Float64Array(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("ceil is not defined for string dtype");
  }

  if (data instanceof BigInt64Array) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = contiguous
        ? t.offset + i
        : offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
      out[i] = Math.ceil(bigintToNumberSafe(getBigIntElement(data, srcOffset)));
    }
  } else {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = contiguous
        ? t.offset + i
        : offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
      out[i] = Math.ceil(getNumericElement(data, srcOffset));
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
 * Element-wise round to nearest integer.
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function round(t: Tensor): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("round is not defined for string dtype");
  }

  const out = new Float64Array(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("round is not defined for string dtype");
  }

  if (data instanceof BigInt64Array) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = contiguous
        ? t.offset + i
        : offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
      out[i] = Math.round(bigintToNumberSafe(getBigIntElement(data, srcOffset)));
    }
  } else {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = contiguous
        ? t.offset + i
        : offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
      out[i] = Math.round(getNumericElement(data, srcOffset));
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
 * Element-wise truncate (round toward zero).
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function trunc(t: Tensor): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("trunc is not defined for string dtype");
  }

  const out = new Float64Array(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("trunc is not defined for string dtype");
  }

  if (data instanceof BigInt64Array) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = contiguous
        ? t.offset + i
        : offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
      out[i] = Math.trunc(bigintToNumberSafe(getBigIntElement(data, srcOffset)));
    }
  } else {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = contiguous
        ? t.offset + i
        : offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
      out[i] = Math.trunc(getNumericElement(data, srcOffset));
    }
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: t.shape,
    dtype: "float64",
    device: t.device,
  });
}
