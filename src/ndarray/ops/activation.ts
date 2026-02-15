import {
  type Axis,
  DTypeError,
  dtypeToTypedArrayCtor,
  getBigIntElement,
  getNumericElement,
  normalizeAxis,
  type ScalarDType,
  type Shape,
} from "../../core";
import { isContiguous } from "../tensor/strides";
import { computeStrides, Tensor } from "../tensor/Tensor";
import { bigintToNumberSafe, flatOffset } from "./_internal";

function floatOutputDType(dtype: Tensor["dtype"]): "float32" | "float64" {
  // Preserve float32 precision, promote all other types to float64
  if (dtype === "float32") return "float32";
  return "float64";
}

function softplusScalar(x: number): number {
  if (x > 0) {
    return x + Math.log1p(Math.exp(-x));
  }
  return Math.log1p(Math.exp(x));
}

function toFloat64Dense(t: Tensor): Float64Array {
  const out = new Float64Array(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("Tensor must have numeric data");
  }

  if (data instanceof BigInt64Array) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      out[i] = bigintToNumberSafe(getBigIntElement(data, srcOffset));
    }
  } else {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      out[i] = getNumericElement(data, srcOffset);
    }
  }
  return out;
}

/**
 * Sigmoid activation function.
 *
 * Applies element-wise: sigmoid(x) = 1 / (1 + exp(-x))
 *
 * **Properties**:
 * - Output range: (0, 1)
 * - Smooth gradient
 * - Can suffer from vanishing gradients
 *
 * @example
 * ```ts
 * import { sigmoid, tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([-1, 0, 1]);
 * const result = sigmoid(x);  // [0.268..., 0.5, 0.731...]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-activations | Deepbox Activation Functions}
 */
export function sigmoid(t: Tensor): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("sigmoid is not defined for string dtype");
  }

  const dtype = floatOutputDType(t.dtype);
  const Ctor = dtypeToTypedArrayCtor(dtype);
  const out = new Ctor(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("sigmoid is not defined for string dtype");
  }

  if (data instanceof BigInt64Array) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      const val = bigintToNumberSafe(getBigIntElement(data, srcOffset));
      out[i] = 1 / (1 + Math.exp(-val));
    }
  } else {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      const val = getNumericElement(data, srcOffset);
      out[i] = 1 / (1 + Math.exp(-val));
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
 * Rectified Linear Unit activation.
 *
 * Applies element-wise: relu(x) = max(0, x)
 *
 * **Properties**:
 * - Output range: [0, ∞)
 * - Non-linear but simple
 * - Can suffer from dying ReLU problem
 *
 * @example
 * ```ts
 * import { relu, tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([-1, 0, 1]);
 * const result = relu(x);  // [0, 0, 1]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-activations | Deepbox Activation Functions}
 */
export function relu(t: Tensor): Tensor<Shape, ScalarDType> {
  if (t.dtype === "string") {
    throw new DTypeError("relu is not defined for string dtype");
  }

  const dtype = floatOutputDType(t.dtype);
  const Ctor = dtypeToTypedArrayCtor(dtype);
  const out = new Ctor(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("relu is not defined for string dtype");
  }

  if (data instanceof BigInt64Array) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      const val = bigintToNumberSafe(getBigIntElement(data, srcOffset));
      out[i] = Math.max(0, val);
    }
  } else {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      const val = getNumericElement(data, srcOffset);
      out[i] = Math.max(0, val);
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
 * Leaky ReLU activation.
 *
 * Applies element-wise: leaky_relu(x) = max(alpha * x, x)
 *
 * **Parameters**:
 * @param t - Input tensor
 * @param alpha - Slope for negative values (default: 0.01)
 *
 * @example
 * ```ts
 * import { leakyRelu, tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([-1, 0, 1]);
 * const result = leakyRelu(x, 0.1);  // [-0.1, 0, 1]
 * ```
 */
export function leakyRelu(t: Tensor, alpha = 0.01): Tensor<Shape, ScalarDType> {
  if (t.dtype === "string") {
    throw new DTypeError("leakyRelu is not defined for string dtype");
  }

  const dtype = floatOutputDType(t.dtype);
  const Ctor = dtypeToTypedArrayCtor(dtype);
  const out = new Ctor(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("leakyRelu is not defined for string dtype");
  }

  if (data instanceof BigInt64Array) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      const val = bigintToNumberSafe(getBigIntElement(data, srcOffset));
      out[i] = val > 0 ? val : alpha * val;
    }
  } else {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      const val = getNumericElement(data, srcOffset);
      out[i] = val > 0 ? val : alpha * val;
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
 * Exponential Linear Unit activation.
 *
 * Applies element-wise:
 * - elu(x) = x if x > 0
 * - elu(x) = alpha * (exp(x) - 1) if x <= 0
 *
 * **Parameters**:
 * @param t - Input tensor
 * @param alpha - Scale for negative values (default: 1.0)
 *
 * @example
 * ```ts
 * import { elu, tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([-1, 0, 1]);
 * const result = elu(x);
 * ```
 */
export function elu(t: Tensor, alpha: number = 1.0): Tensor<Shape, ScalarDType> {
  if (t.dtype === "string") {
    throw new DTypeError("elu is not defined for string dtype");
  }

  const out = new Float64Array(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("elu is not defined for string dtype");
  }

  if (data instanceof BigInt64Array) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      const val = bigintToNumberSafe(getBigIntElement(data, srcOffset));
      out[i] = val > 0 ? val : alpha * (Math.exp(val) - 1);
    }
  } else {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      const val = getNumericElement(data, srcOffset);
      out[i] = val > 0 ? val : alpha * (Math.exp(val) - 1);
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
 * Gaussian Error Linear Unit activation.
 *
 * Applies element-wise: gelu(x) = x * Φ(x)
 * where Φ(x) is the cumulative distribution function of the standard normal distribution.
 *
 * **Algorithm**: Can use tanh approximation or erf function
 *
 * @example
 * ```ts
 * import { gelu, tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([-1, 0, 1]);
 * const result = gelu(x);
 * ```
 *
 * @see Hendrycks & Gimpel (2016) "Gaussian Error Linear Units (GELUs)"
 */
export function gelu(t: Tensor): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("gelu is not defined for string dtype");
  }

  const out = new Float64Array(t.size);
  const sqrt2OverPi = Math.sqrt(2 / Math.PI);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("gelu is not defined for string dtype");
  }

  if (data instanceof BigInt64Array) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      const x = bigintToNumberSafe(getBigIntElement(data, srcOffset));
      const x3 = x * x * x;
      out[i] = 0.5 * x * (1 + Math.tanh(sqrt2OverPi * (x + 0.044715 * x3)));
    }
  } else {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      const x = getNumericElement(data, srcOffset);
      const x3 = x * x * x;
      out[i] = 0.5 * x * (1 + Math.tanh(sqrt2OverPi * (x + 0.044715 * x3)));
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
 * Softmax activation function.
 *
 * Normalizes input to probability distribution over classes.
 * Applies along specified axis: softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))
 *
 * Uses numerically stable computation by subtracting the maximum value
 * along the axis before exponentiating.
 *
 * **Parameters**:
 * @param t - Input tensor of any dimensionality
 * @param axis - Axis along which to compute softmax (default: -1, i.e., last axis)
 *
 * **Properties**:
 * - Output sums to 1 along the specified axis
 * - Output values in (0, 1)
 * - Supports tensors of any dimensionality
 *
 * Output dtype:
 * - `float64`
 *
 * @example
 * ```ts
 * import { softmax, tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([[1, 2, 3], [1, 2, 3]]);
 * const result = softmax(x, 1);  // Each row sums to 1
 *
 * const x3d = tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
 * const result3d = softmax(x3d, -1);  // Softmax along last axis
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-activations | Deepbox Activation Functions}
 */
export function softmax(t: Tensor, axis: Axis = -1): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("softmax is not defined for string dtype");
  }

  // Normalize negative axis
  const actualAxis = normalizeAxis(axis, t.ndim);

  const out = new Float64Array(t.size);

  // Convert data to contiguous float64 values in logical order
  const data = toFloat64Dense(t);

  // Compute the size of dimensions before, at, and after the softmax axis
  let outerSize = 1;
  for (let i = 0; i < actualAxis; i++) {
    outerSize *= t.shape[i] ?? 1;
  }
  const axisSize = t.shape[actualAxis] ?? 1;
  let innerSize = 1;
  for (let i = actualAxis + 1; i < t.ndim; i++) {
    innerSize *= t.shape[i] ?? 1;
  }

  // Reuse expBuffer across all slices to minimize allocations
  // This reduces memory allocations from O(outerSize * innerSize) to O(1)
  const expBuffer = new Float64Array(axisSize);

  // For each "slice" perpendicular to the softmax axis
  for (let outer = 0; outer < outerSize; outer++) {
    for (let inner = 0; inner < innerSize; inner++) {
      // Compute base offset for this slice
      // baseOffset = outer * (axisSize * innerSize) + inner
      const baseOffset = outer * axisSize * innerSize + inner;

      // Step 1: Find max along axis for numerical stability
      let maxVal = -Infinity;
      for (let k = 0; k < axisSize; k++) {
        const idx = baseOffset + k * innerSize;
        const val = data[idx] ?? 0;
        if (val > maxVal) maxVal = val;
      }

      // Step 2: Compute exp(x - max) and sum
      let sumExp = 0;
      for (let k = 0; k < axisSize; k++) {
        const idx = baseOffset + k * innerSize;
        const expVal = Math.exp((data[idx] ?? 0) - maxVal);
        expBuffer[k] = expVal;
        sumExp += expVal;
      }

      // Step 3: Normalize by sum
      for (let k = 0; k < axisSize; k++) {
        const idx = baseOffset + k * innerSize;
        out[idx] = (expBuffer[k] ?? 0) / sumExp;
      }
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
 * Log-Softmax activation function.
 *
 * Computes log(softmax(x)) in a numerically stable way.
 * Uses the identity: log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
 *
 * **Parameters**:
 * @param t - Input tensor of any dimensionality
 * @param axis - Axis along which to compute log-softmax (default: -1, i.e., last axis)
 *
 * **Properties**:
 * - More numerically stable than computing log(softmax(x)) directly
 * - Output values are log probabilities (negative values, sum to 0 when exp'd)
 * - Supports tensors of any dimensionality
 *
 * Output dtype:
 * - `float64`
 *
 * **Performance**:
 * - Time complexity: O(n) where n is the tensor size
 * - Space complexity: O(axisSize) for temporary buffer (reused across slices)
 * - More efficient than computing log(softmax(x)) separately
 *
 * @example
 * ```ts
 * import { logSoftmax, tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([[1, 2, 3]]);
 * const result = logSoftmax(x, 1);
 *
 * const x3d = tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
 * const result3d = logSoftmax(x3d, -1);  // Log-softmax along last axis
 * ```
 */
export function logSoftmax(t: Tensor, axis: Axis = -1): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("logSoftmax is not defined for string dtype");
  }

  // Normalize negative axis
  const actualAxis = normalizeAxis(axis, t.ndim);

  const out = new Float64Array(t.size);

  // Convert data to contiguous float64 values in logical order
  const data = toFloat64Dense(t);

  // Compute the size of dimensions before, at, and after the softmax axis
  let outerSize = 1;
  for (let i = 0; i < actualAxis; i++) {
    outerSize *= t.shape[i] ?? 1;
  }
  const axisSize = t.shape[actualAxis] ?? 1;
  let innerSize = 1;
  for (let i = actualAxis + 1; i < t.ndim; i++) {
    innerSize *= t.shape[i] ?? 1;
  }

  // For each "slice" perpendicular to the softmax axis
  for (let outer = 0; outer < outerSize; outer++) {
    for (let inner = 0; inner < innerSize; inner++) {
      // Compute base offset for this slice
      const baseOffset = outer * axisSize * innerSize + inner;

      // Step 1: Find max along axis for numerical stability
      let maxVal = -Infinity;
      for (let k = 0; k < axisSize; k++) {
        const idx = baseOffset + k * innerSize;
        const val = data[idx] ?? 0;
        if (val > maxVal) maxVal = val;
      }

      // Step 2: Compute log(sum(exp(x - max)))
      let sumExp = 0;
      for (let k = 0; k < axisSize; k++) {
        const idx = baseOffset + k * innerSize;
        sumExp += Math.exp((data[idx] ?? 0) - maxVal);
      }
      const logSumExp = maxVal + Math.log(sumExp);

      // Step 3: Compute log_softmax = x - log_sum_exp
      for (let k = 0; k < axisSize; k++) {
        const idx = baseOffset + k * innerSize;
        out[idx] = (data[idx] ?? 0) - logSumExp;
      }
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
 * Swish activation function (also known as SiLU).
 *
 * Applies element-wise: swish(x) = x * sigmoid(x)
 *
 * @example
 * ```ts
 * import { swish, tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([-1, 0, 1]);
 * const result = swish(x);
 * ```
 *
 * @see Ramachandran et al. (2017) "Searching for Activation Functions"
 */
export function swish(t: Tensor): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("swish is not defined for string dtype");
  }

  const out = new Float64Array(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("swish is not defined for string dtype");
  }

  if (data instanceof BigInt64Array) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      const val = bigintToNumberSafe(getBigIntElement(data, srcOffset));
      out[i] = val / (1 + Math.exp(-val));
    }
  } else {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      const val = getNumericElement(data, srcOffset);
      out[i] = val / (1 + Math.exp(-val));
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
 * Mish activation function.
 *
 * Applies element-wise: mish(x) = x * tanh(softplus(x))
 * where softplus(x) = log(1 + exp(x))
 *
 * @example
 * ```ts
 * import { mish, tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([-1, 0, 1]);
 * const result = mish(x);
 * ```
 *
 * @see Misra (2019) "Mish: A Self Regularized Non-Monotonic Activation Function"
 */
export function mish(t: Tensor): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("mish is not defined for string dtype");
  }

  const out = new Float64Array(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("mish is not defined for string dtype");
  }

  if (data instanceof BigInt64Array) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      const x = bigintToNumberSafe(getBigIntElement(data, srcOffset));
      out[i] = x * Math.tanh(softplusScalar(x));
    }
  } else {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      const x = getNumericElement(data, srcOffset);
      out[i] = x * Math.tanh(softplusScalar(x));
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
 * Softplus activation function.
 *
 * Smooth approximation of ReLU: softplus(x) = log(1 + exp(x))
 *
 * @example
 * ```ts
 * import { softplus, tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([-1, 0, 1]);
 * const result = softplus(x);
 * ```
 */
export function softplus(t: Tensor): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("softplus is not defined for string dtype");
  }

  const out = new Float64Array(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("softplus is not defined for string dtype");
  }

  if (data instanceof BigInt64Array) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      const val = bigintToNumberSafe(getBigIntElement(data, srcOffset));
      out[i] = softplusScalar(val);
    }
  } else {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      const val = getNumericElement(data, srcOffset);
      out[i] = softplusScalar(val);
    }
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: t.shape,
    dtype: "float64",
    device: t.device,
  });
}
