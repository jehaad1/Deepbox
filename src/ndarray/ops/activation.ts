/**
 * Activation functions on tensors: sigmoid, relu, relu6, leakyRelu, elu, selu,
 * celu, gelu, softmax, logSoftmax, swish, mish, softplus, softsign, logSigmoid,
 * hardsigmoid, hardswish, hardtanh, hardshrink, softshrink and tanhshrink.
 *
 * Output dtypes: float input (float16, bfloat16, float32, float64) keeps its
 * dtype; values are computed in float64 and rounded to that dtype. Integer and
 * bool input computes in `float32`. `relu`, `relu6` and `hardtanh` are clips, so
 * integer input keeps its integer dtype (`bool` gives `int32` for `relu` and
 * `relu6`).
 *
 * @see {@link https://deepbox.dev/docs/ndarray-activations | Deepbox Activation Functions}
 */

import { type Axis, DTypeError, InvalidParameterError, normalizeAxis } from "../../core";
import { toFloatDType } from "../../core/utils/dtype_utils";
import { isContiguous } from "../tensor/strides";
import { computeStrides, Tensor } from "../tensor/Tensor";
import { allocFloat, type FloatDType, flatOffset, floatResult, readNumbers } from "./_internal";
import { add, addScalar, clip as clipOp, mul, mulScalar, neg } from "./arithmetic";
import { dispatchUnary } from "./device_dispatch";
import { exp as expOp } from "./math";
import { tanh as tanhOp } from "./trigonometry";

function requireNotString(t: Tensor, name: string): void {
  if (t.dtype === "string") {
    throw new DTypeError(`${name} is not defined for string dtype`);
  }
}

function requireFiniteParam(value: number, param: string, fn: string): void {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new InvalidParameterError(
      `${fn}: ${param} must be a finite number; received ${String(value)}`,
      param,
      value
    );
  }
}

/** Numerically stable logistic function: never exponentiates a positive argument. */
function sigmoidScalar(x: number): number {
  if (x >= 0) return 1 / (1 + Math.exp(-x));
  const e = Math.exp(x);
  return e / (1 + e);
}

/** log(1 + exp(x)) without overflow for large x or cancellation for very negative x. */
function softplusScalar(x: number): number {
  if (x > 0) {
    return x + Math.log1p(Math.exp(-x));
  }
  return Math.log1p(Math.exp(x));
}

/** sqrt(2 / pi), the scale inside the tanh approximation of GELU. */
const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);

/**
 * Tanh-approximation GELU: 0.5 x (1 + tanh(u)) with u = sqrt(2/pi) (x + 0.044715 x^3).
 *
 * Since 1 + tanh(u) = 2 / (1 + exp(-2u)), the result is evaluated as
 * x / (1 + exp(-2u)). That avoids the cancellation in `1 + tanh(u)` for negative
 * x, where the textbook form loses all significant digits, and it stays finite
 * for huge |x|. The limit at -Infinity is 0.
 */
function geluScalar(x: number): number {
  if (x === Number.NEGATIVE_INFINITY) return 0;
  const u = SQRT_2_OVER_PI * (x + 0.044715 * x * x * x);
  return x / (1 + Math.exp(-2 * u));
}

const INV_SQRT_2PI = 1 / Math.sqrt(2 * Math.PI);
const TWO_OVER_SQRT_PI = 2 / Math.sqrt(Math.PI);

/** erf(z) for 0 <= z < 1 from its Maclaurin series; all terms are positive, so no cancellation. */
function erfSeries(z: number): number {
  const z2 = 2 * z * z;
  let term = z;
  let total = z;
  for (let n = 1; n < 100; n++) {
    term *= z2 / (2 * n + 1);
    total += term;
    if (term < total * 1e-17) break;
  }
  return TWO_OVER_SQRT_PI * Math.exp(-z * z) * total;
}

/** erfc(z) for z >= 1 from its continued fraction z + (1/2)/(z + (2/2)/(z + (3/2)/(z + ...))). */
function erfcContinuedFraction(z: number): number {
  const depth = z < 1.5 ? 200 : z < 3 ? 80 : 40;
  let f = z;
  for (let n = depth; n >= 1; n--) f = z + n / 2 / f;
  return Math.exp(-z * z) / Math.sqrt(Math.PI) / f;
}

/**
 * Standard normal CDF, accurate to about 2e-15 relative error over the whole
 * real line (the lower tail is computed from erfc, so it does not cancel).
 */
function normalCdfScalar(x: number): number {
  const z = Math.abs(x) * Math.SQRT1_2;
  if (z < 1) {
    const e = erfSeries(z);
    return x >= 0 ? 0.5 + 0.5 * e : 0.5 - 0.5 * e;
  }
  const tail = 0.5 * erfcContinuedFraction(z);
  return x >= 0 ? 1 - tail : tail;
}

/** Exact GELU, `x * Phi(x)`; the limit at -Infinity is 0. */
function geluExactScalar(x: number): number {
  if (x === Number.NEGATIVE_INFINITY) return 0;
  return x * normalCdfScalar(x);
}

/** Derivative of the tanh-approximation GELU. */
function geluTanhDerivativeScalar(x: number): number {
  if (!Number.isFinite(x)) return x === Number.POSITIVE_INFINITY ? 1 : x === -Infinity ? 0 : x;
  const inner = SQRT_2_OVER_PI * (x + 0.044715 * x * x * x);
  const th = Math.tanh(inner);
  const cdf = 0.5 * (1 + th);
  const pdf = SQRT_2_OVER_PI * (1 + 3 * 0.044715 * x * x) * (1 - th * th);
  return cdf + 0.5 * x * pdf;
}

/** Derivative of the exact GELU, `Phi(x) + x * phi(x)`. */
function geluExactDerivativeScalar(x: number): number {
  if (!Number.isFinite(x)) return x === Number.POSITIVE_INFINITY ? 1 : x === -Infinity ? 0 : x;
  return normalCdfScalar(x) + x * INV_SQRT_2PI * Math.exp(-0.5 * x * x);
}

/**
 * Taylor coefficients of x - tanh(x) = c0 x^3 + c1 x^5 + c2 x^7 + ... (radius of
 * convergence pi / 2). Twenty terms are accurate to machine precision for |x| < 0.5.
 */
const X_MINUS_TANH_COEFFS: readonly number[] = [
  0.3333333333333333, -0.13333333333333333, 0.05396825396825397, -0.021869488536155203,
  0.008863235529902197, -0.003592128036572481, 0.0014558343870513183, -0.000590027440945586,
  0.00023912911424355248, -9.691537956929451e-5, 3.927832388331683e-5, -1.5918905069328964e-5,
  6.451689215655431e-6, -2.6147711512907546e-6, 1.0597268320104654e-6, -4.294911078273806e-7,
  1.7406618963571648e-7, -7.054636946400968e-8, 2.859136662305254e-8, -1.1587644432798853e-8,
];

/**
 * x - tanh(x). Direct subtraction cancels almost completely for small |x|
 * (the true value is about x^3 / 3), so small arguments use the Taylor series.
 */
function tanhshrinkScalar(x: number): number {
  const ax = Math.abs(x);
  if (ax < 0.5) {
    const x2 = x * x;
    let acc = 0;
    for (let k = X_MINUS_TANH_COEFFS.length - 1; k >= 0; k--) {
      acc = acc * x2 + (X_MINUS_TANH_COEFFS[k] as number);
    }
    return acc * x2 * x;
  }
  return x - Math.tanh(x);
}

/**
 * Apply a scalar function to every element, in float64, and store the results in the
 * float dtype of the input (integer and bool input gives `float32`).
 */
function mapFloat(t: Tensor, name: string, f: (x: number) => number): Tensor {
  requireNotString(t, name);
  const dtype = toFloatDType(t.dtype);
  const src = readNumbers(t, name);
  const n = t.size;
  const out = allocFloat(dtype, n);
  for (let i = 0; i < n; i++) out[i] = f(src[i] as number);
  return floatResult(out, t.shape, dtype, t.device);
}

/**
 * Shared kernel of {@link relu} and {@link relu6}: `min(max(x, 0), hi)`. Float input keeps
 * its dtype, integer input keeps its integer dtype (`bool` gives `int32`). NaN propagates.
 */
function reluKernel(t: Tensor, name: string, hi: number): Tensor {
  requireNotString(t, name);
  const n = t.size;

  if (t.data instanceof BigInt64Array) {
    const data = t.data;
    const out = new BigInt64Array(n);
    const hiBig = Number.isFinite(hi) ? BigInt(hi) : undefined;
    const logicalStrides = computeStrides(t.shape);
    const contiguous = isContiguous(t.shape, t.strides);
    for (let i = 0; i < n; i++) {
      let v = data[flatOffset(i, t.offset, contiguous, logicalStrides, t.strides)] as bigint;
      if (v < 0n) v = 0n;
      if (hiBig !== undefined && v > hiBig) v = hiBig;
      out[i] = v;
    }
    return Tensor.fromTypedArray({ data: out, shape: t.shape, dtype: "int64", device: t.device });
  }

  if (t.dtype === "int32" || t.dtype === "uint8" || t.dtype === "bool") {
    const src = readNumbers(t, name);
    const isUint8 = t.dtype === "uint8";
    const out = isUint8 ? new Uint8Array(n) : new Int32Array(n);
    for (let i = 0; i < n; i++) out[i] = Math.min(Math.max(src[i] as number, 0), hi);
    return Tensor.fromTypedArray({
      data: out,
      shape: t.shape,
      dtype: isUint8 ? "uint8" : "int32",
      device: t.device,
    });
  }

  const dtype = toFloatDType(t.dtype);
  const out = allocFloat(dtype, n);
  const src = readNumbers(t, name);
  for (let i = 0; i < n; i++) out[i] = Math.min(Math.max(src[i] as number, 0), hi);
  return floatResult(out, t.shape, dtype, t.device);
}

/**
 * Sigmoid activation function.
 *
 * Applies element-wise: sigmoid(x) = 1 / (1 + exp(-x)), evaluated so that
 * large negative inputs do not lose the (denormal) tail.
 *
 * **Properties**:
 * - Output range: (0, 1)
 * - Smooth gradient
 * - Can suffer from vanishing gradients
 *
 * @param t - Input tensor
 * @returns Tensor of the same shape (float dtypes are kept, integer and bool input gives `float32`)
 * @throws {DTypeError} For `string` tensors
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
  requireNotString(t, "sigmoid");

  if (t.device !== "cpu") {
    const onDevice = dispatchUnary("sigmoid", t);
    if (onDevice) return onDevice;
  }

  const dtype = toFloatDType(t.dtype);
  const out = allocFloat(dtype, t.size);
  const src = readNumbers(t, "sigmoid");
  for (let i = 0; i < t.size; i++) {
    out[i] = sigmoidScalar(src[i] as number);
  }

  return floatResult(out, t.shape, dtype, t.device);
}

/**
 * Rectified Linear Unit activation.
 *
 * Applies element-wise: relu(x) = max(0, x). NaN propagates.
 *
 * **Properties**:
 * - Output range: [0, ∞)
 * - Non-linear but simple
 * - Can suffer from dying ReLU problem
 *
 * @param t - Input tensor
 * @returns Tensor of the same shape and dtype (`bool` input gives `int32`)
 * @throws {DTypeError} For `string` tensors
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
export function relu(t: Tensor): Tensor {
  requireNotString(t, "relu");

  if (t.device !== "cpu") {
    const onDevice = dispatchUnary("relu", t);
    if (onDevice) return onDevice;
  }

  return reluKernel(t, "relu", Number.POSITIVE_INFINITY);
}

/**
 * ReLU6 activation: relu capped at 6.
 *
 * Applies element-wise: relu6(x) = min(max(0, x), 6). NaN propagates.
 *
 * @param t - Input tensor
 * @returns Tensor of the same shape and dtype (`bool` input gives `int32`)
 * @throws {DTypeError} For `string` tensors
 *
 * @example
 * ```ts
 * import { relu6, tensor } from 'deepbox/ndarray';
 *
 * relu6(tensor([-2, 3, 8]));  // [0, 3, 6]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-activations | Deepbox Activation Functions}
 */
export function relu6(t: Tensor): Tensor {
  return reluKernel(t, "relu6", 6);
}

/**
 * Leaky ReLU activation.
 *
 * Applies element-wise: leaky_relu(x) = x for x > 0, alpha * x otherwise.
 * NaN propagates.
 *
 * @param t - Input tensor
 * @param alpha - Slope for negative values (default: 0.01)
 * @returns Tensor of the same shape (float dtypes are kept, integer and bool input gives `float32`)
 * @throws {InvalidParameterError} If `alpha` is not a finite number
 * @throws {DTypeError} For `string` tensors
 *
 * @example
 * ```ts
 * import { leakyRelu, tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([-1, 0, 1]);
 * const result = leakyRelu(x, 0.1);  // [-0.1, 0, 1]
 * ```
 */
export function leakyRelu(t: Tensor, alpha = 0.01): Tensor {
  requireNotString(t, "leakyRelu");
  requireFiniteParam(alpha, "alpha", "leakyRelu");
  return mapFloat(t, "leakyRelu", (val) => (val > 0 ? val : alpha * val));
}

/**
 * Exponential Linear Unit activation.
 *
 * Applies element-wise:
 * - elu(x) = x if x > 0
 * - elu(x) = alpha * (exp(x) - 1) if x <= 0
 *
 * The negative branch uses `expm1`, so values near zero keep full precision.
 *
 * @param t - Input tensor
 * @param alpha - Scale for negative values (default: 1.0)
 * @returns Tensor of the same shape (float dtypes are kept, integer and bool input gives `float32`)
 * @throws {InvalidParameterError} If `alpha` is not a finite number
 * @throws {DTypeError} For `string` tensors
 *
 * @example
 * ```ts
 * import { elu, tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([-1, 0, 1]);
 * const result = elu(x);  // [-0.632..., 0, 1]
 * ```
 */
export function elu(t: Tensor, alpha: number = 1.0): Tensor {
  requireNotString(t, "elu");
  requireFiniteParam(alpha, "alpha", "elu");
  return mapFloat(t, "elu", (val) => (val > 0 ? val : alpha * Math.expm1(val)));
}

/** Alpha of the SELU activation (Klambauer et al., 2017), as used by PyTorch. */
const SELU_ALPHA = 1.6732632423543772;
/** Scale of the SELU activation, as used by PyTorch. */
const SELU_SCALE = 1.0507009873554805;

/**
 * Scaled Exponential Linear Unit activation.
 *
 * Applies element-wise: selu(x) = scale * (x if x > 0, else alpha * (exp(x) - 1)) with
 * alpha = 1.6732632423543772 and scale = 1.0507009873554805 (PyTorch's constants).
 *
 * @param t - Input tensor
 * @returns Tensor of the same shape (float dtypes are kept, integer and bool input gives `float32`)
 * @throws {DTypeError} For `string` tensors
 *
 * @example
 * ```ts
 * import { selu, tensor } from 'deepbox/ndarray';
 *
 * selu(tensor([-1, 0, 1]));  // [-1.1113..., 0, 1.0507...]
 * ```
 *
 * @see Klambauer et al. (2017) "Self-Normalizing Neural Networks"
 */
export function selu(t: Tensor): Tensor {
  return mapFloat(t, "selu", (val) =>
    val > 0 ? SELU_SCALE * val : SELU_SCALE * SELU_ALPHA * Math.expm1(val)
  );
}

/**
 * Continuously differentiable Exponential Linear Unit activation.
 *
 * Applies element-wise: celu(x) = max(0, x) + min(0, alpha * (exp(x / alpha) - 1)).
 *
 * @param t - Input tensor
 * @param alpha - Scale of the negative branch; must be finite and non-zero (default: 1.0)
 * @returns Tensor of the same shape (float dtypes are kept, integer and bool input gives `float32`)
 * @throws {InvalidParameterError} If `alpha` is zero or not a finite number
 * @throws {DTypeError} For `string` tensors
 *
 * @example
 * ```ts
 * import { celu, tensor } from 'deepbox/ndarray';
 *
 * celu(tensor([-1, 0, 1]), 2);  // [-0.7869..., 0, 1]
 * ```
 *
 * @see Barron (2017) "Continuously Differentiable Exponential Linear Units"
 */
export function celu(t: Tensor, alpha: number = 1.0): Tensor {
  requireNotString(t, "celu");
  requireFiniteParam(alpha, "alpha", "celu");
  if (alpha === 0) {
    throw new InvalidParameterError("celu: alpha must be non-zero", "alpha", alpha);
  }
  return mapFloat(t, "celu", (val) => (val > 0 ? val : alpha * Math.expm1(val / alpha)));
}

/** Which GELU formula to use: the tanh approximation, or the exact erf form (`"none"`). */
export type GeluApproximation = "tanh" | "none";

/** Options of {@link gelu}. */
export type GeluOptions = {
  /** `"tanh"` (default) for the tanh approximation, `"none"` for the exact erf form. */
  approximate?: GeluApproximation;
};

function requireGeluApproximation(approximate: unknown, fn: string): GeluApproximation {
  if (approximate !== "tanh" && approximate !== "none") {
    throw new InvalidParameterError(
      `${fn}: approximate must be "tanh" or "none"; received ${String(approximate)}`,
      "approximate",
      approximate
    );
  }
  return approximate;
}

/**
 * Gaussian Error Linear Unit activation.
 *
 * With `approximate: "tanh"` (the default) this applies element-wise
 * gelu(x) ~ 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))), the approximation
 * from Hendrycks & Gimpel (2016). It matches PyTorch's `gelu(approximate="tanh")`,
 * the device kernels and the autograd gradient. With `approximate: "none"` it
 * is the exact form x * Phi(x) = 0.5 x (1 + erf(x / sqrt(2))), which is what
 * PyTorch's `gelu` and `nn.GELU()` compute by default. The two differ by up to
 * about 5e-4. `gelu(-Infinity)` is 0.
 *
 * Tensors on a kernel device run the tanh form as a single kernel and the exact
 * form from the `erf` kernel.
 *
 * @param t - Input tensor
 * @param options - `{ approximate?: "tanh" | "none" }`. A bare `"tanh"` or `"none"`
 *   string is also accepted, for backward compatibility
 * @returns Tensor of the same shape (float dtypes are kept, integer and bool input gives `float32`)
 * @throws {InvalidParameterError} If `approximate` is neither `"tanh"` nor `"none"`
 * @throws {DTypeError} For `string` tensors
 *
 * @example
 * ```ts
 * import { gelu, tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([-1, 0, 1]);
 * const result = gelu(x);                          // [-0.158..., 0, 0.841...]
 * const exact = gelu(x, { approximate: "none" });  // [-0.1586..., 0, 0.8413...]
 * ```
 *
 * @see Hendrycks & Gimpel (2016) "Gaussian Error Linear Units (GELUs)"
 */
export function gelu(t: Tensor, options: GeluApproximation | GeluOptions = {}): Tensor {
  const approximate = typeof options === "string" ? options : (options?.approximate ?? "tanh");
  return geluCore(t, approximate);
}

function geluCore(t: Tensor, approximate: GeluApproximation): Tensor {
  requireNotString(t, "gelu");
  const mode = requireGeluApproximation(approximate, "gelu");

  if (t.device !== "cpu") {
    if (mode === "tanh") {
      const onDevice = dispatchUnary("gelu", t);
      if (onDevice) return onDevice;
    } else if (t.isDeviceTensor) {
      // x * Phi(x) with Phi(x) = 0.5 * (1 + erf(x / sqrt(2))).
      const erfT = dispatchUnary("erf", mulScalar(t, Math.SQRT1_2));
      if (erfT) return mul(t, addScalar(mulScalar(erfT, 0.5), 0.5));
    }
  }

  const dtype = toFloatDType(t.dtype);
  const out = allocFloat(dtype, t.size);
  const src = readNumbers(t, "gelu");
  if (mode === "tanh") {
    for (let i = 0; i < t.size; i++) out[i] = geluScalar(src[i] as number);
  } else {
    for (let i = 0; i < t.size; i++) out[i] = geluExactScalar(src[i] as number);
  }

  return floatResult(out, t.shape, dtype, t.device);
}

/**
 * Derivative of {@link gelu} with respect to its input, for the autograd layer.
 *
 * Host tensors give a tensor of the input's float dtype (`float32` for integer input).
 * Device tensors are differentiated with device operations (`tanh`, `erf`, `exp` and
 * element-wise arithmetic), so no host readback is needed and the result keeps the
 * input's device dtype.
 *
 * @param t - Input of the forward pass
 * @param approximate - Same choice that was used for the forward pass
 * @internal
 */
export function geluDerivative(t: Tensor, approximate: GeluApproximation = "tanh"): Tensor {
  requireNotString(t, "gelu");
  const mode = requireGeluApproximation(approximate, "gelu");

  if (t.isDeviceTensor) {
    const x2 = mul(t, t);
    if (mode === "tanh") {
      // cdf + 0.5 x pdf with cdf = 0.5 (1 + th), pdf = k (1 + 3 c x^2) (1 - th^2).
      const inner = mulScalar(mul(t, addScalar(mulScalar(x2, 0.044715), 1)), SQRT_2_OVER_PI);
      const th = tanhOp(inner);
      const cdf = mulScalar(addScalar(th, 1), 0.5);
      const sech2 = addScalar(neg(mul(th, th)), 1);
      const poly = addScalar(mulScalar(x2, 3 * 0.044715), 1);
      const pdf = mulScalar(mul(poly, sech2), SQRT_2_OVER_PI);
      return add(cdf, mulScalar(mul(t, pdf), 0.5));
    }
    const erfT = dispatchUnary("erf", mulScalar(t, Math.SQRT1_2));
    if (erfT) {
      const cdf = addScalar(mulScalar(erfT, 0.5), 0.5);
      const pdf = mulScalar(expOp(mulScalar(x2, -0.5)), INV_SQRT_2PI);
      return add(cdf, mul(t, pdf));
    }
  }

  const dtype = toFloatDType(t.dtype);
  const out = allocFloat(dtype, t.size);
  const src = readNumbers(t, "gelu");
  if (mode === "tanh") {
    for (let i = 0; i < t.size; i++) out[i] = geluTanhDerivativeScalar(src[i] as number);
  } else {
    for (let i = 0; i < t.size; i++) out[i] = geluExactDerivativeScalar(src[i] as number);
  }
  return floatResult(out, t.shape, dtype, t.device);
}

/**
 * Shared kernel of {@link softmax} and {@link logSoftmax}. Each 1-D slice along
 * `axis` is shifted by its maximum before exponentiating, so large inputs do
 * not overflow. A 0-d tensor is treated as a single-element slice. The sums are
 * accumulated in float64 and the result is stored in the input's float dtype.
 */
function softmaxKernel(t: Tensor, axis: Axis, name: string, logSpace: boolean): Tensor {
  requireNotString(t, name);

  if (t.ndim === 0 && typeof axis === "number" && axis !== 0 && axis !== -1) {
    throw new InvalidParameterError(
      `${name}: axis ${String(axis)} is out of range for a 0-d tensor (expected 0 or -1)`,
      "axis",
      axis
    );
  }
  const shape = t.ndim === 0 ? [1] : t.shape;
  const actualAxis = normalizeAxis(axis, shape.length);

  const dtype: FloatDType = toFloatDType(t.dtype);
  const out = allocFloat(dtype, t.size);
  const src = readNumbers(t, name);

  // Sizes of the dimensions before, at, and after the axis.
  let outerSize = 1;
  for (let i = 0; i < actualAxis; i++) {
    outerSize *= shape[i] ?? 1;
  }
  const axisSize = shape[actualAxis] ?? 1;
  let innerSize = 1;
  for (let i = actualAxis + 1; i < shape.length; i++) {
    innerSize *= shape[i] ?? 1;
  }

  // Reused across all slices: O(axisSize) memory instead of O(size).
  const expBuffer = logSpace ? null : new Float64Array(axisSize);

  for (let outer = 0; outer < outerSize; outer++) {
    for (let inner = 0; inner < innerSize; inner++) {
      const baseOffset = outer * axisSize * innerSize + inner;

      let maxVal = Number.NEGATIVE_INFINITY;
      for (let k = 0; k < axisSize; k++) {
        const val = src[baseOffset + k * innerSize] as number;
        if (val > maxVal) maxVal = val;
      }

      let sumExp = 0;
      if (expBuffer) {
        for (let k = 0; k < axisSize; k++) {
          const e = Math.exp((src[baseOffset + k * innerSize] as number) - maxVal);
          expBuffer[k] = e;
          sumExp += e;
        }
        for (let k = 0; k < axisSize; k++) {
          out[baseOffset + k * innerSize] = (expBuffer[k] as number) / sumExp;
        }
      } else {
        for (let k = 0; k < axisSize; k++) {
          sumExp += Math.exp((src[baseOffset + k * innerSize] as number) - maxVal);
        }
        const logSumExp = maxVal + Math.log(sumExp);
        for (let k = 0; k < axisSize; k++) {
          const idx = baseOffset + k * innerSize;
          out[idx] = (src[idx] as number) - logSumExp;
        }
      }
    }
  }

  return floatResult(out, t.shape, dtype, t.device);
}

/**
 * Softmax activation function.
 *
 * Normalizes input to a probability distribution along `axis`:
 * softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x))).
 *
 * Subtracting the maximum along the axis before exponentiating keeps large
 * inputs from overflowing. A slice that contains NaN, or consists only of
 * infinities, yields NaN for that slice (as in PyTorch).
 *
 * **Properties**:
 * - Output sums to 1 along the specified axis
 * - Output values in [0, 1]
 * - Supports tensors of any dimensionality, including 0-d (result 1)
 *
 * @param t - Input tensor of any dimensionality
 * @param axis - Axis along which to compute softmax (default: -1, the last axis)
 * @returns Tensor of the same shape (float dtypes are kept, integer and bool input gives `float32`)
 * @throws {InvalidParameterError} If `axis` is out of range
 * @throws {DTypeError} For `string` tensors
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
  return softmaxKernel(t, axis, "softmax", false);
}

/**
 * Log-Softmax activation function.
 *
 * Computes log(softmax(x)) in a numerically stable way using
 * log_softmax(x) = x - max(x) - log(sum(exp(x - max(x)))).
 *
 * **Properties**:
 * - More accurate than computing log(softmax(x)) directly (no log of a
 *   value that underflowed to 0)
 * - Output values are log probabilities (<= 0; exp of them sums to 1)
 * - Time O(n), extra memory O(1)
 *
 * @param t - Input tensor of any dimensionality
 * @param axis - Axis along which to compute log-softmax (default: -1, the last axis)
 * @returns Tensor of the same shape (float dtypes are kept, integer and bool input gives `float32`)
 * @throws {InvalidParameterError} If `axis` is out of range
 * @throws {DTypeError} For `string` tensors
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
  return softmaxKernel(t, axis, "logSoftmax", true);
}

/**
 * Swish activation function (also known as SiLU).
 *
 * Applies element-wise: swish(x) = x * sigmoid(x). The limit at -Infinity is 0.
 *
 * @param t - Input tensor
 * @returns Tensor of the same shape (float dtypes are kept, integer and bool input gives `float32`)
 * @throws {DTypeError} For `string` tensors
 *
 * @example
 * ```ts
 * import { swish, tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([-1, 0, 1]);
 * const result = swish(x);  // [-0.268..., 0, 0.731...]
 * ```
 *
 * @see Ramachandran et al. (2017) "Searching for Activation Functions"
 */
export function swish(t: Tensor): Tensor {
  // swish(-Inf) limit is 0; the raw formula gives -Inf/Inf = NaN
  return mapFloat(t, "swish", (val) =>
    val === Number.NEGATIVE_INFINITY ? 0 : val / (1 + Math.exp(-val))
  );
}

/**
 * Mish activation function.
 *
 * Applies element-wise: mish(x) = x * tanh(softplus(x))
 * where softplus(x) = log(1 + exp(x)). The limit at -Infinity is 0.
 *
 * @param t - Input tensor
 * @returns Tensor of the same shape (float dtypes are kept, integer and bool input gives `float32`)
 * @throws {DTypeError} For `string` tensors
 *
 * @example
 * ```ts
 * import { mish, tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([-1, 0, 1]);
 * const result = mish(x);  // [-0.303..., 0, 0.865...]
 * ```
 *
 * @see Misra (2019) "Mish: A Self Regularized Non-Monotonic Activation Function"
 */
export function mish(t: Tensor): Tensor {
  return mapFloat(t, "mish", (x) =>
    x === Number.NEGATIVE_INFINITY ? 0 : x * Math.tanh(softplusScalar(x))
  );
}

/**
 * Softplus activation function.
 *
 * Smooth approximation of ReLU: softplus(x) = log(1 + exp(x)), computed without
 * overflow for large x or cancellation for very negative x.
 *
 * @param t - Input tensor
 * @returns Tensor of the same shape (float dtypes are kept, integer and bool input gives
 *   `float32`; device tensors keep their dtype)
 * @throws {DTypeError} For `string` tensors
 *
 * @example
 * ```ts
 * import { softplus, tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([-1, 0, 1]);
 * const result = softplus(x);  // [0.313..., 0.693..., 1.313...]
 * ```
 */
export function softplus(t: Tensor): Tensor {
  requireNotString(t, "softplus");

  if (t.device !== "cpu") {
    const onDevice = dispatchUnary("softplus", t);
    if (onDevice) return onDevice;
  }

  return mapFloat(t, "softplus", softplusScalar);
}

/**
 * Softsign activation: x / (1 + |x|).
 *
 * Output range is (-1, 1). Infinite inputs give NaN, as in PyTorch.
 *
 * @param t - Input tensor
 * @returns Tensor of the same shape (float dtypes are kept, integer and bool input gives `float32`)
 * @throws {DTypeError} For `string` tensors
 *
 * @example
 * ```ts
 * import { softsign, tensor } from 'deepbox/ndarray';
 *
 * softsign(tensor([-2, 0, 2]));  // [-0.666..., 0, 0.666...]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-activations | Deepbox Activation Functions}
 */
export function softsign(t: Tensor): Tensor {
  return mapFloat(t, "softsign", (x) => x / (1 + Math.abs(x)));
}

/**
 * Log-sigmoid activation: log(1 / (1 + exp(-x))).
 *
 * Computed as min(x, 0) - log1p(exp(-|x|)), so it neither overflows for very
 * negative x nor loses precision for large positive x.
 *
 * @param t - Input tensor
 * @returns Tensor of the same shape (float dtypes are kept, integer and bool input gives `float32`)
 * @throws {DTypeError} For `string` tensors
 *
 * @example
 * ```ts
 * import { logSigmoid, tensor } from 'deepbox/ndarray';
 *
 * logSigmoid(tensor([-1, 0, 1]));  // [-1.3132..., -0.6931..., -0.3132...]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-activations | Deepbox Activation Functions}
 */
export function logSigmoid(t: Tensor): Tensor {
  return mapFloat(t, "logSigmoid", (x) => Math.min(x, 0) - Math.log1p(Math.exp(-Math.abs(x))));
}

/**
 * Hard sigmoid activation: relu6(x + 3) / 6.
 *
 * Piecewise linear: 0 for x <= -3, 1 for x >= 3 and x / 6 + 1/2 in between.
 *
 * @param t - Input tensor
 * @returns Tensor of the same shape (float dtypes are kept, integer and bool input gives `float32`)
 * @throws {DTypeError} For `string` tensors
 *
 * @example
 * ```ts
 * import { hardsigmoid, tensor } from 'deepbox/ndarray';
 *
 * hardsigmoid(tensor([-4, 0, 1.5, 4]));  // [0, 0.5, 0.75, 1]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-activations | Deepbox Activation Functions}
 */
export function hardsigmoid(t: Tensor): Tensor {
  return mapFloat(t, "hardsigmoid", (x) => Math.min(Math.max(x + 3, 0), 6) / 6);
}

/**
 * Hard swish activation: x * relu6(x + 3) / 6.
 *
 * @param t - Input tensor
 * @returns Tensor of the same shape (float dtypes are kept, integer and bool input gives `float32`)
 * @throws {DTypeError} For `string` tensors
 *
 * @example
 * ```ts
 * import { hardswish, tensor } from 'deepbox/ndarray';
 *
 * hardswish(tensor([-4, 0, 1, 4]));  // [0, 0, 0.666..., 4]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-activations | Deepbox Activation Functions}
 */
export function hardswish(t: Tensor): Tensor {
  return mapFloat(t, "hardswish", (x) => (x * Math.min(Math.max(x + 3, 0), 6)) / 6);
}

/**
 * Applies the HardTanh activation function element-wise.
 *
 * HardTanh(x) = max(minVal, min(maxVal, x)). This is a clip, so the input
 * dtype is kept (see {@link clip} for integer-tensor rules).
 *
 * @param t - Input tensor
 * @param minVal - Lower bound (default: -1)
 * @param maxVal - Upper bound (default: 1)
 * @returns Tensor of the same shape
 * @throws {InvalidParameterError} If `minVal > maxVal`
 * @throws {DTypeError} For `string` tensors
 *
 * @example
 * ```ts
 * import { hardtanh, tensor } from 'deepbox/ndarray';
 *
 * hardtanh(tensor([-2, 0.5, 3]));  // [-1, 0.5, 1]
 * ```
 */
export function hardtanh(t: Tensor, minVal = -1, maxVal = 1): Tensor {
  if (minVal > maxVal) {
    throw new InvalidParameterError(
      `hardtanh: minVal (${minVal}) must be <= maxVal (${maxVal})`,
      "minVal/maxVal",
      { minVal, maxVal }
    );
  }
  return clipOp(t, minVal, maxVal);
}

function requireShrinkLambda(lambd: number, fn: string): void {
  if (typeof lambd !== "number" || Number.isNaN(lambd) || lambd < 0) {
    throw new InvalidParameterError(
      `${fn}: lambd must be a non-negative number; received ${String(lambd)}`,
      "lambd",
      lambd
    );
  }
}

/**
 * Hard shrinkage: x where |x| > lambd, otherwise 0.
 *
 * Values exactly at +-lambd become 0. NaN propagates.
 *
 * @param t - Input tensor
 * @param lambd - Threshold, must be non-negative (default: 0.5)
 * @returns Tensor of the same shape (float dtypes are kept, integer and bool input gives `float32`)
 * @throws {InvalidParameterError} If `lambd` is negative or NaN
 * @throws {DTypeError} For `string` tensors
 *
 * @example
 * ```ts
 * import { hardshrink, tensor } from 'deepbox/ndarray';
 *
 * hardshrink(tensor([-2, -0.3, 0.3, 2]));  // [-2, 0, 0, 2]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-activations | Deepbox Activation Functions}
 */
export function hardshrink(t: Tensor, lambd = 0.5): Tensor {
  requireNotString(t, "hardshrink");
  requireShrinkLambda(lambd, "hardshrink");
  return mapFloat(t, "hardshrink", (x) => (x >= -lambd && x <= lambd ? 0 : x));
}

/**
 * Soft shrinkage: x - lambd for x > lambd, x + lambd for x < -lambd, otherwise 0.
 *
 * NaN propagates.
 *
 * @param t - Input tensor
 * @param lambd - Threshold, must be non-negative (default: 0.5)
 * @returns Tensor of the same shape (float dtypes are kept, integer and bool input gives `float32`)
 * @throws {InvalidParameterError} If `lambd` is negative or NaN
 * @throws {DTypeError} For `string` tensors
 *
 * @example
 * ```ts
 * import { softshrink, tensor } from 'deepbox/ndarray';
 *
 * softshrink(tensor([-2, -0.3, 0.3, 2]));  // [-1.5, 0, 0, 1.5]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-activations | Deepbox Activation Functions}
 */
export function softshrink(t: Tensor, lambd = 0.5): Tensor {
  requireNotString(t, "softshrink");
  requireShrinkLambda(lambd, "softshrink");
  return mapFloat(t, "softshrink", (x) =>
    x > lambd ? x - lambd : x < -lambd ? x + lambd : Number.isNaN(x) ? x : 0
  );
}

/**
 * Applies the Tanhshrink activation element-wise.
 *
 * Tanhshrink(x) = x - tanh(x). Small inputs use a Taylor series because the
 * direct subtraction cancels almost every digit (the true value is about
 * x³ / 3).
 *
 * @param t - Input tensor
 * @returns Tensor of the same shape (float dtypes are kept, integer and bool input gives `float32`)
 * @throws {DTypeError} For `string` tensors
 *
 * @example
 * ```ts
 * import { tanhshrink, tensor } from 'deepbox/ndarray';
 *
 * tanhshrink(tensor([-1, 0, 1]));  // [-0.238..., 0, 0.238...]
 * ```
 */
export function tanhshrink(t: Tensor): Tensor {
  return mapFloat(t, "tanhshrink", tanhshrinkScalar);
}
