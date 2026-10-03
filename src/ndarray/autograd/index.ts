/**
 * Autograd module for automatic differentiation.
 *
 * Implements reverse-mode automatic differentiation (backpropagation)
 * for `Tensor` operations.
 *
 * ## Gradient state
 *
 * A **module-level singleton** `gradEnabled` controls whether new
 * operations record their backward graph.  Use {@link noGrad} to
 * temporarily disable gradient tracking (e.g. during inference).
 * `noGrad` only accepts **synchronous** callbacks: passing an async
 * function will throw, because the flag would be restored before the
 * async work completes.
 *
 * ## max / min backward: tie-breaking
 *
 * When multiple elements share the maximum (or minimum) value along the
 * reduced axis, the gradient is **divided equally** among all tied
 * positions.  This preserves the total gradient magnitude and is a
 * valid subgradient of the max/min operation.
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox documentation}
 */

import type { Axis, Device, DType, Shape, TypedArray } from "../../core";
import {
  DeepboxError,
  DeviceError,
  DTypeError,
  getBigIntElement,
  getNumericElement,
  InvalidParameterError,
  normalizeAxes,
  normalizeAxis,
  ShapeError,
  shapeToSize,
} from "../../core";
import { dot } from "../linalg";
import { readNumbers } from "../ops/_internal";
import {
  elu,
  type GeluApproximation,
  type GeluOptions,
  gelu,
  geluDerivative,
  hardtanh as hardtanhOp,
  leakyRelu,
  mish as mishOp,
  relu,
  sigmoid,
  softplus as softplusOp,
  swish as swishOp,
  tanhshrink as tanhshrinkOp,
} from "../ops/activation";
import {
  abs as absOp,
  add,
  addScalar,
  clip as clipOp,
  div,
  maximum as maximumOp,
  minimum as minimumOp,
  mul,
  mulScalar,
  neg,
  pow,
  sub,
} from "../ops/arithmetic";
import { equal, greater, greaterEqual, isnan, less, lessEqual, notEqual } from "../ops/comparison";
import { col2im, im2col as im2colOp } from "../ops/conv";
import { dispatchUnary } from "../ops/device_dispatch";
import { logicalAnd, logicalOr } from "../ops/logical";
import { concatenate as concatOp, stack as stackOp } from "../ops/manipulation";
import {
  ceil as ceilOp,
  exp,
  expm1 as expm1Op,
  floor as floorOp,
  log,
  log1p as log1pOp,
  round as roundOp,
} from "../ops/math";
import { dropoutMask } from "../ops/random";
import {
  all as allOp,
  any as anyOp,
  argmax as argmaxOp,
  argmin as argminOp,
  cumsum as cumsumOp,
  max,
  mean as meanOp,
  min,
  prod as prodOp,
  std as stdOp,
  sum,
  variance as varianceOp,
} from "../ops/reduction";
import { argsort as argsortOp, sort as sortOp } from "../ops/sorting";
import { cos as cosOp, sin as sinOp, tanh, tan as tanOp } from "../ops/trigonometry";
import { clone as cloneOp, flip as flipOp, where as whereOp } from "../ops/utils";
import { tensor, zeros } from "../tensor/creation";
import { gather, type SliceRange, slice } from "../tensor/indexing";
import { reshape, transpose } from "../tensor/shape";
import { squeeze as squeezeOp, unsqueeze as unsqueezeOp } from "../tensor/shape_ops";
import { isContiguous, offsetFromFlatIndex } from "../tensor/strides";
import {
  computeStrides,
  dtypeToTypedArrayCtor,
  scalarOperand,
  type Tensor,
  Tensor as TensorClass,
} from "../tensor/Tensor";

/** Options for creating a {@link GradTensor}. */
export type GradTensorOptions = {
  readonly requiresGrad?: boolean;
  readonly dtype?: Exclude<DType, "string">;
};

type BackwardFn = () => void;

/**
 * Module-level singleton that controls gradient tracking.
 *
 * When `false`, newly created `GradTensor` operations will not record
 * backward functions regardless of the `requiresGrad` flag on their
 * inputs.  Toggled by {@link noGrad}. Creating a leaf tensor is not an
 * operation: its `requiresGrad` flag is never changed by this switch.
 *
 * **Thread-safety note**: because JavaScript is single-threaded this
 * global flag is safe in synchronous code, but it must **never** be
 * relied upon across async boundaries, hence `noGrad` rejects async
 * callbacks.
 */
let gradEnabled = true;

type NumericDType = Exclude<DType, "string">;

function ensureNumericDType(dtype: DType, context: string): NumericDType {
  if (dtype === "string") {
    throw new DTypeError(`${context} does not support string dtype`);
  }
  return dtype;
}

function ensureNumericTensor(t: Tensor, context: string): asserts t is Tensor<Shape, NumericDType> {
  if (t.dtype === "string") {
    throw new DTypeError(`${context} does not support string dtype`);
  }
}

function onesLike(t: Tensor): Tensor {
  ensureNumericTensor(t, "autograd");
  const dtype = t.dtype;
  const Ctor = dtypeToTypedArrayCtor(dtype);
  const out = new Ctor(t.size);
  if (out instanceof BigInt64Array) {
    out.fill(1n);
  } else {
    out.fill(1);
  }
  return TensorClass.fromTypedArray({
    data: out,
    shape: t.shape,
    dtype,
    device: t.device,
  });
}

function ensureSameSize(a: Tensor, b: Tensor, context: string): void {
  if (a.size !== b.size) {
    throw ShapeError.mismatch(a.shape, b.shape, context);
  }
}

function shapesEqual(a: Shape, b: Shape): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

function castTensor(t: Tensor, dtype: Exclude<DType, "string">): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("autograd does not support string dtype");
  }
  return t.astype(dtype);
}

function isFloatDType(dtype: DType): boolean {
  return dtype === "float16" || dtype === "bfloat16" || dtype === "float32" || dtype === "float64";
}

/**
 * Cast an op result back to the input's floating dtype. Several elementwise
 * kernels always compute in float64; without this a float32 graph would pick
 * up float64 nodes and fail on the next dtype-strict accumulate. Non-float
 * inputs keep the promoted result, because truncating a real-valued result
 * to an integer dtype would silently destroy it.
 */
function keepFloatDtype(result: Tensor, srcDtype: DType, context: string): Tensor {
  if (result.dtype === srcDtype || !isFloatDType(srcDtype)) return result;
  return castTensor(result, ensureNumericDType(srcDtype, context));
}

/**
 * Copy a tensor so the copy owns its buffer. Used for leaf gradients, which
 * users and optimizers (for example gradient clipping) modify in place and
 * which must therefore never share memory with another node's gradient.
 */
function ownTensor(t: Tensor): Tensor {
  if (t.isDeviceTensor) {
    const copied = dispatchUnary("copy", t);
    return copied ?? t;
  }
  return cloneOp(t);
}

/** Axis argument for the reduction kernels: a single axis stays a number, lists are copied. */
function kernelAxis(axes: readonly number[] | undefined): number | number[] | undefined {
  if (axes === undefined) return undefined;
  return axes.length === 1 ? (axes[0] as number) : [...axes];
}

/**
 * Normalize an axis argument to a sorted-as-given list of unique non-negative
 * axes, or `undefined` for "reduce everything".
 */
function resolveAxisList(
  axis: Axis | readonly Axis[] | undefined,
  ndim: number
): number[] | undefined {
  return axis === undefined ? undefined : normalizeAxes(axis, ndim);
}

/** Number of elements combined into each output element of a reduction. */
function reducedCount(shape: Shape, axes: readonly number[] | undefined): number {
  if (axes === undefined) return shapeToSize(shape);
  let n = 1;
  for (const a of axes) n *= shape[a] ?? 1;
  return n;
}

/** Shape of `shape` with every reduced axis set to 1 (all axes when `axes` is undefined). */
function keepdimsShape(shape: Shape, axes: readonly number[] | undefined): number[] {
  if (axes === undefined) return new Array<number>(shape.length).fill(1);
  const out = [...shape];
  for (const a of axes) out[a] = 1;
  return out;
}

/** Convert `t` to `dtype` when it is not already of that dtype. */
function asDtype(t: Tensor, dtype: DType): Tensor {
  return t.dtype === dtype ? t : castTensor(t, ensureNumericDType(dtype, "autograd"));
}

/** Float64 view of `t` for derivative formulas (a no-op for float64 tensors). */
function asF64(t: Tensor): Tensor {
  return asDtype(t, "float64");
}

/** A 0-d float64 constant, broadcast against float64 tensors by the binary ops. */
function f64Const(value: number): Tensor {
  return tensor(value, { dtype: "float64" });
}

/**
 * Right-hand side of a fluent method as a tensor: a `GradTensor` gives its
 * tensor, a number follows the scalar rules of `Tensor` (it never upcasts).
 */
function rawOperand(ref: Tensor, other: number | Tensor | GradTensor, op: string): Tensor {
  return GradTensor.isGradTensor(other) ? other.tensor : scalarOperand(ref, other, op);
}

/** Right-hand side of a fluent method as a `GradTensor`; a number or tensor becomes a constant leaf. */
function gradOperand(ref: GradTensor, other: number | Tensor | GradTensor, op: string): GradTensor {
  if (GradTensor.isGradTensor(other)) return other;
  return GradTensor.fromTensor(scalarOperand(ref.tensor, other, op));
}

/** A `Tensor` or `GradTensor` argument as a `GradTensor` (no numbers). */
function gradFromTensorLike(other: Tensor | GradTensor, op: string): GradTensor {
  if (GradTensor.isGradTensor(other)) return other;
  if (!(other instanceof TensorClass)) {
    throw new InvalidParameterError(
      `${op}: operand must be a Tensor or GradTensor; received ${typeof other}`,
      "other",
      other
    );
  }
  return GradTensor.fromTensor(other);
}

/** Copy a readonly axis list into the mutable form the reduction ops take. */
function axisArg(axis: Axis | readonly Axis[] | undefined): Axis | Axis[] | undefined {
  return axis === undefined || !Array.isArray(axis) ? (axis as Axis | undefined) : [...axis];
}

/** dtype of a gradient for a tensor of `dtype`: floats keep theirs, everything else uses float64. */
function gradDtypeFor(dtype: DType): NumericDType {
  return isFloatDType(dtype) ? ensureNumericDType(dtype, "autograd") : "float64";
}

/**
 * Attach a one-input backward rule to an already computed output. `gradFn`
 * maps the upstream gradient of the output to the gradient of `input`
 * (same shape as `input`). Nothing is recorded when gradients are disabled or
 * `input` does not require them.
 */
function unaryNode(
  input: GradTensor,
  outTensor: Tensor,
  name: string,
  gradFn: (go: Tensor) => Tensor
): GradTensor {
  const requiresGrad = gradEnabled && input.requiresGrad;
  const out: GradTensor = new GradTensor({
    tensor: outTensor,
    requiresGrad,
    prev: requiresGrad ? [input] : [],
    backward: () => {
      if (!requiresGrad) return;
      const go = out.grad;
      if (go === null) {
        throw new DeepboxError(`Internal error: missing gradient for ${name} backward`);
      }
      input.accumulateGrad(gradFn(go));
    },
  });
  return out;
}

/** Multiply an upstream gradient by a derivative tensor, in the gradient's dtype. */
function chain(go: Tensor, deriv: Tensor): Tensor {
  return mul(go, asDtype(deriv, go.dtype));
}

/**
 * Lane layout of a reduction over `axes` (all axes when undefined): for lane
 * `l` (the `l`-th output element in row-major order) and reduced position `r`,
 * the row-major flat input index is `bases[l] + offsets[r]`.
 */
function reductionLanes(
  shape: Shape,
  axes: readonly number[] | undefined
): { bases: Float64Array; offsets: Float64Array } {
  const ndim = shape.length;
  const reduced = new Array<boolean>(ndim).fill(axes === undefined);
  if (axes !== undefined) for (const a of axes) reduced[a] = true;
  const strides = computeStrides(shape);
  const kept: number[] = [];
  const red: number[] = [];
  for (let d = 0; d < ndim; d++) (reduced[d] ? red : kept).push(d);

  const enumerate = (dims: readonly number[]): Float64Array => {
    let count = 1;
    for (const d of dims) count *= shape[d] ?? 1;
    const result = new Float64Array(count);
    const coord = new Int32Array(dims.length);
    let off = 0;
    for (let i = 0; i < count; i++) {
      result[i] = off;
      for (let k = dims.length - 1; k >= 0; k--) {
        const d = dims[k] as number;
        const stride = strides[d] ?? 0;
        coord[k] = (coord[k] ?? 0) + 1;
        off += stride;
        if ((coord[k] as number) < (shape[d] ?? 1)) break;
        off -= (coord[k] as number) * stride;
        coord[k] = 0;
      }
    }
    return result;
  };
  return { bases: enumerate(kept), offsets: enumerate(red) };
}

const SELU_ALPHA = 1.6732632423543772;
const SELU_SCALE = 1.0507009873554805;

/**
 * Shared backward of max/min. The upstream gradient is split equally among all
 * positions that attain the extremum; a NaN extremum is attributed to the NaN
 * element(s), matching how the forward reduction propagates NaN.
 */
function extremumGrad(
  input: Tensor,
  result: Tensor,
  go: Tensor,
  axes: readonly number[] | undefined,
  keepdims: boolean,
  context: string
): Tensor {
  let resultK = result;
  let goK = go;
  if (!keepdims || axes === undefined) {
    const targetShape = keepdimsShape(input.shape, axes);
    resultK = result.reshape(targetShape);
    goK = toContiguous(go).reshape(targetShape);
  }

  const dtype = ensureNumericDType(input.dtype, context);
  let maskBool = equal(input, resultK);
  if (isFloatDType(dtype)) {
    maskBool = logicalOr(maskBool, logicalAnd(isnan(input), isnan(resultK)));
  }
  const mask = castTensor(maskBool, dtype);
  const tieCount = castTensor(sum(mask, kernelAxis(axes), true), dtype);
  return mul(div(mask, tieCount), goK);
}

function reduceBroadcastGrad(grad: Tensor, targetShape: Shape): Tensor {
  if (grad.dtype === "string") {
    throw new DTypeError("autograd does not support string dtype");
  }
  if (shapesEqual(grad.shape, targetShape)) {
    return grad;
  }
  if (shapeToSize(grad.shape) === 0 || shapeToSize(targetShape) === 0) {
    return zeros(targetShape, { dtype: grad.dtype, device: grad.device });
  }

  const gradShape = grad.shape;
  const gradNdim = gradShape.length;
  const targetNdim = targetShape.length;
  if (gradNdim < targetNdim) {
    throw ShapeError.mismatch(targetShape, gradShape, "broadcast");
  }

  const expandedTarget = new Array<number>(gradNdim);
  const leading = gradNdim - targetNdim;
  for (let i = 0; i < gradNdim; i++) {
    const targetDim = i < leading ? 1 : (targetShape[i - leading] ?? 1);
    const gradDim = gradShape[i] ?? 1;
    if (targetDim !== gradDim && targetDim !== 1) {
      throw ShapeError.mismatch(targetShape, gradShape, "broadcast");
    }
    expandedTarget[i] = targetDim;
  }

  let result = grad;
  for (let axis = 0; axis < gradNdim; axis++) {
    const targetDim = expandedTarget[axis] ?? 1;
    const gradDim = gradShape[axis] ?? 1;
    if (targetDim === 1 && gradDim !== 1) {
      result = sum(result, axis, true);
    }
  }

  if (!shapesEqual(result.shape, targetShape)) {
    result = reshape(result, targetShape);
  }

  const gradDtype = grad.dtype;
  if (result.dtype !== gradDtype) {
    result = castTensor(result, gradDtype);
  }

  return result;
}

function asFloat64Dense(t: Tensor): Float64Array {
  if (t.dtype === "string") {
    throw new DTypeError("autograd does not support string dtype");
  }
  const out = new Float64Array(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);
  const data = t.data;

  if (Array.isArray(data)) {
    throw new DTypeError("autograd does not support string dtype");
  }

  for (let flat = 0; flat < t.size; flat++) {
    const offset = contiguous
      ? t.offset + flat
      : offsetFromFlatIndex(flat, logicalStrides, t.strides, t.offset);
    if (data instanceof BigInt64Array) {
      out[flat] = Number(getBigIntElement(data, offset));
    } else {
      out[flat] = getNumericElement(data, offset);
    }
  }
  return out;
}

function toContiguous(t: Tensor): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("autograd does not support string dtype");
  }
  if (isContiguous(t.shape, t.strides)) {
    return t;
  }
  if (t.isDeviceTensor) {
    // Materialize a strided device view with the `copy` kernel (writes a
    // contiguous buffer in logical row-major order), with no host readback.
    const copied = dispatchUnary("copy", t);
    if (copied) return copied;
  }
  const Ctor = dtypeToTypedArrayCtor(t.dtype);
  const out = new Ctor(t.size);
  const logicalStrides = computeStrides(t.shape);

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("autograd does not support string dtype");
  }

  if (out instanceof BigInt64Array) {
    if (data instanceof BigInt64Array) {
      for (let i = 0; i < t.size; i++) {
        const offset = offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
        out[i] = getBigIntElement(data, offset);
      }
    } else {
      // Should not happen if dtype matches
      throw new DTypeError("Internal error: dtype mismatch in toContiguous");
    }
  } else {
    if (data instanceof BigInt64Array) {
      throw new DTypeError("Internal error: dtype mismatch in toContiguous");
    }
    for (let i = 0; i < t.size; i++) {
      const offset = offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
      out[i] = getNumericElement(data, offset);
    }
  }
  return TensorClass.fromTypedArray({
    data: out,
    shape: t.shape,
    dtype: t.dtype,
    device: t.device,
  });
}

function fromFloat64Dense(shape: Shape, device: Tensor["device"], data: Float64Array): Tensor {
  if (shapeToSize(shape) !== data.length) {
    throw new ShapeError("Internal error: dense buffer does not match shape");
  }
  return TensorClass.fromTypedArray({
    data,
    shape,
    dtype: "float64",
    device,
  });
}

/**
 * Tensor wrapper that records a computation graph for reverse-mode autodiff.
 *
 * Gradients accumulate into leaf tensors across `backward()` calls until
 * `zeroGrad()` is called; gradients of intermediate nodes are recomputed on
 * every pass.
 *
 * The method surface matches `Tensor`, so code reads the same with or without
 * gradient tracking. Results that cannot carry a gradient (`argmax`, comparisons,
 * `argsort`, `any`, `all`) are plain `Tensor`s, as in PyTorch.
 *
 * @example
 * ```ts
 * const w = parameter([1, 2]);
 * const loss = w.mul(w).sum();
 * loss.backward();
 * console.log(loss.item()); // 5
 * console.log(w.grad?.toArray()); // [2, 4]
 * ```
 */
export class GradTensor {
  readonly tensor: Tensor;
  requiresGrad: boolean;

  private _grad: Tensor | null;
  private readonly _prev: readonly GradTensor[];
  private readonly _backward: BackwardFn;

  /** Check if a value is a GradTensor (works across module boundaries). */
  static isGradTensor(value: unknown): value is GradTensor {
    return (
      typeof value === "object" &&
      value !== null &&
      "tensor" in value &&
      "requiresGrad" in value &&
      "backward" in value &&
      typeof (value as { backward: unknown }).backward === "function"
    );
  }

  constructor(data: number | number[] | number[][] | number[][][], options?: GradTensorOptions);
  constructor(args: {
    readonly tensor: Tensor;
    readonly requiresGrad: boolean;
    readonly prev: readonly GradTensor[];
    readonly backward: BackwardFn;
  });
  constructor(
    dataOrArgs:
      | number
      | number[]
      | number[][]
      | number[][][]
      | {
          readonly tensor: Tensor;
          readonly requiresGrad: boolean;
          readonly prev: readonly GradTensor[];
          readonly backward: BackwardFn;
        },
    options?: GradTensorOptions
  ) {
    if (typeof dataOrArgs === "object" && dataOrArgs !== null && "tensor" in dataOrArgs) {
      this.tensor = dataOrArgs.tensor;
      this.requiresGrad = dataOrArgs.requiresGrad;
      this._prev = dataOrArgs.prev;
      this._backward = dataOrArgs.backward;
    } else {
      const data = dataOrArgs as number | number[] | number[][] | number[][][];
      const t =
        options?.dtype === undefined ? tensor(data) : tensor(data, { dtype: options.dtype });
      this.tensor = t;
      // PyTorch semantics: the flag is what the caller asked for, also inside
      // noGrad(). Only operations executed inside noGrad() skip recording.
      this.requiresGrad = options?.requiresGrad ?? false;
      this._prev = [];
      this._backward = () => {};
    }
    this._grad = null;
  }

  /** Build a GradTensor from an existing tensor, graph parents and backward rule. */
  static create(args: {
    readonly tensor: Tensor;
    readonly requiresGrad: boolean;
    readonly prev: readonly GradTensor[];
    readonly backward: BackwardFn;
  }): GradTensor {
    return new GradTensor(args);
  }

  /**
   * Wrap a tensor as a leaf GradTensor without copying its data.
   *
   * @param t - Numeric tensor to wrap.
   * @param options - `requiresGrad` (default false; kept as given, also inside
   *   {@link noGrad}, like PyTorch) and an optional `dtype` that must equal the tensor's dtype.
   * @throws {DTypeError} For string tensors or a `dtype` that does not match.
   */
  static fromTensor(t: Tensor, options: GradTensorOptions = {}): GradTensor {
    if (t.dtype === "string") {
      throw new DTypeError("autograd does not support string dtype");
    }
    if (options.dtype !== undefined && options.dtype !== t.dtype) {
      throw new DTypeError(
        `GradTensor dtype mismatch: expected ${options.dtype}, received ${t.dtype}`
      );
    }
    const requiresGrad = options.requiresGrad ?? false;
    return new GradTensor({
      tensor: t,
      requiresGrad,
      prev: [],
      backward: () => {},
    });
  }

  /** Create a 0-d GradTensor holding `value` (default dtype float32). */
  static scalar(value: number, options: GradTensorOptions = {}): GradTensor {
    const dtype = options.dtype ?? "float32";
    const Ctor = dtypeToTypedArrayCtor(dtype);
    const out = new Ctor(1);
    if (out instanceof BigInt64Array) {
      out[0] = BigInt(Math.round(value));
    } else {
      out[0] = value;
    }
    const t = TensorClass.fromTypedArray({
      data: out,
      shape: [],
      dtype,
      device: "cpu",
    });
    return GradTensor.fromTensor(t, options);
  }

  /**
   * Get the shape of the underlying tensor.
   * Implements TensorLike interface for compatibility with Tensor.
   */
  get shape(): Shape {
    return this.tensor.shape;
  }

  /**
   * Get the total number of elements.
   * Implements TensorLike interface for compatibility with Tensor.
   */
  get size(): number {
    return this.tensor.size;
  }

  /**
   * Get the number of dimensions.
   * Implements TensorLike interface for compatibility with Tensor.
   */
  get ndim(): number {
    return this.tensor.ndim;
  }

  /**
   * Get the data type of the underlying tensor.
   * Implements TensorLike interface for compatibility with Tensor.
   */
  get dtype(): DType {
    return this.tensor.dtype;
  }

  /**
   * Convert to a nested JS array (delegates to the underlying tensor).
   * Implements TensorLike interface for compatibility with Tensor.
   */
  toArray(): ReturnType<Tensor["toArray"]> {
    return this.tensor.toArray();
  }

  /** Read a single element by multi-index (delegates to the underlying tensor). */
  at(...indices: number[]): ReturnType<Tensor["at"]> {
    return this.tensor.at(...indices);
  }

  /**
   * Cast to another numeric dtype, differentiably. The backward casts the
   * gradient back to this tensor's dtype.
   */
  astype(dtype: DType): GradTensor {
    if (dtype === "string") {
      throw new DTypeError("autograd does not support string dtype");
    }
    if (this.tensor.dtype === dtype) return this;
    const outTensor = castTensor(this.tensor, dtype);
    const requiresGrad = gradEnabled && this.requiresGrad;
    const srcDtype = ensureNumericDType(this.tensor.dtype, "astype");
    let out: GradTensor;
    out = new GradTensor({
      tensor: outTensor,
      requiresGrad,
      prev: requiresGrad ? [this] : [],
      backward: () => {
        if (!requiresGrad) return;
        const go = out._grad;
        if (go === null) throw new DeepboxError("Internal error: missing gradient for astype");
        this.accumulateGrad(castTensor(go, srcDtype));
      },
    });
    return out;
  }

  /**
   * Get the device where the tensor resides.
   * Implements TensorLike interface for compatibility with Tensor.
   */
  get device(): Tensor["device"] {
    return this.tensor.device;
  }

  /** `true` when the tensor's storage lives in device memory (see `Tensor.isDeviceTensor`). */
  get isDeviceTensor(): boolean {
    return this.tensor.isDeviceTensor;
  }

  /**
   * Get the memory strides of the underlying tensor.
   * Implements TensorLike interface for compatibility with Tensor.
   */
  get strides(): readonly number[] {
    return this.tensor.strides;
  }

  /**
   * Get the offset into the underlying data buffer.
   * Implements TensorLike interface for compatibility with Tensor.
   */
  get offset(): number {
    return this.tensor.offset;
  }

  /**
   * Get the underlying data buffer.
   * Implements TensorLike interface for compatibility with Tensor.
   */
  get data(): TypedArray {
    if (this.tensor.dtype === "string") {
      throw new DTypeError("GradTensor does not support string tensors");
    }
    const data = this.tensor.data;
    if (Array.isArray(data)) {
      throw new DTypeError("GradTensor does not support string tensors");
    }
    return data;
  }

  /**
   * Get the accumulated gradient for this tensor.
   * Returns null if no gradient has been computed yet.
   */
  get grad(): Tensor | null {
    return this._grad;
  }

  /**
   * Replace the stored gradient with `grad`.
   *
   * @param grad - Gradient tensor; its shape must equal this tensor's shape.
   * @throws {InvalidParameterError} If `requiresGrad` is false.
   * @throws {ShapeError} If the shape of `grad` differs from the tensor's shape.
   */
  setGrad(grad: Tensor): void {
    if (!this.requiresGrad) {
      throw new InvalidParameterError(
        "Cannot set gradient on tensor with requiresGrad=false",
        "requiresGrad",
        this.requiresGrad
      );
    }
    if (!shapesEqual(grad.shape, this.tensor.shape)) {
      throw new ShapeError(
        `setGrad shape mismatch: gradient has shape [${grad.shape}] but tensor has shape [${this.tensor.shape}]`
      );
    }
    this._grad = grad;
  }

  /**
   * Reset the gradient to zeros (shape and dtype of the tensor). Does nothing
   * when `requiresGrad` is false.
   */
  zeroGrad(): void {
    if (!this.requiresGrad) return;
    this._grad = zeros(this.tensor.shape, {
      dtype: this.tensor.dtype,
      device: this.tensor.device,
    });
  }

  /**
   * Return a GradTensor that shares this tensor's data but is cut off from the
   * graph (`requiresGrad` is false).
   */
  detach(): GradTensor {
    return GradTensor.fromTensor(this.tensor, { requiresGrad: false });
  }

  /**
   * Turn gradient tracking on or off for this tensor. Turning it off also
   * clears the stored gradient. The flag is set as given, also inside
   * {@link noGrad} (like `requires_grad_()` in PyTorch); operations run inside
   * `noGrad` still do not record a graph.
   */
  setRequiresGrad(value: boolean): void {
    this.requiresGrad = value;
    if (!this.requiresGrad) {
      this._grad = null;
    }
  }

  /** Whether a gradient has been stored on this tensor. */
  hasGrad(): boolean {
    return this._grad !== null;
  }

  /**
   * Add `grad` into this tensor's gradient.
   *
   * A floating-point gradient of another float dtype is cast to the dtype of this tensor.
   * The first contribution to a leaf is copied, so a leaf never shares a
   * buffer with another node's gradient (in-place edits such as gradient
   * clipping would otherwise be applied twice to shared storage).
   *
   * @internal
   */
  accumulateGrad(grad: Tensor): void {
    if (!this.requiresGrad) return;

    if (!shapesEqual(grad.shape, this.tensor.shape)) {
      throw new ShapeError(
        `accumulateGrad shape mismatch: gradient has shape [${grad.shape}] but tensor has shape [${this.tensor.shape}]`
      );
    }

    // Hand-written kernels (custom ops, layers) may compute in float64. A floating-point
    // gradient is stored in the dtype of the tensor it belongs to, so a float32 graph never
    // collects float64 gradients that the next accumulation would reject.
    const sameFloatKind =
      isFloatDType(grad.dtype) &&
      isFloatDType(this.tensor.dtype) &&
      grad.dtype !== this.tensor.dtype;
    const normalizedGrad = toContiguous(
      sameFloatKind
        ? castTensor(grad, ensureNumericDType(this.tensor.dtype, "accumulateGrad"))
        : grad
    );

    if (this._grad === null) {
      this._grad = this._prev.length === 0 ? ownTensor(normalizedGrad) : normalizedGrad;
      return;
    }

    ensureSameSize(this._grad, normalizedGrad, "accumulateGrad");
    if (this._grad.dtype !== normalizedGrad.dtype) {
      throw new DTypeError(
        `accumulateGrad dtype mismatch: ${this._grad.dtype} vs ${normalizedGrad.dtype}`
      );
    }
    this._grad = add(this._grad, normalizedGrad);
  }

  /**
   * Backpropagate gradients from this node through the recorded graph.
   */
  backward(grad?: Tensor): void {
    if (!this.requiresGrad) {
      return;
    }

    let seedGrad: Tensor;
    if (grad === undefined) {
      seedGrad = onesLike(this.tensor);
    } else {
      ensureNumericDType(grad.dtype, "backward");
      // A seed with the same number of elements but another shape (for example
      // [1] for a scalar loss) is reshaped; any other size is an error.
      ensureSameSize(this.tensor, grad, "backward");
      seedGrad = shapesEqual(grad.shape, this.tensor.shape)
        ? grad
        : toContiguous(grad).reshape(this.tensor.shape);
    }
    if (this._prev.length === 0) {
      // Calling backward() on a leaf accumulates into .grad like any other
      // contribution (PyTorch semantics) instead of overwriting it.
      this.accumulateGrad(seedGrad);
    } else {
      this._grad = seedGrad;
    }

    const topo: GradTensor[] = [];
    const visited = new Set<GradTensor>();

    // Iterative post-order DFS (explicit stack, not recursion): the graph
    // depth for an unrolled model (a long RNN over thousands of timesteps or
    // a deep residual stack) equals the recursion depth, which overflows
    // V8's ~10k-frame call stack. An explicit worklist scales to arbitrary
    // depth. Ordering is identical to the previous recursive post-order.
    const stack: Array<{ node: GradTensor; idx: number }> = [{ node: this, idx: 0 }];
    visited.add(this);
    while (stack.length > 0) {
      const frame = stack[stack.length - 1] as { node: GradTensor; idx: number };
      const prev = frame.node._prev;
      if (frame.idx < prev.length) {
        const child = prev[frame.idx] as GradTensor;
        frame.idx++;
        if (!visited.has(child)) {
          visited.add(child);
          stack.push({ node: child, idx: 0 });
        }
      } else {
        topo.push(frame.node);
        stack.pop();
      }
    }
    topo.reverse();

    // Intermediate (non-leaf) gradients are not retained across backward
    // calls (PyTorch semantics). Without this reset, a second backward()
    // would accumulate into stale grads left over from the previous pass
    // and propagate wrong values into the leaves.
    for (const v of topo) {
      if (v !== this && v._prev.length > 0) {
        v._grad = null;
      }
    }

    for (const v of topo) {
      v._backward();
    }
  }

  /**
   * Elementwise sum with broadcasting. `other` may be a `GradTensor`, a `Tensor`
   * or a number; the last two are constants and receive no gradient.
   */
  add(otherArg: number | Tensor | GradTensor): GradTensor {
    const other = gradOperand(this, otherArg, "add");
    const outTensor = add(this.tensor, other.tensor);
    const requiresGrad = gradEnabled && (this.requiresGrad || other.requiresGrad);

    const out = new GradTensor({
      tensor: outTensor,
      requiresGrad,
      prev: requiresGrad ? [this, other] : [],
      backward: () => {
        if (!requiresGrad) return;
        const go = out._grad;
        if (go === null) {
          throw new DeepboxError("Internal error: missing gradient for add backward");
        }
        if (this.requiresGrad) this.accumulateGrad(reduceBroadcastGrad(go, this.tensor.shape));
        if (other.requiresGrad) other.accumulateGrad(reduceBroadcastGrad(go, other.tensor.shape));
      },
    });

    return out;
  }

  /**
   * Elementwise difference with broadcasting. `other` may be a `GradTensor`, a `Tensor`
   * or a number; the last two are constants and receive no gradient.
   */
  sub(otherArg: number | Tensor | GradTensor): GradTensor {
    const other = gradOperand(this, otherArg, "sub");
    const outTensor = sub(this.tensor, other.tensor);
    const requiresGrad = gradEnabled && (this.requiresGrad || other.requiresGrad);

    const out = new GradTensor({
      tensor: outTensor,
      requiresGrad,
      prev: requiresGrad ? [this, other] : [],
      backward: () => {
        if (!requiresGrad) return;
        const go = out._grad;
        if (go === null) {
          throw new DeepboxError("Internal error: missing gradient for sub backward");
        }
        if (this.requiresGrad) this.accumulateGrad(reduceBroadcastGrad(go, this.tensor.shape));
        if (other.requiresGrad) {
          const grad = reduceBroadcastGrad(neg(go), other.tensor.shape);
          other.accumulateGrad(grad);
        }
      },
    });

    return out;
  }

  /**
   * Elementwise product with broadcasting. `other` may be a `GradTensor`, a `Tensor`
   * or a number; the last two are constants and receive no gradient.
   */
  mul(otherArg: number | Tensor | GradTensor): GradTensor {
    const other = gradOperand(this, otherArg, "mul");
    const outTensor = mul(this.tensor, other.tensor);
    const requiresGrad = gradEnabled && (this.requiresGrad || other.requiresGrad);

    const out = new GradTensor({
      tensor: outTensor,
      requiresGrad,
      prev: requiresGrad ? [this, other] : [],
      backward: () => {
        if (!requiresGrad) return;
        const go = out._grad;
        if (go === null) {
          throw new DeepboxError("Internal error: missing gradient for mul backward");
        }
        if (this.requiresGrad) {
          const grad = reduceBroadcastGrad(mul(go, other.tensor), this.tensor.shape);
          this.accumulateGrad(grad);
        }
        if (other.requiresGrad) {
          const grad = reduceBroadcastGrad(mul(go, this.tensor), other.tensor.shape);
          other.accumulateGrad(grad);
        }
      },
    });

    return out;
  }

  /** Elementwise negation. */
  neg(): GradTensor {
    const outTensor = neg(this.tensor);
    const requiresGrad = gradEnabled && this.requiresGrad;

    const out = new GradTensor({
      tensor: outTensor,
      requiresGrad,
      prev: requiresGrad ? [this] : [],
      backward: () => {
        if (!requiresGrad) return;
        const go = out._grad;
        if (go === null) {
          throw new DeepboxError("Internal error: missing gradient for neg backward");
        }
        this.accumulateGrad(neg(go));
      },
    });

    return out;
  }

  /**
   * Sum over `axis`, or over all elements when `axis` is omitted.
   *
   * @param axis - Axis or list of axes to reduce (negative values count from the end).
   * @param keepdims - Keep the reduced axes with size 1.
   * @throws {InvalidParameterError} If an axis is out of range or listed twice.
   *
   * @example
   * ```ts
   * const x = parameter([[1, 2], [3, 4]]);
   * x.sum([0, 1]).toArray(); // 10
   * x.sum(1, true).shape; // [2, 1]
   * ```
   */
  sum(axisArg?: Axis | readonly Axis[], keepdims = false): GradTensor {
    const axes = resolveAxisList(axisArg, this.tensor.ndim);
    // A one-element list takes the single-axis kernel path.
    const axis: Axis | undefined = axes !== undefined && axes.length === 1 ? axes[0] : undefined;
    const multiAxes = axes !== undefined && axes.length !== 1 ? axes : undefined;
    let outTensor = sum(
      this.tensor,
      multiAxes === undefined ? (axis ?? undefined) : [...multiAxes],
      keepdims
    );
    const targetDtype = ensureNumericDType(this.tensor.dtype, "sum");
    if (outTensor.dtype !== targetDtype) {
      outTensor = castTensor(outTensor, targetDtype);
    }
    const requiresGrad = gradEnabled && this.requiresGrad;

    const out = new GradTensor({
      tensor: outTensor,
      requiresGrad,
      prev: requiresGrad ? [this] : [],
      backward: () => {
        if (!requiresGrad) return;
        const go = out._grad;
        if (go === null) {
          throw new DeepboxError("Internal error: missing gradient for sum backward");
        }

        // d/dx sum(x) = 1, so the input gradient is the upstream gradient broadcast
        // to the input shape.
        if (multiAxes !== undefined) {
          // Several axes (or none): put size-1 axes back and broadcast with a multiply.
          const goKeep = keepdims
            ? go
            : toContiguous(go).reshape(keepdimsShape(this.tensor.shape, multiAxes));
          this.accumulateGrad(
            asDtype(
              mul(onesLike(this.tensor), goKeep),
              ensureNumericDType(this.tensor.dtype, "sum")
            )
          );
          return;
        }
        if (axis === undefined) {
          if (this.tensor.isDeviceTensor) {
            // Device path: broadcast the 0-D upstream gradient across the
            // input shape with a device kernel instead of reading host data.
            this.accumulateGrad(mul(onesLike(this.tensor), go));
            return;
          }
          const goData = go.data;
          let g0: number;
          if (Array.isArray(goData)) {
            throw new DTypeError("autograd does not support string dtype");
          } else if (goData instanceof BigInt64Array) {
            g0 = Number(getBigIntElement(goData, go.offset));
          } else {
            g0 = getNumericElement(goData, go.offset);
          }
          const ones = new Float64Array(this.tensor.size);
          ones.fill(g0);
          const onesTensor = fromFloat64Dense(this.tensor.shape, this.tensor.device, ones);
          const grad =
            onesTensor.dtype === this.tensor.dtype
              ? onesTensor
              : castTensor(onesTensor, ensureNumericDType(this.tensor.dtype, "sum"));
          this.accumulateGrad(grad);
          return;
        }

        const ax = normalizeAxis(axis, this.tensor.ndim);

        if (this.tensor.isDeviceTensor) {
          // sum-over-axis backward = broadcast the upstream gradient back along
          // the reduced axis. Reshape `go` to hold a size-1 reduced axis
          // (keepdims already has it), then multiply by a ones tensor of the
          // input shape so the device binary kernel broadcasts it, with no host
          // readback. This is the path softmax/layernorm/cross-entropy hit.
          const keepShape = this.tensor.shape.map((d, i) => (i === ax ? 1 : d));
          const goKeep = keepdims ? go : go.reshape(keepShape);
          this.accumulateGrad(mul(onesLike(this.tensor), goKeep));
          return;
        }

        // go shape is the input shape with the reduced axis removed
        // (keepdims=false) or set to 1 (keepdims=true).
        // Allocation-free scatter of the upstream gradient back to the input
        // shape. sum() backward broadcasts `go` along the reduced axis: each
        // input element maps to the go element at the same coordinates with the
        // reduced axis removed (keepdims=false) or held at 0 (keepdims=true).
        // The previous implementation allocated two JS arrays (inCoord/outCoord)
        // per input element, tens of millions of short-lived allocations per
        // backward on softmax/cross-entropy/layernorm/attention logits, GC-bound.
        // Here we walk the input in row-major order with a single reused
        // coordinate odometer and track the physical go offset incrementally:
        // zero per-element allocation. `go.strides` are physical offsets into
        // go.data, so this handles contiguous and non-contiguous `go` uniformly.
        const ndim = this.tensor.ndim;
        const shape = this.tensor.shape;
        const goStridePerInputDim = new Int32Array(ndim);
        for (let d = 0; d < ndim; d++) {
          if (d === ax) {
            // Reduced axis contributes nothing: removed (keepdims=false) or
            // pinned to index 0 in a size-1 go dimension (keepdims=true).
            goStridePerInputDim[d] = 0;
          } else {
            const goDim = keepdims ? d : d < ax ? d : d - 1;
            goStridePerInputDim[d] = go.strides[goDim] ?? 0;
          }
        }

        const goDataBuf = go.data;
        if (Array.isArray(goDataBuf)) {
          throw new DTypeError("autograd does not support string dtype");
        }

        const inDense = new Float64Array(this.tensor.size);
        const coord = new Int32Array(ndim);
        let goOffset = go.offset;
        for (let inFlat = 0; inFlat < this.tensor.size; inFlat++) {
          inDense[inFlat] =
            goDataBuf instanceof BigInt64Array
              ? Number(getBigIntElement(goDataBuf, goOffset))
              : getNumericElement(goDataBuf, goOffset);
          // Advance the row-major odometer (last axis fastest), updating the go
          // offset incrementally instead of recomputing coordinates each step.
          for (let d = ndim - 1; d >= 0; d--) {
            const dim = shape[d] ?? 1;
            const stride = goStridePerInputDim[d] ?? 0;
            const nc = (coord[d] ?? 0) + 1;
            if (nc < dim) {
              coord[d] = nc;
              goOffset += stride;
              break;
            }
            coord[d] = 0;
            goOffset -= stride * (dim - 1);
          }
        }

        const inDenseTensor = fromFloat64Dense(this.tensor.shape, this.tensor.device, inDense);
        const grad =
          inDenseTensor.dtype === this.tensor.dtype
            ? inDenseTensor
            : castTensor(inDenseTensor, ensureNumericDType(this.tensor.dtype, "sum"));
        this.accumulateGrad(grad);
      },
    });

    return out;
  }

  /**
   * Elementwise quotient with broadcasting. `other` may be a `GradTensor`, a `Tensor`
   * or a number; the last two are constants and receive no gradient.
   */
  div(otherArg: number | Tensor | GradTensor): GradTensor {
    const other = gradOperand(this, otherArg, "div");
    const outTensor = div(this.tensor, other.tensor);
    const requiresGrad = gradEnabled && (this.requiresGrad || other.requiresGrad);

    const out = new GradTensor({
      tensor: outTensor,
      requiresGrad,
      prev: requiresGrad ? [this, other] : [],
      backward: () => {
        if (!requiresGrad) return;
        const go = out._grad;
        if (go === null) {
          throw new DeepboxError("Internal error: missing gradient for div backward");
        }
        const gradDtype = ensureNumericDType(go.dtype, "div");
        const denom =
          other.tensor.dtype === gradDtype ? other.tensor : castTensor(other.tensor, gradDtype);
        if (this.requiresGrad) {
          const grad = reduceBroadcastGrad(div(go, denom), this.tensor.shape);
          this.accumulateGrad(grad);
        }
        if (other.requiresGrad) {
          // d(a/b)/db = -((a/b)/b). Dividing the stored quotient by b instead of
          // forming a / b^2 avoids overflowing b * b (and underflowing it to
          // zero) for large or tiny denominators. The upstream gradient is
          // applied last, as PyTorch does, so it cannot overflow an
          // intermediate product.
          const quotient =
            outTensor.dtype === gradDtype ? outTensor : castTensor(outTensor, gradDtype);
          const grad = reduceBroadcastGrad(neg(mul(go, div(quotient, denom))), other.tensor.shape);
          other.accumulateGrad(grad);
        }
      },
    });

    return out;
  }

  /**
   * Raise every element to a constant power.
   *
   * Integer and bool tensors raised to a negative or fractional power are
   * promoted to float64 first, because the result is not an integer.
   *
   * @param exponent - Constant number, or a `Tensor` / `GradTensor` exponent (broadcast).
   *   A `GradTensor` exponent receives its own gradient, `out * log(x)`.
   * @throws {DTypeError} For int64 tensors with a number exponent.
   */
  pow(exponent: number | Tensor | GradTensor): GradTensor {
    if (typeof exponent !== "number") {
      return powByTensor(this, gradOperand(this, exponent, "pow"));
    }
    if (!this.tensor.isDeviceTensor && this.tensor.data instanceof BigInt64Array) {
      throw new DTypeError(
        "pow() backward is not supported for int64 tensors. " +
          "Cast to float32/float64 before calling pow() if gradients are needed."
      );
    }
    const srcDtype = this.tensor.dtype;
    if (
      (srcDtype === "int32" || srcDtype === "uint8" || srcDtype === "bool") &&
      (!Number.isInteger(exponent) || exponent < 1)
    ) {
      return this.astype("float64").pow(exponent);
    }
    const exponentTensor = tensor(exponent, { dtype: this.tensor.dtype });
    const outTensor = pow(this.tensor, exponentTensor);
    const requiresGrad = gradEnabled && this.requiresGrad;

    const out = new GradTensor({
      tensor: outTensor,
      requiresGrad,
      prev: requiresGrad ? [this] : [],
      backward: () => {
        if (!requiresGrad) return;
        const go = out._grad;
        if (go === null) {
          throw new DeepboxError("Internal error: missing gradient for pow backward");
        }
        if (exponent === 0) {
          // x^0 is the constant 1, so its derivative is 0 everywhere (including
          // x = 0, where 0 * 0^-1 would otherwise evaluate to NaN).
          this.accumulateGrad(
            zeros(this.tensor.shape, { dtype: this.tensor.dtype, device: this.tensor.device })
          );
          return;
        }
        const powMinusOne = pow(this.tensor, tensor(exponent - 1, { dtype: this.tensor.dtype }));
        const grad = mul(mul(go, tensor(exponent, { dtype: outTensor.dtype })), powMinusOne);
        this.accumulateGrad(grad);
      },
    });

    return out;
  }

  /** Elementwise square root, `pow(0.5)`. */
  sqrt(): GradTensor {
    return this.pow(0.5);
  }

  /**
   * Matrix product with `numpy.matmul` / `torch.matmul` semantics.
   *
   * A 1-D left operand acts as a row vector and a 1-D right operand as a column
   * vector, and the promoted axis is dropped from the result. Leading (batch)
   * dimensions are right-aligned and broadcast, so shapes such as
   * `[1, m, k] @ [3, k, n]`, `[2, 1, m, k] @ [3, k, n]` and `[b, m, k] @ [k, n]`
   * are valid. In the backward pass the gradient of an operand whose batch
   * dimensions were broadcast is summed over them (`reduceBroadcastGrad`).
   *
   * @example
   * ```ts
   * const a = parameter(ones([1, 2, 3]));
   * const b = parameter(ones([4, 3, 5]));
   * const y = a.matmul(b); // shape [4, 2, 5]
   * y.sum().backward();
   * a.grad?.shape; // [1, 2, 3]
   * ```
   */
  matmul(otherArg: Tensor | GradTensor): GradTensor {
    const other = gradFromTensorLike(otherArg, "matmul");
    const outTensor = dot(this.tensor, other.tensor);
    const requiresGrad = gradEnabled && (this.requiresGrad || other.requiresGrad);
    const leftDtype = ensureNumericDType(this.tensor.dtype, "matmul");
    const rightDtype = ensureNumericDType(other.tensor.dtype, "matmul");

    const swapLastTwo = (t: Tensor): Tensor => {
      const axes: number[] = [];
      for (let i = 0; i < t.ndim; i++) {
        axes.push(i);
      }
      const last = t.ndim - 1;
      const secondLast = t.ndim - 2;
      axes[last] = secondLast;
      axes[secondLast] = last;
      return transpose(t, axes);
    };

    const out = new GradTensor({
      tensor: outTensor,
      requiresGrad,
      prev: requiresGrad ? [this, other] : [],
      backward: () => {
        if (!requiresGrad) return;
        const rawGo = out._grad;
        if (rawGo === null) {
          throw new DeepboxError("Internal error: missing gradient for matmul backward");
        }
        // 1-D operands follow NumPy matmul semantics: a 1-D left operand is
        // promoted to [1, k] and a 1-D right operand to [k, 1], with the
        // corresponding dim removed from the output. Re-insert those dims on
        // the upstream gradient and operands so the matrix formulas apply, then
        // squeeze the promoted dim back off each computed gradient.
        const leftIs1d = this.tensor.ndim === 1;
        const rightIs1d = other.tensor.ndim === 1;
        const leftMat = leftIs1d ? this.tensor.reshape([1, this.tensor.size]) : this.tensor;
        const rightMat = rightIs1d ? other.tensor.reshape([other.tensor.size, 1]) : other.tensor;
        let go = rawGo;
        if (leftIs1d || rightIs1d) {
          const goShape = [...rawGo.shape];
          if (rightIs1d) goShape.push(1);
          if (leftIs1d) goShape.splice(Math.max(0, goShape.length - 1), 0, 1);
          go = rawGo.reshape(goShape);
        }
        // Each full gradient has the broadcast batch shape of the output; sum
        // it down to the shape of the operand it belongs to.
        if (this.requiresGrad) {
          const full = dot(go, swapLastTwo(rightMat));
          let grad = castTensor(reduceBroadcastGrad(full, leftMat.shape), leftDtype);
          if (leftIs1d) grad = grad.reshape([this.tensor.size]);
          this.accumulateGrad(grad);
        }
        if (other.requiresGrad) {
          const full = dot(swapLastTwo(leftMat), go);
          let grad = castTensor(reduceBroadcastGrad(full, rightMat.shape), rightDtype);
          if (rightIs1d) grad = grad.reshape([other.tensor.size]);
          other.accumulateGrad(grad);
        }
      },
    });

    return out;
  }

  /** Rectified linear unit, `max(x, 0)`; the gradient at 0 is 0. */
  relu(): GradTensor {
    if (!this.tensor.isDeviceTensor && this.tensor.data instanceof BigInt64Array) {
      const out = new BigInt64Array(this.tensor.size);
      const logicalStrides = computeStrides(this.tensor.shape);
      const contiguous = isContiguous(this.tensor.shape, this.tensor.strides);
      for (let i = 0; i < this.tensor.size; i++) {
        const offset = contiguous
          ? this.tensor.offset + i
          : offsetFromFlatIndex(i, logicalStrides, this.tensor.strides, this.tensor.offset);
        const val = getBigIntElement(this.tensor.data, offset);
        out[i] = val > 0n ? val : 0n;
      }
      if (gradEnabled && this.requiresGrad) {
        // Silently detaching the graph would zero all upstream gradients;
        // fail loudly like pow() does for int64.
        throw new DTypeError(
          "relu gradients are not supported for int64 tensors. " +
            "Cast to float32/float64 before calling relu() if gradients are needed."
        );
      }
      const outTensor = TensorClass.fromTypedArray({
        data: out,
        shape: this.tensor.shape,
        dtype: "int64",
        device: this.tensor.device,
      });
      return GradTensor.fromTensor(outTensor, { requiresGrad: false });
    }
    const outTensor = relu(this.tensor);
    const outDtype = ensureNumericDType(outTensor.dtype, "relu");
    const requiresGrad = gradEnabled && this.requiresGrad;

    const out = new GradTensor({
      tensor: outTensor,
      requiresGrad,
      prev: requiresGrad ? [this] : [],
      backward: () => {
        if (!requiresGrad) return;
        const go = out._grad;
        if (go === null) {
          throw new DeepboxError("Internal error: missing gradient for relu backward");
        }
        if (this.tensor.isDeviceTensor) {
          // Device path: Heaviside mask via the `step` kernel.
          const mask = dispatchUnary("step", this.tensor);
          if (mask === null) {
            throw new DeviceError("relu backward: device kernel unavailable");
          }
          this.accumulateGrad(mul(go, mask));
          return;
        }
        const maskData = new (dtypeToTypedArrayCtor(outDtype))(this.tensor.size);
        const inputDense = asFloat64Dense(this.tensor);
        if (maskData instanceof BigInt64Array) {
          for (let i = 0; i < inputDense.length; i++) {
            const val = inputDense[i] ?? 0;
            maskData[i] = val > 0 ? 1n : 0n;
          }
        } else {
          // maskData is numeric typed array
          for (let i = 0; i < inputDense.length; i++) {
            const val = inputDense[i] ?? 0;
            maskData[i] = val > 0 ? 1 : 0;
          }
        }
        const maskTensor = TensorClass.fromTypedArray({
          data: maskData,
          shape: this.tensor.shape,
          dtype: outDtype,
          device: this.tensor.device,
        });
        const grad = mul(go, maskTensor);
        this.accumulateGrad(grad);
      },
    });

    return out;
  }

  /** Logistic function `1 / (1 + exp(-x))`. */
  sigmoid(): GradTensor {
    const outTensor = sigmoid(this.tensor);
    const requiresGrad = gradEnabled && this.requiresGrad;

    const out = new GradTensor({
      tensor: outTensor,
      requiresGrad,
      prev: requiresGrad ? [this] : [],
      backward: () => {
        if (!requiresGrad) return;
        const go = out._grad;
        if (go === null) {
          throw new DeepboxError("Internal error: missing gradient for sigmoid backward");
        }
        const one = tensor(1, { dtype: outTensor.dtype });
        const sigmoidGrad = mul(outTensor, sub(one, outTensor));
        const grad = mul(go, sigmoidGrad);
        this.accumulateGrad(grad);
      },
    });

    return out;
  }

  /** Elementwise square, `pow(2)`. */
  square(): GradTensor {
    return this.pow(2);
  }

  /** Elementwise exponential. */
  exp(): GradTensor {
    let outTensor = exp(this.tensor);
    const targetDtype = ensureNumericDType(this.tensor.dtype, "exp");
    if (outTensor.dtype !== targetDtype) {
      outTensor = castTensor(outTensor, targetDtype);
    }
    const requiresGrad = gradEnabled && this.requiresGrad;

    const out = new GradTensor({
      tensor: outTensor,
      requiresGrad,
      prev: requiresGrad ? [this] : [],
      backward: () => {
        if (!requiresGrad) return;
        const go = out._grad;
        if (go === null) {
          throw new DeepboxError("Internal error: missing gradient for exp backward");
        }
        // d/dx exp(x) = exp(x)
        const grad = mul(go, outTensor);
        this.accumulateGrad(grad);
      },
    });

    return out;
  }

  /** Elementwise natural logarithm. */
  log(): GradTensor {
    const outTensor = log(this.tensor);
    const requiresGrad = gradEnabled && this.requiresGrad;

    const out = new GradTensor({
      tensor: outTensor,
      requiresGrad,
      prev: requiresGrad ? [this] : [],
      backward: () => {
        if (!requiresGrad) return;
        const go = out._grad;
        if (go === null) {
          throw new DeepboxError("Internal error: missing gradient for log backward");
        }
        // d/dx log(x) = 1/x
        const denom =
          this.tensor.dtype === go.dtype
            ? this.tensor
            : castTensor(this.tensor, ensureNumericDType(go.dtype, "log"));
        const grad = div(go, denom);
        this.accumulateGrad(grad);
      },
    });

    return out;
  }

  /** Elementwise hyperbolic tangent (keeps the input's float dtype). */
  tanh(): GradTensor {
    // Preserve input dtype (the underlying tanh op always emits float64),
    // matching sigmoid/relu so mixed-dtype graphs (e.g. float32 RNNs) don't
    // throw on the next elementwise op.
    const raw = tanh(this.tensor);
    const outTensor =
      raw.dtype === this.tensor.dtype
        ? raw
        : castTensor(raw, ensureNumericDType(this.tensor.dtype, "tanh"));
    const requiresGrad = gradEnabled && this.requiresGrad;

    const out = new GradTensor({
      tensor: outTensor,
      requiresGrad,
      prev: requiresGrad ? [this] : [],
      backward: () => {
        if (!requiresGrad) return;
        const go = out._grad;
        if (go === null) {
          throw new DeepboxError("Internal error: missing gradient for tanh backward");
        }
        // d/dx tanh(x) = 1 - tanh(x)^2
        const tanhSq = mul(outTensor, outTensor);
        const one = tensor(1, { dtype: outTensor.dtype });
        const grad = mul(go, sub(one, tanhSq));
        this.accumulateGrad(grad);
      },
    });

    return out;
  }

  /**
   * Slice the tensor (see {@link slice}); the gradient is scattered back into
   * a zero tensor of the original shape.
   *
   * @param args - One range or index per leading axis.
   */
  slice(...args: SliceRange[]): GradTensor {
    const outTensor = slice(this.tensor, ...args);
    const requiresGrad = gradEnabled && this.requiresGrad;

    const out = new GradTensor({
      tensor: outTensor,
      requiresGrad,
      prev: requiresGrad ? [this] : [],
      backward: () => {
        if (!requiresGrad) return;
        const go = out._grad;
        if (go === null) {
          throw new DeepboxError("Internal error: missing gradient for slice backward");
        }
        // Slice a tensor holding each element's own flat position with the
        // same ranges. The result is exactly the source position of every
        // output element, so the scatter below cannot drift from the forward
        // slicing rules (negative steps, clamping, integer indices).
        const inputShape = this.tensor.shape;
        const positions = new Float64Array(this.tensor.size);
        for (let i = 0; i < positions.length; i++) positions[i] = i;
        const source = asFloat64Dense(
          slice(fromFloat64Dense(inputShape, this.tensor.device, positions), ...args)
        );
        const goDense = asFloat64Dense(go);
        const acc = new Float64Array(this.tensor.size);
        for (let i = 0; i < source.length; i++) {
          const at = source[i] ?? 0;
          acc[at] = (acc[at] ?? 0) + (goDense[i] ?? 0);
        }
        const dtype = ensureNumericDType(this.tensor.dtype, "slice");
        this.accumulateGrad(
          castTensor(fromFloat64Dense(inputShape, this.tensor.device, acc), dtype)
        );
      },
    });
    return out;
  }

  /**
   * Select entries along `axis` using a 1-D index tensor.
   *
   * Indices are not differentiable; the gradient is scatter-added back into
   * the positions that were read, so repeated indices accumulate.
   *
   * @param indices - 1-D integer indices (`Tensor` or `GradTensor`).
   * @param axis - Axis to gather along.
   */
  gather(indicesArg: Tensor | GradTensor, axis: Axis): GradTensor {
    const indices = gradFromTensorLike(indicesArg, "gather");
    const outTensor = gather(this.tensor, indices.tensor, axis);
    const ax = normalizeAxis(axis, this.tensor.ndim);
    const requiresGrad = gradEnabled && this.requiresGrad;

    const out = new GradTensor({
      tensor: outTensor,
      requiresGrad,
      prev: requiresGrad ? [this] : [],
      backward: () => {
        if (!requiresGrad) return;
        const go = out._grad;
        if (go === null) {
          throw new DeepboxError("Internal error: missing gradient for gather backward");
        }
        // scatter_add: viewed as [outer, axis, inner], output row j along the
        // axis came from input row idx[j].
        const dtype = ensureNumericDType(this.tensor.dtype, "gather");
        const inShape = this.tensor.shape;
        const axisSize = inShape[ax] ?? 1;
        let outer = 1;
        for (let d = 0; d < ax; d++) outer *= inShape[d] ?? 1;
        let inner = 1;
        for (let d = ax + 1; d < inShape.length; d++) inner *= inShape[d] ?? 1;
        const idxDense = asFloat64Dense(indices.tensor);
        const n = idxDense.length;
        const goDense = asFloat64Dense(go);
        const acc = new Float64Array(this.tensor.size);
        for (let o = 0; o < outer; o++) {
          for (let j = 0; j < n; j++) {
            const row = Math.round(idxDense[j] ?? 0);
            const src = (o * n + j) * inner;
            const dst = (o * axisSize + row) * inner;
            for (let k = 0; k < inner; k++) {
              acc[dst + k] = (acc[dst + k] ?? 0) + (goDense[src + k] ?? 0);
            }
          }
        }
        const gradTensor = castTensor(fromFloat64Dense(inShape, this.tensor.device, acc), dtype);
        this.accumulateGrad(gradTensor);
      },
    });
    return out;
  }

  /**
   * Mean over `axis`, or over all elements when `axis` is omitted.
   *
   * @param axis - Axis or list of axes to reduce (negative values count from the end).
   * @param keepdims - Keep the reduced axes with size 1.
   * @throws {InvalidParameterError} If an axis is out of range or listed twice.
   *
   * @example
   * ```ts
   * const x = parameter([[1, 2], [3, 4]]);
   * x.mean([0, 1]).toArray(); // 2.5
   * ```
   */
  mean(axis?: Axis | readonly Axis[], keepdims = false): GradTensor {
    const n = reducedCount(this.tensor.shape, resolveAxisList(axis, this.tensor.ndim));
    const summed = this.sum(axis, keepdims);
    // Use summed tensor's dtype (which may be promoted by sum) to avoid dtype mismatch in div
    const denom = GradTensor.scalar(n, {
      dtype: ensureNumericDType(summed.tensor.dtype, "mean"),
    });
    return summed.div(denom);
  }

  /**
   * Maximum over `axis`, or over all elements when `axis` is omitted. The
   * gradient is split equally among tied maxima; a NaN maximum sends the
   * gradient to the NaN element(s).
   *
   * @param axis - Axis or list of axes to reduce (negative values count from the end).
   * @param keepdims - Keep the reduced axes with size 1.
   *
   * @example
   * ```ts
   * const x = parameter([[1, 5], [3, 2]]);
   * x.max([0, 1]).toArray(); // 5
   * ```
   */
  max(axis?: Axis | readonly Axis[], keepdims = false): GradTensor {
    const axes = resolveAxisList(axis, this.tensor.ndim);
    const outTensor = max(this.tensor, kernelAxis(axes), keepdims);
    const requiresGrad = gradEnabled && this.requiresGrad;

    const out = new GradTensor({
      tensor: outTensor,
      requiresGrad,
      prev: requiresGrad ? [this] : [],
      backward: () => {
        if (!requiresGrad) return;
        const go = out._grad;
        if (go === null) {
          throw new DeepboxError("Internal error: missing gradient for max backward");
        }

        const grad = extremumGrad(this.tensor, outTensor, go, axes, keepdims, "max");
        this.accumulateGrad(grad);
      },
    });

    return out;
  }

  /**
   * Reshape the GradTensor to a new shape without copying data.
   *
   * Returns a new GradTensor with the specified shape. The underlying tensor
   * is reshaped, and gradient computation is preserved through the reshape operation.
   *
   * @param newShape - The desired shape for the tensor
   * @returns A new GradTensor with the specified shape
   * @throws {ShapeError} If the new shape is incompatible with the tensor's size
   *
   * @example
   * ```ts
   * const t = parameter([1, 2, 3, 4, 5, 6]);
   * const reshaped = t.reshape([2, 3]);
   * console.log(reshaped.shape); // [2, 3]
   * ```
   */
  reshape(newShape: Shape): GradTensor {
    const reshapedTensor = this.tensor.reshape(newShape);
    const requiresGrad = gradEnabled && this.requiresGrad;

    const out = new GradTensor({
      tensor: reshapedTensor,
      requiresGrad,
      prev: requiresGrad ? [this] : [],
      backward: () => {
        if (!requiresGrad) return;
        const go = out._grad;
        if (go === null) {
          throw new DeepboxError("Internal error: missing gradient for reshape backward");
        }
        // Reshape gradient back to original shape
        const reshapedGrad = go.reshape(this.tensor.shape);
        this.accumulateGrad(reshapedGrad);
      },
    });

    return out;
  }

  /**
   * Flatten the GradTensor to a 1-dimensional array.
   *
   * Returns a new 1D GradTensor containing all elements.
   *
   * @returns A 1D GradTensor with shape [size]
   *
   * @example
   * ```ts
   * const matrix = parameter([[1, 2, 3], [4, 5, 6]]);
   * const flat = matrix.flatten();
   * console.log(flat.shape); // [6]
   * ```
   */
  flatten(): GradTensor {
    return this.reshape([this.tensor.size]);
  }

  /**
   * Create a view of the GradTensor with a different shape.
   *
   * Similar to reshape but uses the underlying tensor's view method.
   *
   * @param shape - The desired shape for the view
   * @param strides - Optional custom strides
   * @param offset - Optional offset into the data buffer
   * @returns A new GradTensor view with the specified shape
   */
  view(shape: Shape, strides?: readonly number[], offset?: number): GradTensor {
    const viewTensor = this.tensor.view(shape, strides, offset);
    const requiresGrad = gradEnabled && this.requiresGrad;

    const out = new GradTensor({
      tensor: viewTensor,
      requiresGrad,
      prev: requiresGrad ? [this] : [],
      backward: () => {
        if (!requiresGrad) return;
        const go = out._grad;
        if (go === null) {
          throw new DeepboxError("Internal error: missing gradient for view backward");
        }
        if (strides === undefined && offset === undefined) {
          // Plain reshape view: gradient is just reshaped back
          this.accumulateGrad(go.reshape(this.tensor.shape));
          return;
        }
        // Custom strides/offset: elements of the base may be referenced by
        // zero, one, or several view positions (e.g. overlapping windows), so
        // the gradient must be scatter-ADDED back (a reshape would be wrong).
        const base = this.tensor;
        if (!isContiguous(base.shape, base.strides)) {
          throw new DeepboxError(
            "view backward with custom strides requires a contiguous base tensor"
          );
        }
        const goDense = asFloat64Dense(go);
        const gradData = new Float64Array(base.size);
        const viewShape = viewTensor.shape;
        const viewStrides = viewTensor.strides;
        const relOffset = viewTensor.offset - base.offset;
        const viewLogicalStrides = computeStrides(viewShape);
        for (let i = 0; i < viewTensor.size; i++) {
          let rem = i;
          let off = relOffset;
          for (let d = 0; d < viewShape.length; d++) {
            const ls = viewLogicalStrides[d] ?? 1;
            const coord = Math.floor(rem / ls);
            rem -= coord * ls;
            off += coord * (viewStrides[d] ?? 0);
          }
          gradData[off] = (gradData[off] ?? 0) + (goDense[i] ?? 0);
        }
        const gradTensor = TensorClass.fromTypedArray({
          data: gradData,
          shape: base.shape,
          dtype: "float64",
          device: base.device,
        });
        this.accumulateGrad(castTensor(gradTensor, ensureNumericDType(base.dtype, "view")));
      },
    });

    return out;
  }
  /**
   * Permute the axes (reverse them when `axes` is omitted). Negative axes count
   * from the end.
   */
  transpose(axes?: readonly number[]): GradTensor {
    const outTensor = transpose(this.tensor, axes);
    const requiresGrad = gradEnabled && this.requiresGrad;

    const out = new GradTensor({
      tensor: outTensor,
      requiresGrad,
      prev: requiresGrad ? [this] : [],
      backward: () => {
        if (!requiresGrad) return;
        const go = out._grad;
        if (go === null) {
          throw new DeepboxError("Internal error: missing gradient for transpose backward");
        }
        // Transpose gradient back to original shape
        // For default transpose (reverse axes), we just apply it again
        // For custom axes, we need inverse permutation
        if (axes === undefined) {
          const grad = transpose(go);
          this.accumulateGrad(grad);
        } else {
          // Inverse permutation (normalize negative axes first, e.g.
          // transpose([-1, -2]) must invert as [ndim-1, ndim-2])
          const ndim = this.tensor.ndim;
          const invAxes = new Array<number>(axes.length);
          for (let i = 0; i < axes.length; i++) {
            const axis = axes[i];
            if (axis !== undefined) {
              invAxes[axis < 0 ? axis + ndim : axis] = i;
            }
          }
          const grad = transpose(go, invAxes);
          this.accumulateGrad(grad);
        }
      },
    });

    return out;
  }

  /**
   * Minimum over `axis`, or over all elements when `axis` is omitted. The
   * gradient is split equally among tied minima; a NaN minimum sends the
   * gradient to the NaN element(s).
   *
   * @param axis - Axis or list of axes to reduce (negative values count from the end).
   * @param keepdims - Keep the reduced axes with size 1.
   *
   * @example
   * ```ts
   * const x = parameter([[1, 5], [3, 2]]);
   * x.min(0).toArray(); // [1, 2]
   * ```
   */
  min(axis?: Axis | readonly Axis[], keepdims = false): GradTensor {
    const axes = resolveAxisList(axis, this.tensor.ndim);
    const outTensor = min(this.tensor, kernelAxis(axes), keepdims);
    const requiresGrad = gradEnabled && this.requiresGrad;

    const out = new GradTensor({
      tensor: outTensor,
      requiresGrad,
      prev: requiresGrad ? [this] : [],
      backward: () => {
        if (!requiresGrad) return;
        const go = out._grad;
        if (go === null) {
          throw new DeepboxError("Internal error: missing gradient for min backward");
        }

        const grad = extremumGrad(this.tensor, outTensor, go, axes, keepdims, "min");
        this.accumulateGrad(grad);
      },
    });

    return out;
  }

  /** Elementwise absolute value; the gradient at 0 is 0. */
  abs(): GradTensor {
    const outTensor = absOp(this.tensor);
    const requiresGrad = gradEnabled && this.requiresGrad;

    const out = new GradTensor({
      tensor: outTensor,
      requiresGrad,
      prev: requiresGrad ? [this] : [],
      backward: () => {
        if (!requiresGrad) return;
        const go = out._grad;
        if (go === null) {
          throw new DeepboxError("Internal error: missing gradient for abs backward");
        }
        // d/dx |x| = sign(x), but sign(0) = 0 so gradient is 0 at x=0
        const zeroT = zeros(this.tensor.shape, {
          dtype: this.tensor.dtype,
          device: this.tensor.device,
        });
        const posMask = castTensor(
          greater(this.tensor, zeroT),
          ensureNumericDType(this.tensor.dtype, "abs")
        );
        const negMask = castTensor(
          less(this.tensor, zeroT),
          ensureNumericDType(this.tensor.dtype, "abs")
        );
        const signT = sub(posMask, negMask);
        const grad = mul(go, signT);
        this.accumulateGrad(grad);
      },
    });

    return out;
  }

  /**
   * Clamp to `[minVal, maxVal]`; either bound may be omitted. The gradient is 1
   * inside the interval (ends included) and 0 outside.
   *
   * @param minVal - Lower bound
   * @param maxVal - Upper bound
   */
  clip(minVal?: number, maxVal?: number): GradTensor {
    const outTensor = clipOp(this.tensor, minVal, maxVal);
    const requiresGrad = gradEnabled && this.requiresGrad;

    const out = new GradTensor({
      tensor: outTensor,
      requiresGrad,
      prev: requiresGrad ? [this] : [],
      backward: () => {
        if (!requiresGrad) return;
        const go = out._grad;
        if (go === null) {
          throw new DeepboxError("Internal error: missing gradient for clip backward");
        }
        // Gradient passes through where input is within [min, max], zero elsewhere:
        // mask = (x >= min) && (x <= max), with a missing bound always satisfied.
        const dtype = ensureNumericDType(this.tensor.dtype, "clip");
        let mask = onesLike(this.tensor);
        if (minVal !== undefined) {
          const lowT = tensor(minVal, { dtype: this.tensor.dtype });
          mask = mul(mask, castTensor(greaterEqual(this.tensor, lowT), dtype));
        }
        if (maxVal !== undefined) {
          const highT = tensor(maxVal, { dtype: this.tensor.dtype });
          mask = mul(mask, castTensor(lessEqual(this.tensor, highT), dtype));
        }
        this.accumulateGrad(mul(go, mask));
      },
    });

    return out;
  }

  /** Leaky ReLU: `x` for `x > 0`, otherwise `negativeSlope * x`. */
  leakyRelu(negativeSlope = 0.01): GradTensor {
    const outTensor = leakyRelu(this.tensor, negativeSlope);
    const requiresGrad = gradEnabled && this.requiresGrad;

    const out = new GradTensor({
      tensor: outTensor,
      requiresGrad,
      prev: requiresGrad ? [this] : [],
      backward: () => {
        if (!requiresGrad) return;
        const go = out._grad;
        if (go === null) {
          throw new DeepboxError("Internal error: missing gradient for leakyRelu backward");
        }
        // d/dx leakyRelu(x) = 1 if x > 0, negativeSlope if x <= 0
        const dtype = ensureNumericDType(outTensor.dtype, "leakyRelu");
        const zeroT = zeros(this.tensor.shape, {
          dtype: this.tensor.dtype,
          device: this.tensor.device,
        });
        const posMask = castTensor(greater(this.tensor, zeroT), dtype);
        // slopeVals = negativeSlope * (1 - posMask) + 1 * posMask
        const slopeT = tensor(negativeSlope, { dtype });
        const oneT = tensor(1, { dtype });
        const negMask = sub(oneT, posMask);
        const slopeVals = add(posMask, mul(negMask, slopeT));
        const grad = mul(go, slopeVals);
        this.accumulateGrad(grad);
      },
    });

    return out;
  }

  /** Exponential linear unit: `x` for `x > 0`, otherwise `alpha * (exp(x) - 1)`. */
  elu(alpha = 1.0): GradTensor {
    const outTensor = keepFloatDtype(elu(this.tensor, alpha), this.tensor.dtype, "elu");
    const requiresGrad = gradEnabled && this.requiresGrad;

    const out = new GradTensor({
      tensor: outTensor,
      requiresGrad,
      prev: requiresGrad ? [this] : [],
      backward: () => {
        if (!requiresGrad) return;
        const go = out._grad;
        if (go === null) {
          throw new DeepboxError("Internal error: missing gradient for elu backward");
        }
        // d/dx elu(x) = 1 if x > 0, alpha * exp(x) if x <= 0
        //             = 1 if x > 0, elu(x) + alpha if x <= 0
        const dtype = ensureNumericDType(outTensor.dtype, "elu");
        const zeroT = zeros(this.tensor.shape, {
          dtype: this.tensor.dtype,
          device: this.tensor.device,
        });
        const posMask = castTensor(greater(this.tensor, zeroT), dtype);
        const oneT = tensor(1, { dtype });
        const negMask = sub(oneT, posMask);
        const alphaT = tensor(alpha, { dtype });
        // For x <= 0: derivative = elu(x) + alpha
        const negDeriv = add(outTensor, alphaT);
        const derivVals = add(posMask, mul(negMask, negDeriv));
        const grad = mul(go, derivVals);
        this.accumulateGrad(grad);
      },
    });

    return out;
  }

  /**
   * Gaussian error linear unit. `approximate` selects the tanh approximation
   * (`"tanh"`, the default) or the exact erf form (`"none"`); the gradient is
   * exact for the chosen form. Device tensors are differentiated on the device.
   *
   * @param options - `{ approximate?: "tanh" | "none" }`, or the bare string (same as `Tensor.gelu`)
   */
  gelu(options: GeluApproximation | GeluOptions = {}): GradTensor {
    const approximate: GeluApproximation =
      typeof options === "string" ? options : (options?.approximate ?? "tanh");
    const outTensor = keepFloatDtype(gelu(this.tensor, approximate), this.tensor.dtype, "gelu");
    const requiresGrad = gradEnabled && this.requiresGrad;

    const out = new GradTensor({
      tensor: outTensor,
      requiresGrad,
      prev: requiresGrad ? [this] : [],
      backward: () => {
        if (!requiresGrad) return;
        const go = out._grad;
        if (go === null) {
          throw new DeepboxError("Internal error: missing gradient for gelu backward");
        }
        const dtype = ensureNumericDType(outTensor.dtype, "gelu");
        const derivTensor = geluDerivative(this.tensor, approximate);
        const deriv = derivTensor.dtype === dtype ? derivTensor : castTensor(derivTensor, dtype);
        this.accumulateGrad(mul(go, deriv));
      },
    });

    return out;
  }

  /** Clamp to `[minVal, maxVal]`; the gradient is 1 strictly inside the interval and 0 elsewhere. */
  hardtanh(minVal = -1, maxVal = 1): GradTensor {
    const outTensor = hardtanhOp(this.tensor, minVal, maxVal);
    const outDtype = ensureNumericDType(outTensor.dtype, "hardtanh");
    const requiresGrad = gradEnabled && this.requiresGrad;

    const out = new GradTensor({
      tensor: outTensor,
      requiresGrad,
      prev: requiresGrad ? [this] : [],
      backward: () => {
        if (!requiresGrad) return;
        const go = out._grad;
        if (go === null) {
          throw new DeepboxError("Internal error: missing gradient for hardtanh backward");
        }
        if (this.tensor.isDeviceTensor) {
          // Device path: (x > min) * (x < max) from two Heaviside `step` kernels.
          const aboveMin = dispatchUnary("step", addScalar(this.tensor, -minVal));
          const belowMax = dispatchUnary("step", addScalar(neg(this.tensor), maxVal));
          if (aboveMin === null || belowMax === null) {
            throw new DeviceError("hardtanh backward: device kernel unavailable");
          }
          this.accumulateGrad(mul(go, mul(aboveMin, belowMax)));
          return;
        }
        const maskData = new (dtypeToTypedArrayCtor(outDtype))(this.tensor.size);
        const inputDense = asFloat64Dense(this.tensor);
        if (maskData instanceof BigInt64Array) {
          for (let i = 0; i < inputDense.length; i++) {
            const val = inputDense[i] ?? 0;
            maskData[i] = val > minVal && val < maxVal ? 1n : 0n;
          }
        } else {
          for (let i = 0; i < inputDense.length; i++) {
            const val = inputDense[i] ?? 0;
            maskData[i] = val > minVal && val < maxVal ? 1 : 0;
          }
        }
        const maskTensor = TensorClass.fromTypedArray({
          data: maskData,
          shape: this.tensor.shape,
          dtype: outDtype,
          device: this.tensor.device,
        });
        const grad = mul(go, maskTensor);
        this.accumulateGrad(grad);
      },
    });

    return out;
  }

  /** `x - tanh(x)`. */
  tanhshrink(): GradTensor {
    const outTensor = keepFloatDtype(tanhshrinkOp(this.tensor), this.tensor.dtype, "tanhshrink");
    const requiresGrad = gradEnabled && this.requiresGrad;

    const out = new GradTensor({
      tensor: outTensor,
      requiresGrad,
      prev: requiresGrad ? [this] : [],
      backward: () => {
        if (!requiresGrad) return;
        const go = out._grad;
        if (go === null) {
          throw new DeepboxError("Internal error: missing gradient for tanhshrink backward");
        }
        // d/dx (x - tanh(x)) = tanh(x)^2
        const tanhX = keepFloatDtype(tanh(this.tensor), this.tensor.dtype, "tanhshrink");
        const gradVal = mul(tanhX, tanhX);
        this.accumulateGrad(reduceBroadcastGrad(mul(go, gradVal), this.tensor.shape));
      },
    });

    return out;
  }

  /**
   * Elementwise sine. The gradient is `cos(x)`.
   *
   * @example
   * ```ts
   * const x = parameter([0, Math.PI]);
   * x.sin().sum().backward();
   * x.grad?.toArray(); // [1, -1]
   * ```
   */
  sin(): GradTensor {
    const x = this.tensor;
    const outTensor = keepFloatDtype(sinOp(x), x.dtype, "sin");
    return unaryNode(this, outTensor, "sin", (go) => chain(go, cosOp(asF64(x))));
  }

  /**
   * Elementwise cosine. The gradient is `-sin(x)`.
   *
   * @example
   * ```ts
   * const x = parameter([0]);
   * x.cos().toArray(); // [1]
   * ```
   */
  cos(): GradTensor {
    const x = this.tensor;
    const outTensor = keepFloatDtype(cosOp(x), x.dtype, "cos");
    return unaryNode(this, outTensor, "cos", (go) => neg(chain(go, sinOp(asF64(x)))));
  }

  /**
   * Elementwise tangent. The gradient is `1 + tan(x)^2`.
   *
   * @example
   * ```ts
   * const x = parameter([0]);
   * x.tan().sum().backward();
   * x.grad?.toArray(); // [1]
   * ```
   */
  tan(): GradTensor {
    const x = this.tensor;
    const outTensor = keepFloatDtype(tanOp(x), x.dtype, "tan");
    return unaryNode(this, outTensor, "tan", (go) => {
      const t = asF64(outTensor);
      return chain(go, addScalar(mul(t, t), 1));
    });
  }

  /**
   * `log(1 + x)`, accurate for small `x`. The gradient is `1 / (1 + x)`.
   *
   * @example
   * ```ts
   * const x = parameter([0, 1]);
   * x.log1p().sum().backward();
   * x.grad?.toArray(); // [1, 0.5]
   * ```
   */
  log1p(): GradTensor {
    const x = this.tensor;
    const outTensor = keepFloatDtype(log1pOp(x), x.dtype, "log1p");
    return unaryNode(this, outTensor, "log1p", (go) =>
      mul(asF64(go), div(f64Const(1), addScalar(asF64(x), 1))).astype(go.dtype)
    );
  }

  /**
   * `exp(x) - 1`, accurate for small `x`. The gradient is `exp(x)`.
   *
   * @example
   * ```ts
   * const x = parameter([0]);
   * x.expm1().sum().backward();
   * x.grad?.toArray(); // [1]
   * ```
   */
  expm1(): GradTensor {
    const x = this.tensor;
    const outTensor = keepFloatDtype(expm1Op(x), x.dtype, "expm1");
    return unaryNode(this, outTensor, "expm1", (go) => chain(go, exp(asF64(x))));
  }

  /**
   * Softplus, `log(1 + exp(x))`, computed without overflow. The gradient is
   * `sigmoid(x)`.
   */
  softplus(): GradTensor {
    const x = this.tensor;
    const outTensor = keepFloatDtype(softplusOp(x), x.dtype, "softplus");
    return unaryNode(this, outTensor, "softplus", (go) => chain(go, sigmoid(asF64(x))));
  }

  /**
   * Log of the logistic function, `-softplus(-x)`, stable for large `|x|`. The
   * gradient is `sigmoid(-x)`.
   *
   * @example
   * ```ts
   * const x = parameter([0]);
   * x.logSigmoid().toArray(); // [-0.6931...]
   * ```
   */
  logSigmoid(): GradTensor {
    const x = this.tensor;
    const outTensor = keepFloatDtype(neg(softplusOp(neg(asF64(x)))), x.dtype, "logSigmoid");
    return unaryNode(this, outTensor, "logSigmoid", (go) => chain(go, sigmoid(neg(asF64(x)))));
  }

  /**
   * Swish (SiLU), `x * sigmoid(x)`. The gradient is `s * (1 + x * (1 - s))`
   * with `s = sigmoid(x)`.
   *
   * @example
   * ```ts
   * const x = parameter([0]);
   * x.swish().sum().backward();
   * x.grad?.toArray(); // [0.5]
   * ```
   */
  swish(): GradTensor {
    const x = this.tensor;
    const outTensor = keepFloatDtype(swishOp(x), x.dtype, "swish");
    return unaryNode(this, outTensor, "swish", (go) => {
      const xf = asF64(x);
      const s = sigmoid(xf);
      return chain(go, mul(s, addScalar(mul(xf, sub(f64Const(1), s)), 1)));
    });
  }

  /**
   * Mish, `x * tanh(softplus(x))`.
   *
   * @example
   * ```ts
   * const x = parameter([0]);
   * x.mish().toArray(); // [0]
   * ```
   */
  mish(): GradTensor {
    const x = this.tensor;
    const outTensor = keepFloatDtype(mishOp(x), x.dtype, "mish");
    return unaryNode(this, outTensor, "mish", (go) => {
      const xf = asF64(x);
      const t = tanh(softplusOp(xf));
      // d/dx = tanh(sp) + x * sigmoid(x) * (1 - tanh(sp)^2)
      const tail = mul(mul(xf, sigmoid(xf)), sub(f64Const(1), mul(t, t)));
      return chain(go, add(t, tail));
    });
  }

  /**
   * Scaled ELU, `1.0507... * elu(x, 1.6732...)`, the self-normalizing activation.
   *
   * @example
   * ```ts
   * const x = parameter([1, 0]);
   * x.selu().toArray(); // [1.0507..., 0]
   * ```
   */
  selu(): GradTensor {
    const x = this.tensor;
    const outTensor = keepFloatDtype(
      mulScalar(elu(asF64(x), SELU_ALPHA), SELU_SCALE),
      x.dtype,
      "selu"
    );
    return unaryNode(this, outTensor, "selu", (go) => {
      const xf = asF64(x);
      // x > 0: scale. x <= 0: scale * alpha * exp(x). The exponent is clamped at 0 so the
      // unused branch never overflows.
      const negSide = mulScalar(exp(minimumOp(xf, f64Const(0))), SELU_ALPHA * SELU_SCALE);
      const deriv = whereOp(greater(xf, f64Const(0)), f64Const(SELU_SCALE), negSide);
      return chain(go, deriv);
    });
  }

  /**
   * Softmax along `axis` (default last). See {@link softmax}.
   *
   * @example
   * ```ts
   * const x = parameter([[1, 1], [0, 2]]);
   * x.softmax().toArray(); // [[0.5, 0.5], [0.119, 0.881]]
   * ```
   */
  softmax(axis: Axis = -1): GradTensor {
    return softmax(this, axis);
  }

  /**
   * Log-softmax along `axis` (default last). See {@link logSoftmax}.
   *
   * @example
   * ```ts
   * const x = parameter([[1, 1]]);
   * x.logSoftmax().toArray(); // [[-0.6931..., -0.6931...]]
   * ```
   */
  logSoftmax(axis: Axis = -1): GradTensor {
    return logSoftmax(this, axis);
  }

  /**
   * Cumulative sum along `axis`; the input is flattened first when `axis` is
   * omitted (NumPy semantics). The gradient is the reversed cumulative sum of
   * the upstream gradient.
   *
   * @param axis - Axis to accumulate along (negative values count from the end).
   * @throws {InvalidParameterError} If `axis` is out of range.
   *
   * @example
   * ```ts
   * const x = parameter([1, 2, 3]);
   * const y = x.cumsum(); // [1, 3, 6]
   * y.sum().backward();
   * x.grad?.toArray(); // [3, 2, 1]
   * ```
   */
  cumsum(axis?: Axis): GradTensor {
    const x = this.tensor;
    const ax = axis === undefined ? undefined : normalizeAxis(axis, x.ndim);
    const outTensor = keepFloatDtype(cumsumOp(x, ax), x.dtype, "cumsum");
    return unaryNode(this, outTensor, "cumsum", (go) => {
      const along = ax ?? 0;
      const reversed = cumsumOp(flipOp(go, along), along);
      const grad = toContiguous(flipOp(reversed, along));
      return asDtype(ax === undefined ? grad.reshape(x.shape) : grad, gradDtypeFor(x.dtype));
    });
  }

  /**
   * Product over `axis`, a list of axes, or all elements. The gradient is the
   * product of the other elements of each lane, so it is exact when a lane
   * contains zeros.
   *
   * @param axis - Axis or list of axes to reduce (negative values count from the end).
   * @param keepdims - Keep the reduced axes with size 1.
   *
   * @example
   * ```ts
   * const x = parameter([2, 3, 4]);
   * x.prod().backward();
   * x.grad?.toArray(); // [12, 8, 6]
   * ```
   */
  prod(axis?: Axis | readonly Axis[], keepdims = false): GradTensor {
    const x = this.tensor;
    const axes = resolveAxisList(axis, x.ndim);
    const outTensor = keepFloatDtype(prodOp(x, kernelAxis(axes), keepdims), x.dtype, "prod");
    return unaryNode(this, outTensor, "prod", (go) => {
      const { bases, offsets } = reductionLanes(x.shape, axes);
      const values = asFloat64Dense(x);
      const upstream = asFloat64Dense(go);
      const grad = new Float64Array(x.size);
      const count = offsets.length;
      const prefix = new Float64Array(count + 1);
      const suffix = new Float64Array(count + 1);
      for (let lane = 0; lane < bases.length; lane++) {
        const base = bases[lane] as number;
        prefix[0] = 1;
        for (let r = 0; r < count; r++) {
          prefix[r + 1] = (prefix[r] as number) * (values[base + (offsets[r] as number)] as number);
        }
        suffix[count] = 1;
        for (let r = count - 1; r >= 0; r--) {
          suffix[r] = (suffix[r + 1] as number) * (values[base + (offsets[r] as number)] as number);
        }
        const g = upstream[lane] as number;
        for (let r = 0; r < count; r++) {
          grad[base + (offsets[r] as number)] =
            g * (prefix[r] as number) * (suffix[r + 1] as number);
        }
      }
      return asDtype(fromFloat64Dense(x.shape, x.device, grad), gradDtypeFor(x.dtype));
    });
  }

  /**
   * Variance over `axis`, a list of axes, or all elements.
   *
   * @param axis - Axis or list of axes to reduce.
   * @param keepdims - Keep the reduced axes with size 1.
   * @param ddof - Delta degrees of freedom: the divisor is `n - ddof` (default 0, population variance).
   * @throws {InvalidParameterError} If the tensor is empty, `ddof` is negative or not below `n`, or an axis is invalid.
   *
   * @example
   * ```ts
   * const x = parameter([1, 2, 3, 4]);
   * x.var(undefined, false, 1).toArray(); // 1.666...
   * ```
   */
  var(axis?: Axis | readonly Axis[], keepdims = false, ddof = 0): GradTensor {
    const x = this.tensor;
    const axes = resolveAxisList(axis, x.ndim);
    const outTensor = keepFloatDtype(
      varianceOp(x, kernelAxis(axes), keepdims, ddof),
      x.dtype,
      "var"
    );
    return unaryNode(this, outTensor, "var", (go) => {
      const xf = asF64(x);
      const keep = keepdimsShape(x.shape, axes);
      const centered = sub(xf, meanOp(xf, kernelAxis(axes), true));
      const scale = 2 / (reducedCount(x.shape, axes) - ddof);
      const goK = asF64(toContiguous(go)).reshape(keep);
      return asDtype(mul(mulScalar(centered, scale), goK), gradDtypeFor(x.dtype));
    });
  }

  /**
   * Standard deviation over `axis`, a list of axes, or all elements. The
   * gradient is 0 for a lane with zero spread, as in PyTorch.
   *
   * @param axis - Axis or list of axes to reduce.
   * @param keepdims - Keep the reduced axes with size 1.
   * @param ddof - Delta degrees of freedom: the divisor is `n - ddof` (default 0).
   * @throws {InvalidParameterError} If the tensor is empty, `ddof` is negative or not below `n`, or an axis is invalid.
   *
   * @example
   * ```ts
   * const x = parameter([[1, 2], [3, 5]]);
   * x.std(0).toArray(); // [1, 1.5]
   * ```
   */
  std(axis?: Axis | readonly Axis[], keepdims = false, ddof = 0): GradTensor {
    const x = this.tensor;
    const axes = resolveAxisList(axis, x.ndim);
    const outTensor = keepFloatDtype(stdOp(x, kernelAxis(axes), keepdims, ddof), x.dtype, "std");
    return unaryNode(this, outTensor, "std", (go) => {
      const xf = asF64(x);
      const keep = keepdimsShape(x.shape, axes);
      const centered = sub(xf, meanOp(xf, kernelAxis(axes), true));
      const denom = reducedCount(x.shape, axes) - ddof;
      const goK = asF64(toContiguous(go)).reshape(keep);
      const stdK = asF64(outTensor).reshape(keep);
      const grad = div(mul(centered, goK), mulScalar(stdK, denom));
      // A lane without spread has std 0 and a 0/0 gradient; PyTorch defines it as 0.
      const flatLane = equal(stdK, f64Const(0));
      return asDtype(whereOp(flatLane, f64Const(0), grad), gradDtypeFor(x.dtype));
    });
  }

  /**
   * Elementwise maximum with broadcasting. NaN propagates. Where the two
   * inputs are equal each receives half of the gradient (PyTorch semantics).
   *
   * @throws {ShapeError} If the shapes do not broadcast.
   *
   * @example
   * ```ts
   * const a = parameter([1, 5, 3]);
   * const b = parameter([4, 2, 3]);
   * a.maximum(b).toArray(); // [4, 5, 3]
   * ```
   */
  maximum(other: number | Tensor | GradTensor): GradTensor {
    return pairExtremum(this, gradOperand(this, other, "maximum"), true);
  }

  /**
   * Elementwise minimum with broadcasting. NaN propagates. Where the two
   * inputs are equal each receives half of the gradient (PyTorch semantics).
   *
   * @throws {ShapeError} If the shapes do not broadcast.
   *
   * @example
   * ```ts
   * const a = parameter([1, 5, 3]);
   * const b = parameter([4, 2, 3]);
   * a.minimum(b).toArray(); // [1, 2, 3]
   * ```
   */
  minimum(other: number | Tensor | GradTensor): GradTensor {
    return pairExtremum(this, gradOperand(this, other, "minimum"), false);
  }

  /**
   * Select between two tensors by a condition, with broadcasting:
   * `condition ? a : b`. The condition is not differentiable; each branch
   * receives the upstream gradient where it was selected and zero elsewhere.
   * A number for `a` or `b` is wrapped as a constant of the other operand's dtype.
   *
   * @param condition - Boolean (or numeric, nonzero means true) tensor.
   * @param a - Values where the condition holds.
   * @param b - Values elsewhere.
   * @throws {DTypeError} If `a` and `b` have different dtypes.
   * @throws {ShapeError} If the shapes do not broadcast.
   *
   * @example
   * ```ts
   * const x = parameter([-1, 2, -3]);
   * const y = GradTensor.where(x.tensor.astype("bool"), x, x.neg());
   * y.toArray(); // [1, 2, 3]
   * ```
   */
  static where(
    condition: GradTensor | Tensor,
    a: GradTensor | number,
    b: GradTensor | number
  ): GradTensor {
    const condTensor = condition instanceof GradTensor ? condition.tensor : condition;
    const reference = a instanceof GradTensor ? a : b instanceof GradTensor ? b : undefined;
    let operandDtype: NumericDType =
      reference === undefined ? "float32" : ensureNumericDType(reference.dtype, "where");
    const needsFloat = (v: GradTensor | number): boolean =>
      typeof v === "number" && !Number.isInteger(v) && !isFloatDType(operandDtype);
    let left = a;
    let right = b;
    if (needsFloat(a) || needsFloat(b)) {
      // A fractional constant cannot live in an integer tensor: promote the tensor operand.
      operandDtype = "float64";
      if (left instanceof GradTensor) left = left.astype("float64");
      if (right instanceof GradTensor) right = right.astype("float64");
    }
    const lhs = typeof left === "number" ? GradTensor.scalar(left, { dtype: operandDtype }) : left;
    const rhs =
      typeof right === "number" ? GradTensor.scalar(right, { dtype: operandDtype }) : right;
    const outTensor = whereOp(condTensor, lhs.tensor, rhs.tensor);
    const requiresGrad = gradEnabled && (lhs.requiresGrad || rhs.requiresGrad);
    const out: GradTensor = new GradTensor({
      tensor: outTensor,
      requiresGrad,
      prev: requiresGrad ? [lhs, rhs] : [],
      backward: () => {
        if (!requiresGrad) return;
        const go = out.grad;
        if (go === null)
          throw new DeepboxError("Internal error: missing gradient for where backward");
        const zero = zeros(go.shape, { dtype: go.dtype, device: go.device });
        if (lhs.requiresGrad) {
          lhs.accumulateGrad(reduceBroadcastGrad(whereOp(condTensor, go, zero), lhs.tensor.shape));
        }
        if (rhs.requiresGrad) {
          rhs.accumulateGrad(reduceBroadcastGrad(whereOp(condTensor, zero, go), rhs.tensor.shape));
        }
      },
    });
    return out;
  }

  /**
   * Transpose with the axes reversed (like `.T` in NumPy and PyTorch). One- and
   * zero-dimensional tensors are returned as a view of the same shape.
   */
  get T(): GradTensor {
    return this.transpose();
  }

  /**
   * Differentiable copy: the result owns its own buffer and the gradient flows
   * through unchanged (like `torch.clone`).
   *
   * @example
   * ```ts
   * const x = parameter([1, 2, 3]);
   * const y = x.clone();
   * y.mul(y).sum().backward();
   * x.grad?.toArray(); // [2, 4, 6]
   * ```
   */
  clone(): GradTensor {
    return unaryNode(this, ownTensor(this.tensor), "clone", (go) => go);
  }

  /**
   * Value of a single-element tensor as a JS number (or bigint for `int64`), like
   * `torch.Tensor.item()`. Use it to read a scalar loss.
   *
   * @throws {ShapeError} When the tensor has more than one element.
   *
   * @example
   * ```ts
   * const loss = parameter([1, 2, 3]).sum();
   * console.log(loss.item()); // 6
   * ```
   */
  item(): number | bigint {
    return this.tensor.item() as number | bigint;
  }

  /**
   * Remove axes of size 1 (functional form: `squeeze`). A differentiable view: the gradient
   * is reshaped back.
   *
   * @param axis - Axis or axes to remove (default: all size-1 axes)
   *
   * @example
   * ```ts
   * parameter([[1, 2]]).squeeze().shape; // [2]
   * ```
   */
  squeeze(axis?: Axis | readonly Axis[]): GradTensor {
    const x = this.tensor;
    return unaryNode(this, squeezeOp(x, axis), "squeeze", (go) => go.reshape(x.shape));
  }

  /**
   * Insert an axis of size 1 (functional form: `unsqueeze`). A differentiable view.
   *
   * @param axis - Position of the new axis; negative values count from the end
   *
   * @example
   * ```ts
   * parameter([1, 2]).unsqueeze(0).shape; // [1, 2]
   * ```
   */
  unsqueeze(axis: number): GradTensor {
    const x = this.tensor;
    return unaryNode(this, unsqueezeOp(x, axis), "unsqueeze", (go) => go.reshape(x.shape));
  }

  /**
   * Index of the maximum (functional form: `argmax`). Not differentiable, so the result is a
   * plain `int32` `Tensor`, as in PyTorch.
   *
   * @param axis - Axis to search (default: over the flattened tensor)
   * @param keepdims - Keep the reduced axis with size 1
   */
  argmax(axis?: Axis, keepdims = false): Tensor {
    return argmaxOp(this.tensor, axis, keepdims);
  }

  /**
   * Index of the minimum (functional form: `argmin`). Not differentiable, so the result is a
   * plain `int32` `Tensor`.
   *
   * @param axis - Axis to search (default: over the flattened tensor)
   * @param keepdims - Keep the reduced axis with size 1
   */
  argmin(axis?: Axis, keepdims = false): Tensor {
    return argminOp(this.tensor, axis, keepdims);
  }

  /**
   * Element-wise equality, giving a plain `bool` `Tensor` (functional form: `equal`).
   *
   * @param other - `GradTensor`, `Tensor` (broadcast) or number
   */
  eq(other: number | Tensor | GradTensor): Tensor {
    return equal(this.tensor, rawOperand(this.tensor, other, "eq"));
  }

  /**
   * Element-wise inequality, giving a plain `bool` `Tensor` (functional form: `notEqual`).
   *
   * @param other - `GradTensor`, `Tensor` (broadcast) or number
   */
  ne(other: number | Tensor | GradTensor): Tensor {
    return notEqual(this.tensor, rawOperand(this.tensor, other, "ne"));
  }

  /**
   * Element-wise greater-than, giving a plain `bool` `Tensor` (functional form: `greater`).
   *
   * @param other - `GradTensor`, `Tensor` (broadcast) or number
   */
  gt(other: number | Tensor | GradTensor): Tensor {
    return greater(this.tensor, rawOperand(this.tensor, other, "gt"));
  }

  /**
   * Element-wise greater-or-equal, giving a plain `bool` `Tensor` (functional form: `greaterEqual`).
   *
   * @param other - `GradTensor`, `Tensor` (broadcast) or number
   */
  ge(other: number | Tensor | GradTensor): Tensor {
    return greaterEqual(this.tensor, rawOperand(this.tensor, other, "ge"));
  }

  /**
   * Element-wise less-than, giving a plain `bool` `Tensor` (functional form: `less`).
   *
   * @param other - `GradTensor`, `Tensor` (broadcast) or number
   */
  lt(other: number | Tensor | GradTensor): Tensor {
    return less(this.tensor, rawOperand(this.tensor, other, "lt"));
  }

  /**
   * Element-wise less-or-equal, giving a plain `bool` `Tensor` (functional form: `lessEqual`).
   *
   * @param other - `GradTensor`, `Tensor` (broadcast) or number
   */
  le(other: number | Tensor | GradTensor): Tensor {
    return lessEqual(this.tensor, rawOperand(this.tensor, other, "le"));
  }

  /** Element-wise NaN test, giving a plain `bool` `Tensor` (functional form: `isnan`). */
  isnan(): Tensor {
    return isnan(this.tensor);
  }

  /**
   * True where any element is non-zero, as a plain `bool` `Tensor` (functional form: `any`).
   *
   * @param axis - Axis or axes to reduce (default: all)
   * @param keepdims - Keep reduced axes with size 1
   */
  any(axis?: Axis | readonly Axis[], keepdims = false): Tensor {
    return anyOp(this.tensor, axisArg(axis), keepdims);
  }

  /**
   * True where all elements are non-zero, as a plain `bool` `Tensor` (functional form: `all`).
   *
   * @param axis - Axis or axes to reduce (default: all)
   * @param keepdims - Keep reduced axes with size 1
   */
  all(axis?: Axis | readonly Axis[], keepdims = false): Tensor {
    return allOp(this.tensor, axisArg(axis), keepdims);
  }

  /**
   * Round down (functional form: `floor`). The gradient is zero everywhere, as in PyTorch.
   *
   * @example
   * ```ts
   * const x = parameter([1.7, -1.2]);
   * x.floor().toArray(); // [1, -2]
   * ```
   */
  floor(): GradTensor {
    const x = this.tensor;
    const outTensor = keepFloatDtype(floorOp(x), x.dtype, "floor");
    return unaryNode(this, outTensor, "floor", () => zerosGradLike(x));
  }

  /**
   * Round up (functional form: `ceil`). The gradient is zero everywhere, as in PyTorch.
   *
   * @example
   * ```ts
   * const x = parameter([1.2, -1.7]);
   * x.ceil().toArray(); // [2, -1]
   * ```
   */
  ceil(): GradTensor {
    const x = this.tensor;
    const outTensor = keepFloatDtype(ceilOp(x), x.dtype, "ceil");
    return unaryNode(this, outTensor, "ceil", () => zerosGradLike(x));
  }

  /**
   * Round half to even (functional form: `round`). The gradient is zero everywhere, as in PyTorch.
   *
   * @param decimals - Digits after the decimal point (negative rounds to tens, hundreds, ...)
   *
   * @example
   * ```ts
   * const x = parameter([0.5, 1.5, 2.5]);
   * x.round().toArray(); // [0, 2, 2]
   * ```
   */
  round(decimals = 0): GradTensor {
    const x = this.tensor;
    const outTensor = keepFloatDtype(roundOp(x, decimals), x.dtype, "round");
    return unaryNode(this, outTensor, "round", () => zerosGradLike(x));
  }

  /**
   * Reverse the order of elements along axes (functional form: `flip`). Differentiable: the
   * gradient is flipped back.
   *
   * @param axes - Axis or axes to flip (default: all)
   *
   * @example
   * ```ts
   * const x = parameter([1, 2, 3]);
   * x.flip().toArray(); // [3, 2, 1]
   * ```
   */
  flip(axes?: Axis | readonly Axis[]): GradTensor {
    return unaryNode(this, flipOp(this.tensor, axes), "flip", (go) => flipOp(go, axes));
  }

  /**
   * Sorted values along an axis (functional form: `sort`), like `torch.sort(x).values`.
   * Differentiable: each upstream gradient goes back to the position its value came from.
   * Use {@link GradTensor.argsort} for the indices.
   *
   * @param axis - Axis to sort along (default -1)
   * @param descending - Sort from largest to smallest
   *
   * @example
   * ```ts
   * const x = parameter([3, 1, 2]);
   * x.sort().toArray(); // [1, 2, 3]
   * ```
   */
  sort(axis: Axis | undefined = -1, descending = false): GradTensor {
    const x = this.tensor;
    const outTensor = sortOp(x, axis, descending);
    const requiresGrad = gradEnabled && this.requiresGrad;
    // Computed with the forward pass so a later in-place edit cannot change the routing.
    const order = requiresGrad ? argsortOp(x, axis, descending) : null;
    return unaryNode(this, outTensor, "sort", (go) => {
      if (order === null) throw new DeepboxError("Internal error: missing order for sort backward");
      const goDense = asFloat64Dense(go);
      const acc = new Float64Array(x.size);
      if (x.ndim === 0) {
        acc[0] = goDense[0] ?? 0;
      } else {
        const { outer, len, inner } = laneLayout(x.shape, normalizeAxis(axis ?? -1, x.ndim));
        const orderDense = asFloat64Dense(order);
        for (let o = 0; o < outer; o++) {
          for (let k = 0; k < len; k++) {
            for (let i = 0; i < inner; i++) {
              const pos = (o * len + k) * inner + i;
              const dst = (o * len + (orderDense[pos] as number)) * inner + i;
              acc[dst] = (acc[dst] as number) + (goDense[pos] as number);
            }
          }
        }
      }
      return asDtype(fromFloat64Dense(x.shape, x.device, acc), gradDtypeFor(x.dtype));
    });
  }

  /**
   * Indices that would sort the tensor along an axis (functional form: `argsort`). Not
   * differentiable, so the result is a plain `int32` `Tensor`.
   *
   * @param axis - Axis to sort along (default -1)
   * @param descending - Sort from largest to smallest
   */
  argsort(axis: Axis | undefined = -1, descending = false): Tensor {
    return argsortOp(this.tensor, axis, descending);
  }

  /**
   * Dot product, the same as {@link GradTensor.matmul}.
   *
   * @param other - Right operand
   */
  dot(other: Tensor | GradTensor): GradTensor {
    return this.matmul(other);
  }

  /**
   * Move the tensor, and its gradient if it has one, to another device (see `Tensor.to`).
   *
   * The result is a new leaf with the same `requiresGrad` flag; it is not connected to the
   * graph of this tensor. Returns `this` when the data is already on the target device.
   *
   * @param device - Target device
   * @throws {DeviceError} If the target device has no available backend
   * @throws {DTypeError} If a tensor of another dtype than float32, float16 or bfloat16 is moved to a kernel device
   *
   * @example
   * ```ts
   * const w = await parameter([1, 2, 3]).to("webgpu");
   * ```
   */
  async to(device: Device): Promise<GradTensor> {
    const grad = this._grad;
    const moved = await this.tensor.to(device);
    const movedGrad = grad === null ? null : await grad.to(device);
    if (moved === this.tensor && movedGrad === grad) return this;
    const out = GradTensor.fromTensor(moved, { requiresGrad: this.requiresGrad });
    out._grad = movedGrad;
    return out;
  }

  /** Move the tensor and its gradient to the CPU. Shorthand for `to("cpu")`. */
  cpu(): Promise<GradTensor> {
    return this.to("cpu");
  }

  /**
   * Release the device memory of the tensor and of its gradient (see `Tensor.dispose`).
   * Host tensors ignore this call.
   */
  dispose(): void {
    this.tensor.dispose();
    this._grad?.dispose();
  }

  /**
   * Return a human-readable string representation of this GradTensor.
   *
   * Delegates to the underlying {@link Tensor.toString} and appends
   * gradient metadata.
   *
   * @param maxElements - Maximum elements per dimension before summarizing (default: 6).
   * @returns Formatted string representation
   */
  toString(maxElements = 6): string {
    const base = this.tensor.toString(maxElements);
    const gradInfo = this.requiresGrad ? ", requiresGrad=true" : "";
    return base.replace(/\)$/, `${gradInfo})`);
  }
}

/**
 * Create a leaf GradTensor with `requiresGrad = true`. The flag also holds
 * when the tensor is created inside {@link noGrad} (PyTorch semantics); only
 * operations run inside `noGrad` skip recording.
 *
 * @param data - Nested numbers or an existing tensor.
 * @param options - Optional `dtype` used when `data` is not already a tensor.
 */
export function parameter(
  data: number | number[] | number[][] | number[][][] | Tensor,
  options: GradTensorOptions = {}
): GradTensor {
  const t =
    data instanceof TensorClass
      ? data
      : tensor(data, options.dtype ? { dtype: options.dtype } : undefined);
  return GradTensor.fromTensor(t, { ...options, requiresGrad: true });
}

/**
 * Run `fn` with gradient tracking disabled and return its result.
 *
 * Operations executed inside do not record a graph, so their results have
 * `requiresGrad` false. Leaf tensors created inside (for example with
 * {@link parameter}) keep the `requiresGrad` flag they were created with, as in
 * PyTorch. The previous state is restored afterwards, also when `fn` throws.
 *
 * **Important:** The callback must be synchronous. Passing an async function
 * will cause `gradEnabled` to be restored before the awaited work finishes,
 * silently breaking gradient tracking inside the async continuation.
 *
 * @throws {DeepboxError} If the callback returns a Promise or other thenable
 */
export function noGrad<T>(fn: () => T): T {
  const prev = gradEnabled;
  gradEnabled = false;
  try {
    const result = fn();
    if (
      result instanceof Promise ||
      (typeof result === "object" &&
        result !== null &&
        typeof (result as { then?: unknown }).then === "function")
    ) {
      throw new DeepboxError(
        "noGrad() does not support async callbacks. " +
          "The gradient state would be restored before the async work completes. " +
          "Wrap your async logic so that only the synchronous tensor operations are inside noGrad()."
      );
    }
    return result;
  } finally {
    gradEnabled = prev;
  }
}

/**
 * Differentiable im2col: unfold sliding kernel windows of an NCHW input into
 * columns. The backward pass is {@link col2imGrad}'s forward (col2im).
 */
export function im2col(
  input: GradTensor,
  kernelSize: [number, number],
  stride: [number, number],
  padding: [number, number]
): GradTensor {
  const outTensor = im2colOp(input.tensor, kernelSize, stride, padding);
  const requiresGrad = gradEnabled && input.requiresGrad;

  let result: GradTensor;

  const backward = () => {
    if (!requiresGrad) return;
    const go = result.grad;
    if (go === null) {
      throw new DeepboxError("Internal error: missing gradient for im2col backward");
    }
    // Backward of im2col is col2im
    const gradInput = col2im(go, input.shape, kernelSize, stride, padding);
    input.accumulateGrad(gradInput);
  };

  result = GradTensor.create({
    tensor: outTensor,
    requiresGrad,
    prev: requiresGrad ? [input] : [],
    backward,
  });

  return result;
}

/**
 * Column to Image operation for GradTensor (transpose/adjoint of im2col).
 * The backward of col2im is im2col.
 */
export function col2imGrad(
  cols: GradTensor,
  outputShape: readonly number[],
  kernelSize: [number, number],
  stride: [number, number],
  padding: [number, number]
): GradTensor {
  const outTensor = col2im(cols.tensor, outputShape, kernelSize, stride, padding);
  const requiresGrad = gradEnabled && cols.requiresGrad;
  let result: GradTensor;
  const backward = () => {
    if (!requiresGrad) return;
    const go = result.grad;
    if (go === null) {
      throw new DeepboxError("Internal error: missing gradient for col2im backward");
    }
    const gradCols = im2colOp(go, kernelSize, stride, padding);
    cols.accumulateGrad(gradCols);
  };
  result = GradTensor.create({
    tensor: outTensor,
    requiresGrad,
    prev: requiresGrad ? [cols] : [],
    backward,
  });
  return result;
}

/**
 * Wrap a precomputed output tensor with an explicit reverse-mode rule.
 *
 * Layers that compute their forward value with hand-written kernels (Conv3d,
 * ConvTranspose, pooling, Embedding, RNN cells, …) use this to attach a
 * correct backward instead of returning a detached leaf. Each entry in
 * `grads` is `[input, gradFn]`; `gradFn(outGrad)` returns that input's
 * gradient tensor (already reduced to the input's shape). Honors the global
 * no-grad context and only records the graph when some input requires grad.
 */
export function customOp(
  output: Tensor,
  grads: ReadonlyArray<readonly [GradTensor, (outGrad: Tensor) => Tensor]>
): GradTensor {
  const requiresGrad = gradEnabled && grads.some(([inp]) => inp.requiresGrad);
  let result: GradTensor;
  const backward = () => {
    if (!requiresGrad) return;
    const go = result.grad;
    if (go === null) {
      throw new DeepboxError("Internal error: missing gradient for customOp backward");
    }
    for (const [inp, gradFn] of grads) {
      if (inp.requiresGrad) inp.accumulateGrad(gradFn(go));
    }
  };
  result = GradTensor.create({
    tensor: output,
    requiresGrad,
    prev: requiresGrad ? grads.filter(([inp]) => inp.requiresGrad).map(([inp]) => inp) : [],
    backward,
  });
  return result;
}

/**
 * Stack a list of same-shape GradTensors along a new leading axis.
 * Backward splits the upstream gradient back to each input. Used by the
 * recurrent layers to assemble per-timestep hidden states differentiably.
 *
 * @param parts - Tensors to stack; all must have the same shape. The result
 *   takes the dtype and device of the first one.
 * @throws {DeepboxError} If `parts` is empty.
 * @throws {ShapeError} If the shapes differ.
 */
export function stackGrad(parts: readonly GradTensor[]): GradTensor {
  const first = parts[0];
  if (first === undefined) {
    throw new DeepboxError("stackGrad requires at least one tensor");
  }
  const partShape = first.tensor.shape;
  for (const part of parts) {
    if (!shapesEqual(part.tensor.shape, partShape)) {
      throw ShapeError.mismatch(partShape, part.tensor.shape, "stackGrad");
    }
  }
  // Preserve the parts' dtype so downstream ops (e.g. the next recurrent
  // layer's float32 weights) don't hit a dtype mismatch. Stacking works on the
  // typed storage directly, so int64 values stay exact.
  const outDtype = ensureNumericDType(first.tensor.dtype, "stackGrad");
  const outTensor = stackOp(
    parts.map((p) => castTensor(p.tensor, outDtype)),
    0
  );
  const requiresGrad = gradEnabled && parts.some((p) => p.requiresGrad);
  let result: GradTensor;
  const backward = () => {
    if (!requiresGrad) return;
    const go = result.grad;
    if (go === null) throw new DeepboxError("Internal error: missing gradient for stack backward");
    for (let p = 0; p < parts.length; p++) {
      const part = parts[p] as GradTensor;
      if (!part.requiresGrad) continue;
      const partDtype = ensureNumericDType(part.tensor.dtype, "stackGrad");
      part.accumulateGrad(castTensor(slice(go, p), partDtype));
    }
  };
  result = GradTensor.create({
    tensor: outTensor,
    requiresGrad,
    prev: requiresGrad ? parts.filter((p) => p.requiresGrad) : [],
    backward,
  });
  return result;
}

/**
 * Concatenate GradTensors along an existing axis. Backward slices the
 * gradient back to each input's segment.
 *
 * @param parts - Tensors to join; shapes must agree on every axis but `axis`.
 * @param axis - Axis to join along; negative values count from the end.
 * @throws {DeepboxError} If `parts` is empty.
 */
export function concatGrad(parts: readonly GradTensor[], axis = 0): GradTensor {
  if (parts.length === 0) throw new DeepboxError("concatGrad requires at least one tensor");
  const outTensor = concatOp(
    parts.map((p) => p.tensor),
    axis
  );
  const requiresGrad = gradEnabled && parts.some((p) => p.requiresGrad);
  let result: GradTensor;
  const ax = axis < 0 ? axis + parts[0]!.tensor.ndim : axis;
  const backward = () => {
    if (!requiresGrad) return;
    const go = result.grad;
    if (go === null) throw new DeepboxError("Internal error: missing gradient for concat backward");
    let start = 0;
    for (const part of parts) {
      const len = part.tensor.shape[ax] ?? 0;
      if (part.requiresGrad) {
        const ranges = part.tensor.shape.map((_, d) =>
          d === ax ? { start, end: start + len } : {}
        );
        part.accumulateGrad(go.slice(...ranges));
      }
      start += len;
    }
  };
  result = GradTensor.create({
    tensor: outTensor,
    requiresGrad,
    prev: requiresGrad ? parts.filter((p) => p.requiresGrad) : [],
    backward,
  });
  return result;
}

/** Zero gradient for a non-differentiable rounding op on `x`. */
function zerosGradLike(x: Tensor): Tensor {
  return zeros(x.shape, { dtype: gradDtypeFor(x.dtype), device: x.device });
}

/**
 * `base ** exponent` with a tensor exponent. The base gradient is
 * `y * x^(y-1)` (0 where `y == 0`); the exponent gradient is `out * log(x)`
 * (0 where `x == 0` and `y >= 0`), as in PyTorch.
 */
function powByTensor(base: GradTensor, exponent: GradTensor): GradTensor {
  const needsGrad = gradEnabled && (base.requiresGrad || exponent.requiresGrad);
  const outTensor = pow(base.tensor, exponent.tensor);
  const out: GradTensor = new GradTensor({
    tensor: outTensor,
    requiresGrad: needsGrad,
    prev: needsGrad ? [base, exponent] : [],
    backward: () => {
      if (!needsGrad) return;
      const go = out.grad;
      if (go === null) throw new DeepboxError("Internal error: missing gradient for pow backward");
      const x = asF64(base.tensor);
      const y = asF64(exponent.tensor);
      const goF = asF64(go);
      const zero = f64Const(0);
      if (base.requiresGrad) {
        const slope = whereOp(equal(y, zero), zero, mul(y, pow(x, sub(y, f64Const(1)))));
        const g = reduceBroadcastGrad(mul(goF, slope), base.tensor.shape);
        base.accumulateGrad(asDtype(g, gradDtypeFor(base.dtype)));
      }
      if (exponent.requiresGrad) {
        const flat = logicalAnd(equal(x, zero), greaterEqual(y, zero));
        const slope = whereOp(flat, zero, mul(asF64(outTensor), log(x)));
        const g = reduceBroadcastGrad(mul(goF, slope), exponent.tensor.shape);
        exponent.accumulateGrad(asDtype(g, gradDtypeFor(exponent.dtype)));
      }
    },
  });
  return out;
}

/**
 * Shared implementation of GradTensor.maximum / minimum. Where the operands are
 * equal each gets half of the gradient; a NaN operand sends the gradient to both
 * (PyTorch's rule).
 */
function pairExtremum(a: GradTensor, b: GradTensor, isMax: boolean): GradTensor {
  const name = isMax ? "maximum" : "minimum";
  const outTensor = (isMax ? maximumOp : minimumOp)(a.tensor, b.tensor);
  const requiresGrad = gradEnabled && (a.requiresGrad || b.requiresGrad);
  const out: GradTensor = new GradTensor({
    tensor: outTensor,
    requiresGrad,
    prev: requiresGrad ? [a, b] : [],
    backward: () => {
      if (!requiresGrad) return;
      const go = out.grad;
      if (go === null)
        throw new DeepboxError(`Internal error: missing gradient for ${name} backward`);
      const x = asF64(a.tensor);
      const y = asF64(b.tensor);
      const one = f64Const(1);
      const below = asF64(less(x, y));
      const above = asF64(greater(x, y));
      const half = sub(one, mulScalar(asF64(equal(x, y)), 0.5));
      const wa = mul(half, sub(one, isMax ? below : above));
      const wb = mul(half, sub(one, isMax ? above : below));
      const goF = asF64(go);
      if (a.requiresGrad) {
        const g = asDtype(reduceBroadcastGrad(mul(goF, wa), a.tensor.shape), gradDtypeFor(a.dtype));
        a.accumulateGrad(g);
      }
      if (b.requiresGrad) {
        const g = asDtype(reduceBroadcastGrad(mul(goF, wb), b.tensor.shape), gradDtypeFor(b.dtype));
        b.accumulateGrad(g);
      }
    },
  });
  return out;
}

/** Sizes of the dimensions before, at and after `axis`, for lane-wise kernels. */
function laneLayout(shape: Shape, ax: number): { outer: number; len: number; inner: number } {
  let outer = 1;
  for (let d = 0; d < ax; d++) outer *= shape[d] ?? 1;
  let inner = 1;
  for (let d = ax + 1; d < shape.length; d++) inner *= shape[d] ?? 1;
  return { outer, len: shape[ax] ?? 1, inner };
}

/**
 * Whether the fused host kernels can handle `input`: a float32 or float64 host
 * tensor with at least one dimension. Everything else takes the composite path
 * built from differentiable primitives.
 */
function canFuseSoftmax(input: GradTensor): boolean {
  const t = input.tensor;
  return !t.isDeviceTensor && t.ndim > 0 && (t.dtype === "float32" || t.dtype === "float64");
}

/**
 * Fused softmax / log-softmax for host tensors. The forward pass makes one
 * sweep per lane (max, sum of exponentials, normalize) in float64 and stores
 * the result in the input dtype; the backward pass is the closed form
 * `softmax: y * (g - sum(g * y))` and `log-softmax: g - exp(y) * sum(g)`.
 * A lane that holds NaN, or whose maximum is infinite, gives NaN like the
 * composite formulation.
 */
function fusedSoftmax(input: GradTensor, axis: Axis, logSpace: boolean): GradTensor {
  const t = input.tensor;
  const ax = normalizeAxis(axis, t.ndim);
  const { outer, len, inner } = laneLayout(t.shape, ax);
  const dtype = ensureNumericDType(t.dtype, logSpace ? "logSoftmax" : "softmax");
  const src = readNumbers(t, logSpace ? "logSoftmax" : "softmax");
  const out = new (dtypeToTypedArrayCtor(dtype))(t.size) as Float32Array | Float64Array;

  for (let o = 0; o < outer; o++) {
    for (let k = 0; k < inner; k++) {
      const base = o * len * inner + k;
      let maxVal = Number.NEGATIVE_INFINITY;
      let hasNaN = false;
      for (let j = 0; j < len; j++) {
        const v = src[base + j * inner] as number;
        if (v > maxVal) maxVal = v;
        else if (Number.isNaN(v)) hasNaN = true;
      }
      if (hasNaN || !Number.isFinite(maxVal)) {
        for (let j = 0; j < len; j++) out[base + j * inner] = Number.NaN;
        continue;
      }
      let total = 0;
      for (let j = 0; j < len; j++) total += Math.exp((src[base + j * inner] as number) - maxVal);
      if (logSpace) {
        const lse = maxVal + Math.log(total);
        for (let j = 0; j < len; j++) {
          out[base + j * inner] = (src[base + j * inner] as number) - lse;
        }
      } else {
        for (let j = 0; j < len; j++) {
          out[base + j * inner] = Math.exp((src[base + j * inner] as number) - maxVal) / total;
        }
      }
    }
  }

  const outTensor = TensorClass.fromTypedArray({
    data: out,
    shape: t.shape,
    dtype,
    device: t.device,
  });
  const requiresGrad = gradEnabled && input.requiresGrad;
  let result: GradTensor;
  const backward = () => {
    if (!requiresGrad) return;
    const go = result.grad;
    if (go === null) {
      throw new DeepboxError(
        `Internal error: missing gradient for ${logSpace ? "logSoftmax" : "softmax"} backward`
      );
    }
    const g = readNumbers(go, "softmax backward");
    const gin = new (dtypeToTypedArrayCtor(dtype))(t.size) as Float32Array | Float64Array;
    for (let o = 0; o < outer; o++) {
      for (let k = 0; k < inner; k++) {
        const base = o * len * inner + k;
        let acc = 0;
        if (logSpace) {
          for (let j = 0; j < len; j++) acc += g[base + j * inner] as number;
          for (let j = 0; j < len; j++) {
            const idx = base + j * inner;
            gin[idx] = (g[idx] as number) - Math.exp(out[idx] as number) * acc;
          }
        } else {
          for (let j = 0; j < len; j++) {
            acc += (g[base + j * inner] as number) * (out[base + j * inner] as number);
          }
          for (let j = 0; j < len; j++) {
            const idx = base + j * inner;
            gin[idx] = (out[idx] as number) * ((g[idx] as number) - acc);
          }
        }
      }
    }
    input.accumulateGrad(
      TensorClass.fromTypedArray({ data: gin, shape: t.shape, dtype, device: t.device })
    );
  };
  result = GradTensor.create({
    tensor: outTensor,
    requiresGrad,
    prev: requiresGrad ? [input] : [],
    backward,
  });
  return result;
}

/**
 * Differentiable softmax along `axis`.
 *
 * Host float32 and float64 tensors use a fused kernel (one sweep per lane, in
 * float64, with the closed-form gradient). Other inputs, such as device tensors,
 * are computed as `exp(x - max) / sum(exp(x - max))` from differentiable
 * primitives with the maximum treated as a constant (softmax is shift-invariant),
 * so large inputs do not overflow either way.
 *
 * @param source - Logits. Integer and bool inputs are converted to float32.
 * @param axis - Axis to normalize over (default: last).
 */
export function softmax(source: GradTensor, axis: Axis = -1): GradTensor {
  // Integer and bool logits become float32 (the exponentials are not integers).
  const floated = isFloatDType(source.dtype) ? source : source.astype("float32");
  // A 0-d input is a single lane: softmax over a length-1 axis, like PyTorch.
  if (floated.ndim === 0) {
    normalizeAxis(axis, 1);
    return softmax(floated.reshape([1]), 0).reshape([]);
  }
  const input = floated;
  if (canFuseSoftmax(input)) return fusedSoftmax(input, axis, false);
  const maxVal = max(input.tensor, axis, true);
  const maxT = GradTensor.fromTensor(
    castTensor(maxVal, ensureNumericDType(input.dtype, "softmax")),
    {
      requiresGrad: false,
    }
  );
  const shifted = input.sub(maxT);
  const expT = shifted.exp();
  const sumT = expT.sum(axis, true);
  return expT.div(sumT);
}

/**
 * Differentiable log-softmax along `axis`, computed as
 * `x - max - log(sum(exp(x - max)))` (log-sum-exp form, stable for large inputs).
 *
 * Host float32 and float64 tensors use a fused kernel (one sweep per lane, in
 * float64, with the closed-form gradient `g - exp(y) * sum(g)`); other inputs
 * use the composite formulation from differentiable primitives.
 *
 * @param source - Logits. Integer and bool inputs are converted to float32.
 * @param axis - Axis to normalize over (default: last).
 */
export function logSoftmax(source: GradTensor, axis: Axis = -1): GradTensor {
  // Integer and bool logits become float32 (the logarithms are not integers).
  const floated = isFloatDType(source.dtype) ? source : source.astype("float32");
  // A 0-d input is a single lane: log-softmax over a length-1 axis, like PyTorch.
  if (floated.ndim === 0) {
    normalizeAxis(axis, 1);
    return logSoftmax(floated.reshape([1]), 0).reshape([]);
  }
  const input = floated;
  if (canFuseSoftmax(input)) return fusedSoftmax(input, axis, true);
  const maxVal = max(input.tensor, axis, true);
  const maxT = GradTensor.fromTensor(
    castTensor(maxVal, ensureNumericDType(input.dtype, "logSoftmax")),
    {
      requiresGrad: false,
    }
  );
  const shifted = input.sub(maxT);
  const expT = shifted.exp();
  const sumT = expT.sum(axis, true);
  const logSumExp = sumT.log();
  return shifted.sub(logSumExp);
}

/**
 * Differentiable variance.
 *
 * @param input - Input tensor.
 * @param axis - Axis to reduce; when omitted, all elements are reduced.
 * @param correction - Delta degrees of freedom: the divisor is `n - correction`
 *   (default 1, the unbiased estimator). The divisor is clamped at 0, so
 *   `n <= correction` yields NaN or Infinity instead of a negative variance.
 */
export function variance(input: GradTensor, axis?: Axis, correction = 1): GradTensor {
  const meanVal = input.mean(axis, true);
  const centered = input.sub(meanVal);
  const sq = centered.square();
  const sumSq = sq.sum(axis, false); // Reduce dims

  const n = axis === undefined ? input.size : (input.shape[normalizeAxis(axis, input.ndim)] ?? 1);
  const denom = Math.max(0, n - correction);
  const denomT = GradTensor.scalar(denom, {
    dtype: ensureNumericDType(sumSq.dtype, "variance"),
  });

  return sumSq.div(denomT);
}

/**
 * Inverted dropout: during training each element is zeroed with probability
 * `p` and the survivors are scaled by `1 / (1 - p)`, so the expected value is
 * unchanged. Returns `input` itself when `training` is false or `p` is 0.
 * Integer and bool inputs are converted to float32 first, because the scale is
 * not an integer in general.
 *
 * @param input - Input tensor.
 * @param p - Drop probability in `[0, 1)` (default 0.5).
 * @param training - Apply dropout only when true (default true).
 * @throws {InvalidParameterError} If `p` is outside `[0, 1)`.
 */
export function dropout(input: GradTensor, p = 0.5, training = true): GradTensor {
  if (!Number.isFinite(p) || p < 0 || p >= 1) {
    throw new InvalidParameterError("p must be in [0, 1)", "p", p);
  }
  if (!training || p === 0) return input;

  // The survivors are scaled by 1 / (1 - p), which an integer or bool tensor
  // cannot hold (the scale would be truncated), so those inputs become float32.
  const source = isFloatDType(input.dtype) ? input : input.astype("float32");
  const dtype = ensureNumericDType(source.dtype, "dropout");
  const scale = 1 / (1 - p);
  const mask = dropoutMask(source.shape, p, scale, dtype, source.device);
  const maskGrad = GradTensor.fromTensor(mask, { requiresGrad: false });
  return source.mul(maskGrad);
}
