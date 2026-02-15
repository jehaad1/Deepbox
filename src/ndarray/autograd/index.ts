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
 * `noGrad` only accepts **synchronous** callbacks — passing an async
 * function will throw, because the flag would be restored before the
 * async work completes.
 *
 * ## max / min backward — tie-breaking
 *
 * When multiple elements share the maximum (or minimum) value along the
 * reduced axis, **all** tied positions receive gradient.  This means the
 * gradient is *not* divided among ties — each tied element gets the full
 * upstream gradient.  This matches Deepbox's behaviour and avoids the
 * cost of counting ties, but callers should be aware that the
 * "effective" gradient magnitude is multiplied by the tie count.
 */

import type { Axis, DType, Shape, TypedArray } from "../../core";
import {
  DataValidationError,
  DeepboxError,
  DTypeError,
  getBigIntElement,
  getNumericElement,
  InvalidParameterError,
  normalizeAxis,
  ShapeError,
  shapeToSize,
} from "../../core";
import { dot } from "../linalg";
import { elu, gelu, leakyRelu, relu, sigmoid } from "../ops/activation";
import { abs as absOp, add, clip as clipOp, div, mul, neg, pow, sub } from "../ops/arithmetic";
import { equal, greater, less } from "../ops/comparison";
import { col2im, im2col as im2colOp } from "../ops/conv";
import { exp, log } from "../ops/math";
import { dropoutMask } from "../ops/random";
import { max, min, sum } from "../ops/reduction";
import { tanh } from "../ops/trigonometry";
import { type NestedArray, tensor, zeros } from "../tensor/creation";
import { gather, type SliceRange, slice } from "../tensor/indexing";
import { reshape, transpose } from "../tensor/shape";
import { isContiguous, offsetFromFlatIndex } from "../tensor/strides";
import {
  computeStrides,
  dtypeToTypedArrayCtor,
  type Tensor,
  Tensor as TensorClass,
} from "../tensor/Tensor";

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
 * inputs.  Toggled by {@link noGrad}.
 *
 * **Thread-safety note**: because JavaScript is single-threaded this
 * global flag is safe in synchronous code, but it must **never** be
 * relied upon across async boundaries — hence `noGrad` rejects async
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

function toNumericNestedArray(value: unknown): NestedArray {
  if (typeof value === "number") {
    return value;
  }
  if (typeof value === "bigint") {
    const asNumber = Number(value);
    if (!Number.isSafeInteger(asNumber)) {
      throw new DataValidationError("int64 value is too large to safely convert to number");
    }
    return asNumber;
  }
  if (Array.isArray(value)) {
    return value.map(toNumericNestedArray);
  }
  throw new DTypeError("Expected numeric tensor data");
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
  if (t.dtype === dtype) return t;
  if (t.dtype === "string") {
    throw new DTypeError("autograd does not support string dtype");
  }
  const Ctor = dtypeToTypedArrayCtor(dtype);
  const out = new Ctor(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);
  const toBool = dtype === "bool";

  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("autograd does not support string dtype");
  }

  if (out instanceof BigInt64Array) {
    if (data instanceof BigInt64Array) {
      for (let i = 0; i < t.size; i++) {
        const offset = contiguous
          ? t.offset + i
          : offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
        out[i] = getBigIntElement(data, offset);
      }
    } else {
      for (let i = 0; i < t.size; i++) {
        const offset = contiguous
          ? t.offset + i
          : offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
        out[i] = BigInt(Math.trunc(getNumericElement(data, offset)));
      }
    }
  } else {
    // out is Numeric TypedArray
    if (data instanceof BigInt64Array) {
      for (let i = 0; i < t.size; i++) {
        const offset = contiguous
          ? t.offset + i
          : offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
        const value = getBigIntElement(data, offset);
        out[i] = toBool ? (value !== 0n ? 1 : 0) : Number(value);
      }
    } else {
      for (let i = 0; i < t.size; i++) {
        const offset = contiguous
          ? t.offset + i
          : offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
        const value = getNumericElement(data, offset);
        out[i] = toBool ? (value !== 0 ? 1 : 0) : value;
      }
    }
  }

  return TensorClass.fromTypedArray({
    data: out,
    shape: t.shape,
    dtype,
    device: t.device,
  });
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
      const t = tensor(dataOrArgs as number | number[] | number[][] | number[][][]);
      this.tensor = t;
      this.requiresGrad = (options?.requiresGrad ?? false) && gradEnabled;
      this._prev = [];
      this._backward = () => {};
    }
    this._grad = null;
  }

  static create(args: {
    readonly tensor: Tensor;
    readonly requiresGrad: boolean;
    readonly prev: readonly GradTensor[];
    readonly backward: BackwardFn;
  }): GradTensor {
    return new GradTensor(args);
  }

  static fromTensor(t: Tensor, options: GradTensorOptions = {}): GradTensor {
    if (t.dtype === "string") {
      throw new DTypeError("autograd does not support string dtype");
    }
    if (options.dtype !== undefined && options.dtype !== t.dtype) {
      throw new DTypeError(
        `GradTensor dtype mismatch: expected ${options.dtype}, received ${t.dtype}`
      );
    }
    const requiresGrad = (options.requiresGrad ?? false) && gradEnabled;
    return new GradTensor({
      tensor: t,
      requiresGrad,
      prev: [],
      backward: () => {},
    });
  }

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
   * Get the device where the tensor resides.
   * Implements TensorLike interface for compatibility with Tensor.
   */
  get device(): Tensor["device"] {
    return this.tensor.device;
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

  setGrad(grad: Tensor): void {
    if (!this.requiresGrad) {
      throw new InvalidParameterError(
        "Cannot set gradient on tensor with requiresGrad=false",
        "requiresGrad",
        this.requiresGrad
      );
    }
    ensureSameSize(this.tensor, grad, "setGrad");
    this._grad = grad;
  }

  zeroGrad(): void {
    if (!this.requiresGrad) return;
    this._grad = zeros(this.tensor.shape, {
      dtype: this.tensor.dtype,
      device: this.tensor.device,
    });
  }

  detach(): GradTensor {
    return GradTensor.fromTensor(this.tensor, { requiresGrad: false });
  }

  setRequiresGrad(value: boolean): void {
    this.requiresGrad = value && gradEnabled;
    if (!this.requiresGrad) {
      this._grad = null;
    }
  }

  hasGrad(): boolean {
    return this._grad !== null;
  }

  /** @internal */
  accumulateGrad(grad: Tensor): void {
    if (!this.requiresGrad) return;

    const normalizedGrad = toContiguous(grad);

    if (this._grad === null) {
      this._grad = normalizedGrad;
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

    const seedGrad = grad ?? onesLike(this.tensor);
    this._grad = seedGrad;

    const topo: GradTensor[] = [];
    const visited = new Set<GradTensor>();

    const build = (v: GradTensor): void => {
      if (visited.has(v)) return;
      visited.add(v);
      for (const child of v._prev) build(child);
      topo.push(v);
    };

    build(this);
    topo.reverse();

    for (const v of topo) {
      v._backward();
    }
  }

  add(other: GradTensor): GradTensor {
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

  sub(other: GradTensor): GradTensor {
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

  mul(other: GradTensor): GradTensor {
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

  sum(axis?: Axis, keepdims = false): GradTensor {
    let outTensor = sum(this.tensor, axis, keepdims);
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
        if (axis === undefined) {
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

        // If keepdims=false, output shape is input shape with the reduced axis removed.
        // If keepdims=true, output shape matches input ndim with that axis set to 1.
        const outShape = go.shape;

        const inDense = new Float64Array(this.tensor.size);
        const inLogicalStrides = computeStrides(this.tensor.shape);
        const outLogicalStrides = computeStrides(outShape);
        const goContiguous = isContiguous(go.shape, go.strides);

        for (let inFlat = 0; inFlat < this.tensor.size; inFlat++) {
          // Convert input flat -> input coordinates
          let rem = inFlat;
          const inCoord = new Array<number>(this.tensor.ndim);
          for (let d = 0; d < this.tensor.ndim; d++) {
            const s = inLogicalStrides[d] ?? 1;
            const c = Math.floor(rem / s);
            rem -= c * s;
            inCoord[d] = c;
          }

          // Map to output coordinates by removing or zeroing the reduced axis
          const outCoord: number[] = [];
          for (let d = 0; d < this.tensor.ndim; d++) {
            if (d === ax) {
              if (keepdims) outCoord.push(0);
              continue;
            }
            outCoord.push(inCoord[d] ?? 0);
          }

          // Convert output coordinates -> output flat
          let outFlat = 0;
          for (let d = 0; d < outCoord.length; d++) {
            outFlat += (outCoord[d] ?? 0) * (outLogicalStrides[d] ?? 1);
          }

          const goOffset = goContiguous
            ? go.offset + outFlat
            : offsetFromFlatIndex(outFlat, outLogicalStrides, go.strides, go.offset);
          const goDataBuf = go.data;
          if (Array.isArray(goDataBuf)) {
            throw new DTypeError("autograd does not support string dtype");
          }
          inDense[inFlat] =
            goDataBuf instanceof BigInt64Array
              ? Number(getBigIntElement(goDataBuf, goOffset))
              : getNumericElement(goDataBuf, goOffset);
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

  div(other: GradTensor): GradTensor {
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
        const numer =
          this.tensor.dtype === gradDtype ? this.tensor : castTensor(this.tensor, gradDtype);
        const denom =
          other.tensor.dtype === gradDtype ? other.tensor : castTensor(other.tensor, gradDtype);
        if (this.requiresGrad) {
          const grad = reduceBroadcastGrad(div(go, denom), this.tensor.shape);
          this.accumulateGrad(grad);
        }
        if (other.requiresGrad) {
          const grad = reduceBroadcastGrad(
            mul(div(go, mul(denom, denom)), neg(numer)),
            other.tensor.shape
          );
          other.accumulateGrad(grad);
        }
      },
    });

    return out;
  }

  pow(exponent: number): GradTensor {
    if (this.tensor.data instanceof BigInt64Array) {
      throw new DTypeError(
        "pow() backward is not supported for int64 tensors. " +
          "Cast to float32/float64 before calling pow() if gradients are needed."
      );
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
        const powMinusOne = pow(this.tensor, tensor(exponent - 1, { dtype: this.tensor.dtype }));
        const grad = mul(mul(go, tensor(exponent, { dtype: outTensor.dtype })), powMinusOne);
        this.accumulateGrad(grad);
      },
    });

    return out;
  }

  sqrt(): GradTensor {
    return this.pow(0.5);
  }

  matmul(other: GradTensor): GradTensor {
    const outTensor = dot(this.tensor, other.tensor);
    const requiresGrad = gradEnabled && (this.requiresGrad || other.requiresGrad);
    const leftBatchRank = Math.max(0, this.tensor.ndim - 2);
    const rightBatchRank = Math.max(0, other.tensor.ndim - 2);
    const leftBroadcasted = leftBatchRank === 0 && rightBatchRank > 0;
    const rightBroadcasted = rightBatchRank === 0 && leftBatchRank > 0;
    const leftDtype = ensureNumericDType(this.tensor.dtype, "matmul");
    const rightDtype = ensureNumericDType(other.tensor.dtype, "matmul");

    const swapLastTwo = (t: Tensor): Tensor => {
      if (t.ndim < 2) return t;
      const axes: number[] = [];
      for (let i = 0; i < t.ndim; i++) {
        axes.push(i);
      }
      const last = t.ndim - 1;
      const secondLast = t.ndim - 2;
      const tmp = axes[last];
      axes[last] = axes[secondLast] ?? last;
      axes[secondLast] = tmp ?? secondLast;
      return transpose(t, axes);
    };

    const reduceBatchDims = (t: Tensor, batchRank: number, targetDtype: NumericDType): Tensor => {
      let out = t;
      for (let i = 0; i < batchRank; i++) {
        out = sum(out, 0);
      }
      return castTensor(out, targetDtype);
    };

    const out = new GradTensor({
      tensor: outTensor,
      requiresGrad,
      prev: requiresGrad ? [this, other] : [],
      backward: () => {
        if (!requiresGrad) return;
        const go = out._grad;
        if (go === null) {
          throw new DeepboxError("Internal error: missing gradient for matmul backward");
        }
        if (this.requiresGrad) {
          let grad = dot(go, swapLastTwo(other.tensor));
          if (leftBroadcasted) {
            grad = reduceBatchDims(grad, rightBatchRank, leftDtype);
          } else {
            grad = castTensor(grad, leftDtype);
          }
          this.accumulateGrad(grad);
        }
        if (other.requiresGrad) {
          let grad = dot(swapLastTwo(this.tensor), go);
          if (rightBroadcasted) {
            grad = reduceBatchDims(grad, leftBatchRank, rightDtype);
          } else {
            grad = castTensor(grad, rightDtype);
          }
          other.accumulateGrad(grad);
        }
      },
    });

    return out;
  }

  relu(): GradTensor {
    if (this.tensor.data instanceof BigInt64Array) {
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

  square(): GradTensor {
    return this.pow(2);
  }

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

  tanh(): GradTensor {
    const outTensor = tanh(this.tensor);
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

  slice(...args: SliceRange[]): GradTensor {
    const outTensor = slice(this.tensor, ...args);
    const requiresGrad = gradEnabled && this.requiresGrad;

    // Pre-compute normalized ranges for backward scatter
    const inputShape = this.tensor.shape;
    const ndim = this.tensor.ndim;
    type NR = { start: number; end: number; step: number; isIndex: boolean };
    const nrList: NR[] = [];
    for (let axis = 0; axis < ndim; axis++) {
      const dim = inputShape[axis] ?? 0;
      const r = args[axis];
      if (r === undefined) {
        nrList.push({ start: 0, end: dim, step: 1, isIndex: false });
      } else if (typeof r === "number") {
        const idx = r < 0 ? dim + r : r;
        nrList.push({ start: idx, end: idx + 1, step: 1, isIndex: true });
      } else {
        const step = r.step ?? 1;
        if (step > 0) {
          const s = Math.min(
            Math.max((r.start ?? 0) < 0 ? dim + (r.start ?? 0) : (r.start ?? 0), 0),
            dim
          );
          const e = Math.min(
            Math.max((r.end ?? dim) < 0 ? dim + (r.end ?? dim) : (r.end ?? dim), 0),
            dim
          );
          nrList.push({ start: s, end: e, step, isIndex: false });
        } else {
          let s = (r.start ?? dim - 1) < 0 ? dim + (r.start ?? dim - 1) : (r.start ?? dim - 1);
          if (s >= dim) s = dim - 1;
          if (s < -1) s = -1;
          const endRaw = r.end ?? -1;
          let e: number;
          if (endRaw === -1) {
            e = -1;
          } else if (endRaw < 0) {
            e = dim + endRaw;
            if (e < -1) e = -1;
          } else {
            e = endRaw;
            if (e >= dim) e = dim - 1;
          }
          nrList.push({ start: s, end: e, step, isIndex: false });
        }
      }
    }

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
        // Scatter gradient back into a zero tensor of the original shape
        const dtype = ensureNumericDType(this.tensor.dtype, "slice");
        const gradData = new (dtypeToTypedArrayCtor(dtype))(this.tensor.size);
        const goDense = asFloat64Dense(go);

        // Build output shape (non-index axes)
        const outShape: number[] = [];
        for (const nr of nrList) {
          if (!nr.isIndex) {
            const len =
              nr.step > 0
                ? Math.max(0, Math.ceil((nr.end - nr.start) / nr.step))
                : Math.max(0, Math.ceil((nr.start - nr.end) / -nr.step));
            outShape.push(len);
          }
        }
        const outSize = outShape.length === 0 ? 1 : outShape.reduce((a, b) => a * b, 1);
        const outStrides = computeStrides(outShape);
        const inStrides = computeStrides(inputShape);

        for (let outFlat = 0; outFlat < outSize; outFlat++) {
          // Unravel output flat index
          let rem = outFlat;
          const outIdx: number[] = new Array(outShape.length);
          for (let i = 0; i < outShape.length; i++) {
            const s = outStrides[i] ?? 1;
            outIdx[i] = Math.floor(rem / s);
            rem %= s;
          }

          // Map to input flat index
          let inFlat = 0;
          let outAxis = 0;
          for (let axis = 0; axis < ndim; axis++) {
            const nr = nrList[axis];
            if (!nr) throw new DeepboxError("Internal error: missing normalized slice range");
            const inIdx = nr.isIndex ? nr.start : nr.start + (outIdx[outAxis++] ?? 0) * nr.step;
            inFlat += inIdx * (inStrides[axis] ?? 0);
          }

          const val = goDense[outFlat] ?? 0;
          if (gradData instanceof BigInt64Array) {
            gradData[inFlat] = BigInt(Math.round(val));
          } else {
            gradData[inFlat] = val;
          }
        }

        const gradTensor = TensorClass.fromTypedArray({
          data: gradData,
          shape: inputShape,
          dtype,
          device: this.tensor.device,
        });
        this.accumulateGrad(gradTensor);
      },
    });
    return out;
  }

  gather(indices: GradTensor, axis: Axis): GradTensor {
    const outTensor = gather(this.tensor, indices.tensor, axis);
    const ax = normalizeAxis(axis, this.tensor.ndim);
    const requiresGrad = gradEnabled && (this.requiresGrad || indices.requiresGrad);

    const out = new GradTensor({
      tensor: outTensor,
      requiresGrad,
      prev: requiresGrad ? [this, indices] : [],
      backward: () => {
        if (!requiresGrad) return;
        const go = out._grad;
        if (go === null) {
          throw new DeepboxError("Internal error: missing gradient for gather backward");
        }
        // scatter_add: accumulate gradient into positions that were gathered from
        if (this.requiresGrad) {
          const dtype = ensureNumericDType(this.tensor.dtype, "gather");
          const gradData = new (dtypeToTypedArrayCtor(dtype))(this.tensor.size);
          const goDense = asFloat64Dense(go);

          const outShape = outTensor.shape;
          const inShape = this.tensor.shape;
          const outStrides = computeStrides(outShape);
          const inStrides = computeStrides(inShape);
          const idxDense = asFloat64Dense(indices.tensor);

          for (let outFlat = 0; outFlat < outTensor.size; outFlat++) {
            // Unravel output flat index
            let rem = outFlat;
            const outIdx: number[] = new Array(outShape.length);
            for (let i = 0; i < outShape.length; i++) {
              const s = outStrides[i] ?? 1;
              outIdx[i] = Math.floor(rem / s);
              rem %= s;
            }

            // The input index is the same as outIdx except along the gather axis,
            // where we use the original index value
            const idxVal = Math.round(idxDense[outIdx[ax] ?? 0] ?? 0);
            const inIdx = outIdx.slice();
            inIdx[ax] = idxVal;

            let inFlat = 0;
            for (let i = 0; i < inShape.length; i++) {
              inFlat += (inIdx[i] ?? 0) * (inStrides[i] ?? 0);
            }

            const val = goDense[outFlat] ?? 0;
            if (gradData instanceof BigInt64Array) {
              gradData[inFlat] = (gradData[inFlat] ?? 0n) + BigInt(Math.round(val));
            } else {
              gradData[inFlat] = (gradData[inFlat] ?? 0) + val;
            }
          }

          const gradTensor = TensorClass.fromTypedArray({
            data: gradData,
            shape: inShape,
            dtype,
            device: this.tensor.device,
          });
          this.accumulateGrad(gradTensor);
        }
        // indices gradient is not meaningful (discrete), so we skip it
      },
    });
    return out;
  }

  mean(axis?: Axis, keepdims = false): GradTensor {
    let n: number;
    if (axis === undefined) {
      n = this.tensor.size;
    } else {
      const ax = normalizeAxis(axis, this.tensor.ndim);
      n = this.tensor.shape[ax] ?? 1;
    }
    const summed = this.sum(axis, keepdims);
    // Use summed tensor's dtype (which may be promoted by sum) to avoid dtype mismatch in div
    const denom = GradTensor.scalar(n, {
      dtype: ensureNumericDType(summed.tensor.dtype, "mean"),
    });
    return summed.div(denom);
  }

  max(axis?: Axis, keepdims = false): GradTensor {
    const outTensor = max(this.tensor, axis, keepdims);
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

        let maxReshaped = outTensor;
        let gradReshaped = go;

        if (axis === undefined) {
          // Full reduction
          const allOnes = new Array(this.tensor.ndim).fill(1);
          maxReshaped = outTensor.reshape(allOnes);
          gradReshaped = go.reshape(allOnes);
        } else if (!keepdims) {
          // Axis reduction without keepdims
          const ax = normalizeAxis(axis, this.tensor.ndim);
          const targetShape = [...this.tensor.shape];
          targetShape[ax] = 1;
          maxReshaped = outTensor.reshape(targetShape);
          gradReshaped = go.reshape(targetShape);
        }

        const maskBool = equal(this.tensor, maxReshaped);
        const mask = castTensor(maskBool, ensureNumericDType(this.tensor.dtype, "max"));
        const grad = mul(mask, gradReshaped);
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
        // view preserves total element count; reshape gradient back to original shape
        const gradReshaped = go.reshape(this.tensor.shape);
        this.accumulateGrad(gradReshaped);
      },
    });

    return out;
  }
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
          // Inverse permutation
          const invAxes = new Array<number>(axes.length);
          for (let i = 0; i < axes.length; i++) {
            const axis = axes[i];
            if (axis !== undefined) {
              invAxes[axis] = i;
            }
          }
          const grad = transpose(go, invAxes);
          this.accumulateGrad(grad);
        }
      },
    });

    return out;
  }

  min(axis?: Axis, keepdims = false): GradTensor {
    const outTensor = min(this.tensor, axis, keepdims);
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

        let minReshaped = outTensor;
        let gradReshaped = go;

        if (axis === undefined) {
          const allOnes = new Array(this.tensor.ndim).fill(1);
          minReshaped = outTensor.reshape(allOnes);
          gradReshaped = go.reshape(allOnes);
        } else if (!keepdims) {
          const ax = normalizeAxis(axis, this.tensor.ndim);
          const targetShape = [...this.tensor.shape];
          targetShape[ax] = 1;
          minReshaped = outTensor.reshape(targetShape);
          gradReshaped = go.reshape(targetShape);
        }

        const maskBool = equal(this.tensor, minReshaped);
        const mask = castTensor(maskBool, ensureNumericDType(this.tensor.dtype, "min"));
        const grad = mul(mask, gradReshaped);
        this.accumulateGrad(grad);
      },
    });

    return out;
  }

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

  clip(minVal: number, maxVal: number): GradTensor {
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
        // Gradient passes through where input is within [min, max], zero elsewhere
        const dtype = ensureNumericDType(this.tensor.dtype, "clip");
        const lowT = tensor(minVal, { dtype: this.tensor.dtype });
        const highT = tensor(maxVal, { dtype: this.tensor.dtype });
        const aboveLow = castTensor(greater(this.tensor, lowT), dtype);
        const equalLow = castTensor(equal(this.tensor, lowT), dtype);
        const belowHigh = castTensor(less(this.tensor, highT), dtype);
        const equalHigh = castTensor(equal(this.tensor, highT), dtype);
        // mask = (x > min || x == min) && (x < max || x == max)
        //      = (x >= min) && (x <= max)
        const geMin = add(aboveLow, equalLow);
        const leMax = add(belowHigh, equalHigh);
        const mask = mul(geMin, leMax);
        const grad = mul(go, mask);
        this.accumulateGrad(grad);
      },
    });

    return out;
  }

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

  elu(alpha = 1.0): GradTensor {
    const outTensor = elu(this.tensor, alpha);
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

  gelu(): GradTensor {
    const outTensor = gelu(this.tensor);
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
        // GELU(x) = x * Φ(x), where Φ is the standard normal CDF
        // d/dx GELU(x) ≈ Φ(x) + x * φ(x), where φ is the standard normal PDF
        // Approximate with tanh form:
        // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        // Derivative: use numerical approach via the output
        const dtype = ensureNumericDType(outTensor.dtype, "gelu");
        const inputDense = asFloat64Dense(this.tensor);
        const derivData = new Float64Array(this.tensor.size);
        const k = Math.sqrt(2 / Math.PI);
        for (let i = 0; i < inputDense.length; i++) {
          const x = inputDense[i] ?? 0;
          const inner = k * (x + 0.044715 * x * x * x);
          const tanhVal = Math.tanh(inner);
          const cdf = 0.5 * (1 + tanhVal);
          const pdf = k * (1 + 0.134145 * x * x) * (1 - tanhVal * tanhVal);
          derivData[i] = cdf + x * 0.5 * pdf;
        }
        const derivTensor = TensorClass.fromTypedArray({
          data: derivData,
          shape: this.tensor.shape,
          dtype: "float64",
          device: this.tensor.device,
        });
        const deriv = derivTensor.dtype === dtype ? derivTensor : castTensor(derivTensor, dtype);
        const grad = mul(go, deriv);
        this.accumulateGrad(grad);
      },
    });

    return out;
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
 * Create a GradTensor with requiresGrad=true.
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
 * Context manager to disable gradient calculation.
 *
 * **Important:** The callback must be synchronous. Passing an async function
 * will cause `gradEnabled` to be restored before the awaited work finishes,
 * silently breaking gradient tracking inside the async continuation.
 *
 * @throws {DeepboxError} If the callback returns a Promise (async function detected)
 */
export function noGrad<T>(fn: () => T): T {
  const prev = gradEnabled;
  gradEnabled = false;
  try {
    const result = fn();
    if (result instanceof Promise) {
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
 * Image to Column operation for GradTensor.
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

export function softmax(input: GradTensor, axis = -1): GradTensor {
  // Detach max for numerical stability (softmax is shift-invariant)
  const maxVal = max(input.tensor, axis, true);
  const maxT = GradTensor.fromTensor(maxVal, { requiresGrad: false });
  const shifted = input.sub(maxT);
  const expT = shifted.exp();
  const sumT = expT.sum(axis, true);
  return expT.div(sumT);
}

export function logSoftmax(input: GradTensor, axis = -1): GradTensor {
  const maxVal = max(input.tensor, axis, true);
  const maxTensor =
    maxVal.dtype === input.dtype
      ? maxVal
      : tensor(toNumericNestedArray(maxVal.toArray()), { dtype: input.dtype });
  const maxT = GradTensor.fromTensor(maxTensor, { requiresGrad: false });
  const shifted = input.sub(maxT);
  const expT = shifted.exp();
  const sumT = expT.sum(axis, true);
  const logSumExp = sumT.log();
  return shifted.sub(logSumExp);
}

export function variance(input: GradTensor, axis?: number, correction = 1): GradTensor {
  const meanVal = input.mean(axis, true);
  const centered = input.sub(meanVal);
  const sq = centered.square();
  const sumSq = sq.sum(axis, false); // Reduce dims

  let n: number;
  if (axis === undefined) {
    n = input.size;
  } else {
    const ax = axis < 0 ? input.ndim + axis : axis;
    if (ax < 0 || ax >= input.ndim) {
      throw new InvalidParameterError(
        `variance axis must be in range [0, ${input.ndim})`,
        "axis",
        axis
      );
    }
    n = input.shape[ax] ?? 1;
  }
  const denom = Math.max(0, n - correction);
  const denomT = GradTensor.scalar(denom, {
    dtype: ensureNumericDType(input.dtype, "variance"),
  });

  return sumSq.div(denomT);
}

export function dropout(input: GradTensor, p = 0.5, training = true): GradTensor {
  if (!training || p === 0) return input;

  const dtype = ensureNumericDType(input.dtype, "dropout");
  const scale = 1 / (1 - p);
  const mask = dropoutMask(input.shape, p, scale, dtype, input.device);
  const maskGrad = GradTensor.fromTensor(mask, { requiresGrad: false });
  return input.mul(maskGrad);
}
