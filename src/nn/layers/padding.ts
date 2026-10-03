/**
 * Padding layers: ZeroPad2d, ConstantPad2d, ReflectionPad2d, ReplicationPad2d.
 *
 * All layers accept `(N, C, H, W)` input, or an unbatched `(C, H, W)` input,
 * keep the input dtype, and are differentiable.
 *
 * @module
 * @see {@link https://deepbox.dev/docs/nn-module | Deepbox documentation}
 */

import {
  DTypeError,
  ensureNumericDType,
  InvalidParameterError,
  ShapeError,
  type TypedArray,
} from "../../core";
import { type AnyTensor, customOp, type GradTensor } from "../../ndarray";
import { readNumbers, requireNumericData } from "../../ndarray/ops/_internal";
import { isContiguous, offsetFromFlatIndex } from "../../ndarray/tensor/strides";
import { computeStrides, Tensor as TensorClass } from "../../ndarray/tensor/Tensor";
import { Module } from "../module/Module";
import { allPlain, settle, toGradInput } from "./_shared";

type Padding4 = readonly [number, number, number, number];

type PadMode = "constant" | "reflect" | "replicate";

function normalizePadding(padding: number | Padding4): Padding4 {
  const values: number[] =
    typeof padding === "number" ? [padding, padding, padding, padding] : Array.from(padding);
  if (values.length !== 4) {
    throw new InvalidParameterError(
      "padding must be a number or [left, right, top, bottom]",
      "padding",
      padding
    );
  }
  for (const v of values) {
    if (!Number.isInteger(v)) {
      throw new InvalidParameterError("padding values must be integers", "padding", padding);
    }
  }
  return [values[0] ?? 0, values[1] ?? 0, values[2] ?? 0, values[3] ?? 0];
}

/**
 * For every output position along one spatial axis, the input position it
 * reads from, or -1 when the position is a constant fill.
 *
 * Reflection requires each pad to be smaller than the axis size, which the
 * caller checks beforehand, so a single mirror step always lands in range.
 */
function buildIndexMap(mode: PadMode, outLen: number, before: number, size: number): Int32Array {
  const map = new Int32Array(outLen);
  for (let o = 0; o < outLen; o++) {
    const i = o - before;
    if (i >= 0 && i < size) {
      map[o] = i;
    } else if (mode === "constant") {
      map[o] = -1;
    } else if (mode === "replicate") {
      map[o] = i < 0 ? 0 : size - 1;
    } else {
      map[o] = i < 0 ? -i : 2 * (size - 1) - i;
    }
  }
  return map;
}

/**
 * Differentiable 2-D padding. The forward gathers each output cell from the
 * input cell given by the row and column maps (or writes the constant fill);
 * the backward scatter-adds each output gradient back to its source input
 * cell, so constant cells contribute no gradient.
 */
function pad2dForward(
  inputGrad: GradTensor,
  padding: Padding4,
  mode: PadMode,
  layerName: string,
  constValue = 0
): GradTensor {
  const origDtype = inputGrad.dtype;
  if (origDtype === "string") {
    throw new DTypeError("Padding layers do not support string dtype");
  }
  if (inputGrad.ndim !== 3 && inputGrad.ndim !== 4) {
    throw new ShapeError(
      `${layerName} expects 3D (C, H, W) or 4D (N, C, H, W) input; got ${inputGrad.ndim}D`
    );
  }
  // Half-precision tensors are padded in float32 and cast back afterwards.
  if (origDtype === "float16" || origDtype === "bfloat16") {
    return pad2dForward(inputGrad.astype("float32"), padding, mode, layerName, constValue).astype(
      origDtype
    );
  }
  if (inputGrad.ndim === 3) {
    const batched = inputGrad.reshape([1, ...inputGrad.shape]);
    const out = pad2dForward(batched, padding, mode, layerName, constValue);
    return out.reshape(out.shape.slice(1));
  }

  const input = inputGrad.tensor;
  const [padLeft, padRight, padTop, padBottom] = padding;
  const N = input.shape[0] ?? 0;
  const C = input.shape[1] ?? 0;
  const H = input.shape[2] ?? 0;
  const W = input.shape[3] ?? 0;
  const outH = H + padTop + padBottom;
  const outW = W + padLeft + padRight;

  if (outH <= 0 || outW <= 0) {
    throw new ShapeError(`Padded output dimensions must be positive; got ${outH}x${outW}`);
  }
  if (mode === "reflect") {
    if (padTop >= H || padBottom >= H) {
      throw new ShapeError(
        `${layerName} padding (${padTop}, ${padBottom}) must be less than input height (${H})`
      );
    }
    if (padLeft >= W || padRight >= W) {
      throw new ShapeError(
        `${layerName} padding (${padLeft}, ${padRight}) must be less than input width (${W})`
      );
    }
  } else if (mode === "replicate" && (H === 0 || W === 0)) {
    throw new ShapeError(`${layerName} cannot replicate an empty input; got ${H}x${W}`);
  }

  const rowMap = buildIndexMap(mode, outH, padTop, H);
  const colMap = buildIndexMap(mode, outW, padLeft, W);

  const dtype = ensureNumericDType(input.dtype, layerName);
  const inputData = requireNumericData(input.data, layerName);
  const contig = isContiguous(input.shape, input.strides);
  const logical = computeStrides(input.shape);
  const planeSize = H * W;
  const outSize = N * C * outH * outW;
  const needsGrad = inputGrad.requiresGrad;
  // srcIndex[o] = input logical-flat index the output read from, or -1 (const fill)
  const srcIndex = needsGrad ? new Int32Array(outSize).fill(-1) : null;

  // Visit every non-constant output cell: fn(outputIndex, inputLogicalFlatIndex).
  const forEachSource = (fn: (o: number, flat: number) => void): void => {
    let o = 0;
    for (let n = 0; n < N; n++) {
      for (let c = 0; c < C; c++) {
        const planeBase = (n * C + c) * planeSize;
        for (let oh = 0; oh < outH; oh++) {
          const ih = rowMap[oh] ?? -1;
          if (ih < 0) {
            o += outW;
            continue;
          }
          const rowBase = planeBase + ih * W;
          for (let ow = 0; ow < outW; ow++, o++) {
            const iw = colMap[ow] ?? -1;
            if (iw >= 0) fn(o, rowBase + iw);
          }
        }
      }
    }
  };
  const physical = (flat: number): number =>
    contig ? input.offset + flat : offsetFromFlatIndex(flat, logical, input.strides, input.offset);

  let outTensor: TensorClass;
  if (inputData instanceof BigInt64Array) {
    if (!Number.isFinite(constValue)) {
      throw new InvalidParameterError(
        "padding value must be finite for int64 input",
        "value",
        constValue
      );
    }
    const out = new BigInt64Array(outSize);
    const fill = BigInt(Math.trunc(constValue));
    if (fill !== 0n) out.fill(fill);
    forEachSource((o, flat) => {
      out[o] = inputData[physical(flat)] as bigint;
      if (srcIndex) srcIndex[o] = flat;
    });
    outTensor = TensorClass.fromTypedArray({
      data: out,
      shape: [N, C, outH, outW],
      dtype,
      device: input.device,
    });
  } else {
    const Ctor = inputData.constructor as new (n: number) => Exclude<TypedArray, BigInt64Array>;
    const out = new Ctor(outSize);
    const fill = dtype === "bool" ? (constValue !== 0 ? 1 : 0) : constValue;
    if (fill !== 0) out.fill(fill);
    forEachSource((o, flat) => {
      out[o] = inputData[physical(flat)] as number;
      if (srcIndex) srcIndex[o] = flat;
    });
    outTensor = TensorClass.fromTypedArray({
      data: out,
      shape: [N, C, outH, outW],
      dtype,
      device: input.device,
    });
  }

  const gradDtype = dtype === "float64" ? "float64" : "float32";

  return customOp(outTensor, [
    [
      inputGrad,
      (g: TensorClass): TensorClass => {
        const gd = readNumbers(g, layerName);
        const gi =
          gradDtype === "float64"
            ? new Float64Array(N * C * planeSize)
            : new Float32Array(N * C * planeSize);
        if (srcIndex) {
          for (let o = 0; o < outSize; o++) {
            const si = srcIndex[o] as number;
            if (si >= 0) gi[si] = (gi[si] as number) + (gd[o] as number);
          }
        }
        return TensorClass.fromTypedArray({
          data: gi,
          shape: [N, C, H, W],
          dtype: gradDtype,
          device: input.device,
        });
      },
    ],
  ]);
}

/**
 * Pads the input tensor using zeros.
 *
 * The padding is `[left, right, top, bottom]`, or a single number applied to
 * all four sides. Negative values crop. Accepts `(N, C, H, W)` and unbatched
 * `(C, H, W)` input.
 *
 * @example
 * ```ts
 * const pad = new ZeroPad2d(1); // pad 1 on all sides
 * const pad2 = new ZeroPad2d([1, 1, 2, 2]); // [left, right, top, bottom]
 * ```
 *
 * @category Neural Network Layers
 */
export class ZeroPad2d extends Module {
  private readonly padding: Padding4;

  constructor(padding: number | Padding4) {
    super();
    this.padding = normalizePadding(padding);
  }

  forward(input: GradTensor): GradTensor;
  forward(input: TensorClass): TensorClass;
  forward(input: AnyTensor): AnyTensor;
  forward(input: AnyTensor): AnyTensor {
    return settle(
      pad2dForward(toGradInput(input), this.padding, "constant", "ZeroPad2d", 0),
      allPlain(input)
    );
  }

  override toString(): string {
    return `ZeroPad2d(padding=[${this.padding.join(", ")}])`;
  }
}

/**
 * Pads the input tensor using a constant value.
 *
 * The padding is `[left, right, top, bottom]`, or a single number applied to
 * all four sides. Negative values crop. For integer and boolean inputs the
 * value is truncated to the input dtype.
 *
 * @example
 * ```ts
 * const pad = new ConstantPad2d(1, -1); // pad 1 on all sides with -1
 * ```
 *
 * @category Neural Network Layers
 */
export class ConstantPad2d extends Module {
  private readonly padding: Padding4;
  private readonly value: number;

  constructor(padding: number | Padding4, value: number) {
    super();
    this.padding = normalizePadding(padding);
    this.value = value;
  }

  forward(input: GradTensor): GradTensor;
  forward(input: TensorClass): TensorClass;
  forward(input: AnyTensor): AnyTensor;
  forward(input: AnyTensor): AnyTensor {
    return settle(
      pad2dForward(toGradInput(input), this.padding, "constant", "ConstantPad2d", this.value),
      allPlain(input)
    );
  }

  override toString(): string {
    return `ConstantPad2d(padding=[${this.padding.join(", ")}], value=${this.value})`;
  }
}

/**
 * Pads using reflection of the input boundary (the edge row or column is not
 * repeated). Each pad must be smaller than the corresponding input size.
 *
 * @example
 * ```ts
 * const pad = new ReflectionPad2d(1);
 * ```
 *
 * @category Neural Network Layers
 */
export class ReflectionPad2d extends Module {
  private readonly padding: Padding4;

  constructor(padding: number | Padding4) {
    super();
    this.padding = normalizePadding(padding);
  }

  forward(input: GradTensor): GradTensor;
  forward(input: TensorClass): TensorClass;
  forward(input: AnyTensor): AnyTensor;
  forward(input: AnyTensor): AnyTensor {
    return settle(
      pad2dForward(toGradInput(input), this.padding, "reflect", "ReflectionPad2d"),
      allPlain(input)
    );
  }

  override toString(): string {
    return `ReflectionPad2d(padding=[${this.padding.join(", ")}])`;
  }
}

/**
 * Pads using replication of the input boundary values.
 *
 * @example
 * ```ts
 * const pad = new ReplicationPad2d(1);
 * ```
 *
 * @category Neural Network Layers
 */
export class ReplicationPad2d extends Module {
  private readonly padding: Padding4;

  constructor(padding: number | Padding4) {
    super();
    this.padding = normalizePadding(padding);
  }

  forward(input: GradTensor): GradTensor;
  forward(input: TensorClass): TensorClass;
  forward(input: AnyTensor): AnyTensor;
  forward(input: AnyTensor): AnyTensor {
    return settle(
      pad2dForward(toGradInput(input), this.padding, "replicate", "ReplicationPad2d"),
      allPlain(input)
    );
  }

  override toString(): string {
    return `ReplicationPad2d(padding=[${this.padding.join(", ")}])`;
  }
}
