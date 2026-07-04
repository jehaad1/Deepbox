/**
 * Padding layers: ZeroPad2d, ConstantPad2d, ReflectionPad2d, ReplicationPad2d.
 *
 * @module
 * @see {@link https://deepbox.dev/docs/nn-module | Deepbox documentation}
 */

import { DTypeError, InvalidParameterError, ShapeError } from "../../core";
import { type AnyTensor, customOp, GradTensor } from "../../ndarray";
import { readAsNumber, requireNumericData } from "../../ndarray/ops/_internal";
import { isContiguous, offsetFromFlatIndex } from "../../ndarray/tensor/strides";
import { computeStrides, Tensor as TensorClass } from "../../ndarray/tensor/Tensor";
import { Module } from "../module/Module";

type Padding4 = readonly [number, number, number, number];

function normalizePadding(padding: number | Padding4): Padding4 {
  if (typeof padding === "number") {
    return [padding, padding, padding, padding];
  }
  if (padding.length !== 4) {
    throw new InvalidParameterError(
      "padding must be a number or [left, right, top, bottom]",
      "padding",
      padding
    );
  }
  return padding;
}

/**
 * Differentiable 2-D padding. `resolve(oh, ow, inH, inW)` returns the source
 * (ih, iw) within an (n, c) plane for output cell (oh, ow), or null when the
 * cell is a constant fill. The forward gathers from those sources; the
 * backward scatter-adds each output gradient back to its source input cell
 * (constant cells contribute no gradient).
 */
function pad2dForward(
  inputGrad: GradTensor,
  padding: Padding4,
  resolve: (oh: number, ow: number, inH: number, inW: number) => readonly [number, number] | null,
  constValue = 0
): GradTensor {
  const input = inputGrad.tensor;
  if (input.dtype === "string") {
    throw new DTypeError("Padding layers do not support string dtype");
  }
  if (input.ndim !== 4) {
    throw new ShapeError(`Padding layers expect 4D input (N, C, H, W); got ${input.ndim}D`);
  }

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

  const inputData = requireNumericData(input.data, "Padding");
  const contig = isContiguous(input.shape, input.strides);
  const logical = computeStrides(input.shape);
  const readAt = (n: number, c: number, ih: number, iw: number): number => {
    const planeIdx = ((n * C + c) * H + ih) * W + iw;
    const off = contig
      ? input.offset + planeIdx
      : offsetFromFlatIndex(planeIdx, logical, input.strides, input.offset);
    return readAsNumber(inputData, off);
  };

  const outSize = N * C * outH * outW;
  const outData = new Float64Array(outSize);
  // srcIndex[o] = input logical-flat index the output read from, or -1 (const)
  const srcIndex = new Int32Array(outSize).fill(-1);

  let idx = 0;
  for (let n = 0; n < N; n++) {
    for (let c = 0; c < C; c++) {
      for (let oh = 0; oh < outH; oh++) {
        for (let ow = 0; ow < outW; ow++) {
          const src = resolve(oh, ow, H, W);
          if (src === null) {
            outData[idx] = constValue;
          } else {
            const [ih, iw] = src;
            outData[idx] = readAt(n, c, ih, iw);
            srcIndex[idx] = ((n * C + c) * H + ih) * W + iw;
          }
          idx++;
        }
      }
    }
  }

  const outTensor = TensorClass.fromTypedArray({
    data: outData,
    shape: [N, C, outH, outW],
    dtype: "float64",
    device: input.device,
  });

  return customOp(outTensor, [
    [
      inputGrad,
      (g: TensorClass): TensorClass => {
        const gd = requireNumericData(g.data, "Padding");
        const gContig = isContiguous(g.shape, g.strides);
        const gLogical = computeStrides(g.shape);
        const gi = new Float64Array(N * C * H * W);
        for (let o = 0; o < outSize; o++) {
          const si = srcIndex[o] ?? -1;
          if (si < 0) continue;
          const goff = gContig
            ? g.offset + o
            : offsetFromFlatIndex(o, gLogical, g.strides, g.offset);
          gi[si]! += readAsNumber(gd, goff);
        }
        return TensorClass.fromTypedArray({
          data: gi,
          shape: [N, C, H, W],
          dtype: "float64",
          device: input.device,
        });
      },
    ],
  ]);
}

/**
 * Pads the input tensor using zeros.
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

  forward(input: AnyTensor): GradTensor {
    const inputGrad = GradTensor.isGradTensor(input) ? input : GradTensor.fromTensor(input);
    const [padLeft, , padTop] = this.padding;

    return pad2dForward(inputGrad, this.padding, (oh, ow, inH, inW) => {
      const ih = oh - padTop;
      const iw = ow - padLeft;
      if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) return [ih, iw];
      return null;
    });
  }

  override toString(): string {
    return `ZeroPad2d(padding=[${this.padding.join(", ")}])`;
  }
}

/**
 * Pads the input tensor using a constant value.
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

  forward(input: AnyTensor): GradTensor {
    const inputGrad = GradTensor.isGradTensor(input) ? input : GradTensor.fromTensor(input);
    const [padLeft, , padTop] = this.padding;

    return pad2dForward(
      inputGrad,
      this.padding,
      (oh, ow, inH, inW) => {
        const ih = oh - padTop;
        const iw = ow - padLeft;
        if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) return [ih, iw];
        return null;
      },
      this.value
    );
  }

  override toString(): string {
    return `ConstantPad2d(padding=[${this.padding.join(", ")}], value=${this.value})`;
  }
}

/**
 * Pads using reflection of the input boundary.
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

  forward(input: AnyTensor): GradTensor {
    const t = GradTensor.isGradTensor(input) ? input.tensor : input;
    const [padLeft, , padTop] = this.padding;
    const inH = t.shape[2] ?? 0;
    const inW = t.shape[3] ?? 0;

    if (padTop >= inH || (this.padding[3] ?? 0) >= inH) {
      throw new ShapeError(
        `ReflectionPad2d padding (${padTop}, ${this.padding[3]}) must be less than input height (${inH})`
      );
    }
    if (padLeft >= inW || (this.padding[1] ?? 0) >= inW) {
      throw new ShapeError(
        `ReflectionPad2d padding (${padLeft}, ${this.padding[1]}) must be less than input width (${inW})`
      );
    }

    const inputGrad = GradTensor.isGradTensor(input) ? input : GradTensor.fromTensor(input);
    return pad2dForward(inputGrad, this.padding, (oh, ow, iH, iW) => {
      let ih = oh - padTop;
      let iw = ow - padLeft;
      // Reflect: if out of bounds, mirror
      if (ih < 0) ih = -ih;
      if (ih >= iH) ih = 2 * (iH - 1) - ih;
      if (iw < 0) iw = -iw;
      if (iw >= iW) iw = 2 * (iW - 1) - iw;
      return [ih, iw];
    });
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

  forward(input: AnyTensor): GradTensor {
    const inputGrad = GradTensor.isGradTensor(input) ? input : GradTensor.fromTensor(input);
    const [padLeft, , padTop] = this.padding;

    return pad2dForward(inputGrad, this.padding, (oh, ow, inH, inW) => {
      const ih = Math.min(Math.max(oh - padTop, 0), inH - 1);
      const iw = Math.min(Math.max(ow - padLeft, 0), inW - 1);
      return [ih, iw];
    });
  }

  override toString(): string {
    return `ReplicationPad2d(padding=[${this.padding.join(", ")}])`;
  }
}
