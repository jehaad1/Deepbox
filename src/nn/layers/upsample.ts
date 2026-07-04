/**
 * Upsample layer with nearest and bilinear modes.
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

type UpsampleMode = "nearest" | "bilinear";

/**
 * Upsamples a 4D input (N, C, H, W) using nearest-neighbor or bilinear interpolation.
 *
 * Specify either `scaleFactor` or `size`, not both.
 *
 * @example
 * ```ts
 * const up = new Upsample({ scaleFactor: 2, mode: 'nearest' });
 * // input: (1, 3, 4, 4) -> output: (1, 3, 8, 8)
 *
 * const up2 = new Upsample({ size: [16, 16], mode: 'bilinear' });
 * ```
 *
 * @category Neural Network Layers
 */
export class Upsample extends Module {
  private readonly scaleFactor?: number;
  private readonly size?: readonly [number, number];
  private readonly mode: UpsampleMode;

  constructor(options: {
    readonly scaleFactor?: number;
    readonly size?: readonly [number, number];
    readonly mode?: UpsampleMode;
  }) {
    super();

    if (options.scaleFactor !== undefined && options.size !== undefined) {
      throw new InvalidParameterError(
        "Specify either scaleFactor or size, not both",
        "options",
        options
      );
    }
    if (options.scaleFactor === undefined && options.size === undefined) {
      throw new InvalidParameterError(
        "Must specify either scaleFactor or size",
        "options",
        options
      );
    }

    if (options.scaleFactor !== undefined) {
      if (!Number.isFinite(options.scaleFactor) || options.scaleFactor <= 0) {
        throw new InvalidParameterError(
          "scaleFactor must be a positive number",
          "scaleFactor",
          options.scaleFactor
        );
      }
      this.scaleFactor = options.scaleFactor;
    }

    if (options.size !== undefined) {
      if (
        options.size.length !== 2 ||
        !Number.isInteger(options.size[0]) ||
        !Number.isInteger(options.size[1]) ||
        options.size[0] <= 0 ||
        options.size[1] <= 0
      ) {
        throw new InvalidParameterError(
          "size must be [height, width] with positive integers",
          "size",
          options.size
        );
      }
      this.size = options.size;
    }

    this.mode = options.mode ?? "nearest";
    if (this.mode !== "nearest" && this.mode !== "bilinear") {
      throw new InvalidParameterError("mode must be 'nearest' or 'bilinear'", "mode", this.mode);
    }
  }

  forward(input: AnyTensor): GradTensor {
    const t = GradTensor.isGradTensor(input) ? input.tensor : input;

    if (t.dtype === "string") {
      throw new DTypeError("Upsample does not support string dtype");
    }
    if (t.ndim !== 4) {
      throw new ShapeError(`Upsample expects 4D input (N, C, H, W); got ${t.ndim}D`);
    }

    const N = t.shape[0] ?? 0;
    const C = t.shape[1] ?? 0;
    const inH = t.shape[2] ?? 0;
    const inW = t.shape[3] ?? 0;

    let outH: number;
    let outW: number;
    if (this.size) {
      outH = this.size[0];
      outW = this.size[1];
    } else {
      const sf = this.scaleFactor ?? 1;
      outH = Math.round(inH * sf);
      outW = Math.round(inW * sf);
    }

    const data = requireNumericData(t.data, "Upsample");
    const inContig = isContiguous(t.shape, t.strides);
    const inLogical = computeStrides(t.shape);
    const readPlane = (n: number, c: number, planeIdx: number): number => {
      const logicalIdx = (n * C + c) * inH * inW + planeIdx;
      const off = inContig
        ? t.offset + logicalIdx
        : offsetFromFlatIndex(logicalIdx, inLogical, t.strides, t.offset);
      return readAsNumber(data, off);
    };

    // Precompute spatial contributions (same across n, c): for each output
    // pixel, the list of (input plane index, weight) it interpolates from.
    // This makes both the forward gather and the backward scatter exact.
    const spatial: Array<Array<[number, number]>> = new Array(outH * outW);
    for (let oh = 0; oh < outH; oh++) {
      let hContribs: Array<[number, number]>;
      if (this.mode === "nearest") {
        const ih = Math.min(Math.floor((oh * inH) / outH), inH - 1);
        hContribs = [[ih, 1]];
      } else {
        const srcH = outH > 1 ? (oh * (inH - 1)) / (outH - 1) : 0;
        const h0 = Math.floor(srcH);
        const h1 = Math.min(h0 + 1, inH - 1);
        const hFrac = srcH - h0;
        hContribs = [
          [h0, 1 - hFrac],
          [h1, hFrac],
        ];
      }
      for (let ow = 0; ow < outW; ow++) {
        let wContribs: Array<[number, number]>;
        if (this.mode === "nearest") {
          const iw = Math.min(Math.floor((ow * inW) / outW), inW - 1);
          wContribs = [[iw, 1]];
        } else {
          const srcW = outW > 1 ? (ow * (inW - 1)) / (outW - 1) : 0;
          const w0 = Math.floor(srcW);
          const w1 = Math.min(w0 + 1, inW - 1);
          const wFrac = srcW - w0;
          wContribs = [
            [w0, 1 - wFrac],
            [w1, wFrac],
          ];
        }
        const list: Array<[number, number]> = [];
        for (const [ih, wh] of hContribs) {
          for (const [iw, ww] of wContribs) {
            const weight = wh * ww;
            if (weight === 0) continue;
            list.push([ih * inW + iw, weight]);
          }
        }
        spatial[oh * outW + ow] = list;
      }
    }

    const outArr = new Float64Array(N * C * outH * outW);
    let idx = 0;
    for (let n = 0; n < N; n++) {
      for (let c = 0; c < C; c++) {
        for (let sp = 0; sp < outH * outW; sp++) {
          let acc = 0;
          for (const [planeIdx, weight] of spatial[sp]!) acc += readPlane(n, c, planeIdx) * weight;
          outArr[idx++] = acc;
        }
      }
    }

    const outTensor = TensorClass.fromTypedArray({
      data: outArr,
      shape: [N, C, outH, outW],
      dtype: "float64",
      device: t.device,
    });

    const inputGrad = GradTensor.isGradTensor(input) ? input : GradTensor.fromTensor(input);
    const spatialLen = outH * outW;
    return customOp(outTensor, [
      [
        inputGrad,
        (g: TensorClass): TensorClass => {
          const gd = requireNumericData(g.data, "Upsample");
          const gContig = isContiguous(g.shape, g.strides);
          const gLogical = computeStrides(g.shape);
          const gi = new Float64Array(N * C * inH * inW);
          let oi = 0;
          for (let n = 0; n < N; n++) {
            for (let c = 0; c < C; c++) {
              const planeBase = (n * C + c) * inH * inW;
              for (let sp = 0; sp < spatialLen; sp++) {
                const goff = gContig
                  ? g.offset + oi
                  : offsetFromFlatIndex(oi, gLogical, g.strides, g.offset);
                const gv = readAsNumber(gd, goff);
                oi++;
                for (const [planeIdx, weight] of spatial[sp]!) {
                  gi[planeBase + planeIdx]! += gv * weight;
                }
              }
            }
          }
          return TensorClass.fromTypedArray({
            data: gi,
            shape: [N, C, inH, inW],
            dtype: "float64",
            device: t.device,
          });
        },
      ],
    ]);
  }

  override toString(): string {
    if (this.size) {
      return `Upsample(size=[${this.size.join(", ")}], mode='${this.mode}')`;
    }
    return `Upsample(scale_factor=${this.scaleFactor}, mode='${this.mode}')`;
  }
}
