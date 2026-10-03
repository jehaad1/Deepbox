/**
 * Upsample layer with nearest and bilinear modes.
 *
 * @module
 * @see {@link https://deepbox.dev/docs/nn-module | Deepbox documentation}
 */

import { type DType, DTypeError, InvalidParameterError, ShapeError } from "../../core";
import { type AnyTensor, customOp, GradTensor } from "../../ndarray";
import { readAsNumber, requireNumericData } from "../../ndarray/ops/_internal";
import { isContiguous, offsetFromFlatIndex } from "../../ndarray/tensor/strides";
import { computeStrides, Tensor as TensorClass } from "../../ndarray/tensor/Tensor";
import { Module } from "../module/Module";
import { allPlain, settle } from "./_shared";

type UpsampleMode = "nearest" | "bilinear";

/** Per-axis source indices and interpolation weight for every output position. */
type AxisTable = {
  /** Lower source index. */
  readonly lo: Int32Array;
  /** Upper source index (equal to `lo` when there is nothing to blend). */
  readonly hi: Int32Array;
  /** Weight of the upper index; the lower index has weight `1 - hiWeight`. */
  readonly hiWeight: Float64Array;
};

function isFloatDType(dtype: DType): boolean {
  return dtype === "float16" || dtype === "bfloat16" || dtype === "float32" || dtype === "float64";
}

/** Cast a float64 result to `dtype` (no-op for float64). */
function castResult(t: TensorClass, dtype: DType): TensorClass {
  return t.dtype === dtype ? t : t.astype(dtype);
}

/**
 * Validate one scale factor entry.
 */
function checkScale(value: number): void {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new InvalidParameterError("scaleFactor must be a positive number", "scaleFactor", value);
  }
}

/**
 * Build the source index and weight tables for one spatial axis.
 *
 * Index rules follow `torch.nn.Upsample`: when the output size comes from `size`
 * the ratio is `inSize / outSize`; when it comes from `scaleFactor` the ratio is
 * `1 / scaleFactor`. An axis whose size does not change is copied. Nearest
 * indices are computed with a float32 ratio, as PyTorch does, so that a position
 * that lies exactly on a source pixel boundary (for example `33 / 1.1`) does not
 * fall to the wrong side through double rounding.
 */
function buildAxisTable(
  inSize: number,
  outSize: number,
  scale: number | null,
  mode: UpsampleMode,
  alignCorners: boolean
): AxisTable {
  const lo = new Int32Array(outSize);
  const hi = new Int32Array(outSize);
  const hiWeight = new Float64Array(outSize);
  const last = inSize - 1;
  const copy = outSize === inSize;
  const nearestRatio = Math.fround(scale === null ? inSize / outSize : 1 / scale);

  for (let o = 0; o < outSize; o++) {
    if (copy) {
      lo[o] = o;
      hi[o] = o;
      continue;
    }
    if (mode === "nearest") {
      const src = outSize === 2 * inSize ? o >> 1 : Math.floor(Math.fround(o * nearestRatio));
      const idx = Math.min(src, last);
      lo[o] = idx;
      hi[o] = idx;
      continue;
    }

    let src: number;
    if (alignCorners) {
      src = outSize > 1 ? (o * last) / (outSize - 1) : 0;
    } else {
      const ratio = scale === null ? inSize / outSize : 1 / scale;
      src = Math.max(ratio * (o + 0.5) - 0.5, 0);
    }
    const i0 = Math.min(Math.floor(src), last);
    const i1 = Math.min(i0 + 1, last);
    lo[o] = i0;
    hi[o] = i1;
    hiWeight[o] = Math.min(Math.max(src - i0, 0), 1);
  }
  return { lo, hi, hiWeight };
}

/**
 * Upsamples a 4D input (N, C, H, W) using nearest-neighbor or bilinear interpolation.
 *
 * Specify either `scaleFactor` or `size`, not both. With `scaleFactor`, the output
 * height and width are `floor(input * scaleFactor)`. The output keeps the input's
 * float dtype (integer and bool inputs produce float64) and is differentiable with
 * respect to the input.
 *
 * Bilinear mode follows `alignCorners`. The default is `true` (corner pixels of
 * input and output coincide), which is the convention Deepbox has always used.
 * PyTorch's `nn.Upsample` defaults to `false`; pass `alignCorners: false` for
 * results that match it.
 *
 * @example
 * ```ts
 * const up = new Upsample({ scaleFactor: 2, mode: 'nearest' });
 * // input: (1, 3, 4, 4) -> output: (1, 3, 8, 8)
 *
 * const up2 = new Upsample({ size: [16, 16], mode: 'bilinear', alignCorners: false });
 * ```
 *
 * @category Neural Network Layers
 */
export class Upsample extends Module {
  private readonly scaleFactor?: readonly [number, number];
  private readonly size?: readonly [number, number];
  private readonly mode: UpsampleMode;
  private readonly alignCorners: boolean;

  /**
   * @param options.scaleFactor - Multiplier for height and width, a single number or `[scaleH, scaleW]`
   * @param options.size - Target output size, a single integer or `[height, width]`
   * @param options.mode - `"nearest"` (default) or `"bilinear"`
   * @param options.alignCorners - Bilinear only: align the corner pixels of input and output
   *   (default: `true`; PyTorch's default is `false`)
   * @throws {InvalidParameterError} If both or neither of `scaleFactor` and `size` are given, or a value is invalid
   */
  constructor(options: {
    readonly scaleFactor?: number | readonly [number, number];
    readonly size?: number | readonly [number, number];
    readonly mode?: UpsampleMode;
    readonly alignCorners?: boolean;
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
      const sf = options.scaleFactor;
      if (typeof sf === "number") {
        checkScale(sf);
        this.scaleFactor = [sf, sf];
      } else {
        if (!Array.isArray(sf) || sf.length !== 2) {
          throw new InvalidParameterError(
            "scaleFactor must be a positive number or [scaleH, scaleW]",
            "scaleFactor",
            sf
          );
        }
        checkScale(sf[0]);
        checkScale(sf[1]);
        this.scaleFactor = [sf[0], sf[1]];
      }
    }

    if (options.size !== undefined) {
      const size = typeof options.size === "number" ? [options.size, options.size] : options.size;
      if (
        !Array.isArray(size) ||
        size.length !== 2 ||
        !Number.isInteger(size[0]) ||
        !Number.isInteger(size[1]) ||
        (size[0] ?? 0) <= 0 ||
        (size[1] ?? 0) <= 0
      ) {
        throw new InvalidParameterError(
          "size must be [height, width] with positive integers",
          "size",
          options.size
        );
      }
      this.size = [size[0] as number, size[1] as number];
    }

    this.mode = options.mode ?? "nearest";
    if (this.mode !== "nearest" && this.mode !== "bilinear") {
      throw new InvalidParameterError("mode must be 'nearest' or 'bilinear'", "mode", this.mode);
    }

    const alignCorners = options.alignCorners ?? true;
    if (typeof alignCorners !== "boolean") {
      throw new InvalidParameterError(
        "alignCorners must be a boolean",
        "alignCorners",
        alignCorners
      );
    }
    this.alignCorners = alignCorners;
  }

  forward(input: GradTensor): GradTensor;
  forward(input: TensorClass): TensorClass;
  forward(input: AnyTensor): AnyTensor;
  forward(input: AnyTensor): AnyTensor {
    return settle(this.run(input), allPlain(input));
  }

  private run(input: AnyTensor): GradTensor {
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

    if (inH < 1 || inW < 1) {
      throw new ShapeError(
        `Upsample requires non-empty spatial dimensions; got height ${inH}, width ${inW}`
      );
    }

    let outH: number;
    let outW: number;
    let scaleH: number | null = null;
    let scaleW: number | null = null;
    if (this.size) {
      outH = this.size[0];
      outW = this.size[1];
    } else {
      const [sfH, sfW] = this.scaleFactor ?? [1, 1];
      scaleH = sfH;
      scaleW = sfW;
      outH = Math.floor(inH * sfH);
      outW = Math.floor(inW * sfW);
    }
    if (outH < 1 || outW < 1) {
      throw new ShapeError(
        `Upsample output size must be positive; got ${outH}x${outW} for input ${inH}x${inW}`
      );
    }

    const rows = buildAxisTable(inH, outH, scaleH, this.mode, this.alignCorners);
    const cols = buildAxisTable(inW, outW, scaleW, this.mode, this.alignCorners);

    // Per output pixel, up to four (input plane index, weight) taps in the order
    // (lo row, lo col), (lo row, hi col), (hi row, lo col), (hi row, hi col).
    // Taps with zero weight are dropped so non-finite neighbours never leak in.
    const spatialLen = outH * outW;
    const tapIndex = new Int32Array(spatialLen * 4);
    const tapWeight = new Float64Array(spatialLen * 4);
    const tapCount = new Uint8Array(spatialLen);
    for (let oh = 0; oh < outH; oh++) {
      const h0 = rows.lo[oh] ?? 0;
      const h1 = rows.hi[oh] ?? 0;
      const hw1 = rows.hiWeight[oh] ?? 0;
      const hTaps: ReadonlyArray<readonly [number, number]> = [
        [h0, 1 - hw1],
        [h1, hw1],
      ];
      for (let ow = 0; ow < outW; ow++) {
        const w0 = cols.lo[ow] ?? 0;
        const w1 = cols.hi[ow] ?? 0;
        const ww1 = cols.hiWeight[ow] ?? 0;
        const wTaps: ReadonlyArray<readonly [number, number]> = [
          [w0, 1 - ww1],
          [w1, ww1],
        ];
        const sp = oh * outW + ow;
        let count = 0;
        for (const [ih, wh] of hTaps) {
          for (const [iw, ww] of wTaps) {
            const weight = wh * ww;
            if (weight === 0) continue;
            tapIndex[sp * 4 + count] = ih * inW + iw;
            tapWeight[sp * 4 + count] = weight;
            count++;
          }
        }
        tapCount[sp] = count;
      }
    }

    const data = requireNumericData(t.data, "Upsample");
    const inPlane = inH * inW;
    const planes = N * C;
    let src: ArrayLike<number>;
    let srcBase: number;
    if (isContiguous(t.shape, t.strides) && !(data instanceof BigInt64Array)) {
      src = data;
      srcBase = t.offset;
    } else {
      const dense = new Float64Array(planes * inPlane);
      const logical = computeStrides(t.shape);
      const contiguous = isContiguous(t.shape, t.strides);
      for (let i = 0; i < dense.length; i++) {
        const off = contiguous
          ? t.offset + i
          : offsetFromFlatIndex(i, logical, t.strides, t.offset);
        dense[i] = readAsNumber(data, off);
      }
      src = dense;
      srcBase = 0;
    }

    const outArr = new Float64Array(planes * spatialLen);
    for (let p = 0; p < planes; p++) {
      const inBase = srcBase + p * inPlane;
      const outBase = p * spatialLen;
      for (let sp = 0; sp < spatialLen; sp++) {
        let acc = 0;
        const count = tapCount[sp] ?? 0;
        for (let k = 0; k < count; k++) {
          const slot = sp * 4 + k;
          acc += (src[inBase + (tapIndex[slot] ?? 0)] ?? 0) * (tapWeight[slot] ?? 0);
        }
        outArr[outBase + sp] = acc;
      }
    }

    const outDtype: DType = isFloatDType(t.dtype) ? t.dtype : "float64";
    const outTensor = castResult(
      TensorClass.fromTypedArray({
        data: outArr,
        shape: [N, C, outH, outW],
        dtype: "float64",
        device: t.device,
      }),
      outDtype
    );

    const inputGrad = GradTensor.isGradTensor(input) ? input : GradTensor.fromTensor(input);
    return customOp(outTensor, [
      [
        inputGrad,
        (g: TensorClass): TensorClass => {
          const gd = requireNumericData(g.data, "Upsample");
          const gContig = isContiguous(g.shape, g.strides);
          const gLogical = computeStrides(g.shape);
          const gi = new Float64Array(planes * inPlane);
          let oi = 0;
          for (let p = 0; p < planes; p++) {
            const planeBase = p * inPlane;
            for (let sp = 0; sp < spatialLen; sp++) {
              const goff = gContig
                ? g.offset + oi
                : offsetFromFlatIndex(oi, gLogical, g.strides, g.offset);
              const gv = readAsNumber(gd, goff);
              oi++;
              const count = tapCount[sp] ?? 0;
              for (let k = 0; k < count; k++) {
                const slot = sp * 4 + k;
                const target = planeBase + (tapIndex[slot] ?? 0);
                gi[target] = (gi[target] ?? 0) + gv * (tapWeight[slot] ?? 0);
              }
            }
          }
          return castResult(
            TensorClass.fromTypedArray({
              data: gi,
              shape: [N, C, inH, inW],
              dtype: "float64",
              device: t.device,
            }),
            isFloatDType(t.dtype) ? t.dtype : "float64"
          );
        },
      ],
    ]);
  }

  override toString(): string {
    const target = this.size
      ? `size=[${this.size.join(", ")}]`
      : `scale_factor=${
          this.scaleFactor && this.scaleFactor[0] === this.scaleFactor[1]
            ? this.scaleFactor[0]
            : `[${this.scaleFactor?.join(", ")}]`
        }`;
    const corners = this.mode === "bilinear" ? `, align_corners=${this.alignCorners}` : "";
    return `Upsample(${target}, mode='${this.mode}'${corners})`;
  }
}
