/**
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox documentation}
 */

import type { Shape } from "../../core";
import { DTypeError, dtypeToTypedArrayCtor, InvalidParameterError, ShapeError } from "../../core";
import { computeStrides, Tensor } from "../tensor/Tensor";
import { dispatchCol2im, dispatchIm2col } from "./device_dispatch";

function validateConvPair(
  name: "kernelSize" | "stride" | "padding",
  pair: readonly number[],
  minValue: number
): void {
  if (!Array.isArray(pair) || pair.length !== 2) {
    throw new InvalidParameterError(
      `${name} must be a pair [height, width]; received ${JSON.stringify(pair)}`,
      name,
      pair
    );
  }
  for (const [i, v] of pair.entries()) {
    if (!Number.isFinite(v) || !Number.isInteger(v) || v < minValue) {
      throw new InvalidParameterError(
        `${name}[${i}] must be an integer >= ${minValue}; received ${String(v)}`,
        name,
        pair
      );
    }
  }
}

/** Output spatial size of a sliding window, or an error if the kernel does not fit. */
function convOutputSize(
  op: string,
  height: number,
  width: number,
  kernelSize: readonly number[],
  stride: readonly number[],
  padding: readonly number[]
): [number, number] {
  const [kH, kW] = kernelSize as [number, number];
  const [sH, sW] = stride as [number, number];
  const [pH, pW] = padding as [number, number];
  const outH = Math.floor((height + 2 * pH - kH) / sH) + 1;
  const outW = Math.floor((width + 2 * pW - kW) / sW) + 1;
  if (outH <= 0 || outW <= 0) {
    throw new InvalidParameterError(
      `${op}: invalid output dimensions ${outH}x${outW}; kernel [${kH}, ${kW}] does not fit ` +
        `an input of ${height}x${width} with padding [${pH}, ${pW}]`,
      "output_dimensions",
      { outH, outW }
    );
  }
  return [outH, outW];
}

/**
 * Image to Column operation (im2col).
 *
 * Rearranges image blocks into columns. Positions that fall in the padding are
 * zero. The output keeps the input dtype.
 *
 * @param input - Input tensor of shape (batch, channels, height, width)
 * @param kernelSize - Size of the kernel [kH, kW]
 * @param stride - Stride [sH, sW]
 * @param padding - Padding [pH, pW]
 * @returns Output tensor of shape (batch, outH * outW, channels * kH * kW)
 * @throws {ShapeError} If `input` is not 4-D
 * @throws {DTypeError} If `input` has string dtype
 * @throws {InvalidParameterError} If a parameter is invalid or the kernel does not fit the input
 *
 * @example
 * ```ts
 * const x = tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]);   // (1, 1, 3, 3)
 * im2col(x, [2, 2], [1, 1], [0, 0]).shape;                    // [1, 4, 4]
 * ```
 */
export function im2col(
  input: Tensor,
  kernelSize: [number, number],
  stride: [number, number],
  padding: [number, number]
): Tensor {
  if (input.ndim !== 4) {
    throw new ShapeError(`im2col expects 4D input, got ${input.ndim}D`);
  }
  if (input.dtype === "string") {
    throw new DTypeError("im2col does not support string tensors");
  }

  validateConvPair("kernelSize", kernelSize, 1);
  validateConvPair("stride", stride, 1);
  validateConvPair("padding", padding, 0);

  const batch = input.shape[0] ?? 0;
  const channels = input.shape[1] ?? 0;
  const height = input.shape[2] ?? 0;
  const width = input.shape[3] ?? 0;

  const [kH, kW] = kernelSize;
  const [sH, sW] = stride;
  const [pH, pW] = padding;

  const [outH, outW] = convOutputSize("im2col", height, width, kernelSize, stride, padding);

  if (input.device !== "cpu") {
    const onDevice = dispatchIm2col(input, kernelSize, stride, padding);
    if (onDevice) return onDevice;
  }

  // Output shape: (batch, outH * outW, channels * kH * kW). This layout allows a
  // matmul with a (channels * kH * kW, out_channels) weight matrix.
  const colSize = channels * kH * kW;
  const outPixels = outH * outW;
  const outShape: Shape = [batch, outPixels, colSize];

  const inputData = input.data;
  if (Array.isArray(inputData)) {
    throw new DTypeError("im2col does not support string tensors");
  }
  const Ctor = dtypeToTypedArrayCtor(input.dtype);
  const outData = new Ctor(batch * outPixels * colSize);

  // The gather below only copies elements, so one loop serves every dtype. int64
  // data is a BigInt64Array on both sides; the `Float64Array` type is only used to
  // satisfy the compiler. Padding positions stay at the zero-initialized value.
  const src = inputData as Float64Array;
  const dst = outData as Float64Array;

  const iStride0 = input.strides[0] ?? 0;
  const iStride1 = input.strides[1] ?? 0;
  const iStride2 = input.strides[2] ?? 0;
  const iStride3 = input.strides[3] ?? 0;
  const kArea = kH * kW;

  for (let b = 0; b < batch; b++) {
    const inputBatchOffset = input.offset + b * iStride0;
    let rowOffset = b * outPixels * colSize;

    for (let oh = 0; oh < outH; oh++) {
      const ihBase = oh * sH - pH;
      for (let ow = 0; ow < outW; ow++) {
        const iwBase = ow * sW - pW;
        // Kernel columns whose input column lies inside the image.
        const kwLo = Math.max(0, -iwBase);
        const kwHi = Math.min(kW, width - iwBase);

        for (let c = 0; c < channels; c++) {
          const channelOffset = inputBatchOffset + c * iStride1;
          const colBase = rowOffset + c * kArea;

          for (let kh = 0; kh < kH; kh++) {
            const ih = ihBase + kh;
            if (ih < 0 || ih >= height) continue;
            const rowSrc = channelOffset + ih * iStride2 + iwBase * iStride3;
            const rowDst = colBase + kh * kW;
            for (let kw = kwLo; kw < kwHi; kw++) {
              dst[rowDst + kw] = src[rowSrc + kw * iStride3] as number;
            }
          }
        }
        rowOffset += colSize;
      }
    }
  }

  return Tensor.fromTypedArray({
    data: outData,
    shape: outShape,
    dtype: input.dtype,
    device: input.device,
  });
}

/**
 * Column to Image operation (col2im).
 *
 * Rearranges columns back into image blocks, summing values where windows
 * overlap. This is the adjoint of {@link im2col} and is used for gradients.
 * The output keeps the dtype of `cols`.
 *
 * @param cols - Column tensor of shape (batch, outH * outW, channels * kH * kW)
 * @param inputShape - Shape of the original image (batch, channels, height, width)
 * @param kernelSize - Size of the kernel [kH, kW]
 * @param stride - Stride [sH, sW]
 * @param padding - Padding [pH, pW]
 * @returns Gradient tensor of shape inputShape
 * @throws {ShapeError} If `cols` is not 3-D, `inputShape` does not have 4 entries, or the
 *   column shape does not match the geometry
 * @throws {DTypeError} If `cols` has string dtype
 * @throws {InvalidParameterError} If a parameter is invalid or the kernel does not fit the image
 */
export function col2im(
  cols: Tensor,
  inputShape: Shape,
  kernelSize: [number, number],
  stride: [number, number],
  padding: [number, number]
): Tensor {
  if (cols.ndim !== 3) {
    throw new ShapeError(`col2im expects 3D input, got ${cols.ndim}D`);
  }
  if (cols.dtype === "string") {
    throw new DTypeError("col2im does not support string tensors");
  }
  if (inputShape.length !== 4) {
    throw new ShapeError(`col2im expects inputShape of length 4, got ${inputShape.length}`);
  }
  for (const [i, d] of inputShape.entries()) {
    if (!Number.isInteger(d) || d < 0) {
      throw new InvalidParameterError(
        `inputShape[${i}] must be a non-negative integer; received ${String(d)}`,
        "inputShape",
        inputShape
      );
    }
  }

  validateConvPair("kernelSize", kernelSize, 1);
  validateConvPair("stride", stride, 1);
  validateConvPair("padding", padding, 0);

  const batch = inputShape[0] ?? 0;
  const channels = inputShape[1] ?? 0;
  const height = inputShape[2] ?? 0;
  const width = inputShape[3] ?? 0;

  const [kH, kW] = kernelSize;
  const [sH, sW] = stride;
  const [pH, pW] = padding;

  const [outH, outW] = convOutputSize("col2im", height, width, kernelSize, stride, padding);

  const colSize = channels * kH * kW;
  const outPixels = outH * outW;

  if (
    (cols.shape[0] ?? 0) !== batch ||
    (cols.shape[1] ?? 0) !== outPixels ||
    (cols.shape[2] ?? 0) !== colSize
  ) {
    throw new ShapeError(
      `col2im input shape mismatch: expected [${batch}, ${outPixels}, ${colSize}], got [${cols.shape}]`
    );
  }

  if (cols.device !== "cpu") {
    const onDevice = dispatchCol2im(cols, inputShape, kernelSize, stride, padding);
    if (onDevice) return onDevice;
  }

  const colsData = cols.data;
  if (Array.isArray(colsData)) {
    throw new DTypeError("col2im does not support string tensors");
  }
  const Ctor = dtypeToTypedArrayCtor(cols.dtype);
  const outData = new Ctor(batch * channels * height * width);

  // One accumulation loop serves every dtype; int64 data is a BigInt64Array on
  // both sides (BigInt + BigInt), the `Float64Array` type only satisfies the compiler.
  const src = colsData as Float64Array;
  const dst = outData as Float64Array;

  const outStrides = computeStrides(inputShape);
  const oStride0 = outStrides[0] ?? 0;
  const oStride1 = outStrides[1] ?? 0;
  const oStride2 = outStrides[2] ?? 0;
  const oStride3 = outStrides[3] ?? 0;

  const cStride0 = cols.strides[0] ?? 0;
  const cStride1 = cols.strides[1] ?? 0;
  const cStride2 = cols.strides[2] ?? 0;
  const kArea = kH * kW;

  for (let b = 0; b < batch; b++) {
    const colsBatchOffset = cols.offset + b * cStride0;
    const outBatchOffset = b * oStride0;

    for (let oh = 0; oh < outH; oh++) {
      const ihBase = oh * sH - pH;
      for (let ow = 0; ow < outW; ow++) {
        const iwBase = ow * sW - pW;
        const kwLo = Math.max(0, -iwBase);
        const kwHi = Math.min(kW, width - iwBase);
        const colsRowOffset = colsBatchOffset + (oh * outW + ow) * cStride1;

        for (let c = 0; c < channels; c++) {
          const outChannelOffset = outBatchOffset + c * oStride1;
          const colBase = colsRowOffset + c * kArea * cStride2;

          for (let kh = 0; kh < kH; kh++) {
            const ih = ihBase + kh;
            if (ih < 0 || ih >= height) continue;
            const rowOut = outChannelOffset + ih * oStride2 + iwBase * oStride3;
            const rowSrc = colBase + kh * kW * cStride2;
            for (let kw = kwLo; kw < kwHi; kw++) {
              const o = rowOut + kw * oStride3;
              dst[o] = (dst[o] as number) + (src[rowSrc + kw * cStride2] as number);
            }
          }
        }
      }
    }
  }

  return Tensor.fromTypedArray({
    data: outData,
    shape: inputShape,
    dtype: cols.dtype,
    device: cols.device,
  });
}
