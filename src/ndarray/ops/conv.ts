import type { Shape } from "../../core";
import {
  DTypeError,
  dtypeToTypedArrayCtor,
  getBigIntElement,
  getNumericElement,
  InvalidParameterError,
  ShapeError,
} from "../../core";
import { computeStrides, Tensor } from "../tensor/Tensor";

function validateConvPair(
  name: "kernelSize" | "stride" | "padding",
  pair: [number, number],
  minValue: number
): void {
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

/**
 * Image to Column operation (im2col).
 *
 * Rearranges image blocks into columns.
 *
 * @param input - Input tensor of shape (batch, channels, height, width)
 * @param kernelSize - Size of the kernel [kH, kW]
 * @param stride - Stride [sH, sW]
 * @param padding - Padding [pH, pW]
 * @returns Output tensor of shape (batch, outH * outW, channels * kH * kW)
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

  const outH = Math.floor((height + 2 * pH - kH) / sH) + 1;
  const outW = Math.floor((width + 2 * pW - kW) / sW) + 1;

  if (outH <= 0 || outW <= 0) {
    throw new InvalidParameterError(
      `Invalid output dimensions: ${outH}x${outW}`,
      "output_dimensions",
      { outH, outW }
    );
  }

  // Output shape: (batch, outH * outW, channels * kH * kW)
  // This shape allows matmul with weight matrix (channels * kH * kW, out_channels)
  const colSize = channels * kH * kW;
  const outPixels = outH * outW;
  const outShape: Shape = [batch, outPixels, colSize];

  const Ctor = dtypeToTypedArrayCtor(input.dtype);
  const outData = new Ctor(batch * outPixels * colSize);

  const inputData = input.data;
  if (Array.isArray(inputData)) {
    throw new DTypeError("im2col does not support string tensors");
  }

  const iStride0 = input.strides[0] ?? 0;
  const iStride1 = input.strides[1] ?? 0;
  const iStride2 = input.strides[2] ?? 0;
  const iStride3 = input.strides[3] ?? 0;

  // Optimized loop structure
  // We want to fill the output row by row
  // Each row in output corresponds to one sliding window position
  // The row contains flattened channel, kH, kW data

  // To match typical matmul convention:
  // Input unfolded: (batch, out_pixels, in_features)
  // Weight: (in_features, out_channels)
  // Result: (batch, out_pixels, out_channels)

  if (inputData instanceof BigInt64Array) {
    if (!(outData instanceof BigInt64Array)) {
      throw new DTypeError("im2col expected int64 output buffer");
    }
    const outBig = outData;

    for (let b = 0; b < batch; b++) {
      const batchOffset = b * outPixels * colSize;
      const inputBatchOffset = input.offset + b * iStride0;

      for (let oh = 0; oh < outH; oh++) {
        for (let ow = 0; ow < outW; ow++) {
          const outRowIdx = oh * outW + ow;
          const rowOffset = batchOffset + outRowIdx * colSize;
          let colIdx = 0;

          for (let c = 0; c < channels; c++) {
            const inputChannelOffset = inputBatchOffset + c * iStride1;

            for (let kh = 0; kh < kH; kh++) {
              const ih = oh * sH - pH + kh;

              for (let kw = 0; kw < kW; kw++) {
                const iw = ow * sW - pW + kw;

                let val = 0n;
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                  const inputOffset = inputChannelOffset + ih * iStride2 + iw * iStride3;
                  val = getBigIntElement(inputData, inputOffset);
                }

                outBig[rowOffset + colIdx] = val;
                colIdx++;
              }
            }
          }
        }
      }
    }
  } else {
    if (outData instanceof BigInt64Array) {
      throw new DTypeError("im2col unexpected int64 output buffer");
    }
    const outNum = outData;

    for (let b = 0; b < batch; b++) {
      const batchOffset = b * outPixels * colSize;
      const inputBatchOffset = input.offset + b * iStride0;

      for (let oh = 0; oh < outH; oh++) {
        for (let ow = 0; ow < outW; ow++) {
          const outRowIdx = oh * outW + ow;
          const rowOffset = batchOffset + outRowIdx * colSize;
          let colIdx = 0;

          for (let c = 0; c < channels; c++) {
            const inputChannelOffset = inputBatchOffset + c * iStride1;

            for (let kh = 0; kh < kH; kh++) {
              const ih = oh * sH - pH + kh;

              for (let kw = 0; kw < kW; kw++) {
                const iw = ow * sW - pW + kw;

                let val = 0;
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                  const inputOffset = inputChannelOffset + ih * iStride2 + iw * iStride3;
                  val = getNumericElement(inputData, inputOffset);
                }

                outNum[rowOffset + colIdx] = val;
                colIdx++;
              }
            }
          }
        }
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
 * Rearranges columns back into image blocks (accumulating values).
 * Used for gradient computation.
 *
 * @param cols - Column tensor of shape (batch, outH * outW, channels * kH * kW)
 * @param inputShape - Shape of the original image (batch, channels, height, width)
 * @param kernelSize - Size of the kernel [kH, kW]
 * @param stride - Stride [sH, sW]
 * @param padding - Padding [pH, pW]
 * @returns Gradient tensor of shape inputShape
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

  const outH = Math.floor((height + 2 * pH - kH) / sH) + 1;
  const outW = Math.floor((width + 2 * pW - kW) / sW) + 1;
  if (outH <= 0 || outW <= 0) {
    throw new InvalidParameterError(
      `Invalid output dimensions: ${outH}x${outW}`,
      "output_dimensions",
      { outH, outW }
    );
  }

  const colSize = channels * kH * kW;
  const outPixels = outH * outW;

  // Validate cols shape
  if (
    (cols.shape[0] ?? 0) !== batch ||
    (cols.shape[1] ?? 0) !== outPixels ||
    (cols.shape[2] ?? 0) !== colSize
  ) {
    throw new ShapeError(
      `col2im input shape mismatch: expected [${batch}, ${outPixels}, ${colSize}], got [${cols.shape}]`
    );
  }

  const Ctor = dtypeToTypedArrayCtor(cols.dtype);
  const outData = new Ctor(batch * channels * height * width);

  const colsData = cols.data;
  if (Array.isArray(colsData)) {
    throw new DTypeError("col2im does not support string tensors");
  }

  // Calculate strides for the output tensor (which we are filling)
  const outStrides = computeStrides(inputShape);
  const oStride0 = outStrides[0] ?? 0;
  const oStride1 = outStrides[1] ?? 0;
  const oStride2 = outStrides[2] ?? 0;
  const oStride3 = outStrides[3] ?? 0;

  const cStride0 = cols.strides[0] ?? 0;
  const cStride1 = cols.strides[1] ?? 0;
  const cStride2 = cols.strides[2] ?? 0;

  if (colsData instanceof BigInt64Array) {
    if (!(outData instanceof BigInt64Array)) {
      throw new DTypeError("col2im expected int64 output buffer");
    }
    const outBig = outData;

    for (let b = 0; b < batch; b++) {
      const colsBatchOffset = cols.offset + b * cStride0;
      const outBatchOffset = b * oStride0;

      for (let oh = 0; oh < outH; oh++) {
        for (let ow = 0; ow < outW; ow++) {
          const rowIdx = oh * outW + ow;
          const colsRowOffset = colsBatchOffset + rowIdx * cStride1;

          let colIdx = 0;
          for (let c = 0; c < channels; c++) {
            const outChannelOffset = outBatchOffset + c * oStride1;

            for (let kh = 0; kh < kH; kh++) {
              const ih = oh * sH - pH + kh;
              for (let kw = 0; kw < kW; kw++) {
                const iw = ow * sW - pW + kw;

                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                  const val = getBigIntElement(colsData, colsRowOffset + colIdx * cStride2);
                  const outOffset = outChannelOffset + ih * oStride2 + iw * oStride3;
                  const prev = outBig[outOffset];
                  outBig[outOffset] = (prev === undefined ? 0n : prev) + val;
                }
                colIdx++;
              }
            }
          }
        }
      }
    }
  } else {
    if (outData instanceof BigInt64Array) {
      throw new DTypeError("col2im unexpected int64 output buffer");
    }
    const outNum = outData;

    for (let b = 0; b < batch; b++) {
      const colsBatchOffset = cols.offset + b * cStride0;
      const outBatchOffset = b * oStride0;

      for (let oh = 0; oh < outH; oh++) {
        for (let ow = 0; ow < outW; ow++) {
          const rowIdx = oh * outW + ow;
          const colsRowOffset = colsBatchOffset + rowIdx * cStride1;

          let colIdx = 0;
          for (let c = 0; c < channels; c++) {
            const outChannelOffset = outBatchOffset + c * oStride1;

            for (let kh = 0; kh < kH; kh++) {
              const ih = oh * sH - pH + kh;
              for (let kw = 0; kw < kW; kw++) {
                const iw = ow * sW - pW + kw;

                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                  const val = getNumericElement(colsData, colsRowOffset + colIdx * cStride2);
                  const outOffset = outChannelOffset + ih * oStride2 + iw * oStride3;
                  const prev = outNum[outOffset];
                  outNum[outOffset] = (prev === undefined ? 0 : prev) + val;
                }
                colIdx++;
              }
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
