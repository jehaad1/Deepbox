import {
  DataValidationError,
  type DType,
  DTypeError,
  dtypeToTypedArrayCtor,
  getBigIntElement,
  getNumericElement,
  type Shape,
  ShapeError,
} from "../../core";
import { transpose as tensorTranspose } from "../tensor/shape";
import { type Tensor, Tensor as TensorClass } from "../tensor/Tensor";

const INT64_MIN = -(1n << 63n);
const INT64_MAX = (1n << 63n) - 1n;

function isFloatDType(dtype: DType): boolean {
  return dtype === "float32" || dtype === "float64";
}

type NumericDType = Exclude<DType, "string">;

function resolveDotDtype(a: DType, b: DType): NumericDType {
  if (a === "string" || b === "string") {
    throw new DTypeError("dot is not defined for string dtype");
  }
  if (a === "int64" || b === "int64") {
    if (a !== b) {
      throw new DTypeError(`dot requires matching dtypes; received ${a} and ${b}`);
    }
    return "int64";
  }
  if (a === b) {
    return a;
  }
  if (isFloatDType(a) && isFloatDType(b)) {
    return "float64";
  }
  throw new DTypeError(`dot requires matching dtypes; received ${a} and ${b}`);
}

/**
 * Compute dot product or matrix multiplication.
 *
 * Supported cases:
 * - Both 1-D (vector, vector): inner product, returns a scalar tensor
 * - Both 2-D (matrix, matrix): standard matrix multiplication (m,k) x (k,n) -> (m,n)
 * - 2-D x 1-D (matrix, vector): matrix-vector product (m,k) x (k,) -> (m,)
 * - Both 3-D: batch matrix multiplication (b,m,k) x (b,k,n) -> (b,m,n)
 *
 * Other combinations (e.g., 1-D x 2-D, mixed dimensionalities above 3-D)
 * are not yet implemented and will throw a ShapeError.
 *
 * @param a - First tensor
 * @param b - Second tensor
 * @returns Dot product result
 */
export function dot(a: Tensor, b: Tensor): Tensor {
  const outDtype = resolveDotDtype(a.dtype, b.dtype);
  const isBigInt = outDtype === "int64";

  if (Array.isArray(a.data) || Array.isArray(b.data)) {
    throw new DTypeError("dot is not defined for string dtype");
  }
  const aData = a.data;
  const bData = b.data;

  // Case 1: Both are 1-D vectors (inner product)
  if (a.ndim === 1 && b.ndim === 1) {
    if (a.shape[0] !== b.shape[0]) {
      throw new ShapeError(`shapes ${a.shape} and ${b.shape} not aligned`);
    }
    const size = a.shape[0] ?? 0;
    const aStride = a.strides[0] ?? 0;
    const bStride = b.strides[0] ?? 0;
    if (isBigInt) {
      if (!(aData instanceof BigInt64Array) || !(bData instanceof BigInt64Array)) {
        throw new DTypeError("dot requires int64 dtype");
      }
      const bigA = aData;
      const bigB = bData;
      let sum = 0n;
      for (let i = 0; i < size; i++) {
        sum +=
          getBigIntElement(bigA, a.offset + i * aStride) *
          getBigIntElement(bigB, b.offset + i * bStride);
      }
      if (sum < INT64_MIN || sum > INT64_MAX) {
        throw new DataValidationError("int64 dot overflow");
      }
      const result = new BigInt64Array(1);
      result[0] = sum;
      const scalarShape: Shape = [];
      return TensorClass.fromTypedArray({
        data: result,
        shape: scalarShape,
        dtype: "int64",
        device: a.device,
      });
    }
    if (aData instanceof BigInt64Array || bData instanceof BigInt64Array) {
      throw new DTypeError("dot requires non-int64 dtype");
    }
    const numA = aData;
    const numB = bData;
    let sum = 0;
    for (let i = 0; i < size; i++) {
      sum +=
        getNumericElement(numA, a.offset + i * aStride) *
        getNumericElement(numB, b.offset + i * bStride);
    }
    const Ctor = dtypeToTypedArrayCtor(outDtype);
    const result = new Ctor(1);
    result[0] = sum;
    const scalarShape: Shape = [];
    return TensorClass.fromTypedArray({
      data: result,
      shape: scalarShape,
      dtype: outDtype,
      device: a.device,
    });
  }

  // Case 2: Both are 2-D matrices (matrix multiplication)
  if (a.ndim === 2 && b.ndim === 2) {
    const m = a.shape[0] ?? 0;
    const k1 = a.shape[1] ?? 0;
    const k2 = b.shape[0] ?? 0;
    const n = b.shape[1] ?? 0;

    if (k1 !== k2) {
      throw new ShapeError(
        `shapes ${a.shape} and ${b.shape} not aligned: ${k1} (dim 1) != ${k2} (dim 0)`
      );
    }

    const outSize = m * n;
    const Ctor = dtypeToTypedArrayCtor(outDtype);
    const result = new Ctor(outSize);

    if (isBigInt) {
      if (!(aData instanceof BigInt64Array) || !(bData instanceof BigInt64Array)) {
        throw new DTypeError("dot requires int64 dtype");
      }
      if (!(result instanceof BigInt64Array)) {
        throw new DTypeError("Internal error: expected int64 output buffer");
      }
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          let sum = 0n;
          for (let k = 0; k < k1; k++) {
            const aVal = getBigIntElement(
              aData,
              a.offset + i * (a.strides[0] ?? 0) + k * (a.strides[1] ?? 0)
            );
            const bVal = getBigIntElement(
              bData,
              b.offset + k * (b.strides[0] ?? 0) + j * (b.strides[1] ?? 0)
            );
            sum += aVal * bVal;
          }
          if (sum < INT64_MIN || sum > INT64_MAX) {
            throw new DataValidationError("int64 dot overflow");
          }
          result[i * n + j] = sum;
        }
      }
    } else {
      if (aData instanceof BigInt64Array || bData instanceof BigInt64Array) {
        throw new DTypeError("dot requires non-int64 dtype");
      }
      if (result instanceof BigInt64Array) {
        throw new DTypeError("Internal error: unexpected int64 output buffer");
      }
      // Fast path: contiguous row-major layout — tiled i-k-j loop for cache locality
      const aS0 = a.strides[0] ?? 0;
      const aS1 = a.strides[1] ?? 0;
      const bS0 = b.strides[0] ?? 0;
      const bS1 = b.strides[1] ?? 0;
      const aOff = a.offset;
      const bOff = b.offset;
      const numA = aData;
      const numB = bData;
      const TILE = 32;
      for (let ii = 0; ii < m; ii += TILE) {
        const iEnd = ii + TILE < m ? ii + TILE : m;
        for (let kk = 0; kk < k1; kk += TILE) {
          const kEnd = kk + TILE < k1 ? kk + TILE : k1;
          for (let jj = 0; jj < n; jj += TILE) {
            const jEnd = jj + TILE < n ? jj + TILE : n;
            for (let i = ii; i < iEnd; i++) {
              const aBase = aOff + i * aS0;
              const rBase = i * n;
              for (let k = kk; k < kEnd; k++) {
                const aVal = numA[aBase + k * aS1] as number;
                const bBase = bOff + k * bS0;
                for (let j = jj; j < jEnd; j++) {
                  result[rBase + j] =
                    (result[rBase + j] as number) + aVal * (numB[bBase + j * bS1] as number);
                }
              }
            }
          }
        }
      }
    }

    return TensorClass.fromTypedArray({
      data: result,
      shape: [m, n],
      dtype: outDtype,
      device: a.device,
    });
  }

  // Case 3: Matrix-vector multiplication (2-D x 1-D)
  if (a.ndim === 2 && b.ndim === 1) {
    const m = a.shape[0] ?? 0;
    const k1 = a.shape[1] ?? 0;
    const k2 = b.shape[0] ?? 0;

    if (k1 !== k2) {
      throw new ShapeError(`shapes ${a.shape} and ${b.shape} not aligned`);
    }

    const Ctor = dtypeToTypedArrayCtor(outDtype);
    const result = new Ctor(m);

    if (isBigInt) {
      if (!(aData instanceof BigInt64Array) || !(bData instanceof BigInt64Array)) {
        throw new DTypeError("dot requires int64 dtype");
      }
      if (!(result instanceof BigInt64Array)) {
        throw new DTypeError("Internal error: expected int64 output buffer");
      }
      for (let i = 0; i < m; i++) {
        let sum = 0n;
        for (let k = 0; k < k1; k++) {
          const aVal = getBigIntElement(
            aData,
            a.offset + i * (a.strides[0] ?? 0) + k * (a.strides[1] ?? 0)
          );
          const bVal = getBigIntElement(bData, b.offset + k * (b.strides[0] ?? 0));
          sum += aVal * bVal;
        }
        if (sum < INT64_MIN || sum > INT64_MAX) {
          throw new DataValidationError("int64 dot overflow");
        }
        result[i] = sum;
      }
    } else {
      if (aData instanceof BigInt64Array || bData instanceof BigInt64Array) {
        throw new DTypeError("dot requires non-int64 dtype");
      }
      if (result instanceof BigInt64Array) {
        throw new DTypeError("Internal error: unexpected int64 output buffer");
      }
      for (let i = 0; i < m; i++) {
        let sum = 0;
        for (let k = 0; k < k1; k++) {
          const aVal = getNumericElement(
            aData,
            a.offset + i * (a.strides[0] ?? 0) + k * (a.strides[1] ?? 0)
          );
          const bVal = getNumericElement(bData, b.offset + k * (b.strides[0] ?? 0));
          sum += aVal * bVal;
        }
        result[i] = sum;
      }
    }

    return TensorClass.fromTypedArray({
      data: result,
      shape: [m],
      dtype: outDtype,
      device: a.device,
    });
  }

  // Case 4: Higher dimensional tensors (batched matmul)
  if (a.ndim >= 3 || b.ndim >= 3) {
    if (a.ndim < 2 || b.ndim < 2) {
      throw new ShapeError(`dot not implemented for shapes ${a.shape} and ${b.shape}`);
    }

    const aBatchRank = Math.max(0, a.ndim - 2);
    const bBatchRank = Math.max(0, b.ndim - 2);
    const aBatchShape = a.shape.slice(0, aBatchRank);
    const bBatchShape = b.shape.slice(0, bBatchRank);

    let batchShape: number[];
    if (aBatchRank > 0 && bBatchRank > 0) {
      if (aBatchRank !== bBatchRank) {
        throw new ShapeError(`batch dimensions don't match: [${aBatchShape}] vs [${bBatchShape}]`);
      }
      for (let i = 0; i < aBatchRank; i++) {
        if (aBatchShape[i] !== bBatchShape[i]) {
          throw new ShapeError(
            `batch dimensions don't match: [${aBatchShape}] vs [${bBatchShape}]`
          );
        }
      }
      batchShape = aBatchShape;
    } else if (aBatchRank > 0) {
      batchShape = aBatchShape;
    } else if (bBatchRank > 0) {
      batchShape = bBatchShape;
    } else {
      throw new ShapeError(`dot not implemented for shapes ${a.shape} and ${b.shape}`);
    }

    const m = a.shape[a.ndim - 2] ?? 0;
    const k1 = a.shape[a.ndim - 1] ?? 0;
    const k2 = b.shape[b.ndim - 2] ?? 0;
    const n = b.shape[b.ndim - 1] ?? 0;

    if (k1 !== k2) {
      throw new ShapeError(`shapes not aligned for matmul`);
    }

    let batchSize = 1;
    for (const dim of batchShape) {
      batchSize *= dim;
    }

    const outShape = batchShape.length === 0 ? [m, n] : [...batchShape, m, n];
    const outSize = batchSize * m * n;
    const Ctor = dtypeToTypedArrayCtor(outDtype);
    const result = new Ctor(outSize);

    const aStrideM = a.strides[a.ndim - 2];
    const aStrideK = a.strides[a.ndim - 1];
    const bStrideK = b.strides[b.ndim - 2];
    const bStrideN = b.strides[b.ndim - 1];

    if (aStrideM === undefined || aStrideK === undefined) {
      throw new ShapeError("Internal error: missing strides for left operand");
    }
    if (bStrideK === undefined || bStrideN === undefined) {
      throw new ShapeError("Internal error: missing strides for right operand");
    }

    const aBatchStrides = aBatchRank > 0 ? a.strides.slice(0, aBatchRank) : [];
    const bBatchStrides = bBatchRank > 0 ? b.strides.slice(0, bBatchRank) : [];

    const batchOffset = (
      index: number,
      shape: readonly number[],
      strides: readonly number[],
      baseOffset: number
    ): number => {
      if (shape.length === 0) return baseOffset;
      let offset = baseOffset;
      let remaining = index;
      for (let d = shape.length - 1; d >= 0; d--) {
        const dim = shape[d] ?? 0;
        const stride = strides[d];
        if (stride === undefined) {
          throw new ShapeError("Internal error: missing batch stride");
        }
        if (dim === 0) {
          return baseOffset;
        }
        const idx = remaining % dim;
        remaining = Math.floor(remaining / dim);
        offset += idx * stride;
      }
      return offset;
    };

    for (let b_idx = 0; b_idx < batchSize; b_idx++) {
      const aOffset =
        aBatchRank > 0 ? batchOffset(b_idx, batchShape, aBatchStrides, a.offset) : a.offset;
      const bOffset =
        bBatchRank > 0 ? batchOffset(b_idx, batchShape, bBatchStrides, b.offset) : b.offset;

      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          if (isBigInt) {
            if (!(aData instanceof BigInt64Array) || !(bData instanceof BigInt64Array)) {
              throw new DTypeError("dot requires int64 dtype");
            }
            if (!(result instanceof BigInt64Array)) {
              throw new DTypeError("Internal error: expected int64 output buffer");
            }
            let sum = 0n;
            for (let k = 0; k < k1; k++) {
              const aVal = getBigIntElement(aData, aOffset + i * aStrideM + k * aStrideK);
              const bVal = getBigIntElement(bData, bOffset + k * bStrideK + j * bStrideN);
              sum += aVal * bVal;
            }
            const outIndex = b_idx * (m * n) + i * n + j;
            if (sum < INT64_MIN || sum > INT64_MAX) {
              throw new DataValidationError("int64 dot overflow");
            }
            result[outIndex] = sum;
          } else {
            if (aData instanceof BigInt64Array || bData instanceof BigInt64Array) {
              throw new DTypeError("dot requires non-int64 dtype");
            }
            if (result instanceof BigInt64Array) {
              throw new DTypeError("Internal error: unexpected int64 output buffer");
            }
            let sum = 0;
            for (let k = 0; k < k1; k++) {
              const aVal = getNumericElement(aData, aOffset + i * aStrideM + k * aStrideK);
              const bVal = getNumericElement(bData, bOffset + k * bStrideK + j * bStrideN);
              sum += aVal * bVal;
            }
            const outIndex = b_idx * (m * n) + i * n + j;
            result[outIndex] = sum;
          }
        }
      }
    }

    return TensorClass.fromTypedArray({
      data: result,
      shape: outShape,
      dtype: outDtype,
      device: a.device,
    });
  }

  throw new ShapeError(`dot not implemented for shapes ${a.shape} and ${b.shape}`);
}

/**
 * Transpose a tensor by reversing or permuting its axes.
 *
 * @param t - Input tensor
 * @param axes - Permutation of axes (optional, defaults to reversing all axes)
 * @returns Transposed tensor
 *
 * @example
 * ```ts
 * const t = tensor([[1, 2], [3, 4]]);
 * const tT = transpose(t);  // [[1, 3], [2, 4]]
 * ```
 */
export function transpose(t: Tensor, axes?: number[]): Tensor {
  return tensorTranspose(t, axes);
}
