/**
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox documentation}
 */

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
import { dispatchDot } from "../ops/device_dispatch";
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
 * - 1-D x 2-D (vector, matrix): vector-matrix product (k,) x (k,n) -> (n,)
 * - 3-D and higher: batch matrix multiplication, e.g. (b,m,k) x (b,k,n) -> (b,m,n)
 *
 * Other combinations (e.g., mixing a 1-D operand with a batched 3-D+ operand)
 * are not yet implemented and will throw a ShapeError.
 *
 * @param a - First tensor
 * @param b - Second tensor
 * @returns Dot product result
 */
type NumericOut = Exclude<import("../tensor/Tensor").Tensor["data"], string[] | BigInt64Array>;

/**
 * Register-blocked GEMM for unit inner strides: 2 output rows x 4 B-rows per
 * step. Each rowAcc load/store is amortized over 8 multiply-adds (vs 1 in the
 * plain i-k-j kernel), which is ~3x on large matrices.
 */
function gemmBlocked(
  A: Float64Array | Float32Array,
  aOff: number,
  aS0: number,
  B: Float64Array | Float32Array,
  bOff: number,
  bS0: number,
  out: NumericOut,
  m: number,
  k: number,
  n: number
): void {
  const r0 = new Float64Array(n);
  const r1 = new Float64Array(n);
  let i = 0;
  for (; i + 2 <= m; i += 2) {
    r0.fill(0);
    r1.fill(0);
    const aB0 = aOff + i * aS0;
    const aB1 = aB0 + aS0;
    let p = 0;
    for (; p + 4 <= k; p += 4) {
      const a00 = A[aB0 + p] as number;
      const a01 = A[aB0 + p + 1] as number;
      const a02 = A[aB0 + p + 2] as number;
      const a03 = A[aB0 + p + 3] as number;
      const a10 = A[aB1 + p] as number;
      const a11 = A[aB1 + p + 1] as number;
      const a12 = A[aB1 + p + 2] as number;
      const a13 = A[aB1 + p + 3] as number;
      const b0 = bOff + p * bS0;
      const b1 = b0 + bS0;
      const b2 = b1 + bS0;
      const b3 = b2 + bS0;
      for (let j = 0; j < n; j++) {
        const x0 = B[b0 + j] as number;
        const x1 = B[b1 + j] as number;
        const x2 = B[b2 + j] as number;
        const x3 = B[b3 + j] as number;
        r0[j] = (r0[j] as number) + a00 * x0 + a01 * x1 + a02 * x2 + a03 * x3;
        r1[j] = (r1[j] as number) + a10 * x0 + a11 * x1 + a12 * x2 + a13 * x3;
      }
    }
    for (; p < k; p++) {
      const a0 = A[aB0 + p] as number;
      const a1 = A[aB1 + p] as number;
      const bB = bOff + p * bS0;
      for (let j = 0; j < n; j++) {
        const x = B[bB + j] as number;
        r0[j] = (r0[j] as number) + a0 * x;
        r1[j] = (r1[j] as number) + a1 * x;
      }
    }
    const rBase = i * n;
    for (let j = 0; j < n; j++) {
      out[rBase + j] = r0[j] as number;
      out[rBase + n + j] = r1[j] as number;
    }
  }
  for (; i < m; i++) {
    r0.fill(0);
    const aBase = aOff + i * aS0;
    for (let p = 0; p < k; p++) {
      const aVal = A[aBase + p] as number;
      const bB = bOff + p * bS0;
      for (let j = 0; j < n; j++) {
        r0[j] = (r0[j] as number) + aVal * (B[bB + j] as number);
      }
    }
    const rBase = i * n;
    for (let j = 0; j < n; j++) out[rBase + j] = r0[j] as number;
  }
}

function gemmF64(
  A: Float64Array,
  aOff: number,
  aS0: number,
  aS1: number,
  B: Float64Array,
  bOff: number,
  bS0: number,
  bS1: number,
  out: NumericOut,
  m: number,
  k: number,
  n: number
): void {
  if (aS1 === 1 && bS1 === 1) {
    gemmBlocked(A, aOff, aS0, B, bOff, bS0, out, m, k, n);
    return;
  }
  const rowAcc = new Float64Array(n);
  for (let i = 0; i < m; i++) {
    rowAcc.fill(0);
    const aBase = aOff + i * aS0;
    for (let p = 0; p < k; p++) {
      const aVal = A[aBase + p * aS1] ?? 0;
      if (aVal === 0) continue;
      const bBase = bOff + p * bS0;
      if (bS1 === 1) {
        for (let j = 0; j < n; j++) rowAcc[j]! += aVal * B[bBase + j]!;
      } else {
        for (let j = 0; j < n; j++) rowAcc[j]! += aVal * B[bBase + j * bS1]!;
      }
    }
    const rBase = i * n;
    for (let j = 0; j < n; j++) out[rBase + j] = rowAcc[j]!;
  }
}

function gemmF32(
  A: Float32Array,
  aOff: number,
  aS0: number,
  aS1: number,
  B: Float32Array,
  bOff: number,
  bS0: number,
  bS1: number,
  out: NumericOut,
  m: number,
  k: number,
  n: number
): void {
  if (aS1 === 1 && bS1 === 1) {
    gemmBlocked(A, aOff, aS0, B, bOff, bS0, out, m, k, n);
    return;
  }
  const rowAcc = new Float64Array(n);
  for (let i = 0; i < m; i++) {
    rowAcc.fill(0);
    const aBase = aOff + i * aS0;
    for (let p = 0; p < k; p++) {
      const aVal = A[aBase + p * aS1] ?? 0;
      if (aVal === 0) continue;
      const bBase = bOff + p * bS0;
      if (bS1 === 1) {
        for (let j = 0; j < n; j++) rowAcc[j]! += aVal * B[bBase + j]!;
      } else {
        for (let j = 0; j < n; j++) rowAcc[j]! += aVal * B[bBase + j * bS1]!;
      }
    }
    const rBase = i * n;
    for (let j = 0; j < n; j++) out[rBase + j] = rowAcc[j]!;
  }
}

function gemmGeneric(
  A: ArrayLike<number>,
  aOff: number,
  aS0: number,
  aS1: number,
  B: ArrayLike<number>,
  bOff: number,
  bS0: number,
  bS1: number,
  out: NumericOut,
  m: number,
  k: number,
  n: number
): void {
  const rowAcc = new Float64Array(n);
  for (let i = 0; i < m; i++) {
    rowAcc.fill(0);
    const aBase = aOff + i * aS0;
    for (let p = 0; p < k; p++) {
      const aVal = A[aBase + p * aS1] ?? 0;
      if (aVal === 0) continue;
      const bBase = bOff + p * bS0;
      for (let j = 0; j < n; j++) rowAcc[j]! += aVal * (B[bBase + j * bS1] ?? 0);
    }
    const rBase = i * n;
    for (let j = 0; j < n; j++) out[rBase + j] = rowAcc[j]!;
  }
}

export function dot(a: Tensor, b: Tensor): Tensor {
  const outDtype = resolveDotDtype(a.dtype, b.dtype);
  const isBigInt = outDtype === "int64";

  if (a.device !== "cpu" || b.device !== "cpu") {
    const onDevice = dispatchDot(a, b);
    if (onDevice) return onDevice;
  }

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
      // Monomorphic i-k-j kernels with a float64 row accumulator (see
      // gemmF64/gemmF32): avoids read-modify-write on the output per FLOP,
      // walks B rows contiguously, and keeps each kernel's typed-array
      // accesses monomorphic for V8.
      const aS0 = a.strides[0] ?? 0;
      const aS1 = a.strides[1] ?? 0;
      const bS0 = b.strides[0] ?? 0;
      const bS1 = b.strides[1] ?? 0;
      if (aData instanceof Float64Array && bData instanceof Float64Array) {
        gemmF64(aData, a.offset, aS0, aS1, bData, b.offset, bS0, bS1, result, m, k1, n);
      } else if (aData instanceof Float32Array && bData instanceof Float32Array) {
        gemmF32(aData, a.offset, aS0, aS1, bData, b.offset, bS0, bS1, result, m, k1, n);
      } else {
        gemmGeneric(aData, a.offset, aS0, aS1, bData, b.offset, bS0, bS1, result, m, k1, n);
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

  // Case 4: Vector-matrix multiplication (1-D x 2-D)
  if (a.ndim === 1 && b.ndim === 2) {
    const k1 = a.shape[0] ?? 0;
    const k2 = b.shape[0] ?? 0;
    const n = b.shape[1] ?? 0;

    if (k1 !== k2) {
      throw new ShapeError(`shapes ${a.shape} and ${b.shape} not aligned`);
    }

    const Ctor = dtypeToTypedArrayCtor(outDtype);
    const result = new Ctor(n);

    if (isBigInt) {
      if (!(aData instanceof BigInt64Array) || !(bData instanceof BigInt64Array)) {
        throw new DTypeError("dot requires int64 dtype");
      }
      if (!(result instanceof BigInt64Array)) {
        throw new DTypeError("Internal error: expected int64 output buffer");
      }
      for (let j = 0; j < n; j++) {
        let sum = 0n;
        for (let k = 0; k < k1; k++) {
          const aVal = getBigIntElement(aData, a.offset + k * (a.strides[0] ?? 0));
          const bVal = getBigIntElement(
            bData,
            b.offset + k * (b.strides[0] ?? 0) + j * (b.strides[1] ?? 0)
          );
          sum += aVal * bVal;
        }
        if (sum < INT64_MIN || sum > INT64_MAX) {
          throw new DataValidationError("int64 dot overflow");
        }
        result[j] = sum;
      }
    } else {
      if (aData instanceof BigInt64Array || bData instanceof BigInt64Array) {
        throw new DTypeError("dot requires non-int64 dtype");
      }
      if (result instanceof BigInt64Array) {
        throw new DTypeError("Internal error: unexpected int64 output buffer");
      }
      for (let j = 0; j < n; j++) {
        let sum = 0;
        for (let k = 0; k < k1; k++) {
          const aVal = getNumericElement(aData, a.offset + k * (a.strides[0] ?? 0));
          const bVal = getNumericElement(
            bData,
            b.offset + k * (b.strides[0] ?? 0) + j * (b.strides[1] ?? 0)
          );
          sum += aVal * bVal;
        }
        result[j] = sum;
      }
    }

    return TensorClass.fromTypedArray({
      data: result,
      shape: [n],
      dtype: outDtype,
      device: a.device,
    });
  }

  // Case 5: Higher dimensional tensors (batched matmul)
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
