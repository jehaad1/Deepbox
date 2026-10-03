/**
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox documentation}
 */

import {
  DataValidationError,
  type DType,
  DTypeError,
  dtypeToTypedArrayCtor,
  getBigIntElement,
  type Shape,
} from "../../core";
import { roundHalfResult } from "../ops/_internal";
import { dispatchDot } from "../ops/device_dispatch";
import { transpose as tensorTranspose } from "../tensor/shape";
import { type Tensor, Tensor as TensorClass } from "../tensor/Tensor";
import { type MatmulOp, planMatmul } from "./matmul_plan";

const INT64_MIN = -(1n << 63n);
const INT64_MAX = (1n << 63n) - 1n;

function isFloatDType(dtype: DType): boolean {
  return dtype === "float32" || dtype === "float64";
}

type NumericDType = Exclude<DType, "string">;

type NumericOut = Exclude<import("../tensor/Tensor").Tensor["data"], string[] | BigInt64Array>;

/** Name of the public operation, used in error messages. */
export type ContractionOp = MatmulOp;

/**
 * Output dtype of a matrix product.
 *
 * Equal dtypes are kept as they are, float32 mixed with float64 gives float64,
 * and every other combination (including int64 with a non-int64 dtype) is
 * rejected so integer products are never silently routed through floats.
 */
function resolveDotDtype(op: ContractionOp, a: DType, b: DType): NumericDType {
  if (a === "string" || b === "string") {
    throw new DTypeError(`${op} is not defined for string dtype`);
  }
  if (a === "int64" || b === "int64") {
    if (a !== b) {
      throw new DTypeError(`${op} requires matching dtypes; received ${a} and ${b}`);
    }
    return "int64";
  }
  if (a === b) {
    return a;
  }
  if (isFloatDType(a) && isFloatDType(b)) {
    return "float64";
  }
  throw new DTypeError(`${op} requires matching dtypes; received ${a} and ${b}`);
}

/**
 * Register-blocked GEMM for unit inner strides: 2 output rows x 4 B-rows per
 * step. Each rowAcc load/store is amortized over 8 multiply-adds (vs 1 in the
 * plain i-k-j kernel), which is ~3x on large matrices.
 *
 * Every output element is accumulated in float64 in ascending `p` order, so
 * the result matches a plain sequential dot product bit for bit.
 */
function gemmBlocked(
  A: Float64Array | Float32Array,
  aOff: number,
  aS0: number,
  B: Float64Array | Float32Array,
  bOff: number,
  bS0: number,
  out: NumericOut,
  oOff: number,
  m: number,
  k: number,
  n: number,
  r0: Float64Array,
  r1: Float64Array
): void {
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
    const rBase = oOff + i * n;
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
    const rBase = oOff + i * n;
    for (let j = 0; j < n; j++) out[rBase + j] = r0[j] as number;
  }
}

/**
 * Strided i-k-j kernel for float64 operands with a float64 row accumulator.
 *
 * Zero entries of A are deliberately not skipped: `0 * Infinity` and
 * `0 * NaN` must propagate NaN exactly like a reference matmul does.
 */
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
  oOff: number,
  m: number,
  k: number,
  n: number,
  rowAcc: Float64Array
): void {
  for (let i = 0; i < m; i++) {
    rowAcc.fill(0);
    const aBase = aOff + i * aS0;
    for (let p = 0; p < k; p++) {
      const aVal = A[aBase + p * aS1] as number;
      const bBase = bOff + p * bS0;
      if (bS1 === 1) {
        for (let j = 0; j < n; j++)
          rowAcc[j] = (rowAcc[j] as number) + aVal * (B[bBase + j] as number);
      } else {
        for (let j = 0; j < n; j++) {
          rowAcc[j] = (rowAcc[j] as number) + aVal * (B[bBase + j * bS1] as number);
        }
      }
    }
    const rBase = oOff + i * n;
    for (let j = 0; j < n; j++) out[rBase + j] = rowAcc[j] as number;
  }
}

/** Strided i-k-j kernel for float32 operands (float64 accumulation). */
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
  oOff: number,
  m: number,
  k: number,
  n: number,
  rowAcc: Float64Array
): void {
  for (let i = 0; i < m; i++) {
    rowAcc.fill(0);
    const aBase = aOff + i * aS0;
    for (let p = 0; p < k; p++) {
      const aVal = A[aBase + p * aS1] as number;
      const bBase = bOff + p * bS0;
      if (bS1 === 1) {
        for (let j = 0; j < n; j++)
          rowAcc[j] = (rowAcc[j] as number) + aVal * (B[bBase + j] as number);
      } else {
        for (let j = 0; j < n; j++) {
          rowAcc[j] = (rowAcc[j] as number) + aVal * (B[bBase + j * bS1] as number);
        }
      }
    }
    const rBase = oOff + i * n;
    for (let j = 0; j < n; j++) out[rBase + j] = rowAcc[j] as number;
  }
}

/** Strided i-k-j kernel for any other pair of numeric typed arrays. */
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
  oOff: number,
  m: number,
  k: number,
  n: number,
  rowAcc: Float64Array
): void {
  for (let i = 0; i < m; i++) {
    rowAcc.fill(0);
    const aBase = aOff + i * aS0;
    for (let p = 0; p < k; p++) {
      const aVal = A[aBase + p * aS1] as number;
      const bBase = bOff + p * bS0;
      for (let j = 0; j < n; j++) {
        rowAcc[j] = (rowAcc[j] as number) + aVal * (B[bBase + j * bS1] as number);
      }
    }
    const rBase = oOff + i * n;
    for (let j = 0; j < n; j++) out[rBase + j] = rowAcc[j] as number;
  }
}

/** Matrix-vector kernel (n === 1): one sequential float64 dot product per output row. */
function gemv(
  A: ArrayLike<number>,
  aOff: number,
  aS0: number,
  aS1: number,
  B: ArrayLike<number>,
  bOff: number,
  bS0: number,
  out: NumericOut,
  oOff: number,
  m: number,
  k: number
): void {
  for (let i = 0; i < m; i++) {
    const aBase = aOff + i * aS0;
    let sum = 0;
    for (let p = 0; p < k; p++) {
      sum += (A[aBase + p * aS1] as number) * (B[bOff + p * bS0] as number);
    }
    out[oOff + i] = sum;
  }
}

/**
 * int32 kernel with exact two's-complement wraparound (NumPy int32 semantics).
 * Accumulating in float64 would lose the low bits once partial sums pass 2^53.
 */
function gemmInt32(
  A: Int32Array,
  aOff: number,
  aS0: number,
  aS1: number,
  B: Int32Array,
  bOff: number,
  bS0: number,
  bS1: number,
  out: Int32Array,
  oOff: number,
  m: number,
  k: number,
  n: number,
  rowAcc: Int32Array
): void {
  for (let i = 0; i < m; i++) {
    rowAcc.fill(0);
    const aBase = aOff + i * aS0;
    for (let p = 0; p < k; p++) {
      const aVal = A[aBase + p * aS1] as number;
      const bBase = bOff + p * bS0;
      for (let j = 0; j < n; j++) {
        rowAcc[j] = ((rowAcc[j] as number) + Math.imul(aVal, B[bBase + j * bS1] as number)) | 0;
      }
    }
    const rBase = oOff + i * n;
    for (let j = 0; j < n; j++) out[rBase + j] = rowAcc[j] as number;
  }
}

/** bool kernel: logical OR of ANDs (NumPy bool matmul). */
function gemmBool(
  A: Uint8Array,
  aOff: number,
  aS0: number,
  aS1: number,
  B: Uint8Array,
  bOff: number,
  bS0: number,
  bS1: number,
  out: Uint8Array,
  oOff: number,
  m: number,
  k: number,
  n: number
): void {
  for (let i = 0; i < m; i++) {
    const aBase = aOff + i * aS0;
    for (let j = 0; j < n; j++) {
      let any = 0;
      for (let p = 0; p < k; p++) {
        if (A[aBase + p * aS1] !== 0 && B[bOff + p * bS0 + j * bS1] !== 0) {
          any = 1;
          break;
        }
      }
      out[oOff + i * n + j] = any;
    }
  }
}

/** Flat offsets of every batch element, in row-major batch order. */
function batchOffsetTable(
  batchShape: readonly number[],
  strides: readonly number[],
  base: number
): number[] {
  let offsets = [base];
  for (let d = 0; d < batchShape.length; d++) {
    const dim = batchShape[d] ?? 0;
    const stride = strides[d] ?? 0;
    const next: number[] = new Array(offsets.length * dim);
    let idx = 0;
    for (const o of offsets) {
      for (let i = 0; i < dim; i++) next[idx++] = o + i * stride;
    }
    offsets = next;
  }
  return offsets;
}

/**
 * Shared implementation of `dot` and the strict 2-D `matmul`.
 *
 * Every supported shape combination is reduced to a batch of (m, k) x (k, n)
 * products with explicit strides, so one set of kernels serves vectors,
 * matrices and batches, and non-contiguous views are read in place.
 *
 * @internal
 */
export function contract(a: Tensor, b: Tensor, op: ContractionOp): Tensor {
  return roundHalfResult(contractKernel(a, b, op));
}

function contractKernel(a: Tensor, b: Tensor, op: ContractionOp): Tensor {
  const outDtype = resolveDotDtype(op, a.dtype, b.dtype);

  const aData = a.data;
  const bData = b.data;
  if (Array.isArray(aData) || Array.isArray(bData)) {
    throw new DTypeError(`${op} is not defined for string dtype`);
  }

  const plan = planMatmul(a, b, op);
  const { k, n, outShape, aSM, aSK, bSK, bSN } = plan;
  let { m, batchShape, aBatchStrides, bBatchStrides } = plan;

  // A right operand shared by the whole batch (every batch stride is 0) lets
  // the batch fold into the rows of `a` whenever its batch and row axes merge
  // into one strided axis: (B, m, k) @ (k, n) is then a single (B*m, k) product.
  if (batchShape.length > 0 && bBatchStrides.every((s) => s === 0)) {
    let expected = aSM * m;
    let foldable = true;
    let rows = m;
    for (let d = batchShape.length - 1; d >= 0; d--) {
      const dim = batchShape[d] ?? 1;
      if (dim === 1) continue;
      if ((aBatchStrides[d] ?? 0) !== expected) {
        foldable = false;
        break;
      }
      rows *= dim;
      expected *= dim;
    }
    if (foldable) {
      m = rows;
      batchShape = [];
      aBatchStrides = [];
      bBatchStrides = [];
    }
  }

  let batchSize = 1;
  for (const dim of batchShape) batchSize *= dim;

  const aOffsets =
    batchShape.length > 0 ? batchOffsetTable(batchShape, aBatchStrides, a.offset) : null;
  const bOffsets =
    batchShape.length > 0 ? batchOffsetTable(batchShape, bBatchStrides, b.offset) : null;

  const Ctor = dtypeToTypedArrayCtor(outDtype);
  const result = new Ctor(batchSize * m * n);
  const blockSize = m * n;

  if (blockSize > 0) {
    if (outDtype === "int64") {
      if (
        !(aData instanceof BigInt64Array) ||
        !(bData instanceof BigInt64Array) ||
        !(result instanceof BigInt64Array)
      ) {
        throw new DTypeError(`${op} requires int64 dtype`);
      }
      for (let bi = 0; bi < batchSize; bi++) {
        const aBase = aOffsets ? (aOffsets[bi] as number) : a.offset;
        const bBase = bOffsets ? (bOffsets[bi] as number) : b.offset;
        const oBase = bi * blockSize;
        for (let i = 0; i < m; i++) {
          for (let j = 0; j < n; j++) {
            let sum = 0n;
            for (let p = 0; p < k; p++) {
              sum +=
                getBigIntElement(aData, aBase + i * aSM + p * aSK) *
                getBigIntElement(bData, bBase + p * bSK + j * bSN);
            }
            if (sum < INT64_MIN || sum > INT64_MAX) {
              throw new DataValidationError(`int64 ${op} overflow`);
            }
            result[oBase + i * n + j] = sum;
          }
        }
      }
    } else {
      if (
        aData instanceof BigInt64Array ||
        bData instanceof BigInt64Array ||
        result instanceof BigInt64Array
      ) {
        throw new DTypeError(`${op} requires non-int64 dtype`);
      }
      // Scratch buffers are shared across all batch elements.
      const r0 = new Float64Array(n);
      const r1 = new Float64Array(n);
      const rowAcc32 = outDtype === "int32" ? new Int32Array(n) : null;
      for (let bi = 0; bi < batchSize; bi++) {
        const aBase = aOffsets ? (aOffsets[bi] as number) : a.offset;
        const bBase = bOffsets ? (bOffsets[bi] as number) : b.offset;
        const oBase = bi * blockSize;
        if (outDtype === "bool") {
          gemmBool(
            aData as Uint8Array,
            aBase,
            aSM,
            aSK,
            bData as Uint8Array,
            bBase,
            bSK,
            bSN,
            result as Uint8Array,
            oBase,
            m,
            k,
            n
          );
        } else if (outDtype === "int32") {
          gemmInt32(
            aData as Int32Array,
            aBase,
            aSM,
            aSK,
            bData as Int32Array,
            bBase,
            bSK,
            bSN,
            result as Int32Array,
            oBase,
            m,
            k,
            n,
            rowAcc32 as Int32Array
          );
        } else if (n === 1) {
          gemv(aData, aBase, aSM, aSK, bData, bBase, bSK, result, oBase, m, k);
        } else if (aData instanceof Float64Array && bData instanceof Float64Array) {
          if (aSK === 1 && bSN === 1) {
            gemmBlocked(aData, aBase, aSM, bData, bBase, bSK, result, oBase, m, k, n, r0, r1);
          } else {
            gemmF64(aData, aBase, aSM, aSK, bData, bBase, bSK, bSN, result, oBase, m, k, n, r0);
          }
        } else if (aData instanceof Float32Array && bData instanceof Float32Array) {
          if (aSK === 1 && bSN === 1) {
            gemmBlocked(aData, aBase, aSM, bData, bBase, bSK, result, oBase, m, k, n, r0, r1);
          } else {
            gemmF32(aData, aBase, aSM, aSK, bData, bBase, bSK, bSN, result, oBase, m, k, n, r0);
          }
        } else {
          gemmGeneric(aData, aBase, aSM, aSK, bData, bBase, bSK, bSN, result, oBase, m, k, n, r0);
        }
      }
    }
  }

  const shape: Shape = outShape;
  return TensorClass.fromTypedArray({
    data: result,
    shape,
    dtype: outDtype,
    device: a.device,
  });
}

/**
 * Compute dot product or matrix multiplication.
 *
 * Supported cases:
 * - Both 1-D (vector, vector): inner product, returns a 0-d tensor
 * - Both 2-D (matrix, matrix): standard matrix multiplication (m,k) x (k,n) -> (m,n)
 * - 2-D x 1-D (matrix, vector): matrix-vector product (m,k) x (k,) -> (m,)
 * - 1-D x 2-D (vector, matrix): vector-matrix product (k,) x (k,n) -> (n,)
 * - 3-D and higher: batched matrix multiplication with `numpy.matmul` rules. The last
 *   two axes are the matrices and the leading axes are batch dimensions, aligned from
 *   the right and broadcast: two batch dimensions are compatible when they are equal or
 *   one of them is 1, and a missing batch dimension counts as 1. For example
 *   (b,m,k) x (b,k,n) -> (b,m,n), (1,m,k) x (3,k,n) -> (3,m,n),
 *   (2,1,m,k) x (3,k,n) -> (2,3,m,n), and (b,m,k) x (k,n) -> (b,m,n).
 * - 1-D with 3-D or higher: the vector is promoted like `numpy.matmul`, so
 *   (b,m,k) x (k,) -> (b,m) and (k,) x (b,k,n) -> (b,n).
 *
 * Both operands must have the same dtype. The only mixed case allowed is
 * float32 with float64, which gives float64. Float products are accumulated in
 * float64. int32 results wrap around like NumPy, bool results are the logical
 * product (OR of ANDs), and int64 results throw if they leave the int64 range.
 * `0 * Infinity` and `0 * NaN` give NaN as usual.
 *
 * Unlike `numpy.dot`, 0-d operands are rejected (use `mul`), and N-D operands are
 * multiplied as broadcast batches (the `numpy.matmul` rule) rather than summed over
 * every leading axis of the left operand and the second-to-last axis of the right.
 *
 * @param a - First tensor
 * @param b - Second tensor
 * @returns Dot product result
 *
 * @throws {DTypeError} If a dtype is `string` or the dtypes are incompatible
 * @throws {ShapeError} If an operand is 0-d, the inner dimensions differ, or the batch
 * dimensions cannot be broadcast
 * @throws {DataValidationError} If an int64 result overflows
 *
 * @example
 * ```ts
 * import { dot, ones, tensor } from "deepbox/ndarray";
 *
 * dot(tensor([1, 2, 3]), tensor([4, 5, 6])); // 32 (0-d tensor)
 * dot(tensor([[1, 2], [3, 4]]), tensor([[5, 6], [7, 8]])); // [[19, 22], [43, 50]]
 *
 * // Batch dimensions broadcast: (1, 2, 2) x (3, 2, 2) -> (3, 2, 2)
 * dot(ones([1, 2, 2]), ones([3, 2, 2])).shape; // [3, 2, 2]
 * ```
 */
export function dot(a: Tensor, b: Tensor): Tensor {
  if (a.device !== "cpu" || b.device !== "cpu") {
    // Resolve the dtype first so unsupported dtypes report the same error on every device.
    resolveDotDtype("dot", a.dtype, b.dtype);
    const onDevice = dispatchDot(a, b);
    if (onDevice) return onDevice;
  }
  return contract(a, b, "dot");
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
