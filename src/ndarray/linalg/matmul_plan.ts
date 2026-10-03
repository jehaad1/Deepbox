/**
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox documentation}
 */

import { ShapeError } from "../../core";

/** Name of the public operation, used in error messages. */
export type MatmulOp = "dot" | "matmul";

/** Shape and strides of one operand. */
export type StridedShape = {
  readonly shape: readonly number[];
  readonly strides: readonly number[];
};

/**
 * Everything a kernel needs to run a (possibly broadcast) batched matrix product.
 *
 * The product is a batch of `(m, k) x (k, n)` matrices. `batchShape` is the
 * broadcast batch shape, and the batch strides of each operand are 0 on every
 * dimension that is broadcast (missing on the left, or of size 1).
 *
 * @internal
 */
export type MatmulPlan = {
  /** Broadcast batch shape (empty for plain vector and matrix products). */
  batchShape: number[];
  m: number;
  k: number;
  n: number;
  /** Shape of the result, with the promoted vector axes dropped. */
  outShape: number[];
  /** Stride of `a` along each batch dimension (0 where broadcast). */
  aBatchStrides: number[];
  /** Stride of `b` along each batch dimension (0 where broadcast). */
  bBatchStrides: number[];
  /** Row stride, inner stride of `a` (0 row stride for a promoted 1-D operand). */
  aSM: number;
  aSK: number;
  /** Inner stride, column stride of `b` (0 column stride for a promoted 1-D operand). */
  bSK: number;
  bSN: number;
};

/**
 * NumPy-style shape text: `(3,)` for 1-D, `(2, 3)` otherwise.
 *
 * @example
 * ```ts
 * formatMatmulShape([3]); // "(3,)"
 * formatMatmulShape([2, 3]); // "(2, 3)"
 * ```
 */
export function formatMatmulShape(shape: readonly number[]): string {
  if (shape.length === 1) return `(${shape[0]},)`;
  return `(${shape.join(", ")})`;
}

/**
 * Plan a product with `numpy.matmul` shape rules.
 *
 * A 1-D left operand acts as a row vector `(1, k)` and a 1-D right operand as a
 * column vector `(k, 1)`; the promoted axis is dropped from the result. Leading
 * (batch) dimensions are aligned from the right and broadcast: two batch
 * dimensions are compatible when they are equal or one of them is 1.
 *
 * @param a - Shape and strides of the left operand
 * @param b - Shape and strides of the right operand
 * @param op - Operation name for error messages
 * @returns The plan
 *
 * @throws {ShapeError} If an operand is 0-d, the inner dimensions differ, or the
 * batch dimensions cannot be broadcast
 *
 * @example
 * ```ts
 * const plan = planMatmul(
 *   { shape: [1, 2, 3], strides: [6, 3, 1] },
 *   { shape: [4, 3, 5], strides: [15, 5, 1] },
 *   "dot"
 * );
 * plan.outShape; // [4, 2, 5]
 * plan.aBatchStrides; // [0]
 * ```
 *
 * @internal
 */
export function planMatmul(a: StridedShape, b: StridedShape, op: MatmulOp): MatmulPlan {
  const aNdim = a.shape.length;
  const bNdim = b.shape.length;
  if (aNdim === 0 || bNdim === 0) {
    throw new ShapeError(
      `${op} does not accept 0-d tensors (shapes ${formatMatmulShape(a.shape)} and ${formatMatmulShape(b.shape)}); ` +
        "use mul for scalar multiplication"
    );
  }

  const m = aNdim === 1 ? 1 : (a.shape[aNdim - 2] ?? 0);
  const k = a.shape[aNdim - 1] ?? 0;
  const kb = b.shape[bNdim === 1 ? 0 : bNdim - 2] ?? 0;
  const n = bNdim === 1 ? 1 : (b.shape[bNdim - 1] ?? 0);
  if (k !== kb) {
    throw new ShapeError(
      `shapes ${formatMatmulShape(a.shape)} and ${formatMatmulShape(b.shape)} not aligned: ` +
        `${k} (dim ${aNdim - 1}) != ${kb} (dim ${bNdim === 1 ? 0 : bNdim - 2})`
    );
  }

  const aBatchRank = Math.max(0, aNdim - 2);
  const bBatchRank = Math.max(0, bNdim - 2);
  const batchRank = Math.max(aBatchRank, bBatchRank);
  const batchShape: number[] = new Array<number>(batchRank);
  const aBatchStrides: number[] = new Array<number>(batchRank).fill(0);
  const bBatchStrides: number[] = new Array<number>(batchRank).fill(0);
  for (let i = 0; i < batchRank; i++) {
    const ai = i - (batchRank - aBatchRank);
    const bi = i - (batchRank - bBatchRank);
    const da = ai >= 0 ? (a.shape[ai] ?? 1) : 1;
    const db = bi >= 0 ? (b.shape[bi] ?? 1) : 1;
    if (da !== db && da !== 1 && db !== 1) {
      throw new ShapeError(
        `batch dimensions don't match: [${a.shape.slice(0, aBatchRank)}] vs ` +
          `[${b.shape.slice(0, bBatchRank)}] cannot be broadcast together`
      );
    }
    batchShape[i] = da === 1 ? db : da;
    if (ai >= 0 && da !== 1) aBatchStrides[i] = a.strides[ai] ?? 0;
    if (bi >= 0 && db !== 1) bBatchStrides[i] = b.strides[bi] ?? 0;
  }

  const outShape = [...batchShape];
  if (aNdim >= 2) outShape.push(m);
  if (bNdim >= 2) outShape.push(n);

  return {
    batchShape,
    m,
    k,
    n,
    outShape,
    aBatchStrides,
    bBatchStrides,
    aSM: aNdim === 1 ? 0 : (a.strides[aNdim - 2] ?? 0),
    aSK: a.strides[aNdim - 1] ?? 0,
    bSK: b.strides[bNdim === 1 ? 0 : bNdim - 2] ?? 0,
    bSN: bNdim === 1 ? 0 : (b.strides[bNdim - 1] ?? 0),
  };
}
