/**
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox documentation}
 */

import type { DType, Shape } from "../../core";
import {
  DataValidationError,
  DTypeError,
  dtypeToTypedArrayCtor,
  getBigIntElement,
  getNumericElement,
  ShapeError,
} from "../../core";

import { dispatchMatmul } from "../ops/device_dispatch";
import { Tensor } from "../tensor/Tensor";

const INT64_MIN = -(1n << 63n);
const INT64_MAX = (1n << 63n) - 1n;

function isFloatDType(dtype: DType): boolean {
  return dtype === "float32" || dtype === "float64";
}

type NumericDType = Exclude<DType, "string">;

// Unified dtype promotion for matrix operations (matches linalg/index.ts resolveDotDtype)
function resolveMatmulDtype(opName: "dot" | "matmul", a: DType, b: DType): NumericDType {
  if (a === "string" || b === "string") {
    throw new DTypeError(`${opName} is not defined for string dtype`);
  }
  if (a === "int64" || b === "int64") {
    if (a !== b) {
      throw new DTypeError(`${opName} requires matching dtypes; received ${a} and ${b}`);
    }
    return "int64";
  }
  if (a === b) {
    return a;
  }
  if (isFloatDType(a) && isFloatDType(b)) {
    return "float64";
  }
  throw new DTypeError(`${opName} requires matching dtypes; received ${a} and ${b}`);
}

/**
 * Matrix multiplication.
 *
 * Supported (initial foundation):
 * - 2D x 2D
 * - All numeric dtypes except `string`
 *
 * Output dtype:
 * - int64 when both are int64
 * - float64 when mixing float32/float64
 * - otherwise matches the input dtype (requires matching dtypes)
 */
export function matmul(a: Tensor, b: Tensor): Tensor {
  if (a.device !== "cpu" || b.device !== "cpu") {
    const onDevice = dispatchMatmul(a, b);
    if (onDevice) return onDevice;
  }

  if (a.ndim !== 2 || b.ndim !== 2) {
    throw new ShapeError("matmul requires 2D tensors");
  }

  const m = a.shape[0] ?? 0;
  const k1 = a.shape[1] ?? 0;
  const k2 = b.shape[0] ?? 0;
  const n = b.shape[1] ?? 0;

  if (m === undefined || k1 === undefined || k2 === undefined || n === undefined) {
    throw new ShapeError("Internal error: missing shape");
  }

  if (k1 !== k2) {
    throw ShapeError.mismatch(a.shape, b.shape, "matmul");
  }

  const outShape: Shape = [m, n];
  const outDtype = resolveMatmulDtype("matmul", a.dtype, b.dtype);

  if (Array.isArray(a.data) || Array.isArray(b.data)) {
    throw new DTypeError("matmul not defined for string dtype");
  }
  const aData = a.data;
  const bData = b.data;

  if (outDtype === "int64") {
    // Both must be int64 based on resolveMatmulDtype logic for int64 result
    // But to be safe and satisfy TS, we can assert or check.
    if (!(aData instanceof BigInt64Array) || !(bData instanceof BigInt64Array)) {
      throw new DTypeError("Internal error: int64 matmul requires BigInt64Array data");
    }

    const out = new BigInt64Array(m * n);
    const aStride0 = a.strides[0] ?? 0;
    const aStride1 = a.strides[1] ?? 0;
    const bStride0 = b.strides[0] ?? 0;
    const bStride1 = b.strides[1] ?? 0;
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let acc = 0n;
        for (let k = 0; k < k1; k++) {
          const av = getBigIntElement(aData, a.offset + i * aStride0 + k * aStride1);
          const bv = getBigIntElement(bData, b.offset + k * bStride0 + j * bStride1);
          acc += av * bv;
        }
        if (acc < INT64_MIN || acc > INT64_MAX) {
          throw new DataValidationError("int64 matmul overflow");
        }
        out[i * n + j] = acc;
      }
    }

    return Tensor.fromTypedArray({
      data: out,
      shape: outShape,
      dtype: outDtype,
      device: a.device,
    });
  }

  const OutCtor = dtypeToTypedArrayCtor(outDtype);
  const out = new OutCtor(m * n);

  const aStride0 = a.strides[0] ?? 0;
  const aStride1 = a.strides[1] ?? 0;
  const bStride0 = b.strides[0] ?? 0;
  const bStride1 = b.strides[1] ?? 0;

  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let acc = 0;
      for (let k = 0; k < k1; k++) {
        const aIdx = a.offset + i * aStride0 + k * aStride1;
        const bIdx = b.offset + k * bStride0 + j * bStride1;

        const av =
          aData instanceof BigInt64Array
            ? Number(getBigIntElement(aData, aIdx))
            : getNumericElement(aData, aIdx);
        const bv =
          bData instanceof BigInt64Array
            ? Number(getBigIntElement(bData, bIdx))
            : getNumericElement(bData, bIdx);
        acc += av * bv;
      }
      out[i * n + j] = acc;
    }
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: outDtype,
    device: a.device,
  });
}

export function dot(a: Tensor, b: Tensor): Tensor {
  if (a.ndim !== 1 || b.ndim !== 1) {
    throw new ShapeError("dot requires 1D tensors");
  }

  if (a.size !== b.size) {
    throw ShapeError.mismatch(a.shape, b.shape, "dot");
  }

  const outDtype = resolveMatmulDtype("dot", a.dtype, b.dtype);

  if (Array.isArray(a.data) || Array.isArray(b.data)) {
    throw new DTypeError("dot not defined for string dtype");
  }
  const aData = a.data;
  const bData = b.data;

  if (outDtype === "int64") {
    if (!(aData instanceof BigInt64Array) || !(bData instanceof BigInt64Array)) {
      throw new DTypeError("Internal error: int64 dot requires BigInt64Array data");
    }
    const aStride = a.strides[0] ?? 0;
    const bStride = b.strides[0] ?? 0;
    let acc = 0n;
    for (let i = 0; i < a.size; i++) {
      const av = getBigIntElement(aData, a.offset + i * aStride);
      const bv = getBigIntElement(bData, b.offset + i * bStride);
      acc += av * bv;
    }
    if (acc < INT64_MIN || acc > INT64_MAX) {
      throw new DataValidationError("int64 dot overflow");
    }

    const out = new BigInt64Array(1);
    out[0] = acc;

    return Tensor.fromTypedArray({
      data: out,
      shape: [],
      dtype: "int64",
      device: a.device,
    });
  }

  let acc = 0;
  const aStride = a.strides[0] ?? 0;
  const bStride = b.strides[0] ?? 0;
  for (let i = 0; i < a.size; i++) {
    const av =
      aData instanceof BigInt64Array
        ? Number(getBigIntElement(aData, a.offset + i * aStride))
        : getNumericElement(aData, a.offset + i * aStride);
    const bv =
      bData instanceof BigInt64Array
        ? Number(getBigIntElement(bData, b.offset + i * bStride))
        : getNumericElement(bData, b.offset + i * bStride);
    acc += av * bv;
  }

  const OutCtor = dtypeToTypedArrayCtor(outDtype);
  const out = new OutCtor(1);
  out[0] = acc;

  return Tensor.fromTypedArray({
    data: out,
    shape: [],
    dtype: outDtype,
    device: a.device,
  });
}

/**
 * Tensor dot product along specified axes.
 *
 * Contracts `a` and `b` over the given axes, generalizing matmul and dot.
 *
 * If `axes` is a number N, contracts the last N axes of `a` with the first N axes of `b`.
 * If `axes` is a pair of arrays, contracts the specified axes of each tensor.
 *
 * @param a - First tensor
 * @param b - Second tensor
 * @param axes - Number of axes to contract, or [axesA, axesB] arrays
 * @returns Contracted tensor
 *
 * @throws {ShapeError} If contracted axes have mismatched sizes
 * @throws {DTypeError} If tensors have string dtype
 */
export function tensordot(a: Tensor, b: Tensor, axes: number | [number[], number[]] = 2): Tensor {
  if (a.dtype === "string" || b.dtype === "string") {
    throw new DTypeError("tensordot is not defined for string dtype");
  }

  let axesA: number[];
  let axesB: number[];

  if (typeof axes === "number") {
    if (axes < 0 || !Number.isInteger(axes)) {
      throw new ShapeError("axes must be a non-negative integer");
    }
    axesA = [];
    axesB = [];
    for (let i = 0; i < axes; i++) {
      axesA.push(a.ndim - axes + i);
      axesB.push(i);
    }
  } else {
    axesA = axes[0];
    axesB = axes[1];
    if (axesA.length !== axesB.length) {
      throw new ShapeError("axes[0] and axes[1] must have the same length");
    }
  }

  // Validate contracted dimensions match
  for (let i = 0; i < axesA.length; i++) {
    const dimA = a.shape[axesA[i]!];
    const dimB = b.shape[axesB[i]!];
    if (dimA !== dimB) {
      throw new ShapeError(
        `tensordot: contracted axis ${i} has mismatched sizes: ${dimA} vs ${dimB}`
      );
    }
  }

  // Build free axes (non-contracted)
  const freeA: number[] = [];
  const freeB: number[] = [];
  const axesASet = new Set(axesA);
  const axesBSet = new Set(axesB);
  for (let i = 0; i < a.ndim; i++) {
    if (!axesASet.has(i)) freeA.push(i);
  }
  for (let i = 0; i < b.ndim; i++) {
    if (!axesBSet.has(i)) freeB.push(i);
  }

  // Compute output shape
  const outShape: number[] = [];
  for (const ax of freeA) outShape.push(a.shape[ax]!);
  for (const ax of freeB) outShape.push(b.shape[ax]!);

  // Compute contracted size
  let contractedSize = 1;
  for (const ax of axesA) contractedSize *= a.shape[ax]!;

  // Free sizes
  let freeASize = 1;
  for (const ax of freeA) freeASize *= a.shape[ax]!;
  let freeBSize = 1;
  for (const ax of freeB) freeBSize *= b.shape[ax]!;

  // Flatten into (freeASize, contractedSize) and (contractedSize, freeBSize) then matmul
  // We need to read elements in the right order using stride-based indexing
  const aData = a.data;
  const bData = b.data;
  if (Array.isArray(aData) || Array.isArray(bData)) {
    throw new DTypeError("tensordot not defined for string dtype");
  }

  const out = new Float64Array(freeASize * freeBSize);

  // Build multi-index iterators
  const aFreeShape = freeA.map((ax) => a.shape[ax]!);
  const bFreeShape = freeB.map((ax) => b.shape[ax]!);
  const contractedShape = axesA.map((ax) => a.shape[ax]!);

  for (let fi = 0; fi < freeASize; fi++) {
    const aFreeIdx = unravelIndex(fi, aFreeShape);
    for (let fj = 0; fj < freeBSize; fj++) {
      const bFreeIdx = unravelIndex(fj, bFreeShape);
      let acc = 0;
      for (let ci = 0; ci < contractedSize; ci++) {
        const cIdx = unravelIndex(ci, contractedShape);

        // Build full index for a
        let aOff = a.offset;
        for (let k = 0; k < freeA.length; k++) {
          aOff += (aFreeIdx[k] ?? 0) * (a.strides[freeA[k]!] ?? 0);
        }
        for (let k = 0; k < axesA.length; k++) {
          aOff += (cIdx[k] ?? 0) * (a.strides[axesA[k]!] ?? 0);
        }

        // Build full index for b
        let bOff = b.offset;
        for (let k = 0; k < freeB.length; k++) {
          bOff += (bFreeIdx[k] ?? 0) * (b.strides[freeB[k]!] ?? 0);
        }
        for (let k = 0; k < axesB.length; k++) {
          bOff += (cIdx[k] ?? 0) * (b.strides[axesB[k]!] ?? 0);
        }

        const av =
          aData instanceof BigInt64Array
            ? Number(getBigIntElement(aData, aOff))
            : getNumericElement(aData, aOff);
        const bv =
          bData instanceof BigInt64Array
            ? Number(getBigIntElement(bData, bOff))
            : getNumericElement(bData, bOff);
        acc += av * bv;
      }
      out[fi * freeBSize + fj] = acc;
    }
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: outShape.length === 0 ? [] : outShape,
    dtype: "float64",
    device: a.device,
  });
}

function unravelIndex(flatIdx: number, shape: number[]): number[] {
  const idx: number[] = new Array(shape.length);
  let remaining = flatIdx;
  for (let i = shape.length - 1; i >= 0; i--) {
    const dim = shape[i]!;
    idx[i] = remaining % dim;
    remaining = Math.floor(remaining / dim);
  }
  return idx;
}

/**
 * Compute the Pearson correlation coefficient matrix.
 *
 * Given a 1D or 2D tensor, returns the correlation matrix.
 * For a 2D input of shape (m, n), each row is a variable and each column
 * is an observation (like NumPy).
 *
 * @param x - Input tensor (1D or 2D)
 * @returns Correlation matrix of shape (m, m)
 *
 * @throws {ShapeError} If input is not 1D or 2D
 * @throws {DTypeError} If input has string dtype
 */
export function corrcoef(x: Tensor): Tensor {
  const covMatrix = cov(x);
  const m = covMatrix.shape[0] ?? 0;
  const covData = covMatrix.data;
  if (Array.isArray(covData) || covData instanceof BigInt64Array) {
    throw new DTypeError("corrcoef not defined for string or int64 dtype");
  }

  const out = new Float64Array(m * m);

  // corr[i][j] = cov[i][j] / sqrt(cov[i][i] * cov[j][j])
  const diag = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    const val = getNumericElement(covData, i * m + i);
    diag[i] = val;
  }

  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      const covIJ = getNumericElement(covData, i * m + j);
      const denom = Math.sqrt(diag[i]! * diag[j]!);
      out[i * m + j] = denom === 0 ? (i === j ? 1 : 0) : covIJ / denom;
    }
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: [m, m],
    dtype: "float64",
    device: x.device,
  });
}

/**
 * Compute the covariance matrix.
 *
 * Given a 1D or 2D tensor, returns the covariance matrix.
 * For a 2D input of shape (m, n), each row is a variable and each column
 * is an observation.
 *
 * Uses Bessel's correction (ddof=1) by default.
 *
 * @param x - Input tensor (1D or 2D)
 * @param ddof - Delta degrees of freedom (default: 1)
 * @returns Covariance matrix of shape (m, m)
 *
 * @throws {ShapeError} If input is not 1D or 2D
 * @throws {DTypeError} If input has string dtype
 */
export function cov(x: Tensor, ddof: number = 1): Tensor {
  if (x.dtype === "string") {
    throw new DTypeError("cov is not defined for string dtype");
  }
  if (x.ndim > 2) {
    throw new ShapeError(`cov requires 1D or 2D input; got ndim=${x.ndim}`);
  }

  const xData = x.data;
  if (Array.isArray(xData)) {
    throw new DTypeError("cov not defined for string dtype");
  }

  // Treat 1D as (1, n)
  const m = x.ndim === 1 ? 1 : (x.shape[0] ?? 0);
  const n = x.ndim === 1 ? x.size : (x.shape[1] ?? 0);

  if (n < 2) {
    // Return zeros for single observation
    const out = new Float64Array(m * m);
    return Tensor.fromTypedArray({
      data: out,
      shape: [m, m],
      dtype: "float64",
      device: x.device,
    });
  }

  // Read all values into flat row-major array
  const values = new Float64Array(m * n);
  if (x.ndim === 1) {
    const stride = x.strides[0] ?? 0;
    for (let j = 0; j < n; j++) {
      values[j] =
        xData instanceof BigInt64Array
          ? Number(getBigIntElement(xData, x.offset + j * stride))
          : getNumericElement(xData, x.offset + j * stride);
    }
  } else {
    const s0 = x.strides[0] ?? 0;
    const s1 = x.strides[1] ?? 0;
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        values[i * n + j] =
          xData instanceof BigInt64Array
            ? Number(getBigIntElement(xData, x.offset + i * s0 + j * s1))
            : getNumericElement(xData, x.offset + i * s0 + j * s1);
      }
    }
  }

  // Compute means
  const means = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    let sum = 0;
    for (let j = 0; j < n; j++) sum += values[i * n + j]!;
    means[i] = sum / n;
  }

  // Compute covariance matrix
  const denom = n - ddof;
  const out = new Float64Array(m * m);
  for (let i = 0; i < m; i++) {
    for (let j = i; j < m; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) {
        sum += (values[i * n + k]! - means[i]!) * (values[j * n + k]! - means[j]!);
      }
      const val = denom > 0 ? sum / denom : 0;
      out[i * m + j] = val;
      out[j * m + i] = val;
    }
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: [m, m],
    dtype: "float64",
    device: x.device,
  });
}
