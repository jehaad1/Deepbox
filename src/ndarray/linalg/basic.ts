import type { DType, Shape } from "../../core";
import {
  DataValidationError,
  DTypeError,
  dtypeToTypedArrayCtor,
  getBigIntElement,
  getNumericElement,
  ShapeError,
} from "../../core";

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
