/**
 * Element-wise GCD/LCM and 1D set operations on tensors.
 *
 * - gcd: Element-wise greatest common divisor
 * - lcm: Element-wise least common multiple
 * - union1d: Sorted unique union of two 1D tensors
 * - intersect1d: Sorted unique intersection of two 1D tensors
 * - setdiff1d: Sorted set difference of two 1D tensors
 *
 * @module ndarray/ops/setops
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox documentation}
 */

import { DataValidationError, DTypeError, InvalidParameterError } from "../../core";
import { Tensor } from "../tensor/Tensor";
import { broadcastApply, getBroadcastShape } from "./broadcast";

// ---- Internal helpers ----

const INT32_MAX = 2147483647;
const INT64_MAX = (1n << 63n) - 1n;
const INT64_MIN = -(1n << 63n);

/** Reject dtypes that have no meaningful integer or ordering semantics here. */
function assertNumeric(t: Tensor, name: string): void {
  if (t.dtype === "string") {
    throw new DTypeError(`${name} requires numeric tensor`);
  }
  if (t.dtype === "complex64" || t.dtype === "complex128") {
    throw new DTypeError(`${name} does not support complex dtypes`);
  }
}

/** True for dtypes whose values are exact integers. */
function isIntegerDType(dtype: Tensor["dtype"]): boolean {
  return dtype === "int32" || dtype === "int64" || dtype === "uint8" || dtype === "bool";
}

/** Greatest common divisor of two non-negative integers held in doubles. */
function gcdInt(a: number, b: number): number {
  while (b !== 0) {
    const t = b;
    b = a % b;
    a = t;
  }
  return a;
}

function gcdBig(a: bigint, b: bigint): bigint {
  while (b !== 0n) {
    const t = b;
    b = a % b;
    a = t;
  }
  return a;
}

function absBig(x: bigint): bigint {
  return x < 0n ? -x : x;
}

/** Truncate a value toward zero, rejecting NaN and the infinities. */
function toIntegerValue(x: number, name: string): number {
  if (!Number.isFinite(x)) {
    throw new DataValidationError(`${name} requires finite values; received ${String(x)}`);
  }
  return Math.abs(Math.trunc(x));
}

function toBigIntValue(x: number | bigint, name: string): bigint {
  if (typeof x === "bigint") return absBig(x);
  return BigInt(toIntegerValue(x, name));
}

/**
 * Apply an integer binary function with NumPy broadcasting.
 *
 * Inputs are cast to integers (floats are truncated toward zero). If either
 * input is int64 the work is done in BigInt and the result is int64;
 * otherwise the result is int32. Results that do not fit the output dtype
 * throw instead of wrapping.
 */
function integerBinary(
  a: Tensor,
  b: Tensor,
  name: string,
  numFn: (x: number, y: number) => number,
  bigFn: (x: bigint, y: bigint) => bigint
): Tensor {
  assertNumeric(a, name);
  assertNumeric(b, name);

  const outShape = getBroadcastShape(a.shape, b.shape);
  const size = outShape.reduce((p, d) => p * d, 1);
  const useBig = a.dtype === "int64" || b.dtype === "int64";
  const aData = a.data as ArrayLike<number | bigint>;
  const bData = b.data as ArrayLike<number | bigint>;

  if (useBig) {
    const out = new BigInt64Array(size);
    const result = Tensor.fromTypedArray({
      data: out,
      shape: outShape,
      dtype: "int64",
      device: a.device,
    });
    broadcastApply(a, b, result, (oa, ob, oo) => {
      const r = bigFn(
        toBigIntValue(aData[oa] as number | bigint, name),
        toBigIntValue(bData[ob] as number | bigint, name)
      );
      if (r > INT64_MAX || r < INT64_MIN) {
        throw new DataValidationError(`${name} result does not fit in int64`);
      }
      out[oo] = r;
    });
    return result;
  }

  const out = new Int32Array(size);
  const result = Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: "int32",
    device: a.device,
  });
  broadcastApply(a, b, result, (oa, ob, oo) => {
    const r = numFn(
      toIntegerValue(aData[oa] as number, name),
      toIntegerValue(bData[ob] as number, name)
    );
    if (!(r <= INT32_MAX)) {
      throw new DataValidationError(
        `${name} result does not fit in int32; convert the inputs to int64 for larger values`
      );
    }
    out[oo] = r;
  });
  return result;
}

// ---- Element-wise GCD / LCM ----

/**
 * Element-wise greatest common divisor.
 *
 * Both inputs are cast to integers (floats are truncated toward zero) and the
 * result is always non-negative, with `gcd(0, 0) = 0`. Inputs broadcast
 * against each other under NumPy rules. If either input is int64 the result is
 * int64, otherwise int32.
 *
 * @param a - First tensor
 * @param b - Second tensor (broadcastable with `a`)
 * @returns Tensor of GCD values with dtype int32 (int64 if an input is int64)
 * @throws {DTypeError} If an input has string or complex dtype
 * @throws {ShapeError} If the shapes do not broadcast
 * @throws {DataValidationError} If an input is NaN or infinite, or a result does not fit the output dtype
 *
 * @example
 * ```ts
 * const a = tensor([12, 15, 20]);
 * const b = tensor([8, 10, 25]);
 * gcd(a, b); // [4, 5, 5]
 * ```
 */
export function gcd(a: Tensor, b: Tensor): Tensor {
  return integerBinary(a, b, "gcd", gcdInt, gcdBig);
}

/**
 * Element-wise least common multiple.
 *
 * Both inputs are cast to integers (floats are truncated toward zero) and the
 * result is always non-negative, with `lcm(x, 0) = 0`. Inputs broadcast
 * against each other under NumPy rules. If either input is int64 the result is
 * int64, otherwise int32.
 *
 * @param a - First tensor
 * @param b - Second tensor (broadcastable with `a`)
 * @returns Tensor of LCM values with dtype int32 (int64 if an input is int64)
 * @throws {DTypeError} If an input has string or complex dtype
 * @throws {ShapeError} If the shapes do not broadcast
 * @throws {DataValidationError} If an input is NaN or infinite, or a result does not fit the output dtype
 *
 * @example
 * ```ts
 * const a = tensor([4, 6, 12]);
 * const b = tensor([6, 8, 15]);
 * lcm(a, b); // [12, 24, 60]
 * ```
 */
export function lcm(a: Tensor, b: Tensor): Tensor {
  return integerBinary(
    a,
    b,
    "lcm",
    (x, y) => (x === 0 || y === 0 ? 0 : (x / gcdInt(x, y)) * y),
    (x, y) => (x === 0n || y === 0n ? 0n : (x / gcdBig(x, y)) * y)
  );
}

// ---- 1D set operations ----

/** Values of a 1-D tensor, as exact BigInt or as doubles. */
type SetValues = Float64Array | BigInt64Array;

function isNaNValue(v: number | bigint): boolean {
  return typeof v === "number" && Number.isNaN(v);
}

function require1D(t: Tensor, name: string, which: "a" | "b"): void {
  if (t.ndim !== 1) {
    throw new InvalidParameterError(
      `${name} requires 1D tensor; ${which} has shape [${t.shape.join(", ")}]`,
      which,
      t.shape
    );
  }
}

/**
 * Read both operands of a set operation. When either is int64 and the other
 * holds exact integers, both are read as BigInt so values above 2^53 are not
 * merged by rounding; otherwise both are read as doubles.
 */
function readOperands(a: Tensor, b: Tensor, name: string): [SetValues, SetValues] {
  assertNumeric(a, name);
  assertNumeric(b, name);
  require1D(a, name, "a");
  require1D(b, name, "b");
  const exact =
    (a.dtype === "int64" || b.dtype === "int64") &&
    isIntegerDType(a.dtype) &&
    isIntegerDType(b.dtype);
  return [read1D(a, exact), read1D(b, exact)];
}

function read1D(t: Tensor, exact: boolean): SetValues {
  const n = t.shape[0] ?? 0;
  const stride = t.strides[0] ?? 1;
  const data = t.data as ArrayLike<number | bigint>;
  if (exact) {
    const out = new BigInt64Array(n);
    for (let i = 0; i < n; i++) out[i] = BigInt(data[t.offset + i * stride] as number | bigint);
    return out;
  }
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = Number(data[t.offset + i * stride]);
  return out;
}

/** Sort ascending (NaN last, -0 before 0) and build a 1-D tensor. */
function toSortedTensor(values: Iterable<number | bigint>, exact: boolean, a: Tensor): Tensor {
  if (exact) {
    const out = BigInt64Array.from(values as Iterable<bigint>).sort();
    return Tensor.fromTypedArray({
      data: out,
      shape: [out.length],
      dtype: "int64",
      device: a.device,
    });
  }
  const out = Float64Array.from(values as Iterable<number>).sort();
  return Tensor.fromTypedArray({
    data: out,
    shape: [out.length],
    dtype: "float64",
    device: a.device,
  });
}

/**
 * Compute the sorted, unique union of two 1D tensors.
 *
 * NaN values collapse into a single NaN that sorts last, as in NumPy. The
 * result is float64, or int64 when one input is int64 and the other holds
 * integers.
 *
 * @param a - First 1D tensor
 * @param b - Second 1D tensor
 * @returns Sorted 1D tensor containing the unique union of elements
 * @throws {DTypeError} If an input has string or complex dtype
 * @throws {InvalidParameterError} If an input is not 1D
 *
 * @example
 * ```ts
 * union1d(tensor([1, 2, 3]), tensor([2, 3, 4])); // [1, 2, 3, 4]
 * ```
 */
export function union1d(a: Tensor, b: Tensor): Tensor {
  const [va, vb] = readOperands(a, b, "union1d");
  const exact = va instanceof BigInt64Array;
  const combined = new Set<number | bigint>(va);
  for (const v of vb) combined.add(v);
  return toSortedTensor(combined, exact, a);
}

/**
 * Compute the sorted, unique intersection of two 1D tensors.
 *
 * NaN never matches another NaN (as in NumPy), so NaN is not part of the
 * result. The result is float64, or int64 when one input is int64 and the
 * other holds integers.
 *
 * @param a - First 1D tensor
 * @param b - Second 1D tensor
 * @returns Sorted 1D tensor containing the unique intersection of elements
 * @throws {DTypeError} If an input has string or complex dtype
 * @throws {InvalidParameterError} If an input is not 1D
 *
 * @example
 * ```ts
 * intersect1d(tensor([1, 2, 3, 4]), tensor([2, 4, 6])); // [2, 4]
 * ```
 */
export function intersect1d(a: Tensor, b: Tensor): Tensor {
  const [va, vb] = readOperands(a, b, "intersect1d");
  const exact = va instanceof BigInt64Array;
  const inB = new Set<number | bigint>(vb);
  const result = new Set<number | bigint>();
  for (const v of va) {
    if (inB.has(v) && !isNaNValue(v)) result.add(v);
  }
  return toSortedTensor(result, exact, a);
}

/**
 * Compute the sorted set difference of two 1D tensors.
 *
 * Returns the unique elements of `a` that are not in `b`. NaN never matches
 * another NaN (as in NumPy), so a NaN in `a` is always kept. The result is
 * float64, or int64 when one input is int64 and the other holds integers.
 *
 * @param a - First 1D tensor
 * @param b - Second 1D tensor
 * @returns Sorted 1D tensor of unique elements in `a` not in `b`
 * @throws {DTypeError} If an input has string or complex dtype
 * @throws {InvalidParameterError} If an input is not 1D
 *
 * @example
 * ```ts
 * setdiff1d(tensor([1, 2, 3, 4]), tensor([2, 4])); // [1, 3]
 * ```
 */
export function setdiff1d(a: Tensor, b: Tensor): Tensor {
  const [va, vb] = readOperands(a, b, "setdiff1d");
  const exact = va instanceof BigInt64Array;
  const inB = new Set<number | bigint>(vb);
  const result = new Set<number | bigint>();
  for (const v of va) {
    if (isNaNValue(v) || !inB.has(v)) result.add(v);
  }
  return toSortedTensor(result, exact, a);
}
