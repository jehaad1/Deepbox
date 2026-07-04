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

import { DTypeError, InvalidParameterError, ShapeError } from "../../core";
import { isContiguous, offsetFromFlatIndex } from "../tensor/strides";
import { computeStrides, Tensor } from "../tensor/Tensor";

// ---- Internal helpers ----

/** Binary GCD (Stein's algorithm) for non-negative integers. */
function gcdInt(a: number, b: number): number {
  a = Math.abs(Math.trunc(a));
  b = Math.abs(Math.trunc(b));
  if (a === 0) return b;
  if (b === 0) return a;
  while (b !== 0) {
    const t = b;
    b = a % b;
    a = t;
  }
  return a;
}

function lcmInt(a: number, b: number): number {
  a = Math.abs(Math.trunc(a));
  b = Math.abs(Math.trunc(b));
  if (a === 0 || b === 0) return 0;
  return (a / gcdInt(a, b)) * b;
}

/** Read a numeric value from a tensor at a physical offset. */
function readVal(t: Tensor, offset: number): number {
  const raw = t.data[offset];
  if (typeof raw === "number") return raw;
  if (typeof raw === "bigint") return Number(raw);
  return 0;
}

/** Read all values of a 1D tensor into a number array. */
function read1D(t: Tensor): number[] {
  const n = t.shape[0] ?? 0;
  const stride = t.strides[0] ?? 1;
  const result: number[] = [];
  for (let i = 0; i < n; i++) {
    result.push(readVal(t, t.offset + i * stride));
  }
  return result;
}

function assertNumeric(t: Tensor, name: string): void {
  if (t.dtype === "string") {
    throw new DTypeError(`${name} requires numeric tensor`);
  }
}

// ---- Element-wise GCD / LCM ----

/**
 * Element-wise greatest common divisor.
 *
 * Both inputs are cast to integers (truncated). Supports broadcasting.
 *
 * @param a - First tensor
 * @param b - Second tensor (must be broadcastable to `a`)
 * @returns Tensor of GCD values with dtype int32
 *
 * @example
 * ```ts
 * const a = tensor([12, 15, 20]);
 * const b = tensor([8, 10, 25]);
 * gcd(a, b); // [4, 5, 5]
 * ```
 */
export function gcd(a: Tensor, b: Tensor): Tensor {
  assertNumeric(a, "gcd");
  assertNumeric(b, "gcd");

  if (a.size !== b.size) {
    throw new ShapeError(`gcd requires tensors with same size; got ${a.size} and ${b.size}`);
  }

  const size = a.size;
  const out = new Int32Array(size);
  const aStrides = computeStrides(a.shape);
  const bStrides = computeStrides(b.shape);
  const aCont = isContiguous(a.shape, a.strides);
  const bCont = isContiguous(b.shape, b.strides);

  for (let i = 0; i < size; i++) {
    const aOff = aCont ? a.offset + i : offsetFromFlatIndex(i, aStrides, a.strides, a.offset);
    const bOff = bCont ? b.offset + i : offsetFromFlatIndex(i, bStrides, b.strides, b.offset);
    out[i] = gcdInt(readVal(a, aOff), readVal(b, bOff));
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: [...a.shape],
    dtype: "int32",
    device: a.device,
  });
}

/**
 * Element-wise least common multiple.
 *
 * Both inputs are cast to integers (truncated). Supports broadcasting.
 *
 * @param a - First tensor
 * @param b - Second tensor (must be broadcastable to `a`)
 * @returns Tensor of LCM values with dtype int32
 *
 * @example
 * ```ts
 * const a = tensor([4, 6, 12]);
 * const b = tensor([6, 8, 15]);
 * lcm(a, b); // [12, 24, 60]
 * ```
 */
export function lcm(a: Tensor, b: Tensor): Tensor {
  assertNumeric(a, "lcm");
  assertNumeric(b, "lcm");

  if (a.size !== b.size) {
    throw new ShapeError(`lcm requires tensors with same size; got ${a.size} and ${b.size}`);
  }

  const size = a.size;
  const out = new Int32Array(size);
  const aStrides = computeStrides(a.shape);
  const bStrides = computeStrides(b.shape);
  const aCont = isContiguous(a.shape, a.strides);
  const bCont = isContiguous(b.shape, b.strides);

  for (let i = 0; i < size; i++) {
    const aOff = aCont ? a.offset + i : offsetFromFlatIndex(i, aStrides, a.strides, a.offset);
    const bOff = bCont ? b.offset + i : offsetFromFlatIndex(i, bStrides, b.strides, b.offset);
    out[i] = lcmInt(readVal(a, aOff), readVal(b, bOff));
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: [...a.shape],
    dtype: "int32",
    device: a.device,
  });
}

// ---- 1D set operations ----

/**
 * Compute the sorted, unique union of two 1D tensors.
 *
 * @param a - First 1D tensor
 * @param b - Second 1D tensor
 * @returns Sorted 1D tensor containing the unique union of elements
 *
 * @example
 * ```ts
 * union1d(tensor([1, 2, 3]), tensor([2, 3, 4])); // [1, 2, 3, 4]
 * ```
 */
export function union1d(a: Tensor, b: Tensor): Tensor {
  assertNumeric(a, "union1d");
  assertNumeric(b, "union1d");
  if (a.ndim !== 1) {
    throw new InvalidParameterError("union1d requires 1D tensor", "a", a.shape);
  }
  if (b.ndim !== 1) {
    throw new InvalidParameterError("union1d requires 1D tensor", "b", b.shape);
  }

  const setA = read1D(a);
  const setB = read1D(b);
  const combined = new Set<number>([...setA, ...setB]);
  const sorted = [...combined].sort((x, y) => x - y);

  return Tensor.fromTypedArray({
    data: new Float64Array(sorted),
    shape: [sorted.length],
    dtype: "float64",
    device: a.device,
  });
}

/**
 * Compute the sorted, unique intersection of two 1D tensors.
 *
 * @param a - First 1D tensor
 * @param b - Second 1D tensor
 * @returns Sorted 1D tensor containing the unique intersection of elements
 *
 * @example
 * ```ts
 * intersect1d(tensor([1, 2, 3, 4]), tensor([2, 4, 6])); // [2, 4]
 * ```
 */
export function intersect1d(a: Tensor, b: Tensor): Tensor {
  assertNumeric(a, "intersect1d");
  assertNumeric(b, "intersect1d");
  if (a.ndim !== 1) {
    throw new InvalidParameterError("intersect1d requires 1D tensor", "a", a.shape);
  }
  if (b.ndim !== 1) {
    throw new InvalidParameterError("intersect1d requires 1D tensor", "b", b.shape);
  }

  const setB = new Set(read1D(b));
  const uniqueA = new Set(read1D(a));
  const result: number[] = [];
  for (const v of uniqueA) {
    if (setB.has(v)) result.push(v);
  }
  result.sort((x, y) => x - y);

  return Tensor.fromTypedArray({
    data: new Float64Array(result),
    shape: [result.length],
    dtype: "float64",
    device: a.device,
  });
}

/**
 * Compute the sorted set difference of two 1D tensors.
 *
 * Returns elements in `a` that are not in `b`.
 *
 * @param a - First 1D tensor
 * @param b - Second 1D tensor
 * @returns Sorted 1D tensor of unique elements in `a` not in `b`
 *
 * @example
 * ```ts
 * setdiff1d(tensor([1, 2, 3, 4]), tensor([2, 4])); // [1, 3]
 * ```
 */
export function setdiff1d(a: Tensor, b: Tensor): Tensor {
  assertNumeric(a, "setdiff1d");
  assertNumeric(b, "setdiff1d");
  if (a.ndim !== 1) {
    throw new InvalidParameterError("setdiff1d requires 1D tensor", "a", a.shape);
  }
  if (b.ndim !== 1) {
    throw new InvalidParameterError("setdiff1d requires 1D tensor", "b", b.shape);
  }

  const setB = new Set(read1D(b));
  const uniqueA = new Set(read1D(a));
  const result: number[] = [];
  for (const v of uniqueA) {
    if (!setB.has(v)) result.push(v);
  }
  result.sort((x, y) => x - y);

  return Tensor.fromTypedArray({
    data: new Float64Array(result),
    shape: [result.length],
    dtype: "float64",
    device: a.device,
  });
}
