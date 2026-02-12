/**
 * Test helper utilities for type-safe test assertions.
 */

import type { GradTensor, Tensor } from "../src/ndarray";

/**
 * Extracts typed array data as a number[] from a Tensor or GradTensor.
 * Handles BigInt64Array conversion to numbers safely.
 */
export function numData(t: Tensor | GradTensor): number[] {
  const raw = "tensor" in t ? t.tensor.data : t.data;
  if (Array.isArray(raw)) {
    return raw.map(Number);
  }
  if (raw instanceof BigInt64Array) {
    return Array.from({ length: raw.length }, (_, i) => Number(raw[i]));
  }
  return Array.from(raw);
}

/**
 * Type guard: ensures a numeric typed array (not BigInt64Array, not string[]).
 */
export function expectFloatTypedArray(
  data: ArrayLike<number | bigint | string> | BigInt64Array | string[],
  context = "data"
): Float32Array | Float64Array | Int32Array | Uint8Array {
  if (
    data instanceof Float32Array ||
    data instanceof Float64Array ||
    data instanceof Int32Array ||
    data instanceof Uint8Array
  ) {
    return data;
  }
  throw new Error(`${context}: expected numeric typed array`);
}

/**
 * Converts a raw typed array (from tensor.data) to a number[].
 * Handles BigInt64Array/BigUint64Array by converting to Number.
 * Returns empty array for non-typed-array inputs.
 */
export function numRawData(data: unknown): number[] {
  if (
    !(
      data instanceof Float64Array ||
      data instanceof Float32Array ||
      data instanceof Int32Array ||
      data instanceof Uint32Array ||
      data instanceof Int16Array ||
      data instanceof Uint16Array ||
      data instanceof Int8Array ||
      data instanceof Uint8Array ||
      data instanceof Uint8ClampedArray ||
      data instanceof BigInt64Array ||
      data instanceof BigUint64Array
    )
  ) {
    return [];
  }
  const out: number[] = [];
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    out.push(typeof v === "bigint" ? Number(v) : v);
  }
  return out;
}

/**
 * Narrows `toArray()` result to a flat number array. Throws if not an array.
 */
export function toNumArr(value: unknown): number[] {
  if (!Array.isArray(value)) {
    if (typeof value === "number") return [value];
    throw new Error(`expected array, got ${typeof value}`);
  }
  return value.map(Number);
}

/**
 * Narrows `toArray()` result to a 2D number array. Throws if not a 2D array.
 */
export function toNum2D(value: unknown): number[][] {
  if (!Array.isArray(value)) throw new Error(`expected 2D array, got ${typeof value}`);
  return value.map((row: unknown) => {
    if (!Array.isArray(row)) throw new Error(`expected 2D array row, got ${typeof row}`);
    return row.map(Number);
  });
}

/**
 * Narrows `toArray()` result to a 2D string array. Throws if not a 2D array of strings.
 */
export function toStr2D(value: unknown): string[][] {
  if (!Array.isArray(value)) throw new Error(`expected 2D array, got ${typeof value}`);
  return value.map((row: unknown) => {
    if (!Array.isArray(row)) throw new Error(`expected 2D array row, got ${typeof row}`);
    return row.map(String);
  });
}
