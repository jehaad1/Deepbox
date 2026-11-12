/**
 * Order-preserving integer codes for string tensors.
 *
 * Operations such as `where`, `flip`, `unique` or `sort` only move or compare elements, so a
 * string tensor can run through the numeric implementation: every distinct string is replaced
 * by its rank in code point order, the numeric operation runs on the int32 ranks, and the
 * result is mapped back to strings. Equal strings share a rank, and a smaller rank is a
 * smaller string, so equality, ordering, stability and NaN-free semantics carry over.
 *
 * @internal
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox documentation}
 */

import { DeepboxError } from "../../core";
import { isContiguous, offsetFromFlatIndex } from "../tensor/strides";
import { computeStrides, Tensor } from "../tensor/Tensor";

/**
 * Compare two strings by Unicode code point, the order NumPy uses for `str` arrays. The
 * default `<` operator orders by UTF-16 code unit, which puts a character outside the Basic
 * Multilingual Plane (a surrogate pair) before U+E000 to U+FFFF.
 */
export function compareCodePoints(a: string, b: string): number {
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    const ca = a.codePointAt(i) as number;
    const cb = b.codePointAt(j) as number;
    if (ca !== cb) return ca < cb ? -1 : 1;
    i += ca > 0xffff ? 2 : 1;
    j += cb > 0xffff ? 2 : 1;
  }
  if (i < a.length) return 1;
  if (j < b.length) return -1;
  return 0;
}

/** Logical (row-major) elements of a string tensor. */
export function stringValues(t: Tensor): string[] {
  const data = t.data;
  if (!Array.isArray(data)) {
    throw new DeepboxError("Internal error: expected a string tensor");
  }
  const size = t.size;
  const out = new Array<string>(size);
  if (isContiguous(t.shape, t.strides)) {
    for (let i = 0; i < size; i++) out[i] = data[t.offset + i] as string;
    return out;
  }
  const logicalStrides = computeStrides(t.shape);
  for (let i = 0; i < size; i++) {
    out[i] = data[offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset)] as string;
  }
  return out;
}

/** String tensors encoded as int32 rank tensors, plus the table that decodes them. */
export type StringCodes = {
  /** One int32 tensor per input, with the shape of the input. */
  readonly codes: Tensor[];
  /** Distinct strings in code point order; `table[rank]` is the string with that rank. */
  readonly table: readonly string[];
};

/**
 * Encode string tensors with shared ranks, so values from different tensors compare correctly.
 *
 * @throws {DeepboxError} If an input is not a string tensor
 */
export function encodeStrings(tensors: readonly Tensor[]): StringCodes {
  const values = tensors.map(stringValues);
  const distinct = new Set<string>();
  for (const list of values) for (const v of list) distinct.add(v);
  const table = [...distinct].sort(compareCodePoints);
  const rank = new Map<string, number>();
  for (let i = 0; i < table.length; i++) rank.set(table[i] as string, i);
  const codes = tensors.map((t, k) => {
    const list = values[k] as string[];
    const data = new Int32Array(list.length);
    for (let i = 0; i < list.length; i++) data[i] = rank.get(list[i] as string) as number;
    return Tensor.fromTypedArray({ data, shape: [...t.shape], dtype: "int32", device: t.device });
  });
  return { codes, table };
}

/** Map a tensor of ranks back to a string tensor of the same shape. */
export function decodeStrings(ranks: Tensor, table: readonly string[]): Tensor {
  const size = ranks.size;
  const data = new Array<string>(size);
  const logicalStrides = computeStrides(ranks.shape);
  const src = ranks.data as ArrayLike<number | bigint>;
  const dense = isContiguous(ranks.shape, ranks.strides);
  for (let i = 0; i < size; i++) {
    const at = dense
      ? ranks.offset + i
      : offsetFromFlatIndex(i, logicalStrides, ranks.strides, ranks.offset);
    data[i] = table[Number(src[at])] as string;
  }
  return Tensor.fromStringArray({ data, shape: [...ranks.shape], device: ranks.device });
}
