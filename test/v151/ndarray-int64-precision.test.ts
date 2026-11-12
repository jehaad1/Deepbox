import { describe, expect, it } from "vitest";
import { allclose, isclose, searchsorted, Tensor } from "../../src/ndarray";

// Reference: numpy.searchsorted on int64 arrays compares the integers exactly (values below
// were checked against numpy 2.x). numpy.isclose converts int64 to double and so reports
// 2**53 + 1 and 2**53 as close even with zero tolerances; this library compares the exact
// integer difference instead, as agreed for this fix.

function i64(values: bigint[]): Tensor {
  return Tensor.fromTypedArray({
    data: BigInt64Array.from(values),
    shape: [values.length],
    dtype: "int64",
    device: "cpu",
  });
}

const BIG = 9007199254740992n; // 2^53

describe("int64 beyond 2^53", () => {
  it("searchsorted distinguishes neighbouring values", () => {
    const arr = i64([BIG, BIG + 1n, BIG + 2n]);
    // numpy.searchsorted([2**53, 2**53 + 1, 2**53 + 2], [2**53 + 1, 2**53 + 2], "left") -> [1, 2]
    expect(searchsorted(arr, i64([BIG + 1n, BIG + 2n])).toArray()).toEqual([1, 2]);
    // side="right" -> [2, 3]
    expect(searchsorted(arr, i64([BIG + 1n, BIG + 2n]), "right").toArray()).toEqual([2, 3]);
  });

  it("searchsorted works on strided int64 views", () => {
    const arr = i64([BIG, 0n, BIG + 1n, 0n, BIG + 2n]);
    const view = arr.slice({ start: 0, end: 5, step: 2 });
    expect(searchsorted(view, i64([BIG + 1n])).toArray()).toEqual([1]);
  });

  it("isclose with zero tolerances is exact", () => {
    expect(isclose(i64([BIG + 1n]), i64([BIG]), 0, 0).toArray()).toEqual([0]);
    expect(isclose(i64([BIG + 1n]), i64([BIG + 1n]), 0, 0).toArray()).toEqual([1]);
    expect(allclose(i64([BIG + 1n]), i64([BIG]), 0, 0)).toBe(false);
    expect(allclose(i64([BIG + 1n, 5n]), i64([BIG + 1n, 5n]), 0, 0)).toBe(true);
  });

  it("isclose uses the exact difference with tolerances", () => {
    // |a - b| = 1 <= atol 1
    expect(isclose(i64([BIG + 1n]), i64([BIG]), 0, 1).toArray()).toEqual([1]);
    expect(isclose(i64([BIG + 3n]), i64([BIG]), 0, 2).toArray()).toEqual([0]);
    // rtol scales with |b|: 1e-15 * 2^53 is about 9.007
    expect(isclose(i64([BIG + 9n]), i64([BIG]), 1e-15, 0).toArray()).toEqual([1]);
    expect(isclose(i64([BIG + 10n]), i64([BIG]), 1e-15, 0).toArray()).toEqual([0]);
  });

  it("isclose broadcasts a scalar int64", () => {
    const a = i64([BIG, BIG + 1n]);
    const b = Tensor.fromTypedArray({
      data: BigInt64Array.from([BIG]),
      shape: [],
      dtype: "int64",
      device: "cpu",
    });
    expect(isclose(a, b, 0, 0).toArray()).toEqual([1, 0]);
    expect(allclose(a, b, 0, 0)).toBe(false);
  });
});
