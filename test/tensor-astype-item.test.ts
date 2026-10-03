import { describe, expect, it } from "vitest";
import { DTypeError, ShapeError, setSeed } from "../src/core";
import { argsort, sort, sum, tensor, transpose } from "../src/ndarray";
import { randn } from "../src/random";

describe("Tensor.astype", () => {
  it("converts float32 to float64 and back", () => {
    const t = tensor([1.5, -2.25, 0]);
    const f64 = t.astype("float64");
    expect(f64.dtype).toBe("float64");
    expect(f64.toArray()).toEqual([1.5, -2.25, 0]);
    expect(f64.astype("float32").dtype).toBe("float32");
  });

  it("truncates float to int32", () => {
    expect(tensor([1.7, -2.3, 0.9]).astype("int32").toArray()).toEqual([1, -2, 0]);
  });

  it("maps nonzero to 1 for bool", () => {
    expect(tensor([2, 0, -3, Number.NaN]).astype("bool").toArray()).toEqual([1, 0, 1, 1]);
  });

  it("converts to int64 bigints, truncating toward zero", () => {
    expect(tensor([5.9, -5.9], { dtype: "float64" }).astype("int64").toArray()).toEqual([5n, -5n]);
  });

  it("throws converting non-finite values to int64", () => {
    expect(() => tensor([Number.NaN]).astype("int64")).toThrow(DTypeError);
    expect(() => tensor([Number.POSITIVE_INFINITY]).astype("int64")).toThrow(DTypeError);
  });

  it("converts int64 to float64 and bool", () => {
    const t = tensor([0, 3, -1], { dtype: "int64" });
    expect(t.astype("float64").toArray()).toEqual([0, 3, -1]);
    expect(t.astype("bool").toArray()).toEqual([0, 1, 1]);
  });

  it("stringifies numeric tensors and parses string tensors", () => {
    expect(tensor([1.5, -2], { dtype: "float64" }).astype("string").toArray()).toEqual([
      "1.5",
      "-2",
    ]);
    expect(tensor(["1.5", "x", "3"]).astype("float64").toArray()).toEqual([1.5, Number.NaN, 3]);
  });

  it("materializes strided views in logical order", () => {
    const v = transpose(
      tensor(
        [
          [1, 2],
          [3, 4],
        ],
        { dtype: "float64" }
      )
    );
    expect(v.astype("int32").toArray()).toEqual([
      [1, 3],
      [2, 4],
    ]);
  });

  it("returns the same tensor when the dtype already matches", () => {
    const t = tensor([1, 2], { dtype: "float64" });
    expect(t.astype("float64")).toBe(t);
  });

  it("preserves shape for 2-D inputs", () => {
    const t = tensor([
      [1.1, 2.2],
      [3.3, 4.4],
    ]);
    const c = t.astype("float64");
    expect(c.shape).toEqual([2, 2]);
  });
});

describe("Tensor.item", () => {
  it("extracts 0-D and single-element values", () => {
    expect(sum(tensor([1, 2, 3], { dtype: "float64" })).item()).toBe(6);
    expect(tensor([42], { dtype: "int32" }).item()).toBe(42);
  });

  it("throws on multi-element tensors", () => {
    expect(() => tensor([1, 2]).item()).toThrow(ShapeError);
  });

  it("respects views and offsets", () => {
    const t = tensor([10, 20, 30], { dtype: "float64" });
    expect(t.slice({ start: 2, end: 3 }).item()).toBe(30);
  });
});

describe("radix sort paths (large lanes)", () => {
  it("matches the small-lane path across the radix threshold", () => {
    setSeed(4242);
    const big = randn([9000]);
    const raw = Array.from(big.data as Float32Array);
    raw[5] = Number.NaN;
    raw[100] = Number.POSITIVE_INFINITY;
    raw[200] = Number.NEGATIVE_INFINITY;
    raw[300] = raw[400] ?? 0; // tie
    const t64 = tensor(raw, { dtype: "float64" });

    const sorted = sort(t64).toArray() as number[];
    // ascending with NaNs last
    for (let i = 1; i < sorted.length; i++) {
      const prev = sorted[i - 1] as number;
      const cur = sorted[i] as number;
      if (Number.isNaN(prev)) expect(Number.isNaN(cur)).toBe(true);
      else if (!Number.isNaN(cur)) expect(prev <= cur).toBe(true);
    }
    expect(sorted.filter((v) => Number.isNaN(v)).length).toBe(1);

    // argsort must be a permutation that sorts the input identically
    const idx = argsort(t64).toArray() as number[];
    expect(new Set(idx).size).toBe(raw.length);
    const viaIdx = idx.map((i) => raw[i] as number);
    expect(viaIdx.map((v) => (Number.isNaN(v) ? "nan" : v))).toEqual(
      sorted.map((v) => (Number.isNaN(v) ? "nan" : v))
    );

    // stability: tied values keep original index order
    const tieVal = raw[300] as number;
    const tiePositions = idx.filter((i) => raw[i] === tieVal);
    for (let i = 1; i < tiePositions.length; i++) {
      expect((tiePositions[i - 1] as number) < (tiePositions[i] as number)).toBe(true);
    }
  });

  it("descending reverses the ascending order", () => {
    setSeed(99);
    const t = tensor(Array.from(randn([8500]).data as Float32Array), { dtype: "float64" });
    const asc = sort(t).toArray() as number[];
    const desc = sort(t, -1, true).toArray() as number[];
    expect(desc).toEqual([...asc].reverse());
  });

  it("treats signed zeros as equal, consistently across packed/radix/comparator paths", () => {
    // -0 and +0 must compare equal (matching numpy and the comparator path);
    // the packed float32 fast path and the radix path must not distinguish
    // them, or the same argsort would return different orders by dtype/size.
    // Use f32-exact inputs (small integers + 0.5 steps) so the float32 and
    // float64 sorts also compare value-for-value, not just by order.
    for (const size of [5, 9000]) {
      const raw = new Array<number>(size).fill(1);
      raw[2] = 0; // +0 at a lower index
      raw[3] = -0; // -0 at a higher index
      if (size > 10) {
        for (let i = 4; i < size; i++) raw[i] = ((i * 7) % 101) - 50 + 0.5; // f32-exact
        raw[100] = -0;
        raw[200] = 0;
      }
      const f32 = tensor(raw, { dtype: "float32" });
      const f64 = tensor(raw, { dtype: "float64" });
      // argsort order is where signed zero actually matters, so they must agree.
      expect(argsort(f32).toArray()).toEqual(argsort(f64).toArray());
      // sorted values agree because every input is exactly representable in f32.
      expect(sort(f32).toArray()).toEqual(sort(f64).toArray());
    }
  });
});
