import { describe, expect, it, vi } from "vitest";
import {
  allclose,
  atan2,
  div,
  floorDiv,
  gather,
  mod,
  randn,
  reciprocal,
  repeat,
  split,
  tensor,
  tile,
} from "../src/ndarray";
import { softplus } from "../src/ndarray/ops/activation";
import { numData } from "./_helpers";

describe("ndarray regression fixes", () => {
  it("mod uses floor semantics for negatives", () => {
    const a = tensor([-3, 3], { dtype: "float32" });
    const b = tensor([2, -2], { dtype: "float32" });
    const r = mod(a, b);
    expect(numData(r)).toEqual([1, -1]);

    const ai = tensor([-3, 3], { dtype: "int64" });
    const bi = tensor([2, -2], { dtype: "int64" });
    const ri = mod(ai, bi);
    expect(Array.from(ri.data as BigInt64Array)).toEqual([1n, -1n]);
  });

  it("floorDiv uses floor semantics for int64 negatives", () => {
    const a = tensor([-3, 3], { dtype: "int64" });
    const b = tensor([2, -2], { dtype: "int64" });
    const r = floorDiv(a, b);
    expect(Array.from(r.data as BigInt64Array)).toEqual([-2n, -2n]);
  });

  it("div and reciprocal promote int64 to float64", () => {
    const a = tensor([3, -3], { dtype: "int64" });
    const b = tensor([2, 2], { dtype: "int64" });
    const d = div(a, b);
    expect(d.dtype).toBe("float64");
    const dVals = numData(d);
    expect(dVals[0]).toBeCloseTo(1.5);
    expect(dVals[1]).toBeCloseTo(-1.5);

    const r = reciprocal(a);
    expect(r.dtype).toBe("float64");
    const rVals = numData(r);
    expect(rVals[0]).toBeCloseTo(1 / 3);
    expect(rVals[1]).toBeCloseTo(-1 / 3);
  });

  it("allclose uses broadcasting semantics", () => {
    const a = tensor([[1], [2]], { dtype: "float64" });
    const b = tensor([[1, 2]], { dtype: "float64" });
    expect(allclose(a, b)).toBe(false);

    const scalar = tensor(1, { dtype: "float64" });
    const c = tensor(
      [
        [1, 1],
        [1, 1],
      ],
      { dtype: "float64" }
    );
    expect(allclose(scalar, c)).toBe(true);
  });

  it("atan2 broadcasts and returns correct shape", () => {
    const y = tensor([[1], [2]], { dtype: "float64" });
    const x = tensor([[1, 2]], { dtype: "float64" });
    const z = atan2(y, x);
    expect(z.shape).toEqual([2, 2]);
    const values = numData(z);
    const expected = [Math.atan2(1, 1), Math.atan2(1, 2), Math.atan2(2, 1), Math.atan2(2, 2)];
    for (let i = 0; i < expected.length; i++) {
      expect(values[i]).toBeCloseTo(expected[i]);
    }
  });

  it("repeat validates repeats and axis", () => {
    const t = tensor([1, 2]);
    expect(() => repeat(t, -1)).toThrow(/repeats must be a non-negative integer/i);
    expect(() => repeat(t, 2.5)).toThrow(/repeats must be a non-negative integer/i);
    expect(() => repeat(t, 2, 0.5)).toThrow(/axis must be an integer/i);
  });

  it("tile validates reps", () => {
    const t = tensor([1, 2]);
    expect(() => tile(t, [-1])).toThrow(/reps\[0\] must be a non-negative integer/i);
    expect(() => tile(t, [1.5])).toThrow(/reps\[0\] must be a non-negative integer/i);
  });

  it("split validates sections and indices", () => {
    const t = tensor([1, 2, 3]);
    expect(() => split(t, 0)).toThrow(/positive integer/i);
    expect(() => split(t, [2, 1])).toThrow(/non-decreasing/i);
    expect(() => split(t, [4])).toThrow(/out of bounds/i);
  });

  it("gather supports negative axis", () => {
    const t = tensor(
      [
        [1, 2],
        [3, 4],
      ],
      { dtype: "float32" }
    );
    const indices = tensor([1], { dtype: "int32" });
    const g = gather(t, indices, -1);
    expect(g.shape).toEqual([2, 1]);
    expect(numData(g)).toEqual([2, 4]);
  });

  it("randn avoids log(0) edge case", () => {
    const spy = vi
      .spyOn(Math, "random")
      .mockImplementationOnce(() => 0)
      .mockImplementationOnce(() => 0.5)
      .mockImplementation(() => 0.5);

    const t = randn([2], { dtype: "float64" });
    const values = numData(t);
    for (const v of values) {
      expect(Number.isFinite(v)).toBe(true);
    }
    spy.mockRestore();
  });

  it("softplus is numerically stable for large positive values", () => {
    const t = tensor([1000], { dtype: "float64" });
    const out = softplus(t);
    const v = Number(out.data[0]);
    expect(Number.isFinite(v)).toBe(true);
    expect(v).toBeCloseTo(1000, 6);
  });
});
