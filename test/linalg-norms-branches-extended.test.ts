import { describe, expect, it } from "vitest";
import { norm } from "../src/linalg/norms";
import type { Tensor } from "../src/ndarray";
import { tensor, zeros } from "../src/ndarray";

function asTensor(v: Tensor | number): Tensor {
  if (typeof v === "number") throw new Error("expected Tensor, got number");
  return v;
}

function asNumber(v: Tensor | number): number {
  if (typeof v !== "number") throw new Error("expected number, got Tensor");
  return v;
}

describe("linalg norms branch coverage", () => {
  it("rejects string tensors and invalid axis usage", () => {
    expect(() => norm(tensor(["a", "b"]))).toThrow(/string/i);

    const m = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(() => norm(m, "nuc", 0)).toThrow(/axis is only supported/i);
    expect(() => norm(m, 2, 2)).toThrow(/out of bounds/i);
    expect(() => norm(m, 2, [0, 0])).toThrow(/duplicate axis/i);
  });

  it("handles axis reductions with keepdims and special orders", () => {
    const m = tensor([
      [1, -2, 3],
      [4, 0, -6],
    ]);

    const l0 = asTensor(norm(m, 0, 1, true));
    expect(l0.shape).toEqual([2, 1]);
    expect(l0.toArray()).toEqual([[3], [2]]);

    const linf = asTensor(norm(m, Number.POSITIVE_INFINITY, 0));
    expect(linf.toArray()).toEqual([4, 2, 6]);

    const lfro = asTensor(norm(m, "fro", 1));
    const l2 = asTensor(norm(m, 2, 1));
    expect(lfro.toArray()).toEqual(l2.toArray());
  });

  it("handles empty reductions and negative orders", () => {
    const empty = zeros([0, 2]);
    const ninf = asTensor(norm(empty, Number.NEGATIVE_INFINITY, 0));
    expect(ninf.toArray()).toEqual([0, 0]);

    const v = zeros([0]);
    expect(() => norm(v, -1)).toThrow(/Vector norm order/i);
  });

  it("rejects >2D when axis is omitted", () => {
    const t = zeros([2, 2, 2]);
    expect(() => norm(t)).toThrow(/1D or 2D/i);
  });

  it("covers matrix norms without axis", () => {
    const m = tensor([
      [1, -2],
      [3, -4],
    ]);
    expect(norm(m, 1)).toBe(6); // max column sum
    expect(norm(m, -1)).toBe(4); // min column sum
    expect(norm(m, Number.POSITIVE_INFINITY)).toBe(7); // max row sum
    expect(norm(m, Number.NEGATIVE_INFINITY)).toBe(3); // min row sum
    expect(norm(m, 2)).toBeGreaterThan(0);
    expect(norm(m, -2)).toBeGreaterThanOrEqual(0);
    expect(norm(m, "nuc")).toBeGreaterThan(0);
  });

  it("supports negative axes and multi-axis reductions", () => {
    const m = tensor([
      [1, 2],
      [3, 4],
    ]);
    const n = asTensor(norm(m, 0, [-1]));
    expect(n.toArray()).toEqual([2, 2]);

    const ord1 = asNumber(norm(m, 1, [0, 1]));
    expect(ord1).toBe(6);

    const ord2 = asNumber(norm(m, 2, [0, 1]));
    expect(ord2).toBeCloseTo(Math.sqrt(15 + Math.sqrt(221)), 6);
  });
});
