import { describe, expect, it } from "vitest";
import { cond, norm } from "../src/linalg/norms";
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

describe("linalg norms branches 2", () => {
  it("handles axis=[] and negative orders", () => {
    const m = tensor([
      [0, 0],
      [0, 0],
    ]);
    const v = asNumber(norm(m, 2, []));
    expect(v).toBe(0);

    // Negative vector-norm orders are valid (NumPy semantics). On a 1-D
    // vector with axis reduction we get a scalar back.
    expect(asNumber(norm(tensor([1, 2, 3]), -1, 0))).toBeCloseTo(0.5454545, 5);
    // 2-D input with an axis returns a per-slice tensor of the harmonic-style
    // norm; all-zero columns give 0.
    const negNorm = norm(m, -1, 0) as Tensor;
    expect(negNorm.toArray()).toEqual([0, 0]);
  });

  it("rejects unknown matrix ord values", () => {
    const m = tensor([
      [1, 2],
      [3, 4],
    ]);
    // @ts-expect-error - invalid norm order should be rejected at runtime
    expect(() => norm(m, "bad")).toThrow(/Invalid norm order/i);
  });

  it("handles cond frobenius with singular matrices", () => {
    const singular = tensor([
      [1, 2],
      [2, 4],
    ]);
    // Numerically singular → smallest singular value ~machine epsilon, so the
    // condition number is very large but finite (matching NumPy/LAPACK).
    const c = cond(singular, "fro");
    expect(c).toBeGreaterThan(1e12);

    const finite = cond(
      tensor([
        [1, 0],
        [0, 2],
      ]),
      "fro"
    );
    expect(finite).toBeGreaterThan(0);
  });

  it("handles axis reductions with keepdims=false", () => {
    const t = zeros([2, 3]);
    const out = asTensor(norm(t, 0, 1, false));
    expect(out.shape).toEqual([2]);
    expect(out.toArray()).toEqual([0, 0]);
  });
});
