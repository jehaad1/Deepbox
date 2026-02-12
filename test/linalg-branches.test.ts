import { describe, expect, it } from "vitest";
import { inv, pinv } from "../src/linalg/inverse";
import { cond, norm } from "../src/linalg/norms";
import { det, matrixRank, trace } from "../src/linalg/properties";
import { solve, solveTriangular } from "../src/linalg/solvers";
import { lstsq } from "../src/linalg/solvers/lstsq";
import { allclose, eye, tensor } from "../src/ndarray";
import { matmul } from "../src/ndarray/linalg/basic";

describe("Linalg - Branch Coverage", () => {
  it("covers properties branches", () => {
    const m = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(det(m)).toBeCloseTo(-2, 6);
    expect(trace(m).toArray()).toEqual([5]);
    expect(matrixRank(m)).toBe(2);
    expect(cond(m)).toBeCloseTo(14.933034, 5);
    expect(cond(eye(2))).toBeCloseTo(1, 8);

    expect(() => trace(tensor([1, 2]))).toThrow(/at least 2D/i);
  });

  it("covers inverse branches", () => {
    const m = tensor([
      [1, 2],
      [3, 4],
    ]);
    const invM = inv(m);
    expect(invM.shape).toEqual([2, 2]);
    expect(allclose(matmul(m, invM), eye(2), 1e-8, 1e-8)).toBe(true);

    const pinvM = pinv(m);
    expect(pinvM.shape).toEqual([2, 2]);

    expect(() => inv(tensor([[1, 2, 3]]))).toThrow(/square/);
  });

  it("covers norm branches", () => {
    const v = tensor([1, -2, 3]);
    expect(norm(v)).toBeGreaterThan(0);
    expect(norm(v, 1)).toBe(6);
    expect(norm(v, Infinity)).toBe(3);

    const m = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(norm(m, "fro")).toBeCloseTo(5.477225, 5);
    expect(norm(m, "nuc")).toBeCloseTo(5.8309519, 5);
  });

  it("covers solvers branches", () => {
    const A = tensor([
      [3, 1],
      [1, 2],
    ]);
    const b = tensor([9, 8]);
    const x = solve(A, b);
    expect(x.shape).toEqual([2]);
    expect(Number(x.data[x.offset])).toBeCloseTo(2, 6);
    expect(Number(x.data[x.offset + 1])).toBeCloseTo(3, 6);

    const tri = tensor([
      [2, 0],
      [3, 4],
    ]);
    const xt = solveTriangular(tri, tensor([6, 18]));
    expect(xt.shape).toEqual([2]);
    expect(Number(xt.data[xt.offset])).toBeCloseTo(3, 6);
    expect(Number(xt.data[xt.offset + 1])).toBeCloseTo(2.25, 6);

    const ls = lstsq(A, b);
    expect(ls.x.shape).toEqual([2]);
    expect(Number(ls.x.data[ls.x.offset])).toBeCloseTo(2, 6);
    expect(Number(ls.x.data[ls.x.offset + 1])).toBeCloseTo(3, 6);
  });
});
