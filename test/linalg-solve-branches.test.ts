import { describe, expect, it } from "vitest";
import { solve, solveTriangular } from "../src/linalg/solvers";
import { allclose, tensor } from "../src/ndarray";
import { matmul } from "../src/ndarray/linalg/basic";
import { toNumArr } from "./_helpers";

describe("solve branches", () => {
  it("validates solve input shapes", () => {
    expect(() => solve(tensor([1, 2]), tensor([1, 2]))).toThrow(/2D matrix/i);
    expect(() => solve(tensor([[1, 2, 3]]), tensor([1]))).toThrow(/square/i);
    expect(() =>
      solve(
        tensor([
          [1, 2],
          [3, 4],
        ]),
        tensor([[1]])
      )
    ).toThrow(/dimensions do not match/i);
    expect(() =>
      solve(
        tensor([
          [1, 2],
          [3, 4],
        ]),
        tensor([
          [1, 2],
          [3, 4],
          [5, 6],
        ])
      )
    ).toThrow(/dimensions do not match/i);
    expect(() =>
      solve(
        tensor([
          [1, 2],
          [3, 4],
        ]),
        tensor([[[1]]])
      )
    ).toThrow(/1D or 2D/i);
  });

  it("solves with multiple RHS", () => {
    const A = tensor([
      [2, 0],
      [0, 2],
    ]);
    const B = tensor([
      [2, 4],
      [6, 8],
    ]);
    const X = solve(A, B);
    expect(X.toArray()).toEqual([
      [1, 2],
      [3, 4],
    ]);
  });

  it("validates triangular solver and handles upper/lower", () => {
    const L = tensor([
      [2, 0],
      [3, 4],
    ]);
    const b = tensor([6, 18]);
    const x = solveTriangular(L, b, true);
    expect(x.toArray()).toEqual([3, 2.25]);

    const U = tensor([
      [2, 1],
      [0, 4],
    ]);
    const y = solveTriangular(U, b, false);
    expect(toNumArr(y.toArray())[0]).toBeCloseTo(0.75, 6);

    const B = tensor([
      [6, 8],
      [18, 20],
    ]);
    const X = solveTriangular(L, B, true);
    expect(X.toArray()).toEqual([
      [3, 4],
      [2.25, 2],
    ]);

    const U2 = tensor([
      [2, 1],
      [0, 4],
    ]);
    const Bu = tensor([
      [6, 8],
      [18, 20],
    ]);
    const Xu = solveTriangular(U2, Bu, false);
    expect(Xu.shape).toEqual([2, 2]);
    const reconstructed = matmul(U2, Xu);
    expect(allclose(reconstructed, Bu, 1e-8, 1e-8)).toBe(true);
  });

  it("detects singular triangular matrices", () => {
    const U = tensor([
      [0, 1],
      [0, 2],
    ]);
    expect(() => solveTriangular(U, tensor([1, 2]), false)).toThrow(/singular/i);

    const L = tensor([
      [0, 0],
      [1, 2],
    ]);
    expect(() => solveTriangular(L, tensor([1, 2]), true)).toThrow(/singular/i);
  });
});
