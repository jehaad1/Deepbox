import { describe, expect, it } from "vitest";
import { solve, solveTriangular } from "../src/linalg/solvers";
import { tensor } from "../src/ndarray";

describe("linalg solvers error branches", () => {
  it("validates solve inputs", () => {
    const A = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(() => solve(tensor([1, 2]), tensor([1, 2]))).toThrow(/2D matrix/);
    expect(() => solve(tensor([[1, 2, 3]]), tensor([1]))).toThrow(/square/);
    expect(() => solve(A, tensor([[1], [2], [3]]))).toThrow(/dimensions/);
    expect(() =>
      solve(
        A,
        tensor([
          [1, 2],
          [3, 4],
          [5, 6],
        ])
      )
    ).toThrow(/dimensions/);
  });

  it("validates solveTriangular inputs and singularity", () => {
    const tri = tensor([
      [1, 2],
      [0, 0],
    ]);
    expect(() => solveTriangular(tri, tensor([[1], [2]]))).toThrow(/singular/i);
    expect(() => solveTriangular(tri, tensor([1, 2, 3]))).toThrow(/dimensions/);
    expect(() => solveTriangular(tri, tensor([1, 2]))).toThrow(/singular/i);
  });
});
