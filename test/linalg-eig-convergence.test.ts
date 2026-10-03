import { describe, expect, it } from "vitest";
import { ConvergenceError } from "../src/core";
import { eig } from "../src/linalg/decomposition/eig";
import { tensor } from "../src/ndarray";

describe("eig convergence safeguards", () => {
  it("throws ConvergenceError when maxIter is too small", () => {
    // 2x2 blocks are solved in closed form and need no QR sweeps, so use a 5x5
    // matrix with real eigenvalues 1..5 that needs several sweeps.
    const A = tensor([
      [4.75, -1.5, 0.75, -0.75, 1.0],
      [-0.25, 1.5, 0.75, -0.75, 1.0],
      [-1.0, 0.0, 3.0, 1.0, 0.0],
      [-1.75, -0.5, 0.25, 3.75, 3.0],
      [3.75, -1.5, 0.75, -0.75, 2.0],
    ]);
    expect(() => eig(A, { maxIter: 1, tol: 1e-12 })).toThrow(ConvergenceError);
  });
});
