import { describe, expect, it } from "vitest";
import { ConvergenceError } from "../src/core";
import { eig } from "../src/linalg/decomposition/eig";
import { tensor } from "../src/ndarray";

describe("eig convergence safeguards", () => {
  it("throws ConvergenceError when maxIter is too small", () => {
    const A = tensor([
      [4, 1],
      [2, 3],
    ]);
    expect(() => eig(A, { maxIter: 1, tol: 1e-12 })).toThrow(ConvergenceError);
  });
});
