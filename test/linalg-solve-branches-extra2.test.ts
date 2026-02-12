import { describe, expect, it } from "vitest";
import { solveTriangular } from "../src/linalg/solvers";
import { tensor } from "../src/ndarray";

describe("linalg solve branches extra 2", () => {
  it("throws on singular triangular with 2D rhs", () => {
    const U = tensor([
      [0, 1],
      [0, 2],
    ]);
    const B = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(() => solveTriangular(U, B, false)).toThrow(/singular/i);

    const L = tensor([
      [0, 0],
      [1, 2],
    ]);
    expect(() => solveTriangular(L, B, true)).toThrow(/singular/i);
  });
});
