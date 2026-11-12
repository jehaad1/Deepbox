import { describe, expect, it } from "vitest";
import { MemoryError } from "../../src/core";
import { denseToCSR, sparseSolve } from "../../src/linalg";
import { tensor, zeros } from "../../src/ndarray";

// sparseSolve expands the matrix into a dense n x n workspace. Above 2^26 elements
// (n = 8192, 512 MiB of float64) it throws a MemoryError that points to sparseCholeskySolve.

function identityCsr(n: number) {
  const rowPointers = new Int32Array(n + 1);
  const colIndices = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    rowPointers[i + 1] = i + 1;
    colIndices[i] = i;
  }
  return { n, values: new Float64Array(n).fill(1), colIndices, rowPointers };
}

describe("sparseSolve dense workspace limit", () => {
  it("throws MemoryError above the limit and names sparseCholeskySolve", () => {
    const n = 8193;
    expect(() => sparseSolve(identityCsr(n), zeros([n]))).toThrow(MemoryError);
    expect(() => sparseSolve(identityCsr(n), zeros([n]))).toThrow(/sparseCholeskySolve/);
  });

  it("still solves systems below the limit", () => {
    const A = tensor(
      [
        [4, 1, 0],
        [1, 3, 1],
        [0, 1, 2],
      ],
      { dtype: "float64" }
    );
    const x = sparseSolve(denseToCSR(A), tensor([1, 2, 3], { dtype: "float64" }));
    // numpy.linalg.solve(A, [1, 2, 3]) -> [0.22222222, 0.11111111, 1.44444444]
    const got = Array.from(x.toArray() as number[]);
    expect(got[0]).toBeCloseTo(2 / 9, 12);
    expect(got[1]).toBeCloseTo(1 / 9, 12);
    expect(got[2]).toBeCloseTo(13 / 9, 12);
  });
});
