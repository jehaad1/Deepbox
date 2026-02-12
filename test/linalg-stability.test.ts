import { describe, expect, it } from "vitest";
import { pinv, solve } from "../src/linalg";
import { dot, tensor } from "../src/ndarray";
import { matmul } from "../src/ndarray/linalg/basic";

function randomMatrix(rows: number, cols: number, seed: number = 0): ReturnType<typeof tensor> {
  const data = new Float64Array(rows * cols);
  // Simple LCG for deterministic random numbers in tests
  let state = seed;
  const next = () => {
    state = (state * 1664525 + 1013904223) % 4294967296;
    return (state / 4294967296) * 2 - 1; // [-1, 1]
  };

  for (let i = 0; i < data.length; i++) {
    data[i] = next();
  }
  return tensor(data).view([rows, cols]);
}

/**
 * Helper to compute Frobenius norm of difference between two tensors
 */
function frobeniusDiff(A: ReturnType<typeof tensor>, B: ReturnType<typeof tensor>): number {
  let sum = 0;
  for (let i = 0; i < A.size; i++) {
    const diff = Number(A.data[A.offset + i]) - Number(B.data[B.offset + i]);
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

describe("deepbox/linalg - numerical stability", () => {
  it("should satisfy A * pinv(A) * A ≈ A for singular matrix", () => {
    const A = tensor([
      [1, 2],
      [2, 4],
    ]);
    const Ap = pinv(A);
    const reconstructed = dot(dot(A, Ap), A);

    for (let i = 0; i < A.size; i++) {
      const v = Number(A.data[A.offset + i]);
      const r = Number(reconstructed.data[reconstructed.offset + i]);
      expect(r).toBeCloseTo(v, 4);
    }
  });

  it("pinv throws on non-finite values", () => {
    const A = tensor([
      [1, NaN],
      [2, 3],
    ]);
    expect(() => pinv(A)).toThrow(/non-finite/i);
  });

  it("should solve random linear systems A * x = b correctly", () => {
    const n = 10;
    // Generate a random matrix with likely non-zero determinant
    const A = randomMatrix(n, n, 42);
    // Ensure diagonal dominance for stability (though solve should handle general cases)
    if (!(A.data instanceof Float64Array)) {
      throw new Error("Expected Float64Array");
    }
    for (let i = 0; i < n; i++) {
      A.data[A.offset + i * n + i] += 10;
    }

    const x_expected = randomMatrix(n, 1, 123);
    const b = matmul(A, x_expected);

    const x_computed = solve(A, b);

    expect(frobeniusDiff(x_computed, x_expected)).toBeLessThan(1e-7);
  });

  it("should handle larger random systems (50x50)", () => {
    const n = 50;
    const A = randomMatrix(n, n, 999);
    // Add identity to ensure non-singularity
    if (!(A.data instanceof Float64Array)) {
      throw new Error("Expected Float64Array");
    }
    for (let i = 0; i < n; i++) {
      A.data[A.offset + i * n + i] += 5;
    }

    const x_expected = randomMatrix(n, 1, 888);
    const b = matmul(A, x_expected);

    const x_computed = solve(A, b);

    // Tolerance might need to be slightly looser for larger systems due to accumulated error
    expect(frobeniusDiff(x_computed, x_expected)).toBeLessThan(1e-6);
  });

  it("should maintain A * pinv(A) * A ≈ A for random rectangular matrices", () => {
    // 10x5 matrix (tall)
    const A = randomMatrix(10, 5, 555);
    const Ap = pinv(A);
    const reconstructed = matmul(matmul(A, Ap), A);
    expect(frobeniusDiff(reconstructed, A)).toBeLessThan(1e-8);

    // 5x10 matrix (wide)
    const B = randomMatrix(5, 10, 777);
    const Bp = pinv(B);
    const reconstructedB = matmul(matmul(B, Bp), B);
    expect(frobeniusDiff(reconstructedB, B)).toBeLessThan(1e-8);
  });
});
