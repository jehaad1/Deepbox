import { describe, expect, it } from "vitest";
import { block_diag, expm, logm, sqrtm } from "../src/linalg";
import { tensor } from "../src/ndarray";

describe("block_diag", () => {
  it("constructs block diagonal from two matrices", () => {
    const A = tensor([
      [1, 2],
      [3, 4],
    ]);
    const B = tensor([[5]]);
    const D = block_diag(A, B);
    expect(D.shape).toEqual([3, 3]);
    expect(D.toArray()).toEqual([
      [1, 2, 0],
      [3, 4, 0],
      [0, 0, 5],
    ]);
  });

  it("constructs block diagonal from three matrices", () => {
    const A = tensor([[1]]);
    const B = tensor([
      [2, 3],
      [4, 5],
    ]);
    const C = tensor([[6]]);
    const D = block_diag(A, B, C);
    expect(D.shape).toEqual([4, 4]);
    expect(D.toArray()).toEqual([
      [1, 0, 0, 0],
      [0, 2, 3, 0],
      [0, 4, 5, 0],
      [0, 0, 0, 6],
    ]);
  });

  it("handles single matrix", () => {
    const A = tensor([
      [1, 2],
      [3, 4],
    ]);
    const D = block_diag(A);
    expect(D.toArray()).toEqual([
      [1, 2],
      [3, 4],
    ]);
  });

  it("handles no arguments", () => {
    const D = block_diag();
    expect(D.shape).toEqual([0, 0]);
  });

  it("throws on non-2D input", () => {
    expect(() => block_diag(tensor([1, 2]))).toThrow();
  });
});

describe("expm", () => {
  it("exp of zero matrix is identity", () => {
    const A = tensor([
      [0, 0],
      [0, 0],
    ]);
    const E = expm(A);
    expect(E.shape).toEqual([2, 2]);
    const arr = E.toArray() as number[][];
    expect(arr[0]![0]!).toBeCloseTo(1, 5);
    expect(arr[0]![1]!).toBeCloseTo(0, 5);
    expect(arr[1]![0]!).toBeCloseTo(0, 5);
    expect(arr[1]![1]!).toBeCloseTo(1, 5);
  });

  it("exp of diagonal matrix", () => {
    const A = tensor([
      [1, 0],
      [0, 2],
    ]);
    const E = expm(A);
    const arr = E.toArray() as number[][];
    expect(arr[0]![0]!).toBeCloseTo(Math.exp(1), 5);
    expect(arr[0]![1]!).toBeCloseTo(0, 5);
    expect(arr[1]![0]!).toBeCloseTo(0, 5);
    expect(arr[1]![1]!).toBeCloseTo(Math.exp(2), 5);
  });

  it("exp of symmetric matrix", () => {
    // Symmetric matrix: [[2, 1], [1, 2]]
    // Eigenvalues: 3, 1; exp: e^3, e^1
    const A = tensor([
      [2, 1],
      [1, 2],
    ]);
    const E = expm(A);
    expect(E.shape).toEqual([2, 2]);
    // Verify E is symmetric
    const arr = E.toArray() as number[][];
    expect(arr[0]![1]!).toBeCloseTo(arr[1]![0]!, 5);
  });
});

describe("logm", () => {
  it("log of identity is zero matrix", () => {
    const I = tensor([
      [1, 0],
      [0, 1],
    ]);
    const L = logm(I);
    const arr = L.toArray() as number[][];
    expect(arr[0]![0]!).toBeCloseTo(0, 5);
    expect(arr[0]![1]!).toBeCloseTo(0, 5);
    expect(arr[1]![0]!).toBeCloseTo(0, 5);
    expect(arr[1]![1]!).toBeCloseTo(0, 5);
  });

  it("log of diagonal matrix", () => {
    const A = tensor([
      [Math.E, 0],
      [0, Math.E * Math.E],
    ]);
    const L = logm(A);
    const arr = L.toArray() as number[][];
    expect(arr[0]![0]!).toBeCloseTo(1, 5);
    expect(arr[1]![1]!).toBeCloseTo(2, 5);
    expect(arr[0]![1]!).toBeCloseTo(0, 5);
    expect(arr[1]![0]!).toBeCloseTo(0, 5);
  });
});

describe("sqrtm", () => {
  it("sqrt of identity is identity", () => {
    const I = tensor([
      [1, 0],
      [0, 1],
    ]);
    const S = sqrtm(I);
    const arr = S.toArray() as number[][];
    expect(arr[0]![0]!).toBeCloseTo(1, 5);
    expect(arr[0]![1]!).toBeCloseTo(0, 5);
    expect(arr[1]![0]!).toBeCloseTo(0, 5);
    expect(arr[1]![1]!).toBeCloseTo(1, 5);
  });

  it("sqrt of diagonal matrix", () => {
    const A = tensor([
      [4, 0],
      [0, 9],
    ]);
    const S = sqrtm(A);
    const arr = S.toArray() as number[][];
    expect(arr[0]![0]!).toBeCloseTo(2, 5);
    expect(arr[1]![1]!).toBeCloseTo(3, 5);
    expect(arr[0]![1]!).toBeCloseTo(0, 5);
    expect(arr[1]![0]!).toBeCloseTo(0, 5);
  });

  it("sqrtm(A) * sqrtm(A) ≈ A for symmetric positive definite", () => {
    const A = tensor([
      [5, 2],
      [2, 5],
    ]);
    const S = sqrtm(A);
    // Multiply S * S
    const s = S.toArray() as number[][];
    const product: number[][] = [
      [0, 0],
      [0, 0],
    ];
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        for (let k = 0; k < 2; k++) {
          product[i]![j]! += s[i]![k]! * s[k]![j]!;
        }
      }
    }
    const orig = A.toArray() as number[][];
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        expect(product[i]![j]!).toBeCloseTo(orig[i]![j]!, 5);
      }
    }
  });
});
