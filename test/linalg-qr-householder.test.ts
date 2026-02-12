import { describe, expect, it } from "vitest";
import { qr } from "../src/linalg/decomposition/qr";
import { allclose, dot, eye, tensor, transpose } from "../src/ndarray";

describe("QR Decomposition (Householder)", () => {
  it("factorizes square matrix", () => {
    const A = tensor([
      [12, -51, 4],
      [6, 167, -68],
      [-4, 24, -41],
    ]);
    const [Q, R] = qr(A);

    // Check dimensions
    expect(Q.shape).toEqual([3, 3]);
    expect(R.shape).toEqual([3, 3]);

    // Check orthogonality: Q^T Q = I
    const I = eye(3);
    const QTQ = dot(transpose(Q), Q);
    expect(allclose(QTQ, I, 1e-10, 1e-10)).toBe(true);

    // Check upper triangularity of R
    // R[1,0], R[2,0], R[2,1] should be 0
    expect(Number(R.data[R.offset + 1 * 3 + 0])).toBeCloseTo(0, 10);
    expect(Number(R.data[R.offset + 2 * 3 + 0])).toBeCloseTo(0, 10);
    expect(Number(R.data[R.offset + 2 * 3 + 1])).toBeCloseTo(0, 10);

    // Check reconstruction: Q R = A
    const QR = dot(Q, R);
    expect(allclose(QR, A, 1e-10, 1e-10)).toBe(true);
  });

  it("factorizes tall matrix (reduced mode)", () => {
    // 4x3 matrix
    const A = tensor([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
      [10, 11, 12],
    ]);
    const [Q, R] = qr(A, "reduced");

    // Q should be 4x3, R should be 3x3
    expect(Q.shape).toEqual([4, 3]);
    expect(R.shape).toEqual([3, 3]);

    // Q^T Q = I (3x3)
    const I = eye(3);
    const QTQ = dot(transpose(Q), Q);
    expect(allclose(QTQ, I, 1e-10, 1e-10)).toBe(true);

    // Check reconstruction
    const QR = dot(Q, R);
    expect(allclose(QR, A, 1e-10, 1e-10)).toBe(true);
  });

  it("factorizes tall matrix (complete mode)", () => {
    // 4x3 matrix
    const A = tensor([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
      [10, 11, 12],
    ]);
    const [Q, R] = qr(A, "complete");

    // Q should be 4x4, R should be 4x3
    expect(Q.shape).toEqual([4, 4]);
    expect(R.shape).toEqual([4, 3]);

    // Q^T Q = I (4x4)
    const I = eye(4);
    const QTQ = dot(transpose(Q), Q);
    expect(allclose(QTQ, I, 1e-10, 1e-10)).toBe(true);

    // Check reconstruction
    const QR = dot(Q, R);
    expect(allclose(QR, A, 1e-10, 1e-10)).toBe(true);
  });

  it("factorizes wide matrix", () => {
    // 2x3 matrix
    const A = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const [Q, R] = qr(A, "reduced");

    // min(2,3) = 2
    // Q should be 2x2, R should be 2x3
    expect(Q.shape).toEqual([2, 2]);
    expect(R.shape).toEqual([2, 3]);

    // Q^T Q = I (2x2)
    const I = eye(2);
    const QTQ = dot(transpose(Q), Q);
    expect(allclose(QTQ, I, 1e-10, 1e-10)).toBe(true);

    // Check reconstruction
    const QR = dot(Q, R);
    expect(allclose(QR, A, 1e-10, 1e-10)).toBe(true);
  });

  it("handles singular matrix", () => {
    // Rank deficient matrix
    const A = tensor([
      [1, 1],
      [1, 1],
    ]);
    const [Q, R] = qr(A);

    expect(allclose(dot(Q, R), A, 1e-10, 1e-10)).toBe(true);
  });
});
