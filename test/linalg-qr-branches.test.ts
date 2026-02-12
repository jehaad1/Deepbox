import { describe, expect, it } from "vitest";
import { qr } from "../src/linalg/decomposition/qr";
import { allclose, tensor, transpose } from "../src/ndarray";
import { matmul } from "../src/ndarray/linalg/basic";
import { toNum2D } from "./_helpers";

describe("qr branch coverage", () => {
  it("handles reduced QR with zero columns", () => {
    const A = tensor([
      [0, 0],
      [0, 0],
    ]);
    const [Q, R] = qr(A, "reduced");
    expect(Q.shape).toEqual([2, 2]);
    expect(R.shape).toEqual([2, 2]);
    expect(R.toArray()).toEqual([
      [0, 0],
      [0, 0],
    ]);
  });

  it("handles complete QR with dependent columns", () => {
    const A = tensor([[0], [2]]);
    const [Q, R] = qr(A, "complete");
    expect(Q.shape).toEqual([2, 2]);
    expect(R.shape).toEqual([2, 1]);

    const q = toNum2D(Q.toArray());
    const r = toNum2D(R.toArray());
    const reconstructed = [
      [q[0][0] * r[0][0] + q[0][1] * r[1][0]],
      [q[1][0] * r[0][0] + q[1][1] * r[1][0]],
    ];
    expect(reconstructed[0][0]).toBeCloseTo(0, 6);
    expect(reconstructed[1][0]).toBeCloseTo(2, 6);
  });

  it("handles m < n inputs", () => {
    const A = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const [Q, R] = qr(A);
    expect(Q.shape).toEqual([2, 2]);
    expect(R.shape).toEqual([2, 3]);
    const reconstructed = matmul(Q, R);
    expect(allclose(reconstructed, A, 1e-8, 1e-8)).toBe(true);
  });

  it("produces orthonormal Q for rank-deficient inputs", () => {
    const A = tensor([
      [1, 2],
      [2, 4],
    ]);
    const [Q, _R] = qr(A, "reduced");
    const QtQ = toNum2D(matmul(transpose(Q), Q).toArray());
    for (let i = 0; i < QtQ.length; i++) {
      const row = QtQ[i];
      expect(row).toBeDefined();
      if (!row) continue;
      for (let j = 0; j < row.length; j++) {
        const val = row[j];
        expect(val).toBeDefined();
        if (val === undefined) continue;
        const expected = i === j ? 1 : 0;
        expect(val).toBeCloseTo(expected, 6);
      }
    }
  });
});
