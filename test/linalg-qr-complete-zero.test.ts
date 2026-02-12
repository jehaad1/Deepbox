import { describe, expect, it } from "vitest";
import { qr } from "../src/linalg/decomposition/qr";
import { allclose, eye, tensor, transpose } from "../src/ndarray";
import { matmul } from "../src/ndarray/linalg/basic";

describe("linalg qr complete zero", () => {
  it("handles complete QR for zero matrix", () => {
    const A = tensor([
      [0, 0],
      [0, 0],
    ]);
    const [Q, R] = qr(A, "complete");
    expect(Q.shape).toEqual([2, 2]);
    expect(R.shape).toEqual([2, 2]);
    const QTQ = matmul(transpose(Q), Q);
    expect(allclose(QTQ, eye(2), 1e-8, 1e-8)).toBe(true);
    expect(R.toArray()).toEqual([
      [0, 0],
      [0, 0],
    ]);
  });
});
