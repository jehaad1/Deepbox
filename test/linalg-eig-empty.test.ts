import { describe, expect, it } from "vitest";
import { eig, eigh, eigvals } from "../src/linalg/decomposition/eig";
import { zeros } from "../src/ndarray";

describe("linalg eig empty branches", () => {
  it("handles empty matrices", () => {
    const A = zeros([0, 0]);
    const [vals, vecs] = eig(A);
    expect(vals.shape).toEqual([0]);
    expect(vecs.shape).toEqual([0, 0]);

    const valsOnly = eigvals(A);
    expect(valsOnly.shape).toEqual([0]);

    const [svals, svecs] = eigh(A);
    expect(svals.shape).toEqual([0]);
    expect(svecs.shape).toEqual([0, 0]);
  });
});
