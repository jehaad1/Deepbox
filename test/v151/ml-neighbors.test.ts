import { describe, expect, it } from "vitest";
import { BallTree, KDTree } from "../../src/ml";

describe("ML-11: KDTree and BallTree copy the input buffer", () => {
  it("KDTree queries are unaffected by later changes to the source buffer", () => {
    const data = new Float64Array([0, 0, 1, 0, 5, 5]);
    const tree = new KDTree(data, 3, 2);
    const before = tree.query(new Float64Array([0.9, 0.1]), 1);
    data.set([100, 100, 100, 101, 100, 102]);
    expect(tree.query(new Float64Array([0.9, 0.1]), 1)).toEqual(before);
    expect(before.indices).toEqual([1]);
  });

  it("BallTree queries are unaffected by later changes to the source buffer", () => {
    const data = new Float64Array([0, 0, 1, 0, 5, 5]);
    const tree = new BallTree(data, 3, 2);
    const before = tree.query(new Float64Array([0.9, 0.1]), 1);
    data.set([100, 100, 100, 101, 100, 102]);
    expect(tree.query(new Float64Array([0.9, 0.1]), 1)).toEqual(before);
    expect(before.indices).toEqual([1]);
  });

  it("only the indexed prefix of a longer buffer is used", () => {
    const data = new Float64Array([0, 0, 1, 0, 5, 5, 9, 9]);
    const tree = new KDTree(data, 3, 2);
    expect(tree.query(new Float64Array([9, 9]), 1).indices).toEqual([2]);
  });
});
