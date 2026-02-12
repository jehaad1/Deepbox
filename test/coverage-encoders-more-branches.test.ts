import { describe, expect, it } from "vitest";
import { CSRMatrix, zeros } from "../src/ndarray";
import { MultiLabelBinarizer } from "../src/preprocess/encoders";

describe("preprocess encoders extra branches", () => {
  it("MultiLabelBinarizer constructor and fit errors", () => {
    expect(() => new MultiLabelBinarizer({ sparse: true, sparseOutput: false })).toThrow(/sparse/);
    expect(() => new MultiLabelBinarizer({ classes: [] })).toThrow(/at least one value/i);

    const mlb = new MultiLabelBinarizer();
    expect(() => mlb.fit([])).toThrow(/empty array/);
    expect(() => mlb.fit([1 as unknown as string[]])).toThrow(/label arrays/);
    expect(() => mlb.fit([[true as unknown as string]])).toThrow(/labels must be/);
  });

  it("MultiLabelBinarizer transform and inverse branches", () => {
    const mlb = new MultiLabelBinarizer({ classes: ["a", "b"] });
    expect(() => mlb.transform([["a"]])).toThrow(/fitted/);
    expect(() => mlb.fit([["a", "c"]])).toThrow(/Unknown label/);

    mlb.fit([["a"], ["b"]]);
    const dense = mlb.transform([["a", "b"], []]);
    if (dense instanceof CSRMatrix) {
      throw new Error("Expected dense Tensor output");
    }
    expect(dense.shape).toEqual([2, 2]);
    expect(() => mlb.transform([["c"]])).toThrow(/Unknown label/);

    const inv = mlb.inverseTransform(dense);
    expect(inv[0]).toEqual(["a", "b"]);
    expect(inv[1]).toEqual([]);

    expect(() => mlb.inverseTransform(zeros([1, 3]))).toThrow(/column count/);

    const sparse = new MultiLabelBinarizer({ sparse: true });
    sparse.fit([["x"], ["y", "x"]]);
    const sparseOut = sparse.transform([["x", "x"], []]);
    expect(sparseOut).toBeInstanceOf(CSRMatrix);
    const invSparse = sparse.inverseTransform(sparseOut as CSRMatrix);
    expect(invSparse.length).toBe(2);
  });

  it("MultiLabelBinarizer empty transforms", () => {
    const sparse = new MultiLabelBinarizer({ sparse: true });
    sparse.fit([["a"]]);
    const emptySparse = sparse.transform([]);
    expect(emptySparse).toBeInstanceOf(CSRMatrix);

    const dense = new MultiLabelBinarizer();
    dense.fit([["a"]]);
    const emptyDense = dense.transform([]);
    if (emptyDense instanceof CSRMatrix) {
      throw new Error("Expected dense Tensor output");
    }
    expect(emptyDense.shape).toEqual([0, 1]);
  });
});
