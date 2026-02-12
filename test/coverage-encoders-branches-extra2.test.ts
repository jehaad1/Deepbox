import { describe, expect, it } from "vitest";
import { CSRMatrix, tensor, zeros } from "../src/ndarray";
import {
  LabelBinarizer,
  LabelEncoder,
  OneHotEncoder,
  OrdinalEncoder,
} from "../src/preprocess/encoders";
import { toNumberMatrix } from "./preprocess-test-helpers";

describe("preprocess encoders additional branch coverage", () => {
  it("LabelEncoder handles bigint labels", () => {
    const y = tensor(new BigInt64Array([1n, 2n, 1n, 3n]), { dtype: "int64" });
    const encoder = new LabelEncoder();
    const encoded = encoder.fitTransform(y);
    expect(encoded.dtype).toBe("float64");

    const decoded = encoder.inverseTransform(encoded);
    expect(decoded.dtype).toBe("int64");
    expect(decoded.toArray()).toEqual([1n, 2n, 1n, 3n]);
  });

  it("OneHotEncoder ignores unknown categories in sparse mode", () => {
    const X = tensor([["red"], ["blue"]]);
    const encoder = new OneHotEncoder({ sparse: true, handleUnknown: "ignore" });
    encoder.fit(X);

    const out = encoder.transform(tensor([["green"], ["red"]]));
    expect(out).toBeInstanceOf(CSRMatrix);
    if (!(out instanceof CSRMatrix)) {
      throw new Error("Expected CSRMatrix output");
    }
    const dense = out.toDense();
    const arr = toNumberMatrix(dense, "dense");
    expect(arr[0]).toEqual([0, 0]);
    const rowSum = (arr[1] ?? []).reduce((sum, v) => sum + v, 0);
    expect(rowSum).toBe(1);
  });

  it("OneHotEncoder inverseTransform maps all-zero rows when drop is set", () => {
    const X = tensor([["a"], ["b"]]);
    const encoder = new OneHotEncoder({ drop: "first", sparse: false });
    encoder.fit(X);

    const zerosInput = zeros([1, 1]);
    const decoded = encoder.inverseTransform(zerosInput);
    expect(decoded.toArray()).toEqual([["a"]]);
  });

  it("OrdinalEncoder inverseTransform on empty inputs preserves category dtype", () => {
    const stringEnc = new OrdinalEncoder();
    stringEnc.fit(tensor([["a"], ["b"]]));
    const empty = zeros([0, 1]);
    const stringOut = stringEnc.inverseTransform(empty);
    expect(stringOut.shape).toEqual([0, 1]);
    expect(stringOut.dtype).toBe("string");

    const bigEnc = new OrdinalEncoder();
    const bigX = tensor(new BigInt64Array([1n, 2n]), { dtype: "int64" }).reshape([2, 1]);
    bigEnc.fit(bigX);
    const bigOut = bigEnc.inverseTransform(zeros([0, 1]));
    expect(bigOut.dtype).toBe("int64");
  });

  it("OrdinalEncoder round-trips bigint categories", () => {
    const bigEnc = new OrdinalEncoder();
    const bigX = tensor(new BigInt64Array([1n, 2n, 1n, 2n]), { dtype: "int64" }).reshape([2, 2]);
    const encoded = bigEnc.fitTransform(bigX);
    const decoded = bigEnc.inverseTransform(encoded);
    expect(decoded.dtype).toBe("int64");
    expect(decoded.toArray()).toEqual([
      [1n, 2n],
      [1n, 2n],
    ]);
  });

  it("LabelBinarizer handles empty transforms for dense and sparse outputs", () => {
    const dense = new LabelBinarizer({ sparse: false });
    dense.fit(tensor(["a", "b"]));
    const denseOut = dense.transform(tensor([]));
    if (denseOut instanceof CSRMatrix) {
      throw new Error("Expected dense output");
    }
    expect(denseOut.shape).toEqual([0, 2]);

    const sparse = new LabelBinarizer({ sparse: true });
    sparse.fit(tensor(["a", "b"]));
    const sparseOut = sparse.transform(tensor([]));
    expect(sparseOut).toBeInstanceOf(CSRMatrix);
  });
});
