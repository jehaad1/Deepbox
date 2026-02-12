import { describe, expect, it } from "vitest";
import { DeepboxError } from "../src/core/errors";
import { CSRMatrix, tensor } from "../src/ndarray";
import {
  LabelBinarizer,
  LabelEncoder,
  MultiLabelBinarizer,
  OneHotEncoder,
  OrdinalEncoder,
} from "../src/preprocess";

describe("LabelEncoder", () => {
  it("should encode and decode numeric labels", () => {
    const encoder = new LabelEncoder();
    const y = tensor([1, 2, 1, 3]);
    encoder.fit(y);
    const yEncoded = encoder.transform(y);
    expect(yEncoded.shape).toEqual([4]);
    const yDecoded = encoder.inverseTransform(yEncoded);
    expect(yDecoded.shape).toEqual([4]);
  });

  it("should encode and decode string labels", () => {
    const encoder = new LabelEncoder();
    const y = tensor(["cat", "dog", "cat", "bird"]);
    encoder.fit(y);
    const yEncoded = encoder.transform(y);
    const yDecoded = encoder.inverseTransform(yEncoded);
    expect(yEncoded.shape).toEqual([4]);
    expect(yDecoded.dtype).toBe("string");
  });
});

describe("OneHotEncoder", () => {
  it("should encode to dense matrix when sparse=false", () => {
    const X = tensor([
      ["red", "S"],
      ["blue", "M"],
    ]);
    const encoder = new OneHotEncoder({ sparse: false });
    const out = encoder.fitTransform(X);
    expect(out.shape[0]).toBe(2);
  });

  it("should return CSRMatrix when sparse=true", () => {
    const X = tensor([
      ["red", "S"],
      ["blue", "M"],
      ["red", "M"],
    ]);
    const encoder = new OneHotEncoder({ sparse: true });
    const out = encoder.fitTransform(X);
    expect(out).toBeInstanceOf(CSRMatrix);
    if (!(out instanceof CSRMatrix)) {
      throw new DeepboxError("Expected CSRMatrix output");
    }
    const csr = out;
    expect(csr.indptr).toBeDefined();
    // inverse should work via dense conversion
    const inv = encoder.inverseTransform(csr);
    expect(inv.dtype).toBe("string");
    expect(inv.shape).toEqual([3, 2]);
  });
});

describe("OrdinalEncoder", () => {
  it("should encode categorical features as ordinal integers", () => {
    const encoder = new OrdinalEncoder();
    const X = tensor([
      ["red", "S"],
      ["blue", "M"],
      ["red", "L"],
    ]);
    const XEncoded = encoder.fitTransform(X);
    expect(XEncoded.shape).toEqual([3, 2]);
  });
});

describe("LabelBinarizer", () => {
  it("should binarize labels", () => {
    const binarizer = new LabelBinarizer();
    binarizer.fit(tensor([0, 1, 2]));
    const out = binarizer.transform(tensor([0, 1, 2]));
    expect(out.shape[0]).toBe(3);
  });
});

describe("MultiLabelBinarizer", () => {
  it("should binarize multi-label data", () => {
    const binarizer = new MultiLabelBinarizer();
    binarizer.fit([
      ["a", "b"],
      ["b", "c"],
    ]);
    const out = binarizer.transform([["a"], ["c"]]);
    expect(out.shape[0]).toBe(2);
  });
});
