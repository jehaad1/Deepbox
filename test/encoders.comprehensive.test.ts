import { describe, expect, it } from "vitest";
import { DeepboxError } from "../src/core/errors";
import { CSRMatrix, tensor, transpose, zeros } from "../src/ndarray";
import {
  LabelBinarizer,
  LabelEncoder,
  MultiLabelBinarizer,
  OneHotEncoder,
  OrdinalEncoder,
} from "../src/preprocess";
import {
  getFloat64Data,
  getStringData,
  toNumberArray,
  toNumberMatrix,
} from "./preprocess-test-helpers";

describe("LabelEncoder - Comprehensive Tests", () => {
  describe("Basic Functionality", () => {
    it("should encode numeric labels correctly", () => {
      const encoder = new LabelEncoder();
      const y = tensor([1, 2, 1, 3]);
      encoder.fit(y);
      const yEncoded = encoder.transform(y);

      expect(yEncoded.shape).toEqual([4]);
      expect(Array.from(getFloat64Data(yEncoded))).toEqual([0, 1, 0, 2]);
    });

    it("should encode string labels correctly", () => {
      const encoder = new LabelEncoder();
      const y = tensor(["cat", "dog", "cat", "bird"]);
      encoder.fit(y);
      const yEncoded = encoder.transform(y);

      expect(yEncoded.shape).toEqual([4]);
      expect(yEncoded.dtype).toBe("float64");
    });

    it("should handle fitTransform correctly", () => {
      const encoder = new LabelEncoder();
      const y = tensor(["a", "b", "c", "a"]);
      const yEncoded = encoder.fitTransform(y);

      expect(yEncoded.shape).toEqual([4]);
      expect(Array.from(getFloat64Data(yEncoded))).toEqual([0, 1, 2, 0]);
    });

    it("should inverse transform correctly for numeric labels", () => {
      const encoder = new LabelEncoder();
      const y = tensor([1, 2, 1, 3]);
      const yEncoded = encoder.fitTransform(y);
      const yDecoded = encoder.inverseTransform(yEncoded);

      expect(yDecoded.shape).toEqual([4]);
      expect(toNumberArray(yDecoded, "yDecoded")).toEqual([1, 2, 1, 3]);
    });

    it("should inverse transform correctly for string labels", () => {
      const encoder = new LabelEncoder();
      const y = tensor(["cat", "dog", "cat", "bird"]);
      const yEncoded = encoder.fitTransform(y);
      const yDecoded = encoder.inverseTransform(yEncoded);

      expect(yDecoded.dtype).toBe("string");
      expect(getStringData(yDecoded)).toEqual(["cat", "dog", "cat", "bird"]);
    });
  });

  describe("Edge Cases", () => {
    it("should throw error when fitting empty array", () => {
      const encoder = new LabelEncoder();
      const y = tensor([]);
      expect(() => encoder.fit(y)).toThrow("Cannot fit LabelEncoder on empty array");
    });

    it("should reject non-1D inputs", () => {
      const encoder = new LabelEncoder();
      const y = tensor([[1], [2]]);
      expect(() => encoder.fit(y)).toThrow(/1D/);
    });

    it("should handle single unique value", () => {
      const encoder = new LabelEncoder();
      const y = tensor([5, 5, 5, 5]);
      encoder.fit(y);
      const yEncoded = encoder.transform(y);

      expect(Array.from(getFloat64Data(yEncoded))).toEqual([0, 0, 0, 0]);
    });

    it("should handle large number of unique classes", () => {
      const encoder = new LabelEncoder();
      const labels = Array.from({ length: 1000 }, (_, i) => i);
      const y = tensor(labels);
      encoder.fit(y);
      const yEncoded = encoder.transform(y);

      expect(yEncoded.shape).toEqual([1000]);
      expect(Math.max(...Array.from(getFloat64Data(yEncoded)))).toBe(999);
    });

    it("should maintain consistent ordering across multiple fits", () => {
      const encoder1 = new LabelEncoder();
      const encoder2 = new LabelEncoder();
      const y = tensor(["zebra", "apple", "mango", "banana"]);

      encoder1.fit(y);
      encoder2.fit(y);

      const y1 = encoder1.transform(y);
      const y2 = encoder2.transform(y);

      expect(Array.from(getFloat64Data(y1))).toEqual(Array.from(getFloat64Data(y2)));
    });

    it("should handle mixed case strings", () => {
      const encoder = new LabelEncoder();
      const y = tensor(["Apple", "apple", "APPLE"]);
      encoder.fit(y);
      const yEncoded = encoder.transform(y);

      expect(yEncoded.shape).toEqual([3]);
    });
  });

  describe("Error Handling", () => {
    it("should throw error when transforming before fitting", () => {
      const encoder = new LabelEncoder();
      const y = tensor([1, 2, 3]);
      expect(() => encoder.transform(y)).toThrow("must be fitted before transform");
    });

    it("should throw error when inverse transforming before fitting", () => {
      const encoder = new LabelEncoder();
      const y = tensor([0, 1, 2]);
      expect(() => encoder.inverseTransform(y)).toThrow("must be fitted before inverse_transform");
    });

    it("should throw error for unknown labels during transform", () => {
      const encoder = new LabelEncoder();
      const yTrain = tensor([1, 2, 3]);
      encoder.fit(yTrain);

      const yTest = tensor([1, 2, 4]);
      expect(() => encoder.transform(yTest)).toThrow("Unknown label");
    });

    it("should throw error for invalid indices during inverse transform", () => {
      const encoder = new LabelEncoder();
      const y = tensor([1, 2, 3]);
      encoder.fit(y);

      const yInvalid = tensor([0, 1, 5]);
      expect(() => encoder.inverseTransform(yInvalid)).toThrow("Invalid label index");
    });

    it("should throw error for negative indices during inverse transform", () => {
      const encoder = new LabelEncoder();
      const y = tensor([1, 2, 3]);
      encoder.fit(y);

      const yInvalid = tensor([-1, 0, 1]);
      expect(() => encoder.inverseTransform(yInvalid)).toThrow("Invalid label index");
    });
  });

  describe("Empty Data Handling", () => {
    it("should handle empty transform after fitting", () => {
      const encoder = new LabelEncoder();
      const yTrain = tensor([1, 2, 3]);
      encoder.fit(yTrain);

      const yEmpty = tensor([]);
      const result = encoder.transform(yEmpty);
      expect(result.size).toBe(0);
    });

    it("should handle empty inverse transform after fitting", () => {
      const encoder = new LabelEncoder();
      const y = tensor([1, 2, 3]);
      encoder.fit(y);

      const yEmpty = tensor([]);
      const result = encoder.inverseTransform(yEmpty);
      expect(result.size).toBe(0);
    });
  });
});

describe("OneHotEncoder - Comprehensive Tests", () => {
  describe("Basic Functionality - Dense", () => {
    it("should encode 2D categorical data correctly", () => {
      const X = tensor([
        ["red", "S"],
        ["blue", "M"],
        ["red", "M"],
      ]);
      const encoder = new OneHotEncoder({ sparse: false });
      const encoded = encoder.fitTransform(X);

      expect(encoded.shape[0]).toBe(3);
      expect(encoded.shape[1]).toBeGreaterThan(0);
    });

    it("should handle numeric categories", () => {
      const X = tensor([
        [1, 10],
        [2, 20],
        [1, 20],
      ]);
      const encoder = new OneHotEncoder({ sparse: false });
      const encoded = encoder.fitTransform(X);

      expect(encoded.shape[0]).toBe(3);
    });

    it("should inverse transform correctly", () => {
      const X = tensor([
        ["red", "S"],
        ["blue", "M"],
        ["red", "L"],
      ]);
      const encoder = new OneHotEncoder({ sparse: false });
      const encoded = encoder.fitTransform(X);
      const decoded = encoder.inverseTransform(encoded);

      expect(decoded.shape).toEqual([3, 2]);
      expect(decoded.dtype).toBe("string");
    });

    it("supports handleUnknown='ignore' in dense mode", () => {
      const X = tensor([["a"], ["b"]]);
      const encoder = new OneHotEncoder({
        sparse: false,
        handleUnknown: "ignore",
      });
      encoder.fit(X);
      const out = encoder.transform(tensor([["c"]]));
      if (out instanceof CSRMatrix) {
        throw new DeepboxError("Expected dense output");
      }
      expect(toNumberMatrix(out, "out")).toEqual([[0, 0]]);
    });

    it("supports drop='first' and inverse transform", () => {
      const X = tensor([["a"], ["b"]]);
      const encoder = new OneHotEncoder({ drop: "first", sparse: false });
      const encoded = encoder.fitTransform(X);
      expect(encoded.shape[1]).toBe(1);
      const decoded = encoder.inverseTransform(encoded);
      expect(decoded.toArray()).toEqual(X.toArray());
    });

    it("supports drop='if_binary' for binary features", () => {
      const X = tensor([["a"], ["b"], ["a"]]);
      const encoder = new OneHotEncoder({ drop: "if_binary", sparse: false });
      const encoded = encoder.fitTransform(X);
      expect(encoded.shape[1]).toBe(1);
    });

    it("supports explicit categories", () => {
      const X = tensor([["b"], ["a"]]);
      const encoder = new OneHotEncoder({
        categories: [["b", "a"]],
        sparse: false,
      });
      const out = encoder.fitTransform(X);
      if (out instanceof CSRMatrix) {
        throw new DeepboxError("Expected dense output");
      }
      const encoded = toNumberMatrix(out, "encoded");
      expect(encoded[0]).toEqual([1, 0]);
      expect(encoded[1]).toEqual([0, 1]);
    });
  });

  describe("Basic Functionality - Sparse", () => {
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
      expect(csr.indices).toBeDefined();
      expect(csr.data).toBeDefined();
    });

    it("accepts sparseOutput alias", () => {
      const X = tensor([
        ["red", "S"],
        ["blue", "M"],
      ]);
      const encoder = new OneHotEncoder({ sparseOutput: true });
      const out = encoder.fitTransform(X);
      expect(out).toBeInstanceOf(CSRMatrix);
    });

    it("should inverse transform sparse correctly", () => {
      const X = tensor([
        ["red", "S"],
        ["blue", "M"],
        ["red", "M"],
      ]);
      const encoder = new OneHotEncoder({ sparse: true });
      const encoded = encoder.fitTransform(X);
      const decoded = encoder.inverseTransform(encoded);

      expect(decoded.shape).toEqual([3, 2]);
    });
  });

  describe("Edge Cases", () => {
    it("should throw error when fitting empty array", () => {
      const encoder = new OneHotEncoder();
      const X = tensor([[]]);
      expect(() => encoder.fit(X)).toThrow("Cannot fit OneHotEncoder on empty array");
    });

    it("should handle single feature", () => {
      const X = tensor([["a"], ["b"], ["a"]]);
      const encoder = new OneHotEncoder({ sparse: false });
      const encoded = encoder.fitTransform(X);

      expect(encoded.shape[0]).toBe(3);
      expect(encoded.shape[1]).toBe(2);
    });

    it("should handle single sample", () => {
      const X = tensor([["a", "b"]]);
      const encoder = new OneHotEncoder({ sparse: false });
      const encoded = encoder.fitTransform(X);

      expect(encoded.shape[0]).toBe(1);
    });

    it("should handle high cardinality features", () => {
      const categories = Array.from({ length: 100 }, (_, i) => [`cat${i}`, `val${i}`]);
      const X = tensor(categories);
      const encoder = new OneHotEncoder({ sparse: true });
      const encoded = encoder.fitTransform(X);

      expect(encoded).toBeDefined();
    });

    it("should handle strided (transposed) tensors", () => {
      const X = tensor([
        ["red", "S", "x"],
        ["blue", "M", "y"],
      ]);
      const Xt = transpose(X);
      const encoder = new OneHotEncoder({ sparse: false });
      const encoded = encoder.fitTransform(Xt);
      const decoded = encoder.inverseTransform(encoded);
      expect(decoded.toArray()).toEqual(Xt.toArray());
    });

    it("inverseTransform maps unknowns to dropped category when handleUnknown='ignore'", () => {
      const X = tensor([["a"], ["b"]]);
      const encoder = new OneHotEncoder({
        handleUnknown: "ignore",
        drop: "first",
      });
      encoder.fit(X);
      const encoded = encoder.transform(tensor([["c"]]));
      const decoded = encoder.inverseTransform(encoded);
      expect(decoded.toArray()).toEqual([["a"]]);
    });
  });

  describe("Error Handling", () => {
    it("should throw error when transforming before fitting", () => {
      const encoder = new OneHotEncoder();
      const X = tensor([["a", "b"]]);
      expect(() => encoder.transform(X)).toThrow("must be fitted before transform");
    });

    it("should throw error for unknown categories", () => {
      const XTrain = tensor([
        ["a", "x"],
        ["b", "y"],
      ]);
      const encoder = new OneHotEncoder();
      encoder.fit(XTrain);

      const XTest = tensor([["c", "x"]]);
      expect(() => encoder.transform(XTest)).toThrow("Unknown category");
    });

    it("should throw error when inverse transforming before fitting", () => {
      const encoder = new OneHotEncoder();
      const X = tensor([[1, 0, 1, 0]]);
      expect(() => encoder.inverseTransform(X)).toThrow("must be fitted before inverse_transform");
    });

    it("should throw error for mismatched feature count", () => {
      const encoder = new OneHotEncoder();
      const XTrain = tensor([
        ["a", "b"],
        ["c", "d"],
      ]);
      encoder.fit(XTrain);
      const XBad = tensor([["a"]]);
      expect(() => encoder.transform(XBad)).toThrow(/feature count/i);
    });

    it("should throw when explicit categories miss training values", () => {
      const XTrain = tensor([["a"], ["c"]]);
      const encoder = new OneHotEncoder({
        categories: [["a", "b"]],
      });
      expect(() => encoder.fit(XTrain)).toThrow(/Unknown category/);
    });

    it("should reject conflicting sparse options", () => {
      expect(() => new OneHotEncoder({ sparse: true, sparseOutput: false })).toThrow(/sparse/i);
    });
  });

  describe("Empty Data Handling", () => {
    it("should handle empty transform after fitting - dense", () => {
      const XTrain = tensor([["a", "b"]]);
      const encoder = new OneHotEncoder({ sparse: false });
      encoder.fit(XTrain);

      const XEmpty = zeros([0, 2]);
      const result = encoder.transform(XEmpty);
      expect(result.shape[0]).toBe(0);
      expect(result.shape[1]).toBe(2);
    });

    it("should handle empty transform after fitting - sparse", () => {
      const XTrain = tensor([["a", "b"]]);
      const encoder = new OneHotEncoder({ sparse: true });
      encoder.fit(XTrain);

      const XEmpty = zeros([0, 2]);
      const result = encoder.transform(XEmpty);
      expect(result).toBeInstanceOf(CSRMatrix);
      if (!(result instanceof CSRMatrix)) {
        throw new DeepboxError("Expected CSRMatrix output");
      }
      expect(result.shape[0]).toBe(0);
      expect(result.shape[1]).toBe(2);
    });
  });
});

describe("OrdinalEncoder - Comprehensive Tests", () => {
  describe("Basic Functionality", () => {
    it("should encode ordinal data correctly", () => {
      const X = tensor([
        [1, 2],
        [2, 3],
        [1, 3],
      ]);
      const encoder = new OrdinalEncoder();
      const encoded = encoder.fitTransform(X);

      expect(encoded.shape).toEqual([3, 2]);
    });

    it("should inverse transform correctly", () => {
      const X = tensor([
        [1, 2],
        [2, 3],
        [1, 3],
      ]);
      const encoder = new OrdinalEncoder();
      const encoded = encoder.fitTransform(X);
      const decoded = encoder.inverseTransform(encoded);

      expect(decoded.shape).toEqual([3, 2]);
    });
  });

  describe("Edge Cases", () => {
    it("should handle single feature", () => {
      const X = tensor([[1], [2], [1]]);
      const encoder = new OrdinalEncoder();
      const encoded = encoder.fitTransform(X);

      expect(encoded.shape).toEqual([3, 1]);
    });

    it("should maintain order of categories", () => {
      const X = tensor([
        [3, 1],
        [1, 2],
        [2, 3],
      ]);
      const encoder = new OrdinalEncoder();
      encoder.fit(X);
      const encoded = encoder.transform(X);

      expect(encoded.shape).toEqual([3, 2]);
    });

    it("supports explicit categories ordering", () => {
      const X = tensor([["b"], ["a"]]);
      const encoder = new OrdinalEncoder({ categories: [["b", "a"]] });
      const encoded = toNumberMatrix(encoder.fitTransform(X), "encoded");
      expect(encoded).toEqual([[0], [1]]);
    });

    it("supports handleUnknown='useEncodedValue'", () => {
      const X = tensor([["a"], ["b"]]);
      const encoder = new OrdinalEncoder({
        handleUnknown: "useEncodedValue",
        unknownValue: -1,
      });
      encoder.fit(X);
      const out = encoder.transform(tensor([["c"]]));
      expect(toNumberMatrix(out, "out")).toEqual([[-1]]);
      expect(() => encoder.inverseTransform(out)).toThrow(/unknown|invalid/i);
    });
  });

  describe("Error Handling", () => {
    it("should throw error when transforming before fitting", () => {
      const encoder = new OrdinalEncoder();
      const X = tensor([[1, 2]]);
      expect(() => encoder.transform(X)).toThrow("must be fitted before transform");
    });

    it("should throw error for unknown categories", () => {
      const XTrain = tensor([
        [1, 2],
        [2, 3],
      ]);
      const encoder = new OrdinalEncoder();
      encoder.fit(XTrain);

      const XTest = tensor([[5, 2]]);
      expect(() => encoder.transform(XTest)).toThrow("Unknown category");
    });

    it("should throw error when inverse transforming before fitting", () => {
      const encoder = new OrdinalEncoder();
      const X = tensor([[0, 1]]);
      expect(() => encoder.inverseTransform(X)).toThrow("must be fitted before inverse_transform");
    });

    it("should reject non-2D inputs", () => {
      const encoder = new OrdinalEncoder();
      const X = tensor([1, 2, 3]);
      expect(() => encoder.fit(X)).toThrow(/2D/);
    });

    it("should throw error for invalid encoded values", () => {
      const X = tensor([
        [1, 2],
        [2, 3],
      ]);
      const encoder = new OrdinalEncoder();
      encoder.fit(X);

      const XInvalid = tensor([[0, 5]]);
      expect(() => encoder.inverseTransform(XInvalid)).toThrow("Invalid encoded value");
    });

    it("should throw error for mismatched feature count", () => {
      const X = tensor([
        [1, 2],
        [2, 3],
      ]);
      const encoder = new OrdinalEncoder();
      encoder.fit(X);
      const XBad = tensor([[1]]);
      expect(() => encoder.transform(XBad)).toThrow(/feature count/i);
    });
  });
});

describe("LabelBinarizer - Comprehensive Tests", () => {
  describe("Basic Functionality", () => {
    it("should binarize labels correctly", () => {
      const y = tensor([0, 1, 2, 0, 1]);
      const binarizer = new LabelBinarizer();
      const yBin = binarizer.fitTransform(y);

      expect(yBin.shape[0]).toBe(5);
      expect(yBin.shape[1]).toBe(3);
    });

    it("should inverse transform correctly", () => {
      const y = tensor([0, 1, 2, 0, 1]);
      const binarizer = new LabelBinarizer();
      const yBin = binarizer.fitTransform(y);
      const yDecoded = binarizer.inverseTransform(yBin);

      expect(yDecoded.shape).toEqual([5]);
    });

    it("supports custom posLabel/negLabel", () => {
      const y = tensor([0, 1, 0]);
      const binarizer = new LabelBinarizer({ posLabel: 2, negLabel: -1 });
      const out = binarizer.fitTransform(y);
      if (out instanceof CSRMatrix) {
        throw new DeepboxError("Expected dense output");
      }
      const arr = toNumberMatrix(out, "yBin");
      expect(arr).toEqual([
        [2, -1],
        [-1, 2],
        [2, -1],
      ]);
    });
  });

  describe("Edge Cases", () => {
    it("should handle binary classification", () => {
      const y = tensor([0, 1, 0, 1]);
      const binarizer = new LabelBinarizer();
      const yBin = binarizer.fitTransform(y);

      expect(yBin.shape[0]).toBe(4);
      expect(yBin.shape[1]).toBe(2);
    });

    it("should handle multi-class classification", () => {
      const y = tensor([0, 1, 2, 3, 4]);
      const binarizer = new LabelBinarizer();
      const yBin = binarizer.fitTransform(y);

      expect(yBin.shape[1]).toBe(5);
    });

    it("supports sparse output and inverseTransform", () => {
      const y = tensor([0, 1, 0, 1]);
      const binarizer = new LabelBinarizer({ sparse: true });
      const out = binarizer.fitTransform(y);
      expect(out).toBeInstanceOf(CSRMatrix);
      if (!(out instanceof CSRMatrix)) {
        throw new DeepboxError("Expected CSRMatrix output");
      }
      expect(out.data.length).toBe(4);
      const decoded = binarizer.inverseTransform(out);
      expect(toNumberArray(decoded, "decoded")).toEqual([0, 1, 0, 1]);
    });

    it("accepts sparseOutput alias", () => {
      const y = tensor([0, 1]);
      const binarizer = new LabelBinarizer({ sparseOutput: true });
      const out = binarizer.fitTransform(y);
      expect(out).toBeInstanceOf(CSRMatrix);
    });
  });

  describe("Error Handling", () => {
    it("should throw error when transforming before fitting", () => {
      const binarizer = new LabelBinarizer();
      const y = tensor([0, 1, 2]);
      expect(() => binarizer.transform(y)).toThrow("must be fitted before transform");
    });

    it("should throw error when inverse transforming before fitting", () => {
      const binarizer = new LabelBinarizer();
      const Y = tensor([
        [1, 0],
        [0, 1],
      ]);
      expect(() => binarizer.inverseTransform(Y)).toThrow(
        "must be fitted before inverse_transform"
      );
    });

    it("should reject non-1D inputs", () => {
      const binarizer = new LabelBinarizer();
      const y = tensor([[0], [1]]);
      expect(() => binarizer.fit(y)).toThrow(/1D/);
    });

    it("should throw error when columns do not match classes", () => {
      const y = tensor([0, 1, 2]);
      const binarizer = new LabelBinarizer();
      binarizer.fit(y);
      const YBad = tensor([[1, 0]]);
      expect(() => binarizer.inverseTransform(YBad)).toThrow(/column count/i);
    });

    it("should reject sparse output when negLabel != 0", () => {
      expect(() => new LabelBinarizer({ sparse: true, negLabel: -1 })).toThrow(/negLabel/i);
    });

    it("should reject conflicting sparse options", () => {
      expect(() => new LabelBinarizer({ sparse: true, sparseOutput: false })).toThrow(/sparse/i);
    });
  });
});

describe("MultiLabelBinarizer - Comprehensive Tests", () => {
  describe("Basic Functionality", () => {
    it("should binarize multi-label data correctly", () => {
      const y = [["sci-fi", "action"], ["comedy"], ["action", "drama"]];
      const binarizer = new MultiLabelBinarizer();
      const yBin = binarizer.fitTransform(y);

      expect(yBin.shape[0]).toBe(3);
      expect(yBin.shape[1]).toBeGreaterThan(0);
    });

    it("should inverse transform correctly", () => {
      const y = [["a", "b"], ["b", "c"], ["a"]];
      const binarizer = new MultiLabelBinarizer();
      const yBin = binarizer.fitTransform(y);
      const yDecoded = binarizer.inverseTransform(yBin);

      expect(yDecoded.length).toBe(3);
      expect(Array.isArray(yDecoded[0])).toBe(true);
    });

    it("supports explicit classes ordering", () => {
      const y = [["a"], ["b"]];
      const binarizer = new MultiLabelBinarizer({ classes: ["b", "a"] });
      const out = binarizer.fitTransform(y);
      if (out instanceof CSRMatrix) {
        throw new DeepboxError("Expected dense output");
      }
      const arr = toNumberMatrix(out, "out");
      expect(arr).toEqual([
        [0, 1],
        [1, 0],
      ]);
    });
  });

  describe("Edge Cases", () => {
    it("should handle empty label sets", () => {
      const y = [["a"], [], ["b"]];
      const binarizer = new MultiLabelBinarizer();
      const yBin = binarizer.fitTransform(y);

      expect(yBin.shape[0]).toBe(3);
    });

    it("should reject NaN labels", () => {
      const binarizer = new MultiLabelBinarizer();
      expect(() => binarizer.fit([["a"], [Number.NaN]])).toThrow(/finite/i);
    });

    it("should handle single label per sample", () => {
      const y = [["a"], ["b"], ["c"]];
      const binarizer = new MultiLabelBinarizer();
      const yBin = binarizer.fitTransform(y);

      expect(yBin.shape[0]).toBe(3);
      expect(yBin.shape[1]).toBe(3);
    });

    it("should handle numeric labels", () => {
      const y = [[1, 2], [2, 3], [1]];
      const binarizer = new MultiLabelBinarizer();
      const yBin = binarizer.fitTransform(y);

      expect(yBin.shape[0]).toBe(3);
    });

    it("supports sparse output and inverseTransform", () => {
      const y = [["a", "b"], ["b"], []];
      const binarizer = new MultiLabelBinarizer({ sparse: true });
      const out = binarizer.fitTransform(y);
      expect(out).toBeInstanceOf(CSRMatrix);
      if (!(out instanceof CSRMatrix)) {
        throw new DeepboxError("Expected CSRMatrix output");
      }
      const decoded = binarizer.inverseTransform(out);
      expect(decoded[0]?.slice().sort()).toEqual(["a", "b"]);
      expect(decoded[1]?.slice().sort()).toEqual(["b"]);
      expect(decoded[2]?.length).toBe(0);
    });

    it("accepts sparseOutput alias", () => {
      const y = [["a"], ["b"]];
      const binarizer = new MultiLabelBinarizer({ sparseOutput: true });
      const out = binarizer.fitTransform(y);
      expect(out).toBeInstanceOf(CSRMatrix);
    });

    it("rejects labels not present in explicit classes", () => {
      const binarizer = new MultiLabelBinarizer({ classes: ["a"] });
      expect(() => binarizer.fit([["a", "b"]])).toThrow(/Unknown label/i);
    });
  });

  describe("Error Handling", () => {
    it("should throw error when transforming before fitting", () => {
      const binarizer = new MultiLabelBinarizer();
      const y = [["a"], ["b"]];
      expect(() => binarizer.transform(y)).toThrow("must be fitted before transform");
    });

    it("should throw error when inverse transforming before fitting", () => {
      const binarizer = new MultiLabelBinarizer();
      const Y = tensor([
        [1, 0],
        [0, 1],
      ]);
      expect(() => binarizer.inverseTransform(Y)).toThrow(
        "must be fitted before inverse_transform"
      );
    });

    it("should throw error when columns do not match classes", () => {
      const y = [["a"], ["b"]];
      const binarizer = new MultiLabelBinarizer();
      binarizer.fit(y);
      const YBad = tensor([[1, 0, 0]]);
      expect(() => binarizer.inverseTransform(YBad)).toThrow(/column count/i);
    });

    it("should reject conflicting sparse options", () => {
      expect(() => new MultiLabelBinarizer({ sparse: true, sparseOutput: false })).toThrow(
        /sparse/i
      );
    });

    it("should throw error for unknown labels during transform", () => {
      const binarizer = new MultiLabelBinarizer();
      binarizer.fit([["a"], ["b"]]);
      expect(() => binarizer.transform([["a", "c"]])).toThrow(/Unknown label/i);
    });
  });
});
