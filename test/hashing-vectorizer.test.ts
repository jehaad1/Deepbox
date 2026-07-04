import { describe, expect, it } from "vitest";
import { HashingVectorizer } from "../src/preprocess";

describe("HashingVectorizer", () => {
  describe("constructor", () => {
    it("creates with default options", () => {
      const hv = new HashingVectorizer();
      expect(hv).toBeDefined();
      const params = hv.getParams();
      expect(params.nFeatures).toBe(1 << 20);
      expect(params.lowercase).toBe(true);
      expect(params.binary).toBe(false);
      expect(params.alternateSign).toBe(true);
      expect(params.norm).toBe("l2");
    });

    it("creates with custom nFeatures", () => {
      const hv = new HashingVectorizer({ nFeatures: 256 });
      expect(hv.getParams().nFeatures).toBe(256);
    });

    it("validates nFeatures is positive integer", () => {
      expect(() => new HashingVectorizer({ nFeatures: 0 })).toThrow();
      expect(() => new HashingVectorizer({ nFeatures: -1 })).toThrow();
      expect(() => new HashingVectorizer({ nFeatures: 1.5 })).toThrow();
    });

    it("validates ngramRange", () => {
      expect(() => new HashingVectorizer({ ngramRange: [0, 1] })).toThrow();
      expect(() => new HashingVectorizer({ ngramRange: [2, 1] })).toThrow();
    });
  });

  describe("transformText", () => {
    it("produces correct output shape", () => {
      const hv = new HashingVectorizer({ nFeatures: 64 });
      const X = hv.transformText(["hello world", "foo bar baz"]);
      expect(X.shape).toEqual([2, 64]);
    });

    it("is deterministic", () => {
      const hv = new HashingVectorizer({ nFeatures: 128 });
      const X1 = hv.transformText(["the quick brown fox"]);
      const X2 = hv.transformText(["the quick brown fox"]);
      expect(X1.toArray()).toEqual(X2.toArray());
    });

    it("does not require fitting", () => {
      const hv = new HashingVectorizer({ nFeatures: 32 });
      expect(() => hv.transformText(["test"])).not.toThrow();
    });

    it("handles empty document list", () => {
      const hv = new HashingVectorizer({ nFeatures: 16 });
      const X = hv.transformText([]);
      expect(X.shape).toEqual([0, 16]);
    });

    it("handles empty string document", () => {
      const hv = new HashingVectorizer({ nFeatures: 16 });
      const X = hv.transformText([""]);
      expect(X.shape).toEqual([1, 16]);
      // All zeros since no tokens
      const data = X.toArray() as number[][];
      const row = data[0] as number[];
      const allZero = row.every((v) => v === 0);
      expect(allZero).toBe(true);
    });

    it("different documents produce different feature vectors", () => {
      const hv = new HashingVectorizer({ nFeatures: 128, norm: undefined });
      const X = hv.transformText(["hello world", "completely different text"]);
      const data = X.toArray() as number[][];
      const row0 = data[0] as number[];
      const row1 = data[1] as number[];
      const same = row0.every((v, i) => v === row1[i]);
      expect(same).toBe(false);
    });
  });

  describe("binary mode", () => {
    it("produces only 0s and 1s", () => {
      const hv = new HashingVectorizer({
        nFeatures: 64,
        binary: true,
        norm: undefined,
      });
      const X = hv.transformText(["hello hello hello world world"]);
      const data = X.toArray() as number[][];
      const row = data[0] as number[];
      for (const v of row) {
        expect(v === 0 || v === 1).toBe(true);
      }
    });
  });

  describe("alternateSign", () => {
    it("with alternateSign=false, all values are non-negative", () => {
      const hv = new HashingVectorizer({
        nFeatures: 64,
        alternateSign: false,
        norm: undefined,
      });
      const X = hv.transformText(["hello world foo bar baz qux"]);
      const data = X.toArray() as number[][];
      const row = data[0] as number[];
      for (const v of row) {
        expect(v).toBeGreaterThanOrEqual(0);
      }
    });
  });

  describe("normalization", () => {
    it("l2 normalization produces unit-length rows", () => {
      const hv = new HashingVectorizer({ nFeatures: 64, norm: "l2" });
      const X = hv.transformText(["hello world foo"]);
      const data = X.toArray() as number[][];
      const row = data[0] as number[];
      let sumSq = 0;
      for (const v of row) {
        sumSq += v * v;
      }
      expect(Math.abs(sumSq - 1)).toBeLessThan(1e-6);
    });

    it("l1 normalization produces rows summing to 1 in absolute value", () => {
      const hv = new HashingVectorizer({ nFeatures: 64, norm: "l1" });
      const X = hv.transformText(["hello world foo"]);
      const data = X.toArray() as number[][];
      const row = data[0] as number[];
      let sumAbs = 0;
      for (const v of row) {
        sumAbs += Math.abs(v);
      }
      expect(Math.abs(sumAbs - 1)).toBeLessThan(1e-6);
    });

    it("no normalization preserves raw counts", () => {
      const hv = new HashingVectorizer({
        nFeatures: 64,
        norm: undefined,
        alternateSign: false,
      });
      const X = hv.transformText(["hello hello hello"]);
      const data = X.toArray() as number[][];
      const row = data[0] as number[];
      const nonZero = row.filter((v) => v !== 0);
      // "hello" repeated 3 times should hash to one bucket with count 3
      expect(nonZero.length).toBe(1);
      expect(nonZero[0]).toBe(3);
    });
  });

  describe("lowercase", () => {
    it("case insensitive by default", () => {
      const hv = new HashingVectorizer({ nFeatures: 64 });
      const X1 = hv.transformText(["Hello World"]);
      const X2 = hv.transformText(["hello world"]);
      expect(X1.toArray()).toEqual(X2.toArray());
    });

    it("case sensitive when lowercase=false", () => {
      const hv = new HashingVectorizer({ nFeatures: 64, lowercase: false });
      const X1 = hv.transformText(["Hello World"]);
      const X2 = hv.transformText(["hello world"]);
      expect(X1.toArray()).not.toEqual(X2.toArray());
    });
  });

  describe("stopWords", () => {
    it("removes stop words", () => {
      const hv = new HashingVectorizer({
        nFeatures: 64,
        stopWords: ["the", "is"],
        norm: undefined,
        alternateSign: false,
      });
      const X1 = hv.transformText(["the cat is big"]);
      const X2 = hv.transformText(["cat big"]);
      expect(X1.toArray()).toEqual(X2.toArray());
    });
  });

  describe("ngramRange", () => {
    it("supports bigrams", () => {
      const hv = new HashingVectorizer({
        nFeatures: 128,
        ngramRange: [2, 2],
        norm: undefined,
        alternateSign: false,
      });
      const X = hv.transformText(["hello world foo"]);
      const data = X.toArray() as number[][];
      const row = data[0] as number[];
      const nonZero = row.filter((v) => v !== 0);
      // "hello world" and "world foo" → 2 bigrams
      expect(nonZero.length).toBeGreaterThanOrEqual(1);
      expect(nonZero.length).toBeLessThanOrEqual(2);
    });
  });

  describe("fitTransformText", () => {
    it("is equivalent to transformText", () => {
      const hv = new HashingVectorizer({ nFeatures: 64 });
      const docs = ["hello world", "foo bar"];
      const X1 = hv.transformText(docs);
      const X2 = hv.fitTransformText(docs);
      expect(X1.toArray()).toEqual(X2.toArray());
    });
  });
});
