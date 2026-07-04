import { describe, expect, it } from "vitest";
import { CountVectorizer, TfidfVectorizer } from "../src/preprocess";

describe("CountVectorizer", () => {
  it("basic fit and transform", () => {
    const cv = new CountVectorizer();
    const docs = ["hello world", "hello deepbox world"];
    cv.fitText(docs);
    const X = cv.transformText(docs);
    expect(X.shape[0]).toBe(2);
    expect(X.shape[1]).toBe(3); // deepbox, hello, world (alphabetical)
  });

  it("fitTransformText convenience", () => {
    const cv = new CountVectorizer();
    const X = cv.fitTransformText(["foo bar", "bar baz"]);
    expect(X.shape[0]).toBe(2);
    expect(X.shape[1]).toBe(3); // bar, baz, foo
  });

  it("vocabulary is sorted alphabetically", () => {
    const cv = new CountVectorizer();
    cv.fitText(["banana apple cherry"]);
    const names = cv.getFeatureNames();
    expect(names).toEqual(["apple", "banana", "cherry"]);
  });

  it("counts word occurrences correctly", () => {
    const cv = new CountVectorizer();
    const X = cv.fitTransformText(["the cat sat on the mat"]);
    const vocab = cv.vocabulary;
    const theIdx = vocab.get("the")!;
    // "the" appears twice
    expect(Number(X.data[X.offset + theIdx])).toBe(2);
  });

  it("handles multiple documents", () => {
    const cv = new CountVectorizer();
    const docs = ["hello world", "goodbye world", "hello goodbye"];
    const X = cv.fitTransformText(docs);
    expect(X.shape[0]).toBe(3);
    const names = cv.getFeatureNames();
    expect(names).toContain("hello");
    expect(names).toContain("world");
    expect(names).toContain("goodbye");
  });

  it("lowercase=true by default", () => {
    const cv = new CountVectorizer();
    cv.fitText(["Hello WORLD"]);
    const names = cv.getFeatureNames();
    expect(names).toContain("hello");
    expect(names).toContain("world");
    expect(names).not.toContain("Hello");
  });

  it("lowercase=false preserves case", () => {
    const cv = new CountVectorizer({ lowercase: false });
    cv.fitText(["Hello WORLD"]);
    const names = cv.getFeatureNames();
    expect(names).toContain("Hello");
    expect(names).toContain("WORLD");
  });

  it("binary mode", () => {
    const cv = new CountVectorizer({ binary: true });
    const X = cv.fitTransformText(["the cat the cat the cat"]);
    const theIdx = cv.vocabulary.get("the")!;
    const catIdx = cv.vocabulary.get("cat")!;
    expect(Number(X.data[X.offset + theIdx])).toBe(1);
    expect(Number(X.data[X.offset + catIdx])).toBe(1);
  });

  it("maxFeatures limits vocabulary", () => {
    const cv = new CountVectorizer({ maxFeatures: 2 });
    cv.fitText(["aaa bbb ccc aaa bbb aaa"]);
    expect(cv.getFeatureNames().length).toBe(2);
  });

  it("stopWords are removed", () => {
    const cv = new CountVectorizer({ stopWords: ["the", "is", "and"] });
    cv.fitText(["the cat is big and fluffy"]);
    const names = cv.getFeatureNames();
    expect(names).not.toContain("the");
    expect(names).not.toContain("is");
    expect(names).not.toContain("and");
    expect(names).toContain("cat");
    expect(names).toContain("big");
    expect(names).toContain("fluffy");
  });

  it("minDf filters rare terms (absolute)", () => {
    const cv = new CountVectorizer({ minDf: 2 });
    cv.fitText(["hello world", "hello deepbox", "goodbye world"]);
    const names = cv.getFeatureNames();
    // "hello" appears in 2 docs, "world" in 2, "goodbye" in 1, "deepbox" in 1
    expect(names).toContain("hello");
    expect(names).toContain("world");
    expect(names).not.toContain("deepbox");
    expect(names).not.toContain("goodbye");
  });

  it("maxDf filters common terms (proportion)", () => {
    const cv = new CountVectorizer({ maxDf: 0.5 });
    cv.fitText(["hello world", "hello deepbox", "hello goodbye"]);
    const names = cv.getFeatureNames();
    // "hello" appears in all 3 docs (100%), should be filtered
    expect(names).not.toContain("hello");
    expect(names).toContain("world");
  });

  it("bigrams with ngramRange", () => {
    const cv = new CountVectorizer({ ngramRange: [2, 2] });
    cv.fitText(["hello big world"]);
    const names = cv.getFeatureNames();
    expect(names).toContain("hello big");
    expect(names).toContain("big world");
    expect(names.length).toBe(2);
  });

  it("unigrams and bigrams", () => {
    const cv = new CountVectorizer({ ngramRange: [1, 2] });
    cv.fitText(["aa bb cc"]);
    const names = cv.getFeatureNames();
    expect(names).toContain("aa");
    expect(names).toContain("bb");
    expect(names).toContain("cc");
    expect(names).toContain("aa bb");
    expect(names).toContain("bb cc");
    expect(names.length).toBe(5);
  });

  it("transform ignores unknown terms", () => {
    const cv = new CountVectorizer();
    cv.fitText(["hello world"]);
    const X = cv.transformText(["hello unknown"]);
    const nF = cv.getFeatureNames().length;
    expect(X.shape).toEqual([1, nF]);
    const helloIdx = cv.vocabulary.get("hello")!;
    expect(Number(X.data[X.offset + helloIdx])).toBe(1);
    // "unknown" is not in vocabulary, total count should be 1
    let total = 0;
    for (let j = 0; j < nF; j++) total += Number(X.data[X.offset + j]);
    expect(total).toBe(1);
  });

  it("throws when not fitted", () => {
    const cv = new CountVectorizer();
    expect(() => cv.transformText(["hello"])).toThrow();
    expect(() => cv.getFeatureNames()).toThrow();
  });

  it("getParams returns options", () => {
    const cv = new CountVectorizer({ maxFeatures: 100, lowercase: false });
    const params = cv.getParams();
    expect(params.maxFeatures).toBe(100);
    expect(params.lowercase).toBe(false);
  });

  it("handles empty documents", () => {
    const cv = new CountVectorizer();
    const X = cv.fitTransformText(["hello world", "", "world"]);
    expect(X.shape[0]).toBe(3);
    // Row 1 (empty doc) should be all zeros
    const nF = cv.getFeatureNames().length;
    let sum = 0;
    for (let j = 0; j < nF; j++) sum += Number(X.data[X.offset + 1 * nF + j]);
    expect(sum).toBe(0);
  });
});

describe("TfidfVectorizer", () => {
  it("basic fit and transform", () => {
    const tfidf = new TfidfVectorizer();
    const docs = ["hello world", "hello deepbox"];
    const X = tfidf.fitTransformText(docs);
    expect(X.shape[0]).toBe(2);
    expect(X.shape[1]).toBe(3); // deepbox, hello, world
  });

  it("fitTransformText convenience", () => {
    const tfidf = new TfidfVectorizer();
    const X = tfidf.fitTransformText(["foo bar", "bar baz"]);
    expect(X.shape[0]).toBe(2);
    expect(X.shape[1]).toBe(3);
  });

  it("idf downweights common terms", () => {
    const tfidf = new TfidfVectorizer({ norm: undefined });
    const docs = ["cat dog", "cat bird", "cat fish"];
    tfidf.fitText(docs);
    const idf = tfidf.idf;
    const vocab = tfidf.vocabulary;
    const catIdf = idf[vocab.get("cat")!]!;
    const dogIdf = idf[vocab.get("dog")!]!;
    // "cat" appears in all 3 docs, "dog" in 1 → dog should have higher IDF
    expect(dogIdf).toBeGreaterThan(catIdf);
  });

  it("l2 normalization produces unit vectors", () => {
    const tfidf = new TfidfVectorizer({ norm: "l2" });
    const docs = ["hello world deepbox", "test data science"];
    const X = tfidf.fitTransformText(docs);
    const nF = tfidf.getFeatureNames().length;

    for (let i = 0; i < 2; i++) {
      let norm = 0;
      for (let j = 0; j < nF; j++) {
        const val = Number(X.data[X.offset + i * nF + j]);
        norm += val * val;
      }
      norm = Math.sqrt(norm);
      expect(norm).toBeCloseTo(1.0, 5);
    }
  });

  it("l1 normalization", () => {
    const tfidf = new TfidfVectorizer({ norm: "l1" });
    const docs = ["hello world"];
    const X = tfidf.fitTransformText(docs);
    const nF = tfidf.getFeatureNames().length;
    let norm = 0;
    for (let j = 0; j < nF; j++) {
      norm += Math.abs(Number(X.data[X.offset + j]));
    }
    expect(norm).toBeCloseTo(1.0, 5);
  });

  it("no normalization", () => {
    const tfidf = new TfidfVectorizer({ norm: undefined });
    const docs = ["hello"];
    const X = tfidf.fitTransformText(docs);
    const val = Number(X.data[X.offset]);
    // With smoothIdf and single doc: idf = log((1+1)/(1+1)) + 1 = 1
    // tf = 1, so tfidf = 1
    expect(val).toBeCloseTo(1.0, 5);
  });

  it("sublinearTf applies log scaling", () => {
    const tfidfLinear = new TfidfVectorizer({ norm: undefined, sublinearTf: false });
    const tfidfSublinear = new TfidfVectorizer({ norm: undefined, sublinearTf: true });
    const docs = ["word word word word"]; // tf=4

    const X1 = tfidfLinear.fitTransformText(docs);
    const X2 = tfidfSublinear.fitTransformText(docs);

    const v1 = Number(X1.data[X1.offset]);
    const v2 = Number(X2.data[X2.offset]);
    // sublinear: 1 + log(4) ≈ 2.386 vs linear: 4
    expect(v2).toBeLessThan(v1);
    expect(v2).toBeGreaterThan(0);
  });

  it("smoothIdf=false changes IDF calculation", () => {
    const tfidfSmooth = new TfidfVectorizer({ norm: undefined, smoothIdf: true });
    const tfidfNoSmooth = new TfidfVectorizer({ norm: undefined, smoothIdf: false });
    const docs = ["hello world", "hello deepbox"];

    tfidfSmooth.fitText(docs);
    tfidfNoSmooth.fitText(docs);

    // IDF values should differ
    const sIdf = tfidfSmooth.idf;
    const nsIdf = tfidfNoSmooth.idf;
    expect(sIdf.length).toBe(nsIdf.length);
    // At least some values should differ
    let anyDiff = false;
    for (let i = 0; i < sIdf.length; i++) {
      if (Math.abs((sIdf[i] ?? 0) - (nsIdf[i] ?? 0)) > 1e-10) anyDiff = true;
    }
    expect(anyDiff).toBe(true);
  });

  it("vocabulary is accessible after fitting", () => {
    const tfidf = new TfidfVectorizer();
    tfidf.fitText(["aa bb cc"]);
    const names = tfidf.getFeatureNames();
    expect(names).toEqual(["aa", "bb", "cc"]);
  });

  it("passes options to CountVectorizer", () => {
    const tfidf = new TfidfVectorizer({ stopWords: ["the"], maxFeatures: 2 });
    tfidf.fitText(["the big red fox", "the small blue cat"]);
    const names = tfidf.getFeatureNames();
    expect(names).not.toContain("the");
    expect(names.length).toBeLessThanOrEqual(2);
  });

  it("throws when not fitted", () => {
    const tfidf = new TfidfVectorizer();
    expect(() => tfidf.transformText(["hello"])).toThrow();
    expect(() => tfidf.idf).toThrow();
  });

  it("getParams returns options", () => {
    const tfidf = new TfidfVectorizer({ norm: "l1", sublinearTf: true });
    const params = tfidf.getParams();
    expect(params.norm).toBe("l1");
    expect(params.sublinearTf).toBe(true);
    expect(params.smoothIdf).toBe(true);
  });

  it("transform on new documents uses learned vocabulary", () => {
    const tfidf = new TfidfVectorizer();
    tfidf.fitText(["hello world", "hello deepbox"]);
    const X = tfidf.transformText(["deepbox world something_new"]);
    expect(X.shape[0]).toBe(1);
    expect(X.shape[1]).toBe(3);
  });
});
