import { describe, expect, it } from "vitest";
import { InvalidParameterError } from "../../src/core";
import { CountVectorizer } from "../../src/preprocess";

// Reference: scikit-learn CountVectorizer(ngram_range=(1, 2)).fit(["a bb cc", "bb cc dd"])
// builds the vocabulary ["bb", "bb cc", "cc", "cc dd", "dd"] (single letters are not tokens).

describe("CountVectorizer ngramRange", () => {
  it("does not loop over n-gram sizes larger than the document", () => {
    const cv = new CountVectorizer({ ngramRange: [1, 2 ** 40] });
    const start = performance.now();
    cv.fitText(["aa bb cc", "bb cc dd"]);
    expect(performance.now() - start).toBeLessThan(500);
    expect(cv.vocabulary.get("aa bb cc")).toBeDefined();
    expect(cv.vocabulary.get("bb cc dd")).toBeDefined();
    expect(cv.vocabulary.size).toBe(4 + 3 + 2);
  });

  it("keeps the bigram vocabulary of scikit-learn", () => {
    const cv = new CountVectorizer({ ngramRange: [1, 2] }).fitText(["a bb cc", "bb cc dd"]);
    expect([...cv.vocabulary.keys()].sort()).toEqual(["bb", "bb cc", "cc", "cc dd", "dd"]);
  });

  for (const range of [[0, 1], [2, 1], [1.5, 2], [1, Infinity], [1], "12"]) {
    it(`rejects ngramRange ${JSON.stringify(range)}`, () => {
      expect(
        () => new CountVectorizer({ ngramRange: range as unknown as [number, number] })
      ).toThrow(InvalidParameterError);
    });
  }
});
