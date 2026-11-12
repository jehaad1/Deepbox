import { describe, expect, it } from "vitest";
import { crossValScore, LogisticRegression, Pipeline } from "../../src/ml";
import { tensor } from "../../src/ndarray";
import { TfidfVectorizer } from "../../src/preprocess";

const docs = [
  "good movie great acting",
  "great plot good fun",
  "fun and good story",
  "wonderful great film",
  "good good great",
  "lovely fun film",
  "bad movie awful acting",
  "awful plot bad fun",
  "boring and bad story",
  "terrible awful film",
  "bad bad awful",
  "dull boring film",
];
const labels = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0];

describe("ML-16 cross-validation over raw text", () => {
  it("scores a vectorizer pipeline on a 1-D string tensor", () => {
    const pipe = new Pipeline([
      ["tfidf", new TfidfVectorizer()],
      ["clf", new LogisticRegression()],
    ]);
    const scores = crossValScore(
      pipe,
      tensor(docs, { dtype: "string" }),
      tensor(labels, { dtype: "int32" }),
      { cv: 3 }
    );
    const values = Array.from(scores as ArrayLike<number>);
    expect(values).toHaveLength(3);
    for (const v of values) expect(Number.isFinite(v)).toBe(true);
  });
});
