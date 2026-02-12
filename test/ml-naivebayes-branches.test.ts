import { describe, expect, it } from "vitest";
import { GaussianNB } from "../src/ml/naive_bayes";
import { tensor } from "../src/ndarray";
import { toNum2D } from "./_helpers";

describe("GaussianNB branches", () => {
  it("validates fit inputs", () => {
    const nb = new GaussianNB();
    expect(() => nb.fit(tensor([1, 2]), tensor([0, 1]))).toThrow(/2-dimensional/i);
    expect(() => nb.fit(tensor([[1, 2]]), tensor([[0]]))).toThrow(/1-dimensional/i);
    expect(() => nb.fit(tensor([[1, 2]]), tensor([0, 1]))).toThrow(/same number of samples/i);
  });

  it("predicts and scores", () => {
    const X = tensor([
      [1, 2],
      [1, 3],
      [4, 5],
      [4, 6],
    ]);
    const y = tensor([0, 0, 1, 1]);
    const nb = new GaussianNB();
    nb.fit(X, y);

    const preds = nb.predict(
      tensor([
        [1, 2],
        [4, 5],
      ])
    );
    expect(preds.toArray()).toEqual([0, 1]);

    const proba = nb.predictProba(tensor([[1, 2]]));
    const probs = toNum2D(proba.toArray())[0];
    const sum = probs.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 6);

    const score = nb.score(X, y);
    expect(score).toBeGreaterThan(0.9);

    const classes = nb.classes;
    expect(classes?.toArray()).toEqual([0, 1]);
  });

  it("throws when predicting before fit", () => {
    const nb = new GaussianNB();
    expect(() => nb.predict(tensor([[1, 2]]))).toThrow(/fitted/i);
    expect(() => nb.predictProba(tensor([[1, 2]]))).toThrow(/fitted/i);
  });

  it("validates feature count on predict", () => {
    const X = tensor([
      [1, 2],
      [2, 3],
      [4, 5],
      [4, 6],
    ]);
    const y = tensor([0, 0, 1, 1]);
    const nb = new GaussianNB();
    nb.fit(X, y);
    expect(() => nb.predict(tensor([[1, 2, 3]]))).toThrow(/features/i);
  });
});
