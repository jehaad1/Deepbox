import { describe, expect, it } from "vitest";
import { GaussianNB } from "../src/ml";
import { tensor } from "../src/ndarray";

describe("GaussianNB", () => {
  it("should classify simple data", () => {
    const X = tensor([
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 5],
    ]);
    const y = tensor([0, 0, 1, 1]);

    const nb = new GaussianNB();
    nb.fit(X, y);

    const predictions = nb.predict(tensor([[2.5, 3.5]]));
    expect(predictions.shape).toEqual([1]);
  });

  it("should predict probabilities", () => {
    const X = tensor([
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 5],
    ]);
    const y = tensor([0, 0, 1, 1]);

    const nb = new GaussianNB();
    nb.fit(X, y);

    const proba = nb.predictProba(tensor([[2.5, 3.5]]));
    expect(proba.ndim).toBe(2);
    expect(proba.shape[1]).toBe(2);
  });

  it("should calculate accuracy score", () => {
    const X = tensor([
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 5],
    ]);
    const y = tensor([0, 0, 1, 1]);

    const nb = new GaussianNB();
    nb.fit(X, y);

    const score = nb.score(X, y);
    expect(score).toBeGreaterThan(0);
    expect(score).toBeLessThanOrEqual(1);
  });
});
