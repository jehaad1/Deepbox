import { describe, expect, it } from "vitest";
import { LogisticRegression } from "../src/ml/linear/LogisticRegression";
import { OneVsOneClassifier } from "../src/ml/multiclass";
import { tensor } from "../src/ndarray";

describe("OneVsOneClassifier.predictProba()", () => {
  it("returns normalized probabilities for all classes", () => {
    // Simple 3-class dataset
    const X = tensor([
      [1, 0],
      [1.1, 0.1],
      [0, 1],
      [0.1, 1.1],
      [1, 1],
      [1.1, 1.1],
    ]);
    const y = tensor([0, 0, 1, 1, 2, 2]);

    const ovo = new OneVsOneClassifier({ estimator: new LogisticRegression() });
    ovo.fit(X, y);

    const proba = ovo.predictProba(X);
    expect(proba.shape).toEqual([6, 3]);

    // Each row should sum to ~1.0
    for (let i = 0; i < 6; i++) {
      let rowSum = 0;
      for (let j = 0; j < 3; j++) {
        const p = Number(proba.data[proba.offset + i * 3 + j]);
        expect(p).toBeGreaterThanOrEqual(0);
        expect(p).toBeLessThanOrEqual(1);
        rowSum += p;
      }
      expect(rowSum).toBeCloseTo(1.0, 5);
    }
  });

  it("throws NotFittedError if not fitted", () => {
    const ovo = new OneVsOneClassifier({ estimator: new LogisticRegression() });
    expect(() => ovo.predictProba(tensor([[0, 0]]))).toThrow();
  });

  it("works with binary classification", () => {
    const X = tensor([
      [0, 0],
      [0.1, 0],
      [1, 1],
      [1.1, 1],
    ]);
    const y = tensor([0, 0, 1, 1]);

    const ovo = new OneVsOneClassifier({ estimator: new LogisticRegression() });
    ovo.fit(X, y);

    const proba = ovo.predictProba(X);
    expect(proba.shape).toEqual([4, 2]);

    // Each row should sum to ~1.0
    for (let i = 0; i < 4; i++) {
      let rowSum = 0;
      for (let j = 0; j < 2; j++) {
        rowSum += Number(proba.data[proba.offset + i * 2 + j]);
      }
      expect(rowSum).toBeCloseTo(1.0, 5);
    }
  });
});
