import { describe, expect, it } from "vitest";
import { KNeighborsClassifier, KNeighborsRegressor } from "../src/ml/neighbors";
import { tensor } from "../src/ndarray";
import { toNum2D } from "./_helpers";

describe("KNeighbors branches", () => {
  it("validates constructor and fit inputs", () => {
    expect(() => new KNeighborsClassifier({ nNeighbors: 0 })).toThrow(/>= 1/);

    const knn = new KNeighborsClassifier();
    expect(() => knn.fit(tensor([1, 2]), tensor([0, 1]))).toThrow(/2-dimensional/i);
    expect(() => knn.fit(tensor([[1, 2]]), tensor([[0]]))).toThrow(/1-dimensional/i);
    expect(() => knn.fit(tensor([[1, 2]]), tensor([0, 1]))).toThrow(/same number of samples/i);
  });

  it("rejects nNeighbors greater than n_samples and feature mismatch", () => {
    const X = tensor([[0], [1], [2]]);
    const y = tensor([0, 0, 1]);
    const tooMany = new KNeighborsClassifier({ nNeighbors: 4 });
    expect(() => tooMany.fit(X, y)).toThrow(/nNeighbors.*n_samples/i);

    const knn = new KNeighborsClassifier({ nNeighbors: 1 });
    knn.fit(X, y);
    expect(() => knn.predict(tensor([[1, 2]]))).toThrow(/features/i);
  });

  it("predicts with uniform and distance weights", () => {
    const X = tensor([[0], [1], [2]]);
    const y = tensor([0, 0, 1]);

    const knn = new KNeighborsClassifier({
      nNeighbors: 1,
      weights: "uniform",
      metric: "manhattan",
    });
    knn.fit(X, y);
    const pred = knn.predict(tensor([[1.8]]));
    expect(pred.toArray()).toEqual([1]);

    const knnDist = new KNeighborsClassifier({ nNeighbors: 2, weights: "distance" });
    knnDist.fit(X, y);
    const proba = knnDist.predictProba(tensor([[0.2]]));
    const probs = toNum2D(proba.toArray())[0];
    const sum = probs.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 6);

    const score = knn.score(X, y);
    expect(score).toBeGreaterThan(0.9);
  });

  it("predicts regression and score", () => {
    const X = tensor([[0], [1], [2]]);
    const y = tensor([0, 1, 4]);

    const knn = new KNeighborsRegressor({ nNeighbors: 2, weights: "distance" });
    knn.fit(X, y);
    const pred = knn.predict(tensor([[1.5]]));
    expect(pred.shape).toEqual([1]);

    const score = knn.score(X, y);
    expect(score).toBeGreaterThan(0.8);
  });

  it("throws when predicting before fit", () => {
    const knn = new KNeighborsClassifier();
    expect(() => knn.predict(tensor([[1]]))).toThrow(/fitted/i);
    expect(() => knn.predictProba(tensor([[1]]))).toThrow(/fitted/i);

    const knr = new KNeighborsRegressor();
    expect(() => knr.predict(tensor([[1]]))).toThrow(/fitted/i);
  });
});
