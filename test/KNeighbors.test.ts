import { describe, expect, it } from "vitest";
import { KNeighborsClassifier, KNeighborsRegressor } from "../src/ml";
import { tensor } from "../src/ndarray";

describe("KNeighborsClassifier", () => {
  it("should classify simple data", () => {
    const X = tensor([
      [0, 0],
      [1, 1],
      [2, 2],
      [3, 3],
    ]);
    const y = tensor([0, 0, 1, 1]);

    const knn = new KNeighborsClassifier({ nNeighbors: 3 });
    knn.fit(X, y);

    const predictions = knn.predict(tensor([[1.5, 1.5]]));
    expect(predictions.shape).toEqual([1]);
  });

  it("should calculate accuracy score", () => {
    const X = tensor([
      [0, 0],
      [1, 1],
      [2, 2],
      [3, 3],
    ]);
    const y = tensor([0, 0, 1, 1]);

    const knn = new KNeighborsClassifier({ nNeighbors: 3 });
    knn.fit(X, y);

    const score = knn.score(X, y);
    expect(score).toBeGreaterThan(0.5);
    expect(score).toBeLessThanOrEqual(1.0);
  });

  it("should predict probabilities", () => {
    const X = tensor([
      [0, 0],
      [1, 1],
      [2, 2],
      [3, 3],
    ]);
    const y = tensor([0, 0, 1, 1]);

    const knn = new KNeighborsClassifier({ nNeighbors: 3 });
    knn.fit(X, y);

    const proba = knn.predictProba(tensor([[1.5, 1.5]]));
    expect(proba.ndim).toBe(2);
    expect(proba.shape[0]).toBe(1);
  });
});

describe("KNeighborsRegressor", () => {
  it("should predict continuous values", () => {
    const X = tensor([[0], [1], [2], [3]]);
    const y = tensor([0, 1, 4, 9]);

    const knn = new KNeighborsRegressor({ nNeighbors: 2 });
    knn.fit(X, y);

    const predictions = knn.predict(tensor([[1.5]]));
    expect(predictions.shape).toEqual([1]);
  });

  it("should calculate R2 score", () => {
    const X = tensor([[0], [1], [2], [3]]);
    const y = tensor([0, 1, 4, 9]);

    const knn = new KNeighborsRegressor({ nNeighbors: 2 });
    knn.fit(X, y);

    const score = knn.score(X, y);
    expect(typeof score).toBe("number");
  });
});
