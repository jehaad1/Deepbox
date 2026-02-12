import { describe, expect, it } from "vitest";
import { LogisticRegression } from "../src/ml/linear/LogisticRegression";
import { tensor } from "../src/ndarray";

describe("LogisticRegression Multiclass (OvR)", () => {
  it("fits and predicts on multiclass data (3 classes)", () => {
    // 3 classes, 2 features
    // Class 0: near (0,0)
    // Class 1: near (5,5)
    // Class 2: near (10,0)
    const X = tensor([
      [0, 0],
      [0.1, 0.1],
      [-0.1, -0.1], // Class 0
      [5, 5],
      [5.1, 5.1],
      [4.9, 4.9], // Class 1
      [10, 0],
      [10.1, 0.1],
      [9.9, -0.1], // Class 2
    ]);
    const y = tensor([0, 0, 0, 1, 1, 1, 2, 2, 2]);

    const lr = new LogisticRegression({ multiClass: "ovr", C: 1.0 });
    lr.fit(X, y);

    expect(lr.coef.shape).toEqual([3, 2]);
    expect(Array.isArray(lr.intercept)).toBe(true);
    if (!Array.isArray(lr.intercept)) throw new Error("unreachable");
    expect(lr.intercept.length).toBe(3);

    // Predict on training data
    const preds = lr.predict(X);
    expect(preds.shape).toEqual([9]);

    // Check accuracy
    const score = lr.score(X, y);
    expect(score).toBe(1.0);

    // Predict proba
    const proba = lr.predictProba(X);
    expect(proba.shape).toEqual([9, 3]);

    // Check that probabilities sum to 1 (approximately)
    for (let i = 0; i < 9; i++) {
      let sum = 0;
      for (let j = 0; j < 3; j++) {
        sum += Number(proba.data[proba.offset + i * 3 + j]);
      }
      expect(sum).toBeCloseTo(1.0, 5);
    }
  });

  it("handles binary classification correctly (fallback)", () => {
    const X = tensor([[0], [1], [0], [1]]);
    const y = tensor([0, 1, 0, 1]);

    const lr = new LogisticRegression({ multiClass: "ovr" }); // explicit ovr but binary data
    // Should default to binary logic if only 2 classes found?
    // Wait, my implementation checks uniqueClasses.length <= 2 -> binary logic.
    // So even if multiClass="ovr", if data is binary, it uses binary implementation?
    // Let's verify the implementation details.
    // Line 267: if (uniqueClasses.length <= 2) { ... binary ... } else { ... multiclass ... }
    // So yes, it uses binary implementation.

    lr.fit(X, y);
    expect(lr.coef.shape).toEqual([1]); // Binary coef shape is (n_features,)
    expect(typeof lr.intercept).toBe("number");

    const preds = lr.predict(X);
    expect(preds.shape).toEqual([4]);
    expect(lr.score(X, y)).toBe(1.0);
  });
});
