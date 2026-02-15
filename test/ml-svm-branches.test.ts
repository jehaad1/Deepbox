import { describe, expect, it } from "vitest";
import { LinearSVC, LinearSVR } from "../src/ml/svm/SVM";
import { tensor } from "../src/ndarray";

describe("SVM branches", () => {
  it("validates constructor params", () => {
    expect(() => new LinearSVC({ C: 0 })).toThrow(/C must be positive/i);
    expect(() => new LinearSVC({ maxIter: 0 })).toThrow(/maxIter must be a positive integer/i);
    expect(() => new LinearSVC({ tol: -1 })).toThrow(/tol must be >= 0/i);
  });

  it("handles fit validation and binary class requirement", () => {
    const svc = new LinearSVC();
    expect(() => svc.fit(tensor([1, 2]), tensor([0, 1]))).toThrow(/2-dimensional/i);
    expect(() => svc.fit(tensor([[1, 2]]), tensor([[0]]))).toThrow(/1-dimensional/i);
    expect(() =>
      svc.fit(
        tensor([
          [1, 2],
          [3, 4],
        ]),
        tensor([0, 1, 2])
      )
    ).toThrow(/same number of samples/i);

    // Multiclass now supported via OvR; single-class should still throw
    const X = tensor([
      [1, 2],
      [2, 3],
      [3, 4],
    ]);
    expect(() => svc.fit(X, tensor([0, 0, 0]))).toThrow(/at least 2 classes/i);
  });

  it("predicts and exposes coef/intercept", () => {
    const X = tensor([
      [1, 1],
      [2, 2],
      [3, 3],
      [4, 4],
    ]);
    const y = tensor([0, 0, 1, 1]);
    const svc = new LinearSVC({ maxIter: 200 });
    svc.fit(X, y);

    const preds = svc.predict(
      tensor([
        [1.5, 1.5],
        [3.5, 3.5],
      ])
    );
    expect(preds.shape[0]).toBe(2);

    const proba = svc.predictProba(tensor([[2, 2]]));
    expect(proba.shape).toEqual([1, 2]);

    expect(svc.coef.shape).toEqual([1, 2]);
    expect(svc.intercept.shape).toEqual([1]);

    expect(() => svc.predict(tensor([[1, 2, 3]]))).toThrow(/features/i);
  });

  it("throws on predict before fit", () => {
    const svc = new LinearSVC();
    expect(() => svc.predict(tensor([[1, 2]]))).toThrow(/fitted/i);
    expect(() => svc.predictProba(tensor([[1, 2]]))).toThrow(/fitted/i);
    expect(() => svc.coef).toThrow(/fitted/i);
    expect(() => svc.intercept).toThrow(/fitted/i);
  });

  it("covers SVR branches and constant-score path", () => {
    const svr = new LinearSVR({ epsilon: 0.5, maxIter: 50 });
    const X = tensor([[0], [0]]);
    const y = tensor([0, 0]);
    svr.fit(X, y);

    const preds = svr.predict(X);
    expect(preds.toArray()).toEqual([0, 0]);

    const score = svr.score(X, y);
    expect(score).toBe(1);

    expect(() => svr.predict(tensor([[0, 1]]))).toThrow(/features/i);
  });

  it("validates SVR constructor params", () => {
    expect(() => new LinearSVR({ C: 0 })).toThrow(/C must be positive/i);
    expect(() => new LinearSVR({ epsilon: -1 })).toThrow(/epsilon must be/i);
    expect(() => new LinearSVR({ maxIter: 0 })).toThrow(/maxIter must be positive/i);
    expect(() => new LinearSVR({ tol: -1 })).toThrow(/tol must be >= 0/i);
  });
});
