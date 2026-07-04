import { describe, expect, it } from "vitest";
import { LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis } from "../src/ml";
import { tensor } from "../src/ndarray";

describe("LinearDiscriminantAnalysis", () => {
  const X = tensor([
    [1, 2],
    [2, 3],
    [3, 3],
    [8, 7],
    [9, 8],
    [10, 8],
  ]);
  const y = tensor([0, 0, 0, 1, 1, 1]);

  it("should classify well-separated data", () => {
    const lda = new LinearDiscriminantAnalysis();
    lda.fit(X, y);

    const predictions = lda.predict(X);
    expect(predictions.shape).toEqual([6]);
    // Well-separated data should be classified perfectly
    for (let i = 0; i < 6; i++) {
      expect(Number(predictions.data[i])).toBe(Number(y.data[i]));
    }
  });

  it("should predict probabilities summing to 1", () => {
    const lda = new LinearDiscriminantAnalysis();
    lda.fit(X, y);

    const proba = lda.predictProba(tensor([[5, 5]]));
    expect(proba.ndim).toBe(2);
    expect(proba.shape[1]).toBe(2);
    const p0 = Number(proba.data[0]);
    const p1 = Number(proba.data[1]);
    expect(p0 + p1).toBeCloseTo(1, 5);
    expect(p0).toBeGreaterThanOrEqual(0);
    expect(p1).toBeGreaterThanOrEqual(0);
  });

  it("should compute accuracy score", () => {
    const lda = new LinearDiscriminantAnalysis();
    lda.fit(X, y);

    const score = lda.score(X, y);
    expect(score).toBe(1); // Perfect score on well-separated data
  });

  it("should transform data for dimensionality reduction", () => {
    const lda = new LinearDiscriminantAnalysis();
    lda.fit(X, y);

    const Xt = lda.transform(X);
    expect(Xt.shape[0]).toBe(6);
    expect(Xt.shape[1]).toBe(1); // min(n_classes - 1, n_features) = 1
  });

  it("should support fitTransform", () => {
    const lda = new LinearDiscriminantAnalysis();
    const Xt = lda.fitTransform(X, y);
    expect(Xt.shape[0]).toBe(6);
    expect(Xt.shape[1]).toBe(1);
  });

  it("should expose classes after fitting", () => {
    const lda = new LinearDiscriminantAnalysis();
    lda.fit(X, y);
    const classes = lda.classes;
    expect(classes).toBeDefined();
    expect(classes!.shape).toEqual([2]);
  });

  it("should support shrinkage='auto'", () => {
    const lda = new LinearDiscriminantAnalysis({ shrinkage: "auto" });
    lda.fit(X, y);
    const score = lda.score(X, y);
    expect(score).toBeGreaterThan(0.5);
  });

  it("should support numeric shrinkage", () => {
    const lda = new LinearDiscriminantAnalysis({ shrinkage: 0.5 });
    lda.fit(X, y);
    const score = lda.score(X, y);
    expect(score).toBeGreaterThan(0.5);
  });

  it("should throw on invalid nComponents", () => {
    expect(() => new LinearDiscriminantAnalysis({ nComponents: 0 })).toThrow();
    expect(() => new LinearDiscriminantAnalysis({ nComponents: -1 })).toThrow();
  });

  it("should throw on invalid shrinkage", () => {
    expect(() => new LinearDiscriminantAnalysis({ shrinkage: -0.1 })).toThrow();
    expect(() => new LinearDiscriminantAnalysis({ shrinkage: 1.5 })).toThrow();
  });

  it("should throw when predicting before fitting", () => {
    const lda = new LinearDiscriminantAnalysis();
    expect(() => lda.predict(tensor([[1, 2]]))).toThrow();
  });

  it("should throw when transforming before fitting", () => {
    const lda = new LinearDiscriminantAnalysis();
    expect(() => lda.transform(tensor([[1, 2]]))).toThrow();
  });

  it("should throw on less than 2 classes", () => {
    const lda = new LinearDiscriminantAnalysis();
    expect(() =>
      lda.fit(
        tensor([
          [1, 2],
          [3, 4],
        ]),
        tensor([0, 0])
      )
    ).toThrow();
  });

  it("should get and set params", () => {
    const lda = new LinearDiscriminantAnalysis({ nComponents: 1 });
    const params = lda.getParams();
    expect(params.nComponents).toBe(1);

    lda.setParams({ shrinkage: 0.3 });
    expect(lda.getParams().shrinkage).toBe(0.3);
  });

  it("should throw on unknown parameter", () => {
    const lda = new LinearDiscriminantAnalysis();
    lda.fit(X, y);
    expect(() => lda.setParams({ invalidParam: 42 })).toThrow();
  });

  it("should handle multi-class data", () => {
    const Xmc = tensor([
      [1, 1],
      [1, 2],
      [2, 1],
      [10, 1],
      [10, 2],
      [11, 1],
      [5, 10],
      [5, 11],
      [6, 10],
    ]);
    const ymc = tensor([0, 0, 0, 1, 1, 1, 2, 2, 2]);

    const lda = new LinearDiscriminantAnalysis();
    lda.fit(Xmc, ymc);

    const pred = lda.predict(Xmc);
    expect(pred.shape).toEqual([9]);
    const score = lda.score(Xmc, ymc);
    expect(score).toBeGreaterThan(0.5);
  });
});

describe("QuadraticDiscriminantAnalysis", () => {
  const X = tensor([
    [1, 2],
    [2, 3],
    [3, 3],
    [8, 7],
    [9, 8],
    [10, 8],
  ]);
  const y = tensor([0, 0, 0, 1, 1, 1]);

  it("should classify well-separated data", () => {
    const qda = new QuadraticDiscriminantAnalysis();
    qda.fit(X, y);

    const predictions = qda.predict(X);
    expect(predictions.shape).toEqual([6]);
    for (let i = 0; i < 6; i++) {
      expect(Number(predictions.data[i])).toBe(Number(y.data[i]));
    }
  });

  it("should predict probabilities summing to 1", () => {
    const qda = new QuadraticDiscriminantAnalysis();
    qda.fit(X, y);

    const proba = qda.predictProba(tensor([[5, 5]]));
    expect(proba.ndim).toBe(2);
    expect(proba.shape[1]).toBe(2);
    const p0 = Number(proba.data[0]);
    const p1 = Number(proba.data[1]);
    expect(p0 + p1).toBeCloseTo(1, 5);
  });

  it("should compute accuracy score", () => {
    const qda = new QuadraticDiscriminantAnalysis();
    qda.fit(X, y);

    const score = qda.score(X, y);
    expect(score).toBe(1);
  });

  it("should support regularization", () => {
    const qda = new QuadraticDiscriminantAnalysis({ regParam: 0.1 });
    qda.fit(X, y);
    const score = qda.score(X, y);
    expect(score).toBeGreaterThan(0.5);
  });

  it("should throw on invalid regParam", () => {
    expect(() => new QuadraticDiscriminantAnalysis({ regParam: -1 })).toThrow();
  });

  it("should throw when predicting before fitting", () => {
    const qda = new QuadraticDiscriminantAnalysis();
    expect(() => qda.predict(tensor([[1, 2]]))).toThrow();
  });

  it("should throw on less than 2 classes", () => {
    const qda = new QuadraticDiscriminantAnalysis();
    expect(() =>
      qda.fit(
        tensor([
          [1, 2],
          [3, 4],
        ]),
        tensor([0, 0])
      )
    ).toThrow();
  });

  it("should expose classes after fitting", () => {
    const qda = new QuadraticDiscriminantAnalysis();
    qda.fit(X, y);
    expect(qda.classes).toBeDefined();
    expect(qda.classes!.shape).toEqual([2]);
  });

  it("should get and set params", () => {
    const qda = new QuadraticDiscriminantAnalysis({ regParam: 0.5 });
    expect(qda.getParams().regParam).toBe(0.5);
    qda.setParams({ regParam: 0.1 });
    expect(qda.getParams().regParam).toBe(0.1);
  });

  it("should throw on unknown parameter", () => {
    const qda = new QuadraticDiscriminantAnalysis();
    expect(() => qda.setParams({ invalidParam: 42 })).toThrow();
  });
});
