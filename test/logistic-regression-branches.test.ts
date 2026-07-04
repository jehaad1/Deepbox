import { describe, expect, it } from "vitest";
import { LogisticRegression } from "../src/ml/linear/LogisticRegression";
import { tensor } from "../src/ndarray";

const X = tensor([
  [1, 2],
  [2, 3],
  [3, 4],
  [5, 6],
  [6, 7],
  [7, 8],
]);
const yBin = tensor([0, 0, 0, 1, 1, 1]);
const yMulti = tensor([0, 0, 1, 1, 2, 2]);

// ────── Binary classification ──────
describe("LogisticRegression binary", () => {
  it("fits and predicts", () => {
    const lr = new LogisticRegression({ maxIter: 200, learningRate: 0.1 });
    lr.fit(X, yBin);
    const pred = lr.predict(X);
    expect(pred.shape).toEqual([6]);
  });

  it("predictProba returns 2 columns", () => {
    const lr = new LogisticRegression({ maxIter: 200 });
    lr.fit(X, yBin);
    const proba = lr.predictProba(X);
    expect(proba.shape).toEqual([6, 2]);
  });

  it("score", () => {
    const lr = new LogisticRegression({ maxIter: 200 });
    lr.fit(X, yBin);
    const s = lr.score(X, yBin);
    expect(s).toBeGreaterThanOrEqual(0);
    expect(s).toBeLessThanOrEqual(1);
  });

  it("coef and intercept accessors", () => {
    const lr = new LogisticRegression({ maxIter: 200 });
    lr.fit(X, yBin);
    expect(lr.coef.shape).toEqual([2]);
    expect(typeof lr.intercept).toBe("number");
  });

  it("classes accessor", () => {
    const lr = new LogisticRegression();
    lr.fit(X, yBin);
    expect(lr.classes).toBeDefined();
  });

  it("fitIntercept=false", () => {
    const lr = new LogisticRegression({ fitIntercept: false, maxIter: 100 });
    lr.fit(X, yBin);
    expect(lr.intercept).toBe(0);
  });

  it("penalty=none", () => {
    const lr = new LogisticRegression({ penalty: "none", maxIter: 100 });
    lr.fit(X, yBin);
    expect(lr.predict(X).shape).toEqual([6]);
  });

  it("penalty=l1 with saga solver", () => {
    const lr = new LogisticRegression({
      penalty: "l1",
      solver: "saga",
      maxIter: 100,
    });
    lr.fit(X, yBin);
    expect(lr.predict(X).shape).toEqual([6]);
  });

  it("penalty=l1 with liblinear solver", () => {
    const lr = new LogisticRegression({
      penalty: "l1",
      solver: "liblinear",
      maxIter: 100,
    });
    lr.fit(X, yBin);
    expect(lr.predict(X).shape).toEqual([6]);
  });

  it("classWeight=balanced", () => {
    const lr = new LogisticRegression({
      classWeight: "balanced",
      maxIter: 100,
    });
    lr.fit(X, yBin);
    expect(lr.predict(X).shape).toEqual([6]);
  });

  it("classWeight=custom dict", () => {
    const lr = new LogisticRegression({
      classWeight: { 0: 1, 1: 2 },
      maxIter: 100,
    });
    lr.fit(X, yBin);
    expect(lr.predict(X).shape).toEqual([6]);
  });

  it("non-0/1 binary labels", () => {
    const y = tensor([3, 3, 3, 7, 7, 7]);
    const lr = new LogisticRegression({ maxIter: 100 });
    lr.fit(X, y);
    const pred = lr.predict(X);
    expect(pred.shape).toEqual([6]);
  });
});

// ────── Multiclass classification ──────
describe("LogisticRegression multiclass", () => {
  it("fits and predicts", () => {
    const lr = new LogisticRegression({ maxIter: 200 });
    lr.fit(X, yMulti);
    const pred = lr.predict(X);
    expect(pred.shape).toEqual([6]);
  });

  it("predictProba returns n_classes columns", () => {
    const lr = new LogisticRegression({ maxIter: 200 });
    lr.fit(X, yMulti);
    const proba = lr.predictProba(X);
    expect(proba.shape).toEqual([6, 3]);
  });

  it("score", () => {
    const lr = new LogisticRegression({ maxIter: 200 });
    lr.fit(X, yMulti);
    const s = lr.score(X, yMulti);
    expect(typeof s).toBe("number");
  });

  it("multiClass=ovr", () => {
    const lr = new LogisticRegression({ multiClass: "ovr", maxIter: 100 });
    lr.fit(X, yMulti);
    expect(lr.predict(X).shape).toEqual([6]);
  });

  it("classWeight=balanced multiclass", () => {
    const lr = new LogisticRegression({
      classWeight: "balanced",
      maxIter: 100,
    });
    lr.fit(X, yMulti);
    expect(lr.predict(X).shape).toEqual([6]);
  });
});

// ────── Error handling ──────
describe("LogisticRegression errors", () => {
  it("throws for not fitted predict", () => {
    expect(() => new LogisticRegression().predict(X)).toThrow(/fitted/i);
  });

  it("throws for not fitted predictProba", () => {
    expect(() => new LogisticRegression().predictProba(X)).toThrow(/fitted/i);
  });

  it("throws for not fitted coef", () => {
    expect(() => new LogisticRegression().coef).toThrow(/fitted/i);
  });

  it("throws for not fitted intercept", () => {
    expect(() => new LogisticRegression().intercept).toThrow(/fitted/i);
  });

  it("throws for C <= 0", () => {
    expect(() => new LogisticRegression({ C: 0 })).toThrow();
    expect(() => new LogisticRegression({ C: -1 })).toThrow();
  });

  it("throws for invalid maxIter", () => {
    expect(() => new LogisticRegression({ maxIter: 0 })).toThrow();
    expect(() => new LogisticRegression({ maxIter: -1 })).toThrow();
  });

  it("throws for invalid tol", () => {
    expect(() => new LogisticRegression({ tol: -1 })).toThrow();
  });

  it("throws for invalid learningRate", () => {
    expect(() => new LogisticRegression({ learningRate: 0 })).toThrow();
    expect(() => new LogisticRegression({ learningRate: -1 })).toThrow();
  });

  it("throws for lbfgs + l1", () => {
    expect(() => new LogisticRegression({ solver: "lbfgs", penalty: "l1" })).toThrow();
  });
});

// ────── getParams / setParams ──────
describe("LogisticRegression params", () => {
  it("getParams returns defaults", () => {
    const lr = new LogisticRegression();
    const params = lr.getParams();
    expect(params.penalty).toBe("l2");
    expect(params.solver).toBe("lbfgs");
  });

  it("setParams updates maxIter", () => {
    const lr = new LogisticRegression();
    lr.setParams({ maxIter: 500 });
    expect(lr.getParams().maxIter).toBe(500);
  });

  it("setParams updates tol", () => {
    const lr = new LogisticRegression();
    lr.setParams({ tol: 0.001 });
    expect(lr.getParams().tol).toBe(0.001);
  });

  it("setParams updates C", () => {
    const lr = new LogisticRegression();
    lr.setParams({ C: 10 });
    expect(lr.getParams().C).toBe(10);
  });

  it("setParams updates learningRate", () => {
    const lr = new LogisticRegression();
    lr.setParams({ learningRate: 0.5 });
    expect(lr.getParams().learningRate).toBe(0.5);
  });

  it("setParams updates fitIntercept", () => {
    const lr = new LogisticRegression();
    lr.setParams({ fitIntercept: false });
    expect(lr.getParams().fitIntercept).toBe(false);
  });

  it("setParams updates penalty", () => {
    const lr = new LogisticRegression();
    lr.setParams({ penalty: "none" });
    expect(lr.getParams().penalty).toBe("none");
  });

  it("setParams updates solver", () => {
    const lr = new LogisticRegression();
    lr.setParams({ solver: "saga" });
    expect(lr.getParams().solver).toBe("saga");
  });

  it("setParams updates multiClass", () => {
    const lr = new LogisticRegression();
    lr.setParams({ multiClass: "ovr" });
    expect(lr.getParams().multiClass).toBe("ovr");
  });

  it("setParams rejects invalid maxIter", () => {
    expect(() => new LogisticRegression().setParams({ maxIter: -1 })).toThrow();
  });

  it("setParams rejects invalid tol", () => {
    expect(() => new LogisticRegression().setParams({ tol: -1 })).toThrow();
  });

  it("setParams rejects invalid C", () => {
    expect(() => new LogisticRegression().setParams({ C: 0 })).toThrow();
  });

  it("setParams rejects invalid learningRate", () => {
    expect(() => new LogisticRegression().setParams({ learningRate: 0 })).toThrow();
  });

  it("setParams rejects invalid fitIntercept", () => {
    expect(() => new LogisticRegression().setParams({ fitIntercept: "yes" })).toThrow();
  });

  it("setParams rejects invalid penalty", () => {
    expect(() => new LogisticRegression().setParams({ penalty: "elastic" })).toThrow();
  });

  it("setParams rejects invalid solver", () => {
    expect(() => new LogisticRegression().setParams({ solver: "adam" })).toThrow();
  });

  it("setParams rejects invalid multiClass", () => {
    expect(() => new LogisticRegression().setParams({ multiClass: "bad" })).toThrow();
  });

  it("setParams rejects unknown param", () => {
    expect(() => new LogisticRegression().setParams({ unknown: 1 })).toThrow();
  });
});
