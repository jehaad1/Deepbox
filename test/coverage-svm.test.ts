import { describe, expect, it } from "vitest";
import { SVC, SVR } from "../src/ml/svm/KernelSVM";
import { LinearSVC, LinearSVR } from "../src/ml/svm/SVM";
import { tensor } from "../src/ndarray";
import { setSeed } from "../src/random";

// ---- LinearSVC extended branch coverage ----

describe("LinearSVC extended branches", () => {
  const X = tensor([
    [1, 2],
    [2, 3],
    [3, 1],
    [4, 2],
    [5, 6],
    [6, 7],
    [7, 5],
    [8, 6],
  ]);
  const y = tensor([0, 0, 0, 0, 1, 1, 1, 1]);

  it("binary classification fit/predict", () => {
    const svm = new LinearSVC({ C: 1.0, maxIter: 500 });
    svm.fit(X, y);
    const preds = svm.predict(X);
    expect(preds.shape).toEqual([8]);
  });

  it("predictProba for binary returns 2 columns", () => {
    const svm = new LinearSVC({ C: 1.0, maxIter: 500 });
    svm.fit(X, y);
    const proba = svm.predictProba(X);
    expect(proba.shape[1]).toBe(2);
    // Each row sums to 1
    const data = proba.toArray() as number[][];
    for (const row of data) {
      const sum = (row as number[]).reduce((a: number, b: number) => a + b, 0);
      expect(sum).toBeCloseTo(1.0, 5);
    }
  });

  it("score returns accuracy", () => {
    const svm = new LinearSVC({ C: 1.0, maxIter: 500 });
    svm.fit(X, y);
    const score = svm.score(X, y);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("coef and intercept accessible after fit", () => {
    const svm = new LinearSVC({ C: 1.0, maxIter: 500 });
    svm.fit(X, y);
    const coef = svm.coef;
    expect(coef.shape.length).toBeGreaterThanOrEqual(1);
    const intercept = svm.intercept;
    expect(intercept.shape.length).toBeGreaterThanOrEqual(1);
  });

  it("coef throws before fit", () => {
    const svm = new LinearSVC();
    expect(() => svm.coef).toThrow(/fitted/i);
    expect(() => svm.intercept).toThrow(/fitted/i);
  });

  it("multiclass OvR classification", () => {
    const Xm = tensor([
      [1, 0],
      [1, 1],
      [0, 1],
      [5, 0],
      [5, 1],
      [5, 2],
      [10, 0],
      [10, 1],
      [10, 2],
    ]);
    const ym = tensor([0, 0, 0, 1, 1, 1, 2, 2, 2]);
    const svm = new LinearSVC({ C: 1.0, maxIter: 500 });
    svm.fit(Xm, ym);
    const preds = svm.predict(Xm);
    expect(preds.shape).toEqual([9]);
  });

  it("multiclass predictProba returns nClasses columns", () => {
    const Xm = tensor([
      [1, 0],
      [1, 1],
      [0, 1],
      [5, 0],
      [5, 1],
      [5, 2],
      [10, 0],
      [10, 1],
      [10, 2],
    ]);
    const ym = tensor([0, 0, 0, 1, 1, 1, 2, 2, 2]);
    const svm = new LinearSVC({ C: 1.0, maxIter: 500 });
    svm.fit(Xm, ym);
    const proba = svm.predictProba(Xm);
    expect(proba.shape[1]).toBe(3);
  });

  it("classWeight balanced", () => {
    const svm = new LinearSVC({ C: 1.0, maxIter: 200, classWeight: "balanced" });
    svm.fit(X, y);
    const preds = svm.predict(X);
    expect(preds.shape).toEqual([8]);
  });

  it("classWeight as record", () => {
    const svm = new LinearSVC({ C: 1.0, maxIter: 200, classWeight: { 0: 1, 1: 2 } });
    svm.fit(X, y);
    const preds = svm.predict(X);
    expect(preds.shape).toEqual([8]);
  });

  it("throws when predicting before fit", () => {
    const svm = new LinearSVC();
    expect(() => svm.predict(tensor([[1, 2]]))).toThrow(/fitted/i);
    expect(() => svm.predictProba(tensor([[1, 2]]))).toThrow(/fitted/i);
  });

  it("rejects single class", () => {
    const svm = new LinearSVC();
    expect(() =>
      svm.fit(
        tensor([
          [1, 2],
          [3, 4],
        ]),
        tensor([0, 0])
      )
    ).toThrow(/2 classes/i);
  });

  it("constructor rejects invalid C", () => {
    expect(() => new LinearSVC({ C: 0 })).toThrow();
    expect(() => new LinearSVC({ C: -1 })).toThrow();
    expect(() => new LinearSVC({ C: NaN })).toThrow();
  });

  it("constructor rejects invalid maxIter", () => {
    expect(() => new LinearSVC({ maxIter: 0 })).toThrow();
    expect(() => new LinearSVC({ maxIter: -1 })).toThrow();
    expect(() => new LinearSVC({ maxIter: 1.5 })).toThrow();
  });

  it("constructor rejects invalid tol", () => {
    expect(() => new LinearSVC({ tol: -1 })).toThrow();
    expect(() => new LinearSVC({ tol: NaN })).toThrow();
  });

  it("getParams returns correct values", () => {
    const svm = new LinearSVC({ C: 2.0, maxIter: 500, tol: 1e-3 });
    const params = svm.getParams();
    expect(params.C).toBe(2.0);
    expect(params.maxIter).toBe(500);
    expect(params.tol).toBe(1e-3);
  });

  it("setParams updates C", () => {
    const svm = new LinearSVC();
    svm.setParams({ C: 5.0 });
    expect(svm.getParams().C).toBe(5.0);
  });

  it("setParams updates maxIter", () => {
    const svm = new LinearSVC();
    svm.setParams({ maxIter: 2000 });
    expect(svm.getParams().maxIter).toBe(2000);
  });

  it("setParams updates tol", () => {
    const svm = new LinearSVC();
    svm.setParams({ tol: 0.001 });
    expect(svm.getParams().tol).toBe(0.001);
  });

  it("setParams rejects invalid C", () => {
    const svm = new LinearSVC();
    expect(() => svm.setParams({ C: 0 })).toThrow();
    expect(() => svm.setParams({ C: -1 })).toThrow();
  });

  it("setParams rejects invalid maxIter", () => {
    const svm = new LinearSVC();
    expect(() => svm.setParams({ maxIter: 0 })).toThrow();
    expect(() => svm.setParams({ maxIter: 1.5 })).toThrow();
  });

  it("setParams rejects invalid tol", () => {
    const svm = new LinearSVC();
    expect(() => svm.setParams({ tol: -1 })).toThrow();
  });

  it("setParams rejects unknown param", () => {
    const svm = new LinearSVC();
    expect(() => svm.setParams({ unknown: 42 })).toThrow(/unknown/i);
  });

  it("score rejects non-1D y", () => {
    const svm = new LinearSVC({ maxIter: 100 });
    svm.fit(X, y);
    expect(() => svm.score(X, tensor([[0, 1]]))).toThrow(/1-dimensional/i);
  });

  it("score rejects non-finite y", () => {
    const svm = new LinearSVC({ maxIter: 100 });
    svm.fit(X, y);
    expect(() => svm.score(X, tensor([0, NaN, 0, 0, 1, 1, 1, 1]))).toThrow(/non-finite/i);
  });
});

// ---- LinearSVR extended branch coverage ----

describe("LinearSVR extended branches", () => {
  const X = tensor([[1], [2], [3], [4], [5]]);
  const y = tensor([1.5, 2.5, 3.5, 4.5, 5.5]);

  it("fit and predict", () => {
    const svr = new LinearSVR({ C: 1.0, maxIter: 500 });
    svr.fit(X, y);
    const preds = svr.predict(X);
    expect(preds.shape).toEqual([5]);
  });

  it("score returns R2", () => {
    const svr = new LinearSVR({ C: 10.0, maxIter: 2000 });
    svr.fit(X, y);
    const r2 = svr.score(X, y);
    expect(r2).toBeGreaterThan(-1);
    expect(r2).toBeLessThanOrEqual(1);
  });

  it("throws when predicting before fit", () => {
    const svr = new LinearSVR();
    expect(() => svr.predict(tensor([[1]]))).toThrow(/fitted/i);
  });

  it("constructor rejects invalid C", () => {
    expect(() => new LinearSVR({ C: 0 })).toThrow();
    expect(() => new LinearSVR({ C: -1 })).toThrow();
  });

  it("constructor rejects invalid epsilon", () => {
    expect(() => new LinearSVR({ epsilon: -1 })).toThrow();
    expect(() => new LinearSVR({ epsilon: NaN })).toThrow();
  });

  it("constructor rejects invalid maxIter", () => {
    expect(() => new LinearSVR({ maxIter: 0 })).toThrow();
    expect(() => new LinearSVR({ maxIter: 1.5 })).toThrow();
  });

  it("constructor rejects invalid tol", () => {
    expect(() => new LinearSVR({ tol: -1 })).toThrow();
  });

  it("getParams returns correct values", () => {
    const svr = new LinearSVR({ C: 2.0, epsilon: 0.2, maxIter: 500, tol: 1e-3 });
    const params = svr.getParams();
    expect(params.C).toBe(2.0);
    expect(params.epsilon).toBe(0.2);
    expect(params.maxIter).toBe(500);
    expect(params.tol).toBe(1e-3);
  });

  it("setParams updates C", () => {
    const svr = new LinearSVR();
    svr.setParams({ C: 5.0 });
    expect(svr.getParams().C).toBe(5.0);
  });

  it("setParams updates epsilon", () => {
    const svr = new LinearSVR();
    svr.setParams({ epsilon: 0.5 });
    expect(svr.getParams().epsilon).toBe(0.5);
  });

  it("setParams updates maxIter", () => {
    const svr = new LinearSVR();
    svr.setParams({ maxIter: 2000 });
    expect(svr.getParams().maxIter).toBe(2000);
  });

  it("setParams updates tol", () => {
    const svr = new LinearSVR();
    svr.setParams({ tol: 0.01 });
    expect(svr.getParams().tol).toBe(0.01);
  });

  it("setParams rejects invalid C", () => {
    const svr = new LinearSVR();
    expect(() => svr.setParams({ C: 0 })).toThrow();
  });

  it("setParams rejects invalid epsilon", () => {
    const svr = new LinearSVR();
    expect(() => svr.setParams({ epsilon: -1 })).toThrow();
  });

  it("setParams rejects invalid maxIter", () => {
    const svr = new LinearSVR();
    expect(() => svr.setParams({ maxIter: 0 })).toThrow();
  });

  it("setParams rejects invalid tol", () => {
    const svr = new LinearSVR();
    expect(() => svr.setParams({ tol: -1 })).toThrow();
  });

  it("setParams rejects unknown param", () => {
    const svr = new LinearSVR();
    expect(() => svr.setParams({ unknown: 42 })).toThrow(/unknown/i);
  });

  it("score rejects non-1D y", () => {
    const svr = new LinearSVR({ maxIter: 100 });
    svr.fit(X, y);
    expect(() => svr.score(X, tensor([[1, 2]]))).toThrow(/1-dimensional/i);
  });

  it("score rejects non-finite y", () => {
    const svr = new LinearSVR({ maxIter: 100 });
    svr.fit(X, y);
    expect(() => svr.score(X, tensor([1, NaN, 3, 4, 5]))).toThrow(/non-finite/i);
  });
});

// ---- SVC (Kernel SVM) extended branch coverage ----

describe("SVC (KernelSVM) extended branches", () => {
  const X = tensor([
    [0, 0],
    [1, 1],
    [0, 1],
    [1, 0],
    [5, 5],
    [6, 6],
    [5, 6],
    [6, 5],
  ]);
  const y = tensor([0, 0, 0, 0, 1, 1, 1, 1]);

  it("binary rbf classification", () => {
    setSeed(42);
    const svc = new SVC({ kernel: "rbf", C: 10, maxIter: 200 });
    svc.fit(X, y);
    const preds = svc.predict(X);
    expect(preds.shape).toEqual([8]);
  });

  it("linear kernel", () => {
    setSeed(42);
    const svc = new SVC({ kernel: "linear", C: 10, maxIter: 200 });
    svc.fit(X, y);
    const preds = svc.predict(X);
    expect(preds.shape).toEqual([8]);
  });

  it("poly kernel", () => {
    setSeed(42);
    const svc = new SVC({ kernel: "poly", C: 10, maxIter: 200, degree: 2, gamma: 1.0, coef0: 1 });
    svc.fit(X, y);
    const preds = svc.predict(X);
    expect(preds.shape).toEqual([8]);
  });

  it("sigmoid kernel", () => {
    setSeed(42);
    const svc = new SVC({ kernel: "sigmoid", C: 10, maxIter: 200, gamma: 0.01 });
    svc.fit(X, y);
    const preds = svc.predict(X);
    expect(preds.shape).toEqual([8]);
  });

  it("gamma=auto", () => {
    setSeed(42);
    const svc = new SVC({ gamma: "auto", maxIter: 100 });
    svc.fit(X, y);
    const preds = svc.predict(X);
    expect(preds.shape).toEqual([8]);
  });

  it("gamma as number", () => {
    setSeed(42);
    const svc = new SVC({ gamma: 0.5, maxIter: 100 });
    svc.fit(X, y);
    const preds = svc.predict(X);
    expect(preds.shape).toEqual([8]);
  });

  it("predictProba binary", () => {
    setSeed(42);
    const svc = new SVC({ C: 10, maxIter: 200 });
    svc.fit(X, y);
    const proba = svc.predictProba(X);
    expect(proba.shape[1]).toBe(2);
    const data = proba.toArray() as number[][];
    for (const row of data) {
      const sum = (row as number[]).reduce((a: number, b: number) => a + b, 0);
      expect(sum).toBeCloseTo(1.0, 5);
    }
  });

  it("multiclass OvR", () => {
    setSeed(42);
    const Xm = tensor([
      [0, 0],
      [0, 1],
      [5, 5],
      [5, 6],
      [10, 0],
      [10, 1],
    ]);
    const ym = tensor([0, 0, 1, 1, 2, 2]);
    const svc = new SVC({ C: 10, maxIter: 200 });
    svc.fit(Xm, ym);
    const preds = svc.predict(Xm);
    expect(preds.shape).toEqual([6]);
  });

  it("multiclass predictProba", () => {
    setSeed(42);
    const Xm = tensor([
      [0, 0],
      [0, 1],
      [5, 5],
      [5, 6],
      [10, 0],
      [10, 1],
    ]);
    const ym = tensor([0, 0, 1, 1, 2, 2]);
    const svc = new SVC({ C: 10, maxIter: 200 });
    svc.fit(Xm, ym);
    const proba = svc.predictProba(Xm);
    expect(proba.shape[1]).toBe(3);
  });

  it("score", () => {
    setSeed(42);
    const svc = new SVC({ C: 10, maxIter: 200 });
    svc.fit(X, y);
    const score = svc.score(X, y);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("classes accessor", () => {
    const svc = new SVC();
    expect(svc.classes).toBeUndefined();
    setSeed(42);
    svc.fit(X, y);
    expect(svc.classes?.toArray()).toEqual([0, 1]);
  });

  it("throws before fit", () => {
    const svc = new SVC();
    expect(() => svc.predict(tensor([[1, 2]]))).toThrow(/fitted/i);
    expect(() => svc.predictProba(tensor([[1, 2]]))).toThrow(/fitted/i);
  });

  it("rejects single class", () => {
    const svc = new SVC();
    expect(() =>
      svc.fit(
        tensor([
          [1, 2],
          [3, 4],
        ]),
        tensor([0, 0])
      )
    ).toThrow(/2 classes/i);
  });

  it("constructor rejects invalid C", () => {
    expect(() => new SVC({ C: 0 })).toThrow();
    expect(() => new SVC({ C: -1 })).toThrow();
  });

  it("constructor rejects invalid maxIter", () => {
    expect(() => new SVC({ maxIter: 0 })).toThrow();
    expect(() => new SVC({ maxIter: 1.5 })).toThrow();
  });

  it("getParams returns correct values", () => {
    const svc = new SVC({
      C: 2.0,
      kernel: "poly",
      gamma: 0.5,
      coef0: 1,
      degree: 2,
      maxIter: 500,
      tol: 0.01,
    });
    const params = svc.getParams();
    expect(params.C).toBe(2.0);
    expect(params.kernel).toBe("poly");
    expect(params.gamma).toBe(0.5);
    expect(params.coef0).toBe(1);
    expect(params.degree).toBe(2);
    expect(params.maxIter).toBe(500);
    expect(params.tol).toBe(0.01);
  });

  it("setParams updates C", () => {
    const svc = new SVC();
    svc.setParams({ C: 5.0 });
    expect(svc.getParams().C).toBe(5.0);
  });

  it("setParams updates kernel", () => {
    const svc = new SVC();
    svc.setParams({ kernel: "linear" });
    expect(svc.getParams().kernel).toBe("linear");
  });

  it("setParams updates gamma", () => {
    const svc = new SVC();
    svc.setParams({ gamma: "auto" });
    expect(svc.getParams().gamma).toBe("auto");
    svc.setParams({ gamma: 0.5 });
    expect(svc.getParams().gamma).toBe(0.5);
  });

  it("setParams updates coef0", () => {
    const svc = new SVC();
    svc.setParams({ coef0: 2 });
    expect(svc.getParams().coef0).toBe(2);
  });

  it("setParams updates degree", () => {
    const svc = new SVC();
    svc.setParams({ degree: 4 });
    expect(svc.getParams().degree).toBe(4);
  });

  it("setParams updates maxIter", () => {
    const svc = new SVC();
    svc.setParams({ maxIter: 2000 });
    expect(svc.getParams().maxIter).toBe(2000);
  });

  it("setParams updates tol", () => {
    const svc = new SVC();
    svc.setParams({ tol: 0.01 });
    expect(svc.getParams().tol).toBe(0.01);
  });

  it("setParams rejects invalid C", () => {
    const svc = new SVC();
    expect(() => svc.setParams({ C: 0 })).toThrow();
  });

  it("setParams rejects invalid kernel", () => {
    const svc = new SVC();
    expect(() => svc.setParams({ kernel: "bad" })).toThrow();
  });

  it("setParams rejects invalid gamma", () => {
    const svc = new SVC();
    expect(() => svc.setParams({ gamma: -1 })).toThrow();
    expect(() => svc.setParams({ gamma: "bad" })).toThrow();
  });

  it("setParams rejects invalid coef0", () => {
    const svc = new SVC();
    expect(() => svc.setParams({ coef0: "bad" as unknown as number })).toThrow();
  });

  it("setParams rejects invalid degree", () => {
    const svc = new SVC();
    expect(() => svc.setParams({ degree: 0 })).toThrow();
    expect(() => svc.setParams({ degree: 1.5 })).toThrow();
  });

  it("setParams rejects invalid maxIter", () => {
    const svc = new SVC();
    expect(() => svc.setParams({ maxIter: 0 })).toThrow();
  });

  it("setParams rejects invalid tol", () => {
    const svc = new SVC();
    expect(() => svc.setParams({ tol: -1 })).toThrow();
  });

  it("setParams rejects unknown param", () => {
    const svc = new SVC();
    expect(() => svc.setParams({ unknown: 42 })).toThrow(/unknown/i);
  });

  it("score rejects non-1D y", () => {
    setSeed(42);
    const svc = new SVC({ maxIter: 100 });
    svc.fit(X, y);
    expect(() => svc.score(X, tensor([[0, 1]]))).toThrow(/1-dimensional/i);
  });

  it("score rejects non-finite y", () => {
    setSeed(42);
    const svc = new SVC({ maxIter: 100 });
    svc.fit(X, y);
    expect(() => svc.score(X, tensor([0, NaN, 0, 0, 1, 1, 1, 1]))).toThrow(/non-finite/i);
  });
});

// ---- SVR (Kernel SVR) extended branch coverage ----

describe("SVR (KernelSVM) extended branches", () => {
  const X = tensor([[1], [2], [3], [4], [5]]);
  const y = tensor([1.2, 2.1, 2.9, 4.0, 5.1]);

  it("fit and predict", () => {
    setSeed(42);
    const svr = new SVR({ kernel: "rbf", C: 10, maxIter: 200 });
    svr.fit(X, y);
    const preds = svr.predict(X);
    expect(preds.shape).toEqual([5]);
  });

  it("linear kernel", () => {
    setSeed(42);
    const svr = new SVR({ kernel: "linear", C: 10, maxIter: 200 });
    svr.fit(X, y);
    const preds = svr.predict(X);
    expect(preds.shape).toEqual([5]);
  });

  it("poly kernel", () => {
    setSeed(42);
    const svr = new SVR({ kernel: "poly", C: 10, maxIter: 200, degree: 2, gamma: 1.0 });
    svr.fit(X, y);
    const preds = svr.predict(X);
    expect(preds.shape).toEqual([5]);
  });

  it("gamma=auto", () => {
    setSeed(42);
    const svr = new SVR({ gamma: "auto", maxIter: 100 });
    svr.fit(X, y);
    const preds = svr.predict(X);
    expect(preds.shape).toEqual([5]);
  });

  it("score returns R2", () => {
    setSeed(42);
    const svr = new SVR({ C: 100, maxIter: 500 });
    svr.fit(X, y);
    const r2 = svr.score(X, y);
    expect(r2).toBeLessThanOrEqual(1);
  });

  it("throws before fit", () => {
    const svr = new SVR();
    expect(() => svr.predict(tensor([[1]]))).toThrow(/fitted/i);
  });

  it("constructor rejects invalid C", () => {
    expect(() => new SVR({ C: 0 })).toThrow();
    expect(() => new SVR({ C: -1 })).toThrow();
  });

  it("constructor rejects invalid epsilon", () => {
    expect(() => new SVR({ epsilon: -1 })).toThrow();
  });

  it("getParams and setParams", () => {
    const svr = new SVR({ C: 2.0, epsilon: 0.2 });
    expect(svr.getParams().C).toBe(2.0);
    expect(svr.getParams().epsilon).toBe(0.2);
    svr.setParams({ C: 5.0 });
    expect(svr.getParams().C).toBe(5.0);
  });

  it("setParams rejects unknown", () => {
    const svr = new SVR();
    expect(() => svr.setParams({ unknown: 42 })).toThrow(/unknown/i);
  });

  it("score rejects non-1D y", () => {
    setSeed(42);
    const svr = new SVR({ maxIter: 100 });
    svr.fit(X, y);
    expect(() => svr.score(X, tensor([[1, 2]]))).toThrow(/1-dimensional/i);
  });
});
