import { describe, expect, it } from "vitest";
import { SVC, SVR } from "../src/ml/svm/KernelSVM";
import { LinearSVC, LinearSVR } from "../src/ml/svm/SVM";
import { tensor } from "../src/ndarray";

describe("SVC (KernelSVM)", () => {
  const X = tensor([
    [0, 0],
    [1, 1],
    [2, 2],
    [3, 3],
  ]);
  const y = tensor([0, 0, 1, 1]);

  it("fits and predicts with rbf kernel", () => {
    const svc = new SVC({ kernel: "rbf", C: 10, maxIter: 200 });
    svc.fit(X, y);
    const preds = svc.predict(X);
    expect(preds.shape).toEqual([4]);
  });

  it("fits and predicts with linear kernel", () => {
    const svc = new SVC({ kernel: "linear", C: 10, maxIter: 200 });
    svc.fit(X, y);
    const preds = svc.predict(X);
    expect(preds.shape).toEqual([4]);
  });

  it("fits and predicts with poly kernel", () => {
    const svc = new SVC({ kernel: "poly", degree: 2, C: 10, maxIter: 200 });
    svc.fit(X, y);
    const preds = svc.predict(X);
    expect(preds.shape).toEqual([4]);
  });

  it("fits and predicts with sigmoid kernel", () => {
    const svc = new SVC({ kernel: "sigmoid", C: 10, maxIter: 200 });
    svc.fit(X, y);
    const preds = svc.predict(X);
    expect(preds.shape).toEqual([4]);
  });

  it("predictProba returns probabilities", () => {
    const svc = new SVC({ kernel: "rbf", maxIter: 100 });
    svc.fit(X, y);
    const proba = svc.predictProba(X);
    expect(proba.shape).toEqual([4, 2]);
  });

  it("score computes accuracy", () => {
    const svc = new SVC({ kernel: "rbf", C: 10, maxIter: 200 });
    svc.fit(X, y);
    const score = svc.score(X, y);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("classes getter", () => {
    const svc = new SVC();
    expect(svc.classes).toBeUndefined();
    svc.fit(X, y);
    expect(svc.classes?.toArray()).toEqual([0, 1]);
  });

  it("handles multiclass OvR", () => {
    const Xm = tensor([
      [0, 0],
      [1, 1],
      [2, 0],
      [0, 2],
      [1, 2],
      [2, 1],
    ]);
    const ym = tensor([0, 1, 2, 0, 1, 2]);
    const svc = new SVC({ kernel: "rbf", maxIter: 100 });
    svc.fit(Xm, ym);
    const preds = svc.predict(Xm);
    expect(preds.shape).toEqual([6]);
    const proba = svc.predictProba(Xm);
    expect(proba.shape).toEqual([6, 3]);
  });

  it("gamma auto and numeric", () => {
    const svc1 = new SVC({ gamma: "auto", maxIter: 50 });
    svc1.fit(X, y);
    expect(svc1.predict(X).shape).toEqual([4]);

    const svc2 = new SVC({ gamma: 0.5, maxIter: 50 });
    svc2.fit(X, y);
    expect(svc2.predict(X).shape).toEqual([4]);
  });

  it("throws before fit", () => {
    const svc = new SVC();
    expect(() => svc.predict(tensor([[1, 2]]))).toThrow(/fitted/i);
    expect(() => svc.predictProba(tensor([[1, 2]]))).toThrow(/fitted/i);
  });

  it("validates feature count", () => {
    const svc = new SVC({ maxIter: 50 });
    svc.fit(X, y);
    expect(() => svc.predict(tensor([[1, 2, 3]]))).toThrow(/features/i);
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
    ).toThrow(/at least 2 classes/i);
  });

  it("validates constructor params", () => {
    expect(() => new SVC({ C: 0 })).toThrow(/C must be positive/i);
    expect(() => new SVC({ C: -1 })).toThrow(/C must be positive/i);
    expect(() => new SVC({ maxIter: 0 })).toThrow(/maxIter/i);
  });

  it("getParams returns all options", () => {
    const svc = new SVC({
      C: 2,
      kernel: "poly",
      gamma: 0.5,
      coef0: 1,
      degree: 2,
      maxIter: 500,
      tol: 0.01,
    });
    const p = svc.getParams();
    expect(p.C).toBe(2);
    expect(p.kernel).toBe("poly");
    expect(p.gamma).toBe(0.5);
    expect(p.coef0).toBe(1);
    expect(p.degree).toBe(2);
    expect(p.maxIter).toBe(500);
    expect(p.tol).toBe(0.01);
  });

  describe("setParams", () => {
    it("sets valid params", () => {
      const svc = new SVC();
      svc.setParams({
        C: 5,
        kernel: "poly",
        gamma: 0.1,
        coef0: 2,
        degree: 4,
        maxIter: 2000,
        tol: 0.001,
      });
      const p = svc.getParams();
      expect(p.C).toBe(5);
      expect(p.kernel).toBe("poly");
      expect(p.gamma).toBe(0.1);
      expect(p.coef0).toBe(2);
      expect(p.degree).toBe(4);
      expect(p.maxIter).toBe(2000);
      expect(p.tol).toBe(0.001);
    });

    it("accepts gamma scale/auto", () => {
      const svc = new SVC();
      svc.setParams({ gamma: "scale" });
      expect(svc.getParams().gamma).toBe("scale");
      svc.setParams({ gamma: "auto" });
      expect(svc.getParams().gamma).toBe("auto");
    });

    it("accepts all kernels", () => {
      const svc = new SVC();
      for (const k of ["rbf", "linear", "poly", "sigmoid"] as const) {
        svc.setParams({ kernel: k });
        expect(svc.getParams().kernel).toBe(k);
      }
    });

    it("rejects invalid C", () => {
      const svc = new SVC();
      expect(() => svc.setParams({ C: 0 })).toThrow(/C/);
      expect(() => svc.setParams({ C: -1 })).toThrow(/C/);
      expect(() => svc.setParams({ C: "1" })).toThrow(/C/);
    });

    it("rejects invalid kernel", () => {
      const svc = new SVC();
      expect(() => svc.setParams({ kernel: "bad" })).toThrow(/kernel/);
    });

    it("rejects invalid gamma", () => {
      const svc = new SVC();
      expect(() => svc.setParams({ gamma: 0 })).toThrow(/gamma/);
      expect(() => svc.setParams({ gamma: -1 })).toThrow(/gamma/);
      expect(() => svc.setParams({ gamma: "bad" })).toThrow(/gamma/);
    });

    it("rejects invalid coef0", () => {
      const svc = new SVC();
      expect(() => svc.setParams({ coef0: "0" })).toThrow(/coef0/);
    });

    it("rejects invalid degree", () => {
      const svc = new SVC();
      expect(() => svc.setParams({ degree: 0 })).toThrow(/degree/);
      expect(() => svc.setParams({ degree: 1.5 })).toThrow(/degree/);
    });

    it("rejects invalid maxIter", () => {
      const svc = new SVC();
      expect(() => svc.setParams({ maxIter: 0 })).toThrow(/maxIter/);
      expect(() => svc.setParams({ maxIter: 1.5 })).toThrow(/maxIter/);
    });

    it("rejects invalid tol", () => {
      const svc = new SVC();
      expect(() => svc.setParams({ tol: -1 })).toThrow(/tol/);
      expect(() => svc.setParams({ tol: "0" })).toThrow(/tol/);
    });

    it("rejects unknown params", () => {
      const svc = new SVC();
      expect(() => svc.setParams({ foo: 1 })).toThrow(/Unknown parameter/);
    });
  });
});

describe("SVR (KernelSVM)", () => {
  const X = tensor([[1], [2], [3], [4], [5]]);
  const y = tensor([1.0, 2.0, 3.0, 4.0, 5.0]);

  it("fits and predicts with rbf kernel", () => {
    const svr = new SVR({ kernel: "rbf", C: 10, maxIter: 200 });
    svr.fit(X, y);
    const preds = svr.predict(X);
    expect(preds.shape).toEqual([5]);
  });

  it("fits with linear kernel", () => {
    const svr = new SVR({ kernel: "linear", C: 10, maxIter: 200 });
    svr.fit(X, y);
    const preds = svr.predict(X);
    expect(preds.shape).toEqual([5]);
  });

  it("fits with poly kernel", () => {
    const svr = new SVR({ kernel: "poly", degree: 2, C: 10, maxIter: 200 });
    svr.fit(X, y);
    expect(svr.predict(X).shape).toEqual([5]);
  });

  it("fits with sigmoid kernel", () => {
    const svr = new SVR({ kernel: "sigmoid", C: 10, maxIter: 200 });
    svr.fit(X, y);
    expect(svr.predict(X).shape).toEqual([5]);
  });

  it("score computes R²", () => {
    const svr = new SVR({ kernel: "rbf", C: 100, maxIter: 500 });
    svr.fit(X, y);
    const score = svr.score(X, y);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("gamma auto and numeric", () => {
    const svr1 = new SVR({ gamma: "auto", maxIter: 50 });
    svr1.fit(X, y);
    expect(svr1.predict(X).shape).toEqual([5]);

    const svr2 = new SVR({ gamma: 0.5, maxIter: 50 });
    svr2.fit(X, y);
    expect(svr2.predict(X).shape).toEqual([5]);
  });

  it("throws before fit", () => {
    const svr = new SVR();
    expect(() => svr.predict(tensor([[1]]))).toThrow(/fitted/i);
  });

  it("validates feature count", () => {
    const svr = new SVR({ maxIter: 50 });
    svr.fit(X, y);
    expect(() => svr.predict(tensor([[1, 2]]))).toThrow(/features/i);
  });

  it("validates constructor params", () => {
    expect(() => new SVR({ C: 0 })).toThrow(/C must be positive/i);
    expect(() => new SVR({ epsilon: -1 })).toThrow(/epsilon/i);
  });

  it("getParams returns all options", () => {
    const svr = new SVR({ C: 2, kernel: "poly", epsilon: 0.5 });
    const p = svr.getParams();
    expect(p.C).toBe(2);
    expect(p.kernel).toBe("poly");
    expect(p.epsilon).toBe(0.5);
  });

  describe("setParams", () => {
    it("sets valid params including epsilon", () => {
      const svr = new SVR();
      svr.setParams({
        C: 5,
        kernel: "linear",
        epsilon: 0.2,
        gamma: "auto",
        coef0: 1,
        degree: 2,
        maxIter: 500,
        tol: 0.01,
      });
      const p = svr.getParams();
      expect(p.C).toBe(5);
      expect(p.kernel).toBe("linear");
      expect(p.epsilon).toBe(0.2);
      expect(p.gamma).toBe("auto");
      expect(p.coef0).toBe(1);
      expect(p.degree).toBe(2);
      expect(p.maxIter).toBe(500);
      expect(p.tol).toBe(0.01);
    });

    it("rejects invalid epsilon", () => {
      const svr = new SVR();
      expect(() => svr.setParams({ epsilon: -1 })).toThrow(/epsilon/);
      expect(() => svr.setParams({ epsilon: "0.1" })).toThrow(/epsilon/);
    });

    it("rejects unknown params", () => {
      const svr = new SVR();
      expect(() => svr.setParams({ foo: 1 })).toThrow(/Unknown parameter/);
    });
  });
});

describe("LinearSVC setParams", () => {
  it("sets valid params", () => {
    const svc = new LinearSVC();
    svc.setParams({ C: 5, maxIter: 500, tol: 0.01 });
    const p = svc.getParams();
    expect(p.C).toBe(5);
    expect(p.maxIter).toBe(500);
    expect(p.tol).toBe(0.01);
  });

  it("rejects invalid C", () => {
    const svc = new LinearSVC();
    expect(() => svc.setParams({ C: 0 })).toThrow(/C/);
  });

  it("rejects unknown params", () => {
    const svc = new LinearSVC();
    expect(() => svc.setParams({ foo: 1 })).toThrow(/Unknown parameter/);
  });

  it("handles multiclass OvR", () => {
    const Xm = tensor([
      [0, 0],
      [1, 1],
      [2, 0],
      [0, 2],
      [1, 2],
      [2, 1],
    ]);
    const ym = tensor([0, 1, 2, 0, 1, 2]);
    const svc = new LinearSVC({ maxIter: 200 });
    svc.fit(Xm, ym);
    const preds = svc.predict(Xm);
    expect(preds.shape).toEqual([6]);
    const proba = svc.predictProba(Xm);
    expect(proba.shape).toEqual([6, 3]);
  });
});

describe("LinearSVR setParams", () => {
  it("sets valid params", () => {
    const svr = new LinearSVR();
    svr.setParams({ C: 5, epsilon: 0.2, maxIter: 500, tol: 0.01 });
    const p = svr.getParams();
    expect(p.C).toBe(5);
    expect(p.epsilon).toBe(0.2);
    expect(p.maxIter).toBe(500);
    expect(p.tol).toBe(0.01);
  });

  it("rejects unknown params", () => {
    const svr = new LinearSVR();
    expect(() => svr.setParams({ foo: 1 })).toThrow(/Unknown parameter/);
  });

  it("score with varying data", () => {
    const X = tensor([[1], [2], [3], [4], [5]]);
    const y = tensor([1.0, 2.0, 3.0, 4.0, 5.0]);
    const svr = new LinearSVR({ maxIter: 200 });
    svr.fit(X, y);
    const score = svr.score(X, y);
    expect(score).toBeLessThanOrEqual(1);
  });
});
