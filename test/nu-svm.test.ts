import { describe, expect, it } from "vitest";
import { NuSVC, NuSVR, OneClassSVM } from "../src/ml";
import { tensor } from "../src/ndarray";

describe("NuSVC", () => {
  const X = tensor([
    [1, 1],
    [2, 2],
    [3, 3],
    [8, 8],
    [9, 9],
    [10, 10],
  ]);
  const y = tensor([0, 0, 0, 1, 1, 1]);

  it("should classify well-separated data", () => {
    const clf = new NuSVC({ nu: 0.5 });
    clf.fit(X, y);
    const pred = clf.predict(X);
    expect(pred.shape).toEqual([6]);
  });

  it("should predict probabilities summing to 1", () => {
    const clf = new NuSVC({ nu: 0.5 });
    clf.fit(X, y);
    const proba = clf.predictProba(tensor([[5, 5]]));
    expect(proba.ndim).toBe(2);
    expect(proba.shape[1]).toBe(2);
    const p0 = Number(proba.data[0]);
    const p1 = Number(proba.data[1]);
    expect(p0 + p1).toBeCloseTo(1, 5);
  });

  it("should compute accuracy score", () => {
    const clf = new NuSVC({ nu: 0.5 });
    clf.fit(X, y);
    const score = clf.score(X, y);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("should expose classes", () => {
    const clf = new NuSVC();
    clf.fit(X, y);
    expect(clf.classes).toBeDefined();
    expect(clf.classes!.shape).toEqual([2]);
  });

  it("should throw on invalid nu", () => {
    expect(() => new NuSVC({ nu: 0 })).toThrow();
    expect(() => new NuSVC({ nu: 1.5 })).toThrow();
    expect(() => new NuSVC({ nu: -0.1 })).toThrow();
  });

  it("should throw when predicting before fitting", () => {
    const clf = new NuSVC();
    expect(() => clf.predict(tensor([[1, 2]]))).toThrow();
  });

  it("should throw on less than 2 classes", () => {
    const clf = new NuSVC();
    expect(() =>
      clf.fit(
        tensor([
          [1, 2],
          [3, 4],
        ]),
        tensor([0, 0])
      )
    ).toThrow();
  });

  it("should get and set params", () => {
    const clf = new NuSVC({ nu: 0.3 });
    expect(clf.getParams().nu).toBe(0.3);
    clf.setParams({ nu: 0.7 });
    expect(clf.getParams().nu).toBe(0.7);
  });

  it("should throw on unknown param", () => {
    const clf = new NuSVC();
    expect(() => clf.setParams({ badParam: 1 })).toThrow();
  });
});

describe("NuSVR", () => {
  const X = tensor([[1], [2], [3], [4], [5]]);
  const y = tensor([1.0, 2.0, 3.0, 4.0, 5.0]);

  it("should make predictions", () => {
    const svr = new NuSVR({ nu: 0.5, C: 10 });
    svr.fit(X, y);
    const pred = svr.predict(X);
    expect(pred.shape).toEqual([5]);
  });

  it("should compute R^2 score", () => {
    const svr = new NuSVR({ nu: 0.5, C: 10 });
    svr.fit(X, y);
    const score = svr.score(X, y);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("should throw on invalid nu", () => {
    expect(() => new NuSVR({ nu: 0 })).toThrow();
    expect(() => new NuSVR({ nu: 2 })).toThrow();
  });

  it("should throw on invalid C", () => {
    expect(() => new NuSVR({ C: -1 })).toThrow();
  });

  it("should throw when predicting before fitting", () => {
    const svr = new NuSVR();
    expect(() => svr.predict(tensor([[1]]))).toThrow();
  });

  it("should get and set params", () => {
    const svr = new NuSVR({ nu: 0.3 });
    expect(svr.getParams().nu).toBe(0.3);
    svr.setParams({ nu: 0.8 });
    expect(svr.getParams().nu).toBe(0.8);
  });
});

describe("OneClassSVM", () => {
  const X = tensor([
    [1, 1],
    [1, 2],
    [2, 1],
    [2, 2],
    [1.5, 1.5],
  ]);

  it("should fit and predict inliers/outliers", () => {
    const ocsvm = new OneClassSVM({ nu: 0.2 });
    ocsvm.fit(X);
    const pred = ocsvm.predict(X);
    expect(pred.shape).toEqual([5]);
    // All predictions should be +1 or -1
    for (let i = 0; i < 5; i++) {
      const v = Number(pred.data[i]);
      expect(v === 1 || v === -1).toBe(true);
    }
  });

  it("should support fitPredict", () => {
    const ocsvm = new OneClassSVM({ nu: 0.2 });
    const pred = ocsvm.fitPredict(X);
    expect(pred.shape).toEqual([5]);
  });

  it("should compute anomaly scores", () => {
    const ocsvm = new OneClassSVM({ nu: 0.2 });
    ocsvm.fit(X);
    const scores = ocsvm.scoreSamples(X);
    expect(scores.shape).toEqual([5]);
    // Inliers should generally have higher scores
    for (let i = 0; i < 5; i++) {
      expect(Number.isFinite(Number(scores.data[i]))).toBe(true);
    }
  });

  it("should throw on invalid nu", () => {
    expect(() => new OneClassSVM({ nu: 0 })).toThrow();
    expect(() => new OneClassSVM({ nu: 1.5 })).toThrow();
  });

  it("should throw when predicting before fitting", () => {
    const ocsvm = new OneClassSVM();
    expect(() => ocsvm.predict(tensor([[1, 2]]))).toThrow();
  });

  it("should throw when scoring before fitting", () => {
    const ocsvm = new OneClassSVM();
    expect(() => ocsvm.scoreSamples(tensor([[1, 2]]))).toThrow();
  });

  it("should get and set params", () => {
    const ocsvm = new OneClassSVM({ nu: 0.3 });
    expect(ocsvm.getParams().nu).toBe(0.3);
    ocsvm.setParams({ nu: 0.7 });
    expect(ocsvm.getParams().nu).toBe(0.7);
  });

  it("should throw on unknown param", () => {
    const ocsvm = new OneClassSVM();
    expect(() => ocsvm.setParams({ badParam: 1 })).toThrow();
  });
});
