import { describe, expect, it } from "vitest";
import { LogisticRegression, OneVsOneClassifier, OneVsRestClassifier } from "../src/ml";
import { tensor } from "../src/ndarray";

describe("OneVsRestClassifier", () => {
  // 3-class problem
  const X = tensor([
    [1, 0],
    [2, 0],
    [3, 0],
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 1],
    [2, 2],
    [3, 3],
  ]);
  const y = tensor([0, 0, 0, 1, 1, 1, 2, 2, 2]);

  it("fit and predict", () => {
    const ovr = new OneVsRestClassifier({
      estimator: new LogisticRegression({ maxIter: 200 }),
    });
    ovr.fit(X, y);
    const pred = ovr.predict(X);
    expect(pred.size).toBe(9);
  });

  it("score is reasonable on training data", () => {
    const ovr = new OneVsRestClassifier({
      estimator: new LogisticRegression({ maxIter: 200 }),
    });
    ovr.fit(X, y);
    const acc = ovr.score(X, y);
    expect(acc).toBeGreaterThan(0.5);
  });

  it("predictProba returns valid probabilities", () => {
    const ovr = new OneVsRestClassifier({
      estimator: new LogisticRegression({ maxIter: 200 }),
    });
    ovr.fit(X, y);
    const proba = ovr.predictProba(X);
    expect(proba.shape[0]).toBe(9);
    expect(proba.shape[1]).toBe(3);

    for (let i = 0; i < 9; i++) {
      let sum = 0;
      for (let c = 0; c < 3; c++) {
        const p = Number(proba.data[proba.offset + i * 3 + c]);
        expect(p).toBeGreaterThanOrEqual(0);
        sum += p;
      }
      expect(sum).toBeCloseTo(1.0, 1);
    }
  });

  it("classes are accessible after fit", () => {
    const ovr = new OneVsRestClassifier({
      estimator: new LogisticRegression({ maxIter: 200 }),
    });
    ovr.fit(X, y);
    const cls = ovr.classes;
    expect(cls.size).toBe(3);
  });

  it("throws when not fitted", () => {
    const ovr = new OneVsRestClassifier({
      estimator: new LogisticRegression(),
    });
    expect(() => ovr.predict(X)).toThrow();
    expect(() => ovr.predictProba(X)).toThrow();
    expect(() => ovr.classes).toThrow();
  });
});

describe("OneVsOneClassifier", () => {
  const X = tensor([
    [1, 0],
    [2, 0],
    [3, 0],
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 1],
    [2, 2],
    [3, 3],
  ]);
  const y = tensor([0, 0, 0, 1, 1, 1, 2, 2, 2]);

  it("fit and predict", () => {
    const ovo = new OneVsOneClassifier({
      estimator: new LogisticRegression({ maxIter: 200 }),
    });
    ovo.fit(X, y);
    const pred = ovo.predict(X);
    expect(pred.size).toBe(9);
  });

  it("score is reasonable on training data", () => {
    const ovo = new OneVsOneClassifier({
      estimator: new LogisticRegression({ maxIter: 200 }),
    });
    ovo.fit(X, y);
    const acc = ovo.score(X, y);
    expect(acc).toBeGreaterThan(0.5);
  });

  it("classes are accessible after fit", () => {
    const ovo = new OneVsOneClassifier({
      estimator: new LogisticRegression({ maxIter: 200 }),
    });
    ovo.fit(X, y);
    expect(ovo.classes.size).toBe(3);
  });

  it("predictProba returns probabilities", () => {
    const ovo = new OneVsOneClassifier({
      estimator: new LogisticRegression({ maxIter: 200 }),
    });
    ovo.fit(X, y);
    const proba = ovo.predictProba(X);
    expect(proba.shape[0]).toBe(X.shape[0]);
    expect(proba.shape[1]).toBe(3);
  });

  it("throws when not fitted", () => {
    const ovo = new OneVsOneClassifier({ estimator: new LogisticRegression() });
    expect(() => ovo.predict(X)).toThrow();
    expect(() => ovo.classes).toThrow();
  });
});
