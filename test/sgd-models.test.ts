import { describe, expect, it } from "vitest";
import { SGDClassifier, SGDRegressor } from "../src/ml";
import { tensor } from "../src/ndarray";

describe("SGDClassifier", () => {
  // Linearly separable binary data
  const Xbin = tensor([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
    [2, 2],
    [2, 3],
    [3, 2],
    [3, 3],
  ]);
  const ybin = tensor([0, 0, 0, 0, 1, 1, 1, 1]);

  it("fit and predict on binary data with hinge loss", () => {
    const clf = new SGDClassifier({ loss: "hinge", randomState: 42, maxIter: 500, eta0: 0.01 });
    clf.fit(Xbin, ybin);
    const pred = clf.predict(Xbin);
    let correct = 0;
    for (let i = 0; i < ybin.size; i++) {
      if (Number(pred.data[pred.offset + i]) === Number(ybin.data[ybin.offset + i])) correct++;
    }
    expect(correct / ybin.size).toBeGreaterThanOrEqual(0.75);
  });

  it("fit and predict with log_loss", () => {
    const clf = new SGDClassifier({ loss: "log_loss", randomState: 42, maxIter: 500 });
    clf.fit(Xbin, ybin);
    const pred = clf.predict(Xbin);
    let correct = 0;
    for (let i = 0; i < ybin.size; i++) {
      if (Number(pred.data[pred.offset + i]) === Number(ybin.data[ybin.offset + i])) correct++;
    }
    expect(correct / ybin.size).toBeGreaterThanOrEqual(0.75);
  });

  it("predictProba with log_loss returns valid probabilities", () => {
    const clf = new SGDClassifier({ loss: "log_loss", randomState: 42, maxIter: 500 });
    clf.fit(Xbin, ybin);
    const proba = clf.predictProba(Xbin);
    expect(proba.shape[0]).toBe(8);
    expect(proba.shape[1]).toBe(2);
    // Probabilities should sum to ~1 per row
    for (let i = 0; i < 8; i++) {
      let sum = 0;
      for (let j = 0; j < 2; j++) {
        const p = Number(proba.data[proba.offset + i * 2 + j]);
        expect(p).toBeGreaterThanOrEqual(0);
        expect(p).toBeLessThanOrEqual(1);
        sum += p;
      }
      expect(sum).toBeCloseTo(1.0, 3);
    }
  });

  it("predictProba throws for hinge loss", () => {
    const clf = new SGDClassifier({ loss: "hinge", randomState: 42, maxIter: 200 });
    clf.fit(Xbin, ybin);
    expect(() => clf.predictProba(Xbin)).toThrow();
  });

  it("score returns accuracy", () => {
    const clf = new SGDClassifier({ loss: "log_loss", randomState: 42, maxIter: 500 });
    clf.fit(Xbin, ybin);
    const acc = clf.score(Xbin, ybin);
    expect(acc).toBeGreaterThanOrEqual(0.75);
    expect(acc).toBeLessThanOrEqual(1);
  });

  it("multiclass classification", () => {
    const Xmulti = tensor([
      [0, 0],
      [0, 1],
      [1, 0],
      [3, 3],
      [3, 4],
      [4, 3],
      [0, 4],
      [1, 4],
      [0, 3],
    ]);
    const ymulti = tensor([0, 0, 0, 1, 1, 1, 2, 2, 2]);
    const clf = new SGDClassifier({ loss: "log_loss", randomState: 42, maxIter: 1000 });
    clf.fit(Xmulti, ymulti);
    const pred = clf.predict(Xmulti);
    expect(pred.size).toBe(9);
    // Should get at least some correct
    let correct = 0;
    for (let i = 0; i < ymulti.size; i++) {
      if (Number(pred.data[pred.offset + i]) === Number(ymulti.data[ymulti.offset + i])) correct++;
    }
    expect(correct).toBeGreaterThan(3);
  });

  it("squared_hinge loss works", () => {
    const clf = new SGDClassifier({ loss: "squared_hinge", randomState: 42, maxIter: 500 });
    clf.fit(Xbin, ybin);
    const pred = clf.predict(Xbin);
    expect(pred.size).toBe(8);
  });

  it("modified_huber loss works", () => {
    const clf = new SGDClassifier({ loss: "modified_huber", randomState: 42, maxIter: 500 });
    clf.fit(Xbin, ybin);
    const pred = clf.predict(Xbin);
    expect(pred.size).toBe(8);
  });

  it("l1 penalty works", () => {
    const clf = new SGDClassifier({ penalty: "l1", randomState: 42, maxIter: 500 });
    clf.fit(Xbin, ybin);
    const pred = clf.predict(Xbin);
    expect(pred.size).toBe(8);
  });

  it("elasticnet penalty works", () => {
    const clf = new SGDClassifier({
      penalty: "elasticnet",
      l1Ratio: 0.5,
      randomState: 42,
      maxIter: 500,
    });
    clf.fit(Xbin, ybin);
    const pred = clf.predict(Xbin);
    expect(pred.size).toBe(8);
  });

  it("no penalty works", () => {
    const clf = new SGDClassifier({ penalty: "none", randomState: 42, maxIter: 500 });
    clf.fit(Xbin, ybin);
    const pred = clf.predict(Xbin);
    expect(pred.size).toBe(8);
  });

  it("coef and intercept accessible after fit", () => {
    const clf = new SGDClassifier({ randomState: 42, maxIter: 200 });
    clf.fit(Xbin, ybin);
    expect(clf.coef.length).toBe(2);
    expect(typeof clf.intercept).toBe("number");
  });

  it("classes accessible after fit", () => {
    const clf = new SGDClassifier({ randomState: 42, maxIter: 200 });
    clf.fit(Xbin, ybin);
    expect(clf.classes).toBeDefined();
    expect(clf.classes!.size).toBe(2);
  });

  it("throws when not fitted", () => {
    const clf = new SGDClassifier();
    expect(() => clf.predict(Xbin)).toThrow();
    expect(() => clf.coef).toThrow();
    expect(() => clf.intercept).toThrow();
  });

  it("throws for invalid alpha", () => {
    expect(() => new SGDClassifier({ alpha: -1 })).toThrow();
  });

  it("throws for invalid maxIter", () => {
    expect(() => new SGDClassifier({ maxIter: 0 })).toThrow();
  });

  it("getParams returns options", () => {
    const clf = new SGDClassifier({ loss: "log_loss", alpha: 0.001 });
    const params = clf.getParams();
    expect(params.loss).toBe("log_loss");
    expect(params.alpha).toBe(0.001);
  });

  it("deterministic with randomState", () => {
    const clf1 = new SGDClassifier({ randomState: 123, maxIter: 100 });
    const clf2 = new SGDClassifier({ randomState: 123, maxIter: 100 });
    clf1.fit(Xbin, ybin);
    clf2.fit(Xbin, ybin);
    const pred1 = clf1.predict(Xbin);
    const pred2 = clf2.predict(Xbin);
    for (let i = 0; i < pred1.size; i++) {
      expect(Number(pred1.data[pred1.offset + i])).toBe(Number(pred2.data[pred2.offset + i]));
    }
  });

  it("different learning rate schedules", () => {
    for (const lr of ["constant", "optimal", "invscaling"] as const) {
      const clf = new SGDClassifier({ learningRate: lr, randomState: 42, maxIter: 200 });
      clf.fit(Xbin, ybin);
      expect(clf.predict(Xbin).size).toBe(8);
    }
  });
});

describe("SGDRegressor", () => {
  // Simple linear relationship: y = 2*x1 + 3*x2 + 1
  const Xreg = tensor([
    [1, 1],
    [2, 1],
    [3, 1],
    [1, 2],
    [2, 2],
    [3, 2],
    [1, 3],
    [2, 3],
    [3, 3],
    [4, 4],
  ]);
  const yreg = tensor([6, 8, 10, 9, 11, 13, 12, 14, 16, 20]);

  it("fit and predict with squared_error loss", () => {
    const reg = new SGDRegressor({
      loss: "squared_error",
      randomState: 42,
      maxIter: 2000,
      eta0: 0.001,
    });
    reg.fit(Xreg, yreg);
    const pred = reg.predict(Xreg);
    expect(pred.size).toBe(10);
    // Should produce reasonable predictions
    const r2 = reg.score(Xreg, yreg);
    expect(r2).toBeGreaterThan(0.5);
  });

  it("huber loss works", () => {
    const reg = new SGDRegressor({ loss: "huber", randomState: 42, maxIter: 2000, eta0: 0.001 });
    reg.fit(Xreg, yreg);
    const pred = reg.predict(Xreg);
    expect(pred.size).toBe(10);
  });

  it("epsilon_insensitive loss works", () => {
    const reg = new SGDRegressor({
      loss: "epsilon_insensitive",
      randomState: 42,
      maxIter: 2000,
      eta0: 0.001,
    });
    reg.fit(Xreg, yreg);
    const pred = reg.predict(Xreg);
    expect(pred.size).toBe(10);
  });

  it("score returns R² value", () => {
    const reg = new SGDRegressor({ randomState: 42, maxIter: 2000, eta0: 0.001 });
    reg.fit(Xreg, yreg);
    const r2 = reg.score(Xreg, yreg);
    expect(r2).toBeGreaterThan(0);
    expect(r2).toBeLessThanOrEqual(1);
  });

  it("coef and intercept accessible after fit", () => {
    const reg = new SGDRegressor({ randomState: 42, maxIter: 200 });
    reg.fit(Xreg, yreg);
    expect(reg.coef.length).toBe(2);
    expect(typeof reg.intercept).toBe("number");
  });

  it("l1 penalty works", () => {
    const reg = new SGDRegressor({ penalty: "l1", randomState: 42, maxIter: 500 });
    reg.fit(Xreg, yreg);
    expect(reg.predict(Xreg).size).toBe(10);
  });

  it("elasticnet penalty works", () => {
    const reg = new SGDRegressor({
      penalty: "elasticnet",
      l1Ratio: 0.5,
      randomState: 42,
      maxIter: 500,
    });
    reg.fit(Xreg, yreg);
    expect(reg.predict(Xreg).size).toBe(10);
  });

  it("throws when not fitted", () => {
    const reg = new SGDRegressor();
    expect(() => reg.predict(Xreg)).toThrow();
    expect(() => reg.coef).toThrow();
    expect(() => reg.intercept).toThrow();
  });

  it("throws for invalid alpha", () => {
    expect(() => new SGDRegressor({ alpha: -1 })).toThrow();
  });

  it("throws for invalid maxIter", () => {
    expect(() => new SGDRegressor({ maxIter: 0 })).toThrow();
  });

  it("getParams returns options", () => {
    const reg = new SGDRegressor({ loss: "huber", alpha: 0.01, epsilon: 0.5 });
    const params = reg.getParams();
    expect(params.loss).toBe("huber");
    expect(params.alpha).toBe(0.01);
    expect(params.epsilon).toBe(0.5);
  });

  it("deterministic with randomState", () => {
    const reg1 = new SGDRegressor({ randomState: 123, maxIter: 200 });
    const reg2 = new SGDRegressor({ randomState: 123, maxIter: 200 });
    reg1.fit(Xreg, yreg);
    reg2.fit(Xreg, yreg);
    const pred1 = reg1.predict(Xreg);
    const pred2 = reg2.predict(Xreg);
    for (let i = 0; i < pred1.size; i++) {
      expect(Number(pred1.data[pred1.offset + i])).toBeCloseTo(
        Number(pred2.data[pred2.offset + i]),
        10
      );
    }
  });

  it("constant learning rate schedule", () => {
    const reg = new SGDRegressor({
      learningRate: "constant",
      eta0: 0.001,
      randomState: 42,
      maxIter: 500,
    });
    reg.fit(Xreg, yreg);
    expect(reg.predict(Xreg).size).toBe(10);
  });
});
