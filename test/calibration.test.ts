import { describe, expect, it } from "vitest";
import { CalibratedClassifierCV, calibrationCurve, LogisticRegression } from "../src/ml";
import { tensor } from "../src/ndarray";

describe("CalibratedClassifierCV", () => {
  // Simple binary classification data
  const X = tensor([
    [1, 0],
    [2, 0],
    [3, 0],
    [4, 0],
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 4],
  ]);
  const y = tensor([0, 0, 0, 0, 1, 1, 1, 1]);

  it("fit and predict with sigmoid method", () => {
    const base = new LogisticRegression({ maxIter: 200 });
    const cal = new CalibratedClassifierCV({ estimator: base, method: "sigmoid" });
    cal.fit(X, y);
    const pred = cal.predict(X);
    expect(pred.size).toBe(8);
  });

  it("predictProba returns valid probabilities", () => {
    const base = new LogisticRegression({ maxIter: 200 });
    const cal = new CalibratedClassifierCV({ estimator: base, method: "sigmoid" });
    cal.fit(X, y);
    const proba = cal.predictProba(X);
    expect(proba.shape[0]).toBe(8);
    expect(proba.shape[1]).toBe(2);

    // Probabilities should sum to ~1
    for (let i = 0; i < 8; i++) {
      let sum = 0;
      for (let c = 0; c < 2; c++) {
        const p = Number(proba.data[proba.offset + i * 2 + c]);
        expect(p).toBeGreaterThanOrEqual(0);
        expect(p).toBeLessThanOrEqual(1);
        sum += p;
      }
      expect(sum).toBeCloseTo(1.0, 1);
    }
  });

  it("fit and predict with isotonic method", () => {
    const base = new LogisticRegression({ maxIter: 200 });
    const cal = new CalibratedClassifierCV({ estimator: base, method: "isotonic" });
    cal.fit(X, y);
    const pred = cal.predict(X);
    expect(pred.size).toBe(8);
  });

  it("score returns accuracy", () => {
    const base = new LogisticRegression({ maxIter: 200 });
    const cal = new CalibratedClassifierCV({ estimator: base, method: "sigmoid" });
    cal.fit(X, y);
    const acc = cal.score(X, y);
    expect(acc).toBeGreaterThan(0.5);
  });

  it("throws when not fitted", () => {
    const base = new LogisticRegression();
    const cal = new CalibratedClassifierCV({ estimator: base });
    expect(() => cal.predictProba(X)).toThrow();
  });

  it("throws for invalid cv", () => {
    const base = new LogisticRegression();
    expect(() => new CalibratedClassifierCV({ estimator: base, cv: 1 })).toThrow();
  });

  it("getParams returns options", () => {
    const base = new LogisticRegression();
    const cal = new CalibratedClassifierCV({ estimator: base, method: "isotonic", cv: 3 });
    const params = cal.getParams();
    expect(params.method).toBe("isotonic");
    expect(params.cv).toBe(3);
  });
});

describe("calibrationCurve", () => {
  it("returns correct bins for uniform strategy", () => {
    const yTrue = tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 1]);
    const yProb = tensor([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]);
    const { meanPredicted, fractionPositives } = calibrationCurve(yTrue, yProb, { nBins: 5 });

    expect(meanPredicted.length).toBeGreaterThan(0);
    expect(fractionPositives.length).toBe(meanPredicted.length);

    // Mean predicted should be within [0, 1]
    for (const mp of meanPredicted) {
      expect(mp).toBeGreaterThanOrEqual(0);
      expect(mp).toBeLessThanOrEqual(1);
    }

    // Fraction of positives should be within [0, 1]
    for (const fp of fractionPositives) {
      expect(fp).toBeGreaterThanOrEqual(0);
      expect(fp).toBeLessThanOrEqual(1);
    }
  });

  it("quantile strategy works", () => {
    const yTrue = tensor([0, 0, 1, 1, 1]);
    const yProb = tensor([0.1, 0.3, 0.6, 0.8, 0.9]);
    const { meanPredicted, fractionPositives } = calibrationCurve(yTrue, yProb, {
      nBins: 3,
      strategy: "quantile",
    });
    expect(meanPredicted.length).toBeGreaterThan(0);
    expect(fractionPositives.length).toBe(meanPredicted.length);
  });

  it("perfectly calibrated model", () => {
    // All predictions are exactly right
    const yTrue = tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);
    const yProb = tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    const { meanPredicted, fractionPositives } = calibrationCurve(yTrue, yProb, { nBins: 2 });
    expect(meanPredicted.length).toBe(2);
    // First bin: all 0s predicted, all actually 0
    expect(fractionPositives[0]).toBeCloseTo(0, 1);
    // Last bin: all 1s predicted, all actually 1
    expect(fractionPositives[fractionPositives.length - 1]).toBeCloseTo(1, 1);
  });

  it("uses cross-validation (custom cv) and stays well-calibrated", () => {
    const rows: number[][] = [];
    const labels: number[] = [];
    for (let i = 0; i < 20; i++) {
      rows.push([i % 10, 0]);
      labels.push(0);
      rows.push([0, i % 10]);
      labels.push(1);
    }
    const Xb = tensor(rows);
    const yb = tensor(labels);
    const cal = new CalibratedClassifierCV({
      estimator: new LogisticRegression({ maxIter: 200 }),
      method: "sigmoid",
      cv: 3,
    });
    cal.fit(Xb, yb);
    const proba = cal.predictProba(Xb);
    const n = proba.shape[0] ?? 0;
    const nc = proba.shape[1] ?? 0;
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let c = 0; c < nc; c++) {
        const p = Number(proba.data[proba.offset + i * nc + c]);
        expect(p).toBeGreaterThanOrEqual(0);
        expect(p).toBeLessThanOrEqual(1);
        sum += p;
      }
      expect(sum).toBeCloseTo(1, 5);
    }
    expect(cal.score(Xb, yb)).toBeGreaterThan(0.7);
  });

  it("setParams updates method and cv, rejects unknown / invalid", () => {
    const cal = new CalibratedClassifierCV({ estimator: new LogisticRegression() });
    cal.setParams({ method: "isotonic", cv: 4 });
    expect(cal.getParams().method).toBe("isotonic");
    expect(cal.getParams().cv).toBe(4);
    expect(() => cal.setParams({ cv: 1 })).toThrow();
    expect(() => cal.setParams({ method: "bogus" })).toThrow();
    expect(() => cal.setParams({ unknown: 1 })).toThrow();
    expect(() => cal.setParams({ estimator: new LogisticRegression() })).toThrow();
  });

  it("throws for mismatched sizes", () => {
    const yTrue = tensor([0, 1, 1]);
    const yProb = tensor([0.1, 0.9]);
    expect(() => calibrationCurve(yTrue, yProb)).toThrow();
  });

  it("handles empty bins gracefully", () => {
    const yTrue = tensor([0, 1]);
    const yProb = tensor([0.1, 0.9]);
    const { meanPredicted, fractionPositives } = calibrationCurve(yTrue, yProb, { nBins: 10 });
    // Only 2 non-empty bins
    expect(meanPredicted.length).toBe(2);
    expect(fractionPositives.length).toBe(2);
  });
});
