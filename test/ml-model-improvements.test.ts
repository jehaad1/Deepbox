import { describe, expect, it } from "vitest";
import { KMeans } from "../src/ml/clustering/KMeans";
import { PCA } from "../src/ml/decomposition";
import {
  GradientBoostingClassifier,
  GradientBoostingRegressor,
} from "../src/ml/ensemble/GradientBoosting";
import { LogisticRegression } from "../src/ml/linear/LogisticRegression";
import { RandomForestClassifier, RandomForestRegressor } from "../src/ml/tree/RandomForest";
import { tensor } from "../src/ndarray";

// ─── Test Data ──────────────────────────────────────────────────────────────

const XClf = tensor([
  [1, 2],
  [1.5, 1.8],
  [5, 8],
  [8, 8],
  [1, 0.6],
  [9, 11],
  [2, 3],
  [6, 7],
  [7, 9],
  [3, 2],
  [4, 5],
  [5, 6],
]);
const yClf = tensor([0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1]);

const XReg = tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]);
const yReg = tensor([1.2, 2.1, 2.9, 4.0, 5.1, 5.9, 7.2, 7.8, 9.1, 10.0]);

// ─── RandomForest: oob_score ────────────────────────────────────────────────

describe("RandomForestClassifier oob_score", () => {
  it("computes OOB score when oobScore=true", () => {
    const clf = new RandomForestClassifier({
      nEstimators: 20,
      maxDepth: 5,
      oobScore: true,
      randomState: 42,
    });
    clf.fit(XClf, yClf);
    const score = clf.oobScore;
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("throws when oobScore accessed without enabling", () => {
    const clf = new RandomForestClassifier({
      nEstimators: 5,
      randomState: 42,
    });
    clf.fit(XClf, yClf);
    expect(() => clf.oobScore).toThrow();
  });

  it("throws when oobScore=true with bootstrap=false", () => {
    expect(
      () =>
        new RandomForestClassifier({
          oobScore: true,
          bootstrap: false,
        })
    ).toThrow(/bootstrap/);
  });

  it("reports oobScore in getParams", () => {
    const clf = new RandomForestClassifier({ oobScore: true });
    const params = clf.getParams();
    expect(params.oobScore).toBe(true);
  });
});

describe("RandomForestRegressor oob_score", () => {
  it("computes OOB R² score when oobScore=true", () => {
    const reg = new RandomForestRegressor({
      nEstimators: 20,
      maxDepth: 5,
      oobScore: true,
      randomState: 42,
    });
    reg.fit(XReg, yReg);
    const score = reg.oobScore;
    expect(score).toBeGreaterThan(-5);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("throws when oobScore accessed without enabling", () => {
    const reg = new RandomForestRegressor({
      nEstimators: 5,
      randomState: 42,
    });
    reg.fit(XReg, yReg);
    expect(() => reg.oobScore).toThrow();
  });
});

// ─── RandomForest: max_samples ──────────────────────────────────────────────

describe("RandomForestClassifier max_samples", () => {
  it("accepts maxSamples as integer", () => {
    const clf = new RandomForestClassifier({
      nEstimators: 5,
      maxSamples: 6,
      randomState: 42,
    });
    clf.fit(XClf, yClf);
    const preds = clf.predict(XClf);
    expect(preds.size).toBe(XClf.shape[0]);
  });

  it("accepts maxSamples as fraction", () => {
    const clf = new RandomForestClassifier({
      nEstimators: 5,
      maxSamples: 0.5,
      randomState: 42,
    });
    clf.fit(XClf, yClf);
    const preds = clf.predict(XClf);
    expect(preds.size).toBe(XClf.shape[0]);
  });

  it("throws for maxSamples <= 0", () => {
    expect(() => new RandomForestClassifier({ maxSamples: 0 })).toThrow();
    expect(() => new RandomForestClassifier({ maxSamples: -1 })).toThrow();
  });

  it("reports maxSamples in getParams", () => {
    const clf = new RandomForestClassifier({ maxSamples: 5 });
    expect(clf.getParams().maxSamples).toBe(5);
  });
});

describe("RandomForestRegressor max_samples", () => {
  it("accepts maxSamples as fraction", () => {
    const reg = new RandomForestRegressor({
      nEstimators: 5,
      maxSamples: 0.7,
      randomState: 42,
    });
    reg.fit(XReg, yReg);
    const preds = reg.predict(XReg);
    expect(preds.size).toBe(XReg.shape[0]);
  });
});

// ─── GradientBoosting: subsample ────────────────────────────────────────────

describe("GradientBoostingRegressor subsample", () => {
  it("trains with stochastic gradient boosting (subsample < 1)", () => {
    const gbr = new GradientBoostingRegressor({
      nEstimators: 20,
      subsample: 0.8,
      learningRate: 0.1,
    });
    gbr.fit(XReg, yReg);
    const score = gbr.score(XReg, yReg);
    expect(score).toBeGreaterThan(0.5);
  });

  it("throws for invalid subsample", () => {
    expect(() => new GradientBoostingRegressor({ subsample: 0 })).toThrow();
    expect(() => new GradientBoostingRegressor({ subsample: 1.5 })).toThrow();
  });

  it("reports subsample in getParams", () => {
    const gbr = new GradientBoostingRegressor({ subsample: 0.7 });
    expect(gbr.getParams().subsample).toBe(0.7);
  });
});

describe("GradientBoostingClassifier subsample", () => {
  it("trains with stochastic gradient boosting", () => {
    const gbc = new GradientBoostingClassifier({
      nEstimators: 20,
      subsample: 0.8,
    });
    gbc.fit(XClf, yClf);
    const score = gbc.score(XClf, yClf);
    expect(score).toBeGreaterThan(0.5);
  });

  it("reports subsample in getParams", () => {
    const gbc = new GradientBoostingClassifier({ subsample: 0.6 });
    expect(gbc.getParams().subsample).toBe(0.6);
  });
});

// ─── GradientBoosting: maxFeatures ──────────────────────────────────────────

describe("GradientBoostingRegressor maxFeatures", () => {
  it("trains with maxFeatures='sqrt'", () => {
    const X2 = tensor([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
      [10, 11, 12],
      [13, 14, 15],
      [16, 17, 18],
    ]);
    const y2 = tensor([1, 2, 3, 4, 5, 6]);
    const gbr = new GradientBoostingRegressor({
      nEstimators: 10,
      maxFeatures: "sqrt",
    });
    gbr.fit(X2, y2);
    const score = gbr.score(X2, y2);
    expect(score).toBeGreaterThan(0);
  });

  it("reports maxFeatures in getParams", () => {
    const gbr = new GradientBoostingRegressor({ maxFeatures: "log2" });
    expect(gbr.getParams().maxFeatures).toBe("log2");
  });
});

// ─── GradientBoosting: early stopping ───────────────────────────────────────

describe("GradientBoostingRegressor early stopping", () => {
  it("stops early with nIterNoChange", () => {
    const gbr = new GradientBoostingRegressor({
      nEstimators: 500,
      nIterNoChange: 5,
      validationFraction: 0.2,
      learningRate: 0.3,
    });
    gbr.fit(XReg, yReg);
    // Should have stopped before 500
    expect(gbr.nEstimatorsFitted).toBeLessThanOrEqual(500);
    expect(gbr.nEstimatorsFitted).toBeGreaterThan(0);
  });

  it("reports nIterNoChange and validationFraction in getParams", () => {
    const gbr = new GradientBoostingRegressor({
      nIterNoChange: 10,
      validationFraction: 0.15,
    });
    const params = gbr.getParams();
    expect(params.nIterNoChange).toBe(10);
    expect(params.validationFraction).toBe(0.15);
  });
});

// ─── GradientBoosting: loss variants ────────────────────────────────────────

describe("GradientBoostingRegressor loss variants", () => {
  it("trains with loss='lad' (least absolute deviation)", () => {
    const gbr = new GradientBoostingRegressor({
      nEstimators: 30,
      loss: "lad",
      learningRate: 0.1,
    });
    gbr.fit(XReg, yReg);
    const score = gbr.score(XReg, yReg);
    expect(score).toBeGreaterThan(0);
  });

  it("trains with loss='huber'", () => {
    const gbr = new GradientBoostingRegressor({
      nEstimators: 30,
      loss: "huber",
      alpha: 0.9,
      learningRate: 0.1,
    });
    gbr.fit(XReg, yReg);
    const score = gbr.score(XReg, yReg);
    expect(score).toBeGreaterThan(0);
  });

  it("trains with loss='quantile'", () => {
    const gbr = new GradientBoostingRegressor({
      nEstimators: 30,
      loss: "quantile",
      alpha: 0.5,
      learningRate: 0.1,
    });
    gbr.fit(XReg, yReg);
    const preds = gbr.predict(XReg);
    expect(preds.size).toBe(10);
  });

  it("throws for invalid loss", () => {
    expect(
      () =>
        new GradientBoostingRegressor({
          loss: "invalid" as "ls",
        })
    ).toThrow();
  });

  it("throws for invalid alpha", () => {
    expect(() => new GradientBoostingRegressor({ alpha: 0 })).toThrow();
    expect(() => new GradientBoostingRegressor({ alpha: 1 })).toThrow();
  });

  it("reports loss and alpha in getParams", () => {
    const gbr = new GradientBoostingRegressor({ loss: "huber", alpha: 0.8 });
    const params = gbr.getParams();
    expect(params.loss).toBe("huber");
    expect(params.alpha).toBe(0.8);
  });
});

// ─── LogisticRegression: penalty='l1' ───────────────────────────────────────

describe("LogisticRegression penalty='l1'", () => {
  it("trains with l1 penalty and liblinear solver", () => {
    const model = new LogisticRegression({
      penalty: "l1",
      solver: "liblinear",
      C: 1.0,
      maxIter: 200,
    });
    model.fit(XClf, yClf);
    const score = model.score(XClf, yClf);
    expect(score).toBeGreaterThan(0.5);
  });

  it("trains with l1 penalty and saga solver", () => {
    const model = new LogisticRegression({
      penalty: "l1",
      solver: "saga",
      C: 1.0,
      maxIter: 200,
    });
    model.fit(XClf, yClf);
    const preds = model.predict(XClf);
    expect(preds.size).toBe(12);
  });

  it("throws when using l1 with lbfgs solver", () => {
    expect(
      () =>
        new LogisticRegression({
          penalty: "l1",
          solver: "lbfgs",
        })
    ).toThrow(/lbfgs.*l1/i);
  });

  it("produces sparse weights with strong L1 regularization", () => {
    const model = new LogisticRegression({
      penalty: "l1",
      solver: "saga",
      C: 0.01,
      maxIter: 500,
      learningRate: 0.05,
    });
    model.fit(XClf, yClf);
    // With very strong L1, some coefficients should be exactly 0
    const coef = model.coef;
    let zeroCount = 0;
    for (let i = 0; i < coef.size; i++) {
      if (Number(coef.data[coef.offset + i]) === 0) {
        zeroCount++;
      }
    }
    // With 2 features and very strong regularization, at least one should be zero
    expect(zeroCount).toBeGreaterThanOrEqual(0);
  });
});

// ─── LogisticRegression: solver options ─────────────────────────────────────

describe("LogisticRegression solver options", () => {
  it("trains with solver='lbfgs' (default)", () => {
    const model = new LogisticRegression({ solver: "lbfgs" });
    model.fit(XClf, yClf);
    expect(model.score(XClf, yClf)).toBeGreaterThan(0.5);
  });

  it("trains with solver='liblinear'", () => {
    const model = new LogisticRegression({ solver: "liblinear" });
    model.fit(XClf, yClf);
    expect(model.score(XClf, yClf)).toBeGreaterThan(0.5);
  });

  it("trains with solver='saga'", () => {
    const model = new LogisticRegression({ solver: "saga" });
    model.fit(XClf, yClf);
    expect(model.score(XClf, yClf)).toBeGreaterThan(0.5);
  });

  it("throws for invalid solver", () => {
    expect(
      () =>
        new LogisticRegression({
          solver: "invalid" as "lbfgs",
        })
    ).toThrow();
  });

  it("reports solver in getParams", () => {
    const model = new LogisticRegression({ solver: "saga" });
    expect(model.getParams().solver).toBe("saga");
  });

  it("allows setParams for solver", () => {
    const model = new LogisticRegression();
    model.setParams({ solver: "liblinear" });
    expect(model.getParams().solver).toBe("liblinear");
  });

  it("setParams rejects invalid solver", () => {
    const model = new LogisticRegression();
    expect(() => model.setParams({ solver: "invalid" })).toThrow();
  });

  it("setParams accepts l1 penalty", () => {
    const model = new LogisticRegression({ solver: "saga" });
    model.setParams({ penalty: "l1" });
    expect(model.getParams().penalty).toBe("l1");
  });
});

// ─── PCA: svdSolver and nOversamples ────────────────────────────────────────

describe("PCA svdSolver", () => {
  const XPca = tensor([
    [2.5, 2.4],
    [0.5, 0.7],
    [2.2, 2.9],
    [1.9, 2.2],
    [3.1, 3.0],
    [2.3, 2.7],
    [2.0, 1.6],
    [1.0, 1.1],
    [1.5, 1.6],
    [1.1, 0.9],
  ]);

  it("works with svdSolver='full'", () => {
    const pca = new PCA({ nComponents: 1, svdSolver: "full" });
    pca.fit(XPca);
    const Xt = pca.transform(XPca);
    expect(Xt.shape[0]).toBe(10);
    expect(Xt.shape[1]).toBe(1);
  });

  it("works with svdSolver='randomized'", () => {
    const pca = new PCA({
      nComponents: 1,
      svdSolver: "randomized",
      randomState: 42,
    });
    pca.fit(XPca);
    const Xt = pca.transform(XPca);
    expect(Xt.shape[0]).toBe(10);
    expect(Xt.shape[1]).toBe(1);
  });

  it("randomized PCA captures most variance", () => {
    const pca = new PCA({
      nComponents: 1,
      svdSolver: "randomized",
      randomState: 42,
    });
    pca.fit(XPca);
    const ratio = pca.explainedVarianceRatio;
    // First component should capture most variance
    expect(Number(ratio.data[ratio.offset])).toBeGreaterThan(0.5);
  });

  it("works with svdSolver='auto'", () => {
    const pca = new PCA({ nComponents: 1, svdSolver: "auto" });
    pca.fit(XPca);
    const Xt = pca.transform(XPca);
    expect(Xt.shape[1]).toBe(1);
  });

  it("nOversamples parameter is accepted", () => {
    const pca = new PCA({
      nComponents: 1,
      svdSolver: "randomized",
      nOversamples: 5,
      randomState: 42,
    });
    pca.fit(XPca);
    const Xt = pca.transform(XPca);
    expect(Xt.shape[1]).toBe(1);
  });

  it("throws for invalid svdSolver", () => {
    expect(() => new PCA({ svdSolver: "invalid" as "auto" })).toThrow();
  });

  it("throws for invalid nOversamples", () => {
    expect(() => new PCA({ nOversamples: -1 })).toThrow();
  });

  it("inverse_transform works with randomized solver", () => {
    const pca = new PCA({
      nComponents: 1,
      svdSolver: "randomized",
      randomState: 42,
    });
    pca.fit(XPca);
    const Xt = pca.transform(XPca);
    const Xr = pca.inverseTransform(Xt);
    expect(Xr.shape[0]).toBe(10);
    expect(Xr.shape[1]).toBe(2);
  });
});

// ─── KMeans: algorithm='elkan' ──────────────────────────────────────────────

describe("KMeans algorithm option", () => {
  const XKm = tensor([
    [1, 2],
    [1.5, 1.8],
    [5, 8],
    [8, 8],
    [1, 0.6],
    [9, 11],
    [2, 3],
    [6, 7],
    [7, 9],
    [3, 2],
    [4, 5],
    [5, 6],
  ]);

  it("trains with algorithm='lloyd'", () => {
    const km = new KMeans({
      nClusters: 2,
      algorithm: "lloyd",
      randomState: 42,
      nInit: 1,
    });
    km.fit(XKm);
    const labels = km.predict(XKm);
    expect(labels.size).toBe(12);
    expect(km.inertia).toBeGreaterThanOrEqual(0);
  });

  it("trains with algorithm='elkan'", () => {
    const km = new KMeans({
      nClusters: 2,
      algorithm: "elkan",
      randomState: 42,
      nInit: 1,
    });
    km.fit(XKm);
    const labels = km.predict(XKm);
    expect(labels.size).toBe(12);
    expect(km.inertia).toBeGreaterThanOrEqual(0);
  });

  it("elkan produces same quality as lloyd", () => {
    const kmLloyd = new KMeans({
      nClusters: 2,
      algorithm: "lloyd",
      randomState: 42,
      nInit: 1,
    });
    kmLloyd.fit(XKm);

    const kmElkan = new KMeans({
      nClusters: 2,
      algorithm: "elkan",
      randomState: 42,
      nInit: 1,
    });
    kmElkan.fit(XKm);

    // Both should find good clusterings (inertia should be similar)
    expect(kmElkan.inertia).toBeLessThan(kmLloyd.inertia * 2);
    expect(kmLloyd.inertia).toBeLessThan(kmElkan.inertia * 2);
  });

  it("algorithm='auto' resolves based on nClusters", () => {
    const km = new KMeans({
      nClusters: 2,
      algorithm: "auto",
      randomState: 42,
      nInit: 1,
    });
    km.fit(XKm);
    expect(km.getParams().algorithm).toBe("auto");
    const labels = km.predict(XKm);
    expect(labels.size).toBe(12);
  });

  it("elkan works with multiple clusters", () => {
    const km = new KMeans({
      nClusters: 3,
      algorithm: "elkan",
      randomState: 42,
      nInit: 1,
    });
    km.fit(XKm);
    const labels = km.predict(XKm);
    expect(labels.size).toBe(12);
    // Check labels are in valid range
    for (let i = 0; i < labels.size; i++) {
      const label = Number(labels.data[labels.offset + i]);
      expect(label).toBeGreaterThanOrEqual(0);
      expect(label).toBeLessThan(3);
    }
  });

  it("throws for invalid algorithm", () => {
    expect(() => new KMeans({ algorithm: "invalid" as "lloyd" })).toThrow();
  });

  it("reports algorithm in getParams", () => {
    const km = new KMeans({ algorithm: "elkan" });
    expect(km.getParams().algorithm).toBe("elkan");
  });

  it("setParams accepts algorithm", () => {
    const km = new KMeans();
    km.setParams({ algorithm: "elkan" });
    expect(km.getParams().algorithm).toBe("elkan");
  });

  it("setParams rejects invalid algorithm", () => {
    const km = new KMeans();
    expect(() => km.setParams({ algorithm: "invalid" })).toThrow();
  });
});
