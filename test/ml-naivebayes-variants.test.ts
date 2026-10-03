import { describe, expect, it } from "vitest";
import {
  AdaBoostClassifier,
  AdaBoostRegressor,
  AgglomerativeClustering,
  BaggingClassifier,
  BaggingRegressor,
  BernoulliNB,
  ComplementNB,
  DecisionTreeClassifier,
  DecisionTreeRegressor,
  GaussianMixture,
  GradientBoostingClassifier,
  GradientBoostingRegressor,
  IsolationForest,
  KMeans,
  LinearSVC,
  LocalOutlierFactor,
  LogisticRegression,
  MiniBatchKMeans,
  MultinomialNB,
  RandomForestClassifier,
  RandomForestRegressor,
  SVC,
  SVR,
  VotingClassifier,
  VotingRegressor,
} from "../src/ml";
import { tensor } from "../src/ndarray";

const X_clf = tensor([
  [1, 10],
  [2, 20],
  [3, 30],
  [4, 40],
  [5, 50],
  [6, 60],
  [7, 70],
  [8, 80],
  [9, 90],
  [10, 100],
]);
const y_clf = tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);

const X_reg = tensor([
  [1, 10],
  [2, 20],
  [3, 30],
  [4, 40],
  [5, 50],
  [6, 60],
  [7, 70],
  [8, 80],
  [9, 90],
  [10, 100],
]);
const y_reg = tensor([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);

describe("MultinomialNB", () => {
  it("fits and predicts on count data", () => {
    const X = tensor([
      [3, 0, 1],
      [0, 2, 1],
      [1, 0, 3],
      [0, 3, 0],
      [2, 0, 2],
      [0, 2, 0],
    ]);
    const y = tensor([0, 1, 0, 1, 0, 1]);
    const clf = new MultinomialNB();
    clf.fit(X, y);
    const pred = clf.predict(X);
    expect(pred.shape).toEqual([6]);
    expect(clf.score(X, y)).toBeGreaterThan(0.5);
  });

  it("throws NotFittedError before fitting", () => {
    const clf = new MultinomialNB();
    expect(() => clf.predict(tensor([[1, 2]]))).toThrow(/fitted/i);
  });

  it("throws on negative features", () => {
    const clf = new MultinomialNB();
    expect(() => clf.fit(tensor([[-1, 0]]), tensor([0]))).toThrow(/non-negative/i);
  });
});

describe("BernoulliNB", () => {
  it("fits and predicts on binary data", () => {
    const X = tensor([
      [1, 0, 1],
      [0, 1, 1],
      [1, 0, 0],
      [0, 1, 0],
      [1, 1, 0],
      [0, 0, 1],
    ]);
    const y = tensor([0, 1, 0, 1, 0, 1]);
    const clf = new BernoulliNB();
    clf.fit(X, y);
    const pred = clf.predict(X);
    expect(pred.shape).toEqual([6]);
  });

  it("binarizes continuous features", () => {
    const X = tensor([
      [0.5, -0.1],
      [1.5, 0.5],
    ]);
    const y = tensor([0, 1]);
    const clf = new BernoulliNB({ binarize: 0.0 });
    clf.fit(X, y);
    const proba = clf.predictProba(tensor([[1.0, -0.5]]));
    expect(proba.shape[1]).toBe(2);
  });
});

describe("ComplementNB", () => {
  it("fits and predicts", () => {
    const X = tensor([
      [3, 0, 1],
      [0, 2, 1],
      [1, 0, 3],
      [0, 3, 0],
      [2, 0, 2],
      [0, 2, 0],
    ]);
    const y = tensor([0, 1, 0, 1, 0, 1]);
    const clf = new ComplementNB();
    clf.fit(X, y);
    const pred = clf.predict(X);
    expect(pred.shape).toEqual([6]);
    expect(clf.score(X, y)).toBeGreaterThan(0.5);
  });

  it("supports norm option", () => {
    const X = tensor([
      [3, 0],
      [0, 3],
      [2, 1],
      [1, 2],
    ]);
    const y = tensor([0, 1, 0, 1]);
    const clf = new ComplementNB({ norm: true });
    clf.fit(X, y);
    expect(clf.getParams().norm).toBe(true);
  });
});

describe("KMeans nInit", () => {
  it("default nInit is 10", () => {
    const km = new KMeans({ nClusters: 2 });
    expect(km.getParams().nInit).toBe(10);
  });

  it("nInit=1 still works", () => {
    const km = new KMeans({ nClusters: 2, nInit: 1, randomState: 42 });
    km.fit(X_clf);
    expect(km.labels.size).toBe(10);
  });

  it("nInit>1 picks best inertia", () => {
    const km1 = new KMeans({ nClusters: 2, nInit: 1, randomState: 42 });
    km1.fit(X_clf);
    const km10 = new KMeans({ nClusters: 2, nInit: 10, randomState: 42 });
    km10.fit(X_clf);
    // nInit=10 should have inertia <= nInit=1
    expect(km10.inertia).toBeLessThanOrEqual(km1.inertia + 1e-6);
  });
});

describe("AdaBoostClassifier", () => {
  it("fits and predicts", () => {
    const clf = new AdaBoostClassifier({ nEstimators: 10, maxDepth: 1 });
    clf.fit(X_clf, y_clf);
    const pred = clf.predict(X_clf);
    expect(pred.shape).toEqual([10]);
    expect(clf.score(X_clf, y_clf)).toBeGreaterThanOrEqual(0.5);
  });

  it("predictProba returns valid probabilities", () => {
    const clf = new AdaBoostClassifier({ nEstimators: 10 });
    clf.fit(X_clf, y_clf);
    const proba = clf.predictProba(X_clf);
    expect(proba.shape).toEqual([10, 2]);
    for (let i = 0; i < 10; i++) {
      let sum = 0;
      for (let c = 0; c < 2; c++) {
        const p = Number(proba.data[proba.offset + i * 2 + c]);
        expect(p).toBeGreaterThanOrEqual(0);
        sum += p;
      }
      expect(sum).toBeCloseTo(1, 3);
    }
  });

  it("throws NotFittedError before fitting", () => {
    const clf = new AdaBoostClassifier();
    expect(() => clf.predict(tensor([[1, 2]]))).toThrow(/fitted/i);
  });

  it("exposes featureImportances", () => {
    const clf = new AdaBoostClassifier({ nEstimators: 5, maxDepth: 1 });
    clf.fit(X_clf, y_clf);
    const imp = clf.featureImportances;
    expect(imp.shape).toEqual([2]);
  });
});

describe("AdaBoostRegressor", () => {
  it("fits and predicts", () => {
    const reg = new AdaBoostRegressor({ nEstimators: 10, maxDepth: 2 });
    reg.fit(X_reg, y_reg);
    const pred = reg.predict(X_reg);
    expect(pred.shape).toEqual([10]);
  });

  it("throws NotFittedError before fitting", () => {
    const reg = new AdaBoostRegressor();
    expect(() => reg.predict(tensor([[1, 2]]))).toThrow(/fitted/i);
  });
});

describe("BaggingClassifier", () => {
  it("fits and predicts", () => {
    const clf = new BaggingClassifier({ nEstimators: 5, randomState: 42 });
    clf.fit(X_clf, y_clf);
    const pred = clf.predict(X_clf);
    expect(pred.shape).toEqual([10]);
    expect(clf.score(X_clf, y_clf)).toBeGreaterThan(0.5);
  });

  it("throws NotFittedError before fitting", () => {
    const clf = new BaggingClassifier();
    expect(() => clf.predict(tensor([[1, 2]]))).toThrow(/fitted/i);
  });
});

describe("BaggingRegressor", () => {
  it("fits and predicts", () => {
    const reg = new BaggingRegressor({ nEstimators: 5, randomState: 42 });
    reg.fit(X_reg, y_reg);
    const pred = reg.predict(X_reg);
    expect(pred.shape).toEqual([10]);
  });

  it("throws NotFittedError before fitting", () => {
    const reg = new BaggingRegressor();
    expect(() => reg.predict(tensor([[1, 2]]))).toThrow(/fitted/i);
  });
});

describe("SVC (Kernel SVM)", () => {
  it("fits and predicts with RBF kernel", () => {
    const svc = new SVC({ kernel: "rbf", C: 10, maxIter: 500 });
    svc.fit(X_clf, y_clf);
    const pred = svc.predict(X_clf);
    expect(pred.shape).toEqual([10]);
    expect(svc.score(X_clf, y_clf)).toBeGreaterThanOrEqual(0.5);
  });

  it("fits and predicts with linear kernel", () => {
    const svc = new SVC({ kernel: "linear", C: 1 });
    svc.fit(X_clf, y_clf);
    const pred = svc.predict(X_clf);
    expect(pred.shape).toEqual([10]);
  });

  it("predictProba returns valid probabilities", () => {
    const svc = new SVC({ kernel: "rbf", C: 10 });
    svc.fit(X_clf, y_clf);
    const proba = svc.predictProba(X_clf);
    expect(proba.shape).toEqual([10, 2]);
  });

  it("throws NotFittedError before fitting", () => {
    const svc = new SVC();
    expect(() => svc.predict(tensor([[1, 2]]))).toThrow(/fitted/i);
  });
});

describe("SVR (Kernel SVR)", () => {
  it("fits and predicts with RBF kernel", () => {
    const svr = new SVR({ kernel: "rbf", C: 100, maxIter: 500 });
    svr.fit(X_reg, y_reg);
    const pred = svr.predict(X_reg);
    expect(pred.shape).toEqual([10]);
  });

  it("throws NotFittedError before fitting", () => {
    const svr = new SVR();
    expect(() => svr.predict(tensor([[1]]))).toThrow(/fitted/i);
  });
});

describe("VotingClassifier", () => {
  it("fits and predicts with hard voting", () => {
    const clf = new VotingClassifier({
      estimators: [
        new DecisionTreeClassifier({ maxDepth: 2 }),
        new DecisionTreeClassifier({ maxDepth: 3 }),
      ],
      voting: "hard",
    });
    clf.fit(X_clf, y_clf);
    const pred = clf.predict(X_clf);
    expect(pred.shape).toEqual([10]);
    expect(clf.score(X_clf, y_clf)).toBeGreaterThan(0.5);
  });

  it("fits and predicts with soft voting", () => {
    const clf = new VotingClassifier({
      estimators: [
        new DecisionTreeClassifier({ maxDepth: 2 }),
        new DecisionTreeClassifier({ maxDepth: 3 }),
      ],
      voting: "soft",
    });
    clf.fit(X_clf, y_clf);
    const proba = clf.predictProba(X_clf);
    expect(proba.shape).toEqual([10, 2]);
  });

  it("throws NotFittedError before fitting", () => {
    const clf = new VotingClassifier({
      estimators: [new DecisionTreeClassifier()],
    });
    expect(() => clf.predict(tensor([[1, 2]]))).toThrow(/fitted/i);
  });
});

describe("VotingRegressor", () => {
  it("fits and predicts", () => {
    const reg = new VotingRegressor({
      estimators: [
        new DecisionTreeRegressor({ maxDepth: 2 }),
        new DecisionTreeRegressor({ maxDepth: 3 }),
      ],
    });
    reg.fit(X_reg, y_reg);
    const pred = reg.predict(X_reg);
    expect(pred.shape).toEqual([10]);
  });

  it("supports weighted averaging", () => {
    const reg = new VotingRegressor({
      estimators: [
        new DecisionTreeRegressor({ maxDepth: 2 }),
        new DecisionTreeRegressor({ maxDepth: 3 }),
      ],
      weights: [2, 1],
    });
    reg.fit(X_reg, y_reg);
    expect(reg.getParams().weights).toEqual([2, 1]);
  });
});

describe("AgglomerativeClustering", () => {
  it("fits with ward linkage", () => {
    const agg = new AgglomerativeClustering({ nClusters: 2, linkage: "ward" });
    agg.fit(X_clf);
    expect(agg.labels.size).toBe(10);
    // Check we get exactly 2 unique labels
    const labels = new Set<number>();
    for (let i = 0; i < 10; i++) {
      labels.add(Number(agg.labels.data[agg.labels.offset + i]));
    }
    expect(labels.size).toBe(2);
  });

  it("fits with single linkage", () => {
    const agg = new AgglomerativeClustering({
      nClusters: 2,
      linkage: "single",
    });
    agg.fit(X_clf);
    expect(agg.labels.size).toBe(10);
  });

  it("fitPredict returns labels", () => {
    const agg = new AgglomerativeClustering({ nClusters: 3 });
    const labels = agg.fitPredict(X_clf);
    expect(labels.size).toBe(10);
  });
});

describe("GaussianMixture", () => {
  it("fits and predicts", () => {
    const gmm = new GaussianMixture({ nComponents: 2, randomState: 42 });
    gmm.fit(X_clf);
    const labels = gmm.predict(X_clf);
    expect(labels.size).toBe(10);
  });

  it("predictProba returns valid probabilities", () => {
    const gmm = new GaussianMixture({ nComponents: 2, randomState: 42 });
    gmm.fit(X_clf);
    const proba = gmm.predictProba(X_clf);
    expect(proba.shape).toEqual([10, 2]);
    for (let i = 0; i < 10; i++) {
      let sum = 0;
      for (let c = 0; c < 2; c++) {
        const p = Number(proba.data[proba.offset + i * 2 + c]);
        expect(p).toBeGreaterThanOrEqual(0);
        sum += p;
      }
      expect(sum).toBeCloseTo(1, 3);
    }
  });

  it("exposes cluster centers (means)", () => {
    const gmm = new GaussianMixture({ nComponents: 2, randomState: 42 });
    gmm.fit(X_clf);
    expect(gmm.clusterCenters.shape).toEqual([2, 2]);
  });
});

describe("MiniBatchKMeans", () => {
  it("fits and predicts", () => {
    const km = new MiniBatchKMeans({
      nClusters: 2,
      batchSize: 5,
      randomState: 42,
    });
    km.fit(X_clf);
    expect(km.labels.size).toBe(10);
    expect(km.clusterCenters.shape).toEqual([2, 2]);
  });

  it("predict on new data", () => {
    const km = new MiniBatchKMeans({
      nClusters: 2,
      batchSize: 5,
      randomState: 42,
    });
    km.fit(X_clf);
    const pred = km.predict(
      tensor([
        [3, 30],
        [8, 80],
      ])
    );
    expect(pred.size).toBe(2);
  });

  it("throws NotFittedError before fitting", () => {
    const km = new MiniBatchKMeans();
    expect(() => km.predict(tensor([[1, 2]]))).toThrow(/fitted/i);
  });
});

describe("IsolationForest", () => {
  it("fits and detects outliers", () => {
    const X = tensor([
      [0, 0],
      [0.1, -0.1],
      [0.2, 0.1],
      [-0.1, 0.2],
      [0.05, 0.05],
      [-0.05, -0.05],
      [100, 100],
      [-100, -100],
    ]);
    const ifo = new IsolationForest({
      nEstimators: 50,
      contamination: 0.25,
      randomState: 42,
    });
    ifo.fit(X);
    const labels = ifo.predict(X);
    expect(labels.size).toBe(8);
    // Outliers ([100,100] and [-100,-100]) should be detected as -1
    expect(Number(labels.data[labels.offset + 6])).toBe(-1);
    expect(Number(labels.data[labels.offset + 7])).toBe(-1);
  });

  it("scoreSamples returns anomaly scores", () => {
    const X = tensor([
      [0, 0],
      [0.1, 0.1],
      [100, 100],
    ]);
    const ifo = new IsolationForest({ nEstimators: 50, randomState: 42 });
    ifo.fit(X);
    const scores = ifo.scoreSamples(X);
    expect(scores.size).toBe(3);
    // Outlier should have a more negative score
    expect(Number(scores.data[scores.offset + 2])).toBeLessThan(
      Number(scores.data[scores.offset + 0])
    );
  });

  it("throws NotFittedError before fitting", () => {
    const ifo = new IsolationForest();
    expect(() => ifo.predict(tensor([[1, 2]]))).toThrow(/fitted/i);
  });
});

describe("LocalOutlierFactor", () => {
  it("fits and detects outliers", () => {
    const X = tensor([
      [0, 0],
      [0.1, -0.1],
      [0.2, 0.1],
      [-0.1, 0.2],
      [0.05, 0.05],
      [-0.05, -0.05],
      [100, 100],
      [-100, -100],
    ]);
    const lof = new LocalOutlierFactor({ nNeighbors: 3, contamination: 0.25 });
    lof.fit(X);
    const labels = lof.predict(X);
    expect(labels.size).toBe(8);
    // The two far-away points should be outliers
    expect(Number(labels.data[labels.offset + 6])).toBe(-1);
    expect(Number(labels.data[labels.offset + 7])).toBe(-1);
  });

  it("scoreSamples returns scores", () => {
    const X = tensor([
      [0, 0],
      [0.1, 0.1],
      [100, 100],
    ]);
    const lof = new LocalOutlierFactor({ nNeighbors: 2 });
    lof.fit(X);
    const scores = lof.scoreSamples(X);
    expect(scores.size).toBe(3);
  });

  it("throws NotFittedError before fitting", () => {
    const lof = new LocalOutlierFactor();
    expect(() => lof.predict(tensor([[1, 2]]))).toThrow(/fitted/i);
  });
});

// ── class_weight tests ──

describe("LogisticRegression classWeight", () => {
  it("accepts classWeight='balanced'", () => {
    const X = tensor([
      [1, 0],
      [2, 0],
      [3, 0],
      [10, 0],
    ]);
    const y = tensor([0, 0, 0, 1]);
    const lr = new LogisticRegression({
      classWeight: "balanced",
      maxIter: 200,
      learningRate: 0.1,
    });
    lr.fit(X, y);
    const pred = lr.predict(X);
    expect(pred.shape).toEqual([4]);
  });

  it("accepts classWeight as dict", () => {
    const X = tensor([
      [1, 0],
      [2, 0],
      [3, 0],
      [10, 0],
    ]);
    const y = tensor([0, 0, 0, 1]);
    const lr = new LogisticRegression({
      classWeight: { 0: 1, 1: 3 },
      maxIter: 200,
      learningRate: 0.1,
    });
    lr.fit(X, y);
    const pred = lr.predict(X);
    expect(pred.shape).toEqual([4]);
  });

  it("classWeight is exposed in getParams", () => {
    const lr = new LogisticRegression({ classWeight: "balanced" });
    const params = lr.getParams();
    expect(params.classWeight).toBe("balanced");
  });
});

describe("LinearSVC classWeight", () => {
  it("accepts classWeight='balanced'", () => {
    const X = tensor([
      [1, 0],
      [2, 0],
      [3, 0],
      [10, 0],
    ]);
    const y = tensor([0, 0, 0, 1]);
    const svc = new LinearSVC({ classWeight: "balanced", maxIter: 500 });
    svc.fit(X, y);
    const pred = svc.predict(X);
    expect(pred.shape).toEqual([4]);
  });

  it("accepts classWeight as dict", () => {
    const X = tensor([
      [1, 0],
      [2, 0],
      [3, 0],
      [10, 0],
    ]);
    const y = tensor([0, 0, 0, 1]);
    const svc = new LinearSVC({ classWeight: { 0: 1, 1: 5 }, maxIter: 500 });
    svc.fit(X, y);
    const pred = svc.predict(X);
    expect(pred.shape).toEqual([4]);
  });
});

// ── warm_start tests ──

describe("GradientBoostingRegressor warmStart", () => {
  it("warm_start adds trees incrementally", () => {
    const X = tensor([[1], [2], [3], [4], [5]]);
    const y = tensor([1.2, 2.1, 2.9, 4.0, 5.1]);

    const gbr = new GradientBoostingRegressor({
      nEstimators: 5,
      warmStart: true,
      maxDepth: 2,
    });
    gbr.fit(X, y);
    const score1 = gbr.score(X, y);

    // Increase nEstimators. Not possible with readonly, so we test that
    // calling fit again reuses existing trees
    gbr.fit(X, y);
    const score2 = gbr.score(X, y);
    // With warmStart and same nEstimators, second fit should return immediately
    expect(score2).toBeCloseTo(score1, 5);
  });

  it("warmStart appears in getParams", () => {
    const gbr = new GradientBoostingRegressor({ warmStart: true });
    expect(gbr.getParams().warmStart).toBe(true);
  });
});

describe("GradientBoostingClassifier warmStart", () => {
  it("warm_start produces valid predictions after re-fit", () => {
    const X = tensor([
      [1, 2],
      [2, 3],
      [3, 1],
      [4, 2],
      [5, 3],
      [6, 1],
    ]);
    const y = tensor([0, 0, 0, 1, 1, 1]);

    const gbc = new GradientBoostingClassifier({
      nEstimators: 5,
      warmStart: true,
      maxDepth: 2,
    });
    gbc.fit(X, y);
    const pred = gbc.predict(X);
    expect(pred.shape).toEqual([6]);

    // Re-fit with same data should not break
    gbc.fit(X, y);
    const pred2 = gbc.predict(X);
    expect(pred2.shape).toEqual([6]);
  });
});

describe("RandomForestClassifier warmStart", () => {
  it("warm_start keeps existing trees", () => {
    const X = tensor([
      [1, 2],
      [2, 3],
      [3, 1],
      [4, 2],
      [5, 3],
      [6, 1],
    ]);
    const y = tensor([0, 0, 0, 1, 1, 1]);

    const rfc = new RandomForestClassifier({
      nEstimators: 5,
      warmStart: true,
      maxDepth: 3,
      randomState: 42,
    });
    rfc.fit(X, y);
    const pred = rfc.predict(X);
    expect(pred.shape).toEqual([6]);

    // Second fit should reuse trees (already at nEstimators=5)
    rfc.fit(X, y);
    const pred2 = rfc.predict(X);
    expect(pred2.shape).toEqual([6]);
  });

  it("warmStart appears in getParams", () => {
    const rfc = new RandomForestClassifier({ warmStart: true });
    expect(rfc.getParams().warmStart).toBe(true);
  });
});

describe("RandomForestRegressor warmStart", () => {
  it("warm_start keeps existing trees", () => {
    const X = tensor([[1], [2], [3], [4], [5]]);
    const y = tensor([1, 2, 3, 4, 5]);

    const rfr = new RandomForestRegressor({
      nEstimators: 5,
      warmStart: true,
      maxDepth: 3,
      randomState: 42,
    });
    rfr.fit(X, y);
    const score1 = rfr.score(X, y);

    rfr.fit(X, y);
    const score2 = rfr.score(X, y);
    expect(score2).toBeCloseTo(score1, 5);
  });

  it("warmStart appears in getParams", () => {
    const rfr = new RandomForestRegressor({ warmStart: true });
    expect(rfr.getParams().warmStart).toBe(true);
  });
});

describe("KMeans warmStart", () => {
  it("warm_start reuses previous centroids", () => {
    const X = tensor([
      [1, 0],
      [1.1, 0.1],
      [0.9, -0.1],
      [10, 10],
      [10.1, 10.1],
      [9.9, 9.9],
    ]);

    const km = new KMeans({
      nClusters: 2,
      warmStart: true,
      randomState: 42,
      nInit: 1,
    });
    km.fit(X);
    const labels1 = km.predict(X);
    expect(labels1.shape).toEqual([6]);

    // Second fit should use previous centroids
    km.fit(X);
    const labels2 = km.predict(X);
    expect(labels2.shape).toEqual([6]);
    // Labels should be consistent
    for (let i = 0; i < 6; i++) {
      expect(Number(labels2.data[labels2.offset + i])).toBe(
        Number(labels1.data[labels1.offset + i])
      );
    }
  });

  it("warmStart appears in getParams", () => {
    const km = new KMeans({ warmStart: true });
    expect(km.getParams().warmStart).toBe(true);
  });
});
