import { describe, expect, it } from "vitest";
import {
  AffinityPropagation,
  AgglomerativeClustering,
  Birch,
  DBSCAN,
  ElasticNet,
  Lasso,
  LinearRegression,
  LogisticRegression,
  MeanShift,
  OPTICS,
  Ridge,
  SpectralClustering,
} from "../src/ml";
import {
  AdaBoostClassifier,
  AdaBoostRegressor,
  GradientBoostingClassifier,
  GradientBoostingRegressor,
  StackingClassifier,
  StackingRegressor,
} from "../src/ml/ensemble";
import { tensor } from "../src/ndarray";

// ---- SpectralClustering setParams branches ----
describe("SpectralClustering setParams branches", () => {
  it("set all valid params", () => {
    const sc = new SpectralClustering({ nClusters: 2 });
    sc.setParams({ nClusters: 3 });
    sc.setParams({ affinity: "nearest_neighbors" });
    sc.setParams({ gamma: 2.0 });
    sc.setParams({ nNeighbors: 5 });
    sc.setParams({ randomState: 42 });
    sc.setParams({ nInit: 3 });
    const p = sc.getParams();
    expect(p.nClusters).toBe(3);
    expect(p.affinity).toBe("nearest_neighbors");
    expect(p.gamma).toBe(2.0);
    expect(p.nNeighbors).toBe(5);
    expect(p.randomState).toBe(42);
    expect(p.nInit).toBe(3);
  });

  it("invalid nClusters", () => {
    const sc = new SpectralClustering();
    expect(() => sc.setParams({ nClusters: 0 })).toThrow();
    expect(() => sc.setParams({ nClusters: -1 })).toThrow();
    expect(() => sc.setParams({ nClusters: 1.5 })).toThrow();
  });

  it("invalid affinity", () => {
    const sc = new SpectralClustering();
    expect(() => sc.setParams({ affinity: "bad" })).toThrow();
  });

  it("invalid gamma", () => {
    const sc = new SpectralClustering();
    expect(() => sc.setParams({ gamma: 0 })).toThrow();
    expect(() => sc.setParams({ gamma: -1 })).toThrow();
  });

  it("invalid nNeighbors", () => {
    const sc = new SpectralClustering();
    expect(() => sc.setParams({ nNeighbors: 0 })).toThrow();
    expect(() => sc.setParams({ nNeighbors: 1.5 })).toThrow();
  });

  it("invalid randomState", () => {
    const sc = new SpectralClustering();
    expect(() => sc.setParams({ randomState: "abc" as unknown })).toThrow();
  });

  it("invalid nInit", () => {
    const sc = new SpectralClustering();
    expect(() => sc.setParams({ nInit: 0 })).toThrow();
  });

  it("unknown key", () => {
    const sc = new SpectralClustering();
    expect(() => sc.setParams({ unknown: 1 })).toThrow();
  });

  it("affinity nearest_neighbors fit", () => {
    const sc = new SpectralClustering({
      nClusters: 2,
      affinity: "nearest_neighbors",
      nNeighbors: 3,
      randomState: 42,
    });
    const X = tensor([
      [1, 2],
      [1.5, 1.8],
      [1.2, 2.1],
      [5, 8],
      [6, 7],
      [5.5, 8.2],
    ]);
    sc.fit(X);
    expect(sc.labels.size).toBe(6);
  });
});

// ---- MeanShift setParams branches ----
describe("MeanShift setParams branches", () => {
  it("set all valid params", () => {
    const ms = new MeanShift();
    ms.setParams({ bandwidth: 2.0 });
    ms.setParams({ maxIter: 200 });
    ms.setParams({ binSeeding: true });
    const p = ms.getParams();
    expect(p.bandwidth).toBe(2.0);
    expect(p.maxIter).toBe(200);
  });

  it("invalid bandwidth", () => {
    const ms = new MeanShift();
    expect(() => ms.setParams({ bandwidth: -1 })).toThrow();
  });

  it("invalid maxIter", () => {
    const ms = new MeanShift();
    expect(() => ms.setParams({ maxIter: 0 })).toThrow();
  });

  it("unknown key", () => {
    const ms = new MeanShift();
    expect(() => ms.setParams({ unknown: 1 })).toThrow();
  });
});

// ---- Birch setParams branches ----
describe("Birch setParams branches", () => {
  it("set all valid params", () => {
    const b = new Birch({ nClusters: 2 });
    b.setParams({ nClusters: 5 });
    b.setParams({ threshold: 0.5 });
    b.setParams({ branchingFactor: 100 });
    const p = b.getParams();
    expect(p.nClusters).toBe(5);
  });

  it("invalid nClusters", () => {
    const b = new Birch();
    expect(() => b.setParams({ nClusters: 0 })).toThrow();
  });

  it("invalid threshold", () => {
    const b = new Birch();
    expect(() => b.setParams({ threshold: -1 })).toThrow();
  });

  it("invalid branchingFactor", () => {
    const b = new Birch();
    expect(() => b.setParams({ branchingFactor: 0 })).toThrow();
  });

  it("unknown key", () => {
    const b = new Birch();
    expect(() => b.setParams({ unknown: 1 })).toThrow();
  });
});

// ---- DBSCAN setParams branches ----
describe("DBSCAN setParams branches", () => {
  it("set valid params", () => {
    const db = new DBSCAN();
    db.setParams({ eps: 1.0 });
    db.setParams({ minSamples: 3 });
    const p = db.getParams();
    expect(p.eps).toBe(1.0);
    expect(p.minSamples).toBe(3);
  });

  it("invalid eps", () => {
    const db = new DBSCAN();
    expect(() => db.setParams({ eps: -1 })).toThrow();
  });

  it("invalid minSamples", () => {
    const db = new DBSCAN();
    expect(() => db.setParams({ minSamples: 0 })).toThrow();
  });

  it("unknown key", () => {
    const db = new DBSCAN();
    expect(() => db.setParams({ unknown: 1 })).toThrow();
  });
});

// ---- OPTICS setParams branches ----
describe("OPTICS setParams branches", () => {
  it("set valid params", () => {
    const op = new OPTICS();
    op.setParams({ minSamples: 3 });
    op.setParams({ maxEps: 5.0 });
    op.setParams({ clusterMethod: "xi" });
    op.setParams({ xi: 0.1 });
    const p = op.getParams();
    expect(p.minSamples).toBe(3);
  });

  it("invalid minSamples", () => {
    const op = new OPTICS();
    expect(() => op.setParams({ minSamples: 0 })).toThrow();
  });

  it("invalid maxEps", () => {
    const op = new OPTICS();
    expect(() => op.setParams({ maxEps: -1 })).toThrow();
  });

  it("invalid clusterMethod", () => {
    const op = new OPTICS();
    expect(() => op.setParams({ clusterMethod: "bad" })).toThrow();
  });

  it("invalid xi", () => {
    const op = new OPTICS();
    expect(() => op.setParams({ xi: -1 })).toThrow();
  });

  it("unknown key", () => {
    const op = new OPTICS();
    expect(() => op.setParams({ unknown: 1 })).toThrow();
  });
});

// ---- AffinityPropagation setParams branches ----
describe("AffinityPropagation setParams branches", () => {
  it("set valid params", () => {
    const ap = new AffinityPropagation();
    ap.setParams({ damping: 0.8 });
    ap.setParams({ maxIter: 200 });
    ap.setParams({ convergenceIter: 20 });
    const p = ap.getParams();
    expect(p.damping).toBe(0.8);
  });

  it("invalid damping", () => {
    const ap = new AffinityPropagation();
    expect(() => ap.setParams({ damping: 0.3 })).toThrow();
    expect(() => ap.setParams({ damping: 1.1 })).toThrow();
  });

  it("invalid maxIter", () => {
    const ap = new AffinityPropagation();
    expect(() => ap.setParams({ maxIter: 0 })).toThrow();
  });

  it("unknown key", () => {
    const ap = new AffinityPropagation();
    expect(() => ap.setParams({ unknown: 1 })).toThrow();
  });
});

// ---- AgglomerativeClustering setParams branches ----
describe("AgglomerativeClustering setParams branches", () => {
  it("set valid params", () => {
    const ac = new AgglomerativeClustering();
    ac.setParams({ nClusters: 5 });
    ac.setParams({ linkage: "complete" });
    const p = ac.getParams();
    expect(p.nClusters).toBe(5);
  });

  it("invalid nClusters", () => {
    const ac = new AgglomerativeClustering();
    expect(() => ac.setParams({ nClusters: 0 })).toThrow();
  });

  it("invalid linkage", () => {
    const ac = new AgglomerativeClustering();
    expect(() => ac.setParams({ linkage: "bad" })).toThrow();
  });

  it("unknown key", () => {
    const ac = new AgglomerativeClustering();
    expect(() => ac.setParams({ unknown: 1 })).toThrow();
  });
});

// ---- Ridge setParams branches ----
describe("Ridge setParams branches", () => {
  it("set valid params", () => {
    const r = new Ridge();
    r.setParams({ alpha: 2.0 });
    r.setParams({ fitIntercept: false });
    r.setParams({ maxIter: 500 });
    r.setParams({ tol: 0.01 });
    r.setParams({ solver: "cholesky" });
    const p = r.getParams();
    expect(p.alpha).toBe(2.0);
  });

  it("invalid alpha", () => {
    const r = new Ridge();
    expect(() => r.setParams({ alpha: NaN })).toThrow();
    expect(() => r.setParams({ alpha: "bad" as unknown })).toThrow();
  });

  it("invalid maxIter", () => {
    const r = new Ridge();
    expect(() => r.setParams({ maxIter: NaN })).toThrow();
    expect(() => r.setParams({ maxIter: "bad" as unknown })).toThrow();
  });

  it("invalid tol", () => {
    const r = new Ridge();
    expect(() => r.setParams({ tol: NaN })).toThrow();
  });

  it("invalid fitIntercept", () => {
    const r = new Ridge();
    expect(() => r.setParams({ fitIntercept: 1 as unknown })).toThrow();
  });

  it("invalid normalize", () => {
    const r = new Ridge();
    expect(() => r.setParams({ normalize: 1 as unknown })).toThrow();
  });

  it("invalid solver", () => {
    const r = new Ridge();
    expect(() => r.setParams({ solver: "bad" })).toThrow();
  });

  it("unknown key", () => {
    const r = new Ridge();
    expect(() => r.setParams({ unknown: 1 })).toThrow();
  });

  it("coef and intercept getters", () => {
    const r = new Ridge();
    const Xr = tensor([[1], [2], [3], [4]]);
    const yr = tensor([2, 4, 6, 8]);
    r.fit(Xr, yr);
    expect(r.coef.size).toBe(1);
    expect(typeof r.intercept).toBe("number");
  });

  it("coef throws before fit", () => {
    const r = new Ridge();
    expect(() => r.coef).toThrow();
  });

  it("intercept throws before fit", () => {
    const r = new Ridge();
    expect(() => r.intercept).toThrow();
  });

  it("nIter throws before fit", () => {
    const r = new Ridge();
    expect(() => r.nIter).toThrow();
  });
});

// ---- Lasso setParams branches ----
describe("Lasso setParams branches", () => {
  it("set valid params", () => {
    const l = new Lasso();
    l.setParams({ alpha: 2.0 });
    l.setParams({ fitIntercept: false });
    l.setParams({ maxIter: 500 });
    l.setParams({ tol: 0.01 });
    const p = l.getParams();
    expect(p.alpha).toBe(2.0);
  });

  it("invalid alpha", () => {
    const l = new Lasso();
    expect(() => l.setParams({ alpha: NaN })).toThrow();
    expect(() => l.setParams({ alpha: "bad" as unknown })).toThrow();
  });

  it("unknown key", () => {
    const l = new Lasso();
    expect(() => l.setParams({ unknown: 1 })).toThrow();
  });
});

// ---- ElasticNet setParams branches ----
describe("ElasticNet setParams branches", () => {
  it("set valid params", () => {
    const en = new ElasticNet();
    en.setParams({ alpha: 2.0 });
    en.setParams({ l1Ratio: 0.3 });
    en.setParams({ fitIntercept: false });
    en.setParams({ maxIter: 500 });
    en.setParams({ tol: 0.01 });
    const p = en.getParams();
    expect(p.alpha).toBe(2.0);
    expect(p.l1Ratio).toBe(0.3);
  });

  it("invalid alpha", () => {
    const en = new ElasticNet();
    expect(() => en.setParams({ alpha: NaN })).toThrow();
    expect(() => en.setParams({ alpha: "bad" as unknown })).toThrow();
  });

  it("invalid l1Ratio", () => {
    const en = new ElasticNet();
    expect(() => en.setParams({ l1Ratio: -0.1 })).toThrow();
    expect(() => en.setParams({ l1Ratio: 1.1 })).toThrow();
    expect(() => en.setParams({ l1Ratio: "bad" as unknown })).toThrow();
  });

  it("unknown key", () => {
    const en = new ElasticNet();
    expect(() => en.setParams({ unknown: 1 })).toThrow();
  });
});

// ---- AdaBoost setParams branches ----
describe("AdaBoost setParams branches", () => {
  it("AdaBoostClassifier set valid params", () => {
    const ab = new AdaBoostClassifier();
    ab.setParams({ nEstimators: 100 });
    ab.setParams({ learningRate: 0.5 });
    const p = ab.getParams();
    expect(p.nEstimators).toBe(100);
    expect(p.learningRate).toBe(0.5);
  });

  it("AdaBoostClassifier invalid nEstimators", () => {
    const ab = new AdaBoostClassifier();
    expect(() => ab.setParams({ nEstimators: 0 })).toThrow();
  });

  it("AdaBoostClassifier invalid learningRate", () => {
    const ab = new AdaBoostClassifier();
    expect(() => ab.setParams({ learningRate: 0 })).toThrow();
    expect(() => ab.setParams({ learningRate: -1 })).toThrow();
  });

  it("AdaBoostClassifier unknown key", () => {
    const ab = new AdaBoostClassifier();
    expect(() => ab.setParams({ unknown: 1 })).toThrow();
  });

  it("AdaBoostRegressor set valid params", () => {
    const ab = new AdaBoostRegressor();
    ab.setParams({ nEstimators: 100 });
    ab.setParams({ learningRate: 0.5 });
    const p = ab.getParams();
    expect(p.nEstimators).toBe(100);
  });

  it("AdaBoostRegressor invalid params", () => {
    const ab = new AdaBoostRegressor();
    expect(() => ab.setParams({ nEstimators: 0 })).toThrow();
    expect(() => ab.setParams({ learningRate: 0 })).toThrow();
    expect(() => ab.setParams({ unknown: 1 })).toThrow();
  });
});

// ---- GradientBoosting setParams branches ----
describe("GradientBoosting setParams branches", () => {
  it("GBClassifier set valid params", () => {
    const gb = new GradientBoostingClassifier();
    gb.setParams({ nEstimators: 100 });
    gb.setParams({ learningRate: 0.05 });
    gb.setParams({ maxDepth: 5 });
    gb.setParams({ subsample: 0.8 });
    const p = gb.getParams();
    expect(p.nEstimators).toBe(100);
    expect(p.maxDepth).toBe(5);
  });

  it("GBClassifier invalid params", () => {
    const gb = new GradientBoostingClassifier();
    expect(() => gb.setParams({ nEstimators: 0 })).toThrow();
    expect(() => gb.setParams({ learningRate: 0 })).toThrow();
    expect(() => gb.setParams({ maxDepth: 0 })).toThrow();
    expect(() => gb.setParams({ subsample: 0 })).toThrow();
    expect(() => gb.setParams({ subsample: 1.5 })).toThrow();
    expect(() => gb.setParams({ unknown: 1 })).toThrow();
  });

  it("GBRegressor set valid params", () => {
    const gb = new GradientBoostingRegressor();
    gb.setParams({ nEstimators: 100 });
    gb.setParams({ learningRate: 0.05 });
    gb.setParams({ maxDepth: 5 });
    const p = gb.getParams();
    expect(p.nEstimators).toBe(100);
  });

  it("GBRegressor invalid params", () => {
    const gb = new GradientBoostingRegressor();
    expect(() => gb.setParams({ nEstimators: 0 })).toThrow();
    expect(() => gb.setParams({ learningRate: 0 })).toThrow();
    expect(() => gb.setParams({ maxDepth: 0 })).toThrow();
    expect(() => gb.setParams({ unknown: 1 })).toThrow();
  });
});

// ---- Stacking setParams branches ----
describe("Stacking setParams branches", () => {
  it("StackingClassifier set valid params", () => {
    const sc = new StackingClassifier({
      estimators: [new LogisticRegression({ maxIter: 200 })],
      finalEstimator: new LogisticRegression({ maxIter: 200 }),
    });
    sc.setParams({ passthrough: true });
    const p = sc.getParams();
    expect(p.passthrough).toBe(true);
  });

  it("StackingClassifier invalid passthrough", () => {
    const sc = new StackingClassifier({
      estimators: [new LogisticRegression({ maxIter: 200 })],
      finalEstimator: new LogisticRegression({ maxIter: 200 }),
    });
    expect(() => sc.setParams({ passthrough: "yes" as unknown })).toThrow();
  });

  it("StackingClassifier unknown key", () => {
    const sc = new StackingClassifier({
      estimators: [new LogisticRegression({ maxIter: 200 })],
      finalEstimator: new LogisticRegression({ maxIter: 200 }),
    });
    expect(() => sc.setParams({ unknown: 1 })).toThrow();
  });

  it("StackingRegressor set valid params", () => {
    const sr = new StackingRegressor({
      estimators: [new LinearRegression()],
      finalEstimator: new LinearRegression(),
    });
    sr.setParams({ passthrough: true });
    const p = sr.getParams();
    expect(p.passthrough).toBe(true);
  });

  it("StackingRegressor invalid params", () => {
    const sr = new StackingRegressor({
      estimators: [new LinearRegression()],
      finalEstimator: new LinearRegression(),
    });
    expect(() => sr.setParams({ passthrough: 1 as unknown })).toThrow();
    expect(() => sr.setParams({ unknown: 1 })).toThrow();
  });
});
