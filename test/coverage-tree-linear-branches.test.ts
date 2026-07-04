import { describe, expect, it } from "vitest";
import { BayesianRidge, ElasticNet, Lasso, LogisticRegression, Ridge } from "../src/ml";
import {
  DecisionTreeClassifier,
  DecisionTreeRegressor,
  ExtraTreesClassifier,
  ExtraTreesRegressor,
  export_text,
  RandomForestClassifier,
  RandomForestRegressor,
} from "../src/ml/tree";
import { tensor } from "../src/ndarray";

const Xc = tensor([
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
  [2, 2],
  [2, 3],
  [3, 2],
  [3, 3],
]);
const yc = tensor([0, 0, 0, 0, 1, 1, 1, 1]);

const Xr = tensor([[1], [2], [3], [4], [5], [6], [7], [8]]);
const yr = tensor([2, 4, 6, 8, 10, 12, 14, 16]);

// ---- DecisionTreeClassifier ----
describe("DecisionTreeClassifier branches", () => {
  it("fit and predict", () => {
    const dt = new DecisionTreeClassifier({ maxDepth: 5, randomState: 42 });
    dt.fit(Xc, yc);
    const pred = dt.predict(Xc);
    expect(pred.shape).toEqual([8]);
  });

  it("predictProba", () => {
    const dt = new DecisionTreeClassifier({ maxDepth: 5, randomState: 42 });
    dt.fit(Xc, yc);
    const proba = dt.predictProba(Xc);
    expect(proba.shape[0]).toBe(8);
    expect(proba.shape[1]).toBe(2);
  });

  it("score", () => {
    const dt = new DecisionTreeClassifier({ maxDepth: 5, randomState: 42 });
    dt.fit(Xc, yc);
    const s = dt.score(Xc, yc);
    expect(s).toBeGreaterThanOrEqual(0.5);
  });

  it("different criteria (gini vs entropy)", () => {
    const dtG = new DecisionTreeClassifier({
      criterion: "gini",
      randomState: 42,
    });
    dtG.fit(Xc, yc);
    expect(dtG.predict(Xc).shape).toEqual([8]);

    const dtE = new DecisionTreeClassifier({
      criterion: "entropy",
      randomState: 42,
    });
    dtE.fit(Xc, yc);
    expect(dtE.predict(Xc).shape).toEqual([8]);
  });

  it("minSamplesSplit and minSamplesLeaf", () => {
    const dt = new DecisionTreeClassifier({
      minSamplesSplit: 3,
      minSamplesLeaf: 2,
      randomState: 42,
    });
    dt.fit(Xc, yc);
    expect(dt.predict(Xc).shape).toEqual([8]);
  });

  it("maxFeatures", () => {
    const dt = new DecisionTreeClassifier({ maxFeatures: 1, randomState: 42 });
    dt.fit(Xc, yc);
    expect(dt.predict(Xc).shape).toEqual([8]);
  });

  it("getParams", () => {
    const dt = new DecisionTreeClassifier({ maxDepth: 3, minSamplesSplit: 4 });
    const p = dt.getParams();
    expect(p.maxDepth).toBe(3);
    expect(p.minSamplesSplit).toBe(4);
  });

  it("setParams valid", () => {
    const dt = new DecisionTreeClassifier();
    dt.setParams({ maxDepth: 10 });
    expect(dt.getParams().maxDepth).toBe(10);
    dt.setParams({ minSamplesSplit: 5 });
    expect(dt.getParams().minSamplesSplit).toBe(5);
    dt.setParams({ minSamplesLeaf: 3 });
    expect(dt.getParams().minSamplesLeaf).toBe(3);
    dt.setParams({ maxFeatures: 2 });
    expect(dt.getParams().maxFeatures).toBe(2);
    dt.setParams({ randomState: 123 });
    expect(dt.getParams().randomState).toBe(123);
  });

  it("setParams invalid values", () => {
    const dt = new DecisionTreeClassifier();
    expect(() => dt.setParams({ maxDepth: 0 })).toThrow();
    expect(() => dt.setParams({ minSamplesSplit: 1 })).toThrow();
    expect(() => dt.setParams({ minSamplesLeaf: 0 })).toThrow();
    expect(() => dt.setParams({ maxFeatures: 0 })).toThrow();
    expect(() => dt.setParams({ randomState: "abc" as unknown })).toThrow();
    expect(() => dt.setParams({ unknown: 1 })).toThrow();
  });

  it("export_text", () => {
    const dt = new DecisionTreeClassifier({ maxDepth: 3, randomState: 42 });
    dt.fit(Xc, yc);
    const text = export_text(dt);
    expect(typeof text).toBe("string");
    expect(text.length).toBeGreaterThan(0);
  });

  it("tree_ and nFeatures_ accessors", () => {
    const dt = new DecisionTreeClassifier({ maxDepth: 3, randomState: 42 });
    dt.fit(Xc, yc);
    expect(dt.tree_).toBeDefined();
    expect(dt.nFeatures_).toBe(2);
  });

  it("classes getter", () => {
    const dt = new DecisionTreeClassifier({ randomState: 42 });
    dt.fit(Xc, yc);
    expect(dt.classes!.size).toBe(2);
  });
});

// ---- DecisionTreeRegressor ----
describe("DecisionTreeRegressor branches", () => {
  it("fit and predict", () => {
    const dt = new DecisionTreeRegressor({ maxDepth: 5, randomState: 42 });
    dt.fit(Xr, yr);
    const pred = dt.predict(Xr);
    expect(pred.shape).toEqual([8]);
  });

  it("score", () => {
    const dt = new DecisionTreeRegressor({ maxDepth: 5, randomState: 42 });
    dt.fit(Xr, yr);
    const s = dt.score(Xr, yr);
    expect(s).toBeGreaterThanOrEqual(0.5);
  });

  it("criterion mse (default)", () => {
    const dt = new DecisionTreeRegressor({ randomState: 42 });
    dt.fit(Xr, yr);
    expect(dt.predict(Xr).shape).toEqual([8]);
  });

  it("setParams valid", () => {
    const dt = new DecisionTreeRegressor();
    dt.setParams({ maxDepth: 10 });
    expect(dt.getParams().maxDepth).toBe(10);
  });
});

// ---- ExtraTreesClassifier ----
describe("ExtraTreesClassifier branches", () => {
  it("fit and predict", () => {
    const et = new ExtraTreesClassifier({ nEstimators: 5, randomState: 42 });
    et.fit(Xc, yc);
    const pred = et.predict(Xc);
    expect(pred.shape).toEqual([8]);
  });

  it("predictProba", () => {
    const et = new ExtraTreesClassifier({ nEstimators: 5, randomState: 42 });
    et.fit(Xc, yc);
    const proba = et.predictProba(Xc);
    expect(proba.shape[0]).toBe(8);
  });

  it("score", () => {
    const et = new ExtraTreesClassifier({ nEstimators: 5, randomState: 42 });
    et.fit(Xc, yc);
    expect(et.score(Xc, yc)).toBeGreaterThanOrEqual(0.5);
  });

  it("featureImportances", () => {
    const et = new ExtraTreesClassifier({ nEstimators: 5, randomState: 42 });
    et.fit(Xc, yc);
    const imp = et.featureImportances;
    expect(imp.shape).toEqual([2]);
  });

  it("getParams/setParams", () => {
    const et = new ExtraTreesClassifier({ nEstimators: 5 });
    expect(et.getParams().nEstimators).toBe(5);
    et.setParams({ nEstimators: 10 });
    expect(et.getParams().nEstimators).toBe(10);
  });
});

// ---- ExtraTreesRegressor ----
describe("ExtraTreesRegressor branches", () => {
  it("fit and predict", () => {
    const et = new ExtraTreesRegressor({ nEstimators: 5, randomState: 42 });
    et.fit(Xr, yr);
    const pred = et.predict(Xr);
    expect(pred.shape).toEqual([8]);
  });

  it("score", () => {
    const et = new ExtraTreesRegressor({ nEstimators: 5, randomState: 42 });
    et.fit(Xr, yr);
    expect(et.score(Xr, yr)).toBeGreaterThanOrEqual(0.5);
  });

  it("featureImportances", () => {
    const et = new ExtraTreesRegressor({ nEstimators: 5, randomState: 42 });
    et.fit(Xr, yr);
    const imp = et.featureImportances;
    expect(imp.shape).toEqual([1]);
  });
});

// ---- RandomForestClassifier ----
describe("RandomForestClassifier branches", () => {
  it("fit and predict", () => {
    const rf = new RandomForestClassifier({ nEstimators: 5, randomState: 42 });
    rf.fit(Xc, yc);
    const pred = rf.predict(Xc);
    expect(pred.shape).toEqual([8]);
  });

  it("predictProba", () => {
    const rf = new RandomForestClassifier({ nEstimators: 5, randomState: 42 });
    rf.fit(Xc, yc);
    const proba = rf.predictProba(Xc);
    expect(proba.shape[0]).toBe(8);
  });

  it("score", () => {
    const rf = new RandomForestClassifier({ nEstimators: 5, randomState: 42 });
    rf.fit(Xc, yc);
    expect(rf.score(Xc, yc)).toBeGreaterThanOrEqual(0.5);
  });

  it("featureImportances", () => {
    const rf = new RandomForestClassifier({ nEstimators: 5, randomState: 42 });
    rf.fit(Xc, yc);
    const imp = rf.featureImportances;
    expect(imp.shape).toEqual([2]);
  });

  it("maxFeatures sqrt", () => {
    const rf = new RandomForestClassifier({
      nEstimators: 5,
      maxFeatures: "sqrt",
      randomState: 42,
    });
    rf.fit(Xc, yc);
    expect(rf.predict(Xc).shape).toEqual([8]);
  });

  it("getParams/setParams", () => {
    const rf = new RandomForestClassifier({ nEstimators: 5 });
    expect(rf.getParams().nEstimators).toBe(5);
    rf.setParams({ nEstimators: 10 });
    expect(rf.getParams().nEstimators).toBe(10);
  });
});

// ---- RandomForestRegressor ----
describe("RandomForestRegressor branches", () => {
  it("fit and predict", () => {
    const rf = new RandomForestRegressor({ nEstimators: 5, randomState: 42 });
    rf.fit(Xr, yr);
    const pred = rf.predict(Xr);
    expect(pred.shape).toEqual([8]);
  });

  it("score", () => {
    const rf = new RandomForestRegressor({ nEstimators: 5, randomState: 42 });
    rf.fit(Xr, yr);
    expect(rf.score(Xr, yr)).toBeGreaterThanOrEqual(0.5);
  });

  it("featureImportances", () => {
    const rf = new RandomForestRegressor({ nEstimators: 5, randomState: 42 });
    rf.fit(Xr, yr);
    const imp = rf.featureImportances;
    expect(imp.shape).toEqual([1]);
  });
});

// ---- Ridge ----
describe("Ridge branches", () => {
  it("fit and predict", () => {
    const ridge = new Ridge({ alpha: 1.0 });
    ridge.fit(Xr, yr);
    const pred = ridge.predict(Xr);
    expect(pred.shape).toEqual([8]);
  });

  it("score", () => {
    const ridge = new Ridge({ alpha: 1.0 });
    ridge.fit(Xr, yr);
    const s = ridge.score(Xr, yr);
    expect(s).toBeGreaterThan(0.9);
  });

  it("getParams/setParams", () => {
    const ridge = new Ridge({ alpha: 2.0 });
    expect(ridge.getParams().alpha).toBe(2.0);
    ridge.setParams({ alpha: 0.5 });
    expect(ridge.getParams().alpha).toBe(0.5);
  });

  it("fitIntercept false", () => {
    const ridge = new Ridge({ alpha: 1.0, fitIntercept: false });
    ridge.fit(Xr, yr);
    expect(ridge.predict(Xr).shape).toEqual([8]);
  });

  it("multivariate features", () => {
    const ridge = new Ridge({ alpha: 1.0 });
    ridge.fit(Xc, tensor([1, 2, 3, 4, 5, 6, 7, 8]));
    const pred = ridge.predict(Xc);
    expect(pred.shape).toEqual([8]);
  });
});

// ---- Lasso ----
describe("Lasso branches", () => {
  it("fit and predict", () => {
    const lasso = new Lasso({ alpha: 0.1 });
    lasso.fit(Xr, yr);
    const pred = lasso.predict(Xr);
    expect(pred.shape).toEqual([8]);
  });

  it("score", () => {
    const lasso = new Lasso({ alpha: 0.01 });
    lasso.fit(Xr, yr);
    const s = lasso.score(Xr, yr);
    expect(s).toBeGreaterThan(0.5);
  });

  it("getParams/setParams", () => {
    const lasso = new Lasso({ alpha: 2.0 });
    expect(lasso.getParams().alpha).toBe(2.0);
  });
});

// ---- ElasticNet ----
describe("ElasticNet branches", () => {
  it("fit and predict", () => {
    const en = new ElasticNet({ alpha: 0.1, l1Ratio: 0.5 });
    en.fit(Xr, yr);
    const pred = en.predict(Xr);
    expect(pred.shape).toEqual([8]);
  });

  it("score", () => {
    const en = new ElasticNet({ alpha: 0.01, l1Ratio: 0.5 });
    en.fit(Xr, yr);
    const s = en.score(Xr, yr);
    expect(s).toBeGreaterThan(0.5);
  });

  it("getParams/setParams", () => {
    const en = new ElasticNet({ alpha: 2.0 });
    expect(en.getParams().alpha).toBe(2.0);
  });
});

// ---- BayesianRidge ----
describe("BayesianRidge branches", () => {
  it("fit and predict", () => {
    const br = new BayesianRidge();
    br.fit(Xr, yr);
    const pred = br.predict(Xr);
    expect(pred.shape).toEqual([8]);
  });

  it("score", () => {
    const br = new BayesianRidge();
    br.fit(Xr, yr);
    const s = br.score(Xr, yr);
    expect(s).toBeGreaterThan(0.9);
  });

  it("getParams/setParams", () => {
    const br = new BayesianRidge({ maxIter: 200 });
    expect(br.getParams().maxIter).toBe(200);
    br.setParams({ maxIter: 100 });
    expect(br.getParams().maxIter).toBeGreaterThanOrEqual(100);
  });

  it("fitIntercept false", () => {
    const br = new BayesianRidge({ fitIntercept: false });
    br.fit(Xr, yr);
    expect(br.predict(Xr).shape).toEqual([8]);
  });
});

// ---- LogisticRegression ----
describe("LogisticRegression branches", () => {
  it("fit and predict", () => {
    const lr = new LogisticRegression({ maxIter: 200 });
    lr.fit(Xc, yc);
    const pred = lr.predict(Xc);
    expect(pred.shape).toEqual([8]);
  });

  it("predictProba", () => {
    const lr = new LogisticRegression({ maxIter: 200 });
    lr.fit(Xc, yc);
    const proba = lr.predictProba(Xc);
    expect(proba.shape[0]).toBe(8);
  });

  it("score", () => {
    const lr = new LogisticRegression({ maxIter: 200 });
    lr.fit(Xc, yc);
    expect(lr.score(Xc, yc)).toBeGreaterThanOrEqual(0.5);
  });

  it("multiclass", () => {
    const ym = tensor([0, 0, 1, 1, 2, 2, 2, 1]);
    const lr = new LogisticRegression({ maxIter: 200 });
    lr.fit(Xc, ym);
    const pred = lr.predict(Xc);
    expect(pred.shape).toEqual([8]);
  });

  it("penalty l1 with liblinear solver", () => {
    const lr = new LogisticRegression({
      penalty: "l1",
      solver: "liblinear",
      maxIter: 200,
      C: 1.0,
    });
    lr.fit(Xc, yc);
    expect(lr.predict(Xc).shape).toEqual([8]);
  });

  it("getParams/setParams", () => {
    const lr = new LogisticRegression({ maxIter: 200 });
    expect(lr.getParams().maxIter).toBe(200);
    lr.setParams({ maxIter: 100 });
    expect(lr.getParams().maxIter).toBe(100);
  });
});
