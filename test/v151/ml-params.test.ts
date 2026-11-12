import { describe, expect, it } from "vitest";
import * as ml from "../../src/ml";

type Estimator = {
  getParams?(): Record<string, unknown>;
  setParams?(params: Record<string, unknown>): unknown;
};
type Ctor = new (options?: Record<string, unknown>) => Estimator;

const base = new ml.LogisticRegression();
const reg = new ml.Ridge();
const REQUIRED: Record<string, Record<string, unknown>> = {
  CalibratedClassifierCV: { estimator: base },
  StackingClassifier: { estimators: [base] },
  StackingRegressor: { estimators: [reg] },
  VotingClassifier: { estimators: [base] },
  VotingRegressor: { estimators: [reg] },
  OneVsRestClassifier: { estimator: base },
  OneVsOneClassifier: { estimator: base },
  SelfTrainingClassifier: { baseEstimator: base },
};
const NOT_ESTIMATORS = new Set([
  "Pipeline",
  "FeatureUnion",
  "ColumnTransformer",
  "KDTree",
  "BallTree",
  "GridSearchCV",
  "RandomizedSearchCV",
]);

const classes = Object.entries(ml)
  .filter(
    ([name, value]) =>
      /^[A-Z]/.test(name) &&
      typeof value === "function" &&
      !NOT_ESTIMATORS.has(name) &&
      !/Error$/.test(name)
  )
  .map(([name, value]) => [name, value as unknown as Ctor] as const);

describe("ML-4: setParams(getParams()) is a no-op round trip", () => {
  for (const [name, C] of classes) {
    it(name, () => {
      const m = new C(REQUIRED[name] ?? {});
      if (typeof m.getParams !== "function" || typeof m.setParams !== "function") return;
      const before = m.getParams();
      expect(() => m.setParams?.(before)).not.toThrow();
      expect(m.getParams()).toEqual(before);
    });
  }

  it("changing the estimators through setParams is still rejected", () => {
    const vote = new ml.VotingClassifier({ estimators: [base] });
    expect(() => vote.setParams({ estimators: [new ml.GaussianNB()] })).toThrow(/estimators/);
    expect(() => vote.setParams({ nEstimators: 3 })).toThrow(/nEstimators/);
    const stack = new ml.StackingRegressor({ estimators: [reg], finalEstimator: new ml.Ridge() });
    expect(() => stack.setParams({ nEstimators: 3 })).toThrow(/nEstimators/);
    const cal = new ml.CalibratedClassifierCV({ estimator: base });
    expect(() => cal.setParams({ estimator: new ml.GaussianNB() })).toThrow(/estimator/);
  });

  it("the search wrappers round trip too", () => {
    const grid = new ml.GridSearchCV(new ml.Ridge(), { alpha: [1, 2] }, { cv: 3 });
    const gp = grid.getParams();
    expect(() => grid.setParams(gp)).not.toThrow();
    expect(grid.getParams()).toEqual(gp);
    const rand = new ml.RandomizedSearchCV(new ml.Ridge(), { alpha: [1, 2] }, { cv: 3, nIter: 2 });
    const rp = rand.getParams();
    expect(() => rand.setParams(rp)).not.toThrow();
    expect(rand.getParams()).toEqual(rp);
  });
});
