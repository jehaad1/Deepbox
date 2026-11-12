import { describe, expect, it } from "vitest";
import { InvalidParameterError } from "../../src/core";
import * as ml from "../../src/ml";
import { tensor } from "../../src/ndarray";

type Ctor = new (
  options?: Record<string, unknown>
) => {
  getParams?(): Record<string, unknown>;
};

// Required constructor options of the meta-estimators.
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

// Exported classes whose constructor does not take an options object.
const NO_OPTIONS = new Set([
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
      !NO_OPTIONS.has(name) &&
      !/Error$/.test(name)
  )
  .map(([name, value]) => [name, value as unknown as Ctor] as const);

describe("ML-3: constructors reject unknown options", () => {
  it("covers a realistic number of estimators", () => {
    expect(classes.length).toBeGreaterThan(70);
  });

  for (const [name, C] of classes) {
    it(`${name} rejects an unknown option and names it`, () => {
      const required = REQUIRED[name] ?? {};
      expect(() => new C({ ...required, __bogus: 1 })).toThrow(InvalidParameterError);
      expect(() => new C({ ...required, __bogus: 1 })).toThrow(/__bogus/);
    });

    it(`${name} accepts its own getParams() output`, () => {
      const required = REQUIRED[name] ?? {};
      const m = new C(required);
      if (typeof m.getParams !== "function") return;
      // The meta-estimators report a derived nEstimators that is not a constructor option.
      const { nEstimators: _derived, ...params } = m.getParams();
      expect(() => new C({ ...required, ...params })).not.toThrow();
    });
  }

  it("suggests the camelCase name for a snake_case option", () => {
    expect(() => new ml.KMeans({ n_clusters: 3 } as never)).toThrow(/did you mean "nClusters"/);
    expect(() => new ml.RandomForestClassifier({ n_estimators: 3 } as never)).toThrow(
      /did you mean "nEstimators"/
    );
  });

  it("suggests a near miss for a typo and lists the options otherwise", () => {
    expect(() => new ml.KMeans({ nCluster: 3 } as never)).toThrow(/did you mean "nClusters"/);
    expect(() => new ml.KMeans({ zzzzzzzz: 3 } as never)).toThrow(/valid options are/);
  });

  it("also checks the option object of the search wrappers", () => {
    expect(() => new ml.GridSearchCV(new ml.Ridge(), { alpha: [1] }, { cvv: 3 } as never)).toThrow(
      InvalidParameterError
    );
    expect(
      () => new ml.RandomizedSearchCV(new ml.Ridge(), { alpha: [1] }, { n_iter: 3 } as never)
    ).toThrow(/did you mean "nIter"/);
  });

  it("also checks the options of ColumnTransformer", () => {
    expect(() => new ml.ColumnTransformer([], { remaindr: "drop" } as never)).toThrow();
  });
});

describe("ML-13: TSNE accepts maxIter as an alias of nIter", () => {
  const X = tensor(
    Array.from({ length: 12 }, (_, i) => [Math.sin(i), Math.cos(i * 0.7), i * 0.1]),
    { dtype: "float64" }
  );

  it("constructor maxIter equals nIter", () => {
    const a = new ml.TSNE({ maxIter: 250, perplexity: 3, randomState: 1 });
    const b = new ml.TSNE({ nIter: 250, perplexity: 3, randomState: 1 });
    expect(a.getParams()["nIter"]).toBe(250);
    expect(a.fitTransform(X).toArray()).toEqual(b.fitTransform(X).toArray());
  });

  it("setParams maxIter updates nIter and does not leak into getParams", () => {
    const t = new ml.TSNE({ nIter: 300, perplexity: 3 });
    t.setParams({ maxIter: 260 });
    expect(t.getParams()["nIter"]).toBe(260);
    expect("maxIter" in t.getParams()).toBe(false);
  });

  it("conflicting maxIter and nIter are rejected", () => {
    expect(() => new ml.TSNE({ maxIter: 250, nIter: 300 })).toThrow(InvalidParameterError);
    expect(() => new ml.TSNE().setParams({ maxIter: 250, nIter: 300 })).toThrow(
      InvalidParameterError
    );
  });
});
