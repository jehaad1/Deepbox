import { describe, expect, it } from "vitest";
import { InvalidParameterError, NotFittedError } from "../../src/core";
import { crossValScore, LinearRegression, LogisticRegression, Pipeline } from "../../src/ml";
import { cloneEstimator } from "../../src/ml/_internal";
import { tensor } from "../../src/ndarray";
import * as P from "../../src/preprocess";

// Every exported class of deepbox/preprocess that has a fit method must be copyable by the helper
// that Pipeline.clone, crossValScore and GridSearchCV use.

type Case = {
  make: () => object;
  /** Parameters that differ from the defaults; the clone must carry them over. */
  nonDefault?: Record<string, unknown>;
  /** A parameter object that the constructor and setParams must both reject. */
  invalid?: Record<string, unknown>;
};

const cases: Record<string, Case> = {
  KBinsDiscretizer: {
    make: () => new P.KBinsDiscretizer({ nBins: 3, strategy: "uniform" }),
    nonDefault: { nBins: 4 },
    invalid: { nBins: 1 },
  },
  LabelEncoder: { make: () => new P.LabelEncoder() },
  OneHotEncoder: {
    make: () => new P.OneHotEncoder({ handleUnknown: "ignore" }),
    nonDefault: { drop: "first" },
    invalid: { handleUnknown: "nope" },
  },
  OrdinalEncoder: {
    make: () => new P.OrdinalEncoder({ handleUnknown: "useEncodedValue", unknownValue: -2 }),
    nonDefault: { unknownValue: -5 },
    invalid: { handleUnknown: "nope" },
  },
  LabelBinarizer: {
    make: () => new P.LabelBinarizer({ posLabel: 2 }),
    nonDefault: { negLabel: -1 },
    invalid: { posLabel: -3 },
  },
  MultiLabelBinarizer: {
    make: () => new P.MultiLabelBinarizer({ classes: ["a", "b"] }),
    nonDefault: { sparse: true },
    invalid: { sparse: "yes" },
  },
  TargetEncoder: {
    make: () => new P.TargetEncoder({ smooth: 2 }),
    nonDefault: { cv: 3 },
    invalid: { cv: 1 },
  },
  StandardScaler: { make: () => new P.StandardScaler(), nonDefault: { withMean: false } },
  MinMaxScaler: { make: () => new P.MinMaxScaler(), nonDefault: { featureRange: [0, 2] } },
  MaxAbsScaler: { make: () => new P.MaxAbsScaler() },
  RobustScaler: { make: () => new P.RobustScaler(), nonDefault: { withCentering: false } },
  Normalizer: { make: () => new P.Normalizer(), nonDefault: { norm: "l1" } },
  QuantileTransformer: { make: () => new P.QuantileTransformer(), nonDefault: { nQuantiles: 50 } },
  PowerTransformer: { make: () => new P.PowerTransformer(), nonDefault: { standardize: false } },
  PolynomialFeatures: { make: () => new P.PolynomialFeatures(), nonDefault: { degree: 3 } },
  Binarizer: { make: () => new P.Binarizer(), nonDefault: { threshold: 1 } },
  FunctionTransformer: { make: () => new P.FunctionTransformer() },
  SplineTransformer: { make: () => new P.SplineTransformer(), nonDefault: { degree: 2 } },
  SimpleImputer: { make: () => new P.SimpleImputer(), nonDefault: { strategy: "median" } },
  KNNImputer: { make: () => new P.KNNImputer(), nonDefault: { nNeighbors: 3 } },
  MissingIndicator: { make: () => new P.MissingIndicator() },
  VarianceThreshold: { make: () => new P.VarianceThreshold(), nonDefault: { threshold: 0.5 } },
  SelectKBest: { make: () => new P.SelectKBest({ k: 2 }), nonDefault: { k: 3 } },
  SelectFromModel: { make: () => new P.SelectFromModel({ estimator: new LinearRegression() }) },
  RFE: { make: () => new P.RFE({ estimator: new LinearRegression() }) },
  CountVectorizer: { make: () => new P.CountVectorizer(), nonDefault: { lowercase: false } },
  TfidfVectorizer: { make: () => new P.TfidfVectorizer(), nonDefault: { lowercase: false } },
  HashingVectorizer: { make: () => new P.HashingVectorizer(), nonDefault: { nFeatures: 64 } },
  RFECV: { make: () => new P.RFECV({ estimator: new LinearRegression() }) },
};

type Cloneable = {
  getParams(): Record<string, unknown>;
  setParams(params: Record<string, unknown>): unknown;
  clone?(): unknown;
};

/** Parameters with nested estimators replaced by their own parameters, so two copies compare equal. */
function comparableParams(est: Cloneable): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(est.getParams())) {
    out[key] =
      typeof value === "object" && value !== null && "getParams" in value
        ? comparableParams(value as Cloneable)
        : value;
  }
  return out;
}

describe("preprocess estimators can be cloned", () => {
  const fitted = Object.entries(P)
    .filter(
      ([, value]) => typeof value === "function" && typeof value.prototype?.fit === "function"
    )
    .map(([name]) => name);

  it("the case table covers every exported class with fit", () => {
    expect(Object.keys(cases).sort()).toEqual(fitted.sort());
  });

  for (const [name, spec] of Object.entries(cases)) {
    describe(name, () => {
      it("has getParams, setParams and clone", () => {
        const est = spec.make() as Cloneable;
        expect(typeof est.getParams).toBe("function");
        expect(typeof est.setParams).toBe("function");
        expect(typeof est.clone).toBe("function");
      });

      it("cloneEstimator returns an unfitted copy with the same parameters", () => {
        const est = spec.make() as Cloneable;
        const copy = cloneEstimator(est, "test") as Cloneable;
        expect(copy).not.toBe(est);
        expect(copy.constructor).toBe(est.constructor);
        expect(comparableParams(copy)).toEqual(comparableParams(est));
      });

      if (spec.nonDefault !== undefined) {
        const overrides = spec.nonDefault;
        it("cloneEstimator applies overrides without touching the original", () => {
          const est = spec.make() as Cloneable;
          const before = est.getParams();
          const copy = cloneEstimator(est, "test", overrides) as Cloneable;
          expect(est.getParams()).toEqual(before);
          for (const [key, value] of Object.entries(overrides)) {
            expect(copy.getParams()[key]).toEqual(value);
          }
        });
      }

      it("setParams rejects an unknown parameter", () => {
        const est = spec.make() as Cloneable;
        expect(() => est.setParams({ definitelyNotAParam: 1 })).toThrow(InvalidParameterError);
      });

      if (spec.invalid !== undefined) {
        const invalid = spec.invalid;
        it("setParams validates values and leaves the estimator unchanged on failure", () => {
          const est = spec.make() as Cloneable;
          const before = est.getParams();
          expect(() => est.setParams(invalid)).toThrow(InvalidParameterError);
          expect(est.getParams()).toEqual(before);
        });
      }
    });
  }

  it("a fitted encoder clones to an unfitted one", () => {
    const enc = new P.LabelEncoder().fit(["a", "b"]);
    const copy = cloneEstimator(enc, "test");
    expect(() => copy.transform(["a"])).toThrow(NotFittedError);
    expect(copy.fit(["x", "y"]).transform(["y"]).toArray()).toEqual([1]);

    const ohe = new P.OneHotEncoder().fit([["a"], ["b"]]);
    expect(() => cloneEstimator(ohe, "test").transform([["a"]])).toThrow(NotFittedError);

    const kb = new P.KBinsDiscretizer().fit(tensor([[1], [2], [3], [4], [5], [6]]));
    expect(() => cloneEstimator(kb, "test").transform(tensor([[1]]))).toThrow(NotFittedError);
  });

  it("setParams on a fitted encoder discards the fitted state", () => {
    const ohe = new P.OneHotEncoder().fit([["a"], ["b"]]);
    ohe.setParams({ handleUnknown: "ignore" });
    expect(() => ohe.transform([["a"]])).toThrow(NotFittedError);
    expect(ohe.getParams().handleUnknown).toBe("ignore");
  });

  it("a Pipeline with a KBinsDiscretizer step runs under crossValScore", () => {
    const rows: number[][] = [];
    const labels: number[] = [];
    for (let i = 0; i < 20; i++) {
      rows.push([i, (i * 7) % 5]);
      labels.push(i < 10 ? 0 : 1);
    }
    const pipe = new Pipeline([
      ["bins", new P.KBinsDiscretizer({ nBins: 3 })],
      ["clf", new LogisticRegression()],
    ]);
    const scores = crossValScore(pipe, tensor(rows), tensor(labels), 4);
    expect(scores.length).toBe(4);
  });
});
