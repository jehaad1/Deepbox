import { describe, expect, it } from "vitest";
import {
  DataValidationError,
  DTypeError,
  InvalidParameterError,
  MemoryError,
  NotFittedError,
  ShapeError,
} from "../../src/core";
import { LinearRegression, LogisticRegression, RandomForestClassifier, Ridge } from "../../src/ml";
import { Tensor, tensor, transpose, zeros } from "../../src/ndarray";
import {
  Binarizer,
  FunctionTransformer,
  f_classif,
  f_regression,
  KNNImputer,
  MissingIndicator,
  mutual_info_classif,
  mutual_info_regression,
  PolynomialFeatures,
  RFE,
  RFECV,
  SelectFromModel,
  SelectKBest,
  SimpleImputer,
  VarianceThreshold,
} from "../../src/preprocess";

const f64 = (data: number[] | number[][]): Tensor => tensor(data, { dtype: "float64" });

function expectClose(actual: readonly number[], expected: readonly number[], tol = 1e-9): void {
  expect(actual).toHaveLength(expected.length);
  for (let i = 0; i < expected.length; i++) {
    const a = actual[i] as number;
    const e = expected[i] as number;
    if (Number.isNaN(e)) expect(Number.isNaN(a)).toBe(true);
    else expect(Math.abs(a - e)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(e)));
  }
}

// Reference values computed with scikit-learn 1.8 / NumPy 2.4 (see comments per test).
const FX = {
  X: [
    [-0.802, -1.324, -0.248, 0.42, 1.136],
    [0.11, -0.553, -0.785, 0.749, 1.635],
    [0.273, -1.233, -0.958, 1.6, 0.203],
    [-1.732, -0.084, -1.163, -0.629, -0.488],
    [-0.713, 0.553, -0.063, -0.589, 0.41],
    [0.83, -1.643, -0.257, -0.981, -0.173],
    [-1.289, 0.021, -0.038, -0.304, -1.048],
    [-0.396, -1.091, -1.355, 0.225, -1.109],
    [1.17, 0.717, -1.998, 0.272, -1.102],
    [0.033, 0.044, -1.988, -0.233, -0.256],
    [0.962, -1.181, 0.738, -1.099, -0.331],
    [-0.84, 1.449, 0.568, 2.432, 0.642],
    [0.845, 0.841, -0.607, -0.07, 1.35],
    [-0.397, 0.189, -0.021, 0.609, -0.365],
    [-0.152, 0.242, 0.103, -0.865, 0.896],
    [-1.298, -1.201, -1.282, 0.967, -0.361],
    [-0.971, -1.136, 0.421, -1.055, -1.272],
    [0.614, -1.197, -0.322, -0.007, -0.445],
    [-0.054, 1.339, -0.517, -1.259, -1.837],
    [-0.205, -0.352, 0.265, -0.464, -0.479],
    [-0.721, -0.52, 0.16, -0.38, 0.1],
    [1.901, 0.479, -1.576, 1.734, 0.348],
    [-0.941, 0.907, 0.018, -0.615, -0.634],
    [-0.993, 0.048, 1.069, -0.325, 0.421],
  ],
  y: [
    -0.902, 1.388, 2.837, -2.929, -2.241, 1.76, -3.292, 1.263, 5.871, 2.988, 1.075, -1.89, 3.256,
    -0.579, -0.734, -0.387, -3.294, 2.054, 0.35, -1.371, -2.296, 8.048, -2.436, -3.751,
  ],
  yc: [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 2, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1],
  fc: [
    11.254548125909674, 0.6274583141335014, 1.3369583485722971, 0.7087291455480678,
    0.04758565777160346,
  ],
  fr: [
    84.77994484202189, 0.002546909184322842, 19.598641879530195, 3.3488863231626875,
    0.20560295016228639,
  ],
  rfeA: [1, 4, 1, 2, 3],
  rfeB: [1, 3, 2, 2, 3],
  rfeDefault: [1, 4, 1, 2, 3],
  rfeFrac: [1, 2, 1, 1, 2],
  cvA: {
    n: 3,
    ranking: [1, 3, 1, 1, 2],
    mean: [
      0.9822957753059924, 0.983025636960617, 0.9865373046216278, 0.9535046618775563,
      0.6960352915631404,
    ],
    std: [
      0.011632444926114057, 0.011437437246233905, 0.010196540566349095, 0.020666896380099047,
      0.084316381606536,
    ],
  },
  cvB: {
    n: 3,
    ranking: [1, 2, 1, 1, 2],
    mean: [-0.12160414269445496, -0.0886557765222156, -0.32455072181099076],
  },
  strat_y: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
  strat_folds: [
    [0, 1, 2, 9, 16, 17],
    [3, 4, 10, 11, 18],
    [5, 6, 12, 13, 19],
    [7, 8, 14, 15, 20],
  ],
  strat_y2: [0, 1, 2, 0, 0, 1, 2, 2, 1, 0, 1, 0, 2, 0, 1, 2, 0, 1],
  strat_folds2: [
    [0, 1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10, 11],
    [12, 13, 14, 15, 16, 17],
  ],
  kfold_folds: [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7],
    [8, 9],
  ],
};

/** Mock with fixed importances per original column, found through the first row of X. */
class KeyedImportance {
  featureImportances_: number[] | undefined;
  constructor(
    private readonly importances: number[],
    private readonly keys: number[]
  ) {}
  fit(X: Tensor, _y: Tensor): this {
    const first = (X.toArray() as number[][])[0] ?? [];
    this.featureImportances_ = first.map((v) => this.importances[this.keys.indexOf(v)] ?? 0);
    return this;
  }
}

describe("c53 VarianceThreshold", () => {
  it("computes the variance without E[x^2] - E[x]^2 cancellation", () => {
    // np.var([1e9, 1e9 + 1, 1e9 + 2]) == 2/3. The one-pass formula returns 0 here.
    const X = f64([[1e9], [1e9 + 1], [1e9 + 2]]);
    const vt = new VarianceThreshold().fit(X);
    expect(Math.abs((vt.variances[0] as number) - 2 / 3)).toBeLessThan(1e-9);
    expect(vt.getSupport()).toEqual([true]);
  });

  it("removes a constant column even when its mean is not exact", () => {
    // 0.1 * 3 / 3 is not exactly 0.1, so the naive variance is tiny but non-zero.
    const X = f64([
      [0.1, 1],
      [0.1, 2],
      [0.1, 4],
    ]);
    const vt = new VarianceThreshold().fit(X);
    expect(vt.variances[0]).toBe(0);
    expect(vt.getSupport()).toEqual([false, true]);
  });

  it("ignores NaN like np.nanvar and returns NaN for an all-NaN column", () => {
    const X = f64([
      [1, Number.NaN],
      [2, Number.NaN],
      [4, Number.NaN],
    ]);
    const vt = new VarianceThreshold().fit(X);
    // np.nanvar([1, 2, 4]) = 1.5555555555555556
    expectClose([vt.variances[0] as number], [1.5555555555555556]);
    expect(Number.isNaN(vt.variances[1])).toBe(true);
    expect(vt.getSupport()).toEqual([true, false]);
    const withNan = f64([[1], [Number.NaN], [3]]);
    // np.nanvar([1, 3]) = 1
    expect(new VarianceThreshold().fit(withNan).variances[0]).toBe(1);
  });

  it("rejects infinity and string input", () => {
    expect(() => new VarianceThreshold().fit(f64([[1], [Number.POSITIVE_INFINITY]]))).toThrow(
      DataValidationError
    );
    expect(() => new VarianceThreshold().fit(tensor([["a"], ["b"]]))).toThrow(DTypeError);
  });

  it("keeps the dtype of the input and supports inverseTransform and index support", () => {
    const X = tensor(
      [
        [1, 5, 2],
        [2, 5, 4],
        [3, 5, 8],
      ],
      { dtype: "int32" }
    );
    const vt = new VarianceThreshold();
    const Xt = vt.fitTransform(X);
    expect(Xt.dtype).toBe("int32");
    expect(Xt.toArray()).toEqual([
      [1, 2],
      [2, 4],
      [3, 8],
    ]);
    expect(vt.getSupport(true)).toEqual([0, 2]);
    expect(vt.nFeaturesIn).toBe(3);
    const back = vt.inverseTransform(Xt);
    expect(back.toArray()).toEqual([
      [1, 0, 2],
      [2, 0, 4],
      [3, 0, 8],
    ]);
    const f32 = new VarianceThreshold().fitTransform(
      tensor(
        [
          [1, 2],
          [3, 2],
        ],
        { dtype: "float32" }
      )
    );
    expect(f32.dtype).toBe("float32");
  });

  it("works on non-contiguous views", () => {
    const base = f64([
      [1, 2, 3],
      [9, 9, 9],
      [7, 7, 7],
      [3, 8, 1],
    ]);
    const view = transpose(base); // shape [3, 4], strides [1, 3]
    expect(view.strides).toEqual([1, 3]);
    const vt = new VarianceThreshold().fit(view);
    expect(vt.getSupport()).toEqual([true, false, false, true]);
    expect(vt.transform(view).toArray()).toEqual([
      [1, 3],
      [2, 8],
      [3, 1],
    ]);
  });

  it("setParams re-applies the threshold to fitted variances", () => {
    const X = f64([
      [0, 0],
      [1, 10],
      [2, 20],
    ]);
    const vt = new VarianceThreshold().fit(X);
    expect(vt.getSupport()).toEqual([true, true]);
    vt.setParams({ threshold: 1 });
    expect(vt.getSupport()).toEqual([false, true]);
    expect(vt.getParams()).toEqual({ threshold: 1 });
    expect(() => vt.setParams({ nope: 1 })).toThrow(InvalidParameterError);
    expect(() => vt.setParams({ threshold: -1 })).toThrow(InvalidParameterError);
    expect(vt.clone().getParams()).toEqual({ threshold: 1 });
  });
});

describe("c53 f_classif / f_regression", () => {
  const X = f64(FX.X);
  it("match sklearn.feature_selection.f_classif", () => {
    expectClose(f_classif(X, f64(FX.yc)), FX.fc, 1e-10);
  });
  it("match sklearn.feature_selection.f_regression", () => {
    expectClose(f_regression(X, f64(FX.y)), FX.fr, 1e-10);
  });
  it("read y through its strides", () => {
    const strided = (values: number[]): Tensor =>
      Tensor.fromTypedArray({
        data: Float64Array.from(values.flatMap((v) => [v, 99])),
        shape: [values.length],
        strides: [2],
        dtype: "float64",
        device: "cpu",
      });
    expectClose(f_classif(X, strided(FX.yc)), FX.fc, 1e-10);
    expectClose(f_regression(X, strided(FX.y)), FX.fr, 1e-10);
    // A transposed X view gives the same scores as the contiguous matrix.
    const Xt = transpose(
      f64(FX.X[0].map((_: number, j: number) => FX.X.map((r: number[]) => r[j])))
    );
    expectClose(f_regression(Xt, f64(FX.y)), FX.fr, 1e-10);
  });
  it("validate y length, shape, dtype and finiteness", () => {
    expect(() => f_classif(X, f64([0, 1]))).toThrow(ShapeError);
    expect(() => f_regression(X, f64([[1, 2]]))).toThrow(ShapeError);
    expect(() => f_classif(X, tensor(FX.yc.map(String)))).toThrow(DTypeError);
    const yBad = [...FX.y];
    yBad[3] = Number.NaN;
    expect(() => f_regression(X, f64(yBad))).toThrow(DataValidationError);
    const Xbad = f64([
      [1, Number.NaN],
      [2, 3],
    ]);
    expect(() => f_classif(Xbad, f64([0, 1]))).toThrow(DataValidationError);
  });
  it("accept an [n, 1] target column", () => {
    expectClose(f_regression(X, f64(FX.y.map((v: number) => [v]))), FX.fr, 1e-10);
  });
});

describe("c53 SelectKBest", () => {
  const X = f64([
    [1, 5, 0],
    [2, 6, 0],
    [3, 7, 1],
    [4, 8, 1],
  ]);
  const y = f64([0, 0, 1, 1]);

  it("ranks NaN scores last", () => {
    const skb = new SelectKBest({ scoreFunc: () => [Number.NaN, 1, 2], k: 2 }).fit(X, y);
    expect(skb.getSupport()).toEqual([false, true, true]);
    expect(Number.isNaN(skb.scores[0])).toBe(true);
  });

  it("keeps the lowest index among equal scores", () => {
    const skb = new SelectKBest({ scoreFunc: () => [1, 1, 1], k: 2 }).fit(X, y);
    expect(skb.getSupport(true)).toEqual([0, 1]);
  });

  it('supports k = "all"', () => {
    const skb = new SelectKBest({ k: "all" }).fit(X, y);
    expect(skb.getSupport()).toEqual([true, true, true]);
    expect(() => new SelectKBest({ k: "most" as unknown as "all" })).toThrow(InvalidParameterError);
  });

  it("includes scoreFunc in getParams so a rebuilt selector keeps it", () => {
    const scoreFunc = (): number[] => [0, 0, 5];
    const skb = new SelectKBest({ scoreFunc, k: 1 });
    expect(skb.getParams()).toEqual({ k: 1, scoreFunc });
    const rebuilt = new SelectKBest(skb.getParams() as { k: number });
    expect(rebuilt.fit(X, y).getSupport()).toEqual([false, false, true]);
    expect(skb.clone().fit(X, y).getSupport()).toEqual([false, false, true]);
  });

  it("rejects a scoreFunc that returns the wrong number of scores", () => {
    const skb = new SelectKBest({ scoreFunc: () => [1, 2], k: 1 });
    expect(() => skb.fit(X, y)).toThrow(InvalidParameterError);
  });

  it("setParams applies k to the fitted scores", () => {
    const skb = new SelectKBest({ scoreFunc: () => [3, 1, 2], k: 1 }).fit(X, y);
    expect(skb.getSupport()).toEqual([true, false, false]);
    skb.setParams({ k: 2 });
    expect(skb.getSupport()).toEqual([true, false, true]);
    skb.setParams({ scoreFunc: () => [0, 0, 0] });
    expect(() => skb.getSupport()).toThrow(NotFittedError);
  });

  it("keeps the input dtype and offers inverseTransform", () => {
    const Xi = tensor(
      [
        [1, 5, 0],
        [2, 6, 0],
      ],
      { dtype: "int32" }
    );
    const skb = new SelectKBest({ scoreFunc: () => [0, 2, 1], k: 2 }).fit(Xi, f64([0, 1]));
    const Xt = skb.transform(Xi);
    expect(Xt.dtype).toBe("int32");
    expect(Xt.toArray()).toEqual([
      [5, 0],
      [6, 0],
    ]);
    expect(skb.inverseTransform(Xt).toArray()).toEqual([
      [0, 5, 0],
      [0, 6, 0],
    ]);
  });
});

describe("c53 SelectFromModel", () => {
  it("works with the library's own estimators", () => {
    const X = f64(FX.X);
    const rf = new RandomForestClassifier({ nEstimators: 10, randomState: 0 });
    const sel = new SelectFromModel({ estimator: rf, threshold: "mean" }).fit(X, f64(FX.yc));
    expect(sel.getSupport()).toHaveLength(5);
    expect(sel.importances).toHaveLength(5);
    // Feature 0 and 2 generate the labels.
    expect(sel.getSupport()[0]).toBe(true);
    const lr = new SelectFromModel({ estimator: new LogisticRegression() }).fit(
      X,
      f64(FX.yc.map((v: number) => (v > 0 ? 1 : 0)))
    );
    expect(lr.getSupport().some(Boolean)).toBe(true);
  });

  it("does not modify the estimator it was given", () => {
    const est = new LinearRegression();
    const sel = new SelectFromModel({ estimator: est }).fit(f64(FX.X), f64(FX.y));
    expect(() => est.coef).toThrow();
    expect(sel.fittedEstimator).not.toBe(est);
    expect((sel.fittedEstimator as LinearRegression).coef.shape).toEqual([5]);
  });

  it("sums |coef| over targets for 2D coefficients", () => {
    class Multi {
      coef_ = tensor(
        [
          [1, -2, 0],
          [1, 2, 0.5],
        ],
        { dtype: "float64" }
      );
      fit(): this {
        return this;
      }
    }
    const sel = new SelectFromModel({ estimator: new Multi(), threshold: 0.8 });
    sel.fit(f64([[0, 0, 0]]), f64([0]));
    // sum |coef| over targets = [2, 4, 0.5]
    expect(sel.importances).toEqual([2, 4, 0.5]);
    expect(sel.getSupport()).toEqual([true, true, false]);
  });

  it("rejects estimators with mismatched or NaN importances", () => {
    class Short {
      featureImportances_ = [1, 2];
      fit(): this {
        return this;
      }
    }
    class Nan {
      featureImportances_ = [1, Number.NaN, 2];
      fit(): this {
        return this;
      }
    }
    const X = f64([[1, 2, 3]]);
    expect(() => new SelectFromModel({ estimator: new Short() }).fit(X, f64([0]))).toThrow(
      InvalidParameterError
    );
    expect(() => new SelectFromModel({ estimator: new Nan() }).fit(X, f64([0]))).toThrow(
      DataValidationError
    );
    expect(() => new SelectFromModel({ estimator: { fit: () => 0 } }).fit(X, f64([0]))).toThrow(
      InvalidParameterError
    );
  });

  it("validates threshold values and keeps string columns intact", () => {
    const est = { featureImportances_: [0.9, 0.1], fit: () => 0 };
    expect(() => new SelectFromModel({ estimator: est, threshold: Number.NaN })).toThrow(
      InvalidParameterError
    );
    expect(
      () => new SelectFromModel({ estimator: est, threshold: "max" as unknown as "mean" })
    ).toThrow(InvalidParameterError);
    const Xs = tensor([
      ["a", "b"],
      ["c", "d"],
    ]);
    const sel = new SelectFromModel({ estimator: est, threshold: 0.5 }).fit(Xs, f64([0, 1]));
    expect(sel.transform(Xs).toArray()).toEqual([["a"], ["c"]]);
  });

  it("includes the estimator in getParams, and setParams re-thresholds", () => {
    const est = { featureImportances_: [0.9, 0.5, 0.1], fit: () => 0 };
    const sel = new SelectFromModel({ estimator: est, threshold: 0.4 });
    expect(sel.getParams()["estimator"]).toBe(est);
    sel.fit(f64([[1, 2, 3]]), f64([0]));
    expect(sel.getSupport()).toEqual([true, true, false]);
    sel.setParams({ threshold: 0.7 });
    expect(sel.getSupport()).toEqual([true, false, false]);
    sel.setParams({ threshold: 0, maxFeatures: 1 });
    expect(sel.getSupport()).toEqual([true, false, false]);
  });
});

describe("c53 RFE", () => {
  const X = f64(FX.X);
  const y = f64(FX.y);

  it("assigns the largest rank to the feature eliminated first (sklearn ranking_)", () => {
    const X4 = f64([
      [1, 10, 100, 1000],
      [2, 20, 200, 2000],
    ]);
    const est = new KeyedImportance([0.4, 0.1, 0.3, 0.2], [1, 10, 100, 1000]);
    const rfe = new RFE({ estimator: est, nFeaturesToSelect: 1 }).fit(X4, f64([0, 1]));
    // sklearn: [1, 4, 2, 3]
    expect(rfe.ranking).toEqual([1, 4, 2, 3]);
    expect(rfe.getSupport()).toEqual([true, false, false, false]);
    expect(rfe.nFeatures).toBe(1);
  });

  it("matches sklearn.feature_selection.RFE with LinearRegression", () => {
    expect(
      new RFE({ estimator: new LinearRegression(), nFeaturesToSelect: 2 }).fit(X, y).ranking
    ).toEqual(FX.rfeA);
    expect(
      new RFE({ estimator: new LinearRegression(), nFeaturesToSelect: 1, step: 2 }).fit(X, y)
        .ranking
    ).toEqual(FX.rfeB);
  });

  it("selects half of the features by default and accepts fractions", () => {
    // sklearn: n_features_to_select=None keeps n // 2 features (here 2 of 5).
    const def = new RFE({ estimator: new LinearRegression() }).fit(X, y);
    expect(def.ranking).toEqual(FX.rfeDefault);
    expect(def.nFeatures).toBe(2);
    const frac = new RFE({
      estimator: new LinearRegression(),
      nFeaturesToSelect: 0.6,
      step: 0.4,
    }).fit(X, y);
    expect(frac.ranking).toEqual(FX.rfeFrac);
    expect(() => new RFE({ estimator: new LinearRegression(), nFeaturesToSelect: 0 })).toThrow(
      InvalidParameterError
    );
    expect(() => new RFE({ estimator: new LinearRegression(), step: 1.5 })).toThrow(
      InvalidParameterError
    );
  });

  it("works with tree ensembles and typed linear classifiers", () => {
    const yc = f64(FX.yc);
    const rf = new RFE({
      estimator: new RandomForestClassifier({ nEstimators: 8, randomState: 1 }),
      nFeaturesToSelect: 2,
    }).fit(X, yc);
    expect(rf.getSupport(true)).toHaveLength(2);
    const lr = new RFE({ estimator: new LogisticRegression(), nFeaturesToSelect: 3 }).fit(
      X,
      f64(FX.yc.map((v: number) => (v > 0 ? 1 : 0)))
    );
    expect(lr.getSupport(true)).toHaveLength(3);
    const ridge = new RFE({ estimator: new Ridge(), nFeaturesToSelect: 2 }).fit(X, y);
    expect(ridge.nFeatures).toBe(2);
  });

  it("leaves the given estimator unfitted", () => {
    const est = new LinearRegression();
    new RFE({ estimator: est, nFeaturesToSelect: 2 }).fit(X, y);
    expect(() => est.coef).toThrow();
  });

  it("rejects an estimator whose importances do not match the active columns", () => {
    const est = { featureImportances_: [1, 2, 3, 4, 5], fit: () => 0 };
    expect(() => new RFE({ estimator: est, nFeaturesToSelect: 2 }).fit(X, y)).toThrow(
      InvalidParameterError
    );
  });

  it("keeps the dtype of the input and inverts the selection", () => {
    const Xi = tensor(
      [
        [1, 10, 100],
        [2, 20, 200],
      ],
      { dtype: "int32" }
    );
    const est = new KeyedImportance([0.1, 0.9, 0.5], [1, 10, 100]);
    const rfe = new RFE({ estimator: est, nFeaturesToSelect: 2 });
    const Xt = rfe.fitTransform(Xi, f64([0, 1]));
    expect(Xt.dtype).toBe("int32");
    expect(Xt.toArray()).toEqual([
      [10, 100],
      [20, 200],
    ]);
    expect(rfe.inverseTransform(Xt).toArray()).toEqual([
      [0, 10, 100],
      [0, 20, 200],
    ]);
  });

  it("setParams validates and resets the fit; getParams round-trips the estimator", () => {
    const est = new LinearRegression();
    const rfe = new RFE({ estimator: est, nFeaturesToSelect: 2 }).fit(X, y);
    expect(rfe.getParams()["estimator"]).toBe(est);
    rfe.setParams({ nFeaturesToSelect: 3 });
    expect(() => rfe.ranking).toThrow(NotFittedError);
    expect(rfe.fit(X, y).nFeatures).toBe(3);
    expect(() => rfe.setParams({ step: 0 })).toThrow(InvalidParameterError);
    expect(() => rfe.setParams({ foo: 1 })).toThrow(InvalidParameterError);
    expect(rfe.clone().fit(X, y).nFeatures).toBe(3);
  });
});

describe("c53 RFECV", () => {
  const X = f64(FX.X);
  const y = f64(FX.y);

  it("matches sklearn.feature_selection.RFECV (regressor, default scoring)", () => {
    // sklearn RFECV(LinearRegression(), cv=KFold(4)): scores are R^2 per feature count.
    const r = new RFECV({ estimator: new LinearRegression(), cv: 4 }).fit(X, y);
    expect(r.nFeatures).toBe(FX.cvA.n);
    expect(r.ranking).toEqual(FX.cvA.ranking);
    const grid = r.gridScores;
    const counts = [...grid.keys()];
    expect(counts).toEqual([5, 4, 3, 2, 1]);
    expectClose([...grid.values()], FX.cvA.mean, 1e-9);
    expectClose([...r.gridScoresStd.values()], FX.cvA.std, 1e-6);
  });

  it("supports scoring names, min features and fractional step", () => {
    const r = new RFECV({
      estimator: new LinearRegression(),
      cv: 5,
      minFeaturesToSelect: 2,
      step: 2,
      scoring: "neg_mean_squared_error",
    }).fit(X, y);
    expect(r.nFeatures).toBe(FX.cvB.n);
    expect(r.ranking).toEqual(FX.cvB.ranking);
    expectClose([...r.gridScores.values()], FX.cvB.mean, 1e-8);
  });

  it("accepts a scoring function and rejects unknown scoring names", () => {
    const calls: number[] = [];
    const r = new RFECV({
      estimator: new LinearRegression(),
      cv: 3,
      scoring: (yTrue, yPred) => {
        calls.push(yTrue.size);
        expect(yPred.size).toBe(yTrue.size);
        return -1;
      },
    }).fit(X, y);
    expect(calls.length).toBeGreaterThan(0);
    // Equal scores everywhere: the smallest feature count wins.
    expect(r.nFeatures).toBe(1);
    expect(
      () => new RFECV({ estimator: new LinearRegression(), scoring: "f1" as unknown as "r2" })
    ).toThrow(InvalidParameterError);
  });

  it("does not leak validation data into the elimination ranking", () => {
    // Every fit must see only training rows; row ids are stored in column 0.
    const seen: number[][] = [];
    const est = {
      featureImportances_: undefined as number[] | undefined,
      fit(Xtr: Tensor, _y: Tensor) {
        // Only the first fit of each fold sees all three columns.
        if (Xtr.shape[1] === 3) {
          seen.push((Xtr.toArray() as number[][]).map((row) => row[0] as number));
        }
        this.featureImportances_ = new Array<number>(Xtr.shape[1] as number)
          .fill(0)
          .map((_, i) => i);
        return this;
      },
      predict(Xv: Tensor) {
        return tensor(new Array<number>(Xv.shape[0] as number).fill(0));
      },
    };
    const ids = Array.from({ length: 12 }, (_, i) => [i, i % 3, 1]);
    new RFECV({ estimator: est, cv: 3, minFeaturesToSelect: 2 }).fit(
      f64(ids),
      f64(new Array(12).fill(0))
    );
    // Folds 1-3 hold out rows 0-3, 4-7 and 8-11; the last fit uses all rows.
    expect(seen).toHaveLength(4);
    for (const [fold, held] of [
      [0, [0, 1, 2, 3]],
      [1, [4, 5, 6, 7]],
      [2, [8, 9, 10, 11]],
    ] as const) {
      const rows = seen[fold] as number[];
      expect(rows).toHaveLength(8);
      for (const r of held) expect(rows).not.toContain(r);
    }
    expect(seen[3]).toHaveLength(12);
  });

  it("uses scikit-learn's StratifiedKFold for classifiers with accuracy scoring", () => {
    const labels: number[] = FX.strat_y;
    const n = labels.length;
    const rows = labels.map((_, i) => [i, 0.5]);
    const validations: number[][] = [];
    const est = {
      // Explicit tag, so the test does not depend on how tags are inferred from the interface.
      _getTags: () => ({ estimatorType: "classifier" as const }),
      featureImportances_: [1, 0] as number[],
      fit() {
        return this;
      },
      predict(Xv: Tensor) {
        if (Xv.shape[1] === 2)
          validations.push((Xv.toArray() as number[][]).map((r) => r[0] as number));
        return tensor(new Array<number>(Xv.shape[0] as number).fill(0));
      },
    };
    new RFECV({ estimator: est, cv: 4 }).fit(f64(rows), f64(labels));
    expect(validations.slice(0, 4).map((v) => [...v].sort((a, b) => a - b))).toEqual(
      FX.strat_folds
    );
    expect(n).toBe(21);
  });

  it("uses contiguous KFold blocks otherwise (first n % cv folds get one extra sample)", () => {
    const validations: number[][] = [];
    const est = {
      featureImportances_: [1, 0] as number[],
      fit() {
        return this;
      },
      predict(Xv: Tensor) {
        if (Xv.shape[1] === 2)
          validations.push((Xv.toArray() as number[][]).map((r) => r[0] as number));
        return tensor(new Array<number>(Xv.shape[0] as number).fill(0));
      },
    };
    const rows = Array.from({ length: 10 }, (_, i) => [i, 0.5]);
    new RFECV({ estimator: est, cv: 4, scoring: "accuracy", stratified: false }).fit(
      f64(rows),
      f64(new Array(10).fill(0))
    );
    expect(validations.slice(0, 4)).toEqual(FX.kfold_folds);
  });

  it("validates cv, minFeaturesToSelect and y", () => {
    const est = { featureImportances_: [1], fit: () => 0, predict: () => tensor([0]) };
    expect(() => new RFECV({ estimator: est, cv: 30 }).fit(X, y)).toThrow(InvalidParameterError);
    expect(() => new RFECV({ estimator: est, minFeaturesToSelect: 6 }).fit(X, y)).toThrow(
      InvalidParameterError
    );
    expect(() => new RFECV({ estimator: est, cv: 2 }).fit(X, f64([1, 2, 3]))).toThrow(ShapeError);
    expect(() => new RFECV({ estimator: { fit: () => 0 } as never })).toThrow(
      InvalidParameterError
    );
  });

  it("does not set partial state when fit fails", () => {
    const est = { featureImportances_: [1], fit: () => 0, predict: () => tensor([0]) };
    const r = new RFECV({ estimator: est, cv: 30 });
    expect(() => r.fit(X, y)).toThrow();
    expect(() => r.transform(X)).toThrow(NotFittedError);
  });

  it("clones and round-trips parameters", () => {
    const r = new RFECV({ estimator: new LinearRegression(), cv: 3, scoring: "r2" });
    const params = r.getParams();
    expect(params["scoring"]).toBe("r2");
    expect(r.clone().getParams()["cv"]).toBe(3);
    r.setParams({ cv: 4 });
    expect(r.getParams()["cv"]).toBe(4);
  });
});

describe("c53 mutual information", () => {
  // Data without tied values or tied distances, so the tiny tie-breaking noise cannot matter.
  const Xr = [
    [0.034193, 679.874],
    [1.224721, -255.1535],
    [-0.29797, -263.692],
    [0.569726, -28.032],
    [0.746886, -923.6625],
    [1.566549, -48.216],
    [0.680378, -68.283],
    [-0.379099, 231.555],
    [0.824514, -101.265],
    [-0.152786, 342.8495],
    [-0.870341, -757.192],
    [0.394982, -335.283],
    [-1.920341, -407.027],
    [-0.467598, -596.601],
    [-1.492464, 18.319],
    [0.897249, -116.566],
    [-0.743596, 192.497],
    [0.717236, -150.0055],
    [0.544668, 521.4375],
    [-0.206956, -406.758],
    [0.347651, 123.773],
    [1.098813, -642.2905],
    [-0.661613, -419.0835],
    [-1.734015, 63.2175],
    [0.527804, -369.395],
  ];
  const yr = [
    0.622645, 2.778212, -0.344989, 1.300135, 1.87604, 2.600306, 1.606328, -0.517087, 0.941941,
    -0.16676, -1.840851, 1.102573, -4.016307, -0.942492, -2.847787, 1.443993, -1.247753, 1.392487,
    1.286329, -0.622647, 1.129783, 2.439707, -1.394436, -3.215247, 1.55951,
  ];
  const disc = [1, 1, 0, 2, 2, 0, 0, 0, 1, 0, 1, 2, 1, 2, 1, 1, 2, 1, 2, 0, 2, 2, 2, 0, 0];
  const yc = [2, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 2, 1, 1, 1, 0, 0, 1];
  // sklearn.feature_selection.mutual_info_regression(X, y, n_neighbors=3, random_state=0)
  const refRegression = [1.2933232571185864, 0.04508652085863041];

  it("mutual_info_regression matches scikit-learn", () => {
    expectClose(mutual_info_regression(f64(Xr), f64(yr)), refRegression, 1e-9);
  });

  it("mutual_info_regression does not depend on the scale of y or X", () => {
    // sklearn scales X and y to unit std before the Chebyshev search; without scaling
    // the estimate changes when y is multiplied by 1000.
    expectClose(mutual_info_regression(f64(Xr), f64(yr.map((v) => v * 1000))), refRegression, 1e-9);
    expectClose(
      mutual_info_regression(f64(Xr.map(([a, b]) => [a as number, (b as number) * 1e-4])), f64(yr)),
      refRegression,
      1e-9
    );
  });

  it("mutual_info_regression supports discrete features", () => {
    const X = f64(Xr.map(([a], i) => [a as number, disc[i] as number]));
    // sklearn with discrete_features=[False, True]
    const scores = mutual_info_regression(X, f64(yr), { discreteFeatures: [1] });
    expectClose(scores, [1.2933232571185864, 0.12867113146646036], 1e-9);
    expectClose(mutual_info_regression(X, f64(yr), { discreteFeatures: [false, true] }), scores);
    expect(() => mutual_info_regression(X, f64(yr), { discreteFeatures: [5] })).toThrow(
      InvalidParameterError
    );
    expect(() => mutual_info_regression(X, f64(yr), { discreteFeatures: [true] })).toThrow(
      InvalidParameterError
    );
  });

  it("mutual_info_classif uses the open k-th neighbour radius (Ross estimator)", () => {
    // Brute-force NumPy/scipy digamma implementation of sklearn's _compute_mi_cd with the
    // k-th neighbour distance as an open radius (sklearn's own KD-tree call is
    // rounding-sensitive for 1D Euclidean queries, so it cannot serve as exact reference).
    expectClose(
      mutual_info_classif(f64(Xr), f64(yc), { nNeighbors: 3 }),
      [0.44552389394306147, 0.3175734513687798],
      1e-9
    );
    expectClose(
      mutual_info_classif(f64(Xr), f64(yc), { nNeighbors: 1 }),
      [0.43585137932317863, 0.3786398294351581],
      1e-9
    );
  });

  it("mutual_info_classif ignores samples whose class occurs only once", () => {
    const ys = [...yc];
    ys[0] = 7;
    expectClose(
      mutual_info_classif(f64(Xr), f64(ys), { nNeighbors: 3 }),
      [0.47026928366947307, 0.030301901662449282],
      1e-9
    );
  });

  it("mutual_info_classif scores discrete features from the contingency table", () => {
    const X = f64(Xr.map(([a], i) => [a as number, disc[i] as number]));
    // sklearn.metrics.mutual_info_score(disc, yc) = 0.09343881162160593
    const scores = mutual_info_classif(X, f64(yc), { discreteFeatures: [1] });
    expectClose(scores, [0.44552389394306147, 0.09343881162160593], 1e-9);
  });

  it("is reproducible per seed and validates its options", () => {
    const X = f64(Xr);
    const y = f64(yr);
    expect(mutual_info_regression(X, y, { randomState: 7 })).toEqual(
      mutual_info_regression(X, y, { randomState: 7 })
    );
    expect(() => mutual_info_regression(X, y, { randomState: 1.5 })).toThrow(InvalidParameterError);
    expect(() =>
      mutual_info_regression(
        f64([
          [1, Number.NaN],
          [2, 3],
          [3, 4],
          [4, 5],
        ]),
        f64([1, 2, 3, 4])
      )
    ).toThrow(DataValidationError);
    expect(() =>
      mutual_info_classif(f64([[1], [2], [3], [4]]), f64([0, 1, 0, Number.POSITIVE_INFINITY]))
    ).toThrow(DataValidationError);
    expect(() => mutual_info_classif(tensor([["a"], ["b"]]), f64([0, 1]))).toThrow(DTypeError);
  });

  it("handles strided inputs and constant features", () => {
    const X = transpose(f64([Xr.map((r) => r[0] as number)]));
    const ycol = Tensor.fromTypedArray({
      data: Float64Array.from(yr.flatMap((v) => [v, 99])),
      shape: [25],
      strides: [2],
      dtype: "float64",
      device: "cpu",
    });
    expect(X.strides).toEqual([1, 25]);
    expectClose(mutual_info_regression(X, ycol), [refRegression[0] as number], 1e-9);
    const constant = f64(Array.from({ length: 12 }, (_, i) => [1, i % 2]));
    const scores = mutual_info_classif(constant, f64(Array.from({ length: 12 }, (_, i) => i % 2)));
    expect(scores[0] as number).toBeLessThan(0.1);
    expect(Number.isFinite(scores[1] as number)).toBe(true);
  });
});

describe("c53 SimpleImputer", () => {
  const nan = Number.NaN;
  const Xi = f64([
    [1, nan, nan, 5],
    [3, 4, nan, 5],
    [nan, 4, nan, 7],
    [3, 2, nan, nan],
  ]);

  it("matches sklearn for mean, median and most_frequent (keep_empty_features=True)", () => {
    const mean = new SimpleImputer({ strategy: "mean" });
    expectClose(
      (mean.fitTransform(Xi).toArray() as number[][]).flat(),
      [
        1, 3.3333333333333335, 0, 5, 3, 4, 0, 5, 2.3333333333333335, 4, 0, 7, 3, 2, 0,
        5.666666666666667,
      ]
    );
    expectClose(mean.statistics, [2.3333333333333335, 3.3333333333333335, 0, 5.666666666666667]);
    expectClose(new SimpleImputer({ strategy: "median" }).fit(Xi).statistics, [3, 4, 0, 5]);
    expectClose(new SimpleImputer({ strategy: "most_frequent" }).fit(Xi).statistics, [3, 4, 0, 5]);
  });

  it("drops all-missing columns when keepEmptyFeatures is false", () => {
    const imp = new SimpleImputer({ keepEmptyFeatures: false });
    const out = imp.fitTransform(Xi);
    expect(out.shape).toEqual([4, 3]);
    expectClose(
      (out.toArray() as number[][]).flat(),
      [1, 3.3333333333333335, 5, 3, 4, 5, 2.3333333333333335, 4, 7, 3, 2, 5.666666666666667]
    );
    expect(imp.statistics[2]).toBeNaN();
    // constant strategy always keeps every column
    expect(
      new SimpleImputer({
        strategy: "constant",
        fillValue: 9,
        keepEmptyFeatures: false,
      }).fitTransform(Xi).shape
    ).toEqual([4, 4]);
  });

  it("supports a custom missing value marker", () => {
    const X = f64([
      [1, -1],
      [-1, 3],
      [5, 3],
      [7, -1],
    ]);
    const imp = new SimpleImputer({ strategy: "median", missingValues: -1 });
    // sklearn: statistics_ [5, 3]
    expectClose(imp.fit(X).statistics, [5, 3]);
    expectClose((imp.transform(X).toArray() as number[][]).flat(), [1, 3, 5, 3, 5, 3, 7, 3]);
    expect(() => imp.fit(f64([[nan, 1]]))).toThrow(DataValidationError);
  });

  it("validates input and reports clear errors", () => {
    expect(() => new SimpleImputer().fit(f64([]))).toThrow();
    expect(() => new SimpleImputer().fit(zeros([0, 2]))).toThrow(InvalidParameterError);
    expect(() => new SimpleImputer().fit(tensor([["a"], ["b"]]))).toThrow(DTypeError);
    expect(() => new SimpleImputer({ fillValue: "x" as unknown as number })).toThrow(
      InvalidParameterError
    );
    const imp = new SimpleImputer().fit(Xi);
    expect(() => imp.transform(f64([[1, 2]]))).toThrow(InvalidParameterError);
  });

  it("setParams changes behavior and discards the fit; clone round-trips", () => {
    const imp = new SimpleImputer({ strategy: "mean" }).fit(Xi);
    imp.setParams({ strategy: "median" });
    expect(() => imp.statistics).toThrow(NotFittedError);
    expectClose(imp.fit(Xi).statistics, [3, 4, 0, 5]);
    expect(() => imp.setParams({ strategy: "mode" })).toThrow(InvalidParameterError);
    expect(() => imp.setParams({ unknown: 1 })).toThrow(InvalidParameterError);
    expect(imp.clone().getParams()).toMatchObject({ strategy: "median" });
  });

  it("does not modify its input", () => {
    const X = f64([
      [nan, 1],
      [2, 3],
    ]);
    new SimpleImputer().fitTransform(X);
    expect(Number.isNaN((X.toArray() as number[][])[0]?.[0] as number)).toBe(true);
  });
});

describe("c53 KNNImputer", () => {
  const nan = Number.NaN;

  it("takes donors per column among rows that have the feature (sklearn semantics)", () => {
    const X = f64([
      [1, 2, nan],
      [1.1, 2.1, 10],
      [1.2, nan, 20],
      [5, 6, 30],
      [5.1, 6.1, 40],
      [1.05, 2.05, nan],
    ]);
    // sklearn KNNImputer: rows 0 and 5 get 10, 15, 20 for k = 1, 2, 3
    for (const [k, expected] of [
      [1, 10],
      [2, 15],
      [3, 20],
    ] as const) {
      const out = new KNNImputer({ nNeighbors: k }).fitTransform(X).toArray() as number[][];
      expect(out[0]?.[2]).toBeCloseTo(expected, 12);
      expect(out[5]?.[2]).toBeCloseTo(expected, 12);
    }
  });

  it("uses only exact matches when a donor is at distance 0 (weights='distance')", () => {
    const X = f64([
      [1, 2, 5],
      [1, 2, 7],
      [1, 3, 100],
      [9, 9, 1],
      [1, 2, nan],
    ]);
    // sklearn: [1, 2, 6]
    const out = new KNNImputer({ nNeighbors: 3, weights: "distance" }).fitTransform(X);
    expect((out.toArray() as number[][])[4]).toEqual([1, 2, 6]);
  });

  it("falls back to the column mean for a row with no observed features", () => {
    const X = f64([
      [1, 2],
      [3, 4],
      [5, nan],
      [nan, nan],
    ]);
    // sklearn: [[1, 2], [3, 4], [5, 3], [3, 3]]
    const out = new KNNImputer({ nNeighbors: 2 }).fitTransform(X).toArray();
    expect(out).toEqual([
      [1, 2],
      [3, 4],
      [5, 3],
      [3, 3],
    ]);
  });

  it("matches sklearn on a distance-weighted example", () => {
    const X = f64([
      [1, nan],
      [nan, 2],
      [3, 4],
      [5, 6],
    ]);
    // sklearn: [[1, 4.666666666666666], [3.6666666666666665, 2], [3, 4], [5, 6]]
    const out = new KNNImputer({ nNeighbors: 2, weights: "distance" })
      .fitTransform(X)
      .toArray() as number[][];
    expectClose(out.flat(), [1, 4.666666666666666, 3.6666666666666665, 2, 3, 4, 5, 6]);
  });

  it("never reads stale neighbors when some distances are infinite", () => {
    const X = f64([
      [Number.POSITIVE_INFINITY, 2, 1],
      [Number.POSITIVE_INFINITY, 3, 2],
      [1, 2, 3],
      [2, nan, nan],
    ]);
    const out = new KNNImputer({ nNeighbors: 2 }).fitTransform(X).toArray() as number[][];
    for (const row of out) for (const v of row) expect(Number.isNaN(v)).toBe(false);
  });

  it("supports missingValues, keepEmptyFeatures, params and clone", () => {
    const X = f64([
      [1, -1, -1],
      [2, 5, -1],
      [3, 7, -1],
    ]);
    const imp = new KNNImputer({ nNeighbors: 2, missingValues: -1, keepEmptyFeatures: false });
    const out = imp.fitTransform(X);
    expect(out.shape).toEqual([3, 2]);
    expect((out.toArray() as number[][])[0]?.[1]).toBeCloseTo(6, 12);
    expect(imp.getParams()).toMatchObject({ nNeighbors: 2, missingValues: -1 });
    imp.setParams({ nNeighbors: 1 });
    expect(() => imp.transform(X)).toThrow(NotFittedError);
    expect(imp.clone().getParams()).toMatchObject({ nNeighbors: 1 });
    expect(() => new KNNImputer().fit(zeros([0, 1]))).toThrow(InvalidParameterError);
    expect(() => new KNNImputer().fit(tensor([["a"]]))).toThrow(DTypeError);
  });
});

describe("c53 MissingIndicator", () => {
  const nan = Number.NaN;

  it("raises for new missing columns by default (sklearn error_on_new)", () => {
    const mi = new MissingIndicator().fit(
      f64([
        [1, nan, 3],
        [4, 5, 6],
      ])
    );
    expect(mi.features_).toEqual([1]);
    expect(() => mi.transform(f64([[nan, 1, nan]]))).toThrow(DataValidationError);
    expect(mi.transform(f64([[1, nan, 2]])).toArray()).toEqual([[1]]);
    const lenient = new MissingIndicator({ errorOnNew: false }).fit(
      f64([
        [1, nan, 3],
        [4, 5, 6],
      ])
    );
    expect(lenient.transform(f64([[nan, 1, nan]])).toArray()).toEqual([[0]]);
  });

  it("supports a custom missing marker and empty fits fail clearly", () => {
    const mi = new MissingIndicator({ missingValues: -1, features: "all" });
    expect(
      mi
        .fitTransform(
          f64([
            [1, -1],
            [-1, 2],
          ])
        )
        .toArray()
    ).toEqual([
      [0, 1],
      [1, 0],
    ]);
    expect(() => new MissingIndicator().fit(zeros([0, 1]))).toThrow(InvalidParameterError);
    expect(mi.setParams({ features: "missing-only" }).getParams()).toMatchObject({
      features: "missing-only",
    });
    expect(() => mi.features_).toThrow(NotFittedError);
  });
});

describe("c53 PolynomialFeatures", () => {
  it("orders columns like scikit-learn and reports powers and names", () => {
    const poly = new PolynomialFeatures({ degree: 3 }).fit(f64([[0, 0, 0]]));
    expect(poly.getFeatureNamesOut()).toEqual([
      "1",
      "x0",
      "x1",
      "x2",
      "x0^2",
      "x0 x1",
      "x0 x2",
      "x1^2",
      "x1 x2",
      "x2^2",
      "x0^3",
      "x0^2 x1",
      "x0^2 x2",
      "x0 x1^2",
      "x0 x1 x2",
      "x0 x2^2",
      "x1^3",
      "x1^2 x2",
      "x1 x2^2",
      "x2^3",
    ]);
    expect(poly.powers.slice(0, 6)).toEqual([
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [2, 0, 0],
      [1, 1, 0],
    ]);
    expect(poly.getFeatureNamesOut(["a", "b", "c"])[5]).toBe("a b");
    expect(() => poly.getFeatureNamesOut(["a"])).toThrow(InvalidParameterError);
    expect(poly.transform(f64([[2, 3, 5]])).toArray()).toEqual([
      [1, 2, 3, 5, 4, 6, 10, 9, 15, 25, 8, 12, 20, 18, 30, 50, 27, 45, 75, 125],
    ]);
    const inter = new PolynomialFeatures({ degree: 2, interactionOnly: true, includeBias: false });
    expect(inter.fit(f64([[0, 0, 0]])).getFeatureNamesOut()).toEqual([
      "x0",
      "x1",
      "x2",
      "x0 x1",
      "x0 x2",
      "x1 x2",
    ]);
  });

  it("refuses expansions that would exhaust memory", () => {
    const wide = f64([new Array(100).fill(1)]);
    expect(() => new PolynomialFeatures({ degree: 40 }).fit(wide)).toThrow(MemoryError);
    const many = f64(Array.from({ length: 2 }, () => new Array(50).fill(1)));
    expect(() => new PolynomialFeatures({ degree: 1_000_000_000 }).fit(many)).toThrow(MemoryError);
  });

  it("counts the output size correctly for a single feature and a high degree", () => {
    // One feature at degree 100 gives 101 columns; C(100, 50) must not trip the size guard.
    const out = new PolynomialFeatures({ degree: 100 }).fit(f64([[1.01]]));
    expect(out.nOutputFeatures).toBe(101);
    const Xt = out.transform(f64([[2]]));
    expect(Xt.shape).toEqual([1, 101]);
    expect((Xt.toArray() as number[][])[0]?.[10]).toBe(1024);
  });

  it("handles degree 0 and a single feature", () => {
    expect(new PolynomialFeatures({ degree: 0 }).fitTransform(f64([[1, 2]])).toArray()).toEqual([
      [1],
    ]);
    const none = new PolynomialFeatures({ degree: 0, includeBias: false }).fitTransform(
      f64([[1, 2]])
    );
    expect(none.shape).toEqual([1, 0]);
    expect(new PolynomialFeatures({ degree: 3 }).fitTransform(f64([[2]])).toArray()).toEqual([
      [1, 2, 4, 8],
    ]);
  });

  it("setParams takes effect on the next transform without refitting", () => {
    const poly = new PolynomialFeatures({ degree: 2 }).fit(f64([[1, 2]]));
    expect(poly.nOutputFeatures).toBe(6);
    poly.setParams({ degree: 3, includeBias: false });
    expect(poly.nOutputFeatures).toBe(9);
    expect(poly.transform(f64([[1, 2]])).shape).toEqual([1, 9]);
    expect(() => poly.setParams({ degree: -1 })).toThrow(InvalidParameterError);
    expect(() => poly.setParams({ order: 1 })).toThrow(InvalidParameterError);
    expect(poly.clone().getParams()).toEqual({
      degree: 3,
      interactionOnly: false,
      includeBias: false,
    });
  });

  it("rejects string input and non-2D data", () => {
    expect(() => new PolynomialFeatures().fit(tensor([["a"]]))).toThrow(DTypeError);
    expect(() => new PolynomialFeatures().fit(f64([1, 2]))).toThrow(ShapeError);
  });
});

describe("c53 Binarizer and FunctionTransformer", () => {
  it("Binarizer rejects strings, NaN thresholds and a changed feature count", () => {
    expect(() => new Binarizer().fit(tensor([["a"]]))).toThrow(DTypeError);
    expect(() => new Binarizer({ threshold: Number.NaN })).toThrow(InvalidParameterError);
    const b = new Binarizer({ threshold: 1 }).fit(f64([[0, 2]]));
    expect(() => b.transform(f64([[1, 2, 3]]))).toThrow(InvalidParameterError);
    expect(
      b
        .transform(
          f64([
            [1, 2],
            [Number.NaN, 0],
          ])
        )
        .toArray()
    ).toEqual([
      [0, 1],
      [0, 0],
    ]);
  });

  it("Binarizer setParams updates the threshold and clone copies it", () => {
    const b = new Binarizer().setParams({ threshold: 2 });
    expect(b.getParams()).toEqual({ threshold: 2 });
    expect(b.clone().getParams()).toEqual({ threshold: 2 });
    expect(() => b.setParams({ cutoff: 1 })).toThrow(InvalidParameterError);
  });

  it("FunctionTransformer validates func results and supports setParams", () => {
    const ft = new FunctionTransformer({ func: (X) => X });
    ft.fit(f64([[1]]));
    expect(() => new FunctionTransformer({ func: 3 as never })).toThrow(InvalidParameterError);
    const bad = new FunctionTransformer({ func: (() => [1]) as never }).fit(f64([[1]]));
    expect(() => bad.transform(f64([[1]]))).toThrow(InvalidParameterError);
    const doubler = (X: Tensor): Tensor =>
      f64((X.toArray() as number[][]).map((r) => r.map((v) => v * 2)));
    ft.setParams({ func: doubler });
    expect(ft.transform(f64([[1, 2]])).toArray()).toEqual([[2, 4]]);
    expect(
      ft
        .clone()
        .fit(f64([[1]]))
        .transform(f64([[3]]))
        .toArray()
    ).toEqual([[6]]);
    expect(() => ft.setParams({ funk: doubler })).toThrow(InvalidParameterError);
  });
});

describe("c53 reviewer fixes", () => {
  it("SelectFromModel keeps every feature when all importances are equal", () => {
    // mean(0.1, 0.1, 0.1) is 0.10000000000000002 in floating point, above every importance.
    const est = { featureImportances_: [0.1, 0.1, 0.1], fit: () => 0 };
    const sel = new SelectFromModel({ estimator: est }).fit(f64([[1, 2, 3]]), f64([0]));
    expect(sel.getSupport()).toEqual([true, true, true]);
  });

  it("reads flattened [n_targets * n_features] coefficients (SGDClassifier layout)", () => {
    class Flat {
      coef = Float64Array.from([1, -2, 0, 3, 2, 0.5]);
      fit(): this {
        return this;
      }
    }
    const sel = new SelectFromModel({ estimator: new Flat(), threshold: 0.8 });
    sel.fit(f64([[0, 0, 0]]), f64([0]));
    // [[1, -2, 0], [3, 2, 0.5]] -> sum |coef| over targets
    expect(sel.importances).toEqual([4, 4, 0.5]);
    expect(sel.getSupport()).toEqual([true, true, false]);
    class Bad {
      coef = Float64Array.from([1, 2, 3, 4]);
      fit(): this {
        return this;
      }
    }
    expect(() =>
      new SelectFromModel({ estimator: new Bad() }).fit(f64([[0, 0, 0]]), f64([0]))
    ).toThrow(InvalidParameterError);
  });

  it("SelectKBest.setParams validates k before changing any state", () => {
    const skb = new SelectKBest({ scoreFunc: () => [3, 1, 2], k: 1 }).fit(
      f64([[1, 2, 3]]),
      f64([0])
    );
    expect(() => skb.setParams({ k: 4 })).toThrow(InvalidParameterError);
    expect(skb.getParams()["k"]).toBe(1);
    expect(skb.getSupport()).toEqual([true, false, false]);
  });

  it("PolynomialFeatures keeps its previous fit when a new fit is refused", () => {
    const poly = new PolynomialFeatures({ degree: 3 }).fit(f64([[1, 2]]));
    expect(() => poly.fit(f64([new Array(1000).fill(1)]))).toThrow(MemoryError);
    expect(poly.nOutputFeatures).toBe(10);
    expect(poly.transform(f64([[1, 2]])).shape).toEqual([1, 10]);
    expect(() => poly.setParams({ degree: 1_000_000 })).toThrow(MemoryError);
    expect(poly.getParams()["degree"]).toBe(3);
  });

  it("SimpleImputer allows a NaN fill value together with a custom missing marker", () => {
    const imp = new SimpleImputer({
      strategy: "constant",
      fillValue: Number.NaN,
      missingValues: -1,
    });
    const out = imp.fitTransform(f64([[1, -1]])).toArray() as number[][];
    expect(out[0]?.[0]).toBe(1);
    expect(Number.isNaN(out[0]?.[1] as number)).toBe(true);
  });
});
