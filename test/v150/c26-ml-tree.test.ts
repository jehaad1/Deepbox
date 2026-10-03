/**
 * Regression tests for src/ml/tree/DecisionTree.ts and src/ml/tree/ExtraTrees.ts (v1.5.0 review).
 *
 * Reference values come from scikit-learn 1.8 (DecisionTreeClassifier / DecisionTreeRegressor
 * with random_state=0 on the 16-row data set below, plus the small behavior checks quoted next to
 * the individual tests).
 */
import { describe, expect, it } from "vitest";
import {
  DataValidationError,
  InvalidParameterError,
  NotFittedError,
  ShapeError,
} from "../../src/core";
import {
  DecisionTreeClassifier,
  DecisionTreeRegressor,
  export_text,
} from "../../src/ml/tree/DecisionTree";
import { ExtraTreesClassifier, ExtraTreesRegressor } from "../../src/ml/tree/ExtraTrees";
import { type Tensor, tensor } from "../../src/ndarray";
import { setSeed } from "../../src/random";

const toArray = (t: Tensor): number[] => Array.from(t.data as ArrayLike<number>);
const f64 = (rows: number[][]): Tensor => tensor(rows, { dtype: "float64" });
const i32 = (values: number[]): Tensor => tensor(values, { dtype: "int32" });
const f64v = (values: number[]): Tensor => tensor(values, { dtype: "float64" });
const i64 = (values: bigint[]): Tensor => tensor(BigInt64Array.from(values), { dtype: "int64" });

function expectClose(actual: readonly number[], expected: readonly number[], tol = 1e-9): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(Math.abs((actual[i] as number) - (expected[i] as number))).toBeLessThan(tol);
  }
}

/*
 * import numpy as np
 * rng = np.random.RandomState(7)
 * X = np.round(rng.randn(16, 3) * 2, 1)
 * y = (X[:, 0] > 0.3).astype(int) + (X[:, 1] > 1.0).astype(int)
 * yr = np.round(X[:, 0] * 1.5 - X[:, 2] + 0.2 * rng.randn(16), 2)
 */
const X16 = [
  [3.4, -0.9, 0.1],
  [0.8, -1.6, 0.0],
  [-0.0, -3.5, 2.0],
  [1.2, -1.3, -0.3],
  [1.0, -0.5, -0.5],
  [-2.9, 1.1, 0.2],
  [0.5, -3.1, 3.3],
  [0.3, -0.8, 4.1],
  [-0.1, -2.9, -0.8],
  [-4.6, 2.1, -0.8],
  [-1.5, 2.1, -3.3],
  [1.1, -4.1, -1.3],
  [-2.4, 2.9, 3.5],
  [-0.7, 1.7, -0.4],
  [1.1, -1.5, -3.4],
  [-3.6, 0.8, 4.5],
];
const Y16 = [1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0];
const YR16 = [
  5.05, 1.1, -1.62, 2.15, 2.02, -4.5, -2.58, -3.71, 0.36, -6.0, 1.03, 3.19, -7.17, -1.03, 5.03,
  -9.56,
];

describe("DecisionTree: scikit-learn parity", () => {
  it("classifier (gini) matches predictions, importances, depth and leaves", () => {
    // DecisionTreeClassifier(max_depth=3, random_state=0)
    const clf = new DecisionTreeClassifier({ maxDepth: 3 }).fit(f64(X16), i32(Y16));
    expect(toArray(clf.predict(f64(X16)))).toEqual(Y16);
    expectClose(
      toArray(clf.featureImportances),
      [0.3333333333333333, 0.23809523809523822, 0.42857142857142844]
    );
    expect(clf.getDepth()).toBe(3);
    expect(clf.getNLeaves()).toBe(4);
    const proba = clf.predictProba(f64(X16.slice(0, 4)));
    expect(proba.shape).toEqual([4, 2]);
    expect(toArray(proba)).toEqual([0, 1, 0, 1, 1, 0, 0, 1]);
  });

  it("classifier (entropy) matches importances", () => {
    // DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0)
    const clf = new DecisionTreeClassifier({ maxDepth: 3, criterion: "entropy" }).fit(
      f64(X16),
      i32(Y16)
    );
    expectClose(
      toArray(clf.featureImportances),
      [0.30815572670182784, 0.32999001598621597, 0.3618542573119562]
    );
    expect(toArray(clf.predict(f64(X16)))).toEqual(Y16);
  });

  it("log_loss is an alias of entropy", () => {
    const a = new DecisionTreeClassifier({ maxDepth: 3, criterion: "entropy" }).fit(
      f64(X16),
      i32(Y16)
    );
    const b = new DecisionTreeClassifier({ maxDepth: 3, criterion: "log_loss" }).fit(
      f64(X16),
      i32(Y16)
    );
    expect(toArray(b.featureImportances)).toEqual(toArray(a.featureImportances));
  });

  it("classifier with minSamplesLeaf=3 matches", () => {
    // DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=0)
    const clf = new DecisionTreeClassifier({ maxDepth: 3, minSamplesLeaf: 3 }).fit(
      f64(X16),
      i32(Y16)
    );
    expect(toArray(clf.predict(f64(X16)))).toEqual([
      1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0,
    ]);
    expect(clf.getNLeaves()).toBe(3);
    expectClose(toArray(clf.featureImportances), [0, 0.21390374331550807, 0.786096256684492]);
  });

  it("regressor matches predictions, importances, R2, depth and leaves", () => {
    // DecisionTreeRegressor(max_depth=3, random_state=0)
    const reg = new DecisionTreeRegressor({ maxDepth: 3 }).fit(f64(X16), f64v(YR16));
    expectClose(
      toArray(reg.predict(f64(X16))),
      [
        3.855, 1.56, -2.235, 3.855, 1.56, -4.5, -2.235, -2.235, 0.695, -6.585, 0.695, 3.855, -6.585,
        -2.235, 3.855, -9.56,
      ]
    );
    expectClose(
      toArray(reg.featureImportances),
      [0.9084895139064877, 0.04866247514904966, 0.042848010944462764],
      1e-9
    );
    expect(reg.getDepth()).toBe(3);
    expect(reg.getNLeaves()).toBe(7);
    expect(reg.score(f64(X16), f64v(YR16))).toBeCloseTo(0.9583333333333334, 12);
  });

  it("regressor with minSamplesLeaf=3 matches", () => {
    // DecisionTreeRegressor(max_depth=3, min_samples_leaf=3, random_state=0)
    const reg = new DecisionTreeRegressor({ maxDepth: 3, minSamplesLeaf: 3 }).fit(
      f64(X16),
      f64v(YR16)
    );
    expect(reg.getNLeaves()).toBe(5);
    expectClose(
      toArray(reg.predict(f64(X16))),
      [
        2.766666666666666, 2.766666666666666, -2.6366666666666667, 2.766666666666666,
        3.4133333333333336, -6.8075, -2.6366666666666667, -2.6366666666666667, 0.12000000000000004,
        -6.8075, 0.12000000000000004, 3.4133333333333336, -6.8075, 0.12000000000000004,
        3.4133333333333336, -6.8075,
      ]
    );
  });
});

describe("DecisionTree: numerics and dtypes", () => {
  it("regressor picks the right split when targets sit on a huge offset", () => {
    // y = 1e9 + 0.5 * [i >= 3] + 0.001 * i. The raw sumL^2/nL + sumR^2/nR score loses every
    // significant digit at this scale and used to select the threshold 0.5.
    const X = f64([0, 1, 2, 3, 4, 5, 6, 7].map((i) => [i]));
    const y = tensor(
      [0, 1, 2, 3, 4, 5, 6, 7].map((i) => 1e9 + (i < 3 ? 0 : 0.5) + i * 0.001),
      { dtype: "float64" }
    );
    const reg = new DecisionTreeRegressor({ maxDepth: 1 }).fit(X, y);
    expect(reg.tree_?.threshold).toBe(2.5);
  });

  it("regressor predictions keep float64 precision", () => {
    const X = f64([[1], [2], [3], [4], [5], [6]]);
    const y = tensor([1e8 + 0.1, 1e8 + 0.2, 1e8 + 0.3, 1e8 + 0.4, 1e8 + 0.5, 1e8 + 0.6], {
      dtype: "float64",
    });
    const reg = new DecisionTreeRegressor({ maxDepth: 1 }).fit(X, y);
    const pred = reg.predict(X);
    expect(pred.dtype).toBe("float64");
    expectClose(
      toArray(pred),
      [1e8 + 0.2, 1e8 + 0.2, 1e8 + 0.2, 1e8 + 0.5, 1e8 + 0.5, 1e8 + 0.5],
      1e-6
    );
  });

  it("predictProba is float64", () => {
    const clf = new DecisionTreeClassifier().fit(f64(X16), i32(Y16));
    expect(clf.predictProba(f64(X16)).dtype).toBe("float64");
  });

  it("fractional class labels are kept, integer labels stay int32", () => {
    const X = f64([[1], [2], [3], [4], [5], [6]]);
    const frac = new DecisionTreeClassifier().fit(
      X,
      tensor([0.5, 0.5, 1.5, 1.5, 1.5, 0.5], { dtype: "float64" })
    );
    expect(toArray(frac.predict(X))).toEqual([0.5, 0.5, 1.5, 1.5, 1.5, 0.5]);
    expect(frac.predict(X).dtype).toBe("float64");
    expect(toArray(frac.classes as Tensor)).toEqual([0.5, 1.5]);

    const ints = new DecisionTreeClassifier().fit(X, i32([0, 0, 1, 1, 1, 0]));
    expect(ints.predict(X).dtype).toBe("int32");
    expect((ints.classes as Tensor).dtype).toBe("int32");
  });

  it("exact ties between classes resolve to the smallest label (scikit-learn)", () => {
    // DecisionTreeClassifier().fit(np.ones((4, 1)), [1, 1, 0, 0]).predict(...) -> [0 0 0 0]
    const X = f64([[1], [1], [1], [1]]);
    const clf = new DecisionTreeClassifier().fit(X, i32([1, 1, 0, 0]));
    expect(toArray(clf.predict(X))).toEqual([0, 0, 0, 0]);
    expect(toArray(clf.predictProba(X))).toEqual([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
  });

  it("split thresholds stay finite for huge feature values", () => {
    const X = f64([[-1.7e308], [-1.6e308], [1.6e308], [1.7e308]]);
    const clf = new DecisionTreeClassifier().fit(X, i32([0, 0, 1, 1]));
    expect(Number.isFinite(clf.tree_?.threshold)).toBe(true);
    expect(toArray(clf.predict(X))).toEqual([0, 0, 1, 1]);
  });

  it("does not modify its inputs", () => {
    const X = f64(X16);
    const y = f64v(YR16);
    const xCopy = toArray(X);
    const yCopy = toArray(y);
    const reg = new DecisionTreeRegressor().fit(X, y);
    reg.predict(X);
    expect(toArray(X)).toEqual(xCopy);
    expect(toArray(y)).toEqual(yCopy);
  });
});

describe("DecisionTree: random feature subsets", () => {
  it("keeps drawing features when the drawn subset is constant (scikit-learn)", () => {
    // Feature 0 is constant. DecisionTreeClassifier(max_features=1, random_state=s) scores 1.0
    // for every seed; the old code returned a leaf whenever it drew feature 0.
    const X = f64([1, 2, 3, 4, 5, 6].map((v) => [0, v]));
    const y = i32([0, 0, 0, 1, 1, 1]);
    for (let seed = 0; seed < 10; seed++) {
      const clf = new DecisionTreeClassifier({ maxFeatures: 1, randomState: seed }).fit(X, y);
      expect(clf.score(X, y)).toBe(1);
      const reg = new DecisionTreeRegressor({ maxFeatures: 1, randomState: seed }).fit(
        X,
        tensor([0, 0, 0, 1, 1, 1], { dtype: "float64" })
      );
      expect(reg.score(X, tensor([0, 0, 0, 1, 1, 1], { dtype: "float64" }))).toBe(1);
    }
  });

  it("consecutive seeds give different feature subsets", () => {
    // Ten identical informative columns: the root feature is the first one drawn. The old LCG
    // produced nearly the same first draw for seeds s and s + 1 (Random Forest uses s + t).
    const rows = Array.from({ length: 20 }, (_, i) => Array.from({ length: 10 }, () => i));
    const y = i32(rows.map((_, i) => (i < 10 ? 0 : 1)));
    const roots = new Set<number>();
    for (let seed = 0; seed < 20; seed++) {
      const clf = new DecisionTreeClassifier({ maxFeatures: 1, randomState: seed }).fit(
        f64(rows),
        y
      );
      roots.add(clf.tree_?.featureIndex ?? -1);
    }
    expect(roots.size).toBeGreaterThanOrEqual(6);
  });

  it("is reproducible with randomState and with the global seed", () => {
    const X = f64(X16);
    const a = new DecisionTreeClassifier({ maxFeatures: 1, randomState: 3 }).fit(X, i32(Y16));
    const b = new DecisionTreeClassifier({ maxFeatures: 1, randomState: 3 }).fit(X, i32(Y16));
    expect(export_text(a)).toBe(export_text(b));

    setSeed(11);
    const c = new DecisionTreeClassifier({ maxFeatures: 1 }).fit(X, i32(Y16));
    setSeed(11);
    const d = new DecisionTreeClassifier({ maxFeatures: 1 }).fit(X, i32(Y16));
    expect(export_text(c)).toBe(export_text(d));
  });

  it("supports sqrt and log2 and rejects other strings", () => {
    const X = f64(X16);
    expect(() =>
      new DecisionTreeClassifier({ maxFeatures: "sqrt" }).fit(X, i32(Y16))
    ).not.toThrow();
    expect(() =>
      new DecisionTreeRegressor({ maxFeatures: "log2" }).fit(X, f64v(YR16))
    ).not.toThrow();
    expect(() => new DecisionTreeClassifier({ maxFeatures: "auto" as never })).toThrow(
      InvalidParameterError
    );
  });
});

describe("DecisionTree: validation and parameters", () => {
  it("rejects an unknown criterion in the constructor", () => {
    expect(() => new DecisionTreeClassifier({ criterion: "mse" as never })).toThrow(
      InvalidParameterError
    );
  });

  it("setParams requires an integer maxFeatures", () => {
    const clf = new DecisionTreeClassifier();
    expect(() => clf.setParams({ maxFeatures: 1.5 })).toThrow(InvalidParameterError);
    expect(() => new DecisionTreeRegressor().setParams({ maxFeatures: 0 })).toThrow(
      InvalidParameterError
    );
    expect(clf.setParams({ maxFeatures: 2 }).getParams().maxFeatures).toBe(2);
  });

  it("setParams is atomic", () => {
    const clf = new DecisionTreeClassifier({ maxDepth: 4 });
    expect(() => clf.setParams({ maxDepth: 7, minSamplesLeaf: 0 })).toThrow(InvalidParameterError);
    expect(clf.getParams().maxDepth).toBe(4);
    const reg = new DecisionTreeRegressor({ maxDepth: 4 });
    expect(() => reg.setParams({ maxDepth: 7, bogus: 1 })).toThrow(InvalidParameterError);
    expect(reg.getParams().maxDepth).toBe(4);
  });

  it("maxDepth may be Infinity", () => {
    const clf = new DecisionTreeClassifier({ maxDepth: Number.POSITIVE_INFINITY }).fit(
      f64(X16),
      i32(Y16)
    );
    expect(clf.score(f64(X16), i32(Y16))).toBe(1);
    expect(() => new DecisionTreeRegressor({ maxDepth: Number.NaN })).toThrow(
      InvalidParameterError
    );
    expect(() => new DecisionTreeRegressor({ maxDepth: 2.5 })).toThrow(InvalidParameterError);
  });

  it("clone returns an unfitted copy with the same parameters", () => {
    const clf = new DecisionTreeClassifier({
      maxDepth: 3,
      criterion: "entropy",
      maxFeatures: "sqrt",
    });
    clf.fit(f64(X16), i32(Y16));
    const copy = clf.clone();
    expect(copy.getParams()).toEqual(clf.getParams());
    expect(copy.tree_).toBeUndefined();
    expect(() => copy.predict(f64(X16))).toThrow(NotFittedError);
  });

  it("score rejects mismatched rows and empty targets", () => {
    const clf = new DecisionTreeClassifier().fit(f64(X16), i32(Y16));
    expect(() => clf.score(f64(X16), i32(Y16.slice(0, 5)))).toThrow(ShapeError);
    expect(() => clf.score(f64(X16), i32([]))).toThrow(DataValidationError);
    const reg = new DecisionTreeRegressor().fit(f64(X16), f64v(YR16));
    expect(() => reg.score(f64(X16), f64v(YR16.slice(0, 5)))).toThrow(ShapeError);
    expect(() => reg.score(f64(X16), tensor([] as number[]))).toThrow(DataValidationError);
  });

  it("score accepts int64 targets", () => {
    const X = f64([[1], [2], [3], [4]]);
    const clf = new DecisionTreeClassifier().fit(X, i32([0, 0, 1, 1]));
    const y = i64([0n, 0n, 1n, 1n]);
    expect(clf.score(X, y)).toBe(1);
    const reg = new DecisionTreeRegressor().fit(X, tensor([1, 2, 3, 4], { dtype: "float64" }));
    expect(reg.score(X, i64([1n, 2n, 3n, 4n]))).toBe(1);
  });

  it("a rejected refit keeps the previous model and a refit replaces all state", () => {
    const clf = new DecisionTreeClassifier().fit(f64(X16), i32(Y16));
    expect(() => clf.fit(f64([[1], [2]]), i32([0, 1, 1]))).toThrow(ShapeError);
    expect(clf.score(f64(X16), i32(Y16))).toBe(1);
    expect(toArray(clf.classes as Tensor)).toEqual([0, 1]);

    clf.fit(f64([[1], [2], [3], [4]]), i32([5, 5, 7, 7]));
    expect(toArray(clf.classes as Tensor)).toEqual([5, 7]);
    expect(clf.nFeatures_).toBe(1);
    expect(clf.predictProba(f64([[1], [4]])).shape).toEqual([2, 2]);
  });

  it("predict handles zero rows", () => {
    const clf = new DecisionTreeClassifier().fit(f64(X16), i32(Y16));
    const empty = tensor(new Float64Array(0), { dtype: "float64" }).reshape([0, 3]);
    expect(clf.predict(empty).shape).toEqual([0]);
    expect(clf.predictProba(empty).shape).toEqual([0, 2]);
  });

  it("feature importances are all zero when the tree never splits", () => {
    const reg = new DecisionTreeRegressor().fit(f64(X16), f64v(new Array<number>(16).fill(2.5)));
    expect(toArray(reg.featureImportances)).toEqual([0, 0, 0]);
    expect(reg.getDepth()).toBe(0);
    expect(reg.getNLeaves()).toBe(1);
  });

  it("depth and leaf getters require a fitted tree", () => {
    expect(() => new DecisionTreeClassifier().getDepth()).toThrow(NotFittedError);
    expect(() => new DecisionTreeRegressor().getNLeaves()).toThrow(NotFittedError);
  });
});

describe("DecisionTree: deep trees and traversal order", () => {
  it("grows a tree thousands of levels deep without overflowing the call stack", () => {
    // Labels with period 3 make every split peel off two samples, so the tree is a long chain.
    // 1.0.0 overflowed the call stack from about 9000 samples (depth about 6000) on, and the
    // cost grows with n squared, which coverage instrumentation makes about 9 times slower,
    // so n stays just above that point.
    const n = 10000;
    const X = f64(Array.from({ length: n }, (_, i) => [i]));
    const y = i32(Array.from({ length: n }, (_, i) => (i % 3 === 0 ? 1 : 0)));
    const clf = new DecisionTreeClassifier({ maxDepth: Number.POSITIVE_INFINITY }).fit(X, y);
    expect(clf.getDepth()).toBeGreaterThan(5000);
    expect(clf.getNLeaves()).toBe(clf.getDepth() + 1);
    expect(clf.score(X, y)).toBe(1);
    expect(toArray(clf.featureImportances)).toEqual([1]);
  }, 60_000);

  it("remapLeaves visits leaves left to right and export_text lists branches in order", () => {
    const X = f64([[1], [2], [3], [4], [5], [6]]);
    const reg = new DecisionTreeRegressor({ maxDepth: 2 }).fit(X, f64v([1, 1, 5, 5, 9, 9]));
    const seen: number[] = [];
    reg.remapLeaves((v) => {
      seen.push(v);
      return v * 2;
    });
    expect(seen).toEqual([1, 5, 9]);
    expect(toArray(reg.predict(X))).toEqual([2, 2, 10, 10, 18, 18]);
    expect(export_text(reg, { decimals: 1 })).toBe(
      [
        "|--- feature_0 <= 2.5",
        "|   |--- value: 2",
        "|--- feature_0 > 2.5",
        "|   |--- feature_0 <= 4.5",
        "|   |   |--- value: 10",
        "|   |--- feature_0 > 4.5",
        "|   |   |--- value: 18",
      ].join("\n")
    );
  });
});

describe("export_text", () => {
  it("rejects invalid decimals", () => {
    const clf = new DecisionTreeClassifier({ maxDepth: 2 }).fit(f64(X16), i32(Y16));
    expect(() => export_text(clf, { decimals: -1 })).toThrow(InvalidParameterError);
    expect(() => export_text(clf, { decimals: 101 })).toThrow(InvalidParameterError);
    expect(() => export_text(clf, { decimals: 1.5 })).toThrow(InvalidParameterError);
    expect(() => export_text(new DecisionTreeClassifier())).toThrow(NotFittedError);
  });

  it("prints fractional class labels unchanged", () => {
    const X = f64([[1], [2], [3], [4]]);
    const clf = new DecisionTreeClassifier().fit(
      X,
      tensor([0.25, 0.25, 0.75, 0.75], { dtype: "float64" })
    );
    expect(export_text(clf, { decimals: 0 })).toBe(
      [
        "|--- feature_0 <= 3",
        "|   |--- class: 0.25",
        "|--- feature_0 > 3",
        "|   |--- class: 0.75",
      ].join("\n")
    );
  });
});

describe("ExtraTrees", () => {
  // Deterministic pseudo-random data: label from two informative features, third is noise.
  const {
    X: Xn,
    y: yn,
    yReg,
  } = (() => {
    let s = 12345;
    const next = (): number => {
      s = (s * 1103515245 + 12345) % 2147483648;
      return s / 2147483648;
    };
    const rows: number[][] = [];
    const labels: number[] = [];
    const reg: number[] = [];
    for (let i = 0; i < 80; i++) {
      const a = next() * 4 - 2;
      const b = next() * 4 - 2;
      const c = next();
      rows.push([a, b, c]);
      labels.push((a + b + 0.8 * (next() - 0.5) > 0 ? 1 : 0) + (c > 0.8 ? 1 : 0));
      reg.push(3 * a - b + 0.1 * next());
    }
    return { X: f64(rows), y: i32(labels), yReg: tensor(reg, { dtype: "float64" }) };
  })();

  it("classifier featureImportances are computed (they used to be all zero)", () => {
    const clf = new ExtraTreesClassifier({ nEstimators: 30, randomState: 1 }).fit(Xn, yn);
    const imp = toArray(clf.featureImportances);
    expect(imp.length).toBe(3);
    expect(imp.reduce((p, c) => p + c, 0)).toBeCloseTo(1, 12);
    expect(imp[0] as number).toBeGreaterThan(imp[2] as number);
    expect(imp[1] as number).toBeGreaterThan(imp[2] as number);
    for (const v of imp) expect(v).toBeGreaterThanOrEqual(0);
  });

  it("regressor featureImportances are computed and favor the informative feature", () => {
    const reg = new ExtraTreesRegressor({ nEstimators: 30, randomState: 1 }).fit(Xn, yReg);
    const imp = toArray(reg.featureImportances);
    expect(imp.reduce((p, c) => p + c, 0)).toBeCloseTo(1, 12);
    expect(imp[0] as number).toBeGreaterThan(0.5);
    expect(imp[2] as number).toBeLessThan(0.1);
  });

  it("classifier predict is the argmax of predictProba", () => {
    // predict used to be a hard majority vote that disagreed with predictProba on many rows.
    for (let seed = 0; seed < 10; seed++) {
      const clf = new ExtraTreesClassifier({ nEstimators: 4, maxDepth: 2, randomState: seed }).fit(
        Xn,
        yn
      );
      const pred = toArray(clf.predict(Xn));
      const proba = toArray(clf.predictProba(Xn));
      for (let i = 0; i < pred.length; i++) {
        let best = 0;
        for (let c = 1; c < 3; c++) {
          if ((proba[i * 3 + c] as number) > (proba[i * 3 + best] as number)) best = c;
        }
        expect(pred[i]).toBe(best);
      }
    }
  });

  it("predictProba rows sum to one and are float64", () => {
    const clf = new ExtraTreesClassifier({ nEstimators: 7, randomState: 2 }).fit(Xn, yn);
    const proba = clf.predictProba(Xn);
    expect(proba.dtype).toBe("float64");
    expect(proba.shape).toEqual([80, 3]);
    const p = toArray(proba);
    for (let i = 0; i < 80; i++) {
      expect(
        (p[i * 3] as number) + (p[i * 3 + 1] as number) + (p[i * 3 + 2] as number)
      ).toBeCloseTo(1, 12);
    }
  });

  it("works with negative, fractional and large seeds and is reproducible", () => {
    for (const seed of [-5, 0.5, 0.7, 1e12, 233280]) {
      const a = new ExtraTreesClassifier({ nEstimators: 5, randomState: seed }).fit(Xn, yn);
      const b = new ExtraTreesClassifier({ nEstimators: 5, randomState: seed }).fit(Xn, yn);
      expect(toArray(a.predictProba(Xn))).toEqual(toArray(b.predictProba(Xn)));
      expect(a.score(Xn, yn)).toBeGreaterThan(0.6);
    }
  });

  it("different and consecutive seeds give different forests", () => {
    const probas = [1, 2, 3].map((seed) =>
      toArray(
        new ExtraTreesClassifier({ nEstimators: 3, maxDepth: 3, randomState: seed })
          .fit(Xn, yn)
          .predictProba(Xn)
      ).join(",")
    );
    expect(new Set(probas).size).toBe(3);
  });

  it("fractional seeds are not collapsed onto their integer part", () => {
    const fit = (seed: number): string =>
      toArray(
        new ExtraTreesClassifier({ nEstimators: 3, maxDepth: 3, randomState: seed })
          .fit(Xn, yn)
          .predictProba(Xn)
      ).join(",");
    expect(fit(0.5)).not.toBe(fit(0.7));
  });

  it("honors the global seed when randomState is not set", () => {
    setSeed(21);
    const a = new ExtraTreesRegressor({ nEstimators: 5 }).fit(Xn, yReg);
    setSeed(21);
    const b = new ExtraTreesRegressor({ nEstimators: 5 }).fit(Xn, yReg);
    expect(toArray(a.predict(Xn))).toEqual(toArray(b.predict(Xn)));
  });

  it("keeps drawing features when the drawn one is constant (scikit-learn)", () => {
    // ExtraTreesRegressor(n_estimators=5, max_features=1, random_state=0) reaches R2 = 1.0 on
    // [0, i] -> i; the old code produced a leaf whenever it drew the constant column.
    const X = f64(Array.from({ length: 8 }, (_, i) => [0, i]));
    const y = tensor([0, 1, 2, 3, 4, 5, 6, 7], { dtype: "float64" });
    for (let seed = 0; seed < 5; seed++) {
      const reg = new ExtraTreesRegressor({
        nEstimators: 5,
        maxFeatures: 1,
        maxDepth: Number.POSITIVE_INFINITY,
        randomState: seed,
      }).fit(X, y);
      expect(reg.score(X, y)).toBeCloseTo(1, 12);
    }
    const clf = new ExtraTreesClassifier({ nEstimators: 5, maxFeatures: 1, randomState: 0 }).fit(
      X,
      i32([0, 0, 0, 0, 1, 1, 1, 1])
    );
    expect(clf.score(X, i32([0, 0, 0, 0, 1, 1, 1, 1]))).toBeGreaterThan(0.7);
  });

  it("regressor defaults to all features per split (scikit-learn max_features=1.0)", () => {
    expect(new ExtraTreesRegressor().getParams().maxFeatures).toBeUndefined();
    expect(new ExtraTreesClassifier().getParams().maxFeatures).toBe("sqrt");
  });

  it("regressor predictions are float64", () => {
    const X = f64([[1], [2], [3], [4], [5], [6]]);
    const y = tensor([1e8 + 0.1, 1e8 + 0.2, 1e8 + 0.3, 1e8 + 0.4, 1e8 + 0.5, 1e8 + 0.6], {
      dtype: "float64",
    });
    const reg = new ExtraTreesRegressor({ nEstimators: 10, randomState: 0 }).fit(X, y);
    const pred = reg.predict(X);
    expect(pred.dtype).toBe("float64");
    for (const v of toArray(pred)) {
      expect(v).toBeGreaterThan(1e8);
      expect(v).toBeLessThan(1e8 + 1);
    }
    expect(reg.score(X, y)).toBeGreaterThan(0.5);
  });

  it("fractional class labels are not truncated", () => {
    const X = f64([[1], [2], [3], [4], [5], [6]]);
    const y = tensor([0.5, 0.5, 0.5, 1.5, 1.5, 1.5], { dtype: "float64" });
    const clf = new ExtraTreesClassifier({ nEstimators: 20, randomState: 0 }).fit(X, y);
    expect(toArray(clf.classes as Tensor)).toEqual([0.5, 1.5]);
    const pred = clf.predict(f64([[1], [6]]));
    expect(pred.dtype).toBe("float64");
    expect(toArray(pred)).toEqual([0.5, 1.5]);
  });

  it("ties between classes resolve to the smallest label", () => {
    const X = f64([[1], [1], [1], [1]]);
    const clf = new ExtraTreesClassifier({ nEstimators: 3, randomState: 0 }).fit(
      X,
      i32([1, 1, 0, 0])
    );
    expect(toArray(clf.predict(X))).toEqual([0, 0, 0, 0]);
  });

  it("score rejects mismatched rows and empty targets", () => {
    const clf = new ExtraTreesClassifier({ nEstimators: 3, randomState: 0 }).fit(Xn, yn);
    expect(() => clf.score(Xn, i32([0, 1, 2]))).toThrow(ShapeError);
    expect(() => clf.score(Xn, i32([]))).toThrow(DataValidationError);
    const reg = new ExtraTreesRegressor({ nEstimators: 3, randomState: 0 }).fit(Xn, yReg);
    expect(() => reg.score(Xn, tensor([1, 2, 3]))).toThrow(ShapeError);
    expect(() => reg.score(Xn, tensor([] as number[]))).toThrow(DataValidationError);
  });

  it("score accepts int64 targets", () => {
    const X = f64([[1], [2], [3], [4]]);
    const clf = new ExtraTreesClassifier({ nEstimators: 20, randomState: 0 }).fit(
      X,
      i32([0, 0, 1, 1])
    );
    expect(clf.score(X, i64([0n, 0n, 1n, 1n]))).toBeGreaterThan(0.5);
  });

  it("validates options, including in setParams", () => {
    expect(() => new ExtraTreesClassifier({ nEstimators: 0 })).toThrow(InvalidParameterError);
    expect(() => new ExtraTreesRegressor({ maxDepth: 0 })).toThrow(InvalidParameterError);
    expect(() => new ExtraTreesClassifier({ maxFeatures: 1.5 })).toThrow(InvalidParameterError);
    expect(() => new ExtraTreesClassifier({ criterion: "mse" as never })).toThrow(
      InvalidParameterError
    );
    expect(() => new ExtraTreesClassifier({ randomState: Number.NaN })).toThrow(
      InvalidParameterError
    );
    const clf = new ExtraTreesClassifier();
    expect(() => clf.setParams({ maxFeatures: 1.5 })).toThrow(InvalidParameterError);
    expect(() => clf.setParams({ bootstrap: "yes" })).toThrow(InvalidParameterError);
    expect(() => clf.setParams({ nope: 1 })).toThrow(InvalidParameterError);
    expect(() => new ExtraTreesRegressor().setParams({ criterion: "gini" })).toThrow(
      InvalidParameterError
    );
  });

  it("setParams is atomic and accepts maxFeatures undefined", () => {
    const reg = new ExtraTreesRegressor({ nEstimators: 7 });
    expect(() => reg.setParams({ nEstimators: 9, maxDepth: 0 })).toThrow(InvalidParameterError);
    expect(reg.getParams().nEstimators).toBe(7);
    reg.setParams({ maxFeatures: undefined, maxDepth: Number.POSITIVE_INFINITY });
    expect(reg.getParams().maxFeatures).toBeUndefined();
    expect(reg.getParams().maxDepth).toBe(Number.POSITIVE_INFINITY);
  });

  it("clone returns an unfitted copy with the same parameters", () => {
    const clf = new ExtraTreesClassifier({
      nEstimators: 4,
      criterion: "entropy",
      maxFeatures: 2,
      bootstrap: true,
      randomState: 5,
    }).fit(Xn, yn);
    const copy = clf.clone();
    expect(copy.getParams()).toEqual(clf.getParams());
    expect(copy.classes).toBeUndefined();
    expect(() => copy.predict(Xn)).toThrow(NotFittedError);
    const reg = new ExtraTreesRegressor({ nEstimators: 2 });
    expect(reg.clone().getParams()).toEqual(reg.getParams());
    const all = new ExtraTreesClassifier().setParams({ maxFeatures: undefined });
    expect(all.clone().getParams().maxFeatures).toBeUndefined();
  });

  it("entropy criterion fits, and bootstrap with unlimited depth works", () => {
    const clf = new ExtraTreesClassifier({
      nEstimators: 20,
      criterion: "entropy",
      bootstrap: true,
      maxDepth: Number.POSITIVE_INFINITY,
      randomState: 4,
    }).fit(Xn, yn);
    expect(clf.score(Xn, yn)).toBeGreaterThan(0.7);
  });

  it("a rejected refit keeps the previous model; a refit replaces all state", () => {
    const clf = new ExtraTreesClassifier({ nEstimators: 3, randomState: 0 }).fit(Xn, yn);
    expect(() => clf.fit(f64([[1], [2]]), i32([0, 1, 1]))).toThrow(ShapeError);
    expect(clf.predict(Xn).shape).toEqual([80]);
    clf.fit(f64([[1], [2], [3], [4]]), i32([5, 5, 7, 7]));
    expect(toArray(clf.classes as Tensor)).toEqual([5, 7]);
    expect(clf.nFeatures_).toBe(1);
    const reg = new ExtraTreesRegressor({ nEstimators: 3, randomState: 0 });
    expect(() => reg.fit(f64([[1], [2]]), tensor([1, 2, 3]))).toThrow(ShapeError);
    expect(() => reg.predict(Xn)).toThrow(NotFittedError);
    expect(() => reg.featureImportances).toThrow(NotFittedError);
  });

  it("predict handles zero rows", () => {
    const empty = tensor(new Float64Array(0), { dtype: "float64" }).reshape([0, 3]);
    const clf = new ExtraTreesClassifier({ nEstimators: 3, randomState: 0 }).fit(Xn, yn);
    expect(clf.predict(empty).shape).toEqual([0]);
    expect(clf.predictProba(empty).shape).toEqual([0, 3]);
    const reg = new ExtraTreesRegressor({ nEstimators: 3, randomState: 0 }).fit(Xn, yReg);
    expect(reg.predict(empty).shape).toEqual([0]);
  });

  it("does not modify its inputs", () => {
    const xCopy = toArray(Xn);
    const yCopy = toArray(yReg);
    const reg = new ExtraTreesRegressor({ nEstimators: 3, randomState: 0, bootstrap: true }).fit(
      Xn,
      yReg
    );
    reg.predict(Xn);
    expect(toArray(Xn)).toEqual(xCopy);
    expect(toArray(yReg)).toEqual(yCopy);
  });
});
