/**
 * Regression tests for the ensemble estimators (v1.5.0 review):
 * AdaBoost, Bagging, GradientBoosting and Stacking.
 *
 * Reference values come from scikit-learn 1.8 / NumPy 2.4 / SciPy 1.17. The reference
 * script and its output are quoted next to each value.
 */
import { describe, expect, it } from "vitest";
import {
  DataValidationError,
  InvalidParameterError,
  NotFittedError,
  ShapeError,
} from "../../src/core";
import type { Classifier, Regressor } from "../../src/ml/base";
import { AdaBoostClassifier, AdaBoostRegressor } from "../../src/ml/ensemble/AdaBoost";
import { BaggingClassifier, BaggingRegressor } from "../../src/ml/ensemble/Bagging";
import {
  GradientBoostingClassifier,
  GradientBoostingRegressor,
} from "../../src/ml/ensemble/GradientBoosting";
import { StackingClassifier, StackingRegressor } from "../../src/ml/ensemble/Stacking";
import { DecisionTreeClassifier, DecisionTreeRegressor } from "../../src/ml/tree/DecisionTree";
import type { Tensor } from "../../src/ndarray";
import { tensor } from "../../src/ndarray";
import { setSeed } from "../../src/random";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function mat(rows: number[][]): Tensor {
  return tensor(rows, { dtype: "float64" });
}

function vec(values: number[]): Tensor {
  return tensor(values, { dtype: "float64" });
}

function flat(t: Tensor): number[] {
  return Array.from(t.data as ArrayLike<number>).slice(t.offset, t.offset + t.size);
}

function rows(t: Tensor): number[][] {
  const [n = 0, k = 0] = t.shape;
  const f = flat(t);
  return Array.from({ length: n }, (_, i) => f.slice(i * k, (i + 1) * k));
}

/** Deterministic uniform [0, 1) stream (numerical recipes LCG). */
function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/** `count` copies of each value in `values`, in order: two clusters with duplicated rows. */
function repeated(values: number[], count: number): number[] {
  return values.flatMap((v) => new Array<number>(count).fill(v));
}

function column(values: number[]): Tensor {
  return mat(values.map((v) => [v]));
}

/** Two-cluster data: every bootstrap sample contains both distinct x values. */
const twoClusterX = column(repeated([0, 1], 50));
const twoClusterY = vec(repeated([0, 1], 50));

/** Noisy two-class data in two dimensions. */
function noisyClasses(n: number, seed: number): { X: Tensor; y: Tensor } {
  const u = lcg(seed);
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const label = i % 2;
    X.push([label * 1.2 + (u() - 0.5) * 3, label * 0.8 + (u() - 0.5) * 3]);
    y.push(label);
  }
  return { X: mat(X), y: vec(y) };
}

/** Regressor stub whose prediction is the mean of the training targets. */
class MeanRegressor implements Regressor {
  mean = 0;
  fitted = false;
  fit(_X: Tensor, y: Tensor): this {
    const v = flat(y);
    this.mean = v.reduce((a, b) => a + b, 0) / v.length;
    this.fitted = true;
    return this;
  }
  predict(X: Tensor): Tensor {
    return tensor(new Float64Array(X.shape[0] ?? 0).fill(this.mean));
  }
  score(): number {
    return 0;
  }
  getParams(): Record<string, unknown> {
    return {};
  }
  setParams(): this {
    return this;
  }
  clone(): MeanRegressor {
    return new MeanRegressor();
  }
}

/** Final regressor that records the meta-features it was fitted on. */
class RecordingRegressor implements Regressor {
  seenX: Tensor | undefined;
  seenY: Tensor | undefined;
  fit(X: Tensor, y: Tensor): this {
    this.seenX = X;
    this.seenY = y;
    return this;
  }
  predict(X: Tensor): Tensor {
    return tensor(new Float64Array(X.shape[0] ?? 0));
  }
  score(): number {
    return 0;
  }
  getParams(): Record<string, unknown> {
    return {};
  }
  setParams(): this {
    return this;
  }
}

/** Final classifier that records the meta-features it was fitted on. */
class RecordingClassifier implements Classifier {
  seenX: Tensor | undefined;
  fit(X: Tensor, _y: Tensor): this {
    this.seenX = X;
    return this;
  }
  predict(X: Tensor): Tensor {
    return tensor(new Int32Array(X.shape[0] ?? 0), { dtype: "int32" });
  }
  predictProba(X: Tensor): Tensor {
    return tensor(new Float64Array((X.shape[0] ?? 0) * 2).fill(0.5), { dtype: "float64" }).reshape([
      X.shape[0] ?? 0,
      2,
    ]);
  }
  score(): number {
    return 0;
  }
  getParams(): Record<string, unknown> {
    return {};
  }
  setParams(): this {
    return this;
  }
}

// ---------------------------------------------------------------------------
// AdaBoostClassifier
// ---------------------------------------------------------------------------

describe("AdaBoostClassifier (v1.5.0)", () => {
  it("stops after a round with zero weighted error and gives that tree weight 1", () => {
    const m = new AdaBoostClassifier({ nEstimators: 30, randomState: 0 });
    m.fit(twoClusterX, twoClusterY);
    expect(m.nEstimatorsFitted).toBe(1);
    expect(flat(m.estimatorWeights)).toEqual([1]);
    expect(m.score(twoClusterX, twoClusterY)).toBe(1);
  });

  it("predictProba and decisionFunction follow scikit-learn's SAMME formulas (binary)", () => {
    // sklearn: with one estimator the decision function is +-2 and proba = sigmoid(+-2).
    //   1 / (1 + exp(-2.))  ->  0.8807970779778823
    const m = new AdaBoostClassifier({ nEstimators: 1, randomState: 0 });
    m.fit(twoClusterX, twoClusterY);
    const dec = flat(m.decisionFunction(column([0, 1])));
    expect(dec).toEqual([-2, 2]);
    const proba = rows(m.predictProba(column([0, 1])));
    expect(proba[0]?.[1]).toBeCloseTo(1 - 0.8807970779778823, 12);
    expect(proba[1]?.[1]).toBeCloseTo(0.8807970779778823, 12);
    expect((proba[1]?.[0] ?? 0) + (proba[1]?.[1] ?? 0)).toBeCloseTo(1, 14);
  });

  it("predictProba follows scikit-learn's SAMME formulas (3 classes)", () => {
    // scipy.special.softmax([1, -.5, -.5] / 2)
    //   -> [0.5142093777192813, 0.24289531114035925, 0.24289531114035925]
    const X = column(repeated([0, 1, 2], 30));
    const y = vec(repeated([0, 1, 2], 30));
    const m = new AdaBoostClassifier({ nEstimators: 1, maxDepth: 2, randomState: 0 });
    m.fit(X, y);
    const dec = rows(m.decisionFunction(column([0, 1, 2])));
    expect(dec).toEqual([
      [1, -0.5, -0.5],
      [-0.5, 1, -0.5],
      [-0.5, -0.5, 1],
    ]);
    const proba = rows(m.predictProba(column([1])));
    expect(proba[0]?.[1]).toBeCloseTo(0.5142093777192813, 12);
    expect(proba[0]?.[0]).toBeCloseTo(0.24289531114035925, 12);
    expect(proba[0]?.[2]).toBeCloseTo(0.24289531114035925, 12);
  });

  it("trains on float64 features (no float32 rounding of the bootstrap sample)", () => {
    // 1e8 and 1e8 + 1 are the same float32 value, so a float32 copy cannot be split.
    const X = column(repeated([1e8, 1e8 + 1], 20));
    const y = vec(repeated([0, 1], 20));
    const m = new AdaBoostClassifier({ nEstimators: 5, randomState: 1 });
    m.fit(X, y);
    expect(m.score(X, y)).toBe(1);
  });

  it("is reproducible with randomState and differs between seeds", () => {
    const { X, y } = noisyClasses(60, 7);
    const a = new AdaBoostClassifier({ nEstimators: 10, randomState: 3 }).fit(X, y);
    const b = new AdaBoostClassifier({ nEstimators: 10, randomState: 3 }).fit(X, y);
    const c = new AdaBoostClassifier({ nEstimators: 10, randomState: 4 }).fit(X, y);
    expect(flat(a.estimatorWeights)).toEqual(flat(b.estimatorWeights));
    expect(flat(a.predictProba(X))).toEqual(flat(b.predictProba(X)));
    expect(flat(a.estimatorWeights)).not.toEqual(flat(c.estimatorWeights));
  });

  it("uses the global generator when randomState is not given", () => {
    const { X, y } = noisyClasses(60, 8);
    setSeed(11);
    const a = new AdaBoostClassifier({ nEstimators: 8 }).fit(X, y);
    setSeed(11);
    const b = new AdaBoostClassifier({ nEstimators: 8 }).fit(X, y);
    expect(flat(a.estimatorWeights)).toEqual(flat(b.estimatorWeights));
  });

  it("scales the first tree weight by learningRate", () => {
    const { X, y } = noisyClasses(60, 9);
    const full = new AdaBoostClassifier({ nEstimators: 1, randomState: 5 }).fit(X, y);
    const half = new AdaBoostClassifier({ nEstimators: 1, randomState: 5, learningRate: 0.5 }).fit(
      X,
      y
    );
    const w = flat(full.estimatorWeights)[0] ?? 0;
    expect(w).toBeGreaterThan(0);
    expect(flat(half.estimatorWeights)[0]).toBeCloseTo(w * 0.5, 12);
  });

  it("rejects non-integer class labels instead of silently truncating them", () => {
    const X = column([0, 1, 2, 3]);
    expect(() => new AdaBoostClassifier().fit(X, vec([0.5, 0.5, 1.5, 1.5]))).toThrow(
      DataValidationError
    );
  });

  it("requires two classes", () => {
    expect(() => new AdaBoostClassifier().fit(column([0, 1]), vec([1, 1]))).toThrow(
      InvalidParameterError
    );
  });

  it("a failed refit leaves the previous model untouched", () => {
    const m = new AdaBoostClassifier({ nEstimators: 3, randomState: 0 }).fit(
      twoClusterX,
      twoClusterY
    );
    const before = flat(m.predictProba(twoClusterX));
    expect(() => m.fit(column([Number.NaN, 1]), vec([0, 1]))).toThrow(DataValidationError);
    expect(flat(m.predictProba(twoClusterX))).toEqual(before);
    expect(flat(m.classes as Tensor)).toEqual([0, 1]);
  });

  it("refitting replaces the whole ensemble", () => {
    const m = new AdaBoostClassifier({ nEstimators: 3, randomState: 0 }).fit(
      twoClusterX,
      twoClusterY
    );
    m.fit(twoClusterX, vec(repeated([5, 7], 50)));
    expect(flat(m.classes as Tensor)).toEqual([5, 7]);
    expect(flat(m.predict(column([0, 1])))).toEqual([5, 7]);
  });

  it("score rejects a target whose length differs from X", () => {
    const m = new AdaBoostClassifier({ nEstimators: 2, randomState: 0 }).fit(
      twoClusterX,
      twoClusterY
    );
    expect(() => m.score(twoClusterX, vec([0, 1]))).toThrow(ShapeError);
  });

  it("validates options, round-trips getParams and clones", () => {
    expect(() => new AdaBoostClassifier({ randomState: Number.NaN })).toThrow(
      InvalidParameterError
    );
    expect(() => new AdaBoostClassifier({ maxDepth: 0 })).toThrow(InvalidParameterError);
    expect(() => new AdaBoostClassifier({ learningRate: Number.POSITIVE_INFINITY })).toThrow(
      InvalidParameterError
    );
    const m = new AdaBoostClassifier({ nEstimators: 7, learningRate: 0.3, randomState: 2 });
    const clone = m.clone();
    expect(clone).not.toBe(m);
    expect(clone.getParams()).toEqual(m.getParams());
    const other = new AdaBoostClassifier().setParams(m.getParams());
    expect(other.getParams()).toEqual(m.getParams());
  });
});

// ---------------------------------------------------------------------------
// AdaBoostRegressor
// ---------------------------------------------------------------------------

describe("AdaBoostRegressor (v1.5.0)", () => {
  it("keeps float64 precision of the targets and of the predictions", () => {
    // 1e8 + 0.5 is not representable in float32.
    const X = column(repeated([0, 1], 20));
    const y = vec(repeated([1e8, 1e8 + 0.5], 20));
    const m = new AdaBoostRegressor({ nEstimators: 5, randomState: 0 }).fit(X, y);
    expect(m.nEstimatorsFitted).toBe(1);
    const pred = flat(m.predict(column([0, 1])));
    expect(pred[0]).toBe(1e8);
    expect(pred[1]).toBe(1e8 + 0.5);
    expect(m.score(X, y)).toBe(1);
  });

  it("is reproducible with randomState", () => {
    const u = lcg(21);
    const xs = Array.from({ length: 80 }, () => [u() * 4]);
    const ys = xs.map((r) => Math.sin(r[0] ?? 0) + (u() - 0.5) * 0.3);
    const a = new AdaBoostRegressor({ nEstimators: 15, randomState: 6 }).fit(mat(xs), vec(ys));
    const b = new AdaBoostRegressor({ nEstimators: 15, randomState: 6 }).fit(mat(xs), vec(ys));
    expect(flat(a.predict(mat(xs)))).toEqual(flat(b.predict(mat(xs))));
    expect(flat(a.estimatorWeights)).toEqual(flat(b.estimatorWeights));
    expect(a.score(mat(xs), vec(ys))).toBeGreaterThan(0.8);
  });

  it("scales the first tree weight by learningRate", () => {
    const u = lcg(22);
    const xs = Array.from({ length: 80 }, () => [u() * 4]);
    const ys = xs.map((r) => Math.sin(r[0] ?? 0) + (u() - 0.5) * 0.3);
    const full = new AdaBoostRegressor({ nEstimators: 1, randomState: 2 }).fit(mat(xs), vec(ys));
    const half = new AdaBoostRegressor({ nEstimators: 1, randomState: 2, learningRate: 0.5 }).fit(
      mat(xs),
      vec(ys)
    );
    const w = flat(full.estimatorWeights)[0] ?? 0;
    expect(w).toBeGreaterThan(0);
    expect(flat(half.estimatorWeights)[0]).toBeCloseTo(w * 0.5, 12);
  });

  it("validates loss and randomState, and clones", () => {
    expect(() => new AdaBoostRegressor({ loss: "cubic" as never })).toThrow(InvalidParameterError);
    expect(() => new AdaBoostRegressor({ randomState: Number.POSITIVE_INFINITY })).toThrow(
      InvalidParameterError
    );
    const m = new AdaBoostRegressor({ loss: "square", randomState: 1 });
    expect(m.clone().getParams()).toEqual(m.getParams());
  });

  it("a failed refit leaves the previous model untouched", () => {
    const m = new AdaBoostRegressor({ nEstimators: 2, randomState: 0 }).fit(
      twoClusterX,
      twoClusterY
    );
    const before = flat(m.predict(twoClusterX));
    expect(() => m.fit(twoClusterX, vec([1, 2]))).toThrow(ShapeError);
    expect(flat(m.predict(twoClusterX))).toEqual(before);
  });
});

// ---------------------------------------------------------------------------
// Bagging
// ---------------------------------------------------------------------------

describe("BaggingClassifier / BaggingRegressor (v1.5.0)", () => {
  it("grows unlimited-depth trees by default (they are not capped at depth 10)", () => {
    // Alternating labels on 60 sorted points need a tree far deeper than 10 levels.
    const X = column(Array.from({ length: 60 }, (_, i) => i));
    const y = vec(Array.from({ length: 60 }, (_, i) => i % 2));
    const m = new BaggingClassifier({
      nEstimators: 3,
      bootstrap: false,
      maxSamples: 1,
      maxFeatures: 1,
      randomState: 0,
    }).fit(X, y);
    expect(m.score(X, y)).toBe(1);
    const capped = new BaggingClassifier({
      nEstimators: 3,
      bootstrap: false,
      maxDepth: 2,
      randomState: 0,
    }).fit(X, y);
    expect(capped.score(X, y)).toBeLessThan(1);
  });

  it("predictProba averages the trees' leaf probabilities", () => {
    // sklearn: DecisionTreeClassifier(max_depth=1).fit(arange(8)[:, None], [0,0,0,1,0,1,1,1])
    //   predict_proba -> [1, 0] for x <= 2.5 and [0.2, 0.8] above.
    const X = column([0, 1, 2, 3, 4, 5, 6, 7]);
    const y = vec([0, 0, 0, 1, 0, 1, 1, 1]);
    const m = new BaggingClassifier({
      nEstimators: 1,
      bootstrap: false,
      maxDepth: 1,
      randomState: 0,
    }).fit(X, y);
    const p = rows(m.predictProba(X));
    for (let i = 0; i < 3; i++) {
      expect(p[i]?.[0]).toBeCloseTo(1, 6);
      expect(p[i]?.[1]).toBeCloseTo(0, 6);
    }
    for (let i = 3; i < 8; i++) {
      expect(p[i]?.[0]).toBeCloseTo(0.2, 6);
      expect(p[i]?.[1]).toBeCloseTo(0.8, 6);
    }
    expect(flat(m.predict(X))).toEqual([0, 0, 0, 1, 1, 1, 1, 1]);
  });

  it("trains the trees on float64 features and targets", () => {
    const X = column(repeated([1e8, 1e8 + 1], 20));
    const y = vec(repeated([0, 1], 20));
    const clf = new BaggingClassifier({ nEstimators: 3, bootstrap: false, randomState: 0 }).fit(
      X,
      y
    );
    expect(clf.score(X, y)).toBe(1);

    const yr = vec(repeated([1e8, 1e8 + 0.5], 20));
    const reg = new BaggingRegressor({ nEstimators: 3, bootstrap: false, randomState: 0 }).fit(
      X,
      yr
    );
    const pred = flat(reg.predict(column([1e8, 1e8 + 1])));
    expect(pred[0]).toBe(1e8);
    expect(pred[1]).toBe(1e8 + 0.5);
  });

  it("uses a seeded stream: reproducible per seed, different across seeds", () => {
    const { X, y } = noisyClasses(80, 31);
    const make = (seed: number) =>
      new BaggingClassifier({ nEstimators: 8, maxFeatures: 0.5, maxDepth: 3, randomState: seed });
    expect(flat(make(1).fit(X, y).predictProba(X))).toEqual(
      flat(make(1).fit(X, y).predictProba(X))
    );
    expect(flat(make(1).fit(X, y).predictProba(X))).not.toEqual(
      flat(make(2).fit(X, y).predictProba(X))
    );
    // Seeds that the old 233280-state LCG mapped to the same stream are now distinct.
    expect(flat(make(0).fit(X, y).predictProba(X))).not.toEqual(
      flat(make(233280).fit(X, y).predictProba(X))
    );
  });

  it("samples rows without replacement when bootstrap is false", () => {
    // maxSamples = 0.5 of 8 distinct rows: a depth-unlimited tree reproduces the 4 rows it saw
    // exactly, and with one estimator we can read how many distinct rows it held.
    const X = column([0, 1, 2, 3, 4, 5, 6, 7]);
    const y = vec([0, 1, 2, 3, 4, 5, 6, 7]);
    for (const seed of [0, 1, 2, 3]) {
      const m = new BaggingRegressor({
        nEstimators: 1,
        bootstrap: false,
        maxSamples: 0.5,
        randomState: seed,
      }).fit(X, y);
      const pred = flat(m.predict(X));
      const exact = pred.filter((p, i) => p === i).length;
      expect(exact).toBe(4);
    }
  });

  it("validates every option in the constructor and in setParams", () => {
    expect(() => new BaggingClassifier({ bootstrap: "yes" as never })).toThrow(
      InvalidParameterError
    );
    expect(() => new BaggingClassifier({ randomState: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new BaggingClassifier({ maxDepth: 0 })).toThrow(InvalidParameterError);
    expect(() => new BaggingRegressor({ maxDepth: 2.5 })).toThrow(InvalidParameterError);
    expect(() => new BaggingRegressor({ bootstrap: 1 as never })).toThrow(InvalidParameterError);
    expect(() => new BaggingClassifier().setParams({ maxFeatures: 0 })).toThrow(
      InvalidParameterError
    );
  });

  it("round-trips the default maxDepth (Infinity) through getParams, setParams and clone", () => {
    const m = new BaggingClassifier({ randomState: 4 });
    expect(m.getParams().maxDepth).toBe(Number.POSITIVE_INFINITY);
    const other = new BaggingClassifier({ maxDepth: 3 }).setParams(m.getParams());
    expect(other.getParams()).toEqual(m.getParams());
    expect(m.clone().getParams()).toEqual(m.getParams());
    const r = new BaggingRegressor({ nEstimators: 4 });
    expect(r.clone().getParams()).toEqual(r.getParams());
  });

  it("a failed refit leaves the previous model untouched and score checks the target length", () => {
    const m = new BaggingClassifier({ nEstimators: 2, randomState: 0 }).fit(
      twoClusterX,
      twoClusterY
    );
    const before = flat(m.predictProba(twoClusterX));
    expect(() => m.score(twoClusterX, vec([0, 1]))).toThrow(ShapeError);
    expect(() => m.fit(twoClusterX, vec(repeated([0.5, 1.5], 50)))).toThrow(DataValidationError);
    expect(flat(m.predictProba(twoClusterX))).toEqual(before);
  });

  it("rejects non-integer class labels", () => {
    expect(() =>
      new BaggingClassifier().fit(column([0, 1, 2, 3]), vec([0.2, 0.2, 0.9, 0.9]))
    ).toThrow(DataValidationError);
  });

  it("predicts an empty batch", () => {
    const m = new BaggingClassifier({ nEstimators: 2, randomState: 0 }).fit(
      twoClusterX,
      twoClusterY
    );
    expect(m.predict(mat([]).reshape([0, 1])).shape).toEqual([0]);
    expect(m.predictProba(mat([]).reshape([0, 1])).shape).toEqual([0, 2]);
  });
});

// ---------------------------------------------------------------------------
// GradientBoostingRegressor
// ---------------------------------------------------------------------------

describe("GradientBoostingRegressor (v1.5.0)", () => {
  const initData = { X: column([0, 1, 2, 3, 4]), y: vec([1, 2, 3, 4, 100]) };

  it("starts from the loss-specific constant (sklearn init_)", () => {
    // GradientBoostingRegressor(loss=..., n_estimators=1, learning_rate=1e-9).predict(X)[0]
    //   squared_error -> 22.0, absolute_error -> 3.0, huber(alpha=.9) -> 3.0,
    //   quantile(alpha=.9) -> 61.6, quantile(alpha=.25) -> 2.0
    const first = (loss: "ls" | "lad" | "huber" | "quantile", alpha?: number): number => {
      const m = new GradientBoostingRegressor({
        nEstimators: 1,
        learningRate: 1e-9,
        loss,
        ...(alpha === undefined ? {} : { alpha }),
      }).fit(initData.X, initData.y);
      return flat(m.predict(initData.X))[0] ?? Number.NaN;
    };
    expect(first("ls")).toBeCloseTo(22, 6);
    expect(first("lad")).toBeCloseTo(3, 6);
    expect(first("huber", 0.9)).toBeCloseTo(3, 6);
    expect(first("quantile", 0.9)).toBeCloseTo(61.6, 6);
    expect(first("quantile", 0.25)).toBeCloseTo(2, 6);
  });

  it("squared error matches scikit-learn", () => {
    // sklearn: X = arange(50)/10, y = 1000 + 2000*sin(i/8) + 100*i,
    //   GradientBoostingRegressor(n_estimators=100, max_depth=3).fit(X, y).score(X, y)
    //   -> 0.9999813218407694
    const n = 50;
    const X = column(Array.from({ length: n }, (_, i) => i / 10));
    const y = vec(
      Array.from({ length: n }, (_, i) => 1000 + 2000 * Math.sin(i / 8) + 1000 * (i / 10))
    );
    const m = new GradientBoostingRegressor({ nEstimators: 100, maxDepth: 3 }).fit(X, y);
    expect(m.score(X, y)).toBeCloseTo(0.9999813218407694, 8);
  });

  it("converges for lad, huber and quantile losses (leaf values are line-searched)", () => {
    // Same data as above. sklearn R^2 on the training set:
    //   absolute_error -> 0.9967194066610319, huber -> 0.9999581369568947.
    // Before the leaf update the lad model stopped at R^2 = 0.017 because every stage could
    // only move the prediction by learningRate * (mean of +-1).
    const n = 50;
    const X = column(Array.from({ length: n }, (_, i) => i / 10));
    const y = vec(
      Array.from({ length: n }, (_, i) => 1000 + 2000 * Math.sin(i / 8) + 1000 * (i / 10))
    );
    const fit = (loss: "lad" | "huber" | "quantile") =>
      new GradientBoostingRegressor({ nEstimators: 100, maxDepth: 3, loss }).fit(X, y).score(X, y);
    expect(fit("lad")).toBeGreaterThan(0.99);
    expect(fit("huber")).toBeGreaterThan(0.999);

    // Quantile (alpha = 0.9): R^2 is not the right yardstick, so compare the pinball loss.
    // sklearn: 43.59 for the fitted model against 149.60 for the constant 0.9-quantile.
    const ys = flat(y);
    const quantile = new GradientBoostingRegressor({
      nEstimators: 100,
      maxDepth: 3,
      loss: "quantile",
    }).fit(X, y);
    const pred = flat(quantile.predict(X));
    const pinball =
      pred.reduce((sum, p, i) => {
        const d = (ys[i] ?? 0) - p;
        return sum + (d >= 0 ? 0.9 * d : -0.1 * d);
      }, 0) / n;
    expect(pinball).toBeLessThan(60);
  });

  it("leaf values use the inverted-CDF quantile, the exact pinball minimizer", () => {
    // Constant X gives a single leaf, so one stage with learningRate 1 predicts init + leaf value.
    // sklearn 1.8 on y = [1, 2, 3, 100], n_estimators=1, learning_rate=1, max_depth=1:
    //   absolute_error -> 2.0 (lower median of the residuals), quantile(.9) -> 100.0,
    //   quantile(.25) -> 1.0, huber(.9) -> 26.375
    const X = column([0, 0, 0, 0]);
    const y = vec([1, 2, 3, 100]);
    const first = (loss: "lad" | "quantile" | "huber", alpha: number): number => {
      const m = new GradientBoostingRegressor({
        nEstimators: 1,
        learningRate: 1,
        maxDepth: 1,
        loss,
        alpha,
      }).fit(X, y);
      return flat(m.predict(X))[0] ?? Number.NaN;
    };
    expect(first("lad", 0.9)).toBeCloseTo(2, 10);
    expect(first("quantile", 0.9)).toBeCloseTo(100, 10);
    expect(first("quantile", 0.25)).toBeCloseTo(1, 10);
    expect(first("huber", 0.9)).toBeCloseTo(26.375, 10);
  });

  it("quantile loss produces the requested coverage", () => {
    // Targets of scale 4e4 with uniform noise of width 100. sklearn's quantile model at
    // alpha = 0.9 / 0.1 covers about 90% / 10% of the training targets.
    const u = lcg(12345);
    const n = 400;
    const xs = Array.from({ length: n }, (_, i) => i);
    const ys = xs.map((x) => 100 * x + (u() - 0.5) * 100);
    const X = column(xs);
    const y = vec(ys);
    for (const alpha of [0.1, 0.9]) {
      const m = new GradientBoostingRegressor({
        nEstimators: 200,
        maxDepth: 2,
        loss: "quantile",
        alpha,
      }).fit(X, y);
      const pred = flat(m.predict(X));
      const coverage = pred.filter((p, i) => (ys[i] ?? 0) <= p).length / n;
      expect(Math.abs(coverage - alpha)).toBeLessThan(0.06);
    }
  });

  it("accepts the scikit-learn loss names as aliases", () => {
    const X = column([0, 1, 2, 3, 4, 5]);
    const y = vec([0, 1, 4, 9, 16, 25]);
    const a = new GradientBoostingRegressor({ nEstimators: 5, loss: "squared_error" }).fit(X, y);
    const b = new GradientBoostingRegressor({ nEstimators: 5, loss: "ls" }).fit(X, y);
    expect(flat(a.predict(X))).toEqual(flat(b.predict(X)));
    const c = new GradientBoostingRegressor({ nEstimators: 5, loss: "absolute_error" }).fit(X, y);
    const d = new GradientBoostingRegressor({ nEstimators: 5, loss: "lad" }).fit(X, y);
    expect(flat(c.predict(X))).toEqual(flat(d.predict(X)));
    expect(() => new GradientBoostingRegressor({ loss: "mse" as never })).toThrow(
      InvalidParameterError
    );
  });

  it("subsample draws rows without replacement", () => {
    // One stage, learning rate 1, enough depth to isolate every row: the rows the tree saw are
    // reproduced exactly. 5 of 10 rows are drawn, so exactly 5 predictions equal their target.
    const X = column([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const y = vec([0, 10, 20, 30, 40, 50, 60, 70, 80, 90]);
    for (const seed of [0, 1, 2, 3, 4]) {
      const m = new GradientBoostingRegressor({
        nEstimators: 1,
        learningRate: 1,
        maxDepth: 8,
        subsample: 0.5,
        randomState: seed,
      }).fit(X, y);
      const pred = flat(m.predict(X));
      expect(pred.filter((p, i) => Math.abs(p - i * 10) < 1e-9).length).toBe(5);
    }
  });

  it("trains on float64 features when subsampling", () => {
    // Offsetting x by 1e8 makes neighbouring rows identical in float32.
    const xs = Array.from({ length: 20 }, (_, i) => 1e8 + i);
    const X = column(xs);
    const y = vec(xs.map((_, i) => (i < 10 ? 0 : 1)));
    const m = new GradientBoostingRegressor({
      nEstimators: 30,
      learningRate: 0.5,
      maxDepth: 2,
      subsample: 0.9,
      randomState: 2,
    }).fit(X, y);
    expect(m.score(X, y)).toBeGreaterThan(0.95);
  });

  it("early stopping keeps training on tiny-scale targets (relative tolerance)", () => {
    // Targets of order 1e-6: the validation loss is ~1e-13, far below the old absolute 1e-7
    // improvement threshold, so the old code stopped after nIterNoChange stages.
    const xs = Array.from({ length: 100 }, (_, i) => i);
    const X = column(xs);
    const y = vec(xs.map((x) => x * 1e-6));
    const m = new GradientBoostingRegressor({
      nEstimators: 100,
      maxDepth: 3,
      nIterNoChange: 5,
      validationFraction: 0.2,
      randomState: 1,
    }).fit(X, y);
    expect(m.nEstimatorsFitted).toBeGreaterThan(30);
    expect(m.score(X, y)).toBeGreaterThan(0.95);
  });

  it("early stopping holds out random rows, not the last ones", () => {
    // y is sorted by x. Holding out the last rows (old behaviour) validates on an extrapolation
    // region; random rows validate on points inside the training range, so the model keeps
    // improving and ends up much more accurate.
    const xs = Array.from({ length: 200 }, (_, i) => i);
    const X = column(xs);
    const y = vec(xs.map((x) => Math.sqrt(x)));
    const m = new GradientBoostingRegressor({
      nEstimators: 300,
      maxDepth: 3,
      nIterNoChange: 10,
      validationFraction: 0.2,
      randomState: 0,
    }).fit(X, y);
    expect(m.nEstimatorsFitted).toBeGreaterThan(50);
    expect(m.score(X, y)).toBeGreaterThan(0.99);
    const again = new GradientBoostingRegressor({
      nEstimators: 300,
      maxDepth: 3,
      nIterNoChange: 10,
      validationFraction: 0.2,
      randomState: 0,
    }).fit(X, y);
    expect(again.nEstimatorsFitted).toBe(m.nEstimatorsFitted);
    expect(flat(again.predict(X))).toEqual(flat(m.predict(X)));
  });

  it("warm start adds stages and equals a cold fit with the same total", () => {
    const xs = Array.from({ length: 40 }, (_, i) => i / 4);
    const X = column(xs);
    const y = vec(xs.map((x) => Math.sin(x)));
    const warm = new GradientBoostingRegressor({ nEstimators: 10, warmStart: true }).fit(X, y);
    expect(warm.nEstimatorsFitted).toBe(10);
    warm.setParams({ nEstimators: 25 });
    warm.fit(X, y);
    expect(warm.nEstimatorsFitted).toBe(25);
    const cold = new GradientBoostingRegressor({ nEstimators: 25 }).fit(X, y);
    const a = flat(warm.predict(X));
    const b = flat(cold.predict(X));
    for (let i = 0; i < a.length; i++) expect(a[i]).toBeCloseTo(b[i] ?? 0, 10);
  });

  it("warm start with a different number of features throws a ShapeError", () => {
    const m = new GradientBoostingRegressor({ nEstimators: 3, warmStart: true }).fit(
      column([0, 1, 2, 3]),
      vec([0, 1, 2, 3])
    );
    m.setParams({ nEstimators: 6 });
    expect(() =>
      m.fit(
        mat([
          [0, 1],
          [1, 2],
          [2, 3],
          [3, 4],
        ]),
        vec([0, 1, 2, 3])
      )
    ).toThrow(ShapeError);
  });

  it("is reproducible with maxFeatures under the global seed", () => {
    // Trees used Math.random for feature sampling, so a global seed had no effect.
    const u = lcg(5);
    const xs = Array.from({ length: 60 }, () => [u(), u(), u(), u()]);
    const ys = xs.map((r) => (r[0] ?? 0) * 3 + (r[2] ?? 0));
    const run = () => {
      setSeed(77);
      return flat(
        new GradientBoostingRegressor({ nEstimators: 15, maxFeatures: 2 })
          .fit(mat(xs), vec(ys))
          .predict(mat(xs))
      );
    };
    expect(run()).toEqual(run());
    const seeded = (s: number) =>
      flat(
        new GradientBoostingRegressor({ nEstimators: 15, maxFeatures: 2, randomState: s })
          .fit(mat(xs), vec(ys))
          .predict(mat(xs))
      );
    expect(seeded(1)).toEqual(seeded(1));
    expect(seeded(1)).not.toEqual(seeded(2));
  });

  it("validates every option in the constructor", () => {
    expect(() => new GradientBoostingRegressor({ maxFeatures: 2.5 })).toThrow(
      InvalidParameterError
    );
    expect(() => new GradientBoostingRegressor({ maxFeatures: 0 })).toThrow(InvalidParameterError);
    expect(() => new GradientBoostingRegressor({ nIterNoChange: 0 })).toThrow(
      InvalidParameterError
    );
    expect(() => new GradientBoostingRegressor({ validationFraction: 1.5 })).toThrow(
      InvalidParameterError
    );
    expect(() => new GradientBoostingRegressor({ minSamplesLeaf: 0 })).toThrow(
      InvalidParameterError
    );
    expect(() => new GradientBoostingRegressor({ randomState: Number.NaN })).toThrow(
      InvalidParameterError
    );
    expect(() => new GradientBoostingRegressor({ warmStart: "yes" as never })).toThrow(
      InvalidParameterError
    );
  });

  it("clones with the new options and round-trips getParams", () => {
    const m = new GradientBoostingRegressor({
      nEstimators: 7,
      minSamplesLeaf: 2,
      randomState: 3,
      nIterNoChange: 4,
      loss: "huber",
      alpha: 0.8,
    });
    expect(m.clone().getParams()).toEqual(m.getParams());
    const plain = new GradientBoostingRegressor();
    expect(plain.clone().getParams()).toEqual(plain.getParams());
  });

  it("returns float64 predictions and rejects mismatched score targets", () => {
    const X = column([0, 1, 2, 3]);
    const y = vec([1e8, 1e8 + 0.5, 1e8 + 1, 1e8 + 1.5]);
    const m = new GradientBoostingRegressor({ nEstimators: 1, learningRate: 1e-12 }).fit(X, y);
    expect(m.predict(X).dtype).toBe("float64");
    expect(flat(m.predict(X))[0]).toBeCloseTo(1e8 + 0.75, 5);
    expect(() => m.score(X, vec([1, 2]))).toThrow(ShapeError);
  });

  it("a failed refit leaves the previous model untouched", () => {
    const m = new GradientBoostingRegressor({ nEstimators: 2 }).fit(
      column([0, 1, 2]),
      vec([0, 1, 2])
    );
    const before = flat(m.predict(column([0, 1, 2])));
    expect(() => m.fit(column([0, 1, 2]), vec([0, 1]))).toThrow(ShapeError);
    expect(flat(m.predict(column([0, 1, 2])))).toEqual(before);
    expect(m.nEstimatorsFitted).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// GradientBoostingClassifier
// ---------------------------------------------------------------------------

describe("GradientBoostingClassifier (v1.5.0)", () => {
  it("starts from the log of the class ratio, not a Laplace-smoothed ratio", () => {
    // GradientBoostingClassifier(n_estimators=1, learning_rate=1e-9) on 9 negatives and
    // 1 positive: predict_proba(X)[0] -> [0.9, 0.1]. Laplace smoothing gave 1/6.
    const X = column([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const y = vec([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
    const m = new GradientBoostingClassifier({ nEstimators: 1, learningRate: 1e-9 }).fit(X, y);
    const p = rows(m.predictProba(X));
    expect(p[0]?.[0]).toBeCloseTo(0.9, 6);
    expect(p[0]?.[1]).toBeCloseTo(0.1, 6);
  });

  it("predicts the smaller label on an exact tie, like argmax of the probabilities", () => {
    // Constant features and balanced labels: the log-odds are exactly 0 for every row.
    const X = mat([[1], [1], [1], [1]]);
    const y = vec([0, 1, 0, 1]);
    const m = new GradientBoostingClassifier({ nEstimators: 3 }).fit(X, y);
    expect(flat(m.predict(X))).toEqual([0, 0, 0, 0]);
    expect(rows(m.predictProba(X))[0]).toEqual([0.5, 0.5]);
  });

  it("keeps the tail probability of the other class when the model is very confident", () => {
    // 1 - sigmoid(z) is quantized to multiples of 1.1e-16 (and is 0 for z > 36.7), so the old
    // code reported 2.1e-13 here. The true value is sigmoid(-z) = 7.0e-17.
    const X = column([0, 0, 0, 0, 1, 1, 1, 1]);
    const y = vec([0, 0, 0, 0, 1, 1, 1, 1]);
    const m = new GradientBoostingClassifier({
      nEstimators: 40,
      learningRate: 1,
      maxDepth: 1,
    }).fit(X, y);
    const p = rows(m.predictProba(column([1])))[0] ?? [];
    expect(p[0]).toBeGreaterThan(0);
    expect(p[0]).toBeLessThan(1e-15);
    expect(p[1]).toBe(1);
    expect(flat(m.decisionFunction(column([1])))[0]).toBeGreaterThan(30);
  });

  it("separable multiclass data is classified perfectly", () => {
    const xs = repeated([0, 1, 2], 10);
    const m = new GradientBoostingClassifier({ nEstimators: 20, maxDepth: 2 }).fit(
      column(xs),
      vec(xs)
    );
    expect(m.score(column(xs), vec(xs))).toBe(1);
    const p = rows(m.predictProba(column([0, 1, 2])));
    for (const r of p) expect(r.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(m.decisionFunction(column([0, 1, 2])).shape).toEqual([3, 3]);
    expect(m.nEstimatorsFitted).toBe(20);
  });

  it("subsample draws rows without replacement and is reproducible per seed", () => {
    const { X, y } = noisyClasses(60, 3);
    const make = (seed: number) =>
      new GradientBoostingClassifier({
        nEstimators: 10,
        subsample: 0.5,
        maxDepth: 2,
        randomState: seed,
      }).fit(X, y);
    expect(flat(make(1).predictProba(X))).toEqual(flat(make(1).predictProba(X)));
    expect(flat(make(1).predictProba(X))).not.toEqual(flat(make(2).predictProba(X)));
  });

  it("warm start continues the ensemble and matches a cold fit", () => {
    const { X, y } = noisyClasses(50, 4);
    const warm = new GradientBoostingClassifier({ nEstimators: 5, warmStart: true }).fit(X, y);
    warm.setParams({ nEstimators: 12 });
    warm.fit(X, y);
    expect(warm.nEstimatorsFitted).toBe(12);
    const cold = new GradientBoostingClassifier({ nEstimators: 12 }).fit(X, y);
    const a = flat(warm.predictProba(X));
    const b = flat(cold.predictProba(X));
    for (let i = 0; i < a.length; i++) expect(a[i]).toBeCloseTo(b[i] ?? 0, 10);
  });

  it("warm start rejects changed classes or features", () => {
    const { X, y } = noisyClasses(20, 6);
    const m = new GradientBoostingClassifier({ nEstimators: 2, warmStart: true }).fit(X, y);
    m.setParams({ nEstimators: 4 });
    const shifted = vec(flat(y).map((v) => v + 5));
    expect(() => m.fit(X, shifted)).toThrow(DataValidationError);
    expect(() => m.fit(column(flat(y)), y)).toThrow(ShapeError);
  });

  it("rejects non-integer class labels and a single class", () => {
    const X = column([0, 1, 2, 3]);
    expect(() => new GradientBoostingClassifier().fit(X, vec([0.5, 0.5, 1.5, 1.5]))).toThrow(
      DataValidationError
    );
    expect(() => new GradientBoostingClassifier().fit(X, vec([1, 1, 1, 1]))).toThrow(
      InvalidParameterError
    );
  });

  it("validates options and clones with the new options", () => {
    expect(() => new GradientBoostingClassifier({ maxFeatures: 1.5 })).toThrow(
      InvalidParameterError
    );
    expect(() => new GradientBoostingClassifier({ minSamplesLeaf: 0 })).toThrow(
      InvalidParameterError
    );
    expect(() => new GradientBoostingClassifier({ randomState: Number.NaN })).toThrow(
      InvalidParameterError
    );
    const m = new GradientBoostingClassifier({ nEstimators: 9, minSamplesLeaf: 2, randomState: 1 });
    expect(m.clone().getParams()).toEqual(m.getParams());
  });

  it("a failed refit leaves the previous model untouched", () => {
    const { X, y } = noisyClasses(20, 8);
    const m = new GradientBoostingClassifier({ nEstimators: 2 }).fit(X, y);
    const before = flat(m.predictProba(X));
    expect(() => m.fit(X, vec([0, 1]))).toThrow(ShapeError);
    expect(flat(m.predictProba(X))).toEqual(before);
    expect(flat(m.classes as Tensor)).toEqual([0, 1]);
  });

  it("feature importances sum to one and favour the informative feature", () => {
    const u = lcg(14);
    const xs = Array.from({ length: 100 }, () => [u(), u()]);
    const ys = xs.map((r) => ((r[0] ?? 0) > 0.5 ? 1 : 0));
    const m = new GradientBoostingClassifier({ nEstimators: 20, maxDepth: 2 }).fit(
      mat(xs),
      vec(ys)
    );
    const imp = flat(m.featureImportances);
    expect(imp[0]! + imp[1]!).toBeCloseTo(1, 10);
    expect(imp[0]).toBeGreaterThan(0.8);
  });
});

// ---------------------------------------------------------------------------
// Stacking
// ---------------------------------------------------------------------------

describe("StackingRegressor (v1.5.0)", () => {
  const y10 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  const X10 = column(y10);

  it("fits the final estimator on out-of-fold predictions", () => {
    // With cv = 5 and rows dealt to folds in turn, fold f holds rows {f, f + 5}. A mean
    // predictor trained without them predicts (55 - y[f] - y[f + 5]) / 8 for both rows.
    const final = new RecordingRegressor();
    new StackingRegressor({ estimators: [new MeanRegressor()], finalEstimator: final, cv: 5 }).fit(
      X10,
      vec(y10)
    );
    const seen = flat(final.seenX as Tensor);
    for (let i = 0; i < 10; i++) {
      const f = i % 5;
      expect(seen[i]).toBeCloseTo((55 - y10[f]! - y10[f + 5]!) / 8, 12);
    }
  });

  it("cv: 1 keeps the previous in-sample behaviour", () => {
    const final = new RecordingRegressor();
    new StackingRegressor({ estimators: [new MeanRegressor()], finalEstimator: final, cv: 1 }).fit(
      X10,
      vec(y10)
    );
    expect(flat(final.seenX as Tensor)).toEqual(new Array<number>(10).fill(5.5));
  });

  it("defaults to 5 folds", () => {
    const final = new RecordingRegressor();
    new StackingRegressor({ estimators: [new MeanRegressor()], finalEstimator: final }).fit(
      X10,
      vec(y10)
    );
    expect(flat(final.seenX as Tensor)[0]).toBeCloseTo((55 - 1 - 6) / 8, 12);
  });

  it("keeps float64 precision of base predictions in the meta-features", () => {
    const final = new RecordingRegressor();
    const y = y10.map((v) => 1e8 + v * 0.1);
    new StackingRegressor({ estimators: [new MeanRegressor()], finalEstimator: final, cv: 1 }).fit(
      X10,
      vec(y)
    );
    const mean = y.reduce((a, b) => a + b, 0) / y.length;
    expect(flat(final.seenX as Tensor)[0]).toBeCloseTo(mean, 6);
  });

  it("passthrough appends the original features after the base predictions", () => {
    const final = new RecordingRegressor();
    new StackingRegressor({
      estimators: [new MeanRegressor(), new MeanRegressor()],
      finalEstimator: final,
      passthrough: true,
      cv: 1,
    }).fit(X10, vec(y10));
    const X = final.seenX as Tensor;
    expect(X.shape).toEqual([10, 3]);
    expect(rows(X)[3]).toEqual([5.5, 5.5, 4]);
  });

  it("does not leak the targets through a memorizing base estimator", () => {
    // Pure noise targets: a deep tree memorizes them. In-sample (cv: 1) its prediction equals
    // the target; out-of-fold it carries no information about it.
    const u = lcg(99);
    const xs = Array.from({ length: 60 }, () => [u()]);
    const ys = xs.map(() => u());
    const tree = () => new DecisionTreeRegressor({ maxDepth: 30 });

    const inSample = new RecordingRegressor();
    new StackingRegressor({ estimators: [tree()], finalEstimator: inSample, cv: 1 }).fit(
      mat(xs),
      vec(ys)
    );
    const memorized = flat(inSample.seenX as Tensor);
    for (let i = 0; i < ys.length; i++) expect(memorized[i]).toBeCloseTo(ys[i] ?? 0, 5);

    const oof = new RecordingRegressor();
    new StackingRegressor({ estimators: [tree()], finalEstimator: oof }).fit(mat(xs), vec(ys));
    const meta = flat(oof.seenX as Tensor);
    const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
    const mm = mean(meta);
    const my = mean(ys);
    let sxy = 0;
    let sxx = 0;
    let syy = 0;
    for (let i = 0; i < ys.length; i++) {
      sxy += ((meta[i] ?? 0) - mm) * ((ys[i] ?? 0) - my);
      sxx += ((meta[i] ?? 0) - mm) ** 2;
      syy += ((ys[i] ?? 0) - my) ** 2;
    }
    expect(Math.abs(sxy / Math.sqrt(sxx * syy))).toBeLessThan(0.5);
  });

  it("predicts end to end with the default linear meta-learner", () => {
    const xs = Array.from({ length: 40 }, (_, i) => i / 4);
    const X = column(xs);
    const y = vec(xs.map((x) => 2 * x + 1));
    const m = new StackingRegressor({
      estimators: [new DecisionTreeRegressor({ maxDepth: 4 }), new MeanRegressor()],
    }).fit(X, y);
    expect(m.predict(X).shape).toEqual([40]);
    expect(m.score(X, y)).toBeGreaterThan(0.95);
  });

  it("cannot cross-validate an estimator it cannot clone (explicit cv) and says how to fix it", () => {
    class NoClone implements Regressor {
      constructor(required: { a: number }) {
        if (typeof required.a !== "number") throw new Error("a is required");
      }
      fit(): this {
        return this;
      }
      predict(X: Tensor): Tensor {
        return tensor(new Float64Array(X.shape[0] ?? 0));
      }
      score(): number {
        return 0;
      }
      getParams(): Record<string, unknown> {
        return {};
      }
      setParams(): this {
        return this;
      }
    }
    const make = (cv: number) =>
      new StackingRegressor({ estimators: [new NoClone({ a: 1 })], cv }).fit(X10, vec(y10));
    expect(() => make(5)).toThrow(InvalidParameterError);
    expect(() => make(5)).toThrow(/cv: 1/);
    expect(() => make(1)).not.toThrow();
  });

  it("validates cv and exposes it through getParams/setParams/clone", () => {
    const base = { estimators: [new MeanRegressor()] };
    expect(() => new StackingRegressor({ ...base, cv: 0 })).toThrow(InvalidParameterError);
    expect(() => new StackingRegressor({ ...base, cv: 2.5 })).toThrow(InvalidParameterError);
    const m = new StackingRegressor({ ...base, cv: 3 });
    expect(m.getParams().cv).toBe(3);
    m.setParams({ cv: 4 });
    expect(m.getParams().cv).toBe(4);
    expect(() => m.setParams({ cv: 0 })).toThrow(InvalidParameterError);
    const copy = m.clone();
    expect(copy).not.toBe(m);
    expect(copy.getParams()).toEqual(m.getParams());
  });

  it("is unfitted after a failed refit and score checks lengths", () => {
    const m = new StackingRegressor({ estimators: [new MeanRegressor()], cv: 2 }).fit(
      X10,
      vec(y10)
    );
    expect(() => m.score(X10, vec([1, 2]))).toThrow(ShapeError);
    expect(() => m.fit(X10, vec([1, 2]))).toThrow(ShapeError);
    expect(() => m.predict(X10)).toThrow(NotFittedError);
  });
});

describe("StackingClassifier (v1.5.0)", () => {
  function randomLabelData(n: number, seed: number): { X: Tensor; y: Tensor; yv: number[] } {
    const u = lcg(seed);
    const xs = Array.from({ length: n }, () => [u(), u(), u()]);
    const yv = xs.map(() => (u() < 0.5 ? 0 : 1));
    return { X: mat(xs), y: vec(yv), yv };
  }

  it("meta-features are out-of-fold probabilities (a memorizing tree cannot leak labels)", () => {
    // Labels are pure noise, so a deep tree memorizes the training set. In-sample (cv: 1) its
    // probability column equals y exactly; out-of-fold it is no better than chance.
    const { X, y, yv } = randomLabelData(60, 17);
    const tree = () => new DecisionTreeClassifier({ maxDepth: 30 });

    const inSampleFinal = new RecordingClassifier();
    new StackingClassifier({ estimators: [tree()], finalEstimator: inSampleFinal, cv: 1 }).fit(
      X,
      y
    );
    expect(flat(inSampleFinal.seenX as Tensor)).toEqual(yv);

    const oofFinal = new RecordingClassifier();
    new StackingClassifier({ estimators: [tree()], finalEstimator: oofFinal }).fit(X, y);
    const meta = flat(oofFinal.seenX as Tensor);
    expect(meta.length).toBe(60);
    const agree = meta.filter((p, i) => (p > 0.5 ? 1 : 0) === yv[i]).length / 60;
    expect(agree).toBeLessThan(0.75);
  });

  it("stacks one probability column per estimator for two classes", () => {
    const { X, y } = randomLabelData(30, 2);
    const final = new RecordingClassifier();
    new StackingClassifier({
      estimators: [new DecisionTreeClassifier({ maxDepth: 2 }), new DecisionTreeClassifier()],
      finalEstimator: final,
    }).fit(X, y);
    expect((final.seenX as Tensor).shape).toEqual([30, 2]);
  });

  it("stacks K columns per estimator for K > 2 classes, or the label with stackMethod: predict", () => {
    const u = lcg(3);
    const xs = Array.from({ length: 45 }, () => [u(), u()]);
    const ys = xs.map((_, i) => i % 3);
    const base = () => [new DecisionTreeClassifier({ maxDepth: 2 })];
    const proba = new RecordingClassifier();
    new StackingClassifier({ estimators: base(), finalEstimator: proba }).fit(mat(xs), vec(ys));
    expect((proba.seenX as Tensor).shape).toEqual([45, 3]);
    for (const r of rows(proba.seenX as Tensor)) {
      expect(r.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
    }

    const labels = new RecordingClassifier();
    new StackingClassifier({
      estimators: base(),
      finalEstimator: labels,
      stackMethod: "predict",
    }).fit(mat(xs), vec(ys));
    expect((labels.seenX as Tensor).shape).toEqual([45, 1]);
  });

  it("passthrough appends the features", () => {
    const { X, y } = randomLabelData(30, 4);
    const final = new RecordingClassifier();
    new StackingClassifier({
      estimators: [new DecisionTreeClassifier({ maxDepth: 2 })],
      finalEstimator: final,
      passthrough: true,
    }).fit(X, y);
    expect((final.seenX as Tensor).shape).toEqual([30, 4]);
  });

  it("falls back to in-sample predictions when a class has a single sample", () => {
    const X = column([0, 1, 2, 3, 4, 5, 6, 7]);
    const y = vec([0, 0, 0, 0, 0, 0, 0, 1]);
    const final = new RecordingClassifier();
    const m = new StackingClassifier({
      estimators: [new DecisionTreeClassifier({ maxDepth: 3 })],
      finalEstimator: final,
    });
    expect(() => m.fit(X, y)).not.toThrow();
    expect((final.seenX as Tensor).shape).toEqual([8, 1]);
    expect(m.predict(X).shape).toEqual([8]);
  });

  it("fits and predicts a separable problem with the default logistic meta-learner", () => {
    const xs = repeated([0, 1], 20);
    const m = new StackingClassifier({
      estimators: [new DecisionTreeClassifier({ maxDepth: 2 }), new DecisionTreeClassifier()],
    }).fit(column(xs), vec(xs));
    expect(m.score(column(xs), vec(xs))).toBe(1);
    const p = rows(m.predictProba(column([0, 1])));
    expect(p[0]?.[0]).toBeGreaterThan(0.5);
    expect(p[1]?.[1]).toBeGreaterThan(0.5);
  });

  it("validates stackMethod and cv, exposes them in getParams, and clones", () => {
    const est = () => [new DecisionTreeClassifier()];
    expect(
      () => new StackingClassifier({ estimators: est(), stackMethod: "soft" as never })
    ).toThrow(InvalidParameterError);
    expect(() => new StackingClassifier({ estimators: est(), cv: 0 })).toThrow(
      InvalidParameterError
    );
    const m = new StackingClassifier({ estimators: est(), cv: 3, stackMethod: "predict" });
    expect(m.getParams()).toEqual({
      nEstimators: 1,
      passthrough: false,
      cv: 3,
      stackMethod: "predict",
    });
    m.setParams({ stackMethod: "predictProba", cv: 2 });
    expect(m.getParams().stackMethod).toBe("predictProba");
    expect(() => m.setParams({ stackMethod: "x" })).toThrow(InvalidParameterError);
    const copy = m.clone();
    expect(copy).not.toBe(m);
    expect(copy.getParams()).toEqual(m.getParams());
  });

  it("rejects non-integer labels and mismatched score targets; refit failure leaves it unfitted", () => {
    const { X, y } = randomLabelData(20, 5);
    expect(() =>
      new StackingClassifier({ estimators: [new DecisionTreeClassifier()] }).fit(
        X,
        vec(new Array<number>(20).fill(0.5))
      )
    ).toThrow(DataValidationError);
    const m = new StackingClassifier({
      estimators: [new DecisionTreeClassifier({ maxDepth: 2 })],
    }).fit(X, y);
    expect(() => m.score(X, vec([0, 1]))).toThrow(ShapeError);
    expect(() => m.fit(X, vec([0, 1]))).toThrow(ShapeError);
    expect(() => m.predict(X)).toThrow(NotFittedError);
    expect(m.classes).toBeUndefined();
  });
});
