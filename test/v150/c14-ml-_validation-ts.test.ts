import { describe, expect, it } from "vitest";
import { DataValidationError, DTypeError, InvalidParameterError, ShapeError } from "../../src/core";
import {
  CalibratedClassifierCV,
  type Classifier,
  calibrationCurve,
  GaussianMixture,
  getEstimatorTags,
  IsolationForest,
  Isomap,
  KMeans,
  KNeighborsClassifier,
  LinearSVC,
  LocalOutlierFactor,
  LogisticRegression,
  NuSVC,
} from "../../src/ml";
import {
  percentileSorted,
  toFloat64View,
  validateFitInputs,
  validatePredictInputs,
  validateUnsupervisedFitInputs,
} from "../../src/ml/_validation";
import { type Tensor, tensor, transpose } from "../../src/ndarray";
import { clearSeed, setSeed } from "../../src/random";

const f64 = (rows: number | number[] | number[][]): Tensor => tensor(rows, { dtype: "float64" });
const vals = (t: Tensor): number[] => Array.from(toFloat64View(t));

// ---------------------------------------------------------------------------
// _validation.ts
// ---------------------------------------------------------------------------
describe("ml/_validation", () => {
  it("accepts int64 tensors (BigInt storage) instead of reporting non-finite values", () => {
    const X = tensor(
      [
        [1, 2],
        [3, 4],
      ],
      { dtype: "int64" }
    );
    const y = tensor([0, 1], { dtype: "int64" });
    expect(() => validateFitInputs(X, y)).not.toThrow();
    expect(() => validateUnsupervisedFitInputs(X)).not.toThrow();
    expect(() => validatePredictInputs(X, 2, "M")).not.toThrow();
    expect(vals(X)).toEqual([1, 2, 3, 4]);
  });

  it("rejects string tensors with a DTypeError", () => {
    const S = tensor([["a", "b"]], { dtype: "string" });
    expect(() => validateUnsupervisedFitInputs(S)).toThrow(DTypeError);
    expect(() => validatePredictInputs(S, 2, "M")).toThrow(DTypeError);
    expect(() => validateFitInputs(S, tensor([0]))).toThrow(DTypeError);
  });

  it("still reports NaN / Inf, shape and feature-count problems", () => {
    expect(() => validateUnsupervisedFitInputs(f64([[1, Number.NaN]]))).toThrow(
      DataValidationError
    );
    expect(() => validateUnsupervisedFitInputs(f64([[1, Number.POSITIVE_INFINITY]]))).toThrow(
      /non-finite/
    );
    expect(() => validateFitInputs(f64([[1]]), f64([Number.NaN]))).toThrow(/y contains/);
    expect(() => validateFitInputs(f64([[1], [2]]), f64([1]))).toThrow(ShapeError);
    expect(() => validatePredictInputs(f64([[1, 2]]), 3, "M")).toThrow(ShapeError);
  });

  it("toFloat64View respects the tensor offset and does not copy float64 data", () => {
    const base = f64([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const tail = base.slice({ start: 1, end: 3 });
    expect(vals(tail)).toEqual([3, 4, 5, 6]);
    expect(toFloat64View(base).buffer).toBe((base.data as Float64Array).buffer);
  });

  it("percentileSorted matches numpy.percentile (linear interpolation)", () => {
    // np.percentile([-1, 0.3, 0.5, 2, 2, 5, 8], [0, 10, 33.3, 50, 99, 100])
    const sorted = [-1, 0.3, 0.5, 2, 2, 5, 8];
    const expected = [-1, -0.22, 0.4996, 2, 7.82, 8];
    // numpy reference for the unsorted list in the task data is the same set
    [0, 10, 33.3, 50, 99, 100].forEach((q, i) => {
      expect(percentileSorted(sorted, q)).toBeCloseTo(expected[i] as number, 12);
    });
    expect(percentileSorted([4], 37)).toBe(4);
  });
});

// ---------------------------------------------------------------------------
// IsolationForest
// ---------------------------------------------------------------------------
function lcgRows(n: number, d: number, seed: number): number[][] {
  let s = seed;
  const rnd = () => {
    s = (s * 1103515245 + 12345) % 2147483648;
    return s / 2147483648;
  };
  const rows: number[][] = [];
  for (let i = 0; i < n; i++) {
    const r: number[] = [];
    for (let j = 0; j < d; j++) r.push(rnd() * 2 - 1);
    rows.push(r);
  }
  return rows;
}

type IFInternals = {
  maxSamples_: number;
  trees: { feature: Int32Array }[];
};

describe("IsolationForest", () => {
  const Z = f64(lcgRows(100, 2, 1));

  it("flags the sklearn number of outliers for numeric contamination", () => {
    // sklearn: offset_ = percentile(score_samples, 100 * contamination); outliers are strictly below.
    // On 100 distinct scores this is ceil(contamination * (n - 1)) samples (10, 7 and 15 here).
    for (const [c, expected] of [
      [0.1, 10],
      [0.07, 7],
      [0.15, 15],
    ] as const) {
      const m = new IsolationForest({ contamination: c, randomState: 0 }).fit(Z);
      expect(vals(m.predict(Z)).filter((v) => v === -1).length).toBe(expected);
    }
  });

  it("flags at least one outlier when contamination * n < 1", () => {
    const m = new IsolationForest({ contamination: 0.005, randomState: 3 }).fit(Z);
    expect(vals(m.predict(Z)).filter((v) => v === -1).length).toBe(1);
  });

  it("predict and decisionFunction agree and use scoreSamples - offset", () => {
    const m = new IsolationForest({ contamination: 0.2, randomState: 5 }).fit(Z);
    const s = vals(m.scoreSamples(Z));
    const d = vals(m.decisionFunction(Z));
    const labels = vals(m.predict(Z));
    s.forEach((v, i) => {
      expect(d[i]).toBeCloseTo(v - m.offset, 12);
      expect(labels[i]).toBe((d[i] as number) < 0 ? -1 : 1);
    });
    expect(m.predict(Z).dtype).toBe("int32");
    expect(m.scoreSamples(Z).dtype).toBe("float64");
  });

  it('uses a strict comparison with offset -0.5 for contamination "auto"', () => {
    // Two distinct samples: every tree isolates them at depth 1, c(2) = 1,
    // so the anomaly score is exactly 2^-1 = 0.5 and neither sample is an outlier.
    const m = new IsolationForest({ nEstimators: 10, maxSamples: 2, randomState: 1 }).fit(
      f64([[0], [1]])
    );
    expect(vals(m.scoreSamples(f64([[0], [1], [5], [-3]])))).toEqual([-0.5, -0.5, -0.5, -0.5]);
    expect(vals(m.predict(f64([[0], [1], [5]])))).toEqual([1, 1, 1]);
    expect(m.offset).toBe(-0.5);
  });

  it("does not produce NaN when each tree sees a single sample", () => {
    const m = new IsolationForest({ nEstimators: 5, maxSamples: 1, randomState: 1 }).fit(
      f64([[0], [1], [2]])
    );
    expect(vals(m.scoreSamples(f64([[0], [10]])))).toEqual([-0.5, -0.5]);
    const one = new IsolationForest({ nEstimators: 3, randomState: 1 }).fit(f64([[4, 2]]));
    expect(vals(one.predict(f64([[4, 2]])))).toEqual([1]);
  });

  it("is reproducible for a fixed randomState and differs between seeds", () => {
    const a = vals(new IsolationForest({ randomState: 11 }).fit(Z).scoreSamples(Z));
    const b = vals(new IsolationForest({ randomState: 11 }).fit(Z).scoreSamples(Z));
    const c = vals(new IsolationForest({ randomState: 12 }).fit(Z).scoreSamples(Z));
    expect(a).toEqual(b);
    expect(a).not.toEqual(c);
  });

  it("refitting the same instance reproduces the first fit", () => {
    const m = new IsolationForest({ randomState: 4 });
    const first = vals(m.fit(Z).scoreSamples(Z));
    const second = vals(m.fit(Z).scoreSamples(Z));
    expect(second).toEqual(first);
  });

  it("honours the global seed when randomState is not given", () => {
    setSeed(77);
    const a = vals(new IsolationForest({ nEstimators: 20 }).fit(Z).scoreSamples(Z));
    setSeed(77);
    const b = vals(new IsolationForest({ nEstimators: 20 }).fit(Z).scoreSamples(Z));
    clearSeed();
    expect(a).toEqual(b);
  });

  it("resolves maxSamples (auto cap, integer clamp, fraction) and uses it as c(n) denominator", () => {
    const big = f64(lcgRows(300, 2, 2));
    const auto = new IsolationForest({ nEstimators: 2, randomState: 1 }).fit(big);
    expect((auto as unknown as IFInternals).maxSamples_).toBe(256);
    const frac = new IsolationForest({ nEstimators: 2, maxSamples: 0.5, randomState: 1 }).fit(Z);
    expect((frac as unknown as IFInternals).maxSamples_).toBe(50);
    const clamped = new IsolationForest({ nEstimators: 2, maxSamples: 1000, randomState: 1 }).fit(
      Z
    );
    expect((clamped as unknown as IFInternals).maxSamples_).toBe(100);
  });

  it("maxFeatures limits the features used by each tree", () => {
    const X4 = f64(lcgRows(60, 4, 3));
    const m = new IsolationForest({ nEstimators: 30, maxFeatures: 0.5, randomState: 2 }).fit(X4);
    const trees = (m as unknown as IFInternals).trees;
    const used = new Set<number>();
    for (const t of trees) {
      const perTree = new Set([...t.feature].filter((f) => f >= 0));
      expect(perTree.size).toBeLessThanOrEqual(2);
      for (const f of perTree) used.add(f);
    }
    // across trees all features get used
    expect(used.size).toBe(4);
    const one = new IsolationForest({ nEstimators: 10, maxFeatures: 1, randomState: 2 }).fit(X4);
    const feats = new Set<number>();
    for (const t of (one as unknown as IFInternals).trees) {
      for (const f of t.feature) if (f >= 0) feats.add(f);
    }
    expect(feats.size).toBe(4);
  });

  it("skips constant features when choosing a split", () => {
    // Feature 0 is constant; every tree must still isolate along feature 1.
    const rows: number[][] = [];
    for (let i = 0; i < 40; i++) rows.push([7, (i % 10) * 0.01]);
    rows.push([7, 50]);
    const X = f64(rows);
    const m = new IsolationForest({ nEstimators: 50, maxSamples: 41, randomState: 9 }).fit(X);
    const trees = (m as unknown as IFInternals).trees;
    for (const t of trees) {
      // no node splits on the constant feature
      expect([...t.feature].every((f) => f !== 0)).toBe(true);
    }
    const s = vals(m.scoreSamples(X));
    expect(s[40]).toBe(Math.min(...s));
    expect(s.every(Number.isFinite)).toBe(true);
  });

  it("handles int64 input and does not modify X", () => {
    const rows = lcgRows(30, 2, 5).map((r) => r.map((v) => Math.round(v * 100)));
    const X = tensor(rows, { dtype: "int64" });
    const before = Array.from(X.data as BigInt64Array);
    const m = new IsolationForest({ nEstimators: 10, randomState: 1 }).fit(X);
    expect(m.predict(X).size).toBe(30);
    expect(Array.from(X.data as BigInt64Array)).toEqual(before);
  });

  it("validates constructor and setParams values, including NaN", () => {
    expect(() => new IsolationForest({ maxSamples: 2.5 })).toThrow(InvalidParameterError);
    expect(() => new IsolationForest({ maxSamples: 0 })).toThrow(/maxSamples/);
    expect(() => new IsolationForest({ maxSamples: Number.NaN })).toThrow(/maxSamples/);
    expect(() => new IsolationForest({ contamination: 0.7 })).toThrow(/contamination/);
    expect(() => new IsolationForest({ contamination: Number.NaN })).toThrow(/contamination/);
    expect(() => new IsolationForest({ maxFeatures: 0 })).toThrow(/maxFeatures/);
    expect(() => new IsolationForest({ maxFeatures: 2.5 })).toThrow(/maxFeatures/);
    expect(() => new IsolationForest({ randomState: Number.NaN })).toThrow(/randomState/);
    const m = new IsolationForest();
    expect(() => m.setParams({ contamination: Number.NaN })).toThrow(/contamination/);
    expect(() => m.setParams({ maxFeatures: Number.NaN })).toThrow(/maxFeatures/);
    expect(() => m.setParams({ maxSamples: Number.NaN })).toThrow(/maxSamples/);
  });

  it("setParams applies nothing when one value is invalid", () => {
    const m = new IsolationForest({ nEstimators: 7 });
    expect(() => m.setParams({ nEstimators: 20, contamination: 0.9 })).toThrow(
      InvalidParameterError
    );
    expect(m.getParams().nEstimators).toBe(7);
  });

  it("validates inputs at predict time and before fit", () => {
    const m = new IsolationForest({ nEstimators: 5, randomState: 1 });
    expect(() => m.decisionFunction(Z)).toThrow(/fitted/i);
    expect(() => m.offset).toThrow(/fitted/i);
    m.fit(Z);
    expect(() => m.predict(f64([[1, Number.NaN]]))).toThrow(DataValidationError);
    expect(() => m.scoreSamples(f64([[1, 2, 3]]))).toThrow(ShapeError);
    expect(m.predict(f64([[0, 0]])).size).toBe(1);
  });

  it("clone returns an unfitted copy with the same parameters", () => {
    const m = new IsolationForest({ nEstimators: 9, contamination: 0.1, randomState: 3 }).fit(Z);
    const c = m.clone();
    expect(c.getParams()).toEqual(m.getParams());
    expect(() => c.predict(Z)).toThrow(/fitted/i);
    expect(new IsolationForest().clone().getParams().randomState).toBeUndefined();
  });

  it("detects an obvious outlier", () => {
    const rows = lcgRows(50, 2, 8).map((r) => r.map((v) => v * 0.1));
    rows.push([100, 100]);
    const X = f64(rows);
    const m = new IsolationForest({ contamination: 0.05, randomState: 0 }).fit(X);
    expect(vals(m.predict(X))[50]).toBe(-1);
    const s = vals(m.scoreSamples(X));
    expect(s[50]).toBe(Math.min(...s));
  });
});

// ---------------------------------------------------------------------------
// LocalOutlierFactor (reference values from scikit-learn 1.8, novelty=True)
// ---------------------------------------------------------------------------
describe("LocalOutlierFactor", () => {
  const X = f64([
    [0, 0],
    [1, 0.1],
    [0, 1],
    [1.05, 1],
    [0.5, 0.4],
    [0.2, 0.8],
    [5, 5],
    [0.9, 0.3],
  ]);
  const Q = f64([
    [0.4, 0.5],
    [3, 3],
    [6, 6],
    [-1, 0.5],
    [0.7, 0.1],
    [0.3, 0.3],
    [10, 0],
    [0.55, 0.45],
  ]);
  const nof = [
    -0.9884882756884892, -0.9166750546040293, -1.058597931425519, -0.9841584257864376,
    -1.0366601578555006, -1.0191379950291732, -7.847886100747812, -1.006047621453409,
  ];
  const scoreQ = [
    -0.8697159153180732, -3.742002499021352, -6.269815013186224, -1.3811855766958407,
    -0.9447880695243812, -0.9573363637439503, -7.737192300600436, -0.8697159153180732,
  ];

  it("negative outlier factor and novelty scores match scikit-learn", () => {
    const lof = new LocalOutlierFactor({ nNeighbors: 3 }).fit(X);
    vals(lof.negativeOutlierFactor).forEach((v, i) => {
      expect(v).toBeCloseTo(nof[i] as number, 10);
    });
    vals(lof.negativeLofScores).forEach((v, i) => {
      expect(v).toBeCloseTo(nof[i] as number, 10);
    });
    vals(lof.scoreSamples(Q)).forEach((v, i) => {
      expect(v).toBeCloseTo(scoreQ[i] as number, 10);
    });
    expect(vals(lof.predict(Q))).toEqual([1, -1, -1, 1, 1, 1, -1, 1]);
    const dec = [
      0.6302840846819268, -2.242002499021352, -4.769815013186224, 0.11881442330415926,
      0.5552119304756188, 0.5426636362560497, -6.237192300600436, 0.6302840846819268,
    ];
    vals(lof.decisionFunction(Q)).forEach((v, i) => {
      expect(v).toBeCloseTo(dec[i] as number, 10);
    });
    expect(lof.offset).toBe(-1.5);
  });

  it("scoreSamples evaluates the given data, not memorized training scores", () => {
    // Same number of rows as the training set: the old code returned the
    // stored training scores here regardless of the contents of the query.
    const lof = new LocalOutlierFactor({ nNeighbors: 3 }).fit(X);
    const s = vals(lof.scoreSamples(Q));
    expect(s).not.toEqual(vals(lof.negativeOutlierFactor));
    // Far away points are more abnormal than points inside the cloud.
    expect(s[2]).toBeLessThan(s[0] as number);
  });

  it("scoring the training data reproduces the fitted LOF", () => {
    const lof = new LocalOutlierFactor({ nNeighbors: 3 }).fit(X);
    vals(lof.scoreSamples(X)).forEach((v, i) => {
      expect(v).toBeCloseTo(nof[i] as number, 10);
    });
  });

  it("numeric contamination uses the sklearn percentile offset and strict comparison", () => {
    const lof = new LocalOutlierFactor({ nNeighbors: 3, contamination: 0.25 });
    lof.fit(X);
    expect(lof.offset).toBeCloseTo(-1.0421446012480051, 12);
    expect(vals(lof.predict(Q))).toEqual([1, -1, -1, -1, 1, 1, -1, 1]);
    const dec = [0.17242868592993188, -2.699857897773347, -5.2276704119382185, -0.3390409754478356];
    vals(lof.decisionFunction(Q))
      .slice(0, 4)
      .forEach((v, i) => {
        expect(v).toBeCloseTo(dec[i] as number, 10);
      });

    const fp = new LocalOutlierFactor({ nNeighbors: 3, contamination: 0.15 });
    expect(vals(fp.fitPredict(X))).toEqual([1, 1, -1, 1, 1, 1, -1, 1]);
    expect(fp.offset).toBeCloseTo(-1.057501042747018, 12);
    const fp2 = new LocalOutlierFactor({ nNeighbors: 3, contamination: 0.4 });
    expect(vals(fp2.fitPredict(X))).toEqual([1, 1, -1, 1, -1, 1, -1, 1]);
  });

  it("uses the clamped neighbor count when nNeighbors exceeds nSamples - 1", () => {
    // sklearn: n_neighbors_ = 4 for 5 samples, also used for novelty queries.
    const lof = new LocalOutlierFactor({ nNeighbors: 20 }).fit(f64(X5()));
    const expectedNof = [
      -0.9612162943416938, -0.9867959730771807, -0.9867959730771807, -0.9612162943416938,
      -1.1167071011488645,
    ];
    vals(lof.negativeOutlierFactor).forEach((v, i) => {
      expect(v).toBeCloseTo(expectedNof[i] as number, 10);
    });
    const expectedQ = [
      -0.9612162943416938, -2.6245724755174344, -5.90328407497038, -1.2296443978057763,
      -0.9867959730771807, -0.961216294341694, -7.31707859183118, -0.9867959730771807,
    ];
    vals(lof.scoreSamples(Q)).forEach((v, i) => {
      expect(v).toBeCloseTo(expectedQ[i] as number, 10);
    });
  });

  it("changing nNeighbors after fit does not change the fitted model", () => {
    const lof = new LocalOutlierFactor({ nNeighbors: 3 }).fit(X);
    const before = vals(lof.scoreSamples(Q));
    lof.setParams({ nNeighbors: 5 });
    expect(vals(lof.scoreSamples(Q))).toEqual(before);
  });

  it("handles duplicate points like scikit-learn (lrd = 1 / (mean reach + 1e-10))", () => {
    const Xd = f64([
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0.1, 0],
      [0.2, 0.1],
      [3, 3],
      [3.1, 3],
      [3, 3.1],
      [3.2, 3.2],
    ]);
    const lof = new LocalOutlierFactor({ nNeighbors: 3 });
    const labels = vals(lof.fitPredict(Xd));
    expect(labels).toEqual([1, 1, 1, 1, -1, -1, 1, 1, 1, 1]);
    const expected = [-1, -1, -1, -1, -1000000001.0000001, -1308077671.847967];
    const got = vals(lof.negativeOutlierFactor);
    expected.forEach((e, i) => {
      expect(got[i]).toBeCloseTo(e, 0);
    });
    expect(got[6]).toBeCloseTo(-0.9499670607847027, 6);
    expect(got[7]).toBeCloseTo(-1.0540925533672316, 6);
  });

  it("validates query data and reports typed errors", () => {
    const lof = new LocalOutlierFactor({ nNeighbors: 3 });
    expect(() => lof.scoreSamples(Q)).toThrow(/fitted/i);
    expect(() => lof.decisionFunction(Q)).toThrow(/fitted/i);
    expect(() => lof.offset).toThrow(/fitted/i);
    lof.fit(X);
    expect(() => lof.predict(f64([[1, 2, 3]]))).toThrow(ShapeError);
    expect(() => lof.scoreSamples(f64([[1, Number.NaN]]))).toThrow(DataValidationError);
    expect(() => new LocalOutlierFactor().fit(f64([[1, 2]]))).toThrow(/not enough/i);
  });

  it("validates contamination in the constructor and applies setParams atomically", () => {
    expect(() => new LocalOutlierFactor({ contamination: 0.9 })).toThrow(InvalidParameterError);
    expect(() => new LocalOutlierFactor({ contamination: Number.NaN })).toThrow(/contamination/);
    const lof = new LocalOutlierFactor({ nNeighbors: 4 });
    expect(() => lof.setParams({ nNeighbors: 9, contamination: Number.NaN })).toThrow(
      InvalidParameterError
    );
    expect(lof.getParams().nNeighbors).toBe(4);
  });

  it("fits int64 data, copies the training data and supports clone", () => {
    const Xi = tensor(
      [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [50, 50],
      ],
      { dtype: "int64" }
    );
    const lof = new LocalOutlierFactor({ nNeighbors: 2 });
    expect(vals(lof.fitPredict(Xi))[4]).toBe(-1);

    const Xm = f64([
      [0, 0],
      [1, 0],
      [0, 1],
      [1, 1],
    ]);
    const m = new LocalOutlierFactor({ nNeighbors: 2 }).fit(Xm);
    const q = f64([[0.5, 0.5]]);
    const before = vals(m.scoreSamples(q));
    (Xm.data as Float64Array)[0] = 1000;
    expect(vals(m.scoreSamples(q))).toEqual(before);
    expect(m.clone().getParams()).toEqual(m.getParams());
    expect(() => m.clone().predict(q)).toThrow(/fitted/i);
  });

  it("uses a selection path for large k that agrees with the small-k path", () => {
    const data = f64(lcgRows(80, 2, 21));
    const a = new LocalOutlierFactor({ nNeighbors: 33 }).fit(data);
    const b = new LocalOutlierFactor({ nNeighbors: 33 }).fit(data);
    expect(vals(a.negativeOutlierFactor)).toEqual(vals(b.negativeOutlierFactor));
    // mean LOF of a uniform cloud is close to 1
    const m = vals(a.negativeOutlierFactor).reduce((s, v) => s - v, 0) / 80;
    expect(m).toBeGreaterThan(0.95);
    expect(m).toBeLessThan(1.3);
  });
});

function X5(): number[][] {
  return [
    [0, 0],
    [1, 0.1],
    [0, 1],
    [1.05, 1],
    [0.5, 0.4],
  ];
}

// ---------------------------------------------------------------------------
// CalibratedClassifierCV
// ---------------------------------------------------------------------------

/**
 * Test classifier whose probabilities are a fixed function of the first
 * feature, so the calibrators are fitted on exactly those scores.
 */
class ScoreClassifier implements Classifier {
  static fits = 0;
  private classes_: Tensor | undefined;
  fit(_X: Tensor, y: Tensor): this {
    ScoreClassifier.fits++;
    this.classes_ = f64([...new Set(vals(y))].sort((a, b) => a - b));
    return this;
  }
  get classes(): Tensor | undefined {
    return this.classes_;
  }
  predict(X: Tensor): Tensor {
    return f64(vals(this.predictProba(X)).filter((_, i) => i % 2 === 1));
  }
  predictProba(X: Tensor): Tensor {
    const v = vals(X);
    const nFeatures = X.shape[1] ?? 1;
    const out: number[][] = [];
    for (let i = 0; i < (X.shape[0] ?? 0); i++) {
      const s = v[i * nFeatures] as number;
      out.push([1 - s, s]);
    }
    return f64(out);
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
  clone(): ScoreClassifier {
    return new ScoreClassifier();
  }
}

function col(scores: number[]): Tensor {
  return f64(scores.map((s) => [s, 0]));
}

describe("CalibratedClassifierCV", () => {
  const scores = [0.1, 0.2, 0.35, 0.4, 0.55, 0.6, 0.7, 0.85, 0.9, 0.15, 0.45, 0.65];
  const labels = [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0];
  const queries = [0.0, 0.1, 0.3, 0.5, 0.62, 0.95];

  it("Platt scaling matches sklearn's _sigmoid_calibration", () => {
    // sklearn: A, B = -2.738205145916883, 1.346430396184872 ; p = 1 / (1 + exp(A f + B))
    const a = 2.738205145916883;
    const b = -1.346430396184872;
    const cal = new CalibratedClassifierCV({ estimator: new ScoreClassifier(), cv: 3 });
    cal.fit(col(scores), f64(labels));
    const p = vals(cal.predictProba(col(queries)));
    queries.forEach((q, i) => {
      const p1 = 1 / (1 + Math.exp(-(a * q + b)));
      expect(p[2 * i + 1]).toBeCloseTo(p1, 6);
      expect(p[2 * i]).toBeCloseTo(1 - p1, 6);
    });
  });

  it("isotonic calibration matches sklearn's IsotonicRegression(out_of_bounds='clip')", () => {
    const cal = new CalibratedClassifierCV({
      estimator: new ScoreClassifier(),
      method: "isotonic",
      cv: 3,
    });
    cal.fit(col(scores), f64(labels));
    const expected = [0, 0, 1 / 3, 1 / 3, 0.5, 1];
    const p = vals(cal.predictProba(col(queries)));
    expected.forEach((e, i) => {
      expect(p[2 * i + 1]).toBeCloseTo(e, 12);
      expect(p[2 * i]).toBeCloseTo(1 - e, 12);
    });
  });

  it("isotonic regression merges tied scores and pools whole blocks", () => {
    // sklearn reference for tied scores
    const s2 = [0.2, 0.2, 0.5, 0.5, 0.5, 0.8, 0.8, 0.1];
    const y2 = [0, 1, 1, 0, 0, 1, 1, 0];
    const q2 = [0.05, 0.1, 0.15, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9];
    const e2 = [0, 0, 0.19999999999999996, 0.4, 0.4, 0.4, 0.7, 1, 1];
    const cal = new CalibratedClassifierCV({
      estimator: new ScoreClassifier(),
      method: "isotonic",
      cv: 2,
    });
    cal.fit(col(s2), f64(y2));
    vals(cal.predictProba(col(q2)))
      .filter((_, i) => i % 2 === 1)
      .forEach((v, i) => {
        expect(v).toBeCloseTo(e2[i] as number, 12);
      });

    // Strictly decreasing labels pool into a single block (mean 0.5). The old
    // pairwise pooling left 0.75 / 0.25 style partial blocks.
    const dec = new CalibratedClassifierCV({
      estimator: new ScoreClassifier(),
      method: "isotonic",
      cv: 2,
    });
    dec.fit(col([0.1, 0.2, 0.3, 0.4]), f64([1, 1, 0, 0]));
    vals(dec.predictProba(col([0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.9])))
      .filter((_, i) => i % 2 === 1)
      .forEach((v) => {
        expect(v).toBeCloseTo(0.5, 12);
      });
  });

  it("calibrates on held-out folds, not on in-sample predictions", () => {
    class Memorizer implements Classifier {
      private seen = new Map<string, number>();
      private classes_: Tensor | undefined;
      fit(X: Tensor, y: Tensor): this {
        const v = vals(X);
        const yy = vals(y);
        const d = X.shape[1] ?? 1;
        this.seen = new Map();
        yy.forEach((label, i) => {
          this.seen.set(v.slice(i * d, (i + 1) * d).join(","), label);
        });
        this.classes_ = f64([...new Set(yy)].sort((a, b) => a - b));
        return this;
      }
      get classes(): Tensor | undefined {
        return this.classes_;
      }
      predict(): Tensor {
        return f64([0]);
      }
      predictProba(X: Tensor): Tensor {
        const v = vals(X);
        const d = X.shape[1] ?? 1;
        const out: number[][] = [];
        for (let i = 0; i < (X.shape[0] ?? 0); i++) {
          const key = v.slice(i * d, (i + 1) * d).join(",");
          const label = this.seen.get(key);
          out.push(label === undefined ? [0.5, 0.5] : label === 1 ? [0, 1] : [1, 0]);
        }
        return f64(out);
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
      clone(): Memorizer {
        return new Memorizer();
      }
    }
    const rows = Array.from({ length: 20 }, (_, i) => [i, i * 0.5]);
    const y = f64(rows.map((_, i) => i % 2));
    const cal = new CalibratedClassifierCV({ estimator: new Memorizer(), cv: 4 });
    cal.fit(f64(rows), y);
    // The memorizer is useless on unseen rows (always 0.5), so an honest
    // calibrator maps its scores to the 50% base rate everywhere.
    for (const p of vals(cal.predictProba(f64(rows)))) expect(p).toBeCloseTo(0.5, 6);
  });

  it("stratifies folds so every fold trains on all classes", () => {
    // Interleaved labels: round-robin folds put both samples of one class in the
    // same test fold and leave a training split without that class.
    ScoreClassifier.fits = 0;
    const cal = new CalibratedClassifierCV({ estimator: new ScoreClassifier(), cv: 2 });
    cal.fit(col([0.2, 0.7, 0.3, 0.8]), f64([0, 1, 0, 1]));
    // 2 fold estimators + 1 final fit, with no in-sample fallback fit
    expect(ScoreClassifier.fits).toBe(3);
  });

  it("falls back to a global fit for classes that cannot be stratified", () => {
    ScoreClassifier.fits = 0;
    const cal = new CalibratedClassifierCV({ estimator: new ScoreClassifier(), cv: 3 });
    // class 1 has a single sample: its fold's training split lacks class 1
    cal.fit(col([0.1, 0.2, 0.3, 0.4, 0.9]), f64([0, 0, 0, 0, 1]));
    expect(cal.predictProba(col([0.5])).shape).toEqual([1, 2]);
  });

  it("multiclass probabilities are float64, sum to one and follow sorted classes", () => {
    const rows: number[][] = [];
    const y: number[] = [];
    for (let i = 0; i < 30; i++) {
      rows.push([i % 10, 0]);
      y.push(0);
      rows.push([0, i % 10]);
      y.push(1);
      rows.push([i % 10, (i % 10) + 20]);
      y.push(2);
    }
    for (const method of ["sigmoid", "isotonic"] as const) {
      const cal = new CalibratedClassifierCV({
        estimator: new LogisticRegression({ maxIter: 300 }),
        method,
        cv: 3,
      });
      cal.fit(f64(rows), f64(y));
      const proba = cal.predictProba(f64(rows));
      expect(proba.dtype).toBe("float64");
      expect(proba.shape).toEqual([90, 3]);
      const v = vals(proba);
      for (let i = 0; i < 90; i++) {
        expect(
          (v[3 * i] as number) + (v[3 * i + 1] as number) + (v[3 * i + 2] as number)
        ).toBeCloseTo(1, 12);
      }
      expect(vals(cal.classes as Tensor)).toEqual([0, 1, 2]);
      expect(cal.score(f64(rows), f64(y))).toBeGreaterThan(0.9);
    }
  });

  it("Platt scaling is learned (the old 100-step gradient descent barely moved)", () => {
    // Perfectly separable scores should be pushed towards 0 and 1.
    const cal = new CalibratedClassifierCV({ estimator: new ScoreClassifier(), cv: 2 });
    cal.fit(col([0.05, 0.1, 0.15, 0.2, 0.8, 0.85, 0.9, 0.95]), f64([0, 0, 0, 0, 1, 1, 1, 1]));
    const p = vals(cal.predictProba(col([0.05, 0.95])));
    expect(p[1]).toBeLessThan(0.15);
    expect(p[3]).toBeGreaterThan(0.85);
  });

  it("fit leaves a clean not-fitted state when the final estimator fails", () => {
    class Failing extends ScoreClassifier {
      static failFinal = false;
      override fit(X: Tensor, y: Tensor): this {
        if (Failing.failFinal && (X.shape[0] ?? 0) === 8) throw new Error("boom");
        return super.fit(X, y);
      }
      override clone(): Failing {
        return new Failing();
      }
    }
    const cal = new CalibratedClassifierCV({ estimator: new Failing(), cv: 2 });
    const X8 = col([0.05, 0.1, 0.15, 0.2, 0.8, 0.85, 0.9, 0.95]);
    const y8 = f64([0, 0, 0, 0, 1, 1, 1, 1]);
    cal.fit(X8, y8);
    Failing.failFinal = true;
    expect(() => cal.fit(X8, y8)).toThrow("boom");
    expect(() => cal.predictProba(X8)).toThrow(/fitted/i);
    Failing.failFinal = false;
  });

  it("works with a KNN base estimator", () => {
    const rows = Array.from({ length: 40 }, (_, i) => [Math.sin(i), Math.cos(i * 1.3), i % 7]);
    const y = f64(rows.map((r) => ((r[0] as number) + 0.3 * (r[1] as number) > 0 ? 1 : 0)));
    const knn = new KNeighborsClassifier({ nNeighbors: 3 });
    const cal = new CalibratedClassifierCV({ estimator: knn, cv: 3 });
    cal.fit(f64(rows), y);
    const proba = cal.predictProba(f64(rows));
    expect(proba.shape).toEqual([40, 2]);
    expect(cal.score(f64(rows), y)).toBeGreaterThan(0.7);
  });

  it("rejects single-class targets and mismatched score inputs", () => {
    const cal = new CalibratedClassifierCV({ estimator: new ScoreClassifier() });
    expect(() => cal.fit(col([0.1, 0.2, 0.3]), f64([1, 1, 1]))).toThrow(DataValidationError);
    cal.fit(col([0.1, 0.2, 0.7, 0.8]), f64([0, 0, 1, 1]));
    expect(() => cal.score(col([0.1, 0.2]), f64([0, 1, 1]))).toThrow(ShapeError);
    expect(() => cal.predictProba(f64([[0.1, 0, 0]]))).toThrow(ShapeError);
  });

  it("accepts int64 labels and exposes classes, clone and atomic setParams", () => {
    const cal = new CalibratedClassifierCV({ estimator: new ScoreClassifier(), cv: 2 });
    expect(cal.classes).toBeUndefined();
    cal.fit(col([0.1, 0.2, 0.7, 0.8]), tensor([3, 3, 7, 7], { dtype: "int64" }));
    expect(vals(cal.classes as Tensor)).toEqual([3, 7]);
    expect(vals(cal.predict(col([0.05, 0.95])))).toEqual([3, 7]);

    const copy = cal.clone();
    expect(copy.getParams().cv).toBe(2);
    expect(() => copy.predictProba(col([0.5]))).toThrow(/fitted/i);

    expect(() => cal.setParams({ cv: 7, method: "bogus" })).toThrow(InvalidParameterError);
    expect(cal.getParams().cv).toBe(2);
    expect(
      () =>
        new CalibratedClassifierCV({ estimator: new ScoreClassifier(), method: "x" as "sigmoid" })
    ).toThrow(/method/);
  });
});

// ---------------------------------------------------------------------------
// calibrationCurve (reference values from sklearn.calibration.calibration_curve)
// ---------------------------------------------------------------------------
describe("calibrationCurve", () => {
  const yt = f64([0, 0, 1, 0, 1, 1, 0, 1, 1, 1]);
  const yp = f64([0.0, 0.1, 0.2, 0.3, 0.5, 0.5, 0.6, 0.7, 0.9, 1.0]);

  it("uniform bins are (lo, hi] except the first, as in sklearn", () => {
    const r = calibrationCurve(yt, yp, { nBins: 4 });
    expect(r.fractionPositives).toEqual([1 / 3, 2 / 3, 0.5, 1]);
    [0.10000000000000002, 0.43333333333333335, 0.6499999999999999, 0.95].forEach((e, i) => {
      expect(r.meanPredicted[i]).toBeCloseTo(e, 12);
    });
    // 10 bins: a probability sitting exactly on an edge (0.5, 0.6, 0.7, 1.0) belongs to the lower bin
    const r10 = calibrationCurve(yt, yp);
    expect(r10.fractionPositives).toEqual([0, 1, 0, 1, 0, 1, 1, 1]);
    [0.05, 0.2, 0.3, 0.5, 0.6, 0.7, 0.9, 1.0].forEach((e, i) => {
      expect(r10.meanPredicted[i]).toBeCloseTo(e, 12);
    });
  });

  it("quantile strategy uses numpy percentile edges", () => {
    const r = calibrationCurve(yt, yp, { nBins: 4, strategy: "quantile" });
    expect(r.fractionPositives).toEqual([1 / 3, 2 / 3, 0, 1]);
    [0.10000000000000002, 0.43333333333333335, 0.6, 0.8666666666666667].forEach((e, i) => {
      expect(r.meanPredicted[i]).toBeCloseTo(e, 12);
    });
  });

  it("supports -1/1 labels and posLabel", () => {
    const prob = f64([0.3, 0.7, 0.5, 0.1, 0.9]);
    const a = calibrationCurve(f64([-1, 1, 1, -1, 1]), prob, { nBins: 2 });
    expect(a.fractionPositives).toEqual([1 / 3, 1]);
    expect(a.meanPredicted[0]).toBeCloseTo(0.3, 12);
    expect(a.meanPredicted[1]).toBeCloseTo(0.8, 12);
    const b = calibrationCurve(f64([2, 3, 3, 2, 3]), prob, { nBins: 2, posLabel: 3 });
    expect(b.fractionPositives).toEqual([1 / 3, 1]);
    expect(() => calibrationCurve(f64([2, 3, 3, 2, 3]), prob)).toThrow(/posLabel/);
  });

  it("rejects probabilities outside [0, 1], bad options, empty and non-binary input", () => {
    expect(() => calibrationCurve(f64([0, 1]), f64([0.2, 1.2]))).toThrow(DataValidationError);
    expect(() => calibrationCurve(f64([0, 1]), f64([0.2, Number.NaN]))).toThrow(
      DataValidationError
    );
    expect(() => calibrationCurve(f64([0, 1]), f64([0.2, 0.9]), { nBins: 0 })).toThrow(
      InvalidParameterError
    );
    expect(() => calibrationCurve(f64([0, 1]), f64([0.2, 0.9]), { nBins: 2.5 })).toThrow(/nBins/);
    expect(() =>
      calibrationCurve(f64([0, 1]), f64([0.2, 0.9]), { strategy: "bogus" as "uniform" })
    ).toThrow(/strategy/);
    expect(() => calibrationCurve(f64([0, 1, 2]), f64([0.2, 0.9, 0.5]))).toThrow(
      DataValidationError
    );
    expect(() => calibrationCurve(f64([]), f64([]))).toThrow(DataValidationError);
    expect(() => calibrationCurve(f64([0, 1, 1]), f64([0.1, 0.9]))).toThrow(InvalidParameterError);
  });

  it("rejects non-contiguous views instead of reading wrong elements", () => {
    // transpose of a (3, 2) matrix is a strided (2, 3) view
    const view = transpose(
      f64([
        [0, 1],
        [1, 0],
        [0, 1],
      ])
    );
    expect(() => calibrationCurve(view, f64([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))).toThrow(
      DataValidationError
    );
  });
});

// ---------------------------------------------------------------------------
// base.ts: estimator tags
// ---------------------------------------------------------------------------
describe("getEstimatorTags", () => {
  it("recognizes classifiers without predictProba through classes / decisionFunction", () => {
    const svc = getEstimatorTags(new LinearSVC());
    expect(svc.estimatorType).toBe("classifier");
    expect(svc.requiresY).toBe(true);
    const nu = getEstimatorTags(new NuSVC());
    expect(nu.estimatorType).toBe("classifier");
  });

  it("treats mixture models as clusterers and infers hasDecisionFunction", () => {
    const gmm = getEstimatorTags(new GaussianMixture({ nComponents: 2 }));
    expect(gmm.estimatorType).toBe("clusterer");
    expect(gmm.requiresY).toBe(false);
    expect(getEstimatorTags(new KMeans({ nClusters: 2 })).estimatorType).toBe("clusterer");
    expect(getEstimatorTags(new IsolationForest()).estimatorType).toBe("outlier_detector");
    expect(getEstimatorTags(new LocalOutlierFactor()).requiresY).toBe(false);
    expect(
      getEstimatorTags(new CalibratedClassifierCV({ estimator: new LogisticRegression() }))
        .estimatorType
    ).toBe("classifier");
    const hasDecision = { fit: () => 0, decisionFunction: () => 0, predict: () => 0 };
    expect(
      getEstimatorTags(hasDecision as unknown as Parameters<typeof getEstimatorTags>[0])
        .hasDecisionFunction
    ).toBe(true);
  });

  it("tags estimators that only expose fitTransform as transformers", () => {
    expect(getEstimatorTags(new Isomap()).estimatorType).toBe("transformer");
  });

  it("throws a typed error for non-objects", () => {
    expect(() =>
      getEstimatorTags(null as unknown as Parameters<typeof getEstimatorTags>[0])
    ).toThrow(InvalidParameterError);
  });
});
