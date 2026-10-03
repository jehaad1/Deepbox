/**
 * Regression tests for RandomForestClassifier / RandomForestRegressor (v1.5.0 review).
 *
 * Reference values come from scikit-learn 1.8 (RandomForest with bootstrap=False and all
 * features per split, which makes the forest deterministic apart from tie-breaking).
 */
import { describe, expect, it } from "vitest";
import {
  DataValidationError,
  InvalidParameterError,
  NotFittedError,
  ShapeError,
} from "../../src/core";
import { RandomForestClassifier, RandomForestRegressor } from "../../src/ml/tree";
import type { Tensor } from "../../src/ndarray";
import { tensor } from "../../src/ndarray";

// Three clusters with two flipped labels. Generated with
// rng = np.random.RandomState(3); centers [[0,0],[5,5],[10,0]] + round(randn(6,2), 2).
const X = tensor([
  [1.79, 0.44],
  [0.1, -1.86],
  [-0.28, -0.35],
  [-0.08, -0.63],
  [-0.04, -0.48],
  [-1.31, 0.88],
  [5.88, 6.71],
  [5.05, 4.6],
  [4.45, 3.45],
  [5.98, 3.9],
  [3.81, 4.79],
  [6.49, 5.24],
  [8.98, -0.71],
  [10.63, -0.16],
  [9.23, -0.23],
  [10.75, 1.98],
  [8.76, -0.63],
  [9.2, -2.42],
]);
const Y_LABELS = [0, 0, 1, 0, 0, 0, 1, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2];
const Y = tensor(Y_LABELS, { dtype: "int32" });
const X_TEST = tensor([
  [0.2, 0.1],
  [5.1, 4.8],
  [9.8, 0.3],
  [2.6, 2.6],
  [7.5, 2.5],
]);
const Y_REG = tensor([
  2.628, -1.087, -0.258, -0.475, -0.787, -1.331, 12.068, 9.352, 8.221, 10.743, 7.848, 12.364,
  12.441, 15.785, 14.034, 17.371, 13.157, 12.926,
]);

function toArray(t: Tensor): number[] {
  return Array.from(t.data as ArrayLike<number | bigint>, Number);
}

/** Noisy two-class data: the label depends on the first feature plus noise. */
function noisyData(n: number): { X: Tensor; y: Tensor } {
  let s = 12345;
  const rnd = (): number => {
    s = (s * 1103515245 + 12345) % 2147483648;
    return s / 2147483648;
  };
  const rows: number[][] = [];
  const labels: number[] = [];
  for (let i = 0; i < n; i++) {
    const a = rnd();
    const b = rnd();
    rows.push([a, b]);
    labels.push(a + (rnd() - 0.5) * 0.8 > 0.5 ? 1 : 0);
  }
  return { X: tensor(rows), y: tensor(labels, { dtype: "int32" }) };
}

describe("RandomForestClassifier matches scikit-learn on a deterministic forest", () => {
  // sklearn: RandomForestClassifier(n_estimators=3, bootstrap=False, max_features=None,
  //   max_depth=d, random_state=0).fit(X, y); predict_proba(X_TEST)
  const cases: Array<{ depth: number; proba: number[][]; labels: number[] }> = [
    {
      depth: 1,
      proba: [
        [0.5, 0.5, 0],
        [0.5, 0.5, 0],
        [0, 0.125, 0.875],
        [0.5, 0.5, 0],
        [0, 0.125, 0.875],
      ],
      labels: [0, 0, 2, 0, 2],
    },
  ];
  for (const c of cases) {
    it(`maxDepth=${c.depth}: probabilities and labels`, () => {
      const rf = new RandomForestClassifier({
        nEstimators: 3,
        bootstrap: false,
        maxFeatures: 2,
        maxDepth: c.depth,
        randomState: 0,
      }).fit(X, Y);
      const proba = toArray(rf.predictProba(X_TEST));
      c.proba.flat().forEach((v, i) => {
        expect(proba[i]).toBeCloseTo(v, 5);
      });
      expect(toArray(rf.predict(X_TEST))).toEqual(c.labels);
    });
  }
});

describe("fully grown forest matches scikit-learn where splits are unambiguous", () => {
  it("memorizes the training set and agrees on the well-separated test points", () => {
    // sklearn (4 seeds): predict_proba(X_TEST) rows 0, 1, 2 and 4 are [1,0,0], [0,1,0],
    // [0,0,1] and [0,0,1] for every seed; row 3 depends on tie-breaking between equal splits.
    const rf = new RandomForestClassifier({
      nEstimators: 3,
      bootstrap: false,
      maxFeatures: 2,
      maxDepth: Number.POSITIVE_INFINITY,
      randomState: 0,
    }).fit(X, Y);
    expect(rf.score(X, Y)).toBe(1);
    const pred = toArray(rf.predict(X_TEST));
    expect([pred[0], pred[1], pred[2], pred[4]]).toEqual([0, 1, 2, 2]);
  });
});

describe("RandomForestRegressor matches scikit-learn on a deterministic forest", () => {
  it("maxDepth=3 predictions and importances", () => {
    // sklearn: RandomForestRegressor(n_estimators=3, bootstrap=False, max_features=1.0,
    //   max_depth=3, random_state=0).fit(X, y_reg).predict(X_TEST)
    const rf = new RandomForestRegressor({
      nEstimators: 3,
      bootstrap: false,
      maxDepth: 3,
      randomState: 0,
    }).fit(X, Y_REG);
    const expected = [-0.65175, 8.473667, 12.9844, 2.628, 12.9844];
    toArray(rf.predict(X_TEST)).forEach((v, i) => {
      expect(v).toBeCloseTo(expected[i] ?? 0, 5);
    });
    // sklearn: feature_importances_ = [0.999827, 0.000173]
    const imp = toArray(rf.featureImportances);
    expect(imp[0]).toBeGreaterThan(0.999);
    expect(imp[0] + (imp[1] ?? 0)).toBeCloseTo(1, 12);
  });
});

describe("targets keep their precision", () => {
  it("regressor memorizes float64 targets exactly (they were rounded to float32 before)", () => {
    const y = tensor([1e9 + 0.25, 1e9 + 0.5, 1e9 + 0.75, 1e9 + 1], { dtype: "float64" });
    const rf = new RandomForestRegressor({
      nEstimators: 3,
      bootstrap: false,
      maxDepth: Number.POSITIVE_INFINITY,
      randomState: 1,
    }).fit(tensor([[0], [1], [2], [3]]), y);
    const pred = rf.predict(tensor([[0], [1], [2], [3]]));
    expect(pred.dtype).toBe("float64");
    expect(toArray(pred)).toEqual([1e9 + 0.25, 1e9 + 0.5, 1e9 + 0.75, 1e9 + 1]);
  });

  it("classifier keeps fractional labels (they were truncated to integers before)", () => {
    const y = tensor([0.5, 0.5, 1.5, 1.5], { dtype: "float64" });
    const rf = new RandomForestClassifier({ nEstimators: 5, randomState: 1 }).fit(
      tensor([[0], [1], [2], [3]]),
      y
    );
    expect(toArray(rf.classes as Tensor)).toEqual([0.5, 1.5]);
    const pred = rf.predict(tensor([[0], [3]]));
    expect(pred.dtype).toBe("float64");
    expect(toArray(pred)).toEqual([0.5, 1.5]);
    expect(rf.score(tensor([[0], [3]]), tensor([0.5, 1.5], { dtype: "float64" }))).toBe(1);
  });

  it("integer labels stay int32, negative labels work", () => {
    const rf = new RandomForestClassifier({ nEstimators: 5, randomState: 1 }).fit(
      tensor([[0], [1], [2], [3]]),
      tensor([-1, -1, 4, 4], { dtype: "int32" })
    );
    expect((rf.classes as Tensor).dtype).toBe("int32");
    const pred = rf.predict(tensor([[0], [3]]));
    expect(pred.dtype).toBe("int32");
    expect(toArray(pred)).toEqual([-1, 4]);
  });

  it("accepts int64 (BigInt) labels", () => {
    const rf = new RandomForestClassifier({ nEstimators: 5, randomState: 1 }).fit(
      tensor([[0], [1], [2], [3]]),
      tensor(BigInt64Array.from([0n, 0n, 1n, 1n]))
    );
    expect(rf.score(tensor([[0], [3]]), tensor(BigInt64Array.from([0n, 1n])))).toBe(1);
  });

  it("outputs are float64", () => {
    const rf = new RandomForestClassifier({ nEstimators: 5, randomState: 1 }).fit(X, Y);
    expect(rf.predictProba(X_TEST).dtype).toBe("float64");
    expect(rf.featureImportances.dtype).toBe("float64");
    const reg = new RandomForestRegressor({ nEstimators: 5, randomState: 1 }).fit(X, Y_REG);
    expect(reg.featureImportances.dtype).toBe("float64");
  });
});

describe("seeding", () => {
  it("negative and fractional seeds give valid, reproducible forests (negative seeds broke the LCG)", () => {
    const d = noisyData(80);
    for (const seed of [-3, -1000, 0.5, 2 ** 40]) {
      const a = new RandomForestClassifier({ nEstimators: 20, randomState: seed }).fit(d.X, d.y);
      const b = new RandomForestClassifier({ nEstimators: 20, randomState: seed }).fit(d.X, d.y);
      expect(toArray(a.predictProba(d.X))).toEqual(toArray(b.predictProba(d.X)));
      expect(a.score(d.X, d.y)).toBeGreaterThan(0.85);
    }
  });

  it("different seeds give different forests", () => {
    const d = noisyData(80);
    const a = new RandomForestClassifier({ nEstimators: 20, randomState: 1 }).fit(d.X, d.y);
    const b = new RandomForestClassifier({ nEstimators: 20, randomState: 2 }).fit(d.X, d.y);
    expect(toArray(a.predictProba(d.X))).not.toEqual(toArray(b.predictProba(d.X)));
  });

  it("regressor is reproducible with a seed", () => {
    const a = new RandomForestRegressor({ nEstimators: 10, randomState: 7 }).fit(X, Y_REG);
    const b = new RandomForestRegressor({ nEstimators: 10, randomState: 7 }).fit(X, Y_REG);
    expect(toArray(a.predict(X_TEST))).toEqual(toArray(b.predict(X_TEST)));
  });
});

describe("prediction rule", () => {
  it("predict is the argmax of the averaged probabilities (smallest label on ties)", () => {
    const d = noisyData(120);
    const rf = new RandomForestClassifier({ nEstimators: 15, maxDepth: 1, randomState: 3 }).fit(
      d.X,
      d.y
    );
    const proba = toArray(rf.predictProba(d.X));
    const pred = toArray(rf.predict(d.X));
    for (let i = 0; i < pred.length; i++) {
      const p0 = proba[2 * i] ?? 0;
      const p1 = proba[2 * i + 1] ?? 0;
      expect(pred[i]).toBe(p1 > p0 ? 1 : 0);
    }
  });

  it("predictProba rows sum to 1 and predictLogProba is their logarithm", () => {
    const rf = new RandomForestClassifier({ nEstimators: 10, randomState: 4 }).fit(X, Y);
    const proba = toArray(rf.predictProba(X_TEST));
    const logp = toArray(rf.predictLogProba(X_TEST));
    for (let i = 0; i < 5; i++) {
      expect((proba[3 * i] ?? 0) + (proba[3 * i + 1] ?? 0) + (proba[3 * i + 2] ?? 0)).toBeCloseTo(
        1,
        12
      );
    }
    proba.forEach((v, i) => {
      expect(logp[i]).toBe(Math.log(v));
    });
  });

  it("handles zero-row input", () => {
    const rf = new RandomForestClassifier({ nEstimators: 3, randomState: 4 }).fit(X, Y);
    const empty = tensor([] as number[][], { dtype: "float64" }).reshape([0, 2]);
    expect(rf.predict(empty).shape).toEqual([0]);
    expect(rf.predictProba(empty).shape).toEqual([0, 3]);
    const reg = new RandomForestRegressor({ nEstimators: 3, randomState: 4 }).fit(X, Y_REG);
    expect(reg.predict(empty).shape).toEqual([0]);
  });
});

describe("refit and warm start", () => {
  it("refitting reproduces the out-of-bag score (bags of the old fit leaked into the new one)", () => {
    const d = noisyData(100);
    const rf = new RandomForestClassifier({ nEstimators: 25, oobScore: true, randomState: 5 });
    rf.fit(d.X, d.y);
    const first = rf.oobScore;
    rf.fit(d.X, d.y);
    expect(rf.oobScore).toBe(first);
    const reg = new RandomForestRegressor({ nEstimators: 25, oobScore: true, randomState: 5 });
    reg.fit(d.X, d.y);
    const firstReg = reg.oobScore;
    reg.fit(d.X, d.y);
    expect(reg.oobScore).toBe(firstReg);
  });

  it("disabling oobScore before a refit removes the stale value", () => {
    const d = noisyData(60);
    const rf = new RandomForestClassifier({ nEstimators: 10, oobScore: true, randomState: 5 });
    rf.fit(d.X, d.y);
    expect(Number.isFinite(rf.oobScore)).toBe(true);
    rf.setParams({ oobScore: false }).fit(d.X, d.y);
    expect(() => rf.oobScore).toThrow(NotFittedError);
    expect(() => rf.oobDecisionFunction).toThrow(NotFittedError);
  });

  it("a warm-started forest equals the same forest grown in one go", () => {
    const d = noisyData(60);
    const direct = new RandomForestClassifier({ nEstimators: 12, randomState: 9 }).fit(d.X, d.y);
    const warm = new RandomForestClassifier({ nEstimators: 5, randomState: 9, warmStart: true });
    warm.fit(d.X, d.y);
    warm.setParams({ nEstimators: 12 }).fit(d.X, d.y);
    expect(toArray(warm.predictProba(d.X))).toEqual(toArray(direct.predictProba(d.X)));

    const directReg = new RandomForestRegressor({ nEstimators: 12, randomState: 9 }).fit(d.X, d.y);
    const warmReg = new RandomForestRegressor({ nEstimators: 5, randomState: 9, warmStart: true });
    warmReg.fit(d.X, d.y);
    warmReg.setParams({ nEstimators: 12 }).fit(d.X, d.y);
    expect(toArray(warmReg.predict(d.X))).toEqual(toArray(directReg.predict(d.X)));
  });

  it("warm start with fewer trees than already fitted throws instead of keeping extra trees", () => {
    const rf = new RandomForestClassifier({ nEstimators: 6, randomState: 1, warmStart: true });
    rf.fit(X, Y);
    rf.setParams({ nEstimators: 3 });
    expect(() => rf.fit(X, Y)).toThrow(InvalidParameterError);
    const reg = new RandomForestRegressor({ nEstimators: 6, randomState: 1, warmStart: true });
    reg.fit(X, Y_REG);
    reg.setParams({ nEstimators: 3 });
    expect(() => reg.fit(X, Y_REG)).toThrow(InvalidParameterError);
  });

  it("warm start rejects a different feature count or class set", () => {
    const rf = new RandomForestClassifier({ nEstimators: 3, randomState: 1, warmStart: true });
    rf.fit(X, Y);
    rf.setParams({ nEstimators: 6 });
    expect(() => rf.fit(tensor([[1], [2], [3]]), tensor([0, 1, 2]))).toThrow(ShapeError);
    expect(() => rf.fit(X, tensor(Y_LABELS.map((v) => (v === 2 ? 1 : v))))).toThrow(
      DataValidationError
    );
  });

  it("warm start with oobScore needs bootstrap samples for the existing trees", () => {
    // The trees of the first fit have no bootstrap sample; this used to crash with a TypeError.
    const rf = new RandomForestClassifier({ nEstimators: 3, bootstrap: false, warmStart: true });
    rf.fit(X, Y);
    rf.setParams({ bootstrap: true, oobScore: true, nEstimators: 6 });
    expect(() => rf.fit(X, Y)).toThrow(InvalidParameterError);
    const reg = new RandomForestRegressor({ nEstimators: 3, bootstrap: false, warmStart: true });
    reg.fit(X, Y_REG);
    reg.setParams({ bootstrap: true, oobScore: true, nEstimators: 6 });
    expect(() => reg.fit(X, Y_REG)).toThrow(InvalidParameterError);
  });

  it("a failed fit leaves the previous forest usable", () => {
    const rf = new RandomForestClassifier({ nEstimators: 5, randomState: 1 }).fit(X, Y);
    const before = toArray(rf.predictProba(X_TEST));
    expect(() => rf.fit(tensor([[Number.NaN, 1]]), tensor([0]))).toThrow(DataValidationError);
    expect(toArray(rf.predictProba(X_TEST))).toEqual(before);
  });
});

describe("out-of-bag estimates", () => {
  it("regressor oobScore is the R2 of oobPrediction over the covered samples", () => {
    const d = noisyData(80);
    const rf = new RandomForestRegressor({ nEstimators: 30, oobScore: true, randomState: 2 });
    rf.fit(d.X, d.y);
    const pred = toArray(rf.oobPrediction);
    const truth = toArray(d.y);
    const idx = pred.map((_v, i) => i).filter((i) => !Number.isNaN(pred[i]));
    expect(idx.length).toBeGreaterThan(60);
    const mean = idx.reduce((s, i) => s + (truth[i] ?? 0), 0) / idx.length;
    let ssRes = 0;
    let ssTot = 0;
    for (const i of idx) {
      ssRes += ((truth[i] ?? 0) - (pred[i] ?? 0)) ** 2;
      ssTot += ((truth[i] ?? 0) - mean) ** 2;
    }
    expect(rf.oobScore).toBeCloseTo(1 - ssRes / ssTot, 12);
  });

  it("classifier oobScore is the accuracy of argmax(oobDecisionFunction)", () => {
    const d = noisyData(80);
    const rf = new RandomForestClassifier({ nEstimators: 30, oobScore: true, randomState: 2 });
    rf.fit(d.X, d.y);
    const dec = toArray(rf.oobDecisionFunction);
    const truth = toArray(d.y);
    let correct = 0;
    let covered = 0;
    for (let i = 0; i < truth.length; i++) {
      const p0 = dec[2 * i] ?? Number.NaN;
      const p1 = dec[2 * i + 1] ?? Number.NaN;
      if (Number.isNaN(p0)) continue;
      expect(p0 + p1).toBeCloseTo(1, 12);
      covered++;
      if ((p1 > p0 ? 1 : 0) === truth[i]) correct++;
    }
    expect(rf.oobScore).toBeCloseTo(correct / covered, 12);
    expect(rf.oobScore).toBeGreaterThan(0.7);
  });

  it("oobScore is NaN (not 0) when every sample is in every bag", () => {
    const reg = new RandomForestRegressor({ nEstimators: 3, oobScore: true, randomState: 1 });
    reg.fit(tensor([[0]]), tensor([1]));
    expect(reg.oobScore).toBeNaN();
    expect(toArray(reg.oobPrediction).every(Number.isNaN)).toBe(true);
    const clf = new RandomForestClassifier({ nEstimators: 3, oobScore: true, randomState: 1 });
    clf.fit(tensor([[0]]), tensor([1]));
    expect(clf.oobScore).toBeNaN();
  });
});

describe("parameter validation", () => {
  it("maxSamples must be an integer >= 1 or a fraction in (0, 1)", () => {
    for (const bad of [0, -1, 1.5, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(() => new RandomForestClassifier({ maxSamples: bad })).toThrow(InvalidParameterError);
      expect(() => new RandomForestRegressor({ maxSamples: bad })).toThrow(InvalidParameterError);
    }
    expect(() => new RandomForestClassifier({ maxSamples: 0.5 })).not.toThrow();
    expect(() => new RandomForestClassifier({ maxSamples: 7 })).not.toThrow();
  });

  it("maxSamples requires bootstrap", () => {
    expect(() => new RandomForestClassifier({ maxSamples: 5, bootstrap: false })).toThrow(
      /bootstrap/
    );
    const rf = new RandomForestRegressor();
    rf.setParams({ bootstrap: false, maxSamples: 5 });
    expect(() => rf.fit(X, Y_REG)).toThrow(InvalidParameterError);
  });

  it("maxFeatures accepts integers, fractions, sqrt and log2 and rejects the rest", () => {
    for (const ok of [1, 3, 0.5, "sqrt", "log2"] as const) {
      expect(() => new RandomForestClassifier({ maxFeatures: ok })).not.toThrow();
      expect(() => new RandomForestRegressor({ maxFeatures: ok })).not.toThrow();
    }
    for (const bad of [0, -1, 1.5, 2.5, Number.NaN] as const) {
      expect(() => new RandomForestClassifier({ maxFeatures: bad })).toThrow(/maxFeatures/);
      expect(() => new RandomForestRegressor({ maxFeatures: bad })).toThrow(/maxFeatures/);
      expect(() => new RandomForestClassifier().setParams({ maxFeatures: bad })).toThrow(
        /maxFeatures/
      );
    }
  });

  it("a fractional maxFeatures examines that share of the features", () => {
    // With one feature per split the first tree split can only use the chosen feature; here we
    // just check that a fraction fits and predicts.
    const rf = new RandomForestRegressor({ nEstimators: 10, maxFeatures: 0.5, randomState: 1 });
    rf.fit(X, Y_REG);
    expect(rf.predict(X_TEST).shape).toEqual([5]);
  });

  it("maxDepth accepts Infinity", () => {
    const rf = new RandomForestClassifier({
      nEstimators: 3,
      maxDepth: Infinity,
      bootstrap: false,
      randomState: 1,
    });
    expect(rf.fit(X, Y).score(X, Y)).toBe(1);
    expect(() => new RandomForestClassifier({ maxDepth: 2.5 })).toThrow(InvalidParameterError);
    expect(() => new RandomForestClassifier({ maxDepth: Number.NaN })).toThrow(
      InvalidParameterError
    );
  });

  it("setParams is atomic and the classifier now validates like the constructor", () => {
    const rf = new RandomForestClassifier({ nEstimators: 7 });
    expect(() => rf.setParams({ nEstimators: 5, maxDepth: 0 })).toThrow(InvalidParameterError);
    expect(rf.getParams().nEstimators).toBe(7);
    expect(() => rf.setParams({ nEstimators: 5, bogus: 1 })).toThrow(/Unknown parameter/);
    expect(rf.getParams().nEstimators).toBe(7);
    expect(() => rf.setParams({ maxFeatures: 1.5 })).toThrow(InvalidParameterError);
    expect(() => rf.setParams({ criterion: "nope" })).toThrow(InvalidParameterError);
    rf.setParams({ nEstimators: 5, criterion: "entropy" });
    expect(rf.getParams().nEstimators).toBe(5);
    expect(rf.getParams().criterion).toBe("entropy");
  });

  it("setParams cannot create an inconsistent forest silently", () => {
    const rf = new RandomForestClassifier({ oobScore: true });
    rf.setParams({ bootstrap: false });
    expect(() => rf.fit(X, Y)).toThrow(/bootstrap/);
  });

  it("criterion is passed to the trees", () => {
    const a = new RandomForestClassifier({ nEstimators: 5, criterion: "entropy", randomState: 1 });
    expect(a.fit(X, Y).score(X, Y)).toBeGreaterThan(0.8);
    expect(() => new RandomForestClassifier({ criterion: "x" as never })).toThrow(
      InvalidParameterError
    );
  });

  it("clone keeps the hyperparameters, including the regressor default of all features", () => {
    const rf = new RandomForestClassifier({ nEstimators: 9, randomState: 3, maxSamples: 0.5 });
    expect(rf.clone().getParams()).toEqual(rf.getParams());
    const reg = new RandomForestRegressor();
    expect(reg.clone().getParams()).toEqual(reg.getParams());
    expect(reg.clone()).not.toBe(reg);
  });
});

describe("input handling", () => {
  it("rejects scoring and prediction before fit with NotFittedError", () => {
    expect(() => new RandomForestClassifier().predict(X)).toThrow(NotFittedError);
    expect(() => new RandomForestClassifier().score(X, Y)).toThrow(NotFittedError);
    expect(() => new RandomForestRegressor().predict(X)).toThrow(NotFittedError);
    expect(() => new RandomForestRegressor().featureImportances).toThrow(NotFittedError);
  });

  it("score rejects empty and mismatched targets", () => {
    const rf = new RandomForestClassifier({ nEstimators: 3, randomState: 1 }).fit(X, Y);
    expect(() => rf.score(X_TEST, tensor([], { dtype: "int32" }))).toThrow(DataValidationError);
    expect(() => rf.score(X_TEST, tensor([0, 1], { dtype: "int32" }))).toThrow(ShapeError);
    const reg = new RandomForestRegressor({ nEstimators: 3, randomState: 1 }).fit(X, Y_REG);
    expect(() => reg.score(X_TEST, tensor([], { dtype: "float64" }))).toThrow(DataValidationError);
  });

  it("does not modify X or y", () => {
    const x = tensor(
      [
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
      ],
      { dtype: "float64" }
    );
    const y = tensor([0, 1, 0, 1], { dtype: "int32" });
    const xCopy = toArray(x);
    const yCopy = toArray(y);
    new RandomForestClassifier({ nEstimators: 5, randomState: 1 }).fit(x, y);
    new RandomForestRegressor({ nEstimators: 5, randomState: 1 }).fit(x, y);
    expect(toArray(x)).toEqual(xCopy);
    expect(toArray(y)).toEqual(yCopy);
  });

  it("predict checks the feature count", () => {
    const rf = new RandomForestRegressor({ nEstimators: 3, randomState: 1 }).fit(X, Y_REG);
    expect(() => rf.predict(tensor([[1, 2, 3]]))).toThrow(ShapeError);
  });

  it("feature importances sum to 1 and find the informative feature", () => {
    const d = noisyData(150);
    const rf = new RandomForestClassifier({ nEstimators: 30, randomState: 1 }).fit(d.X, d.y);
    const imp = toArray(rf.featureImportances);
    expect(imp[0] ?? 0).toBeGreaterThan(imp[1] ?? 1);
    expect(imp.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
  });
});
