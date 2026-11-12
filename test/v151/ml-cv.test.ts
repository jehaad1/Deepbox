import { describe, expect, it } from "vitest";
import { InvalidParameterError } from "../../src/core";
import {
  crossValidate,
  crossValScore,
  type Estimator,
  GridSearchCV,
  RandomizedSearchCV,
  Ridge,
} from "../../src/ml";
import { type Tensor, tensor } from "../../src/ndarray";
import { GroupKFold, KFold, TimeSeriesSplit } from "../../src/preprocess";

// Reference values: scikit-learn 1.9.1 cross_val_score(Ridge(alpha=1.0), X, y, cv=...) on the
// data built by makeData(24), with TimeSeriesSplit(n_splits=4), GroupKFold(n_splits=4) and KFold(4).
const SK_TS = [0.8428138248967412, 0.9561631581809866, 0.9211332855828223, 0.9349329571478732];
const SK_GKF = [0.9777175067683163, 0.9677951831193917, 0.9473527042535734, 0.9735106502959928];
const SK_KF = [0.9635459014518096, 0.9595993105551525, 0.9671692431662349, 0.9224124792113596];
// sklearn StratifiedKFold(4) and KFold(4) test indices (shuffle=False) on the labels of makeLabels(26).
const SK_STRAT_TESTS = [
  [0, 1, 2, 3, 4, 5, 7],
  [6, 8, 9, 10, 11, 13, 14],
  [12, 15, 16, 17, 19, 22],
  [18, 20, 21, 23, 24, 25],
];
const SK_KF_TESTS = [
  [0, 1, 2, 3, 4, 5, 6],
  [7, 8, 9, 10, 11, 12, 13],
  [14, 15, 16, 17, 18, 19],
  [20, 21, 22, 23, 24, 25],
];

function makeData(n: number): { X: Tensor; y: Tensor } {
  const rows: number[][] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    const row = [Math.sin(i * 0.7), Math.cos(i * 1.3), (i % 5) * 0.3];
    rows.push(row);
    ys.push(2 * row[0]! - row[1]! + 0.5 * row[2]! + 0.3 * Math.sin(i * 2.1));
  }
  return { X: tensor(rows, { dtype: "float64" }), y: tensor(ys, { dtype: "float64" }) };
}

function makeLabels(n: number): Tensor {
  return tensor(
    Array.from({ length: n }, (_, i) => (i * i + Math.floor(i / 3)) % 3),
    { dtype: "float64" }
  );
}

/** Estimator that records the sample indices (stored in column 0 of X) of every test fold. */
class IndexSpy {
  static tests: number[][] = [];
  static trains: number[][] = [];
  private train: number[] = [];
  classes: Tensor | undefined;
  constructor(private readonly isClassifier = false) {}
  fit(X: Tensor): this {
    this.train = Array.from(X.toArray() as number[][], (r) => r[0] as number);
    if (this.isClassifier) this.classes = tensor([0, 1, 2]);
    return this;
  }
  predict(X: Tensor): Tensor {
    return tensor(new Array<number>(X.shape[0] ?? 0).fill(0));
  }
  score(X: Tensor): number {
    IndexSpy.trains.push(this.train);
    IndexSpy.tests.push(Array.from(X.toArray() as number[][], (r) => r[0] as number));
    return 0;
  }
  getParams(): Record<string, unknown> {
    return { isClassifier: this.isClassifier };
  }
  setParams(): this {
    return this;
  }
  clone(): IndexSpy {
    return new IndexSpy(this.isClassifier);
  }
}

function indexData(n: number): { X: Tensor; y: Tensor } {
  return {
    X: tensor(
      Array.from({ length: n }, (_, i) => [i]),
      { dtype: "float64" }
    ),
    y: tensor(
      Array.from({ length: n }, (_, i) => i),
      { dtype: "float64" }
    ),
  };
}

function expectScores(actual: readonly number[], expected: readonly number[]): void {
  expect(actual).toHaveLength(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(actual[i]).toBeCloseTo(expected[i] as number, 10);
  }
}

function sorted(rows: number[][]): number[][] {
  return rows.map((r) => [...r].sort((a, b) => a - b));
}

describe("ML-1: cross-validation accepts splitter objects and options", () => {
  it("crossValScore with TimeSeriesSplit matches sklearn", () => {
    const { X, y } = makeData(24);
    const scores = crossValScore(
      new Ridge({ alpha: 1 }),
      X,
      y,
      new TimeSeriesSplit({ nSplits: 4 })
    );
    expect(scores).toHaveLength(4);
    expectScores(scores, SK_TS);
  });

  it("crossValScore with GroupKFold and groups matches sklearn", () => {
    const { X, y } = makeData(24);
    const groups = tensor(Array.from({ length: 24 }, (_, i) => i % 4));
    const scores = crossValScore(new Ridge({ alpha: 1 }), X, y, {
      cv: new GroupKFold({ nSplits: 4 }),
      groups,
    });
    expectScores(scores, SK_GKF);
  });

  it("crossValScore with KFold (no shuffle) matches sklearn", () => {
    const { X, y } = makeData(24);
    const scores = crossValScore(new Ridge({ alpha: 1 }), X, y, new KFold({ nSplits: 4 }));
    expectScores(scores, SK_KF);
  });

  it("an options object as 4th argument works and positional form still works", () => {
    const { X, y } = makeData(24);
    const scorer = (est: Estimator, Xt: Tensor, yt: Tensor): number => (est as Ridge).score(Xt, yt);
    const a = crossValScore(new Ridge(), X, y, 3, scorer);
    const b = crossValScore(new Ridge(), X, y, { cv: 3, scoring: scorer });
    expect(b).toEqual(a);
  });

  it("integer cv with shuffle false gives sklearn's contiguous KFold folds", () => {
    const { X, y } = indexData(26);
    IndexSpy.tests = [];
    crossValScore(new IndexSpy() as unknown as Ridge, X, y, { cv: 4, shuffle: false });
    expect(IndexSpy.tests).toEqual(SK_KF_TESTS);
  });

  it("integer cv with shuffle false gives sklearn's StratifiedKFold folds for classifiers", () => {
    const { X } = indexData(26);
    const y = makeLabels(26);
    IndexSpy.tests = [];
    crossValScore(new IndexSpy(true) as unknown as Ridge, X, y, { cv: 4, shuffle: false });
    expect(sorted(IndexSpy.tests)).toEqual(SK_STRAT_TESTS);
  });

  it("randomState seeds the shuffle: same seed repeats, another seed differs, default is unchanged", () => {
    const { X, y } = indexData(26);
    const run = (opts: Record<string, unknown>): number[][] => {
      IndexSpy.tests = [];
      crossValScore(new IndexSpy() as unknown as Ridge, X, y, { cv: 4, ...opts });
      return IndexSpy.tests;
    };
    const base = run({});
    expect(run({ randomState: 0 })).toEqual(base);
    expect(run({ randomState: 5 })).toEqual(run({ randomState: 5 }));
    expect(run({ randomState: 5 })).not.toEqual(base);
    expect(run({ shuffle: true })).toEqual(base);
  });

  it("crossValidate takes a splitter and groups", () => {
    const { X, y } = makeData(24);
    const res = crossValidate(new Ridge({ alpha: 1 }), X, y, {
      cv: new TimeSeriesSplit({ nSplits: 4 }),
    });
    expectScores(res.testScores["score"] as number[], SK_TS);
    const groups = tensor(Array.from({ length: 24 }, (_, i) => i % 4));
    const res2 = crossValidate(new Ridge({ alpha: 1 }), X, y, {
      cv: new GroupKFold({ nSplits: 4 }),
      groups,
    });
    expectScores(res2.testScores["score"] as number[], SK_GKF);
  });

  it("GridSearchCV and RandomizedSearchCV accept a splitter", () => {
    const { X, y } = makeData(24);
    const gs = new GridSearchCV(
      new Ridge(),
      { alpha: [0.1, 1, 10] },
      {
        cv: new TimeSeriesSplit({ nSplits: 4 }),
      }
    ).fit(X, y);
    expect(gs.cvResults).toHaveLength(3);
    expect(gs.cvResults.every((r) => r.scores.length === 4)).toBe(true);
    const alpha1 = gs.cvResults[1]?.scores ?? [];
    expectScores(alpha1, SK_TS);
    const rs = new RandomizedSearchCV(
      new Ridge(),
      { alpha: [0.1, 1, 10] },
      {
        cv: new TimeSeriesSplit({ nSplits: 4 }),
        nIter: 3,
        randomState: 1,
      }
    ).fit(X, y);
    expect(rs.cvResults.every((r) => r.scores.length === 4)).toBe(true);
  });

  it("GridSearchCV fit forwards groups to the splitter", () => {
    const { X, y } = makeData(24);
    const groups = tensor(Array.from({ length: 24 }, (_, i) => i % 4));
    const gs = new GridSearchCV(
      new Ridge(),
      { alpha: [1] },
      {
        cv: new GroupKFold({ nSplits: 4 }),
      }
    ).fit(X, y, groups);
    expectScores(gs.cvResults[0]?.scores ?? [], SK_GKF);
  });

  it("a splitter that needs groups fails clearly without them", () => {
    const { X, y } = makeData(24);
    expect(() => crossValScore(new Ridge(), X, y, new GroupKFold({ nSplits: 4 }))).toThrow();
  });

  it("rejects an invalid cv", () => {
    const { X, y } = makeData(24);
    expect(() => crossValScore(new Ridge(), X, y, { cv: {} as never })).toThrow(
      InvalidParameterError
    );
    expect(() => crossValScore(new Ridge(), X, y, 1)).toThrow(InvalidParameterError);
    expect(() => new GridSearchCV(new Ridge(), { alpha: [1] }, { cv: "x" as never })).toThrow(
      InvalidParameterError
    );
  });

  it("rejects splits with out-of-range or empty folds", () => {
    const { X, y } = makeData(24);
    const bad = { split: () => [{ trainIndex: [0, 1, 99], testIndex: [2] }] };
    expect(() => crossValScore(new Ridge(), X, y, bad)).toThrow(InvalidParameterError);
    const empty = { split: () => [{ trainIndex: [0, 1], testIndex: [] as number[] }] };
    expect(() => crossValScore(new Ridge(), X, y, empty)).toThrow(InvalidParameterError);
  });

  it("getParams and setParams carry the splitter", () => {
    const cv = new TimeSeriesSplit({ nSplits: 3 });
    const gs = new GridSearchCV(new Ridge(), { alpha: [1] }, { cv });
    expect(gs.getParams()["cv"]).toBe(cv);
    gs.setParams({ cv: 4 });
    expect(gs.getParams()["cv"]).toBe(4);
    gs.setParams({ cv });
    expect(gs.getParams()["cv"]).toBe(cv);
  });
});
