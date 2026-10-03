/**
 * Regression tests for the 1.5.0 audit of src/preprocess/split.ts and src/preprocess/text.ts.
 *
 * Reference values come from scikit-learn 1.8 (StratifiedKFold, GroupKFold, TimeSeriesSplit,
 * LeavePOut, StratifiedShuffleSplit, GroupShuffleSplit, CountVectorizer, TfidfVectorizer).
 */
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import {
  DataValidationError,
  InvalidParameterError,
  MemoryError,
  NotFittedError,
  resetConfig,
  ShapeError,
  setDtype,
} from "../../src/core";
import { Tensor, tensor } from "../../src/ndarray";
import {
  CountVectorizer,
  GroupKFold,
  GroupShuffleSplit,
  HashingVectorizer,
  KFold,
  LeaveOneOut,
  LeavePOut,
  RepeatedKFold,
  RepeatedStratifiedKFold,
  ShuffleSplit,
  type SplitResult,
  StratifiedKFold,
  StratifiedShuffleSplit,
  TfidfVectorizer,
  TimeSeriesSplit,
  trainTestSplit,
} from "../../src/preprocess";

beforeAll(() => setDtype("float64"));
afterAll(() => resetConfig());

function column(values: readonly number[]): Tensor {
  return tensor(values.map((v) => [v]));
}

function rows(n: number): Tensor {
  return column(Array.from({ length: n }, (_, i) => i));
}

function pairs(splits: SplitResult[]): number[][][] {
  return splits.map((s) => [s.trainIndex, s.testIndex]);
}

function isAscending(values: readonly number[]): boolean {
  return values.every((v, i) => i === 0 || (values[i - 1] as number) < v);
}

function classCounts(labels: readonly number[], indices: readonly number[], nClasses: number) {
  const counts = new Array<number>(nClasses).fill(0);
  for (const i of indices) counts[labels[i] as number] = (counts[labels[i] as number] ?? 0) + 1;
  return counts;
}

describe("StratifiedKFold", () => {
  it("matches scikit-learn fold assignment without shuffling", () => {
    const y = [...Array(7).fill(0), ...Array(5).fill(1), ...Array(3).fill(2)];
    const splits = new StratifiedKFold({ nSplits: 3 }).split(rows(15), tensor(y));
    expect(pairs(splits)).toEqual([
      [
        [3, 4, 5, 6, 8, 9, 10, 11, 13, 14],
        [0, 1, 2, 7, 12],
      ],
      [
        [0, 1, 2, 5, 6, 7, 10, 11, 12, 14],
        [3, 4, 8, 9, 13],
      ],
      [
        [0, 1, 2, 3, 4, 7, 8, 9, 12, 13],
        [5, 6, 10, 11, 14],
      ],
    ]);
  });

  it("keeps fold sizes within one sample of each other (class remainders are spread)", () => {
    // 5 classes of 7 samples, 5 folds: the old per-class allocation gave folds 0 and 1 two extra
    // samples from every class (sizes 10, 10, 5, 5, 5). scikit-learn gives 7 for every fold.
    const y = [
      2, 2, 0, 4, 4, 4, 3, 3, 3, 4, 2, 3, 0, 0, 3, 3, 1, 4, 3, 4, 2, 2, 0, 1, 1, 4, 0, 2, 0, 2, 0,
      1, 1, 1, 1,
    ];
    for (const shuffle of [false, true]) {
      const splits = new StratifiedKFold({ nSplits: 5, shuffle, randomState: 3 }).split(
        rows(35),
        tensor(y)
      );
      expect(splits.map((s) => s.testIndex.length)).toEqual([7, 7, 7, 7, 7]);
      for (const s of splits) {
        expect(classCounts(y, s.testIndex, 5).every((c) => c === 1 || c === 2)).toBe(true);
      }
    }
  });

  it("returns ascending indices that partition the samples", () => {
    const y = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
    const splits = new StratifiedKFold({ nSplits: 5, shuffle: true, randomState: 1 }).split(
      rows(10),
      tensor(y)
    );
    const seen: number[] = [];
    for (const s of splits) {
      expect(isAscending(s.trainIndex)).toBe(true);
      expect(isAscending(s.testIndex)).toBe(true);
      expect(s.trainIndex.length + s.testIndex.length).toBe(10);
      seen.push(...s.testIndex);
    }
    expect(seen.sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });

  it("supports string labels", () => {
    const y = tensor(["b", "a", "b", "a", "b", "a"]);
    const splits = new StratifiedKFold({ nSplits: 3 }).split(rows(6), y);
    for (const s of splits) expect(s.testIndex.length).toBe(2);
  });

  it("validates nSplits in the constructor", () => {
    expect(() => new StratifiedKFold({ nSplits: 1 })).toThrow(InvalidParameterError);
    expect(() => new KFold({ nSplits: 2.5 })).toThrow(InvalidParameterError);
    expect(() => new GroupKFold({ nSplits: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new KFold({ randomState: -1 })).toThrow(/randomState/);
  });
});

describe("KFold", () => {
  it("returns ascending train and test indices when shuffling (like scikit-learn)", () => {
    const splits = new KFold({ nSplits: 3, shuffle: true, randomState: 0 }).split(rows(7));
    const seen: number[] = [];
    for (const s of splits) {
      expect(isAscending(s.trainIndex)).toBe(true);
      expect(isAscending(s.testIndex)).toBe(true);
      seen.push(...s.testIndex);
    }
    expect(splits.map((s) => s.testIndex.length)).toEqual([3, 2, 2]);
    expect(seen.sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4, 5, 6]);
  });

  it("accepts N-d input and rejects 0-d input", () => {
    expect(new KFold({ nSplits: 2 }).split(tensor([1, 2, 3, 4])).length).toBe(2);
    expect(() => new KFold({ nSplits: 2 }).split(tensor(5))).toThrow(ShapeError);
  });

  it("handles very large folds without call-stack limits", () => {
    const n = 300_000;
    const splits = new KFold({ nSplits: 2 }).split(tensor(new Array<number>(n).fill(0)));
    expect(splits[0]?.trainIndex.length).toBe(n / 2);
  });
});

describe("GroupKFold", () => {
  it("matches scikit-learn fold assignment", () => {
    const groups = [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 4, 4, 5];
    const splits = new GroupKFold({ nSplits: 3 }).split(rows(13), undefined, tensor(groups));
    expect(pairs(splits)).toEqual([
      [
        [0, 1, 2, 3, 4, 10, 11, 12],
        [5, 6, 7, 8, 9],
      ],
      [
        [3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 12],
      ],
      [
        [0, 1, 2, 5, 6, 7, 8, 9, 12],
        [3, 4, 10, 11],
      ],
    ]);
  });

  it("breaks equal-size ties like scikit-learn (later labels first)", () => {
    const splits = new GroupKFold({ nSplits: 3 }).split(
      rows(6),
      undefined,
      tensor([0, 1, 2, 3, 4, 5])
    );
    expect(pairs(splits)).toEqual([
      [
        [0, 1, 3, 4],
        [2, 5],
      ],
      [
        [0, 2, 3, 5],
        [1, 4],
      ],
      [
        [1, 2, 4, 5],
        [0, 3],
      ],
    ]);
  });

  it("accepts string groups as an array and as a tensor", () => {
    const groups = ["b", "a", "b", "c", "a", "c", "d"];
    const expected = [
      [
        [0, 2, 6],
        [1, 3, 4, 5],
      ],
      [
        [1, 3, 4, 5],
        [0, 2, 6],
      ],
    ];
    const fromArray = new GroupKFold({ nSplits: 2 }).split(rows(7), undefined, groups);
    const fromTensor = new GroupKFold({ nSplits: 2 }).split(rows(7), undefined, tensor(groups));
    expect(pairs(fromArray)).toEqual(expected);
    expect(pairs(fromTensor)).toEqual(expected);
  });

  it("supports shuffle with a seed and keeps groups intact", () => {
    const groups = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4];
    const run = () =>
      new GroupKFold({ nSplits: 5, shuffle: true, randomState: 9 }).split(
        rows(10),
        undefined,
        groups
      );
    expect(pairs(run())).toEqual(pairs(run()));
    for (const s of run()) {
      expect(s.testIndex.length).toBe(2);
      const testGroups = new Set(s.testIndex.map((i) => groups[i]));
      expect(testGroups.size).toBe(1);
      expect(s.trainIndex.some((i) => testGroups.has(groups[i]))).toBe(false);
    }
  });

  it("rejects groups of the wrong length or shape", () => {
    const gkf = new GroupKFold({ nSplits: 2 });
    expect(() => gkf.split(rows(4), undefined, [0, 1, 2])).toThrow(/same number of samples/);
    expect(() => gkf.split(rows(4), undefined, column([0, 1, 2, 3]))).toThrow(/1D/);
  });
});

describe("LeaveOneOut and LeavePOut", () => {
  it("LeavePOut matches scikit-learn order and returns ascending train indices", () => {
    const splits = new LeavePOut(2).split(rows(4));
    expect(pairs(splits)).toEqual([
      [
        [2, 3],
        [0, 1],
      ],
      [
        [1, 3],
        [0, 2],
      ],
      [
        [1, 2],
        [0, 3],
      ],
      [
        [0, 3],
        [1, 2],
      ],
      [
        [0, 2],
        [1, 3],
      ],
      [
        [0, 1],
        [2, 3],
      ],
    ]);
  });

  it("rejects p equal to the number of samples (empty train set)", () => {
    expect(() => new LeavePOut(3).split(rows(3))).toThrow(InvalidParameterError);
    expect(() => new LeavePOut(3).getNSplits(rows(3))).toThrow(InvalidParameterError);
    expect(() => new LeavePOut(4).split(rows(3))).toThrow(/greater than number of samples/);
  });

  it("LeavePOut stays iterative for large p and refuses oversized outputs", () => {
    const splits = new LeavePOut(1999).split(rows(2000));
    expect(splits.length).toBe(2000);
    expect(splits[0]?.trainIndex.length).toBe(1);
    expect(() => new LeavePOut(1).split(rows(60_000))).toThrow(MemoryError);
  });

  it("LeaveOneOut needs at least 2 samples and guards memory", () => {
    expect(() => new LeaveOneOut().split(rows(1))).toThrow(InvalidParameterError);
    expect(new LeaveOneOut().split(rows(3))[1]).toEqual({ trainIndex: [0, 2], testIndex: [1] });
    expect(() => new LeaveOneOut().split(rows(10_000))).toThrow(MemoryError);
  });
});

describe("TimeSeriesSplit", () => {
  it("matches scikit-learn with gap, testSize and maxTrainSize", () => {
    const splits = new TimeSeriesSplit({
      nSplits: 3,
      gap: 1,
      maxTrainSize: 3,
      testSize: 2,
    }).split(rows(14));
    expect(pairs(splits)).toEqual([
      [
        [4, 5, 6],
        [8, 9],
      ],
      [
        [6, 7, 8],
        [10, 11],
      ],
      [
        [8, 9, 10],
        [12, 13],
      ],
    ]);
  });

  it("throws instead of silently dropping splits when there are too few samples", () => {
    // 10 - 0 - 4 * 3 < 0: scikit-learn raises, the old code returned fewer splits.
    expect(() => new TimeSeriesSplit({ nSplits: 3, testSize: 4 }).split(rows(10))).toThrow(
      /Too many splits/
    );
    expect(() => new TimeSeriesSplit({ nSplits: 2, gap: 4 }).split(rows(6))).toThrow(
      InvalidParameterError
    );
  });

  it("validates testSize and maxTrainSize", () => {
    expect(() => new TimeSeriesSplit({ testSize: 0 })).toThrow(/testSize/);
    expect(() => new TimeSeriesSplit({ testSize: 1.5 })).toThrow(/testSize/);
    expect(() => new TimeSeriesSplit({ maxTrainSize: 0 })).toThrow(/maxTrainSize/);
  });
});

describe("ShuffleSplit family", () => {
  it("ShuffleSplit defaults to a 10% test set (scikit-learn), not 25%", () => {
    const [split] = new ShuffleSplit({ nSplits: 1, randomState: 0 }).split(rows(25));
    expect(split?.trainIndex.length).toBe(22);
    expect(split?.testIndex.length).toBe(3);
  });

  it("StratifiedShuffleSplit returns exactly the requested sizes with proportional classes", () => {
    // sklearn: sizes 40/10 with class counts [24, 12, 4] / [6, 3, 1]. The old rounding gave
    // 6 + 3 + 1 test samples here only by luck and varied the totals for other sizes.
    const y = [...Array(30).fill(0), ...Array(15).fill(1), ...Array(5).fill(2)];
    const splits = new StratifiedShuffleSplit({ nSplits: 4, testSize: 0.2, randomState: 0 }).split(
      rows(50),
      tensor(y)
    );
    for (const s of splits) {
      expect(s.trainIndex.length).toBe(40);
      expect(s.testIndex.length).toBe(10);
      expect(classCounts(y, s.trainIndex, 3)).toEqual([24, 12, 4]);
      expect(classCounts(y, s.testIndex, 3)).toEqual([6, 3, 1]);
      expect(new Set([...s.trainIndex, ...s.testIndex]).size).toBe(50);
    }
  });

  it("StratifiedShuffleSplit matches the requested total for awkward class sizes", () => {
    const y = [...Array(7).fill(0), ...Array(5).fill(1), ...Array(3).fill(2)];
    const splits = new StratifiedShuffleSplit({ nSplits: 3, testSize: 6, randomState: 4 }).split(
      rows(15),
      tensor(y)
    );
    for (const s of splits) {
      expect(s.testIndex.length).toBe(6);
      expect(s.trainIndex.length).toBe(9);
      expect(classCounts(y, s.testIndex, 3).every((c) => c >= 1)).toBe(true);
      expect(new Set([...s.trainIndex, ...s.testIndex]).size).toBe(15);
    }
  });

  it("StratifiedShuffleSplit rejects singleton classes and 2D labels", () => {
    const sss = new StratifiedShuffleSplit({ nSplits: 1, testSize: 0.5 });
    expect(() => sss.split(rows(4), tensor([0, 0, 0, 1]))).toThrow(/at least 2 samples per class/);
    expect(() => sss.split(rows(4), column([0, 0, 1, 1]))).toThrow(/1D/);
    expect(() => sss.split(rows(4), tensor([0, 1]))).toThrow(/same number of samples/);
    expect(
      () =>
        new StratifiedShuffleSplit({ nSplits: 1, testSize: 2 }).split(
          rows(8),
          tensor([0, 0, 0, 0, 1, 1, 1, 1])
        ).length
    ).not.toThrow();
    expect(() =>
      new StratifiedShuffleSplit({ nSplits: 1, testSize: 1 }).split(
        rows(8),
        tensor([0, 0, 0, 0, 1, 1, 1, 1])
      )
    ).toThrow(/testSize must be at least the number of classes/);
  });

  it("StratifiedShuffleSplit handles string labels", () => {
    const y = tensor(["x", "x", "x", "x", "y", "y", "y", "y"]);
    const [s] = new StratifiedShuffleSplit({ nSplits: 1, testSize: 0.5, randomState: 2 }).split(
      rows(8),
      y
    );
    expect(s?.testIndex.filter((i) => i < 4).length).toBe(2);
  });

  it("GroupShuffleSplit rounds the test group count up and returns ascending indices", () => {
    // 7 groups, testSize 0.2: sklearn uses ceil(1.4) = 2 test groups and floor(5.6) = 5 train
    // groups. The old code used round(1.4) = 1.
    const groups = [0, 0, 1, 1, 2, 3, 3, 4, 5, 6];
    const splits = new GroupShuffleSplit({ nSplits: 5, randomState: 1 }).split(
      rows(10),
      undefined,
      groups
    );
    for (const s of splits) {
      const testGroups = new Set(s.testIndex.map((i) => groups[i]));
      const trainGroups = new Set(s.trainIndex.map((i) => groups[i]));
      expect(testGroups.size).toBe(2);
      expect(trainGroups.size).toBe(5);
      for (const g of testGroups) expect(trainGroups.has(g)).toBe(false);
      expect(isAscending(s.trainIndex)).toBe(true);
      expect(isAscending(s.testIndex)).toBe(true);
    }
  });

  it("GroupShuffleSplit supports trainSize, integer sizes and string groups", () => {
    const groups = ["a", "a", "b", "b", "c", "c", "d", "d"];
    const [s] = new GroupShuffleSplit({
      nSplits: 1,
      testSize: 1,
      trainSize: 2,
      randomState: 0,
    }).split(rows(8), undefined, tensor(groups));
    expect(s?.testIndex.length).toBe(2);
    expect(s?.trainIndex.length).toBe(4);
    const [byFraction] = new GroupShuffleSplit({
      nSplits: 1,
      testSize: 0.25,
      randomState: 0,
    }).split(rows(8), undefined, groups);
    expect(byFraction?.testIndex.length).toBe(2);
    expect(byFraction?.trainIndex.length).toBe(6);
  });

  it("GroupShuffleSplit throws instead of returning an empty train set", () => {
    const gss = new GroupShuffleSplit({ nSplits: 1 });
    expect(() => gss.split(rows(3), undefined, [0, 0, 0])).toThrow(InvalidParameterError);
    expect(() => gss.split(rows(3), undefined, [0, 1])).toThrow(/same number of samples/);
    expect(() => gss.split(rows(3))).toThrow(/groups/);
    expect(() => new GroupShuffleSplit({ nSplits: 1.5 })).toThrow(InvalidParameterError);
    expect(() => new GroupShuffleSplit({ testSize: -1 })).toThrow(InvalidParameterError);
  });

  it("ShuffleSplit and StratifiedShuffleSplit validate randomState up front", () => {
    expect(() => new ShuffleSplit({ randomState: 1.5 })).toThrow(/randomState/);
    expect(() => new StratifiedShuffleSplit({ testSize: 0 })).toThrow(/testSize/);
  });
});

describe("Repeated cross-validators", () => {
  it("are reproducible with a seed and validate options in the constructor", () => {
    const a = new RepeatedKFold({ nSplits: 3, nRepeats: 2, randomState: 5 }).split(rows(9));
    const b = new RepeatedKFold({ nSplits: 3, nRepeats: 2, randomState: 5 }).split(rows(9));
    expect(pairs(a)).toEqual(pairs(b));
    expect(a.length).toBe(6);
    expect(() => new RepeatedKFold({ nRepeats: 0 })).toThrow(/nRepeats/);
    expect(() => new RepeatedStratifiedKFold({ randomState: -2 })).toThrow(/randomState/);
  });
});

describe("trainTestSplit", () => {
  it("rounds fractions correctly: 7% of 100 samples is 7 (not 8 from 7.000000000000001)", () => {
    const [, XTest] = trainTestSplit(rows(100), undefined, { testSize: 0.07 });
    expect(XTest.shape[0]).toBe(7);
    // 0.29 * 100 = 28.999999999999996 in floating point; train must still get 29.
    const [XTrain, XTest2] = trainTestSplit(rows(100), undefined, { trainSize: 0.29 });
    expect(XTrain.shape[0]).toBe(29);
    expect(XTest2.shape[0]).toBe(71);
    // fractions that sum to exactly 1 (up to float noise) are accepted
    const [a, b] = trainTestSplit(rows(100), undefined, { trainSize: 0.57, testSize: 0.43 });
    expect(a.shape[0] + b.shape[0]).toBe(100);
  });

  it("splits N-d data and 1D data along the first axis", () => {
    const data = Float64Array.from({ length: 4 * 2 * 3 }, (_, i) => i);
    const X = Tensor.fromTypedArray({ data, shape: [4, 2, 3], dtype: "float64", device: "cpu" });
    const [XTrain, XTest] = trainTestSplit(X, undefined, { testSize: 1, shuffle: false });
    expect(XTrain.shape).toEqual([3, 2, 3]);
    expect(XTest.shape).toEqual([1, 2, 3]);
    expect(Array.from(XTest.data as Float64Array)).toEqual([18, 19, 20, 21, 22, 23]);

    const [yTrain, yTest] = trainTestSplit(tensor([10, 11, 12, 13, 14]), undefined, {
      testSize: 2,
      shuffle: false,
    });
    expect(yTrain.toArray()).toEqual([10, 11, 12]);
    expect(yTest.toArray()).toEqual([13, 14]);
  });

  it("supports multi-output targets", () => {
    const X = rows(4);
    const y = tensor([
      [0, 1],
      [2, 3],
      [4, 5],
      [6, 7],
    ]);
    const [, , yTrain, yTest] = trainTestSplit(X, y, { testSize: 1, shuffle: false });
    expect(yTrain.toArray()).toEqual([
      [0, 1],
      [2, 3],
      [4, 5],
    ]);
    expect(yTest.toArray()).toEqual([[6, 7]]);
  });

  it("copies strided and offset views correctly", () => {
    // Element (i, j) lives at data[1 + 4 i + 2 j]; shape [3, 2], so rows are not block-contiguous.
    const data = Float64Array.from({ length: 12 }, (_, i) => i);
    const view = Tensor.fromTypedArray({
      data,
      shape: [3, 2],
      dtype: "float64",
      device: "cpu",
      offset: 1,
      strides: [4, 2],
    });
    expect(view.toArray()).toEqual([
      [1, 3],
      [5, 7],
      [9, 11],
    ]);
    const [XTrain, XTest] = trainTestSplit(view, undefined, { testSize: 1, shuffle: false });
    expect(XTrain.toArray()).toEqual([
      [1, 3],
      [5, 7],
    ]);
    expect(XTest.toArray()).toEqual([[9, 11]]);

    // Row-contiguous view whose row stride is not the row length (every other row).
    const everyOther = Tensor.fromTypedArray({
      data: Float64Array.from({ length: 12 }, (_, i) => i),
      shape: [3, 2],
      dtype: "float64",
      device: "cpu",
      strides: [4, 1],
    });
    const [, XTest2] = trainTestSplit(everyOther, undefined, { testSize: 1, shuffle: false });
    expect(XTest2.toArray()).toEqual([[8, 9]]);
  });

  it("keeps the dtype for int64, string and float16 data", () => {
    const y64 = Tensor.fromTypedArray({
      data: new BigInt64Array([5n, 6n, 7n, 8n]),
      shape: [4],
      dtype: "int64",
      device: "cpu",
    });
    const [, , yTrain, yTest] = trainTestSplit(rows(4), y64, { testSize: 1, shuffle: false });
    expect(yTrain.dtype).toBe("int64");
    expect(Array.from(yTrain.data as BigInt64Array)).toEqual([5n, 6n, 7n]);
    expect(Array.from(yTest.data as BigInt64Array)).toEqual([8n]);

    const names = tensor([
      ["a", "b"],
      ["c", "d"],
      ["e", "f"],
    ]);
    const [sTrain, sTest] = trainTestSplit(names, undefined, { testSize: 1, shuffle: false });
    expect(sTrain.dtype).toBe("string");
    expect(sTrain.toArray()).toEqual([
      ["a", "b"],
      ["c", "d"],
    ]);
    expect(sTest.toArray()).toEqual([["e", "f"]]);

    const half = tensor([[1], [2], [3]], { dtype: "float16" });
    const [hTrain] = trainTestSplit(half, undefined, { testSize: 1, shuffle: false });
    expect(hTrain.dtype).toBe("float16");
  });

  it("never modifies its inputs", () => {
    const X = rows(6);
    const y = tensor([0, 1, 0, 1, 0, 1]);
    const before = X.toArray();
    trainTestSplit(X, y, { testSize: 0.5, stratify: y, randomState: 1 });
    expect(X.toArray()).toEqual(before);
    expect(y.toArray()).toEqual([0, 1, 0, 1, 0, 1]);
  });

  it("stratify matches the requested sizes exactly and always includes every class", () => {
    const y = [...Array(30).fill(0), ...Array(15).fill(1), ...Array(5).fill(2)];
    const [, , yTrain, yTest] = trainTestSplit(rows(50), tensor(y), {
      testSize: 0.2,
      stratify: tensor(y),
      randomState: 11,
    });
    const train = yTrain.toArray() as number[];
    const test = yTest.toArray() as number[];
    expect(train.length).toBe(40);
    expect(test.length).toBe(10);
    expect(
      classCounts(
        train,
        train.map((_, i) => i),
        3
      )
    ).toEqual([24, 12, 4]);
    expect(
      classCounts(
        test,
        test.map((_, i) => i),
        3
      )
    ).toEqual([6, 3, 1]);
  });

  it("stratify without shuffling keeps the original sample order", () => {
    const y = tensor([0, 1, 0, 1, 0, 1, 0, 1]);
    const [XTrain, XTest] = trainTestSplit(rows(8), y, {
      testSize: 0.5,
      stratify: y,
      shuffle: false,
    });
    const train = (XTrain.toArray() as number[][]).map((r) => r[0] as number);
    const test = (XTest.toArray() as number[][]).map((r) => r[0] as number);
    expect(isAscending(train)).toBe(true);
    expect(isAscending(test)).toBe(true);
    expect(train.length).toBe(4);
  });

  it("stratify rejects a train set smaller than the number of classes even without trainSize", () => {
    const y = tensor([0, 0, 1, 1, 2, 2]);
    expect(() => trainTestSplit(rows(6), y, { stratify: y, testSize: 5 })).toThrow(
      /trainSize must be at least the number of classes/
    );
  });

  it("stratify handles 250k samples (no spread-argument stack overflow)", () => {
    const n = 250_000;
    const y = Int32Array.from({ length: n }, (_, i) => i % 2);
    const yT = Tensor.fromTypedArray({ data: y, shape: [n], dtype: "int32", device: "cpu" });
    const [, , yTrain, yTest] = trainTestSplit(yT, yT, {
      testSize: 0.2,
      stratify: yT,
      randomState: 0,
    });
    expect(yTrain.shape[0]).toBe(200_000);
    expect(yTest.shape[0]).toBe(50_000);
  });

  it("reports the shape of an empty input", () => {
    expect(() => trainTestSplit(tensor([]))).toThrow(/shape \[0\]/);
    expect(() => trainTestSplit(tensor(3))).toThrow(ShapeError);
  });
});

describe("string and label ordering", () => {
  it("orders string classes by code unit, independent of locale", () => {
    // "B" < "a" in code unit order but "a" < "B" under localeCompare. With classes
    // sorted by code unit the allocation tie-break is deterministic and locale independent.
    const y = tensor(["a", "a", "B", "B"]);
    const [s] = new StratifiedShuffleSplit({ nSplits: 1, testSize: 2, randomState: 0 }).split(
      rows(4),
      y
    );
    expect(s?.testIndex.length).toBe(2);
    expect(s?.testIndex.some((i) => i < 2)).toBe(true);
    expect(s?.testIndex.some((i) => i >= 2)).toBe(true);
  });
});

describe("CountVectorizer", () => {
  it("tokenizes Unicode text like scikit-learn", () => {
    const names = new CountVectorizer()
      .fitText(["Café au lait, naïve Zoë 日本語 テスト"])
      .getFeatureNames();
    expect(names).toEqual(["au", "café", "lait", "naïve", "zoë", "テスト", "日本語"]);
  });

  it("sorts the vocabulary by code unit like scikit-learn (uppercase before lowercase)", () => {
    const cv = new CountVectorizer({ lowercase: false }).fitText([
      "Zebra apple",
      "Zebra Café Éclair",
    ]);
    expect(cv.getFeatureNames()).toEqual(["Café", "Zebra", "apple", "Éclair"]);
  });

  it("applies proportional minDf and maxDf without rounding", () => {
    const docs = ["a1 b1 c1", "a1 b1", "a1 d1", "e1 f1", "e1 g1"];
    // 0.5 * 5 = 2.5 documents: minDf keeps df >= 3, maxDf keeps df <= 2.
    expect(new CountVectorizer({ minDf: 0.5 }).fitText(docs).getFeatureNames()).toEqual(["a1"]);
    expect(new CountVectorizer({ maxDf: 0.5 }).fitText(docs).getFeatureNames()).toEqual([
      "b1",
      "c1",
      "d1",
      "e1",
      "f1",
      "g1",
    ]);
  });

  it("throws when the vocabulary is empty or the df limits contradict each other", () => {
    expect(() => new CountVectorizer({ stopWords: ["a1"] }).fitText(["a1"])).toThrow(
      DataValidationError
    );
    expect(() => new CountVectorizer().fitText([])).toThrow(DataValidationError);
    expect(() => new CountVectorizer({ minDf: 3, maxDf: 2 }).fitText(["aa bb", "aa"])).toThrow(
      /maxDf/
    );
  });

  it("ranks maxFeatures by document frequency when binary is set (scikit-learn)", () => {
    // "aa" occurs 3 times in one document, "bb" and "cc" once in two documents each.
    const docs = ["aa aa aa bb", "bb cc", "cc dd"];
    expect(new CountVectorizer({ maxFeatures: 2 }).fitText(docs).getFeatureNames()).toEqual([
      "aa",
      "bb",
    ]);
    expect(
      new CountVectorizer({ maxFeatures: 2, binary: true }).fitText(docs).getFeatureNames()
    ).toEqual(["bb", "cc"]);
  });

  it("supports the english stop word list", () => {
    const cv = new CountVectorizer({ ngramRange: [1, 2], stopWords: "english" });
    cv.fitText(["the quick brown fox jumps over the lazy dog"]);
    expect(cv.getFeatureNames()).toEqual([
      "brown",
      "brown fox",
      "dog",
      "fox",
      "fox jumps",
      "jumps",
      "jumps lazy",
      "lazy",
      "lazy dog",
      "quick",
      "quick brown",
    ]);
  });

  it("does not hang on non-global or zero-length token patterns", () => {
    const nonGlobal = new CountVectorizer({ tokenPattern: /[a-z]+/ });
    expect(nonGlobal.fitText(["ab cd"]).getFeatureNames()).toEqual(["ab", "cd"]);
    const zeroLength = new CountVectorizer({ tokenPattern: /[a-z]*/g });
    expect(zeroLength.fitText(["ab cd"]).getFeatureNames()).toEqual(["ab", "cd"]);
    const pattern = /[a-z]+/g;
    pattern.lastIndex = 3;
    new CountVectorizer({ tokenPattern: pattern }).fitText(["ab cd"]);
    expect(pattern.lastIndex).toBe(3);
  });

  it("rejects non-array and non-string documents", () => {
    const cv = new CountVectorizer();
    expect(() => cv.fitText("hello world" as unknown as string[])).toThrow(InvalidParameterError);
    expect(() => cv.fitText(["ok", 3 as unknown as string])).toThrow(/element 1/);
    cv.fitText(["ok go"]);
    expect(() => cv.transformText([null as unknown as string])).toThrow(InvalidParameterError);
  });

  it("validates options", () => {
    expect(() => new CountVectorizer({ minDf: 1.5 })).toThrow(/minDf/);
    expect(() => new CountVectorizer({ minDf: -1 })).toThrow(/minDf/);
    expect(() => new CountVectorizer({ maxDf: 0 })).toThrow(/maxDf/);
    expect(() => new CountVectorizer({ maxDf: 2.5 })).toThrow(/maxDf/);
    expect(() => new CountVectorizer({ stopWords: "german" as "english" })).toThrow(/stopWords/);
    expect(() => new CountVectorizer({ ngramRange: [2, 1] })).toThrow(/ngramRange/);
    expect(() => new CountVectorizer({ tokenPattern: "a+" as unknown as RegExp })).toThrow(
      /tokenPattern/
    );
  });

  it("setParams changes parameters, invalidates the fit and rejects bad input", () => {
    const cv = new CountVectorizer().setParams({ lowercase: false, binary: true });
    expect(cv.getParams().lowercase).toBe(false);
    expect(cv.getParams().binary).toBe(true);
    cv.fitText(["Aa aa"]);
    expect(cv.getFeatureNames()).toEqual(["Aa", "aa"]);
    cv.setParams({ maxFeatures: 1 });
    expect(() => cv.getFeatureNames()).toThrow(NotFittedError);
    expect(() => cv.setParams({ nope: 1 })).toThrow(/nope/);
    expect(() => cv.setParams({ minDf: -3 })).toThrow(/minDf/);
    expect(cv.getParams().maxFeatures).toBe(1);
  });

  it("getParams round-trips through the constructor", () => {
    const original = new CountVectorizer({
      maxFeatures: 7,
      minDf: 2,
      stopWords: ["xx"],
      ngramRange: [1, 2],
    });
    const copy = new CountVectorizer(
      original.getParams() as ConstructorParameters<typeof CountVectorizer>[0]
    );
    expect(copy.getParams()).toEqual(original.getParams());
  });

  it("returns a tensor of the default dtype with the counted values", () => {
    const cv = new CountVectorizer();
    const X = cv.fitTransformText(["aa bb aa", "bb cc"]);
    expect(X.toArray()).toEqual([
      [2, 1, 0],
      [0, 1, 1],
    ]);
    expect(X.dtype).toBe("float64");
  });
});

describe("TfidfVectorizer", () => {
  it("matches scikit-learn with default options", () => {
    const tfidf = new TfidfVectorizer();
    const X = tfidf.fitTransformText(["hello world hello", "hello deepbox"]);
    expect(tfidf.getFeatureNames()).toEqual(["deepbox", "hello", "world"]);
    const rowsOut = X.toArray() as number[][];
    const expected = [
      [0.0, 0.8181802073667197, 0.5749618667993135],
      [0.8148024746671689, 0.5797386715376657, 0.0],
    ];
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 3; j++) {
        expect(rowsOut[i]?.[j]).toBeCloseTo(expected[i]?.[j] as number, 12);
      }
    }
    const idf = Array.from(tfidf.idf);
    expect(idf[0]).toBeCloseTo(1.4054651081081644, 12);
    expect(idf[1]).toBeCloseTo(1.0, 12);
  });

  it("matches scikit-learn with l1 norm, sublinear tf and unsmoothed idf", () => {
    const tfidf = new TfidfVectorizer({ norm: "l1", sublinearTf: true, smoothIdf: false });
    const X = tfidf.fitTransformText([
      "hello world hello",
      "hello deepbox",
      "deepbox world world world",
    ]);
    const expected = [
      [0.0, 0.6286872075843678, 0.37131279241563214],
      [0.5, 0.5, 0.0],
      [0.32272511267611165, 0.0, 0.6772748873238884],
    ];
    const got = X.toArray() as number[][];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(got[i]?.[j]).toBeCloseTo(expected[i]?.[j] as number, 12);
      }
    }
  });

  it("useIdf=false leaves plain (normalized) term frequencies", () => {
    const tfidf = new TfidfVectorizer({ useIdf: false, norm: undefined });
    const X = tfidf.fitTransformText(["aa aa bb"]);
    expect(X.toArray()).toEqual([[2, 1]]);
    expect(Array.from(tfidf.idf)).toEqual([1, 1]);
  });

  it("fitTransformText equals fitText followed by transformText", () => {
    const docs = ["red green blue", "green blue", "blue yellow blue"];
    const a = new TfidfVectorizer().fitTransformText(docs).toArray();
    const b = new TfidfVectorizer().fitText(docs).transformText(docs).toArray();
    expect(a).toEqual(b);
  });

  it("reports TfidfVectorizer (not CountVectorizer) when used before fitting", () => {
    const tfidf = new TfidfVectorizer();
    expect(() => tfidf.vocabulary).toThrow(/TfidfVectorizer/);
    expect(() => tfidf.getFeatureNames()).toThrow(/TfidfVectorizer/);
    expect(() => tfidf.transformText(["aa"])).toThrow(NotFittedError);
  });

  it("validates norm instead of treating unknown values as l1", () => {
    expect(() => new TfidfVectorizer({ norm: "l3" as "l1" })).toThrow(/norm/);
    expect(() => new TfidfVectorizer({ sublinearTf: "yes" as unknown as boolean })).toThrow(
      /sublinearTf/
    );
  });

  it("setParams updates tf-idf options, forwards vectorizer options and is atomic", () => {
    const tfidf = new TfidfVectorizer().fitText(["aa bb", "aa cc"]);
    tfidf.setParams({ norm: "l1" });
    expect(tfidf.getParams().norm).toBe("l1");
    expect(() => tfidf.getFeatureNames()).not.toThrow();
    tfidf.setParams({ norm: undefined });
    expect(tfidf.getParams().norm).toBeUndefined();

    tfidf.setParams({ smoothIdf: false });
    expect(() => tfidf.idf).toThrow(NotFittedError);
    tfidf.setParams({ stopWords: ["aa"] });
    expect(tfidf.getParams().stopWords).toEqual(["aa"]);

    expect(() => tfidf.setParams({ norm: "l9", smoothIdf: true })).toThrow(/norm/);
    expect(tfidf.getParams().smoothIdf).toBe(false);
    expect(() => tfidf.setParams({ bogus: 1 })).toThrow(/bogus/);
  });
});

describe("HashingVectorizer", () => {
  it("fitText is a no-op that validates documents; getParams exposes every option", () => {
    const hv = new HashingVectorizer({ nFeatures: 64, stopWords: ["the"] });
    expect(hv.fitText(["the cat"])).toBe(hv);
    expect(() => hv.fitText([5 as unknown as string])).toThrow(InvalidParameterError);
    const params = hv.getParams();
    expect(params.stopWords).toEqual(["the"]);
    expect(params.tokenPattern).toBeInstanceOf(RegExp);
    expect(params.norm).toBe("l2");
  });

  it("setParams takes effect and rejects invalid values", () => {
    const hv = new HashingVectorizer({ nFeatures: 32 });
    hv.setParams({ nFeatures: 16, norm: undefined });
    const X = hv.transformText(["aa bb"]);
    expect(X.shape).toEqual([1, 16]);
    expect(hv.getParams().norm).toBeUndefined();
    expect(() => hv.setParams({ nFeatures: 0 })).toThrow(/nFeatures/);
    expect(() => hv.setParams({ norm: "max" })).toThrow(/norm/);
    expect(() => hv.setParams({ unknownOption: true })).toThrow(/unknownOption/);
    expect(hv.getParams().nFeatures).toBe(16);
  });

  it("tokenizes Unicode text and honours the english stop word list", () => {
    const hv = new HashingVectorizer({ nFeatures: 256, norm: undefined, alternateSign: false });
    const withAccent = (hv.transformText(["café"]).toArray() as number[][])[0] as number[];
    expect(withAccent.reduce((a, b) => a + b, 0)).toBe(1);
    const stop = new HashingVectorizer({ nFeatures: 256, norm: undefined, stopWords: "english" });
    const all = (stop.transformText(["the and of"]).toArray() as number[][])[0] as number[];
    expect(all.every((v) => v === 0)).toBe(true);
  });

  it("returns float64 data when the default dtype is float64", () => {
    const hv = new HashingVectorizer({ nFeatures: 8 });
    expect(hv.transformText(["aa bb"]).dtype).toBe("float64");
  });
});
