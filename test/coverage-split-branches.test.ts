import { describe, expect, it } from "vitest";
import { reshape, tensor } from "../src/ndarray";
import {
  GroupKFold,
  KFold,
  LeaveOneOut,
  LeavePOut,
  StratifiedKFold,
  trainTestSplit,
} from "../src/preprocess";

describe("preprocess split branch coverage", () => {
  it("trainTestSplit validation branches", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ]);

    expect(() => trainTestSplit(reshape(tensor([]), [0, 2]))).toThrow(/empty/i);
    expect(() => trainTestSplit(X, tensor([1, 2]))).toThrow(/same number of samples/i);
    expect(() => trainTestSplit(X, undefined, { testSize: 0 })).toThrow(/positive/i);
    expect(() => trainTestSplit(X, undefined, { trainSize: 1.5 })).toThrow(/integer/i);
    expect(() => trainTestSplit(X, undefined, { trainSize: 0.8, testSize: 0.5 })).toThrow(
      /sum to at most 1/i
    );
    expect(() => trainTestSplit(X, undefined, { trainSize: 10, testSize: 10 })).toThrow(
      /exceed number of samples/i
    );
    expect(() => trainTestSplit(X, undefined, { trainSize: 0.01, testSize: 0.2 })).toThrow(
      /at least 1 sample/i
    );

    expect(() => trainTestSplit(X, undefined, { stratify: tensor([[1], [2], [3], [4]]) })).toThrow(
      /1D/i
    );
    expect(() => trainTestSplit(X, undefined, { stratify: tensor([1, 2]) })).toThrow(
      /same number/i
    );
  });

  it("trainTestSplit stratify branches", () => {
    const X = tensor([
      [1, 1],
      [2, 2],
      [3, 3],
      [4, 4],
    ]);
    const stratify = tensor([0, 0, 1, 2]);

    expect(() => trainTestSplit(X, undefined, { stratify, testSize: 0.75 })).toThrow(
      /at least 2 samples per class/i
    );

    expect(() =>
      trainTestSplit(X, undefined, { stratify: tensor([0, 0, 1, 1]), trainSize: 1 })
    ).toThrow(/trainSize must be at least/i);

    expect(() =>
      trainTestSplit(X, undefined, { stratify: tensor([0, 0, 1, 1]), testSize: 1 })
    ).toThrow(/testSize must be at least/i);

    const [XTrain, XTest] = trainTestSplit(X, undefined, {
      stratify: tensor([0, 0, 1, 1]),
      testSize: 0.5,
      shuffle: false,
      randomState: 42,
    });
    expect(XTrain.shape[0] + XTest.shape[0]).toBe(4);
  });

  it("KFold branches", () => {
    const X = tensor([[1], [2], [3]]);
    expect(() => new KFold({ nSplits: 1 }).split(X)).toThrow(/at least 2/i);
    expect(() => new KFold({ nSplits: 5 }).split(X)).toThrow(/greater than number of samples/i);

    const folds = new KFold({ nSplits: 3, shuffle: true, randomState: 1 }).split(X);
    expect(folds.length).toBe(3);
  });

  it("StratifiedKFold branches", () => {
    const X = tensor([[1], [2], [3], [4]]);
    const y = tensor([0, 0, 1, 1]);

    expect(() => new StratifiedKFold({ nSplits: 1 }).split(X, y)).toThrow(/at least 2/i);
    expect(() => new StratifiedKFold({ nSplits: 10 }).split(X, y)).toThrow(/greater than number/i);
    expect(() =>
      new StratifiedKFold({ nSplits: 2 }).split(X, tensor([[1], [2], [3], [4]]))
    ).toThrow(/1D/i);
    expect(() => new StratifiedKFold({ nSplits: 2 }).split(X, tensor([0, 1]))).toThrow(
      /same number of samples/i
    );

    expect(() => new StratifiedKFold({ nSplits: 3 }).split(X, tensor([0, 0, 0, 1]))).toThrow(
      /at least nSplits samples/i
    );

    const folds = new StratifiedKFold({ nSplits: 2, shuffle: true, randomState: 3 }).split(X, y);
    expect(folds.length).toBe(2);
  });

  it("GroupKFold and leave-out branches", () => {
    const X = tensor([[1], [2], [3], [4]]);
    const groups = tensor([0, 0, 1, 1]);

    expect(() => new GroupKFold({ nSplits: 1 }).split(X, undefined, groups)).toThrow(/at least 2/i);
    expect(() => new GroupKFold({ nSplits: 3 }).split(X, undefined, groups)).toThrow(
      /at least nSplits/i
    );
    expect(() =>
      new GroupKFold({ nSplits: 2 }).split(X, undefined, tensor([[1], [2], [3], [4]]))
    ).toThrow(/1D/i);
    expect(() => new GroupKFold({ nSplits: 2 }).split(X, undefined, tensor([0, 1]))).toThrow(
      /same number of samples/i
    );

    const gfolds = new GroupKFold({ nSplits: 2 }).split(X, undefined, groups);
    expect(gfolds.length).toBe(2);

    const loo = new LeaveOneOut();
    expect(loo.getNSplits(X)).toBe(4);
    expect(loo.split(X).length).toBe(4);

    expect(() => new LeavePOut(0)).toThrow(/positive integer/i);
    expect(() => new LeavePOut(5).split(X)).toThrow(/greater than number of samples/i);
    const lpo = new LeavePOut(2).split(X);
    expect(lpo.length).toBeGreaterThan(0);
  });
});
