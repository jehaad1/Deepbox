import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import {
  GroupKFold,
  KFold,
  LeaveOneOut,
  LeavePOut,
  StratifiedKFold,
  trainTestSplit,
} from "../src/preprocess/split";
import { toNumberArray } from "./preprocess-test-helpers";

describe("deepbox/preprocess - Split Extra", () => {
  it("trainTestSplit splits deterministically with randomState", () => {
    const X = tensor([[1], [2], [3], [4]]);
    const y = tensor([0, 0, 1, 1]);
    const [XTrain1, XTest1, yTrain1, yTest1] = trainTestSplit(X, y, {
      testSize: 0.5,
      randomState: 42,
      shuffle: true,
    });
    const [XTrain2, XTest2, yTrain2, yTest2] = trainTestSplit(X, y, {
      testSize: 0.5,
      randomState: 42,
      shuffle: true,
    });
    expect(XTrain1.toArray()).toEqual(XTrain2.toArray());
    expect(XTest1.toArray()).toEqual(XTest2.toArray());
    expect(yTrain1.toArray()).toEqual(yTrain2.toArray());
    expect(yTest1.toArray()).toEqual(yTest2.toArray());
  });

  it("trainTestSplit throws on empty input", () => {
    const X = tensor([]);
    expect(() => trainTestSplit(X)).toThrow();
  });

  it("trainTestSplit supports trainSize (absolute)", () => {
    const X = tensor([[1], [2], [3], [4], [5]]);
    const [XTrain, XTest] = trainTestSplit(X, undefined, { trainSize: 3 });
    expect(XTrain.shape[0]).toBe(3);
    expect(XTest.shape[0]).toBe(2);
  });

  it("trainTestSplit stratifies by labels", () => {
    const X = tensor([[1], [2], [3], [4], [5], [6]]);
    const y = tensor([0, 0, 0, 1, 1, 1]);
    const [_, __, yTrain, yTest] = trainTestSplit(X, y, {
      testSize: 0.5,
      stratify: y,
      randomState: 7,
      shuffle: true,
    });
    const trainLabels = toNumberArray(yTrain, "yTrain");
    const testLabels = toNumberArray(yTest, "yTest");
    expect(trainLabels.filter((v) => v === 0).length).toBe(1);
    expect(trainLabels.filter((v) => v === 1).length).toBe(2);
    expect(testLabels.filter((v) => v === 0).length).toBe(2);
    expect(testLabels.filter((v) => v === 1).length).toBe(1);
  });

  it("KFold returns correct number of splits", () => {
    const X = tensor([[1], [2], [3], [4], [5]]);
    const kf = new KFold({ nSplits: 5 });
    const splits = kf.split(X);
    expect(splits.length).toBe(5);
    expect(kf.getNSplits()).toBe(5);
  });

  it("StratifiedKFold preserves class distribution", () => {
    const X = tensor([[1], [2], [3], [4]]);
    const y = tensor([0, 0, 1, 1]);
    const skf = new StratifiedKFold({ nSplits: 2 });
    const splits = skf.split(X, y);
    expect(splits.length).toBe(2);
    for (const [, testIdx] of splits) {
      const labels = testIdx.map((i) => Number(y.data[y.offset + i]));
      expect(labels).toContain(0);
      expect(labels).toContain(1);
    }
  });

  it("GroupKFold keeps groups separate", () => {
    const X = tensor([[1], [2], [3], [4]]);
    const groups = tensor([0, 0, 1, 1]);
    const gkf = new GroupKFold({ nSplits: 2 });
    const splits = gkf.split(X, undefined, groups);
    expect(splits.length).toBe(2);
    for (const [trainIdx, testIdx] of splits) {
      const trainGroups = new Set(trainIdx.map((i) => Number(groups.data[groups.offset + i])));
      const testGroups = new Set(testIdx.map((i) => Number(groups.data[groups.offset + i])));
      for (const g of trainGroups) {
        expect(testGroups.has(g)).toBe(false);
      }
    }
  });

  it("GroupKFold supports string groups and balances by sample count", () => {
    const X = tensor([[1], [2], [3], [4], [5]]);
    const groups = tensor(["A", "A", "A", "B", "C"]);
    const gkf = new GroupKFold({ nSplits: 2 });
    const splits = gkf.split(X, undefined, groups);
    const sizes = splits.map(([, testIdx]) => testIdx.length).sort((a, b) => a - b);
    expect(sizes).toEqual([2, 3]);
  });

  it("LeaveOneOut and LeavePOut return expected splits", () => {
    const X = tensor([[1], [2], [3]]);
    const loo = new LeaveOneOut();
    const splits = loo.split(X);
    expect(splits.length).toBe(3);
    expect(loo.getNSplits(X)).toBe(3);

    const lpo = new LeavePOut(2);
    const splitsP = lpo.split(X);
    expect(splitsP.length).toBe(3);
  });
});
