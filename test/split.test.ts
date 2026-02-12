import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import {
  GroupKFold,
  KFold,
  LeaveOneOut,
  LeavePOut,
  StratifiedKFold,
  trainTestSplit,
} from "../src/preprocess";

describe("Train-Test Split", () => {
  it("should split data into train and test sets", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ]);
    const y = tensor([0, 1, 0, 1]);
    const [XTrain, XTest, yTrain, yTest] = trainTestSplit(X, y, {
      testSize: 0.25,
    });
    expect(XTrain.shape[0] + XTest.shape[0]).toBe(4);
    expect(yTrain.shape[0] + yTest.shape[0]).toBe(4);
    expect(XTrain.shape[1]).toBe(2);
    expect(XTest.shape[1]).toBe(2);
  });
});

describe("KFold", () => {
  it("should generate correct number of splits", () => {
    const kfold = new KFold({ nSplits: 3 });
    const X = tensor([[1], [2], [3], [4], [5], [6]]);
    const splits = kfold.split(X);
    expect(splits.length).toBe(3);
    expect(kfold.getNSplits()).toBe(3);
  });

  it("should partition all samples across folds", () => {
    const kfold = new KFold({ nSplits: 3 });
    const X = tensor([[1], [2], [3], [4], [5], [6]]);
    const splits = kfold.split(X);
    for (const [trainIdx, testIdx] of splits) {
      expect(trainIdx.length + testIdx.length).toBe(6);
    }
  });
});

describe("StratifiedKFold", () => {
  it("should generate correct number of stratified splits", () => {
    const skfold = new StratifiedKFold({ nSplits: 2 });
    const X = tensor([[1], [2], [3], [4]]);
    const y = tensor([0, 0, 1, 1]);
    const splits = skfold.split(X, y);
    expect(splits.length).toBe(2);
    expect(skfold.getNSplits()).toBe(2);
  });
});

describe("GroupKFold", () => {
  it("should report correct number of splits", () => {
    const gkfold = new GroupKFold({ nSplits: 3 });
    expect(gkfold.getNSplits()).toBe(3);
  });
});

describe("LeaveOneOut", () => {
  it("should generate n splits for n samples", () => {
    const loo = new LeaveOneOut();
    const X = tensor([[1], [2], [3]]);
    expect(loo.getNSplits(X)).toBe(3);
    const splits = loo.split(X);
    expect(splits.length).toBe(3);
    for (const [trainIdx, testIdx] of splits) {
      expect(testIdx.length).toBe(1);
      expect(trainIdx.length).toBe(2);
    }
  });
});

describe("LeavePOut", () => {
  it("should generate correct number of splits", () => {
    const lpo = new LeavePOut(2);
    const X = tensor([[1], [2], [3], [4]]);
    // C(4,2) = 6 splits
    expect(lpo.getNSplits(X)).toBe(6);
  });
});
