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

describe("preprocess split branches", () => {
  it("validates trainTestSplit sizes and labels", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);

    expect(() => trainTestSplit(tensor([]))).toThrow(/shape|2D/i);
    expect(() => trainTestSplit(X, tensor([1, 2]))).toThrow(/same number of samples/i);
    expect(() => trainTestSplit(X, undefined, { stratify: tensor([0, 1]) })).toThrow(/stratify/i);

    expect(() => trainTestSplit(X, undefined, { testSize: 2, trainSize: 2 })).toThrow(/exceed/i);
    const [XsmallTrain, XsmallTest] = trainTestSplit(X, undefined, {
      testSize: 0.01,
      shuffle: false,
    });
    expect(XsmallTrain.shape[0]).toBe(2);
    expect(XsmallTest.shape[0]).toBe(1);

    const [Xtr, Xte] = trainTestSplit(X, undefined, {
      testSize: 1,
      shuffle: false,
    });
    expect(Xtr.shape[0]).toBe(2);
    expect(Xte.shape[0]).toBe(1);

    const y = tensor([0, 1, 0]);
    const [Xs, Xt, ys, yt] = trainTestSplit(X, y, {
      testSize: 0.34,
      stratify: y,
      randomState: 1,
    });
    expect(Xs.shape[0] + Xt.shape[0]).toBe(3);
    expect(ys.shape[0] + yt.shape[0]).toBe(3);
  });

  it("validates stratified train/test minimum class counts", () => {
    const X = tensor([[1], [2], [3], [4]]);
    const y = tensor([0, 0, 1, 1]);
    expect(() => trainTestSplit(X, y, { testSize: 0.25, stratify: y, shuffle: true })).toThrow(
      /testSize.*classes/i
    );

    const XSmall = tensor([[1], [2], [3]]);
    const ySmall = tensor([0, 1, 1]);
    expect(() =>
      trainTestSplit(XSmall, ySmall, { testSize: 0.5, stratify: ySmall, shuffle: true })
    ).toThrow(/at least 2 samples/i);
  });

  it("validates KFold and StratifiedKFold", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ]);
    const y = tensor([0, 0, 1, 1]);

    expect(() => new KFold({ nSplits: 1 }).split(X)).toThrow(/at least 2/i);
    expect(() => new KFold({ nSplits: 5 }).split(X)).toThrow(/greater than number of samples/i);

    const kf = new KFold({ nSplits: 2, shuffle: true, randomState: 0 });
    const splits = Array.from(kf.split(X));
    expect(splits.length).toBe(2);

    expect(() => new StratifiedKFold({ nSplits: 3 }).split(X, y)).toThrow(/at least nSplits/i);
    expect(() => new StratifiedKFold({ nSplits: 2 }).split(X, tensor([0, 1]))).toThrow(
      /same number/i
    );
  });

  it("validates GroupKFold and leave-one-out variants", () => {
    const X = tensor([[1], [2], [3], [4]]);
    const groups = tensor([0, 0, 1, 1]);

    expect(() => new GroupKFold({ nSplits: 1 }).split(X, undefined, groups)).toThrow(/at least 2/i);
    expect(() => new GroupKFold({ nSplits: 3 }).split(X, undefined, groups)).toThrow(
      /Number of groups/i
    );
    expect(() => new GroupKFold({ nSplits: 2 }).split(X, undefined, tensor([0, 1, 2]))).toThrow(
      /same number/i
    );

    const gkf = new GroupKFold({ nSplits: 2 });
    const gSplits = Array.from(gkf.split(X, undefined, groups));
    expect(gSplits.length).toBe(2);

    const loo = new LeaveOneOut();
    const looSplits = Array.from(loo.split(X));
    expect(looSplits.length).toBe(4);

    expect(() => new LeavePOut(0)).toThrow(/positive integer/i);
    expect(() => new LeavePOut(5).split(X)).toThrow(/greater than number of samples/i);
  });
});
