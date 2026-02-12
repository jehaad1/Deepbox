import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { QuantileTransformer } from "../src/preprocess/scalers";
import { StratifiedKFold, trainTestSplit } from "../src/preprocess/split";
import { toNumberMatrix, toStringArray, toStringMatrix } from "./preprocess-test-helpers";

describe("preprocess regressions", () => {
  it("trainTestSplit preserves string/categorical values for X and y", () => {
    const X = tensor([
      ["red", "S"],
      ["blue", "M"],
      ["green", "L"],
      ["red", "M"],
    ]);
    const y = tensor(["cat", "dog", "cat", "dog"]);

    const [XTrain, XTest, yTrain, yTest] = trainTestSplit(X, y, {
      testSize: 0.5,
      shuffle: true,
      randomState: 3,
    });

    expect(XTrain.dtype).toBe("string");
    expect(XTest.dtype).toBe("string");
    expect(yTrain.dtype).toBe("string");
    expect(yTest.dtype).toBe("string");

    const originalRows = toStringMatrix(X, "X")
      .map((row) => row.join("|"))
      .sort();
    const splitRows = [...toStringMatrix(XTrain, "XTrain"), ...toStringMatrix(XTest, "XTest")]
      .map((row) => row.join("|"))
      .sort();
    expect(splitRows).toEqual(originalRows);

    const originalLabels = toStringArray(y, "y").slice().sort();
    const splitLabels = [
      ...toStringArray(yTrain, "yTrain"),
      ...toStringArray(yTest, "yTest"),
    ].sort();
    expect(splitLabels).toEqual(originalLabels);
  });

  it("fractional split sizing keeps valid train/test counts for small n", () => {
    const X = tensor([[1], [2], [3]]);
    const [XTrain, XTest] = trainTestSplit(X, undefined, {
      testSize: 0.25,
      shuffle: false,
    });

    expect(XTrain.shape[0]).toBe(2);
    expect(XTest.shape[0]).toBe(1);
  });

  it("QuantileTransformer honors nQuantiles and spans max to 1", () => {
    const spanningData = tensor([[1], [2], [3], [4]]);
    const spanning = new QuantileTransformer({
      nQuantiles: 4,
      outputDistribution: "uniform",
    }).fitTransform(spanningData);
    const spanningArr = toNumberMatrix(spanning, "spanning");
    expect(spanningArr[0]?.[0]).toBeCloseTo(0, 8);
    expect(spanningArr[3]?.[0]).toBeCloseTo(1, 8);

    const skewed = tensor([[0], [1], [2], [100]]);
    const coarse = new QuantileTransformer({
      nQuantiles: 2,
      outputDistribution: "uniform",
    }).fitTransform(skewed);
    const fine = new QuantileTransformer({
      nQuantiles: 4,
      outputDistribution: "uniform",
    }).fitTransform(skewed);

    const coarseArr = toNumberMatrix(coarse, "coarse");
    const fineArr = toNumberMatrix(fine, "fine");
    expect(coarseArr[1]?.[0]).toBeCloseTo(0.01, 6);
    expect(fineArr[1]?.[0]).toBeCloseTo(1 / 3, 6);
    expect(coarseArr).not.toEqual(fineArr);
  });

  it("StratifiedKFold applies shuffle/randomState deterministically", () => {
    const X = tensor([[1], [2], [3], [4], [5], [6], [7], [8]]);
    const y = tensor([0, 0, 0, 0, 1, 1, 1, 1]);

    const splitsA = new StratifiedKFold({ nSplits: 4, shuffle: true, randomState: 11 }).split(X, y);
    const splitsB = new StratifiedKFold({ nSplits: 4, shuffle: true, randomState: 11 }).split(X, y);
    const splitsC = new StratifiedKFold({ nSplits: 4, shuffle: true, randomState: 13 }).split(X, y);

    expect(splitsA).toEqual(splitsB);
    expect(splitsA).not.toEqual(splitsC);
  });
});
