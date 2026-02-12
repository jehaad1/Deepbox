import { describe, expect, it } from "vitest";
import { CSRMatrix, reshape, tensor, zeros } from "../src/ndarray";
import {
  LabelBinarizer,
  LabelEncoder,
  MaxAbsScaler,
  MinMaxScaler,
  MultiLabelBinarizer,
  Normalizer,
  OneHotEncoder,
  OrdinalEncoder,
  PowerTransformer,
  QuantileTransformer,
  RobustScaler,
  StandardScaler,
} from "../src/preprocess";
import { toNum2D, toStr2D } from "./_helpers";

describe("preprocess branch coverage", () => {
  it("LabelEncoder error branches", () => {
    const enc = new LabelEncoder();
    expect(() => enc.transform(tensor([1]))).toThrow(/fitted/i);
    expect(() => enc.inverseTransform(tensor([0]))).toThrow(/fitted/i);
    expect(() => enc.fit(tensor([]))).toThrow(/empty/i);
    expect(() => enc.fit(tensor([[1, 2]]))).toThrow(/1D/i);

    enc.fit(tensor(["a", "b"]));
    expect(() => enc.transform(tensor(["c"]))).toThrow(/unknown label/i);
    expect(() => enc.inverseTransform(tensor([1.5]))).toThrow(/invalid label index/i);
    expect(() => enc.inverseTransform(tensor(["a"]))).toThrow(/numeric/i);
  });

  it("OneHotEncoder constructor validation", () => {
    expect(() => new OneHotEncoder({ sparse: true, sparseOutput: false })).toThrow(/must match/i);
    expect(() => new OneHotEncoder({ handleUnknown: "bad" as never })).toThrow(/handleUnknown/i);
    expect(() => new OneHotEncoder({ drop: "bad" as never })).toThrow(/drop/i);
    expect(() => new OneHotEncoder({ sparse: "yes" as never })).toThrow(/sparse/i);
  });

  it("OneHotEncoder fit/transform branches", () => {
    const X = tensor([
      ["red", "S"],
      ["blue", "M"],
    ]);

    expect(() => new OneHotEncoder().fit(tensor([[]]))).toThrow(/empty/i);
    expect(() => new OneHotEncoder({ categories: [["a"]] }).fit(X)).toThrow(/categories length/i);
    expect(() =>
      new OneHotEncoder({
        categories: [["red", "blue"], undefined as unknown as string[]],
      }).fit(X)
    ).toThrow(/missing categories/i);
    expect(() =>
      new OneHotEncoder({
        categories: [["red", "blue"], 5 as unknown as string[]],
      }).fit(X)
    ).toThrow(/category arrays/i);
    expect(() =>
      new OneHotEncoder({
        categories: [["a", "a"], ["b"]],
      }).fit(X)
    ).toThrow(/duplicate/i);
    expect(() =>
      new OneHotEncoder({
        categories: [["a", 1], ["b"]],
      }).fit(X)
    ).toThrow(/mixed category types/i);
    expect(() =>
      new OneHotEncoder({
        categories: [["red"], ["S"]],
      }).fit(tensor([["green", "S"]]))
    ).toThrow(/unknown category/i);

    const notFitted = new OneHotEncoder();
    expect(() => notFitted.transform(X)).toThrow(/fitted/i);

    const enc = new OneHotEncoder();
    enc.fit(X);
    expect(() => enc.transform(tensor([["red"]]))).toThrow(/feature count/i);
    expect(() => enc.transform(tensor([["green", "S"]]))).toThrow(/unknown category/i);

    const ignore = new OneHotEncoder({
      handleUnknown: "ignore",
      drop: "first",
    });
    ignore.fit(X);
    const ignored = ignore.transform(tensor([["green", "S"]]));
    expect(ignored.shape).toEqual([1, 2]);

    const empty = reshape(tensor([]), [0, 2]);
    const emptyOut = ignore.transform(empty);
    expect(emptyOut.shape).toEqual([0, 2]);

    const sparseEnc = new OneHotEncoder({ sparse: true, drop: "if_binary" });
    sparseEnc.fit(X);
    const sparseOut = sparseEnc.transform(X);
    expect(sparseOut).toBeInstanceOf(CSRMatrix);
  });

  it("OneHotEncoder inverseTransform branches", () => {
    const X = tensor([
      ["red", "S"],
      ["blue", "M"],
    ]);

    const noDrop = new OneHotEncoder();
    noDrop.fit(X);
    const zerosNoDrop = zeros([1, 4]);
    expect(() => noDrop.inverseTransform(zerosNoDrop)).toThrow(/all zeros/i);
    expect(() => noDrop.inverseTransform(zeros([1, 3]))).toThrow(/column count/i);

    const ignoreNoDrop = new OneHotEncoder({ handleUnknown: "ignore" });
    ignoreNoDrop.fit(X);
    expect(() => ignoreNoDrop.inverseTransform(zerosNoDrop)).toThrow(/cannot inverse-transform/i);

    const dropFirst = new OneHotEncoder({ drop: "first" });
    dropFirst.fit(X);
    const zerosDrop = zeros([1, 2]);
    const invDrop = toStr2D(dropFirst.inverseTransform(zerosDrop).toArray());
    expect(invDrop[0]?.[0]).toBe("blue");

    const single = tensor([["only"]]);
    const dropSingle = new OneHotEncoder({ drop: "first" });
    dropSingle.fit(single);
    const invSingle = toStr2D(dropSingle.inverseTransform(zeros([1, 0])).toArray());
    expect(invSingle[0]?.[0]).toBe("only");
  });

  it("OrdinalEncoder branches", () => {
    expect(() => new OrdinalEncoder({ handleUnknown: "bad" as never })).toThrow(/handleUnknown/i);
    expect(() => new OrdinalEncoder({ unknownValue: 1.5 })).toThrow(/unknownValue/i);

    const X = tensor([
      ["low", "red"],
      ["high", "blue"],
    ]);

    expect(() => new OrdinalEncoder().fit(reshape(tensor([]), [0, 2]))).toThrow(/empty/i);
    expect(() =>
      new OrdinalEncoder({ categories: [["a"]] as unknown as string[][] }).fit(X)
    ).toThrow(/categories length/i);
    expect(() =>
      new OrdinalEncoder({ categories: [["low"], ["red"]] }).fit(tensor([["medium", "red"]]))
    ).toThrow(/unknown category/i);
    expect(() =>
      new OrdinalEncoder({
        handleUnknown: "useEncodedValue",
        unknownValue: 0,
      }).fit(X)
    ).toThrow(/unknownValue/i);

    const notFitted = new OrdinalEncoder();
    expect(() => notFitted.transform(X)).toThrow(/fitted/i);

    const err = new OrdinalEncoder();
    err.fit(X);
    expect(() => err.transform(tensor([["missing", "red"]]))).toThrow(/unknown category/i);

    const useUnknown = new OrdinalEncoder({
      handleUnknown: "useEncodedValue",
      unknownValue: -1,
    });
    useUnknown.fit(X);
    const unknownOut = toNum2D(useUnknown.transform(tensor([["missing", "red"]])).toArray());
    expect(unknownOut[0]?.[0]).toBe(-1);

    const emptyOut = useUnknown.transform(reshape(tensor([]), [0, 2]));
    expect(emptyOut.shape).toEqual([0, 2]);

    expect(() => useUnknown.inverseTransform(tensor([[0, -1]]))).toThrow(/unknown encoded value/i);
    expect(() => useUnknown.inverseTransform(tensor([[5, 0]]))).toThrow(/invalid encoded value/i);
  });

  it("LabelBinarizer branches", () => {
    expect(() => new LabelBinarizer({ posLabel: 0, negLabel: 1 })).toThrow(/posLabel/i);
    expect(() => new LabelBinarizer({ sparse: true, sparseOutput: false })).toThrow(/must match/i);
    expect(() => new LabelBinarizer({ sparse: true, negLabel: -1 })).toThrow(/negLabel/i);

    const bin = new LabelBinarizer();
    expect(() => bin.transform(tensor([1]))).toThrow(/fitted/i);
    expect(() => bin.fit(tensor([]))).toThrow(/empty/i);

    bin.fit(tensor(["cat", "dog"]));
    expect(() => bin.transform(tensor(["mouse"]))).toThrow(/unknown label/i);

    const sparseBin = new LabelBinarizer({ sparse: true });
    sparseBin.fit(tensor(["yes", "no"]));
    const sparseOut = sparseBin.transform(tensor(["yes"]));
    expect(sparseOut).toBeInstanceOf(CSRMatrix);
    expect(() => sparseBin.inverseTransform(sparseOut)).not.toThrow();

    expect(() => sparseBin.inverseTransform(zeros([1, 3]))).toThrow(/column count/i);
  });

  it("MultiLabelBinarizer branches", () => {
    expect(() => new MultiLabelBinarizer({ sparse: true, sparseOutput: false })).toThrow(
      /must match/i
    );
    expect(() => new MultiLabelBinarizer({ classes: ["a", "a"] })).toThrow(/duplicate/i);

    const mlb = new MultiLabelBinarizer();
    expect(() => mlb.fit([])).toThrow(/empty/i);
    expect(() => mlb.fit(["bad"] as unknown as string[][])).toThrow(/label arrays/i);
    expect(() => mlb.fit([[{} as unknown as string]])).toThrow(/strings, numbers, or bigints/i);

    const withClasses = new MultiLabelBinarizer({ classes: ["x", "y"] });
    expect(() => withClasses.fit([["x", "z"]])).toThrow(/unknown label/i);

    expect(() => mlb.transform([["x"]])).toThrow(/fitted/i);

    const mlbSparse = new MultiLabelBinarizer({ sparse: true });
    mlbSparse.fit([["a", "b"], ["b"]]);
    const sparseOut = mlbSparse.transform([["a"]]);
    expect(sparseOut).toBeInstanceOf(CSRMatrix);

    expect(() => mlbSparse.transform([["c"]])).toThrow(/unknown label/i);
    expect(() => mlbSparse.inverseTransform(sparseOut as CSRMatrix)).not.toThrow();
    expect(() => mlbSparse.inverseTransform(zeros([1, 3]))).toThrow(/column count/i);
  });

  it("Scaler validation branches", () => {
    expect(() => new StandardScaler({ copy: "yes" as never })).toThrow(/copy/i);
    const scaler = new StandardScaler();
    expect(() => scaler.transform(tensor([[1, 2]]))).toThrow(/fitted/i);
    expect(() => scaler.fit(tensor([]))).toThrow(/at least one sample/i);
    expect(() => scaler.fit(tensor([1, 2]))).toThrow(/2D/i);
    expect(() => scaler.fit(tensor([["a"]]))).toThrow(/numeric/i);
    expect(() => scaler.fit(tensor([[1, Number.NaN]]))).toThrow(/NaN|Infinity/i);

    expect(() => new MinMaxScaler({ featureRange: [1, 0] })).toThrow(/featureRange/i);
    const minmax = new MinMaxScaler({ clip: true });
    expect(() => minmax.transform(tensor([[1, 2]]))).toThrow(/fitted/i);

    expect(() => new RobustScaler({ quantileRange: [90, 10] })).toThrow(/quantileRange/i);
    expect(() => new RobustScaler({ unitVariance: "yes" as never })).toThrow(/unitVariance/i);

    expect(() => new Normalizer({ norm: "bad" as never })).toThrow(/norm/i);

    expect(() => new QuantileTransformer({ nQuantiles: 1 })).toThrow(/nQuantiles/i);
    expect(() => new QuantileTransformer({ outputDistribution: "bad" as never })).toThrow(
      /outputDistribution/i
    );
    expect(() => new QuantileTransformer({ subsample: 1 })).toThrow(/subsample/i);

    expect(() => new PowerTransformer({ method: "bad" as never })).toThrow(/method/i);
  });

  it("Scaler transform branches", () => {
    const X = tensor([
      [1, 1],
      [1, 1],
    ]);

    const scaler = new StandardScaler();
    const scaled = toNum2D(scaler.fitTransform(X).toArray());
    expect(scaled[0]?.[0]).toBeCloseTo(0);

    const minmax = new MinMaxScaler({ clip: true });
    const minmaxOut = toNum2D(
      minmax
        .fitTransform(
          tensor([
            [0, 5],
            [10, 15],
          ])
        )
        .toArray()
    );
    expect(minmaxOut[0]?.[0]).toBeGreaterThanOrEqual(0);

    const maxAbs = new MaxAbsScaler();
    maxAbs.fit(
      tensor([
        [-2, 4],
        [2, -8],
      ])
    );
    const maxAbsOut = toNum2D(maxAbs.transform(tensor([[2, -8]])).toArray());
    expect(maxAbsOut[0]?.[0]).toBeCloseTo(1);

    const robust = new RobustScaler({
      quantileRange: [25, 75],
      withCentering: false,
    });
    robust.fit(
      tensor([
        [1, 2],
        [2, 3],
        [3, 4],
      ])
    );
    const robustOut = toNum2D(robust.transform(tensor([[2, 3]])).toArray());
    expect(robustOut[0]?.[0]).toBeDefined();

    const normalizer = new Normalizer({ norm: "max" });
    const normed = toNum2D(
      normalizer
        .transform(
          tensor([
            [0, 0],
            [3, 4],
          ])
        )
        .toArray()
    );
    expect(normed[0]?.[0]).toBe(0);

    const quantile = new QuantileTransformer({
      outputDistribution: "normal",
      subsample: 2,
      randomState: 42,
    });
    quantile.fit(
      tensor([
        [1, 2],
        [2, 3],
        [3, 4],
      ])
    );
    const qt = toNum2D(quantile.transform(tensor([[2, 3]])).toArray());
    expect(qt[0]?.[0]).toBeDefined();
    const qtInv = toNum2D(quantile.inverseTransform(tensor([[0, 0]])).toArray());
    expect(qtInv[0]?.[0]).toBeDefined();

    const power = new PowerTransformer({
      method: "box-cox",
      standardize: true,
    });
    expect(() => power.fit(tensor([[0], [1]])).transform(tensor([[1]]))).toThrow(/box-cox/i);

    const powerYJ = new PowerTransformer({
      method: "yeo-johnson",
      standardize: true,
    });
    powerYJ.fit(
      tensor([
        [-2, 0],
        [2, 3],
      ])
    );
    const pOut = toNum2D(powerYJ.transform(tensor([[0, 1]])).toArray());
    expect(pOut[0]?.[0]).toBeDefined();
    const pInv = toNum2D(powerYJ.inverseTransform(tensor([[0, 0]])).toArray());
    expect(pInv[0]?.[0]).toBeDefined();
  });
});
