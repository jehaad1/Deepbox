import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import {
  MaxAbsScaler,
  MinMaxScaler,
  Normalizer,
  PowerTransformer,
  QuantileTransformer,
  RobustScaler,
  StandardScaler,
} from "../src/preprocess/scalers";

describe("preprocess scalers extra branches", () => {
  it("StandardScaler handles errors and no-op options", () => {
    const scaler = new StandardScaler({ withMean: false, withStd: false });
    expect(() => scaler.transform(tensor([[1]]))).toThrow(/fitted/i);
    expect(() => scaler.fit(tensor([]))).toThrow(/at least one sample/i);
    const X = tensor([
      [1, 2],
      [3, 4],
    ]);
    scaler.fit(X);
    const out = scaler.transform(X);
    expect(out.toArray()).toEqual(X.toArray());
  });

  it("MinMaxScaler validates featureRange and fit state", () => {
    expect(() => new MinMaxScaler({ featureRange: [1, 1] })).toThrow(/featureRange/);
    const scaler = new MinMaxScaler();
    expect(() => scaler.transform(tensor([[1]]))).toThrow(/fitted/i);
    expect(() => scaler.fit(tensor([]))).toThrow(/at least one sample/i);
    scaler.fit(
      tensor([
        [1, 1],
        [1, 1],
      ])
    );
    const out = scaler.transform(tensor([[1, 1]]));
    expect(out.shape).toEqual([1, 2]);
  });

  it("MaxAbsScaler handles zero scale and errors", () => {
    const scaler = new MaxAbsScaler();
    expect(() => scaler.transform(tensor([[1]]))).toThrow(/fitted/i);
    expect(() => scaler.fit(tensor([]))).toThrow(/at least one sample/i);
    scaler.fit(
      tensor([
        [0, 0],
        [0, 0],
      ])
    );
    const out = scaler.transform(tensor([[0, 0]]));
    expect(out.toArray()).toEqual([[0, 0]]);
    const inv = scaler.inverseTransform(out);
    expect(inv.toArray()).toEqual([[0, 0]]);
  });

  it("RobustScaler validates options and skips centering/scaling", () => {
    expect(() => new RobustScaler({ quantileRange: [75, 25] })).toThrow(/quantileRange/);
    const scaler = new RobustScaler({ withCentering: false, withScaling: false });
    expect(() => scaler.transform(tensor([[1]]))).toThrow(/fitted/i);
    expect(() => scaler.fit(tensor([]))).toThrow(/at least one sample/i);
    const X = tensor([
      [1, 2],
      [3, 4],
    ]);
    scaler.fit(X);
    const out = scaler.transform(X);
    expect(out.toArray()).toEqual(X.toArray());
  });

  it("Normalizer and QuantileTransformer validate options", () => {
    expect(() => new Normalizer({ norm: "bad" as never })).toThrow(/norm/);
    expect(() => new QuantileTransformer({ nQuantiles: 1 })).toThrow(/nQuantiles/);
    expect(() => new QuantileTransformer({ outputDistribution: "bad" as never })).toThrow(
      /outputDistribution/
    );
    expect(() => new QuantileTransformer({ subsample: 1 as never })).toThrow(/subsample/);

    const qt = new QuantileTransformer({ nQuantiles: 10 });
    qt.fit(tensor([[1], [2], [3]]));
    const out = qt.transform(tensor([[1], [2], [3]]));
    expect(out.shape).toEqual([3, 1]);
  });

  it("PowerTransformer validates method and fit state", () => {
    expect(() => new PowerTransformer({ method: "bad" as never })).toThrow(/method/);
    const pt = new PowerTransformer({ method: "yeo-johnson", standardize: false });
    expect(() => pt.transform(tensor([[1]]))).toThrow(/fitted/i);
    expect(() => pt.fit(tensor([]))).toThrow(/at least one sample/i);
    pt.fit(tensor([[1], [2], [3]]));
    const out = pt.transform(tensor([[1], [2], [3]]));
    expect(out.shape).toEqual([3, 1]);
  });
});
