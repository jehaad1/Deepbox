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

describe("preprocess scaler error branches", () => {
  it("requires fitting before transform/inverse_transform", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(() => new StandardScaler().transform(X)).toThrow(/must be fitted/);
    expect(() => new StandardScaler().inverseTransform(X)).toThrow(/must be fitted/);
    expect(() => new MinMaxScaler().transform(X)).toThrow(/must be fitted/);
    expect(() => new MinMaxScaler().inverseTransform(X)).toThrow(/must be fitted/);
    expect(() => new MaxAbsScaler().transform(X)).toThrow(/must be fitted/);
    expect(() => new MaxAbsScaler().inverseTransform(X)).toThrow(/must be fitted/);
    expect(() => new RobustScaler().transform(X)).toThrow(/must be fitted/);
    expect(() => new RobustScaler().inverseTransform(X)).toThrow(/must be fitted/);
    expect(() => new QuantileTransformer().transform(X)).toThrow(/must be fitted/);
    expect(() => new QuantileTransformer().inverseTransform(X)).toThrow(/must be fitted/);
    expect(() => new PowerTransformer().transform(X)).toThrow(/must be fitted/);
    expect(() => new PowerTransformer().inverseTransform(X)).toThrow(/must be fitted/);
  });

  it("validates fitting data for scalers and transformers", () => {
    const empty = tensor([]);
    expect(() => new StandardScaler().fit(empty)).toThrow(/at least one sample/i);
    expect(() => new MinMaxScaler().fit(empty)).toThrow(/at least one sample/i);
    expect(() => new MaxAbsScaler().fit(empty)).toThrow(/at least one sample/i);
    expect(() => new RobustScaler().fit(empty)).toThrow(/at least one sample/i);
    expect(() => new QuantileTransformer().fit(empty)).toThrow(/at least one sample/i);
    expect(() => new PowerTransformer().fit(empty)).toThrow(/at least one sample/i);

    const pt = new PowerTransformer({ method: "box-cox" });
    pt.fit(
      tensor([
        [1, 2],
        [3, 4],
      ])
    );
    expect(() =>
      pt.transform(
        tensor([
          [1, 0],
          [2, 3],
        ])
      )
    ).toThrow(/strictly positive/i);
  });

  it("validates constructor options", () => {
    expect(() => new MinMaxScaler({ featureRange: [1, 1] })).toThrow(/featureRange/i);
    expect(() => new RobustScaler({ quantileRange: [80, 20] })).toThrow(/quantileRange/i);
    expect(() => new QuantileTransformer({ nQuantiles: 1 })).toThrow(/nQuantiles/i);
    expect(() => Reflect.construct(Normalizer, [{ norm: "l0" }])).toThrow(/norm/i);
    expect(() => Reflect.construct(PowerTransformer, [{ method: "invalid" }])).toThrow(/method/i);
  });

  it("rejects non-numeric tensors", () => {
    const X = tensor([["a"], ["b"]]);
    const scaler = new StandardScaler();
    expect(() => scaler.fit(X)).toThrow(/numeric/i);

    const pt = new PowerTransformer();
    pt.fit(tensor([[1], [2]]));
    expect(() => pt.transform(X)).toThrow(/numeric/i);
  });
});
