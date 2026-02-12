import { describe, expect, it } from "vitest";
import { InvalidParameterError } from "../src/core/errors";
import { assertEstimator } from "../src/ml/base";
import { tensor } from "../src/ndarray";

describe("ml base runtime helper", () => {
  it("passes through estimator-like objects", () => {
    const mock = {
      fit: () => mock,
      getParams: () => ({}),
      setParams: () => mock,
      predict: (_x: unknown) => tensor([1]),
      score: () => 1,
    };
    expect(assertEstimator(mock)).toBe(mock);
  });

  it("throws typed errors for invalid estimator contracts", () => {
    expect(() =>
      assertEstimator({
        getParams: () => ({}),
        // @ts-expect-error - deliberately invalid return type for runtime validation test
        setParams: (_params: unknown) => null,
      })
    ).toThrow(InvalidParameterError);

    expect(() =>
      // @ts-expect-error - null should be rejected at runtime
      assertEstimator(null)
    ).toThrow(InvalidParameterError);
  });
});
