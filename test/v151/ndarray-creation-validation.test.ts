import { describe, expect, it } from "vitest";
import { InvalidParameterError } from "../../src/core";
import { logspace } from "../../src/ndarray";

// Reference: numpy.logspace(0, 3, 4) -> [1, 10, 100, 1000]; numpy.logspace(0, 3, 4, base=2)
// -> [1, 2, 4, 8]. numpy takes (num, endpoint, base); this library takes (num, base, endpoint).

describe("logspace base validation", () => {
  it("accepts a positive finite base in the documented position", () => {
    expect(logspace(0, 3, 4, 2).toArray()).toEqual([1, 2, 4, 8]);
    expect(logspace(0, 3, 4, 10, false, { dtype: "float64" }).shape).toEqual([4]);
  });

  it("rejects an endpoint passed where the base goes", () => {
    expect(() => logspace(0, 3, 4, true as never, 2 as never)).toThrow(InvalidParameterError);
  });

  it("rejects a base that is not finite and positive", () => {
    for (const base of [0, -2, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(() => logspace(0, 3, 4, base)).toThrow(InvalidParameterError);
    }
  });
});
