import { describe, expect, it } from "vitest";
import { DataValidationError } from "../../src/core";
import { MLPClassifier } from "../../src/ml";
import { tensor } from "../../src/ndarray";

const X = tensor(
  [
    [1, 2],
    [2, 1],
    [3, 4],
    [4, 3],
    [5, 6],
    [6, 5],
    [7, 8],
    [8, 7],
  ],
  { dtype: "float64" }
);

describe("ML-2: MLPClassifier label range", () => {
  it("rejects labels outside the int32 range instead of returning wrapped predictions", () => {
    const y = tensor([3e9, 5e9, 3e9, 5e9, 3e9, 5e9, 3e9, 5e9], { dtype: "float64" });
    expect(() => new MLPClassifier({ maxIter: 3, randomState: 0 }).fit(X, y)).toThrow(
      DataValidationError
    );
    expect(() => new MLPClassifier({ maxIter: 3, randomState: 0 }).fit(X, y)).toThrow(/int32/);
  });

  it("still accepts the int32 boundary labels", () => {
    const y = tensor(
      [
        -2147483648, 2147483647, -2147483648, 2147483647, -2147483648, 2147483647, -2147483648,
        2147483647,
      ],
      {
        dtype: "float64",
      }
    );
    const m = new MLPClassifier({ maxIter: 3, randomState: 0 }).fit(X, y);
    expect(Array.from(m.classes?.toArray() as number[])).toEqual([-2147483648, 2147483647]);
  });
});
