import { describe, expect, it } from "vitest";
import { catchWarnings } from "../../src/core";
import { tensor } from "../../src/ndarray";
import { kendalltau, pearsonr, spearmanr } from "../../src/stats";

// scipy 1.18: pearsonr and spearmanr return (nan, nan) for a constant input and emit a
// ConstantInputWarning ("An input array is constant; the correlation coefficient is not
// defined."). kendalltau returns (nan, nan) without a warning.

const CASES: [number[], number[]][] = [
  [
    [1, 1, 1],
    [2, 2, 2],
  ],
  [
    [1, 1, 1],
    [1, 2, 3],
  ],
  [
    [1, 2, 3],
    [5, 5, 5],
  ],
  [
    [1, 1],
    [2, 2],
  ],
  [
    [0.1, 0.1, 0.1],
    [1, 2, 4],
  ],
];

describe("correlation of constant input", () => {
  for (const [name, fn] of [
    ["pearsonr", pearsonr],
    ["spearmanr", spearmanr],
  ] as const) {
    describe(name, () => {
      for (const [x, y] of CASES) {
        it(`returns [NaN, NaN] and warns for ${JSON.stringify(x)} vs ${JSON.stringify(y)}`, () => {
          let result: [number, number] = [0, 0];
          const warnings = catchWarnings(() => {
            result = fn(tensor(x, { dtype: "float64" }), tensor(y, { dtype: "float64" }));
          });
          expect(result[0]).toBeNaN();
          expect(result[1]).toBeNaN();
          expect(warnings.length).toBe(1);
          expect(warnings[0]?.message).toBe(
            "An input array is constant; the correlation coefficient is not defined."
          );
          expect(warnings[0]?.source).toBe(name);
        });
      }

      it("does not warn for varying input", () => {
        const warnings = catchWarnings(() => {
          fn(tensor([1, 2, 3, 4]), tensor([2, 1, 4, 3]));
        });
        expect(warnings).toEqual([]);
      });
    });
  }

  it("kendalltau still returns [NaN, NaN] without a warning", () => {
    const warnings = catchWarnings(() => {
      const [tau, p] = kendalltau(tensor([1, 1, 1]), tensor([1, 2, 3]));
      expect(tau).toBeNaN();
      expect(p).toBeNaN();
    });
    expect(warnings).toEqual([]);
  });
});
