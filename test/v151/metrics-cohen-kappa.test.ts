import { describe, expect, it } from "vitest";
import { InvalidParameterError } from "../../src/core";
import { cohenKappaScore } from "../../src/metrics";
import { tensor } from "../../src/ndarray";

// Expected values come from scikit-learn 1.9.1 cohen_kappa_score(weights=None | "linear" |
// "quadratic", sample_weight=...).

const yt = [0, 1, 2, 2, 1, 0, 2, 1, 0, 2, 1, 1];
const yp = [0, 2, 2, 1, 1, 0, 1, 1, 0, 0, 2, 1];
const sw = [1, 2, 0.5, 3, 1, 1, 2, 1, 1, 4, 0.5, 1];
const t = tensor(yt);
const p = tensor(yp);

describe("cohenKappaScore sampleWeight", () => {
  it("matches scikit-learn without class weights", () => {
    expect(cohenKappaScore(t, p, { sampleWeight: sw })).toBeCloseTo(0.10195227765726678, 12);
  });

  it("matches scikit-learn with linear and quadratic class weights", () => {
    expect(cohenKappaScore(t, p, { weights: "linear", sampleWeight: sw })).toBeCloseTo(
      0.08823529411764708,
      12
    );
    expect(cohenKappaScore(t, p, { weights: "quadratic", sampleWeight: sw })).toBeCloseTo(
      0.07439824945295404,
      12
    );
  });

  it("still accepts the class weights as the third argument", () => {
    expect(cohenKappaScore(t, p)).toBeCloseTo(0.368421052631579, 12);
    expect(cohenKappaScore(t, p, "linear")).toBeCloseTo(0.4, 12);
    expect(cohenKappaScore(t, p, "quadratic")).toBeCloseTo(0.43529411764705883, 12);
    expect(cohenKappaScore(t, p, { weights: "linear" })).toBeCloseTo(0.4, 12);
  });

  it("weights string labels", () => {
    const ys = tensor(["a", "b", "b", "c", "a", "c", "b"]);
    const ps = tensor(["a", "b", "c", "c", "a", "b", "b"]);
    expect(cohenKappaScore(ys, ps, { sampleWeight: [1, 2, 3, 1, 2, 1, 1] })).toBeCloseTo(
      0.44999999999999996,
      12
    );
  });

  it("names the invalid value in the error message", () => {
    expect(() => cohenKappaScore(t, p, "cubic" as never)).toThrow(/'cubic'/);
    expect(() => cohenKappaScore(t, p, { weights: "cubic" as never })).toThrow(/'cubic'/);
    expect(() => cohenKappaScore(t, p, { weights: "cubic" as never })).not.toThrow(/object Object/);
    expect(() => cohenKappaScore(t, p, { weights: 3 as never })).toThrow(InvalidParameterError);
    expect(() => cohenKappaScore(t, p, [1, 2] as never)).not.toThrow(/object Object/);
    expect(() => cohenKappaScore(t, p, { sampleWeight: [1, 2] })).toThrow(/sampleWeight/);
  });
});
