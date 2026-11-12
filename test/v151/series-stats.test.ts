import { describe, expect, it } from "vitest";
import { DataValidationError } from "../../src/core/errors/index";
import { Series } from "../../src/dataframe/index";

// pandas 3.1:
//   e = pd.Series([], dtype=float)
//   e.sum() -> 0.0; e.mean(), e.min(), e.max(), e.median(), e.std(), e.var() -> nan
//   pd.Series([True, False, True, True]).sum() -> 3; .mean() -> 0.75
describe("Series statistics of an empty Series and of booleans (DF-13)", () => {
  it("sum of an empty Series is 0", () => {
    expect(new Series([]).sum()).toBe(0);
  });

  it("the other statistics of an empty Series are NaN", () => {
    const empty = new Series<number>([]);
    expect(empty.mean()).toBeNaN();
    expect(empty.median()).toBeNaN();
    expect(empty.min()).toBeNaN();
    expect(empty.max()).toBeNaN();
    expect(empty.std()).toBeNaN();
    expect(empty.var()).toBeNaN();
  });

  it("sum and mean count true as 1 and false as 0", () => {
    const flags = new Series([true, false, true, true]);
    expect(flags.sum()).toBe(3);
    expect(flags.mean()).toBe(0.75);
    expect(new Series([true, 2, null, false]).sum()).toBe(3);
  });

  it("other statistics still reject booleans and strings", () => {
    expect(() => new Series([true, false]).median()).toThrow(DataValidationError);
    expect(() => new Series(["a", "b"]).sum()).toThrow(DataValidationError);
  });
});
