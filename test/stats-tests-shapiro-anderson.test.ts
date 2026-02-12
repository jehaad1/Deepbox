import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { anderson, shapiro } from "../src/stats/tests";

describe("stats shapiro/anderson branches", () => {
  it("shapiro errors on invalid sizes and ranges", () => {
    expect(() => shapiro(tensor([1, 2]))).toThrow(/between 3 and 5000/i);
    expect(() => shapiro(tensor([1, 1, 1]))).toThrow(/identical/i);
    expect(() => shapiro(tensor([0, 1e-20, 2e-20]))).toThrow(/range is too small/i);
  });

  it("shapiro handles n=3 path", () => {
    const res = shapiro(tensor([1, 2, 4]));
    expect(res.statistic).toBeGreaterThan(0);
    expect(res.pvalue).toBeGreaterThanOrEqual(0);
    expect(res.pvalue).toBeLessThanOrEqual(1);
  });

  it("anderson errors on empty/constant inputs and returns critical values", () => {
    expect(() => anderson(tensor([]))).toThrow(/at least one element/i);
    expect(() => anderson(tensor([2, 2, 2]))).toThrow(/constant/i);

    const res = anderson(tensor([1, 2, 3, 4, 5]));
    expect(res.critical_values.length).toBe(5);
    expect(res.significance_level.length).toBe(5);
  });
});
