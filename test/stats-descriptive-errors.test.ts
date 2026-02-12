import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import {
  geometricMean,
  harmonicMean,
  median,
  mode,
  quantile,
  std,
  trimMean,
} from "../src/stats/descriptive";

describe("stats descriptive error branches", () => {
  it("throws on empty inputs and invalid parameters", () => {
    const empty = tensor([]);
    expect(() => median(empty)).toThrow(/requires at least one element/i);
    expect(() => mode(empty)).toThrow(/requires at least one element/i);
    expect(() => std(empty)).toThrow(/requires at least one element/i);
    expect(() => quantile(empty, 0.5)).toThrow(/requires at least one element/i);
    expect(() => trimMean(empty, 0.1)).toThrow(/requires at least one element/i);
  });

  it("validates quantile and trimMean parameters", () => {
    const t = tensor([1, 2, 3]);
    expect(() => quantile(t, 2)).toThrow(/q must be in \[0, 1\]/i);
    expect(() => trimMean(t, 0.75)).toThrow(/proportiontocut/i);
  });

  it("validates means for positivity/non-zero", () => {
    expect(() => geometricMean(tensor([1, 0, 2]))).toThrow(/> 0/);
    expect(() => harmonicMean(tensor([1, 0, 2]))).toThrow(/> 0/i);
  });
});
