import { describe, expect, it } from "vitest";
import { InvalidParameterError } from "../../src/core";
import { max, min, nanmax, nanmedian, nanmin, tensor, zeros } from "../../src/ndarray";

// Reference: numpy.nanmax([]) and numpy.nanmin([]) raise ValueError ("zero-size array to
// reduction operation fmax which has no identity"), like numpy.max([]). An all-NaN slice gives
// NaN (with a RuntimeWarning). numpy.nanmedian([]) is NaN (with a RuntimeWarning).

describe("nanmax and nanmin on empty input", () => {
  it("throw the same error as max and min", () => {
    expect(() => nanmax(zeros([0]))).toThrow(InvalidParameterError);
    expect(() => nanmin(zeros([0]))).toThrow(InvalidParameterError);
    expect(() => nanmax(zeros([0]))).toThrow(/requires at least one element/);
    expect(() => max(zeros([0]))).toThrow(/requires at least one element/);
    expect(() => min(zeros([0]))).toThrow(/requires at least one element/);
    expect(() => nanmax(zeros([3, 0]), 1)).toThrow(InvalidParameterError);
  });

  it("keep NaN for an all-NaN slice", () => {
    expect(nanmax(tensor([Number.NaN, Number.NaN])).item()).toBeNaN();
    expect(nanmin(tensor([[Number.NaN, 1]]), 1).toArray()).toEqual([1]);
    expect(nanmin(tensor([[Number.NaN, Number.NaN]]), 1).toArray()).toEqual([Number.NaN]);
  });

  it("reduce a non-empty axis of an array that has an empty other axis", () => {
    // numpy.nanmax(np.zeros((2, 0)), axis=0) -> []
    expect(nanmax(zeros([2, 0]), 0).shape).toEqual([0]);
  });

  it("nanmedian of an empty tensor is NaN, as in numpy", () => {
    expect(nanmedian(zeros([0])).item()).toBeNaN();
  });
});
