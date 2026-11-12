import { describe, expect, it } from "vitest";
import { InvalidParameterError } from "../../src/core";
import { arange, histogram, tensor } from "../../src/ndarray";

describe("slice range objects", () => {
  it("throws on an unknown key and hints at end for stop", () => {
    const x = arange(10);
    expect(() => x.slice({ start: 3, stop: 5 } as never)).toThrow(InvalidParameterError);
    expect(() => x.slice({ start: 3, stop: 5 } as never)).toThrow(/"stop".*"end"/);
    expect(() => x.slice({ begin: 1 } as never)).toThrow(/unknown key "begin"/);
  });

  it("still accepts start, end and step", () => {
    expect(arange(10).slice({ start: 3, end: 5 }).toArray()).toEqual([3, 4]);
    expect(arange(10).slice({ step: 3 }).toArray()).toEqual([0, 3, 6, 9]);
    expect(arange(10).slice({}).size).toBe(10);
  });
});

describe("histogram bins", () => {
  it("rejects bins that are neither a number nor an array", () => {
    const x = tensor([1, 2]);
    expect(() => histogram(x, { bins: 3 } as never)).toThrow(InvalidParameterError);
    expect(() => histogram(x, "auto" as never)).toThrow(InvalidParameterError);
    expect(() => histogram(x, null as never)).toThrow(InvalidParameterError);
  });

  it("accepts a bin count and a list of edges", () => {
    // numpy.histogram([1, 2], bins=2) -> ([1, 1], [1, 1.5, 2])
    expect(histogram(tensor([1, 2]), 2).counts.toArray()).toEqual([1, 1]);
    expect(histogram(tensor([1, 2]), [0, 1.5, 3]).counts.toArray()).toEqual([1, 1]);
  });
});
