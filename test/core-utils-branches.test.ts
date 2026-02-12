import { describe, expect, it } from "vitest";
import { IndexError } from "../src/core";
import { dtypeToTypedArrayCtor } from "../src/core/utils/dtypeUtils";
import {
  getArrayElement,
  getBigIntElement,
  getElementAsNumber,
  getNumericElement,
  getShapeDim,
  getStringElement,
  isBigInt64Array,
  isNumericTypedArray,
} from "../src/core/utils/typedArrayAccess";

describe("core utils branch coverage", () => {
  it("handles dtype constructor mapping and errors", () => {
    expect(dtypeToTypedArrayCtor("bool")).toBe(Uint8Array);
    expect(() => dtypeToTypedArrayCtor("string")).toThrow(/string dtype/i);
    // @ts-expect-error - invalid dtype should be rejected at runtime
    expect(() => dtypeToTypedArrayCtor("bad")).toThrow(/Unsupported dtype/i);
  });

  it("covers typed array access helpers and guards", () => {
    const nums = new Float32Array([1, 2, 3]);
    expect(getNumericElement(nums, 1)).toBe(2);
    expect(() => getNumericElement(nums, 99)).toThrow(IndexError);

    const bigs = new BigInt64Array([1n, 2n]);
    expect(getBigIntElement(bigs, 0)).toBe(1n);
    expect(() => getBigIntElement(bigs, 99)).toThrow(IndexError);

    expect(getElementAsNumber(bigs, 1)).toBe(2);
    expect(() => getElementAsNumber(bigs, 99)).toThrow(IndexError);
    expect(getElementAsNumber(nums, 2)).toBe(3);
    expect(() => getElementAsNumber(nums, 99)).toThrow(IndexError);

    expect(getShapeDim([2, 3], 1)).toBe(3);
    expect(getShapeDim([2, 3], 9)).toBe(1);
    expect(getArrayElement([5], 2, 7)).toBe(7);
    expect(getStringElement(["a"], 3)).toBe("");

    expect(isBigInt64Array(bigs)).toBe(true);
    expect(isBigInt64Array(nums)).toBe(false);
    expect(isNumericTypedArray(nums)).toBe(true);
    expect(isNumericTypedArray(bigs)).toBe(false);
  });
});
