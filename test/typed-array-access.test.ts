import { describe, expect, it } from "vitest";
import { DataValidationError, IndexError } from "../src/core";
import {
  asReadonlyArray,
  getArrayElement,
  getBigIntElement,
  getElementAsNumber,
  getNumericElement,
  getShapeDim,
  getStringElement,
  isBigInt64Array,
  isNumericTypedArray,
} from "../src/core/utils/typedArrayAccess";

describe("deepbox/core - Typed Array Access", () => {
  it("gets numeric elements safely", () => {
    const arr = new Float32Array([1, 2]);
    expect(getNumericElement(arr, 0)).toBe(1);
    expect(() => getNumericElement(arr, 2)).toThrow(IndexError);
  });

  it("gets bigint elements safely", () => {
    const arr = new BigInt64Array([1n, 2n]);
    expect(getBigIntElement(arr, 1)).toBe(2n);
    expect(() => getBigIntElement(arr, 3)).toThrow(IndexError);
  });

  it("gets element as number", () => {
    const arr = new BigInt64Array([3n]);
    expect(getElementAsNumber(arr, 0)).toBe(3);
  });

  it("throws IndexError for out-of-bounds getElementAsNumber", () => {
    const numArr = new Float64Array([1, 2]);
    expect(() => getElementAsNumber(numArr, 5)).toThrow(IndexError);
    const bigArr = new BigInt64Array([1n]);
    expect(() => getElementAsNumber(bigArr, 3)).toThrow(IndexError);
  });

  it("throws on bigint values outside safe integer range", () => {
    const tooLarge = BigInt(Number.MAX_SAFE_INTEGER) + 1n;
    const arr = new BigInt64Array([tooLarge]);
    expect(() => getElementAsNumber(arr, 0)).toThrow(DataValidationError);
  });

  it("throws on bigint values below MIN_SAFE_INTEGER", () => {
    const tooSmall = BigInt(Number.MIN_SAFE_INTEGER) - 1n;
    const arr = new BigInt64Array([tooSmall]);
    expect(() => getElementAsNumber(arr, 0)).toThrow(DataValidationError);
  });

  it("gets shape and array elements", () => {
    expect(getShapeDim([2, 3], 1)).toBe(3);
    expect(getShapeDim([2, 3], 3)).toBe(1);
    expect(getArrayElement([1, 2], 5, 7)).toBe(7);
  });

  it("gets string elements safely", () => {
    expect(getStringElement(["a"], 0)).toBe("a");
    expect(getStringElement(["a"], 1)).toBe("");
  });

  it("checks typed array kinds", () => {
    const n = new Float64Array([1]);
    const b = new BigInt64Array([1n]);
    expect(isBigInt64Array(b)).toBe(true);
    expect(isNumericTypedArray(n)).toBe(true);
    expect(isNumericTypedArray(b)).toBe(false);
  });

  it("creates readonly array view", () => {
    const arr = [1, 2, 3];
    const ro = asReadonlyArray(arr);
    expect(ro).toEqual([1, 2, 3]);
  });
});
