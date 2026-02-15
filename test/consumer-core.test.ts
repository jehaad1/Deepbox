import { describe, expect, it } from "vitest";
import {
  asReadonlyArray,
  BroadcastError,
  ConvergenceError,
  DataValidationError,
  DEVICES,
  DeepboxError,
  DeviceError,
  DTYPES,
  DTypeError,
  dtypeToTypedArrayCtor,
  ensureNumericDType,
  getArrayElement,
  getBigIntElement,
  getConfig,
  getDevice,
  getDtype,
  getElementAsNumber,
  getNumericElement,
  getSeed,
  getShapeDim,
  getStringElement,
  IndexError,
  InvalidParameterError,
  isBigInt64Array,
  isDevice,
  isDType,
  isNumericTypedArray,
  isTypedArray,
  MemoryError,
  NotFittedError,
  NotImplementedError,
  normalizeAxes,
  normalizeAxis,
  resetConfig,
  ShapeError,
  setConfig,
  setDevice,
  setDtype,
  setSeed,
  shapeToSize,
  validateArray,
  validateDevice,
  validateDtype,
  validateInteger,
  validateNonNegative,
  validateOneOf,
  validatePositive,
  validateRange,
  validateShape,
} from "../src/core";

describe("consumer API: core", () => {
  describe("config", () => {
    it("getConfig returns default config", () => {
      resetConfig();
      const cfg = getConfig();
      expect(typeof cfg.defaultDtype).toBe("string");
      expect(typeof cfg.defaultDevice).toBe("string");
    });

    it("setDtype/getDtype", () => {
      setDtype("float64");
      expect(getDtype()).toBe("float64");
      resetConfig();
    });

    it("setDevice/getDevice", () => {
      setDevice("cpu");
      expect(getDevice()).toBe("cpu");
    });

    it("setSeed/getSeed", () => {
      setSeed(42);
      expect(getSeed()).toBe(42);
      resetConfig();
    });

    it("setConfig partial update", () => {
      setConfig({ defaultDtype: "float64" });
      expect(getDtype()).toBe("float64");
      resetConfig();
      expect(getDtype()).toBe("float32");
    });
  });

  describe("errors", () => {
    it("all error classes extend DeepboxError", () => {
      expect(new DeepboxError("test")).toBeInstanceOf(Error);
      expect(new ShapeError("x")).toBeInstanceOf(DeepboxError);
      expect(new DTypeError("x")).toBeInstanceOf(DeepboxError);
      expect(new IndexError("x")).toBeInstanceOf(DeepboxError);
      expect(new BroadcastError([2, 3], [4, 5])).toBeInstanceOf(DeepboxError);
      expect(new InvalidParameterError("x", "alpha", -1)).toBeInstanceOf(DeepboxError);
      expect(new NotFittedError("x")).toBeInstanceOf(DeepboxError);
      expect(new NotImplementedError("x")).toBeInstanceOf(DeepboxError);
      expect(new ConvergenceError("x")).toBeInstanceOf(DeepboxError);
      expect(new DataValidationError("x")).toBeInstanceOf(DeepboxError);
      expect(new DeviceError("x")).toBeInstanceOf(DeepboxError);
      expect(new MemoryError("x")).toBeInstanceOf(DeepboxError);
    });
  });

  describe("type guards", () => {
    it("DTYPES and DEVICES are non-empty", () => {
      expect(DTYPES.length).toBeGreaterThan(0);
      expect(DEVICES.length).toBeGreaterThan(0);
    });

    it("isDType / isDevice", () => {
      expect(isDType("float32")).toBe(true);
      expect(isDType("invalid" as never)).toBe(false);
      expect(isDevice("cpu")).toBe(true);
      expect(isDevice("invalid" as never)).toBe(false);
    });
  });

  describe("utilities", () => {
    it("shapeToSize", () => {
      expect(shapeToSize([2, 3, 4])).toBe(24);
      expect(shapeToSize([])).toBe(1);
    });

    it("validators pass on valid input", () => {
      validateShape([2, 3], "test");
      validatePositive(1, "test");
      validateNonNegative(0, "test");
      validateInteger(5, "test");
      validateRange(0.5, 0, 1, "test");
      validateOneOf("a", ["a", "b", "c"], "test");
      validateArray([1, 2], "test");
      validateDtype("float32", "test");
      validateDevice("cpu", "test");
    });

    it("validators throw on invalid input", () => {
      expect(() => validatePositive(-1, "test")).toThrow();
      expect(() => validateNonNegative(-1, "test")).toThrow();
      expect(() => validateInteger(1.5, "test")).toThrow();
      expect(() => validateRange(5, 0, 1, "test")).toThrow();
      expect(() => validateDtype("invalid" as never, "test")).toThrow();
      expect(() => validateShape([-1, 3], "test")).toThrow();
    });

    it("normalizeAxis handles negative indices", () => {
      expect(normalizeAxis(0, 3)).toBe(0);
      expect(normalizeAxis(-1, 3)).toBe(2);
      const axes = normalizeAxes([0, -1], 3);
      expect(axes[0]).toBe(0);
      expect(axes[1]).toBe(2);
    });

    it("TypedArray checks", () => {
      expect(isTypedArray(new Float32Array(1))).toBe(true);
      expect(isTypedArray(new Float64Array(1))).toBe(true);
      expect(isTypedArray(new Int32Array(1))).toBe(true);
      expect(isTypedArray(new BigInt64Array(1))).toBe(true);
      expect(isTypedArray(new Uint8Array(1))).toBe(true);
      expect(isTypedArray([1, 2])).toBe(false);
      expect(isNumericTypedArray(new Float32Array(1))).toBe(true);
      expect(isBigInt64Array(new BigInt64Array(1))).toBe(true);
    });

    it("dtypeToTypedArrayCtor", () => {
      expect(new (dtypeToTypedArrayCtor("float32"))(1)).toBeInstanceOf(Float32Array);
      expect(new (dtypeToTypedArrayCtor("float64"))(1)).toBeInstanceOf(Float64Array);
    });

    it("ensureNumericDType", () => {
      expect(ensureNumericDType("float32")).toBe("float32");
      expect(ensureNumericDType("int32")).toBe("int32");
    });

    it("getShapeDim", () => {
      expect(getShapeDim([2, 3, 4], 0)).toBe(2);
      expect(getShapeDim([2, 3, 4], -1)).toBe(1);
    });

    it("element accessors", () => {
      expect(getArrayElement([10, 20, 30], 1)).toBe(20);
      expect(getNumericElement(new Float32Array([10, 20, 30]), 0)).toBe(10);
      expect(getElementAsNumber(new Float32Array([10, 20, 30]), 2)).toBe(30);
      expect(getStringElement(["hello", "world"], 0)).toBe("hello");
      expect(getBigIntElement(new BigInt64Array([1n, 2n, 3n]), 1)).toBe(2n);
    });

    it("asReadonlyArray", () => {
      const ro = asReadonlyArray([1, 2, 3]);
      expect(ro.length).toBe(3);
    });
  });
});
