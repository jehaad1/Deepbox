import { beforeEach, describe, expect, it } from "vitest";
import {
  BroadcastError,
  ConvergenceError,
  DataValidationError,
  DEVICES,
  type DeepboxConfig,
  DeepboxError,
  DeviceError,
  DTYPES,
  type DType,
  DTypeError,
  dtypeToTypedArrayCtor,
  getConfig,
  getDevice,
  getDtype,
  getSeed,
  IndexError,
  InvalidParameterError,
  isDevice,
  isDType,
  isTypedArray,
  MemoryError,
  NotFittedError,
  NotImplementedError,
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

describe("deepbox/core", () => {
  describe("Types", () => {
    it("should export DTYPES constant", () => {
      expect(DTYPES).toEqual(["float32", "float64", "int32", "int64", "uint8", "bool", "string"]);
    });

    it("should export DEVICES constant", () => {
      expect(DEVICES).toEqual(["cpu", "webgpu", "wasm"]);
    });

    it("should validate dtype correctly", () => {
      expect(isDType("float32")).toBe(true);
      expect(isDType("float64")).toBe(true);
      expect(isDType("invalid")).toBe(false);
      expect(isDType(123)).toBe(false);
    });

    it("should validate all dtypes", () => {
      for (const dtype of DTYPES) {
        expect(isDType(dtype)).toBe(true);
      }
    });

    it("should reject non-string values as dtype", () => {
      expect(isDType(null)).toBe(false);
      expect(isDType(undefined)).toBe(false);
      expect(isDType({})).toBe(false);
      expect(isDType([])).toBe(false);
    });

    it("should be case-sensitive for dtype", () => {
      expect(isDType("Float32")).toBe(false);
      expect(isDType("FLOAT32")).toBe(false);
    });

    it("should validate device correctly", () => {
      expect(isDevice("cpu")).toBe(true);
      expect(isDevice("webgpu")).toBe(true);
      expect(isDevice("invalid")).toBe(false);
      expect(isDevice(null)).toBe(false);
    });

    it("should validate all devices", () => {
      for (const device of DEVICES) {
        expect(isDevice(device)).toBe(true);
      }
    });

    it("should reject non-string values as device", () => {
      expect(isDevice(undefined)).toBe(false);
      expect(isDevice(123)).toBe(false);
      expect(isDevice({})).toBe(false);
    });

    it("should be case-sensitive for device", () => {
      expect(isDevice("CPU")).toBe(false);
      expect(isDevice("Cpu")).toBe(false);
    });

    it("DTYPES should be an array", () => {
      expect(Array.isArray(DTYPES)).toBe(true);
      expect(DTYPES.length).toBe(7);
    });

    it("DEVICES should be an array", () => {
      expect(Array.isArray(DEVICES)).toBe(true);
      expect(DEVICES.length).toBe(3);
    });
  });

  describe("Configuration", () => {
    beforeEach(() => {
      resetConfig();
    });

    it("should have default configuration", () => {
      const config = getConfig();
      expect(config.defaultDtype).toBe("float32");
      expect(config.defaultDevice).toBe("cpu");
      expect(config.seed).toBeNull();
    });

    it("should set and get configuration", () => {
      setConfig({ defaultDtype: "float64", defaultDevice: "webgpu", seed: 42 });
      const config = getConfig();
      expect(config.defaultDtype).toBe("float64");
      expect(config.defaultDevice).toBe("webgpu");
      expect(config.seed).toBe(42);
    });

    it("should reset configuration", () => {
      setConfig({ defaultDtype: "float64", seed: 123 });
      resetConfig();
      const config = getConfig();
      expect(config.defaultDtype).toBe("float32");
      expect(config.seed).toBeNull();
    });

    it("should set and get device", () => {
      setDevice("webgpu");
      expect(getDevice()).toBe("webgpu");
    });

    it("should set and get dtype", () => {
      setDtype("float64");
      expect(getDtype()).toBe("float64");
    });

    it("should reject invalid device and dtype in setters", () => {
      // @ts-expect-error - invalid device should be rejected at runtime
      expect(() => setDevice("gpu")).toThrow(DataValidationError);
      // @ts-expect-error - invalid dtype should be rejected at runtime
      expect(() => setDtype("float128")).toThrow(DataValidationError);
    });

    it("should set and get seed", () => {
      setSeed(42);
      expect(getSeed()).toBe(42);
    });

    it("should throw error for non-integer seed", () => {
      expect(() => setSeed(3.14)).toThrow(DataValidationError);
      expect(() => setSeed(3.14)).toThrow("seed must be an integer");
    });

    it("should throw error for NaN seed", () => {
      expect(() => setSeed(NaN)).toThrow(DataValidationError);
    });

    it("should throw error for Infinity seed", () => {
      expect(() => setSeed(Infinity)).toThrow(DataValidationError);
    });

    it("should throw error for unsafe integer seed", () => {
      expect(() => setSeed(Number.MAX_SAFE_INTEGER + 1)).toThrow(DataValidationError);
      expect(() => setSeed(Number.MAX_SAFE_INTEGER + 1)).toThrow("safe integer");
    });

    it("should accept negative integer seed", () => {
      setSeed(-42);
      expect(getSeed()).toBe(-42);
    });

    it("should accept zero as seed", () => {
      setSeed(0);
      expect(getSeed()).toBe(0);
    });

    it("should return immutable config copy", () => {
      const config1 = getConfig();
      const config2 = getConfig();
      expect(config1).not.toBe(config2);
      expect(config1).toEqual(config2);
    });

    it("should not allow external mutation of config", () => {
      const config = getConfig() as { defaultDtype: string };
      config.defaultDtype = "int32";
      expect(getConfig().defaultDtype).toBe("float32");
    });

    it("should handle partial config updates", () => {
      setConfig({ seed: 123 });
      expect(getConfig().seed).toBe(123);
      expect(getConfig().defaultDtype).toBe("float32");
      expect(getConfig().defaultDevice).toBe("cpu");
    });

    it("should handle multiple sequential config updates", () => {
      setConfig({ defaultDtype: "float64" });
      setConfig({ seed: 42 });
      setConfig({ defaultDevice: "webgpu" });
      const config = getConfig();
      expect(config.defaultDtype).toBe("float64");
      expect(config.seed).toBe(42);
      expect(config.defaultDevice).toBe("webgpu");
    });

    it("should reject unknown config keys", () => {
      // @ts-expect-error - extra keys should be rejected at runtime
      expect(() => setConfig({ extra: "nope" })).toThrow(DataValidationError);
    });

    it("should reject non-object config inputs", () => {
      // @ts-expect-error - non-object should be rejected at runtime
      expect(() => setConfig([])).toThrow(DataValidationError);
      // @ts-expect-error - non-object should be rejected at runtime
      expect(() => setConfig("string")).toThrow(DataValidationError);
      // @ts-expect-error - non-object should be rejected at runtime
      expect(() => setConfig(123)).toThrow(DataValidationError);
    });

    it("should reject non-plain config objects", () => {
      class Config {
        defaultDtype = "float32" as const;
      }
      const instance = new Config();
      expect(() => setConfig(instance)).toThrow(DataValidationError);

      const proto = { seed: 123 };
      const derived: object = Object.create(proto);
      expect(() => setConfig(derived)).toThrow(DataValidationError);
    });

    it("should accept null-prototype config objects", () => {
      const nullProto: Partial<DeepboxConfig> = Object.assign(Object.create(null), { seed: 7 });
      setConfig(nullProto);
      expect(getConfig().seed).toBe(7);
    });

    it("should reject invalid config values", () => {
      resetConfig();
      // @ts-expect-error - invalid dtype should be rejected at runtime
      expect(() => setConfig({ defaultDtype: "bad" })).toThrow(DataValidationError);
      expect(getConfig().defaultDtype).toBe("float32");

      // @ts-expect-error - invalid device should be rejected at runtime
      expect(() => setConfig({ defaultDevice: "gpu" })).toThrow(DataValidationError);
      expect(getConfig().defaultDevice).toBe("cpu");

      expect(() => setConfig({ seed: 3.14 })).toThrow(DataValidationError);
      expect(() => setConfig({ seed: Number.MAX_SAFE_INTEGER + 1 })).toThrow(DataValidationError);

      setConfig({ seed: null });
      expect(getConfig().seed).toBeNull();
    });
  });

  describe("Validation Functions", () => {
    describe("validateShape", () => {
      it("should accept valid shapes", () => {
        expect(() => validateShape([2, 3, 4])).not.toThrow();
        expect(() => validateShape([1])).not.toThrow();
        expect(() => validateShape([])).not.toThrow();
        expect(() => validateShape([0])).not.toThrow();
      });

      it("should reject invalid shapes", () => {
        expect(() => validateShape([2, -1, 4])).toThrow(DataValidationError);
        expect(() => validateShape([2.5, 3])).toThrow(DataValidationError);
        expect(() => validateShape([NaN, 3])).toThrow(DataValidationError);
        expect(() => validateShape([Infinity, 3])).toThrow(DataValidationError);
      });

      it("should reject shapes with negative infinity", () => {
        expect(() => validateShape([-Infinity, 3])).toThrow(DataValidationError);
      });

      it("should reject shapes with unsafe integer dimensions", () => {
        expect(() => validateShape([Number.MAX_SAFE_INTEGER + 1])).toThrow(DataValidationError);
        expect(() => validateShape([Number.MAX_SAFE_INTEGER + 1])).toThrow("safe integer");
      });

      it("should provide detailed error messages for invalid shapes", () => {
        expect(() => validateShape([2, -1, 4])).toThrow("must be >= 0");
        expect(() => validateShape([2.5, 3])).toThrow("must be a finite integer");
      });

      it("should accept custom parameter name in error messages", () => {
        expect(() => validateShape([2, -1], "myShape")).toThrow("myShape[1]");
      });

      it("should handle very large valid shapes", () => {
        const largeShape = [1000, 1000, 1000];
        expect(() => validateShape(largeShape)).not.toThrow();
      });

      it("should handle shapes with mixed valid dimensions", () => {
        expect(() => validateShape([1, 0, 5])).not.toThrow();
      });

      it("should reject non-array shapes", () => {
        expect(() => validateShape("not-array")).toThrow(DataValidationError);
        expect(() => validateShape(null)).toThrow(DataValidationError);
      });
    });

    describe("validateDtype", () => {
      it("should accept valid dtypes", () => {
        expect(() => validateDtype("float32")).not.toThrow();
        expect(() => validateDtype("int64")).not.toThrow();
      });

      it("should reject invalid dtypes", () => {
        expect(() => validateDtype("invalid")).toThrow(DataValidationError);
        expect(() => validateDtype(123)).toThrow(DataValidationError);
      });

      it("should reject null and undefined as dtype", () => {
        expect(() => validateDtype(null)).toThrow(DataValidationError);
        expect(() => validateDtype(undefined)).toThrow(DataValidationError);
      });

      it("should reject object as dtype", () => {
        expect(() => validateDtype({})).toThrow(DataValidationError);
      });

      it("should provide list of valid dtypes in error message", () => {
        try {
          validateDtype("invalid");
        } catch (error) {
          expect(error).toBeInstanceOf(DataValidationError);
          expect((error as Error).message).toContain("float32");
          expect((error as Error).message).toContain("float64");
        }
      });
    });

    describe("validateDevice", () => {
      it("should accept valid devices", () => {
        expect(() => validateDevice("cpu")).not.toThrow();
        expect(() => validateDevice("webgpu")).not.toThrow();
        expect(() => validateDevice("wasm")).not.toThrow();
      });

      it("should reject invalid devices", () => {
        expect(() => validateDevice("gpu")).toThrow(DataValidationError);
        expect(() => validateDevice(null)).toThrow(DataValidationError);
      });

      it("should reject undefined as device", () => {
        expect(() => validateDevice(undefined)).toThrow(DataValidationError);
      });

      it("should reject number as device", () => {
        expect(() => validateDevice(123)).toThrow(DataValidationError);
      });

      it("should provide list of valid devices in error message", () => {
        try {
          validateDevice("invalid");
        } catch (error) {
          expect(error).toBeInstanceOf(DataValidationError);
          expect((error as Error).message).toContain("cpu");
          expect((error as Error).message).toContain("webgpu");
        }
      });
    });

    describe("validatePositive", () => {
      it("should accept positive numbers", () => {
        expect(() => validatePositive(1, "value")).not.toThrow();
        expect(() => validatePositive(0.001, "value")).not.toThrow();
        expect(() => validatePositive(1000, "value")).not.toThrow();
      });

      it("should reject non-positive numbers", () => {
        expect(() => validatePositive(0, "value")).toThrow(DataValidationError);
        expect(() => validatePositive(-1, "value")).toThrow(DataValidationError);
        expect(() => validatePositive(NaN, "value")).toThrow(DataValidationError);
        expect(() => validatePositive(Infinity, "value")).toThrow(DataValidationError);
      });

      it("should reject negative infinity", () => {
        expect(() => validatePositive(-Infinity, "value")).toThrow(DataValidationError);
      });

      it("should accept very small positive numbers", () => {
        expect(() => validatePositive(Number.MIN_VALUE, "value")).not.toThrow();
        expect(() => validatePositive(1e-100, "value")).not.toThrow();
      });

      it("should provide clear error message for zero", () => {
        expect(() => validatePositive(0, "value")).toThrow("must be positive (> 0)");
      });
    });

    describe("validateNonNegative", () => {
      it("should accept non-negative numbers", () => {
        expect(() => validateNonNegative(0, "value")).not.toThrow();
        expect(() => validateNonNegative(1, "value")).not.toThrow();
        expect(() => validateNonNegative(0.5, "value")).not.toThrow();
      });

      it("should reject negative numbers", () => {
        expect(() => validateNonNegative(-1, "value")).toThrow(DataValidationError);
        expect(() => validateNonNegative(-0.001, "value")).toThrow(DataValidationError);
      });

      it("should reject NaN and Infinity", () => {
        expect(() => validateNonNegative(NaN, "value")).toThrow(DataValidationError);
        expect(() => validateNonNegative(Infinity, "value")).toThrow(DataValidationError);
        expect(() => validateNonNegative(-Infinity, "value")).toThrow(DataValidationError);
      });

      it("should accept very large numbers", () => {
        expect(() => validateNonNegative(Number.MAX_VALUE, "value")).not.toThrow();
      });
    });

    describe("validateRange", () => {
      it("should accept values in range", () => {
        expect(() => validateRange(0.5, 0, 1, "value")).not.toThrow();
        expect(() => validateRange(0, 0, 1, "value")).not.toThrow();
        expect(() => validateRange(1, 0, 1, "value")).not.toThrow();
      });

      it("should reject invalid range bounds", () => {
        expect(() => validateRange(1, 2, 1, "value")).toThrow(DataValidationError);
        expect(() => validateRange(1, NaN, 2, "value")).toThrow(DataValidationError);
        expect(() => validateRange(1, 0, Infinity, "value")).toThrow(DataValidationError);
      });

      it("should reject values out of range", () => {
        expect(() => validateRange(-0.1, 0, 1, "value")).toThrow(DataValidationError);
        expect(() => validateRange(1.1, 0, 1, "value")).toThrow(DataValidationError);
      });

      it("should reject NaN in range validation", () => {
        expect(() => validateRange(NaN, 0, 1, "value")).toThrow(DataValidationError);
      });

      it("should reject Infinity in range validation", () => {
        expect(() => validateRange(Infinity, 0, 1, "value")).toThrow(DataValidationError);
      });

      it("should handle negative ranges", () => {
        expect(() => validateRange(-5, -10, -1, "value")).not.toThrow();
        expect(() => validateRange(-11, -10, -1, "value")).toThrow(DataValidationError);
      });

      it("should provide range in error message", () => {
        expect(() => validateRange(5, 0, 1, "value")).toThrow("[0, 1]");
      });
    });

    describe("validateInteger", () => {
      it("should accept integers", () => {
        expect(() => validateInteger(0, "value")).not.toThrow();
        expect(() => validateInteger(42, "value")).not.toThrow();
        expect(() => validateInteger(-10, "value")).not.toThrow();
      });

      it("should reject non-integers", () => {
        expect(() => validateInteger(3.14, "value")).toThrow(DataValidationError);
        expect(() => validateInteger(NaN, "value")).toThrow(DataValidationError);
      });

      it("should reject Infinity", () => {
        expect(() => validateInteger(Infinity, "value")).toThrow(DataValidationError);
        expect(() => validateInteger(-Infinity, "value")).toThrow(DataValidationError);
      });

      it("should accept very large integers", () => {
        expect(() => validateInteger(Number.MAX_SAFE_INTEGER, "value")).not.toThrow();
        expect(() => validateInteger(Number.MIN_SAFE_INTEGER, "value")).not.toThrow();
      });

      it("should reject integers beyond safe range", () => {
        expect(() => validateInteger(Number.MAX_SAFE_INTEGER + 1, "value")).toThrow(
          DataValidationError
        );
        expect(() => validateInteger(Number.MIN_SAFE_INTEGER - 1, "value")).toThrow(
          DataValidationError
        );
      });

      it("should reject very small fractional parts", () => {
        expect(() => validateInteger(1.0000000001, "value")).toThrow(DataValidationError);
      });
    });

    describe("validateOneOf", () => {
      it("should accept valid options", () => {
        expect(() => validateOneOf("a", ["a", "b", "c"], "value")).not.toThrow();
        expect(() => validateOneOf("c", ["a", "b", "c"], "value")).not.toThrow();
      });

      it("should reject invalid options", () => {
        expect(() => validateOneOf("d", ["a", "b", "c"], "value")).toThrow(DataValidationError);
      });

      it("should reject non-string values", () => {
        expect(() => validateOneOf(123, ["a", "b"], "value")).toThrow(DataValidationError);
        expect(() => validateOneOf(null, ["a", "b"], "value")).toThrow(DataValidationError);
      });

      it("should handle empty options array", () => {
        expect(() => validateOneOf("a", [], "value")).toThrow(DataValidationError);
      });

      it("should be case-sensitive", () => {
        expect(() => validateOneOf("A", ["a", "b", "c"], "value")).toThrow(DataValidationError);
      });

      it("should provide list of valid options in error", () => {
        expect(() => validateOneOf("d", ["a", "b", "c"], "value")).toThrow("a, b, c");
      });
    });

    describe("validateArray", () => {
      it("should accept arrays", () => {
        expect(() => validateArray([], "value")).not.toThrow();
        expect(() => validateArray([1, 2, 3], "value")).not.toThrow();
      });

      it("should reject non-arrays", () => {
        expect(() => validateArray("not-array", "value")).toThrow(DataValidationError);
        expect(() => validateArray(null, "value")).toThrow(DataValidationError);
        expect(() => validateArray({}, "value")).toThrow(DataValidationError);
      });

      it("should reject undefined", () => {
        expect(() => validateArray(undefined, "value")).toThrow(DataValidationError);
      });

      it("should reject numbers", () => {
        expect(() => validateArray(123, "value")).toThrow(DataValidationError);
      });

      it("should accept array-like objects that are arrays", () => {
        expect(() => validateArray(Array.from({ length: 3 }), "value")).not.toThrow();
      });

      it("should reject TypedArrays", () => {
        expect(() => validateArray(new Float32Array(10), "value")).toThrow(DataValidationError);
      });
    });
  });

  describe("Utility Functions", () => {
    describe("shapeToSize", () => {
      it("should calculate size correctly", () => {
        expect(shapeToSize([2, 3])).toBe(6);
        expect(shapeToSize([2, 3, 4])).toBe(24);
        expect(shapeToSize([5])).toBe(5);
        expect(shapeToSize([])).toBe(1);
      });

      it("should handle zero dimensions", () => {
        expect(shapeToSize([0, 3])).toBe(0);
        expect(shapeToSize([2, 0, 4])).toBe(0);
      });

      it("should handle single dimension", () => {
        expect(shapeToSize([10])).toBe(10);
        expect(shapeToSize([1])).toBe(1);
      });

      it("should handle high-dimensional tensors", () => {
        expect(shapeToSize([2, 2, 2, 2, 2])).toBe(32);
        expect(shapeToSize([1, 1, 1, 1, 1, 1])).toBe(1);
      });

      it("should handle very large shapes", () => {
        expect(shapeToSize([1000, 1000])).toBe(1000000);
      });

      it("should handle shapes with ones", () => {
        expect(shapeToSize([1, 5, 1, 3])).toBe(15);
      });

      it("should reject invalid shapes", () => {
        expect(() => shapeToSize("bad")).toThrow(DataValidationError);
        expect(() => shapeToSize("bad")).toThrow("must be an array");
        expect(() => shapeToSize([2, -1])).toThrow(DataValidationError);
        expect(() => shapeToSize([2, -1])).toThrow("must be >= 0");
        expect(() => shapeToSize([2.5, 3])).toThrow(DataValidationError);
        expect(() => shapeToSize([2.5, 3])).toThrow("must be a finite integer");
      });

      it("should throw when size exceeds safe integer range", () => {
        expect(() => shapeToSize([Number.MAX_SAFE_INTEGER, 2])).toThrow(DataValidationError);
        expect(() => shapeToSize([Number.MAX_SAFE_INTEGER, 2])).toThrow("too large");
      });
    });

    describe("isTypedArray", () => {
      it("should identify TypedArrays", () => {
        expect(isTypedArray(new Float32Array(10))).toBe(true);
        expect(isTypedArray(new Float64Array(10))).toBe(true);
        expect(isTypedArray(new Int32Array(10))).toBe(true);
        expect(isTypedArray(new Uint8Array(10))).toBe(true);
      });

      it("should reject unsupported TypedArray subclasses", () => {
        expect(isTypedArray(new Uint16Array(10))).toBe(false);
        expect(isTypedArray(new Int16Array(10))).toBe(false);
        expect(isTypedArray(new Uint32Array(10))).toBe(false);
        expect(isTypedArray(new Int8Array(10))).toBe(false);
      });

      it("should identify BigInt64Array", () => {
        expect(isTypedArray(new BigInt64Array(10))).toBe(true);
      });

      it("should reject non-TypedArrays", () => {
        expect(isTypedArray([])).toBe(false);
        expect(isTypedArray(new DataView(new ArrayBuffer(10)))).toBe(false);
        expect(isTypedArray({})).toBe(false);
        expect(isTypedArray(null)).toBe(false);
      });

      it("should reject undefined", () => {
        expect(isTypedArray(undefined)).toBe(false);
      });

      it("should reject strings", () => {
        expect(isTypedArray("not a typed array")).toBe(false);
      });

      it("should reject numbers", () => {
        expect(isTypedArray(123)).toBe(false);
      });

      it("should reject functions", () => {
        expect(isTypedArray(() => {})).toBe(false);
      });

      it("should handle empty TypedArrays", () => {
        expect(isTypedArray(new Float32Array(0))).toBe(true);
        expect(isTypedArray(new Uint8Array(0))).toBe(true);
      });
    });

    describe("dtypeToTypedArrayCtor", () => {
      it("should return correct constructors", () => {
        expect(dtypeToTypedArrayCtor("float32")).toBe(Float32Array);
        expect(dtypeToTypedArrayCtor("float64")).toBe(Float64Array);
        expect(dtypeToTypedArrayCtor("int32")).toBe(Int32Array);
        expect(dtypeToTypedArrayCtor("int64")).toBe(BigInt64Array);
        expect(dtypeToTypedArrayCtor("uint8")).toBe(Uint8Array);
        expect(dtypeToTypedArrayCtor("bool")).toBe(Uint8Array);
      });

      it("should throw for unsupported dtypes", () => {
        expect(() => dtypeToTypedArrayCtor("string" as DType)).toThrow();
      });

      it("should throw DTypeError for string dtype", () => {
        expect(() => dtypeToTypedArrayCtor("string" as DType)).toThrow(DTypeError);
        expect(() => dtypeToTypedArrayCtor("string" as DType)).toThrow("not supported");
      });

      it("should return constructors that can create arrays", () => {
        const Float32Ctor = dtypeToTypedArrayCtor("float32");
        const arr = new Float32Ctor(5);
        expect(arr).toBeInstanceOf(Float32Array);
        expect(arr.length).toBe(5);
      });

      it("should return Uint8Array for bool dtype", () => {
        const BoolCtor = dtypeToTypedArrayCtor("bool");
        expect(BoolCtor).toBe(Uint8Array);
      });

      it("should handle all numeric dtypes", () => {
        const dtypes: DType[] = ["float32", "float64", "int32", "int64", "uint8", "bool"];
        for (const dtype of dtypes) {
          const Ctor = dtypeToTypedArrayCtor(dtype);
          expect(Ctor).toBeDefined();
          expect(typeof Ctor).toBe("function");
        }
      });
    });
  });

  describe("Error Classes", () => {
    it("should create DeepboxError", () => {
      const error = new DeepboxError("test message");
      expect(error).toBeInstanceOf(Error);
      expect(error.name).toBe("DeepboxError");
      expect(error.message).toBe("test message");
    });

    it("should create DeepboxError with cause", () => {
      const cause = new Error("root cause");
      const error = new DeepboxError("test message", { cause });
      expect(error.cause).toBe(cause);
    });

    it("should create DeepboxError without message", () => {
      const error = new DeepboxError();
      expect(error).toBeInstanceOf(Error);
      expect(error.name).toBe("DeepboxError");
    });

    it("should create ShapeError", () => {
      const error = new ShapeError("shape mismatch");
      expect(error).toBeInstanceOf(DeepboxError);
      expect(error.name).toBe("ShapeError");
    });

    it("should create ShapeError with details", () => {
      const error = new ShapeError("mismatch", {
        expected: [2, 3],
        received: [3, 2],
        context: "matrix multiplication",
      });
      expect(error.expected).toEqual([2, 3]);
      expect(error.received).toEqual([3, 2]);
      expect(error.context).toBe("matrix multiplication");
    });

    it("should snapshot shapes in ShapeError details", () => {
      const expected = [2, 3];
      const received = [3, 2];
      const error = new ShapeError("mismatch", { expected, received });
      expected[0] = 99;
      received[1] = 88;
      expect(error.expected).toEqual([2, 3]);
      expect(error.received).toEqual([3, 2]);
    });

    it("should create ShapeError using static mismatch method", () => {
      const error = ShapeError.mismatch([2, 3], [3, 2], "test context");
      expect(error).toBeInstanceOf(ShapeError);
      expect(error.expected).toEqual([2, 3]);
      expect(error.received).toEqual([3, 2]);
      expect(error.context).toBe("test context");
      expect(error.message).toContain("Shape mismatch");
      expect(error.message).toContain("test context");
    });

    it("should create ShapeError using static mismatch without context", () => {
      const error = ShapeError.mismatch([2, 3], [3, 2]);
      expect(error.expected).toEqual([2, 3]);
      expect(error.received).toEqual([3, 2]);
      expect(error.context).toBeUndefined();
    });

    it("should create DTypeError", () => {
      const error = new DTypeError("dtype mismatch");
      expect(error).toBeInstanceOf(DeepboxError);
      expect(error.name).toBe("DTypeError");
    });

    it("should create DeviceError", () => {
      const error = new DeviceError("device error");
      expect(error).toBeInstanceOf(DeepboxError);
      expect(error.name).toBe("DeviceError");
    });

    it("should create IndexError with details", () => {
      const error = new IndexError("index out of bounds", {
        index: 10,
        validRange: [0, 5],
      });
      expect(error).toBeInstanceOf(DeepboxError);
      expect(error.name).toBe("IndexError");
      expect(error.index).toBe(10);
      expect(error.validRange).toEqual([0, 5]);
    });

    it("should create IndexError without details", () => {
      const error = new IndexError("index out of bounds");
      expect(error.index).toBeUndefined();
      expect(error.validRange).toBeUndefined();
    });

    it("should create BroadcastError", () => {
      const error = new BroadcastError([2, 3], [3, 4]);
      expect(error).toBeInstanceOf(DeepboxError);
      expect(error.name).toBe("BroadcastError");
      expect(error.shape1).toEqual([2, 3]);
      expect(error.shape2).toEqual([3, 4]);
    });

    it("should snapshot shapes in BroadcastError", () => {
      const shape1 = [2, 3];
      const shape2 = [3, 4];
      const error = new BroadcastError(shape1, shape2);
      shape1[0] = 99;
      shape2[1] = 88;
      expect(error.shape1).toEqual([2, 3]);
      expect(error.shape2).toEqual([3, 4]);
    });

    it("should create BroadcastError with context", () => {
      const error = new BroadcastError([2, 3], [3, 4], "addition");
      expect(error.message).toContain("addition");
      expect(error.message).toContain("[2,3]");
      expect(error.message).toContain("[3,4]");
    });

    it("should create InvalidParameterError", () => {
      const error = new InvalidParameterError("invalid param");
      expect(error).toBeInstanceOf(DeepboxError);
      expect(error.name).toBe("InvalidParameterError");
    });

    it("should create InvalidParameterError with parameter details", () => {
      const error = new InvalidParameterError("k must be positive", "k", -1);
      expect(error.parameterName).toBe("k");
      expect(error.value).toBe(-1);
    });

    it("should create DataValidationError", () => {
      const error = new DataValidationError("validation failed");
      expect(error).toBeInstanceOf(DeepboxError);
      expect(error.name).toBe("DataValidationError");
    });

    it("should create NotImplementedError", () => {
      const error = new NotImplementedError("not implemented");
      expect(error).toBeInstanceOf(DeepboxError);
      expect(error.name).toBe("NotImplementedError");
    });

    it("should create NotImplementedError with default message", () => {
      const error = new NotImplementedError();
      expect(error.message).toBe("Not implemented");
    });

    it("should create NotFittedError", () => {
      const error = new NotFittedError("model not fitted", "MyModel");
      expect(error).toBeInstanceOf(DeepboxError);
      expect(error.name).toBe("NotFittedError");
      expect(error.modelName).toBe("MyModel");
    });

    it("should create ConvergenceError", () => {
      const error = new ConvergenceError("did not converge");
      expect(error).toBeInstanceOf(DeepboxError);
      expect(error.name).toBe("ConvergenceError");
    });

    it("should create ConvergenceError with details", () => {
      const error = new ConvergenceError("did not converge", {
        iterations: 1000,
        tolerance: 1e-6,
      });
      expect(error.iterations).toBe(1000);
      expect(error.tolerance).toBe(1e-6);
    });

    it("should create MemoryError", () => {
      const error = new MemoryError("out of memory");
      expect(error).toBeInstanceOf(DeepboxError);
      expect(error.name).toBe("MemoryError");
    });

    it("should create MemoryError with memory details", () => {
      const error = new MemoryError("allocation failed", {
        requestedBytes: 1000000000,
        availableBytes: 500000000,
      });
      expect(error.requestedBytes).toBe(1000000000);
      expect(error.availableBytes).toBe(500000000);
    });
  });
});
