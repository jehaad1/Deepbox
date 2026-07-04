import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { describe, expect, it } from "vitest";
import type {
  SerializedEstimator,
  SerializedModuleState,
  SerializedTensor,
} from "../src/core/serialization";
import { fromJSON, load, save, toJSON } from "../src/core/serialization";

describe("Serialization - toJSON / fromJSON", () => {
  it("round-trips a SerializedTensor", () => {
    const payload: SerializedTensor = {
      __type: "Tensor",
      data: [1, 2, 3, 4],
      shape: [2, 2],
      dtype: "float32",
    };
    const json = toJSON(payload);
    const restored = fromJSON(json);
    expect(restored).toEqual(payload);
  });

  it("round-trips a SerializedModuleState", () => {
    const payload: SerializedModuleState = {
      __type: "ModuleState",
      parameters: {
        "fc1.weight": { data: [0.1, 0.2, 0.3], dtype: "float32", shape: [1, 3] },
        "fc1.bias": { data: [0.5], dtype: "float32", shape: [1] },
      },
      buffers: {},
    };
    const json = toJSON(payload);
    const restored = fromJSON(json);
    expect(restored).toEqual(payload);
  });

  it("round-trips a SerializedEstimator", () => {
    const payload: SerializedEstimator = {
      __type: "Estimator",
      className: "LogisticRegression",
      params: { C: 1.0, penalty: "l2", maxIter: 100 },
      state: { coef: [0.1, 0.2], intercept: 0.5, fitted: true },
    };
    const json = toJSON(payload);
    const restored = fromJSON(json);
    expect(restored).toEqual(payload);
  });

  it("handles BigInt values in tensor data", () => {
    const payload: SerializedTensor = {
      __type: "Tensor",
      data: [BigInt(1), BigInt(2), BigInt(3)],
      shape: [3],
      dtype: "int64",
    };
    const json = toJSON(payload);
    const restored = fromJSON(json);
    expect(restored).toEqual(payload);
  });

  it("throws for invalid JSON", () => {
    expect(() => fromJSON("not json")).toThrow();
  });

  it("throws for unknown __type", () => {
    expect(() => fromJSON('{"__type":"Unknown"}')).toThrow("unknown __type");
  });

  it("throws for non-object payload", () => {
    expect(() => fromJSON('"hello"')).toThrow("expected an object");
  });
});

describe("Serialization - save / load (Node.js)", () => {
  it("saves and loads a tensor payload to disk", async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "deepbox-test-"));
    const filePath = path.join(tmpDir, "tensor.json");

    const payload: SerializedTensor = {
      __type: "Tensor",
      data: [1.5, 2.5, 3.5],
      shape: [3],
      dtype: "float64",
    };

    await save(filePath, payload);
    expect(fs.existsSync(filePath)).toBe(true);

    const restored = await load(filePath);
    expect(restored).toEqual(payload);

    // Cleanup
    fs.unlinkSync(filePath);
    fs.rmdirSync(tmpDir);
  });

  it("saves and loads a module state payload", async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "deepbox-test-"));
    const filePath = path.join(tmpDir, "module.json");

    const payload: SerializedModuleState = {
      __type: "ModuleState",
      parameters: {
        weight: { data: [0.1, 0.2], dtype: "float32", shape: [2] },
      },
      buffers: {
        running_mean: { data: [0.0], dtype: "float32", shape: [1] },
      },
    };

    await save(filePath, payload);
    const restored = await load(filePath);
    expect(restored).toEqual(payload);

    // Cleanup
    fs.unlinkSync(filePath);
    fs.rmdirSync(tmpDir);
  });

  it("throws for empty path", async () => {
    const payload: SerializedTensor = {
      __type: "Tensor",
      data: [1],
      shape: [1],
      dtype: "float32",
    };
    await expect(save("", payload)).rejects.toThrow("non-empty string");
  });

  it("throws for invalid load path", async () => {
    await expect(load("/nonexistent/file.json")).rejects.toThrow();
  });
});
