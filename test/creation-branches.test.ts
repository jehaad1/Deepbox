import { describe, expect, it } from "vitest";
import { arange, full, linspace, logspace, tensor, zeros } from "../src/ndarray";

const toUnknownArray = (value: { toArray: () => unknown }): unknown => value.toArray();

describe("deepbox/ndarray - Creation Branches", () => {
  it("validates string tensor shapes and types", () => {
    expect(() => tensor([["a"], ["b", "c"]], { dtype: "string" })).toThrow(/Ragged/);
    // @ts-expect-error - mixed string/number data should be rejected at runtime
    expect(() => tensor([["a"], [1]], { dtype: "string" })).toThrow(/must be strings/);
  });

  it("validates numeric tensor shapes and types", () => {
    expect(() => tensor([[1], [2, 3]])).toThrow(/Ragged/);
    // @ts-expect-error - mixed number/string data should be rejected at runtime
    expect(() => tensor([1, "x"])).toThrow(/leaf values must be numbers/);
  });

  it("enforces int64 values and bool coercion", () => {
    expect(() => tensor([1.5], { dtype: "int64" })).toThrow(/finite integers/);

    const boolT = tensor([0, 2, -1], { dtype: "bool" });
    expect(boolT.toArray()).toEqual([0, 1, 1]);
  });

  it("handles full with string dtype and value validation", () => {
    const s = full([2], "x", { dtype: "string" });
    expect(s.toArray()).toEqual(["x", "x"]);
    expect(() => full([1], 1, { dtype: "string" })).toThrow(/string fill value/);
  });

  it("validates arange and linspace parameters", () => {
    expect(() => arange(0, 10, 0)).toThrow(/step must be non-zero/);
    expect(() => linspace(0, 1, -1)).toThrow(/non-negative/);

    const noEndpoint = linspace(0, 10, 5, false);
    expect(noEndpoint.toArray()).toEqual([0, 2, 4, 6, 8]);
  });

  it("handles logspace BigInt dtype", () => {
    const t = logspace(0, 2, 3, 10, true, { dtype: "int64" });
    expect(t.dtype).toBe("int64");
  });

  it("handles empty string tensor creation", () => {
    const t = tensor([], { dtype: "string" });
    expect(t.shape).toEqual([0]);
  });

  it("zeros handles string dtype", () => {
    const t = zeros([2], { dtype: "string" });
    expect(toUnknownArray(t)).toEqual(["", ""]);
  });

  it("infers dtype from TypedArray inputs and validates mismatches", () => {
    const f32 = tensor(new Float32Array([1, 2]));
    expect(f32.dtype).toBe("float32");

    const u8 = tensor(new Uint8Array([1, 0]));
    expect(u8.dtype).toBe("uint8");

    const boolT = tensor(new Uint8Array([1, 0]), { dtype: "bool" });
    expect(boolT.dtype).toBe("bool");

    expect(() => tensor(new Float32Array([1, 2]), { dtype: "int32" })).toThrow(/TypedArray/i);
  });
});
