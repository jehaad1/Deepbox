import { describe, expect, it } from "vitest";
import { DataLoader } from "../src/datasets";
import { tensor } from "../src/ndarray";
import { numRawData } from "./_helpers";

describe("DataLoader", () => {
  it("should create DataLoader instance", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ]);
    const y = tensor([0, 1, 0, 1]);
    const loader = new DataLoader(X, y, { batchSize: 2 });
    expect(loader).toBeDefined();
  });

  it("should have correct length", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ]);
    const y = tensor([0, 1, 0, 1]);
    const loader = new DataLoader(X, y, { batchSize: 2 });
    expect(loader.length).toBe(2);
  });

  it("should drop last incomplete batch if dropLast is true", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const loader = new DataLoader(X, undefined, {
      batchSize: 2,
      dropLast: true,
    });
    expect(loader.length).toBe(1);
  });

  it("should iterate over batches", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ]);
    const y = tensor([0, 1, 0, 1]);
    const loader = new DataLoader(X, y, { batchSize: 2 });

    const batches = Array.from(loader);
    expect(batches).toHaveLength(2);
    const b0 = batches[0];
    const b1 = batches[1];
    expect(b0).toBeDefined();
    expect(b1).toBeDefined();
    if (!b0 || !b1) return;
    expect(b0[0].shape[0]).toBe(2);
    expect(b1[0].shape[0]).toBe(2);
    expect(b0.length).toBe(2);
    expect(b1.length).toBe(2);
    expect(b0[1].shape[0]).toBe(2);
    expect(b1[1].shape[0]).toBe(2);
  });

  it("should include a final smaller batch when dropLast is false", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const loader = new DataLoader(X, undefined, {
      batchSize: 2,
      dropLast: false,
    });
    const batches = Array.from(loader);
    expect(batches).toHaveLength(2);
    const b0 = batches[0];
    const b1 = batches[1];
    expect(b0).toBeDefined();
    expect(b1).toBeDefined();
    if (!b0 || !b1) return;
    expect(b0[0].shape[0]).toBe(2);
    expect(b1[0].shape[0]).toBe(1);
  });

  it("should shuffle deterministically when seed is provided", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ]);

    const loaderA = new DataLoader(X, undefined, {
      batchSize: 2,
      shuffle: true,
      seed: 123,
    });
    const loaderB = new DataLoader(X, undefined, {
      batchSize: 2,
      shuffle: true,
      seed: 123,
    });

    const a0 = Array.from(loaderA)[0];
    const b0 = Array.from(loaderB)[0];
    expect(a0).toBeDefined();
    expect(b0).toBeDefined();
    if (!a0 || !b0) return;
    const aX = a0[0];
    const bX = b0[0];
    expect(aX.shape).toEqual(bX.shape);
    const aData = numRawData(aX.data);
    const bData = numRawData(bX.data);
    expect(aData.slice(0, 4)).toEqual(bData.slice(0, 4));
  });

  it("should throw for invalid batchSize", () => {
    const X = tensor([[1, 2]]);
    expect(() => new DataLoader(X, undefined, { batchSize: 0 })).toThrow();
  });

  it("should throw when X and y have different number of samples", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
    ]);
    const y = tensor([1]);
    expect(() => new DataLoader(X, y, { batchSize: 1 })).toThrow();
  });
});
