import { describe, expect, it } from "vitest";
import { filterDataset, loadDigits, mapDataset } from "../../src/datasets";
import { tensor, zeros } from "../../src/ndarray";

describe("DF-14 mapDataset keeps trailing dims and images", () => {
  const ds = {
    data: zeros([4, 2, 3]),
    target: tensor([0, 1, 0, 1], { dtype: "int32" }),
    featureNames: ["a", "b", "c", "d", "e", "f"],
    description: "x",
  };

  it("preserves N-D sample shape like filterDataset", () => {
    const mapped = mapDataset(ds, (d, y) => ({ data: d, target: y }));
    expect(mapped.data.shape).toEqual([4, 2, 3]);
    expect(filterDataset(ds, () => true).data.shape).toEqual([4, 2, 3]);
  });

  it("flattens to [n, width] only when the callback changes the length", () => {
    const mapped = mapDataset(ds, (d, y) => ({ data: d.slice(0, 2), target: y }));
    expect(mapped.data.shape).toEqual([4, 2]);
  });

  it("carries images through", () => {
    const digits = loadDigits();
    const mapped = mapDataset(digits, (d, y) => ({ data: d, target: y }));
    expect(mapped.images).toBeDefined();
    expect(mapped.images?.shape).toEqual([1797, 8, 8]);
    expect(mapped.data.shape).toEqual([1797, 64]);
    expect(mapped.images?.at(5, 3, 3)).toBe(digits.images?.at(5, 3, 3));
  });
});
