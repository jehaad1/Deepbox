import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { KBinsDiscretizer } from "../src/preprocess";

describe("KBinsDiscretizer", () => {
  it("uniform strategy: bins evenly spaced", () => {
    const X = tensor([[0], [5], [10], [15], [20]]);
    const kbd = new KBinsDiscretizer({ nBins: 4, strategy: "uniform" });
    kbd.fit(X);
    expect(kbd.binEdges.length).toBe(1);
    expect(kbd.binEdges[0]!.length).toBe(5); // nBins + 1 edges
    // Edges: 0, 5, 10, 15, 20
    expect(kbd.binEdges[0]![0]).toBeCloseTo(0, 10);
    expect(kbd.binEdges[0]![4]).toBeCloseTo(20, 10);
  });

  it("uniform strategy: transforms correctly", () => {
    const X = tensor([[0], [5], [10], [15], [20]]);
    const kbd = new KBinsDiscretizer({ nBins: 4, strategy: "uniform" });
    const result = kbd.fitTransform(X);
    expect(result.shape).toEqual([5, 1]);
    const arr = result.toArray() as number[][];
    // 0 → bin 0, 5 → bin 1, 10 → bin 2, 15 → bin 3, 20 → bin 3
    expect(arr[0]![0]).toBe(0);
    expect(arr[4]![0]).toBe(3);
  });

  it("quantile strategy: bins have equal frequency", () => {
    const X = tensor([[1], [2], [3], [4], [5], [6], [7], [8]]);
    const kbd = new KBinsDiscretizer({ nBins: 4, strategy: "quantile" });
    const result = kbd.fitTransform(X);
    expect(result.shape).toEqual([8, 1]);
    const arr = result.toArray() as number[][];
    // Each bin should contain roughly 2 samples
    const bins = arr.map((r) => r[0]!);
    expect(new Set(bins).size).toBeGreaterThanOrEqual(2);
  });

  it("handles multiple features", () => {
    const X = tensor([
      [0, 100],
      [5, 200],
      [10, 300],
      [15, 400],
    ]);
    const kbd = new KBinsDiscretizer({ nBins: 2, strategy: "uniform" });
    kbd.fit(X);
    expect(kbd.binEdges.length).toBe(2);
    const result = kbd.transform(X);
    expect(result.shape).toEqual([4, 2]);
  });

  it("default nBins is 5", () => {
    const kbd = new KBinsDiscretizer();
    const X = tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]);
    kbd.fit(X);
    expect(kbd.binEdges[0]!.length).toBe(6); // 5 bins + 1
  });

  it("throws on nBins < 2", () => {
    expect(() => new KBinsDiscretizer({ nBins: 1 })).toThrow();
  });

  it("throws when not fitted", () => {
    const kbd = new KBinsDiscretizer();
    expect(() => kbd.binEdges).toThrow("not fitted");
    expect(() => kbd.transform(tensor([[1]]))).toThrow("not fitted");
  });

  it("throws on wrong feature count in transform", () => {
    const kbd = new KBinsDiscretizer({ nBins: 3 });
    kbd.fit(
      tensor([
        [1, 2],
        [3, 4],
      ])
    );
    expect(() => kbd.transform(tensor([[1]]))).toThrow();
  });
});
