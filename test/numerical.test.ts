import { describe, expect, it } from "vitest";
import {
  column_stack,
  digitize,
  gradient,
  hstack,
  interp,
  tensor,
  trapz,
  vstack,
} from "../src/ndarray";

// ─── interp ───────────────────────────────────────────────────────────────────

describe("interp", () => {
  it("interpolates linearly between data points", () => {
    const xp = tensor([0, 1, 2, 3]);
    const fp = tensor([0, 1, 4, 9]);
    const x = tensor([0.5, 1.5, 2.5]);
    const y = interp(x, xp, fp);
    expect(y.shape).toEqual([3]);
    expect(y.at(0)).toBeCloseTo(0.5);
    expect(y.at(1)).toBeCloseTo(2.5);
    expect(y.at(2)).toBeCloseTo(6.5);
  });

  it("clamps to boundary values by default", () => {
    const xp = tensor([0, 1, 2]);
    const fp = tensor([10, 20, 30]);
    const x = tensor([-1, 3]);
    const y = interp(x, xp, fp);
    expect(y.at(0)).toBeCloseTo(10); // clamp left
    expect(y.at(1)).toBeCloseTo(30); // clamp right
  });

  it("uses custom left/right fill values", () => {
    const xp = tensor([0, 1, 2]);
    const fp = tensor([10, 20, 30]);
    const x = tensor([-1, 3]);
    const y = interp(x, xp, fp, -999, 999);
    expect(y.at(0)).toBeCloseTo(-999);
    expect(y.at(1)).toBeCloseTo(999);
  });

  it("returns exact values at data points", () => {
    const xp = tensor([0, 1, 2, 3]);
    const fp = tensor([0, 10, 20, 30]);
    const x = tensor([0, 1, 2, 3]);
    const y = interp(x, xp, fp);
    for (let i = 0; i < 4; i++) {
      expect(y.at(i)).toBeCloseTo(i * 10);
    }
  });

  it("throws on non-1D xp", () => {
    expect(() => interp(tensor([1]), tensor([[1, 2]]), tensor([1, 2]))).toThrow();
  });

  it("throws on unsorted xp", () => {
    expect(() => interp(tensor([1]), tensor([2, 1]), tensor([1, 2]))).toThrow();
  });

  it("throws on mismatched xp/fp lengths", () => {
    expect(() => interp(tensor([1]), tensor([1, 2, 3]), tensor([1, 2]))).toThrow();
  });
});

// ─── trapz ────────────────────────────────────────────────────────────────────

describe("trapz", () => {
  it("integrates constant function", () => {
    const y = tensor([2, 2, 2, 2]);
    expect(trapz(y)).toBeCloseTo(6); // 3 intervals × 2 = 6
  });

  it("integrates linear function", () => {
    const y = tensor([0, 1, 2, 3]);
    expect(trapz(y)).toBeCloseTo(4.5);
  });

  it("respects dx parameter", () => {
    const y = tensor([0, 1, 2, 3]);
    expect(trapz(y, undefined, 0.5)).toBeCloseTo(2.25);
  });

  it("handles non-uniform spacing with x", () => {
    const y = tensor([1, 2, 3, 4]);
    const x = tensor([0, 1, 3, 5]);
    expect(trapz(y, x)).toBeCloseTo(13.5);
  });

  it("returns 0 for single element", () => {
    expect(trapz(tensor([5]))).toBe(0);
  });

  it("throws on non-1D input", () => {
    expect(() =>
      trapz(
        tensor([
          [1, 2],
          [3, 4],
        ])
      )
    ).toThrow();
  });
});

// ─── gradient ─────────────────────────────────────────────────────────────────

describe("gradient", () => {
  it("computes gradient of linear function", () => {
    const f = tensor([1, 3, 5, 7, 9]);
    const g = gradient(f);
    // Constant gradient = 2
    for (let i = 0; i < 5; i++) {
      expect(g.at(i)).toBeCloseTo(2);
    }
  });

  it("computes gradient of quadratic function", () => {
    const f = tensor([0, 1, 4, 9, 16]); // x^2
    const g = gradient(f);
    expect(g.at(0)).toBeCloseTo(1); // forward: (1-0)/1
    expect(g.at(1)).toBeCloseTo(2); // central: (4-0)/2
    expect(g.at(2)).toBeCloseTo(4); // central: (9-1)/2
    expect(g.at(3)).toBeCloseTo(6); // central: (16-4)/2
    expect(g.at(4)).toBeCloseTo(7); // backward: (16-9)/1
  });

  it("respects scalar spacing", () => {
    const f = tensor([0, 1, 4, 9]);
    const g = gradient(f, 0.5);
    expect(g.at(0)).toBeCloseTo(2); // (1-0)/0.5
    expect(g.at(1)).toBeCloseTo(4); // (4-0)/(2*0.5)
  });

  it("handles non-uniform spacing via tensor", () => {
    const f = tensor([0, 1, 8]);
    const x = tensor([0, 1, 4]);
    const g = gradient(f, x);
    expect(g.shape).toEqual([3]);
    // Forward: (1-0)/(1-0) = 1
    expect(g.at(0)).toBeCloseTo(1);
  });

  it("throws on non-1D input", () => {
    expect(() => gradient(tensor([[1, 2]]))).toThrow();
  });

  it("throws on single element", () => {
    expect(() => gradient(tensor([1]))).toThrow();
  });
});

// ─── digitize ─────────────────────────────────────────────────────────────────

describe("digitize", () => {
  it("bins values correctly (left-closed)", () => {
    const x = tensor([0.5, 1.5, 2.5, 3.5]);
    const bins = tensor([1, 2, 3]);
    const result = digitize(x, bins);
    expect(result.at(0)).toBe(0); // 0.5 < 1
    expect(result.at(1)).toBe(1); // 1 <= 1.5 < 2
    expect(result.at(2)).toBe(2); // 2 <= 2.5 < 3
    expect(result.at(3)).toBe(3); // 3.5 >= 3
  });

  it("bins values correctly (right-closed)", () => {
    const x = tensor([1, 2, 3]);
    const bins = tensor([1, 2, 3]);
    const result = digitize(x, bins, true);
    expect(result.at(0)).toBe(0); // 1 is not > any bin (right=true means bins[i] < val)
    expect(result.at(1)).toBe(1); // 2 > 1, but not > 2
    expect(result.at(2)).toBe(2); // 3 > 1, 3 > 2, but not > 3
  });

  it("handles edge case with exact bin values", () => {
    const x = tensor([1, 2, 3]);
    const bins = tensor([1, 2, 3]);
    const result = digitize(x, bins, false);
    // With right=false (default): bins[i] <= val, so val=1 gets bin 1
    expect(result.at(0)).toBe(1);
    expect(result.at(1)).toBe(2);
    expect(result.at(2)).toBe(3);
  });

  it("preserves input shape", () => {
    const x = tensor([
      [1, 2],
      [3, 4],
    ]);
    const bins = tensor([1.5, 2.5, 3.5]);
    const result = digitize(x, bins);
    expect(result.shape).toEqual([2, 2]);
  });

  it("throws on non-1D bins", () => {
    expect(() => digitize(tensor([1]), tensor([[1, 2]]))).toThrow();
  });

  it("throws on unsorted bins", () => {
    expect(() => digitize(tensor([1]), tensor([3, 1]))).toThrow();
  });
});

// ─── vstack ───────────────────────────────────────────────────────────────────

describe("vstack", () => {
  it("stacks 1D arrays as rows", () => {
    const a = tensor([1, 2, 3]);
    const b = tensor([4, 5, 6]);
    const result = vstack([a, b]);
    expect(result.shape).toEqual([2, 3]);
    expect(result.at(0, 0)).toBeCloseTo(1);
    expect(result.at(0, 2)).toBeCloseTo(3);
    expect(result.at(1, 0)).toBeCloseTo(4);
    expect(result.at(1, 2)).toBeCloseTo(6);
  });

  it("concatenates 2D arrays along axis 0", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
    ]);
    const b = tensor([[5, 6]]);
    const result = vstack([a, b]);
    expect(result.shape).toEqual([3, 2]);
    expect(result.at(2, 0)).toBeCloseTo(5);
  });

  it("throws on empty input", () => {
    expect(() => vstack([])).toThrow();
  });
});

// ─── hstack ───────────────────────────────────────────────────────────────────

describe("hstack", () => {
  it("concatenates 1D arrays", () => {
    const a = tensor([1, 2]);
    const b = tensor([3, 4]);
    const result = hstack([a, b]);
    expect(result.shape).toEqual([4]);
    expect(result.at(0)).toBeCloseTo(1);
    expect(result.at(3)).toBeCloseTo(4);
  });

  it("concatenates 2D arrays along axis 1", () => {
    const a = tensor([[1], [2]]);
    const b = tensor([[3], [4]]);
    const result = hstack([a, b]);
    expect(result.shape).toEqual([2, 2]);
    expect(result.at(0, 1)).toBeCloseTo(3);
    expect(result.at(1, 1)).toBeCloseTo(4);
  });

  it("throws on empty input", () => {
    expect(() => hstack([])).toThrow();
  });
});

// ─── column_stack ─────────────────────────────────────────────────────────────

describe("column_stack", () => {
  it("stacks 1D arrays as columns", () => {
    const a = tensor([1, 2, 3]);
    const b = tensor([4, 5, 6]);
    const result = column_stack([a, b]);
    expect(result.shape).toEqual([3, 2]);
    expect(result.at(0, 0)).toBeCloseTo(1);
    expect(result.at(0, 1)).toBeCloseTo(4);
    expect(result.at(2, 0)).toBeCloseTo(3);
    expect(result.at(2, 1)).toBeCloseTo(6);
  });

  it("handles 2D inputs", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
    ]);
    const b = tensor([[5], [6]]);
    const result = column_stack([a, b]);
    expect(result.shape).toEqual([2, 3]);
    expect(result.at(0, 2)).toBeCloseTo(5);
    expect(result.at(1, 2)).toBeCloseTo(6);
  });

  it("throws on empty input", () => {
    expect(() => column_stack([])).toThrow();
  });

  it("throws on 3D input", () => {
    const a = tensor([[[1]]]);
    expect(() => column_stack([a])).toThrow();
  });
});
