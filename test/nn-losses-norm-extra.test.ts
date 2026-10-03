import { describe, expect, it } from "vitest";
import { GradTensor, tensor } from "../src/ndarray";
import {
  BatchNorm3d,
  InstanceNorm1d,
  InstanceNorm2d,
  InstanceNorm3d,
  LocalResponseNorm,
  marginRankingLoss,
} from "../src/nn";

const f32 = { dtype: "float32" as const };

describe("marginRankingLoss", () => {
  it("should compute zero loss when ranking is correct", () => {
    const x1 = tensor([3, 5, 7], f32);
    const x2 = tensor([1, 2, 3], f32);
    const y = tensor([1, 1, 1], f32);
    const loss = marginRankingLoss(x1, x2, y);
    expect(loss.size).toBe(1);
    expect(Number(loss.data[0])).toBeCloseTo(0, 5);
  });

  it("should compute positive loss when ranking is wrong", () => {
    const x1 = tensor([1, 2], f32);
    const x2 = tensor([3, 5], f32);
    const y = tensor([1, 1], f32);
    const loss = marginRankingLoss(x1, x2, y);
    expect(Number(loss.data[0])).toBeGreaterThan(0);
  });

  it("should support margin parameter", () => {
    const x1 = tensor([3], f32);
    const x2 = tensor([2], f32);
    const y = tensor([1], f32);
    const loss = marginRankingLoss(x1, x2, y, 2.0);
    // -1 * (3-2) + 2 = 1
    expect(Number(loss.data[0])).toBeCloseTo(1, 5);
  });

  it("should support reduction='none'", () => {
    const x1 = tensor([3, 1], f32);
    const x2 = tensor([1, 3], f32);
    const y = tensor([1, 1], f32);
    const loss = marginRankingLoss(x1, x2, y, 0, "none");
    expect(loss.shape).toEqual([2]);
  });

  it("should support reduction='sum'", () => {
    const x1 = tensor([3, 1], f32);
    const x2 = tensor([1, 3], f32);
    const y = tensor([1, 1], f32);
    const loss = marginRankingLoss(x1, x2, y, 0, "sum");
    expect(loss.size).toBe(1);
  });

  it("should throw on shape mismatch", () => {
    const x1 = tensor([1, 2, 3], f32);
    const x2 = tensor([1, 2], f32);
    const y = tensor([1, 1, 1], f32);
    expect(() => marginRankingLoss(x1, x2, y)).toThrow();
  });
});

describe("BatchNorm3d", () => {
  it("should accept 5D input", () => {
    const bn = new BatchNorm3d(2);
    const x = tensor(
      Array.from({ length: 2 * 2 * 2 * 2 * 2 }, (_, i) => i + 1),
      f32
    ).reshape([2, 2, 2, 2, 2]);
    const out = bn.forward(x);
    expect(out.shape).toEqual([2, 2, 2, 2, 2]);
  });

  it("should throw on non-5D input", () => {
    const bn = new BatchNorm3d(2);
    const x = tensor(
      [
        [1, 2],
        [3, 4],
      ],
      f32
    );
    expect(() => bn.forward(x)).toThrow();
  });

  it("should throw on invalid numFeatures", () => {
    expect(() => new BatchNorm3d(0)).toThrow();
    expect(() => new BatchNorm3d(-1)).toThrow();
  });

  it("should have correct string representation", () => {
    const bn = new BatchNorm3d(64);
    expect(bn.toString()).toContain("BatchNorm3d");
    expect(bn.toString()).toContain("64");
  });
});

describe("InstanceNorm1d", () => {
  it("should normalize 3D input", () => {
    const norm = new InstanceNorm1d(2);
    const x = tensor(
      Array.from({ length: 2 * 2 * 4 }, (_, i) => i + 1),
      f32
    ).reshape([2, 2, 4]);
    const out = norm.forward(x);
    expect(out.shape).toEqual([2, 2, 4]);
  });

  it("should have correct string representation", () => {
    const norm = new InstanceNorm1d(32);
    expect(norm.toString()).toContain("InstanceNorm1d");
  });
});

describe("InstanceNorm2d", () => {
  it("should normalize 4D input", () => {
    const norm = new InstanceNorm2d(2);
    const x = tensor(
      Array.from({ length: 2 * 2 * 3 * 3 }, (_, i) => i + 1),
      f32
    ).reshape([2, 2, 3, 3]);
    const out = norm.forward(x);
    expect(out.shape).toEqual([2, 2, 3, 3]);
  });

  it("should have correct string representation", () => {
    const norm = new InstanceNorm2d(64);
    expect(norm.toString()).toContain("InstanceNorm2d");
  });
});

describe("InstanceNorm3d", () => {
  it("should normalize 5D input", () => {
    const norm = new InstanceNorm3d(2);
    const x = tensor(
      Array.from({ length: 1 * 2 * 2 * 2 * 2 }, (_, i) => i + 1),
      f32
    ).reshape([1, 2, 2, 2, 2]);
    const out = norm.forward(x);
    expect(out.shape).toEqual([1, 2, 2, 2, 2]);
  });

  it("should have correct string representation", () => {
    const norm = new InstanceNorm3d(16);
    expect(norm.toString()).toContain("InstanceNorm3d");
  });
});

describe("LocalResponseNorm", () => {
  it("should normalize 3D input across channels", () => {
    const lrn = new LocalResponseNorm(3);
    const x = tensor(
      Array.from({ length: 2 * 4 * 3 }, (_, i) => i + 1),
      f32
    ).reshape([2, 4, 3]);
    const out = lrn.forward(x);
    expect(out.shape).toEqual([2, 4, 3]);
  });

  it("should normalize 4D input (images)", () => {
    const lrn = new LocalResponseNorm(5);
    const x = tensor(
      Array.from({ length: 1 * 3 * 4 * 4 }, (_, i) => i + 1),
      f32
    ).reshape([1, 3, 4, 4]);
    const out = lrn.forward(x);
    expect(out.shape).toEqual([1, 3, 4, 4]);
  });

  it("should reduce values (normalization shrinks magnitudes)", () => {
    const lrn = new LocalResponseNorm(3, { alpha: 1.0, beta: 1.0, k: 0 });
    const x = tensor([1, 2, 3, 4, 5, 6], f32).reshape([1, 2, 3]);
    const out = lrn.forward(x);
    for (let i = 0; i < out.size; i++) {
      expect(Math.abs(Number(out.data[i]))).toBeLessThanOrEqual(Math.abs(Number(x.data[i])) + 1e-6);
    }
  });

  it("should throw on less than 3D input", () => {
    const lrn = new LocalResponseNorm(3);
    const x = tensor(
      [
        [1, 2],
        [3, 4],
      ],
      f32
    );
    expect(() => lrn.forward(x)).toThrow();
  });

  it("should throw on invalid size", () => {
    expect(() => new LocalResponseNorm(0)).toThrow();
    expect(() => new LocalResponseNorm(-1)).toThrow();
  });

  it("should have correct string representation", () => {
    const lrn = new LocalResponseNorm(5);
    expect(lrn.toString()).toContain("LocalResponseNorm");
    expect(lrn.toString()).toContain("5");
  });

  it("propagates gradients to the input (autograd correctness)", () => {
    const lrn = new LocalResponseNorm(3, { alpha: 1.0, beta: 0.75, k: 1 });
    const f64 = { dtype: "float64" as const };
    const x0 = [1, 2, 3, 4, 5, 6, 0.5, -1, 2, 1, 0.3, 0.7];
    const shape = [2, 3, 2];
    const mk = (a: number[]) => tensor(a, f64).reshape(shape);

    const x = GradTensor.fromTensor(mk(x0), { requiresGrad: true });
    lrn.forward(x).sum().backward();
    const grad = x.grad;
    expect(grad).not.toBeNull();
    const g = grad as NonNullable<typeof grad>;
    const analytic = Array.from(g.data as Float64Array).slice(g.offset, g.offset + x0.length);

    // Central-difference numerical gradient of sum(LRN(x)) w.r.t. each input.
    const eps = 1e-5;
    let maxDiff = 0;
    for (let i = 0; i < x0.length; i++) {
      const xp = [...x0];
      xp[i] = (xp[i] ?? 0) + eps;
      const xm = [...x0];
      xm[i] = (xm[i] ?? 0) - eps;
      const fp = Number(
        (lrn.forward(GradTensor.fromTensor(mk(xp))).sum().tensor.data as Float64Array)[0]
      );
      const fm = Number(
        (lrn.forward(GradTensor.fromTensor(mk(xm))).sum().tensor.data as Float64Array)[0]
      );
      const numeric = (fp - fm) / (2 * eps);
      maxDiff = Math.max(maxDiff, Math.abs((analytic[i] ?? 0) - numeric));
    }
    expect(maxDiff).toBeLessThan(1e-5);
  });
});
