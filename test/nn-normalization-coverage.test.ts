import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import {
  BatchNorm1d,
  BatchNorm2d,
  GroupNorm,
  InstanceNorm,
  LayerNorm,
  RMSNorm,
} from "../src/nn/layers/normalization";

const f32 = { dtype: "float32" as const };

describe("BatchNorm1d", () => {
  it("normalizes 2D input in training mode", () => {
    const bn = new BatchNorm1d(3);
    const x = tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ],
      f32
    );
    const out = bn.forward(x);
    expect(out.shape).toEqual([3, 3]);
  });

  it("normalizes 3D input in training mode", () => {
    const bn = new BatchNorm1d(2);
    const x = tensor(
      Array.from({ length: 12 }, (_, i) => i),
      f32
    ).reshape([3, 2, 2]);
    const out = bn.forward(x);
    expect(out.shape).toEqual([3, 2, 2]);
  });

  it("uses running stats in eval mode", () => {
    const bn = new BatchNorm1d(3);
    const x = tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      f32
    );
    bn.train();
    bn.forward(x);
    bn.eval();
    const out = bn.forward(x);
    expect(out.shape).toEqual([2, 3]);
  });

  it("works with affine=false", () => {
    const bn = new BatchNorm1d(3, { affine: false });
    const x = tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      f32
    );
    const out = bn.forward(x);
    expect(out.shape).toEqual([2, 3]);
  });

  it("works with trackRunningStats=false", () => {
    const bn = new BatchNorm1d(3, { trackRunningStats: false });
    const x = tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      f32
    );
    const out = bn.forward(x);
    expect(out.shape).toEqual([2, 3]);
  });

  it("validates constructor params", () => {
    expect(() => new BatchNorm1d(0)).toThrow(/positive integer/);
    expect(() => new BatchNorm1d(-1)).toThrow(/positive integer/);
    expect(() => new BatchNorm1d(3, { eps: 0 })).toThrow(/eps/);
  });

  it("validates input shape", () => {
    const bn = new BatchNorm1d(3);
    expect(() => bn.forward(tensor([1, 2, 3], f32))).toThrow();
  });

  it("validates feature count mismatch", () => {
    const bn = new BatchNorm1d(3);
    expect(() => bn.forward(tensor([[1, 2]], f32))).toThrow(/channels/i);
  });

  it("toString", () => {
    const bn = new BatchNorm1d(3);
    expect(bn.toString()).toContain("BatchNorm1d");
  });
});

describe("LayerNorm", () => {
  it("normalizes with number shape", () => {
    const ln = new LayerNorm(4);
    const x = tensor([[1, 2, 3, 4]], f32);
    const out = ln.forward(x);
    expect(out.shape).toEqual([1, 4]);
  });

  it("normalizes with array shape", () => {
    const ln = new LayerNorm([4]);
    const x = tensor([[1, 2, 3, 4]], f32);
    const out = ln.forward(x);
    expect(out.shape).toEqual([1, 4]);
  });

  it("works with elementwiseAffine=false", () => {
    const ln = new LayerNorm(4, { elementwiseAffine: false });
    const x = tensor([[1, 2, 3, 4]], f32);
    const out = ln.forward(x);
    expect(out.shape).toEqual([1, 4]);
  });

  it("validates normalizedShape", () => {
    expect(() => new LayerNorm([])).toThrow(/at least one/);
    expect(() => new LayerNorm([0])).toThrow(/positive integers/);
    expect(() => new LayerNorm([-1])).toThrow(/positive integers/);
  });

  it("validates eps", () => {
    expect(() => new LayerNorm(4, { eps: 0 })).toThrow(/eps/);
  });

  it("validates input shape suffix", () => {
    const ln = new LayerNorm([4]);
    expect(() => ln.forward(tensor([[1, 2, 3]], f32))).toThrow(/does not end/);
  });

  it("toString", () => {
    expect(new LayerNorm(4).toString()).toContain("LayerNorm");
  });
});

describe("GroupNorm", () => {
  it("normalizes with groups", () => {
    const gn = new GroupNorm(2, 4);
    const x = tensor(
      Array.from({ length: 8 }, (_, i) => i + 1),
      f32
    ).reshape([2, 4]);
    const out = gn.forward(x);
    expect(out.shape).toEqual([2, 4]);
  });

  it("works with affine=false", () => {
    const gn = new GroupNorm(2, 4, { affine: false });
    const x = tensor(
      Array.from({ length: 8 }, (_, i) => i + 1),
      f32
    ).reshape([2, 4]);
    const out = gn.forward(x);
    expect(out.shape).toEqual([2, 4]);
  });

  it("validates constructor params", () => {
    expect(() => new GroupNorm(0, 4)).toThrow(/numGroups/);
    expect(() => new GroupNorm(2, 0)).toThrow(/numChannels/);
    expect(() => new GroupNorm(3, 4)).toThrow(/divisible/);
  });

  it("validates input channel mismatch", () => {
    const gn = new GroupNorm(2, 4);
    expect(() => gn.forward(tensor([[1, 2, 3]], f32))).toThrow(/channels/i);
  });

  it("toString", () => {
    expect(new GroupNorm(2, 4).toString()).toContain("GroupNorm");
  });
});

describe("InstanceNorm", () => {
  it("normalizes each channel independently", () => {
    const inorm = new InstanceNorm(2);
    const x = tensor(
      Array.from({ length: 4 }, (_, i) => i + 1),
      f32
    ).reshape([1, 2, 2]);
    const out = inorm.forward(x);
    expect(out.shape).toEqual([1, 2, 2]);
  });

  it("toString", () => {
    expect(new InstanceNorm(4).toString()).toContain("InstanceNorm");
  });
});

describe("RMSNorm", () => {
  it("normalizes with number shape", () => {
    const rms = new RMSNorm(4);
    const x = tensor([[1, 2, 3, 4]], f32);
    const out = rms.forward(x);
    expect(out.shape).toEqual([1, 4]);
  });

  it("normalizes with array shape", () => {
    const rms = new RMSNorm([4]);
    const x = tensor([[1, 2, 3, 4]], f32);
    const out = rms.forward(x);
    expect(out.shape).toEqual([1, 4]);
  });

  it("validates normalizedShape", () => {
    expect(() => new RMSNorm([])).toThrow(/at least one/);
  });

  it("validates input shape suffix", () => {
    const rms = new RMSNorm([4]);
    expect(() => rms.forward(tensor([[1, 2, 3]], f32))).toThrow(/does not end/);
  });

  it("validates input shape too small", () => {
    const rms = new RMSNorm([2, 3]);
    expect(() => rms.forward(tensor([1, 2, 3], f32))).toThrow(/too small/);
  });

  it("toString", () => {
    expect(new RMSNorm(4).toString()).toContain("RMSNorm");
  });
});

describe("BatchNorm2d", () => {
  it("normalizes 4D input in training mode", () => {
    const bn = new BatchNorm2d(2);
    const x = tensor(
      Array.from({ length: 16 }, (_, i) => i + 1),
      f32
    ).reshape([2, 2, 2, 2]);
    const out = bn.forward(x);
    expect(out.shape).toEqual([2, 2, 2, 2]);
  });

  it("uses running stats in eval mode", () => {
    const bn = new BatchNorm2d(2);
    const x = tensor(
      Array.from({ length: 16 }, (_, i) => i + 1),
      f32
    ).reshape([2, 2, 2, 2]);
    bn.train();
    bn.forward(x);
    bn.eval();
    const out = bn.forward(x);
    expect(out.shape).toEqual([2, 2, 2, 2]);
  });

  it("works with affine=false", () => {
    const bn = new BatchNorm2d(2, { affine: false });
    const x = tensor(
      Array.from({ length: 16 }, (_, i) => i + 1),
      f32
    ).reshape([2, 2, 2, 2]);
    const out = bn.forward(x);
    expect(out.shape).toEqual([2, 2, 2, 2]);
  });

  it("works with trackRunningStats=false", () => {
    const bn = new BatchNorm2d(2, { trackRunningStats: false });
    const x = tensor(
      Array.from({ length: 16 }, (_, i) => i + 1),
      f32
    ).reshape([2, 2, 2, 2]);
    const out = bn.forward(x);
    expect(out.shape).toEqual([2, 2, 2, 2]);
  });

  it("validates constructor params", () => {
    expect(() => new BatchNorm2d(0)).toThrow(/positive integer/);
    expect(() => new BatchNorm2d(-1)).toThrow(/positive integer/);
  });

  it("validates non-4D input", () => {
    const bn = new BatchNorm2d(3);
    expect(() => bn.forward(tensor([[1, 2, 3]], f32))).toThrow(/4D/);
  });

  it("validates channel count mismatch", () => {
    const bn = new BatchNorm2d(3);
    const x = tensor(
      Array.from({ length: 16 }, (_, i) => i + 1),
      f32
    ).reshape([2, 2, 2, 2]);
    expect(() => bn.forward(x)).toThrow(/channels/i);
  });

  it("toString", () => {
    expect(new BatchNorm2d(3).toString()).toContain("BatchNorm2d");
  });
});
