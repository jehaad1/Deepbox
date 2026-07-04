import { describe, expect, it } from "vitest";
import { GradTensor, tensor } from "../src/ndarray";
import { clip_grad_norm_, clip_grad_value_ } from "../src/nn/clip";
import {
  calculateFanInOut,
  calculateGain,
  constant_,
  kaiming_normal_,
  kaiming_uniform_,
  makeRng,
  normal_,
  ones_,
  orthogonal_,
  sparse_,
  uniform_,
  xavier_normal_,
  xavier_uniform_,
  zeros_,
} from "../src/nn/init";

// ────── init functions ──────
describe("uniform_", () => {
  it("fills tensor with uniform values", () => {
    const t = tensor([0, 0, 0, 0]);
    uniform_(t, -1, 1);
    const data = Array.from(t.data as Float64Array);
    for (const v of data) {
      expect(v).toBeGreaterThanOrEqual(-1);
      expect(v).toBeLessThanOrEqual(1);
    }
  });
});

describe("normal_", () => {
  it("fills tensor with normal values", () => {
    const t = tensor([0, 0, 0, 0]);
    normal_(t, 0, 1);
    const data = Array.from(t.data as Float64Array);
    expect(data.some((v) => v !== 0)).toBe(true);
  });
});

describe("constant_", () => {
  it("fills tensor with constant", () => {
    const t = tensor([0, 0, 0]);
    constant_(t, 5);
    expect(Array.from(t.data as Float64Array)).toEqual([5, 5, 5]);
  });
});

describe("zeros_", () => {
  it("fills tensor with zeros", () => {
    const t = tensor([1, 2, 3]);
    zeros_(t);
    expect(Array.from(t.data as Float64Array)).toEqual([0, 0, 0]);
  });
});

describe("ones_", () => {
  it("fills tensor with ones", () => {
    const t = tensor([0, 0, 0]);
    ones_(t);
    expect(Array.from(t.data as Float64Array)).toEqual([1, 1, 1]);
  });
});

describe("xavier_uniform_", () => {
  it("initializes 2D tensor", () => {
    const t = tensor([
      [0, 0, 0],
      [0, 0, 0],
    ]);
    xavier_uniform_(t);
    const data = Array.from(t.data as Float64Array);
    expect(data.some((v) => v !== 0)).toBe(true);
  });

  it("initializes with custom gain", () => {
    const t = tensor([
      [0, 0],
      [0, 0],
    ]);
    xavier_uniform_(t, 2.0);
    const data = Array.from(t.data as Float64Array);
    expect(data.some((v) => v !== 0)).toBe(true);
  });
});

describe("xavier_normal_", () => {
  it("initializes 2D tensor", () => {
    const t = tensor([
      [0, 0, 0],
      [0, 0, 0],
    ]);
    xavier_normal_(t);
    const data = Array.from(t.data as Float64Array);
    expect(data.some((v) => v !== 0)).toBe(true);
  });
});

describe("kaiming_uniform_", () => {
  it("initializes with fan_in", () => {
    const t = tensor([
      [0, 0, 0],
      [0, 0, 0],
    ]);
    kaiming_uniform_(t);
    const data = Array.from(t.data as Float64Array);
    expect(data.some((v) => v !== 0)).toBe(true);
  });

  it("initializes with fan_out", () => {
    const t = tensor([
      [0, 0, 0],
      [0, 0, 0],
    ]);
    kaiming_uniform_(t, 0, "fan_out");
    const data = Array.from(t.data as Float64Array);
    expect(data.some((v) => v !== 0)).toBe(true);
  });

  it("initializes with relu nonlinearity", () => {
    const t = tensor([
      [0, 0],
      [0, 0],
    ]);
    kaiming_uniform_(t, 0, "fan_in", "relu");
    const data = Array.from(t.data as Float64Array);
    expect(data.some((v) => v !== 0)).toBe(true);
  });
});

describe("kaiming_normal_", () => {
  it("initializes with fan_in", () => {
    const t = tensor([
      [0, 0, 0],
      [0, 0, 0],
    ]);
    kaiming_normal_(t);
    const data = Array.from(t.data as Float64Array);
    expect(data.some((v) => v !== 0)).toBe(true);
  });

  it("initializes with fan_out", () => {
    const t = tensor([
      [0, 0, 0],
      [0, 0, 0],
    ]);
    kaiming_normal_(t, 0, "fan_out");
    const data = Array.from(t.data as Float64Array);
    expect(data.some((v) => v !== 0)).toBe(true);
  });
});

describe("orthogonal_", () => {
  it("initializes square matrix", () => {
    const t = tensor([
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ]);
    orthogonal_(t);
    const data = Array.from(t.data as Float64Array);
    expect(data.some((v) => v !== 0)).toBe(true);
  });

  it("initializes wide matrix (rows < cols)", () => {
    const t = tensor([
      [0, 0, 0, 0],
      [0, 0, 0, 0],
    ]);
    orthogonal_(t);
    const data = Array.from(t.data as Float64Array);
    expect(data.some((v) => v !== 0)).toBe(true);
  });

  it("initializes tall matrix (rows > cols)", () => {
    const t = tensor([
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
    ]);
    orthogonal_(t);
    const data = Array.from(t.data as Float64Array);
    expect(data.some((v) => v !== 0)).toBe(true);
  });

  it("with custom gain", () => {
    const t = tensor([
      [0, 0],
      [0, 0],
    ]);
    orthogonal_(t, 2.0);
    const data = Array.from(t.data as Float64Array);
    expect(data.some((v) => v !== 0)).toBe(true);
  });

  it("throws for 1D", () => {
    const t = tensor([1, 2, 3]);
    expect(() => orthogonal_(t)).toThrow();
  });
});

describe("sparse_", () => {
  it("initializes sparse 2D tensor", () => {
    const t = tensor([
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ]);
    sparse_(t, 0.5);
    const data = Array.from(t.data as Float64Array);
    const zeroCount = data.filter((v) => v === 0).length;
    expect(zeroCount).toBeGreaterThan(0);
  });

  it("throws for non-2D", () => {
    const t = tensor([1, 2, 3]);
    expect(() => sparse_(t)).toThrow();
  });
});

describe("calculateGain", () => {
  it("returns 1 for linear", () => {
    expect(calculateGain("linear")).toBe(1);
  });

  it("returns 1 for sigmoid", () => {
    expect(calculateGain("sigmoid")).toBe(1);
  });

  it("returns 5/3 for tanh", () => {
    expect(calculateGain("tanh")).toBeCloseTo(5 / 3);
  });

  it("returns sqrt(2) for relu", () => {
    expect(calculateGain("relu")).toBeCloseTo(Math.SQRT2);
  });

  it("handles leaky_relu with param", () => {
    const g = calculateGain("leaky_relu", 0.2);
    expect(g).toBeGreaterThan(0);
  });

  it("returns 3/4 for selu", () => {
    expect(calculateGain("selu")).toBeCloseTo(0.75);
  });

  it("handles conv variants", () => {
    expect(calculateGain("conv1d")).toBe(1);
    expect(calculateGain("conv2d")).toBe(1);
    expect(calculateGain("conv3d")).toBe(1);
  });

  it("throws for unsupported", () => {
    expect(() => calculateGain("unknown")).toThrow();
  });
});

describe("calculateFanInOut", () => {
  it("1D tensor", () => {
    const t = tensor([1, 2, 3]);
    const { fanIn, fanOut } = calculateFanInOut(t);
    expect(fanIn).toBe(3);
    expect(fanOut).toBe(3);
  });

  it("2D tensor", () => {
    const t = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const { fanIn, fanOut } = calculateFanInOut(t);
    expect(fanIn).toBe(3);
    expect(fanOut).toBe(2);
  });

  it("3D tensor (conv)", () => {
    const t = tensor([
      [
        [1, 2],
        [3, 4],
      ],
      [
        [5, 6],
        [7, 8],
      ],
    ]);
    const { fanIn, fanOut } = calculateFanInOut(t);
    expect(fanIn).toBe(4);
    expect(fanOut).toBe(4);
  });
});

describe("makeRng", () => {
  it("returns Math.random without seed", () => {
    const rng = makeRng();
    const v = rng();
    expect(v).toBeGreaterThanOrEqual(0);
    expect(v).toBeLessThan(1);
  });

  it("returns seeded rng with seed", () => {
    const rng1 = makeRng(42);
    const rng2 = makeRng(42);
    expect(rng1()).toBe(rng2());
  });
});

// ────── clip functions ──────
describe("clip_grad_norm_", () => {
  it("returns 0 for no grads", () => {
    const p = GradTensor.fromTensor(tensor([1, 2, 3]));
    const norm = clip_grad_norm_([p], 1.0);
    expect(norm).toBe(0);
  });

  it("clips gradients by L2 norm", () => {
    const p = GradTensor.fromTensor(tensor([1, 2, 3]), { requiresGrad: true });
    // Manually set grad
    p.setGrad(tensor([3, 4, 0]));
    const norm = clip_grad_norm_([p], 1.0);
    expect(norm).toBeCloseTo(5);
  });

  it("clips with inf norm", () => {
    const p = GradTensor.fromTensor(tensor([1, 2, 3]), { requiresGrad: true });
    p.setGrad(tensor([3, 4, 0]));
    const norm = clip_grad_norm_([p], 1.0, Infinity);
    expect(norm).toBe(4);
  });

  it("validates negative maxNorm", () => {
    expect(() => clip_grad_norm_([], -1)).toThrow(/maxNorm/);
  });
});

describe("clip_grad_value_", () => {
  it("clamps gradient values", () => {
    const p = GradTensor.fromTensor(tensor([1, 2, 3]), { requiresGrad: true });
    p.setGrad(tensor([10, -10, 5]));
    clip_grad_value_([p], 3);
    const gradData = Array.from(p.grad!.data as Float64Array);
    for (const v of gradData) {
      expect(Math.abs(v)).toBeLessThanOrEqual(3);
    }
  });

  it("validates negative clipValue", () => {
    expect(() => clip_grad_value_([], -1)).toThrow(/clipValue/);
  });
});
