import { describe, expect, it } from "vitest";
import { GradTensor, tensor } from "../src/ndarray";
import { ModuleDict, ModuleList } from "../src/nn/containers/ModuleList";
import { Embedding } from "../src/nn/layers/embedding";
import {
  ConstantPad2d,
  ReflectionPad2d,
  ReplicationPad2d,
  ZeroPad2d,
} from "../src/nn/layers/padding";
import { Upsample } from "../src/nn/layers/upsample";
import { Flatten, Identity, Unflatten } from "../src/nn/layers/utility";

// Helper: create a 4D tensor (N=1, C=1, H, W)
function make4D(h: number, w: number, values?: number[]): ReturnType<typeof tensor> {
  const data = values ?? Array.from({ length: h * w }, (_, i) => i + 1);
  return tensor(data, { dtype: "float64" }).reshape([1, 1, h, w]);
}

describe("Identity", () => {
  it("passes input through unchanged", () => {
    const id = new Identity();
    const x = tensor([1, 2, 3]);
    const out = id.forward(x);
    expect(out).toBe(x);
  });

  it("toString", () => {
    expect(new Identity().toString()).toBe("Identity()");
  });
});

describe("Flatten", () => {
  it("flattens default (startDim=1, endDim=-1)", () => {
    const flatten = new Flatten();
    const x = tensor(Array.from({ length: 24 }, (_, i) => i)).reshape([2, 3, 4]);
    const out = flatten.forward(x);
    expect(out.shape).toEqual([2, 12]);
  });

  it("flattens custom dims", () => {
    const flatten = new Flatten(0, 1);
    const x = tensor(Array.from({ length: 24 }, (_, i) => i)).reshape([2, 3, 4]);
    const out = flatten.forward(x);
    expect(out.shape).toEqual([6, 4]);
  });

  it("works with GradTensor input", () => {
    const flatten = new Flatten();
    const x = GradTensor.fromTensor(
      tensor(Array.from({ length: 12 }, (_, i) => i)).reshape([2, 3, 2])
    );
    const out = flatten.forward(x);
    expect(out.shape).toEqual([2, 6]);
  });

  it("throws for invalid startDim", () => {
    const flatten = new Flatten(5, -1);
    const x = tensor([1, 2, 3, 4]).reshape([2, 2]);
    expect(() => flatten.forward(x)).toThrow(/startDim/);
  });

  it("throws for invalid endDim", () => {
    const flatten = new Flatten(0, 5);
    const x = tensor([1, 2, 3, 4]).reshape([2, 2]);
    expect(() => flatten.forward(x)).toThrow(/endDim/);
  });

  it("throws when startDim > endDim", () => {
    const flatten = new Flatten(1, 0);
    const x = tensor(Array.from({ length: 24 }, (_, i) => i)).reshape([2, 3, 4]);
    expect(() => flatten.forward(x)).toThrow(/startDim.*<=.*endDim/);
  });

  it("toString", () => {
    expect(new Flatten().toString()).toBe("Flatten(start_dim=1, end_dim=-1)");
  });
});

describe("Unflatten", () => {
  it("unflattens a dim", () => {
    const unflatten = new Unflatten(1, [2, 3]);
    const x = tensor(Array.from({ length: 12 }, (_, i) => i)).reshape([2, 6]);
    const out = unflatten.forward(x);
    expect(out.shape).toEqual([2, 2, 3]);
  });

  it("handles negative dim", () => {
    const unflatten = new Unflatten(-1, [2, 3]);
    const x = tensor(Array.from({ length: 12 }, (_, i) => i)).reshape([2, 6]);
    const out = unflatten.forward(x);
    expect(out.shape).toEqual([2, 2, 3]);
  });

  it("works with GradTensor input", () => {
    const unflatten = new Unflatten(1, [3, 2]);
    const x = GradTensor.fromTensor(
      tensor(Array.from({ length: 12 }, (_, i) => i)).reshape([2, 6])
    );
    const out = unflatten.forward(x);
    expect(out.shape).toEqual([2, 3, 2]);
  });

  it("throws for empty unflattenedSize", () => {
    expect(() => new Unflatten(0, [])).toThrow(/at least one element/);
  });

  it("throws for non-positive unflattenedSize", () => {
    expect(() => new Unflatten(0, [0])).toThrow(/positive integers/);
    // -1 is the "infer this dimension" marker (as in PyTorch); other negatives are invalid.
    expect(() => new Unflatten(0, [-2])).toThrow(/positive integers/);
  });

  it("throws for dim out of range", () => {
    const unflatten = new Unflatten(5, [2, 3]);
    const x = tensor([1, 2, 3, 4, 5, 6]).reshape([2, 3]);
    expect(() => unflatten.forward(x)).toThrow(/out of range/);
  });

  it("throws for size mismatch", () => {
    const unflatten = new Unflatten(1, [2, 2]);
    const x = tensor(Array.from({ length: 6 }, (_, i) => i)).reshape([2, 3]);
    expect(() => unflatten.forward(x)).toThrow(/product/i);
  });

  it("toString", () => {
    expect(new Unflatten(1, [2, 3]).toString()).toBe("Unflatten(dim=1, unflattened_size=[2, 3])");
  });
});

describe("ZeroPad2d", () => {
  it("pads with zeros (scalar padding)", () => {
    const pad = new ZeroPad2d(1);
    const x = make4D(2, 2, [1, 2, 3, 4]);
    const out = pad.forward(x);
    expect(out.shape).toEqual([1, 1, 4, 4]);
  });

  it("pads with zeros (tuple padding)", () => {
    const pad = new ZeroPad2d([1, 1, 0, 0]);
    const x = make4D(2, 2, [1, 2, 3, 4]);
    const out = pad.forward(x);
    expect(out.shape).toEqual([1, 1, 2, 4]);
  });

  it("works with GradTensor input", () => {
    const pad = new ZeroPad2d(1);
    const x = GradTensor.fromTensor(make4D(2, 2, [1, 2, 3, 4]));
    const out = pad.forward(x);
    expect(out.shape).toEqual([1, 1, 4, 4]);
  });

  it("toString", () => {
    expect(new ZeroPad2d(1).toString()).toBe("ZeroPad2d(padding=[1, 1, 1, 1])");
  });
});

describe("ConstantPad2d", () => {
  it("pads with constant value", () => {
    const pad = new ConstantPad2d(1, -1);
    const x = make4D(2, 2, [1, 2, 3, 4]);
    const out = pad.forward(x);
    expect(out.shape).toEqual([1, 1, 4, 4]);
  });

  it("toString", () => {
    expect(new ConstantPad2d(1, -1).toString()).toBe(
      "ConstantPad2d(padding=[1, 1, 1, 1], value=-1)"
    );
  });
});

describe("ReflectionPad2d", () => {
  it("pads with reflection", () => {
    const pad = new ReflectionPad2d(1);
    const x = make4D(3, 3);
    const out = pad.forward(x);
    expect(out.shape).toEqual([1, 1, 5, 5]);
  });

  it("throws when padding >= input size", () => {
    const pad = new ReflectionPad2d(3);
    const x = make4D(3, 3);
    expect(() => pad.forward(x)).toThrow(/less than input/);
  });

  it("toString", () => {
    expect(new ReflectionPad2d(1).toString()).toBe("ReflectionPad2d(padding=[1, 1, 1, 1])");
  });
});

describe("ReplicationPad2d", () => {
  it("pads with replication", () => {
    const pad = new ReplicationPad2d(1);
    const x = make4D(2, 2, [1, 2, 3, 4]);
    const out = pad.forward(x);
    expect(out.shape).toEqual([1, 1, 4, 4]);
  });

  it("toString", () => {
    expect(new ReplicationPad2d(1).toString()).toBe("ReplicationPad2d(padding=[1, 1, 1, 1])");
  });
});

describe("Upsample", () => {
  it("upsamples with nearest mode and scaleFactor", () => {
    const up = new Upsample({ scaleFactor: 2, mode: "nearest" });
    const x = make4D(2, 2, [1, 2, 3, 4]);
    const out = up.forward(x);
    expect(out.shape).toEqual([1, 1, 4, 4]);
  });

  it("upsamples with bilinear mode and scaleFactor", () => {
    const up = new Upsample({ scaleFactor: 2, mode: "bilinear" });
    const x = make4D(2, 2, [1, 2, 3, 4]);
    const out = up.forward(x);
    expect(out.shape).toEqual([1, 1, 4, 4]);
  });

  it("upsamples with size", () => {
    const up = new Upsample({ size: [6, 6] });
    const x = make4D(2, 2, [1, 2, 3, 4]);
    const out = up.forward(x);
    expect(out.shape).toEqual([1, 1, 6, 6]);
  });

  it("upsamples with bilinear and size", () => {
    const up = new Upsample({ size: [4, 4], mode: "bilinear" });
    const x = make4D(2, 2, [1, 2, 3, 4]);
    const out = up.forward(x);
    expect(out.shape).toEqual([1, 1, 4, 4]);
  });

  it("throws when both scaleFactor and size given", () => {
    expect(() => new Upsample({ scaleFactor: 2, size: [4, 4] })).toThrow(/not both/);
  });

  it("throws when neither scaleFactor nor size given", () => {
    expect(() => new Upsample({})).toThrow(/Must specify/);
  });

  it("throws for invalid scaleFactor", () => {
    expect(() => new Upsample({ scaleFactor: 0 })).toThrow(/positive/);
    expect(() => new Upsample({ scaleFactor: -1 })).toThrow(/positive/);
  });

  it("throws for invalid size", () => {
    expect(() => new Upsample({ size: [0, 4] })).toThrow(/positive integers/);
    expect(() => new Upsample({ size: [-1, 4] })).toThrow(/positive integers/);
  });

  it("throws for non-4D input", () => {
    const up = new Upsample({ scaleFactor: 2 });
    expect(() => up.forward(tensor([1, 2, 3]))).toThrow(/4D/);
  });

  it("toString with scaleFactor", () => {
    expect(new Upsample({ scaleFactor: 2 }).toString()).toContain("scale_factor=2");
  });

  it("toString with size", () => {
    expect(new Upsample({ size: [4, 4] }).toString()).toContain("size=[4, 4]");
  });
});

describe("Embedding", () => {
  it("looks up embeddings", () => {
    const emb = new Embedding(10, 3);
    const indices = tensor([0, 1, 2], { dtype: "int32" });
    const out = emb.forward(indices);
    expect(out.shape).toEqual([3, 3]);
  });

  it("handles paddingIdx", () => {
    const emb = new Embedding(10, 3, { paddingIdx: 0 });
    const indices = tensor([0, 1, 2], { dtype: "int32" });
    const out = emb.forward(indices);
    expect(out.shape).toEqual([3, 3]);
  });

  it("handles 2D input indices", () => {
    const emb = new Embedding(10, 3);
    const indices = tensor([0, 1, 2, 3], { dtype: "int32" }).reshape([2, 2]);
    const out = emb.forward(indices);
    expect(out.shape).toEqual([2, 2, 3]);
  });

  it("validates constructor params", () => {
    expect(() => new Embedding(0, 3)).toThrow(/numEmbeddings/);
    expect(() => new Embedding(10, 0)).toThrow(/embeddingDim/);
    expect(() => new Embedding(10, 3, { paddingIdx: -11 })).toThrow(/paddingIdx/);
    expect(() => new Embedding(10, 3, { paddingIdx: 10 })).toThrow(/paddingIdx/);
  });

  it("getParams and toString", () => {
    const emb = new Embedding(10, 3);
    expect(emb.toString()).toContain("Embedding");
  });
});

describe("ModuleList", () => {
  it("creates empty list", () => {
    const ml = new ModuleList();
    expect(ml.length).toBe(0);
  });

  it("creates with initial modules", () => {
    const ml = new ModuleList([new Identity(), new Identity()]);
    expect(ml.length).toBe(2);
  });

  it("append", () => {
    const ml = new ModuleList();
    ml.append(new Identity());
    ml.append(new Identity());
    expect(ml.length).toBe(2);
  });

  it("get by index", () => {
    const id = new Identity();
    const ml = new ModuleList([id]);
    expect(ml.get(0)).toBe(id);
  });

  it("get by negative index", () => {
    const id = new Identity();
    const ml = new ModuleList([new Identity(), id]);
    expect(ml.get(-1)).toBe(id);
  });

  it("throws for out-of-range index", () => {
    const ml = new ModuleList([new Identity()]);
    expect(() => ml.get(5)).toThrow(/out of range/);
  });

  it("insert", () => {
    const ml = new ModuleList([new Identity(), new Identity()]);
    const inserted = new Flatten();
    ml.insert(1, inserted);
    expect(ml.length).toBe(3);
    expect(ml.get(1)).toBe(inserted);
  });

  it("throws for out-of-range insert index", () => {
    const ml = new ModuleList();
    expect(() => ml.insert(-1, new Identity())).toThrow(/out of range/);
    expect(() => ml.insert(5, new Identity())).toThrow(/out of range/);
  });

  it("is iterable", () => {
    const ml = new ModuleList([new Identity(), new Identity()]);
    const items = [...ml];
    expect(items.length).toBe(2);
  });

  it("forward throws", () => {
    const ml = new ModuleList();
    expect(() => ml.forward(tensor([1]))).toThrow(/not callable/i);
  });

  it("toString", () => {
    const ml = new ModuleList([new Identity()]);
    expect(ml.toString()).toContain("ModuleList");
    expect(ml.toString()).toContain("Identity");
  });
});

describe("ModuleDict", () => {
  it("creates empty dict", () => {
    const md = new ModuleDict();
    expect(md.length).toBe(0);
  });

  it("creates with initial modules", () => {
    const md = new ModuleDict({ a: new Identity(), b: new Identity() });
    expect(md.length).toBe(2);
  });

  it("set and get", () => {
    const md = new ModuleDict();
    const id = new Identity();
    md.set("test", id);
    expect(md.get("test")).toBe(id);
  });

  it("throws for unknown key", () => {
    const md = new ModuleDict();
    expect(() => md.get("missing")).toThrow(/not found/);
  });

  it("has", () => {
    const md = new ModuleDict({ a: new Identity() });
    expect(md.has("a")).toBe(true);
    expect(md.has("b")).toBe(false);
  });

  it("delete", () => {
    const md = new ModuleDict({ a: new Identity() });
    expect(md.delete("a")).toBe(true);
    expect(md.length).toBe(0);
    expect(md.delete("a")).toBe(false);
  });

  it("keys, values, entries", () => {
    const md = new ModuleDict({ a: new Identity(), b: new Identity() });
    expect([...md.keys()]).toEqual(["a", "b"]);
    expect([...md.values()].length).toBe(2);
    expect([...md.entries()].length).toBe(2);
  });

  it("is iterable", () => {
    const md = new ModuleDict({ a: new Identity() });
    const items = [...md];
    expect(items.length).toBe(1);
    expect(items[0]![0]).toBe("a");
  });

  it("forward throws", () => {
    const md = new ModuleDict();
    expect(() => md.forward(tensor([1]))).toThrow(/not callable/i);
  });

  it("toString", () => {
    const md = new ModuleDict({ a: new Identity() });
    expect(md.toString()).toContain("ModuleDict");
    expect(md.toString()).toContain("Identity");
  });
});
