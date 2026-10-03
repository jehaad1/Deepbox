import { describe, expect, it } from "vitest";
import { GradTensor, tensor } from "../src/ndarray";
import {
  ELU,
  GELU,
  GLU,
  Hardsigmoid,
  Hardswish,
  LeakyReLU,
  LogSoftmax,
  Mish,
  PReLU,
  ReLU,
  SELU,
  Sigmoid,
  SiLU,
  Softmax,
  Softplus,
  Softsign,
  Swish,
  Tanh,
} from "../src/nn/layers/activations";

const x = tensor([-2, -1, 0, 1, 2], { dtype: "float64" });
const gx = GradTensor.fromTensor(x);

describe("ReLU", () => {
  const act = new ReLU();
  it("forward Tensor", () => {
    const out = act.forward(x);
    expect((out.toArray() as number[])[0]).toBe(0);
    expect((out.toArray() as number[])[4]).toBe(2);
  });
  it("forward GradTensor", () => {
    const out = act.forward(gx);
    expect(out.shape).toEqual([5]);
  });
  it("toString", () => expect(act.toString()).toBe("ReLU()"));
});

describe("Sigmoid", () => {
  const act = new Sigmoid();
  it("forward Tensor", () => {
    const out = act.forward(x);
    const arr = out.toArray() as number[];
    expect(arr[2]).toBeCloseTo(0.5, 5);
  });
  it("forward GradTensor", () => {
    const out = act.forward(gx);
    expect(out.shape).toEqual([5]);
  });
  it("toString", () => expect(act.toString()).toBe("Sigmoid()"));
});

describe("Tanh", () => {
  const act = new Tanh();
  it("forward Tensor", () => {
    const out = act.forward(x);
    const arr = out.toArray() as number[];
    expect(arr[2]).toBeCloseTo(0, 5);
  });
  it("forward GradTensor", () => {
    const out = act.forward(gx);
    expect(out.shape).toEqual([5]);
  });
  it("toString", () => expect(act.toString()).toBe("Tanh()"));
});

describe("LeakyReLU", () => {
  const act = new LeakyReLU(0.1);
  it("forward Tensor", () => {
    const out = act.forward(x);
    const arr = out.toArray() as number[];
    expect(arr[0]).toBeCloseTo(-0.2, 5);
    expect(arr[4]).toBe(2);
  });
  it("forward GradTensor", () => {
    const out = act.forward(gx);
    expect(out.shape).toEqual([5]);
  });
  it("toString", () => expect(act.toString()).toBe("LeakyReLU(alpha=0.1)"));
});

describe("ELU", () => {
  const act = new ELU(1.0);
  it("forward Tensor", () => {
    const out = act.forward(x);
    const arr = out.toArray() as number[];
    expect(arr[0]).toBeLessThan(0);
    expect(arr[4]).toBe(2);
  });
  it("forward GradTensor", () => {
    const out = act.forward(gx);
    expect(out.shape).toEqual([5]);
  });
  it("toString", () => expect(act.toString()).toBe("ELU(alpha=1)"));
});

describe("GELU", () => {
  const act = new GELU();
  it("forward Tensor", () => {
    const out = act.forward(x);
    const arr = out.toArray() as number[];
    expect(arr[2]).toBeCloseTo(0, 5);
  });
  it("forward GradTensor", () => {
    const out = act.forward(gx);
    expect(out.shape).toEqual([5]);
  });
  it("toString", () => expect(act.toString()).toBe("GELU()"));
});

describe("Softmax", () => {
  const act = new Softmax(-1);
  it("forward Tensor", () => {
    const inp = tensor([[1, 2, 3]], { dtype: "float64" });
    const out = act.forward(inp);
    const arr = (out.toArray() as number[][])[0] as number[];
    const sum = arr.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 5);
  });
  it("forward GradTensor", () => {
    const inp = GradTensor.fromTensor(tensor([[1, 2, 3]], { dtype: "float64" }));
    const out = act.forward(inp);
    expect(out.shape).toEqual([1, 3]);
  });
  it("toString", () => expect(act.toString()).toBe("Softmax(axis=-1)"));
});

describe("LogSoftmax", () => {
  const act = new LogSoftmax(-1);
  it("forward Tensor", () => {
    const inp = tensor([[1, 2, 3]], { dtype: "float64" });
    const out = act.forward(inp);
    const arr = (out.toArray() as number[][])[0] as number[];
    expect(arr[0]).toBeLessThan(0);
  });
  it("forward GradTensor", () => {
    const inp = GradTensor.fromTensor(tensor([[1, 2, 3]], { dtype: "float64" }));
    const out = act.forward(inp);
    expect(out.shape).toEqual([1, 3]);
  });
  it("toString", () => expect(act.toString()).toBe("LogSoftmax(axis=-1)"));
});

describe("Softplus", () => {
  const act = new Softplus();
  it("forward Tensor", () => {
    const out = act.forward(x);
    const arr = out.toArray() as number[];
    expect(arr[2]).toBeCloseTo(Math.log(2), 5);
    for (const v of arr) expect(v).toBeGreaterThan(0);
  });
  it("forward GradTensor", () => {
    const out = act.forward(gx);
    expect(out.shape).toEqual([5]);
  });
  it("toString", () => expect(act.toString()).toBe("Softplus()"));
});

describe("Swish", () => {
  const act = new Swish();
  it("forward Tensor", () => {
    const out = act.forward(x);
    const arr = out.toArray() as number[];
    expect(arr[2]).toBeCloseTo(0, 5);
  });
  it("forward GradTensor", () => {
    const out = act.forward(gx);
    expect(out.shape).toEqual([5]);
  });
  it("toString", () => expect(act.toString()).toBe("Swish()"));
});

describe("Mish", () => {
  const act = new Mish();
  it("forward Tensor", () => {
    const out = act.forward(x);
    const arr = out.toArray() as number[];
    expect(arr[2]).toBeCloseTo(0, 5);
  });
  it("forward GradTensor", () => {
    const out = act.forward(gx);
    expect(out.shape).toEqual([5]);
  });
  it("toString", () => expect(act.toString()).toBe("Mish()"));
});

describe("SiLU", () => {
  const act = new SiLU();
  it("forward Tensor", () => {
    const out = act.forward(x);
    expect(out.shape).toEqual([5]);
  });
  it("forward GradTensor", () => {
    const out = act.forward(gx);
    expect(out.shape).toEqual([5]);
  });
  it("toString", () => expect(act.toString()).toBe("SiLU()"));
});

describe("SELU", () => {
  const act = new SELU();
  it("forward Tensor", () => {
    const out = act.forward(x);
    const arr = out.toArray() as number[];
    expect(arr[0]).toBeLessThan(0);
    expect(arr[4]).toBeGreaterThan(0);
  });
  it("forward GradTensor", () => {
    const out = act.forward(gx);
    expect(out.shape).toEqual([5]);
  });
  it("toString", () => expect(act.toString()).toBe("SELU()"));
});

describe("Hardsigmoid", () => {
  const act = new Hardsigmoid();
  it("forward Tensor", () => {
    const out = act.forward(tensor([-4, 0, 4], { dtype: "float64" }));
    const arr = out.toArray() as number[];
    expect(arr[0]).toBe(0);
    expect(arr[1]).toBeCloseTo(0.5, 5);
    expect(arr[2]).toBe(1);
  });
  it("forward GradTensor", () => {
    const out = act.forward(gx);
    expect(out.shape).toEqual([5]);
  });
  it("toString", () => expect(act.toString()).toBe("Hardsigmoid()"));
});

describe("Hardswish", () => {
  const act = new Hardswish();
  it("forward Tensor", () => {
    const out = act.forward(tensor([-4, 0, 4], { dtype: "float64" }));
    const arr = out.toArray() as number[];
    expect(arr[0]).toBeCloseTo(0, 5);
    expect(arr[1]).toBeCloseTo(0, 5);
    expect(arr[2]).toBeCloseTo(4, 5);
  });
  it("forward GradTensor", () => {
    const out = act.forward(gx);
    expect(out.shape).toEqual([5]);
  });
  it("toString", () => expect(act.toString()).toBe("Hardswish()"));
});

describe("PReLU", () => {
  it("forward Tensor", () => {
    const act = new PReLU(1, 0.25);
    const out = act.forward(x);
    expect(out.shape).toEqual([5]);
  });
  it("forward GradTensor", () => {
    const act = new PReLU(1, 0.25);
    const out = act.forward(gx);
    expect(out.shape).toEqual([5]);
  });
  it("forward float32 input (dtype alignment, no DTypeError)", () => {
    const x32 = tensor([-2, -1, 0, 1, 2], { dtype: "float32" });
    const act = new PReLU(1, 0.25);
    const out = act.forward(x32) as GradTensor; // tracks the slope, so a GradTensor
    expect(out.shape).toEqual([5]);
    expect(out.tensor.dtype).toBe("float32");
    // PReLU(x) = relu(x) - 0.25*relu(-x): [-0.5, -0.25, 0, 1, 2]
    expect(Array.from(out.tensor.data as Float32Array)).toEqual([-0.5, -0.25, 0, 1, 2]);
  });
  it("forward float64 input still works (upcast path)", () => {
    const x64 = tensor([-4, 0, 4], { dtype: "float64" });
    const act = new PReLU(1, 0.5);
    const out = act.forward(x64) as GradTensor; // tracks the slope, so a GradTensor
    expect(out.shape).toEqual([3]);
    expect(Array.from(out.tensor.data as Float32Array | Float64Array)).toEqual([-2, 0, 4]);
  });
  it("validates numParameters", () => {
    expect(() => new PReLU(0)).toThrow(/positive integer/);
    expect(() => new PReLU(-1)).toThrow(/positive integer/);
  });
  it("toString", () => {
    expect(new PReLU(1).toString()).toBe("PReLU(num_parameters=1)");
  });
});

describe("GLU", () => {
  it("forward splits and gates", () => {
    const act = new GLU(-1);
    const inp = tensor([[1, 2, 3, 4, 5, 6]], { dtype: "float64" });
    const out = act.forward(inp);
    expect(out.shape).toEqual([1, 3]);
  });
  it("throws for odd dim size", () => {
    const act = new GLU(-1);
    const inp = tensor([[1, 2, 3]], { dtype: "float64" });
    expect(() => act.forward(inp)).toThrow(/even size/);
  });
  it("throws for dim out of range", () => {
    const act = new GLU(5);
    const inp = tensor([[1, 2]], { dtype: "float64" });
    expect(() => act.forward(inp)).toThrow(/out of range/);
  });
  it("toString", () => expect(new GLU().toString()).toBe("GLU(dim=-1)"));
});

describe("Softsign", () => {
  const act = new Softsign();
  it("forward Tensor", () => {
    const out = act.forward(x);
    const arr = out.toArray() as number[];
    expect(arr[2]).toBeCloseTo(0, 5);
    expect(arr[0]).toBeCloseTo(-2 / 3, 5);
    expect(arr[4]).toBeCloseTo(2 / 3, 5);
  });
  it("forward GradTensor", () => {
    const out = act.forward(gx);
    expect(out.shape).toEqual([5]);
  });
  it("toString", () => expect(act.toString()).toBe("Softsign()"));
});
