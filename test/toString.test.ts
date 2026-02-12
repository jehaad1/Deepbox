import { describe, expect, it } from "vitest";
import { DataFrame, Series } from "../src/dataframe";
import { GradTensor, parameter, tensor } from "../src/ndarray";
import { Linear, ReLU, Sequential, Sigmoid, Tanh } from "../src/nn";

// =============================================================================
// Tensor.toString()
// =============================================================================

describe("Tensor.toString()", () => {
  it("formats a 1D tensor", () => {
    const t = tensor([1, 2, 3]);
    const s = t.toString();
    expect(s).toBe("tensor([1, 2, 3], dtype=float32)");
  });

  it("formats a 2D tensor", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const s = t.toString();
    expect(s).toContain("tensor(");
    expect(s).toContain("[1, 2]");
    expect(s).toContain("[3, 4]");
    expect(s).toContain("dtype=float32");
  });

  it("formats a scalar (0-D) tensor", () => {
    const t = tensor(42);
    const s = t.toString();
    expect(s).toBe("tensor(42, dtype=float32)");
  });

  it("formats an int32 tensor", () => {
    const t = tensor([10, 20], { dtype: "int32" });
    const s = t.toString();
    expect(s).toBe("tensor([10, 20], dtype=int32)");
  });

  it("formats an int64 tensor", () => {
    const t = tensor([1, 2, 3], { dtype: "int64" });
    const s = t.toString();
    expect(s).toContain("dtype=int64");
  });

  it("formats a string tensor", () => {
    const t = tensor(["hello", "world"]);
    const s = t.toString();
    expect(s).toContain('"hello"');
    expect(s).toContain('"world"');
    expect(s).toContain("dtype=string");
  });

  it("summarizes large tensors", () => {
    const data = Array.from({ length: 100 }, (_, i) => i);
    const t = tensor(data);
    const s = t.toString(6);
    expect(s).toContain("...");
  });

  it("formats an empty tensor", () => {
    const t = tensor([]);
    const s = t.toString();
    expect(s).toContain("[]");
  });

  it("formats floating-point values with precision", () => {
    const t = tensor([1.123456789]);
    const s = t.toString();
    expect(s).toContain("1.123");
    expect(s).toContain("dtype=float32");
  });
});

// =============================================================================
// GradTensor.toString()
// =============================================================================

describe("GradTensor.toString()", () => {
  it("includes requiresGrad when true", () => {
    const g = parameter([1, 2, 3]);
    const s = g.toString();
    expect(s).toContain("requiresGrad=true");
    expect(s).toContain("dtype=float32");
  });

  it("omits requiresGrad when false", () => {
    const g = GradTensor.fromTensor(tensor([1, 2]), { requiresGrad: false });
    const s = g.toString();
    expect(s).not.toContain("requiresGrad");
  });
});

// =============================================================================
// DataFrame.toString()
// =============================================================================

describe("DataFrame.toString()", () => {
  it("formats a small DataFrame", () => {
    const df = new DataFrame({
      name: ["Alice", "Bob"],
      age: [25, 30],
    });
    const s = df.toString();
    expect(s).toContain("name");
    expect(s).toContain("age");
    expect(s).toContain("Alice");
    expect(s).toContain("Bob");
    expect(s).toContain("25");
    expect(s).toContain("30");
  });

  it("truncates large DataFrames", () => {
    const data: Record<string, number[]> = { x: [] };
    for (let i = 0; i < 100; i++) data.x.push(i);
    const df = new DataFrame(data);
    const s = df.toString(10);
    expect(s).toContain("...");
  });

  it("formats an empty DataFrame", () => {
    const df = new DataFrame({});
    const s = df.toString();
    expect(typeof s).toBe("string");
  });
});

// =============================================================================
// Series.toString()
// =============================================================================

describe("Series.toString()", () => {
  it("formats a small Series", () => {
    const s = new Series([10, 20, 30], { name: "values" });
    const str = s.toString();
    expect(str).toContain("10");
    expect(str).toContain("20");
    expect(str).toContain("30");
    expect(str).toContain("Name: values");
    expect(str).toContain("Length: 3");
  });

  it("truncates large Series", () => {
    const data = Array.from({ length: 100 }, (_, i) => i);
    const s = new Series(data);
    const str = s.toString(10);
    expect(str).toContain("...");
  });

  it("formats Series without name", () => {
    const s = new Series([1, 2]);
    const str = s.toString();
    expect(str).toContain("Length: 2");
    expect(str).not.toContain("Name:");
  });
});

// =============================================================================
// GradTensor.zeroGrad() dtype regression (previously hardcoded float64)
// =============================================================================

describe("GradTensor.zeroGrad() dtype", () => {
  it("preserves float32 dtype after zeroGrad", () => {
    const v = parameter([1, 2, 3]);
    const loss = v.mul(v).sum();
    loss.backward();
    expect(v.grad).not.toBeNull();
    expect(v.grad?.dtype).toBe("float32");

    v.zeroGrad();
    expect(v.grad).not.toBeNull();
    expect(v.grad?.dtype).toBe("float32");
  });

  it("preserves float64 dtype after zeroGrad", () => {
    const v = parameter(tensor([1, 2, 3], { dtype: "float64" }));
    v.zeroGrad();
    expect(v.grad).not.toBeNull();
    expect(v.grad?.dtype).toBe("float64");
  });

  it("allows backward after zeroGrad without dtype mismatch", () => {
    const v = parameter([1, 2, 3]);

    const loss1 = v.mul(v).sum();
    loss1.backward();

    v.zeroGrad();

    const c = GradTensor.fromTensor(tensor([3, 3, 3]), { requiresGrad: false });
    const loss2 = v.mul(c).sum();
    expect(() => loss2.backward()).not.toThrow();
  });
});

// =============================================================================
// Activation layers GradTensor support regression
// =============================================================================

describe("Activation layers preserve GradTensor in Sequential", () => {
  it("ReLU preserves GradTensor through forward pass", () => {
    const relu = new ReLU();
    const input = GradTensor.fromTensor(tensor([-1, 0, 1]), {
      requiresGrad: false,
    });
    const output = relu.forward(input);
    expect(output).toBeInstanceOf(GradTensor);
  });

  it("Sigmoid preserves GradTensor through forward pass", () => {
    const sig = new Sigmoid();
    const input = GradTensor.fromTensor(tensor([0, 1, 2]), {
      requiresGrad: false,
    });
    const output = sig.forward(input);
    expect(output).toBeInstanceOf(GradTensor);
  });

  it("Tanh preserves GradTensor through forward pass", () => {
    const tanh = new Tanh();
    const input = GradTensor.fromTensor(tensor([0, 1, 2]), {
      requiresGrad: false,
    });
    const output = tanh.forward(input);
    expect(output).toBeInstanceOf(GradTensor);
  });

  it("Sequential with ReLU preserves GradTensor end-to-end", () => {
    const model = new Sequential(new Linear(2, 4), new ReLU(), new Linear(4, 1));
    const input = parameter([
      [1, 2],
      [3, 4],
    ]);
    const output = model.forward(input);
    expect(output).toBeInstanceOf(GradTensor);
  });

  it("Sequential training loop with backward does not throw", () => {
    const model = new Sequential(new Linear(2, 4), new ReLU(), new Linear(4, 1));
    const x = parameter([
      [1, 0],
      [0, 1],
    ]);
    const y = parameter([[1], [2]]);
    const pred = model.forward(x);
    expect(pred).toBeInstanceOf(GradTensor);

    if (!(pred instanceof GradTensor)) throw new Error("unreachable");
    const diff = pred.sub(y);
    const loss = diff.mul(diff).mean();
    expect(() => loss.backward()).not.toThrow();
  });

  it("ReLU returns plain Tensor when given plain Tensor", () => {
    const relu = new ReLU();
    const input = tensor([-1, 0, 1]);
    const output = relu.forward(input);
    expect(output).not.toBeInstanceOf(GradTensor);
  });
});
