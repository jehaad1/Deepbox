import { describe, expect, it } from "vitest";
import { GradTensor, tensor } from "../src/ndarray";
import { Linear } from "../src/nn";

describe("deepbox/nn - Linear layer", () => {
  describe("constructor", () => {
    it("should create layer with correct dimensions", () => {
      const layer = new Linear(10, 5);

      expect(layer.inputSize).toBe(10);
      expect(layer.outputSize).toBe(5);
    });

    it("should initialize with bias by default", () => {
      const layer = new Linear(10, 5);
      const bias = layer.getBias();

      expect(bias).toBeDefined();
      expect(bias?.shape).toEqual([5]);
    });

    it("should create layer without bias when specified", () => {
      const layer = new Linear(10, 5, { bias: false });
      const bias = layer.getBias();

      expect(bias).toBeUndefined();
    });

    it("should initialize weights with correct shape", () => {
      const layer = new Linear(10, 5);
      const weight = layer.getWeight();

      expect(weight.shape).toEqual([5, 10]);
    });

    it("validates dimensions", () => {
      expect(() => new Linear(0, 5)).toThrow();
      expect(() => new Linear(5, 0)).toThrow();
      expect(() => new Linear(2.5 as number, 3)).toThrow();
      expect(() => new Linear(3, 2.5 as number)).toThrow();
    });
  });

  describe("forward", () => {
    it("should validate input dimensions", () => {
      const layer = new Linear(10, 5);
      const input_0d = tensor(42);

      expect(() => layer.forward(input_0d)).toThrow("Linear layer expects at least 1D input");
    });

    it("rejects string inputs", () => {
      const layer = new Linear(2, 1);
      const input = tensor([["a", "b"]]);
      expect(() => layer.forward(input)).toThrow();
    });

    it("should validate input features", () => {
      const layer = new Linear(10, 5);
      const wrong_input = tensor([[1, 2, 3]]); // 3 features instead of 10

      expect(() => layer.forward(wrong_input)).toThrow(
        "Linear layer expects 10 input features; got 3"
      );
    });

    it("should compute correct output shape for 1D input", () => {
      const layer = new Linear(10, 5);
      const input = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
      const output = layer.forward(input);

      expect(output.shape).toEqual([5]);
    });

    it("should compute correct output shape for 2D input", () => {
      const layer = new Linear(10, 5);
      const input = tensor([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
      ]);
      const output = layer.forward(input);

      expect(output.shape).toEqual([2, 5]);
    });

    it("should compute correct output shape for 3D input", () => {
      const layer = new Linear(10, 5);
      const input = tensor([
        [
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
          [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        ],
        [
          [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
          [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        ],
      ]);
      const output = layer.forward(input);

      expect(output.shape).toEqual([2, 2, 5]);
    });
  });

  describe("toString", () => {
    it("should show layer info with bias", () => {
      const layer = new Linear(10, 5);
      const str = layer.toString();

      expect(str).toContain("Linear");
      expect(str).toContain("in_features=10");
      expect(str).toContain("out_features=5");
      expect(str).toContain("bias=true");
    });

    it("should show layer info without bias", () => {
      const layer = new Linear(10, 5, { bias: false });
      const str = layer.toString();

      expect(str).toContain("bias=false");
    });
  });

  describe("training/eval mode", () => {
    it("should be in training mode by default", () => {
      const layer = new Linear(10, 5);
      expect(layer.training).toBe(true);
    });

    it("should switch to eval mode", () => {
      const layer = new Linear(10, 5);
      layer.eval();
      expect(layer.training).toBe(false);
    });

    it("should switch back to training mode", () => {
      const layer = new Linear(10, 5);
      layer.eval();
      layer.train();
      expect(layer.training).toBe(true);
    });
  });
});

describe("Linear integration", () => {
  it("should compute forward pass numerically", () => {
    const layer = new Linear(2, 1, { bias: false });
    const weight = layer.getWeight();
    // Set weights to [2, 3]
    weight.data[weight.offset + 0] = 2;
    weight.data[weight.offset + 1] = 3;

    const input = tensor([1, 2]);
    const output = layer.forward(input);
    const value = Number(output.data[output.offset]);

    expect(value).toBeCloseTo(8, 6);
  });

  it("should compute gradients with GradTensor input", () => {
    const layer = new Linear(2, 1, { bias: false });
    const weight = layer.getWeight();
    weight.data[weight.offset + 0] = 2;
    weight.data[weight.offset + 1] = 3;

    const xTensor = tensor([1, 2]);
    const x = GradTensor.fromTensor(xTensor, { requiresGrad: true });
    const yOut = layer.forward(x);
    if (!(yOut instanceof GradTensor)) throw new Error("Expected GradTensor");
    const y = yOut;
    const loss = y.sum();
    loss.backward();

    // Gradient w.r.t. input should be weights
    const xGrad = x.grad;
    expect(xGrad).not.toBeNull();
    if (xGrad) {
      expect(Number(xGrad.data[xGrad.offset + 0])).toBeCloseTo(2, 6);
      expect(Number(xGrad.data[xGrad.offset + 1])).toBeCloseTo(3, 6);
    }

    // Gradient w.r.t. weights should be inputs
    const params = Array.from(layer.parameters());
    expect(params.length).toBeGreaterThan(0);
    const weightParam = params[0];
    const wGrad = weightParam.grad;
    expect(wGrad).not.toBeNull();
    if (wGrad) {
      expect(Number(wGrad.data[wGrad.offset + 0])).toBeCloseTo(1, 6);
      expect(Number(wGrad.data[wGrad.offset + 1])).toBeCloseTo(2, 6);
    }
  });

  it("should initialize weights with correct dtype", () => {
    const layer = new Linear(2, 3, { dtype: "float64" });
    expect(layer.getWeight().dtype).toBe("float64");
  });

  it("should freeze and unfreeze parameters", () => {
    const layer = new Linear(2, 3);
    layer.freezeParameters();
    for (const param of layer.parameters()) {
      expect(param.requiresGrad).toBe(false);
    }
    layer.unfreezeParameters();
    for (const param of layer.parameters()) {
      expect(param.requiresGrad).toBe(true);
    }
  });
});
