import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import {
  ELU,
  GELU,
  LeakyReLU,
  LogSoftmax,
  Mish,
  ReLU,
  Sigmoid,
  Softmax,
  Softplus,
  Swish,
  Tanh,
} from "../src/nn";
import { numData } from "./_helpers";

describe("deepbox/nn - Activation Layers", () => {
  describe("ReLU", () => {
    it("should create ReLU layer", () => {
      const relu = new ReLU();
      expect(relu).toBeDefined();
      expect(relu.toString()).toBe("ReLU()");
    });

    it("should apply ReLU activation correctly", () => {
      const relu = new ReLU();
      const input = tensor([-2, -1, 0, 1, 2]);
      const output = relu.forward(input);

      expect(output.shape).toEqual([5]);
      expect(numData(output)).toEqual([0, 0, 0, 1, 2]);
    });

    it("should handle 2D tensors", () => {
      const relu = new ReLU();
      const input = tensor([
        [-1, 0, 1],
        [2, -3, 4],
      ]);
      const output = relu.forward(input);

      expect(output.shape).toEqual([2, 3]);
    });

    it("should preserve positive values", () => {
      const relu = new ReLU();
      const input = tensor([1, 2, 3, 4, 5]);
      const output = relu.forward(input);

      expect(numData(output)).toEqual([1, 2, 3, 4, 5]);
    });

    it("should zero negative values", () => {
      const relu = new ReLU();
      const input = tensor([-5, -4, -3, -2, -1]);
      const output = relu.forward(input);

      expect(numData(output)).toEqual([0, 0, 0, 0, 0]);
    });
  });

  describe("Sigmoid", () => {
    it("should create Sigmoid layer", () => {
      const sigmoid = new Sigmoid();
      expect(sigmoid).toBeDefined();
      expect(sigmoid.toString()).toBe("Sigmoid()");
    });

    it("should apply sigmoid activation", () => {
      const sigmoid = new Sigmoid();
      const input = tensor([0]);
      const output = sigmoid.forward(input);

      expect(output.shape).toEqual([1]);
      expect(output.data[0]).toBeCloseTo(0.5, 5);
    });

    it("should output values in (0, 1) range", () => {
      const sigmoid = new Sigmoid();
      const input = tensor([-10, -5, 0, 5, 10]);
      const output = sigmoid.forward(input);

      for (let i = 0; i < output.size; i++) {
        const val = output.data[i] as number;
        expect(val).toBeGreaterThan(0);
        expect(val).toBeLessThan(1);
      }
    });

    it("should be symmetric around 0.5", () => {
      const sigmoid = new Sigmoid();
      const input = tensor([2]);
      const negInput = tensor([-2]);

      const output = sigmoid.forward(input);
      const negOutput = sigmoid.forward(negInput);

      const val = output.data[0] as number;
      const negVal = negOutput.data[0] as number;
      expect(val + negVal).toBeCloseTo(1.0, 5);
    });
  });

  describe("Tanh", () => {
    it("should create Tanh layer", () => {
      const tanh = new Tanh();
      expect(tanh).toBeDefined();
      expect(tanh.toString()).toBe("Tanh()");
    });

    it("should apply tanh activation", () => {
      const tanh = new Tanh();
      const input = tensor([0]);
      const output = tanh.forward(input);

      expect(output.data[0]).toBeCloseTo(0, 5);
    });

    it("should output values in (-1, 1) range", () => {
      const tanh = new Tanh();
      const input = tensor([-10, -5, 0, 5, 10]);
      const output = tanh.forward(input);

      for (let i = 0; i < output.size; i++) {
        const val = output.data[i] as number;
        expect(val).toBeGreaterThan(-1);
        expect(val).toBeLessThan(1);
      }
    });

    it("should be zero-centered", () => {
      const tanh = new Tanh();
      const input = tensor([0]);
      const output = tanh.forward(input);

      expect(output.data[0]).toBeCloseTo(0, 5);
    });
  });

  describe("LeakyReLU", () => {
    it("should create LeakyReLU with default alpha", () => {
      const leaky = new LeakyReLU();
      expect(leaky).toBeDefined();
      expect(leaky.toString()).toBe("LeakyReLU(alpha=0.01)");
    });

    it("should create LeakyReLU with custom alpha", () => {
      const leaky = new LeakyReLU(0.2);
      expect(leaky.toString()).toBe("LeakyReLU(alpha=0.2)");
    });

    it("should allow small negative values", () => {
      const leaky = new LeakyReLU(0.1);
      const input = tensor([-10, -5, 0, 5, 10]);
      const output = leaky.forward(input);

      expect(output.shape).toEqual([5]);
      const data = numData(output);
      expect(data[0]).toBeCloseTo(-1, 1); // -10 * 0.1
      expect(data[1]).toBeCloseTo(-0.5, 1); // -5 * 0.1
      expect(data[2]).toBe(0);
      expect(data[3]).toBe(5);
      expect(data[4]).toBe(10);
    });

    it("should preserve positive values", () => {
      const leaky = new LeakyReLU(0.01);
      const input = tensor([1, 2, 3]);
      const output = leaky.forward(input);

      expect(numData(output)).toEqual([1, 2, 3]);
    });
  });

  describe("ELU", () => {
    it("should create ELU with default alpha", () => {
      const elu = new ELU();
      expect(elu).toBeDefined();
      expect(elu.toString()).toBe("ELU(alpha=1)");
    });

    it("should create ELU with custom alpha", () => {
      const elu = new ELU(0.5);
      expect(elu.toString()).toBe("ELU(alpha=0.5)");
    });

    it("should apply ELU activation", () => {
      const elu = new ELU(1.0);
      const input = tensor([0, 1, 2]);
      const output = elu.forward(input);

      expect(output.shape).toEqual([3]);
    });

    it("should preserve positive values", () => {
      const elu = new ELU(1.0);
      const input = tensor([1, 2, 3]);
      const output = elu.forward(input);

      expect(numData(output)).toEqual([1, 2, 3]);
    });
  });

  describe("GELU", () => {
    it("should create GELU layer", () => {
      const gelu = new GELU();
      expect(gelu).toBeDefined();
      expect(gelu.toString()).toBe("GELU()");
    });

    it("should apply GELU activation", () => {
      const gelu = new GELU();
      const input = tensor([0, 1, -1]);
      const output = gelu.forward(input);

      expect(output.shape).toEqual([3]);
    });

    it("should handle 2D inputs", () => {
      const gelu = new GELU();
      const input = tensor([
        [0, 1],
        [-1, 2],
      ]);
      const output = gelu.forward(input);

      expect(output.shape).toEqual([2, 2]);
    });
  });

  describe("Softmax", () => {
    it("should create Softmax with default axis", () => {
      const softmax = new Softmax();
      expect(softmax).toBeDefined();
      expect(softmax.toString()).toBe("Softmax(axis=-1)");
    });

    it("should create Softmax with custom axis", () => {
      const softmax = new Softmax(0);
      expect(softmax.toString()).toBe("Softmax(axis=0)");
    });

    it("should output probabilities that sum to 1", () => {
      const softmax = new Softmax();
      const input = tensor([1, 2, 3, 4]);
      const output = softmax.forward(input);

      const sum = numData(output).reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1.0, 5);
    });

    it("should handle 2D inputs", () => {
      const softmax = new Softmax(-1);
      const input = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const output = softmax.forward(input);

      expect(output.shape).toEqual([2, 3]);
    });

    it("should output values in (0, 1) range", () => {
      const softmax = new Softmax();
      const input = tensor([1, 2, 3]);
      const output = softmax.forward(input);

      for (let i = 0; i < output.size; i++) {
        const val = output.data[i] as number;
        expect(val).toBeGreaterThan(0);
        expect(val).toBeLessThan(1);
      }
    });
  });

  describe("LogSoftmax", () => {
    it("should create LogSoftmax with default axis", () => {
      const logSoftmax = new LogSoftmax();
      expect(logSoftmax).toBeDefined();
      expect(logSoftmax.toString()).toBe("LogSoftmax(axis=-1)");
    });

    it("should create LogSoftmax with custom axis", () => {
      const logSoftmax = new LogSoftmax(0);
      expect(logSoftmax.toString()).toBe("LogSoftmax(axis=0)");
    });

    it("should apply log softmax activation", () => {
      const logSoftmax = new LogSoftmax();
      const input = tensor([1, 2, 3]);
      const output = logSoftmax.forward(input);

      expect(output.shape).toEqual([3]);
    });

    it("should output negative values", () => {
      const logSoftmax = new LogSoftmax();
      const input = tensor([1, 2, 3, 4]);
      const output = logSoftmax.forward(input);

      for (let i = 0; i < output.size; i++) {
        expect(output.data[i]).toBeLessThanOrEqual(0);
      }
    });
  });

  describe("Softplus", () => {
    it("should create Softplus layer", () => {
      const softplus = new Softplus();
      expect(softplus).toBeDefined();
      expect(softplus.toString()).toBe("Softplus()");
    });

    it("should apply softplus activation", () => {
      const softplus = new Softplus();
      const input = tensor([0]);
      const output = softplus.forward(input);

      expect(output.data[0]).toBeCloseTo(Math.log(2), 5);
    });

    it("should output positive values", () => {
      const softplus = new Softplus();
      const input = tensor([-10, -5, 0, 5, 10]);
      const output = softplus.forward(input);

      for (let i = 0; i < output.size; i++) {
        expect(output.data[i]).toBeGreaterThan(0);
      }
    });

    it("should approximate ReLU for large positive values", () => {
      const softplus = new Softplus();
      const input = tensor([10]);
      const output = softplus.forward(input);

      expect(output.data[0]).toBeCloseTo(10, 0);
    });
  });

  describe("Swish", () => {
    it("should create Swish layer", () => {
      const swish = new Swish();
      expect(swish).toBeDefined();
      expect(swish.toString()).toBe("Swish()");
    });

    it("should apply swish activation", () => {
      const swish = new Swish();
      const input = tensor([0, 1, -1]);
      const output = swish.forward(input);

      expect(output.shape).toEqual([3]);
    });

    it("should handle 2D inputs", () => {
      const swish = new Swish();
      const input = tensor([
        [0, 1],
        [-1, 2],
      ]);
      const output = swish.forward(input);

      expect(output.shape).toEqual([2, 2]);
    });
  });

  describe("Mish", () => {
    it("should create Mish layer", () => {
      const mish = new Mish();
      expect(mish).toBeDefined();
      expect(mish.toString()).toBe("Mish()");
    });

    it("should apply mish activation", () => {
      const mish = new Mish();
      const input = tensor([0, 1, -1]);
      const output = mish.forward(input);

      expect(output.shape).toEqual([3]);
    });

    it("should handle 2D inputs", () => {
      const mish = new Mish();
      const input = tensor([
        [0, 1],
        [-1, 2],
      ]);
      const output = mish.forward(input);

      expect(output.shape).toEqual([2, 2]);
    });
  });

  describe("Training/Eval mode", () => {
    it("should respect training mode for all activations", () => {
      const relu = new ReLU();
      expect(relu.training).toBe(true);

      relu.eval();
      expect(relu.training).toBe(false);

      relu.train();
      expect(relu.training).toBe(true);
    });
  });
});
