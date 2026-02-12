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
} from "../src/nn/layers/activations";
import { expectNumberArray, expectNumberArray2D } from "./nn-test-utils";

describe("deepbox/nn - Activation Layers", () => {
  describe("ReLU", () => {
    it("should apply ReLU activation correctly", () => {
      const relu = new ReLU();
      const input = tensor([-2, -1, 0, 1, 2]);
      const output = relu.forward(input);
      expect(output.toArray()).toEqual([0, 0, 0, 1, 2]);
    });

    it("should handle 2D input", () => {
      const relu = new ReLU();
      const input = tensor([
        [-1, 2],
        [3, -4],
      ]);
      const output = relu.forward(input);
      expect(output.toArray()).toEqual([
        [0, 2],
        [3, 0],
      ]);
    });

    it("should return correct toString", () => {
      const relu = new ReLU();
      expect(relu.toString()).toBe("ReLU()");
    });
  });

  describe("Sigmoid", () => {
    it("should apply sigmoid activation correctly", () => {
      const sigmoid = new Sigmoid();
      const input = tensor([0]);
      const output = sigmoid.forward(input);
      const arr = expectNumberArray(output.toArray(), "Sigmoid output");
      expect(arr[0] ?? 0).toBeCloseTo(0.5, 5);
    });

    it("should squash large values", () => {
      const sigmoid = new Sigmoid();
      const input = tensor([-10, 10]);
      const output = sigmoid.forward(input);
      const arr = expectNumberArray(output.toArray(), "Sigmoid output");
      expect(arr[0] ?? 0).toBeLessThan(0.001);
      expect(arr[1] ?? 0).toBeGreaterThan(0.999);
    });

    it("should return correct toString", () => {
      const sigmoid = new Sigmoid();
      expect(sigmoid.toString()).toBe("Sigmoid()");
    });
  });

  describe("Tanh", () => {
    it("should apply tanh activation correctly", () => {
      const tanh = new Tanh();
      const input = tensor([0]);
      const output = tanh.forward(input);
      const arr = expectNumberArray(output.toArray(), "Tanh output");
      expect(arr[0] ?? 0).toBeCloseTo(0, 5);
    });

    it("should squash to [-1, 1] range", () => {
      const tanh = new Tanh();
      const input = tensor([-10, 10]);
      const output = tanh.forward(input);
      const arr = expectNumberArray(output.toArray(), "Tanh output");
      expect(arr[0]).toBeCloseTo(-1, 3);
      expect(arr[1]).toBeCloseTo(1, 3);
    });

    it("should return correct toString", () => {
      const tanh = new Tanh();
      expect(tanh.toString()).toBe("Tanh()");
    });
  });

  describe("LeakyReLU", () => {
    it("should apply leaky ReLU with default alpha", () => {
      const leakyRelu = new LeakyReLU();
      const input = tensor([-10, 0, 10]);
      const output = leakyRelu.forward(input);
      const arr = expectNumberArray(output.toArray(), "LeakyReLU output");
      expect(arr[0]).toBeCloseTo(-0.1, 5);
      expect(arr[1]).toBe(0);
      expect(arr[2]).toBe(10);
    });

    it("should apply leaky ReLU with custom alpha", () => {
      const leakyRelu = new LeakyReLU(0.2);
      const input = tensor([-5, 5]);
      const output = leakyRelu.forward(input);
      const arr = expectNumberArray(output.toArray(), "LeakyReLU output");
      expect(arr[0]).toBeCloseTo(-1, 5);
      expect(arr[1]).toBe(5);
    });

    it("should return correct toString", () => {
      const leakyRelu = new LeakyReLU(0.1);
      expect(leakyRelu.toString()).toBe("LeakyReLU(alpha=0.1)");
    });
  });

  describe("ELU", () => {
    it("should apply ELU with default alpha", () => {
      const elu = new ELU();
      const input = tensor([0, 1, -1]);
      const output = elu.forward(input);
      const arr = expectNumberArray(output.toArray(), "ELU output");
      expect(arr[0]).toBe(0);
      expect(arr[1]).toBe(1);
      expect(arr[2]).toBeCloseTo(Math.exp(-1) - 1, 5);
    });

    it("should apply ELU with custom alpha", () => {
      const elu = new ELU(2.0);
      const input = tensor([-1]);
      const output = elu.forward(input);
      const arr = expectNumberArray(output.toArray(), "ELU output");
      expect(arr[0]).toBeCloseTo(2.0 * (Math.exp(-1) - 1), 5);
    });

    it("should return correct toString", () => {
      const elu = new ELU(1.5);
      expect(elu.toString()).toBe("ELU(alpha=1.5)");
    });
  });

  describe("GELU", () => {
    it("should apply GELU activation", () => {
      const gelu = new GELU();
      const input = tensor([0, 1, -1]);
      const output = gelu.forward(input);
      const arr = expectNumberArray(output.toArray(), "GELU output");
      expect(arr[0]).toBeCloseTo(0, 5);
      expect(arr[1]).toBeGreaterThan(0.8);
      expect(arr[2]).toBeLessThan(0);
    });

    it("should return correct toString", () => {
      const gelu = new GELU();
      expect(gelu.toString()).toBe("GELU()");
    });
  });

  describe("Softmax", () => {
    it("should compute softmax along last axis by default", () => {
      const softmax = new Softmax();
      const input = tensor([1, 2, 3]);
      const output = softmax.forward(input);
      const arr = expectNumberArray(output.toArray(), "Softmax output");
      const sum = arr.reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1, 5);
    });

    it("should compute softmax along specified axis", () => {
      const softmax = new Softmax(0);
      const input = tensor([
        [1, 2],
        [3, 4],
      ]);
      const output = softmax.forward(input);
      const arr = expectNumberArray2D(output.toArray(), "Softmax output");
      expect((arr[0]?.[0] ?? 0) + (arr[1]?.[0] ?? 0)).toBeCloseTo(1, 5);
      expect((arr[0]?.[1] ?? 0) + (arr[1]?.[1] ?? 0)).toBeCloseTo(1, 5);
    });

    it("should return correct toString", () => {
      const softmax = new Softmax(-1);
      expect(softmax.toString()).toBe("Softmax(axis=-1)");
    });
  });

  describe("LogSoftmax", () => {
    it("should compute log softmax", () => {
      const logSoftmax = new LogSoftmax();
      const input = tensor([1, 2, 3]);
      const output = logSoftmax.forward(input);
      const arr = expectNumberArray(output.toArray(), "LogSoftmax output");
      // All values should be negative (log of probability < 1)
      for (const val of arr) {
        expect(val).toBeLessThan(0);
      }
      // exp(log_softmax) should sum to 1
      const expSum = arr.reduce((a, b) => a + Math.exp(b), 0);
      expect(expSum).toBeCloseTo(1, 5);
    });

    it("should return correct toString", () => {
      const logSoftmax = new LogSoftmax(0);
      expect(logSoftmax.toString()).toBe("LogSoftmax(axis=0)");
    });
  });

  describe("Softplus", () => {
    it("should compute softplus", () => {
      const softplus = new Softplus();
      const input = tensor([0, 1, -1]);
      const output = softplus.forward(input);
      const arr = expectNumberArray(output.toArray(), "Softplus output");
      expect(arr[0]).toBeCloseTo(Math.log(2), 5);
      expect(arr[1]).toBeCloseTo(Math.log(1 + Math.exp(1)), 5);
      expect(arr[2]).toBeCloseTo(Math.log(1 + Math.exp(-1)), 5);
    });

    it("should always be positive", () => {
      const softplus = new Softplus();
      const input = tensor([-10, -5, 0, 5, 10]);
      const output = softplus.forward(input);
      const arr = expectNumberArray(output.toArray(), "Softplus output");
      for (const val of arr) {
        expect(val).toBeGreaterThan(0);
      }
    });

    it("should return correct toString", () => {
      const softplus = new Softplus();
      expect(softplus.toString()).toBe("Softplus()");
    });
  });

  describe("Swish", () => {
    it("should compute swish (x * sigmoid(x))", () => {
      const swish = new Swish();
      const input = tensor([0, 1, -1]);
      const output = swish.forward(input);
      const arr = expectNumberArray(output.toArray(), "Swish output");
      expect(arr[0]).toBeCloseTo(0, 5);
      expect(arr[1]).toBeCloseTo(1 / (1 + Math.exp(-1)), 5);
      expect(arr[2]).toBeCloseTo(-1 / (1 + Math.exp(1)), 5);
    });

    it("should return correct toString", () => {
      const swish = new Swish();
      expect(swish.toString()).toBe("Swish()");
    });
  });

  describe("Mish", () => {
    it("should compute mish (x * tanh(softplus(x)))", () => {
      const mish = new Mish();
      const input = tensor([0, 1, -1]);
      const output = mish.forward(input);
      const arr = expectNumberArray(output.toArray(), "Mish output");
      expect(arr[0]).toBeCloseTo(0, 5);
      expect(arr[1]).toBeGreaterThan(0.8);
      expect(arr[2]).toBeLessThan(0);
    });

    it("should return correct toString", () => {
      const mish = new Mish();
      expect(mish.toString()).toBe("Mish()");
    });
  });

  describe("Training mode inheritance", () => {
    it("should inherit training mode from Module", () => {
      const relu = new ReLU();
      expect(relu.training).toBe(true);
      relu.eval();
      expect(relu.training).toBe(false);
      relu.train();
      expect(relu.training).toBe(true);
    });
  });
});
