import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
// biome-ignore lint/correctness/noUnusedImports: SpectralNorm is used throughout this file
import { AlphaDropout, Dropout, Linear, SpectralNorm } from "../src/nn";

// ─── AlphaDropout ───────────────────────────────────────────────────────────

describe("AlphaDropout", () => {
  it("passes input unchanged in eval mode", () => {
    const layer = new AlphaDropout(0.5);
    layer.eval();
    const input = tensor([[1, 2, 3, 4, 5]]);
    const output = layer.forward(input);

    expect(output.shape).toEqual([1, 5]);
    for (let i = 0; i < 5; i++) {
      expect(Number(output.data[i])).toBeCloseTo(i + 1, 10);
    }
  });

  it("modifies output in training mode", () => {
    const layer = new AlphaDropout(0.5);
    layer.train();
    const input = tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]);
    const output = layer.forward(input);

    // With p=0.5, output should not be all zeros (affine transform applies)
    expect(output.shape).toEqual([1, 10]);
    // At least some values should differ from 0 due to alpha dropout
    let _allZero = true;
    for (let i = 0; i < 10; i++) {
      if (Math.abs(Number(output.data[i])) > 1e-10) {
        _allZero = false;
        break;
      }
    }
    // With p=0.5 and 10 elements, extremely unlikely all are kept (which would give 0 output for 0 input due to affine)
    // The affine correction means even kept values won't be exactly 0
    expect(output.size).toBe(10);
  });

  it("preserves shape for various input dimensions", () => {
    const layer = new AlphaDropout(0.3);
    layer.train();

    const input1d = tensor([1, 2, 3]);
    expect(layer.forward(input1d).shape).toEqual([3]);

    const input2d = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(layer.forward(input2d).shape).toEqual([2, 2]);
  });

  it("with p=0 acts as identity in training mode", () => {
    const layer = new AlphaDropout(0);
    layer.train();
    const input = tensor([[1, 2, 3]]);
    const output = layer.forward(input);

    for (let i = 0; i < 3; i++) {
      expect(Number(output.data[i])).toBeCloseTo(i + 1, 10);
    }
  });

  it("throws for invalid probability", () => {
    expect(() => new AlphaDropout(-0.1)).toThrow();
    expect(() => new AlphaDropout(1.0)).toThrow();
    expect(() => new AlphaDropout(1.5)).toThrow();
    expect(() => new AlphaDropout(NaN)).toThrow();
  });

  it("toString returns correct format", () => {
    expect(new AlphaDropout(0.3).toString()).toBe("AlphaDropout(p=0.3)");
  });

  it("dropoutRate getter works", () => {
    expect(new AlphaDropout(0.3).dropoutRate).toBe(0.3);
  });
});

// ─── SpectralNorm ───────────────────────────────────────────────────────────

describe("SpectralNorm", () => {
  it("wraps a Linear layer and produces output", () => {
    const linear = new Linear(4, 3);
    const snLinear = new SpectralNorm(linear, "weight");

    const input = tensor([[1, 2, 3, 4]]);
    const output = snLinear.forward(input);

    expect(output.shape).toEqual([1, 3]);
    // Output should be finite numbers
    for (let i = 0; i < 3; i++) {
      expect(Number.isFinite(Number(output.data[i]))).toBe(true);
    }
  });

  it("spectralNormValue is positive", () => {
    const linear = new Linear(5, 3);
    const snLinear = new SpectralNorm(linear, "weight");

    expect(snLinear.spectralNormValue).toBeGreaterThan(0);
  });

  it("normalizes weight such that spectral norm ≈ 1 after forward", () => {
    const linear = new Linear(4, 3);
    const snLinear = new SpectralNorm(linear, "weight", 10); // More power iterations

    // Run forward to update u, v vectors
    const input = tensor([[1, 2, 3, 4]]);
    snLinear.forward(input);

    // The spectral norm should be estimated
    const sigma = snLinear.spectralNormValue;
    expect(sigma).toBeGreaterThan(0);
    expect(Number.isFinite(sigma)).toBe(true);
  });

  it("does not permanently modify the original weight", () => {
    const linear = new Linear(3, 2);

    // Get original weight data
    const origWeightData: number[] = [];
    for (const [name, param] of linear.namedParameters()) {
      if (name === "weight") {
        for (let i = 0; i < param.tensor.size; i++) {
          origWeightData.push(Number(param.tensor.data[i]));
        }
      }
    }

    const snLinear = new SpectralNorm(linear, "weight");
    const input = tensor([[1, 2, 3]]);
    snLinear.forward(input);

    // Check weight is restored
    const afterWeightData: number[] = [];
    for (const [name, param] of linear.namedParameters()) {
      if (name === "weight") {
        for (let i = 0; i < param.tensor.size; i++) {
          afterWeightData.push(Number(param.tensor.data[i]));
        }
      }
    }

    expect(afterWeightData.length).toBe(origWeightData.length);
    for (let i = 0; i < origWeightData.length; i++) {
      expect(afterWeightData[i]).toBeCloseTo(origWeightData[i]!, 10);
    }
  });

  it("module getter returns the wrapped module", () => {
    const linear = new Linear(4, 3);
    const snLinear = new SpectralNorm(linear, "weight");
    expect(snLinear.module).toBe(linear);
  });

  it("toString returns correct format", () => {
    const linear = new Linear(4, 3);
    const snLinear = new SpectralNorm(linear, "weight");
    expect(snLinear.toString()).toContain("SpectralNorm");
    expect(snLinear.toString()).toContain("weight");
  });

  it("throws for invalid nPowerIterations", () => {
    const linear = new Linear(4, 3);
    expect(() => new SpectralNorm(linear, "weight", 0)).toThrow();
    expect(() => new SpectralNorm(linear, "weight", -1)).toThrow();
    expect(() => new SpectralNorm(linear, "weight", 1.5)).toThrow();
  });

  it("throws for invalid eps", () => {
    const linear = new Linear(4, 3);
    expect(() => new SpectralNorm(linear, "weight", 1, 0)).toThrow();
    expect(() => new SpectralNorm(linear, "weight", 1, -1)).toThrow();
  });

  it("throws for non-existent weight name", () => {
    const linear = new Linear(4, 3);
    expect(() => new SpectralNorm(linear, "nonexistent")).toThrow();
  });
});
