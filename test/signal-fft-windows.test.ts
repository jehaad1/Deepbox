import { describe, expect, it } from "vitest";
import {
  bartlettWindow,
  blackmanWindow,
  convolve,
  correlate,
  fft,
  fftn,
  hammingWindow,
  hannWindow,
  ifftn,
  kaiserWindow,
  tensor,
} from "../src/ndarray";

const f64 = { dtype: "float64" as const };

// ---------------------------------------------------------------------------
// fftn / ifftn
// ---------------------------------------------------------------------------
describe("fftn", () => {
  it("should match fft for 1D input", () => {
    const t = tensor([1, 2, 3, 4], f64);
    const ref = fft(t);
    const nd = fftn(t);
    for (let i = 0; i < 4; i++) {
      expect(Number(nd.real.data[i])).toBeCloseTo(Number(ref.real.data[i]), 10);
      expect(Number(nd.imag.data[i])).toBeCloseTo(Number(ref.imag.data[i]), 10);
    }
  });

  it("should transform 2D input", () => {
    const t = tensor(
      [
        [1, 0],
        [0, 0],
      ],
      f64
    );
    const result = fftn(t);
    expect(result.real.shape).toEqual([2, 2]);
    expect(result.imag.shape).toEqual([2, 2]);
    // DC component = sum of all elements = 1
    expect(Number(result.real.data[0])).toBeCloseTo(1, 10);
  });

  it("should transform 3D input", () => {
    const t = tensor(
      Array.from({ length: 8 }, (_, i) => i + 1),
      f64
    ).reshape([2, 2, 2]);
    const result = fftn(t);
    expect(result.real.shape).toEqual([2, 2, 2]);
    // DC = sum of all elements = 36
    expect(Number(result.real.data[0])).toBeCloseTo(36, 8);
  });

  it("should support specific axes", () => {
    const t = tensor(
      [
        [1, 2],
        [3, 4],
      ],
      f64
    );
    const result = fftn(t, [1]); // FFT only along columns
    expect(result.real.shape).toEqual([2, 2]);
  });

  it("should throw on string dtype", () => {
    expect(() => fftn(tensor(["a", "b"]))).toThrow();
  });
});

describe("ifftn", () => {
  it("should invert fftn for 2D", () => {
    const t = tensor(
      [
        [1, 2],
        [3, 4],
      ],
      f64
    );
    const freq = fftn(t);
    const inv = ifftn(freq.real, freq.imag);
    for (let i = 0; i < 4; i++) {
      expect(Number(inv.real.data[i])).toBeCloseTo(Number(t.data[i]), 10);
      expect(Math.abs(Number(inv.imag.data[i]))).toBeLessThan(1e-10);
    }
  });

  it("should invert fftn for 3D", () => {
    const t = tensor(
      Array.from({ length: 8 }, (_, i) => i + 1),
      f64
    ).reshape([2, 2, 2]);
    const freq = fftn(t);
    const inv = ifftn(freq.real, freq.imag);
    for (let i = 0; i < 8; i++) {
      expect(Number(inv.real.data[i])).toBeCloseTo(i + 1, 8);
    }
  });
});

// ---------------------------------------------------------------------------
// Window functions
// ---------------------------------------------------------------------------
describe("hannWindow", () => {
  it("should return [1] for n=1", () => {
    const w = hannWindow(1);
    expect(w.shape).toEqual([1]);
    expect(Number(w.data[0])).toBe(1);
  });

  it("should start and end at 0", () => {
    const w = hannWindow(10);
    expect(w.shape).toEqual([10]);
    expect(Number(w.data[0])).toBeCloseTo(0, 10);
    expect(Number(w.data[9])).toBeCloseTo(0, 10);
  });

  it("should be symmetric", () => {
    const w = hannWindow(8);
    for (let i = 0; i < 4; i++) {
      expect(Number(w.data[i])).toBeCloseTo(Number(w.data[7 - i]), 10);
    }
  });

  it("should throw for n < 1", () => {
    expect(() => hannWindow(0)).toThrow();
  });
});

describe("hammingWindow", () => {
  it("should return [1] for n=1", () => {
    const w = hammingWindow(1);
    expect(Number(w.data[0])).toBe(1);
  });

  it("should be symmetric", () => {
    const w = hammingWindow(8);
    for (let i = 0; i < 4; i++) {
      expect(Number(w.data[i])).toBeCloseTo(Number(w.data[7 - i]), 10);
    }
  });

  it("should have non-zero endpoints (unlike Hann)", () => {
    const w = hammingWindow(10);
    expect(Number(w.data[0])).toBeGreaterThan(0);
  });
});

describe("blackmanWindow", () => {
  it("should start and end near 0", () => {
    const w = blackmanWindow(10);
    expect(Math.abs(Number(w.data[0]))).toBeLessThan(1e-10);
    expect(Math.abs(Number(w.data[9]))).toBeLessThan(1e-10);
  });

  it("should be symmetric", () => {
    const w = blackmanWindow(8);
    for (let i = 0; i < 4; i++) {
      expect(Number(w.data[i])).toBeCloseTo(Number(w.data[7 - i]), 10);
    }
  });
});

describe("bartlettWindow", () => {
  it("should start and end at 0", () => {
    const w = bartlettWindow(10);
    expect(Number(w.data[0])).toBeCloseTo(0, 10);
    expect(Number(w.data[9])).toBeCloseTo(0, 10);
  });

  it("should peak at center", () => {
    const w = bartlettWindow(11);
    // Center index = 5 should be 1
    expect(Number(w.data[5])).toBeCloseTo(1, 10);
  });
});

describe("kaiserWindow", () => {
  it("should be symmetric", () => {
    const w = kaiserWindow(8, 5);
    for (let i = 0; i < 4; i++) {
      expect(Number(w.data[i])).toBeCloseTo(Number(w.data[7 - i]), 10);
    }
  });

  it("should be flat when beta=0", () => {
    const w = kaiserWindow(5, 0);
    for (let i = 0; i < 5; i++) {
      expect(Number(w.data[i])).toBeCloseTo(1, 10);
    }
  });

  it("should throw for n < 1", () => {
    expect(() => kaiserWindow(0)).toThrow();
  });
});

// ---------------------------------------------------------------------------
// convolve / correlate
// ---------------------------------------------------------------------------
describe("convolve", () => {
  it("should compute full convolution", () => {
    const a = tensor([1, 2, 3], f64);
    const v = tensor([0, 1, 0.5], f64);
    const out = convolve(a, v);
    // full length = 3 + 3 - 1 = 5
    expect(out.shape).toEqual([5]);
    // Manual: [0*1, 1*1+0*2, 0.5*1+1*2+0*3, 0.5*2+1*3, 0.5*3]
    // = [0, 1, 2.5, 4, 1.5]
    expect(Number(out.data[0])).toBeCloseTo(0, 10);
    expect(Number(out.data[1])).toBeCloseTo(1, 10);
    expect(Number(out.data[2])).toBeCloseTo(2.5, 10);
    expect(Number(out.data[3])).toBeCloseTo(4, 10);
    expect(Number(out.data[4])).toBeCloseTo(1.5, 10);
  });

  it("should compute same-mode convolution", () => {
    const a = tensor([1, 2, 3, 4, 5], f64);
    const v = tensor([1, 1, 1], f64);
    const out = convolve(a, v, "same");
    expect(out.shape).toEqual([5]);
  });

  it("should compute valid-mode convolution", () => {
    const a = tensor([1, 2, 3, 4, 5], f64);
    const v = tensor([1, 1, 1], f64);
    const out = convolve(a, v, "valid");
    // valid length = 5 - 3 + 1 = 3
    expect(out.shape).toEqual([3]);
    expect(Number(out.data[0])).toBeCloseTo(6, 10); // 1+2+3
    expect(Number(out.data[1])).toBeCloseTo(9, 10); // 2+3+4
    expect(Number(out.data[2])).toBeCloseTo(12, 10); // 3+4+5
  });

  it("should throw on non-1D input", () => {
    const a = tensor(
      [
        [1, 2],
        [3, 4],
      ],
      f64
    );
    const v = tensor([1, 2], f64);
    expect(() => convolve(a, v)).toThrow();
  });
});

describe("correlate", () => {
  it("should compute full cross-correlation", () => {
    const a = tensor([1, 2, 3], f64);
    const v = tensor([0, 1, 0.5], f64);
    const out = correlate(a, v);
    expect(out.shape).toEqual([5]);
  });

  it("should find signal in autocorrelation", () => {
    const a = tensor([0, 0, 1, 0, 0], f64);
    const out = correlate(a, a, "full");
    // Autocorrelation peak at center
    const center = 4; // index (len-1)
    expect(Number(out.data[center])).toBeCloseTo(1, 10);
  });

  it("should compute same-mode correlation", () => {
    const a = tensor([1, 2, 3, 4, 5], f64);
    const v = tensor([1, 1, 1], f64);
    const out = correlate(a, v, "same");
    expect(out.shape).toEqual([5]);
  });

  it("should compute valid-mode correlation", () => {
    const a = tensor([1, 2, 3, 4, 5], f64);
    const v = tensor([1, 1, 1], f64);
    const out = correlate(a, v, "valid");
    expect(out.shape).toEqual([3]);
  });
});
