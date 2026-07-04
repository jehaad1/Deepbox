import { describe, expect, it } from "vitest";
import { fft, fft2, ifft, ifft2, irfft, rfft, tensor } from "../src/ndarray";

function expectClose(actual: number[], expected: number[], tol = 1e-10): void {
  expect(actual).toHaveLength(expected.length);
  for (let i = 0; i < actual.length; i++) {
    expect(actual[i]).toBeCloseTo(expected[i]!, tol > 1e-6 ? 4 : 10);
  }
}

describe("fft", () => {
  it("transforms a delta function (all ones in frequency domain)", () => {
    const t = tensor([1, 0, 0, 0]);
    const { real, imag } = fft(t);
    expectClose(real.toArray() as number[], [1, 1, 1, 1]);
    expectClose(imag.toArray() as number[], [0, 0, 0, 0]);
  });

  it("transforms a constant signal (delta in frequency domain)", () => {
    const t = tensor([1, 1, 1, 1]);
    const { real, imag } = fft(t);
    expectClose(real.toArray() as number[], [4, 0, 0, 0]);
    expectClose(imag.toArray() as number[], [0, 0, 0, 0]);
  });

  it("transforms a simple sinusoid (power of 2)", () => {
    // cos(2*pi*k/8) for k=0..7 => frequency bin 1 and 7
    const n = 8;
    const data: number[] = [];
    for (let k = 0; k < n; k++) {
      data.push(Math.cos((2 * Math.PI * k) / n));
    }
    const t = tensor(data);
    const { real, imag } = fft(t);
    const re = real.toArray() as number[];
    const im = imag.toArray() as number[];
    // Bins 1 and 7 should have magnitude n/2 = 4
    expect(Math.abs(re[1]!)).toBeCloseTo(4, 10);
    expect(Math.abs(re[7]!)).toBeCloseTo(4, 10);
    // Other bins should be ~0
    expect(Math.abs(re[0]!)).toBeCloseTo(0, 10);
    expect(Math.abs(re[4]!)).toBeCloseTo(0, 10);
    expect(Math.abs(im[0]!)).toBeCloseTo(0, 10);
  });

  it("handles non-power-of-2 length via Bluestein", () => {
    const t = tensor([1, 0, 0, 0, 0]); // length 5
    const { real, imag } = fft(t);
    expect(real.shape).toEqual([5]);
    expect(imag.shape).toEqual([5]);
    // DC component should be 1
    expect((real.toArray() as number[])[0]).toBeCloseTo(1, 10);
    // All frequency components should have magnitude 1 for a delta
    const re = real.toArray() as number[];
    const im = imag.toArray() as number[];
    for (let i = 0; i < 5; i++) {
      const mag = Math.sqrt(re[i]! ** 2 + im[i]! ** 2);
      expect(mag).toBeCloseTo(1, 10);
    }
  });

  it("supports zero-padding via n parameter", () => {
    const t = tensor([1, 2]);
    const { real } = fft(t, 4);
    expect(real.shape).toEqual([4]);
    // DFT of [1, 2, 0, 0]
    const re = real.toArray() as number[];
    expect(re[0]).toBeCloseTo(3, 10); // sum
  });

  it("works on 2D input (batched along first axis)", () => {
    const t = tensor([
      [1, 0, 0, 0],
      [0, 1, 0, 0],
    ]);
    const { real, imag } = fft(t);
    expect(real.shape).toEqual([2, 4]);
    expect(imag.shape).toEqual([2, 4]);
    // First row: delta at 0 => all ones
    const re = real.toArray() as number[][];
    expectClose(re[0]!, [1, 1, 1, 1]);
  });

  it("preserves float64 dtype", () => {
    const t = tensor([1, 2, 3, 4], { dtype: "float64" });
    const { real } = fft(t);
    expect(real.dtype).toBe("float64");
  });

  // Error cases
  it("throws on 0D input", () => {
    const t = tensor(5);
    expect(() => fft(t)).toThrow();
  });

  it("throws on string dtype", () => {
    const t = tensor(["a", "b"]);
    expect(() => fft(t)).toThrow();
  });
});

describe("ifft", () => {
  it("inverts fft (round-trip)", () => {
    const original = tensor([1, 2, 3, 4]);
    const { real: fReal, imag: fImag } = fft(original);
    const { real: recovered } = ifft(fReal, fImag);
    expectClose(recovered.toArray() as number[], [1, 2, 3, 4]);
  });

  it("round-trips non-power-of-2 length", () => {
    const original = tensor([1, 2, 3, 4, 5]);
    const { real: fReal, imag: fImag } = fft(original);
    const { real: recovered } = ifft(fReal, fImag);
    expectClose(recovered.toArray() as number[], [1, 2, 3, 4, 5]);
  });

  it("round-trips 2D batched input", () => {
    const original = tensor([
      [1, 2, 3, 4],
      [5, 6, 7, 8],
    ]);
    const { real: fReal, imag: fImag } = fft(original);
    const { real: recovered } = ifft(fReal, fImag);
    const re = recovered.toArray() as number[][];
    expectClose(re[0]!, [1, 2, 3, 4]);
    expectClose(re[1]!, [5, 6, 7, 8]);
  });
});

describe("rfft", () => {
  it("returns only positive frequencies", () => {
    const t = tensor([1, 0, 0, 0]);
    const { real, imag } = rfft(t);
    // For n=4, rfft returns floor(4/2)+1 = 3 elements
    expect(real.shape).toEqual([3]);
    expect(imag.shape).toEqual([3]);
    expectClose(real.toArray() as number[], [1, 1, 1]);
    expectClose(imag.toArray() as number[], [0, 0, 0]);
  });

  it("handles odd-length input", () => {
    const t = tensor([1, 2, 3, 4, 5]);
    const { real, imag } = rfft(t);
    // floor(5/2)+1 = 3
    expect(real.shape).toEqual([3]);
    expect(imag.shape).toEqual([3]);
  });
});

describe("irfft", () => {
  it("inverts rfft (round-trip)", () => {
    const original = tensor([1, 2, 3, 4]);
    const { real: rReal, imag: rImag } = rfft(original);
    const { real: recovered } = irfft(rReal, rImag);
    expectClose(recovered.toArray() as number[], [1, 2, 3, 4]);
  });

  it("round-trips with explicit n", () => {
    const original = tensor([1, 2, 3, 4, 5, 6]);
    const { real: rReal, imag: rImag } = rfft(original);
    const { real: recovered } = irfft(rReal, rImag, 6);
    expectClose(recovered.toArray() as number[], [1, 2, 3, 4, 5, 6]);
  });
});

describe("fft2", () => {
  it("transforms a 2D delta function", () => {
    const t = tensor([
      [1, 0],
      [0, 0],
    ]);
    const { real, imag } = fft2(t);
    expect(real.shape).toEqual([2, 2]);
    expect(imag.shape).toEqual([2, 2]);
    // 2D DFT of delta => all ones
    const re = real.toArray() as number[][];
    expectClose(re[0]!, [1, 1]);
    expectClose(re[1]!, [1, 1]);
    const im = imag.toArray() as number[][];
    expectClose(im[0]!, [0, 0]);
    expectClose(im[1]!, [0, 0]);
  });

  it("transforms a 2D constant", () => {
    const t = tensor([
      [1, 1],
      [1, 1],
    ]);
    const { real, imag: _imag } = fft2(t);
    const re = real.toArray() as number[][];
    // DC component = sum = 4
    expect(re[0]![0]).toBeCloseTo(4, 10);
    // Other components should be 0
    expect(re[0]![1]).toBeCloseTo(0, 10);
    expect(re[1]![0]).toBeCloseTo(0, 10);
    expect(re[1]![1]).toBeCloseTo(0, 10);
  });

  it("throws on 1D input", () => {
    const t = tensor([1, 2, 3]);
    expect(() => fft2(t)).toThrow();
  });
});

describe("ifft2", () => {
  it("round-trips with fft2", () => {
    const original = tensor([
      [1, 2],
      [3, 4],
    ]);
    const { real: fReal, imag: fImag } = fft2(original);
    const { real: recovered } = ifft2(fReal, fImag);
    const re = recovered.toArray() as number[][];
    expectClose(re[0]!, [1, 2]);
    expectClose(re[1]!, [3, 4]);
  });

  it("round-trips 4x4 matrix", () => {
    const data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
      [13, 14, 15, 16],
    ];
    const original = tensor(data);
    const { real: fReal, imag: fImag } = fft2(original);
    const { real: recovered } = ifft2(fReal, fImag);
    const re = recovered.toArray() as number[][];
    for (let i = 0; i < 4; i++) {
      expectClose(re[i]!, data[i]!);
    }
  });
});
