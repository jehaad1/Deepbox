import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { fft, fft2, ifft, ifft2, irfft, rfft } from "../src/ndarray/ops/fft";

describe("fft", () => {
  it("computes FFT of power-of-2 length", () => {
    const t = tensor([1, 0, 0, 0]);
    const { real, imag } = fft(t);
    expect(real.shape).toEqual([4]);
    expect(imag.shape).toEqual([4]);
  });

  it("computes FFT of non-power-of-2 length (Bluestein)", () => {
    const t = tensor([1, 2, 3, 4, 5]);
    const { real, imag } = fft(t);
    expect(real.shape).toEqual([5]);
    expect(imag.shape).toEqual([5]);
  });

  it("computes FFT with explicit n", () => {
    const t = tensor([1, 2, 3]);
    const { real } = fft(t, 8);
    expect(real.shape).toEqual([8]);
  });

  it("computes FFT with n < input length (truncation)", () => {
    const t = tensor([1, 2, 3, 4, 5, 6, 7, 8]);
    const { real } = fft(t, 4);
    expect(real.shape).toEqual([4]);
  });
});

describe("ifft", () => {
  it("inverts FFT", () => {
    const t = tensor([1, 2, 3, 4]);
    const { real: fReal, imag: fImag } = fft(t);
    const { real, imag: _imag } = ifft(fReal, fImag);
    expect(real.shape).toEqual([4]);
    const data = Array.from(real.data as Float64Array);
    expect(data[0]).toBeCloseTo(1);
    expect(data[1]).toBeCloseTo(2);
  });

  it("ifft with explicit n", () => {
    const r = tensor([1, 0, 0, 0]);
    const i = tensor([0, 0, 0, 0]);
    const { real } = ifft(r, i, 8);
    expect(real.shape).toEqual([8]);
  });

  it("ifft with non-power-of-2", () => {
    const t = tensor([1, 2, 3, 4, 5]);
    const { real: fR, imag: fI } = fft(t);
    const { real } = ifft(fR, fI);
    expect(real.shape).toEqual([5]);
  });
});

describe("rfft", () => {
  it("computes real FFT", () => {
    const t = tensor([1, 2, 3, 4]);
    const { real, imag } = rfft(t);
    expect(real.shape[0]).toBe(3); // n/2 + 1
    expect(imag.shape[0]).toBe(3);
  });

  it("rfft with explicit n", () => {
    const t = tensor([1, 2, 3]);
    const { real } = rfft(t, 8);
    expect(real.shape[0]).toBe(5); // 8/2 + 1
  });

  it("rfft non-power-of-2", () => {
    const t = tensor([1, 2, 3, 4, 5]);
    const { real } = rfft(t);
    expect(real.shape[0]).toBe(3); // 5/2 + 1
  });
});

describe("irfft", () => {
  it("inverts rfft", () => {
    const t = tensor([1, 2, 3, 4]);
    const { real: fR, imag: fI } = rfft(t);
    const { real } = irfft(fR, fI);
    expect(real.shape[0]).toBe(4);
  });

  it("irfft with explicit n", () => {
    const r = tensor([1, 0, 0]);
    const i = tensor([0, 0, 0]);
    const { real } = irfft(r, i, 8);
    expect(real.shape[0]).toBe(8);
  });
});

describe("fft2", () => {
  it("computes 2D FFT", () => {
    const t = tensor([
      [1, 0],
      [0, 0],
    ]);
    const { real, imag } = fft2(t);
    expect(real.shape).toEqual([2, 2]);
    expect(imag.shape).toEqual([2, 2]);
  });

  it("non-square 2D FFT", () => {
    const t = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const { real } = fft2(t);
    expect(real.shape).toEqual([2, 3]);
  });
});

describe("ifft2", () => {
  it("inverts fft2", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const { real: fR, imag: fI } = fft2(t);
    const { real } = ifft2(fR, fI);
    expect(real.shape).toEqual([2, 2]);
    const data = Array.from(real.data as Float64Array);
    expect(data[0]).toBeCloseTo(1);
  });
});
