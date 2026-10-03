/**
 * v1.5.0 regression tests for src/ndarray/ops: fft, einsum, insert/delete,
 * conv (im2col/col2im), logical ops and device dispatch.
 *
 * Reference values come from NumPy 2.4 (np.fft, np.einsum, np.insert, np.delete).
 */

import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type { DeviceBuffer, KernelBackend } from "../../src/core";
import {
  DeviceError,
  DTypeError,
  InvalidParameterError,
  registerBackend,
  ShapeError,
} from "../../src/core";
import { unregisterBackend } from "../../src/core/backend/registry";
import {
  col2im,
  contiguous,
  delete_,
  einsum,
  fft,
  fft2,
  fftn,
  ifft,
  ifft2,
  ifftn,
  im2col,
  insert,
  irfft,
  logicalAnd,
  logicalNot,
  logicalOr,
  logicalXor,
  rfft,
  slice,
  sum,
  tensor,
  transpose,
  where,
  zeros,
} from "../../src/ndarray";
import { dispatchPool2d } from "../../src/ndarray/ops/device_dispatch";
import { fftfreq, fftshift, ifftshift, rfftfreq } from "../../src/ndarray/ops/fft";
import type { Tensor } from "../../src/ndarray/tensor/Tensor";

/** Logical (row-major) values of a possibly strided tensor as plain numbers. */
function vals(t: Tensor): number[] {
  return Array.from(contiguous(t).data as ArrayLike<number | bigint>, (v) => Number(v));
}

function expectClose(actual: ArrayLike<number>, expected: number[], tol = 1e-10): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(Math.abs((actual[i] as number) - (expected[i] as number))).toBeLessThan(tol);
  }
}

// ---------------------------------------------------------------------------
// fft
// ---------------------------------------------------------------------------

describe("fft: accuracy", () => {
  it("long power-of-two transforms stay at double precision (no twiddle drift)", () => {
    const n = 1 << 17;
    const x = new Float64Array(n);
    for (let i = 0; i < n; i++) x[i] = Math.sin(0.37 * i) + ((i * 7) % 11) - 5;
    const t = tensor(Array.from(x), { dtype: "float64" });
    const { real, imag } = fft(t);
    // Direct DFT of a few bins with exactly reduced angles as the reference.
    for (const k of [1, 777, 40000, n - 3]) {
      let re = 0;
      let im = 0;
      for (let j = 0; j < n; j++) {
        const angle = (2 * Math.PI * ((j * k) % n)) / n;
        re += (x[j] as number) * Math.cos(angle);
        im -= (x[j] as number) * Math.sin(angle);
      }
      expect(Math.abs((real.data as Float64Array)[k]! - re)).toBeLessThan(1e-9);
      expect(Math.abs((imag.data as Float64Array)[k]! - im)).toBeLessThan(1e-9);
    }
  });

  it("Bluestein lengths round-trip to double precision", () => {
    const n = 30030;
    const data = Array.from({ length: n }, (_, i) => Math.cos(i * 0.01) * ((i % 13) - 6));
    const t = tensor(data, { dtype: "float64" });
    const spec = fft(t);
    const back = ifft(spec.real, spec.imag);
    expectClose(back.real.data as Float64Array, data, 1e-10);
  });

  it("stays correct when many large Bluestein plans pass through the plan cache", () => {
    // Each plan holds several MB, so these lengths push older plans out of the cache.
    for (let n = 60001; n <= 60009; n += 2) {
      const data = Array.from({ length: n }, (_, i) => ((i * 31) % 17) - 8);
      const spec = fft(tensor(data, { dtype: "float64" }));
      const k = 12345;
      let re = 0;
      let im = 0;
      for (let j = 0; j < n; j++) {
        const angle = (2 * Math.PI * ((j * k) % n)) / n;
        re += (data[j] as number) * Math.cos(angle);
        im -= (data[j] as number) * Math.sin(angle);
      }
      expect(Math.abs((spec.real.data as Float64Array)[k]! - re)).toBeLessThan(1e-8);
      expect(Math.abs((spec.imag.data as Float64Array)[k]! - im)).toBeLessThan(1e-8);
    }
  });

  it("returns exact values at quarter-turn twiddles", () => {
    const { real, imag } = fft(tensor([1, 2, 3, 4], { dtype: "float64" }));
    expect(Array.from(real.data as Float64Array)).toEqual([10, -2, -2, -2]);
    expect(Array.from(imag.data as Float64Array)).toEqual([0, 2, 0, -2]);
  });
});

describe("fft: dtype", () => {
  it("integer and bool inputs produce float64 (NumPy gives complex128)", () => {
    const x = [2147483, 1, 3, 4];
    for (const dtype of ["int32", "int64", "uint8", "bool"] as const) {
      const t = tensor(dtype === "bool" ? [1, 0, 1, 1] : x, { dtype });
      expect(fft(t).real.dtype).toBe("float64");
    }
  });

  it("keeps float32 for float32 input and float64 when either part is float64", () => {
    const a = tensor([1, 2, 3, 4], { dtype: "float32" });
    const b = tensor([0, 0, 0, 0], { dtype: "float64" });
    expect(fft(a).real.dtype).toBe("float32");
    expect(ifft(a, a).real.dtype).toBe("float32");
    expect(ifft(a, b).real.dtype).toBe("float64");
  });

  it("rejects string dtype with a DTypeError", () => {
    expect(() => fft(tensor(["a", "b"]))).toThrow(DTypeError);
  });
});

describe("fft: options", () => {
  const x = tensor([1, 2, 3, 4, 5], { dtype: "float64" });

  it("supports norm = ortho and forward", () => {
    const ortho = fft(x, undefined, -1, "ortho");
    expectClose(
      ortho.real.data as Float64Array,
      [
        6.7082039324993685, -1.118033988749895, -1.118033988749895, -1.118033988749895,
        -1.118033988749895,
      ]
    );
    expectClose(
      ortho.imag.data as Float64Array,
      [0, 1.5388417685876266, 0.36327126400268045, -0.36327126400268045, -1.5388417685876266]
    );
    const fwd = fft(x, undefined, -1, "forward");
    expectClose(fwd.real.data as Float64Array, [3, -0.5, -0.5, -0.5, -0.5]);
    const inv = ifft(x, tensor([0, 0, 0, 0, 0], { dtype: "float64" }), undefined, -1, "forward");
    expectClose(inv.real.data as Float64Array, [15, -2.5, -2.5, -2.5, -2.5]);
    expectClose(
      inv.imag.data as Float64Array,
      [0, -3.440954801177934, -0.8122992405822659, 0.8122992405822659, 3.440954801177934]
    );
  });

  it("rejects an unknown norm", () => {
    expect(() => fft(x, undefined, -1, "bogus" as never)).toThrow(InvalidParameterError);
  });

  it("transforms along an arbitrary axis", () => {
    const m = tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      { dtype: "float64" }
    );
    const { real, imag } = fft(m, undefined, 0);
    expect(real.shape).toEqual([2, 3]);
    expect(vals(real)).toEqual([5, 7, 9, -3, -3, -3]);
    expect(vals(imag)).toEqual([0, 0, 0, 0, 0, 0]);
  });

  it("reads strided (transposed) input correctly", () => {
    const m = tensor(
      [
        [1, 4],
        [2, 5],
        [3, 6],
      ],
      { dtype: "float64" }
    );
    const { real } = fft(transpose(m), undefined, 0);
    expect(vals(real)).toEqual([5, 7, 9, -3, -3, -3]);
  });

  it("rfft crops with n and keeps floor(n/2)+1 bins", () => {
    const { real, imag } = rfft(tensor([1, 2, 3, 4, 5, 6], { dtype: "float64" }), 4);
    expect(vals(real)).toEqual([10, -2, -2]);
    expect(vals(imag)).toEqual([0, 2, 0]);
  });

  it("validates n and axis", () => {
    expect(() => fft(x, 0)).toThrow(InvalidParameterError);
    expect(() => fft(x, 2.5)).toThrow(InvalidParameterError);
    expect(() => fft(x, undefined, 3)).toThrow(InvalidParameterError);
    expect(() => fft(tensor(5))).toThrow(ShapeError);
  });

  it("rejects an empty transform axis like NumPy", () => {
    const empty = zeros([1, 0], { dtype: "float64" });
    expect(() => fft(empty)).toThrow(InvalidParameterError);
  });
});

describe("ifft: input checks", () => {
  it("rejects real and imaginary parts with different shapes", () => {
    const re = tensor([1, 2, 3, 4], { dtype: "float64" });
    const im = tensor([0, 0, 0], { dtype: "float64" });
    expect(() => ifft(re, im)).toThrow(ShapeError);
    expect(() => ifftn(re, im)).toThrow(ShapeError);
    expect(() => irfft(re, im)).toThrow(ShapeError);
    expect(() =>
      ifft2(tensor([[1, 2]], { dtype: "float64" }), tensor([[1], [2]], { dtype: "float64" }))
    ).toThrow(ShapeError);
  });
});

describe("irfft", () => {
  it("ignores the imaginary parts of the DC and Nyquist terms and returns a zero imag", () => {
    // np.fft.irfft([1, 2, 3] + [5j, 0, 0], 4) == [2, -0.5, 0, -0.5]
    const { real, imag } = irfft(
      tensor([1, 2, 3], { dtype: "float64" }),
      tensor([5, 0, 0], { dtype: "float64" }),
      4
    );
    expectClose(real.data as Float64Array, [2, -0.5, 0, -0.5]);
    expect(Array.from(imag.data as Float64Array)).toEqual([0, 0, 0, 0]);

    // Nyquist imaginary part (the last entry) is ignored for even n.
    const nyq = irfft(
      tensor([1, 2, 3], { dtype: "float64" }),
      tensor([0, 0, 9], { dtype: "float64" }),
      4
    );
    expectClose(nyq.real.data as Float64Array, [2, -0.5, 0, -0.5]);
  });

  it("matches NumPy for odd and truncated output lengths", () => {
    const re = tensor([1, 2, 3, 4], { dtype: "float64" });
    const im = tensor([5, 1, 1, 7], { dtype: "float64" });
    expectClose(
      irfft(re, im, 7).real.data as Float64Array,
      [
        2.714285714285714, -2.090972639495403, 1.3650819828648544, -1.9423142022789108,
        1.7585701645338279, -1.4530758481133985, 0.6484248282033162,
      ]
    );
    expectClose(
      irfft(re, im, 6).real.data as Float64Array,
      [2.5, -1.2440169358562922, 0, -0.16666666666666663, 0, -0.08931639747704097]
    );
  });

  it("explains the unusable default length for a single-bin input", () => {
    const one = tensor([1], { dtype: "float64" });
    expect(() => irfft(one, one)).toThrow(/pass n explicitly/);
  });

  it("round-trips rfft along a non-last axis", () => {
    const m = tensor(
      [
        [1, 2],
        [3, 5],
        [8, 13],
        [21, 34],
      ],
      { dtype: "float64" }
    );
    const spec = rfft(m, undefined, 0);
    expect(spec.real.shape).toEqual([3, 2]);
    const back = irfft(spec.real, spec.imag, 4, 0);
    expectClose(back.real.data as Float64Array, vals(m));
  });
});

describe("fft2 / fftn / ifftn", () => {
  it("fft2 matches NumPy on a non-square matrix", () => {
    const { real, imag } = fft2(
      tensor(
        [
          [0, 1, 2],
          [3, 4, 5],
        ],
        { dtype: "float64" }
      )
    );
    expectClose(vals(real), [15, -3, -3, -9, 0, 0]);
    expectClose(vals(imag), [0, 1.7320508075688772, -1.7320508075688772, 0, 0, 0]);
  });

  it("fft2 accepts explicit axes and round-trips with ifft2", () => {
    const t = tensor(
      Array.from({ length: 24 }, (_, i) => (i * 5) % 7),
      { dtype: "float64" }
    ).reshape([2, 3, 4]);
    const spec = fft2(t, [0, 2]);
    const back = ifft2(spec.real, spec.imag, [0, 2]);
    expectClose(vals(back.real), vals(t));
    expectClose(vals(back.imag), new Array<number>(24).fill(0));
  });

  it("fftn with an empty axes list returns a copy, not the input buffer", () => {
    const t = tensor([1, 2, 3], { dtype: "float64" });
    const { real } = fftn(t, []);
    expect(real.data).not.toBe(t.data);
    expect(vals(real)).toEqual([1, 2, 3]);
  });

  it("ifftn inverts fftn for 3-D data of mixed lengths", () => {
    const t = tensor(
      Array.from({ length: 60 }, (_, i) => Math.sin(i) * 3),
      { dtype: "float64" }
    ).reshape([3, 4, 5]);
    const spec = fftn(t);
    const back = ifftn(spec.real, spec.imag);
    expectClose(vals(back.real), vals(t), 1e-12);
  });
});

describe("frequency helpers", () => {
  it("fftfreq matches NumPy", () => {
    expectClose(fftfreq(5, 0.1).data as Float64Array, [0, 2, 4, -4, -2]);
    expect(Array.from(fftfreq(4).data as Float64Array)).toEqual([0, 0.25, -0.5, -0.25]);
    expectClose(
      fftfreq(6).data as Float64Array,
      [0, 0.16666666666666666, 0.3333333333333333, -0.5, -0.3333333333333333, -0.16666666666666666]
    );
    expect(fftfreq(4).dtype).toBe("float64");
  });

  it("rfftfreq matches NumPy", () => {
    expectClose(rfftfreq(5).data as Float64Array, [0, 0.2, 0.4]);
    expect(Array.from(rfftfreq(8, 0.5).data as Float64Array)).toEqual([0, 0.25, 0.5, 0.75, 1]);
  });

  it("validates n and d", () => {
    expect(() => fftfreq(0)).toThrow(InvalidParameterError);
    expect(() => fftfreq(4, 0)).toThrow(InvalidParameterError);
    expect(() => rfftfreq(-1)).toThrow(InvalidParameterError);
  });

  it("fftshift / ifftshift match NumPy for odd and even lengths and 2-D input", () => {
    const m = tensor(
      Array.from({ length: 12 }, (_, i) => i),
      { dtype: "float64" }
    ).reshape([3, 4]);
    expect(vals(fftshift(m))).toEqual([10, 11, 8, 9, 2, 3, 0, 1, 6, 7, 4, 5]);
    expect(vals(ifftshift(m))).toEqual([6, 7, 4, 5, 10, 11, 8, 9, 2, 3, 0, 1]);
    expect(vals(fftshift(m, [1]))).toEqual([2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9]);
    const v = tensor([0, 1, 2, 3, 4], { dtype: "float64" });
    expect(vals(fftshift(v))).toEqual([3, 4, 0, 1, 2]);
    expect(vals(ifftshift(v))).toEqual([2, 3, 4, 0, 1]);
  });
});

// ---------------------------------------------------------------------------
// einsum
// ---------------------------------------------------------------------------

describe("einsum: dtype", () => {
  it("scalar results keep float64 precision", () => {
    const r = einsum(
      "i,i->",
      tensor([0.1, 0.2], { dtype: "float64" }),
      tensor([1, 1], { dtype: "float64" })
    );
    expect(r.dtype).toBe("float64");
    expect(r.shape).toEqual([]);
    expect((r.data as Float64Array)[0]).toBe(0.30000000000000004);
  });

  it("float32 operands give a float32 result on both the fast and generic paths", () => {
    const a = tensor(
      [
        [1, 2],
        [3, 4],
      ],
      { dtype: "float32" }
    );
    expect(einsum("ij,jk->ik", a, a).dtype).toBe("float32");
    expect(einsum("ij,jk,kl->il", a, a, a).dtype).toBe("float32");
    expect(einsum("ii->", a).dtype).toBe("float32");
  });

  it("int32 operands give an int32 result (NumPy keeps the integer dtype)", () => {
    const a = tensor(
      [
        [0, 1, 2],
        [3, 4, 5],
      ],
      { dtype: "int32" }
    );
    const b = tensor([0, 1, 2], { dtype: "int32" });
    const r = einsum("ij,j", a, b);
    expect(r.dtype).toBe("int32");
    expect(vals(r)).toEqual([5, 14]);
  });

  it("mixed dtypes give float64", () => {
    const r = einsum(
      "i,i->",
      tensor([1, 2], { dtype: "float32" }),
      tensor([3, 4], { dtype: "float64" })
    );
    expect(r.dtype).toBe("float64");
    expect(vals(r)).toEqual([11]);
  });

  it("rejects string operands with a DTypeError", () => {
    expect(() => einsum("i->", tensor(["a"]))).toThrow(DTypeError);
  });
});

describe("einsum: matmul fast path", () => {
  const a = tensor(
    [
      [0, 1, 2],
      [3, 4, 5],
    ],
    { dtype: "float64" }
  );

  it("returns densely packed data when the output labels are swapped", () => {
    const r = einsum("ij,kj->ki", a, a);
    expect(r.shape).toEqual([2, 2]);
    // np.einsum('ij,kj', a, a) == [[5, 14], [14, 50]] (symmetric), check a non-symmetric one.
    const b = tensor(
      [
        [1, 0, 0],
        [0, 1, 1],
      ],
      { dtype: "float64" }
    );
    const s = einsum("ij,kj->ki", a, b);
    // out[k][i] = sum_j a[i][j] * b[k][j] -> [[0, 3], [3, 9]]
    expect(Array.from(s.data as Float64Array)).toEqual([0, 3, 3, 9]);
    expect(r.data).toBeInstanceOf(Float64Array);
  });

  it("falls back instead of failing when a contracted size of 1 broadcasts", () => {
    const x = tensor([[1], [2]], { dtype: "float64" });
    const y = tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ],
      { dtype: "float64" }
    );
    // j has size 1 in x and 3 in y: size-1 dimensions broadcast, as in np.einsum.
    // np.einsum('ij,jk->ik', x, y) == [[12, 15, 18], [24, 30, 36]]
    const r = einsum("ij,jk->ik", x, y);
    expect(r.shape).toEqual([2, 3]);
    expect(vals(r)).toEqual([12, 15, 18, 24, 30, 36]);
  });

  it("still rejects genuinely conflicting sizes", () => {
    const x = tensor([[1, 2]], { dtype: "float64" });
    const y = tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ],
      { dtype: "float64" }
    );
    expect(() => einsum("ij,jk->ik", x, y)).toThrow(/conflicting/);
  });
});

describe("einsum: ellipsis and broadcasting", () => {
  it("supports '...' for batched matmul (NumPy reference)", () => {
    const a = tensor([0, 1, 2, 3, 4, 5, 6, 7], { dtype: "float64" }).reshape([2, 2, 2]);
    const r = einsum("...ij,...jk->...ik", a, a);
    expect(r.shape).toEqual([2, 2, 2]);
    expect(vals(r)).toEqual([2, 3, 6, 11, 46, 55, 66, 79]);
  });

  it("sums over ellipsis dimensions and broadcasts them across operands", () => {
    const a = tensor([0, 1, 2, 3, 4, 5], { dtype: "float64" }).reshape([2, 3]);
    expect(vals(einsum("...j->...", a))).toEqual([3, 12]);
    const w = tensor([1, 10, 100], { dtype: "float64" });
    // np.einsum('...j,j->...', a, w) == [210, 543]
    expect(vals(einsum("...j,j->...", a, w))).toEqual([210, 543]);
  });

  it("implicit mode orders labels alphabetically (np.einsum('ba') transposes)", () => {
    const a = tensor([0, 1, 2, 3, 4, 5], { dtype: "float64" }).reshape([2, 3]);
    const r = einsum("ba", a);
    expect(r.shape).toEqual([3, 2]);
    expect(vals(r)).toEqual([0, 3, 1, 4, 2, 5]);
  });

  it("requires '...' in the output when an operand has extra dimensions", () => {
    const a = tensor([0, 1, 2, 3, 4, 5], { dtype: "float64" }).reshape([2, 3]);
    expect(() => einsum("...i->i", a)).toThrow(InvalidParameterError);
  });

  it("broadcasts size-1 dimensions that share a label", () => {
    const a = tensor([[1], [2], [3]], { dtype: "float64" });
    const b = tensor(
      [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
      ],
      { dtype: "float64" }
    );
    // np.einsum('ij,ij->ij', a, b) == a * b
    expect(vals(einsum("ij,ij->ij", a, b))).toEqual([1, 2, 3, 4, 10, 12, 14, 16, 27, 30, 33, 36]);
  });
});

describe("einsum: validation", () => {
  const a = tensor(
    [
      [1, 2],
      [3, 4],
    ],
    { dtype: "float64" }
  );

  it("rejects labels that are not letters", () => {
    expect(() => einsum("i1,1j->ij", a, a)).toThrow(InvalidParameterError);
    expect(() => einsum("ij,jk->i-k", a, a)).toThrow(InvalidParameterError);
    expect(() => einsum("i.j->ij", a)).toThrow(InvalidParameterError);
  });

  it("rejects repeated or unknown output labels", () => {
    expect(() => einsum("ij->ii", a)).toThrow(/repeated/);
    expect(() => einsum("ij->ik", a)).toThrow(/does not appear/);
  });

  it("reports operand rank mismatches", () => {
    expect(() => einsum("i", a)).toThrow(ShapeError);
    expect(() => einsum("ijk...", a)).toThrow(ShapeError);
  });

  it("handles empty contractions", () => {
    const x = zeros([2, 0], { dtype: "float64" });
    const y = zeros([0, 3], { dtype: "float64" });
    const r = einsum("ij,jk->ik", x, y);
    expect(r.shape).toEqual([2, 3]);
    expect(vals(r)).toEqual([0, 0, 0, 0, 0, 0]);
    expect(vals(einsum("i->", tensor([], { dtype: "float64" })))).toEqual([0]);
  });

  it("honors strides and offsets of views on the generic path", () => {
    const m = tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      { dtype: "float64" }
    );
    const mt = transpose(m);
    const v = slice(tensor([9, 1, 1, 1], { dtype: "float64" }), { start: 1, end: 4 });
    // sum_j mt[i][j] * ... : mt is 3x2; contract with v (3) over i
    expect(vals(einsum("ij,i->j", mt, v))).toEqual([6, 15]);
  });
});

// ---------------------------------------------------------------------------
// insert / delete
// ---------------------------------------------------------------------------

describe("insert: NumPy semantics", () => {
  it("a single index inserts every value of a 1-D values tensor, in order", () => {
    const a = tensor([1, 2, 3], { dtype: "float64" });
    // np.insert(a, 1, [9, 8]) == [1, 9, 8, 2, 3]
    expect(vals(insert(a, 1, tensor([9, 8], { dtype: "float64" })))).toEqual([1, 9, 8, 2, 3]);
    expect(vals(insert(a, [1], tensor([9, 8], { dtype: "float64" })))).toEqual([1, 9, 8, 2, 3]);
  });

  it("several indices pair values with indices as given", () => {
    const a = tensor([1, 2, 3], { dtype: "float64" });
    const v = tensor([9, 8], { dtype: "float64" });
    expect(vals(insert(a, [1, 2], v))).toEqual([1, 9, 2, 8, 3]);
    expect(vals(insert(a, [2, 1], v))).toEqual([1, 8, 2, 9, 3]);
    expect(vals(insert(a, [1, 1], 7))).toEqual([1, 7, 7, 2, 3]);
  });

  it("1-D values broadcast along the trailing dimension of N-D tensors", () => {
    const m = tensor(
      [
        [1, 2],
        [3, 4],
      ],
      { dtype: "float64" }
    );
    // np.insert(m, [0, 1], [5, 6], axis=1) == [[5, 1, 6, 2], [5, 3, 6, 4]]
    const r = insert(m, [0, 1], tensor([5, 6], { dtype: "float64" }), 1);
    expect(r.shape).toEqual([2, 4]);
    expect(vals(r)).toEqual([5, 1, 6, 2, 5, 3, 6, 4]);
    // np.insert(m, 1, [5, 6], axis=1) == [[1, 5, 2], [3, 6, 4]]
    expect(vals(insert(m, 1, tensor([5, 6], { dtype: "float64" }), 1))).toEqual([1, 5, 2, 3, 6, 4]);

    const c = tensor([0, 1, 2, 3, 4, 5, 6, 7], { dtype: "float64" }).reshape([2, 2, 2]);
    // np.insert(c, [0, 1], [5, 6], axis=0): the values run along the LAST axis.
    const r3 = insert(c, [0, 1], tensor([5, 6], { dtype: "float64" }), 0);
    expect(r3.shape).toEqual([4, 2, 2]);
    expect(vals(r3)).toEqual([5, 6, 5, 6, 0, 1, 2, 3, 5, 6, 5, 6, 4, 5, 6, 7]);
  });

  it("2-D values: one slot per index along the axis", () => {
    const m = tensor(
      [
        [0, 1, 2],
        [3, 4, 5],
      ],
      { dtype: "float64" }
    );
    const v = tensor(
      [
        [1, 2],
        [3, 4],
      ],
      { dtype: "float64" }
    );
    // np.insert(m, [3, 0], v, axis=1) == [[2, 0, 1, 2, 1], [4, 3, 4, 5, 3]] with v columns
    // paired with the indices as given.
    const r = insert(m, [3, 0], v, 1);
    expect(r.shape).toEqual([2, 5]);
    expect(vals(r)).toEqual([2, 0, 1, 2, 1, 4, 3, 4, 5, 3]);
  });

  it("rejects values that cannot broadcast to the insertion slab", () => {
    const c = tensor(
      Array.from({ length: 24 }, (_, i) => i),
      { dtype: "float64" }
    ).reshape([2, 3, 4]);
    expect(() => insert(c, 1, tensor([1, 2, 3], { dtype: "float64" }), 0)).toThrow(ShapeError);
  });

  it("flattens strided input when no axis is given", () => {
    const m = transpose(
      tensor(
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        { dtype: "float64" }
      )
    );
    // transpose is [[1, 4], [2, 5], [3, 6]] -> flat [1, 4, 2, 5, 3, 6]
    expect(vals(insert(m, 2, 0))).toEqual([1, 4, 0, 2, 5, 3, 6]);
  });

  it("empty index list returns a copy", () => {
    const a = tensor([1, 2, 3], { dtype: "float64" });
    const r = insert(a, [], 5);
    expect(vals(r)).toEqual([1, 2, 3]);
    expect(r.data).not.toBe(a.data);
  });
});

describe("insert: dtype handling", () => {
  it("converts int64 values into a float tensor instead of leaving zeros", () => {
    const a = tensor([1, 2, 3], { dtype: "float64" });
    const r = insert(a, 1, tensor([7], { dtype: "int64" }));
    expect(vals(r)).toEqual([1, 7, 2, 3]);
  });

  it("converts float values into an int64 tensor (truncating) and rejects non-finite ones", () => {
    const a = tensor([1, 2, 3], { dtype: "int64" });
    const r = insert(a, [0, 3], tensor([5.9, -6.9], { dtype: "float64" }));
    expect(r.dtype).toBe("int64");
    expect(r.toArray()).toEqual([5n, 1n, 2n, 3n, -6n]);
    expect(() => insert(a, 0, Number.NaN)).toThrow(InvalidParameterError);
    expect(() => insert(a, 0, Number.POSITIVE_INFINITY)).toThrow(InvalidParameterError);
  });

  it("stores bool tensors as 0/1 whatever number is inserted", () => {
    const a = tensor([1, 0], { dtype: "bool" });
    expect(vals(insert(a, 1, 5))).toEqual([1, 1, 0]);
    expect(vals(insert(a, 1, 0))).toEqual([1, 0, 0]);
  });

  it("string tensors accept strings only, and numeric tensors reject strings", () => {
    const s = tensor(["a", "b"]);
    expect(insert(s, 1, "x").toArray()).toEqual(["a", "x", "b"]);
    expect(insert(s, 0, tensor(["p", "q"])).toArray()).toEqual(["p", "q", "a", "b"]);
    expect(() => insert(s, 0, 1)).toThrow(DTypeError);
    expect(() => insert(s, 0, tensor([1, 2]))).toThrow(DTypeError);
    expect(() => insert(tensor([1, 2]), 0, "x")).toThrow(DTypeError);
    expect(() => insert(tensor([1, 2]), 0, tensor(["x"]))).toThrow(DTypeError);
  });
});

describe("delete_", () => {
  it("handles int64, string and strided tensors", () => {
    expect(delete_(tensor([1, 2, 3, 4], { dtype: "int64" }), [0, 2]).toArray()).toEqual([2n, 4n]);
    expect(delete_(tensor(["a", "b", "c"]), [0, 2]).toArray()).toEqual(["b"]);
    const m = transpose(
      tensor(
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        { dtype: "float64" }
      )
    );
    // m = [[1, 4], [2, 5], [3, 6]]
    expect(vals(delete_(m, 0, 1))).toEqual([4, 5, 6]);
    expect(vals(delete_(m, 1, 0))).toEqual([1, 4, 3, 6]);
  });

  it("deletes repeated indices once and matches NumPy on 3-D input", () => {
    const c = tensor(
      Array.from({ length: 24 }, (_, i) => i),
      { dtype: "float64" }
    ).reshape([2, 3, 4]);
    const r = delete_(c, [0, 3, 3], 2);
    expect(r.shape).toEqual([2, 3, 2]);
    expect(vals(r)).toEqual([1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22]);
    expect(vals(delete_(tensor([1, 2, 3, 4, 5], { dtype: "float64" }), [0, 3, 3]))).toEqual([
      2, 3, 5,
    ]);
  });

  it("deleting everything yields an empty tensor; no indices returns a copy", () => {
    const a = tensor([1, 2, 3], { dtype: "float64" });
    const none = delete_(a, [0, 1, 2], 0);
    expect(none.shape).toEqual([0]);
    const same = delete_(a, [], 0);
    expect(vals(same)).toEqual([1, 2, 3]);
    expect(same.data).not.toBe(a.data);
  });

  it("rejects out-of-range and non-integer indices", () => {
    const a = tensor([1, 2, 3], { dtype: "float64" });
    expect(() => delete_(a, 3, 0)).toThrow(InvalidParameterError);
    expect(() => delete_(a, -4, 0)).toThrow(InvalidParameterError);
    expect(() => delete_(a, 1.5, 0)).toThrow(/integer/);
  });
});

// ---------------------------------------------------------------------------
// im2col / col2im
// ---------------------------------------------------------------------------

describe("im2col / col2im", () => {
  const x = tensor(
    [
      [
        [
          [1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
        ],
      ],
    ],
    { dtype: "float64" }
  );

  it("unfolds with stride and padding (torch.nn.functional.unfold reference)", () => {
    // torch unfold(x, 2, padding=1, stride=2) -> columns for windows at (0,0), (0,2), (2,0), (2,2)
    const cols = im2col(x, [2, 2], [2, 2], [1, 1]);
    expect(cols.shape).toEqual([1, 4, 4]);
    expect(vals(cols)).toEqual([0, 0, 0, 1, 0, 0, 2, 3, 0, 4, 0, 7, 5, 6, 8, 9]);
  });

  it("is the adjoint of col2im: overlapping windows are summed", () => {
    const cols = im2col(x, [2, 2], [1, 1], [0, 0]);
    expect(cols.shape).toEqual([1, 4, 4]);
    const back = col2im(cols, [1, 1, 3, 3], [2, 2], [1, 1], [0, 0]);
    // Each pixel is counted once per window covering it.
    expect(vals(back)).toEqual([1, 4, 3, 8, 20, 12, 7, 16, 9]);
  });

  it("works on strided int64 input", () => {
    const base = tensor(
      [
        [
          [
            [1, 4, 7],
            [2, 5, 8],
            [3, 6, 9],
          ],
        ],
      ],
      { dtype: "int64" }
    );
    const view = transpose(base, [0, 1, 3, 2]); // equals x
    const cols = im2col(view, [2, 2], [1, 1], [0, 0]);
    expect(cols.dtype).toBe("int64");
    expect(vals(cols)).toEqual(vals(im2col(x, [2, 2], [1, 1], [0, 0])));
  });

  it("rejects malformed geometry with typed errors", () => {
    expect(() => im2col(x, [2] as unknown as [number, number], [1, 1], [0, 0])).toThrow(
      InvalidParameterError
    );
    expect(() => im2col(x, [2, 2], [1, 1, 1] as unknown as [number, number], [0, 0])).toThrow(
      InvalidParameterError
    );
    expect(() => im2col(x, [4, 4], [1, 1], [0, 0])).toThrow(/does not fit/);
    const cols = im2col(x, [2, 2], [1, 1], [0, 0]);
    expect(() => col2im(cols, [1, 1, -3, 3], [2, 2], [1, 1], [0, 0])).toThrow(
      InvalidParameterError
    );
    expect(() => col2im(cols, [1, 1, 4, 4], [2, 2], [1, 1], [0, 0])).toThrow(ShapeError);
  });
});

// ---------------------------------------------------------------------------
// logical ops
// ---------------------------------------------------------------------------

describe("logical ops", () => {
  it("treats NaN as true and negative zero as false", () => {
    const a = tensor([Number.NaN, -0, 0, 2], { dtype: "float64" });
    expect(vals(logicalNot(a))).toEqual([0, 1, 1, 0]);
    expect(vals(logicalAnd(a, tensor(1)))).toEqual([1, 0, 0, 1]);
    expect(vals(logicalOr(a, tensor(0)))).toEqual([1, 0, 0, 1]);
    expect(vals(logicalXor(a, tensor(1)))).toEqual([0, 1, 1, 0]);
  });

  it("scalar operand on either side, including sliced (offset) views", () => {
    const base = tensor([9, 1, 0, 1, 0], { dtype: "float64" });
    const s = slice(base, { start: 1, end: 5 }); // [1, 0, 1, 0], offset 1
    expect(vals(logicalAnd(s, tensor(1)))).toEqual([1, 0, 1, 0]);
    expect(vals(logicalAnd(tensor(1), s))).toEqual([1, 0, 1, 0]);
    expect(vals(logicalOr(tensor(0), s))).toEqual([1, 0, 1, 0]);
    expect(vals(logicalXor(s, tensor(1)))).toEqual([0, 1, 0, 1]);
    expect(vals(logicalAnd(tensor(1), tensor(0)))).toEqual([0]);
  });

  it("matches NumPy on same-shape strided and broadcast operands", () => {
    const m = tensor(
      [
        [1, 0],
        [0, 1],
        [1, 1],
      ],
      { dtype: "float64" }
    );
    const mt = transpose(m); // [[1,0,1],[0,1,1]]
    const other = tensor(
      [
        [1, 1, 0],
        [0, 1, 0],
      ],
      { dtype: "float64" }
    );
    expect(vals(logicalAnd(mt, other))).toEqual([1, 0, 0, 0, 1, 0]);
    expect(vals(logicalXor(mt, other))).toEqual([0, 1, 1, 0, 0, 1]);
    expect(vals(logicalOr(mt, tensor([0, 1, 0], { dtype: "float64" })))).toEqual([
      1, 1, 1, 0, 1, 1,
    ]);
  });

  it("works for int64 operands", () => {
    const a = tensor([0, 5], { dtype: "int64" });
    const b = tensor([1, 1], { dtype: "int64" });
    expect(vals(logicalAnd(a, b))).toEqual([0, 1]);
    expect(vals(logicalNot(a))).toEqual([1, 0]);
  });
});

// ---------------------------------------------------------------------------
// device dispatch (minimal stub kernel backends)
// ---------------------------------------------------------------------------

type StubBuffer = DeviceBuffer & { data: Float32Array };

function makeStubBackend(
  device: "webgpu" | "wasm",
  overrides: Record<string, unknown> = {}
): KernelBackend {
  const wrap = (data: Float32Array): StubBuffer => ({
    device,
    byteLength: data.byteLength,
    size: data.length,
    data,
  });
  const unsupported = (name: string) => (): never => {
    throw new Error(`stub backend: ${name} is not implemented`);
  };
  const stub = {
    info: () => ({
      device,
      name: `stub-${device}`,
      available: true,
      capabilities: ["matmul"],
    }),
    supports: () => true,
    init: async () => {},
    dispose: () => {},
    upload: (data: Float32Array) => wrap(new Float32Array(data)),
    download: async (buffer: DeviceBuffer) => new Float32Array((buffer as StubBuffer).data),
    free: () => {},
    fill: (value: number, size: number) => wrap(new Float32Array(size).fill(value)),
    binary: unsupported("binary"),
    unary: unsupported("unary"),
    matmul: unsupported("matmul"),
    reduce: unsupported("reduce"),
    reduceAxis: unsupported("reduceAxis"),
    matmulBatched: unsupported("matmulBatched"),
    ternary: unsupported("ternary"),
    im2col: unsupported("im2col"),
    col2im: unsupported("col2im"),
    pool2d: unsupported("pool2d"),
    pool2dBackward: unsupported("pool2dBackward"),
    ...overrides,
  };
  return stub as unknown as KernelBackend;
}

describe("device dispatch", () => {
  beforeAll(() => {
    registerBackend("webgpu", makeStubBackend("webgpu"));
    registerBackend("wasm", makeStubBackend("wasm"));
  });
  afterAll(() => {
    unregisterBackend("webgpu");
    unregisterBackend("wasm");
  });

  it("where() names both offending devices instead of the host condition", () => {
    const cond = tensor(1);
    const a = tensor([1, 2], { dtype: "float32", device: "webgpu" });
    const b = tensor([3, 4], { dtype: "float32", device: "wasm" });
    let message = "";
    try {
      where(cond, a, b);
    } catch (e) {
      expect(e).toBeInstanceOf(DeviceError);
      message = (e as Error).message;
    }
    expect(message).toContain("webgpu");
    expect(message).toContain("wasm");
    expect(message).not.toContain("cpu");
  });

  it("pooling parameters are validated before reaching the kernel", () => {
    const img = tensor(new Array<number>(16).fill(1), {
      dtype: "float32",
      device: "webgpu",
    }).reshape([1, 1, 4, 4]);
    expect(() => dispatchPool2d(img, "max", [5, 5], [1, 1], [0, 0])).toThrow(InvalidParameterError);
    expect(() => dispatchPool2d(img, "max", [2, 2], [0, 1], [0, 0])).toThrow(InvalidParameterError);
    const flat = tensor([1, 2, 3, 4], { dtype: "float32", device: "webgpu" });
    expect(() => dispatchPool2d(flat, "max", [2, 2], [1, 1], [0, 0])).toThrow(ShapeError);
  });

  it("frees the intermediate buffer when a multi-axis reduction fails part-way", () => {
    const freed: DeviceBuffer[] = [];
    let calls = 0;
    registerBackend(
      "webgpu",
      makeStubBackend("webgpu", {
        free: (buffer: DeviceBuffer) => {
          freed.push(buffer);
        },
        reduceAxis: (_op: string, _x: DeviceBuffer, layout: { shape: readonly number[] }) => {
          calls++;
          if (calls === 2) throw new Error("kernel failure");
          const size = layout.shape.reduce((a, b) => a * b, 1) / (layout.shape[2] ?? 1);
          return { device: "webgpu", byteLength: size * 4, size, data: new Float32Array(size) };
        },
      })
    );
    try {
      const t = tensor(new Array<number>(24).fill(1), {
        dtype: "float32",
        device: "webgpu",
      }).reshape([2, 3, 4]);
      expect(() => sum(t, [2, 1] as unknown as number)).toThrow(/kernel failure/);
      expect(calls).toBe(2);
      // The buffer produced by the first pass (6 elements) was released.
      expect(freed.some((b) => b.size === 6)).toBe(true);
    } finally {
      registerBackend("webgpu", makeStubBackend("webgpu"));
    }
  });

  it("refuses to combine a string scalar with a device tensor", () => {
    const a = tensor([1, 2], { dtype: "float32", device: "webgpu" });
    const s = tensor("x");
    expect(() => where(tensor(1), a, s)).toThrow(DTypeError);
  });
});
