import { describe, expect, it } from "vitest";
import { eig, eigvals } from "../../src/linalg";
import { tensor } from "../../src/ndarray";

// Reference values: numpy.linalg.eig / numpy.linalg.eigvals (numpy 2.x).

function sortedPairs(re: ArrayLike<number>, im: ArrayLike<number>): [number, number][] {
  const out: [number, number][] = [];
  for (let i = 0; i < re.length; i++) out.push([re[i] as number, im[i] as number]);
  return out.sort((p, q) => p[0] - q[0] || p[1] - q[1]);
}

function rows(t: { toArray(): unknown }): number[][] {
  return t.toArray() as number[][];
}

describe("eig and eigvals with complex: true", () => {
  it("returns the conjugate pair of a rotation matrix", () => {
    const A = tensor([
      [0, -1],
      [1, 0],
    ]);
    const { real, imag } = eigvals(A, { complex: true });
    // numpy.linalg.eigvals([[0, -1], [1, 0]]) -> [0+1j, 0-1j]
    expect(Array.from(real.toArray() as number[])).toEqual([0, 0]);
    const im = Array.from(imag.toArray() as number[]);
    expect(Math.abs(im[0] as number)).toBeCloseTo(1, 12);
    expect((im[0] as number) + (im[1] as number)).toBeCloseTo(0, 12);
  });

  it("matches numpy for a 3x3 matrix with one real and one complex pair", () => {
    const A = tensor([
      [1, -2, 3],
      [4, 5, -6],
      [7, 8, 9],
    ]);
    // numpy.linalg.eigvals -> [6, 4.5+4.44409721j, 4.5-4.44409721j]
    const { real, imag } = eigvals(A, { complex: true });
    const got = sortedPairs(real.toArray() as number[], imag.toArray() as number[]);
    const want: [number, number][] = [
      [4.5, -4.44409721],
      [4.5, 4.44409721],
      [6, 0],
    ];
    got.forEach((p, i) => {
      expect(p[0]).toBeCloseTo((want[i] as [number, number])[0], 6);
      expect(p[1]).toBeCloseTo((want[i] as [number, number])[1], 6);
    });
  });

  it("returns eigenvectors satisfying A v = lambda v", () => {
    const a = [
      [1, -2, 3],
      [4, 5, -6],
      [7, 8, 9],
    ];
    const { real, imag, vectorsReal, vectorsImag } = eig(tensor(a, { dtype: "float64" }), {
      complex: true,
    });
    const lr = real.toArray() as number[];
    const li = imag.toArray() as number[];
    const vr = rows(vectorsReal);
    const vi = rows(vectorsImag);
    for (let k = 0; k < 3; k++) {
      let norm = 0;
      for (let i = 0; i < 3; i++) {
        const xr = (vr[i] as number[])[k] as number;
        const xi = (vi[i] as number[])[k] as number;
        norm += xr * xr + xi * xi;
      }
      expect(norm).toBeCloseTo(1, 10);
      for (let i = 0; i < 3; i++) {
        let ar = 0;
        let ai = 0;
        for (let j = 0; j < 3; j++) {
          ar += ((a[i] as number[])[j] as number) * ((vr[j] as number[])[k] as number);
          ai += ((a[i] as number[])[j] as number) * ((vi[j] as number[])[k] as number);
        }
        const xr = (vr[i] as number[])[k] as number;
        const xi = (vi[i] as number[])[k] as number;
        const lvr = (lr[k] as number) * xr - (li[k] as number) * xi;
        const lvi = (lr[k] as number) * xi + (li[k] as number) * xr;
        expect(ar).toBeCloseTo(lvr, 8);
        expect(ai).toBeCloseTo(lvi, 8);
      }
    }
  });

  it("returns zero imaginary parts for a symmetric matrix", () => {
    const { real, imag, vectorsImag } = eig(
      tensor([
        [2, 1],
        [1, 2],
      ]),
      { complex: true }
    );
    expect(Array.from(real.toArray() as number[])).toEqual([1, 3]);
    expect(Array.from(imag.toArray() as number[])).toEqual([0, 0]);
    expect(rows(vectorsImag)).toEqual([
      [0, 0],
      [0, 0],
    ]);
  });

  it("keeps the default return type and points to the option in the error", () => {
    const A = tensor([
      [0, -1],
      [1, 0],
    ]);
    expect(() => eigvals(A)).toThrow(/complex: true/);
    expect(() => eig(A)).toThrow(/complex: true/);
    const [vals, vecs] = eig(
      tensor([
        [2, 1],
        [1, 2],
      ])
    );
    expect(vals.shape).toEqual([2]);
    expect(vecs.shape).toEqual([2, 2]);
  });

  it("handles random matrices (residual check against the definition)", () => {
    let s = 12345;
    const rnd = (): number => {
      s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
      return s / 4294967296 - 0.5;
    };
    for (let trial = 0; trial < 60; trial++) {
      const n = 2 + (trial % 6);
      const a: number[][] = [];
      for (let i = 0; i < n; i++) {
        const row: number[] = [];
        for (let j = 0; j < n; j++) row.push(rnd() * 4);
        a.push(row);
      }
      const { real, imag, vectorsReal, vectorsImag } = eig(tensor(a, { dtype: "float64" }), {
        complex: true,
      });
      const lr = real.toArray() as number[];
      const li = imag.toArray() as number[];
      const vr = rows(vectorsReal);
      const vi = rows(vectorsImag);
      let scale = 0;
      for (const r of a) for (const v of r) scale = Math.max(scale, Math.abs(v));
      for (let k = 0; k < n; k++) {
        for (let i = 0; i < n; i++) {
          let ar = 0;
          let ai = 0;
          for (let j = 0; j < n; j++) {
            ar += ((a[i] as number[])[j] as number) * ((vr[j] as number[])[k] as number);
            ai += ((a[i] as number[])[j] as number) * ((vi[j] as number[])[k] as number);
          }
          const xr = (vr[i] as number[])[k] as number;
          const xi = (vi[i] as number[])[k] as number;
          expect(Math.abs(ar - ((lr[k] as number) * xr - (li[k] as number) * xi))).toBeLessThan(
            1e-8 * scale * n
          );
          expect(Math.abs(ai - ((lr[k] as number) * xi + (li[k] as number) * xr))).toBeLessThan(
            1e-8 * scale * n
          );
        }
      }
    }
  });
});
