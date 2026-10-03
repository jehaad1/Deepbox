import { afterEach, describe, expect, it, vi } from "vitest";
import { DTypeError, InvalidParameterError } from "../../src/core";
import { Tensor, tensor, transpose } from "../../src/ndarray";
import {
  beta,
  binomial,
  categorical,
  choice,
  clearSeed,
  dirichlet,
  exponential,
  Generator,
  geometric,
  gumbel_softmax,
  laplace,
  multinomial,
  multivariate_normal,
  negative_binomial,
  permutation,
  rand,
  setSeed,
  shuffle,
  uniform,
  zipf,
} from "../../src/random";

const nums = (t: Tensor): number[] => Array.from(t.data as ArrayLike<number | bigint>, Number);

function mean(xs: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += xs[i] as number;
  return s / xs.length;
}

function variance(xs: ArrayLike<number>): number {
  const m = mean(xs);
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += ((xs[i] as number) - m) ** 2;
  return s / xs.length;
}

/** A float64 tensor viewing `data[offset .. offset + shape)` of a larger buffer. */
function view(data: Float64Array, shape: number[], offset: number, strides?: number[]): Tensor {
  return Tensor.fromTypedArray({
    data,
    shape,
    dtype: "float64",
    device: "cpu",
    offset,
    ...(strides ? { strides } : {}),
  });
}

afterEach(() => {
  clearSeed();
  vi.restoreAllMocks();
  vi.resetModules();
  vi.doUnmock("../../src/random/random");
});

describe("c56 float32 uniform never reaches 1", () => {
  it("rand float32 stays below 1 for the largest uint32 draw", () => {
    clearSeed();
    // Force the unseeded crypto path to return the largest possible word.
    vi.spyOn(globalThis.crypto, "getRandomValues").mockImplementation(((arr: Uint32Array) => {
      arr.fill(0xffffffff);
      return arr;
    }) as typeof globalThis.crypto.getRandomValues);
    // 3 * 4096 words guarantee the batched buffer is refilled by the stub.
    const x = rand([3 * 4096], { dtype: "float32" });
    const values = nums(x);
    expect(Math.max(...values)).toBeLessThan(1);
    expect(Math.max(...values)).toBe(0.9999999403953552);
  });

  it("exponential float32 is finite for the largest uint32 draw", () => {
    clearSeed();
    vi.spyOn(globalThis.crypto, "getRandomValues").mockImplementation(((arr: Uint32Array) => {
      arr.fill(0xffffffff);
      return arr;
    }) as typeof globalThis.crypto.getRandomValues);
    const x = exponential(1, [3 * 4096], { dtype: "float32" });
    expect(nums(x).every((v) => Number.isFinite(v))).toBe(true);
  });

  it("seeded float32 output equals the rounded float64 stream", () => {
    setSeed(3);
    const f32 = nums(rand([64], { dtype: "float32" }));
    setSeed(3);
    const f64 = nums(rand([64], { dtype: "float64" }));
    expect(f32).toEqual(f64.map((v) => Math.fround(v)));
  });
});

describe("c56 laplace", () => {
  it("is finite when the underlying uniform draw is exactly 0", async () => {
    vi.resetModules();
    const draws = [0, 0.25, 0.75, 0.5];
    vi.doMock("../../src/random/random", async () => {
      const actual =
        await vi.importActual<typeof import("../../src/random/random")>("../../src/random/random");
      return { ...actual, __random: () => draws.shift() ?? 0.5 };
    });
    const mod = await import("../../src/random/index");
    const out = nums(mod.laplace(1, 2, [3], { dtype: "float64" }));
    // loc + scale * ln(2u) for u < 1/2 and loc - scale * ln(2(1 - u)) otherwise;
    // the 0 draw is skipped.
    expect(out[0]).toBeCloseTo(1 + 2 * Math.log(0.5), 12);
    expect(out[1]).toBeCloseTo(1 - 2 * Math.log(0.5), 12);
    expect(out[2]).toBeCloseTo(1, 12);
  });

  it("matches scipy.stats.laplace(1, 2) moments", () => {
    setSeed(11);
    const x = nums(laplace(1, 2, [40000], { dtype: "float64" }));
    // scipy: mean 1, variance 2 * scale^2 = 8
    expect(mean(x)).toBeCloseTo(1, 1);
    expect(variance(x)).toBeGreaterThan(7.2);
    expect(variance(x)).toBeLessThan(8.8);
  });
});

describe("c56 small-shape gamma ratios", () => {
  it("dirichlet with tiny concentrations gives near one-hot rows, not uniform fallbacks", () => {
    setSeed(5);
    const out = dirichlet(tensor([0.001, 0.001, 0.001]), 300, { dtype: "float64" });
    const v = nums(out);
    let oneHot = 0;
    for (let r = 0; r < 300; r++) {
      const row = [v[3 * r] as number, v[3 * r + 1] as number, v[3 * r + 2] as number];
      expect(Math.abs(row[0] + row[1] + row[2] - 1)).toBeLessThan(1e-12);
      expect(row.every((x) => Number.isFinite(x) && x >= 0)).toBe(true);
      if (Math.max(...row) > 0.999) oneHot++;
    }
    expect(oneHot).toBeGreaterThan(285);
  });

  it("dirichlet matches the Beta marginal of scipy.stats.dirichlet([0.2, 0.5, 1.0])", () => {
    setSeed(6);
    const out = nums(dirichlet(tensor([0.2, 0.5, 1.0]), 30000, { dtype: "float64" }));
    const first: number[] = [];
    for (let i = 0; i < out.length; i += 3) first.push(out[i] as number);
    // marginal Beta(0.2, 1.5): mean 0.2 / 1.7, variance ab / ((a+b)^2 (a+b+1))
    expect(mean(first)).toBeCloseTo(0.2 / 1.7, 2);
    expect(variance(first)).toBeCloseTo((0.2 * 1.5) / (1.7 ** 2 * 2.7), 2);
  });

  it("beta with tiny parameters is near Bernoulli(0.5) with all values in [0, 1]", () => {
    setSeed(8);
    const x = nums(beta(0.001, 0.001, [4000], { dtype: "float64" }));
    expect(x.every((v) => Number.isFinite(v) && v >= 0 && v <= 1)).toBe(true);
    const extreme = x.filter((v) => v < 1e-6 || v > 1 - 1e-6).length;
    expect(extreme / x.length).toBeGreaterThan(0.97);
    expect(Math.abs(mean(x) - 0.5)).toBeLessThan(0.05);
  });

  it("beta and dirichlet stay finite when every gamma draw underflows", () => {
    setSeed(10);
    const b = nums(beta(5e-324, 5e-324, [200], { dtype: "float64" }));
    expect(b.every((v) => v === 0 || v === 1)).toBe(true);
    expect(b.some((v) => v === 0) && b.some((v) => v === 1)).toBe(true);
    const d = nums(
      dirichlet(tensor([5e-324, 5e-324, 5e-324], { dtype: "float64" }), 100, { dtype: "float64" })
    );
    for (let r = 0; r < 100; r++) {
      const row = [d[3 * r] as number, d[3 * r + 1] as number, d[3 * r + 2] as number];
      expect(row.slice().sort()).toEqual([0, 0, 1]);
    }
  });

  it("beta(0.3, 4) keeps the scipy mean and variance", () => {
    setSeed(9);
    const x = nums(beta(0.3, 4, [40000], { dtype: "float64" }));
    // scipy: mean a/(a+b) = 0.0697674, var ab/((a+b)^2 (a+b+1)) = 0.0095
    expect(mean(x)).toBeCloseTo(0.3 / 4.3, 2);
    expect(variance(x)).toBeCloseTo((0.3 * 4) / (4.3 ** 2 * 5.3), 2);
  });
});

describe("c56 shuffle and permutation on views", () => {
  it("shuffle of a view leaves the rest of the buffer untouched", () => {
    const buffer = Float64Array.from([1, 2, 3, 4, 5, 6]);
    const v = view(buffer, [3], 0);
    setSeed(1);
    shuffle(v);
    expect(Array.from(buffer.slice(3))).toEqual([4, 5, 6]);
    expect(Array.from(buffer.slice(0, 3)).sort()).toEqual([1, 2, 3]);
  });

  it("shuffle supports a view with a non-zero offset", () => {
    const buffer = Float64Array.from([10, 1, 2, 3, 4, 99]);
    const v = view(buffer, [4], 1);
    setSeed(2);
    shuffle(v);
    expect(buffer[0]).toBe(10);
    expect(buffer[5]).toBe(99);
    expect(Array.from(buffer.slice(1, 5)).sort()).toEqual([1, 2, 3, 4]);
  });

  it("permutation of a view copies only the view elements", () => {
    const buffer = Float64Array.from([1, 2, 3, 4, 5, 6]);
    const v = view(buffer, [3], 2);
    setSeed(3);
    const p = permutation(v);
    expect(p.shape).toEqual([3]);
    expect(p.data.length).toBe(3);
    expect(nums(p).sort()).toEqual([3, 4, 5]);
    expect(Array.from(buffer)).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it("still rejects genuinely strided tensors", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(() => shuffle(transpose(t))).toThrow(InvalidParameterError);
    expect(() => permutation(transpose(t))).toThrow(InvalidParameterError);
  });

  it("shuffle of an empty tensor is a no-op", () => {
    expect(() => shuffle(tensor([] as number[]))).not.toThrow();
  });
});

describe("c56 categorical", () => {
  it("throws instead of looping forever when too few categories have mass", () => {
    expect(() => categorical(tensor([1, 0, 0]), 2, false)).toThrow(InvalidParameterError);
  });

  it("draws only positive-probability categories without replacement", () => {
    setSeed(4);
    const s = nums(categorical(tensor([0.5, 0.5, 0, 0]), 2, false));
    expect(s.sort()).toEqual([0, 1]);
  });

  it("never selects zero-probability categories with replacement", () => {
    setSeed(4);
    const s = nums(categorical(tensor([0, 0.5, 0.5, 0]), 2000));
    expect(s.every((v) => v === 1 || v === 2)).toBe(true);
  });

  it("without replacement matches sequential proportional sampling", () => {
    // P(first = 0) = 0.5; scipy/numpy reference: numpy choice(p, replace=False) shares this law.
    setSeed(12);
    let first0 = 0;
    const trials = 6000;
    for (let i = 0; i < trials; i++) {
      const s = nums(categorical(tensor([0.5, 0.3, 0.2]), 3, false));
      expect(s.slice().sort()).toEqual([0, 1, 2]);
      if (s[0] === 0) first0++;
    }
    expect(Math.abs(first0 / trials - 0.5)).toBeLessThan(0.03);
  });

  it("honors strides of the probability tensor", () => {
    // Logical probabilities are [1, 0, 0] (every other element of the buffer).
    const buffer = Float64Array.from([1, 9, 0, 9, 0, 9]);
    const probs = view(buffer, [3], 0, [2]);
    setSeed(5);
    expect(nums(categorical(probs, 50)).every((v) => v === 0)).toBe(true);
  });
});

describe("c56 choice", () => {
  it("samples without replacement from a huge integer population in O(size) memory", () => {
    setSeed(6);
    const out = choice(2 ** 31, 5, false);
    const v = nums(out);
    expect(out.dtype).toBe("int32");
    expect(out.shape).toEqual([5]);
    expect(new Set(v).size).toBe(5);
    expect(v.every((x) => x >= 0 && x < 2 ** 31)).toBe(true);
  });

  it("the sparse (large population) path draws distinct in-range values", () => {
    // 100000 > 65536 and > 4 * 8, so this call takes the sparse swap-map path.
    setSeed(7);
    const v = nums(choice(100000, 8, false));
    expect(new Set(v).size).toBe(8);
    expect(v.every((x) => x >= 0 && x < 100000)).toBe(true);
  });

  it("reads tensors with a non-zero offset correctly", () => {
    const buffer = Float64Array.from([100, 200, 1, 2, 3, 300]);
    const t = view(buffer, [3], 2);
    setSeed(8);
    const s = nums(choice(t, 40));
    expect(s.every((x) => x === 1 || x === 2 || x === 3)).toBe(true);
    expect(new Set(s).size).toBe(3);
  });

  it("rejects a non-contiguous tensor without consuming the seeded stream", () => {
    const t = transpose(
      tensor([
        [1, 2],
        [3, 4],
      ])
    );
    setSeed(9);
    expect(() => choice(t, 2)).toThrow(InvalidParameterError);
    const after = nums(rand([4], { dtype: "float64" }));
    setSeed(9);
    expect(nums(rand([4], { dtype: "float64" }))).toEqual(after);
  });

  it("never picks trailing zero-probability entries", () => {
    setSeed(10);
    const s = nums(choice(4, 3000, true, tensor([0.5, 0.5, 0, 0])));
    expect(s.every((v) => v === 0 || v === 1)).toBe(true);
  });

  it("weighted draws without replacement return distinct positive-weight items", () => {
    setSeed(11);
    const s = nums(choice(5, 3, false, tensor([0.2, 0, 0.3, 0, 0.5])));
    expect(s.slice().sort()).toEqual([0, 2, 4]);
  });
});

describe("c56 multinomial", () => {
  it("counts sum to n with scipy.stats.multinomial moments for huge n", () => {
    setSeed(13);
    const draws = 4000;
    const out = multinomial(1_000_000_000, tensor([0.1, 0.2, 0.7]), draws, { dtype: "int64" });
    const v = nums(out);
    const col = (c: number): number[] => v.filter((_, i) => i % 3 === c);
    for (let r = 0; r < 20; r++) {
      expect((v[3 * r] as number) + (v[3 * r + 1] as number) + (v[3 * r + 2] as number)).toBe(1e9);
    }
    // scipy: mean n p = 1e8, var n p (1 - p) = 9e7 (sd ~ 9487)
    expect(Math.abs(mean(col(0)) - 1e8)).toBeLessThan(5 * Math.sqrt(9e7 / draws));
    expect(Math.sqrt(variance(col(0)))).toBeGreaterThan(9487 * 0.93);
    expect(Math.sqrt(variance(col(0)))).toBeLessThan(9487 * 1.07);
    // scipy: Cov(X0, X1) = -n p0 p1 = -2e7 => correlation of -sqrt(p0 p1 / ((1-p0)(1-p1)))
    const c0 = col(0);
    const c1 = col(1);
    const m0 = mean(c0);
    const m1 = mean(c1);
    let cov = 0;
    for (let i = 0; i < draws; i++) cov += ((c0[i] as number) - m0) * ((c1[i] as number) - m1);
    cov /= draws;
    const corr = cov / Math.sqrt(variance(c0) * variance(c1));
    expect(corr).toBeCloseTo(-Math.sqrt((0.1 * 0.2) / (0.9 * 0.8)), 1);
  });

  it("keeps the float default dtype and accepts unnormalized pvals", () => {
    setSeed(14);
    const out = multinomial(10, tensor([1, 2, 7]), 6);
    expect(out.dtype).toBe("float32");
    expect(out.shape).toEqual([6, 3]);
    const v = nums(out);
    for (let r = 0; r < 6; r++) {
      expect((v[3 * r] as number) + (v[3 * r + 1] as number) + (v[3 * r + 2] as number)).toBe(10);
    }
    expect(multinomial(10, tensor([1, 2, 7]), 2, { dtype: "int32" }).dtype).toBe("int32");
  });

  it("widens the default dtype to float64 so counts above 2^24 stay exact", () => {
    setSeed(14);
    const n = 2 ** 24 + 1;
    const out = multinomial(n, tensor([0.5, 0.5]), 20);
    expect(out.dtype).toBe("float64");
    const v = nums(out);
    for (let r = 0; r < 20; r++) expect((v[2 * r] as number) + (v[2 * r + 1] as number)).toBe(n);
  });

  it("never allocates to zero-probability categories, including a zero last entry", () => {
    setSeed(15);
    const v = nums(multinomial(500, tensor([0.3, 0, 0.7, 0]), 50));
    for (let r = 0; r < 50; r++) {
      expect(v[4 * r + 1]).toBe(0);
      expect(v[4 * r + 3]).toBe(0);
      expect((v[4 * r] as number) + (v[4 * r + 2] as number)).toBe(500);
    }
  });

  it("supports a shape for size and float dtypes", () => {
    setSeed(16);
    const out = multinomial(7, tensor([0.5, 0.5]), [2, 3], { dtype: "float64" });
    expect(out.shape).toEqual([2, 3, 2]);
    expect(out.dtype).toBe("float64");
  });

  it("validates its arguments", () => {
    expect(() => multinomial(-1, tensor([1]))).toThrow(InvalidParameterError);
    expect(() => multinomial(3, tensor([0.5, -0.5]))).toThrow(InvalidParameterError);
    expect(() => multinomial(3, tensor([0.5, Number.NaN]))).toThrow(InvalidParameterError);
    expect(() => multinomial(3, tensor([0, 0]))).toThrow(InvalidParameterError);
    expect(() => multinomial(3, tensor([] as number[]))).toThrow(InvalidParameterError);
    expect(() => multinomial(3, tensor([1]), -1)).toThrow(InvalidParameterError);
    expect(() => multinomial(3, tensor([1]), 1.5)).toThrow(InvalidParameterError);
    expect(() => multinomial(3, tensor([1]), 1, { dtype: "uint8" })).toThrow(DTypeError);
  });

  it("reads strided pvals", () => {
    const buffer = Float64Array.from([0, 9, 1, 9]);
    const pv = view(buffer, [2], 0, [2]);
    setSeed(17);
    // logical pvals are [0, 1]: every trial lands in the second category
    expect(nums(multinomial(25, pv, 1))).toEqual([0, 25]);
  });
});

describe("c56 multivariate_normal", () => {
  it("reproduces the target covariance (numpy reference cov [[2, 0.6], [0.6, 1]])", () => {
    setSeed(18);
    const n = 60000;
    const out = nums(
      multivariate_normal(
        tensor([1, -1], { dtype: "float64" }),
        tensor(
          [
            [2, 0.6],
            [0.6, 1],
          ],
          { dtype: "float64" }
        ),
        n,
        { dtype: "float64" }
      )
    );
    const a: number[] = [];
    const b: number[] = [];
    for (let i = 0; i < n; i++) {
      a.push(out[2 * i] as number);
      b.push(out[2 * i + 1] as number);
    }
    const ma = mean(a);
    const mb = mean(b);
    let cov = 0;
    for (let i = 0; i < n; i++) cov += ((a[i] as number) - ma) * ((b[i] as number) - mb);
    cov /= n;
    expect(ma).toBeCloseTo(1, 1);
    expect(mb).toBeCloseTo(-1, 1);
    expect(variance(a)).toBeCloseTo(2, 1);
    expect(variance(b)).toBeCloseTo(1, 1);
    expect(cov).toBeCloseTo(0.6, 1);
  });

  it("rejects a covariance that is not positive semi-definite", () => {
    const mean0 = tensor([0, 0]);
    const bad = tensor([
      [1, 2],
      [2, 1],
    ]);
    expect(() => multivariate_normal(mean0, bad, 3)).toThrow(InvalidParameterError);
  });

  it("rejects an asymmetric covariance", () => {
    const mean0 = tensor([0, 0]);
    const asym = tensor([
      [1, 0.5],
      [0, 1],
    ]);
    expect(() => multivariate_normal(mean0, asym, 3)).toThrow(InvalidParameterError);
  });

  it("rejects a zero pivot with a non-zero column", () => {
    const mean0 = tensor([0, 0]);
    const bad = tensor([
      [0, 1],
      [1, 0],
    ]);
    expect(() => multivariate_normal(mean0, bad, 3)).toThrow(InvalidParameterError);
  });

  it("accepts a singular covariance: perfectly correlated components", () => {
    setSeed(19);
    const out = nums(
      multivariate_normal(
        tensor([0, 0]),
        tensor([
          [1, 1],
          [1, 1],
        ]),
        200
      )
    );
    for (let i = 0; i < 200; i++) expect(out[2 * i]).toBe(out[2 * i + 1]);
  });

  it("honors the dtype option, shape sizes and non-finite entries", () => {
    setSeed(20);
    const eye = tensor([
      [1, 0],
      [0, 1],
    ]);
    const out = multivariate_normal(tensor([0, 0]), eye, [2, 3], { dtype: "float64" });
    expect(out.dtype).toBe("float64");
    expect(out.shape).toEqual([2, 3, 2]);
    expect(() => multivariate_normal(tensor([Number.NaN, 0]), eye, 1)).toThrow(
      InvalidParameterError
    );
    expect(() => multivariate_normal(tensor([0, 0]), eye, -2)).toThrow(InvalidParameterError);
  });

  it("accepts a float32 covariance whose rounding makes a pivot slightly negative", () => {
    // 0.9999999 rounds to 0.99999988 in float32, so the second pivot is -1.2e-7.
    const cov = tensor([
      [1, 1],
      [1, 0.9999999],
    ]);
    expect(cov.dtype).toBe("float32");
    setSeed(24);
    const out = nums(multivariate_normal(tensor([0, 0]), cov, 50));
    expect(out.every((v) => Number.isFinite(v))).toBe(true);
  });

  it("accepts a rank-deficient float32 sample covariance", () => {
    // 3 observations of 5 variables: rank 2, so three pivots are pure rounding noise.
    const rows = [
      [1, 2, 0.5, 4, 3],
      [2, 1, 1.5, 0, 2],
      [0, 3, 2.5, 1, 7],
    ];
    const means = [0, 1, 2, 3, 4].map((j) => mean(rows.map((r) => r[j] as number)));
    const cov = [0, 1, 2, 3, 4].map((i) =>
      [0, 1, 2, 3, 4].map(
        (j) =>
          rows.reduce(
            (a, r) =>
              a +
              ((r[i] as number) - (means[i] as number)) * ((r[j] as number) - (means[j] as number)),
            0
          ) / 2
      )
    );
    setSeed(25);
    const out = multivariate_normal(tensor(means), tensor(cov), 10);
    expect(nums(out).every((v) => Number.isFinite(v))).toBe(true);
  });

  it("still rejects a float64 covariance with a clearly negative pivot", () => {
    const bad = tensor(
      [
        [1, 1],
        [1, 0.999],
      ],
      { dtype: "float64" }
    );
    expect(() => multivariate_normal(tensor([0, 0], { dtype: "float64" }), bad, 1)).toThrow(
      InvalidParameterError
    );
  });

  it("reads a transposed covariance view", () => {
    setSeed(21);
    const cov = transpose(
      tensor([
        [4, 0],
        [0, 9],
      ])
    );
    const out = nums(multivariate_normal(tensor([0, 0]), cov, 20000));
    const a: number[] = [];
    const b: number[] = [];
    for (let i = 0; i < out.length; i += 2) {
      a.push(out[i] as number);
      b.push(out[i + 1] as number);
    }
    expect(variance(a)).toBeGreaterThan(3.7);
    expect(variance(a)).toBeLessThan(4.3);
    expect(variance(b)).toBeGreaterThan(8.4);
    expect(variance(b)).toBeLessThan(9.6);
  });
});

describe("c56 gumbel_softmax", () => {
  it("rejects NaN and +Infinity logits and rows without a finite logit", () => {
    expect(() => gumbel_softmax(tensor([0, Number.NaN]))).toThrow(InvalidParameterError);
    expect(() => gumbel_softmax(tensor([0, Number.POSITIVE_INFINITY]))).toThrow(
      InvalidParameterError
    );
    expect(() =>
      gumbel_softmax(tensor([Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY]))
    ).toThrow(InvalidParameterError);
    expect(() => gumbel_softmax(tensor([] as number[]))).toThrow(InvalidParameterError);
  });

  it("gives masked (-Infinity) categories exactly zero probability", () => {
    setSeed(22);
    const out = nums(gumbel_softmax(tensor([0, Number.NEGATIVE_INFINITY, 1])));
    expect(out[1]).toBe(0);
    expect(out[0] as number).toBeGreaterThan(0);
    expect(Math.abs(out[0] + out[2] - 1)).toBeLessThan(1e-12);
  });

  it("handles transposed 2D logits and returns one-hot rows when hard", () => {
    setSeed(23);
    const logits = transpose(
      tensor([
        [0, 50],
        [50, 0],
        [0, 0],
      ])
    ); // logical shape [2, 3]: rows [0, 50, 0] and [50, 0, 0]
    const out = nums(gumbel_softmax(logits, 1, true));
    expect(out.slice(0, 3)).toEqual([0, 1, 0]);
    expect(out.slice(3, 6)).toEqual([1, 0, 0]);
  });
});

describe("c56 integer dtype overflow", () => {
  it("geometric with a tiny p throws for int32 instead of wrapping", () => {
    setSeed(24);
    expect(() => geometric(1e-10, [200])).toThrow(InvalidParameterError);
  });

  it("geometric supports p below 1e-16 with int64", () => {
    setSeed(24);
    const out = geometric(1e-17, [10], { dtype: "int64" });
    expect(nums(out).every((v) => v >= 1)).toBe(true);
  });

  it("zipf never returns a wrapped (non-positive) int32 value", () => {
    setSeed(25);
    const v = nums(zipf(1.01, [300]));
    expect(v.every((x) => x >= 1 && x <= 2147483647)).toBe(true);
  });

  it("negative_binomial with a huge mean throws for int32 and works for int64", () => {
    setSeed(26);
    expect(() => negative_binomial(1, 1e-12, [8])).toThrow(InvalidParameterError);
    const v = nums(negative_binomial(1, 1e-12, [8], { dtype: "int64" }));
    expect(v.some((x) => x > 2 ** 31)).toBe(true);
  });
});

describe("c56 binomial accuracy", () => {
  it("handles success probabilities below 1e-16 (mean 0.45 for n = 2^53 - 1, p = 5e-17)", () => {
    setSeed(27);
    const x = nums(binomial(2 ** 53 - 1, 5e-17, [30000], { dtype: "int64" }));
    // scipy.stats.binom(2**53 - 1, 5e-17): mean 0.4503599627370495
    expect(Math.abs(mean(x) - 0.45036)).toBeLessThan(0.03);
  });

  it("is accurate for very large n (scipy.stats.binom(1e12, 0.3): sd 458257.57)", () => {
    setSeed(28);
    const x = nums(binomial(1e12, 0.3, [400], { dtype: "int64" }));
    const sd = Math.sqrt(variance(x));
    expect(Math.abs(mean(x) - 3e11)).toBeLessThan(5 * (458257.57 / Math.sqrt(400)));
    expect(sd).toBeGreaterThan(458257.57 * 0.85);
    expect(sd).toBeLessThan(458257.57 * 1.15);
  });

  it("matches the exact binomial pmf mean/variance for n = 1e6, p = 0.3", () => {
    setSeed(29);
    const x = nums(binomial(1e6, 0.3, [20000], { dtype: "int64" }));
    // scipy: mean 300000, var 210000
    expect(Math.abs(mean(x) - 300000)).toBeLessThan(5 * Math.sqrt(210000 / 20000));
    expect(variance(x)).toBeGreaterThan(210000 * 0.95);
    expect(variance(x)).toBeLessThan(210000 * 1.05);
  });
});

describe("c56 uniform", () => {
  it("stays finite when high - low overflows float64", () => {
    setSeed(30);
    const x = nums(uniform(-1.7e308, 1.7e308, [500], { dtype: "float64" }));
    expect(x.every((v) => Number.isFinite(v) && v >= -1.7e308 && v <= 1.7e308)).toBe(true);
    expect(Math.min(...x)).toBeLessThan(-1e307);
    expect(Math.max(...x)).toBeGreaterThan(1e307);
  });
});

describe("c56 Generator", () => {
  it("randint is exactly uniform over wide ranges (no 2^-32 granularity)", () => {
    const g = new Generator(31);
    const draws: number[] = [];
    for (let i = 0; i < 200; i++) draws.push(g.randint(0, 2 ** 40));
    expect(draws.every((v) => Number.isInteger(v) && v >= 0 && v < 2 ** 40)).toBe(true);
    // floor(u * 2^40) with 32-bit u can only produce multiples of 256.
    expect(draws.some((v) => v % 256 !== 0)).toBe(true);
    expect(draws.some((v) => v > 2 ** 32)).toBe(true);
  });

  it("randint rejects unsafe bounds and ranges", () => {
    const g = new Generator(32);
    expect(() => g.randint(0, 2 ** 53 + 2)).toThrow(InvalidParameterError);
    expect(() => g.randint(-(2 ** 53) + 1, 2 ** 53 - 1)).toThrow(InvalidParameterError);
    expect(() => g.randint(1.5, 4)).toThrow(InvalidParameterError);
    expect(() => g.randint(3, 3)).toThrow(InvalidParameterError);
    expect(() => g.randint(Number.NaN, 3)).toThrow(InvalidParameterError);
  });

  it("randintArray keeps the pre-existing stream for small ranges", () => {
    // floor(u * range) over the same generator stream.
    const reference = new Generator(33);
    const expected = Array.from({ length: 50 }, () => Math.floor(reference.random() * 10));
    expect(Array.from(new Generator(33).randintArray(0, 10, 50))).toEqual(expected);
    const ref2 = new Generator(34);
    const expected2 = Array.from({ length: 20 }, () => -5 + Math.floor(ref2.random() * 11));
    expect(new Generator(34).randintArray(-5, 6, 20)).toEqual(Int32Array.from(expected2));
  });

  it("randint of a single value matches floor(u * range) for small ranges", () => {
    const ref = new Generator(35);
    const g = new Generator(35);
    for (let i = 0; i < 30; i++) expect(g.randint(3, 100)).toBe(3 + Math.floor(ref.random() * 97));
  });

  it("randintArray covers int32-wide ranges and rejects bounds that do not fit", () => {
    const g = new Generator(36);
    const wide = g.randintArray(-(2 ** 31), 2 ** 31, 2000);
    expect(Math.min(...wide)).toBeLessThan(-(2 ** 29));
    expect(Math.max(...wide)).toBeGreaterThan(2 ** 29);
    expect(() => g.randintArray(0, 2 ** 31 + 1, 3)).toThrow(InvalidParameterError);
    expect(() => g.randintArray(-(2 ** 31) - 1, 3, 3)).toThrow(InvalidParameterError);
  });

  it("randintArray is uniform for a range that does not divide 2^32", () => {
    // scipy.stats.chi2.ppf(0.9999, 6) = 27.86 (7 equally likely values)
    const g = new Generator(37);
    const counts = new Array<number>(7).fill(0);
    const n = 70000;
    for (const v of g.randintArray(0, 7, n)) counts[v] = (counts[v] as number) + 1;
    const chi2 = counts.reduce((a, c) => a + (c - n / 7) ** 2 / (n / 7), 0);
    expect(chi2).toBeLessThan(27.86);
  });

  it("shuffle and permutation consume the stream like a sequential Fisher-Yates", () => {
    const ref = new Generator(38);
    const expected = Array.from({ length: 20 }, (_, i) => i);
    for (let i = 19; i > 0; i--) {
      const j = Math.floor(ref.random() * (i + 1));
      [expected[i], expected[j]] = [expected[j] as number, expected[i] as number];
    }
    expect(Array.from(new Generator(38).permutation(20))).toEqual(expected);
    const arr = Array.from({ length: 20 }, (_, i) => i);
    new Generator(38).shuffle(arr);
    expect(arr).toEqual(expected);
  });

  it("shuffle accepts typed arrays", () => {
    const arr = Int32Array.from([1, 2, 3, 4, 5, 6, 7, 8]);
    new Generator(39).shuffle(arr);
    expect(Array.from(arr).sort((a, b) => a - b)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    expect(Array.from(arr)).not.toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it("shuffle refuses arrays longer than 2^31 elements before touching them", () => {
    const huge = { length: 2 ** 31 + 1 } as unknown as number[];
    expect(() => new Generator(47).shuffle(huge)).toThrow(InvalidParameterError);
  });

  it("choice never returns a zero-weight index and validates the weights", () => {
    const g = new Generator(40);
    for (let i = 0; i < 500; i++) expect(g.choice([0, 0, 1, 0])).toBe(2);
    expect(() => g.choice([1, -1])).toThrow(InvalidParameterError);
    expect(() => g.choice([1, Number.NaN])).toThrow(InvalidParameterError);
    expect(() => g.choice([1, Number.POSITIVE_INFINITY])).toThrow(InvalidParameterError);
    expect(() => g.choice([0, 0])).toThrow(InvalidParameterError);
    expect(() => g.choice([])).toThrow(InvalidParameterError);
  });

  it("choice follows the requested weights", () => {
    const g = new Generator(41);
    const hits = [0, 0, 0];
    for (let i = 0; i < 30000; i++) {
      const k = g.choice([1, 2, 7]);
      hits[k] = (hits[k] as number) + 1;
    }
    expect((hits[0] as number) / 30000).toBeCloseTo(0.1, 1);
    expect((hits[1] as number) / 30000).toBeCloseTo(0.2, 1);
    expect((hits[2] as number) / 30000).toBeCloseTo(0.7, 1);
  });

  it("rejects NaN and non-finite parameters instead of returning NaN", () => {
    const g = new Generator(42);
    expect(() => g.normal(0, Number.NaN)).toThrow(InvalidParameterError);
    expect(() => g.normal(Number.POSITIVE_INFINITY, 1)).toThrow(InvalidParameterError);
    expect(() => g.normalArray(Number.NaN, 1, 3)).toThrow(InvalidParameterError);
    expect(() => g.uniform(Number.NaN, 1)).toThrow(InvalidParameterError);
    expect(() => g.uniformArray(0, Number.NaN, 3)).toThrow(InvalidParameterError);
    expect(() => g.exponential(Number.NaN)).toThrow(InvalidParameterError);
    expect(() => g.exponential(Number.POSITIVE_INFINITY)).toThrow(InvalidParameterError);
    expect(() => g.bernoulli(Number.NaN)).toThrow(InvalidParameterError);
    expect(() => g.bernoulli(1.5)).toThrow(InvalidParameterError);
  });

  it("uniform stays finite when high - low overflows", () => {
    const g = new Generator(43);
    for (let i = 0; i < 200; i++) {
      const v = g.uniform(-1.7e308, 1.7e308);
      expect(Number.isFinite(v)).toBe(true);
    }
    const arr = g.uniformArray(-1.7e308, 1.7e308, 200);
    expect(arr.every((v) => Number.isFinite(v))).toBe(true);
  });

  it("validates sizes and windows with typed errors", () => {
    const g = new Generator(44);
    expect(() => g.randomArray(-1)).toThrow(InvalidParameterError);
    expect(() => g.randomArray(1.5)).toThrow(InvalidParameterError);
    expect(() => g.normalArray(0, 1, Number.NaN)).toThrow(InvalidParameterError);
    expect(() => g.randintArray(0, 5, -3)).toThrow(InvalidParameterError);
    expect(() => g.permutation(2 ** 31 + 1)).toThrow(InvalidParameterError);
    const out = new Float64Array(4);
    expect(() => g.fillNormalInto(out, 2, 5)).toThrow(InvalidParameterError);
    expect(() => g.fillUniformInto(out, -1, 2)).toThrow(InvalidParameterError);
    expect(out.every((v) => v === 0)).toBe(true);
    g.fillNormalInto(out, 1, 2);
    expect(out[0]).toBe(0);
    expect(out[3]).toBe(0);
    expect(out[1]).not.toBe(0);
  });

  it("fillNormalInto matches normalArray for the same stream", () => {
    const a = new Generator(45).normalArray(0, 1, 6);
    const out = new Float64Array(8);
    new Generator(45).fillNormalInto(out, 2, 6);
    expect(Array.from(out.slice(2))).toEqual(Array.from(a));
  });

  it("normal() follows the standard normal moments (scipy.stats.norm)", () => {
    const g = new Generator(46);
    const x = g.normalArray(2, 3, 60000);
    expect(mean(x)).toBeCloseTo(2, 1);
    expect(Math.sqrt(variance(x))).toBeCloseTo(3, 1);
  });

  it("seed property reports the constructor argument and truncates fractions for the stream", () => {
    const a = new Generator(1.2);
    const b = new Generator(1.9);
    expect(a.seed).toBe(1.2);
    expect(a.random()).toBe(b.random());
  });
});
