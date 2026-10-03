/**
 * Wave 2 regression tests for cross-module issues: non-float global default dtypes,
 * seed handling in the model-level generators, unbiased bounded integers, half-precision
 * results, dense (size-1 stride) views in the ML input checks, float64 gradients on float32
 * graphs, and host-only layers under `Module.to`.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import type { Backend } from "../../src/core";
import {
  DTypeError,
  getConfig,
  isBackendAvailable,
  registerBackend,
  setConfig,
  unregisterBackend,
} from "../../src/core";
import type { ml } from "../../src/index";
import { hilbert, toeplitz } from "../../src/linalg";
import {
  type BayesianRidgeOptions,
  type ElasticNetOptions,
  type HuberRegressorOptions,
  IsolationForest,
  type IsotonicOutOfBounds,
  type IsotonicRegressionOptions,
  type KernelRidgeKernel,
  type KernelRidgeOptions,
  KMeans,
  type LassoOptions,
  LinearRegression,
} from "../../src/ml";
import {
  add,
  clip,
  customOp,
  div,
  dot,
  GradTensor,
  mul,
  sigmoid,
  Tensor,
  tensor,
  transpose,
} from "../../src/ndarray";
import { Conv3d, Embedding, Linear } from "../../src/nn";
import { CountVectorizer, HashingVectorizer, TfidfVectorizer } from "../../src/preprocess";
import { __randomBelow, __SeededRandom, __seedToUint64 } from "../../src/random/random";

const f64 = (v: unknown): Tensor => tensor(v as number[], { dtype: "float64" });
const flat = (t: Tensor): number[] => Array.from(t.data as ArrayLike<number>).slice(0, t.size);

describe("non-float global default dtype never truncates built values", () => {
  const original = getConfig().defaultDtype;
  afterEach(() => {
    setConfig({ defaultDtype: original });
  });

  it("hilbert falls back to float32 under integer, bool, string and complex defaults", () => {
    for (const dtype of ["int32", "int64", "uint8", "bool", "string", "complex64"] as const) {
      setConfig({ defaultDtype: dtype });
      const h = hilbert(3);
      expect(h.dtype).toBe("float32");
      expect(h.toArray()).toEqual([
        [1, 0.5, Math.fround(1 / 3)],
        [0.5, Math.fround(1 / 3), 0.25],
        [Math.fround(1 / 3), 0.25, 0.2].map((v) => Math.fround(v)),
      ]);
    }
  });

  it("matrix constructors keep an integer default when every value fits", () => {
    setConfig({ defaultDtype: "int32" });
    const t = toeplitz([1, 2, 3], [9, 5, 6, 7]);
    expect(t.dtype).toBe("int32");
    // a fractional input would be truncated by int32, so it falls back to float32
    const frac = toeplitz([1.5, 2]);
    expect(frac.dtype).toBe("float32");
    expect(frac.toArray()).toEqual([
      [1.5, 2],
      [2, 1.5],
    ]);
    setConfig({ defaultDtype: "bool" });
    expect(toeplitz([1, 0]).dtype).toBe("bool");
    expect(toeplitz([1, 2]).dtype).toBe("float32");
    setConfig({ defaultDtype: "uint8" });
    expect(toeplitz([1, 300]).dtype).toBe("float32");
  });

  it("text vectorizers keep fractional weights and counts above one", () => {
    const docs = ["alpha beta beta", "beta gamma"];
    setConfig({ defaultDtype: "int32" });
    const counts = new CountVectorizer().fitTransformText(docs);
    expect(counts.dtype).toBe("int32");
    expect(counts.toArray()).toEqual([
      [1, 2, 0],
      [0, 1, 1],
    ]);
    const tfidf = new TfidfVectorizer().fitTransformText(docs);
    expect(tfidf.dtype).toBe("float32");
    const w = flat(tfidf);
    expect(w[0]).toBeCloseTo(0.5749618, 5);
    expect(w[1]).toBeCloseTo(0.8181802, 5);
    const hashed = new HashingVectorizer({ nFeatures: 4 }).fitTransformText(docs);
    expect(hashed.dtype).toBe("float32");
    expect(flat(hashed).some((v) => !Number.isInteger(v))).toBe(true);

    setConfig({ defaultDtype: "bool" });
    const boolCounts = new CountVectorizer().fitTransformText(docs);
    expect(boolCounts.dtype).toBe("float32");
    expect(flat(boolCounts)).toEqual([1, 2, 0, 0, 1, 1]);

    setConfig({ defaultDtype: "float64" });
    expect(new TfidfVectorizer().fitTransformText(docs).dtype).toBe("float64");
  });
});

describe("random module contracts used by other modules", () => {
  it("__SeededRandom takes a uint64 bigint and returns values in [0, 1)", () => {
    const a = new __SeededRandom(42n);
    const b = new __SeededRandom(42n);
    for (let i = 0; i < 100; i++) {
      const x = a.next();
      expect(x).toBe(b.next());
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThan(1);
      expect(Number.isInteger(x * 4294967296)).toBe(true);
    }
  });

  it("__seedToUint64 keeps integer seeds and separates fractional ones", () => {
    expect(__seedToUint64(7)).toBe(7n);
    expect(__seedToUint64(-1)).toBe(2n ** 64n - 1n);
    expect(__seedToUint64(0.5)).not.toBe(__seedToUint64(0.7));
    expect(__seedToUint64(0.5)).not.toBe(__seedToUint64(0));
  });

  it("models tell fractional seeds 0.5 and 0.7 apart (both used to truncate to 0)", () => {
    const X = tensor(
      Array.from({ length: 30 }, (_, i) => [Math.sin(i * 1.3), Math.cos(i * 0.7)]),
      { dtype: "float64" }
    );
    const score = (seed: number): number[] =>
      flat(new IsolationForest({ randomState: seed }).fit(X).scoreSamples(X));
    expect(score(0.5)).not.toEqual(score(0.7));
    expect(score(0.5)).toEqual(score(0.5));
    const centers = (seed: number): number[] =>
      flat(new KMeans({ nClusters: 3, randomState: seed, nInit: 1 }).fit(X).clusterCenters);
    expect(centers(0.5)).not.toEqual(centers(0.7));
  });
});

describe("__randomBelow", () => {
  const fromWords = (words: number[]): (() => number) => {
    let i = 0;
    return () => (words[i++] as number) / 4294967296;
  };

  it("rejects the biased low words that floor(u * bound) would accept", () => {
    // 2^32 mod 3 = 1, so the word 0 is one of the over-represented residues and is redrawn
    expect(Math.floor((0 / 4294967296) * 3)).toBe(0);
    expect(__randomBelow(fromWords([0, 4294967295]), 3)).toBe(2);
    // an unbiased word is used as is
    expect(__randomBelow(fromWords([2147483648]), 4)).toBe(2);
  });

  it("stays in range and covers every value", () => {
    const rng = new __SeededRandom(5n);
    const next = (): number => rng.next();
    const seen = new Set<number>();
    for (let i = 0; i < 2000; i++) {
      const v = __randomBelow(next, 7);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(7);
      seen.add(v);
    }
    expect(seen.size).toBe(7);
    // large bounds use plain rejection on the 32-bit word
    const big = __randomBelow(next, 3_000_000_000);
    expect(big).toBeGreaterThanOrEqual(0);
    expect(big).toBeLessThan(3_000_000_000);
  });

  it("falls back to floor(u * bound) for non-integer bounds and off-grid generators", () => {
    expect(__randomBelow(() => 0.5, 5)).toBe(2);
    expect(__randomBelow(() => 0.25, 2.5)).toBe(0);
    expect(__randomBelow(() => 0.9, 0)).toBe(0);
  });
});

describe("half-precision results stay on the dtype grid", () => {
  const a = tensor([1.1, 2.3, 3.7], { dtype: "float16" });
  const b = tensor([0.1, 0.7, 0.3], { dtype: "float16" });

  it("add, mul and div match numpy float16", () => {
    // np.float16 arithmetic on the same inputs
    expect(flat(add(a, b))).toEqual([1.19921875, 3, 4]);
    expect(flat(mul(a, b))).toEqual([0.10992431640625, 1.611328125, 1.1103515625]);
    expect(flat(div(a, b))).toEqual([11, 3.28515625, 12.328125]);
    expect(add(a, b).dtype).toBe("float16");
  });

  it("sigmoid, clip and dot match numpy and torch float16", () => {
    expect(flat(sigmoid(a))).toEqual([0.75, 0.9091796875, 0.97607421875]);
    expect(flat(clip(a, 0.3, 2))).toEqual([1.099609375, 2, 2]);
    expect(flat(dot(a, b))).toEqual([2.830078125]);
  });

  it("bfloat16 results are rounded too", () => {
    const x = tensor([1.1, 2.3], { dtype: "bfloat16" });
    // torch: (x * x) in bfloat16
    const squared = flat(mul(x, x));
    for (const v of squared) {
      const bits = new Uint32Array(new Float32Array([v]).buffer)[0] as number;
      expect(bits & 0xffff).toBe(0);
    }
  });

  it("float32 results are untouched", () => {
    const x = tensor([1.1, 2.3], { dtype: "float32" });
    expect(flat(mul(x, x))[0]).toBe(Math.fround(Math.fround(1.1) * Math.fround(1.1)));
  });
});

describe("dense views in the ML input checks", () => {
  const X = f64([[0], [1], [2], [3], [4], [5]]);
  const y = f64([0, 1, 2, 3, 4, 5]);

  it("accepts the transpose of a row (dense, only size-1 strides differ)", () => {
    const model = new LinearRegression().fit(X, y);
    const column = transpose(f64([[0, 1, 2, 3, 4, 5]]));
    expect(column.shape).toEqual([6, 1]);
    expect(flat(model.predict(column))).toEqual(flat(model.predict(X)));
  });

  it("still rejects genuinely strided views", () => {
    const model = new LinearRegression().fit(X, y);
    const strided = transpose(
      f64([
        [0, 1],
        [2, 3],
      ])
    );
    expect(() => model.predict(strided)).toThrow(/contiguous/);
  });
});

describe("float64 gradients on float32 graphs", () => {
  it("accumulateGrad casts a float64 gradient to the tensor dtype, also on repeated calls", () => {
    const w = GradTensor.fromTensor(tensor([1, 2], { dtype: "float32" }), { requiresGrad: true });
    w.accumulateGrad(tensor([0.5, 0.25], { dtype: "float64" }));
    w.accumulateGrad(tensor([0.5, 0.25], { dtype: "float32" }));
    expect(w.grad?.dtype).toBe("float32");
    expect(flat(w.grad as Tensor)).toEqual([1, 0.5]);
  });

  it("customOp backward works when the rule computes in float64", () => {
    const x = GradTensor.fromTensor(tensor([1, 2, 3], { dtype: "float32" }), {
      requiresGrad: true,
    });
    const out = customOp(tensor([2, 4, 6], { dtype: "float32" }), [
      [
        x,
        (g) =>
          tensor(
            Array.from(flat(g), (v) => v * 2),
            { dtype: "float64" }
          ),
      ],
    ]);
    out.backward();
    out.backward();
    expect(x.grad?.dtype).toBe("float32");
    expect(flat(x.grad as Tensor)).toEqual([4, 4, 4]);
  });
});

describe("dot on non-contiguous and integer inputs follows NumPy", () => {
  it("propagates Infinity and NaN like numpy through a transposed view", () => {
    const a = transpose(
      f64([
        [Number.POSITIVE_INFINITY, 1],
        [0, 3],
      ])
    );
    const r = flat(
      dot(
        a,
        f64([
          [1, 0],
          [0, 1],
        ])
      )
    );
    // numpy: [[inf, nan], [1, 3]]
    expect(r[0]).toBe(Number.POSITIVE_INFINITY);
    expect(Number.isNaN(r[1])).toBe(true);
    expect(r.slice(2)).toEqual([1, 3]);
  });

  it("keeps int32 exact and float32 gradients in float32", () => {
    const i = tensor(
      [
        [1, 2],
        [3, 4],
      ],
      { dtype: "int32" }
    );
    const j = tensor(
      [
        [5, 6],
        [7, 8],
      ],
      { dtype: "int32" }
    );
    expect(dot(i, j).dtype).toBe("int32");
    expect(dot(i, j).toArray()).toEqual([
      [19, 22],
      [43, 50],
    ]);
    const g = GradTensor.fromTensor(
      tensor(
        [
          [1, 2],
          [3, 4],
        ],
        { dtype: "float32" }
      ),
      {
        requiresGrad: true,
      }
    );
    const w = GradTensor.fromTensor(
      tensor(
        [
          [5, 6],
          [7, 8],
        ],
        { dtype: "float32" }
      ),
      {
        requiresGrad: true,
      }
    );
    g.matmul(w).sum().backward();
    expect(g.grad?.dtype).toBe("float32");
    expect(g.grad?.toArray()).toEqual([
      [11, 15],
      [11, 15],
    ]);
  });
});

describe("complex dtypes are rejected by tensors", () => {
  it("creation, casts and typed-array wrapping name the complex array types", () => {
    expect(() => tensor([1, 2], { dtype: "complex64" })).toThrow(DTypeError);
    expect(() => tensor([1, 2]).astype("complex128")).toThrow(DTypeError);
    expect(() =>
      Tensor.fromTypedArray({
        data: new Float32Array(2),
        shape: [2],
        dtype: "complex64",
        device: "cpu",
      })
    ).toThrow(/Complex64Array/);
  });
});

describe("ml option types are exported", () => {
  it("are importable from the ml barrel and through the root namespace", () => {
    const options: {
      elastic: ElasticNetOptions;
      lasso: LassoOptions;
      ridge: BayesianRidgeOptions;
      huber: HuberRegressorOptions;
      isotonic: IsotonicRegressionOptions;
      kernel: KernelRidgeOptions;
      kind: KernelRidgeKernel;
      bounds: IsotonicOutOfBounds;
      viaRoot: ml.ElasticNetOptions;
    } = {
      elastic: { alpha: 0.1 },
      lasso: { alpha: 0.1 },
      ridge: {},
      huber: {},
      isotonic: {},
      kernel: {},
      kind: "rbf",
      bounds: "clip",
      viaRoot: { alpha: 0.2 },
    };
    expect(options.kind).toBe("rbf");
    expect(options.viaRoot.alpha).toBe(0.2);
  });
});

describe("Module.to leaves host-only layers on the host", () => {
  it("moves Linear weights but not Embedding, Conv3d weights", async () => {
    const hostBackend: Backend = {
      info: () => ({ device: "wasm", name: "noop", available: true, capabilities: [] }),
      supports: () => false,
      init: async () => {},
      dispose: () => {},
    };
    const hadBackend = isBackendAvailable("wasm");
    registerBackend("wasm", hostBackend);
    const moved: Tensor[] = [];
    const spy = vi.spyOn(Tensor.prototype, "to").mockImplementation(function (this: Tensor) {
      moved.push(this);
      return Promise.resolve(this);
    } as unknown as Tensor["to"]);
    try {
      const linear = new Linear(2, 2);
      const embedding = new Embedding(4, 3);
      const conv = new Conv3d(1, 1, 2);
      await linear.to("wasm");
      await embedding.to("wasm");
      await conv.to("wasm");
      const linearWeights = [...linear.parameters()].map((p) => p.tensor);
      expect(linearWeights.every((t) => moved.includes(t))).toBe(true);
      for (const p of embedding.parameters()) expect(moved.includes(p.tensor)).toBe(false);
      for (const p of conv.parameters()) expect(moved.includes(p.tensor)).toBe(false);
    } finally {
      spy.mockRestore();
      if (!hadBackend) unregisterBackend("wasm");
    }
  });
});
