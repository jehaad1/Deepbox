import { afterEach, describe, expect, it } from "vitest";
import { getConfig, setConfig } from "../../src/core";
import {
  DataLoader,
  type DatasetLoadOptions,
  iterableDataset,
  loadHousingMini,
  loadIris,
  loadMoonsMulti,
  makeCircles,
  makeMoons,
  parseCSV,
  SubsetRandomSampler,
  type TextFetchOptions,
  WeightedRandomSampler,
} from "../../src/datasets";
import { createPassRng, createRng } from "../../src/datasets/utils";
import {
  averagePrecisionScore,
  type ConfusionMatrixOptions,
  confusionMatrix,
  mse,
  precisionRecallCurve,
  rocAucScore,
  rocCurve,
} from "../../src/metrics";
import { compensatedSum, euclideanDistance, readFiniteFloat64 } from "../../src/metrics/_internal";
import { Tensor, tensor } from "../../src/ndarray";

function f64(t: { data: unknown }): number[] {
  return Array.from(t.data as ArrayLike<number>);
}

describe("samplers: seed validation and speed", () => {
  it("rejects invalid seeds like every other seeded dataset API", () => {
    expect(() => new SubsetRandomSampler([0, 1], { seed: 1.5 })).toThrow(/seed/);
    expect(() => new SubsetRandomSampler([0, 1], { seed: Number.NaN })).toThrow(/seed/);
    expect(() => new WeightedRandomSampler([1, 1], { seed: 0.5 })).toThrow(/seed/);
    expect(() => new WeightedRandomSampler([1, 1], { seed: Number.POSITIVE_INFINITY })).toThrow(
      /seed/
    );
  });

  it("draws without replacement from a large population in near-linear time", () => {
    const n = 200_000;
    const weights = new Float64Array(n).fill(1);
    weights[0] = 0;
    const sampler = new WeightedRandomSampler(weights, {
      replacement: false,
      numSamples: n - 1,
      seed: 3,
    });
    const start = Date.now();
    const drawn = Array.from(sampler);
    expect(Date.now() - start).toBeLessThan(5000);
    expect(drawn).toHaveLength(n - 1);
    expect(new Set(drawn).size).toBe(n - 1);
    expect(drawn).not.toContain(0);
  });

  it("keeps the without-replacement distribution proportional to the weights", () => {
    const counts = [0, 0, 0, 0];
    const trials = 20000;
    for (let t = 0; t < trials; t++) {
      const sampler = new WeightedRandomSampler([1, 2, 3, 0], {
        replacement: false,
        numSamples: 1,
        seed: t,
      });
      const [first] = Array.from(sampler);
      counts[first as number] = (counts[first as number] as number) + 1;
    }
    expect((counts[0] as number) / trials).toBeCloseTo(1 / 6, 1);
    expect((counts[1] as number) / trials).toBeCloseTo(2 / 6, 1);
    expect((counts[2] as number) / trials).toBeCloseTo(3 / 6, 1);
    expect(counts[3]).toBe(0);
  });

  it("returns every positive-weight index exactly once when drawing them all", () => {
    const sampler = new WeightedRandomSampler([0, 5, 0, 1e-300, 2, 0, 1e300], {
      replacement: false,
      numSamples: 4,
      seed: 11,
    });
    expect(Array.from(sampler).sort((a, b) => a - b)).toEqual([1, 3, 4, 6]);
  });
});

describe("reshuffleEachIteration", () => {
  it("createPassRng restarts per pass by default and continues when asked", () => {
    const restart = createPassRng(5, false);
    expect(restart()()).toBe(restart()());
    const shared = createPassRng(5, true);
    expect(shared()).toBe(shared());
    // Without a seed the option has no effect and each call is a usable generator.
    expect(typeof createPassRng(undefined, true)()()).toBe("number");
  });

  it("SubsetRandomSampler: epochs differ but stay reproducible", () => {
    const idx = Array.from({ length: 20 }, (_, i) => i);
    const plain = new SubsetRandomSampler(idx, { seed: 9 });
    expect(Array.from(plain)).toEqual(Array.from(plain));

    const a = new SubsetRandomSampler(idx, { seed: 9, reshuffleEachIteration: true });
    const b = new SubsetRandomSampler(idx, { seed: 9, reshuffleEachIteration: true });
    const a1 = Array.from(a);
    const a2 = Array.from(a);
    expect(a1).toEqual(Array.from(plain));
    expect(a2).not.toEqual(a1);
    expect([...a2].sort((x, y) => x - y)).toEqual(idx);
    expect(Array.from(b)).toEqual(a1);
    expect(Array.from(b)).toEqual(a2);
    expect(
      () => new SubsetRandomSampler(idx, { reshuffleEachIteration: "yes" as unknown as boolean })
    ).toThrow(/reshuffleEachIteration/);
  });

  it("WeightedRandomSampler: epochs differ but stay reproducible", () => {
    const opts = { numSamples: 12, seed: 4, reshuffleEachIteration: true } as const;
    const w = Array.from({ length: 30 }, (_, i) => i + 1);
    const a = new WeightedRandomSampler(w, opts);
    const b = new WeightedRandomSampler(w, opts);
    const a1 = Array.from(a);
    const a2 = Array.from(a);
    expect(a2).not.toEqual(a1);
    expect(Array.from(b)).toEqual(a1);
    expect(Array.from(b)).toEqual(a2);
    const plain = new WeightedRandomSampler(w, { numSamples: 12, seed: 4 });
    expect(Array.from(plain)).toEqual(a1);
    expect(Array.from(plain)).toEqual(a1);
  });

  it("DataLoader: tensor mode reshuffles every epoch only when asked", () => {
    const X = tensor(
      Array.from({ length: 16 }, (_, i) => [i]),
      { dtype: "float64" }
    );
    const order = (loader: DataLoader): number[] => {
      const out: number[] = [];
      for (const [batch] of loader) out.push(...f64(batch));
      return out;
    };
    const fixed = new DataLoader(X, { batchSize: 4, shuffle: true, seed: 2 });
    expect(order(fixed)).toEqual(order(fixed));

    const make = () =>
      new DataLoader(X, { batchSize: 4, shuffle: true, seed: 2, reshuffleEachIteration: true });
    const a = make();
    const e1 = order(a);
    const e2 = order(a);
    expect(e1).toEqual(order(fixed));
    expect(e2).not.toEqual(e1);
    const b = make();
    expect(order(b)).toEqual(e1);
    expect(order(b)).toEqual(e2);
    expect(() => new DataLoader(X, { reshuffleEachIteration: 1 as unknown as boolean })).toThrow(
      /reshuffleEachIteration/
    );
  });

  it("DataLoader and StreamingDataset: streaming mode reshuffles when asked", () => {
    const stream = iterableDataset(() => Array.from({ length: 30 }, (_, i) => [i]));
    const order = (loader: DataLoader): number[] => {
      const out: number[] = [];
      for (const [batch] of loader) out.push(...f64(batch));
      return out;
    };
    const fixed = new DataLoader(stream, { batchSize: 5, shuffleBufferSize: 30, seed: 8 });
    expect(order(fixed)).toEqual(order(fixed));

    const make = () =>
      new DataLoader(stream, {
        batchSize: 5,
        shuffleBufferSize: 30,
        seed: 8,
        reshuffleEachIteration: true,
      });
    const a = make();
    const e1 = order(a);
    const e2 = order(a);
    expect(e1).toEqual(order(fixed));
    expect(e2).not.toEqual(e1);
    expect([...e2].sort((x, y) => x - y)).toEqual(Array.from({ length: 30 }, (_, i) => i));
    const b = make();
    expect(order(b)).toEqual(e1);
    expect(order(b)).toEqual(e2);

    const direct = stream.shuffle(30, 8, { reshuffleEachIteration: true });
    const d1 = Array.from(direct).map((s) => (s as number[])[0]);
    const d2 = Array.from(direct).map((s) => (s as number[])[0]);
    expect(d2).not.toEqual(d1);
  });
});

describe("makeMoons / makeCircles odd sample counts (scikit-learn split)", () => {
  it("makeMoons gives the extra sample of an odd nSamples to class 1", () => {
    for (const n of [1, 3, 5, 7, 101]) {
      const [X, y] = makeMoons({ nSamples: n, shuffle: false });
      const labels = f64(y);
      expect(X.shape).toEqual([n, 2]);
      expect(labels.filter((l) => l === 0)).toHaveLength(Math.floor(n / 2));
      expect(labels.filter((l) => l === 1)).toHaveLength(n - Math.floor(n / 2));
    }
  });

  it("makeCircles gives the extra sample to class 1 and matches scikit-learn geometry", () => {
    const [X, y] = makeCircles({ nSamples: 5, factor: 0.8, shuffle: false });
    expect(f64(y)).toEqual([0, 0, 1, 1, 1]);
    // Reference: sklearn.datasets.make_circles(5, shuffle=False, factor=0.8).
    const ref = [
      [1, 0],
      [-1, 1.2246467991473532e-16],
      [0.8, 0],
      [-0.4, 0.692820323027551],
      [-0.4, -0.6928203230275507],
    ];
    const d = f64(X);
    ref.forEach(([cx, cy], i) => {
      expect(d[i * 2]).toBeCloseTo(cx as number, 12);
      expect(d[i * 2 + 1]).toBeCloseTo(cy as number, 12);
    });
    const counts = (n: number) => {
      const labels = f64(makeCircles({ nSamples: n, shuffle: false })[1]);
      return [labels.filter((l) => l === 0).length, labels.filter((l) => l === 1).length];
    };
    expect(counts(9)).toEqual([4, 5]);
    expect(counts(10)).toEqual([5, 5]);
  });
});

describe("public type exports", () => {
  it("exposes DatasetLoadOptions, TextFetchOptions and ConfusionMatrixOptions", () => {
    const load: DatasetLoadOptions = { dtype: "float64" };
    const text: TextFetchOptions = { subset: "train", maxSamples: 3 };
    const cm: ConfusionMatrixOptions = { normalize: "true" };
    expect(loadIris(load).data.dtype).toBe("float64");
    expect(text.subset).toBe("train");
    expect(f64(confusionMatrix(tensor([0, 1, 1]), tensor([0, 1, 0]), cm))).toEqual([
      1, 0, 0.5, 0.5,
    ]);
  });
});

describe("dataset values are never truncated by a non-float global dtype", () => {
  const original = getConfig().defaultDtype;
  afterEach(() => {
    setConfig({ defaultDtype: original });
  });

  it("reference and synthetic loaders fall back to float32", () => {
    setConfig({ defaultDtype: "int32" });
    const iris = loadIris();
    expect(iris.data.dtype).toBe("float32");
    expect(f64(iris.data).slice(0, 2)[0]).toBeCloseTo(5.1, 5);
    expect(iris.target.dtype).toBe("int32");

    const housing = loadHousingMini();
    expect(housing.data.dtype).toBe("float32");
    expect(housing.target.dtype).toBe("float32");
    expect(new Set(f64(housing.target)).size).toBeGreaterThan(5);

    const moons = loadMoonsMulti();
    expect(moons.data.dtype).toBe("float32");
    expect(f64(moons.data).some((v) => !Number.isInteger(v))).toBe(true);

    setConfig({ defaultDtype: "bool" });
    expect(loadIris().data.dtype).toBe("float32");
  });

  it("still follows a floating-point global dtype and an explicit option", () => {
    setConfig({ defaultDtype: "float64" });
    expect(loadIris().data.dtype).toBe("float64");
    expect(loadHousingMini().data.dtype).toBe("float64");
    expect(loadIris({ dtype: "float32" }).data.dtype).toBe("float32");
  });

  it("parseCSV keeps fractional values under an integer global dtype", () => {
    setConfig({ defaultDtype: "int32" });
    const ds = parseCSV("a,b,y\n1.5,2.25,0.5\n3.5,4.75,1.5\n");
    expect(ds.data.dtype).toBe("float32");
    expect(f64(ds.data)).toEqual([1.5, 2.25, 3.5, 4.75]);
    expect(f64(ds.target)).toEqual([0.5, 1.5]);
  });
});

describe("createRng bulk fills", () => {
  it("rejects a window that does not fit, with or without a seed", () => {
    for (const seed of [undefined, 1]) {
      const rng = createRng(seed);
      const out = new Float64Array(4);
      expect(() => rng.fillNormal(out, 2, 3)).toThrow(/window|exceeds/);
      expect(() => rng.fillUniform(out, 3, 2)).toThrow(/window|exceeds/);
      rng.fillNormal(out, 1, 3);
      rng.fillUniform(out, 0, 4);
    }
  });
});

describe("metrics against scikit-learn with tied scores and strided views", () => {
  // Reference values from scikit-learn 1.8 (roc_curve(drop_intermediate=False)).
  const y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0];
  const s = [0.5, 0.5, 0.8, 0.2, 0.5, 0.8, 0.1, 0.2, 0.2, 0.5];

  it("rocCurve, rocAucScore and averagePrecisionScore group tied scores", () => {
    const yt = tensor(y);
    const ys = tensor(s, { dtype: "float64" });
    const [fpr, tpr, th] = rocCurve(yt, ys);
    expect(f64(fpr)).toEqual([0, 0.2, 0.6, 1, 1]);
    expect(f64(tpr)).toEqual([0, 0.2, 0.6, 0.8, 1]);
    expect(f64(th)).toEqual([Number.POSITIVE_INFINITY, 0.8, 0.5, 0.2, 0.1]);
    expect(rocAucScore(yt, ys)).toBeCloseTo(0.46, 12);
    expect(averagePrecisionScore(yt, ys)).toBeCloseTo(0.4888888888888888, 12);
    const [prec, rec] = precisionRecallCurve(yt, ys);
    expect(f64(rec)).toEqual([0, 0.2, 0.6, 0.8, 1]);
    expect(f64(prec).slice(1)).toEqual([0.5, 0.5, 0.4444444444444444, 0.5]);
  });

  it("gives the same results for strided and column-vector inputs", () => {
    const strided = (logical: number[], stride: number, offset: number): Tensor => {
      const buf = new Float64Array(offset + logical.length * stride + 1).fill(-99);
      logical.forEach((v, i) => {
        buf[offset + i * stride] = v;
      });
      return Tensor.fromTypedArray({
        data: buf,
        shape: [logical.length],
        dtype: "float64",
        device: "cpu",
        offset,
        strides: [stride],
      });
    };
    const rowY = tensor(y);
    const rowS = tensor(s, { dtype: "float64" });
    const viewY = strided(y, 3, 2);
    const viewS = strided(s, 2, 1);
    const colS = tensor(
      s.map((v) => [v]),
      { dtype: "float64" }
    );
    expect(rocAucScore(viewY, viewS)).toBe(rocAucScore(rowY, rowS));
    expect(rocAucScore(rowY, colS)).toBe(rocAucScore(rowY, rowS));
    expect(averagePrecisionScore(viewY, viewS)).toBe(averagePrecisionScore(rowY, rowS));
    expect(f64(rocCurve(viewY, viewS)[0])).toEqual(f64(rocCurve(rowY, rowS)[0]));
    expect(f64(precisionRecallCurve(viewY, viewS)[0])).toEqual(
      f64(precisionRecallCurve(rowY, rowS)[0])
    );
  });
});

describe("shared metric helpers", () => {
  it("live in _internal and keep their behavior", () => {
    expect(compensatedSum([1e16, 1, -1e16])).toBe(1);
    const a = new Float64Array([0, 0]);
    const b = new Float64Array([3, 4]);
    expect(euclideanDistance(a, 0, b, 0, 2)).toBe(5);
    expect(Array.from(readFiniteFloat64(tensor(BigInt64Array.from([1n, 2n])), "x"))).toEqual([
      1, 2,
    ]);
    expect(() => readFiniteFloat64(tensor([1, Number.NaN]), "x")).toThrow(
      /x must contain only finite numbers; found NaN at index 1/
    );
  });

  it("regression metrics use the shared finite-check wording", () => {
    expect(() => mse(tensor([1, 2]), tensor([1, Number.POSITIVE_INFINITY]))).toThrow(
      /yPred must contain only finite numbers; found Infinity at index 1/
    );
  });
});
