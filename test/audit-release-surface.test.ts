import { afterEach, describe, expect, it, vi } from "vitest";
import { availableCores, createWorkerPool, WorkerPool } from "../src/core/parallel";
import { DataFrame } from "../src/dataframe";
import { fetchCIFAR10, fetchMNIST } from "../src/datasets/image";
import {
  fetchKaggleDataset,
  fetchKaggleDatasetInfo,
  listKaggleFiles,
  searchKaggleDatasets,
} from "../src/datasets/kaggle";
import { fetch20Newsgroups, fetchIMDB } from "../src/datasets/text";
import { tensor } from "../src/ndarray";
import {
  packPaddedSequence,
  packSequence,
  padPackedSequence,
  unpackSequence,
} from "../src/nn/layers/packed_sequence";
import { Animation, createAnimation } from "../src/plot/animation/Animation";
import { figure } from "../src/plot/figure/state";

describe("AUDIT release surface: WorkerPool", () => {
  it("exec runs inline and updates status", async () => {
    const pool = new WorkerPool({ maxWorkers: 2 });
    const r = await pool.exec((x: number) => x * 2, 21);
    expect(r.value).toBe(42);
    expect(r.workerId).toBe(0);
    const s = pool.status();
    expect(s.completedTasks).toBeGreaterThanOrEqual(1);
    expect(s.isTerminated).toBe(false);
    pool.terminate();
  });

  it("map uses chunked path when maxWorkers > 1 and items exceed workers", async () => {
    const pool = new WorkerPool({ maxWorkers: 3 });
    const items = [1, 2, 3, 4, 5, 6, 7, 8];
    const out = await pool.map(items, (x) => x + 1);
    expect(out).toEqual([2, 3, 4, 5, 6, 7, 8, 9]);
    pool.terminate();
  });

  it("map returns [] for empty input", async () => {
    const pool = new WorkerPool({ maxWorkers: 2 });
    expect(await pool.map([], (x: number) => x)).toEqual([]);
    pool.terminate();
  });

  it("reduce combines chunks", async () => {
    const pool = new WorkerPool({ maxWorkers: 2 });
    const sum = await pool.reduce([1, 2, 3, 4], (a, b) => a + b, 0);
    expect(sum).toBe(10);
    const empty = await pool.reduce([] as number[], (a, b) => a + b, 99);
    expect(empty).toBe(99);
    pool.terminate();
  });

  it("all and forEach and filter run", async () => {
    const pool = new WorkerPool({ maxWorkers: 2 });
    const sums = await pool.all([() => 1, () => 2]);
    expect(sums).toEqual([1, 2]);
    const acc: number[] = [];
    await pool.forEach([10, 20], (v, i) => {
      acc.push(v + i);
    });
    expect(acc).toEqual([10, 21]);
    const evens = await pool.filter([1, 2, 3, 4], (n) => n % 2 === 0);
    expect(evens).toEqual([2, 4]);
    pool.terminate();
  });

  it("throws after terminate", async () => {
    const pool = new WorkerPool({ maxWorkers: 1 });
    pool.terminate();
    expect(pool.isTerminated).toBe(true);
    await expect(pool.exec((x: number) => x, 1)).rejects.toThrow(/terminated/);
  });

  it("createWorkerPool and availableCores are defined", () => {
    const p = createWorkerPool(2);
    expect(p.status().maxWorkers).toBe(2);
    p.terminate();
    expect(availableCores()).toBeGreaterThanOrEqual(1);
  });
});

describe("AUDIT release surface: DataFrame plot & style", () => {
  const df = new DataFrame({
    x: [1, 2, 3],
    y: [4, 5, 6],
    z: [7, 8, 9],
    cat: ["a", "b", "c"],
    v: [0.1, 0.5, 0.9],
  });

  it("plot.line, bar, scatter, hist, area produce figures", () => {
    expect(df.plot.line({ x: "x", y: "y" }).renderSVG().svg).toContain("<svg");
    expect(df.plot.bar({ x: "x", y: "y" }).renderSVG().svg).toContain("<svg");
    expect(df.plot.scatter({ x: "x", y: "y" }).renderSVG().svg).toContain("<svg");
    expect(df.plot.hist({ column: "y" }).renderSVG().svg).toContain("<svg");
    expect(df.plot.area({ x: "x", y: "y" }).renderSVG().svg).toContain("<svg");
  });

  it("plot.barh and box and pie", () => {
    const d2 = new DataFrame({ a: [1, 2], b: [3, 4] });
    expect(d2.plot.barh({ x: "b", y: "a" }).renderSVG().svg).toContain("<svg");
    expect(d2.plot.box().renderSVG().svg).toContain("<svg");
    const d3 = new DataFrame({ y: [1, 2, 3], lab: ["x", "y", "z"] });
    expect(d3.plot.pie({ y: "y", labels: "lab" }).renderSVG().svg).toContain("<svg");
  });

  it("style chaining and renders", () => {
    const html = df.style
      .setCaption("t")
      .highlight_max()
      .highlight_min()
      .highlight_null()
      .background_gradient()
      .bar()
      .format("x", (v) => `n=${v}`)
      .applymap(() => ({ fontWeight: "bold" }))
      .toHTML();
    expect(html).toContain("<table>");
    expect(html).toContain("n=1");
    const ansi = df.style.toANSI();
    expect(ansi).toContain("x");
  });
});

describe("AUDIT release surface: text datasets", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("fetch20Newsgroups throws when fetch fails by default", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 404, statusText: "Not Found" })
    );
    await expect(fetch20Newsgroups({ maxSamples: 24 })).rejects.toThrow(
      /Failed to fetch 20 Newsgroups/
    );
  });

  it("fetch20Newsgroups uses synthetic data with allowSyntheticFallback", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 404, statusText: "Not Found" })
    );
    const d = await fetch20Newsgroups({ maxSamples: 24, allowSyntheticFallback: true });
    expect(d.texts.length).toBeGreaterThan(0);
    expect(d.nClasses).toBe(20);
    expect(d.description).toContain("fallback");
    expect(d.isSynthetic).toBe(true);
  });

  it("fetchIMDB throws when fetch fails by default", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 404, statusText: "Not Found" })
    );
    await expect(fetchIMDB({ maxSamples: 10 })).rejects.toThrow(/Failed to fetch IMDB/);
  });

  it("fetchIMDB uses synthetic data with allowSyntheticFallback", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 404, statusText: "Not Found" })
    );
    const d = await fetchIMDB({ maxSamples: 10, allowSyntheticFallback: true });
    expect(d.texts.length).toBeGreaterThan(0);
    expect(d.nClasses).toBe(2);
    expect(d.description).toContain("fallback");
    expect(d.isSynthetic).toBe(true);
  });
});

describe("AUDIT release surface: packed_sequence", () => {
  it("packs, unpacks, and round-trips 2D sequences", () => {
    const seqs = [
      tensor([
        [1, 2],
        [3, 4],
        [5, 6],
      ]),
      tensor([[7, 8]]),
      tensor([
        [9, 10],
        [11, 12],
      ]),
    ];
    const packed = packSequence(seqs);
    expect(packed.batchSizes).toEqual([3, 2, 1]);
    const [unpacked, lengths] = unpackSequence(packed);
    expect(lengths).toEqual([3, 1, 2]);
    expect(unpacked[0]!.shape).toEqual([3, 2]);
    expect(unpacked[1]!.shape).toEqual([1, 2]);
  });

  it("supports 1D sequences and enforcesSorted", () => {
    const seqs = [tensor([1, 2, 3]), tensor([4])];
    const packed = packSequence(seqs, true);
    const [back] = unpackSequence(packed);
    expect(
      Array.from((back[0]!.data as Float32Array).slice(back[0]!.offset, back[0]!.offset + 3))
    ).toEqual([1, 2, 3]);
  });

  it("padPackedSequence and packPaddedSequence", () => {
    const seqs = [
      tensor([
        [1, 2],
        [3, 4],
      ]),
      tensor([[5, 6]]),
    ];
    const packed = packSequence(seqs);
    const [padded, lens] = padPackedSequence(packed, 3);
    expect(padded.shape).toEqual([2, 3, 2]);
    expect(lens).toEqual([2, 1]);
    const again = packPaddedSequence(padded, lens);
    expect(again.batchSizes).toEqual(packed.batchSizes);
  });

  it("rejects invalid inputs", () => {
    expect(() => packSequence([])).toThrow(/at least one/);
    expect(() => packSequence([tensor([[[1, 2]]])])).toThrow(/1D or 2D/);
    expect(() => packSequence([tensor([[1, 2]]), tensor([[3, 4, 5]])])).toThrow(/same feature/);
    expect(() => packSequence([tensor([])])).toThrow(/zero-length/);
    expect(() => packPaddedSequence(tensor([1, 2, 3]), [1])).toThrow(/3D/);
    expect(() => packPaddedSequence(tensor([[[1, 2]]]).reshape([1, 1, 2]), [])).toThrow(/lengths/);
    expect(() =>
      packPaddedSequence(
        tensor([
          [
            [1, 2],
            [3, 4],
          ],
        ]).reshape([1, 2, 2]),
        [0]
      )
    ).toThrow(/out of range/);
  });
});

describe("AUDIT release surface: plot Animation", () => {
  it("validates options and renders frames", () => {
    expect(() => new Animation({ fps: 0 })).toThrow(/FPS/);
    expect(() => new Animation({ duration: 0 })).toThrow(/Duration/);
    const anim = new Animation({ fps: 10, duration: 200, loop: false, easing: "ease-in" });
    expect(() => anim.render()).toThrow(/animate\(\)/);
    anim.animate((_i, n) => {
      const fig = figure({ width: 120, height: 80 });
      const ax = fig.addAxes();
      const t = _i / Math.max(1, n - 1);
      ax.plot(tensor([0, 1]), tensor([0, t]));
      return fig;
    });
    const res = anim.render();
    expect(res.frameCount).toBeGreaterThan(0);
    expect(res.frames.length).toBe(res.frameCount);
    expect(anim.getFrame(0)?.svg.svg).toContain("<svg");
    const svgAnim = anim.toAnimatedSVG();
    expect(svgAnim).toContain("<svg");
    expect(anim.toFrames().length).toBe(res.frameCount);
    expect(createAnimation({ fps: 5, duration: 100 }).info().totalFrames).toBeGreaterThan(0);
  });
});

describe("AUDIT release surface: image datasets (mocked fetch)", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("fetchMNIST throws on failed image response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 500, statusText: "err" })
    );
    await expect(fetchMNIST({ baseUrl: "https://x/" })).rejects.toThrow(/MNIST/);
  });

  it("fetchCIFAR10 throws on failed response", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 404, statusText: "nf" }));
    await expect(fetchCIFAR10({ baseUrl: "https://x/" })).rejects.toThrow(/CIFAR-10/);
  });
});

describe("AUDIT release surface: Kaggle (mocked fetch)", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  const creds = { username: "u", key: "k" };

  it("requires credentials", async () => {
    await expect(fetchKaggleDatasetInfo("a/b")).rejects.toThrow(/credentials/);
  });

  it("rejects invalid dataset id", async () => {
    await expect(fetchKaggleDatasetInfo("nope", { credentials: creds })).rejects.toThrow(
      /owner\/dataset/
    );
  });

  it("fetchKaggleDatasetInfo parses JSON", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          ref: "o/d",
          title: "T",
          subtitle: "S",
          totalBytes: 100,
          fileCount: 2,
          lastUpdated: "now",
          downloadCount: 1,
          usabilityRating: 5,
        }),
      })
    );
    const info = await fetchKaggleDatasetInfo("owner/name", { credentials: creds });
    expect(info.title).toBe("T");
    expect(info.totalBytes).toBe(100);
  });

  it("fetchKaggleDataset and maxBytes branch", async () => {
    const buf = new Uint8Array([1, 2, 3, 4]);
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
        headers: { get: () => "application/zip" },
      })
    );
    const full = await fetchKaggleDataset("o/n", { credentials: creds });
    expect(full.totalBytes).toBe(4);
    const part = await fetchKaggleDataset("o/n", { credentials: creds, maxBytes: 2 });
    expect(part.totalBytes).toBe(2);
    expect(part.data.length).toBe(2);
  });

  it("listKaggleFiles and searchKaggleDatasets", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            datasetFiles: [{ name: "a.csv", totalBytes: 10 }],
          }),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => [
            {
              ref: "x/y",
              title: "Q",
              subtitle: "",
              totalBytes: 0,
              fileCount: 0,
              lastUpdated: "",
              downloadCount: 0,
              usabilityRating: 0,
            },
          ],
        })
    );
    const files = await listKaggleFiles("o/n", { credentials: creds });
    expect(files[0]!.name).toBe("a.csv");
    const found = await searchKaggleDatasets("q", { credentials: creds });
    expect(found[0]!.id).toBe("x/y");
  });
});
