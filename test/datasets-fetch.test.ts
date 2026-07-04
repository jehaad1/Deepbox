/**
 * Offline tests for the remote dataset fetchers: a local HTTP server serves
 * synthetic MNIST (IDX + gzip), CIFAR-10 (tar + gzip) and CSV fixtures so
 * the full download → decompress → parse pipelines run without network.
 */

import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { gzipSync } from "node:zlib";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { DeepboxError, InvalidParameterError } from "../src/core";
import {
  fetchCIFAR10,
  fetchCSVDataset,
  fetchKaggleDataset,
  fetchKaggleDatasetInfo,
  fetchMNIST,
  listKaggleFiles,
  parseCSV,
  searchKaggleDatasets,
} from "../src/datasets";

// ─── Fixtures ────────────────────────────────────────────────────────────────

function u32be(value: number): number[] {
  return [(value >>> 24) & 0xff, (value >>> 16) & 0xff, (value >>> 8) & 0xff, value & 0xff];
}

/** IDX image file: magic 0x803, n images of rows×cols pixel bytes. */
function idxImages(n: number, rows: number, cols: number): Uint8Array {
  const header = [...u32be(0x00000803), ...u32be(n), ...u32be(rows), ...u32be(cols)];
  const pixels = Array.from({ length: n * rows * cols }, (_, i) => (i * 17) % 256);
  return new Uint8Array([...header, ...pixels]);
}

/** IDX label file: magic 0x801, n label bytes. */
function idxLabels(labels: number[]): Uint8Array {
  return new Uint8Array([...u32be(0x00000801), ...u32be(labels.length), ...labels]);
}

/** Minimal POSIX tar archive (regular files only). */
function makeTar(files: { name: string; data: Uint8Array }[]): Uint8Array {
  const blocks: Uint8Array[] = [];
  for (const file of files) {
    const header = new Uint8Array(512);
    const nameBytes = new TextEncoder().encode(file.name);
    header.set(nameBytes.subarray(0, 100), 0);
    const sizeOctal = file.data.length.toString(8).padStart(11, "0");
    header.set(new TextEncoder().encode(sizeOctal), 124);
    header[135] = 0x20;
    header[156] = 0x30; // typeflag '0' = regular file
    // checksum: spaces while computing
    header.fill(0x20, 148, 156);
    let sum = 0;
    for (const b of header) sum += b;
    const chk = sum.toString(8).padStart(6, "0");
    header.set(new TextEncoder().encode(chk), 148);
    header[154] = 0;
    header[155] = 0x20;
    blocks.push(header, file.data);
    const pad = (512 - (file.data.length % 512)) % 512;
    if (pad > 0) blocks.push(new Uint8Array(pad));
  }
  blocks.push(new Uint8Array(1024)); // end-of-archive
  let total = 0;
  for (const b of blocks) total += b.length;
  const out = new Uint8Array(total);
  let off = 0;
  for (const b of blocks) {
    out.set(b, off);
    off += b.length;
  }
  return out;
}

/** CIFAR-10 batch: records of [label, 3072 pixel bytes]. */
function cifarBatch(labels: number[]): Uint8Array {
  const rec = 1 + 32 * 32 * 3;
  const out = new Uint8Array(labels.length * rec);
  for (let i = 0; i < labels.length; i++) {
    out[i * rec] = labels[i]!;
    for (let p = 0; p < rec - 1; p++) out[i * rec + 1 + p] = (i + p) % 256;
  }
  return out;
}

// ─── Local server ────────────────────────────────────────────────────────────

let server: Server;
let base: string;
let lastKaggleAuth: string | undefined;

beforeAll(async () => {
  const mnistImages = gzipSync(idxImages(3, 2, 2));
  const mnistLabels = gzipSync(idxLabels([7, 1, 3]));
  const mismatchLabels = gzipSync(idxLabels([7, 1]));
  const cifar = gzipSync(
    Buffer.from(
      makeTar([
        { name: "cifar-10-batches-bin/data_batch_1.bin", data: cifarBatch([3, 5]) },
        { name: "cifar-10-batches-bin/data_batch_2.bin", data: cifarBatch([9]) },
        { name: "cifar-10-batches-bin/test_batch.bin", data: cifarBatch([1]) },
      ])
    )
  );

  server = createServer((req, res) => {
    const url = req.url ?? "";
    const send = (body: Uint8Array | string) => {
      res.writeHead(200);
      res.end(Buffer.from(body));
    };
    if (url.includes("missing")) {
      res.writeHead(404);
      return res.end("nope");
    }
    if (url.includes("train-images")) return send(mnistImages);
    if (url.includes("t10k-images")) return send(mnistImages);
    if (url.includes("train-labels")) return send(mnistLabels);
    if (url.includes("t10k-labels")) return send(mnistLabels);
    if (url.includes("mismatch-labels")) return send(mismatchLabels);
    if (url.includes("cifar-10-binary.tar.gz")) return send(cifar);
    if (url.includes("data.csv")) return send("a,b,y\n1,2,3\n4,5,6\n");
    if (url.includes("quoted.csv")) return send('name,y\n"x, ""q""",1\nplain,2\n');
    if (url.includes("big.csv")) return send(`a,y\n${"1,2\n".repeat(10_000)}`);
    if (url.includes("kaggle-api") || url.includes("kaggle-broken")) {
      lastKaggleAuth = req.headers.authorization;
      if (url.includes("kaggle-broken")) {
        res.writeHead(403);
        return res.end("forbidden");
      }
      if (url.includes("/datasets/view/")) {
        res.writeHead(200, { "content-type": "application/json" });
        return res.end(
          JSON.stringify({ ref: "owner/data", title: "Test Dataset", fileCount: 2, totalBytes: 30 })
        );
      }
      if (url.includes("/files")) {
        res.writeHead(200, { "content-type": "application/json" });
        return res.end(
          JSON.stringify({
            datasetFiles: [
              { name: "a.csv", totalBytes: 10 },
              { name: "b.csv", totalBytes: 20 },
            ],
          })
        );
      }
      if (url.includes("/datasets/download/")) {
        res.writeHead(200, { "content-type": "text/csv" });
        return res.end("x,y\n1,2\n");
      }
      if (url.includes("/datasets/list?search=")) {
        res.writeHead(200, { "content-type": "application/json" });
        return res.end(JSON.stringify([{ ref: "owner/data", title: "Test Dataset" }]));
      }
      res.writeHead(404);
      return res.end();
    }
    if (url.includes("slow")) return; // never respond -> timeout
    res.writeHead(500);
    return res.end();
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}/`;
});

afterAll(() => {
  server.closeAllConnections?.();
  server.close();
});

// ─── MNIST ───────────────────────────────────────────────────────────────────

describe("fetchMNIST", () => {
  it("downloads, gunzips and parses the IDX format", async () => {
    const ds = await fetchMNIST({ baseUrl: base });
    expect(ds.data.shape).toEqual([3, 4]);
    expect(ds.target.toArray()).toEqual([7, 1, 3]);
    expect(ds.nClasses).toBe(10);
    expect(ds.imageDims).toEqual([2, 2, 1]);
    // pixels normalized to [0, 1]
    const first = ds.data.toArray() as number[][];
    expect(first[0]?.[1]).toBeCloseTo(17 / 255, 6); // stored as float32
  });

  it("supports maxSamples and the test split", async () => {
    const ds = await fetchMNIST({ baseUrl: base, maxSamples: 2, split: "test" });
    expect(ds.data.shape).toEqual([2, 4]);
    expect(ds.target.toArray()).toEqual([7, 1]);
  });

  it("throws on HTTP errors and unreachable hosts", async () => {
    await expect(fetchMNIST({ baseUrl: `${base}missing/` })).rejects.toThrow(DeepboxError);
    await expect(fetchMNIST({ baseUrl: "http://127.0.0.1:1/" })).rejects.toThrow(DeepboxError);
  });
});

// ─── CIFAR-10 ────────────────────────────────────────────────────────────────

describe("fetchCIFAR10", () => {
  it("downloads and extracts train batches from the tar", async () => {
    const ds = await fetchCIFAR10({ baseUrl: base });
    expect(ds.data.shape).toEqual([3, 3072]);
    expect(ds.target.toArray()).toEqual([3, 5, 9]);
    expect(ds.classNames).toHaveLength(10);
    expect(ds.imageDims).toEqual([32, 32, 3]);
  });

  it("reads the test split and honors maxSamples", async () => {
    const test = await fetchCIFAR10({ baseUrl: base, split: "test" });
    expect(test.target.toArray()).toEqual([1]);
    const limited = await fetchCIFAR10({ baseUrl: base, maxSamples: 1 });
    expect(limited.data.shape).toEqual([1, 3072]);
  });

  it("throws on HTTP errors", async () => {
    await expect(fetchCIFAR10({ baseUrl: `${base}missing/` })).rejects.toThrow(DeepboxError);
  });
});

// ─── CSV ─────────────────────────────────────────────────────────────────────

describe("fetchCSVDataset", () => {
  it("fetches and parses a CSV with header", async () => {
    const ds = await fetchCSVDataset({ url: `${base}data.csv` });
    expect(ds.data.shape).toEqual([2, 2]);
    expect(ds.target.toArray()).toEqual([3, 6]);
    expect(ds.featureNames).toEqual(["a", "b"]);
  });

  it("supports targetColumn selection", async () => {
    const ds = await fetchCSVDataset({ url: `${base}data.csv`, targetColumn: 0 });
    expect(ds.target.toArray()).toEqual([1, 4]);
  });

  it("rejects oversized responses via maxBytes", async () => {
    await expect(fetchCSVDataset({ url: `${base}big.csv`, maxBytes: 100 })).rejects.toThrow(
      DeepboxError
    );
  });

  it("times out slow servers", async () => {
    await expect(fetchCSVDataset({ url: `${base}slow.csv`, timeout: 150 })).rejects.toThrow(
      /Timed out|aborted|Failed to fetch/
    );
  });

  it("throws on HTTP errors and validates inputs", async () => {
    await expect(fetchCSVDataset({ url: `${base}missing.csv` })).rejects.toThrow(/HTTP 404/);
    await expect(fetchCSVDataset({ url: "" })).rejects.toThrow(InvalidParameterError);
    await expect(fetchCSVDataset({ url: `${base}data.csv`, maxBytes: -5 })).rejects.toThrow(
      InvalidParameterError
    );
  });

  it("honors an abort signal", async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(
      fetchCSVDataset({ url: `${base}data.csv`, abortSignal: controller.signal })
    ).rejects.toThrow(DeepboxError);
  });
});

describe("parseCSV", () => {
  it("parses quoted fields and rejects non-numeric values loudly", () => {
    const ds = parseCSV('"1","2"\n"3","4"\n', { header: false });
    expect(ds.data.shape).toEqual([2, 1]);
    expect(ds.target.toArray()).toEqual([2, 4]);
    // Only numeric CSV datasets are supported; strings throw instead of NaN.
    expect(() => parseCSV('name,y\n"x, ""q""",1\n', {})).toThrow(/Non-numeric value/);
  });

  it("supports header:false and custom separators", () => {
    const ds = parseCSV("1;2\n3;4\n", { header: false, separator: ";" });
    expect(ds.data.shape).toEqual([2, 1]);
    expect(ds.target.toArray()).toEqual([2, 4]);
  });
});

// ─── Kaggle (offline validation paths) ───────────────────────────────────────

describe("kaggle validation", () => {
  const credentials = { username: "u", key: "k" };

  it("rejects malformed dataset ids before any network call", async () => {
    await expect(fetchKaggleDataset("no-slash", { credentials })).rejects.toThrow(
      InvalidParameterError
    );
    await expect(fetchKaggleDataset("bad id/with spaces", { credentials })).rejects.toThrow(
      InvalidParameterError
    );
    await expect(fetchKaggleDatasetInfo("", { credentials })).rejects.toThrow(
      InvalidParameterError
    );
  });
});

// ─── Kaggle API against the local server (apiBaseUrl override) ───────────────

describe("kaggle client with apiBaseUrl override", () => {
  const credentials = { username: "user", key: "secret" };

  it("fetches dataset info, files, downloads, and search results", async () => {
    const opts = { credentials, apiBaseUrl: `${base}kaggle-api` };
    const info = await fetchKaggleDatasetInfo("owner/data", opts);
    expect(info.title).toBe("Test Dataset");
    expect(info.fileCount).toBe(2);

    const files = await listKaggleFiles("owner/data", opts);
    expect(files).toEqual([
      { name: "a.csv", totalBytes: 10 },
      { name: "b.csv", totalBytes: 20 },
    ]);

    const dl = await fetchKaggleDataset("owner/data", { ...opts, file: "a.csv" });
    expect(dl.filename).toBe("a.csv");
    expect(new TextDecoder().decode(dl.data)).toBe("x,y\n1,2\n");

    const whole = await fetchKaggleDataset("owner/data", opts);
    expect(whole.filename).toBe("owner_data.zip");

    const truncated = await fetchKaggleDataset("owner/data", {
      ...opts,
      file: "a.csv",
      maxBytes: 3,
    });
    expect(truncated.totalBytes).toBe(3);
    expect(new TextDecoder().decode(truncated.data)).toBe("x,y");

    const found = await searchKaggleDatasets("test", opts);
    expect(found).toHaveLength(1);
    expect(found[0]?.title).toBe("Test Dataset");
  });

  it("propagates HTTP errors and sends Basic auth", async () => {
    const opts = { credentials, apiBaseUrl: `${base}kaggle-broken` };
    await expect(fetchKaggleDatasetInfo("owner/data", opts)).rejects.toThrow(/Kaggle API error/);
    await expect(fetchKaggleDataset("owner/data", opts)).rejects.toThrow(/Kaggle download error/);
    await expect(listKaggleFiles("owner/data", opts)).rejects.toThrow(/Kaggle API error/);
    // The auth header carries base64(username:key); the server asserts it.
    expect(lastKaggleAuth).toBe(`Basic ${Buffer.from("user:secret").toString("base64")}`);
  });
});
