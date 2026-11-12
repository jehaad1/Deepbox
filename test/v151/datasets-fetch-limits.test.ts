import { createHash } from "node:crypto";
import { gzipSync } from "node:zlib";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  DataValidationError,
  DeepboxError,
  InvalidParameterError,
  MemoryError,
} from "../../src/core";
import { fetchCIFAR10, fetchIMDB, fetchMNIST } from "../../src/datasets";
import { decompressGzip } from "../../src/datasets/_archive";

// Every fetch is mocked: no test here touches the network.

function u32be(value: number): number[] {
  return [(value >>> 24) & 0xff, (value >>> 16) & 0xff, (value >>> 8) & 0xff, value & 0xff];
}

function idxImages(n: number, rows: number, cols: number): Uint8Array {
  const header = [...u32be(0x803), ...u32be(n), ...u32be(rows), ...u32be(cols)];
  return new Uint8Array([...header, ...new Array<number>(n * rows * cols).fill(7)]);
}

function idxLabels(labels: number[]): Uint8Array {
  return new Uint8Array([...u32be(0x801), ...u32be(labels.length), ...labels]);
}

function sha256(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}

/** Serve `files` by path suffix from a mocked fetch. */
function mockFetch(files: Record<string, Uint8Array>): ReturnType<typeof vi.fn> {
  const fn = vi.fn(async (input: string | URL | Request) => {
    const url = String(input);
    const key = Object.keys(files).find((k) => url.endsWith(k));
    if (key === undefined) return new Response("missing", { status: 404, statusText: "Not Found" });
    return new Response(files[key] as Uint8Array<ArrayBuffer>, { status: 200 });
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

const BASE = "https://mirror.test/mnist/";

function mnistFiles(labels: number[]): Record<string, Uint8Array> {
  return {
    "train-images-idx3-ubyte.gz": gzipSync(idxImages(labels.length, 2, 2)),
    "train-labels-idx1-ubyte.gz": gzipSync(idxLabels(labels)),
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("DF-16 fetchMNIST validates labels", () => {
  it("rejects labels above 9", async () => {
    mockFetch(mnistFiles([0, 1, 10]));
    await expect(fetchMNIST({ baseUrl: BASE })).rejects.toThrow(DataValidationError);
    await expect(fetchMNIST({ baseUrl: BASE })).rejects.toThrow(/label 10 at index 2/);
  });

  it("accepts labels 0 to 9", async () => {
    mockFetch(mnistFiles([0, 9, 5]));
    const ds = await fetchMNIST({ baseUrl: BASE });
    expect(Array.from(ds.target.data as Int32Array)).toEqual([0, 9, 5]);
  });
});

describe("DF-15 bounded archive downloads", () => {
  it("decompressGzip stops at the output cap", async () => {
    const bomb = gzipSync(new Uint8Array(8 * 1024 * 1024));
    expect(bomb.length).toBeLessThan(100_000);
    await expect(decompressGzip(bomb, 1024 * 1024)).rejects.toThrow(MemoryError);
    const ok = await decompressGzip(bomb, 8 * 1024 * 1024);
    expect(ok.length).toBe(8 * 1024 * 1024);
  });

  it("fetchMNIST rejects an expanding file with maxDecompressedBytes", async () => {
    mockFetch(mnistFiles([0, 1, 2]));
    await expect(fetchMNIST({ baseUrl: BASE, maxDecompressedBytes: 20 })).rejects.toThrow(
      MemoryError
    );
  });

  it("fetchMNIST rejects an oversized download with maxBytes", async () => {
    mockFetch(mnistFiles([0, 1, 2]));
    await expect(fetchMNIST({ baseUrl: BASE, maxBytes: 10 })).rejects.toThrow(/maxBytes limit 10/);
  });

  it("fetchCIFAR10 rejects an oversized download with maxBytes", async () => {
    mockFetch({ "cifar-10-binary.tar.gz": gzipSync(new Uint8Array(4096)) });
    await expect(fetchCIFAR10({ baseUrl: "https://m.test/", maxBytes: 16 })).rejects.toThrow(
      /maxBytes limit 16/
    );
  });

  it("fetchIMDB rejects an oversized archive with maxBytes", async () => {
    mockFetch({ "a.tar.gz": gzipSync(new Uint8Array(4096)) });
    await expect(
      fetchIMDB({ archiveUrl: "https://m.test/a.tar.gz", maxBytes: 16 })
    ).rejects.toThrow(/maxBytes limit 16/);
  });

  it("fetchIMDB rejects a gzip bomb with maxDecompressedBytes", async () => {
    mockFetch({ "a.tar.gz": gzipSync(new Uint8Array(4 * 1024 * 1024)) });
    await expect(
      fetchIMDB({ archiveUrl: "https://m.test/a.tar.gz", maxDecompressedBytes: 1024 })
    ).rejects.toThrow(/Decompressed data exceeds/);
  });

  it("times out a stalled download", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(
        (_url: string, init?: RequestInit) =>
          new Promise<Response>((_resolve, reject) => {
            init?.signal?.addEventListener("abort", () => reject(init.signal?.reason));
          })
      )
    );
    await expect(fetchMNIST({ baseUrl: BASE, timeout: 20 })).rejects.toThrow(/Timed out/);
    await expect(fetchCIFAR10({ baseUrl: BASE, timeout: 20 })).rejects.toThrow(/Timed out/);
  });

  it("validates the new options", async () => {
    for (const bad of [{ timeout: -1 }, { maxBytes: 0 }, { maxDecompressedBytes: 1.5 }]) {
      await expect(fetchMNIST({ baseUrl: BASE, ...bad })).rejects.toThrow(InvalidParameterError);
      await expect(fetchIMDB({ ...bad })).rejects.toThrow(InvalidParameterError);
    }
    await expect(fetchCIFAR10({ checksum: "abc" })).rejects.toThrow(InvalidParameterError);
    await expect(fetchIMDB({ checksum: "abc" })).rejects.toThrow(InvalidParameterError);
  });
});

describe("DF-15 optional SHA-256 checksum", () => {
  it("accepts a matching MNIST digest pair and rejects a wrong one", async () => {
    const files = mnistFiles([0, 1, 2]);
    mockFetch(files);
    const images = sha256(files["train-images-idx3-ubyte.gz"] as Uint8Array);
    const labels = sha256(files["train-labels-idx1-ubyte.gz"] as Uint8Array);
    const ds = await fetchMNIST({ baseUrl: BASE, checksum: { images, labels } });
    expect(ds.data.shape).toEqual([3, 4]);
    await expect(
      fetchMNIST({
        baseUrl: BASE,
        checksum: { images: images.toUpperCase(), labels: "0".repeat(64) },
      })
    ).rejects.toThrow(DataValidationError);
  });

  it("verifies the CIFAR-10 archive digest before decompressing", async () => {
    const archive = gzipSync(new Uint8Array(1024));
    mockFetch({ "cifar-10-binary.tar.gz": archive });
    await expect(
      fetchCIFAR10({ baseUrl: "https://m.test/", checksum: "f".repeat(64) })
    ).rejects.toThrow(/checksum mismatch/);
    // A matching digest passes the check and fails later on the empty archive.
    await expect(
      fetchCIFAR10({ baseUrl: "https://m.test/", checksum: sha256(archive) })
    ).rejects.toThrow(/missing data_batch_1\.bin/);
  });

  it("verifies the text archive digest", async () => {
    mockFetch({ "a.tar.gz": gzipSync(new Uint8Array(1024)) });
    await expect(
      fetchIMDB({ archiveUrl: "https://m.test/a.tar.gz", checksum: "0".repeat(64) })
    ).rejects.toThrow(/checksum mismatch/);
  });

  it("rejects a checksum of the wrong shape", async () => {
    await expect(fetchMNIST({ baseUrl: BASE, checksum: "a".repeat(64) })).rejects.toThrow(
      InvalidParameterError
    );
    await expect(
      fetchCIFAR10({ checksum: { images: "a".repeat(64), labels: "a".repeat(64) } })
    ).rejects.toThrow(InvalidParameterError);
  });
});

describe("errors keep their Deepbox type", () => {
  it("download failures are DeepboxError", async () => {
    mockFetch({});
    await expect(fetchMNIST({ baseUrl: BASE })).rejects.toThrow(DeepboxError);
  });
});
