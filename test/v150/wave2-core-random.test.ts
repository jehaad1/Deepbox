/**
 * Wave 2 regression tests for the core-random group: WebGPU backend typing and
 * runtime independence from ambient globals, and the random `choice` path that
 * once failed with a ReferenceError.
 */

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { WebGpuBackend } from "../../src/core/backend/WebGpuBackend";
import { GPU_BUFFER_USAGE, GPU_MAP_MODE } from "../../src/core/backend/webgpu_types";
import { tensor } from "../../src/ndarray";
import { choice, permutation, randint, setSeed } from "../../src/random";

type GpuProvider = ConstructorParameters<typeof WebGpuBackend>[0] extends
  | { gpu?: infer G }
  | undefined
  ? G
  : never;

async function loadDawn(): Promise<{ gpu: GpuProvider; globals: Record<string, unknown> } | null> {
  try {
    const dawn = await import("webgpu");
    return {
      gpu: dawn.create([]) as unknown as GpuProvider,
      globals: dawn.globals as unknown as Record<string, unknown>,
    };
  } catch {
    return null;
  }
}

const dawn = await loadDawn();

describe("WebGPU constants and shipped types", () => {
  it("buffer usage and map mode constants follow the WebGPU specification", () => {
    expect(GPU_BUFFER_USAGE).toEqual({
      MAP_READ: 1,
      MAP_WRITE: 2,
      COPY_SRC: 4,
      COPY_DST: 8,
      INDEX: 16,
      VERTEX: 32,
      UNIFORM: 64,
      STORAGE: 128,
      INDIRECT: 256,
      QUERY_RESOLVE: 512,
    });
    expect(GPU_MAP_MODE).toEqual({ READ: 1, WRITE: 2 });
  });

  it("matches the constants of a real WebGPU implementation when one is installed", () => {
    if (!dawn) return;
    const usage = dawn.globals["GPUBufferUsage"] as Record<string, number>;
    const mode = dawn.globals["GPUMapMode"] as Record<string, number>;
    for (const [key, value] of Object.entries(GPU_BUFFER_USAGE)) expect(usage[key]).toBe(value);
    for (const [key, value] of Object.entries(GPU_MAP_MODE)) expect(mode[key]).toBe(value);
  });

  it("WebGpuBackend.ts does not reference ambient GPU* globals in code", () => {
    const source = readFileSync(
      new URL("../../src/core/backend/WebGpuBackend.ts", import.meta.url),
      "utf8"
    );
    // Strip block comments, line comments and string/template literals.
    const code = source
      .replace(/\/\*[\s\S]*?\*\//g, "")
      .replace(/\/\/[^\n]*/g, "")
      .replace(/`(?:\\[\s\S]|[^`\\])*`/g, "``")
      .replace(/"(?:\\.|[^"\\])*"/g, '""');
    expect(code.match(/\bGPU[A-Za-z]*\b/g) ?? []).toEqual([]);
  });
});

describe.skipIf(!dawn)("WebGpuBackend without ambient WebGPU globals", () => {
  it("runs a buffer round trip when GPUBufferUsage and GPUMapMode are not defined", async () => {
    const g = globalThis as Record<string, unknown>;
    expect(g["GPUBufferUsage"]).toBeUndefined();
    expect(g["GPUMapMode"]).toBeUndefined();
    const backend = new WebGpuBackend({ gpu: (dawn as { gpu: GpuProvider }).gpu });
    await backend.init();
    if (!backend.info().available) return;
    try {
      const buf = backend.upload(new Float32Array([1, 2, 3, 4]));
      const out = await backend.download(buf);
      expect(Array.from(out)).toEqual([1, 2, 3, 4]);
      backend.free(buf);
    } finally {
      backend.dispose();
    }
  });
});

describe("random choice (validateContiguous regression)", () => {
  it("samples from a range and from a tensor, with and without replacement", () => {
    setSeed(7);
    const a = choice(10, 4, false);
    expect(a.size).toBe(4);
    expect(new Set(Array.from(a.data as Int32Array | Float64Array)).size).toBe(4);

    const pool = tensor([5, 6, 7, 8]);
    const b = choice(pool, [2, 3]);
    expect(b.shape).toEqual([2, 3]);
    for (const v of Array.from(b.data as Float64Array | Float32Array)) {
      expect([5, 6, 7, 8]).toContain(v);
    }
  });

  it("honors probabilities and stays reproducible for a fixed seed", () => {
    setSeed(11);
    const first = Array.from(choice(5, 6, true, tensor([0, 0, 1, 0, 0])).data as Float64Array);
    expect(first).toEqual([2, 2, 2, 2, 2, 2]);
    setSeed(3);
    const x = Array.from(choice(100, 5).data as Float64Array);
    setSeed(3);
    const y = Array.from(choice(100, 5).data as Float64Array);
    expect(x).toEqual(y);
  });

  it("permutation and randint still work next to choice", () => {
    setSeed(1);
    expect(permutation(6).size).toBe(6);
    expect(randint(0, 5, [3]).size).toBe(3);
  });
});
