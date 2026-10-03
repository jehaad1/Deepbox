/**
 * WASM SIMD backend tests. The embedded kernels run in any Node with WASM
 * SIMD support (the suite self-skips otherwise). Verifies the kernels are
 * bit-identical to scalar CPU arithmetic and that the ops layer routes
 * `wasm`-device tensors through them with safe CPU fallback.
 */

import { describe, expect, it } from "vitest";
import { registerBackend, WasmBackend } from "../src/core";
import { add, div, mul, sub, tensor, transpose } from "../src/ndarray";

const wasm = new WasmBackend();
await wasm.init();
if (wasm.info().available) registerBackend("wasm", wasm);

describe.skipIf(!wasm.info().available)("WASM SIMD backend", () => {
  const flat = (t: unknown): number[] => [t].flat(Infinity) as number[];

  it("binary kernels are bit-identical to scalar float32 arithmetic (incl. tails)", () => {
    for (const n of [1, 3, 4, 7, 1024, 1027]) {
      const a = new Float32Array(n).map((_, i) => (i - n / 2) * 0.37);
      const b = new Float32Array(n).map((_, i) => (i % 11) + 0.5);
      for (const op of ["add", "sub", "mul", "div"] as const) {
        const out = wasm.binaryContiguous(op, a, b);
        expect(out).not.toBeNull();
        for (let i = 0; i < n; i++) {
          const x = a[i] ?? 0;
          const y = b[i] ?? 0;
          const want = Math.fround(
            op === "add" ? x + y : op === "sub" ? x - y : op === "mul" ? x * y : x / y
          );
          expect(out?.[i]).toBe(want);
        }
      }
    }
  });

  it("dot and sum agree with sequential references within accumulation tolerance", () => {
    const n = 2049;
    const a = new Float32Array(n).map((_, i) => Math.sin(i));
    const b = new Float32Array(n).map((_, i) => Math.cos(i * 0.3));
    const dotRef = a.reduce((s, v, i) => s + v * (b[i] ?? 0), 0);
    const sumRef = a.reduce((s, v) => s + v, 0);
    expect(Math.abs((wasm.dotContiguous(a, b) ?? NaN) - dotRef)).toBeLessThan(1e-3);
    expect(Math.abs((wasm.sumContiguous(a) ?? NaN) - sumRef)).toBeLessThan(1e-3);
  });

  it("grows memory for large inputs", () => {
    const big = new Float32Array(2_000_000).fill(1.5);
    const out = wasm.binaryContiguous("add", big, big);
    expect(out?.[1_999_999]).toBe(3);
  });

  it("wasm-device tensors route eligible ops through SIMD", () => {
    const data = Array.from({ length: 1024 }, (_, i) => i + 1);
    const a = tensor(data, { device: "wasm" });
    const b = tensor(data, { device: "wasm" });
    expect(a.isDeviceTensor).toBe(false); // host storage, zero copy
    const out = add(a, b);
    expect(out.device).toBe("wasm");
    expect(flat(out.toArray()).slice(0, 3)).toEqual([2, 4, 6]);
    expect(flat(sub(a, b).toArray())[0]).toBe(0);
    expect(flat(mul(a, b).toArray())[1]).toBe(4);
    expect(flat(div(a, b).toArray())[2]).toBe(1);
  });

  it("ineligible shapes fall back to the CPU with identical results", () => {
    // Small tensors, views and broadcasts are not SIMD-eligible.
    const small = tensor([1, 2, 3], { device: "wasm" });
    expect(flat(add(small, small).toArray())).toEqual([2, 4, 6]);

    const m = tensor(
      [
        [1, 2],
        [3, 4],
      ],
      { device: "wasm" }
    );
    const tr = transpose(m);
    expect(flat(add(tr, tr).toArray())).toEqual([2, 6, 4, 8]);

    // wasm + cpu interop shares host memory and is allowed.
    const c = tensor([1, 2, 3]);
    expect(flat(add(small, c).toArray())).toEqual([2, 4, 6]);
  });

  it("dispose makes the backend unavailable without breaking CPU fallback", async () => {
    const local = new WasmBackend();
    await local.init();
    local.dispose();
    expect(local.info().available).toBe(false);
    expect(local.binaryContiguous("add", new Float32Array(4), new Float32Array(4))).toBeNull();
  });
});
