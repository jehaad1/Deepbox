/**
 * v1.5.0 regression tests for src/core/backend (registry, CPU/WASM/WebGPU backends).
 *
 * The WebGPU suites run the real WGSL kernels through the `webgpu` (Dawn) dev
 * dependency and skip themselves when no adapter is available. Reference values
 * come from NumPy/SciPy in float64.
 */

import { afterAll, describe, expect, it, vi } from "vitest";
import {
  type BinaryKernelOp,
  CpuBackend,
  type DeviceBuffer,
  DeviceError,
  getBackend,
  InvalidParameterError,
  isBackendAvailable,
  type KernelLayout,
  registerBackend,
  type UnaryKernelOp,
  unregisterBackend,
  WasmBackend,
  WebGpuBackend,
} from "../../src/core";

// ─── Registry and CPU backend ────────────────────────────────────────────────

describe("registry and CpuBackend", () => {
  it("CpuBackend stays available after dispose()", () => {
    const cpu = new CpuBackend();
    cpu.dispose();
    expect(cpu.isDisposed).toBe(true);
    expect(cpu.info().available).toBe(true);
    // The registered CPU backend must never become unusable.
    getBackend("cpu").dispose();
    expect(isBackendAvailable("cpu")).toBe(true);
  });

  it("registerBackend rejects unknown devices and non-backends", () => {
    expect(() => registerBackend("tpu" as never, new CpuBackend())).toThrow(InvalidParameterError);
    expect(() => registerBackend("wasm", {} as never)).toThrow(InvalidParameterError);
    expect(() => registerBackend("wasm", null as never)).toThrow(InvalidParameterError);
    expect(isBackendAvailable("wasm")).toBe(false);
  });

  it("getBackend names the registered backends in its error", () => {
    unregisterBackend("webgpu");
    expect(() => getBackend("webgpu")).toThrow(/Registered backends: .*cpu/);
  });
});

// ─── WASM backend ────────────────────────────────────────────────────────────

describe("WasmBackend", () => {
  const probe = new WasmBackend();
  const simd = probe.isSimdSupported();

  it("dispose() during init() leaves no state behind", async () => {
    const wasm = new WasmBackend();
    const pending = wasm.init();
    wasm.dispose();
    await pending;
    expect(wasm.isDisposed).toBe(true);
    expect(wasm.isInitialized).toBe(false);
    expect(wasm.info().available).toBe(false);
    expect(wasm.hasModule("simdAdd")).toBe(false);
    expect(wasm.binaryContiguous("add", new Float32Array(4), new Float32Array(4))).toBeNull();
  });

  it("a disposed backend cannot be initialized again", async () => {
    const wasm = new WasmBackend();
    wasm.dispose();
    await wasm.init();
    expect(wasm.info().available).toBe(false);
  });

  it.skipIf(!simd)("returns null instead of throwing when memory cannot grow", async () => {
    const wasm = new WasmBackend();
    await wasm.init();
    const grow = vi.spyOn(WebAssembly.Memory.prototype, "grow").mockImplementation(() => {
      throw new RangeError("Out of memory");
    });
    try {
      // 3 * 400k floats needs ~4.8 MB, more than the 1 MiB initial memory.
      const a = new Float32Array(400_000).fill(1);
      expect(wasm.binaryContiguous("add", a, a)).toBeNull();
      expect(wasm.dotContiguous(a, a)).toBeNull();
      expect(wasm.sumContiguous(new Float32Array(400_000))).toBeNull();
    } finally {
      grow.mockRestore();
    }
    // The backend is still usable for inputs that fit.
    const small = new Float32Array([1, 2, 3, 4, 5]);
    expect(Array.from(wasm.binaryContiguous("add", small, small) ?? [])).toEqual([2, 4, 6, 8, 10]);
    wasm.dispose();
  });

  it.skipIf(!simd)("handles empty inputs", async () => {
    const wasm = new WasmBackend();
    await wasm.init();
    expect(wasm.binaryContiguous("mul", new Float32Array(0), new Float32Array(0))?.length).toBe(0);
    expect(wasm.sumContiguous(new Float32Array(0))).toBe(0);
    expect(wasm.dotContiguous(new Float32Array(0), new Float32Array(0))).toBe(0);
    wasm.dispose();
  });
});

// ─── WebGPU backend (real GPU through Dawn) ──────────────────────────────────

type Gpu = ConstructorParameters<typeof WebGpuBackend>[0] extends { gpu?: infer G } | undefined
  ? G
  : never;

async function loadDawn(): Promise<(() => Gpu) | null> {
  try {
    const dawn = await import("webgpu");
    Object.assign(globalThis, dawn.globals);
    return () => dawn.create([]) as unknown as Gpu;
  } catch {
    return null;
  }
}

async function makeBackend(createGpu: () => Gpu): Promise<WebGpuBackend | null> {
  try {
    const backend = new WebGpuBackend({ gpu: createGpu() });
    await backend.init();
    return backend.info().available ? backend : null;
  } catch {
    return null;
  }
}

const createGpu = await loadDawn();
const gpu = createGpu ? await makeBackend(createGpu) : null;

afterAll(() => {
  gpu?.dispose();
});

const lay = (n: number): KernelLayout => ({ shape: [n], strides: [1], offset: 0 });

describe.skipIf(!gpu)("WebGpuBackend kernels", () => {
  const g = (): WebGpuBackend => gpu as WebGpuBackend;

  const unary = async (op: UnaryKernelOp, xs: readonly number[]): Promise<number[]> => {
    const be = g();
    const x = be.upload(new Float32Array(xs));
    const out = be.unary(op, x, lay(xs.length));
    const res = Array.from(await be.download(out));
    be.free(x);
    be.free(out);
    return res;
  };

  const binary = async (
    op: BinaryKernelOp,
    xs: readonly number[],
    ys: readonly number[]
  ): Promise<number[]> => {
    const be = g();
    const a = be.upload(new Float32Array(xs));
    const b = be.upload(new Float32Array(ys));
    const out = be.binary(op, a, lay(xs.length), b, lay(ys.length), [xs.length]);
    const res = Array.from(await be.download(out));
    be.free(a);
    be.free(b);
    be.free(out);
    return res;
  };

  const rel = (got: number, want: number): number => Math.abs(got - want) / Math.abs(want);

  describe("NaN propagation (shader compilers fold x != x to false)", () => {
    const xs = [Number.NaN, 1, -1];

    it("relu and sign return NaN for NaN", async () => {
      const r = await unary("relu", xs);
      expect(r[0]).toBeNaN();
      expect(r.slice(1)).toEqual([1, 0]);
      const s = await unary("sign", xs);
      expect(s[0]).toBeNaN();
      expect(s.slice(1)).toEqual([1, -1]);
    });

    it("maximum and minimum propagate NaN from either operand", async () => {
      const max = await binary("maximum", [Number.NaN, 1, 2], [1, Number.NaN, 3]);
      expect(max[0]).toBeNaN();
      expect(max[1]).toBeNaN();
      expect(max[2]).toBe(3);
      const min = await binary("minimum", [Number.NaN, 1, 2], [1, Number.NaN, 3]);
      expect(min[0]).toBeNaN();
      expect(min[1]).toBeNaN();
      expect(min[2]).toBe(2);
    });

    it("tanh and erf return NaN for NaN", async () => {
      expect((await unary("tanh", [Number.NaN]))[0]).toBeNaN();
      expect((await unary("erf", [Number.NaN]))[0]).toBeNaN();
    });

    it("max/min reductions return NaN when any element is NaN", async () => {
      const be = g();
      for (const data of [
        [1, Number.NaN, 3],
        [Number.NaN, 1, 3],
      ]) {
        const x = be.upload(new Float32Array(data));
        for (const op of ["max", "min"] as const) {
          const out = be.reduce(op, x, lay(3));
          expect((await be.download(out))[0]).toBeNaN();
          be.free(out);
        }
        be.free(x);
      }
      const m = be.upload(new Float32Array([1, Number.NaN, 3, 4]));
      const out = be.reduceAxis("max", m, { shape: [2, 2], strides: [2, 1], offset: 0 }, 1);
      const res = Array.from(await be.download(out));
      expect(res[0]).toBeNaN();
      expect(res[1]).toBe(4);
    });
  });

  describe("activation accuracy", () => {
    it("tanh stays finite for large arguments (the built-in returns NaN past ~44)", async () => {
      expect(await unary("tanh", [44.7, 100, 1e6, -100, Number.POSITIVE_INFINITY])).toEqual([
        1, 1, 1, -1, 1,
      ]);
    });

    it("tanh is accurate for small arguments", async () => {
      // numpy.tanh in float64
      const xs = [1e-4, 1e-3, 0.1, 0.4];
      const want = [
        9.999999966666668e-5, 0.0009999996666668, 0.09966799462495582, 0.3799489622552249,
      ];
      const got = await unary("tanh", xs);
      for (let i = 0; i < xs.length; i++) expect(rel(got[i] ?? 0, want[i] ?? 1)).toBeLessThan(5e-7);
    });

    it("gelu is finite for large positive and negative inputs", async () => {
      const got = await unary("gelu", [10, 12, 50, 100, 1000, -100, -1000]);
      expect(got.slice(0, 5)).toEqual([10, 12, 50, 100, 1000]);
      expect(Math.abs(got[5] ?? 1)).toBe(0);
      expect(Math.abs(got[6] ?? 1)).toBe(0);
    });

    it("gelu keeps relative precision for negative inputs", async () => {
      // x * expit(2u) in float64, u = sqrt(2/pi) (x + 0.044715 x^3)
      const xs = [-4, -2, 0.5];
      const want = [-7.024594819237266e-5, -0.045402305912224966, 0.34571400982514394];
      const got = await unary("gelu", xs);
      for (let i = 0; i < xs.length; i++) expect(rel(got[i] ?? 0, want[i] ?? 1)).toBeLessThan(5e-6);
    });

    it("expm1 does not cancel for small arguments", async () => {
      const xs = [1e-8, 1e-4, -1e-4, 0.3, -0.3];
      const want = [
        1.0000000050000001e-8, 0.00010000500016667084, -9.999500016666251e-5, 0.3498588075760031,
        -0.2591817793182821,
      ];
      const got = await unary("expm1", xs);
      for (let i = 0; i < xs.length; i++) expect(rel(got[i] ?? 0, want[i] ?? 1)).toBeLessThan(5e-7);
    });

    it("log1p does not cancel for small arguments", async () => {
      const xs = [1e-8, 1e-4, -1e-4, 0.3, -0.3];
      const want = [
        9.999999950000001e-9, 9.999500033330834e-5, -0.00010000500033335834, 0.26236426446749106,
        -0.35667494393873234,
      ];
      const got = await unary("log1p", xs);
      for (let i = 0; i < xs.length; i++) expect(rel(got[i] ?? 0, want[i] ?? 1)).toBeLessThan(5e-7);
      expect((await unary("log1p", [-1]))[0]).toBe(Number.NEGATIVE_INFINITY);
    });

    it("softplus keeps the negative tail and the linear positive branch", async () => {
      const xs = [-30, -20, 0, 5, 50];
      const want = [9.357622968839737e-14, 2.061153620314381e-9, Math.LN2, 5.006715348489118, 50];
      const got = await unary("softplus", xs);
      for (let i = 0; i < xs.length; i++) expect(rel(got[i] ?? 0, want[i] ?? 1)).toBeLessThan(2e-6);
    });

    it("erf has small relative error near zero and around the series/rational switch", async () => {
      const xs = [1e-4, 1e-3, 0.3, 0.9, 1.5, 3, -1.5, 5];
      const want = [
        0.00011283791633342487, 0.0011283787909692363, 0.3286267594591274, 0.7969082124228319,
        0.9661051464753108, 0.9999779095030014, -0.9661051464753108, 1,
      ];
      const got = await unary("erf", xs);
      for (let i = 0; i < xs.length; i++) expect(rel(got[i] ?? 0, want[i] ?? 1)).toBeLessThan(3e-7);
    });
  });

  describe("pow special cases (match Math.pow)", () => {
    it("zero base, zero exponent and sign handling", async () => {
      const pairs: [number, number][] = [
        [0, 0],
        [0, -1],
        [0, 2],
        [0, 0.5],
        [Number.NaN, 0],
        [0, Number.NaN],
        [-2, 3],
        [-2, 2],
        [-2, 0.5],
        [3, -2],
        [2, 0.5],
        [2, 3],
        [-1, 1e10],
      ];
      const got = await binary(
        "pow",
        pairs.map((p) => p[0]),
        pairs.map((p) => p[1])
      );
      for (let i = 0; i < pairs.length; i++) {
        const [x, y] = pairs[i] ?? [0, 0];
        const want = x ** y;
        if (Number.isNaN(want)) expect(got[i]).toBeNaN();
        else expect(got[i]).toBeCloseTo(want, 5);
      }
      expect(got[1]).toBe(Number.POSITIVE_INFINITY);
    });

    it("small integer exponents are accurate", async () => {
      // 10^7 and 1.1^16 computed in float64
      const got = await binary("pow", [10, 1.1], [7, 16]);
      expect(rel(got[0] ?? 0, 1e7)).toBeLessThan(3e-7);
      expect(rel(got[1] ?? 0, Math.fround(1.1) ** 16)).toBeLessThan(3e-7);
    });
  });

  describe("large launches", () => {
    it("element-wise ops and reductions beyond 65535 workgroups (16.7M elements)", async () => {
      const be = g();
      const n = 16_800_000;
      const a = be.upload(new Float32Array(n).fill(1.5));
      const sumBuf = be.binary("add", a, lay(n), a, lay(n), [n]);
      const host = await be.download(sumBuf);
      expect(host[0]).toBe(3);
      expect(host[n - 1]).toBe(3);
      expect(host[n >> 1]).toBe(3);
      const total = be.reduce("sum", sumBuf, lay(n));
      expect((await be.download(total))[0]).toBe(3 * n);
      const mean = be.reduce("mean", a, lay(n));
      expect((await be.download(mean))[0]).toBeCloseTo(1.5, 6);
      for (const buf of [a, sumBuf, total, mean]) be.free(buf);
    }, 60_000);

    it("buffers above the 128 MiB default limit when the device allows them", async () => {
      const be = g();
      const limits = (
        be.getDevice() as unknown as { limits?: { maxStorageBufferBindingSize?: number } }
      )?.limits;
      if ((limits?.maxStorageBufferBindingSize ?? 0) <= 134_217_728) return;
      const n = 34_000_000;
      const a = be.upload(new Float32Array(n).fill(2));
      const out = be.binary("mul", a, lay(n), a, lay(n), [n]);
      const host = await be.download(out);
      expect(host[n - 1]).toBe(4);
      be.free(a);
      be.free(out);
    }, 60_000);

    it("reports a DeviceError when a buffer cannot fit", () => {
      expect(() => g().fill(0, 3_000_000_000)).toThrow(DeviceError);
    });
  });

  describe("mean reductions", () => {
    it("float16 mean does not overflow through the partial sums", async () => {
      if (!g().supportsF16()) return;
      const be = g();
      // The f16 sum would be 1e6 (> 65504) but the mean is representable.
      const x = be.upload(new Float32Array(1000).fill(1000), "float16");
      const out = be.reduce("mean", x, lay(1000));
      expect((await be.download(out))[0]).toBe(1000);
      const many = be.upload(new Float32Array(100_000).fill(0.001), "float16");
      const out2 = be.reduce("mean", many, lay(100_000));
      expect(rel((await be.download(out2))[0] ?? 0, 0.001)).toBeLessThan(5e-3);
    });

    it("float16 axis mean accumulates in float32", async () => {
      if (!g().supportsF16()) return;
      const be = g();
      const x = be.upload(new Float32Array(4 * 3000).fill(5000), "float16");
      const out = be.reduceAxis("mean", x, { shape: [4, 3000], strides: [3000, 1], offset: 0 }, 1);
      expect(Array.from(await be.download(out))).toEqual([5000, 5000, 5000, 5000]);
    });

    it("mean over an axis divides exactly", async () => {
      const be = g();
      const x = be.upload(Float32Array.from({ length: 12 }, (_, i) => i));
      const layout = { shape: [3, 4], strides: [4, 1], offset: 0 };
      expect(Array.from(await be.download(be.reduceAxis("mean", x, layout, 1)))).toEqual([
        1.5, 5.5, 9.5,
      ]);
      expect(Array.from(await be.download(be.reduceAxis("mean", x, layout, 0)))).toEqual([
        4, 5, 6, 7,
      ]);
    });

    it("full mean of 0..999", async () => {
      const be = g();
      const x = be.upload(Float32Array.from({ length: 1000 }, (_, i) => i));
      expect((await be.download(be.reduce("mean", x, lay(1000))))[0]).toBeCloseTo(499.5, 4);
    });
  });

  describe("argument validation", () => {
    const f = (): WebGpuBackend => g();

    it("matmul rejects mismatched inner dimensions instead of reading garbage", () => {
      const be = f();
      const a = be.upload(new Float32Array(6));
      const b = be.upload(new Float32Array(4));
      expect(() =>
        be.matmul(a, { shape: [2, 3], strides: [3, 1], offset: 0 }, b, {
          shape: [2, 2],
          strides: [2, 1],
          offset: 0,
        })
      ).toThrow(/inner dimensions/);
    });

    it("matmulBatched rejects layouts that disagree with batch/m/k/n", () => {
      const be = f();
      const a = be.upload(new Float32Array(12));
      const b = be.upload(new Float32Array(12));
      expect(() =>
        be.matmulBatched(
          a,
          { shape: [2, 2, 3], strides: [6, 3, 1], offset: 0 },
          b,
          { shape: [2, 3, 2], strides: [6, 2, 1], offset: 0 },
          3,
          2,
          3,
          2
        )
      ).toThrow(DeviceError);
    });

    it("fill rejects negative and fractional sizes", () => {
      expect(() => f().fill(0, -1)).toThrow(DeviceError);
      expect(() => f().fill(0, 1.5)).toThrow(DeviceError);
    });

    it("binary rejects operand layouts that were not broadcast to the output", () => {
      const be = f();
      const x = be.upload(new Float32Array(12));
      expect(() => be.binary("add", x, lay(12), x, lay(12), [3, 4])).toThrow(/broadcast/);
    });

    it("reduceAxis rejects a fractional axis", () => {
      const be = f();
      const x = be.upload(new Float32Array(12));
      expect(() =>
        be.reduceAxis("sum", x, { shape: [3, 4], strides: [4, 1], offset: 0 }, 1.5)
      ).toThrow(DeviceError);
    });

    it("pooling and im2col reject inconsistent or degenerate geometry", () => {
      const be = f();
      const x = be.upload(new Float32Array(12));
      const layout = { shape: [1, 1, 3, 4], strides: [12, 12, 4, 1], offset: 0 };
      const base = {
        batch: 1,
        channels: 1,
        height: 3,
        width: 4,
        outH: 2,
        outW: 3,
        kH: 2,
        kW: 2,
        strideH: 1,
        strideW: 1,
        padH: 0,
        padW: 0,
      };
      expect(() => be.pool2d(x, layout, "max", { ...base, strideH: 0 })).toThrow(DeviceError);
      expect(() => be.im2col(x, layout, { ...base, outH: 1, outW: 1 })).toThrow(/geometry/);
      // A valid configuration still runs.
      const out = be.pool2d(x, layout, "max", base);
      expect(out.size).toBe(6);
    });

    it("col2im and pool2dBackward reject undersized buffers", () => {
      const be = f();
      const small = be.upload(new Float32Array(2));
      const params = {
        batch: 1,
        channels: 1,
        height: 3,
        width: 4,
        outH: 2,
        outW: 3,
        kH: 2,
        kW: 2,
        strideH: 1,
        strideW: 1,
        padH: 0,
        padW: 0,
      };
      expect(() => be.col2im(small, params)).toThrow(/column buffer/);
      const x = be.upload(new Float32Array(12));
      expect(() =>
        be.pool2dBackward(
          x,
          { shape: [1, 1, 3, 4], strides: [12, 12, 4, 1], offset: 0 },
          small,
          "avg",
          params
        )
      ).toThrow(/gradient buffer/);
    });

    it("rejects an unknown device dtype", () => {
      expect(() => f().upload(new Float32Array(1), "int8" as never)).toThrow(DeviceError);
    });

    it("readBuffer rejects sizes that are not a multiple of 4", async () => {
      const be = f();
      const raw = be.createBuffer(new Float32Array([1, 2]), GPUBufferUsage.STORAGE);
      expect(raw).not.toBeNull();
      if (raw) {
        await expect(be.readBuffer(raw, 6)).rejects.toThrow(DeviceError);
        expect(Array.from(await be.readBuffer(raw, 8))).toEqual([1, 2]);
        raw.destroy();
      }
    });
  });

  describe("convolution and pooling kernels", () => {
    const geometry = (
      B: number,
      C: number,
      H: number,
      W: number,
      k: [number, number],
      stride: [number, number],
      pad: [number, number]
    ) => ({
      batch: B,
      channels: C,
      height: H,
      width: W,
      outH: Math.floor((H + 2 * pad[0] - k[0]) / stride[0]) + 1,
      outW: Math.floor((W + 2 * pad[1] - k[1]) / stride[1]) + 1,
      kH: k[0],
      kW: k[1],
      strideH: stride[0],
      strideW: stride[1],
      padH: pad[0],
      padW: pad[1],
    });
    const nchw = (B: number, C: number, H: number, W: number): KernelLayout => ({
      shape: [B, C, H, W],
      strides: [C * H * W, H * W, W, 1],
      offset: 0,
    });

    it("avg pool backward splits the gradient over in-range taps (no padding)", async () => {
      const be = g();
      const p = geometry(1, 1, 3, 3, [2, 2], [1, 1], [0, 0]);
      const x = be.upload(new Float32Array(9));
      const go = be.upload(new Float32Array(4).fill(1));
      const out = be.pool2dBackward(x, nchw(1, 1, 3, 3), go, "avg", p);
      expect(Array.from(await be.download(out))).toEqual([
        0.25, 0.5, 0.25, 0.5, 1, 0.5, 0.25, 0.5, 0.25,
      ]);
    });

    it("avg pool backward excludes padding from the divisor", async () => {
      const be = g();
      // 2x2 input, 2x2 window, padding 1 -> 3x3 output. Each pixel lies in four windows
      // holding 1, 2, 2 and 4 in-range taps: 1 + 1/2 + 1/2 + 1/4 = 2.25.
      const p = geometry(1, 1, 2, 2, [2, 2], [1, 1], [1, 1]);
      const x = be.upload(new Float32Array(4));
      const go = be.upload(new Float32Array(9).fill(1));
      const out = be.pool2dBackward(x, nchw(1, 1, 2, 2), go, "avg", p);
      expect(Array.from(await be.download(out))).toEqual([2.25, 2.25, 2.25, 2.25]);
    });

    it("max pool backward routes the gradient to the first maximum", async () => {
      const be = g();
      const p = geometry(1, 1, 2, 2, [2, 2], [1, 1], [0, 0]);
      const x = be.upload(new Float32Array([1, 1, 1, 1]));
      const go = be.upload(new Float32Array([5]));
      const out = be.pool2dBackward(x, nchw(1, 1, 2, 2), go, "max", p);
      expect(Array.from(await be.download(out))).toEqual([5, 0, 0, 0]);
    });

    it("im2col, col2im and pooling match a reference with stride, padding and views", async () => {
      const be = g();
      const [B, C, H, W] = [2, 3, 5, 6];
      const p = geometry(B, C, H, W, [3, 2], [1, 2], [1, 0]);
      let seed = 42;
      const rnd = (): number => {
        seed = (seed * 1664525 + 1013904223) % 4294967296;
        return seed / 4294967296 - 0.5;
      };
      const x = Float32Array.from({ length: B * C * H * W }, rnd);
      const at = (b: number, c: number, h: number, w: number): number =>
        x[((b * C + c) * H + h) * W + w] ?? 0;
      // The same logical tensor stored NHWC (a permuted NCHW view).
      const nhwc = new Float32Array(x.length);
      for (let b = 0; b < B; b++)
        for (let c = 0; c < C; c++)
          for (let h = 0; h < H; h++)
            for (let w = 0; w < W; w++) nhwc[((b * H + h) * W + w) * C + c] = at(b, c, h, w);
      const views: [DeviceBuffer, KernelLayout][] = [
        [be.upload(x), nchw(B, C, H, W)],
        [be.upload(nhwc), { shape: [B, C, H, W], strides: [H * W * C, 1, W * C, C], offset: 0 }],
      ];

      const cols: number[] = [];
      const poolMax: number[] = [];
      const poolAvg: number[] = [];
      for (let b = 0; b < B; b++)
        for (let oh = 0; oh < p.outH; oh++)
          for (let ow = 0; ow < p.outW; ow++)
            for (let c = 0; c < C; c++)
              for (let kh = 0; kh < p.kH; kh++)
                for (let kw = 0; kw < p.kW; kw++) {
                  const ih = oh * p.strideH + kh - p.padH;
                  const iw = ow * p.strideW + kw - p.padW;
                  const inside = ih >= 0 && ih < H && iw >= 0 && iw < W;
                  cols.push(inside ? at(b, c, ih, iw) : 0);
                }
      for (let b = 0; b < B; b++)
        for (let c = 0; c < C; c++)
          for (let oh = 0; oh < p.outH; oh++)
            for (let ow = 0; ow < p.outW; ow++) {
              const taps: number[] = [];
              for (let kh = 0; kh < p.kH; kh++)
                for (let kw = 0; kw < p.kW; kw++) {
                  const ih = oh * p.strideH + kh - p.padH;
                  const iw = ow * p.strideW + kw - p.padW;
                  if (ih >= 0 && ih < H && iw >= 0 && iw < W) taps.push(at(b, c, ih, iw));
                }
              poolMax.push(Math.max(...taps));
              poolAvg.push(taps.reduce((s, v) => s + v, 0) / taps.length);
            }
      const close = (got: Float32Array, want: number[]): void => {
        expect(got.length).toBe(want.length);
        for (let i = 0; i < want.length; i++) expect(got[i]).toBeCloseTo(want[i] ?? 0, 6);
      };
      for (const [buf, layout] of views) {
        close(await be.download(be.im2col(buf, layout, p)), cols);
        close(await be.download(be.pool2d(buf, layout, "max", p)), poolMax);
        close(await be.download(be.pool2d(buf, layout, "avg", p)), poolAvg);
      }

      // col2im is the adjoint of im2col: <im2col(x), y> == <x, col2im(y)>.
      const y = Float32Array.from({ length: cols.length }, rnd);
      const folded = await be.download(be.col2im(be.upload(y), p));
      let lhs = 0;
      for (let i = 0; i < cols.length; i++) lhs += (cols[i] ?? 0) * (y[i] ?? 0);
      let rhs = 0;
      for (let i = 0; i < x.length; i++) rhs += (x[i] ?? 0) * (folded[i] ?? 0);
      expect(rhs).toBeCloseTo(lhs, 4);
    });
  });

  describe("buffers and dtypes", () => {
    it("empty buffers round-trip", async () => {
      const be = g();
      const empty = be.upload(new Float32Array(0));
      expect((await be.download(empty)).length).toBe(0);
      const filled = be.fill(7, 0);
      expect((await be.download(filled)).length).toBe(0);
    });

    it("upload copies views and leaves the source intact", async () => {
      const be = g();
      const base = new Float32Array([9, 1, 2, 3, 9]);
      const view = base.subarray(1, 4);
      const buf = be.upload(view);
      expect(Array.from(await be.download(buf))).toEqual([1, 2, 3]);
      const whole = new Float32Array([4, 5, 6]);
      const buf2 = be.upload(whole);
      whole[0] = 100;
      expect(Array.from(await be.download(buf2))).toEqual([4, 5, 6]);
    });

    it("where keeps the bfloat16 dtype of its selected operands", async () => {
      const be = g();
      const c = be.upload(new Float32Array([1, 0, 1]));
      const a = be.upload(new Float32Array([1.1, 2.2, 3.3]), "bfloat16");
      const b = be.upload(new Float32Array([9, 8, 7]), "bfloat16");
      const out = be.ternary("where", c, lay(3), a, lay(3), b, lay(3), [3]);
      expect(out.dtype).toBe("bfloat16");
      expect(Array.from(await be.download(out))).toEqual([1.1015625, 8, 3.296875]);
    });

    it("conv/pool kernels keep the bfloat16 dtype of their input", async () => {
      const be = g();
      const x = be.upload(new Float32Array(12).fill(1), "bfloat16");
      const layout = { shape: [1, 1, 3, 4], strides: [12, 12, 4, 1], offset: 0 };
      const params = {
        batch: 1,
        channels: 1,
        height: 3,
        width: 4,
        outH: 2,
        outW: 3,
        kH: 2,
        kW: 2,
        strideH: 1,
        strideW: 1,
        padH: 0,
        padW: 0,
      };
      const pooled: DeviceBuffer = be.pool2d(x, layout, "avg", params);
      expect(pooled.dtype).toBe("bfloat16");
      expect(be.im2col(x, layout, params).dtype).toBe("bfloat16");
    });

    it("advertises conv2d support", () => {
      expect(g().supports("conv2d")).toBe(true);
    });
  });

  describe("lifecycle", () => {
    it("dispose() during init() does not leak a device", async () => {
      if (!createGpu) return;
      const be = new WebGpuBackend({ gpu: createGpu() });
      const pending = be.init();
      be.dispose();
      await pending;
      expect(be.info().available).toBe(false);
      expect(be.getDevice()).toBeNull();
    });

    it("a lost device makes the backend unavailable", async () => {
      if (!createGpu) return;
      const be = await makeBackend(createGpu);
      if (!be) return;
      be.getDevice()?.destroy();
      // The `lost` promise settles asynchronously.
      for (let i = 0; i < 50 && be.info().available; i++) {
        await new Promise((resolve) => setTimeout(resolve, 20));
      }
      expect(be.info().available).toBe(false);
      expect(() => be.fill(0, 4)).toThrow(DeviceError);
      be.dispose();
    });
  });
});
