import { describe, expect, it } from "vitest";
import {
  CpuBackend,
  DeviceError,
  getBackend,
  getDevice,
  isBackendAvailable,
  listBackends,
  registerBackend,
  resetConfig,
  setDevice,
  unregisterBackend,
  WasmBackend,
  WebGpuBackend,
} from "../src/core";
import { tensor } from "../src/ndarray";

describe("Backend abstraction layer", () => {
  describe("CpuBackend", () => {
    it("reports correct info", () => {
      const cpu = new CpuBackend();
      const info = cpu.info();
      expect(info.device).toBe("cpu");
      expect(info.name).toBe("Deepbox CPU Backend");
      expect(info.available).toBe(true);
      expect(info.capabilities.length).toBeGreaterThan(0);
    });

    it("supports all standard capabilities", () => {
      const cpu = new CpuBackend();
      expect(cpu.supports("matmul")).toBe(true);
      expect(cpu.supports("elementwise")).toBe(true);
      expect(cpu.supports("reduction")).toBe(true);
      expect(cpu.supports("random")).toBe(true);
      expect(cpu.supports("blas")).toBe(true);
      expect(cpu.supports("conv2d")).toBe(true);
      expect(cpu.supports("fft")).toBe(true);
    });

    it("init resolves without error", async () => {
      const cpu = new CpuBackend();
      await expect(cpu.init()).resolves.toBeUndefined();
    });

    it("dispose marks backend as disposed", () => {
      const cpu = new CpuBackend();
      expect(cpu.isDisposed).toBe(false);
      cpu.dispose();
      expect(cpu.isDisposed).toBe(true);
    });
  });

  describe("registry", () => {
    it("cpu backend is registered by default", () => {
      const cpu = getBackend("cpu");
      expect(cpu.info().device).toBe("cpu");
    });

    it("listBackends includes cpu", () => {
      const backends = listBackends();
      expect(backends).toContain("cpu");
    });

    it("isBackendAvailable returns true for cpu, false for unregistered", () => {
      // Ensure a clean slate regardless of test execution order.
      unregisterBackend("webgpu");
      unregisterBackend("wasm");
      expect(isBackendAvailable("cpu")).toBe(true);
      expect(isBackendAvailable("webgpu")).toBe(false);
      expect(isBackendAvailable("wasm")).toBe(false);
    });

    it("getBackend throws for unregistered device", () => {
      unregisterBackend("webgpu");
      unregisterBackend("wasm");
      expect(() => getBackend("webgpu")).toThrow(/No backend registered/);
      expect(() => getBackend("wasm")).toThrow(/No backend registered/);
    });

    it("setDevice rejects devices without usable backends", () => {
      resetConfig();
      unregisterBackend("webgpu");
      expect(() => setDevice("webgpu")).toThrow(DeviceError);
      expect(getDevice()).toBe("cpu");
    });

    it("tensor creation rejects unavailable device metadata", () => {
      unregisterBackend("webgpu");
      expect(() => tensor([1, 2, 3], { device: "webgpu" })).toThrow(DeviceError);
    });

    it("registerBackend and unregisterBackend manage custom backends", () => {
      const mockBackend = new CpuBackend();
      registerBackend("wasm", mockBackend);
      expect(isBackendAvailable("wasm")).toBe(true);
      expect(getBackend("wasm")).toBe(mockBackend);
      expect(listBackends()).toContain("wasm");

      expect(unregisterBackend("wasm")).toBe(true);
      expect(unregisterBackend("wasm")).toBe(false); // already removed
      expect(isBackendAvailable("wasm")).toBe(false);
      expect(() => unregisterBackend("cpu")).toThrow(DeviceError);
    });
  });

  describe("accelerated backends", () => {
    it("WebGPU does not report available until a GPU device is acquired", async () => {
      const gpu = new WebGpuBackend();
      expect(gpu.info().available).toBe(false);
      // No WebGPU in this runtime and no explicit provider: init succeeds
      // but the backend stays unavailable.
      await gpu.init();
      if (typeof navigator === "undefined" || !("gpu" in navigator)) {
        expect(gpu.info().available).toBe(false);
      }
    });

    it("WASM SIMD backend initializes and actually computes", async () => {
      const wasm = new WasmBackend();
      expect(wasm.info().available).toBe(false);
      await wasm.init();
      if (!wasm.isSimdSupported()) {
        expect(wasm.info().available).toBe(false);
        return;
      }
      expect(wasm.info().available).toBe(true);

      // Element-wise kernels are bit-identical to the scalar CPU results,
      // including the non-multiple-of-4 tail.
      const n = 1027;
      const a = new Float32Array(n).map((_, i) => i * 0.25);
      const b = new Float32Array(n).map((_, i) => (i % 5) + 1);
      const out = wasm.binaryContiguous("add", a, b);
      expect(out).not.toBeNull();
      for (let i = 0; i < n; i++) {
        expect(out?.[i]).toBe(Math.fround((a[i] ?? 0) + (b[i] ?? 0)));
      }

      wasm.dispose();
      expect(wasm.info().available).toBe(false);
    });
  });
});

describe("WebGPU backend guards (no GPU required)", () => {
  it("kernel calls on an uninitialized backend throw DeviceError", () => {
    const gpu = new WebGpuBackend();
    const layout = { shape: [1], strides: [1], offset: 0 };
    const buf = { device: "webgpu" as const, byteLength: 4, size: 1 };
    expect(() => gpu.upload(new Float32Array(1))).toThrow(DeviceError);
    expect(() => gpu.fill(0, 1)).toThrow(DeviceError);
    expect(() => gpu.binary("add", buf, layout, buf, layout, [1])).toThrow(DeviceError);
    expect(() => gpu.unary("neg", buf, layout)).toThrow(DeviceError);
    expect(() =>
      gpu.matmul(buf, { ...layout, shape: [1, 1], strides: [1, 1] }, buf, {
        ...layout,
        shape: [1, 1],
        strides: [1, 1],
      })
    ).toThrow(DeviceError);
    expect(() => gpu.reduce("sum", buf, layout)).toThrow(DeviceError);
    expect(gpu.getPipeline("add")).toBeNull();
    expect(gpu.getDevice()).toBeNull();
    expect(gpu.listPipelines()).toEqual([]);
    expect(gpu.createBuffer(new Float32Array(1), 0)).toBeNull();
  });

  it("exposes the built-in shader catalog and capability set", () => {
    const gpu = new WebGpuBackend();
    const shaders = gpu.listShaders();
    for (const name of ["add", "mul", "exp", "relu", "matmul", "reduceSum", "fill", "step"]) {
      expect(shaders).toContain(name);
    }
    expect(gpu.supports("matmul")).toBe(true);
    expect(gpu.supports("fft")).toBe(false);
  });

  it("disposed backends report disposed and free() is a no-op", () => {
    const gpu = new WebGpuBackend();
    gpu.dispose();
    gpu.dispose();
    expect(gpu.isDisposed).toBe(true);
    expect(() => gpu.upload(new Float32Array(1))).toThrow(/disposed/);
    gpu.free({ device: "webgpu", byteLength: 4, size: 1 }); // no-op
  });
});

describe("WASM backend introspection", () => {
  it("lists modules and reports state before/after init", async () => {
    const wasm = new WasmBackend();
    expect(wasm.isInitialized).toBe(false);
    expect(wasm.hasModule("simdAdd")).toBe(false);
    expect(wasm.getModule("simdAdd")).toBeUndefined();
    expect(wasm.listModules()).toContain("simdDiv");
    expect(wasm.supports("elementwise")).toBe(true);
    expect(wasm.supports("fft")).toBe(false);
    // Kernel calls before init return null (callers fall back to CPU)
    expect(wasm.binaryContiguous("add", new Float32Array(2), new Float32Array(2))).toBeNull();
    expect(wasm.dotContiguous(new Float32Array(2), new Float32Array(2))).toBeNull();
    expect(wasm.sumContiguous(new Float32Array(2))).toBeNull();

    await wasm.init();
    await wasm.init(); // idempotent
    if (wasm.isSimdSupported()) {
      expect(wasm.isInitialized).toBe(true);
      expect(wasm.hasModule("simdAdd")).toBe(true);
      expect(wasm.getModule("simdSum")?.name).toBe("simdSum");
      // Length-mismatched operands are rejected as null, not misread.
      expect(wasm.binaryContiguous("add", new Float32Array(2), new Float32Array(3))).toBeNull();
      expect(wasm.dotContiguous(new Float32Array(2), new Float32Array(3))).toBeNull();
    }
    wasm.dispose();
  });
});
