// CORE-1: unregisterBackend resets a default device that pointed at the removed backend, and
// the mandatory CPU backend cannot be replaced.
import { afterEach, describe, expect, it } from "vitest";
import {
  CpuBackend,
  getBackend,
  getDevice,
  InvalidParameterError,
  registerBackend,
  resetConfig,
  setDevice,
  unregisterBackend,
} from "../../src/core";

describe("CORE-1: backend registry and default device", () => {
  afterEach(() => {
    unregisterBackend("wasm");
    resetConfig();
  });

  it("falls back to cpu when the default device's backend is unregistered", () => {
    registerBackend("wasm", new CpuBackend());
    setDevice("wasm");
    expect(getDevice()).toBe("wasm");
    expect(unregisterBackend("wasm")).toBe(true);
    expect(getDevice()).toBe("cpu");
  });

  it("leaves the default device alone when another backend is removed", () => {
    registerBackend("wasm", new CpuBackend());
    expect(unregisterBackend("wasm")).toBe(true);
    expect(getDevice()).toBe("cpu");
    registerBackend("wasm", new CpuBackend());
    setDevice("cpu");
    expect(unregisterBackend("wasm")).toBe(true);
    expect(getDevice()).toBe("cpu");
  });

  it("does not touch the default device when nothing was registered", () => {
    expect(unregisterBackend("wasm")).toBe(false);
    expect(getDevice()).toBe("cpu");
  });

  it("refuses to replace the cpu backend", () => {
    const original = getBackend("cpu");
    expect(() => registerBackend("cpu", new CpuBackend())).toThrow(InvalidParameterError);
    expect(getBackend("cpu")).toBe(original);
  });
});
