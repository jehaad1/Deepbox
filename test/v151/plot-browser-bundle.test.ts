import { resolve } from "node:path";
import { build } from "esbuild";
import { afterEach, describe, expect, it, vi } from "vitest";

// esbuild ships with tsup (a devDependency). It stands in for any browser bundler here.
describe("PKG-5 plot and datasets bundle for browsers", () => {
  for (const entry of ["src/plot/index.ts", "src/datasets/index.ts"]) {
    it(`${entry} resolves with platform=browser`, async () => {
      const result = await build({
        entryPoints: [resolve(entry)],
        bundle: true,
        platform: "browser",
        format: "esm",
        write: false,
        logLevel: "silent",
      });
      expect(result.errors).toEqual([]);
      const code = result.outputFiles[0]?.text ?? "";
      expect(code).toContain("webpackIgnore");
      expect(code).toContain("@vite-ignore");
    });
  }
});

describe("PKG-5 saveFig without a file system", () => {
  afterEach(() => {
    vi.doUnmock("node:fs/promises");
    vi.resetModules();
  });

  it("throws a DeviceError that names Node.js", async () => {
    vi.resetModules();
    vi.doMock("node:fs/promises", () => {
      throw new Error("no fs here");
    });
    const { saveFig, figure } = await import("../../src/plot");
    const { DeviceError } = await import("../../src/core");
    figure();
    const error = await saveFig("out.svg").catch((e: unknown) => e);
    expect(error).toBeInstanceOf(DeviceError);
    expect((error as Error).message).toMatch(/needs Node\.js/);
  }, 60_000);
});
