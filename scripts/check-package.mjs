#!/usr/bin/env node
/**
 * Packs the built package and checks that every entry point's types resolve under
 * node10, node16 (ESM and CommonJS) and bundler resolution. Run after `npm run build`.
 */
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const dir = mkdtempSync(join(tmpdir(), "deepbox-pack-"));
try {
  const out = execFileSync(
    "npm",
    ["pack", "--ignore-scripts", "--json", "--pack-destination", dir],
    {
      encoding: "utf8",
    }
  );
  const [{ filename }] = JSON.parse(out);
  execFileSync("npx", ["attw", join(dir, filename)], { stdio: "inherit" });
} finally {
  rmSync(dir, { recursive: true, force: true });
}
