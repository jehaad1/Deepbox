#!/usr/bin/env node
// Ensures src/core/**/*.ts @see targets match DeepboxDocs core.json slugs (path → expected URL path).
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const pkgRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const srcCore = path.join(pkgRoot, "src", "core");

/**
 * @param {string} relFromPkg e.g. src/core/logger.ts
 * @returns {string[] | null} required URL path segments after deepbox.dev/docs/
 */
function requiredDocPaths(relFromPkg) {
  if (!relFromPkg.startsWith("src/core/")) return null;
  const rest = relFromPkg.slice("src/core/".length);
  if (rest === "index.ts") {
    return ["core-types", "core-config", "core-errors", "core-utils"];
  }
  if (rest === "logger.ts" || rest === "warnings.ts" || rest.startsWith("errors/")) {
    if (rest === "errors/broadcast.ts") {
      return ["core-errors", "ndarray-ops"];
    }
    return ["core-errors"];
  }
  if (rest.startsWith("config/")) {
    return ["core-config"];
  }
  if (
    rest === "backend/WebGpuBackend.ts" ||
    rest === "backend/WasmBackend.ts" ||
    rest === "backend/kernels.ts" ||
    rest === "backend/wasm_modules.generated.ts"
  ) {
    return ["devices-and-execution"];
  }
  if (rest.startsWith("backend/")) {
    return ["core-config"];
  }
  if (rest.startsWith("utils/") || rest === "serialization.ts" || rest.startsWith("parallel/")) {
    return ["core-utils"];
  }
  if (rest.startsWith("types/")) {
    return ["core-types"];
  }
  return null;
}

/** @param {string} dir @param {{ rel: string[] }} acc */
function collect(dir, acc) {
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, ent.name);
    if (ent.isDirectory()) collect(full, acc);
    else if (ent.isFile() && ent.name.endsWith(".ts") && !ent.name.endsWith(".d.ts")) {
      acc.rel.push(path.relative(pkgRoot, full).split(path.sep).join("/"));
    }
  }
}

const allFiles = { rel: [] };
collect(srcCore, allFiles);
allFiles.rel.sort();

const failures = [];
for (const rel of allFiles.rel) {
  const required = requiredDocPaths(rel);
  if (!required) continue;
  const text = fs.readFileSync(path.join(pkgRoot, rel), "utf8");
  for (const slug of required) {
    const needle = `deepbox.dev/docs/${slug}`;
    if (!text.includes(needle)) {
      failures.push({ rel, missing: needle });
    }
  }
}

console.log(
  `JSDoc core slug check: ${failures.length === 0 ? "ok" : `${failures.length} mismatch(es)`} across src/core/**/*.ts\n`
);
for (const f of failures) {
  console.log(`${f.rel}: missing reference to ${f.missing}`);
}

if (failures.length > 0) {
  process.exitCode = 1;
}
