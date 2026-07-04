#!/usr/bin/env node
// Adds @see https://deepbox.dev/docs/... to src/**/*.ts files that omit "deepbox.dev/docs".
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const pkgRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const srcRoot = path.join(pkgRoot, "src");

/** @type {Record<string, string>} */
const defaultDocSlugByTopDir = {
  core: "core-types",
  ndarray: "ndarray-tensor",
  linalg: "linalg-properties",
  dataframe: "dataframe-overview",
  stats: "stats-descriptive",
  metrics: "metrics-classification",
  preprocess: "preprocess-scalers",
  ml: "ml-linear",
  nn: "nn-module",
  optim: "optim-optimizers",
  random: "random-generation",
  datasets: "datasets-builtin",
  plot: "plot-basic",
};

/** @param {string} relPosix e.g. src/stats/kde.ts */
function docUrlFor(relPosix) {
  if (relPosix === "src/index.ts") {
    return "https://deepbox.dev/docs/introduction";
  }
  const parts = relPosix.split("/");
  if (parts[0] !== "src" || parts.length < 2) {
    return "https://deepbox.dev/docs/introduction";
  }
  const top = parts[1];
  if (top === "core") {
    const rest = parts.slice(2).join("/");
    if (rest.startsWith("errors/") || rest === "logger.ts" || rest === "warnings.ts") {
      return "https://deepbox.dev/docs/core-errors";
    }
    if (rest.startsWith("config/")) {
      return "https://deepbox.dev/docs/core-config";
    }
    if (rest === "backend/WebGpuBackend.ts" || rest === "backend/WasmBackend.ts") {
      return "https://deepbox.dev/docs/devices-and-execution";
    }
    if (rest.startsWith("backend/")) {
      return "https://deepbox.dev/docs/core-config";
    }
    if (rest.startsWith("utils/") || rest === "serialization.ts" || rest.startsWith("parallel/")) {
      return "https://deepbox.dev/docs/core-utils";
    }
    if (rest.startsWith("types/")) {
      return "https://deepbox.dev/docs/core-types";
    }
    return "https://deepbox.dev/docs/core-types";
  }
  if (top === "stats") {
    const base = parts[parts.length - 1];
    if (base === "distributions.ts" || base === "kde.ts") {
      return "https://deepbox.dev/docs/stats-distributions";
    }
    if (
      base === "power.ts" ||
      base === "tests.ts" ||
      base === "confidence.ts" ||
      base === "multiple.ts"
    ) {
      return "https://deepbox.dev/docs/stats-tests";
    }
    return "https://deepbox.dev/docs/stats-descriptive";
  }
  const slug = defaultDocSlugByTopDir[top] ?? "introduction";
  return `https://deepbox.dev/docs/${slug}`;
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

/** @param {string} content @param {string} url */
function injectSee(content, url) {
  if (content.includes("deepbox.dev/docs")) return content;
  const see = ` * @see {@link ${url} | Deepbox documentation}`;

  const trimmed = content.trimStart();
  if (trimmed.startsWith("/**")) {
    const start = content.indexOf("/**");
    const end = content.indexOf("*/", start);
    if (end !== -1) {
      const before = content.slice(0, end);
      if (before.includes("deepbox.dev/docs")) return content;
      const after = content.slice(end);
      return `${before.trimEnd()}\n${see}\n${after}`;
    }
  }

  return `/**\n${see}\n */\n\n${content}`;
}

const allFiles = { rel: [] };
collect(srcRoot, allFiles);
allFiles.rel.sort();

let updated = 0;
for (const rel of allFiles.rel) {
  if (rel.endsWith("env.d.ts")) continue;
  const abs = path.join(pkgRoot, rel);
  const text = fs.readFileSync(abs, "utf8");
  if (text.includes("deepbox.dev/docs")) continue;
  const next = injectSee(text, docUrlFor(rel));
  if (next !== text) {
    fs.writeFileSync(abs, next);
    updated += 1;
  }
}

console.log(`inject-jsdoc-deepbox-dev: updated ${updated} files.`);
