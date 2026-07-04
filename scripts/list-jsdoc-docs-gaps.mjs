#!/usr/bin/env node
// Lists every src/**/*.ts file whose contents omit a DeepboxDocs URL path (`deepbox.dev/docs`).
// Stronger than a bare `deepbox.dev` check: ensures files point at documentation, not only the site root.
// Exits 1 when any file fails (use in validate:all).
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const pkgRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const srcRoot = path.join(pkgRoot, "src");
const requiredSubstring = "deepbox.dev/docs";

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
collect(srcRoot, allFiles);
allFiles.rel.sort();

const missing = [];
for (const rel of allFiles.rel) {
  if (rel.endsWith("env.d.ts")) continue;
  const text = fs.readFileSync(path.join(pkgRoot, rel), "utf8");
  if (!text.includes(requiredSubstring)) missing.push(rel);
}

console.log(
  `JSDoc / docs URL gap report: ${missing.length} of ${allFiles.rel.length} src *.ts files omit "${requiredSubstring}" (add @see links as you touch modules).\n`
);
for (const p of missing) console.log(p);

if (missing.length > 0) {
  process.exitCode = 1;
}
