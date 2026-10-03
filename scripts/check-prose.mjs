#!/usr/bin/env node
// Fails when project text contains an em dash (U+2014).
// Project style uses commas, colons, parentheses or periods instead.
// Scans source, tests, scripts, benchmarks, docs and the top-level Markdown files.
// Exits 1 and lists every offending line when any are found.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const pkgRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const EM_DASH = "\u2014";

const roots = ["src", "test", "scripts", "benchmarks", "docs", ".github"];
const topLevelFiles = [
  "README.md",
  "CHANGELOG.md",
  "CONTRIBUTING.md",
  "SECURITY.md",
  "CODE_OF_CONDUCT.md",
  "SKILL.md",
  "package.json",
];
const extensions = new Set([".ts", ".mts", ".mjs", ".js", ".md", ".json", ".yml", ".yaml", ".py"]);
const skipDirs = new Set(["node_modules", "dist", "coverage", "output", "results", "__pycache__"]);

/** @param {string} dir @param {string[]} acc */
function collect(dir, acc) {
  if (!fs.existsSync(dir)) return;
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    if (skipDirs.has(ent.name)) continue;
    const full = path.join(dir, ent.name);
    if (ent.isDirectory()) collect(full, acc);
    else if (ent.isFile() && extensions.has(path.extname(ent.name))) acc.push(full);
  }
}

const files = [];
for (const root of roots) collect(path.join(pkgRoot, root), files);
for (const name of topLevelFiles) {
  const full = path.join(pkgRoot, name);
  if (fs.existsSync(full)) files.push(full);
}

const hits = [];
for (const file of files.sort()) {
  const lines = fs.readFileSync(file, "utf8").split("\n");
  lines.forEach((line, i) => {
    if (line.includes(EM_DASH)) {
      hits.push(`${path.relative(pkgRoot, file)}:${i + 1}: ${line.trim().slice(0, 120)}`);
    }
  });
}

if (hits.length > 0) {
  console.error(
    `Found ${hits.length} line(s) with an em dash (U+2014). Use a comma, colon, parentheses or a period instead.\n`
  );
  for (const hit of hits) console.error(hit);
  process.exit(1);
}
console.log(`Prose check passed: no em dashes in ${files.length} files.`);
