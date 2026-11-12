#!/usr/bin/env node
/**
 * Makes the emitted declarations valid under every TypeScript module resolution mode.
 *
 * `tsc --emitDeclarationOnly` copies the extensionless relative specifiers used in
 * `src` (for example `from "./tensor"`). The `bundler` and `node10` modes accept them,
 * but `node16` and `nodenext` require explicit extensions, so they report TS2834 and
 * every import from the package becomes untyped.
 *
 * This script runs after the declaration build and:
 * 1. Rewrites each relative specifier in `dist/**\/*.d.ts` to its explicit target:
 *    `./tensor` becomes `./tensor.js`, and a directory `./ops` becomes `./ops/index.js`.
 * 2. Writes a `.d.cts` twin next to every `.d.ts`, with `.cjs` specifiers, so the
 *    `require` condition gets CommonJS-flavored types instead of ESM types.
 *
 * It fails when a specifier resolves to neither a file nor a directory index.
 */
import { existsSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const dist = join(root, "dist");

// Matches the module specifier in `from "./x"`, `import("./x")` and `import "./x"`.
const SPECIFIER = /(\bfrom\s+|\bimport\s*\(\s*|\bimport\s+)(["'])(\.{1,2}\/[^"']*)\2/g;

/** @param {string} dir */
function listDeclarations(dir) {
  /** @type {string[]} */
  const out = [];
  for (const name of readdirSync(dir)) {
    const full = join(dir, name);
    if (statSync(full).isDirectory()) out.push(...listDeclarations(full));
    else if (name.endsWith(".d.ts")) out.push(full);
  }
  return out;
}

/**
 * @param {string} fromFile
 * @param {string} spec
 */
function resolveSpecifier(fromFile, spec) {
  if (/\.(c|m)?js$/.test(spec)) return spec;
  const base = resolve(dirname(fromFile), spec);
  if (existsSync(`${base}.d.ts`)) return `${spec}.js`;
  if (existsSync(join(base, "index.d.ts"))) return `${spec.replace(/\/$/, "")}/index.js`;
  throw new Error(`fix-declarations: cannot resolve "${spec}" in ${relative(root, fromFile)}`);
}

if (!existsSync(dist)) {
  throw new Error("fix-declarations: dist/ does not exist. Run the build first.");
}

const files = listDeclarations(dist);
let rewritten = 0;
for (const file of files) {
  const source = readFileSync(file, "utf8");
  const esm = source.replace(SPECIFIER, (_match, lead, quote, spec) => {
    rewritten++;
    return `${lead}${quote}${resolveSpecifier(file, spec)}${quote}`;
  });
  if (esm !== source) writeFileSync(file, esm);
  const cjs = esm.replace(
    SPECIFIER,
    (_match, lead, quote, spec) => `${lead}${quote}${spec.replace(/\.js$/, ".cjs")}${quote}`
  );
  writeFileSync(file.replace(/\.d\.ts$/, ".d.cts"), cjs);
}

console.log(
  `fix-declarations: ${files.length} declaration files, ${rewritten} specifiers rewritten, ${files.length} .d.cts files written`
);
