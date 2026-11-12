#!/usr/bin/env node
/**
 * Writes the root entry points `dist/index.js` and `dist/index.cjs`.
 *
 * The root package exports each module as a namespace (`db.ndarray`, `db.ml`, ...).
 * When tsup bundled `src/index.ts` together with the subpath entries, it emitted the
 * namespace objects into the shared chunks, so `import { tensor } from "deepbox/ndarray"`
 * also pulled in every namespace object and the code they reference. The root entry is
 * therefore written here as thin re-exports of the subpath builds: native ESM namespace
 * re-exports and plain CommonJS `require` calls. Both entries share the subpath modules,
 * so `db.ndarray.Tensor` is the same class as the one from `deepbox/ndarray`.
 *
 * The module list comes from the `exports` map in package.json.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const pkg = JSON.parse(readFileSync(join(root, "package.json"), "utf8"));
const modules = Object.keys(pkg.exports)
  .filter((key) => key !== "." && key !== "./package.json")
  .map((key) => key.slice(2))
  .sort();

const esm = modules.map((m) => `export * as ${m} from "./${m}/index.js";`).join("\n");
const cjs = [
  '"use strict";',
  ...modules.map((m) => `exports.${m} = require("./${m}/index.cjs");`),
].join("\n");

writeFileSync(join(root, "dist", "index.js"), `${esm}\n`);
writeFileSync(join(root, "dist", "index.cjs"), `${cjs}\n`);
console.log(`write-root-entry: ${modules.length} namespaces (${modules.join(", ")})`);
