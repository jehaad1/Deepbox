#!/usr/bin/env node
// Runs every code fence inside an `@example` JSDoc block under src/ and fails when one throws.
//
// Convention
//   - A fence marked ```ts (or ```js) must run as written, as an ES module with top-level await.
//   - Imports of `deepbox` and `deepbox/<module>` are rewritten to the source index files, so no build is needed.
//   - A true snippet that cannot stand alone (a method body excerpt, network or file IO) is marked
//     with the word `fragment` in the fence info string: ```ts fragment. Fragments are skipped but counted.
//   - Prefer making an example runnable. Examples that write files must use os.tmpdir().
//
// Usage
//   node scripts/check-jsdoc-examples.mjs [--skip a,b] [--only substring] [--concurrency N] [--keep]
//
// Every src directory is checked. --skip excludes directories for a local run.
import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const pkgRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const srcRoot = path.join(pkgRoot, "src");

const args = process.argv.slice(2);
function flagValue(name) {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : undefined;
}
const skipDirs = flagValue("--skip")?.split(",").filter(Boolean) ?? [];
const only = flagValue("--only");
const keep = args.includes("--keep");
const concurrency = Number(flagValue("--concurrency") ?? Math.max(1, os.cpus().length - 1));
const TIMEOUT_MS = 60_000;

/** @param {string} dir @param {string[]} acc */
function collect(dir, acc) {
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, ent.name);
    if (ent.isDirectory()) collect(full, acc);
    else if (ent.isFile() && ent.name.endsWith(".ts") && !ent.name.endsWith(".d.ts"))
      acc.push(full);
  }
}

/**
 * Extract fenced examples from the JSDoc blocks of one file.
 * @param {string} file
 * @returns {{ file: string, line: number, lang: string, fragment: boolean, code: string }[]}
 */
function extract(file) {
  const rel = path.relative(pkgRoot, file).split(path.sep).join("/");
  const lines = fs.readFileSync(file, "utf8").split("\n");
  const out = [];
  let inDoc = false;
  let inExample = false;
  let fence = null;
  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i];
    if (!inDoc) {
      if (/^\s*\/\*\*/.test(raw)) {
        inDoc = true;
        inExample = false;
        fence = null;
      } else continue;
    }
    const text = raw
      .replace(/^\s*\/\*\*\s?/, "")
      .replace(/^\s*\*\/?\s?/, "")
      .replace(/\*\/\s*$/, "");
    if (fence) {
      if (/^\s*```\s*$/.test(text)) {
        out.push({
          file: rel,
          line: fence.line,
          lang: fence.lang,
          fragment: fence.fragment,
          code: fence.body.join("\n"),
        });
        fence = null;
      } else fence.body.push(text);
    } else if (/^\s*@example\b/.test(text)) {
      inExample = true;
    } else if (/^\s*@[a-zA-Z]/.test(text)) {
      inExample = false;
    } else if (inExample) {
      const m = /^\s*```(\w*)\s*(.*)$/.exec(text);
      if (m) {
        const info = (m[2] ?? "").trim().split(/\s+/);
        fence = { line: i + 2, lang: m[1] || "ts", fragment: info.includes("fragment"), body: [] };
      }
    }
    if (/\*\//.test(raw)) {
      inDoc = false;
      inExample = false;
      fence = null;
    }
  }
  return out;
}

const modules = fs
  .readdirSync(srcRoot, { withFileTypes: true })
  .filter((e) => e.isDirectory() && fs.existsSync(path.join(srcRoot, e.name, "index.ts")))
  .map((e) => e.name);

/** @param {string} code */
function rewriteImports(code) {
  return code.replace(/(["'])deepbox(?:\/([\w-]+))?\1/g, (all, q, mod) => {
    if (!mod) return `${q}${path.join(srcRoot, "index.ts")}${q}`;
    if (!modules.includes(mod)) return all;
    return `${q}${path.join(srcRoot, mod, "index.ts")}${q}`;
  });
}

const files = [];
collect(srcRoot, files);
const examples = [];
for (const f of files.sort()) {
  const top = path.relative(srcRoot, f).split(path.sep)[0];
  if (skipDirs.includes(top)) {
    const n = extract(f).length;
    if (n > 0) examples.push({ skipped: true, top, count: n });
    continue;
  }
  for (const ex of extract(f)) {
    if (only && !`${ex.file}:${ex.line}`.includes(only)) continue;
    examples.push(ex);
  }
}

const skippedCount = examples.filter((e) => e.skipped).reduce((s, e) => s + e.count, 0);
const fragments = examples.filter((e) => !e.skipped && e.fragment);
const runnable = examples.filter((e) => !e.skipped && !e.fragment);

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "deepbox-jsdoc-"));
const tsxLoader = pathToFileURL(
  path.join(pkgRoot, "node_modules", "tsx", "dist", "loader.mjs")
).href;
const tsxArgs = ["--import", tsxLoader];

/** @param {{file:string,line:number,code:string}} ex @param {number} idx */
function run(ex, idx) {
  const target = path.join(tmp, `ex${idx}.mts`);
  fs.writeFileSync(target, `${rewriteImports(ex.code)}\n`);
  return new Promise((resolve) => {
    const child = spawn(process.execPath, [...tsxArgs, target], {
      cwd: pkgRoot,
      stdio: ["ignore", "pipe", "pipe"],
      env: { ...process.env, NO_COLOR: "1", FORCE_COLOR: "0" },
    });
    let stderr = "";
    let stdout = "";
    child.stdout.on("data", (d) => {
      stdout += d;
    });
    child.stderr.on("data", (d) => {
      stderr += d;
    });
    const timer = setTimeout(() => child.kill("SIGKILL"), TIMEOUT_MS);
    child.on("close", (code, signal) => {
      clearTimeout(timer);
      if (code === 0) return resolve(null);
      const text = (stderr || stdout)
        .split("\n")
        .map((l) => l.trim())
        .filter(Boolean);
      const errLine =
        text.find((l) => /^[\w.]*(Error|Exception)(\s\[\w+\])?:/.test(l)) ??
        text.find((l) => /(Error|error TS|ERROR)/.test(l) && !l.startsWith("at ")) ??
        text.find((l) => !l.startsWith("at ") && !l.startsWith("file:")) ??
        (signal ? `killed by ${signal} (timeout)` : `exit code ${code}`);
      resolve(
        errLine.replace(new RegExp(tmp.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "g"), "<example>")
      );
    });
  });
}

const failures = [];
let next = 0;
let passed = 0;
async function worker() {
  while (next < runnable.length) {
    const idx = next++;
    const ex = runnable[idx];
    const err = await run(ex, idx);
    if (err === null) passed++;
    else failures.push({ ex, err });
  }
}
const started = Date.now();
await Promise.all(
  Array.from({ length: Math.min(concurrency, Math.max(1, runnable.length)) }, worker)
);
if (!keep) fs.rmSync(tmp, { recursive: true, force: true });

failures.sort((a, b) =>
  a.ex.file === b.ex.file ? a.ex.line - b.ex.line : a.ex.file < b.ex.file ? -1 : 1
);
for (const { ex, err } of failures) console.error(`${ex.file}:${ex.line}: ${err}`);

const seconds = ((Date.now() - started) / 1000).toFixed(1);
const summary = `JSDoc examples: run ${runnable.length}, passed ${passed}, failed ${failures.length}, fragments ${fragments.length}, skipped ${skippedCount} (dirs: ${skipDirs.join(",") || "none"}) in ${seconds}s`;
if (failures.length > 0) {
  console.error(`\n${summary}`);
  process.exit(1);
}
console.log(summary);
