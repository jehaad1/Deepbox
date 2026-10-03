/**
 * Compile the Deepbox WASM SIMD kernels (WAT → wasm) and embed them as
 * base64 in `src/core/backend/wasm_modules.generated.ts`.
 *
 * The generated file is committed so the published package needs neither a
 * WAT compiler nor a build step at install time. Re-run after editing the
 * WAT sources below:
 *
 *   node scripts/build-wasm.mjs
 */

import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import wabtFactory from "wabt";

/**
 * Binary element-wise kernel: out[i] = a[i] OP b[i] for i in [0, count).
 * SIMD main loop (4 f32 lanes) + scalar tail so every count is handled.
 * Offsets are byte offsets into the shared imported memory.
 */
const binaryWat = (simdOp, scalarOp) => `(module
  (import "env" "memory" (memory 1))
  (func (export "run") (param $a i32) (param $b i32) (param $out i32) (param $count i32)
    (local $i i32)
    (local $vn i32)
    (local.set $vn (i32.and (local.get $count) (i32.const -4)))
    (block $break
      (loop $loop
        (br_if $break (i32.ge_u (local.get $i) (local.get $vn)))
        (v128.store
          (i32.add (local.get $out) (i32.shl (local.get $i) (i32.const 2)))
          (${simdOp}
            (v128.load (i32.add (local.get $a) (i32.shl (local.get $i) (i32.const 2))))
            (v128.load (i32.add (local.get $b) (i32.shl (local.get $i) (i32.const 2))))
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 4)))
        (br $loop)
      )
    )
    (block $tail_break
      (loop $tail
        (br_if $tail_break (i32.ge_u (local.get $i) (local.get $count)))
        (f32.store
          (i32.add (local.get $out) (i32.shl (local.get $i) (i32.const 2)))
          (${scalarOp}
            (f32.load (i32.add (local.get $a) (i32.shl (local.get $i) (i32.const 2))))
            (f32.load (i32.add (local.get $b) (i32.shl (local.get $i) (i32.const 2))))
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $tail)
      )
    )
  )
)`;

/**
 * Dot product: returns sum(a[i] * b[i]). 4-lane SIMD accumulation with a
 * scalar tail; lane order of accumulation differs from a sequential loop,
 * which is why callers must opt in explicitly (see WasmBackend docs).
 */
const DOT_WAT = `(module
  (import "env" "memory" (memory 1))
  (func (export "run") (param $a i32) (param $b i32) (param $count i32) (result f32)
    (local $i i32)
    (local $vn i32)
    (local $acc v128)
    (local $s f32)
    (local.set $vn (i32.and (local.get $count) (i32.const -4)))
    (local.set $acc (f32x4.splat (f32.const 0)))
    (block $break
      (loop $loop
        (br_if $break (i32.ge_u (local.get $i) (local.get $vn)))
        (local.set $acc
          (f32x4.add
            (local.get $acc)
            (f32x4.mul
              (v128.load (i32.add (local.get $a) (i32.shl (local.get $i) (i32.const 2))))
              (v128.load (i32.add (local.get $b) (i32.shl (local.get $i) (i32.const 2))))
            )
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 4)))
        (br $loop)
      )
    )
    (local.set $s
      (f32.add
        (f32.add (f32x4.extract_lane 0 (local.get $acc)) (f32x4.extract_lane 1 (local.get $acc)))
        (f32.add (f32x4.extract_lane 2 (local.get $acc)) (f32x4.extract_lane 3 (local.get $acc)))
      )
    )
    (block $tail_break
      (loop $tail
        (br_if $tail_break (i32.ge_u (local.get $i) (local.get $count)))
        (local.set $s
          (f32.add
            (local.get $s)
            (f32.mul
              (f32.load (i32.add (local.get $a) (i32.shl (local.get $i) (i32.const 2))))
              (f32.load (i32.add (local.get $b) (i32.shl (local.get $i) (i32.const 2))))
            )
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $tail)
      )
    )
    (local.get $s)
  )
)`;

/** Sum reduction with the same accumulation caveats as the dot product. */
const SUM_WAT = `(module
  (import "env" "memory" (memory 1))
  (func (export "run") (param $a i32) (param $count i32) (result f32)
    (local $i i32)
    (local $vn i32)
    (local $acc v128)
    (local $s f32)
    (local.set $vn (i32.and (local.get $count) (i32.const -4)))
    (local.set $acc (f32x4.splat (f32.const 0)))
    (block $break
      (loop $loop
        (br_if $break (i32.ge_u (local.get $i) (local.get $vn)))
        (local.set $acc
          (f32x4.add
            (local.get $acc)
            (v128.load (i32.add (local.get $a) (i32.shl (local.get $i) (i32.const 2))))
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 4)))
        (br $loop)
      )
    )
    (local.set $s
      (f32.add
        (f32.add (f32x4.extract_lane 0 (local.get $acc)) (f32x4.extract_lane 1 (local.get $acc)))
        (f32.add (f32x4.extract_lane 2 (local.get $acc)) (f32x4.extract_lane 3 (local.get $acc)))
      )
    )
    (block $tail_break
      (loop $tail
        (br_if $tail_break (i32.ge_u (local.get $i) (local.get $count)))
        (local.set $s
          (f32.add (local.get $s) (f32.load (i32.add (local.get $a) (i32.shl (local.get $i) (i32.const 2)))))
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $tail)
      )
    )
    (local.get $s)
  )
)`;

const MODULES = {
  simdAdd: binaryWat("f32x4.add", "f32.add"),
  simdSub: binaryWat("f32x4.sub", "f32.sub"),
  simdMul: binaryWat("f32x4.mul", "f32.mul"),
  simdDiv: binaryWat("f32x4.div", "f32.div"),
  simdDot: DOT_WAT,
  simdSum: SUM_WAT,
};

const wabt = await wabtFactory();
const binaries = {};
for (const [name, source] of Object.entries(MODULES)) {
  const mod = wabt.parseWat(`${name}.wat`, source, { simd: true });
  const { buffer } = mod.toBinary({});
  mod.destroy();
  binaries[name] = Buffer.from(buffer).toString("base64");
  console.log(`${name}: ${buffer.length} bytes`);
}

const out = `/**
 * GENERATED FILE. Do not edit by hand.
 *
 * WASM SIMD kernel sources and their compiled binaries, generated by
 * \`node scripts/build-wasm.mjs\` (wabt). The binaries are embedded as
 * base64 so the published package needs no WAT compiler at runtime.
 *
 * @module core/backend
 * @see {@link https://deepbox.dev/docs/devices-and-execution | Devices & execution}
 */

/**
 * Hand-written WAT sources for the SIMD kernels (single source of truth in
 * \`scripts/build-wasm.mjs\`). All modules import a shared memory
 * (\`env.memory\`) and use a 4-lane f32 SIMD main loop with a scalar tail.
 */
export const WAT_MODULES = {
${Object.entries(MODULES)
  .map(([name, src]) => `  ${name}: ${JSON.stringify(src)},`)
  .join("\n")}
} as const;

/** Available WASM module names. */
export type WasmModuleName = keyof typeof WAT_MODULES;

/** Compiled wasm binaries (base64), one per {@link WAT_MODULES} entry. */
export const WASM_BINARIES: Record<WasmModuleName, string> = {
${Object.entries(binaries)
  .map(([name, b64]) => `  ${name}: ${JSON.stringify(b64)},`)
  .join("\n")}
};
`;

const here = dirname(fileURLToPath(import.meta.url));
const target = join(here, "..", "src", "core", "backend", "wasm_modules.generated.ts");
writeFileSync(target, out);
console.log(`wrote ${target}`);
