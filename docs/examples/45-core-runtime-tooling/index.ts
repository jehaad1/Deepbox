/**
 * Example 45: Core Runtime Tooling
 *
 * Focuses on v1.0.0 runtime primitives in `deepbox/core`: logger,
 * warnings, backend registry, and JSON/file serialization.
 */

import { mkdir } from "node:fs/promises";
import {
  catchWarnings,
  filterWarnings,
  fromJSON,
  isBackendAvailable,
  Logger,
  listBackends,
  load,
  registerBackend,
  resetWarnings,
  save,
  setLogHandler,
  toJSON,
  WasmBackend,
  warn,
} from "deepbox/core";

const OUTPUT_DIR = "docs/examples/45-core-runtime-tooling/output";

console.log("=".repeat(60));
console.log("Example 45: Core Runtime Tooling");
console.log("=".repeat(60));

await mkdir(OUTPUT_DIR, { recursive: true });

// ============================================================================
// Part 1: Structured logging
// ============================================================================
console.log("\n🪵 Part 1: Logger");
console.log("-".repeat(60));

const capturedLogs: string[] = [];
setLogHandler((entry) => {
  capturedLogs.push(
    `L${entry.level} @ ${new Date(entry.timestamp).toISOString()} :: ${entry.message}`
  );
});

const logger = new Logger(2, "Example45");
logger.info("Starting serialization and backend checks");
logger.debug("Using verbosity level 2 to emit summary + progress events");
logger.trace("This trace entry is recorded but not emitted at level 2");

console.log(`Captured log entries: ${capturedLogs.length}`);
for (const line of capturedLogs) {
  console.log(`  ${line}`);
}
console.log(`Recorded entries (including trace): ${logger.getEntries().length}`);

setLogHandler(undefined);

// ============================================================================
// Part 2: Warning filtering and collection
// ============================================================================
console.log("\n⚠️  Part 2: Warnings");
console.log("-".repeat(60));

resetWarnings();
filterWarnings("once", {
  category: "ConvergenceWarning",
  message: /max iterations/i,
});

const warnings = catchWarnings(() => {
  warn("solver hit max iterations", "ConvergenceWarning", "Example45");
  warn("solver hit max iterations", "ConvergenceWarning", "Example45");
  warn("probabilities were clipped into [0, 1]", "DataConversionWarning", "Example45");
});

console.log(`Warnings collected after applying 'once' filter: ${warnings.length}`);
for (const warning of warnings) {
  console.log(`  [${warning.category}] ${warning.message}`);
}
resetWarnings();

// ============================================================================
// Part 3: In-memory and file serialization
// ============================================================================
console.log("\n💾 Part 3: Serialization");
console.log("-".repeat(60));

const tensorPayload = {
  __type: "Tensor" as const,
  data: [1.5, 2.5, 3.5, 4.5],
  shape: [2, 2],
  dtype: "float64",
};

const modulePayload = {
  __type: "ModuleState" as const,
  parameters: {
    "encoder.weight": {
      data: [0.1, 0.2, 0.3, 0.4],
      dtype: "float32",
      shape: [2, 2],
    },
  },
  buffers: {
    running_mean: {
      data: [0.0, 0.0],
      dtype: "float32",
      shape: [2],
    },
  },
};

const tensorJson = toJSON(tensorPayload);
const restoredTensor = fromJSON(tensorJson);
console.log(`Tensor payload JSON length: ${tensorJson.length} chars`);
if (restoredTensor.__type === "Tensor") {
  console.log(`  Restored tensor shape: [${restoredTensor.shape.join(", ")}]`);
}

const tensorPath = `${OUTPUT_DIR}/tensor-payload.json`;
const modulePath = `${OUTPUT_DIR}/module-state.json`;

await save(tensorPath, tensorPayload);
await save(modulePath, modulePayload);

const loadedTensor = await load(tensorPath);
const loadedModule = await load(modulePath);

console.log(`Saved tensor payload: ${tensorPath}`);
console.log(`Saved module state:   ${modulePath}`);
console.log(`Loaded payload types: ${loadedTensor.__type}, ${loadedModule.__type}`);

// ============================================================================
// Part 4: Backend registry
// ============================================================================
console.log("\n🧠 Part 4: Backend Registry");
console.log("-".repeat(60));

console.log(`Backends before registration: ${listBackends().join(", ")}`);
console.log(`Is WebGPU available? ${isBackendAvailable("webgpu") ? "yes" : "no"}`);
console.log(`Is WASM available?   ${isBackendAvailable("wasm") ? "yes" : "no"}`);

// The WASM SIMD backend ships precompiled kernels: init() instantiates
// them, and once registered, contiguous float32 arithmetic on
// `wasm`-device tensors runs through 4-lane SIMD.
const wasm = new WasmBackend();
await wasm.init();
if (wasm.info().available) {
  registerBackend("wasm", wasm);
}

console.log(`Backends after registration:  ${listBackends().join(", ")}`);
console.log(`Is WASM available now? ${isBackendAvailable("wasm") ? "yes" : "no"}`);
console.log(`WASM SIMD kernels: ${wasm.listModules().join(", ")}`);

const simdA = new Float32Array([1, 2, 3, 4, 5]);
const simdB = new Float32Array([10, 20, 30, 40, 50]);
const simdOut = wasm.binaryContiguous("add", simdA, simdB);
console.log(`SIMD add result: [${simdOut ? Array.from(simdOut).join(", ") : "unavailable"}]`);

// ============================================================================
// Summary
// ============================================================================
console.log("\n💡 Key Takeaways");
console.log("-".repeat(60));
console.log("• Logger captures structured events independently from console output");
console.log("• Warning filters let you silence, dedupe, or escalate numerical issues");
console.log("• Serialization helpers round-trip Deepbox payloads in memory or on disk");
console.log(
  "• The backend registry makes CPU mandatory; WebGPU/WASM backends plug in real kernels"
);

console.log("\n✅ Core Runtime Tooling Example Complete!");
console.log("=".repeat(60));
