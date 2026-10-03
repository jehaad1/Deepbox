# Core Runtime Tooling

> **View online:** https://deepbox.dev/examples/45-core-runtime-tooling

Walks through the runtime tools in `deepbox/core`: the logger, warning filters, JSON and file serialization, and the backend registry with the WASM SIMD backend. The WebGPU and WASM backends accelerate a subset of operations. An operation that a device cannot run throws a `DeviceError` that says to move the tensor with `await t.cpu()`.

## Deepbox Modules Used

| Module         | Features Used                                                                                                                        |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `deepbox/core` | Logger, setLogHandler, warn, filterWarnings, catchWarnings, resetWarnings, save, load, toJSON, fromJSON, WasmBackend, registerBackend, listBackends, isBackendAvailable |

## Usage

```bash
npm run example:45
```

## Output

- Console output: captured log entries, collected warnings, serialization round trips, registered backends and one SIMD addition.
- Two JSON files written to `output/`: `tensor-payload.json` and `module-state.json`.

## Files

```text
45-core-runtime-tooling/
├── index.ts
├── README.md
└── output/
```
