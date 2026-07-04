# Core Runtime Tooling

> **View online:** https://deepbox.dev/examples/45-core-runtime-tooling

A runtime-focused walkthrough for the v1.0.0 `deepbox/core` surface: structured logging, warning policies, backend registration, and JSON/file serialization.

## Deepbox Modules Used

| Module          | Features Used                                                                 |
| --------------- | ----------------------------------------------------------------------------- |
| `deepbox/core`  | Logger, warnings, save/load, toJSON/fromJSON, CpuBackend, backend registry    |

## Usage

```bash
npm run example:45
```

## Output

- Console walkthrough of logging, warning capture, serialization, and backend registration
- JSON payloads written to `output/`

## Architecture

```text
45-core-runtime-tooling/
├── index.ts
├── README.md
└── output/
```
