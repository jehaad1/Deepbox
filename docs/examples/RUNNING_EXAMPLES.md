# Running Deepbox Examples

> **Browse online:** https://deepbox.dev/examples

This directory contains **50 runnable examples (00-49)**. Execute them from the Deepbox package root.

## Prerequisites

- Node.js `>= 24.13.0`
- `npm install`
- Working directory: the `Deepbox` package root

## Individual Example Commands

```bash
# Foundations
npm run example:00
npm run example:01
npm run example:02
npm run example:03
npm run example:04
npm run example:05

# Classical ML
npm run example:06
npm run example:07
npm run example:08
npm run example:09
npm run example:10
npm run example:11
npm run example:12

# Neural nets and optimization
npm run example:13
npm run example:14
npm run example:15
npm run example:16
npm run example:27
npm run example:28
npm run example:29
npm run example:30
npm run example:32
npm run example:39
npm run example:40
npm run example:41

# Data, statistics, and math
npm run example:17
npm run example:18
npm run example:19
npm run example:20
npm run example:21
npm run example:22
npm run example:23
npm run example:24
npm run example:26
npm run example:31
npm run example:33
npm run example:35
npm run example:37
npm run example:38
npm run example:42
npm run example:43
npm run example:48
npm run example:49

# Specialized v1.0 additions
npm run example:25
npm run example:34
npm run example:36
npm run example:44
npm run example:45
npm run example:46
npm run example:47
```

## Run Everything

```bash
npm run examples:all
```

## Notable Outputs

- `example:03`, `example:12`, `example:15`, `example:25`, and `example:44` render SVG charts.
- `example:45` writes serialized JSON payloads under `docs/examples/45-core-runtime-tooling/output/`.
- `example:47` writes JSON, XLSX, Parquet, HTML, ANSI, and SVG artifacts under `docs/examples/47-dataframe-io-styling/output/`.
- `example:48` writes inference-oriented SVG artifacts under `docs/examples/48-statistical-inference-playbook/output/`.

## Troubleshooting

If you hit module resolution errors:

1. Run from the Deepbox package root, not from inside `docs/examples/`.
2. Use the provided npm scripts so `docs/examples/tsconfig.json` path aliases are applied.
3. If you need a clean rebuild first, run `npm run build`.

## Development Notes

- Add new example entry points under `docs/examples/NN-name/index.ts`.
- Register a matching `example:NN` script in [`package.json`](../../package.json).
- Update this runbook and [`docs/examples/README.md`](./README.md).

## License

MIT. See the LICENSE file in the repository root.
