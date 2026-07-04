# Running Deepbox Projects

> **Browse online:** https://deepbox.dev/projects

This directory contains **9 production-style projects**. Run them from the Deepbox package root.

## Prerequisites

- Node.js `>= 24.13.0`
- `npm install`
- Working directory: the `Deepbox` package root

## Individual Project Commands

```bash
npm run project:01
npm run project:02
npm run project:03
npm run project:04
npm run project:05
npm run project:06
npm run project:07
npm run project:08
npm run project:09
```

## Run Everything

```bash
npm run projects:all
```

## Output Highlights

- `project:01`-`project:06` continue to generate the SVG artifacts already documented in their local READMEs.
- `project:07` writes calibration and feature-importance SVGs plus a JSON model report.
- `project:08` writes a labeled confusion-matrix SVG plus JSON summaries for model comparison and vocabulary preview.
- `project:09` writes rollout scorecards, decision reports, grouped rate charts, and KDE/significance SVGs.

## Troubleshooting

If a project cannot resolve `deepbox/...` imports:

1. Run it from the package root, not from inside `docs/projects/`.
2. Use the npm scripts so `docs/projects/tsconfig.json` aliases are applied.
3. If needed, rebuild first with `npm run build`.

## Development Notes

- Add new project entry points under `docs/projects/0X-name/index.ts`.
- Register a matching `project:0X` script in [`package.json`](../../package.json).
- Update this runbook and [`docs/projects/README.md`](./README.md).

## License

MIT — See the parent directory.
