# Running Deepbox Projects

> **Browse online:** https://deepbox.dev/projects

This directory holds nine projects. Run them from the Deepbox package root.

## Prerequisites

- Node.js `>= 24.13.0`
- `npm install`
- Working directory: the Deepbox package root

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

Each script runs `tsx --tsconfig docs/projects/tsconfig.json docs/projects/<name>/index.ts`. The tsconfig maps `deepbox` and `deepbox/*` imports to `src/`, so no build is needed.

## Run Everything

```bash
npm run projects:all
```

## Type Check

```bash
npm run typecheck:docs
```

This checks the examples and the projects. To check only the projects, run `npx tsc --noEmit -p docs/projects/tsconfig.json`.

## Output

Every run rewrites the files in the project's `output/` folder.

- `project:01` to `project:06` write SVG charts, listed in each project README.
- `project:07` writes a calibration curve, a feature-importance chart and `model-report.json`.
- `project:08` writes a confusion-matrix SVG and two JSON files: the model comparison and a preview of the TF-IDF vocabulary.
- `project:09` writes two JSON reports (variant scorecard and decision report) and three SVG charts.

## Troubleshooting

If a project cannot resolve `deepbox/...` imports:

1. Run it from the package root, not from inside `docs/projects/`.
2. Use the npm scripts, or pass `--tsconfig docs/projects/tsconfig.json` to `tsx`, so the path aliases apply.

## Development Notes

- Add new project entry points under `docs/projects/0X-name/index.ts`.
- Register a matching `project:0X` script in [`package.json`](../../package.json).
- Update this runbook and [`docs/projects/README.md`](./README.md).

## License

MIT. See the parent directory.
