# Deepbox Docs

This directory holds the documentation that ships with the repository, written for Deepbox 1.5.0.

## What Lives Here

- [`examples`](./examples/README.md): numbered, runnable examples that go from tensor basics to advanced workflows. Each one is a single `index.ts` with a README.
- [`projects`](./projects/README.md): larger end-to-end projects that combine several Deepbox modules.

The full guides and API reference are at [deepbox.dev/docs](https://deepbox.dev/docs).

## Running the Code

Examples and projects import from the package subpaths (`deepbox/ndarray`, `deepbox/ml` and so on). Their `tsconfig.json` files map those paths to `src/`, so you can run them from a clone without building:

```bash
npx tsx --tsconfig docs/examples/tsconfig.json docs/examples/00-quick-start/index.ts
npm run typecheck:docs
```

[`examples/RUNNING_EXAMPLES.md`](./examples/RUNNING_EXAMPLES.md) and [`projects/RUNNING_PROJECTS.md`](./projects/RUNNING_PROJECTS.md) list the per-folder commands.

## Recommended Path

1. Start with [`docs/examples/00-quick-start`](./examples/00-quick-start/README.md).
2. Use [`docs/examples/README.md`](./examples/README.md) to pick an example track for the module you need.
3. Move to [`docs/projects/README.md`](./projects/README.md) when you want patterns for a complete application.

## Writing Rules

Text in this directory follows the rules in [CONTRIBUTING.md](../CONTRIBUTING.md#writing-rules): plain wording, no em dashes, no emoji. `npm run prose:check` fails on em dashes.
