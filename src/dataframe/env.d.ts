/**
 * Placeholder kept for tooling that skips this file by name.
 *
 * This file used to declare minimal `process`, `fetch`, `document`, `Blob` and
 * `URL` globals. They were not needed (the file I/O helpers in `./io` read
 * `globalThis` instead, and the project tsconfig already includes the DOM and
 * Node.js type libraries) and they hid the real Node.js `process` type from
 * every file in the program, so `process.env` did not type-check. Nothing in
 * `src` depends on them any more.
 *
 * @see {@link https://deepbox.dev/docs/dataframe-overview | Deepbox documentation}
 */

export {};
