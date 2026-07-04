/**
 * Global configuration exports for Deepbox Core.
 *
 * This file is a barrel that re-exports the global configuration type and helpers.
 * @see {@link https://deepbox.dev/docs/core-config | Config & backends}
 */

export type { DeepboxConfig } from "./global";
export {
  getConfig,
  getDevice,
  getDtype,
  getSeed,
  resetConfig,
  setConfig,
  setDevice,
  setDtype,
  setSeed,
} from "./global";
