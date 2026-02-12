/**
 * Global configuration exports for Deepbox Core.
 *
 * This file is a barrel that re-exports the global configuration type and helpers.
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
