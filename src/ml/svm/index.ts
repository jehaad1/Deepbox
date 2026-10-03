/**
 * Support Vector Machine (SVM) implementations.
 *
 * - Kernel methods: `SVC`, `SVR`, `NuSVC`, `NuSVR`, `OneClassSVM`
 * - Linear methods: `LinearSVC`, `LinearSVR`
 *
 * @see {@link https://deepbox.dev/docs/ml-svm | Deepbox SVM}
 */
export type {
  ClassWeightOption,
  GammaOption,
  KernelType,
  SVCOptions,
  SVROptions,
} from "./KernelSVM";
export { SVC, SVR } from "./KernelSVM";
export type { NuSVCOptions, NuSVROptions, OneClassSVMOptions } from "./NuSVM";
export { NuSVC, NuSVR, OneClassSVM } from "./NuSVM";
export type {
  LinearSVCLoss,
  LinearSVCOptions,
  LinearSVRLoss,
  LinearSVROptions,
} from "./SVM";
export { LinearSVC, LinearSVR } from "./SVM";
