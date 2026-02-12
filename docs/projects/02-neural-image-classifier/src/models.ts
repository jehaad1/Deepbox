/**
 * Neural Network Model Definitions
 *
 * Provides model creation utilities and architecture configurations
 * for the image classifier project.
 */

import { InvalidParameterError } from "deepbox/core";
import { Dropout, GELU, LeakyReLU, Linear, ReLU, Sequential } from "deepbox/nn";

/**
 * Model configuration options
 */
export interface ModelConfig {
  inputSize: number;
  hiddenSize: number;
  numClasses: number;
  architecture: "simple" | "gelu" | "leaky";
}

/**
 * Create a simple MLP with ReLU activation
 */
export function createSimpleMLP(
  inputSize: number,
  hiddenSize: number,
  numClasses: number
): Sequential {
  return new Sequential(
    new Linear(inputSize, hiddenSize),
    new ReLU(),
    new Dropout(0.2),
    new Linear(hiddenSize, numClasses)
  );
}

/**
 * Create a model based on configuration
 */
export function createModel(config: ModelConfig): Sequential {
  const { inputSize, hiddenSize, numClasses, architecture } = config;

  switch (architecture) {
    case "simple":
      return new Sequential(
        new Linear(inputSize, hiddenSize),
        new ReLU(),
        new Dropout(0.2),
        new Linear(hiddenSize, numClasses)
      );

    case "gelu":
      return new Sequential(
        new Linear(inputSize, hiddenSize),
        new GELU(),
        new Dropout(0.2),
        new Linear(hiddenSize, numClasses)
      );

    case "leaky":
      return new Sequential(
        new Linear(inputSize, hiddenSize),
        new LeakyReLU(0.01),
        new Dropout(0.2),
        new Linear(hiddenSize, numClasses)
      );

    default:
      throw new InvalidParameterError(
        `Unknown architecture: ${architecture}`,
        "architecture",
        architecture
      );
  }
}

/**
 * Get model summary statistics
 */
export function getModelSummary(model: Sequential): {
  numLayers: number;
  numParameters: number;
} {
  const params = Array.from(model.parameters());

  let totalParams = 0;
  for (const param of params) {
    const size = param.tensor.shape.reduce((a, b) => a * b, 1);
    totalParams += size;
  }

  return {
    numLayers: model.length,
    numParameters: totalParams,
  };
}
