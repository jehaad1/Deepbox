export type FloatArray = Float32Array | Float64Array;

type HasData = { data: unknown };
type HasTensorData = { tensor: { data: unknown } };

function getFloatArray(value: unknown, context: string): FloatArray {
  if (value instanceof Float32Array || value instanceof Float64Array) {
    return value;
  }
  throw new Error(`${context} must be a float32 or float64 array`);
}

export function getTensorData(container: HasData, context: string): FloatArray {
  return getFloatArray(container.data, context);
}

export function getParamData(container: HasTensorData, context: string): FloatArray {
  return getFloatArray(container.tensor.data, context);
}

export function getTensorValue(container: HasData, index: number, context: string): number {
  const data = getTensorData(container, context);
  const value = data[index];
  if (value === undefined) {
    throw new Error(`${context} missing value at index ${index}`);
  }
  return value;
}

export function getParamValue(container: HasTensorData, index: number, context: string): number {
  const data = getParamData(container, context);
  const value = data[index];
  if (value === undefined) {
    throw new Error(`${context} missing value at index ${index}`);
  }
  return value;
}
