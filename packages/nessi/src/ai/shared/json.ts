export const safeJsonParse = <T>(input: string): T | null => {
  try {
    return JSON.parse(input) as T;
  } catch {
    return null;
  }
};

export const stringifyJson = (value: unknown) =>
  typeof value === "string" ? value : JSON.stringify(value);

export const ensureRecord = (value: unknown): Record<string, unknown> => {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return {};
};


/**
 * Streamed tool arguments as text for the normalizer, which checks the completed text. Servers
 * that send objects instead of JSON strings keep their value; absent or empty input stays empty.
 */
export const toolArgumentsText = (value: unknown): string => {
  if (value === undefined || value === null) return "";
  if (typeof value === "string") return value;
  if (typeof value === "object" && !Array.isArray(value) && Object.keys(value).length === 0) return "";
  return JSON.stringify(value);
};


/** A stream record or response body: a JSON object, or `null` for anything else. */
export const parseJsonRecord = <T>(input: string): T | null => {
  const value = safeJsonParse<unknown>(input);
  return value && typeof value === "object" && !Array.isArray(value) ? value as T : null;
};

export const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value);
