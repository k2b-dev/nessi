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

/** A stream record or response body: a JSON object, or `null` for anything else. */
export const parseJsonRecord = <T>(input: string): T | null => {
  const value = safeJsonParse<unknown>(input);
  return value && typeof value === "object" && !Array.isArray(value) ? value as T : null;
};
