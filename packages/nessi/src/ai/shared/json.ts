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
 * Tool arguments as an executable object. Absent arguments mean a call without input (`{}`);
 * supplied arguments that are not a JSON object return `undefined`, so they never run a tool
 * with its input defaults.
 */
export const parseToolArguments = (value: unknown): Record<string, unknown> | undefined => {
  if (value === undefined || value === null) return {};
  if (typeof value === "string") {
    if (!value.trim()) return {};
    const parsed = safeJsonParse<unknown>(value);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? parsed as Record<string, unknown>
      : undefined;
  }
  return typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : undefined;
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

export const malformedToolArgumentsError = (label: string, name: string) =>
  new Error(`${label} returned malformed JSON arguments for "${name}".`);
