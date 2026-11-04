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

export const malformedToolArgumentsError = (label: string, name: string) =>
  new Error(`${label} returned malformed JSON arguments for "${name}".`);
