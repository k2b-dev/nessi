import { safeJsonParse } from "./json.js";

const parseErrorMessage = (rawText: string) => {
  const parsed = safeJsonParse<Record<string, unknown>>(rawText);
  if (!parsed) return { message: rawText };

  const nestedError = parsed.error as Record<string, unknown> | undefined;
  const message =
    (typeof nestedError?.message === "string" && nestedError.message) ||
    (typeof parsed.message === "string" && parsed.message) ||
    rawText;
  const code =
    (typeof nestedError?.code === "string" && nestedError.code) ||
    (typeof parsed.code === "string" && parsed.code) ||
    undefined;

  return { message, code };
};

export const isRetryableStatus = (status: number) =>
  status === 408 || status === 409 || status === 425 || status === 429 || status >= 500;

const CONTEXT_OVERFLOW_PATTERNS = [
  /context[ _](length|window|size)/,
  /maximum (context|model) length/,
  /(prompt|input|context|messages?) (is )?too long/,
  /max(imum)? tokens exceeded/,
  /token limit/,
  /reduce the length/,
  /exceeds the maximum number of tokens/,
  /too large for model/,
];

export const isContextOverflow = (status: number, message: string) => {
  if (status !== 400 && status !== 413 && status !== 422) return false;
  const lower = message.toLowerCase();
  return CONTEXT_OVERFLOW_PATTERNS.some((pattern) => pattern.test(lower));
};

const toInt = (value: string) => parseInt(value.replace(/,/g, ""), 10);

/**
 * Try to extract an overflow ratio from a provider error message.
 * Many providers include token counts like:
 * "maximum context length is 32768 tokens ... your prompt contains 45000 tokens"
 * Anthropic uses "prompt is too long: 210000 tokens > 200000 maximum".
 */
export const parseOverflowRatio = (message: string): number | undefined => {
  const anthropicMatch = message.match(/([\d,]+)\s*tokens\s*>\s*([\d,]+)\s*maximum/i);
  if (anthropicMatch?.[1] && anthropicMatch[2]) {
    const max = toInt(anthropicMatch[2]);
    return max > 0 ? toInt(anthropicMatch[1]) / max : undefined;
  }
  const maxMatch = message.match(
    /(?:maximum|max)\s+(?:context\s+)?(?:length|window|limit)\s+(?:is|of)\s+([\d,]+)/i,
  );
  const actualMatch = message.match(
    /(?:resulted?\s+in|contains?\s+(?:at\s+least\s+)?|received|totale?\s+of\s+(?:at\s+least\s+)?)\s*([\d,]+)\s*(?:tokens|input)/i,
  );
  if (!maxMatch?.[1] || !actualMatch?.[1]) return undefined;
  const max = toInt(maxMatch[1]);
  const actual = toInt(actualMatch[1]);
  return max > 0 ? actual / max : undefined;
};

/** An error body only feeds the error message; anything past this budget is dropped. */
export const ERROR_BODY_LIMIT_BYTES = 64 * 1024;

export type ErrorBodyLimits = {
  /** Gives up when one read stalls this long. */
  idleMs?: number;
  /** Gives up at this time (`Date.now()` milliseconds), however the body flows. */
  deadlineAt?: number;
};

/**
 * Reads at most `ERROR_BODY_LIMIT_BYTES` of an error body. The status is already known, so a
 * stalled, failed, or oversized body still yields the error with what arrived.
 */
export const readErrorBody = async (response: Response, limits: ErrorBodyLimits = {}): Promise<string> => {
  const reader = response.body?.getReader();
  if (!reader) return "";
  const decoder = new TextDecoder();
  let text = "";
  let bytes = 0;
  let truncated = false;
  let timeout: ReturnType<typeof setTimeout> | undefined;
  try {
    while (true) {
      const waits = [limits.idleMs, limits.deadlineAt === undefined ? undefined : limits.deadlineAt - Date.now()]
        .filter((ms): ms is number => ms !== undefined && Number.isFinite(ms));
      const waitMs = waits.length > 0 ? Math.min(...waits) : undefined;
      if (waitMs !== undefined && waitMs <= 0) break;
      const stalled = new Promise<"stalled">((resolve) => {
        if (waitMs !== undefined) timeout = setTimeout(() => resolve("stalled"), waitMs);
      });
      const result = await Promise.race([reader.read(), stalled]);
      clearTimeout(timeout);
      if (result === "stalled" || result.done) break;
      const chunk = result.value.subarray(0, ERROR_BODY_LIMIT_BYTES - bytes);
      bytes += chunk.byteLength;
      text += decoder.decode(chunk, { stream: true });
      if (chunk.byteLength < result.value.byteLength || bytes >= ERROR_BODY_LIMIT_BYTES) {
        truncated = true;
        break;
      }
    }
  } catch {
    // A failed body read leaves the diagnostics read so far.
  } finally {
    clearTimeout(timeout);
    void reader.cancel().catch(() => {});
  }
  // A cut body drops its incomplete last character instead of flushing a replacement character.
  return truncated ? text : text + decoder.decode();
};

export const normalizeHttpError = async (
  label: string,
  response: Response,
  retryableOverride?: boolean,
  limits?: ErrorBodyLimits,
) => {
  const rawText = await readErrorBody(response, limits);
  const { message, code } = parseErrorMessage(rawText);
  const fullMessage = code ? `${message} (code: ${code})` : message || `HTTP ${response.status}`;
  const contextOverflow = isContextOverflow(response.status, fullMessage);
  return {
    error: `${label} ${response.status}: ${fullMessage}`,
    retryable: retryableOverride ?? isRetryableStatus(response.status),
    contextOverflow,
    overflowRatio: contextOverflow ? parseOverflowRatio(fullMessage) : undefined,
  };
};

export const formatConnectionError = (label: string, error: unknown) =>
  `${label} connection failed: ${error instanceof Error ? error.message : String(error)}`;

export const streamEndedError = (label: string) => ({
  type: "error" as const,
  error: `${label} stream ended unexpectedly before completion.`,
  retryable: true,
});
