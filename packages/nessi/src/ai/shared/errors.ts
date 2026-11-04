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

/** `retry-after-ms` (OpenAI) wins over the standard `retry-after` seconds or HTTP date. */
export const parseRetryAfterMs = (headers: Headers, now = Date.now()): number | undefined => {
  const milliseconds = Number(headers.get("retry-after-ms")?.trim() || Number.NaN);
  if (Number.isFinite(milliseconds) && milliseconds >= 0) return milliseconds;
  const value = headers.get("retry-after")?.trim();
  if (!value) return undefined;
  const seconds = Number(value);
  if (Number.isFinite(seconds)) return seconds >= 0 ? seconds * 1_000 : undefined;
  const date = Date.parse(value);
  return Number.isFinite(date) ? Math.max(0, date - now) : undefined;
};

export const normalizeHttpError = async (
  label: string,
  response: Response,
  retryableOverride?: boolean,
) => {
  const rawText = await response.text().catch(() => "");
  const { message, code } = parseErrorMessage(rawText);
  const fullMessage = code ? `${message} (code: ${code})` : message || `HTTP ${response.status}`;
  const contextOverflow = isContextOverflow(response.status, fullMessage);
  return {
    error: `${label} ${response.status}: ${fullMessage}`,
    retryable: retryableOverride ?? isRetryableStatus(response.status),
    status: response.status,
    ...(code !== undefined ? { code } : {}),
    ...withRetryAfter(parseRetryAfterMs(response.headers)),
    contextOverflow,
    overflowRatio: contextOverflow ? parseOverflowRatio(fullMessage) : undefined,
  };
};

const withRetryAfter = (retryAfterMs: number | undefined) => (retryAfterMs !== undefined ? { retryAfterMs } : {});

export const formatConnectionError = (label: string, error: unknown) =>
  `${label} connection failed: ${error instanceof Error ? error.message : String(error)}`;

const errorCode = (error: unknown) =>
  typeof error === "object" && error !== null && "code" in error && typeof error.code === "string" ? error.code : undefined;

/**
 * A request whose connection failed; `code` is the runtime's, such as `ECONNREFUSED`. Bun sets it
 * on the error, Node on its `cause`.
 */
export const normalizeConnectionError = (label: string, error: unknown) => {
  const code = errorCode(error) ?? (error instanceof Error ? errorCode(error.cause) : undefined);
  return { error: formatConnectionError(label, error), retryable: true, ...(code !== undefined ? { code } : {}) };
};

export type ProviderErrorDetails = {
  /** `timeout` when a provider deadline ended the request; `provider_error` otherwise. */
  kind: "provider_error" | "timeout";
  retryable: boolean;
  /** HTTP status of a rejected request. */
  status?: number;
  /** Provider error code from the body, or the runtime's connection error code. */
  code?: string;
  /** How long the provider asked the caller to wait (`retry-after-ms` or `retry-after`). */
  retryAfterMs?: number;
  contextOverflow?: boolean;
  overflowRatio?: number;
  /** The deadline that ended the request, when `kind` is `timeout`. */
  scope?: "provider_first_byte" | "provider_idle" | "provider_total";
};

/**
 * A failed provider request from `complete()` or `completeFromStream()`. It carries what a caller
 * needs to decide on a retry; nessi itself never retries. Caller aborts throw the signal's reason.
 */
export class ProviderError extends Error implements ProviderErrorDetails {
  override readonly name = "ProviderError";
  readonly kind: ProviderErrorDetails["kind"];
  readonly retryable: boolean;
  readonly status?: number;
  readonly code?: string;
  readonly retryAfterMs?: number;
  readonly contextOverflow?: boolean;
  readonly overflowRatio?: number;
  readonly scope?: ProviderErrorDetails["scope"];

  constructor(message: string, details: ProviderErrorDetails, options?: { cause?: unknown }) {
    super(message, options);
    this.kind = details.kind;
    this.retryable = details.retryable;
    if (details.status !== undefined) this.status = details.status;
    if (details.code !== undefined) this.code = details.code;
    if (details.retryAfterMs !== undefined) this.retryAfterMs = details.retryAfterMs;
    if (details.contextOverflow !== undefined) this.contextOverflow = details.contextOverflow;
    if (details.overflowRatio !== undefined) this.overflowRatio = details.overflowRatio;
    if (details.scope !== undefined) this.scope = details.scope;
  }
}

/** Throwable form of `normalizeHttpError()` or `normalizeConnectionError()`. */
export const providerError = (
  { error, ...details }: { error: string } & Omit<ProviderErrorDetails, "kind">,
  options?: { cause?: unknown },
) => new ProviderError(error, { kind: "provider_error", ...details }, options);

export const invalidResponseError = (label: string) =>
  new ProviderError(`${label} returned invalid JSON.`, { kind: "provider_error", retryable: true });

export const streamEndedError = (label: string) => ({
  type: "error" as const,
  error: `${label} stream ended unexpectedly before completion.`,
  retryable: true,
});
