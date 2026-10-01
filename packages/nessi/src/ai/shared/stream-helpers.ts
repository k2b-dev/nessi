import { formatConnectionError, normalizeHttpError } from "./errors.js";
import { parseSSE, SSETimeoutError } from "./sse.js";
import type { SSEEvent } from "./sse.js";
import type { ProviderTimeouts, RawStreamEvent } from "../types.js";

type SSEStreamResult =
  | { ok: true; events: AsyncGenerator<SSEEvent> }
  | { ok: false; error: Extract<RawStreamEvent, { type: "error" | "timeout" }> };

export const openSSEStream = async (
  url: string,
  headers: Record<string, string>,
  body: unknown,
  label: string,
  signal?: AbortSignal,
  contextWindow?: number,
  timeouts?: ProviderTimeouts,
): Promise<SSEStreamResult> => {
  const serializedBody = JSON.stringify(body);
  let response: Response;
  const controller = new AbortController();
  const abortExternal = () => controller.abort(signal?.reason);
  const cleanupExternalAbort = () => {
    if (signal) signal.removeEventListener("abort", abortExternal);
  };
  if (signal) {
    if (signal.aborted) controller.abort(signal.reason);
    else signal.addEventListener("abort", abortExternal, { once: true });
  }
  let firstByteTimeout: ReturnType<typeof setTimeout> | undefined;
  const firstByteDeadline = timeouts?.firstByteMs && timeouts.firstByteMs > 0
    ? Date.now() + timeouts.firstByteMs
    : undefined;
  try {
    response = await Promise.race([
      fetch(url, {
        method: "POST",
        headers,
        body: serializedBody,
        signal: controller.signal,
      }),
      new Promise<never>((_, reject) => {
        if (!timeouts?.firstByteMs || timeouts.firstByteMs <= 0) return;
        firstByteTimeout = setTimeout(() => {
          controller.abort();
          reject(new SSETimeoutError("provider_first_byte", timeouts.firstByteMs!));
        }, timeouts.firstByteMs);
      }),
    ]);
  } catch (error) {
    cleanupExternalAbort();
    if (error instanceof SSETimeoutError) {
      return {
        ok: false,
        error: { type: "timeout", scope: error.scope, message: error.message, retryable: true },
      };
    }
    // Heuristic: if the request body is large relative to the context window,
    // a network error likely means the server rejected it for context overflow
    // (browsers hide the actual HTTP 400 body behind CORS on error responses).
    // Browsers surface that as a TypeError; aborts never count.
    const estimatedTokens = serializedBody.length / 4;
    const isLikelyOverflow = !controller.signal.aborted
      && error instanceof TypeError
      && typeof contextWindow === "number"
      && contextWindow > 0
      && estimatedTokens > contextWindow * 0.85;
    if (isLikelyOverflow) {
      const ratio = estimatedTokens / contextWindow;
      return {
        ok: false,
        error: {
          type: "error",
          error: `${label}: context window likely exceeded (~${Math.round(estimatedTokens)} tokens estimated, limit ${contextWindow})`,
          retryable: false,
          contextOverflow: true,
          overflowRatio: ratio,
        },
      };
    }
    return {
      ok: false,
      error: { type: "error", error: formatConnectionError(label, error), retryable: true },
    };
  } finally {
    if (firstByteTimeout) clearTimeout(firstByteTimeout);
  }

  if (!response.ok) {
    const normalized = await normalizeHttpError(label, response);
    cleanupExternalAbort();
    return { ok: false, error: { type: "error", ...normalized } };
  }

  const reader = response.body?.getReader() as ReadableStreamDefaultReader<Uint8Array> | undefined;
  if (!reader) {
    cleanupExternalAbort();
    return {
      ok: false,
      error: { type: "error", error: `${label} response body missing`, retryable: false },
    };
  }

  const streamTimeouts = timeouts ? { ...timeouts } : undefined;
  if (firstByteDeadline && streamTimeouts) {
    streamTimeouts.firstByteMs = Math.max(1, firstByteDeadline - Date.now());
  }

  const events = async function* (): AsyncGenerator<SSEEvent> {
    try {
      yield* parseSSE(reader, streamTimeouts);
    } finally {
      cleanupExternalAbort();
      controller.abort();
      await reader.cancel().catch(() => {});
    }
  };

  return { ok: true, events: events() };
};
