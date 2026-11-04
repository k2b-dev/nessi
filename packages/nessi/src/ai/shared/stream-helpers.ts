import { assertHeaderValues, errorEnvelope, normalizeConnectionError, normalizeHttpError, parseRetryAfterMs, readErrorBody, runtimeErrorCode, serializeRequest } from "./errors.js";
import { parseJsonRecord } from "./json.js";
import { parseSSE, SSETimeoutError } from "./sse.js";
import type { SSEEvent } from "./sse.js";
import type { ProviderTimeouts, RawStreamEvent, TransportReport } from "../types.js";

type SSEStreamResult =
  | { ok: true; events: AsyncGenerator<SSEEvent> }
  | { ok: false; error: Extract<RawStreamEvent, { type: "error" | "timeout" }> };

type RuntimeGlobals = {
  navigator?: { userAgent?: string };
  process?: { versions?: { node?: string } };
  Bun?: unknown;
  Deno?: unknown;
};

/**
 * True in browsers and their web workers, whose user agents start with "Mozilla/". Node, Bun,
 * Deno and edge runtimes such as Cloudflare Workers report their own names.
 */
export const isBrowserRuntime = (runtime: RuntimeGlobals = globalThis as RuntimeGlobals) =>
  /^Mozilla\//.test(runtime.navigator?.userAgent ?? "")
  && runtime.process?.versions?.node === undefined
  && runtime.Bun === undefined
  && runtime.Deno === undefined;

export const openSSEStream = async (
  url: string,
  headers: Record<string, string>,
  body: unknown,
  label: string,
  signal?: AbortSignal,
  contextWindow?: number,
  timeouts?: ProviderTimeouts,
  onTransport?: (report: TransportReport) => void,
): Promise<SSEStreamResult> => {
  // An already aborted caller sends nothing; the stream reports the abort with the caller's reason.
  signal?.throwIfAborted();
  const serializedBody = serializeRequest(label, body);
  assertHeaderValues(label, headers);
  const startedAt = Date.now();
  let reported = false;
  const report = (fields: Omit<TransportReport, "startedAt">) => {
    if (reported || !onTransport) return;
    reported = true;
    try {
      onTransport({ startedAt, ...fields });
    } catch {
      // Observation only: a failing hook must not change the request's outcome.
    }
  };
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
    const request = fetch(url, {
      method: "POST",
      headers,
      body: serializedBody,
      // Credential headers such as x-api-key would follow a redirect to another origin.
      redirect: "error",
      signal: controller.signal,
    });
    // A fetch that ignores the signal can still answer after the deadline; release that body.
    void request.then((late) => {
      if (controller.signal.aborted) void late.body?.cancel().catch(() => {});
    }, () => {});
    response = await Promise.race([
      request,
      new Promise<never>((_, reject) => {
        if (!timeouts?.firstByteMs || timeouts.firstByteMs <= 0) return;
        firstByteTimeout = setTimeout(() => {
          // Reject first: the aborted fetch would otherwise win the race as a connection error.
          const timeout = new SSETimeoutError("provider_first_byte", timeouts.firstByteMs!);
          reject(timeout);
          controller.abort(timeout);
        }, timeouts.firstByteMs);
      }),
    ]);
  } catch (error) {
    cleanupExternalAbort();
    const connectionCode = error instanceof SSETimeoutError || controller.signal.aborted ? undefined : runtimeErrorCode(error);
    report(connectionCode !== undefined ? { connectionCode } : {});
    if (error instanceof SSETimeoutError) {
      return {
        ok: false,
        error: { type: "timeout", scope: error.scope, message: error.message, retryable: true },
      };
    }
    // Heuristic: if the request body is large relative to the context window,
    // a network error likely means the server rejected it for context overflow
    // (browsers hide the actual HTTP 400 body behind CORS on error responses).
    // Browsers surface that as a TypeError; aborts never count. Server runtimes report refused
    // connections as a TypeError too, so the guess only applies in browsers.
    const estimatedTokens = serializedBody.length / 4;
    const isLikelyOverflow = !controller.signal.aborted
      && isBrowserRuntime()
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
      error: { type: "error", ...normalizeConnectionError(label, error) },
    };
  } finally {
    if (firstByteTimeout) clearTimeout(firstByteTimeout);
  }

  const headersAt = Date.now();
  const retryAfterMs = parseRetryAfterMs(response.headers);
  let firstByteAt: number | undefined;
  const noteFirstByte = () => { firstByteAt ??= Date.now(); };
  const answered = () => ({
    headersAt,
    status: response.status,
    ...(retryAfterMs !== undefined ? { retryAfterMs } : {}),
    ...(firstByteAt !== undefined ? { firstByteAt } : {}),
  });
  // An abort settles the request now, not when the consumer next reads the stream.
  controller.signal.addEventListener("abort", () => report(answered()), { once: true });
  const bodyLimits = { idleMs: timeouts?.idleMs, deadlineAt: firstByteDeadline, signal: controller.signal, onFirstByte: noteFirstByte };

  if (!response.ok) {
    // Startup timers are cleared by now; a stalled error body must not hang the request.
    const normalized = await normalizeHttpError(label, response, undefined, bodyLimits);
    cleanupExternalAbort();
    report(answered());
    return { ok: false, error: { type: "error", ...normalized } };
  }

  // Some gateways answer a stream request with a plain JSON body, usually an error envelope.
  if (response.headers.get("content-type")?.toLowerCase().includes("application/json")) {
    const text = await readErrorBody(response, bodyLimits);
    cleanupExternalAbort();
    report(answered());
    const record = parseJsonRecord<Record<string, unknown>>(text);
    const envelope = record ? errorEnvelope(label, record) : undefined;
    return {
      ok: false,
      error: envelope
        ? { type: "error", ...envelope }
        : { type: "error", error: `${label} answered a stream request with a JSON body instead of a stream.`, retryable: true },
    };
  }

  const bodyReader = response.body?.getReader();
  if (!bodyReader) {
    cleanupExternalAbort();
    report(answered());
    return {
      ok: false,
      error: { type: "error", error: `${label} response body missing`, retryable: false },
    };
  }

  const streamTimeouts = timeouts ? { ...timeouts } : undefined;
  if (firstByteDeadline && streamTimeouts) {
    streamTimeouts.firstByteMs = Math.max(1, firstByteDeadline - Date.now());
  }

  // An abort also ends the body read, even when a wrapped fetch does not tie the body to the signal.
  controller.signal.addEventListener("abort", () => void bodyReader.cancel(controller.signal.reason).catch(() => {}), { once: true });

  const reader = {
    async read() {
      const result = await bodyReader.read();
      if (!result.done && result.value?.byteLength) noteFirstByte();
      return result;
    },
    cancel: (reason?: unknown) => bodyReader.cancel(reason),
  };

  const events = async function* (): AsyncGenerator<SSEEvent> {
    try {
      yield* parseSSE(reader, streamTimeouts);
    } finally {
      cleanupExternalAbort();
      controller.abort();
      await reader.cancel().catch(() => {});
      report(answered());
    }
  };

  return { ok: true, events: events() };
};
