import type { ProviderTimeouts } from "../types.js";
import { normalizeConnectionError, ProviderError, providerError } from "./errors.js";
import type { ByteStreamReader } from "./sse.js";

/**
 * Buffer a native response with a total deadline and idle limits after headers.
 * Redirects fail: credential headers such as x-api-key would follow them to another origin.
 */
export const fetchCompleteResponse = async (
  url: string,
  init: RequestInit,
  label: string,
  timeouts?: ProviderTimeouts,
): Promise<Response> => {
  const signal = init.signal;
  signal?.throwIfAborted();
  const controller = new AbortController();
  const aborted = new Promise<never>((_, reject) => {
    controller.signal.addEventListener("abort", () => reject(controller.signal.reason), { once: true });
  });
  const abortExternal = () => controller.abort(signal?.reason);
  signal?.addEventListener("abort", abortExternal, { once: true });
  let totalTimeout: ReturnType<typeof setTimeout> | undefined;
  let idleTimeout: ReturnType<typeof setTimeout> | undefined;
  let reader: (ByteStreamReader & { releaseLock(): void }) | undefined;

  const scheduleTimeout = (scope: "provider_total" | "provider_idle", timeoutMs?: number) => {
    if (!timeoutMs || timeoutMs <= 0) return;
    return setTimeout(() => {
      controller.abort(new ProviderError(
        `${label} response ${scope === "provider_total" ? "total" : "idle"} timeout after ${timeoutMs}ms.`,
        { kind: "timeout", scope, retryable: true },
      ));
    }, timeoutMs);
  };

  try {
    totalTimeout = scheduleTimeout("provider_total", timeouts?.totalMs);
    const response = await Promise.race([
      fetch(url, { ...init, redirect: "error", signal: controller.signal }).then((response) => {
        // A response can arrive just after an abort, including with custom fetch implementations.
        if (controller.signal.aborted) {
          void response.body?.cancel(controller.signal.reason).catch(() => {});
          throw controller.signal.reason;
        }
        return response;
      }).catch((error: unknown) => {
        if (controller.signal.aborted) throw controller.signal.reason;
        throw providerError(normalizeConnectionError(label, error), { cause: error });
      }),
      aborted,
    ]);
    reader = response.body?.getReader();
    if (!reader) {
      signal?.throwIfAborted();
      return response;
    }

    const decoder = new TextDecoder();
    let text = "";
    while (true) {
      clearTimeout(idleTimeout);
      idleTimeout = scheduleTimeout("provider_idle", timeouts?.idleMs);
      const { done, value } = await Promise.race([reader.read(), aborted]);
      if (done) break;
      text += decoder.decode(value, { stream: true });
    }
    text += decoder.decode();
    // The last read can win the race against an abort that already happened.
    signal?.throwIfAborted();
    return new Response(text, { status: response.status, statusText: response.statusText, headers: response.headers });
  } catch (error) {
    controller.abort(error);
    void reader?.cancel?.(error).catch(() => {});
    if (signal?.aborted) throw signal.reason;
    if (error instanceof ProviderError) throw error;
    // The connection failed while the body was read.
    throw providerError(normalizeConnectionError(label, error), { cause: error });
  } finally {
    clearTimeout(totalTimeout);
    clearTimeout(idleTimeout);
    signal?.removeEventListener("abort", abortExternal);
    reader?.releaseLock();
  }
};
