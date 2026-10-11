import type { ProviderTimeouts } from "../types.js";
import { formatConnectionError } from "./errors.js";
import type { ByteStreamReader } from "./sse.js";

/** Buffer a non-streaming response, bounding headers and each body read separately. */
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
  let timeout: ReturnType<typeof setTimeout> | undefined;
  let reader: (ByteStreamReader & { releaseLock(): void }) | undefined;

  const armTimeout = (scope: "provider_first_byte" | "provider_idle", timeoutMs?: number) => {
    clearTimeout(timeout);
    if (!timeoutMs || timeoutMs <= 0) return;
    timeout = setTimeout(() => {
      controller.abort(Object.assign(
        new Error(`${label} response ${scope === "provider_first_byte" ? "first byte" : "idle"} timeout after ${timeoutMs}ms.`),
        { kind: "timeout", scope, retryable: true },
      ));
    }, timeoutMs);
  };

  try {
    armTimeout("provider_first_byte", timeouts?.firstByteMs);
    const response = await Promise.race([
      fetch(url, { ...init, signal: controller.signal }).then((response) => {
        // A response can arrive just after an abort, including with custom fetch implementations.
        if (controller.signal.aborted) {
          void response.body?.cancel(controller.signal.reason).catch(() => {});
          throw controller.signal.reason;
        }
        return response;
      }).catch((error: unknown) => {
        if (controller.signal.aborted) throw controller.signal.reason;
        throw new Error(formatConnectionError(label, error));
      }),
      aborted,
    ]);
    clearTimeout(timeout);
    reader = response.body?.getReader();
    if (!reader) return response;

    const decoder = new TextDecoder();
    let text = "";
    while (true) {
      armTimeout("provider_idle", timeouts?.idleMs);
      const { done, value } = await Promise.race([reader.read(), aborted]);
      if (done) break;
      text += decoder.decode(value, { stream: true });
    }
    text += decoder.decode();
    return new Response(text, { status: response.status, statusText: response.statusText, headers: response.headers });
  } catch (error) {
    controller.abort(error);
    void reader?.cancel?.(error).catch(() => {});
    throw error;
  } finally {
    clearTimeout(timeout);
    signal?.removeEventListener("abort", abortExternal);
    reader?.releaseLock();
  }
};
