import { afterEach, describe, expect, it } from "bun:test";
import { anthropic, completeFromStream, gemini, mistral, ollama, openAICompatible, ProviderError } from "../../../src/ai/index.js";
import type { Provider } from "../../../src/ai/types.js";
import { parseRetryAfterMs } from "../../../src/ai/shared/errors.js";
import { stubFetch } from "../helpers/fixtures.js";

const originalFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = originalFetch; });

const providers: Array<{ name: string; create: () => Provider }> = [
  { name: "openai-compatible", create: () => openAICompatible({ name: "test", model: "test", baseURL: "https://example.test" }) },
  { name: "anthropic", create: () => anthropic("test", { apiKey: "test", baseURL: "https://example.test" }) },
  { name: "mistral", create: () => mistral("test", { apiKey: "test", baseURL: "https://example.test" }) },
  { name: "gemini", create: () => gemini("test", { apiKey: "test", baseURL: "https://example.test" }) },
  { name: "ollama", create: () => ollama("test", { baseURL: "https://example.test" }) },
];

const request = { messages: [{ role: "user" as const, content: [{ type: "text" as const, text: "Hi" }] }] };

const rejection = async (promise: Promise<unknown>) => {
  try {
    await promise;
  } catch (error) {
    return error;
  }
  throw new Error("expected a rejection");
};

const rateLimited = () => new Response(JSON.stringify({ error: { message: "Slow down", code: "rate_limit_exceeded" } }), {
  status: 429,
  headers: { "Content-Type": "application/json", "retry-after": "2" },
});

const overflow = () => new Response(
  JSON.stringify({ error: { message: "This model's maximum context length is 1000 tokens. However, your messages resulted in 2000 tokens" } }),
  { status: 400, headers: { "Content-Type": "application/json" } },
);

describe("ProviderError", () => {
  for (const { name, create } of providers) {
    for (const [mode, run] of [
      ["complete()", (provider: Provider, signal?: AbortSignal) => provider.complete({ ...request, signal })],
      ["completeFromStream()", (provider: Provider, signal?: AbortSignal) => completeFromStream(provider, { ...request, signal })],
    ] as const) {
      it(`${name} ${mode} keeps status, code, and Retry-After of a rejected request`, async () => {
        stubFetch(rateLimited);
        const error = await rejection(run(create()));
        expect(error).toBeInstanceOf(ProviderError);
        expect(error).toMatchObject({ kind: "provider_error", retryable: true, status: 429, code: "rate_limit_exceeded", retryAfterMs: 2_000 });
      });

      it(`${name} ${mode} marks a context overflow as not retryable`, async () => {
        stubFetch(overflow);
        expect(await rejection(run(create()))).toMatchObject({ kind: "provider_error", retryable: false, status: 400, contextOverflow: true, overflowRatio: 2 });
      });

      it(`${name} ${mode} keeps the runtime code of a failed connection`, async () => {
        stubFetch(() => { throw Object.assign(new TypeError("Unable to connect"), { code: "ECONNREFUSED" }); });
        const error = await rejection(run(create()));
        expect(error).toBeInstanceOf(ProviderError);
        expect(error).toMatchObject({ kind: "provider_error", retryable: true, code: "ECONNREFUSED" });
      });

      it(`${name} ${mode} keeps the runtime code from a Node-style cause`, async () => {
        stubFetch(() => { throw new TypeError("fetch failed", { cause: Object.assign(new Error("connect"), { code: "ECONNRESET" }) }); });
        expect(await rejection(run(create()))).toMatchObject({ kind: "provider_error", retryable: true, code: "ECONNRESET" });
      });

      it(`${name} ${mode} reports a connection that drops during the body`, async () => {
        stubFetch(() => new Response(new ReadableStream({
          start(controller) {
            controller.enqueue(new TextEncoder().encode("{"));
            controller.error(Object.assign(new TypeError("socket closed"), { code: "ECONNRESET" }));
          },
        }), { headers: { "Content-Type": mode === "complete()" ? "application/json" : "text/event-stream" } }));
        const error = await rejection(run(create()));
        expect(error).toBeInstanceOf(ProviderError);
        expect(error).toMatchObject({ retryable: true });
      });

      it(`${name} ${mode} rethrows an abort that lands as the body ends`, async () => {
        const controller = new AbortController();
        const reason = { stoppedBy: "caller" };
        stubFetch(() => new Response(new ReadableStream({
          start(stream) {
            stream.enqueue(new TextEncoder().encode("{}"));
          },
          pull(stream) {
            // The final read resolves before the abort reaches the request.
            stream.close();
            controller.abort(reason);
          },
        }), { headers: { "Content-Type": "application/json" } }));
        expect(await rejection(run(create(), controller.signal))).toBe(reason);
      });

      it(`${name} ${mode} rethrows the caller's abort reason`, async () => {
        const controller = new AbortController();
        const reason = new Error("caller stopped");
        stubFetch((_input, init) => new Promise<Response>((_, reject) => {
          init?.signal?.addEventListener("abort", () => reject(init.signal?.reason), { once: true });
          controller.abort(reason);
        }));
        expect(await rejection(run(create(), controller.signal))).toBe(reason);
      });
    }
  }

  it("completeFromStream() reports a stream deadline as a timeout, also when fetch rejects on abort", async () => {
    // Like a real fetch, reject in the same tick as the abort.
    globalThis.fetch = Object.assign((_input: string | URL | Request, init?: RequestInit) => new Promise<Response>((_, reject) => {
      init?.signal?.addEventListener("abort", () => reject(new DOMException("The operation was aborted.", "AbortError")), { once: true });
    }), { preconnect: () => {} });
    for (const provider of [
      openAICompatible({ name: "test", model: "test", baseURL: "https://example.test", timeouts: { firstByteMs: 5 } }),
      ollama("test", { baseURL: "https://example.test", timeouts: { firstByteMs: 5 } }),
    ]) {
      const error = await rejection(completeFromStream(provider, request));
      expect(error).toBeInstanceOf(ProviderError);
      expect(error).toMatchObject({ kind: "timeout", retryable: true, scope: "provider_first_byte" });
    }
  });

  it("complete() reports a response deadline as a timeout", async () => {
    stubFetch(() => new Promise<Response>(() => {}));
    const provider = openAICompatible({ name: "test", model: "test", baseURL: "https://example.test", timeouts: { totalMs: 5 } });
    const error = await rejection(provider.complete(request));
    expect(error).toBeInstanceOf(ProviderError);
    expect(error).toMatchObject({ kind: "timeout", retryable: true, scope: "provider_total" });
  });
});

describe("parseRetryAfterMs", () => {
  const now = Date.parse("2026-11-04T10:00:00Z");
  it.each([
    [{ "retry-after-ms": "1500", "retry-after": "9" }, 1_500],
    [{ "retry-after": "3" }, 3_000],
    [{ "retry-after": "Wed, 04 Nov 2026 10:00:05 GMT" }, 5_000],
    [{ "retry-after": "Wed, 04 Nov 2026 09:59:00 GMT" }, 0],
    [{ "retry-after": "soon" }, undefined],
    [{}, undefined],
  ])("reads %j", (headers, expected) => {
    expect(parseRetryAfterMs(new Headers(headers), now)).toBe(expected);
  });
});
