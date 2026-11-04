import { afterEach, describe, expect, it } from "bun:test";
import { complete, openAICompatible } from "../../../src/ai/index.js";
import type { Provider, ProviderTimeouts } from "../../../src/ai/types.js";
import { endpoint, ENDPOINT_NAMES } from "../helpers/endpoints.js";
import { stubFetch } from "../helpers/fixtures.js";

const originalFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = originalFetch; });

const encoder = new TextEncoder();
const answer = [
  `data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: "Hello 🌊" }, finish_reason: null }] })}\n\n`,
  `data: ${JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 10, completion_tokens: 3, total_tokens: 13 } })}\n\n`,
  "data: [DONE]\n\n",
].join("");

// A broken implementation must fail the test rather than leave the runner hanging.
async function bounded<T>(promise: Promise<T>, ms = 1000): Promise<T> {
  let timeout: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_, reject) => { timeout = setTimeout(() => reject(new Error("Test deadline exceeded")), ms); }),
    ]);
  } finally {
    clearTimeout(timeout);
  }
}

for (const name of ENDPOINT_NAMES) {
  const create = (timeouts: ProviderTimeouts): Provider => endpoint(name, { baseURL: "https://example.test", timeouts });

  describe(`${name} stream timeouts`, () => {
    it("bounds the wait for the first byte and aborts the request", async () => {
      let requestSignal: AbortSignal | null | undefined;
      stubFetch((_url, init) => {
        requestSignal = init?.signal;
        return new Promise<Response>(() => {});
      });
      await expect(bounded(complete(create({ firstByteMs: 10 }), { messages: [] })))
        .rejects.toMatchObject({ kind: "timeout", scope: "provider_first_byte", retryable: true });
      expect(requestSignal?.aborted).toBe(true);
    });

    it("bounds a stalled body after a chunk and cancels the reader", async () => {
      let cancelled = false;
      stubFetch(() => new Response(new ReadableStream<Uint8Array>({
        start(controller) { controller.enqueue(encoder.encode(answer.slice(0, 40))); },
        cancel() { cancelled = true; },
      })));
      await expect(bounded(complete(create({ idleMs: 20 }), { messages: [] })))
        .rejects.toMatchObject({ kind: "timeout", scope: "provider_idle", retryable: true });
      expect(cancelled).toBe(true);
    });

    it("resets the idle timeout for each chunk and keeps a character split across chunks", async () => {
      const bytes = encoder.encode(answer);
      const emoji = bytes.indexOf(0xf0);
      const chunks = [bytes.slice(0, emoji + 1), bytes.slice(emoji + 1, emoji + 3), bytes.slice(emoji + 3)];
      let index = 0;
      stubFetch(() => new Response(new ReadableStream<Uint8Array>({
        async pull(controller) {
          await new Promise((resolve) => setTimeout(resolve, 20));
          const chunk = chunks[index++];
          if (chunk) controller.enqueue(chunk);
          else controller.close();
        },
      })));
      const result = await bounded(complete(create({ firstByteMs: 500, idleMs: 300 }), { messages: [] }));
      expect(result.message.content).toContainEqual({ type: "text", text: "Hello 🌊" });
      expect(result.finishReason).toBe("stop");
    });

    for (const phase of ["headers", "body"] as const) {
      it(`keeps the caller's abort reason while waiting for ${phase}`, async () => {
        const caller = new AbortController();
        const reason = new Error("Caller cancelled");
        let cancelled = false;
        stubFetch((_url, init) => {
          setTimeout(() => caller.abort(reason), 1);
          if (phase === "headers") {
            return new Promise<Response>((_, reject) => init?.signal?.addEventListener("abort", () => reject(init.signal?.reason), { once: true }));
          }
          return new Response(new ReadableStream<Uint8Array>({ cancel() { cancelled = true; } }));
        });
        await expect(bounded(complete(create({ firstByteMs: 500, idleMs: 500 }), { messages: [], signal: caller.signal })))
          .rejects.toBe(reason);
        if (phase === "body") expect(cancelled).toBe(true);
      });
    }

    it("does not dispatch a request for an already aborted signal", async () => {
      let called = false;
      stubFetch(() => { called = true; return new Response(answer); });
      const reason = new Error("Already cancelled");
      await expect(complete(create({}), { messages: [], signal: AbortSignal.abort(reason) })).rejects.toBe(reason);
      expect(called).toBe(false);
    });
  });
}

describe("stream timeouts over HTTP", () => {
  for (const phase of ["first byte", "body"] as const) {
    it(`aborts a real local HTTP request stalled at the ${phase}`, async () => {
      const server = Bun.serve({
        hostname: "127.0.0.1",
        port: 0,
        fetch() {
          return phase === "first byte" ? new Promise<Response>(() => {}) : new Response(new ReadableStream<Uint8Array>({
            start(controller) { controller.enqueue(encoder.encode(answer.slice(0, 40))); },
          }), { headers: { "Content-Type": "text/event-stream" } });
        },
      });
      try {
        const provider = openAICompatible({ name: "local", model: "test", baseURL: server.url.toString(), timeouts: { firstByteMs: 100, idleMs: 20 } });
        await expect(bounded(complete(provider, { messages: [] })))
          .rejects.toMatchObject({ kind: "timeout", scope: phase === "first byte" ? "provider_first_byte" : "provider_idle", retryable: true });
      } finally {
        await server.stop(true);
      }
    });
  }

  it("lets a streamed answer outlast firstByteMs when chunks keep coming", async () => {
    const server = Bun.serve({
      hostname: "127.0.0.1",
      port: 0,
      fetch() {
        let chunks = 0;
        return new Response(new ReadableStream<Uint8Array>({
          async pull(controller) {
            await new Promise((resolve) => setTimeout(resolve, 50));
            const frame = ++chunks < 5
              ? { choices: [{ index: 0, delta: { content: "Hi" }, finish_reason: null }] }
              : { choices: [{ index: 0, delta: {}, finish_reason: "stop" }] };
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(frame)}\n\n`));
            if (chunks === 5) controller.close();
          },
        }), { headers: { "Content-Type": "text/event-stream" } });
      },
    });
    try {
      // 5 × 50 ms is past firstByteMs; each gap stays far below idleMs.
      const provider = openAICompatible({ name: "local", model: "test", baseURL: server.url.toString(), timeouts: { firstByteMs: 200, idleMs: 400 } });
      const result = await bounded(complete(provider, { messages: [] }), 3000);
      expect(result.message.content).toEqual([{ type: "text", text: "HiHiHiHi" }]);
      expect(result.finishReason).toBe("stop");
    } finally {
      await server.stop(true);
    }
  });
});
