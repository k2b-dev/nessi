import { afterEach, describe, expect, it } from "bun:test";
import { anthropic, gemini, mistral, ollama, openAICompatible } from "../../../src/ai/index.js";
import type { Provider, ProviderTimeouts } from "../../../src/ai/types.js";
import { stubFetch } from "../helpers/fixtures.js";

const originalFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = originalFetch; });

const providers: Array<{ name: string; create: (timeouts: ProviderTimeouts) => Provider; payload: unknown }> = [
  {
    name: "openai-compatible",
    create: (timeouts) => openAICompatible({ name: "test", model: "test", baseURL: "https://example.test", timeouts }),
    payload: { choices: [{ message: { content: "Hello 🌊" }, finish_reason: "stop" }] },
  },
  {
    name: "anthropic",
    create: (timeouts) => anthropic("test", { apiKey: "test", baseURL: "https://example.test", timeouts }),
    payload: { content: [{ type: "text", text: "Hello 🌊" }], stop_reason: "end_turn" },
  },
  {
    name: "mistral",
    create: (timeouts) => mistral("test", { apiKey: "test", baseURL: "https://example.test", timeouts }),
    payload: { choices: [{ message: { content: "Hello 🌊" }, finish_reason: "stop" }] },
  },
  {
    name: "gemini",
    create: (timeouts) => gemini("test", { apiKey: "test", baseURL: "https://example.test", timeouts }),
    payload: { candidates: [{ content: { parts: [{ text: "Hello 🌊" }] }, finishReason: "STOP" }] },
  },
  {
    name: "ollama",
    create: (timeouts) => ollama("test", { baseURL: "https://example.test", timeouts }),
    payload: { message: { content: "Hello 🌊" }, done: true, done_reason: "stop" },
  },
];

// A broken implementation must fail the test rather than leave the test runner hanging.
async function bounded<T>(promise: Promise<T>): Promise<T> {
  let timeout: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_, reject) => { timeout = setTimeout(() => reject(new Error("Test deadline exceeded")), 1000); }),
    ]);
  } finally {
    clearTimeout(timeout);
  }
}

for (const testCase of providers) {
  describe(`${testCase.name} complete timeouts`, () => {
    it("bounds the wait for headers and aborts the request", async () => {
      let requestSignal: AbortSignal | null | undefined;
      stubFetch((_url, init) => {
        requestSignal = init?.signal;
        return new Promise<Response>(() => {});
      });
      await expect(bounded(testCase.create({ firstByteMs: 10 }).complete({ messages: [] })))
        .rejects.toMatchObject({ scope: "provider_first_byte" });
      expect(requestSignal?.aborted).toBe(true);
    });

    for (const hasFirstChunk of [false, true]) {
      it(`bounds body reads ${hasFirstChunk ? "after a chunk" : "immediately after headers"} and cancels the reader`, async () => {
        let cancelled = false;
        stubFetch(() => new Response(new ReadableStream<Uint8Array>({
          start(controller) { if (hasFirstChunk) controller.enqueue(new TextEncoder().encode('{"partial":')); },
          cancel() { cancelled = true; },
        })));
        await expect(bounded(testCase.create({ firstByteMs: 100, idleMs: 10 }).complete({ messages: [] })))
          .rejects.toMatchObject({ scope: "provider_idle" });
        expect(cancelled).toBe(true);
      });
    }

    it("resets idle timeouts for each chunk and preserves split UTF-8", async () => {
      const bytes = new TextEncoder().encode(JSON.stringify(testCase.payload));
      const emojiIndex = bytes.indexOf(0xf0);
      const chunks = [bytes.slice(0, emojiIndex + 1), bytes.slice(emojiIndex + 1, emojiIndex + 3), bytes.slice(emojiIndex + 3)];
      let index = 0;
      stubFetch(() => new Response(new ReadableStream<Uint8Array>({
        async pull(controller) {
          await new Promise((resolve) => setTimeout(resolve, 40));
          const chunk = chunks[index++];
          if (chunk) controller.enqueue(chunk);
          else controller.close();
        },
      }), { headers: { "x-request-id": "request-1" } }));
      const result = await bounded(testCase.create({ firstByteMs: 10, idleMs: 100 }).complete({ messages: [] }));
      expect(result.message.content).toContainEqual({ type: "text", text: "Hello 🌊" });
      expect(result.finishReason).toBe("stop");
      if (testCase.name === "openai-compatible") expect(result.providerMeta?.requestId).toBe("request-1");
    });

    for (const phase of ["headers", "body"] as const) {
      it(`preserves caller abort reasons while waiting for ${phase}`, async () => {
        const caller = new AbortController();
        const reason = new Error("Caller cancelled");
        let cancelled = false;
        stubFetch(() => {
          setTimeout(() => caller.abort(reason), 1);
          return phase === "headers" ? new Promise<Response>(() => {}) : new Response(new ReadableStream<Uint8Array>({
            cancel() { cancelled = true; },
          }));
        });
        await expect(bounded(testCase.create({ firstByteMs: 100, idleMs: 100 }).complete({ messages: [], signal: caller.signal })))
          .rejects.toBe(reason);
        if (phase === "body") expect(cancelled).toBe(true);
      });
    }

    it("does not dispatch a request for an already aborted signal", async () => {
      let called = false;
      stubFetch(() => { called = true; return new Response(JSON.stringify(testCase.payload)); });
      const reason = new Error("Already cancelled");
      await expect(testCase.create({}).complete({ messages: [], signal: AbortSignal.abort(reason) })).rejects.toBe(reason);
      expect(called).toBe(false);
    });

    it("times out stalled HTTP error bodies too", async () => {
      stubFetch(() => new Response(new ReadableStream<Uint8Array>(), { status: 500 }));
      await expect(bounded(testCase.create({ idleMs: 10 }).complete({ messages: [] })))
        .rejects.toMatchObject({ scope: "provider_idle" });
    });

    it("cleans up deadlines and caller abort listeners after success", async () => {
      const caller = new AbortController();
      let requestSignal: AbortSignal | null | undefined;
      stubFetch((_url, init) => {
        requestSignal = init?.signal;
        return new Response(JSON.stringify(testCase.payload));
      });
      await testCase.create({ firstByteMs: 10, idleMs: 10 }).complete({ messages: [], signal: caller.signal });
      caller.abort();
      await new Promise((resolve) => setTimeout(resolve, 20));
      expect(requestSignal?.aborted).toBe(false);
    });
  });
}

describe("complete timeouts over HTTP", () => {
  for (const phase of ["headers", "body"] as const) {
    it(`aborts a real local HTTP request stalled at ${phase}`, async () => {
      const server = Bun.serve({
        hostname: "127.0.0.1",
        port: 0,
        fetch() {
          return phase === "headers" ? new Promise<Response>(() => {}) : new Response(new ReadableStream<Uint8Array>({
            start(controller) { controller.enqueue(new TextEncoder().encode('{"choices":')); },
          }));
        },
      });
      try {
        const provider = openAICompatible({
          name: "local", model: "test", baseURL: server.url.toString(), timeouts: { firstByteMs: 100, idleMs: 20 },
        });
        await expect(bounded(provider.complete({ messages: [] })))
          .rejects.toMatchObject({ kind: "timeout", scope: phase === "headers" ? "provider_first_byte" : "provider_idle", retryable: true });
      } finally {
        await server.stop(true);
      }
    });
  }
});
