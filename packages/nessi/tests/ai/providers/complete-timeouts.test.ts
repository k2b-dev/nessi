import { afterEach, describe, expect, it } from "bun:test";
import { completeFromStream, openAICompatible } from "../../../src/ai/index.js";
import { endpoint, ENDPOINT_NAMES } from "../helpers/endpoints.js";
import type { Provider, ProviderTimeouts } from "../../../src/ai/types.js";
import { stubFetch } from "../helpers/fixtures.js";

const originalFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = originalFetch; });

const chatPayload = { choices: [{ message: { content: "Hello 🌊" }, finish_reason: "stop" }], usage: { prompt_tokens: 10, completion_tokens: 3, total_tokens: 13 } };

// Every endpoint variant runs on the same adapter; the matrix keeps each variant's settings covered.
const providers: Array<{ name: string; create: (timeouts: ProviderTimeouts) => Provider; payload: unknown }> = ENDPOINT_NAMES.map((name) => ({
  name,
  create: (timeouts) => endpoint(name, { baseURL: "https://example.test", timeouts }),
  payload: chatPayload,
}));

// A broken implementation must fail the test rather than leave the test runner hanging.
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

for (const testCase of providers) {
  describe(`${testCase.name} complete timeouts`, () => {
    it("ignores the streaming first-byte deadline during native generation", async () => {
      stubFetch(async () => {
        await new Promise((resolve) => setTimeout(resolve, 40));
        return new Response(JSON.stringify(testCase.payload), { headers: { "x-request-id": "request-1" } });
      });
      const result = await bounded(testCase.create({ firstByteMs: 10, idleMs: 10 }).complete({ messages: [] }));
      expect(result.message.content).toContainEqual({ type: "text", text: "Hello 🌊" });
      expect(result.finishReason).toBe("stop");
      expect(result.usage).toMatchObject({ input: 10, output: 3, total: 13 });
      expect(result.providerMeta?.requestId).toBe("request-1");
    });

    it("bounds native generation with totalMs and aborts the request", async () => {
      let requestSignal: AbortSignal | null | undefined;
      stubFetch((_url, init) => {
        requestSignal = init?.signal;
        return new Promise<Response>(() => {});
      });
      await expect(bounded(testCase.create({ totalMs: 10 }).complete({ messages: [] })))
        .rejects.toMatchObject({ kind: "timeout", scope: "provider_total", retryable: true });
      expect(requestSignal?.aborted).toBe(true);
    });

    for (const hasFirstChunk of [false, true]) {
      it(`bounds body reads ${hasFirstChunk ? "after a chunk" : "immediately after headers"} and cancels the reader`, async () => {
        let cancelled = false;
        stubFetch(() => new Response(new ReadableStream<Uint8Array>({
          start(controller) { if (hasFirstChunk) controller.enqueue(new TextEncoder().encode('{"partial":')); },
          cancel() { cancelled = true; },
        })));
        await expect(bounded(testCase.create({ totalMs: 100, idleMs: 10 }).complete({ messages: [] })))
          .rejects.toMatchObject({ scope: "provider_idle" });
        expect(cancelled).toBe(true);
      });
    }

    it("keeps the total deadline after headers when idleMs is unset", async () => {
      let cancelled = false;
      stubFetch(() => new Response(new ReadableStream<Uint8Array>({
        start(controller) { controller.enqueue(new TextEncoder().encode('{"partial":')); },
        cancel() { cancelled = true; },
      })));
      await expect(bounded(testCase.create({ totalMs: 10 }).complete({ messages: [] })))
        .rejects.toMatchObject({ kind: "timeout", scope: "provider_total", retryable: true });
      expect(cancelled).toBe(true);
    });

    it("does not reset the total deadline when body chunks make progress", async () => {
      let cancelled = false;
      let chunks = 0;
      stubFetch(() => new Response(new ReadableStream<Uint8Array>({
        start(controller) { controller.enqueue(new TextEncoder().encode('{"partial":')); },
        async pull(controller) {
          await new Promise((resolve) => setTimeout(resolve, 20));
          if (!cancelled) {
            chunks++;
            controller.enqueue(new TextEncoder().encode(" "));
          }
        },
        cancel() { cancelled = true; },
      })));
      // The idle limit is far above the 20 ms chunk gap, so only the total deadline can end it.
      await expect(bounded(testCase.create({ totalMs: 100, idleMs: 900 }).complete({ messages: [] })))
        .rejects.toMatchObject({ kind: "timeout", scope: "provider_total", retryable: true });
      expect(cancelled).toBe(true);
      // Progress happened before the deadline, so a total timer that reset on chunks would not fire.
      expect(chunks).toBeGreaterThanOrEqual(2);
    });

    for (const totalMs of [undefined, 0, -1]) {
      it(`disables the total deadline when totalMs is ${totalMs}`, async () => {
        stubFetch(async () => {
          await new Promise((resolve) => setTimeout(resolve, 20));
          return new Response(JSON.stringify(testCase.payload));
        });
        const result = await bounded(testCase.create({ totalMs, firstByteMs: 1, idleMs: 0 }).complete({ messages: [] }));
        expect(result.finishReason).toBe("stop");
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
      const result = await bounded(testCase.create({ totalMs: 500, firstByteMs: 10, idleMs: 100 }).complete({ messages: [] }));
      expect(result.message.content).toContainEqual({ type: "text", text: "Hello 🌊" });
      expect(result.finishReason).toBe("stop");
      expect(result.providerMeta?.requestId).toBe("request-1");
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
        await expect(bounded(testCase.create({ totalMs: 100, idleMs: 100 }).complete({ messages: [], signal: caller.signal })))
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

    for (const scope of ["provider_total", "provider_idle"] as const) {
      it(`times out stalled HTTP error bodies with ${scope}`, async () => {
        stubFetch(() => new Response(new ReadableStream<Uint8Array>(), { status: 500 }));
        await expect(bounded(testCase.create(scope === "provider_total" ? { totalMs: 10 } : { idleMs: 10 }).complete({ messages: [] })))
          .rejects.toMatchObject({ kind: "timeout", scope, retryable: true });
      });
    }

    it("cancels a response that arrives after the total deadline", async () => {
      let cancelled = false;
      stubFetch(async () => {
        await new Promise((resolve) => setTimeout(resolve, 20));
        return new Response(new ReadableStream<Uint8Array>({ cancel() { cancelled = true; } }));
      });
      await expect(bounded(testCase.create({ totalMs: 5 }).complete({ messages: [] })))
        .rejects.toMatchObject({ scope: "provider_total" });
      await new Promise((resolve) => setTimeout(resolve, 30));
      expect(cancelled).toBe(true);
    });

    it("cleans up deadlines and caller abort listeners after success", async () => {
      const caller = new AbortController();
      let requestSignal: AbortSignal | null | undefined;
      stubFetch((_url, init) => {
        requestSignal = init?.signal;
        return new Response(JSON.stringify(testCase.payload));
      });
      await testCase.create({ totalMs: 10, idleMs: 10 }).complete({ messages: [], signal: caller.signal });
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
          name: "local", model: "test", baseURL: server.url.toString(), timeouts: { totalMs: 100, idleMs: 20 },
        });
        await expect(bounded(provider.complete({ messages: [] })))
          .rejects.toMatchObject({ kind: "timeout", scope: phase === "headers" ? "provider_total" : "provider_idle", retryable: true });
      } finally {
        await server.stop(true);
      }
    });
  }

  it("allows both native and streamed answers to outlast firstByteMs", async () => {
    let nativeRequests = 0;
    let streamedRequests = 0;
    const server = Bun.serve({
      hostname: "127.0.0.1", port: 0,
      async fetch(request) {
        const body: unknown = await request.json();
        if (typeof body === "object" && body !== null && "stream" in body && body.stream === true) {
          streamedRequests++;
          let cancelled = false;
          let chunks = 0;
          return new Response(new ReadableStream<Uint8Array>({
            async pull(controller) {
              await new Promise((resolve) => setTimeout(resolve, 50));
              if (cancelled) return;
              const frame = ++chunks < 5
                ? { choices: [{ index: 0, delta: { content: "Hi" }, finish_reason: null }] }
                : { choices: [{ index: 0, delta: {}, finish_reason: "stop" }] };
              controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify(frame)}\n\n`));
              if (chunks === 5) controller.close();
            },
            cancel() { cancelled = true; },
          }), { headers: { "Content-Type": "text/event-stream" } });
        }
        nativeRequests++;
        await new Promise((resolve) => setTimeout(resolve, 300));
        return Response.json({ choices: [{ message: { content: "HiHiHiHi" }, finish_reason: "stop" }] });
      },
    });
    try {
      const provider = openAICompatible({
        // Both answers take 250-300 ms, past firstByteMs; each gap is 50 ms, far below the other limits.
        name: "local", model: "test", baseURL: server.url.toString(), timeouts: { firstByteMs: 200, idleMs: 400, totalMs: 900 },
      });
      const native = await bounded(provider.complete({ messages: [] }), 3000);
      const streamed = await bounded(completeFromStream(provider, { messages: [] }), 3000);
      expect(native.message.content).toEqual(streamed.message.content);
      expect(native.finishReason).toBe("stop");
      expect(streamed.finishReason).toBe("stop");
      expect(nativeRequests).toBe(1);
      expect(streamedRequests).toBe(1);
    } finally {
      await server.stop(true);
    }
  });
});
