import { afterEach, describe, expect, it } from "bun:test";
import { anthropic, gemini, mistral, ollama, openAICompatible } from "../../../src/ai/index.js";
import type { Provider, ProviderTimeouts, StreamEvent } from "../../../src/ai/types.js";
import { ERROR_BODY_LIMIT_BYTES, readErrorBody } from "../../../src/ai/shared/errors.js";
import { stubFetch } from "../helpers/fixtures.js";

const originalFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = originalFetch; });

const providers: Array<{ name: string; create: (timeouts?: ProviderTimeouts) => Provider }> = [
  { name: "openai-compatible", create: (timeouts) => openAICompatible({ name: "test", model: "test", baseURL: "https://example.test", timeouts }) },
  { name: "anthropic", create: (timeouts) => anthropic("test", { apiKey: "test", baseURL: "https://example.test", timeouts }) },
  { name: "mistral", create: (timeouts) => mistral("test", { apiKey: "test", baseURL: "https://example.test", timeouts }) },
  { name: "gemini", create: (timeouts) => gemini("test", { apiKey: "test", baseURL: "https://example.test", timeouts }) },
  { name: "ollama", create: (timeouts) => ollama("test", { baseURL: "https://example.test", timeouts }) },
];

const request = { messages: [{ role: "user" as const, content: [{ type: "text" as const, text: "Hi" }] }] };
const encoder = new TextEncoder();

/** A gateway that sends error headers and part of a body, then never finishes it. */
const stalledErrorBody = () => new Response(new ReadableStream<Uint8Array>({
  start(controller) {
    controller.enqueue(encoder.encode('{"error":{"message":"overloaded'));
  },
}), { status: 503, headers: { "Content-Type": "application/json" } });

/** An error body that never ends; `pulls` counts how much of it was read. */
const endlessErrorBody = () => {
  const counter = { pulls: 0 };
  const chunk = encoder.encode("x".repeat(16 * 1024));
  const response = new Response(new ReadableStream<Uint8Array>({
    pull(controller) {
      counter.pulls++;
      controller.enqueue(chunk);
    },
  }), { status: 429, headers: { "Content-Type": "text/plain" } });
  return { response, counter };
};

async function bounded<T>(promise: Promise<T>, ms = 2_000): Promise<T> {
  let timeout: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_, reject) => { timeout = setTimeout(() => reject(new Error(`still waiting after ${ms}ms`)), ms); }),
    ]);
  } finally {
    clearTimeout(timeout);
  }
}

const streamIssues = async (provider: Provider) => {
  const issues = [];
  for await (const event of provider.stream(request) as AsyncIterable<StreamEvent>) {
    if (event.type === "issue") issues.push(event.issue);
  }
  return issues;
};

describe("streaming error bodies", () => {
  for (const { name, create } of providers) {
    for (const timeouts of [{ firstByteMs: 20, idleMs: 20 }, { firstByteMs: 20 }] satisfies ProviderTimeouts[]) {
      it(`${name} reports a stalled error body within its deadline (${Object.keys(timeouts).join(", ")})`, async () => {
        stubFetch(stalledErrorBody);
        const issues = await bounded(streamIssues(create(timeouts)));
        expect(issues).toEqual([expect.objectContaining({ kind: "provider_error", retryable: true, message: expect.stringContaining("503") })]);
      });
    }

    it(`${name} stream stops reading an endless error body`, async () => {
      const { response, counter } = endlessErrorBody();
      stubFetch(() => response);
      const issues = await bounded(streamIssues(create()));
      expect(issues).toEqual([expect.objectContaining({ kind: "provider_error", retryable: true })]);
      expect(counter.pulls * 16 * 1024).toBeLessThanOrEqual(ERROR_BODY_LIMIT_BYTES + 3 * 16 * 1024);
    });

    it(`${name} complete() stops reading an endless error body`, async () => {
      const { response, counter } = endlessErrorBody();
      stubFetch(() => response);
      await expect(bounded(create().complete(request))).rejects.toThrow("429");
      expect(counter.pulls * 16 * 1024).toBeLessThanOrEqual(ERROR_BODY_LIMIT_BYTES + 3 * 16 * 1024);
    });
  }
});

describe("readErrorBody", () => {
  it("keeps at most the byte budget and decodes split characters", async () => {
    const text = "ü".repeat(ERROR_BODY_LIMIT_BYTES);
    const body = await readErrorBody(new Response(text, { status: 500 }));
    expect(encoder.encode(body).byteLength).toBeLessThanOrEqual(ERROR_BODY_LIMIT_BYTES);
    expect(body.startsWith("üüü")).toBe(true);
  });

  it("returns what arrived when the body fails", async () => {
    let pulls = 0;
    const response = new Response(new ReadableStream<Uint8Array>({
      pull(controller) {
        if (pulls++ === 0) controller.enqueue(encoder.encode("partial"));
        else controller.error(new TypeError("socket closed"));
      },
    }), { status: 502 });
    expect(await readErrorBody(response)).toBe("partial");
  });
});
