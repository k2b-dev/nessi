import { afterEach, describe, expect, it } from "bun:test";
import { complete, openAICompatible, openAICompatibleTranscription, systemOneDecision } from "../../../src/ai/index.js";
import { endpoint, ENDPOINT_NAMES } from "../helpers/endpoints.js";
import type { Provider, ProviderTimeouts, StreamEvent } from "../../../src/ai/types.js";
import { ERROR_BODY_LIMIT_BYTES, readErrorBody } from "../../../src/ai/shared/errors.js";
import { stubFetch } from "../helpers/fixtures.js";

const originalFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = originalFetch; });

const providers: Array<{ name: string; create: (timeouts?: ProviderTimeouts) => Provider }> = ENDPOINT_NAMES.map((name) => ({
  name,
  create: (timeouts?: ProviderTimeouts) => endpoint(name, { baseURL: "https://example.test", timeouts }),
}));

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

async function bounded<T>(promise: Promise<T>, ms = 500): Promise<T> {
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

    it(`${name} keeps the first-byte deadline absolute when headers arrive late`, async () => {
      stubFetch(() => new Promise<Response>((resolve) => setTimeout(() => resolve(stalledErrorBody()), 60)));
      const startedAt = Date.now();
      await bounded(streamIssues(create({ firstByteMs: 100 })));
      // Headers at 60 ms leave 40 ms of the budget for the body; restarting it would end near 160 ms.
      expect(Date.now() - startedAt).toBeLessThan(140);
    });

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
      await expect(bounded(complete(create(), request))).rejects.toThrow("429");
      expect(counter.pulls * 16 * 1024).toBeLessThanOrEqual(ERROR_BODY_LIMIT_BYTES + 3 * 16 * 1024);
    });
  }
});

const oneChunk = (bytes: Uint8Array, status = 500) => new Response(new ReadableStream<Uint8Array>({
  start(controller) {
    controller.enqueue(bytes);
    controller.close();
  },
}), { status });

describe("readErrorBody", () => {
  it("keeps at most the byte budget and drops a character cut at the boundary", async () => {
    const bytes = encoder.encode(`${"a".repeat(ERROR_BODY_LIMIT_BYTES - 1)}€${"b".repeat(1024)}`);
    const body = await readErrorBody(oneChunk(bytes));
    expect(body).toBe("a".repeat(ERROR_BODY_LIMIT_BYTES - 1));
  });

  it("caps error bodies by bytes, also in one large chunk", async () => {
    stubFetch(() => oneChunk(encoder.encode("€".repeat(ERROR_BODY_LIMIT_BYTES)), 500));
    const error = await complete(openAICompatible({ name: "test", model: "test", baseURL: "https://example.test" }), request)
      .catch((caught: unknown) => caught);
    expect(error).toBeInstanceOf(Error);
    const message = error instanceof Error ? error.message : "";
    expect(encoder.encode(message).byteLength).toBeLessThanOrEqual(ERROR_BODY_LIMIT_BYTES + 100);
    expect(message).not.toContain("\uFFFD");
  });

  it("bounds transcription and decision error bodies", async () => {
    for (const run of [
      (): Promise<unknown> => openAICompatibleTranscription("whisper", { baseURL: "https://example.test" }).transcribe({ file: new Blob(["a"], { type: "audio/wav" }) }),
      () => systemOneDecision("clef", { baseURL: "https://example.test" }).decide({ state: "x", questions: { urgent: { type: "noul", instructions: "Urgent?" } } }),
    ]) {
      const { response, counter } = endlessErrorBody();
      stubFetch(() => response);
      await expect(bounded(run())).rejects.toThrow("429");
      expect(counter.pulls * 16 * 1024).toBeLessThanOrEqual(ERROR_BODY_LIMIT_BYTES + 3 * 16 * 1024);
    }
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
