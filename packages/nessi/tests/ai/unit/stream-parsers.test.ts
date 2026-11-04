import { afterEach, describe, expect, it } from "bun:test";
import { openAICompatible } from "../../../src/ai/index.js";
import { endpoint } from "../helpers/endpoints.js";
import { stubFetch, textResponse } from "../helpers/fixtures.js";
import { MAX_STREAM_RECORD_CHARS, parseSSE, readRecordBatches, SSETimeoutError, StreamRecordTooLargeError } from "../../../src/ai/shared/sse.js";

const encoder = new TextEncoder();

const readerFromChunks = (chunks: string[]) => new ReadableStream<Uint8Array>({
  start(controller) {
    for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
    controller.close();
  },
}).getReader();

/** A WHATWG reader whose pending read resolves as a clean end when cancelled. */
const stallingReader = (chunks: string[] = []) => new ReadableStream<Uint8Array>({
  start(controller) {
    for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
  },
}).getReader();

const drain = async <T>(events: AsyncIterable<T>) => {
  const all: T[] = [];
  for await (const event of events) all.push(event);
  return all;
};

const failure = async (events: AsyncIterable<unknown>) => {
  try {
    await drain(events);
  } catch (error) {
    return error;
  }
  return undefined;
};

describe("stream timeouts", () => {
  it("parseSSE times out on a WHATWG reader instead of ending cleanly", async () => {
    expect(await failure(parseSSE(stallingReader(), { firstByteMs: 5 }))).toMatchObject({ scope: "provider_first_byte" });
    const idle = await failure(parseSSE(stallingReader(['data: {"a":1}\n\n']), { idleMs: 5 }));
    expect(idle).toBeInstanceOf(SSETimeoutError);
    expect(idle).toMatchObject({ scope: "provider_idle" });
  });
});

describe("record splitting", () => {
  it("joins CRLF split across chunks", async () => {
    expect(await drain(parseSSE(readerFromChunks(["data: a\r", "\n\r", "\ndata: b\r\n\r\n"])))).toEqual([
      { event: undefined, data: "a", id: undefined },
      { event: undefined, data: "b", id: undefined },
    ]);
  });

  it("finds a separator split across chunks", async () => {
    expect((await drain(parseSSE(readerFromChunks(["data: a\n", "\ndata: b\n", "\n"])))).map((event) => event.data)).toEqual(["a", "b"]);
    expect((await drain(readRecordBatches(readerFromChunks(['{"a":1}', '\n{"b"', ":2}\n"]), "\n"))).flat()).toEqual(['{"a":1}', '{"b":2}']);
  });

  it("matches a plain split for any chunking (seeded)", async () => {
    let seed = 42;
    const random = () => {
      seed = (seed * 1_103_515_245 + 12_345) % 2_147_483_648;
      return seed / 2_147_483_648;
    };
    const alphabet = ["a", "b", "\n", "\r", "\r\n", "€"];
    for (let round = 0; round < 300; round++) {
      let text = "";
      const length = Math.floor(random() * 60);
      for (let index = 0; index < length; index++) text += alphabet[Math.floor(random() * alphabet.length)];
      const chunks: string[] = [];
      for (let index = 0; index < text.length;) {
        const size = 1 + Math.floor(random() * 5);
        chunks.push(text.slice(index, index + size));
        index += size;
      }
      for (const separator of ["\n", "\n\n"]) {
        const expected = text.replace(/\r\n/g, "\n").split(separator);
        if (expected.at(-1) === "") expected.pop();
        expect((await drain(readRecordBatches(readerFromChunks(chunks), separator))).flat()).toEqual(expected);
      }
    }
  });

  it("keeps one large record whole across many small chunks", async () => {
    const value = "x".repeat(200_000);
    const record = `data: ${JSON.stringify({ value })}\n\n`;
    const chunks: string[] = [];
    for (let index = 0; index < record.length; index += 512) chunks.push(record.slice(index, index + 512));
    const events = await drain(parseSSE(readerFromChunks(chunks)));
    expect(events).toHaveLength(1);
    expect(JSON.parse(events[0]!.data).value).toHaveLength(200_000);
  });

  it("rejects a completed record past the budget and accepts one at it", async () => {
    const atLimit = "x".repeat(MAX_STREAM_RECORD_CHARS);
    for (const chunks of [[`${atLimit}\n`, "\n"], [`${atLimit}\n\n`], [atLimit.slice(0, 100), `${atLimit.slice(100)}\n\n`]]) {
      expect((await drain(readRecordBatches(readerFromChunks(chunks), "\n\n"))).flat().map((record) => record.length)).toEqual([MAX_STREAM_RECORD_CHARS]);
    }
    for (const chunks of [[`${atLimit}x\n\n`], ["x", `${atLimit}\n\nnext`], [`${atLimit}x`]]) {
      expect(await failure(readRecordBatches(readerFromChunks(chunks), "\n\n"))).toBeInstanceOf(StreamRecordTooLargeError);
    }
  });

  it("splits multi-byte characters across byte chunks", async () => {
    const bytes = encoder.encode('{"a":"€ü"}\n{"b":"ß"}\n');
    const reader = new ReadableStream<Uint8Array>({
      start(controller) {
        for (const byte of bytes) controller.enqueue(new Uint8Array([byte]));
        controller.close();
      },
    }).getReader();
    expect((await drain(readRecordBatches(reader, "\n"))).flat()).toEqual(['{"a":"€ü"}', '{"b":"ß"}']);
  });

  it("stops a record that grows past the budget", async () => {
    let pulls = 0;
    const chunk = encoder.encode("x".repeat(1024 * 1024));
    const reader = () => new ReadableStream<Uint8Array>({
      pull(controller) {
        pulls++;
        controller.enqueue(chunk);
      },
    }).getReader();
    for (const parse of [(source: ReturnType<typeof reader>) => parseSSE(source), (source: ReturnType<typeof reader>) => readRecordBatches(source, "\n")]) {
      pulls = 0;
      const error = await failure(parse(reader()));
      expect(error).toBeInstanceOf(StreamRecordTooLargeError);
      expect(error).toMatchObject({ retryable: false });
      expect(pulls).toBeLessThanOrEqual(MAX_STREAM_RECORD_CHARS / chunk.byteLength + 2);
    }
  });
});

describe("oversized records through adapters", () => {
  const originalFetch = globalThis.fetch;
  afterEach(() => { globalThis.fetch = originalFetch; });

  it("end the stream with a non-retryable provider error", async () => {
    const huge = `data: {"choices":[{"delta":{"content":"${"x".repeat(MAX_STREAM_RECORD_CHARS)}"}}]}\n\n`;
    for (const provider of [
      openAICompatible({ name: "test", model: "test", baseURL: "https://example.test" }),
      endpoint("mistral", { baseURL: "https://example.test" }),
    ]) {
      stubFetch(() => textResponse(huge, "text/event-stream"));
      const issues = [];
      for await (const event of provider.stream({ messages: [{ role: "user", content: [{ type: "text", text: "Hi" }] }] })) {
        if (event.type === "issue") issues.push(event.issue);
      }
      expect(issues).toEqual([expect.objectContaining({ kind: "provider_error", retryable: false })]);
    }
  });
});
