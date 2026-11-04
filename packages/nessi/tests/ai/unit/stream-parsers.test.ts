import { describe, expect, it } from "bun:test";
import { parseNDJSON } from "../../../src/ai/shared/ndjson.js";
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

  it("parseNDJSON times out on a WHATWG reader instead of ending cleanly", async () => {
    expect(await failure(parseNDJSON(stallingReader(), { firstByteMs: 5 }))).toMatchObject({ scope: "provider_first_byte" });
    expect(await failure(parseNDJSON(stallingReader(['{"a":1}\n']), { idleMs: 5 }))).toMatchObject({ scope: "provider_idle" });
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
    expect(await drain(parseNDJSON(readerFromChunks(['{"a":1}', '\n{"b"', ":2}\n"])))).toEqual([{ a: 1 }, { b: 2 }]);
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

  it("stops a record that grows past the budget", async () => {
    let pulls = 0;
    const chunk = encoder.encode("x".repeat(1024 * 1024));
    const reader = () => new ReadableStream<Uint8Array>({
      pull(controller) {
        pulls++;
        controller.enqueue(chunk);
      },
    }).getReader();
    for (const parse of [parseSSE, parseNDJSON]) {
      pulls = 0;
      expect(await failure(parse(reader()))).toBeInstanceOf(StreamRecordTooLargeError);
      expect(pulls).toBeLessThanOrEqual(MAX_STREAM_RECORD_CHARS / chunk.byteLength + 2);
    }
  });
});
