import { describe, expect, it } from "bun:test";
import type { ByteStreamReader } from "../../../src/ai/shared/sse.js";
import { parseSSE } from "../../../src/ai/shared/sse.js";

function readerFromChunks(chunks: string[]): ByteStreamReader {
  const encoder = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  }).getReader();
}

describe("parseSSE", () => {
  it("parses multi-line data frames", async () => {
    const events = [];
    for await (const event of parseSSE(readerFromChunks([
      "event: ping\ndata: {\"a\":1}\n",
      "data: {\"b\":2}\n\n",
    ]))) {
      events.push(event);
    }

    expect(events).toEqual([{ event: "ping", data: "{\"a\":1}\n{\"b\":2}", id: undefined }]);
  });

  it("handles chunk boundaries inside frames", async () => {
    const events = [];
    for await (const event of parseSSE(readerFromChunks([
      "data: {\"hel",
      "lo\":\"world\"}\n\n",
    ]))) {
      events.push(event);
    }

    expect(events[0]?.data).toBe("{\"hello\":\"world\"}");
  });

  it("throws a structured first-byte timeout", async () => {
    const reader: ByteStreamReader = {
      read: () => new Promise<never>(() => {}),
    };

    let error: unknown;
    try {
      for await (const _event of parseSSE(reader, { firstByteMs: 1 })) {
        // consume
      }
    } catch (caught) {
      error = caught;
    }

    expect(error).toMatchObject({ scope: "provider_first_byte" });
  });
});
