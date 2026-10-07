import { describe, expect, it } from "bun:test";
import type { ByteStreamReader } from "../../../src/ai/shared/sse.js";
import { parseNDJSON } from "../../../src/ai/shared/ndjson.js";

function readerFromChunks(chunks: string[]): ByteStreamReader {
  const encoder = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  }).getReader();
}

describe("parseNDJSON", () => {
  it("parses objects across chunk boundaries", async () => {
    const values: Array<{ a: number }> = [];
    for await (const value of parseNDJSON<{ a: number }>(readerFromChunks([
      "{\"a\":1}\n{\"a",
      "\":2}\n",
    ]))) {
      values.push(value);
    }

    expect(values).toEqual([{ a: 1 }, { a: 2 }]);
  });
});
