import { describe, expect, it } from "bun:test";
import { appendAssistantContentBlock } from "../../../src/ai/shared/messages.js";
import { normalizeProviderStream } from "../../../src/ai/shared/tool-stream-normalizer.js";
import type { AssistantContentBlock, RawStreamEvent } from "../../../src/ai/types.js";

async function* source(events: RawStreamEvent[]) {
  for (const event of events) yield event;
}

const blocksOf = async (events: RawStreamEvent[]) => {
  const blocks: AssistantContentBlock[] = [];
  for await (const event of normalizeProviderStream(source(events))) {
    if (event.type === "block_end") blocks.push(event.block);
  }
  return blocks;
};

describe("signed blocks in the stream normalizer", () => {
  it("completes a thinking block with its signature and starts a new one afterwards", async () => {
    expect(await blocksOf([
      { type: "thinking", delta: "first" },
      { type: "thinking", delta: "", signature: "s1" },
      { type: "thinking", delta: "second" },
      { type: "thinking", delta: "", signature: "s2" },
    ])).toEqual([
      { type: "thinking", thinking: "first", signature: "s1" },
      { type: "thinking", thinking: "second", signature: "s2" },
    ]);
  });

  it("keeps signature-only and redacted thinking without text", async () => {
    expect(await blocksOf([
      { type: "thinking", delta: "", redacted: "enc" },
      { type: "thinking", delta: "", signature: "only-sig" },
      { type: "text", delta: "Hi" },
    ])).toEqual([
      { type: "thinking", thinking: "", redacted: "enc" },
      { type: "thinking", thinking: "", signature: "only-sig" },
      { type: "text", text: "Hi" },
    ]);
  });

  it("attaches text and tool call signatures", async () => {
    expect(await blocksOf([
      { type: "text", delta: "Answer" },
      { type: "text", delta: "", signature: "t-sig" },
      { type: "tool_start", callId: "c1", name: "lookup" },
      { type: "tool_call", callId: "c1", name: "lookup", args: {}, signature: "fc-sig" },
    ])).toEqual([
      { type: "text", text: "Answer", signature: "t-sig" },
      { type: "tool_call", id: "c1", name: "lookup", args: {}, signature: "fc-sig" },
    ]);
  });

  it("emits unsigned blocks exactly as before", async () => {
    expect(await blocksOf([{ type: "thinking", delta: "plain" }, { type: "text", delta: "text" }]))
      .toEqual([{ type: "thinking", thinking: "plain" }, { type: "text", text: "text" }]);
  });
});

describe("appendAssistantContentBlock", () => {
  it("merges unsigned thinking but keeps signed and redacted blocks separate", () => {
    const content: AssistantContentBlock[] = [];
    appendAssistantContentBlock(content, { type: "thinking", thinking: "a" });
    appendAssistantContentBlock(content, { type: "thinking", thinking: "b" });
    appendAssistantContentBlock(content, { type: "thinking", thinking: "c", signature: "s" });
    appendAssistantContentBlock(content, { type: "thinking", thinking: "", redacted: "enc" });
    appendAssistantContentBlock(content, { type: "thinking", thinking: "" });

    expect(content).toEqual([
      { type: "thinking", thinking: "ab" },
      { type: "thinking", thinking: "c", signature: "s" },
      { type: "thinking", thinking: "", redacted: "enc" },
    ]);
  });

  it("merges text and keeps the latest text signature", () => {
    const content: AssistantContentBlock[] = [];
    appendAssistantContentBlock(content, { type: "text", text: "Hel" });
    appendAssistantContentBlock(content, { type: "text", text: "lo", signature: "t" });
    appendAssistantContentBlock(content, { type: "text", text: "" });

    expect(content).toEqual([{ type: "text", text: "Hello", signature: "t" }]);
  });
});
