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

  it("keeps leading whitespace of thinking byte for byte", async () => {
    expect(await blocksOf([
      { type: "thinking", delta: "\n " },
      { type: "thinking", delta: "Reason" },
      { type: "thinking", delta: "", signature: "s" },
    ])).toEqual([{ type: "thinking", thinking: "\n Reason", signature: "s" }]);
  });

  it("keeps signed text parts apart and restores their leading whitespace", async () => {
    expect(await blocksOf([
      { type: "text", delta: "first", signature: "s1" },
      { type: "text", delta: "second", signature: "s2" },
      { type: "text", delta: "\n " },
      { type: "text", delta: "Answer" },
      { type: "text", delta: "", signature: "s3" },
      { type: "text", delta: "tail" },
    ])).toEqual([
      { type: "text", text: "first", signature: "s1" },
      { type: "text", text: "second", signature: "s2" },
      { type: "text", text: "\n Answer", signature: "s3" },
      { type: "text", text: "tail" },
    ]);
  });

  it("drops remembered whitespace when other content intervenes", async () => {
    expect(await blocksOf([
      { type: "thinking", delta: "\n " },
      { type: "text", delta: " " },
      { type: "thinking", delta: "R", signature: "s" },
    ])).toEqual([{ type: "thinking", thinking: "R", signature: "s" }]);
  });

  it("leaves unsigned blocks without leading whitespace as before", async () => {
    expect(await blocksOf([{ type: "thinking", delta: "\n" }, { type: "thinking", delta: "Plain" }, { type: "text", delta: " \n" }, { type: "text", delta: "Hi" }]))
      .toEqual([{ type: "thinking", thinking: "Plain" }, { type: "text", text: "Hi" }]);
  });

  it("does not open a block for whitespace-only text", async () => {
    expect(await blocksOf([{ type: "text", delta: "\n" }, { type: "text", delta: "Hi" }])).toEqual([{ type: "text", text: "Hi" }]);
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

  it("merges unsigned text but keeps signed text in its own block", () => {
    const content: AssistantContentBlock[] = [];
    appendAssistantContentBlock(content, { type: "text", text: "Hel" });
    appendAssistantContentBlock(content, { type: "text", text: "lo" });
    appendAssistantContentBlock(content, { type: "text", text: "first", signature: "s1" });
    appendAssistantContentBlock(content, { type: "text", text: "second", signature: "s2" });
    appendAssistantContentBlock(content, { type: "text", text: "" });

    expect(content).toEqual([
      { type: "text", text: "Hello" },
      { type: "text", text: "first", signature: "s1" },
      { type: "text", text: "second", signature: "s2" },
    ]);
  });
});
