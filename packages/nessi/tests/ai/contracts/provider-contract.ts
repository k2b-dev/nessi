import { expect } from "bun:test";
import { complete } from "../../../src/ai/index.js";
import type { AssistantContentBlock, GenerateRequest, Provider, StreamEvent } from "../../../src/ai/types.js";

export async function collectStream(provider: Provider, request: GenerateRequest): Promise<StreamEvent[]> {
  const events: StreamEvent[] = [];
  for await (const event of provider.stream(request)) {
    events.push(event);
  }
  return events;
}

/**
 * The parts of an answer both paths must agree on; ids and signatures are compared where present.
 * An empty, unsigned reasoning block only carries `details` (OpenRouter's stream attaches them at
 * the end); replay sends the same details either way, so it is left out of the comparison.
 */
const comparable = (blocks: AssistantContentBlock[]) => blocks.filter((block) =>
  !(block.type === "thinking" && !block.thinking && block.signature === undefined && block.redacted === undefined)
).map((block) => {
  if (block.type === "text") return { type: "text", text: block.text };
  if (block.type === "thinking") return { type: "thinking", thinking: block.thinking, signature: block.signature };
  return { type: "tool_call", id: block.id, name: block.name, args: block.args };
});

/**
 * `complete()` must describe exactly the answer the stream emitted: blocks in order (text,
 * reasoning with its signature, tool calls with id, name, and arguments), replay `details`,
 * finish reason, and usage, with no issue on the way. The fetch stub answers every request.
 */
export async function expectProviderContract(provider: Provider, request: GenerateRequest) {
  const events = await collectStream(provider, request);
  const result = await complete(provider, request);

  expect(events.some((event) => event.type === "issue")).toBe(false);
  const streamedBlocks = events.flatMap((event) => (event.type === "block_end" ? [event.block] : []));
  expect(comparable(result.message.content)).toEqual(comparable(streamedBlocks));
  expect(details(result.message.content)).toEqual(details(streamedBlocks));
  const final = events.at(-1);
  expect(final?.type).toBe("usage");
  if (final?.type === "usage") {
    expect(result.finishReason).toBe(final.finishReason ?? "stop");
    expect(result.usage).toEqual(final.usage);
  }
  return result;
}

/** Replay data (`details`) of all reasoning blocks, whichever block carries it. */
const details = (blocks: AssistantContentBlock[]) =>
  blocks.flatMap((block) => (block.type === "thinking" && block.details ? block.details : []));
