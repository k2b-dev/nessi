import { expect } from "bun:test";
import { completeFromStream } from "../../../src/ai/index.js";
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
 * `complete()` and the stream must describe the same answer: blocks in order (text, reasoning
 * with its signature, tool calls with id, name, and arguments), finish reason, and usage. The
 * fetch stub must answer the first request with the native body and later ones with the stream.
 */
export async function expectProviderContract(provider: Provider, request: GenerateRequest) {
  const result = await provider.complete(request);
  const events = await collectStream(provider, request);
  const streamed = await completeFromStream(provider, request);

  expect(events.some((event) => event.type === "issue")).toBe(false);
  expect(comparable(streamed.message.content)).toEqual(comparable(result.message.content));
  expect(details(streamed.message.content)).toEqual(details(result.message.content));
  expect(streamed.finishReason).toBe(result.finishReason);
  expect(streamed.usage).toEqual(result.usage);

  const streamedBlocks = events.flatMap((event) => (event.type === "block_end" ? [event.block] : []));
  expect(comparable(streamedBlocks)).toEqual(comparable(result.message.content));
  return result;
}

/** Replay data (`details`) of all reasoning blocks, whichever block carries it. */
const details = (blocks: AssistantContentBlock[]) =>
  blocks.flatMap((block) => (block.type === "thinking" && block.details ? block.details : []));
