import { describe, expect, it } from "bun:test";
import { normalizeProviderStream } from "../../../src/ai/shared/tool-stream-normalizer.js";
import type { RawStreamEvent, StreamEvent } from "../../../src/ai/types.js";

async function collect(events: RawStreamEvent[]): Promise<StreamEvent[]> {
  const out: StreamEvent[] = [];
  async function* source() {
    for (const event of events) yield event;
  }
  for await (const event of normalizeProviderStream(source(), { suppressTextAfterMalformedTool: true })) {
    out.push(event);
  }
  return out;
}

describe("normalizeProviderStream", () => {
  it("emits valid streamed tool calls as executable content blocks", async () => {
    const events = await collect([
      { type: "tool_start", callId: "c1", name: "search" },
      { type: "tool_delta", callId: "c1", argsDelta: "{\"q\":\"x\"}" },
      { type: "tool_call", callId: "c1", name: "search", args: { q: "x" } },
    ]);

    expect(events.map((event) => event.type)).toEqual(["block_start", "block_delta", "block_end"]);
    const end = events.find((event) => event.type === "block_end");
    expect(end?.type === "block_end" ? end.block : undefined).toEqual({
      type: "tool_call",
      id: "c1",
      name: "search",
      args: { q: "x" },
    });
  });

  it("marks pending tool calls malformed when text arrives", async () => {
    const events = await collect([
      { type: "tool_start", callId: "c1", name: "card" },
      { type: "tool_delta", callId: "c1", argsDelta: "{\"title\":\"Hi\"" },
      { type: "text", delta: "</invoke>" },
      { type: "text", delta: "</think>" },
      { type: "tool_call", callId: "c1", name: "card", args: {} },
      { type: "usage", usage: { input: 1, output: 2, total: 3 }, finishReason: "stop" },
    ]);

    expect(events.map((event) => event.type)).toEqual(["issue", "usage"]);
    const issue = events.find((event) => event.type === "issue");
    expect(issue?.type === "issue" && "reason" in issue.issue ? issue.issue.reason : undefined).toBe("text_during_tool_call");
    expect(issue?.type === "issue" && "textDelta" in issue.issue ? issue.issue.textDelta : undefined).toBe("</invoke>");
    expect(events.some((event) => event.type === "block_end" && event.block.type === "tool_call")).toBe(false);
    expect(events.some((event) => event.type === "block_end" && event.block.type === "text")).toBe(false);
  });

  it("does not turn malformed argument JSON into an executable tool call", async () => {
    const events = await collect([
      { type: "tool_start", callId: "c1", name: "card" },
      { type: "tool_delta", callId: "c1", argsDelta: "{\"title\":\"Hi\"" },
      { type: "tool_call", callId: "c1", name: "card", args: {} },
    ]);

    expect(events.map((event) => event.type)).toEqual(["issue"]);
    const issue = events.find((event) => event.type === "issue");
    expect(issue?.type === "issue" && "reason" in issue.issue ? issue.issue.reason : undefined).toBe("invalid_tool_arguments");
  });

  it("cancels pending tool calls when the stream ends", async () => {
    const events = await collect([
      { type: "tool_start", callId: "c1", name: "search" },
      { type: "tool_delta", callId: "c1", argsDelta: "{\"q\":" },
    ]);

    expect(events.map((event) => event.type)).toEqual(["issue"]);
    const cancel = events.find((event) => event.type === "issue");
    expect(cancel?.type === "issue" && "reason" in cancel.issue ? cancel.issue.reason : undefined).toBe("stream_ended_before_tool_call");
    expect(cancel?.type === "issue" ? cancel.issue.kind : undefined).toBe("cancelled_tool_call");
  });

  it("downgrades tool_use finish reason when malformed calls suppressed all executable tools", async () => {
    const events = await collect([
      { type: "tool_start", callId: "c1", name: "card" },
      { type: "tool_delta", callId: "c1", argsDelta: "{\"title\":\"Hi\"" },
      { type: "text", delta: "</invoke>" },
      { type: "usage", usage: { input: 1, output: 2, total: 3 }, finishReason: "tool_use" },
    ]);

    const usage = events.find((event) => event.type === "usage") as Extract<StreamEvent, { type: "usage" }>;
    expect(usage.finishReason).toBe("stop");
  });

  it("cancels pending tool calls before provider errors", async () => {
    const events = await collect([
      { type: "tool_start", callId: "c1", name: "search" },
      { type: "tool_delta", callId: "c1", argsDelta: "{\"q\":\"x\"" },
      { type: "error", error: "network", retryable: true },
    ]);

    expect(events.map((event) => event.type)).toEqual(["issue", "issue"]);
    const cancel = events[0];
    expect(cancel?.type === "issue" && "reason" in cancel.issue ? cancel.issue.reason : undefined).toBe("provider_error_before_tool_call");
    const providerError = events[1];
    expect(providerError?.type === "issue" ? providerError.issue.kind : undefined).toBe("provider_error");
  });
});
