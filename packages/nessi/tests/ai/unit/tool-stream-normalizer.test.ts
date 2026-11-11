import { describe, expect, it } from "bun:test";
import { normalizeProviderStream } from "../../../src/ai/shared/tool-stream-normalizer.js";
import type { RawStreamEvent, StreamEvent } from "../../../src/ai/types.js";

async function collect(events: RawStreamEvent[], toolArgsPreview?: boolean): Promise<StreamEvent[]> {
  const out: StreamEvent[] = [];
  async function* source() {
    for (const event of events) yield event;
  }
  for await (const event of normalizeProviderStream(source(), { suppressTextAfterMalformedTool: true, toolArgsPreview })) {
    out.push(event);
  }
  return out;
}

describe("normalizeProviderStream", () => {
  it("previews each argument delta before the unchanged final tool block only when opted in", async () => {
    const raw: RawStreamEvent[] = [
      { type: "tool_start", callId: "c1", name: "code_write" },
      { type: "tool_delta", callId: "c1", argsDelta: '{"path":"a.ts",' },
      { type: "tool_delta", callId: "c1", argsDelta: '"content":"hello' },
      { type: "tool_delta", callId: "c1", argsDelta: ' world"}' },
      { type: "tool_call", callId: "c1", name: "code_write", args: { path: "a.ts", content: "hello world" } },
    ];
    const finalBlock: StreamEvent[] = [
      { type: "block_start", blockId: "block-0", index: 0, kind: "tool_call", callId: "c1", name: "code_write" },
      { type: "block_delta", blockId: "block-0", delta: '{"path":"a.ts","content":"hello world"}' },
      { type: "block_end", blockId: "block-0", index: 0, block: {
        type: "tool_call", id: "c1", name: "code_write", args: { path: "a.ts", content: "hello world" },
      } },
    ];
    expect(JSON.stringify(await collect(raw))).toBe(JSON.stringify(finalBlock));
    expect(JSON.stringify(await collect(raw, false))).toBe(JSON.stringify(finalBlock));
    expect(await collect(raw, true)).toEqual([
      { type: "tool_args_delta", callId: "c1", name: "code_write", delta: '{"path":"a.ts",' },
      { type: "tool_args_delta", callId: "c1", name: "code_write", delta: '"content":"hello' },
      { type: "tool_args_delta", callId: "c1", name: "code_write", delta: ' world"}' },
      ...finalBlock,
    ]);
  });

  it.each(["invalid_json", "text", "end", "provider_error"] as const)(
    "keeps previews provisional when the call ends with %s",
    async (ending) => {
      const tail: RawStreamEvent[] = ending === "invalid_json"
        ? [{ type: "tool_call", callId: "c1", name: "code_write", args: {} }]
        : ending === "text"
          ? [
              { type: "text", delta: "interrupted" },
              { type: "tool_delta", callId: "c1", argsDelta: '"ignored"}' },
              { type: "tool_call", callId: "c1", name: "code_write", args: {} },
            ]
          : ending === "provider_error"
            ? [{ type: "error", error: "connection lost", retryable: true }]
            : [];
      const raw: RawStreamEvent[] = [
        { type: "tool_start", callId: "c1", name: "code_write" },
        { type: "tool_delta", callId: "c1", argsDelta: '{"content":' },
        ...tail,
      ];
      const events = await collect(raw, true);
      expect(events[0]).toEqual({ type: "tool_args_delta", callId: "c1", name: "code_write", delta: '{"content":' });
      expect(events[1]).toMatchObject({ type: "issue", issue: {
        kind: ending === "end" || ending === "provider_error" ? "cancelled_tool_call" : "malformed_tool_call",
        callId: "c1",
      } });
      expect(events.filter((event) => event.type === "tool_args_delta")).toHaveLength(1);
      expect(events.some((event) => event.type.startsWith("block_"))).toBe(false);
      expect(await collect(raw)).toEqual(events.filter((event) => event.type !== "tool_args_delta"));
    },
  );

  it("preserves provider order across calls and does not preview a delta without a pending start", async () => {
    const events = await collect([
      { type: "tool_start", callId: "c1", name: "first" },
      { type: "tool_start", callId: "c2", name: "second" },
      { type: "tool_delta", callId: "c2", argsDelta: "{" },
      { type: "tool_delta", callId: "c1", argsDelta: "{}" },
      { type: "tool_delta", callId: "c2", argsDelta: "}" },
      { type: "tool_delta", callId: "unknown", argsDelta: "{}" },
      { type: "tool_call", callId: "c1", name: "first", args: {} },
      { type: "tool_call", callId: "c2", name: "second", args: {} },
    ], true);
    expect(events.filter((event) => event.type === "tool_args_delta")).toEqual([
      { type: "tool_args_delta", callId: "c2", name: "second", delta: "{" },
      { type: "tool_args_delta", callId: "c1", name: "first", delta: "{}" },
      { type: "tool_args_delta", callId: "c2", name: "second", delta: "}" },
    ]);
    expect(events.filter((event) => event.type === "block_end").map((event) => event.index)).toEqual([0, 1]);
  });

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

describe("tool call arguments", () => {
  it("never emits an executable call for non-object arguments", async () => {
    // A broken adapter: the wire value reaches the normalizer unchecked.
    const wireArgs: unknown = ["all"];
    const events = await collect([
      { type: "tool_call", callId: "c1", name: "wipe", args: wireArgs as Record<string, unknown> },
      { type: "usage", usage: { input: 1, output: 1, total: 2 }, finishReason: "tool_use" },
    ]);
    expect(events.some((event) => event.type === "block_end")).toBe(false);
    expect(events).toContainEqual(expect.objectContaining({
      type: "issue",
      issue: expect.objectContaining({ kind: "malformed_tool_call", reason: "invalid_tool_arguments", callId: "c1", name: "wipe" }),
    }));
  });
});
