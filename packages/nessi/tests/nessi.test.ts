import { describe, it, expect } from "bun:test";
import { completeFromStream } from "../src/ai/index.js";
import { z } from "zod";
import { nessi } from "../src/nessi.js";
import { defineTool } from "../src/tools.js";
import { memoryStore } from "../src/stores.js";
import { mockProvider, mockProviderMultiTurn } from "./mock-provider.js";
import type {
  CreditStore,
  OutboundEvent,
  Provider,
  ProviderEvent,
  ProviderRequest,
  SessionStore,
  Usage,
} from "../src/types.js";

// Helper: collect all events from a loop
async function collectEvents(loop: ReturnType<typeof nessi>): Promise<OutboundEvent[]> {
  const events: OutboundEvent[] = [];
  for await (const event of loop) {
    events.push(event);
  }
  return events;
}

// Helper: simple server tool
const echoTool = defineTool({
  name: "echo",
  description: "Echoes input",
  inputSchema: z.object({ text: z.string() }),
  outputSchema: z.object({ echoed: z.string() }),
}).server(async (input) => ({ echoed: input.text }));

// Helper: tool with approval
const dangerTool = defineTool({
  name: "danger",
  description: "Needs approval",
  inputSchema: z.object({ action: z.string() }),
  needsApproval: true,
}).server(async (input) => ({ done: input.action }));

// Helper: client tool
const toastTool = defineTool({
  name: "toast",
  description: "Show toast",
  inputSchema: z.object({ message: z.string() }),
  outputSchema: z.object({ shown: z.boolean() }),
}).client((input) => ({ shown: true }));

// Helper: server tool that requests a client-side survey mid-execution
const askSurveyTool = defineTool({
  name: "ask_survey",
  description: "Ask a survey via client tool bridge",
  inputSchema: z.object({}),
  outputSchema: z.object({ result: z.string() }),
}).server(async (_input, ctx) => {
  const response = await ctx.requestClientTool<{ result: string }>("survey", {
    title: "Quick check",
    questions: [{ question: "Pick one", options: ["A", "B"] }],
  });
  return { result: response.result };
});

const surveyTool = defineTool({
  name: "survey",
  description: "Ask a survey on the client",
  inputSchema: z.object({
    title: z.string(),
    questions: z.array(z.object({
      question: z.string(),
      options: z.array(z.string()),
    })),
  }),
  outputSchema: z.object({ result: z.string() }),
}).client(() => ({ result: "A" }));

describe("nessi core loop", () => {
  it("rejects duplicate tool names", () => {
    const first = defineTool({
      name: "dup",
      description: "First",
      inputSchema: z.object({}),
    }).server(async () => ({ ok: true }));
    const second = defineTool({
      name: "dup",
      description: "Second",
      inputSchema: z.object({}),
    }).server(async () => ({ ok: true }));

    expect(() =>
      nessi({
        provider: mockProvider([]),
        store: memoryStore(),
        systemPrompt: "test",
        input: "Hi",
        tools: [first, second],
      }),
    ).toThrow("Duplicate tool name: dup");
  });

  it("keeps static tools available across provider turns", async () => {
    const requestedToolNames: string[][] = [];
    let executions = 0;
    const tool = defineTool({
      name: "static_tool",
      description: "A static tool",
      inputSchema: z.object({}),
    }).server(async () => {
      executions++;
      return { ok: true };
    });
    const provider = mockProviderMultiTurn((request, callIndex) => {
      requestedToolNames.push(request.tools?.map((item) => item.name) ?? []);
      return callIndex === 0
        ? [
            { type: "tool_start", callId: "static-1", name: "static_tool" },
            { type: "tool_call", callId: "static-1", name: "static_tool", args: {} },
          ]
        : [{ type: "text", delta: "Done" }];
    });

    await collectEvents(nessi({
      provider,
      store: memoryStore(),
      systemPrompt: "test",
      input: "Start",
      tools: [tool],
    }));

    expect(requestedToolNames).toEqual([["static_tool"], ["static_tool"]]);
    expect(executions).toBe(1);
  });

  it("uses one dynamic tool snapshot per provider turn", async () => {
    let resolverCalls = 0;
    let legacyExecutions = 0;
    let addedExecutions = 0;
    const requestedToolNames: string[][] = [];

    const unlockTool = defineTool({
      name: "unlock",
      description: "Unlock the next toolset",
      inputSchema: z.object({}),
    }).server(async () => {
      activeTools.splice(0, activeTools.length, unlockTool, addedTool);
      return { unlocked: true };
    });
    const legacyTool = defineTool({
      name: "legacy",
      description: "Only available in the first snapshot",
      inputSchema: z.object({}),
    }).server(async () => {
      legacyExecutions++;
      return { legacy: true };
    });
    const addedTool = defineTool({
      name: "added",
      description: "Available after unlocking",
      inputSchema: z.object({}),
    }).server(async () => {
      addedExecutions++;
      return { added: true };
    });
    const activeTools = [unlockTool, legacyTool];

    const provider = mockProviderMultiTurn((request, callIndex) => {
      requestedToolNames.push(request.tools?.map((item) => item.name) ?? []);
      if (callIndex === 0) {
        return [
          { type: "tool_start", callId: "unlock-1", name: "unlock" },
          { type: "tool_call", callId: "unlock-1", name: "unlock", args: {} },
          { type: "tool_start", callId: "legacy-1", name: "legacy" },
          { type: "tool_call", callId: "legacy-1", name: "legacy", args: {} },
        ];
      }
      if (callIndex === 1) {
        return [
          { type: "tool_start", callId: "added-1", name: "added" },
          { type: "tool_call", callId: "added-1", name: "added", args: {} },
        ];
      }
      return [{ type: "text", delta: "Done" }];
    });

    await collectEvents(nessi({
      provider,
      store: memoryStore(),
      systemPrompt: "test",
      input: "Start",
      tools: async () => {
        resolverCalls++;
        return activeTools;
      },
    }));

    expect(requestedToolNames).toEqual([
      ["unlock", "legacy"],
      ["unlock", "added"],
      ["unlock", "added"],
    ]);
    expect(resolverCalls).toBe(3);
    expect(legacyExecutions).toBe(1);
    expect(addedExecutions).toBe(1);
  });

  it("reports duplicate names from a dynamic tool snapshot", async () => {
    const first = defineTool({
      name: "dup",
      description: "First",
      inputSchema: z.object({}),
    }).server(async () => ({ ok: true }));
    const second = defineTool({
      name: "dup",
      description: "Second",
      inputSchema: z.object({}),
    }).server(async () => ({ ok: true }));
    let providerCalled = false;

    const events = await collectEvents(nessi({
      provider: mockProvider([], { onRequest: () => { providerCalled = true; } }),
      store: memoryStore(),
      systemPrompt: "test",
      input: "Start",
      tools: () => [first, second],
    }));

    expect(providerCalled).toBe(false);
    expect(events.find((event) => event.type === "issue")).toMatchObject({
      type: "issue",
      issue: { kind: "runtime_error", message: "Duplicate tool name: dup" },
    });
    expect(events.at(-1)).toMatchObject({ type: "loop_end", reason: "error" });
  });

  it("reports dynamic tool resolver failures without calling the provider", async () => {
    let providerCalled = false;
    const events = await collectEvents(nessi({
      provider: mockProvider([], { onRequest: () => { providerCalled = true; } }),
      store: memoryStore(),
      systemPrompt: "test",
      input: "Start",
      tools: async () => {
        throw new Error("registry unavailable");
      },
    }));

    expect(providerCalled).toBe(false);
    expect(events.find((event) => event.type === "issue")).toMatchObject({
      type: "issue",
      issue: { kind: "runtime_error", message: "registry unavailable" },
    });
    expect(events.at(-1)).toMatchObject({ type: "loop_end", reason: "error" });
  });

  it("passes the provider tool call ID to server tools", async () => {
    let receivedCallId: string | undefined;
    const tool = defineTool({
      name: "inspect_call",
      description: "Inspect the current call",
      inputSchema: z.object({}),
    }).server(async (_input, ctx) => {
      receivedCallId = ctx.callId;
      return { ok: true };
    });
    const provider = mockProviderMultiTurn((_request, callIndex) => callIndex === 0
      ? [
          { type: "tool_start", callId: "provider-call-42", name: "inspect_call" },
          { type: "tool_call", callId: "provider-call-42", name: "inspect_call", args: {} },
        ]
      : [{ type: "text", delta: "Done" }]);

    await collectEvents(nessi({
      provider,
      store: memoryStore(),
      systemPrompt: "test",
      input: "Start",
      tools: [tool],
    }));

    expect(receivedCallId).toBe("provider-call-42");
  });

  it("emits issue and loop_end(error) when the store throws", async () => {
    const brokenStore: SessionStore = {
      async load() {
        return [];
      },
      async append() {
        throw new Error("store failed");
      },
    };

    const events = await collectEvents(
      nessi({
        provider: mockProvider([{ type: "text", delta: "never reaches provider" }]),
        store: brokenStore,
        systemPrompt: "test",
        input: "Hi",
      }),
    );

    const issue = events.find((event) => event.type === "issue") as Extract<OutboundEvent, { type: "issue" }>;
    expect(issue).toMatchObject({
      type: "issue",
      agentId: "main",
      issue: { kind: "runtime_error", message: "store failed", retryable: false },
    });
    const done = events.find((event) => event.type === "loop_end") as Extract<OutboundEvent, { type: "loop_end" }>;
    expect(done).toMatchObject({
      type: "loop_end",
      agentId: "main",
      reason: "error",
      aggregate: {
        turns: [],
        toolCallCount: 0,
        toolErrorCount: 0,
        assistantMessageCount: 0,
      },
    });
    expect(issue.loopId).toBe(done.loopId);
  });

  it("handles simple text response", async () => {
    const events = await collectEvents(
      nessi({
        provider: mockProvider([
          { type: "text", delta: "Hello " },
          { type: "text", delta: "world!" },
          { type: "usage", usage: { input: 10, output: 5, total: 15 } },
        ]),
        systemPrompt: "test",
        store: memoryStore(),
        input: "Hi",
      }),
    );

    const types = events.map((e) => e.type);
    expect(types).toContain("turn_start");
    expect(types).toContain("block_end");
    expect(types).toContain("turn_end");
    expect(types).toContain("loop_end");

    const textEvents = events.filter((e) => e.type === "block_end" && e.block.type === "text");
    expect(textEvents).toHaveLength(1);
    expect(textEvents[0]?.type === "block_end" && textEvents[0].block.type === "text" ? textEvents[0].block.text : "").toBe("Hello world!");

    const done = events.find((e) => e.type === "loop_end") as any;
    expect(done.reason).toBe("stop");
  });

  it("passes root generation options to the provider", async () => {
    let capturedRequest: any;
    const events = await collectEvents(
      nessi({
        provider: mockProvider(
          [
            { type: "text", delta: "ok" },
            { type: "usage", usage: { input: 1, output: 1, total: 2 } },
          ],
          { onRequest: (request) => { capturedRequest = request; } },
        ),
        systemPrompt: "test",
        store: memoryStore(),
        input: "Hi",
        temperature: 0,
        maxOutputTokens: 64,
        disableReasoning: true,
      }),
    );

    expect(events.some((event) => event.type === "loop_end")).toBe(true);
    expect(capturedRequest.temperature).toBe(0);
    expect(capturedRequest.maxOutputTokens).toBe(64);
    expect(capturedRequest.disableReasoning).toBe(true);
  });

  it("passes reasoningEffort and extraBody to the provider", async () => {
    let capturedRequest: any;
    await collectEvents(
      nessi({
        provider: mockProvider(
          [
            { type: "text", delta: "ok" },
            { type: "usage", usage: { input: 1, output: 1, total: 2 } },
          ],
          { onRequest: (request) => { capturedRequest = request; } },
        ),
        systemPrompt: "test",
        store: memoryStore(),
        input: "Hi",
        reasoningEffort: "high",
        extraBody: { top_k: 20 },
      }),
    );

    expect(capturedRequest.reasoningEffort).toBe("high");
    expect(capturedRequest.extraBody).toEqual({ top_k: 20 });
  });

  it("coalesces adjacent block deltas when requested", async () => {
    const events = await collectEvents(
      nessi({
        provider: mockProvider([
          { type: "text", delta: "a" },
          { type: "text", delta: "b" },
          { type: "text", delta: "c" },
          { type: "usage", usage: { input: 1, output: 3, total: 4 } },
        ]),
        systemPrompt: "test",
        store: memoryStore(),
        input: "Hi",
        coalesce: { maxChars: 2 },
      }),
    );

    const deltas = events
      .filter((event): event is Extract<OutboundEvent, { type: "block_delta" }> => event.type === "block_delta")
      .map((event) => event.delta);
    expect(deltas).toEqual(["ab", "c"]);
  });

  it("classifies malformed half-open tool streams without durable tool UI events", async () => {
    const store = memoryStore();
    const provider = mockProvider([
      { type: "tool_start", callId: "c1", name: "card" },
      { type: "tool_delta", callId: "c1", argsDelta: "{\"title\":\"Hi\"" },
      { type: "text", delta: "</invoke>" },
      { type: "text", delta: "</think>" },
      { type: "tool_call", callId: "c1", name: "card", args: {} },
      { type: "usage", usage: { input: 10, output: 3, total: 13 }, finishReason: "tool_use" },
    ]);

    const events = await collectEvents(
      nessi({
        provider,
        systemPrompt: "test",
        store,
        tools: [echoTool],
        input: "Create card",
      }),
    );

    expect(events.some((event) => event.type === "tool_execution_start")).toBe(false);
    expect(events.some((event) => event.type === "block_end" && event.block.type === "tool_call")).toBe(false);
    expect(events.some((event) => event.type === "block_end" && event.block.type === "text")).toBe(false);

    const issue = events.find((event) => event.type === "issue") as Extract<OutboundEvent, { type: "issue" }>;
    expect(issue.issue.kind).toBe("malformed_tool_call");
    expect(issue.issue.kind === "malformed_tool_call" ? issue.issue.reason : undefined).toBe("text_during_tool_call");
    expect(issue.issue.kind === "malformed_tool_call" ? issue.issue.textDelta : undefined).toBe("</invoke>");

    const done = events.find((event) => event.type === "loop_end") as Extract<OutboundEvent, { type: "loop_end" }>;
    expect(done.aggregate?.toolIssueCount).toBe(1);
    expect(done.aggregate?.toolMalformedCount).toBe(1);
    expect(done.aggregate?.toolCancelledCount).toBe(0);
    expect(done.aggregate?.toolIssues[0]?.reason).toBe("text_during_tool_call");
    expect(done.aggregate?.turns[0]?.stopReason).toBe("stop");

    const assistantMessages = (await store.load()).filter((entry) => entry.message.role === "assistant");
    const persistedText = assistantMessages
      .flatMap((entry) => entry.message.role === "assistant" ? entry.message.content : [])
      .filter((block) => block.type === "text")
      .map((block) => block.text)
      .join("");
    expect(persistedText).not.toContain("</invoke>");
    expect(persistedText).not.toContain("</think>");
  });

  it("aggregates cancelled pending tool streams", async () => {
    const provider = mockProvider([
      { type: "tool_start", callId: "c1", name: "card" },
      { type: "tool_delta", callId: "c1", argsDelta: "{\"title\":\"Hi\"}" },
      { type: "usage", usage: { input: 10, output: 3, total: 13 }, finishReason: "stop" },
    ]);

    const events = await collectEvents(
      nessi({
        provider,
        systemPrompt: "test",
        store: memoryStore(),
        tools: [echoTool],
        input: "Create card",
      }),
    );

    expect(events.some((event) => event.type === "tool_execution_start")).toBe(false);
    expect(events.some((event) => event.type === "block_end" && event.block.type === "tool_call")).toBe(false);

    const cancel = events.find((event) => event.type === "issue") as Extract<OutboundEvent, { type: "issue" }>;
    expect(cancel.issue.kind).toBe("cancelled_tool_call");
    expect(cancel.issue.kind === "cancelled_tool_call" ? cancel.issue.reason : undefined).toBe("stream_ended_before_tool_call");

    const done = events.find((event) => event.type === "loop_end") as Extract<OutboundEvent, { type: "loop_end" }>;
    expect(done.aggregate?.toolIssueCount).toBe(1);
    expect(done.aggregate?.toolMalformedCount).toBe(0);
    expect(done.aggregate?.toolCancelledCount).toBe(1);
    expect(done.aggregate?.toolIssues[0]?.reason).toBe("stream_ended_before_tool_call");
  });

  it("adds a generated loopId to every outbound event", async () => {
    const events = await collectEvents(
      nessi({
        provider: mockProvider([
          { type: "text", delta: "Hello" },
          { type: "usage", usage: { input: 10, output: 5, total: 15 } },
        ]),
        systemPrompt: "test",
        store: memoryStore(),
        input: "Hi",
      }),
    );

    const loopIds = new Set(events.map((event) => event.loopId));
    const loopId = [...loopIds][0];
    expect(loopIds.size).toBe(1);
    expect(typeof loopId).toBe("string");
    expect(loopId.length).toBeGreaterThan(0);
  });

  it("emits loop aggregate metadata on loop_end for multi-turn tool loops", async () => {
    const provider = mockProviderMultiTurn((request, callIndex) => {
      if (callIndex === 0) {
        return [
          { type: "tool_start", callId: "c1", name: "echo" },
          { type: "tool_call", callId: "c1", name: "echo", args: { text: "hello" } },
          { type: "usage", usage: { input: 10, output: 5, total: 15, cacheRead: 2, creditsUsed: 1.5 } },
        ];
      }
      return [
        { type: "text", delta: "The echo said: hello" },
        { type: "usage", usage: { input: 20, output: 10, total: 30, creditsUsed: 2.5 } },
      ];
    });

    const events = await collectEvents(
      nessi({
        loopId: "test-loop-aggregate",
        provider,
        systemPrompt: "test",
        store: memoryStore(),
        tools: [echoTool],
        input: "Echo hello",
      }),
    );

    const done = events.find((e) => e.type === "loop_end") as Extract<OutboundEvent, { type: "loop_end" }>;

    expect(events.every((event) => event.loopId === "test-loop-aggregate")).toBe(true);
    expect(done.loopId).toBe("test-loop-aggregate");
    expect(done.reason).toBe("stop");
    expect(done.aggregate?.assistantMessageCount).toBe(2);
    expect(done.aggregate?.toolCallCount).toBe(1);
    expect(done.aggregate?.toolErrorCount).toBe(0);
    expect(done.aggregate?.usage).toEqual({
      input: 30,
      output: 15,
      total: 45,
      cacheRead: 2,
      creditsUsed: 4,
    });
    expect(done.aggregate?.turns[0]?.stopReason).toBe("tool_use");
    expect(done.aggregate?.turns[0]?.toolCalls).toEqual([
      {
        callId: "c1",
        name: "echo",
        args: { text: "hello" },
        result: { echoed: "hello" },
      },
    ]);
    expect(done.aggregate?.turns[1]?.toolCalls).toEqual([]);
  });

  it("reports loop timing without counting action waits as generation or total elapsed", async () => {
    const originalNow = Date.now;
    let now = 0;
    Date.now = () => now;

    try {
      const timedTool = defineTool({
        name: "timed_danger",
        description: "Needs approval and takes time",
        inputSchema: z.object({ action: z.string() }),
        outputSchema: z.object({ done: z.string() }),
        needsApproval: true,
      }).server(async (input) => {
        now += 300;
        return { done: input.action };
      });

      let streamCalls = 0;
      const timedEvent = <T extends ProviderEvent>(ms: number, event: T): T => {
        now += ms;
        return event;
      };
      const provider: Provider = {
        name: "timed-mock",
        family: "openai-compatible",
        model: "timed-mock",
        capabilities: {
          streaming: true,
          tools: true,
          images: true,
          thinking: true,
          usage: true,
          structuredOutput: true,
        },
        async *stream() {
          const callIndex = streamCalls++;
          if (callIndex === 0) {
            yield timedEvent(100, {
              type: "block_end",
              blockId: "tool-1",
              index: 0,
              block: { type: "tool_call", id: "c1", name: "timed_danger", args: { action: "delete" } },
            });
            yield timedEvent(50, { type: "usage", usage: { input: 10, output: 5, total: 15 } });
            return;
          }
          yield timedEvent(250, {
            type: "block_end",
            blockId: "text-1",
            index: 0,
            block: { type: "text", text: "Done." },
          });
          yield timedEvent(100, { type: "usage", usage: { input: 20, output: 10, total: 30 } });
        },
        complete(request) {
          return completeFromStream(provider, request);
        },
      };

      const loop = nessi({
        provider,
        systemPrompt: "test",
        store: memoryStore(),
        tools: [timedTool],
        input: "Do timed work",
      });

      const events: OutboundEvent[] = [];
      for await (const event of loop) {
        events.push(event);
        if (event.type === "tool_action_request" && event.kind === "approval") {
          now += 500;
          loop.push({ type: "approval_response", callId: event.callId, approved: true });
        } else {
          now += 25;
        }
      }

      const done = events.find((event) => event.type === "loop_end") as Extract<OutboundEvent, { type: "loop_end" }>;
      expect(done.aggregate.timing).toEqual({
        wallMs: 1575,
        totalElapsedMs: 800,
        generationMs: 500,
        toolExecutionMs: 300,
        actionWaitMs: 500,
        outputTokensPerSecond: 30,
      });
    } finally {
      Date.now = originalNow;
    }
  });

  it("stores the provider model name on assistant messages", async () => {
    const provider = {
      ...mockProvider([
        { type: "text" as const, delta: "Hello model" },
        { type: "usage" as const, usage: { input: 10, output: 5, total: 15 } },
      ], { name: "openai" }),
      model: "gpt-4o-mini",
    };

    const events = await collectEvents(
      nessi({
        provider,
        systemPrompt: "test",
        store: memoryStore(),
        input: "Hi",
      }),
    );

    const turnEnd = events.find((event) => event.type === "turn_end");
    expect(turnEnd && turnEnd.type === "turn_end" ? turnEnd.message.model : undefined).toBe("gpt-4o-mini");
  });

  it("handles thinking events", async () => {
    const store = memoryStore();
    const events = await collectEvents(
      nessi({
        provider: mockProvider([
          { type: "thinking", delta: "Let me think..." },
          { type: "text", delta: "Here's my answer." },
          { type: "usage", usage: { input: 10, output: 5, total: 15 } },
        ]),
        systemPrompt: "test",
        store,
        input: "Think about this",
      }),
    );

    const thinkingEvents = events.filter((e) => e.type === "block_end" && e.block.type === "thinking");
    expect(thinkingEvents).toHaveLength(1);
    expect(
      thinkingEvents[0]?.type === "block_end" && thinkingEvents[0].block.type === "thinking"
        ? thinkingEvents[0].block.thinking
        : undefined,
    ).toBe("Let me think...");

    const turnEnd = events.find((event) => event.type === "turn_end") as Extract<OutboundEvent, { type: "turn_end" }>;
    expect(turnEnd.message.content.map((block) => block.type)).toEqual(["thinking", "text"]);

    const assistantMessage = (await store.load()).find((entry) => entry.message.role === "assistant")?.message;
    expect(assistantMessage?.role).toBe("assistant");
    if (assistantMessage?.role === "assistant") {
      expect(assistantMessage.content.map((block) => block.type)).toEqual(["thinking", "text"]);
    }
  });

  it("executes server tool and continues", async () => {
    const provider = mockProviderMultiTurn((request, callIndex) => {
      if (callIndex === 0) {
        return [
          { type: "tool_start", callId: "c1", name: "echo" },
          { type: "tool_delta", callId: "c1", argsDelta: '{"text":' },
          { type: "tool_delta", callId: "c1", argsDelta: '"hello"}' },
          { type: "tool_call", callId: "c1", name: "echo", args: { text: "hello" } },
          { type: "usage", usage: { input: 10, output: 5, total: 15 } },
        ];
      }
      return [
        { type: "text", delta: "The echo said: hello" },
        { type: "usage", usage: { input: 20, output: 10, total: 30 } },
      ];
    });

    const events = await collectEvents(
      nessi({
        provider,
        systemPrompt: "test",
        store: memoryStore(),
        tools: [echoTool],
        input: "Echo hello",
      }),
    );

    const types = events.map((e) => e.type);
    expect(types).toContain("tool_execution_start");
    expect(types).toContain("block_end");
    expect(types).toContain("tool_execution_end");
    // Should have two turn_starts (original + after tool)
    expect(types.filter((t) => t === "turn_start")).toHaveLength(2);

    const toolEnd = events.find((e) => e.type === "tool_execution_end") as any;
    expect(toolEnd.result).toEqual({ echoed: "hello" });

    // turn_end must come after the assistant tool_call block and tool execution.
    const firstTurnEndIdx = types.indexOf("turn_end");
    const toolCallIdx = events.findIndex((event) => event.type === "block_end" && event.block.type === "tool_call");
    const toolEndIdx = types.indexOf("tool_execution_end");
    expect(toolCallIdx).toBeLessThan(firstTurnEndIdx);
    expect(toolEndIdx).toBeLessThan(firstTurnEndIdx);

    const done = events.find((e) => e.type === "loop_end") as any;
    expect(done.reason).toBe("stop");
  });

  it("handles client tool with push()", async () => {
    const provider = mockProviderMultiTurn((request, callIndex) => {
      if (callIndex === 0) {
        return [
          { type: "tool_start", callId: "c1", name: "toast" },
          { type: "tool_call", callId: "c1", name: "toast", args: { message: "Done!" } },
          { type: "usage", usage: { input: 10, output: 5, total: 15 } },
        ];
      }
      return [
        { type: "text", delta: "Toast shown." },
        { type: "usage", usage: { input: 20, output: 10, total: 30 } },
      ];
    });

    const loop = nessi({
      provider,
      systemPrompt: "test",
      store: memoryStore(),
      tools: [toastTool],
      input: "Show a toast",
    });

    const events: OutboundEvent[] = [];
    for await (const event of loop) {
      events.push(event);
      if (event.type === "tool_action_request" && event.kind === "client_tool") {
        loop.push({ type: "tool_result", callId: event.callId, result: { shown: true } });
      }
    }

    const actionReq = events.find((e) => e.type === "tool_action_request") as any;
    expect(actionReq.kind).toBe("client_tool");
    expect(actionReq.name).toBe("toast");

    const toolEnd = events.find((e) => e.type === "tool_execution_end") as any;
    expect(toolEnd.result).toEqual({ shown: true });
  });

  it("validates pushed client tool results", async () => {
    const provider = mockProviderMultiTurn((_request, callIndex) => {
      if (callIndex === 0) {
        return [
          { type: "tool_start", callId: "c1", name: "toast" },
          { type: "tool_call", callId: "c1", name: "toast", args: { message: "Done!" } },
          { type: "usage", usage: { input: 10, output: 5, total: 15 } },
        ];
      }
      return [
        { type: "text", delta: "Handled." },
        { type: "usage", usage: { input: 20, output: 10, total: 30 } },
      ];
    });

    const loop = nessi({
      provider,
      systemPrompt: "test",
      store: memoryStore(),
      tools: [toastTool],
      input: "Show a toast",
    });

    const events: OutboundEvent[] = [];
    for await (const event of loop) {
      events.push(event);
      if (event.type === "tool_action_request" && event.kind === "client_tool") {
        loop.push({ type: "tool_result", callId: event.callId, result: { shown: "yes" } });
      }
    }

    const toolEnd = events.find((event) => event.type === "tool_execution_end") as any;
    expect(toolEnd.isError).toBe(true);
    expect(toolEnd.result).toContain("Output validation error");
    const issue = events.find((event) => event.type === "issue") as Extract<OutboundEvent, { type: "issue" }>;
    expect(issue.issue.kind).toBe("tool_execution_error");
    expect(issue.issue.kind === "tool_execution_error" ? issue.issue.reason : undefined).toBe("output_validation_failed");
  });

  it("times out client tools only when timeoutMs is set on the tool", async () => {
    const slowClientTool = defineTool({
      name: "slow_client",
      description: "Waits for a client result",
      inputSchema: z.object({}),
      outputSchema: z.object({ ok: z.boolean() }),
      timeoutMs: 1,
    }).client(() => ({ ok: true }));
    const provider = mockProviderMultiTurn((_request, callIndex) => {
      if (callIndex === 0) {
        return [
          { type: "tool_start", callId: "c1", name: "slow_client" },
          { type: "tool_call", callId: "c1", name: "slow_client", args: {} },
          { type: "usage", usage: { input: 10, output: 5, total: 15 } },
        ];
      }
      return [
        { type: "text", delta: "Timed out and continued." },
        { type: "usage", usage: { input: 20, output: 10, total: 30 } },
      ];
    });

    const events = await collectEvents(
      nessi({
        provider,
        systemPrompt: "test",
        store: memoryStore(),
        tools: [slowClientTool],
        input: "Run slow client",
      }),
    );

    const issue = events.find((event) => event.type === "issue") as Extract<OutboundEvent, { type: "issue" }>;
    expect(issue.issue.kind).toBe("timeout");
    expect(issue.issue.kind === "timeout" ? issue.issue.scope : undefined).toBe("tool");
    const toolEnd = events.find((event) => event.type === "tool_execution_end") as any;
    expect(toolEnd.isError).toBe(true);
    const done = events.find((event) => event.type === "loop_end") as any;
    expect(done.reason).toBe("stop");
    expect(done.aggregate.toolErrorCount).toBe(1);
  });

  it("removes timed-out inbound waiters before later client tool results", async () => {
    const slowClientTool = defineTool({
      name: "slow_client",
      description: "Waits for a client result",
      inputSchema: z.object({ index: z.number() }),
      outputSchema: z.object({ ok: z.boolean(), index: z.number() }),
      timeoutMs: 20,
    }).client(() => ({ ok: true, index: 0 }));
    const provider = mockProviderMultiTurn((_request, callIndex) => {
      if (callIndex === 0) {
        return [
          { type: "tool_start", callId: "c1", name: "slow_client" },
          { type: "tool_call", callId: "c1", name: "slow_client", args: { index: 1 } },
          { type: "tool_start", callId: "c2", name: "slow_client" },
          { type: "tool_call", callId: "c2", name: "slow_client", args: { index: 2 } },
          { type: "usage", usage: { input: 10, output: 5, total: 15 } },
        ];
      }
      return [
        { type: "text", delta: "Continued." },
        { type: "usage", usage: { input: 20, output: 10, total: 30 } },
      ];
    });

    const loop = nessi({
      provider,
      systemPrompt: "test",
      store: memoryStore(),
      tools: [slowClientTool],
      input: "Run slow clients",
    });

    const events: OutboundEvent[] = [];
    for await (const event of loop) {
      events.push(event);
      if (event.type === "tool_action_request" && event.callId === "c2") {
        loop.push({ type: "tool_result", callId: event.callId, result: { ok: true, index: 2 } });
      }
    }

    const toolEnds = events.filter((event) => event.type === "tool_execution_end") as Array<any>;
    expect(toolEnds.map((event) => [event.callId, event.isError, event.result])).toEqual([
      ["c1", true, expect.stringContaining("timed out")],
      ["c2", undefined, { ok: true, index: 2 }],
    ]);
  });

  it("supports server tool -> requestClientTool() bridge", async () => {
    const provider = mockProviderMultiTurn((request, callIndex) => {
      if (callIndex === 0) {
        return [
          { type: "tool_start", callId: "c1", name: "ask_survey" },
          { type: "tool_call", callId: "c1", name: "ask_survey", args: {} },
          { type: "usage", usage: { input: 10, output: 5, total: 15 } },
        ];
      }
      return [
        { type: "text", delta: "Thanks!" },
        { type: "usage", usage: { input: 20, output: 10, total: 30 } },
      ];
    });

    const loop = nessi({
      provider,
      systemPrompt: "test",
      store: memoryStore(),
      tools: [askSurveyTool],
      input: "Ask survey",
    });

    const events: OutboundEvent[] = [];
    for await (const event of loop) {
      events.push(event);
      if (event.type === "tool_action_request" && event.kind === "client_tool" && event.name === "survey") {
        loop.push({ type: "tool_result", callId: event.callId, result: { result: "Pick one\nA" } });
      }
    }

    const actionReq = events.find((e) => e.type === "tool_action_request") as any;
    expect(actionReq.kind).toBe("client_tool");
    expect(actionReq.name).toBe("survey");

    const toolEnd = events.find((e) => e.type === "tool_execution_end") as any;
    expect(toolEnd.result).toEqual({ result: "Pick one\nA" });
  });

  it("validates registered requestClientTool() bridge results", async () => {
    const provider = mockProviderMultiTurn((_request, callIndex) => {
      if (callIndex === 0) {
        return [
          { type: "tool_start", callId: "c1", name: "ask_survey" },
          { type: "tool_call", callId: "c1", name: "ask_survey", args: {} },
          { type: "usage", usage: { input: 10, output: 5, total: 15 } },
        ];
      }
      return [
        { type: "text", delta: "Handled." },
        { type: "usage", usage: { input: 20, output: 10, total: 30 } },
      ];
    });

    const loop = nessi({
      provider,
      systemPrompt: "test",
      store: memoryStore(),
      tools: [askSurveyTool, surveyTool],
      input: "Ask survey",
    });

    const events: OutboundEvent[] = [];
    for await (const event of loop) {
      events.push(event);
      if (event.type === "tool_action_request" && event.kind === "client_tool" && event.name === "survey") {
        loop.push({ type: "tool_result", callId: event.callId, result: { result: 123 } });
      }
    }

    const issue = events.find((event) =>
      event.type === "issue"
      && event.issue.kind === "tool_execution_error"
      && event.issue.callId === "c1-client-0"
    ) as Extract<OutboundEvent, { type: "issue" }>;
    expect(issue.issue.kind === "tool_execution_error" ? issue.issue.reason : undefined).toBe("output_validation_failed");

    const toolEnd = events.find((event) => event.type === "tool_execution_end" && event.callId === "c1") as any;
    expect(toolEnd.isError).toBe(true);
    expect(toolEnd.result).toContain('Output validation error for client tool "survey"');
  });

  it("buffers out-of-order inbound events by callId", async () => {
    const provider = mockProviderMultiTurn((request, callIndex) => {
      if (callIndex === 0) {
        return [
          { type: "tool_start", callId: "c1", name: "toast" },
          { type: "tool_call", callId: "c1", name: "toast", args: { message: "first" } },
          { type: "tool_start", callId: "c2", name: "toast" },
          { type: "tool_call", callId: "c2", name: "toast", args: { message: "second" } },
          { type: "usage", usage: { input: 10, output: 5, total: 15 } },
        ];
      }
      return [
        { type: "text", delta: "Done." },
        { type: "usage", usage: { input: 20, output: 10, total: 30 } },
      ];
    });

    const loop = nessi({
      provider,
      systemPrompt: "test",
      store: memoryStore(),
      tools: [toastTool],
      input: "Run two client tools",
    });

    const events: OutboundEvent[] = [];
    for await (const event of loop) {
      events.push(event);
      if (event.type === "tool_action_request" && event.kind === "client_tool" && event.callId === "c1") {
        // Push c2 first, then c1. c2 should be buffered until requested.
        loop.push({ type: "tool_result", callId: "c2", result: { shown: true, which: 2 } });
        loop.push({ type: "tool_result", callId: "c1", result: { shown: true, which: 1 } });
      }
    }

    const toolEnds = events.filter((e) => e.type === "tool_execution_end") as Array<any>;
    expect(toolEnds).toHaveLength(2);
    expect(toolEnds[0].callId).toBe("c1");
    expect(toolEnds[0].result).toEqual({ shown: true, which: 1 });
    expect(toolEnds[1].callId).toBe("c2");
    expect(toolEnds[1].result).toEqual({ shown: true, which: 2 });
  });

  it("handles approval flow — approved", async () => {
    const provider = mockProviderMultiTurn((request, callIndex) => {
      if (callIndex === 0) {
        return [
          { type: "tool_start", callId: "c1", name: "danger" },
          { type: "tool_call", callId: "c1", name: "danger", args: { action: "delete" } },
          { type: "usage", usage: { input: 10, output: 5, total: 15 } },
        ];
      }
      return [
        { type: "text", delta: "Done." },
        { type: "usage", usage: { input: 20, output: 10, total: 30 } },
      ];
    });

    const loop = nessi({
      provider,
      systemPrompt: "test",
      store: memoryStore(),
      tools: [dangerTool],
      input: "Do something dangerous",
    });

    const events: OutboundEvent[] = [];
    for await (const event of loop) {
      events.push(event);
      if (event.type === "tool_action_request" && event.kind === "approval") {
        loop.push({ type: "approval_response", callId: event.callId, approved: true });
      }
    }

    const actionReq = events.find((e) => e.type === "tool_action_request") as any;
    expect(actionReq.kind).toBe("approval");

    const toolEnd = events.find((e) => e.type === "tool_execution_end") as any;
    expect(toolEnd.isError).toBeUndefined();
  });

  it("handles approval flow — denied", async () => {
    const provider = mockProviderMultiTurn((request, callIndex) => {
      if (callIndex === 0) {
        return [
          { type: "tool_start", callId: "c1", name: "danger" },
          { type: "tool_call", callId: "c1", name: "danger", args: { action: "delete" } },
          { type: "usage", usage: { input: 10, output: 5, total: 15 } },
        ];
      }
      return [
        { type: "text", delta: "OK, cancelled." },
        { type: "usage", usage: { input: 20, output: 10, total: 30 } },
      ];
    });

    const loop = nessi({
      provider,
      systemPrompt: "test",
      store: memoryStore(),
      tools: [dangerTool],
      input: "Do something dangerous",
    });

    const events: OutboundEvent[] = [];
    for await (const event of loop) {
      events.push(event);
      if (event.type === "tool_action_request" && event.kind === "approval") {
        loop.push({ type: "approval_response", callId: event.callId, approved: false });
      }
    }

    const toolEnd = events.find((e) => e.type === "tool_execution_end") as any;
    expect(toolEnd.isError).toBe(true);
    expect(toolEnd.result).toContain("denied");
  });

  it("stops at maxTurns", async () => {
    // Provider always returns tool calls → infinite loop without maxTurns
    const provider = mockProviderMultiTurn(() => [
      { type: "tool_start", callId: "c1", name: "echo" },
      { type: "tool_call", callId: "c1", name: "echo", args: { text: "loop" } },
      { type: "usage", usage: { input: 10, output: 5, total: 15 } },
    ]);

    const events = await collectEvents(
      nessi({
        provider,
        systemPrompt: "test",
        store: memoryStore(),
        tools: [echoTool],
        maxTurns: 2,
        input: "Loop forever",
      }),
    );

    const done = events.find((e) => e.type === "loop_end") as any;
    expect(done.reason).toBe("max_turns");
    expect(done.aggregate?.assistantMessageCount).toBe(2);
    expect(done.aggregate?.toolCallCount).toBe(2);
    expect(done.aggregate?.toolErrorCount).toBe(0);
    expect(done.aggregate?.usage).toEqual({ input: 20, output: 10, total: 30 });
  });

  it("stops when credits run out", async () => {
    const creditStore: CreditStore = {
      async remaining() {
        return 0;
      },
      async deduct() {},
    };

    const events = await collectEvents(
      nessi({
        provider: mockProvider([
          { type: "text", delta: "hello" },
          { type: "usage", usage: { input: 10, output: 5, total: 15 } },
        ]),
        systemPrompt: "test",
        store: memoryStore(),
        creditStore,
        input: "Hi",
      }),
    );

    const done = events.find((e) => e.type === "loop_end") as any;
    expect(done.reason).toBe("no_credits");
    // Should not have any text events — stopped before provider call
    const textEvents = events.filter((e) => e.type === "block_end" && e.block.type === "text");
    expect(textEvents).toHaveLength(0);
  });

  it("deducts credits after turn", async () => {
    let deducted = 0;
    const creditStore: CreditStore = {
      async remaining() {
        return 100;
      },
      async deduct(credits) {
        deducted += credits;
      },
    };

    await collectEvents(
      nessi({
        provider: mockProvider([
          { type: "text", delta: "hello" },
          { type: "usage", usage: { input: 10, output: 5, total: 15, creditsUsed: 7 } },
        ]),
        systemPrompt: "test",
        store: memoryStore(),
        creditStore,
        input: "Hi",
      }),
    );

    expect(deducted).toBe(7);
  });

  it("includes persisted assistant turn in aggregate when credit deduction fails", async () => {
    const store = memoryStore();
    const creditStore: CreditStore = {
      async remaining() {
        return 100;
      },
      async deduct() {
        throw new Error("credit write failed");
      },
    };

    const events = await collectEvents(
      nessi({
        provider: mockProvider([
          { type: "text", delta: "Charged answer." },
          { type: "usage", usage: { input: 10, output: 5, total: 15, creditsUsed: 7 } },
        ]),
        systemPrompt: "test",
        store,
        creditStore,
        input: "Hi",
      }),
    );

    const done = events.find((e) => e.type === "loop_end") as any;
    expect(done.reason).toBe("error");
    expect(done.aggregate?.assistantMessageCount).toBe(1);
    expect(done.aggregate?.usage).toEqual({ input: 10, output: 5, total: 15, creditsUsed: 7 });

    const entries = await store.load();
    expect(entries.filter((entry) => entry.message.role === "assistant")).toHaveLength(1);
  });

  it("handles abort signal", async () => {
    const controller = new AbortController();
    controller.abort(); // abort immediately

    const events = await collectEvents(
      nessi({
        provider: mockProvider([
          { type: "text", delta: "hello" },
          { type: "usage", usage: { input: 10, output: 5, total: 15 } },
        ]),
        systemPrompt: "test",
        store: memoryStore(),
        signal: controller.signal,
        input: "Hi",
      }),
    );

    const done = events.find((e) => e.type === "loop_end") as any;
    expect(done.reason).toBe("aborted");
  });

  it("includes loopId on abort() loop events", async () => {
    const loop = nessi({
      loopId: "abort-loop",
      provider: mockProvider([
        { type: "text", delta: "hello" },
        { type: "usage", usage: { input: 10, output: 5, total: 15 } },
      ]),
      systemPrompt: "test",
      store: memoryStore(),
      input: "Hi",
    });
    const subscribedEvents: OutboundEvent[] = [];
    loop.subscribe((event) => subscribedEvents.push(event));

    loop.abort();
    const events = await collectEvents(loop);

    const done = events.find((e) => e.type === "loop_end");
    expect(done?.loopId).toBe("abort-loop");
    expect(events.every((event) => event.loopId === "abort-loop")).toBe(true);
    expect(subscribedEvents.every((event) => event.loopId === "abort-loop")).toBe(true);
  });

  it("preserves stream block order for interrupted assistant messages", async () => {
    const store = memoryStore();
    const loop = nessi({
      provider: mockProvider([
        { type: "thinking", delta: "Reason first." },
        { type: "text", delta: "Partial answer." },
        { type: "usage", usage: { input: 10, output: 5, total: 15 } },
      ]),
      systemPrompt: "test",
      store,
      input: "Think then answer",
    });

    loop.subscribe((event) => {
      if (event.type === "usage") loop.abort();
    });

    const events = await collectEvents(loop);

    const turnEnd = events.find((event) => event.type === "turn_end") as Extract<OutboundEvent, { type: "turn_end" }>;
    expect(turnEnd.message.stopReason).toBe("interrupted");
    expect(turnEnd.message.content.map((block) => block.type)).toEqual(["thinking", "text"]);

    const assistantMessage = (await store.load()).find((entry) => entry.message.role === "assistant")?.message;
    expect(assistantMessage?.role).toBe("assistant");
    if (assistantMessage?.role === "assistant") {
      expect(assistantMessage.stopReason).toBe("interrupted");
      expect(assistantMessage.content.map((block) => block.type)).toEqual(["thinking", "text"]);
    }
  });

  it("does not persist empty interrupted assistant messages from empty deltas", async () => {
    const store = memoryStore();
    const loop = nessi({
      provider: mockProvider([
        { type: "thinking", delta: "" },
        { type: "text", delta: "" },
        { type: "usage", usage: { input: 10, output: 0, total: 10 } },
      ]),
      systemPrompt: "test",
      store,
      input: "Empty stream",
    });

    loop.subscribe((event) => {
      if (event.type === "usage") loop.abort();
    });

    const events = await collectEvents(loop);

    // The turn is closed, but nothing empty is stored; the reported usage still counts.
    const turnEnd = events.find((event) => event.type === "turn_end") as Extract<OutboundEvent, { type: "turn_end" }>;
    expect(turnEnd.message).toMatchObject({ content: [], stopReason: "interrupted" });
    const done = events.find((event) => event.type === "loop_end") as Extract<OutboundEvent, { type: "loop_end" }>;
    expect(done.reason).toBe("aborted");
    expect(done.aggregate.usage).toMatchObject({ input: 10, output: 0, total: 10 });

    const entries = await store.load();
    expect(entries.filter((entry) => entry.message.role === "assistant")).toHaveLength(0);
  });

  it("handles input validation error", async () => {
    const provider = mockProviderMultiTurn((request, callIndex) => {
      if (callIndex === 0) {
        return [
          { type: "tool_start", callId: "c1", name: "echo" },
          // Invalid args: number instead of string
          { type: "tool_call", callId: "c1", name: "echo", args: { text: 123 } },
          { type: "usage", usage: { input: 10, output: 5, total: 15 } },
        ];
      }
      return [
        { type: "text", delta: "Sorry about that." },
        { type: "usage", usage: { input: 20, output: 10, total: 30 } },
      ];
    });

    const events = await collectEvents(
      nessi({
        provider,
        systemPrompt: "test",
        store: memoryStore(),
        tools: [echoTool],
        input: "Echo something",
      }),
    );

    const toolEnd = events.find((e) => e.type === "tool_execution_end") as any;
    expect(toolEnd.isError).toBe(true);
    expect(toolEnd.result).toContain("Validation");
    // The assistant tool_call block is preserved, and execution attempts remain start/end paired.
    const toolCalls = events.filter((e) => e.type === "block_end" && e.block.type === "tool_call");
    expect(toolCalls).toHaveLength(1);
    const toolStart = events.find((event) => event.type === "tool_execution_start") as any;
    expect(toolStart.args).toEqual({ text: 123 });

    const done = events.find((e) => e.type === "loop_end") as any;
    expect(done.aggregate?.toolCallCount).toBe(1);
    expect(done.aggregate?.toolErrorCount).toBe(1);
    expect(done.aggregate?.turns[0]?.toolCalls[0]?.args).toEqual({ text: 123 });
    expect(done.aggregate?.turns[0]?.toolCalls[0]?.isError).toBe(true);
    expect(done.aggregate?.turns[0]?.toolCalls[0]?.result).toContain("Validation");
  });

  it("handles output validation error", async () => {
    const badTool = defineTool({
      name: "bad",
      description: "Returns wrong type",
      inputSchema: z.object({}),
      outputSchema: z.object({ count: z.number() }),
    }).server(async () => ({ count: "not a number" }) as any);

    const provider = mockProviderMultiTurn((request, callIndex) => {
      if (callIndex === 0) {
        return [
          { type: "tool_start", callId: "c1", name: "bad" },
          { type: "tool_call", callId: "c1", name: "bad", args: {} },
          { type: "usage", usage: { input: 10, output: 5, total: 15 } },
        ];
      }
      return [
        { type: "text", delta: "Handled." },
        { type: "usage", usage: { input: 20, output: 10, total: 30 } },
      ];
    });

    const events = await collectEvents(
      nessi({
        provider,
        systemPrompt: "test",
        store: memoryStore(),
        tools: [badTool],
        input: "Do it",
      }),
    );

    const toolEnd = events.find((e) => e.type === "tool_execution_end") as any;
    expect(toolEnd.isError).toBe(true);
    expect(toolEnd.result).toContain("Output validation");
  });

  it("handles unknown tool name", async () => {
    const provider = mockProviderMultiTurn((request, callIndex) => {
      if (callIndex === 0) {
        return [
          { type: "tool_start", callId: "c1", name: "nonexistent" },
          { type: "tool_call", callId: "c1", name: "nonexistent", args: {} },
          { type: "usage", usage: { input: 10, output: 5, total: 15 } },
        ];
      }
      return [
        { type: "text", delta: "OK." },
        { type: "usage", usage: { input: 20, output: 10, total: 30 } },
      ];
    });

    const events = await collectEvents(
      nessi({
        provider,
        systemPrompt: "test",
        store: memoryStore(),
        tools: [echoTool],
        input: "Call nonexistent",
      }),
    );

    const toolEnd = events.find((e) => e.type === "tool_execution_end") as any;
    expect(toolEnd.isError).toBe(true);
    expect(toolEnd.result).toContain("Unknown tool");
  });

  it("handles compaction before provider call", async () => {
    let compactCalled = false;
    const store = memoryStore();

    const events = await collectEvents(
      nessi({
        provider: mockProvider([
          { type: "text", delta: "Compacted!" },
          { type: "usage", usage: { input: 10, output: 5, total: 15 } },
        ]),
        systemPrompt: "test",
        store,
        compact(ctx) {
          if (compactCalled) return null;
          compactCalled = true;
          return (async () => {
            await ctx.store.append(
              { role: "user", content: [{ type: "text", text: "Summary" }] },
              { seq: 0, kind: "summary" },
            );
          })();
        },
        input: "Hi",
      }),
    );

    expect(compactCalled).toBe(true);
    const types = events.map((e) => e.type);
    expect(types).toContain("compaction_start");
    expect(types).toContain("compaction_end");
    // compaction_start should come before turn_start
    const csIdx = types.indexOf("compaction_start");
    const tsIdx = types.indexOf("turn_start");
    expect(csIdx).toBeLessThan(tsIdx);
  });

  it("emits compaction_end before issue when root compaction rejects asynchronously", async () => {
    const events = await collectEvents(
      nessi({
        provider: mockProvider([
          { type: "text", delta: "Should not run" },
          { type: "usage", usage: { input: 10, output: 5, total: 15 } },
        ]),
        systemPrompt: "test",
        store: memoryStore(),
        compact: async () => {
          await Promise.resolve();
          throw new Error("async compact failed");
        },
        input: "Hi",
      }),
    );

    const types = events.map((event) => event.type);
    const startIdx = types.indexOf("compaction_start");
    const endIdx = types.indexOf("compaction_end");
    const issueIdx = types.indexOf("issue");
    const loopEndIdx = types.indexOf("loop_end");
    expect(startIdx).toBeGreaterThan(-1);
    expect(endIdx).toBeGreaterThan(startIdx);
    expect(issueIdx).toBeGreaterThan(endIdx);
    expect(loopEndIdx).toBeGreaterThan(issueIdx);
    const issue = events[issueIdx];
    expect(issue.type).toBe("issue");
    if (issue.type !== "issue") return;
    expect(issue.issue.message).toContain("async compact failed");
  });

  it("handles context overflow with compaction retry", async () => {
    let callCount = 0;
    const provider: typeof echoTool extends never ? never : import("../src/types.js").Provider = {
      name: "mock",
      family: "openai-compatible",
      model: "mock",
      capabilities: {
        streaming: true,
        tools: true,
        images: true,
        thinking: true,
        usage: true,
      },
      contextWindow: 100_000,
      async *stream() {
        callCount++;
        if (callCount === 1) {
          yield {
            type: "issue" as const,
            issue: { kind: "provider_error" as const, message: "context too long", retryable: false, contextOverflow: true },
          };
          return;
        }
        yield { type: "block_start" as const, blockId: "b0", index: 0, kind: "text" as const };
        yield { type: "block_delta" as const, blockId: "b0", delta: "After compaction" };
        yield { type: "block_end" as const, blockId: "b0", index: 0, block: { type: "text" as const, text: "After compaction" } };
        yield { type: "usage" as const, usage: { input: 10, output: 5, total: 15 } };
      },
      complete(request) {
        return completeFromStream(provider, request);
      },
    };

    const events = await collectEvents(
      nessi({
        provider,
        systemPrompt: "test",
        store: memoryStore(),
        compact(ctx) {
          if (!ctx.force) return null;
          return (async () => {
            await ctx.store.append(
              { role: "user", content: [{ type: "text", text: "Compacted summary" }] },
              { seq: 0, kind: "summary" },
            );
          })();
        },
        input: "Hi",
      }),
    );

    expect(callCount).toBe(2);
    const types = events.map((e) => e.type);
    expect(types).toContain("compaction_start");
    expect(types).toContain("compaction_end");
    const done = events.find((e) => e.type === "loop_end") as any;
    expect(done.reason).toBe("stop");
  });

  it("context overflow without compact -> loop_end(context_overflow)", async () => {
    const events = await collectEvents(
      nessi({
        provider: mockProvider([{ type: "error", error: "context too long", retryable: false, contextOverflow: true }]),
        systemPrompt: "test",
        store: memoryStore(),
        input: "Hi",
      }),
    );

    const done = events.find((e) => e.type === "loop_end") as any;
    expect(done.reason).toBe("context_overflow");
  });

  it("context overflow with compaction retry that also fails -> no infinite loop", async () => {
    const provider: import("../src/types.js").Provider = {
      name: "mock",
      family: "openai-compatible",
      model: "mock",
      capabilities: {
        streaming: true,
        tools: true,
        images: true,
        thinking: true,
        usage: true,
      },
      contextWindow: 100_000,
      async *stream() {
        // Always overflow
        yield {
          type: "issue" as const,
          issue: { kind: "provider_error" as const, message: "context too long", retryable: false, contextOverflow: true },
        };
      },
      complete(request) {
        return completeFromStream(provider, request);
      },
    };

    const events = await collectEvents(
      nessi({
        provider,
        systemPrompt: "test",
        store: memoryStore(),
        compact(ctx) {
          return (async () => {
            await ctx.store.append(
              { role: "user", content: [{ type: "text", text: "Summary" }] },
              { seq: 0, kind: "summary" },
            );
          })();
        },
        input: "Hi",
      }),
    );

    const done = events.find((e) => e.type === "loop_end") as any;
    expect(done.reason).toBe("context_overflow");
    // Should have exactly 2 compaction rounds (normal + force retry), then give up
    const compactionStarts = events.filter((e) => e.type === "compaction_start");
    expect(compactionStarts.length).toBeLessThanOrEqual(2);
  });

  it("emits compaction_end before issue when context-overflow compaction rejects asynchronously", async () => {
    const provider: import("../src/types.js").Provider = {
      name: "mock",
      family: "openai-compatible",
      model: "mock",
      capabilities: {
        streaming: true,
        tools: true,
        images: true,
        thinking: true,
        usage: true,
      },
      contextWindow: 100_000,
      async *stream() {
        yield {
          type: "issue" as const,
          issue: { kind: "provider_error" as const, message: "context too long", retryable: false, contextOverflow: true },
        };
      },
      complete(request) {
        return completeFromStream(provider, request);
      },
    };

    const events = await collectEvents(
      nessi({
        provider,
        systemPrompt: "test",
        store: memoryStore(),
        compact(ctx) {
          if (!ctx.force) return null;
          return (async () => {
            await Promise.resolve();
            throw new Error("overflow compact failed");
          })();
        },
        input: "Hi",
      }),
    );

    const types = events.map((event) => event.type);
    const startIdx = types.indexOf("compaction_start");
    const endIdx = types.indexOf("compaction_end");
    const issueIdx = types.findIndex((type, index) => type === "issue" && index > endIdx);
    const loopEndIdx = types.indexOf("loop_end");
    expect(startIdx).toBeGreaterThan(-1);
    expect(endIdx).toBeGreaterThan(startIdx);
    expect(issueIdx).toBeGreaterThan(endIdx);
    expect(loopEndIdx).toBeGreaterThan(issueIdx);
    const issue = events[issueIdx];
    expect(issue.type).toBe("issue");
    if (issue.type !== "issue") return;
    expect(issue.issue.message).toContain("overflow compact failed");
  });

  it("subscribe() receives events in parallel", async () => {
    const subscribedEvents: OutboundEvent[] = [];

    const loop = nessi({
      provider: mockProvider([
        { type: "text", delta: "Hello" },
        { type: "usage", usage: { input: 10, output: 5, total: 15 } },
      ]),
      systemPrompt: "test",
      store: memoryStore(),
      input: "Hi",
    });

    const unsub = loop.subscribe((event) => subscribedEvents.push(event));

    const iteratedEvents: OutboundEvent[] = [];
    for await (const event of loop) {
      iteratedEvents.push(event);
    }

    unsub();

    // Both should have received the same events
    expect(subscribedEvents.length).toBe(iteratedEvents.length);
    expect(subscribedEvents.map((e) => e.type)).toEqual(iteratedEvents.map((e) => e.type));
  });

  it("handles multiple tools in one turn", async () => {
    const provider = mockProviderMultiTurn((request, callIndex) => {
      if (callIndex === 0) {
        return [
          { type: "tool_start", callId: "c1", name: "echo" },
          { type: "tool_call", callId: "c1", name: "echo", args: { text: "first" } },
          { type: "tool_start", callId: "c2", name: "echo" },
          { type: "tool_call", callId: "c2", name: "echo", args: { text: "second" } },
          { type: "usage", usage: { input: 10, output: 5, total: 15 } },
        ];
      }
      return [
        { type: "text", delta: "Both done." },
        { type: "usage", usage: { input: 20, output: 10, total: 30 } },
      ];
    });

    const events = await collectEvents(
      nessi({
        provider,
        systemPrompt: "test",
        store: memoryStore(),
        tools: [echoTool],
        input: "Echo two things",
      }),
    );

    const toolEnds = events.filter((e) => e.type === "tool_execution_end");
    expect(toolEnds).toHaveLength(2);
    expect((toolEnds[0] as any).result).toEqual({ echoed: "first" });
    expect((toolEnds[1] as any).result).toEqual({ echoed: "second" });
  });

  it("stores messages correctly across turns", async () => {
    const store = memoryStore();

    const provider = mockProviderMultiTurn((request, callIndex) => {
      if (callIndex === 0) {
        return [
          { type: "tool_start", callId: "c1", name: "echo" },
          { type: "tool_call", callId: "c1", name: "echo", args: { text: "test" } },
          { type: "usage", usage: { input: 10, output: 5, total: 15 } },
        ];
      }
      return [
        { type: "text", delta: "Final." },
        { type: "usage", usage: { input: 20, output: 10, total: 30 } },
      ];
    });

    await collectEvents(
      nessi({
        provider,
        systemPrompt: "test",
        store,
        tools: [echoTool],
        input: "Echo test",
      }),
    );

    const entries = await store.load();
    // user message, assistant (tool_call), tool_result, assistant (text)
    expect(entries).toHaveLength(4);
    expect(entries[0].message.role).toBe("user");
    expect(entries[1].message.role).toBe("assistant");
    expect(entries[2].message.role).toBe("tool_result");
    expect(entries[3].message.role).toBe("assistant");
  });

  it("handles tool execution error gracefully", async () => {
    const crashTool = defineTool({
      name: "crash",
      description: "Always throws",
      inputSchema: z.object({}),
    }).server(async () => {
      throw new Error("boom");
    });

    const provider = mockProviderMultiTurn((request, callIndex) => {
      if (callIndex === 0) {
        return [
          { type: "tool_start", callId: "c1", name: "crash" },
          { type: "tool_call", callId: "c1", name: "crash", args: {} },
          { type: "usage", usage: { input: 10, output: 5, total: 15 } },
        ];
      }
      return [
        { type: "text", delta: "Handled the error." },
        { type: "usage", usage: { input: 20, output: 10, total: 30 } },
      ];
    });

    const events = await collectEvents(
      nessi({
        provider,
        systemPrompt: "test",
        store: memoryStore(),
        tools: [crashTool],
        input: "Crash",
      }),
    );

    const toolEnd = events.find((e) => e.type === "tool_execution_end") as any;
    expect(toolEnd.isError).toBe(true);
    expect(toolEnd.result).toContain("boom");

    // Should still continue to next turn
    const done = events.find((e) => e.type === "loop_end") as any;
    expect(done.reason).toBe("stop");
    expect(done.aggregate?.toolErrorCount).toBe(1);
    expect(done.aggregate?.turns[0]?.toolCalls[0]?.result).toContain("boom");
    expect(done.aggregate?.turns[0]?.toolCalls[0]?.isError).toBe(true);
  });

  it("sets agentId on all events", async () => {
    const events = await collectEvents(
      nessi({
        agentId: "custom-agent",
        provider: mockProvider([
          { type: "text", delta: "Hello" },
          { type: "usage", usage: { input: 10, output: 5, total: 15 } },
        ]),
        systemPrompt: "test",
        store: memoryStore(),
        input: "Hi",
      }),
    );

    for (const event of events) {
      expect((event as any).agentId).toBe("custom-agent");
    }
  });

  it("defaults agentId to 'main'", async () => {
    const events = await collectEvents(
      nessi({
        provider: mockProvider([
          { type: "text", delta: "Hello" },
          { type: "usage", usage: { input: 10, output: 5, total: 15 } },
        ]),
        systemPrompt: "test",
        store: memoryStore(),
        input: "Hi",
      }),
    );

    for (const event of events) {
      expect((event as any).agentId).toBe("main");
    }
  });

  it("provider error (non-retryable) -> loop_end(error)", async () => {
    const events = await collectEvents(
      nessi({
        provider: mockProvider([{ type: "error", error: "API key invalid", retryable: false }]),
        systemPrompt: "test",
        store: memoryStore(),
        input: "Hi",
      }),
    );

    const types = events.map((e) => e.type);
    expect(types).toContain("issue");
    const done = events.find((e) => e.type === "loop_end") as any;
    expect(done.reason).toBe("error");
    expect(done.aggregate?.assistantMessageCount).toBe(0);
    expect(done.aggregate?.usage).toBeUndefined();
  });

  it("provider error (retryable) closes the turn without storing an empty assistant message", async () => {
    const store = memoryStore();
    const events = await collectEvents(
      nessi({
        provider: mockProvider([{ type: "error", error: "Rate limited", retryable: true }]),
        systemPrompt: "test",
        store,
        input: "Hi",
      }),
    );

    const types = events.map((e) => e.type);
    expect(types).toEqual(["loop_start", "turn_start", "issue", "turn_end", "loop_end"]);
    expect((events.find((e) => e.type === "turn_end") as any).message.stopReason).toBe("error");
    const done = events.find((e) => e.type === "loop_end") as any;
    expect(done.reason).toBe("error");

    const entries = await store.load();
    const assistantMessages = entries.filter((entry) => entry.message.role === "assistant");
    expect(assistantMessages).toHaveLength(0);
  });

  it("steer() injects message before next provider call", async () => {
    const requests: import("../src/types.js").ProviderRequest[] = [];
    const store = memoryStore();

    const provider = mockProviderMultiTurn((request, callIndex) => {
      requests.push(request);
      if (callIndex === 0) {
        return [
          { type: "tool_start", callId: "c1", name: "echo" },
          { type: "tool_call", callId: "c1", name: "echo", args: { text: "hi" } },
          { type: "usage", usage: { input: 10, output: 5, total: 15 } },
        ];
      }
      return [
        { type: "text", delta: "Done." },
        { type: "usage", usage: { input: 20, output: 10, total: 30 } },
      ];
    });

    const loop = nessi({
      provider,
      systemPrompt: "test",
      store,
      tools: [echoTool],
      input: "Start",
    });

    const events: OutboundEvent[] = [];
    for await (const event of loop) {
      events.push(event);
      // Steer after first tool completes
      if (event.type === "tool_execution_end") {
        loop.steer("Focus on X");
      }
    }

    // steer_applied should be in the event stream
    const steerEvents = events.filter((e) => e.type === "steer_applied");
    expect(steerEvents).toHaveLength(1);
    expect((steerEvents[0] as any).message).toBe("Focus on X");

    // The second provider call should see the steer message
    expect(requests).toHaveLength(2);
    const secondMessages = requests[1].messages;
    const steerMsg = secondMessages.find(
      (m) => m.role === "user" && m.content.some((c: any) => c.text === "Focus on X"),
    );
    expect(steerMsg).toBeTruthy();
  });

  it("applies supplied steering messages in order before a provider call", async () => {
    const requests: import("../src/types.js").ProviderRequest[] = [];
    const store = memoryStore();
    let steeringCalls = 0;
    const provider = mockProviderMultiTurn((request) => {
      requests.push(request);
      return [
        { type: "text", delta: "Done." },
        { type: "usage", usage: { input: 20, output: 10, total: 30 } },
      ];
    });

    const events = await collectEvents(nessi({
      provider,
      systemPrompt: "test",
      store,
      input: "Start",
      steering: ({ agentId, loopId, signal }) => {
        steeringCalls++;
        expect(agentId).toBe("main");
        expect(loopId).toBeTruthy();
        expect(signal.aborted).toBe(false);
        return steeringCalls === 1 ? ["First", "Second"] : undefined;
      },
    }));

    expect(requests).toHaveLength(1);
    const userTexts = requests[0]!.messages
      .filter((message) => message.role === "user")
      .flatMap((message) => message.content)
      .filter((part): part is Extract<typeof part, { type: "text" }> => typeof part !== "string" && part.type === "text")
      .map((part) => part.text);
    expect(userTexts).toEqual(["Start", "First", "Second"]);
    expect(events.filter((event) => event.type === "steer_applied").map((event) => event.message)).toEqual([
      "First",
      "Second",
    ]);
  });

  it("checks supplied steering before a normal loop end", async () => {
    const requests: import("../src/types.js").ProviderRequest[] = [];
    let steeringCalls = 0;
    const provider = mockProviderMultiTurn((request, callIndex) => {
      requests.push(request);
      return [
        { type: "text", delta: callIndex === 0 ? "Initial answer." : "Revised answer." },
        { type: "usage", usage: { input: 20, output: 10, total: 30 } },
      ];
    });

    const events = await collectEvents(nessi({
      provider,
      systemPrompt: "test",
      store: memoryStore(),
      input: "Start",
      steering: () => {
        steeringCalls++;
        return steeringCalls === 2 ? "Revise the answer" : undefined;
      },
    }));

    expect(requests).toHaveLength(2);
    expect(requests[1]!.messages.some(
      (message) => message.role === "user"
        && message.content.some((part) => typeof part !== "string" && part.type === "text" && part.text === "Revise the answer"),
    )).toBe(true);
    expect(events.filter((event) => event.type === "turn_end")).toHaveLength(2);
    expect(events.filter((event) => event.type === "steer_applied")).toHaveLength(1);
    expect(events.at(-1)?.type).toBe("loop_end");
  });

  it("applies local steer() before a normal loop end", async () => {
    const requests: import("../src/types.js").ProviderRequest[] = [];
    const provider = mockProviderMultiTurn((request, callIndex) => {
      requests.push(request);
      return [
        { type: "text", delta: callIndex === 0 ? "Initial answer." : "Revised answer." },
        { type: "usage", usage: { input: 20, output: 10, total: 30 } },
      ];
    });
    const loop = nessi({
      provider,
      systemPrompt: "test",
      store: memoryStore(),
      input: "Start",
    });

    const events: OutboundEvent[] = [];
    for await (const event of loop) {
      events.push(event);
      if (event.type === "turn_end" && requests.length === 1) loop.steer("Revise the answer");
    }

    expect(requests).toHaveLength(2);
    expect(events.filter((event) => event.type === "steer_applied")).toHaveLength(1);
  });

  it("steer() resets turn counter", async () => {
    let callCount = 0;
    const provider = mockProviderMultiTurn((request, callIndex) => {
      callCount++;
      // Always return a tool call to keep looping
      if (callIndex < 3) {
        return [
          { type: "tool_start", callId: `c${callIndex}`, name: "echo" },
          { type: "tool_call", callId: `c${callIndex}`, name: "echo", args: { text: `${callIndex}` } },
          { type: "usage", usage: { input: 10, output: 5, total: 15 } },
        ];
      }
      return [
        { type: "text", delta: "Done." },
        { type: "usage", usage: { input: 20, output: 10, total: 30 } },
      ];
    });

    const loop = nessi({
      provider,
      systemPrompt: "test",
      store: memoryStore(),
      tools: [echoTool],
      input: "Go",
      maxTurns: 2,
    });

    const events: OutboundEvent[] = [];
    for await (const event of loop) {
      events.push(event);
      // After first tool_end, steer to reset turns
      if (event.type === "tool_execution_end" && (event as any).callId === "c0") {
        loop.steer("Keep going");
      }
    }

    // Without steer, maxTurns=2 would stop after 2 provider calls.
    // With steer resetting turn counter, we get more calls.
    expect(callCount).toBeGreaterThan(2);
  });

  it("steer() ignores empty strings", async () => {
    const loop = nessi({
      provider: mockProvider([
        { type: "text", delta: "Hello" },
        { type: "usage", usage: { input: 10, output: 5, total: 15 } },
      ]),
      systemPrompt: "test",
      store: memoryStore(),
      input: "Hi",
    });

    loop.steer("");
    loop.steer("   ");

    const events: OutboundEvent[] = [];
    for await (const event of loop) {
      events.push(event);
    }

    const steerEvents = events.filter((e) => e.type === "steer_applied");
    expect(steerEvents).toHaveLength(0);
  });

  it("pre-emptive compaction when fillRatio >= 0.85", async () => {
    let compactCtx: { force: boolean; fillRatio?: number } | null = null;
    // contextWindow=50 with ~200 char input → fillRatio ≈ 1.3 (well above 0.85)
    const provider = mockProvider(
      [
        { type: "text", delta: "OK" },
        { type: "usage", usage: { input: 10, output: 5, total: 15 } },
      ],
      { contextWindow: 50 },
    );

    const store = memoryStore();

    const events = await collectEvents(
      nessi({
        provider,
        systemPrompt: "test",
        store,
        compact(ctx) {
          compactCtx = { force: ctx.force, fillRatio: ctx.fillRatio };
          return null;
        },
        input: "A ".repeat(100),
      }),
    );

    // Should have attempted compaction with force=true due to high fillRatio
    expect(compactCtx).not.toBeNull();
    expect(compactCtx!.force).toBe(true);
    expect(compactCtx!.fillRatio).toBeDefined();
    expect(compactCtx!.fillRatio!).toBeGreaterThanOrEqual(0.85);

    // Should still complete the turn
    const done = events.find((e) => e.type === "loop_end") as any;
    expect(done.reason).toBe("stop");
  });

  it("compact called with force=false and low fillRatio when context is small", async () => {
    let compactCtx: { force: boolean; fillRatio?: number } | null = null;
    const provider = mockProvider(
      [
        { type: "text", delta: "OK" },
        { type: "usage", usage: { input: 10, output: 5, total: 15 } },
      ],
      { contextWindow: 1_000_000 },
    );

    const events = await collectEvents(
      nessi({
        provider,
        systemPrompt: "test",
        store: memoryStore(),
        compact(ctx) {
          compactCtx = { force: ctx.force, fillRatio: ctx.fillRatio };
          return null; // no compaction needed
        },
        input: "Hi",
      }),
    );

    // compact IS called, but with force=false and very low fillRatio
    expect(compactCtx).not.toBeNull();
    expect(compactCtx!.force).toBe(false);
    expect(compactCtx!.fillRatio).toBeDefined();
    expect(compactCtx!.fillRatio!).toBeLessThan(0.01);
    // No compaction events should have been emitted (compact returned null)
    const types = events.map((e) => e.type);
    expect(types).not.toContain("compaction_start");
    const done = events.find((e) => e.type === "loop_end") as any;
    expect(done.reason).toBe("stop");
  });

  it("estimates compaction pressure after historical projection and tool-result truncation", async () => {
    const store = memoryStore();
    await store.append({
      role: "assistant",
      content: [{ type: "tool_call", id: "large-1", name: "large", args: {} }],
    });
    await store.append({
      role: "tool_result",
      callId: "large-1",
      name: "large",
      result: `FULL:${"f".repeat(10_000)}`,
      historicalResult: { originLoopId: "old-loop", value: `HISTORY:${"h".repeat(100)}` },
    });
    let compactCtx: { force: boolean; fillRatio?: number } | null = null;

    await collectEvents(nessi({
      loopId: "new-loop",
      provider: mockProvider(
        [
          { type: "text", delta: "OK" },
          { type: "usage", usage: { input: 100, output: 5, total: 105 } },
        ],
        { contextWindow: 1_000 },
      ),
      systemPrompt: "test",
      store,
      maxToolResultChars: 80,
      compact(ctx) {
        compactCtx = { force: ctx.force, fillRatio: ctx.fillRatio };
        return null;
      },
      input: "Continue",
    }));

    expect(compactCtx).not.toBeNull();
    expect(compactCtx!.force).toBe(false);
    expect(compactCtx!.fillRatio).toBeLessThan(0.5);
  });

  it("does not treat prior output tokens as input context pressure", async () => {
    const compactCalls: Array<{ force: boolean; fillRatio?: number }> = [];
    const tool = defineTool({
      name: "small_tool",
      description: "Return a small result",
      inputSchema: z.object({}),
      outputSchema: z.object({ ok: z.boolean() }),
    }).server(async () => ({ ok: true }));
    const provider = mockProviderMultiTurn(
      (_request, callIndex) => callIndex === 0
        ? [
            { type: "tool_start", callId: "small-1", name: "small_tool" },
            { type: "tool_call", callId: "small-1", name: "small_tool", args: {} },
            { type: "usage", usage: { input: 20, output: 900, total: 920 } },
          ]
        : [
            { type: "text", delta: "Done" },
            { type: "usage", usage: { input: 80, output: 5, total: 85 } },
          ],
      { contextWindow: 1_000 },
    );

    await collectEvents(nessi({
      provider,
      systemPrompt: "test",
      store: memoryStore(),
      tools: [tool],
      compact(ctx) {
        compactCalls.push({ force: ctx.force, fillRatio: ctx.fillRatio });
        return null;
      },
      input: "Run it",
    }));

    expect(compactCalls).toHaveLength(2);
    expect(compactCalls[1]?.force).toBe(false);
    expect(compactCalls[1]?.fillRatio).toBeLessThan(0.85);
  });

  it("context overflow error includes overflowRatio from provider", async () => {
    const events = await collectEvents(
      nessi({
        provider: mockProvider([
          { type: "error", error: "context too long", retryable: false, contextOverflow: true, overflowRatio: 1.5 },
        ]),
        systemPrompt: "test",
        store: memoryStore(),
        input: "Hi",
      }),
    );

    const error = events.find((e) => e.type === "issue") as any;
    expect(error).toBeDefined();
    expect(error.issue.contextOverflow).toBe(true);
    expect(error.issue.overflowRatio).toBeDefined();

    const done = events.find((e) => e.type === "loop_end") as any;
    expect(done.reason).toBe("context_overflow");
  });

  it("context overflow compaction retry passes fillRatio to compact", async () => {
    let callCount = 0;
    let retryFillRatio: number | undefined;
    const provider: import("../src/types.js").Provider = {
      name: "mock",
      family: "openai-compatible",
      model: "mock",
      capabilities: { streaming: true, tools: true, images: true, thinking: true, usage: true },
      contextWindow: 100_000,
      async *stream() {
        callCount++;
        if (callCount === 1) {
          yield {
            type: "issue" as const,
            issue: {
              kind: "provider_error" as const,
              message: "context too long",
              retryable: false,
              contextOverflow: true,
              overflowRatio: 1.3,
            },
          };
          return;
        }
        yield { type: "block_start" as const, blockId: "b0", index: 0, kind: "text" as const };
        yield { type: "block_delta" as const, blockId: "b0", delta: "After compaction" };
        yield { type: "block_end" as const, blockId: "b0", index: 0, block: { type: "text" as const, text: "After compaction" } };
        yield { type: "usage" as const, usage: { input: 10, output: 5, total: 15 } };
      },
      complete(request) {
        return completeFromStream(provider, request);
      },
    };

    const events = await collectEvents(
      nessi({
        provider,
        systemPrompt: "test",
        store: memoryStore(),
        compact(ctx) {
          if (!ctx.force) return null;
          retryFillRatio = ctx.fillRatio;
          return (async () => {
            await ctx.store.append(
              { role: "user", content: [{ type: "text", text: "Compacted" }] },
              { seq: 0, kind: "summary" },
            );
          })();
        },
        input: "Hi",
      }),
    );

    expect(callCount).toBe(2);
    expect(retryFillRatio).toBeDefined();
    const done = events.find((e) => e.type === "loop_end") as any;
    expect(done.reason).toBe("stop");
  });

  it("persists historical tool results once and selects them only for later loops", async () => {
    const store = memoryStore();
    let projectionCalls = 0;
    const fullResult = { stdout: "full output", exitCode: 0 };
    const historicalResult = { summary: "exit 0" };
    const shellTool = defineTool({
      name: "shell",
      description: "Run a command",
      inputSchema: z.object({ command: z.string() }),
      outputSchema: z.object({ stdout: z.string(), exitCode: z.number() }),
      async toHistoricalResult({ input, output, callId }) {
        projectionCalls++;
        expect(input).toEqual({ command: "pwd" });
        expect(output).toEqual(fullResult);
        expect(callId).toBe("shell-1");
        return historicalResult;
      },
    }).server(async () => fullResult);

    const originRequests: ProviderRequest[] = [];
    const originProvider = mockProviderMultiTurn((request, callIndex) => {
      originRequests.push(request);
      if (callIndex === 0) {
        return [
          { type: "tool_start", callId: "shell-1", name: "shell" },
          { type: "tool_call", callId: "shell-1", name: "shell", args: { command: "pwd" } },
        ];
      }
      return [{ type: "text", delta: "done" }];
    });

    const originEvents = await collectEvents(nessi({
      loopId: "origin-loop",
      provider: originProvider,
      systemPrompt: "test",
      store,
      tools: [shellTool],
      input: "run pwd",
    }));

    expect(projectionCalls).toBe(1);
    const storedToolResult = (await store.load()).find((entry) => entry.message.role === "tool_result")?.message;
    expect(storedToolResult).toEqual({
      role: "tool_result",
      callId: "shell-1",
      name: "shell",
      result: fullResult,
      historicalResult: { originLoopId: "origin-loop", value: historicalResult },
      isError: false,
    });
    const originToolResult = originRequests[1]?.messages.find((message) => message.role === "tool_result");
    expect(originToolResult?.role === "tool_result" ? originToolResult.result : undefined).toEqual(fullResult);
    const originMessages = originRequests[1]?.messages ?? [];
    const toolCallIndex = originMessages.findIndex((message) =>
      message.role === "assistant"
      && message.content.some((block) => block.type === "tool_call" && block.id === "shell-1")
    );
    const toolResultIndex = originMessages.findIndex((message) =>
      message.role === "tool_result" && message.callId === "shell-1"
    );
    expect(toolResultIndex).toBe(toolCallIndex + 1);
    const originDone = originEvents.find((event) => event.type === "loop_end");
    expect(originDone?.type === "loop_end" ? originDone.aggregate.turns[0]?.toolCalls[0]?.result : undefined).toEqual(fullResult);

    let resumedRequest: ProviderRequest | undefined;
    await collectEvents(nessi({
      loopId: "origin-loop",
      provider: mockProvider([{ type: "text", delta: "resumed" }], { onRequest: (request) => { resumedRequest = request; } }),
      systemPrompt: "test",
      store,
      input: "continue the same loop",
    }));
    const resumedToolResult = resumedRequest?.messages.find((message) => message.role === "tool_result");
    expect(resumedToolResult?.role === "tool_result" ? resumedToolResult.result : undefined).toEqual(fullResult);

    let laterRequest: ProviderRequest | undefined;
    await collectEvents(nessi({
      loopId: "later-loop",
      provider: mockProvider([{ type: "text", delta: "later" }], { onRequest: (request) => { laterRequest = request; } }),
      systemPrompt: "test",
      store,
      input: "new request",
    }));
    const laterToolResult = laterRequest?.messages.find((message) => message.role === "tool_result");
    expect(laterToolResult?.role === "tool_result" ? laterToolResult.result : undefined).toEqual(historicalResult);
    expect(projectionCalls).toBe(1);
  });

  it("keeps successful tool execution when historical result derivation fails", async () => {
    const store = memoryStore();
    const tool = defineTool({
      name: "fragile_projection",
      description: "Returns a full result",
      inputSchema: z.object({}),
      toHistoricalResult() {
        throw new Error("cannot summarize");
      },
    }).server(async () => ({ full: true }));
    const provider = mockProviderMultiTurn((_request, callIndex) => callIndex === 0
      ? [
          { type: "tool_start", callId: "fragile-1", name: "fragile_projection" },
          { type: "tool_call", callId: "fragile-1", name: "fragile_projection", args: {} },
        ]
      : [{ type: "text", delta: "continued" }]);

    const events = await collectEvents(nessi({
      loopId: "projection-error-loop",
      provider,
      systemPrompt: "test",
      store,
      tools: [tool],
      input: "run it",
    }));

    const issue = events.find((event) => event.type === "issue");
    expect(issue?.type === "issue" ? issue.issue : undefined).toMatchObject({
      kind: "tool_historical_result_error",
      callId: "fragile-1",
      name: "fragile_projection",
      retryable: false,
    });
    const toolEnd = events.find((event) => event.type === "tool_execution_end");
    expect(toolEnd?.type === "tool_execution_end" ? toolEnd.result : undefined).toEqual({ full: true });
    expect(toolEnd?.type === "tool_execution_end" ? toolEnd.isError : undefined).toBeUndefined();
    const done = events.find((event) => event.type === "loop_end");
    expect(done?.type === "loop_end" ? done.reason : undefined).toBe("stop");
    expect(done?.type === "loop_end" ? done.aggregate.issueCount : undefined).toBe(1);
    const stored = (await store.load()).find((entry) => entry.message.role === "tool_result")?.message;
    expect(stored).toEqual({
      role: "tool_result",
      callId: "fragile-1",
      name: "fragile_projection",
      result: { full: true },
      isError: false,
    });
  });

  it("derives historical results for top-level client tools", async () => {
    const store = memoryStore();
    const clientTool = defineTool({
      name: "client_report",
      description: "Collect a report in the client",
      inputSchema: z.object({ reportId: z.string() }),
      outputSchema: z.object({ rows: z.array(z.string()) }),
      toHistoricalResult: ({ input, output }) => ({ reportId: input.reportId, rowCount: output.rows.length }),
    }).client(async () => ({ rows: [] }));
    const provider = mockProviderMultiTurn((_request, callIndex) => callIndex === 0
      ? [
          { type: "tool_start", callId: "client-report-1", name: "client_report" },
          { type: "tool_call", callId: "client-report-1", name: "client_report", args: { reportId: "r1" } },
        ]
      : [{ type: "text", delta: "done" }]);
    const loop = nessi({
      loopId: "client-origin",
      provider,
      systemPrompt: "test",
      store,
      tools: [clientTool],
      input: "collect report",
    });

    for await (const event of loop) {
      if (event.type === "tool_action_request" && event.kind === "client_tool") {
        loop.push({ type: "tool_result", callId: event.callId, result: { rows: ["a", "b"] } });
      }
    }

    const stored = (await store.load()).find((entry) => entry.message.role === "tool_result")?.message;
    expect(stored?.role === "tool_result" ? stored.historicalResult : undefined).toEqual({
      originLoopId: "client-origin",
      value: { reportId: "r1", rowCount: 2 },
    });
  });

  it("applies maxToolResultChars after selecting a historical result", async () => {
    const store = memoryStore();
    await store.append({
      role: "assistant",
      content: [{ type: "tool_call", id: "large-1", name: "large", args: {} }],
    });
    await store.append({
      role: "tool_result",
      callId: "large-1",
      name: "large",
      result: `FULL:${"f".repeat(500)}`,
      historicalResult: { originLoopId: "old-loop", value: `HISTORY:${"h".repeat(100)}` },
    });
    let request: ProviderRequest | undefined;

    await collectEvents(nessi({
      loopId: "new-loop",
      provider: mockProvider([{ type: "text", delta: "ok" }], { onRequest: (value) => { request = value; } }),
      systemPrompt: "test",
      store,
      maxToolResultChars: 40,
      input: "continue",
    }));

    const result = request?.messages.find((message) => message.role === "tool_result");
    const projected = result?.role === "tool_result" ? result.result : undefined;
    expect(projected).toContain("HISTORY:");
    expect(projected).toContain("characters omitted");
    expect(projected).not.toContain("FULL:");
  });
});
