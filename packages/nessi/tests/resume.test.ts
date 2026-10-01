import { describe, it, expect } from "bun:test";
import { z } from "zod";
import { nessi } from "../src/nessi.js";
import { defineTool } from "../src/tools.js";
import { memoryStore } from "../src/stores.js";
import { mockProvider, mockProviderMultiTurn } from "./mock-provider.js";
import type { OutboundEvent, SessionStore } from "../src/types.js";

async function collectEvents(loop: ReturnType<typeof nessi>): Promise<OutboundEvent[]> {
  const events: OutboundEvent[] = [];
  for await (const event of loop) {
    events.push(event);
  }
  return events;
}

const echoTool = defineTool({
  name: "echo",
  description: "Echoes input",
  inputSchema: z.object({ text: z.string() }),
  outputSchema: z.object({ echoed: z.string() }),
}).server(async (input) => ({ echoed: input.text }));

const dangerTool = defineTool({
  name: "danger",
  description: "Needs approval",
  inputSchema: z.object({ action: z.string() }),
  needsApproval: true,
}).server(async (input) => ({ done: input.action }));

const toastTool = defineTool({
  name: "toast",
  description: "Show toast",
  inputSchema: z.object({ message: z.string() }),
  outputSchema: z.object({ shown: z.boolean() }),
}).client(() => ({ shown: true }));

/** Store pre-seeded with a user question so run-from-history has a prompt. */
const storeWithHistory = async (messages: Parameters<SessionStore["append"]>[0][]): Promise<SessionStore> => {
  const store = memoryStore();
  for (const message of messages) {
    await store.append(message);
  }
  return store;
};

describe("nessi run-from-history (input omitted)", () => {
  it("does not append a user message and answers from history", async () => {
    const store = await storeWithHistory([{ role: "user", content: [{ type: "text", text: "Hi from history" }] }]);
    let seenMessages: unknown[] = [];
    const provider = mockProvider(
      [
        { type: "text", delta: "Hello back" },
        { type: "usage", usage: { input: 1, output: 1, total: 2 }, finishReason: "stop" },
      ],
      {
        onRequest: (request) => {
          seenMessages = request.messages;
        },
      },
    );

    const events = await collectEvents(nessi({ provider, store, systemPrompt: "sys" }));

    expect(events.some((event) => event.type === "block_end" && event.block.type === "text")).toBe(true);
    expect(events.at(-1)).toMatchObject({ type: "loop_end", reason: "stop" });
    // Provider saw exactly the historical user message — nothing was appended.
    expect(seenMessages).toHaveLength(1);
    const entries = await store.load();
    expect(entries.filter((entry) => entry.message.role === "user")).toHaveLength(1);
    expect(entries.at(-1)?.message.role).toBe("assistant");
  });

  it("throws no error for an empty store and finishes cleanly", async () => {
    const store = memoryStore();
    const provider = mockProvider([
      { type: "text", delta: "ok" },
      { type: "usage", usage: { input: 0, output: 1, total: 1 }, finishReason: "stop" },
    ]);
    const events = await collectEvents(nessi({ provider, store, systemPrompt: "sys" }));
    expect(events.at(-1)).toMatchObject({ type: "loop_end", reason: "stop" });
  });
});

describe("nessi resume of unresolved tool calls", () => {
  const historyWithPendingCall = () => [
    { role: "user" as const, content: [{ type: "text" as const, text: "Do something dangerous" }] },
    {
      role: "assistant" as const,
      content: [
        { type: "text" as const, text: "I will run the tool." },
        { type: "tool_call" as const, id: "call-1", name: "danger", args: { action: "wipe" } },
      ],
      stopReason: "tool_use" as const,
    },
  ];

  it("resolves dynamic tools for pending calls and again for the following provider turn", async () => {
    const store = await storeWithHistory([
      { role: "user", content: [{ type: "text", text: "Echo this" }] },
      {
        role: "assistant",
        content: [{ type: "tool_call", id: "resume-call-1", name: "dynamic_echo", args: { text: "hello" } }],
        stopReason: "tool_use",
      },
    ]);
    let resolverCalls = 0;
    let receivedCallId: string | undefined;
    const dynamicEcho = defineTool({
      name: "dynamic_echo",
      description: "Echoes input",
      inputSchema: z.object({ text: z.string() }),
    }).server(async (input, ctx) => {
      receivedCallId = ctx.callId;
      return { echoed: input.text };
    });
    const provider = mockProvider([{ type: "text", delta: "Done" }], {
      onRequest: (request) => {
        expect(request.tools?.map((tool) => tool.name)).toEqual(["dynamic_echo"]);
      },
    });

    const events = await collectEvents(nessi({
      provider,
      store,
      systemPrompt: "sys",
      tools: () => {
        resolverCalls++;
        return [dynamicEcho];
      },
    }));

    expect(resolverCalls).toBe(2);
    expect(receivedCallId).toBe("resume-call-1");
    expect(events.find((event) => event.type === "turn_start" && event.resumed)).toBeTruthy();
    expect(events.at(-1)).toMatchObject({ type: "loop_end", reason: "stop" });
  });

  it("executes a pending approval tool call with a seeded approval", async () => {
    const store = await storeWithHistory(historyWithPendingCall());
    const provider = mockProvider([
      { type: "text", delta: "Tool done, all good." },
      { type: "usage", usage: { input: 1, output: 1, total: 2 }, finishReason: "stop" },
    ]);

    const loop = nessi({ provider, store, systemPrompt: "sys", tools: [dangerTool] });
    // Seed the approval before iterating — the channel buffers it.
    loop.push({ type: "approval_response", callId: "call-1", approved: true });

    const events = await collectEvents(loop);
    const types = events.map((event) => event.type);

    expect(types).toContain("tool_execution_start");
    expect(types).toContain("tool_execution_end");
    // No tool_action_request should surface because the seeded approval resolved it.
    expect(types).not.toContain("tool_action_request");
    expect(events.at(-1)).toMatchObject({ type: "loop_end", reason: "stop" });

    const entries = await store.load();
    const toolResult = entries.find((entry) => entry.message.role === "tool_result");
    expect(toolResult?.message).toMatchObject({ callId: "call-1", isError: false });
    // Only the original user message — resume must not add one.
    expect(entries.filter((entry) => entry.message.role === "user")).toHaveLength(1);
  });

  it("records a rejection result for a seeded denial", async () => {
    const store = await storeWithHistory(historyWithPendingCall());
    const provider = mockProvider([
      { type: "text", delta: "Understood, not doing it." },
      { type: "usage", usage: { input: 1, output: 1, total: 2 }, finishReason: "stop" },
    ]);

    const loop = nessi({ provider, store, systemPrompt: "sys", tools: [dangerTool] });
    loop.push({ type: "approval_response", callId: "call-1", approved: false });

    const events = await collectEvents(loop);
    const toolEnd = events.find((event) => event.type === "tool_execution_end");
    expect(toolEnd).toMatchObject({ isError: true, result: "User denied this action" });
    expect(events.at(-1)).toMatchObject({ type: "loop_end", reason: "stop" });
  });

  it("resumes a pending client tool with a seeded result", async () => {
    const store = await storeWithHistory([
      { role: "user", content: [{ type: "text", text: "Show a toast" }] },
      {
        role: "assistant",
        content: [{ type: "tool_call", id: "call-9", name: "toast", args: { message: "hi" } }],
        stopReason: "tool_use",
      },
    ]);
    const provider = mockProvider([
      { type: "text", delta: "Toast shown." },
      { type: "usage", usage: { input: 1, output: 1, total: 2 }, finishReason: "stop" },
    ]);

    const loop = nessi({ provider, store, systemPrompt: "sys", tools: [toastTool] });
    loop.push({ type: "tool_result", callId: "call-9", result: { shown: true } });

    const events = await collectEvents(loop);
    const toolEnd = events.find((event) => event.type === "tool_execution_end");
    expect(toolEnd).toMatchObject({ callId: "call-9", result: { shown: true } });
    expect(events.at(-1)).toMatchObject({ type: "loop_end", reason: "stop" });
  });

  it("skips already-resolved calls and resumes only unresolved ones", async () => {
    const store = await storeWithHistory([
      { role: "user", content: [{ type: "text", text: "Run both" }] },
      {
        role: "assistant",
        content: [
          { type: "tool_call", id: "call-a", name: "echo", args: { text: "one" } },
          { type: "tool_call", id: "call-b", name: "echo", args: { text: "two" } },
        ],
        stopReason: "tool_use",
      },
      { role: "tool_result", callId: "call-a", name: "echo", result: { echoed: "one" }, isError: false },
    ]);
    const provider = mockProvider([
      { type: "text", delta: "Both done." },
      { type: "usage", usage: { input: 1, output: 1, total: 2 }, finishReason: "stop" },
    ]);

    const events = await collectEvents(nessi({ provider, store, systemPrompt: "sys", tools: [echoTool] }));
    const toolEnds = events.filter((event) => event.type === "tool_execution_end");
    expect(toolEnds).toHaveLength(1);
    expect(toolEnds[0]).toMatchObject({ callId: "call-b" });

    const entries = await store.load();
    const results = entries.filter((entry) => entry.message.role === "tool_result");
    expect(results).toHaveLength(2);
  });

  it("emits a fresh tool_action_request when the resume has no seeded response", async () => {
    const store = await storeWithHistory(historyWithPendingCall());
    const provider = mockProvider([
      { type: "text", delta: "done" },
      { type: "usage", usage: { input: 1, output: 1, total: 2 }, finishReason: "stop" },
    ]);

    const loop = nessi({ provider, store, systemPrompt: "sys", tools: [dangerTool] });
    const events: OutboundEvent[] = [];
    for await (const event of loop) {
      events.push(event);
      if (event.type === "tool_action_request") {
        loop.push({ type: "approval_response", callId: event.callId, approved: true });
      }
    }

    expect(events.some((event) => event.type === "tool_action_request")).toBe(true);
    expect(events.at(-1)).toMatchObject({ type: "loop_end", reason: "stop" });
  });

  it("continues into further tool turns after a resume", async () => {
    const store = await storeWithHistory(historyWithPendingCall());
    const provider = mockProviderMultiTurn((_request, callIndex) => {
      if (callIndex === 0) {
        return [
          { type: "tool_start", callId: "call-2", name: "echo" },
          { type: "tool_call", callId: "call-2", name: "echo", args: { text: "next" } },
          { type: "usage", usage: { input: 1, output: 1, total: 2 }, finishReason: "tool_use" },
        ];
      }
      return [
        { type: "text", delta: "All finished." },
        { type: "usage", usage: { input: 1, output: 1, total: 2 }, finishReason: "stop" },
      ];
    });

    const loop = nessi({ provider, store, systemPrompt: "sys", tools: [dangerTool, echoTool] });
    loop.push({ type: "approval_response", callId: "call-1", approved: true });

    const events = await collectEvents(loop);
    const toolEnds = events.filter((event) => event.type === "tool_execution_end");
    expect(toolEnds.map((event) => event.type === "tool_execution_end" && event.callId)).toEqual(["call-1", "call-2"]);
    expect(events.at(-1)).toMatchObject({ type: "loop_end", reason: "stop" });
  });
});

describe("nessi abort while waiting for inbound events", () => {
  it("ends with loop_end(aborted) when aborted while awaiting an approval", async () => {
    const store = memoryStore();
    const provider = mockProvider([
      { type: "tool_start", callId: "call-1", name: "danger" },
      { type: "tool_call", callId: "call-1", name: "danger", args: { action: "wipe" } },
      { type: "usage", usage: { input: 1, output: 1, total: 2 }, finishReason: "tool_use" },
    ]);

    const loop = nessi({ provider, store, systemPrompt: "sys", input: "Do it", tools: [dangerTool] });
    const events: OutboundEvent[] = [];
    for await (const event of loop) {
      events.push(event);
      if (event.type === "tool_action_request") {
        loop.abort();
      }
    }

    expect(events.at(-1)).toMatchObject({ type: "loop_end", reason: "aborted" });
    // The pending tool must NOT be recorded as failed — no tool_result may exist.
    const entries = await store.load();
    expect(entries.some((entry) => entry.message.role === "tool_result")).toBe(false);
  });

  it("ends with loop_end(aborted) when aborted while awaiting a client tool result", async () => {
    const store = memoryStore();
    const provider = mockProvider([
      { type: "tool_start", callId: "call-1", name: "toast" },
      { type: "tool_call", callId: "call-1", name: "toast", args: { message: "hi" } },
      { type: "usage", usage: { input: 1, output: 1, total: 2 }, finishReason: "tool_use" },
    ]);

    const loop = nessi({ provider, store, systemPrompt: "sys", input: "Toast me", tools: [toastTool] });
    const events: OutboundEvent[] = [];
    for await (const event of loop) {
      events.push(event);
      if (event.type === "tool_action_request") {
        loop.abort();
      }
    }

    expect(events.at(-1)).toMatchObject({ type: "loop_end", reason: "aborted" });
    const entries = await store.load();
    expect(entries.some((entry) => entry.message.role === "tool_result")).toBe(false);
  });
});

describe("nessi history after an aborted tool call", () => {
  it("answers the orphaned call before the next user message without changing the store", async () => {
    const store = memoryStore();
    const firstProvider = mockProvider([
      { type: "tool_start", callId: "call-1", name: "danger" },
      { type: "tool_call", callId: "call-1", name: "danger", args: { action: "wipe" } },
      { type: "usage", usage: { input: 1, output: 1, total: 2 }, finishReason: "tool_use" },
    ]);
    const first = nessi({ provider: firstProvider, store, systemPrompt: "sys", input: "Do it", tools: [dangerTool] });
    for await (const event of first) {
      if (event.type === "tool_action_request") first.abort();
    }

    let sentRoles: string[] = [];
    let sentResult: unknown;
    const secondProvider = mockProvider([
      { type: "text", delta: "Okay." },
      { type: "usage", usage: { input: 1, output: 1, total: 2 } },
    ], {
      onRequest: (request) => {
        sentRoles = request.messages.map((message) => message.role);
        sentResult = request.messages.find((message) => message.role === "tool_result");
      },
    });
    const events = await collectEvents(nessi({
      provider: secondProvider,
      store,
      systemPrompt: "sys",
      input: "Never mind",
      tools: [dangerTool],
    }));

    expect(events.at(-1)).toMatchObject({ type: "loop_end", reason: "stop" });
    expect(sentRoles).toEqual(["user", "assistant", "tool_result", "user"]);
    expect(sentResult).toMatchObject({ callId: "call-1", name: "danger", isError: true });
    const entries = await store.load();
    expect(entries.some((entry) => entry.message.role === "tool_result")).toBe(false);
  });
});
