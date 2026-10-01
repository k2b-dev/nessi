import { describe, expect, it } from "bun:test";
import { z } from "zod";
import { defineTool, nessi, StructuredOutputError } from "../src/index.js";
import { mockProvider, mockProviderMultiTurn } from "./mock-provider.js";
import type { GenerateResult, Provider } from "../src/ai/index.js";
import type { ProviderRequest } from "../src/types.js";

const textResult = (text: string, index: number): GenerateResult => ({
  message: {
    role: "assistant",
    model: "mock",
    content: [{ type: "text", text }],
    usage: { input: 1, output: 1, total: 2 },
    stopReason: "stop",
  },
  usage: { input: 1, output: 1, total: 2 },
  finishReason: "stop",
  providerMeta: { model: "mock", requestId: `req-${index}` },
});

const completeProvider = (responses: string[], structuredOutput: boolean) => {
  const requests: ProviderRequest[] = [];
  let callIndex = 0;
  const provider: Provider = {
    name: "complete-mock",
    family: "openai-compatible",
    model: "mock",
    capabilities: {
      streaming: false,
      tools: false,
      images: true,
      thinking: false,
      usage: true,
      structuredOutput,
    },
    async *stream() {},
    async complete(request) {
      requests.push(request);
      const index = callIndex++;
      return textResult(responses[Math.min(index, responses.length - 1)] ?? "{}", index);
    },
  };
  return { provider, requests };
};

describe("nessi.structured", () => {
  it("uses native responseFormat and returns parsed output", async () => {
    let capturedRequest: ProviderRequest | undefined;
    const provider = mockProvider(
      [
        { type: "text", delta: "{\"title\":\"Launch\",\"count\":2}" },
        { type: "usage", usage: { input: 3, output: 4, total: 7 } },
      ],
      { onRequest: (request) => { capturedRequest = request; } },
    );

    const result = await nessi.structured({
      provider,
      input: [
        { type: "text", text: "Extract the card." },
        { type: "file", mediaType: "image/png", data: "abc123" },
      ],
      outputName: "card",
      output: z.object({
        title: z.string(),
        count: z.number(),
      }),
      tools: [],
      maxOutputTokens: 64,
      temperature: 0,
    });

    expect(result.output).toEqual({ title: "Launch", count: 2 });
    expect(result.reason).toBe("stop");
    expect(result.aggregate.assistantMessageCount).toBe(1);
    expect(result.usage?.total).toBe(7);
    expect(result.structuredMeta).toEqual({
      mode: "native",
      repaired: false,
      attempts: 1,
      usedResponseFormat: true,
    });
    expect(capturedRequest?.responseFormat?.type).toBe("json_schema");
    expect(capturedRequest?.responseFormat?.name).toBe("card");
    expect((capturedRequest?.responseFormat?.schema as Record<string, unknown> | undefined)?.$schema).toBeUndefined();
    expect(capturedRequest?.maxOutputTokens).toBe(64);
    expect(capturedRequest?.temperature).toBe(0);
    expect(capturedRequest?.messages[0]?.role).toBe("user");
    expect(capturedRequest?.messages[0]?.role === "user" ? capturedRequest.messages[0].content[1] : undefined)
      .toEqual({ type: "file", mediaType: "image/png", data: "abc123" });
  });

  it("falls back to schema instructions and repairs invalid output once", async () => {
    const { provider, requests } = completeProvider([
      "Sure, title is Launch",
      "```json\n{\"title\":\"Launch\",\"count\":2}\n```",
    ], false);

    const result = await nessi.structured({
      provider,
      input: "Extract the card.",
      output: z.object({
        title: z.string(),
        count: z.number(),
      }),
    });

    expect(result.output).toEqual({ title: "Launch", count: 2 });
    expect(result.structuredMeta).toEqual({
      mode: "repair",
      repaired: true,
      attempts: 2,
      usedResponseFormat: false,
    });
    expect(requests).toHaveLength(2);
    expect(requests[0]?.responseFormat).toBeUndefined();
    expect(requests[0]?.systemPrompt).toContain("JSON Schema");
    expect(requests[1]?.messages).toHaveLength(3);
    expect(result.aggregate.assistantMessageCount).toBe(2);
  });

  it("adds timing metadata to direct structured aggregates", async () => {
    const originalNow = Date.now;
    let now = 100;
    Date.now = () => now;

    try {
      const { provider } = completeProvider(["{\"title\":\"Launch\",\"count\":2}"], true);
      const baseComplete = provider.complete;
      provider.complete = async (request) => {
        now += 250;
        return baseComplete(request);
      };

      const result = await nessi.structured({
        provider,
        input: "Extract the card.",
        output: z.object({
          title: z.string(),
          count: z.number(),
        }),
      });

      expect(result.aggregate.timing).toEqual({
        wallMs: 250,
        totalElapsedMs: 250,
        generationMs: 250,
        toolExecutionMs: 0,
        actionWaitMs: 0,
        outputTokensPerSecond: 4,
      });
    } finally {
      Date.now = originalNow;
    }
  });

  it("wraps direct-mode provider failures in StructuredOutputError", async () => {
    const provider: Provider = {
      ...mockProvider([]),
      complete: async () => {
        throw new Error("upstream unavailable");
      },
    };

    const caught = await nessi.structured({
      provider,
      input: "Return ok.",
      output: z.object({ ok: z.boolean() }),
    }).catch((error: unknown) => error);

    expect(caught).toBeInstanceOf(StructuredOutputError);
    expect(caught).toMatchObject({ code: "loop_failed", message: "upstream unavailable" });
  });

  it("falls back when a native provider receives a schema outside strict native constraints", async () => {
    const { provider, requests } = completeProvider([
      "{\"title\":\"Launch\"}",
    ], true);

    const result = await nessi.structured({
      provider,
      input: "Extract the card.",
      output: z.object({
        title: z.string(),
        optionalNote: z.string().optional(),
      }),
    });

    expect(result.output).toEqual({ title: "Launch" });
    expect(result.structuredMeta).toEqual({
      mode: "fallback",
      repaired: false,
      attempts: 1,
      usedResponseFormat: false,
    });
    expect(requests[0]?.responseFormat).toBeUndefined();
  });

  it("runs a bounded server-tool loop and stops after submit_result", async () => {
    const lookup = defineTool({
      name: "lookup",
      description: "Look up a count",
      inputSchema: z.object({ id: z.string() }),
      outputSchema: z.object({ count: z.number() }),
    }).server(async () => ({ count: 2 }));

    const requests: ProviderRequest[] = [];
    const provider = mockProviderMultiTurn((request, callIndex) => {
      requests.push(request);
      if (callIndex === 0) return [
        { type: "tool_call", callId: "lookup-1", name: "lookup", args: { id: "launch" } },
        { type: "usage", usage: { input: 2, output: 1, total: 3 }, finishReason: "tool_use" },
      ];
      return [
        {
          type: "tool_call",
          callId: "submit-1",
          name: "submit_result",
          args: { title: "Launch", count: 2 },
        },
        { type: "usage", usage: { input: 3, output: 2, total: 5 }, finishReason: "tool_use" },
      ];
    });

    const result = await nessi.structured({
      provider,
      input: "Use tools and return the final card.",
      output: z.object({
        title: z.string(),
        count: z.number(),
      }),
      tools: [lookup],
      maxTurns: 4,
    });

    expect(result.output).toEqual({ title: "Launch", count: 2 });
    expect(result.structuredMeta).toEqual({
      mode: "tool_loop",
      repaired: false,
      attempts: 2,
      usedResponseFormat: false,
    });
    expect(result.aggregate.toolCallCount).toBe(2);
    expect(result.aggregate.toolErrorCount).toBe(0);
    expect(requests).toHaveLength(2);
    expect(requests[0]?.responseFormat).toBeUndefined();
    expect(requests[0]?.tools?.map((tool) => tool.name)).toEqual(["lookup", "submit_result"]);
  });

  it("uses dynamic structured tool snapshots across provider turns", async () => {
    const requestedToolNames: string[][] = [];
    let resolverCalls = 0;
    let lookupCallId: string | undefined;

    const unlock = defineTool({
      name: "unlock",
      description: "Unlock lookup",
      inputSchema: z.object({}),
    }).server(async () => {
      activeTools.splice(0, activeTools.length, unlock, lookup);
      return { unlocked: true };
    });
    const lookup = defineTool({
      name: "lookup",
      description: "Look up the final value",
      inputSchema: z.object({}),
    }).server(async (_input, ctx) => {
      lookupCallId = ctx.callId;
      activeTools.splice(0, activeTools.length, unlock);
      return { value: "resolved" };
    });
    const activeTools = [unlock];

    const provider = mockProviderMultiTurn((request, callIndex) => {
      requestedToolNames.push(request.tools?.map((tool) => tool.name) ?? []);
      if (callIndex === 0) return [
        { type: "tool_call", callId: "unlock-1", name: "unlock", args: {} },
        { type: "usage", usage: { input: 1, output: 1, total: 2 }, finishReason: "tool_use" },
      ];
      if (callIndex === 1) return [
        { type: "tool_call", callId: "lookup-1", name: "lookup", args: {} },
        { type: "usage", usage: { input: 1, output: 1, total: 2 }, finishReason: "tool_use" },
      ];
      return [
        { type: "tool_call", callId: "submit-1", name: "submit_result", args: { value: "resolved" } },
        { type: "usage", usage: { input: 1, output: 1, total: 2 }, finishReason: "tool_use" },
      ];
    });

    const result = await nessi.structured({
      provider,
      input: "Resolve the value.",
      output: z.object({ value: z.string() }),
      tools: async () => {
        resolverCalls++;
        return activeTools;
      },
    });

    expect(result.output).toEqual({ value: "resolved" });
    expect(result.structuredMeta.mode).toBe("tool_loop");
    expect(requestedToolNames).toEqual([
      ["unlock", "submit_result"],
      ["unlock", "lookup", "submit_result"],
      ["unlock", "submit_result"],
    ]);
    expect(resolverCalls).toBe(3);
    expect(lookupCallId).toBe("lookup-1");
  });

  it("uses tool_loop mode when a structured tool resolver returns an empty snapshot", async () => {
    const provider = mockProvider([
      { type: "tool_call", callId: "submit-1", name: "submit_result", args: { ok: true } },
      { type: "usage", usage: { input: 1, output: 1, total: 2 }, finishReason: "tool_use" },
    ], {
      onRequest: (request) => {
        expect(request.tools?.map((tool) => tool.name)).toEqual(["submit_result"]);
      },
    });

    const result = await nessi.structured({
      provider,
      input: "Return ok.",
      output: z.object({ ok: z.boolean() }),
      tools: () => [],
    });

    expect(result.output).toEqual({ ok: true });
    expect(result.structuredMeta.mode).toBe("tool_loop");
  });

  it("reports structured tool resolver failures without calling the provider", async () => {
    let providerCalled = false;
    let caught: unknown;
    try {
      await nessi.structured({
        provider: mockProvider([], { onRequest: () => { providerCalled = true; } }),
        input: "Return ok.",
        output: z.object({ ok: z.boolean() }),
        tools: async () => {
          throw new Error("tool registry unavailable");
        },
      });
    } catch (error) {
      caught = error;
    }

    expect(providerCalled).toBe(false);
    expect(caught).toBeInstanceOf(StructuredOutputError);
    expect((caught as StructuredOutputError).code).toBe("loop_failed");
    expect((caught as StructuredOutputError).details).toMatchObject({
      aggregate: {
        issues: [{ kind: "runtime_error", message: "tool registry unavailable" }],
      },
    });
  });

  it("validates reserved names in every dynamic structured tool snapshot", async () => {
    const reserved = defineTool({
      name: "submit_result",
      description: "Conflicts with nessi.structured()",
      inputSchema: z.object({}),
    }).server(async () => ({ ok: true }));

    await expect(nessi.structured({
      provider: mockProvider([]),
      input: "Return ok.",
      output: z.object({ ok: z.boolean() }),
      tools: () => [reserved],
    })).rejects.toThrow('Tool name "submit_result" is reserved by nessi.structured().');
  });

  it("keeps the tool loop open after invalid submit_result args and aggregates the validation error", async () => {
    const noop = defineTool({
      name: "noop",
      description: "No-op server tool",
      inputSchema: z.object({}),
    }).server(async () => ({ ok: true }));

    const provider = mockProviderMultiTurn((_request, callIndex) => {
      if (callIndex === 0) return [
        {
          type: "tool_call",
          callId: "submit-1",
          name: "submit_result",
          args: { title: "Launch", count: "wrong" },
        },
        { type: "usage", usage: { input: 2, output: 1, total: 3 }, finishReason: "tool_use" },
      ];
      return [
        {
          type: "tool_call",
          callId: "submit-2",
          name: "submit_result",
          args: { title: "Launch", count: 2 },
        },
        { type: "usage", usage: { input: 3, output: 2, total: 5 }, finishReason: "tool_use" },
      ];
    });

    const result = await nessi.structured({
      provider,
      input: "Return the final card.",
      output: z.object({
        title: z.string(),
        count: z.number(),
      }),
      tools: [noop],
      maxTurns: 4,
    });

    expect(result.output).toEqual({ title: "Launch", count: 2 });
    expect(result.structuredMeta).toEqual({
      mode: "tool_loop",
      repaired: true,
      attempts: 2,
      usedResponseFormat: false,
    });
    expect(result.aggregate.toolCallCount).toBe(2);
    expect(result.aggregate.toolErrorCount).toBe(1);
    expect(result.aggregate.issues[0]).toMatchObject({
      kind: "tool_execution_error",
      reason: "input_validation_failed",
      name: "submit_result",
    });
  });

  it("throws max_turns when a tool loop never calls submit_result", async () => {
    const ping = defineTool({
      name: "ping",
      description: "Ping",
      inputSchema: z.object({}),
    }).server(async () => ({ ok: true }));

    const provider = mockProviderMultiTurn(() => [
      { type: "tool_call", callId: crypto.randomUUID(), name: "ping", args: {} },
      { type: "usage", usage: { input: 1, output: 1, total: 2 }, finishReason: "tool_use" },
    ]);

    let caught: unknown;
    try {
      await nessi.structured({
        provider,
        input: "Keep trying.",
        output: z.object({ ok: z.boolean() }),
        tools: [ping],
        maxTurns: 2,
      });
    } catch (error) {
      caught = error;
    }

    expect(caught).toBeInstanceOf(StructuredOutputError);
    expect((caught as StructuredOutputError).code).toBe("max_turns");
    expect((caught as StructuredOutputError).details).toMatchObject({
      aggregate: {
        toolCallCount: 2,
        assistantMessageCount: 2,
      },
    });
  });

  it("rejects a user tool named submit_result", async () => {
    const reserved = defineTool({
      name: "submit_result",
      description: "Conflicts with nessi.structured()",
      inputSchema: z.object({}),
    }).server(async () => ({ ok: true }));

    let caught: unknown;
    try {
      await nessi.structured({
        provider: mockProvider([]),
        input: "Return ok.",
        output: z.object({ ok: z.boolean() }),
        tools: [reserved],
      });
    } catch (error) {
      caught = error;
    }

    expect(caught).toBeInstanceOf(StructuredOutputError);
    expect((caught as StructuredOutputError).code).toBe("unsupported_tool");
  });

  it("rejects client tools", async () => {
    const clientTool = defineTool({
      name: "client_pick",
      description: "Pick on the client",
      inputSchema: z.object({}),
    }).client(() => ({ ok: true }));

    await expect(nessi.structured({
      provider: mockProvider([]),
      input: "Pick",
      output: z.object({ ok: z.boolean() }),
      tools: [clientTool as any],
    })).rejects.toThrow(StructuredOutputError);
  });
});
