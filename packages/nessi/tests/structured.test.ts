import { describe, expect, it } from "bun:test";
import { z } from "zod";
import { defineTool, nessi, StructuredOutputError } from "../src/index.js";
import { mockProvider, mockProviderMultiTurn } from "./mock-provider.js";
import { openai, vllm } from "../src/ai/index.js";
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
  describe("native schema support", () => {
    const usesNative = async (
      output: z.ZodType,
      capabilities: Partial<Provider["capabilities"]> = {},
      answer = "{}",
    ) => {
      const { provider, requests } = completeProvider([answer], true);
      provider.capabilities = { ...provider.capabilities, ...capabilities };
      const result = await nessi.structured({ provider, input: "Extract.", output }).catch(() => undefined);
      const native = requests[0]?.responseFormat !== undefined;
      if (native && answer !== "{}") expect(result?.structuredMeta.mode).toBe("native");
      return native;
    };

    it("sends nullable fields and unions natively", async () => {
      expect(await usesNative(z.object({
        title: z.string().nullable(),
        status: z.enum(["open", "closed"]).nullable(),
        amount: z.union([z.string(), z.number()]),
        lines: z.array(z.object({ total: z.number().nullable() })),
      }), {}, '{"title":null,"status":"open","amount":3,"lines":[{"total":null}]}')).toBe(true);
    });

    it("falls back for a union at the root and for unsupported branches", async () => {
      expect(await usesNative(z.union([z.object({ a: z.string() }), z.object({ b: z.string() })]))).toBe(false);
      expect(await usesNative(z.object({ value: z.union([z.object({ a: z.string().optional() }), z.null()]) }))).toBe(false);
      expect(await usesNative(z.object({ value: z.looseObject({ name: z.string() }).nullable() }))).toBe(false);
      expect(await usesNative(z.object({ tags: z.record(z.string(), z.string()).nullable() }))).toBe(false);
      expect(await usesNative(z.object({ tags: z.record(z.string(), z.string()) }))).toBe(false);
    });

    it("needs every property required only for strict providers", async () => {
      const optional = z.object({ title: z.string(), note: z.string().optional() });
      expect(await usesNative(optional)).toBe(false);
      expect(await usesNative(optional, { strictStructuredOutput: false }, '{"title":"x"}')).toBe(true);
    });

    it("marks vLLM guided decoding as not strict", () => {
      expect(vllm("test").capabilities).toMatchObject({ structuredOutput: true, strictStructuredOutput: false });
      expect(openai("test", { apiKey: "test" }).capabilities.strictStructuredOutput).toBeUndefined();
    });
  });

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
      reasoningEffort: "low",
      extraBody: { top_k: 1 },
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
    expect(capturedRequest?.reasoningEffort).toBe("low");
    expect(capturedRequest?.extraBody).toEqual({ top_k: 1 });
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

  for (const [stopReason, code] of [
    ["max_tokens", "truncated"],
    ["error", "provider_stopped"],
    ["interrupted", "provider_stopped"],
    ["aborted", "aborted"],
  ] as const) {
    it(`never accepts or repairs a ${stopReason} response`, async () => {
      // Each field alone must stop it, with valid output (no accept) and invalid output (no repair).
      for (const [structuredOutput, field, text] of [
        [true, "finishReason", '{"ok":true}'],
        [false, "stopReason", '{"ok":true}'],
        [false, "finishReason", "not json"],
        [true, "stopReason", "not json"],
      ] as const) {
        let calls = 0;
        const provider: Provider = {
          ...completeProvider([], structuredOutput).provider,
          complete: async () => {
            const result = textResult(text, calls++);
            return field === "finishReason"
              ? { ...result, finishReason: stopReason }
              : { ...result, message: { ...result.message, stopReason } };
          },
        };
        const caught = await nessi.structured({ provider, input: "Return ok.", output: z.object({ ok: z.boolean() }) })
          .catch((error: unknown) => error);
        expect(caught).toBeInstanceOf(StructuredOutputError);
        expect(caught).toMatchObject({ code, details: { attempts: 1, aggregate: { usage: { input: 1, output: 1, total: 2 } } } });
        expect(calls).toBe(1);
      }
    });
  }

  it("checks the stop reason of the repair attempt too", async () => {
    let calls = 0;
    const provider: Provider = {
      ...completeProvider([], false).provider,
      complete: async () => {
        const index = calls++;
        const result = textResult(index === 0 ? "not json" : '{"ok":true}', index);
        return index === 0 ? result : { ...result, finishReason: "max_tokens", message: { ...result.message, stopReason: "max_tokens" } };
      },
    };
    const caught = await nessi.structured({ provider, input: "Return ok.", output: z.object({ ok: z.boolean() }) })
      .catch((error: unknown) => error);
    expect(caught).toMatchObject({ code: "truncated", details: { attempts: 2, aggregate: { usage: { total: 4 } } } });
  });

  for (const [stopReason, code] of [["max_tokens", "truncated"], ["interrupted", "provider_stopped"], ["aborted", "aborted"]] as const) {
    it(`tool-loop mode rejects a result submitted in a ${stopReason} response`, async () => {
      const lookup = defineTool({ name: "lookup", description: "Look up", inputSchema: z.object({}) }).server(async () => "x");
      const provider = mockProviderMultiTurn(() => [
        { type: "tool_call", callId: "submit-1", name: "submit_result", args: { ok: true } },
        { type: "usage", usage: { input: 3, output: 2, total: 5 }, finishReason: stopReason },
      ]);
      const caught = await nessi.structured({ provider, input: "Return ok.", output: z.object({ ok: z.boolean() }), tools: [lookup] })
        .catch((error: unknown) => error);
      expect(caught).toBeInstanceOf(StructuredOutputError);
      expect(caught).toMatchObject({ code, details: { aggregate: { usage: { total: 5 } } } });
    });
  }

  it("reports the generation time of a failed request", async () => {
    const originalNow = Date.now;
    let now = 1_000;
    Date.now = () => now;
    try {
      const provider: Provider = {
        ...completeProvider([], false).provider,
        complete: async () => {
          now += 500;
          throw new Error("upstream unavailable");
        },
      };
      const caught = await nessi.structured({ provider, input: "Return ok.", output: z.object({ ok: z.boolean() }) })
        .catch((error: unknown) => error);
      expect(caught).toMatchObject({ code: "loop_failed", details: { aggregate: { timing: { wallMs: 500, generationMs: 500 } } } });
    } finally {
      Date.now = originalNow;
    }
  });

  it("keeps the usage of the first attempt when the repair request fails", async () => {
    let calls = 0;
    const provider: Provider = {
      ...completeProvider([], false).provider,
      complete: async () => {
        if (calls++ === 0) return textResult("not json", 0);
        throw new Error("upstream unavailable");
      },
    };
    const caught = await nessi.structured({ provider, input: "Return ok.", output: z.object({ ok: z.boolean() }) })
      .catch((error: unknown) => error);
    expect(caught).toMatchObject({ code: "loop_failed", details: { attempts: 1, aggregate: { usage: { total: 2 } } } });
  });

  it("reports a caller abort that lands as the response returns", async () => {
    const controller = new AbortController();
    const provider: Provider = {
      ...completeProvider([], true).provider,
      complete: async () => {
        controller.abort();
        return textResult('{"ok":true}', 0);
      },
    };
    const caught = await nessi.structured({ provider, input: "Return ok.", output: z.object({ ok: z.boolean() }), signal: controller.signal })
      .catch((error: unknown) => error);
    expect(caught).toMatchObject({ code: "aborted" });
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
