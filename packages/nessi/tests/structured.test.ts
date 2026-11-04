import { describe, expect, it } from "bun:test";
import { z } from "zod";
import { defineTool, nessi, StructuredOutputError } from "../src/index.js";
import { mockProvider, mockProviderMultiTurn, resultStream, streamFromResult } from "./mock-provider.js";
import { endpoint } from "./ai/helpers/endpoints.js";
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
    stream(request) {
      requests.push(request);
      const index = callIndex++;
      return streamFromResult(textResult(responses[Math.min(index, responses.length - 1)] ?? "{}", index));
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
      expect(endpoint("vllm", { model: "test" }).capabilities).toMatchObject({ structuredOutput: true, strictStructuredOutput: false });
      expect(endpoint("openai", { model: "test" }).capabilities.strictStructuredOutput).toBeUndefined();
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
      const baseStream = provider.stream;
      provider.stream = (request) => {
        now += 250;
        return baseStream(request);
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
      stream: () => resultStream(async () => {
        throw new Error("upstream unavailable");
      }),
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
      // Valid output must not be accepted and invalid output must not be repaired, in both modes.
      for (const [structuredOutput, text] of [
        [true, '{"ok":true}'],
        [false, '{"ok":true}'],
        [false, "not json"],
        [true, "not json"],
      ] as const) {
        let calls = 0;
        const provider: Provider = {
          ...completeProvider([], structuredOutput).provider,
          stream: () => resultStream(async () => ({ ...textResult(text, calls++), finishReason: stopReason })),
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
      stream: () => resultStream(async () => {
        const index = calls++;
        const result = textResult(index === 0 ? "not json" : '{"ok":true}', index);
        return index === 0 ? result : { ...result, finishReason: "max_tokens", message: { ...result.message, stopReason: "max_tokens" } };
      }),
    };
    const caught = await nessi.structured({ provider, input: "Return ok.", output: z.object({ ok: z.boolean() }) })
      .catch((error: unknown) => error);
    expect(caught).toMatchObject({ code: "truncated", details: { attempts: 2, aggregate: { usage: { total: 4 } } } });
  });

  for (const [stopReason, code] of [["max_tokens", "truncated"], ["interrupted", "provider_stopped"], ["aborted", "aborted"]] as const) {
  }

  it("reports the generation time of a failed request", async () => {
    const originalNow = Date.now;
    let now = 1_000;
    Date.now = () => now;
    try {
      const provider: Provider = {
        ...completeProvider([], false).provider,
        stream: () => resultStream(async () => {
          now += 500;
          throw new Error("upstream unavailable");
        }),
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
      stream: () => resultStream(async () => {
        if (calls++ === 0) return textResult("not json", 0);
        throw new Error("upstream unavailable");
      }),
    };
    const caught = await nessi.structured({ provider, input: "Return ok.", output: z.object({ ok: z.boolean() }) })
      .catch((error: unknown) => error);
    expect(caught).toMatchObject({ code: "loop_failed", details: { attempts: 1, aggregate: { usage: { total: 2 } } } });
  });

  it("reports a caller abort that lands as the response returns", async () => {
    const controller = new AbortController();
    const provider: Provider = {
      ...completeProvider([], true).provider,
      stream: () => resultStream(async () => {
        controller.abort();
        return textResult('{"ok":true}', 0);
      }),
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

});
