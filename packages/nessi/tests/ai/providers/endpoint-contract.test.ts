import { afterEach, describe, expect, it } from "bun:test";
import { expectProviderContract } from "../contracts/provider-contract.js";
import { endpoint, ENDPOINT_NAMES } from "../helpers/endpoints.js";
import type { EndpointName } from "../helpers/endpoints.js";
import { jsonResponse, stubFetch, textResponse } from "../helpers/fixtures.js";

const originalFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = originalFetch; });

const sse = (...records: unknown[]) => records.map((record) => `data: ${typeof record === "string" ? record : JSON.stringify(record)}\n\n`).join("");
const usage = { prompt_tokens: 12, completion_tokens: 5, total_tokens: 17, prompt_tokens_details: { cached_tokens: 4 } };
const toolCall = { id: "call_00001", type: "function", function: { name: "lookup", arguments: '{"query":"cats"}' } };

/** One answer with reasoning, text, a tool call, and usage, in each endpoint's own reasoning format. */
const answers: Record<EndpointName, { native: unknown; stream: string }> = {
  openai: {
    native: { choices: [{ index: 0, message: { content: "Looking.", tool_calls: [toolCall] }, finish_reason: "tool_calls" }], usage },
    stream: sse(
      { choices: [{ index: 0, delta: { content: "Look" }, finish_reason: null }] },
      { choices: [{ index: 0, delta: { content: "ing." }, finish_reason: null }] },
      { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, ...toolCall, function: { name: "lookup", arguments: '{"query":' } }] }, finish_reason: null }] },
      { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: '"cats"}' } }] }, finish_reason: null }] },
      { choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }] },
      { choices: [], usage },
      "[DONE]",
    ),
  },
  openrouter: {
    native: {
      choices: [{
        index: 0,
        message: { content: "Looking.", reasoning_details: [{ type: "reasoning.text", text: "Need data.", index: 0 }], tool_calls: [toolCall] },
        finish_reason: "tool_calls",
      }],
      usage,
    },
    stream: sse(
      { choices: [{ index: 0, delta: { reasoning_details: [{ type: "reasoning.text", text: "Need ", index: 0 }] }, finish_reason: null }] },
      { choices: [{ index: 0, delta: { reasoning_details: [{ type: "reasoning.text", text: "data.", index: 0 }] }, finish_reason: null }] },
      { choices: [{ index: 0, delta: { content: "Looking." }, finish_reason: null }] },
      { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, ...toolCall }] }, finish_reason: null }] },
      { choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }], usage },
      "[DONE]",
    ),
  },
  mistral: {
    native: {
      choices: [{
        index: 0,
        message: {
          content: [{ type: "thinking", thinking: [{ type: "text", text: "Need data." }], signature: "m-sig" }, { type: "text", text: "Looking." }],
          tool_calls: [toolCall],
        },
        finish_reason: "tool_calls",
      }],
      usage,
    },
    stream: sse(
      { choices: [{ index: 0, delta: { content: [{ type: "thinking", thinking: [{ type: "text", text: "Need data." }] }] }, finish_reason: null }] },
      { choices: [{ index: 0, delta: { content: [{ type: "thinking", thinking: [], signature: "m-sig" }] }, finish_reason: null }] },
      { choices: [{ index: 0, delta: { content: [{ type: "text", text: "Looking." }] }, finish_reason: null }] },
      { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, ...toolCall }] }, finish_reason: null }] },
      { choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }], usage },
      "[DONE]",
    ),
  },
  vllm: {
    native: { choices: [{ index: 0, message: { reasoning_content: "Need data.", content: "Looking.", tool_calls: [toolCall] }, finish_reason: "tool_calls" }], usage },
    stream: sse(
      { choices: [{ index: 0, delta: { reasoning_content: "Need " }, finish_reason: null }] },
      { choices: [{ index: 0, delta: { reasoning_content: "data." }, finish_reason: null }] },
      { choices: [{ index: 0, delta: { content: "Looking." }, finish_reason: null }] },
      { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, ...toolCall }] }, finish_reason: null }] },
      { choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }] },
      { choices: [], usage },
      "[DONE]",
    ),
  },
};

/** What every variant must produce for that answer; Mistral's reasoning keeps its signature. */
const expectedBlocks = (name: EndpointName): unknown[] => [
  ...(name === "openai" ? [] : [{ type: "thinking", thinking: "Need data.", signature: name === "mistral" ? "m-sig" : undefined }]),
  { type: "text", text: "Looking." },
  { type: "tool_call", id: "call_00001", name: "lookup", args: { query: "cats" } },
];

/** Request fields each variant must send (or leave out) for the same call. */
const expectedRequest: Record<EndpointName, (native: Record<string, unknown>, streamed: Record<string, unknown>) => void> = {
  openai: (native, streamed) => {
    expect(native).toMatchObject({ max_completion_tokens: 64, response_format: { type: "json_schema" }, reasoning_effort: "low" });
    expect(streamed).toMatchObject({ stream: true, stream_options: { include_usage: true } });
  },
  openrouter: (native, streamed) => {
    expect(native).toMatchObject({ max_tokens: 64, response_format: { type: "json_schema" }, reasoning: { effort: "low" } });
    expect(native).not.toHaveProperty("reasoning_effort");
    expect(streamed).toMatchObject({ stream: true, stream_options: { include_usage: true } });
  },
  mistral: (native, streamed) => {
    expect(native).toMatchObject({ max_tokens: 64, response_format: { type: "json_schema" }, reasoning_effort: "low" });
    expect(streamed).toMatchObject({ stream: true });
    expect(streamed).not.toHaveProperty("stream_options");
  },
  vllm: (native, streamed) => {
    expect(native).toMatchObject({ max_tokens: 64, structured_outputs: { json: { type: "object" } }, reasoning_effort: "low" });
    expect(native).not.toHaveProperty("response_format");
    expect(streamed).toMatchObject({ stream: true, stream_options: { include_usage: true } });
  },
};

describe("endpoint contract", () => {
  for (const name of ENDPOINT_NAMES) {
    it(`${name}: complete() and the stream describe the expected answer and send the endpoint's fields`, async () => {
      const bodies: Array<Record<string, unknown>> = [];
      stubFetch((_input, init) => {
        bodies.push(JSON.parse(String(init?.body)));
        return bodies.length === 1 ? jsonResponse(answers[name].native) : textResponse(answers[name].stream, "text/event-stream");
      });
      const result = await expectProviderContract(endpoint(name, { baseURL: "https://example.test" }), {
        messages: [{ role: "user", content: [{ type: "text", text: "Find cats" }] }],
        maxOutputTokens: 64,
        reasoningEffort: "low",
        responseFormat: { type: "json_schema", name: "answer", schema: { type: "object", properties: {}, required: [] } },
      });

      // `details` are compared between both paths by the contract; here only the visible answer counts.
      const visible = result.message.content.map((block) => (block.type === "thinking" ? { ...block, details: undefined } : block));
      expect(JSON.parse(JSON.stringify(visible))).toEqual(JSON.parse(JSON.stringify(expectedBlocks(name))));
      expect(result.finishReason).toBe("tool_use");
      expect(result.usage).toMatchObject({ input: 12, output: 5, total: 17, cacheRead: 4 });
      expectedRequest[name](bodies[0]!, bodies[1]!);
    });
  }
});
