import { openAICompatible } from "../../../src/ai/index.js";
import type { OpenAICompat, OpenAICompatibleConfig, Provider } from "../../../src/ai/index.js";

/**
 * The endpoint settings Cloud uses, as plain `openAICompatible()` options. Tests run their
 * matrices over these variants so each endpoint quirk stays covered.
 */
export const ENDPOINTS = {
  openai: {
    baseURL: "https://api.openai.com/v1",
    compat: { supportsUsageInStreaming: true, thinkingFormat: "none", maxTokensField: "max_completion_tokens", structuredOutput: "response_format" },
  },
  openrouter: {
    baseURL: "https://openrouter.ai/api/v1",
    compat: { thinkingFormat: "reasoning_details", reasoningFormat: "openrouter", maxTokensField: "max_tokens", structuredOutput: "response_format" },
  },
  mistral: {
    baseURL: "https://api.mistral.ai/v1",
    compat: {
      toolCallIdPolicy: "strict9",
      requiresToolResultName: true,
      thinkingFormat: "content_chunks",
      supportsUsageInStreaming: false,
      maxTokensField: "max_tokens",
      structuredOutput: "response_format",
    },
  },
  vllm: {
    baseURL: "http://localhost:8000/v1",
    compat: { supportsUsageInStreaming: true, thinkingFormat: "text", maxTokensField: "max_tokens", structuredOutput: "vllm_structured_outputs" },
  },
} satisfies Record<string, { baseURL: string; compat: OpenAICompat }>;

export type EndpointName = keyof typeof ENDPOINTS;

export const ENDPOINT_NAMES = Object.keys(ENDPOINTS) as EndpointName[];

/** An `openAICompatible()` provider with one endpoint's settings; `overrides` win. */
export const endpoint = (name: EndpointName, overrides: Partial<OpenAICompatibleConfig> = {}): Provider =>
  openAICompatible({
    name,
    model: "test",
    apiKey: "test",
    ...ENDPOINTS[name],
    ...overrides,
    compat: { ...ENDPOINTS[name].compat, ...overrides.compat },
  });
