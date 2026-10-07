import { openAICompatible } from "./openai-compatible.js";
import type { OpenAICompatibleConfig, Provider, ProviderRequestDefaults, ProviderTimeouts } from "../types.js";

export type OpenAIOptions = ProviderRequestDefaults & {
  apiKey?: string;
  baseURL?: string;
  contextWindow?: number;
  temperature?: number;
  creditsPerInputToken?: number;
  creditsPerOutputToken?: number;
  /** "auto" (default) and "never" keep OpenAI's IDs; "strict9" maps them to 9-character IDs. */
  normalizeToolCallIds?: "auto" | "never" | "strict9";
  timeouts?: ProviderTimeouts;
};

export const openai = (model: string, options?: OpenAIOptions): Provider => {
  const config: OpenAICompatibleConfig = {
    name: "openai",
    model,
    baseURL: options?.baseURL ?? "https://api.openai.com/v1",
    apiKey: options?.apiKey ?? globalThis.process?.env?.OPENAI_API_KEY,
    contextWindow: options?.contextWindow,
    temperature: options?.temperature,
    creditsPerInputToken: options?.creditsPerInputToken,
    creditsPerOutputToken: options?.creditsPerOutputToken,
    timeouts: options?.timeouts,
    reasoningEffort: options?.reasoningEffort,
    extraBody: options?.extraBody,
    compat: {
      toolCallIdPolicy: options?.normalizeToolCallIds === "strict9" ? "strict9" : "passthrough",
      supportsUsageInStreaming: true,
      thinkingFormat: "none",
      maxTokensField: "max_completion_tokens",
      structuredOutput: "response_format",
    },
  };

  return openAICompatible(config);
};
