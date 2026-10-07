import { openAICompatible } from "./openai-compatible.js";
import type { OpenAICompatibleConfig, Provider, ProviderRequestDefaults, ProviderTimeouts } from "../types.js";

export type VLLMOptions = ProviderRequestDefaults & {
  apiKey?: string;
  baseURL?: string;
  contextWindow?: number;
  temperature?: number;
  creditsPerInputToken?: number;
  creditsPerOutputToken?: number;
  timeouts?: ProviderTimeouts;
};

export const vllm = (model: string, options?: VLLMOptions): Provider => {
  const config: OpenAICompatibleConfig = {
    name: "vllm",
    model,
    baseURL: options?.baseURL ?? "http://localhost:8000/v1",
    apiKey: options?.apiKey,
    contextWindow: options?.contextWindow,
    temperature: options?.temperature,
    creditsPerInputToken: options?.creditsPerInputToken,
    creditsPerOutputToken: options?.creditsPerOutputToken,
    timeouts: options?.timeouts,
    reasoningEffort: options?.reasoningEffort,
    extraBody: options?.extraBody,
    compat: {
      toolCallIdPolicy: "passthrough",
      supportsUsageInStreaming: true,
      thinkingFormat: "none",
      maxTokensField: "max_tokens",
      structuredOutput: "vllm_structured_outputs",
    },
  };

  return openAICompatible(config);
};
