import { openAICompatible } from "./openai-compatible.js";
import type { OpenAICompatibleConfig, Provider, ProviderRequestDefaults, ProviderTimeouts } from "../types.js";

export type OpenRouterOptions = ProviderRequestDefaults & {
  apiKey?: string;
  baseURL?: string;
  contextWindow?: number;
  temperature?: number;
  referer?: string;
  title?: string;
  creditsPerInputToken?: number;
  creditsPerOutputToken?: number;
  timeouts?: ProviderTimeouts;
};

export const openrouter = (model: string, options?: OpenRouterOptions): Provider => {
  const headers: Record<string, string> = {};
  if (options?.referer) headers["HTTP-Referer"] = options.referer;
  if (options?.title) headers["X-Title"] = options.title;

  const config: OpenAICompatibleConfig = {
    name: "openrouter",
    model,
    baseURL: options?.baseURL ?? "https://openrouter.ai/api/v1",
    apiKey: options?.apiKey ?? globalThis.process?.env?.OPENROUTER_API_KEY,
    contextWindow: options?.contextWindow,
    temperature: options?.temperature,
    creditsPerInputToken: options?.creditsPerInputToken,
    creditsPerOutputToken: options?.creditsPerOutputToken,
    timeouts: options?.timeouts,
    reasoningEffort: options?.reasoningEffort,
    extraBody: options?.extraBody,
    headers,
    compat: {
      toolCallIdPolicy: "passthrough",
      supportsUsageInStreaming: true,
      thinkingFormat: "reasoning_details",
      reasoningFormat: "openrouter",
      maxTokensField: "max_tokens",
      structuredOutput: "response_format",
    },
  };

  return openAICompatible(config);
};
