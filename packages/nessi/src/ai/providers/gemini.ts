import { formatConnectionError, normalizeHttpError, streamEndedError } from "../shared/errors.js";
import {
  appendAssistantContentBlock,
  assertOnlySupportedFiles,
  buildAssistantMessageFromContent,
} from "../shared/messages.js";
import { safeJsonParse } from "../shared/json.js";
import { resolveReasoning, withExtraBody } from "../shared/request-options.js";
import { openSSEStream } from "../shared/stream-helpers.js";
import { normalizeProviderStream } from "../shared/tool-stream-normalizer.js";
import { toGeminiTools } from "../shared/tools.js";
import { applyCredits, makeUsage } from "../shared/usage.js";
import type {
  AssistantContentBlock,
  GenerateRequest,
  GenerateResult,
  Message,
  Provider,
  ProviderRequestDefaults,
  ProviderTimeouts,
  RawStreamEvent,
  StreamEvent,
} from "../types.js";

type GeminiPart = {
  text?: string;
  /** Marks a thought summary part. */
  thought?: boolean;
  /** Opaque signature that must be returned in the same part (required for Gemini 3 function calls). */
  thoughtSignature?: string;
  inlineData?: { mimeType: string; data: string };
  functionCall?: { name: string; args?: Record<string, unknown> };
  functionResponse?: { name: string; response: Record<string, unknown> };
};

type GeminiContent = {
  role: "user" | "model";
  parts: GeminiPart[];
};

type GeminiResponse = {
  promptFeedback?: { blockReason?: string };
  candidates?: Array<{
    content?: GeminiContent;
    finishReason?: string;
  }>;
  usageMetadata?: {
    promptTokenCount?: number;
    candidatesTokenCount?: number;
    thoughtsTokenCount?: number;
    totalTokenCount?: number;
  };
};

export type GeminiOptions = ProviderRequestDefaults & {
  apiKey?: string;
  baseURL?: string;
  contextWindow?: number;
  temperature?: number;
  maxOutputTokens?: number;
  creditsPerInputToken?: number;
  creditsPerOutputToken?: number;
  timeouts?: ProviderTimeouts;
};

const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value);

// Gemini expects an object; "output" and "error" are the documented keys for wrapped values.
const functionResponseBody = (result: unknown, isError: boolean | undefined): Record<string, unknown> => {
  if (isError) return { error: result };
  return isPlainObject(result) ? result : { output: result };
};

const FOREIGN_CALL_SIGNATURE = "skip_thought_signature_validator";

const convertMessages = (messages: Message[]) => {
  const out: GeminiContent[] = [];
  for (const message of messages) {
    if (message.role === "user") {
      assertOnlySupportedFiles(message.content, true, "gemini");
      out.push({
        role: "user",
        parts: message.content.map((part) => {
          if (typeof part === "string") return { text: part };
          if (part.type === "text") return { text: part.text };
          return { inlineData: { mimeType: part.mediaType, data: part.data } };
        }),
      });
      continue;
    }

    if (message.role === "assistant") {
      // Signatures go back only to Gemini, in the part they came with. Unsigned thought summaries are not resent.
      const ownMessage = message.provider === "gemini";
      const signed = (signature: string | undefined) => (ownMessage && signature !== undefined ? { thoughtSignature: signature } : {});
      const parts: GeminiPart[] = [];
      let firstCall = true;
      for (const block of message.content) {
        if (block.type === "thinking") {
          if (ownMessage && block.signature !== undefined) {
            parts.push({ text: block.thinking, thought: true, thoughtSignature: block.signature });
          }
        } else if (block.type === "text") parts.push({ text: block.text, ...signed(block.signature) });
        else if (block.type === "tool_call") {
          const part: GeminiPart = { functionCall: { name: block.name, args: block.args }, ...signed(block.signature) };
          // Gemini 3 rejects unsigned function calls; this documented value skips the check for
          // history that Gemini did not produce.
          if (!ownMessage && firstCall) part.thoughtSignature = FOREIGN_CALL_SIGNATURE;
          firstCall = false;
          parts.push(part);
        }
      }
      if (parts.length > 0) out.push({ role: "model", parts });
      continue;
    }

    const part: GeminiPart = {
      functionResponse: {
        name: message.name,
        response: functionResponseBody(message.result, message.isError),
      },
    };
    // Responses to parallel calls must share one content entry.
    const last = out.at(-1);
    if (last?.role === "user" && last.parts.every((existing) => existing.functionResponse)) last.parts.push(part);
    else out.push({ role: "user", parts: [part] });
  }
  return out;
};

const usageFromResponse = (response: GeminiResponse, options?: GeminiOptions) =>
  applyCredits(
    makeUsage(
      response.usageMetadata?.promptTokenCount ?? 0,
      // Thinking tokens are billed as output; other providers include them in output too.
      (response.usageMetadata?.candidatesTokenCount ?? 0) + (response.usageMetadata?.thoughtsTokenCount ?? 0),
    ),
    options?.creditsPerInputToken,
    options?.creditsPerOutputToken,
  );

/**
 * Levels go to `thinkingLevel` (Gemini 3); "none" and the deprecated `disableReasoning` set a zero
 * thinking budget. Models that cannot turn thinking off or do not know a level answer with a 400.
 */
const applyReasoning = (generationConfig: Record<string, unknown>, request: GenerateRequest, options?: GeminiOptions) => {
  const reasoning = resolveReasoning(request, options);
  if (reasoning.legacyDisable || reasoning.effort === "none") generationConfig.thinkingConfig = { thinkingBudget: 0 };
  else if (reasoning.effort !== undefined) generationConfig.thinkingConfig = { thinkingLevel: reasoning.effort };
};

const signatureOf = (part: GeminiPart) => (part.thoughtSignature !== undefined ? { signature: part.thoughtSignature } : {});

const mapFinishReason = (reason: string | undefined, hasTools: boolean) => {
  if (reason === "MAX_TOKENS") return "max_tokens" as const;
  if (hasTools) return "tool_use" as const;
  return "stop" as const;
};

const createRequestId = () =>
  globalThis.crypto?.randomUUID?.() ?? `gemini-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;

export const gemini = (model: string, options?: GeminiOptions): Provider => {
  const baseURL = (options?.baseURL ?? "https://generativelanguage.googleapis.com/v1beta/models").replace(/\/+$/, "");
  const apiKey = options?.apiKey ?? globalThis.process?.env?.GEMINI_API_KEY ?? globalThis.process?.env?.GOOGLE_API_KEY;

  const urlFor = (path: "generateContent" | "streamGenerateContent") => {
    const key = apiKey ? `?key=${encodeURIComponent(apiKey)}` : "";
    const alt = path === "streamGenerateContent" ? `${key ? "&" : "?"}alt=sse` : "";
    return `${baseURL}/${model}:${path}${key}${alt}`;
  };

  const buildBody = (request: GenerateRequest) => {
    const body: Record<string, unknown> = {
      contents: convertMessages(request.messages),
    };
    if (request.systemPrompt) {
      body.systemInstruction = { parts: [{ text: request.systemPrompt }] };
    }
    if (request.tools?.length) body.tools = toGeminiTools(request.tools);
    const generationConfig: Record<string, unknown> = {};
    const temperature = request.temperature ?? options?.temperature;
    if (temperature !== undefined) generationConfig.temperature = temperature;
    const maxOutputTokens = request.maxOutputTokens ?? options?.maxOutputTokens;
    if (maxOutputTokens !== undefined) generationConfig.maxOutputTokens = maxOutputTokens;
    applyReasoning(generationConfig, request, options);
    if (request.responseFormat) {
      generationConfig.responseMimeType = "application/json";
      generationConfig.responseJsonSchema = request.responseFormat.schema;
    }
    if (Object.keys(generationConfig).length > 0) body.generationConfig = generationConfig;
    return withExtraBody(body, request, options);
  };

  return {
    name: "gemini",
    family: "gemini",
    model,
    contextWindow: options?.contextWindow ?? 1_000_000,
    capabilities: {
      streaming: true,
      tools: true,
      images: true,
      thinking: true,
      usage: true,
      structuredOutput: true,
    },

    async complete(request: GenerateRequest): Promise<GenerateResult> {
      const requestId = createRequestId();
      const response = await fetch(urlFor("generateContent"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildBody(request)),
        signal: request.signal,
      }).catch((error: unknown) => {
        throw new Error(formatConnectionError("gemini", error));
      });

      if (!response.ok) {
        const normalized = await normalizeHttpError("gemini", response);
        throw new Error(normalized.error);
      }

      const payload = safeJsonParse<GeminiResponse>(await response.text());
      if (!payload) throw new Error("gemini returned invalid JSON.");
      const candidate = payload.candidates?.[0];
      const content: AssistantContentBlock[] = [];
      let toolCount = 0;
      for (const part of candidate?.content?.parts ?? []) {
        if (part.functionCall) {
          content.push({
            type: "tool_call",
            id: `${requestId}-${toolCount++}`,
            name: part.functionCall.name,
            args: part.functionCall.args ?? {},
            ...signatureOf(part),
          });
        } else if (part.thought) {
          appendAssistantContentBlock(content, { type: "thinking", thinking: part.text ?? "", ...signatureOf(part) });
        } else if (part.text !== undefined || part.thoughtSignature !== undefined) {
          appendAssistantContentBlock(content, { type: "text", text: part.text ?? "", ...signatureOf(part) });
        }
      }
      const usage = usageFromResponse(payload, options);
      const finishReason = mapFinishReason(candidate?.finishReason, toolCount > 0);

      return {
        message: buildAssistantMessageFromContent(model, content, usage, finishReason, "gemini"),
        usage,
        finishReason,
        providerMeta: { model },
      };
    },

    stream(request: GenerateRequest): AsyncIterable<StreamEvent> {
      const raw = async function* (): AsyncIterable<RawStreamEvent> {
      const result = await openSSEStream(
        urlFor("streamGenerateContent"),
        { "Content-Type": "application/json" },
        buildBody(request),
        "gemini",
        request.signal,
        undefined,
        options?.timeouts,
      );

      if (!result.ok) {
        yield result.error;
        return;
      }

      let toolCounter = 0;
      const requestId = createRequestId();
      let latestUsage: ReturnType<typeof usageFromResponse> | undefined;
      let rawFinishReason: string | undefined;
      for await (const event of result.events) {
        if (event.data === "[DONE]") break;
        const payload = safeJsonParse<GeminiResponse>(event.data);
        if (!payload) continue;
        if (payload.usageMetadata) latestUsage = usageFromResponse(payload, options);
        // A blocked prompt has no candidates and therefore no finish reason; it is not a truncated stream.
        const blockReason = payload.promptFeedback?.blockReason;
        if (blockReason) {
          if (latestUsage) yield { type: "usage", usage: latestUsage };
          yield { type: "error", error: `gemini blocked the prompt (${blockReason}).`, retryable: false };
          return;
        }
        const candidate = payload.candidates?.[0];
        const parts = candidate?.content?.parts ?? [];
        if (candidate?.finishReason) rawFinishReason = candidate.finishReason;
        for (const part of parts) {
          if (part.functionCall) {
            const callId = `${requestId}-${toolCounter++}`;
            yield { type: "tool_start", callId, name: part.functionCall.name };
            yield {
              type: "tool_call",
              callId,
              name: part.functionCall.name,
              args: part.functionCall.args ?? {},
              ...signatureOf(part),
            };
          } else if (part.thought) {
            yield { type: "thinking", delta: part.text ?? "", ...signatureOf(part) };
          } else if (part.text || part.thoughtSignature !== undefined) {
            // A signature can arrive in a part with empty text; it belongs to the current text.
            yield { type: "text", delta: part.text ?? "", ...signatureOf(part) };
          }
        }
        // usageMetadata is cumulative; it is reported once at the end so text blocks stay intact.
      }
      if (!rawFinishReason) {
        yield streamEndedError("gemini");
        return;
      }
      yield {
        type: "usage",
        usage: latestUsage ?? makeUsage(),
        finishReason: mapFinishReason(rawFinishReason, toolCounter > 0),
      };
      };
      return normalizeProviderStream(raw(), { suppressTextAfterMalformedTool: true });
    },
  };
};
