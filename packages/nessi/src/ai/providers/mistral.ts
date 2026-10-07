import { formatConnectionError, normalizeHttpError, streamEndedError } from "../shared/errors.js";
import { assertOnlySupportedFiles, buildAssistantMessage } from "../shared/messages.js";
import { ensureRecord, safeJsonParse, stringifyJson } from "../shared/json.js";
import { resolveReasoning, withExtraBody } from "../shared/request-options.js";
import { openSSEStream } from "../shared/stream-helpers.js";
import { normalizeProviderStream } from "../shared/tool-stream-normalizer.js";
import { createStrictToolCallIdFactory } from "../shared/tool-call-ids.js";
import { toOpenAITools } from "../shared/tools.js";
import { applyCredits, makeUsage } from "../shared/usage.js";
import type {
  GenerateRequest,
  GenerateResult,
  Message,
  Provider,
  ProviderRequestDefaults,
  ProviderTimeouts,
  RawStreamEvent,
  StreamEvent,
  ToolCallBlock,
  Usage,
} from "../types.js";

type MistralMessage = {
  role: "system" | "user" | "assistant" | "tool";
  content: string | Array<{ type: "text"; text: string } | { type: "image_url"; image_url: { url: string } }> | null;
  tool_calls?: Array<{
    id: string;
    type: "function";
    function: { name: string; arguments: string };
  }>;
  tool_call_id?: string;
  name?: string;
};

// Reasoning models (Magistral) return content as typed chunks instead of a string.
type MistralContent =
  | string
  | null
  | Array<
    | { type: "text"; text: string }
    | { type: "thinking"; thinking: Array<{ type: "text"; text: string }> }
  >;

type MistralChunk = {
  choices?: Array<{
    index: number;
    delta: {
      content?: MistralContent;
      tool_calls?: Array<{
        index: number;
        id?: string;
        function?: { name?: string; arguments?: string };
      }>;
    };
    message?: {
      content?: MistralContent;
      tool_calls?: Array<{
        id?: string;
        function?: { name?: string; arguments?: string };
      }>;
    };
    finish_reason: string | null;
  }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
  };
};

const usageFromChunk = (chunk: MistralChunk, options?: MistralOptions): Usage | undefined => {
  if (!chunk.usage) return undefined;
  return applyCredits(
    makeUsage(chunk.usage.prompt_tokens ?? 0, chunk.usage.completion_tokens ?? 0),
    options?.creditsPerInputToken,
    options?.creditsPerOutputToken,
  );
};

const splitContent = (content: MistralContent | undefined) => {
  if (typeof content === "string") return { text: content, thinking: "" };
  let text = "";
  let thinking = "";
  for (const chunk of content ?? []) {
    if (chunk.type === "text") text += chunk.text;
    else if (chunk.type === "thinking") thinking += chunk.thinking.map((part) => part.text ?? "").join("");
  }
  return { text, thinking };
};

const convertMessages = (messages: Message[], systemPrompt: string | undefined, options?: MistralOptions) => {
  const out: MistralMessage[] = [];
  const pendingToolIds = new Map<string, string[]>();
  const makeStrictId = options?.normalizeToolCallIds === "strict9" ? createStrictToolCallIdFactory() : null;

  if (systemPrompt) out.push({ role: "system", content: systemPrompt });

  for (const message of messages) {
    if (message.role === "user") {
      assertOnlySupportedFiles(message.content, true, "mistral");
      const parts = message.content.map((part) => {
        if (typeof part === "string") return { type: "text" as const, text: part };
        if (part.type === "text") return { type: "text" as const, text: part.text };
        return { type: "image_url" as const, image_url: { url: `data:${part.mediaType};base64,${part.data}` } };
      });
      if (parts.length === 1 && parts[0]?.type === "text") out.push({ role: "user", content: parts[0].text });
      else out.push({ role: "user", content: parts });
      continue;
    }

    if (message.role === "assistant") {
      let text = "";
      const toolCalls: NonNullable<MistralMessage["tool_calls"]> = [];
      for (const block of message.content) {
        if (block.type === "text") text += block.text;
        else if (block.type === "tool_call") {
          const mappedId = makeStrictId ? makeStrictId(block.id) : block.id;
          if (makeStrictId) {
            const queue = pendingToolIds.get(block.id) ?? [];
            queue.push(mappedId);
            pendingToolIds.set(block.id, queue);
          }
          toolCalls.push({
            id: mappedId,
            type: "function",
            function: { name: block.name, arguments: stringifyJson(block.args) },
          });
        }
      }
      const next: MistralMessage = { role: "assistant", content: text || null };
      if (toolCalls.length > 0) next.tool_calls = toolCalls;
      out.push(next);
      continue;
    }

    const queue = pendingToolIds.get(message.callId);
    const mappedId = makeStrictId ? queue?.shift() : message.callId;
    if (!mappedId) continue;
    if (makeStrictId && queue && queue.length === 0) pendingToolIds.delete(message.callId);
    out.push({
      role: "tool",
      content: stringifyJson(message.result),
      name: message.name,
      tool_call_id: mappedId,
    });
  }

  return out;
};

const mapFinishReason = (reason: string | null | undefined, hasTools: boolean) => {
  if (reason === "tool_calls") return "tool_use" as const;
  if (reason === "length") return "max_tokens" as const;
  if (hasTools) return "tool_use" as const;
  return "stop" as const;
};

const responseFormatName = (name: string | undefined) => {
  const safe = (name ?? "structured_output").replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 64);
  return safe || "structured_output";
};

const applyResponseFormat = (body: Record<string, unknown>, request: GenerateRequest) => {
  if (!request.responseFormat) return;
  body.response_format = {
    type: "json_schema",
    json_schema: {
      name: responseFormatName(request.responseFormat.name),
      schema: request.responseFormat.schema,
      strict: true,
    },
  };
};

export type MistralOptions = ProviderRequestDefaults & {
  apiKey?: string;
  baseURL?: string;
  contextWindow?: number;
  temperature?: number;
  normalizeToolCallIds?: "strict9" | "never";
  creditsPerInputToken?: number;
  creditsPerOutputToken?: number;
  timeouts?: ProviderTimeouts;
};

export const mistral = (model: string, options?: MistralOptions): Provider => {
  const baseURL = (options?.baseURL ?? "https://api.mistral.ai/v1").replace(/\/+$/, "");
  const resolveTemperature = (request: GenerateRequest) => request.temperature ?? options?.temperature;

  return {
    name: "mistral",
    family: "mistral",
    model,
    contextWindow: options?.contextWindow ?? 128_000,
    capabilities: {
      streaming: true,
      tools: true,
      images: true,
      thinking: true,
      usage: true,
      structuredOutput: true,
    },

    async complete(request: GenerateRequest): Promise<GenerateResult> {
      const body: Record<string, unknown> = {
        model,
        messages: convertMessages(request.messages, request.systemPrompt, options),
        stream: false,
      };
      if (request.tools?.length) {
        body.tools = toOpenAITools(request.tools);
        body.parallel_tool_calls = true;
      }
      applyResponseFormat(body, request);
      const temperature = resolveTemperature(request);
      if (temperature !== undefined) body.temperature = temperature;
      if (request.maxOutputTokens !== undefined) body.max_tokens = request.maxOutputTokens;

      const response = await fetch(`${baseURL}/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${options?.apiKey ?? globalThis.process?.env?.MISTRAL_API_KEY ?? ""}`,
        },
        body: JSON.stringify(withExtraBody(body, request, options)),
        signal: request.signal,
      }).catch((error: unknown) => {
        throw new Error(formatConnectionError("mistral", error));
      });

      if (!response.ok) {
        const normalized = await normalizeHttpError("mistral", response);
        throw new Error(normalized.error);
      }

      const payload = safeJsonParse<MistralChunk>(await response.text());
      if (!payload) throw new Error("mistral returned invalid JSON.");
      const choice = payload.choices?.[0];
      const toolCalls: ToolCallBlock[] = (choice?.message?.tool_calls ?? []).map((call, index) => ({
        type: "tool_call",
        id: call.id ?? `mistral-${index}`,
        name: call.function?.name ?? "",
        args: ensureRecord(safeJsonParse(call.function?.arguments ?? "{}")),
      }));
      const usage = usageFromChunk(payload, options);
      const finishReason = mapFinishReason(choice?.finish_reason, toolCalls.length > 0);
      const { text, thinking } = splitContent(choice?.message?.content);

      return {
        message: buildAssistantMessage(model, text, thinking, toolCalls, usage, finishReason, "mistral"),
        usage,
        finishReason,
        providerMeta: { model },
      };
    },

    stream(request: GenerateRequest): AsyncIterable<StreamEvent> {
      const raw = async function* (): AsyncIterable<RawStreamEvent> {
      const body: Record<string, unknown> = {
        model,
        messages: convertMessages(request.messages, request.systemPrompt, options),
        stream: true,
      };
      if (request.tools?.length) {
        body.tools = toOpenAITools(request.tools);
        body.parallel_tool_calls = true;
      }
      applyResponseFormat(body, request);
      const temperature = resolveTemperature(request);
      if (temperature !== undefined) body.temperature = temperature;
      if (request.maxOutputTokens !== undefined) body.max_tokens = request.maxOutputTokens;

      const result = await openSSEStream(
        `${baseURL}/chat/completions`,
        {
          "Content-Type": "application/json",
          Authorization: `Bearer ${options?.apiKey ?? globalThis.process?.env?.MISTRAL_API_KEY ?? ""}`,
        },
        withExtraBody(body, request, options),
        "mistral",
        request.signal,
        undefined,
        options?.timeouts,
      );

      if (!result.ok) {
        yield result.error;
        return;
      }

      type ToolBuffer = { callId: string; name: string; argsBuffer: string; started: boolean };
      const buffers = new Map<number, ToolBuffer>();
      const pendingToolCalls: ToolBuffer[] = [];
      let latestUsage: Usage | undefined;
      let latestFinishReason: GenerateResult["finishReason"] | undefined;
      let sawDone = false;
      const startToolCall = function* (buffer: ToolBuffer) {
        if (buffer.started || !buffer.name.trim()) return;
        buffer.started = true;
        yield { type: "tool_start" as const, callId: buffer.callId, name: buffer.name };
        if (buffer.argsBuffer) yield { type: "tool_delta" as const, callId: buffer.callId, argsDelta: buffer.argsBuffer };
      };
      const flush = function* (): Generator<RawStreamEvent> {
        for (const buffer of pendingToolCalls) {
          yield* startToolCall(buffer);
          yield {
            type: "tool_call" as const,
            callId: buffer.callId,
            name: buffer.name,
            args: ensureRecord(safeJsonParse(buffer.argsBuffer || "{}")),
          };
        }
        buffers.clear();
        pendingToolCalls.length = 0;
      };

      for await (const event of result.events) {
        if (event.data === "[DONE]") {
          sawDone = true;
          break;
        }
        const chunk = safeJsonParse<MistralChunk>(event.data);
        if (!chunk) continue;
        const choice = chunk.choices?.[0];
        const usage = usageFromChunk(chunk, options);
        if (!choice) {
          if (usage) {
            latestUsage = usage;
            yield { type: "usage", usage };
          }
          continue;
        }
        if (typeof choice.delta.content === "string") {
          if (choice.delta.content) yield { type: "text", delta: choice.delta.content };
        } else {
          for (const chunk of choice.delta.content ?? []) {
            if (chunk.type === "text") yield { type: "text", delta: chunk.text };
            else if (chunk.type === "thinking") {
              for (const part of chunk.thinking) {
                if (part.text) yield { type: "thinking", delta: part.text };
              }
            }
          }
        }
        if (choice.delta.tool_calls) {
          for (const toolCall of choice.delta.tool_calls) {
            const current = buffers.get(toolCall.index);
            // Some servers reuse one index for parallel calls; a new id starts a new call.
            const existing = current && (!toolCall.id || toolCall.id === current.callId) ? current : undefined;
            if (!existing) {
              const callId = toolCall.id ?? `mistral-${toolCall.index}`;
              const name = toolCall.function?.name ?? "";
              const argsDelta = toolCall.function?.arguments ?? "";
              const buffer = {
                callId,
                name,
                argsBuffer: argsDelta,
                started: false,
              };
              buffers.set(toolCall.index, buffer);
              pendingToolCalls.push(buffer);
              yield* startToolCall(buffer);
            } else {
              if (toolCall.function?.name) existing.name = toolCall.function.name;
              const argsDelta = toolCall.function?.arguments ?? "";
              if (argsDelta) existing.argsBuffer += argsDelta;
              const wasStarted = existing.started;
              yield* startToolCall(existing);
              if (wasStarted && argsDelta) {
                yield { type: "tool_delta", callId: existing.callId, argsDelta };
              }
            }
          }
        }
        if (choice.finish_reason === "tool_calls") {
          yield* flush();
        }
        if (choice.finish_reason) latestFinishReason = mapFinishReason(choice.finish_reason, false);
        if (usage) {
          latestUsage = usage;
          yield { type: "usage", usage };
        }
      }

      if (!latestFinishReason && !sawDone) {
        yield streamEndedError("mistral");
        return;
      }

      if (pendingToolCalls.length > 0) {
        latestFinishReason = "tool_use";
        yield* flush();
      }

      yield {
        type: "usage",
        usage: latestUsage ?? makeUsage(),
        finishReason: latestFinishReason ?? "stop",
      };
      };
      return normalizeProviderStream(raw(), { suppressTextAfterMalformedTool: true });
    },
  };
};
