// ============================================================================
// nessi.structured() - schema-valid task helper
// ============================================================================

import { z } from "zod";
import { aggregateFromTurns, buildLoopTiming } from "./aggregates.js";
import { extractAssistantText } from "./ai/shared/messages.js";
import { createLoopId, rejectRemovedOptions, toErrorMessage } from "./utils.js";
import type {
  AssistantMessage,
  AssistantStopReason,
  ContentPart,
  JsonSchemaObject,
  LoopAggregate,
  LoopTurnAggregate,
  ResponseFormat,
  StructuredInput,
  StructuredMeta,
  StructuredOptions,
  StructuredResult,
  UserMessage,
} from "./types.js";
import { complete } from "./ai/complete.js";
import type { GenerateRequest, GenerateResult } from "./ai/index.js";

const DEFAULT_SYSTEM_PROMPT = "You produce schema-valid structured data for the caller.";

export type StructuredOutputErrorCode =
  | "invalid_output"
  | "loop_failed"
  | "aborted"
  /** The response stopped at the output token limit. */
  | "truncated"
  /** The provider ended the response with an error, a refusal, or an interruption. */
  | "provider_stopped";

export class StructuredOutputError extends Error {
  readonly code: StructuredOutputErrorCode;
  readonly details?: unknown;

  constructor(message: string, code: StructuredOutputErrorCode, details?: unknown) {
    super(message);
    this.name = "StructuredOutputError";
    this.code = code;
    this.details = details;
  }
}

type ParsedOutput<T> =
  | { ok: true; value: T; jsonText: string }
  | { ok: false; message: string; details?: unknown; jsonText?: string };

const normalizeContentParts = (parts: ContentPart[]) =>
  parts.map((part) => (typeof part === "string" ? { type: "text" as const, text: part } : part));

const normalizeStructuredInput = (input: StructuredInput): UserMessage => {
  if (typeof input === "string") return { role: "user", content: [{ type: "text", text: input }] };
  if (Array.isArray(input)) return { role: "user", content: normalizeContentParts(input) };
  return { role: "user", content: normalizeContentParts(input.content) };
};

const schemaFor = (schema: z.ZodType): JsonSchemaObject =>
  stripJsonSchemaMetadata(z.toJSONSchema(schema, { target: "draft-07" })) as JsonSchemaObject;

const stripJsonSchemaMetadata = (value: unknown): unknown => {
  if (Array.isArray(value)) return value.map(stripJsonSchemaMetadata);
  if (!isRecord(value)) return value;
  const out: Record<string, unknown> = {};
  for (const [key, child] of Object.entries(value)) {
    if (key === "$schema") continue;
    out[key] = stripJsonSchemaMetadata(child);
  }
  return out;
};

const createResponseFormat = (schema: JsonSchemaObject, name?: string): ResponseFormat => ({
  type: "json_schema",
  name: name ?? "structured_output",
  schema,
});

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value);

const hasUnsupportedComposition = (schema: Record<string, unknown>) =>
  "oneOf" in schema || "allOf" in schema || "not" in schema;

/**
 * Whether native structured output can enforce the schema. `strict` providers also need closed
 * objects with every property required. `anyOf` (such as Zod's `.nullable()`) works below the root.
 */
const isNativeSchemaCompatible = (schema: unknown, strict: boolean, root = true): boolean => {
  if (!isRecord(schema)) return false;
  if (hasUnsupportedComposition(schema)) return false;

  if ("anyOf" in schema) {
    const { anyOf, ...rest } = schema;
    return !root
      && Array.isArray(anyOf)
      && anyOf.length > 0
      && Object.keys(rest).every((key) => key === "description" || key === "default")
      && anyOf.every((branch) => isNativeSchemaCompatible(branch, strict, false));
  }

  const type = schema.type;
  if (type === "object") {
    const properties = schema.properties;
    const required = schema.required;
    // Open objects and records (`additionalProperties` other than false) are not strict.
    if (strict && schema.additionalProperties !== false) return false;
    if (!isRecord(properties)) return true;
    const requiredKeys: unknown[] = Array.isArray(required) ? required : [];
    for (const key of Object.keys(properties)) {
      if (strict && !requiredKeys.includes(key)) return false;
      if (!isNativeSchemaCompatible(properties[key], strict, false)) return false;
    }
    return true;
  }

  if (type === "array") {
    return schema.items === undefined || isNativeSchemaCompatible(schema.items, strict, false);
  }

  if (Array.isArray(type)) {
    return type.every((entry) => typeof entry === "string" && ["string", "number", "integer", "boolean", "null"].includes(entry));
  }

  if (typeof type === "string") {
    return ["string", "number", "integer", "boolean", "null"].includes(type);
  }

  return Array.isArray(schema.enum) || Object.prototype.hasOwnProperty.call(schema, "const");
};

const textFromMessage = (message: AssistantMessage) => extractAssistantText(message);

const tryParseJson = (text: string): unknown | undefined => {
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
};

const findBalancedJson = (text: string): string | undefined => {
  let start = -1;
  for (let index = 0; index < text.length; index++) {
    const char = text[index]!;
    if (char === "{" || char === "[") {
      start = index;
      break;
    }
  }
  if (start < 0) return undefined;

  const stack: string[] = [];
  let inString = false;
  let escaped = false;

  for (let index = start; index < text.length; index++) {
    const char = text[index]!;
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (char === "\\") {
        escaped = true;
      } else if (char === "\"") {
        inString = false;
      }
      continue;
    }

    if (char === "\"") {
      inString = true;
      continue;
    }
    if (char === "{" || char === "[") {
      stack.push(char === "{" ? "}" : "]");
      continue;
    }
    if (char !== "}" && char !== "]") continue;

    const expected = stack.pop();
    if (expected !== char) return undefined;
    if (stack.length === 0) return text.slice(start, index + 1);
  }

  return undefined;
};

const extractJsonText = (raw: string): string | undefined => {
  const trimmed = raw.trim();
  if (!trimmed) return undefined;

  const fenced = trimmed.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i)?.[1]?.trim();
  if (fenced && tryParseJson(fenced) !== undefined) return fenced;
  if (tryParseJson(trimmed) !== undefined) return trimmed;
  return findBalancedJson(trimmed);
};

const parseStructuredOutput = <TOutput extends z.ZodType>(
  schema: TOutput,
  raw: string,
): ParsedOutput<z.infer<TOutput>> => {
  const jsonText = extractJsonText(raw);
  if (!jsonText) {
    return { ok: false, message: "No JSON value was found in the assistant response." };
  }

  let json: unknown;
  try {
    json = JSON.parse(jsonText);
  } catch (error) {
    return {
      ok: false,
      message: `Assistant response contained invalid JSON: ${toErrorMessage(error)}`,
      jsonText,
    };
  }

  const parsed = schema.safeParse(json);
  if (!parsed.success) {
    return {
      ok: false,
      message: "Assistant response did not match the requested schema.",
      details: parsed.error.issues,
      jsonText,
    };
  }

  return { ok: true, value: parsed.data, jsonText };
};

const schemaInstruction = (schema: JsonSchemaObject, outputName?: string) => [
  "Return exactly one JSON value that satisfies the following JSON Schema.",
  "Do not wrap the JSON in markdown fences. Do not include explanatory text.",
  outputName ? `Schema name: ${outputName}` : "",
  JSON.stringify(schema, null, 2),
].filter(Boolean).join("\n");

const repairInstruction = (failure: Extract<ParsedOutput<unknown>, { ok: false }>) => [
  "The previous response did not match the requested structured output.",
  failure.message,
  failure.details ? JSON.stringify(failure.details, null, 2) : "",
  "Return only corrected JSON that satisfies the schema.",
].filter(Boolean).join("\n");

const resultToTurn = (result: GenerateResult): LoopTurnAggregate => {
  const toolCalls = result.message.content
    .filter((block) => block.type === "tool_call")
    .map((block) => ({
      callId: block.id,
      name: block.name,
      args: block.args,
    }));

  return {
    message: result.message,
    usage: result.usage,
    stopReason: result.message.stopReason,
    toolCalls,
  };
};

/** A response that did not end normally; its text is never accepted or repaired. */
const unfinishedResponse = (
  ...stopReasons: Array<AssistantStopReason | undefined>
): { message: string; code: StructuredOutputErrorCode } | undefined => {
  const reasons = new Set(stopReasons);
  if (reasons.has("aborted")) return { message: "Provider response was aborted.", code: "aborted" };
  if (reasons.has("max_tokens")) {
    return { message: "Provider response stopped at the output token limit before the structured output was complete.", code: "truncated" };
  }
  for (const reason of ["error", "interrupted"] as const) {
    if (reasons.has(reason)) return { message: `Provider ended the response with reason: ${reason}.`, code: "provider_stopped" };
  }
  return undefined;
};

const directStructured = async <TOutput extends z.ZodType>(
  options: StructuredOptions<TOutput>,
  inputMessage: UserMessage,
  jsonSchema: JsonSchemaObject,
  loopId: string,
): Promise<StructuredResult<z.infer<TOutput>>> => {
  const startedAt = Date.now();
  let generationMs = 0;
  const attempts: GenerateResult[] = [];
  const aggregateOfAttempts = () => {
    const aggregate = aggregateFromTurns(attempts.map(resultToTurn));
    aggregate.timing = buildLoopTiming({
      wallMs: Math.max(0, Date.now() - startedAt),
      generationMs,
      toolExecutionMs: 0,
      actionWaitMs: 0,
    }, aggregate.usage);
    return aggregate;
  };
  // Every failure reports the attempts that already ran, so their usage is not lost.
  const failure = (message: string, code: StructuredOutputErrorCode, details?: Record<string, unknown>) =>
    new StructuredOutputError(message, code, { ...details, attempts: attempts.length, aggregate: aggregateOfAttempts() });
  const completeWithTiming = async (request: GenerateRequest) => {
    const requestStartedAt = Date.now();
    let result: GenerateResult;
    try {
      result = await complete(options.provider, request);
    } catch (error) {
      generationMs += Math.max(0, Date.now() - requestStartedAt);
      // Provider failures surface as StructuredOutputError with the ProviderError as cause.
      const code = options.signal?.aborted ? "aborted" : "loop_failed";
      throw failure(toErrorMessage(error), code, { cause: error });
    }
    generationMs += Math.max(0, Date.now() - requestStartedAt);
    attempts.push(result);
    if (options.signal?.aborted) throw failure("Structured output was aborted.", "aborted");
    const unfinished = unfinishedResponse(result.finishReason, result.message.stopReason);
    if (unfinished) throw failure(unfinished.message, unfinished.code);
    return result;
  };
  const { structuredOutput, strictStructuredOutput } = options.provider.capabilities;
  const useResponseFormat = structuredOutput === true
    && isNativeSchemaCompatible(jsonSchema, strictStructuredOutput !== false);
  const responseFormat = useResponseFormat ? createResponseFormat(jsonSchema, options.outputName) : undefined;
  const systemPrompt = useResponseFormat
    ? (options.systemPrompt ?? DEFAULT_SYSTEM_PROMPT)
    : [
        options.systemPrompt ?? DEFAULT_SYSTEM_PROMPT,
        schemaInstruction(jsonSchema, options.outputName),
      ].join("\n\n");

  const first = await completeWithTiming({
    systemPrompt,
    messages: [inputMessage],
    responseFormat,
    temperature: options.temperature,
    maxOutputTokens: options.maxOutputTokens,
    reasoningEffort: options.reasoningEffort,
    extraBody: options.extraBody,
    signal: options.signal,
  });

  let parsed = parseStructuredOutput(options.output, textFromMessage(first.message));
  if (!parsed.ok) {
    const repair = await completeWithTiming({
      systemPrompt: [
        options.systemPrompt ?? DEFAULT_SYSTEM_PROMPT,
        schemaInstruction(jsonSchema, options.outputName),
      ].join("\n\n"),
      messages: [
        inputMessage,
        first.message,
        { role: "user", content: [{ type: "text", text: repairInstruction(parsed) }] },
      ],
      responseFormat,
      temperature: options.temperature,
      maxOutputTokens: options.maxOutputTokens,
      reasoningEffort: options.reasoningEffort,
      extraBody: options.extraBody,
      signal: options.signal,
    });
    parsed = parseStructuredOutput(options.output, textFromMessage(repair.message));
  }

  if (!parsed.ok) throw failure("Provider returned invalid structured output.", "invalid_output", { failure: parsed });
  const aggregate = aggregateOfAttempts();

  const last = attempts[attempts.length - 1]!;
  const repaired = attempts.length > 1;
  const meta: StructuredMeta = {
    mode: repaired ? "repair" : useResponseFormat ? "native" : "fallback",
    repaired,
    attempts: attempts.length,
    usedResponseFormat: useResponseFormat,
  };

  return {
    output: parsed.value,
    message: last.message,
    aggregate,
    reason: "stop",
    loopId,
    usage: aggregate.usage,
    providerMeta: last.providerMeta,
    structuredMeta: meta,
  };
};

export const structured = async <TOutput extends z.ZodType>(
  options: StructuredOptions<TOutput>,
): Promise<StructuredResult<z.infer<TOutput>>> => {
  rejectRemovedOptions(options, {
    tools: "nessi.structured() makes one provider call; use nessi() for tool work.",
    maxTurns: "nessi.structured() makes one provider call; use nessi() for tool work.",
    onEvent: "nessi.structured() makes one provider call; use nessi() for tool work.",
    disableReasoning: 'Use reasoningEffort: "none".',
  }, "nessi.structured()");
  const loopId = options.loopId?.trim() ? options.loopId : createLoopId();
  const inputMessage = normalizeStructuredInput(options.input);
  const jsonSchema = schemaFor(options.output);
  return directStructured(options, inputMessage, jsonSchema, loopId);
};
