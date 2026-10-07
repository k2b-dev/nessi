// ============================================================================
// nessi.structured() - schema-valid task helper
// ============================================================================

import { z } from "zod";
import { aggregateFromTurns, buildLoopTiming } from "./aggregates.js";
import { extractAssistantText } from "./ai/shared/messages.js";
import { nessi } from "./nessi.js";
import { memoryStore } from "./stores.js";
import { defineTool } from "./tools.js";
import { createLoopId, toErrorMessage } from "./utils.js";
import type {
  AssistantMessage,
  ContentPart,
  DoneReason,
  JsonSchemaObject,
  LoopAggregate,
  LoopTurnAggregate,
  OutboundEvent,
  ResponseFormat,
  ServerTool,
  StructuredInput,
  StructuredMeta,
  StructuredOptions,
  StructuredResult,
  Tool,
  ToolContext,
  UserMessage,
} from "./types.js";
import type { GenerateResult } from "./ai/index.js";

const DEFAULT_SYSTEM_PROMPT = "You produce schema-valid structured data for the caller.";
const SUBMIT_RESULT_TOOL_NAME = "submit_result";

export type StructuredOutputErrorCode =
  | "invalid_output"
  | "unsupported_tool"
  | "loop_failed"
  | "max_turns"
  | "aborted";

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
  "anyOf" in schema || "oneOf" in schema || "allOf" in schema || "not" in schema;

const isStrictNativeSchemaCompatible = (schema: unknown): boolean => {
  if (!isRecord(schema)) return false;
  if (hasUnsupportedComposition(schema)) return false;

  const type = schema.type;
  if (type === "object") {
    const properties = schema.properties;
    const required = schema.required;
    if (!isRecord(properties)) return true;
    if (!Array.isArray(required)) return false;
    for (const key of Object.keys(properties)) {
      if (!required.includes(key)) return false;
      if (!isStrictNativeSchemaCompatible(properties[key])) return false;
    }
    return true;
  }

  if (type === "array") {
    return schema.items === undefined || isStrictNativeSchemaCompatible(schema.items);
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

const unsupportedToolError = (message: string) =>
  new StructuredOutputError(message, "unsupported_tool");

const validateStructuredTools = (tools: ServerTool[]) => {
  for (const tool of tools as Tool[]) {
    if (tool.kind !== "server") {
      throw unsupportedToolError("nessi.structured() only accepts server tools. Use nessi() for client tools.");
    }
    if (tool.def.needsApproval) {
      throw unsupportedToolError(`nessi.structured() does not support approval tools: ${tool.def.name}`);
    }
    if (tool.def.name === SUBMIT_RESULT_TOOL_NAME) {
      throw unsupportedToolError(`Tool name "${SUBMIT_RESULT_TOOL_NAME}" is reserved by nessi.structured().`);
    }
  }
};

const wrapStructuredTool = (tool: ServerTool): ServerTool => ({
  kind: "server",
  def: tool.def,
  execute(input: unknown, ctx: ToolContext) {
    return tool.execute(input, {
      callId: ctx.callId,
      signal: ctx.signal,
      async requestApproval() {
        throw new Error("nessi.structured() does not support tool approvals. Use nessi() for interactive tools.");
      },
      async requestClientTool() {
        throw new Error("nessi.structured() does not support client tool bridges. Use nessi() for interactive tools.");
      },
    });
  },
});

const directStructured = async <TOutput extends z.ZodType>(
  options: StructuredOptions<TOutput>,
  inputMessage: UserMessage,
  jsonSchema: JsonSchemaObject,
  loopId: string,
): Promise<StructuredResult<z.infer<TOutput>>> => {
  const startedAt = Date.now();
  let generationMs = 0;
  const completeWithTiming = async (request: Parameters<typeof options.provider.complete>[0]) => {
    const requestStartedAt = Date.now();
    try {
      return await options.provider.complete(request);
    } catch (error) {
      // Match the tool-loop mode, which reports provider failures as StructuredOutputError.
      const code = options.signal?.aborted ? "aborted" : "loop_failed";
      throw new StructuredOutputError(toErrorMessage(error), code, { cause: error });
    } finally {
      generationMs += Math.max(0, Date.now() - requestStartedAt);
    }
  };
  const useResponseFormat = options.provider.capabilities.structuredOutput === true
    && isStrictNativeSchemaCompatible(jsonSchema);
  const responseFormat = useResponseFormat ? createResponseFormat(jsonSchema, options.outputName) : undefined;
  const systemPrompt = useResponseFormat
    ? (options.systemPrompt ?? DEFAULT_SYSTEM_PROMPT)
    : [
        options.systemPrompt ?? DEFAULT_SYSTEM_PROMPT,
        schemaInstruction(jsonSchema, options.outputName),
      ].join("\n\n");

  const attempts: GenerateResult[] = [];
  const first = await completeWithTiming({
    systemPrompt,
    messages: [inputMessage],
    responseFormat,
    temperature: options.temperature,
    maxOutputTokens: options.maxOutputTokens,
    disableReasoning: options.disableReasoning,
    reasoningEffort: options.reasoningEffort,
    extraBody: options.extraBody,
    signal: options.signal,
  });
  attempts.push(first);

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
      disableReasoning: options.disableReasoning,
      reasoningEffort: options.reasoningEffort,
      extraBody: options.extraBody,
      signal: options.signal,
    });
    attempts.push(repair);
    parsed = parseStructuredOutput(options.output, textFromMessage(repair.message));
  }

  const aggregate = aggregateFromTurns(attempts.map(resultToTurn));
  aggregate.timing = buildLoopTiming({
    wallMs: Math.max(0, Date.now() - startedAt),
    generationMs,
    toolExecutionMs: 0,
    actionWaitMs: 0,
  }, aggregate.usage);
  if (!parsed.ok) {
    throw new StructuredOutputError("Provider returned invalid structured output.", "invalid_output", {
      attempts: attempts.length,
      failure: parsed,
      aggregate,
    });
  }

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

const loopErrorCode = (reason: DoneReason): StructuredOutputErrorCode => {
  if (reason === "max_turns") return "max_turns";
  if (reason === "aborted") return "aborted";
  return "loop_failed";
};

const toolLoopStructured = async <TOutput extends z.ZodType>(
  options: StructuredOptions<TOutput>,
  inputMessage: UserMessage,
  jsonSchema: JsonSchemaObject,
  loopId: string,
): Promise<StructuredResult<z.infer<TOutput>>> => {
  const toolSource = options.tools ?? [];

  let submitted: z.infer<TOutput> | undefined;
  const submitDef = defineTool({
    name: SUBMIT_RESULT_TOOL_NAME,
    description: "Submit the final structured result. Call this exactly once when the task is complete.",
    inputSchema: options.output,
    outputSchema: z.object({ accepted: z.literal(true) }),
  });
  const submitTool = submitDef.server(async (input) => {
    submitted = input;
    return { accepted: true };
  });
  (submitTool.def as { terminal?: boolean }).terminal = true;

  let structuredToolError: StructuredOutputError | undefined;
  const prepareTools = (value: unknown): ServerTool[] => {
    if (!Array.isArray(value)) throw new Error("Structured tool resolver must return an array");
    const tools = [...value] as ServerTool[];
    try {
      validateStructuredTools(tools);
    } catch (error) {
      if (error instanceof StructuredOutputError) structuredToolError = error;
      throw error;
    }
    return [...tools.map(wrapStructuredTool), submitTool];
  };
  const tools = typeof toolSource === "function"
    ? async () => prepareTools(await toolSource())
    : prepareTools(toolSource);

  const store = memoryStore();
  await store.append(inputMessage);

  const loop = nessi({
    agentId: options.agentId,
    loopId,
    provider: options.provider,
    store,
    systemPrompt: [
      options.systemPrompt ?? DEFAULT_SYSTEM_PROMPT,
      "Use the available tools when needed.",
      `When the final answer is ready, call ${SUBMIT_RESULT_TOOL_NAME} with the structured result.`,
      "Do not return the final structured result as normal assistant text.",
      schemaInstruction(jsonSchema, options.outputName),
    ].join("\n\n"),
    tools,
    maxTurns: options.maxTurns ?? 8,
    temperature: options.temperature,
    maxOutputTokens: options.maxOutputTokens,
    disableReasoning: options.disableReasoning,
    reasoningEffort: options.reasoningEffort,
    extraBody: options.extraBody,
    signal: options.signal,
  });

  let loopEnd: Extract<OutboundEvent, { type: "loop_end" }> | undefined;
  let message: AssistantMessage | undefined;

  for await (const event of loop) {
    options.onEvent?.(event);
    if (event.type === "turn_end") message = event.message;
    if (event.type === "loop_end") loopEnd = event;
  }

  if (!loopEnd) {
    throw new StructuredOutputError("Structured tool loop ended without loop_end.", "loop_failed");
  }
  if (loopEnd.reason !== "stop") {
    if (structuredToolError) {
      throw new StructuredOutputError(structuredToolError.message, structuredToolError.code, {
        aggregate: loopEnd.aggregate,
      });
    }
    throw new StructuredOutputError(`Structured tool loop ended with reason: ${loopEnd.reason}`, loopErrorCode(loopEnd.reason), {
      aggregate: loopEnd.aggregate,
    });
  }
  if (submitted === undefined) {
    throw new StructuredOutputError("Structured tool loop ended without a submitted result.", "invalid_output", {
      aggregate: loopEnd.aggregate,
    });
  }

  const lastMessage = message ?? loopEnd.aggregate.turns.at(-1)?.message;
  if (!lastMessage) {
    throw new StructuredOutputError("Structured tool loop ended without an assistant message.", "loop_failed", {
      aggregate: loopEnd.aggregate,
    });
  }

  return {
    output: submitted,
    message: lastMessage,
    aggregate: loopEnd.aggregate,
    reason: loopEnd.reason,
    loopId,
    usage: loopEnd.aggregate.usage,
    structuredMeta: {
      mode: "tool_loop",
      repaired: loopEnd.aggregate.toolErrorCount > 0,
      attempts: loopEnd.aggregate.assistantMessageCount,
      usedResponseFormat: false,
    },
  };
};

export const structured = async <TOutput extends z.ZodType>(
  options: StructuredOptions<TOutput>,
): Promise<StructuredResult<z.infer<TOutput>>> => {
  const loopId = options.loopId?.trim() ? options.loopId : createLoopId();
  const inputMessage = normalizeStructuredInput(options.input);
  const jsonSchema = schemaFor(options.output);
  const hasTools = typeof options.tools === "function" || (options.tools?.length ?? 0) > 0;
  if (hasTools) return toolLoopStructured(options, inputMessage, jsonSchema, loopId);
  return directStructured(options, inputMessage, jsonSchema, loopId);
};
