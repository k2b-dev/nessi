// ============================================================================
// nessi – Tool Definitions
// ============================================================================

import { z } from "zod";
import type { ToolDefinition, ServerTool, ClientTool, ToolContext } from "./types.js";

// ----------------------------------------------------------------------------
// defineTool()
// ----------------------------------------------------------------------------

export const defineTool = <TInput extends z.ZodType, TOutput extends z.ZodType = z.ZodAny>(config: {
  name: string;
  description: string;
  inputSchema: TInput;
  outputSchema?: TOutput;
  needsApproval?: boolean;
  timeoutMs?: number | false;
  replay?: "safe" | "unsafe";
  concurrency?: "parallel" | "serial";
  toHistoricalResult?: (context: {
    input: z.infer<TInput>;
    output: z.infer<TOutput>;
    callId: string;
  }) => unknown | Promise<unknown>;
}): ToolDefinition<TInput, TOutput> => {
  const def: ToolDefinition<TInput, TOutput> = {
    ...config,
    needsApproval: config.needsApproval ?? false,
    server(execute: (input: z.infer<TInput>, ctx: ToolContext) => Promise<z.infer<TOutput>>) {
      return { kind: "server" as const, def, execute };
    },
    client(execute: (input: z.infer<TInput>) => z.infer<TOutput> | Promise<z.infer<TOutput>>) {
      // Client tools run in the application, which owns their confirmation UI.
      if (def.needsApproval) {
        throw new Error(`Tool "${def.name}": needsApproval is only supported for server tools; ask for confirmation in the client tool itself.`);
      }
      return { kind: "client" as const, def, execute };
    },
  };
  return def;
}

// ----------------------------------------------------------------------------
// Tool → JSON Schema (for provider adapters)
// ----------------------------------------------------------------------------

const deepFreeze = <T>(value: T): T => {
  if (value && typeof value === "object" && !Object.isFrozen(value)) {
    Object.freeze(value);
    for (const child of Object.values(value)) deepFreeze(child);
  }
  return value;
};

const toJsonSchema = (schema: z.ZodType) => z.toJSONSchema(schema, { target: "draft-07" });

export const toolToJsonSchema = (tool: ServerTool | ClientTool) => ({
  name: tool.def.name,
  description: tool.def.description,
  parameters: toJsonSchema(tool.def.inputSchema),
})

// Zod schemas are immutable, so the loop converts each one once, also when a tool resolver builds
// new tools around the same schemas every turn. Metadata registered on a schema after its first
// turn is not picked up.
const loopJsonSchemas = new WeakMap<z.ZodType, ReturnType<typeof toJsonSchema>>();

/** The provider spec the loop sends; its schema is shared between turns and therefore frozen. */
export const cachedToolSpec = (tool: ServerTool | ClientTool) => {
  let inputSchema = loopJsonSchemas.get(tool.def.inputSchema);
  if (!inputSchema) {
    inputSchema = deepFreeze(toJsonSchema(tool.def.inputSchema));
    loopJsonSchemas.set(tool.def.inputSchema, inputSchema);
  }
  return { name: tool.def.name, description: tool.def.description, inputSchema };
};

export const toolToSpec = (tool: ServerTool | ClientTool) => {
  const schema = toolToJsonSchema(tool);
  return {
    name: schema.name,
    description: schema.description,
    inputSchema: schema.parameters,
  };
}
