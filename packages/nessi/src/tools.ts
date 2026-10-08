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

export const toolToJsonSchema = (tool: ServerTool | ClientTool) => ({
  name: tool.def.name,
  description: tool.def.description,
  parameters: z.toJSONSchema(tool.def.inputSchema, { target: "draft-07" }),
})

export const toolToSpec = (tool: ServerTool | ClientTool) => {
  const schema = toolToJsonSchema(tool);
  return {
    name: schema.name,
    description: schema.description,
    inputSchema: schema.parameters,
  };
}
