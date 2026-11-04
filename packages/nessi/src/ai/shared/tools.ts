import type { ToolSpec } from "../types.js";

export const toOpenAITools = (tools: ToolSpec[]) =>
  tools.map((tool) => ({
    type: "function" as const,
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.inputSchema,
    },
  }));
