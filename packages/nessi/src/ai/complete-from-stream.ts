import type {
  AssistantContentBlock,
  GenerateRequest,
  GenerateResult,
  Provider,
  ToolCallBlock,
  Usage,
} from "./types.js";
import { appendAssistantContentBlock, buildAssistantMessageFromContent } from "./shared/messages.js";

export const completeFromStream = async (
  /** `name`, when given, is recorded as the message's producer. */
  provider: Pick<Provider, "model" | "stream"> & { name?: string },
  request: GenerateRequest,
): Promise<GenerateResult> => {
  let usage: Usage | undefined;
  let finishReason: GenerateResult["finishReason"] | undefined;
  const content: AssistantContentBlock[] = [];
  const toolCalls: ToolCallBlock[] = [];

  for await (const event of provider.stream(request)) {
    switch (event.type) {
      case "block_start":
      case "block_delta":
        break;
      case "block_end":
        if (event.block.type === "tool_call") toolCalls.push(event.block);
        appendAssistantContentBlock(content, event.block);
        break;
      case "issue":
        if (event.issue.kind === "provider_error") throw new Error(event.issue.message);
        if (event.issue.kind === "timeout" && event.issue.scope !== "tool") throw new Error(event.issue.message);
        break;
      case "usage":
        usage = event.usage;
        finishReason = event.finishReason ?? finishReason;
        break;
    }
  }

  finishReason ??= toolCalls.length > 0 ? "tool_use" : "stop";
  return {
    message: buildAssistantMessageFromContent(provider.model, content, usage, finishReason, provider.name),
    usage,
    finishReason,
    providerMeta: { model: provider.model },
  };
};
