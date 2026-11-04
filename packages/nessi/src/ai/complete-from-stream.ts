import type {
  AssistantContentBlock,
  GenerateRequest,
  GenerateResult,
  Provider,
  ToolCallBlock,
  Usage,
} from "./types.js";
import { ProviderError } from "./shared/errors.js";
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
      case "issue": {
        const { issue } = event;
        // A caller abort surfaces from the stream as a failed request; keep the caller's reason.
        if (issue.kind === "timeout" && issue.scope !== "tool") {
          request.signal?.throwIfAborted();
          throw new ProviderError(issue.message, { kind: "timeout", retryable: issue.retryable, scope: issue.scope });
        }
        if (issue.kind === "provider_error") {
          request.signal?.throwIfAborted();
          const { kind, message, ...details } = issue;
          throw new ProviderError(message, { kind, ...details });
        }
        break;
      }
      case "usage":
        usage = event.usage;
        finishReason = event.finishReason ?? finishReason;
        break;
    }
  }

  request.signal?.throwIfAborted();
  finishReason ??= toolCalls.length > 0 ? "tool_use" : "stop";
  return {
    message: buildAssistantMessageFromContent(provider.model, content, usage, finishReason, provider.name),
    usage,
    finishReason,
    providerMeta: { model: provider.model },
  };
};
