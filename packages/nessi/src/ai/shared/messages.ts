import { invalidRequestError } from "./errors.js";
import type {
  AssistantMessage,
  AssistantContentBlock,
  ContentPart,
  Message,
  TextBlock,
  ThinkingBlock,
} from "../types.js";

const hasThinkingData = (block: ThinkingBlock) =>
  block.signature !== undefined || block.redacted !== undefined || block.details !== undefined;

export const appendAssistantContentBlock = (
  content: AssistantContentBlock[],
  block: AssistantContentBlock,
) => {
  if (block.type === "text" && block.text.length === 0 && block.signature === undefined) return;
  if (block.type === "thinking" && block.thinking.length === 0 && !hasThinkingData(block)) return;

  const last = content.at(-1);
  // Signed text keeps its own block so the signature stays with the exact text it covers.
  if (block.type === "text" && last?.type === "text" && block.signature === undefined && last.signature === undefined) {
    last.text += block.text;
    return;
  }
  // Signed or encrypted reasoning must go back to the provider exactly as received.
  if (block.type === "thinking" && last?.type === "thinking" && !hasThinkingData(block) && !hasThinkingData(last)) {
    last.thinking += block.thinking;
    return;
  }
  content.push(block);
};

const cloneAssistantContentBlock = (block: AssistantContentBlock): AssistantContentBlock => ({ ...block });

export const assertOnlySupportedFiles = (
  parts: ContentPart[],
  supportImages: boolean,
  label: string,
) => {
  for (const part of parts) {
    if (typeof part === "string" || part.type === "text") continue;
    if (supportImages && part.mediaType.startsWith("image/")) continue;
    throw invalidRequestError(`${label} does not support input file type '${part.mediaType}'.`);
  }
};

export const buildAssistantMessageFromContent = (
  model: string,
  content: AssistantContentBlock[],
  usage?: AssistantMessage["usage"],
  stopReason?: AssistantMessage["stopReason"],
  provider?: string,
): AssistantMessage => {
  const clonedContent = content.map(cloneAssistantContentBlock);
  return { role: "assistant", content: clonedContent, model, ...(provider ? { provider } : {}), usage, stopReason };
};

export const extractAssistantText = (message: Message) => {
  if (message.role !== "assistant") return "";
  return message.content
    .filter((block): block is TextBlock => block.type === "text")
    .map((block) => block.text)
    .join("");
};
