import type {
  AssistantMessage,
  AssistantContentBlock,
  ContentPart,
  InputFilePart,
  Message,
  TextBlock,
  ThinkingBlock,
  ToolCallBlock,
} from "../types.js";

export const textPart = (text: string) => ({ type: "text" as const, text });

export const textBlock = (text: string): TextBlock => ({ type: "text", text });

export const thinkingBlock = (thinking: string): ThinkingBlock => ({ type: "thinking", thinking });

export const toolCallBlock = (id: string, name: string, args: Record<string, unknown>): ToolCallBlock =>
  ({ type: "tool_call", id, name, args });

const hasThinkingData = (block: ThinkingBlock) => block.signature !== undefined || block.redacted !== undefined;

export const appendAssistantContentBlock = (
  content: AssistantContentBlock[],
  block: AssistantContentBlock,
) => {
  if (block.type === "text" && block.text.length === 0 && block.signature === undefined) return;
  if (block.type === "thinking" && block.thinking.length === 0 && !hasThinkingData(block)) return;

  const last = content.at(-1);
  if (block.type === "text" && last?.type === "text") {
    last.text += block.text;
    if (block.signature !== undefined) last.signature = block.signature;
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

export const contentPartToText = (part: ContentPart) => {
  if (typeof part === "string") return part;
  if (part.type === "text") return part.text;
  return "";
};

export const isImageFilePart = (part: ContentPart): part is InputFilePart =>
  typeof part !== "string" && part.type === "file" && part.mediaType.startsWith("image/");

export const assertOnlySupportedFiles = (
  parts: ContentPart[],
  supportImages: boolean,
  label: string,
) => {
  for (const part of parts) {
    if (typeof part === "string" || part.type === "text") continue;
    if (supportImages && part.mediaType.startsWith("image/")) continue;
    throw new Error(`${label} does not support input file type '${part.mediaType}'.`);
  }
};

export const buildAssistantMessage = (
  model: string,
  text: string,
  thinking: string,
  toolCalls: ToolCallBlock[],
  usage?: AssistantMessage["usage"],
  stopReason?: AssistantMessage["stopReason"],
  provider?: string,
): AssistantMessage => {
  const content = [
    ...(thinking ? [thinkingBlock(thinking)] : []),
    ...(text ? [textBlock(text)] : []),
    ...toolCalls,
  ];
  return buildAssistantMessageFromContent(model, content, usage, stopReason, provider);
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
