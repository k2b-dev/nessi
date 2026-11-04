import { describe, expect, it } from "bun:test";
import { nessi } from "../src/nessi.js";
import { memoryStore } from "../src/stores.js";
import type { Message } from "../src/types.js";
import { estimateTokens, FILE_PART_TOKEN_ESTIMATE } from "../src/utils.js";
import { mockProvider } from "./mock-provider.js";

// 1 MiB of base64: about 262k tokens if it were counted as text.
const image = { type: "file" as const, mediaType: "image/png", data: "A".repeat(1024 * 1024) };
const withImage: Message[] = [{ role: "user", content: [{ type: "text", text: "What is in this picture?" }, image] }];

describe("context estimate for files", () => {
  it("counts a file part as a fixed allowance, not by its base64 size", () => {
    const tokens = estimateTokens(withImage);
    expect(tokens).toBeGreaterThanOrEqual(FILE_PART_TOKEN_ESTIMATE);
    expect(tokens).toBeLessThan(FILE_PART_TOKEN_ESTIMATE + 100);
    expect(image.data).toHaveLength(1024 * 1024);
  });

  it("adds one allowance per file and keeps text estimates unchanged", () => {
    const text: Message[] = [{ role: "user", content: [{ type: "text", text: "x".repeat(4_000) }] }];
    expect(estimateTokens(text)).toBe(Math.ceil(JSON.stringify(text).length / 4));
    const twoImages: Message[] = [{ role: "user", content: [image, { ...image, data: "B".repeat(10) }] }];
    const emptyImages: Message[] = [{ role: "user", content: [{ ...image, data: "" }, { ...image, data: "" }] }];
    expect(estimateTokens(twoImages)).toBe(estimateTokens(emptyImages));
    expect(estimateTokens(twoImages)).toBeGreaterThanOrEqual(2 * FILE_PART_TOKEN_ESTIMATE);
  });

  it("counts file-shaped tool results as the text the provider receives", () => {
    const toolResult: Message[] = [{ role: "tool_result", callId: "c1", name: "fetch", result: { type: "file", data: "A".repeat(400_000) } }];
    expect(estimateTokens(toolResult)).toBeGreaterThan(100_000);
  });

  it("does not force compaction for one large image in a 128k context", async () => {
    let compactContext: { force: boolean; fillRatio?: number } | undefined;
    const provider = mockProvider(
      [{ type: "text", delta: "A cat." }, { type: "usage", usage: { input: 10, output: 2, total: 12 }, finishReason: "stop" }],
      { contextWindow: 128_000 },
    );
    const store = memoryStore();
    await store.append(withImage[0]!);
    for await (const _event of nessi({
      provider,
      store,
      systemPrompt: "test",
      compact: (context) => {
        compactContext = { force: context.force, fillRatio: context.fillRatio };
        return null;
      },
    })) { /* drain */ }
    expect(compactContext?.force).toBe(false);
    expect(compactContext?.fillRatio).toBeLessThan(0.1);
  });
});
