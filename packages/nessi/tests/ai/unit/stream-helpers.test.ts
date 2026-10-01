import { afterEach, describe, expect, it } from "bun:test";
import { openSSEStream } from "../../../src/ai/shared/stream-helpers.js";

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe("openSSEStream", () => {
  const largeBody = { input: "x".repeat(4_000) };

  it("does not classify a user abort of a large request as context overflow", async () => {
    globalThis.fetch = (async () => {
      throw new DOMException("The operation was aborted.", "AbortError");
    }) as unknown as typeof fetch;
    const controller = new AbortController();
    controller.abort();

    const result = await openSSEStream("https://example.com", {}, largeBody, "custom", controller.signal, 1_000);

    expect(result.ok).toBe(false);
    expect(!result.ok && result.error.type === "error" ? result.error.contextOverflow : "missing").toBeUndefined();
  });

  it("keeps the browser network-error heuristic for large requests", async () => {
    globalThis.fetch = (async () => {
      throw new TypeError("Failed to fetch");
    }) as unknown as typeof fetch;

    const result = await openSSEStream("https://example.com", {}, largeBody, "custom", undefined, 1_000);

    expect(!result.ok && result.error.type === "error" ? result.error.contextOverflow : undefined).toBe(true);
  });

  it("does not classify other connection failures as context overflow", async () => {
    globalThis.fetch = (async () => {
      throw new Error("socket hang up");
    }) as unknown as typeof fetch;

    const result = await openSSEStream("https://example.com", {}, largeBody, "custom", undefined, 1_000);

    expect(!result.ok && result.error.type === "error" ? result.error.retryable : undefined).toBe(true);
    expect(!result.ok && result.error.type === "error" ? result.error.contextOverflow : "missing").toBeUndefined();
  });
});
