import { afterEach, describe, expect, it } from "bun:test";
import { isBrowserRuntime, openSSEStream } from "../../../src/ai/shared/stream-helpers.js";
import { stubFetch } from "../helpers/fixtures.js";

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe("openSSEStream", () => {
  const largeBody = { input: "x".repeat(4_000) };

  it("does not classify a user abort of a large request as context overflow", async () => {
    stubFetch(async () => {
      throw new DOMException("The operation was aborted.", "AbortError");
    });
    const controller = new AbortController();
    controller.abort();

    const result = await openSSEStream("https://example.com", {}, largeBody, "custom", controller.signal, 1_000);

    expect(result.ok).toBe(false);
    expect(!result.ok && result.error.type === "error" ? result.error.contextOverflow : "missing").toBeUndefined();
  });

  it("keeps the network-error overflow guess to browsers", async () => {
    stubFetch(async () => {
      throw new TypeError("Failed to fetch");
    });

    // Bun reports refused connections as TypeError too; there it stays a retryable connection error.
    const result = await openSSEStream("https://example.com", {}, largeBody, "custom", undefined, 1_000);

    expect(!result.ok && result.error.type === "error" ? result.error.retryable : undefined).toBe(true);
    expect(!result.ok && result.error.type === "error" ? result.error.contextOverflow : "missing").toBeUndefined();
  });

  it("recognizes browser and server runtimes", () => {
    expect(isBrowserRuntime({})).toBe(true);
    expect(isBrowserRuntime({ process: { versions: { node: "24.0.0" } } })).toBe(false);
    expect(isBrowserRuntime({ Bun: {} })).toBe(false);
    expect(isBrowserRuntime({ Deno: {} })).toBe(false);
    expect(isBrowserRuntime()).toBe(false);
  });

  it("does not classify other connection failures as context overflow", async () => {
    stubFetch(async () => {
      throw new Error("socket hang up");
    });

    const result = await openSSEStream("https://example.com", {}, largeBody, "custom", undefined, 1_000);

    expect(!result.ok && result.error.type === "error" ? result.error.retryable : undefined).toBe(true);
    expect(!result.ok && result.error.type === "error" ? result.error.contextOverflow : "missing").toBeUndefined();
  });
});
