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
    const controller = new AbortController();
    // The caller aborts while the request is in flight.
    stubFetch(async () => {
      controller.abort();
      throw new DOMException("The operation was aborted.", "AbortError");
    });

    const result = await openSSEStream("https://example.com", {}, largeBody, "custom", controller.signal, 1_000);

    expect(result.ok).toBe(false);
    expect(!result.ok && result.error.type === "error" ? result.error.contextOverflow : "missing").toBeUndefined();
  });

  it("sends nothing for a signal that is already aborted", async () => {
    let sent = false;
    stubFetch(async () => { sent = true; return new Response(""); });
    const reason = new Error("caller stopped");
    await expect(openSSEStream("https://example.com", {}, largeBody, "custom", AbortSignal.abort(reason), 1_000)).rejects.toBe(reason);
    expect(sent).toBe(false);
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
    const browser = { navigator: { userAgent: "Mozilla/5.0 (X11; Linux x86_64) Chrome/140" } };
    expect(isBrowserRuntime(browser)).toBe(true);
    expect(isBrowserRuntime({})).toBe(false);
    expect(isBrowserRuntime({ navigator: { userAgent: "Cloudflare-Workers" }, process: {} })).toBe(false);
    expect(isBrowserRuntime({ ...browser, process: { versions: { node: "24.0.0" } } })).toBe(false);
    expect(isBrowserRuntime({ ...browser, Bun: {} })).toBe(false);
    expect(isBrowserRuntime({ ...browser, Deno: {} })).toBe(false);
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
