import { afterEach, describe, expect, it } from "bun:test";
import {
  anthropic,
  cloudflareDecision,
  gemini,
  mistral,
  ollama,
  openAICompatible,
  openAICompatibleTranscription,
  systemOneDecision,
} from "../../../src/ai/index.js";
import type { Provider, StreamEvent } from "../../../src/ai/types.js";
import { jsonResponse, stubFetch } from "../helpers/fixtures.js";

const SECRET = "synthetic-credential";

const servers: ReturnType<typeof Bun.serve>[] = [];
const originalFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = originalFetch;
  for (const server of servers.splice(0)) server.stop(true);
});

/** One origin that redirects every request to a second origin, which records what reached it. */
const redirectingEndpoint = () => {
  const leaked: Array<Record<string, string>> = [];
  const target = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    fetch(request) {
      leaked.push(Object.fromEntries(request.headers));
      return Response.json({});
    },
  });
  // A different host name is a different origin, also on loopback.
  const origin = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    fetch: (request) => new Response(null, {
      status: 307,
      headers: { location: `http://localhost:${target.port}${new URL(request.url).pathname}` },
    }),
  });
  servers.push(target, origin);
  return { baseURL: `http://127.0.0.1:${origin.port}`, leaked };
};

const providers: Array<{ name: string; create: (baseURL: string) => Provider }> = [
  { name: "anthropic", create: (baseURL) => anthropic("test", { apiKey: SECRET, baseURL }) },
  { name: "mistral", create: (baseURL) => mistral("test", { apiKey: SECRET, baseURL }) },
  { name: "gemini", create: (baseURL) => gemini("test", { apiKey: SECRET, baseURL }) },
  { name: "ollama", create: (baseURL) => ollama("test", { baseURL }) },
  {
    name: "openai-compatible with custom headers",
    create: (baseURL) => openAICompatible({ name: "test", model: "test", baseURL, apiKey: SECRET, headers: { "x-api-key": SECRET } }),
  },
];

const request = { messages: [{ role: "user" as const, content: [{ type: "text" as const, text: "Hi" }] }] };

describe("provider redirects", () => {
  for (const { name, create } of providers) {
    it(`${name} completion refuses a redirect instead of resending credentials`, async () => {
      const { baseURL, leaked } = redirectingEndpoint();
      await expect(create(baseURL).complete(request)).rejects.toThrow();
      expect(leaked).toEqual([]);
    });

    it(`${name} stream reports a redirect as a provider error instead of resending credentials`, async () => {
      const { baseURL, leaked } = redirectingEndpoint();
      const events: StreamEvent[] = [];
      for await (const event of create(baseURL).stream(request)) events.push(event);
      expect(events.some((event) => event.type === "issue" && event.issue.kind === "provider_error")).toBe(true);
      expect(leaked).toEqual([]);
    });
  }

  it("transcription refuses a redirect instead of resending credentials", async () => {
    const { baseURL, leaked } = redirectingEndpoint();
    const provider = openAICompatibleTranscription("whisper", { baseURL, apiKey: SECRET, headers: { "x-api-key": SECRET } });
    await expect(provider.transcribe({ file: new Blob(["audio"], { type: "audio/wav" }) })).rejects.toThrow();
    expect(leaked).toEqual([]);
  });

  it("decision providers refuse a redirect instead of resending credentials", async () => {
    const questions = { urgent: { type: "noul", instructions: "Is this urgent?" } } as const;
    const systemOne = redirectingEndpoint();
    const cloudflare = redirectingEndpoint();
    await expect(systemOneDecision("clef", { baseURL: systemOne.baseURL, apiKey: SECRET }).decide({ state: "x", questions }))
      .rejects.toThrow();
    await expect(cloudflareDecision("clef", { baseURL: cloudflare.baseURL, accountId: "a", apiToken: SECRET }).decide({ state: "x", questions }))
      .rejects.toThrow();
    expect([...systemOne.leaked, ...cloudflare.leaked]).toEqual([]);
  });
});

describe("gemini API key", () => {
  it("is sent in the x-goog-api-key header and never in the URL", async () => {
    const seen: Array<{ url: string; key: string | null }> = [];
    stubFetch((input, init) => {
      seen.push({ url: String(input), key: new Headers(init?.headers).get("x-goog-api-key") });
      return String(input).includes("alt=sse")
        ? new Response('data: {"candidates":[{"content":{"parts":[{"text":"Hi"}]},"finishReason":"STOP"}]}\n\n', {
          headers: { "Content-Type": "text/event-stream" },
        })
        : jsonResponse({ candidates: [{ content: { parts: [{ text: "Hi" }] }, finishReason: "STOP" }] });
    });
    const provider = gemini("test", { apiKey: SECRET, baseURL: "https://example.test" });
    await provider.complete(request);
    for await (const _event of provider.stream(request)) { /* drain */ }

    expect(seen).toEqual([
      { url: "https://example.test/test:generateContent", key: SECRET },
      { url: "https://example.test/test:streamGenerateContent?alt=sse", key: SECRET },
    ]);
  });
});
