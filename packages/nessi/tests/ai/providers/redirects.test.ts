import { afterEach, describe, expect, it } from "bun:test";
import { cloudflareDecision, complete, openAICompatibleTranscription, systemOneDecision } from "../../../src/ai/index.js";
import { endpoint, ENDPOINT_NAMES } from "../helpers/endpoints.js";
import type { Provider, StreamEvent } from "../../../src/ai/types.js";

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
  // Another port is another origin.
  const origin = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    fetch: (request) => new Response(null, {
      status: 307,
      headers: { location: `http://127.0.0.1:${target.port}${new URL(request.url).pathname}` },
    }),
  });
  servers.push(target, origin);
  return { baseURL: `http://127.0.0.1:${origin.port}`, leaked };
};

it("the redirect harness forwards custom headers when fetch follows redirects", async () => {
  const { baseURL, leaked } = redirectingEndpoint();
  await fetch(`${baseURL}/v1/messages`, { method: "POST", headers: { "x-api-key": SECRET }, body: "{}" });
  expect(leaked).toEqual([expect.objectContaining({ "x-api-key": SECRET })]);
});

// Every endpoint variant, each also with a custom credential header that `fetch` would forward.
const providers: Array<{ name: string; create: (baseURL: string) => Provider }> = ENDPOINT_NAMES.map((name) => ({
  name: `openai-compatible (${name})`,
  create: (baseURL: string) => endpoint(name, { baseURL, apiKey: SECRET, headers: { "x-api-key": SECRET } }),
}));

const request = { messages: [{ role: "user" as const, content: [{ type: "text" as const, text: "Hi" }] }] };

describe("provider redirects", () => {
  for (const { name, create } of providers) {
    it(`${name} completion refuses a redirect instead of resending credentials`, async () => {
      const { baseURL, leaked } = redirectingEndpoint();
      await expect(complete(create(baseURL), request)).rejects.toThrow();
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
