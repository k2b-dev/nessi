import { afterEach, describe, expect, it } from "bun:test";
import {
  cloudflareDecision,
  complete,
  openAICompatible,
  openAICompatibleTranscription,
  systemOneDecision,
  vllmRealtimeTranscription,
} from "../../../src/ai/index.js";
import { stubFetch } from "../helpers/fixtures.js";

const originalFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = originalFetch;
});

/** A key with a line break inside, such as two lines pasted from a secret file; the canary must never appear in an error. */
const BROKEN_KEY = "KEY-CANARY-1234\nsecond-line";
const BASE = "https://example.test/v1";
const question = { q: { type: "noul" as const, instructions: "?" } };
const audio = new Blob([new Uint8Array(8)], { type: "audio/wav" });

const failure = async (run: () => unknown) => {
  try {
    await run();
  } catch (error) {
    return error instanceof Error ? error.message : String(error);
  }
  throw new Error("expected a failure");
};

/** Every provider path that builds request headers from an API key or custom headers. */
const paths: Array<[string, (key: string, headers?: Record<string, string>) => unknown]> = [
  ["chat", (key, headers) => complete(openAICompatible({ name: "chat", model: "m", baseURL: BASE, apiKey: key, headers }), { messages: [] })],
  ["systemOne decision", (key, headers) => systemOneDecision("clef", { baseURL: BASE, apiKey: key, headers }).decide({ state: "x", questions: question })],
  ["Cloudflare decision", (key) => cloudflareDecision("clef", { accountId: "acct", apiToken: key }).decide({ state: "x", questions: question })],
  ["transcription", (key, headers) => openAICompatibleTranscription("whisper", { baseURL: BASE, apiKey: key, headers }).transcribe({ file: audio })],
];

describe("invalid header values", () => {
  for (const [label, run] of paths) {
    it(`${label}: rejects a broken API key before sending, without repeating it`, async () => {
      let sent = 0;
      stubFetch(() => {
        sent++;
        return new Response("{}");
      });
      const message = await failure(() => run(BROKEN_KEY));
      expect(message).not.toContain("KEY-CANARY");
      expect(message).toContain('"Authorization" header');
      expect(sent).toBe(0);
    });

    it(`${label}: never passes a runtime header message through`, async () => {
      // A runtime that rejects a header itself repeats the value in its message.
      stubFetch(() => {
        throw new TypeError("Header 'Authorization' has invalid value: 'Bearer KEY-CANARY-1234'");
      });
      const message = await failure(() => run("valid-key"));
      expect(message).not.toContain("KEY-CANARY");
    });
  }

  for (const [label, run] of paths.filter(([name]) => name !== "Cloudflare decision")) {
    it(`${label}: rejects a broken custom header without repeating it`, async () => {
      stubFetch(() => new Response("{}"));
      const message = await failure(() => run("valid-key", { "X-Tenant": "TENANT-CANARY\r\nX-Evil: 1" }));
      expect(message).not.toContain("TENANT-CANARY");
      expect(message).toContain('"X-Tenant" header');
    });
  }

  it("realtime transcription: rejects a broken key before opening the socket, without repeating it", async () => {
    let opened = 0;
    const provider = vllmRealtimeTranscription("voxtral-realtime", {
      baseURL: BASE,
      apiKey: BROKEN_KEY,
      webSocket: () => {
        opened++;
        throw new TypeError("unreachable");
      },
    });
    const message = await failure(async () => {
      for await (const _event of provider.stream({ audio: (async function* () {})() })) {
        // The stream must fail before any event.
      }
    });
    expect(message).not.toContain("KEY-CANARY");
    expect(opened).toBe(0);
  });

  it("realtime transcription: never passes a runtime header message through", async () => {
    const provider = vllmRealtimeTranscription("voxtral-realtime", {
      baseURL: BASE,
      apiKey: "valid-key",
      webSocket: () => {
        throw new TypeError("Header 'Authorization' has invalid value: 'Bearer KEY-CANARY-1234'");
      },
    });
    const message = await failure(async () => {
      for await (const _event of provider.stream({ audio: (async function* () {})() })) {
        // The stream must fail before any event.
      }
    });
    expect(message).not.toContain("KEY-CANARY");
  });

  it("transcription: checks a custom Authorization header that the API key replaces", async () => {
    const message = await failure(() =>
      openAICompatibleTranscription("whisper", { baseURL: BASE, apiKey: "valid-key", headers: { Authorization: "AUTH-CANARY\u0100" } }));
    expect(message).not.toContain("AUTH-CANARY");
  });

  it("realtime transcription: a socket error event never passes a header message through", async () => {
    const listeners: Record<string, (event: unknown) => void> = {};
    const provider = vllmRealtimeTranscription("voxtral-realtime", {
      baseURL: BASE,
      apiKey: "valid-key",
      webSocket: () => {
        const socket = {
          readyState: 0,
          send: () => {},
          close: () => {},
          addEventListener: (type: string, listener: (event: unknown) => void) => {
            listeners[type] = listener;
          },
        };
        queueMicrotask(() => {
          listeners.error?.({ message: "Header 'Authorization' has invalid value: 'Bearer KEY-CANARY-1234'" });
          listeners.close?.({ code: 1006, reason: "" });
        });
        return socket as never;
      },
    });
    const message = await failure(async () => {
      for await (const _event of provider.stream({ audio: (async function* () {})() })) {
        // The stream must fail before any event.
      }
    });
    expect(message).not.toContain("KEY-CANARY");
  });

  it("accepts what fetch accepts: surrounding whitespace is trimmed, Latin-1 is allowed", async () => {
    let sent = 0;
    stubFetch(() => {
      sent++;
      return Response.json({ answers: { q: { type: "noul", noul: 0.9 } } });
    });
    // A key read from a file with its trailing line break worked before and keeps working.
    for (const apiKey of ["sk-trailing-newline\n", " sk-padded\t", "sk-latin\u00ff"]) {
      await systemOneDecision("clef", { baseURL: BASE, apiKey }).decide({ state: "x", questions: question });
    }
    expect(sent).toBe(3);
    expect(await failure(() => systemOneDecision("clef", { baseURL: BASE, apiKey: "sk-\u0100" }).decide({ state: "x", questions: question })))
      .toContain('"Authorization" header');
  });

  it("accepts ordinary keys and header values, tabs included", async () => {
    let sent = 0;
    stubFetch(() => {
      sent++;
      return Response.json({ answers: { q: { type: "noul", noul: 0.9 } } });
    });
    await systemOneDecision("clef", { baseURL: BASE, apiKey: "sk-abc_123.XYZ", headers: { "X-Note": "a\tb ü" } })
      .decide({ state: "x", questions: question });
    expect(sent).toBe(1);
  });
});
