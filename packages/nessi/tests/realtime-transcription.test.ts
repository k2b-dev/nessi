import { afterEach, describe, expect, it } from "bun:test";
import type { ServerWebSocket } from "bun";
import { vllmRealtimeTranscription, type RealtimeTranscriptionEvent } from "../src/ai/index.js";

type ClientEvent = { type: string; model?: string; audio?: string; final?: boolean };
type Session = {
  headers: Headers;
  path: string;
  events: ClientEvent[];
  closed: Promise<void>;
  markClosed: () => void;
};

const servers: ReturnType<typeof Bun.serve>[] = [];
afterEach(() => {
  for (const server of servers.splice(0)) server.stop(true);
});

/** Minimal vLLM /v1/realtime server. `onEvent` scripts the replies per client event. */
const realtimeServer = (onEvent: (ws: ServerWebSocket<Session>, event: ClientEvent) => void) => {
  const sessions: Session[] = [];
  const server = Bun.serve<Session>({
    hostname: "127.0.0.1",
    port: 0,
    fetch(request, server) {
      const closed = Promise.withResolvers<void>();
      const session: Session = {
        headers: request.headers,
        path: new URL(request.url).pathname,
        events: [],
        closed: closed.promise,
        markClosed: closed.resolve,
      };
      sessions.push(session);
      return server.upgrade(request, { data: session }) ? undefined : new Response("upgrade failed", { status: 400 });
    },
    websocket: {
      open(ws) {
        ws.send(JSON.stringify({ type: "session.created" }));
      },
      message(ws, message) {
        const event = JSON.parse(String(message)) as ClientEvent;
        ws.data.events.push(event);
        onEvent(ws, event);
      },
      close(ws) {
        ws.data.markClosed();
      },
    },
  });
  servers.push(server);
  return { baseURL: `http://127.0.0.1:${server.port}/v1/`, sessions };
};

/** Echo server: one delta per append, done after the final commit. */
const echoServer = () =>
  realtimeServer((ws, event) => {
    if (event.type === "input_audio_buffer.append") {
      ws.send(JSON.stringify({ type: "transcription.delta", delta: `[${atob(event.audio ?? "").length}]` }));
    }
    if (event.type === "input_audio_buffer.commit" && event.final) {
      ws.send(JSON.stringify({
        type: "transcription.done",
        text: "Hallo Welt",
        usage: { prompt_tokens: 3, completion_tokens: 4, total_tokens: 7 },
      }));
    }
  });

async function* chunks(...parts: Array<Int16Array | Uint8Array>) {
  for (const part of parts) yield part;
}

const collect = async (events: AsyncIterable<RealtimeTranscriptionEvent>) => {
  const out: RealtimeTranscriptionEvent[] = [];
  for await (const event of events) out.push(event);
  return out;
};

describe("vLLM realtime transcription", () => {
  it("streams PCM16 audio and yields deltas and the final transcript", async () => {
    const { baseURL, sessions } = echoServer();
    const provider = vllmRealtimeTranscription("voxtral-realtime", { baseURL, apiKey: "test-key" });

    const events = await collect(provider.stream({
      audio: chunks(new Int16Array(16_000), new Uint8Array(0), new Uint8Array(3_200)),
    }));

    expect(provider.sampleRate).toBe(16_000);
    expect(events).toEqual([
      { type: "delta", text: "[32000]" },
      { type: "delta", text: "[3200]" },
      { type: "done", text: "Hallo Welt", audioMs: 1_100, usage: { input: 3, output: 4, total: 7 } },
    ]);
    const session = sessions[0]!;
    expect(session.path).toBe("/v1/realtime");
    expect(session.headers.get("authorization")).toBe("Bearer test-key");
    expect(session.events.map((event) => event.type)).toEqual([
      "session.update",
      "input_audio_buffer.commit",
      "input_audio_buffer.append",
      "input_audio_buffer.append",
      "input_audio_buffer.commit",
    ]);
    expect(session.events[0]).toEqual({ type: "session.update", model: "voxtral-realtime" });
    expect(session.events.at(-1)).toEqual({ type: "input_audio_buffer.commit", final: true });
    await session.closed;
  });

  it("uses a custom WebSocket factory with the realtime URL and headers", async () => {
    const { baseURL } = echoServer();
    let opened: { url: string; headers: Record<string, string> } | undefined;
    const provider = vllmRealtimeTranscription("voxtral-realtime", {
      baseURL,
      headers: { "X-Tenant": "test" },
      webSocket: (url, headers) => {
        opened = { url, headers };
        return new WebSocket(url);
      },
    });

    const events = await collect(provider.stream({ audio: chunks(new Int16Array(160)) }));

    expect(opened).toEqual({ url: baseURL.replace("http:", "ws:").replace(/\/$/, "") + "/realtime", headers: { "X-Tenant": "test" } });
    expect(events.at(-1)).toMatchObject({ type: "done", text: "Hallo Welt", audioMs: 10 });
  });

  it("rejects with the server error", async () => {
    const { baseURL } = realtimeServer((ws, event) => {
      if (event.type === "session.update") {
        ws.send(JSON.stringify({ type: "error", error: "The model `nope` does not exist.", code: "model_not_found" }));
      }
    });
    const provider = vllmRealtimeTranscription("nope", { baseURL });

    await expect(collect(provider.stream({ audio: chunks(new Int16Array(160)) })))
      .rejects.toThrow("vllm realtime error (model_not_found): The model `nope` does not exist.");
  });

  it("rejects when the connection closes before the transcript is done", async () => {
    const { baseURL } = realtimeServer((ws, event) => {
      if (event.type === "input_audio_buffer.append") ws.close(1011, "engine crashed");
    });
    const provider = vllmRealtimeTranscription("voxtral-realtime", { baseURL });

    await expect(collect(provider.stream({ audio: chunks(new Int16Array(160)) })))
      .rejects.toThrow("vllm realtime connection closed before the transcript was done: engine crashed");
  });

  it("reports a rejected handshake as a connection failure", async () => {
    const server = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch: () => new Response("unauthorized", { status: 401 }) });
    servers.push(server);
    const provider = vllmRealtimeTranscription("voxtral-realtime", { baseURL: `http://127.0.0.1:${server.port}/v1`, apiKey: "wrong" });

    await expect(collect(provider.stream({ audio: chunks(new Int16Array(160)) })))
      .rejects.toThrow("vllm realtime connection failed:");
  });

  it("rejects audio chunks with a partial sample", async () => {
    const { baseURL } = echoServer();
    const provider = vllmRealtimeTranscription("voxtral-realtime", { baseURL });

    await expect(collect(provider.stream({ audio: chunks(new Uint8Array(3)) })))
      .rejects.toThrow("whole 16-bit samples");
  });

  it("aborts with the signal reason and closes the connection", async () => {
    const { baseURL, sessions } = echoServer();
    const provider = vllmRealtimeTranscription("voxtral-realtime", { baseURL });
    const controller = new AbortController();
    async function* endless() {
      while (true) {
        yield new Int16Array(160);
        await Bun.sleep(5);
      }
    }

    const run = (async () => {
      for await (const event of provider.stream({ audio: endless(), signal: controller.signal })) {
        if (event.type === "delta") controller.abort(new Error("user stopped"));
      }
    })();

    await expect(run).rejects.toThrow("user stopped");
    await sessions[0]!.closed;
  });

  it("closes the connection on abort while the consumer is still handling an event", async () => {
    const { baseURL, sessions } = echoServer();
    const provider = vllmRealtimeTranscription("voxtral-realtime", { baseURL });
    const controller = new AbortController();
    async function* endless() {
      while (true) {
        yield new Int16Array(160);
        await Bun.sleep(5);
      }
    }

    const run = (async () => {
      for await (const event of provider.stream({ audio: endless(), signal: controller.signal })) {
        if (event.type !== "delta") continue;
        controller.abort(new Error("user stopped"));
        // The consumer is busy; the socket must still close without another next() call.
        await sessions[0]!.closed;
      }
    })();

    await expect(run).rejects.toThrow("user stopped");
  });

  it("closes the connection when the consumer stops early", async () => {
    const { baseURL, sessions } = echoServer();
    const provider = vllmRealtimeTranscription("voxtral-realtime", { baseURL });

    for await (const event of provider.stream({ audio: chunks(new Int16Array(160), new Int16Array(160)) })) {
      if (event.type === "delta") break;
    }

    await sessions[0]!.closed;
  });
});
