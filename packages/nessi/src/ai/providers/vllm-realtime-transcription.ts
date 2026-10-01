import type {
  RealtimeTranscriptionProvider,
  RealtimeWebSocket,
  RealtimeWebSocketFactory,
} from "../transcription.js";

export type VllmRealtimeTranscriptionOptions = {
  /** HTTP(S) or WS(S) base URL including `/v1`; `/realtime` is appended. */
  baseURL: string;
  apiKey?: string;
  headers?: Record<string, string>;
  /** Opens the WebSocket. Defaults to the global `WebSocket`. */
  webSocket?: RealtimeWebSocketFactory;
  name?: string;
};

type ServerEvent =
  | { type: "session.created" }
  | { type: "transcription.delta"; delta?: string }
  | {
      type: "transcription.done";
      text?: string;
      usage?: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number };
    }
  | { type: "error"; error?: string; code?: string | null };

type Inbound =
  | { kind: "event"; event: ServerEvent }
  | { kind: "closed"; code: number; reason: string }
  | { kind: "failed"; error: unknown }
  | { kind: "aborted" };

type WebSocketConstructor = new (url: string, init?: { headers: Record<string, string> }) => RealtimeWebSocket;

const globalWebSocket: RealtimeWebSocketFactory = (url, headers) => {
  // Header options are a runtime extension (Bun, Node), so the global is typed locally.
  const WebSocketImpl = (globalThis as { WebSocket?: WebSocketConstructor }).WebSocket;
  if (!WebSocketImpl) throw new Error("No global WebSocket is available; pass the webSocket option.");
  return Object.keys(headers).length > 0 ? new WebSocketImpl(url, { headers }) : new WebSocketImpl(url);
};

const realtimeURL = (baseURL: string) => {
  const url = new URL(`${baseURL.replace(/\/+$/, "")}/realtime`);
  if (url.protocol === "http:") url.protocol = "ws:";
  if (url.protocol === "https:") url.protocol = "wss:";
  return url.toString();
};

const toBase64 = (bytes: Uint8Array) => {
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(binary);
};

// vLLM decodes realtime audio as 16 kHz PCM16; the protocol has no rate negotiation.
const SAMPLE_RATE = 16_000;

const isServerEvent = (value: unknown): value is ServerEvent =>
  typeof value === "object" && value !== null && "type" in value && typeof value.type === "string";

/**
 * Live transcription over vLLM's `/v1/realtime` WebSocket, e.g. with
 * `mistralai/Voxtral-Mini-4B-Realtime-2602`. Audio streams in while text streams out.
 */
export const vllmRealtimeTranscription = (
  model: string,
  options: VllmRealtimeTranscriptionOptions,
): RealtimeTranscriptionProvider => {
  const name = options.name ?? "vllm";
  const url = realtimeURL(options.baseURL);
  const openWebSocket = options.webSocket ?? globalWebSocket;
  const headers: Record<string, string> = { ...options.headers };
  if (options.apiKey) {
    for (const key of Object.keys(headers)) if (key.toLowerCase() === "authorization") delete headers[key];
    headers.Authorization = `Bearer ${options.apiKey}`;
  }

  return {
    name,
    model,
    sampleRate: SAMPLE_RATE,
    async *stream({ audio, signal }) {
      signal?.throwIfAborted();

      const inbox: Inbound[] = [];
      let wake: (() => void) | undefined;
      const push = (item: Inbound) => {
        inbox.push(item);
        wake?.();
        wake = undefined;
      };
      const nextInbound = async (): Promise<Inbound> => {
        while (inbox.length === 0) await new Promise<void>((resolve) => (wake = resolve));
        return inbox.shift()!;
      };

      const audioIterator = audio[Symbol.asyncIterator]();
      let socket: RealtimeWebSocket;
      try {
        socket = openWebSocket(url, headers);
      } catch (error) {
        throw new Error(`${name} realtime connection failed: ${error instanceof Error ? error.message : String(error)}`);
      }

      let socketError = "";
      socket.addEventListener("message", ({ data }) => {
        let event: unknown;
        try {
          event = JSON.parse(String(data));
        } catch {
          return;
        }
        if (isServerEvent(event)) push({ kind: "event", event });
      });
      socket.addEventListener("error", (event) => {
        // Some runtimes (e.g. Bun) explain handshake failures such as a rejected API key here.
        if (typeof event === "object" && event !== null && "message" in event && typeof event.message === "string") {
          socketError = event.message;
        }
      });
      socket.addEventListener("close", ({ code, reason }) => push({ kind: "closed", code, reason }));

      let sessionStarted = false;
      let stopped = false;
      let audioBytes = 0;
      // Stop sending and close right away, even while the consumer is still handling an event.
      const stop = () => {
        if (stopped) return;
        stopped = true;
        socket.close();
        void Promise.resolve(audioIterator.return?.()).catch(() => {});
      };
      const send = (event: Record<string, unknown>) => {
        if (!stopped) socket.send(JSON.stringify(event));
      };
      const pumpAudio = async () => {
        send({ type: "session.update", model });
        send({ type: "input_audio_buffer.commit" });
        while (!stopped) {
          const next = await audioIterator.next();
          if (next.done || stopped) break;
          const chunk = next.value;
          const bytes = chunk instanceof Int16Array
            ? new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength)
            : chunk;
          if (bytes.byteLength % 2 !== 0) throw new Error("Realtime audio chunks must contain whole 16-bit samples.");
          // vLLM rejects empty appends.
          if (bytes.byteLength === 0) continue;
          audioBytes += bytes.byteLength;
          send({ type: "input_audio_buffer.append", audio: toBase64(bytes) });
        }
        if (!stopped) send({ type: "input_audio_buffer.commit", final: true });
      };

      const onAbort = () => {
        stop();
        push({ kind: "aborted" });
      };
      signal?.addEventListener("abort", onAbort, { once: true });

      try {
        while (true) {
          const inbound = await nextInbound();
          if (inbound.kind === "aborted") signal?.throwIfAborted();
          if (inbound.kind === "failed") throw inbound.error;
          if (inbound.kind === "closed") {
            const detail = socketError || inbound.reason || `code ${inbound.code}`;
            throw new Error(sessionStarted
              ? `${name} realtime connection closed before the transcript was done: ${detail}`
              : `${name} realtime connection failed: ${detail}`);
          }
          if (inbound.kind !== "event") continue;

          const event = inbound.event;
          if (event.type === "session.created") {
            sessionStarted = true;
            void pumpAudio().catch((error: unknown) => push({ kind: "failed", error }));
          } else if (event.type === "transcription.delta") {
            if (event.delta) yield { type: "delta", text: event.delta };
          } else if (event.type === "transcription.done") {
            const usage = event.usage;
            yield {
              type: "done",
              text: event.text ?? "",
              audioMs: Math.round((audioBytes / 2 / SAMPLE_RATE) * 1000),
              ...(usage
                ? {
                    usage: {
                      input: usage.prompt_tokens ?? 0,
                      output: usage.completion_tokens ?? 0,
                      total: usage.total_tokens ?? (usage.prompt_tokens ?? 0) + (usage.completion_tokens ?? 0),
                    },
                  }
                : {}),
            };
            return;
          } else if (event.type === "error") {
            const code = event.code ? ` (${event.code})` : "";
            throw new Error(`${name} realtime error${code}: ${event.error ?? "unknown error"}`);
          }
        }
      } finally {
        signal?.removeEventListener("abort", onAbort);
        stop();
      }
    },
  };
};
