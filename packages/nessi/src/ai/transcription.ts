import type { Usage } from "./types.js";

export type TranscriptionRequest = {
  file: Blob;
  /** Override the file name, or supply one for an unnamed Blob. */
  filename?: string;
  /** ISO-639-1 language code, for example "de". Omit for automatic detection. */
  language?: string;
  /** Optional vocabulary or context hint, subject to model support. */
  prompt?: string;
  signal?: AbortSignal;
};

export type TranscriptionResult = {
  text: string;
};

export type TranscriptionProvider = {
  name: string;
  model: string;
  transcribe(request: TranscriptionRequest): Promise<TranscriptionResult>;
};

export type RealtimeTranscriptionRequest = {
  /** Mono 16-bit little-endian PCM at the provider's `sampleRate`, sent as it arrives. */
  audio: AsyncIterable<Int16Array | Uint8Array>;
  signal?: AbortSignal;
};

export type RealtimeTranscriptionEvent =
  /** Newly recognized text; append it to the previous deltas. */
  | { type: "delta"; text: string }
  /** Final transcript after the audio ended. `audioMs` is the duration of the sent audio. */
  | { type: "done"; text: string; audioMs: number; usage?: Usage };

export type RealtimeTranscriptionProvider = {
  name: string;
  model: string;
  /** Required input sample rate in Hz. */
  sampleRate: number;
  stream(request: RealtimeTranscriptionRequest): AsyncIterable<RealtimeTranscriptionEvent>;
};

/** The subset of the standard WebSocket API used by realtime providers. */
export type RealtimeWebSocket = {
  send(data: string): void;
  close(code?: number, reason?: string): void;
  addEventListener(type: "message", listener: (event: { data: unknown }) => void): void;
  addEventListener(type: "close", listener: (event: { code: number; reason: string }) => void): void;
  addEventListener(type: "error", listener: (event: unknown) => void): void;
};

/**
 * Opens a WebSocket. The default uses the global `WebSocket` and passes `headers` only when
 * an API key or headers are configured, which requires a runtime that accepts them (e.g. Bun).
 */
export type RealtimeWebSocketFactory = (url: string, headers: Record<string, string>) => RealtimeWebSocket;
