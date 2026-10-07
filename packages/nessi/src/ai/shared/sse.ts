export type SSEEvent = {
  event?: string;
  data: string;
  id?: string;
};

/** The part of a stream reader the parsers use; any WHATWG or runtime-specific reader fits. */
export type ByteStreamReader = {
  read(): Promise<{ done: boolean; value?: Uint8Array }>;
  cancel?(reason?: unknown): Promise<void>;
};

export type SSETimeoutScope = "provider_first_byte" | "provider_idle";

export class SSETimeoutError extends Error {
  readonly scope: SSETimeoutScope;

  constructor(scope: SSETimeoutScope, timeoutMs: number) {
    super(`SSE stream ${scope === "provider_first_byte" ? "first byte" : "idle"} timeout after ${timeoutMs}ms.`);
    this.name = "SSETimeoutError";
    this.scope = scope;
  }
}

export type SSETimeouts = {
  firstByteMs?: number;
  idleMs?: number;
};

const parseFrame = (frame: string): SSEEvent | null => {
  const lines = frame.split(/\r?\n/);
  const dataLines: string[] = [];
  let event: string | undefined;
  let id: string | undefined;

  for (const rawLine of lines) {
    if (!rawLine || rawLine.startsWith(":")) continue;
    const idx = rawLine.indexOf(":");
    const field = idx === -1 ? rawLine : rawLine.slice(0, idx);
    let value = idx === -1 ? "" : rawLine.slice(idx + 1);
    if (value.startsWith(" ")) value = value.slice(1);

    if (field === "event") event = value;
    else if (field === "data") dataLines.push(value);
    else if (field === "id") id = value;
  }

  if (dataLines.length === 0) return null;
  return { event, data: dataLines.join("\n"), id };
};

const readWithTimeout = async (
  reader: ByteStreamReader,
  scope: SSETimeoutScope,
  timeoutMs: number | undefined,
) => {
  if (!timeoutMs || timeoutMs <= 0) return reader.read();
  let timeout: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      reader.read(),
      new Promise<never>((_, reject) => {
        timeout = setTimeout(() => {
          const error = new SSETimeoutError(scope, timeoutMs);
          void reader.cancel?.(error).catch(() => {});
          reject(error);
        }, timeoutMs);
      }),
    ]);
  } finally {
    if (timeout) clearTimeout(timeout);
  }
}

export const parseSSE = async function* (
  reader: ByteStreamReader,
  timeouts: SSETimeouts = {},
): AsyncGenerator<SSEEvent> {
  const decoder = new TextDecoder();
  let buffer = "";
  let readCount = 0;

  while (true) {
    const scope = readCount === 0 ? "provider_first_byte" : "provider_idle";
    const timeoutMs = readCount === 0 ? timeouts.firstByteMs : timeouts.idleMs;
    const { done, value } = await readWithTimeout(reader, scope, timeoutMs);
    readCount++;
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const normalized = buffer.replace(/\r\n/g, "\n");
    const frames = normalized.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      const parsed = parseFrame(frame);
      if (parsed) yield parsed;
    }
  }

  const finalFrame = buffer.replace(/\r\n/g, "\n");
  const parsed = parseFrame(finalFrame);
  if (parsed) yield parsed;
};
