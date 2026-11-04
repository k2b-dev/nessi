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

/**
 * Largest single stream record nessi buffers. Provider records carry deltas or one chunk of an
 * answer, far below this; the budget only stops a broken or hostile stream from growing memory.
 */
export const MAX_STREAM_RECORD_CHARS = 8 * 1024 * 1024;

export class StreamRecordTooLargeError extends Error {
  constructor() {
    super(`Stream record exceeds ${MAX_STREAM_RECORD_CHARS} characters.`);
    this.name = "StreamRecordTooLargeError";
  }
}

const readWithTimeout = async (
  reader: ByteStreamReader,
  scope: SSETimeoutScope,
  timeoutMs: number | undefined,
) => {
  if (!timeoutMs || timeoutMs <= 0) return reader.read();
  let timeout: ReturnType<typeof setTimeout> | undefined;
  let timedOut: SSETimeoutError | undefined;
  try {
    const result = await Promise.race([
      reader.read(),
      new Promise<never>((_, reject) => {
        timeout = setTimeout(() => {
          timedOut = new SSETimeoutError(scope, timeoutMs);
          // Reject before cancelling: cancelling resolves the pending read as a clean end.
          reject(timedOut);
          void reader.cancel?.(timedOut).catch(() => {});
        }, timeoutMs);
      }),
    ]);
    if (timedOut) throw timedOut;
    return result;
  } finally {
    if (timeout) clearTimeout(timeout);
  }
};

/**
 * Splits a byte stream into records at `separator`. Each chunk is scanned once and kept as a
 * fragment until its record ends, so fragmented large records cost linear time. CRLF line
 * endings become LF first. The first read uses `firstByteMs`, later reads `idleMs`. Yields the
 * records completed by each read as one batch.
 */
export const readRecordBatches = async function* (
  reader: ByteStreamReader,
  separator: string,
  timeouts: SSETimeouts = {},
): AsyncGenerator<string[]> {
  const decoder = new TextDecoder();
  const tailLength = separator.length - 1;
  let fragments: string[] = [];
  let pendingLength = 0;
  // The last characters of the pending record, where a separator split across chunks begins.
  let tail = "";
  let carriedCR = false;
  let readCount = 0;

  const normalize = (text: string) => {
    if (carriedCR) text = `\r${text}`;
    carriedCR = text.endsWith("\r");
    if (carriedCR) text = text.slice(0, -1);
    return text.replace(/\r\n/g, "\n");
  };

  const split = (text: string): string[] => {
    const scan = tail + text;
    let start = 0;
    let end = scan.indexOf(separator);
    if (end < 0) {
      fragments.push(text);
      pendingLength += text.length;
      tail = tailLength > 0 ? scan.slice(-tailLength) : "";
      return [];
    }
    // The first record ends here; its start is in the fragments, of which `tail` is the end.
    const records = [fragments.join("").slice(0, pendingLength - tail.length) + scan.slice(0, end)];
    start = end + separator.length;
    while ((end = scan.indexOf(separator, start)) >= 0) {
      records.push(scan.slice(start, end));
      start = end + separator.length;
    }
    const rest = scan.slice(start);
    fragments = rest ? [rest] : [];
    pendingLength = rest.length;
    tail = tailLength > 0 ? rest.slice(-tailLength) : "";
    return records;
  };

  while (true) {
    const scope = readCount === 0 ? "provider_first_byte" : "provider_idle";
    const timeoutMs = readCount === 0 ? timeouts.firstByteMs : timeouts.idleMs;
    const { done, value } = await readWithTimeout(reader, scope, timeoutMs);
    readCount++;
    if (done) break;
    const records = split(normalize(decoder.decode(value, { stream: true })));
    if (records.length > 0) yield records;
    if (pendingLength > MAX_STREAM_RECORD_CHARS) throw new StreamRecordTooLargeError();
  }

  const records = split(normalize(decoder.decode()));
  const rest = fragments.join("") + (carriedCR ? "\r" : "");
  if (rest) records.push(rest);
  if (records.length > 0) yield records;
};

export const parseSSE = async function* (
  reader: ByteStreamReader,
  timeouts: SSETimeouts = {},
): AsyncGenerator<SSEEvent> {
  for await (const frames of readRecordBatches(reader, "\n\n", timeouts)) {
    for (const frame of frames) {
      const parsed = parseFrame(frame);
      if (parsed) yield parsed;
    }
  }
};
