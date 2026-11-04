import { parseJsonRecord } from "./json.js";
import type { ByteStreamReader } from "./sse.js";

export const parseNDJSON = async function* <T>(
  reader: ByteStreamReader,
  timeouts: { firstByteMs?: number; idleMs?: number } = {},
): AsyncGenerator<T> {
  const decoder = new TextDecoder();
  let buffer = "";
  let readCount = 0;

  const readWithTimeout = async () => {
    const scope = readCount === 0 ? "provider_first_byte" : "provider_idle";
    const timeoutMs = readCount === 0 ? timeouts.firstByteMs : timeouts.idleMs;
    if (!timeoutMs || timeoutMs <= 0) return reader.read();
    let timeout: ReturnType<typeof setTimeout> | undefined;
    try {
      return await Promise.race([
        reader.read(),
        new Promise<never>((_, reject) => {
          timeout = setTimeout(() => {
            const error = {
              scope,
              message: `NDJSON stream ${scope === "provider_first_byte" ? "first byte" : "idle"} timeout after ${timeoutMs}ms.`,
            };
            void reader.cancel?.(error).catch(() => {});
            reject(error);
          }, timeoutMs);
        }),
      ]);
    } finally {
      if (timeout) clearTimeout(timeout);
    }
  };

  while (true) {
    const { done, value } = await readWithTimeout();
    readCount++;
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed) yield parseRecord<T>(trimmed);
    }
  }

  const trimmed = buffer.trim();
  if (trimmed) yield parseRecord<T>(trimmed);
};

/** A line that is not a JSON object; content it carried would otherwise vanish. */
export class MalformedRecordError extends Error {
  constructor() {
    super("Received a malformed NDJSON record.");
    this.name = "MalformedRecordError";
  }
}

const parseRecord = <T>(line: string): T => {
  const record = parseJsonRecord<T>(line);
  if (!record) throw new MalformedRecordError();
  return record;
};
