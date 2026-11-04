import { parseJsonRecord } from "./json.js";
import { readRecordBatches } from "./sse.js";
import type { ByteStreamReader } from "./sse.js";

export const parseNDJSON = async function* <T>(
  reader: ByteStreamReader,
  timeouts: { firstByteMs?: number; idleMs?: number } = {},
): AsyncGenerator<T> {
  for await (const lines of readRecordBatches(reader, "\n", timeouts)) {
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed) yield parseRecord<T>(trimmed);
    }
  }
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
