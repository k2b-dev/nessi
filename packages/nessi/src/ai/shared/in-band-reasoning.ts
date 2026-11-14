import type { RawStreamEvent } from "../types.js";

const markers = [
  { value: "<|open|>think<|sep|>", kind: "thinking" },
  { value: "<|close|>think<|sep|>", kind: "text" },
  { value: "<think>", kind: "thinking" },
  { value: "</think>", kind: "text" },
] as const;

const maxMarkerLength = Math.max(...markers.map((marker) => marker.value.length));

type Span = Extract<RawStreamEvent, { type: "text" | "thinking" }>;

/** Parse content deltas before exposing them as canonical blocks. */
export function inBandReasoning(model: string, implicit = true) {
  let kind: "text" | "thinking" = "text";
  let buffer = "";
  let searchFrom = 0;
  let atStart = true;
  // Kimi K3 can omit the opener and send reasoning as content up to the closing marker.
  // Keep only this ambiguous first span until a marker, structured reasoning or a boundary.
  let undecided = implicit && /(?:^|\/)kimi[-_]k3(?:$|[-_:])/i.test(model);

  const eligible = (value: string, index: number) => {
    if (value === "<think>") return atStart && !buffer.slice(0, index).trim();
    if (value === "</think>") return kind === "thinking";
    return true;
  };
  const suffixLength = () => {
    for (let size = Math.min(buffer.length, maxMarkerLength); size > 0; size--) {
      if (markers.some((marker) => eligible(marker.value, buffer.length - size) && marker.value.startsWith(buffer.slice(-size)))) return size;
    }
    return 0;
  };
  const emit = (text: string, type = kind): Span[] => {
    if (text.trim()) atStart = false;
    return text ? [{ type, delta: text }] : [];
  };

  return {
    push(delta: string): Span[] {
      buffer += delta;
      const spans: Span[] = [];
      for (;;) {
        const match = markers.map((marker) => ({ ...marker, index: buffer.indexOf(marker.value, searchFrom) }))
          .filter((marker) => marker.index >= 0 && eligible(marker.value, marker.index)).sort((a, b) => a.index - b.index)[0];
        if (!match) {
          searchFrom = Math.max(0, buffer.length - maxMarkerLength + 1);
          break;
        }
        spans.push(...emit(buffer.slice(0, match.index), undecided && match.kind === "text" ? "thinking" : kind));
        buffer = buffer.slice(match.index + match.value.length);
        searchFrom = 0;
        atStart = false;
        kind = match.kind;
        undecided = false;
      }
      if (!undecided) {
        const keep = suffixLength();
        spans.push(...emit(buffer.slice(0, buffer.length - keep)));
        buffer = buffer.slice(buffer.length - keep);
        searchFrom = 0;
      }
      return spans;
    },
    structured(): Span[] {
      undecided = false;
      return this.push("");
    },
    flush(failed = false): Span[] {
      const keep = suffixLength();
      // Incomplete control syntax at EOF must not become visible. A lone '<' is ordinary text.
      const drop = keep > 1 ? keep : 0;
      const tail = buffer.slice(buffer.length - drop);
      const type = undecided && (failed || (drop > 1 && "<|close|>think<|sep|>".startsWith(tail))) ? "thinking" : kind;
      const spans = emit(buffer.slice(0, buffer.length - drop), type);
      buffer = "";
      searchFrom = 0;
      undecided = false;
      return spans;
    },
  };
}
