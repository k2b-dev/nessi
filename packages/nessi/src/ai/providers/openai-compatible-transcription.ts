import { formatConnectionError, normalizeHttpError } from "../shared/errors.js";
import type { TranscriptionProvider } from "../transcription.js";

export type OpenAICompatibleTranscriptionOptions = {
  baseURL: string;
  apiKey?: string;
  headers?: Record<string, string>;
  name?: string;
};

export const openAICompatibleTranscription = (
  model: string,
  options: OpenAICompatibleTranscriptionOptions,
): TranscriptionProvider => {
  const baseURL = options.baseURL.replace(/\/+$/, "");
  const name = options.name ?? "openai-compatible";
  const headers = new Headers(options.headers);
  // Fetch must supply the multipart boundary, including with custom headers.
  headers.delete("Content-Type");
  if (options.apiKey) headers.set("Authorization", `Bearer ${options.apiKey}`);

  return {
    name,
    model,
    async transcribe(request) {
      request.signal?.throwIfAborted();
      const body = new FormData();
      // Bun.file().name can contain a local path; only upload the basename.
      const fileName = "name" in request.file && typeof request.file.name === "string"
        ? request.file.name.split(/[\\/]/).at(-1)
        : undefined;
      const filename = request.filename ?? fileName;
      if (filename !== undefined) body.append("file", request.file, filename);
      else body.append("file", request.file);
      body.append("model", model);
      body.append("response_format", "json");
      if (request.language !== undefined) body.append("language", request.language);
      if (request.prompt !== undefined) body.append("prompt", request.prompt);

      const response = await fetch(`${baseURL}/audio/transcriptions`, {
        method: "POST",
        headers,
        body,
        signal: request.signal,
      }).catch((error: unknown) => {
        request.signal?.throwIfAborted();
        throw new Error(formatConnectionError(name, error));
      });
      if (!response.ok) {
        const normalized = await normalizeHttpError(name, response);
        request.signal?.throwIfAborted();
        throw new Error(normalized.error);
      }

      const raw = await response.text();
      let payload: unknown;
      try {
        payload = JSON.parse(raw);
      } catch {
        throw new Error(`${name} returned invalid transcription JSON.`);
      }
      if (typeof payload !== "object" || payload === null
        || !("text" in payload) || typeof payload.text !== "string") {
        throw new Error(`${name} returned a transcription without a string text field.`);
      }
      return { text: payload.text };
    },
  };
};
