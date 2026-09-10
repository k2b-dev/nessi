import { afterEach, describe, expect, it } from "bun:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { openAICompatibleTranscription, type TranscriptionProvider } from "../src/ai/index.js";

const servers: ReturnType<typeof Bun.serve>[] = [];
afterEach(() => {
  for (const server of servers.splice(0)) server.stop(true);
});

function endpoint(handler: (request: Request) => Response | Promise<Response>) {
  const server = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch: handler });
  servers.push(server);
  return `${server.url}v1///`;
}

describe("audio transcription", () => {
  it("uploads multipart audio, options and custom authentication over HTTP", async () => {
    let received: Request | undefined;
    let form: FormData | undefined;
    const baseURL = endpoint(async (request) => {
      received = request;
      form = await request.formData();
      return Response.json({ text: "Hallo Welt", extra: "ignored" });
    });
    const provider: TranscriptionProvider = openAICompatibleTranscription("whisper-large-v3", {
      baseURL, apiKey: "test-key", name: "test-speech",
      headers: { "content-type": "application/json", authorization: "old", "X-Tenant": "test" },
    });
    expect(await provider.transcribe({
      file: new Blob(["audio bytes"], { type: "audio/mpeg" }), filename: "aufnahme.mp3",
      language: "de", prompt: "Nessi",
    })).toEqual({ text: "Hallo Welt" });
    expect(received?.method).toBe("POST");
    if (!received) throw new Error("Missing HTTP request");
    expect(new URL(received.url).pathname).toBe("/v1/audio/transcriptions");
    expect(received?.headers.get("authorization")).toBe("Bearer test-key");
    expect(received?.headers.get("x-tenant")).toBe("test");
    expect(received?.headers.get("content-type")).toContain("multipart/form-data; boundary=");
    expect(form?.get("model")).toBe("whisper-large-v3");
    expect(form?.get("response_format")).toBe("json");
    expect(form?.get("language")).toBe("de");
    expect(form?.get("prompt")).toBe("Nessi");
    const file = form?.get("file");
    if (!(file instanceof File)) throw new Error("Missing uploaded file");
    expect(file.name).toBe("aufnahme.mp3");
    expect(file.type).toBe("audio/mpeg");
    expect(await file.text()).toBe("audio bytes");
  });

  it("preserves File names, omits optional fields and accepts empty transcripts", async () => {
    let form: FormData | undefined;
    let authorization: string | null = null;
    const provider = openAICompatibleTranscription("local-whisper", {
      baseURL: endpoint(async (request) => {
        authorization = request.headers.get("authorization");
        form = await request.formData();
        return Response.json({ text: "" });
      }),
    });
    expect(await provider.transcribe({ file: new File(["silence"], "silence.wav") })).toEqual({ text: "" });
    expect(authorization).toBeNull();
    expect(form?.has("language")).toBe(false);
    expect(form?.has("prompt")).toBe(false);
    const file = form?.get("file");
    if (!(file instanceof File)) throw new Error("Missing uploaded file");
    expect(file.name).toBe("silence.wav");
  });

  for (const [body, expected] of [
    ["not json", "invalid transcription JSON"],
    ["null", "without a string text field"],
    ['{"text":42}', "without a string text field"],
    ["{}", "without a string text field"],
  ]) {
    it(`rejects malformed success response: ${body}`, async () => {
      const provider = openAICompatibleTranscription("whisper", {
        baseURL: endpoint(() => new Response(body)),
      });
      await expect(provider.transcribe({ file: new Blob() })).rejects.toThrow(expected);
    });
  }

  it("reports provider HTTP errors", async () => {
    const provider = openAICompatibleTranscription("whisper", {
      name: "speech", baseURL: endpoint(() => Response.json(
        { error: { message: "Quota exceeded", code: "quota" } }, { status: 429 },
      )),
    });
    await expect(provider.transcribe({ file: new Blob() })).rejects.toThrow("speech 429: Quota exceeded (code: quota)");
  });

  it("uploads a Bun.file with its original filename", async () => {
    const directory = await mkdtemp(join(tmpdir(), "nessi-audio-"));
    try {
      const path = join(directory, "recording.mp3");
      await Bun.write(path, "audio fixture");
      let form: FormData | undefined;
      const provider = openAICompatibleTranscription("whisper", {
        baseURL: endpoint(async (request) => {
          form = await request.formData();
          return Response.json({ text: "transcript" });
        }),
      });
      await provider.transcribe({ file: Bun.file(path) });
      const file = form?.get("file");
      if (!(file instanceof File)) throw new Error("Missing uploaded file");
      expect(file.name).toBe("recording.mp3");
      expect(await file.text()).toBe("audio fixture");
    } finally {
      await rm(directory, { recursive: true });
    }
  });

  it("reports connection failures", async () => {
    const baseURL = endpoint(() => Response.json({ text: "" }));
    servers.at(-1)?.stop(true);
    const provider = openAICompatibleTranscription("whisper", { baseURL, name: "speech" });
    await expect(provider.transcribe({ file: new Blob() })).rejects.toThrow("speech connection failed:");
  });

  it("does not upload when already aborted", async () => {
    let calls = 0;
    const provider = openAICompatibleTranscription("whisper", {
      baseURL: endpoint(() => { calls++; return Response.json({ text: "" }); }),
    });
    const reason = new Error("cancelled");
    await expect(provider.transcribe({ file: new Blob(), signal: AbortSignal.abort(reason) })).rejects.toBe(reason);
    expect(calls).toBe(0);
  });

  it("aborts an in-flight request", async () => {
    const controller = new AbortController();
    const reason = new Error("cancelled in flight");
    const provider = openAICompatibleTranscription("whisper", {
      baseURL: endpoint(() => {
        controller.abort(reason);
        return Response.json({ text: "late" });
      }),
    });
    await expect(provider.transcribe({ file: new Blob(), signal: controller.signal })).rejects.toBe(reason);
  });
});
