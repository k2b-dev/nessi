import { afterEach, describe, expect, it } from "bun:test";
import { complete, ProviderError } from "../../../src/ai/index.js";
import type { TransportReport } from "../../../src/ai/index.js";
import { endpoint } from "../helpers/endpoints.js";
import { stubFetch } from "../helpers/fixtures.js";

const originalFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = originalFetch; });

const encoder = new TextEncoder();
const answer = [
  `data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: "Hi" }, finish_reason: "stop" }] })}\n\n`,
  "data: [DONE]\n\n",
].join("");

/** Runs one request and returns every report it produced, plus its error if it failed. */
const reportsOf = async (signal?: AbortSignal, timeouts?: { firstByteMs?: number; idleMs?: number }) => {
  const reports: TransportReport[] = [];
  const error = await complete(endpoint("openai", { baseURL: "https://example.test", timeouts }), {
    messages: [],
    signal,
    onTransport: (report) => reports.push(report),
  }).then(() => undefined, (caught: unknown) => caught);
  return { reports, error };
};

describe("onTransport", () => {
  it("reports one answered request with its timings", async () => {
    stubFetch(async () => {
      await new Promise((resolve) => setTimeout(resolve, 5));
      return new Response(new ReadableStream<Uint8Array>({
        async pull(controller) {
          await new Promise((resolve) => setTimeout(resolve, 5));
          controller.enqueue(encoder.encode(answer));
          controller.close();
        },
      }), { status: 200, headers: { "Content-Type": "text/event-stream" } });
    });
    const { reports, error } = await reportsOf();
    expect(error).toBeUndefined();
    expect(reports).toHaveLength(1);
    const [report] = reports;
    expect(report).toMatchObject({ status: 200 });
    expect(report!.headersAt!).toBeGreaterThanOrEqual(report!.startedAt);
    expect(report!.firstByteAt!).toBeGreaterThanOrEqual(report!.headersAt!);
  });

  it("reports a rejected request with its status and Retry-After before the error surfaces", async () => {
    const order: string[] = [];
    stubFetch(() => new Response(JSON.stringify({ error: { message: "Slow down" } }), { status: 429, headers: { "retry-after": "3" } }));
    const error = await complete(endpoint("openai", { baseURL: "https://example.test" }), {
      messages: [],
      onTransport: (report) => order.push(`report ${report.status} ${report.retryAfterMs}`),
    }).then(() => undefined, (caught: unknown) => {
      order.push("error");
      return caught;
    });
    expect(error).toBeInstanceOf(ProviderError);
    expect(order).toEqual(["report 429 3000", "error"]);
  });

  it("reports a failed connection with the runtime code and no response", async () => {
    stubFetch(() => { throw Object.assign(new TypeError("Unable to connect"), { code: "ECONNREFUSED" }); });
    const { reports } = await reportsOf();
    expect(reports).toEqual([expect.objectContaining({ connectionCode: "ECONNREFUSED" })]);
    expect(reports[0]).not.toHaveProperty("headersAt");
  });

  it("reports a first-byte timeout without a connection code", async () => {
    stubFetch(() => new Promise<Response>(() => {}));
    const { reports, error } = await reportsOf(undefined, { firstByteMs: 5 });
    expect(error).toMatchObject({ kind: "timeout" });
    expect(reports).toHaveLength(1);
    expect(reports[0]).not.toHaveProperty("connectionCode");
    expect(reports[0]).not.toHaveProperty("headersAt");
  });

  it("reports once when the caller aborts while the body streams", async () => {
    const caller = new AbortController();
    stubFetch(() => new Response(new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: "Hi" } }] })}\n\n`));
        setTimeout(() => caller.abort(new Error("stop")), 5);
      },
    }), { headers: { "Content-Type": "text/event-stream" } }));
    const { reports, error } = await reportsOf(caller.signal);
    expect(error).toMatchObject({ message: "stop" });
    expect(reports).toHaveLength(1);
    expect(reports[0]).toMatchObject({ status: 200 });
    expect(reports[0]!.firstByteAt).toBeDefined();
  });

  it("ignores an error thrown by the hook", async () => {
    stubFetch(() => new Response(answer, { headers: { "Content-Type": "text/event-stream" } }));
    const result = await complete(endpoint("openai", { baseURL: "https://example.test" }), {
      messages: [],
      onTransport: () => { throw new Error("telemetry down"); },
    });
    expect(result.message.content).toEqual([{ type: "text", text: "Hi" }]);
  });
});
