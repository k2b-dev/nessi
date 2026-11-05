import { afterEach, describe, expect, it } from "bun:test";
import { cloudflareDecision, systemOneDecision } from "../../../src/ai/index.js";
import type { TransportReport } from "../../../src/ai/index.js";
import { stubFetch } from "../helpers/fixtures.js";

const originalFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = originalFetch;
});

const questions = { urgent: { type: "noul" as const, instructions: "Urgent?" } };
const answer = () => Response.json({ model: "clef", answers: { urgent: { type: "noul", noul: 0.8 } }, usage: { input_tokens: 5 } });
const router = () => systemOneDecision("clef", { baseURL: "https://decide.example.test/v1", apiKey: "key" });

describe("decision onTransport", () => {
  it("reports one answered request with its timings, before the result", async () => {
    const order: string[] = [];
    const reports: TransportReport[] = [];
    stubFetch(answer);
    const result = await router().decide({
      state: "x",
      questions,
      onTransport: (report) => {
        reports.push(report);
        order.push("report");
      },
    });
    order.push("result");
    expect(result.answers.urgent.value).toBe(true);
    expect(order).toEqual(["report", "result"]);
    expect(reports).toHaveLength(1);
    const [report] = reports;
    expect(report).toMatchObject({ status: 200 });
    expect(report!.headersAt!).toBeGreaterThanOrEqual(report!.startedAt);
    expect(report!.firstByteAt!).toBeGreaterThanOrEqual(report!.headersAt!);
    expect(report).not.toHaveProperty("connectionCode");
  });

  it("reports a rejected request with its status and Retry-After before the error surfaces", async () => {
    const order: string[] = [];
    stubFetch(() => Response.json({ error: { message: "Slow down" } }, { status: 429, headers: { "retry-after": "3" } }));
    await router()
      .decide({ state: "x", questions, onTransport: (report) => order.push(`report ${report.status} ${report.retryAfterMs} ${report.firstByteAt !== undefined}`) })
      .catch(() => order.push("error"));
    expect(order).toEqual(["report 429 3000 true", "error"]);
  });

  it("reports a failed connection with the runtime code and no response", async () => {
    stubFetch(() => {
      throw Object.assign(new TypeError("Unable to connect"), { code: "ECONNREFUSED" });
    });
    const reports: TransportReport[] = [];
    await expect(router().decide({ state: "x", questions, onTransport: (report) => reports.push(report) })).rejects.toThrow("connection failed");
    expect(reports).toEqual([expect.objectContaining({ connectionCode: "ECONNREFUSED" })]);
    expect(reports[0]).not.toHaveProperty("headersAt");
  });

  it("reports an aborted request once, without a connection code", async () => {
    const caller = new AbortController();
    const reason = new Error("stop");
    stubFetch((_url, init) => new Promise<Response>((_, reject) => {
      init?.signal?.addEventListener("abort", () => reject(init.signal?.reason), { once: true });
      setTimeout(() => caller.abort(reason), 1);
    }));
    const reports: TransportReport[] = [];
    await expect(router().decide({ state: "x", questions, signal: caller.signal, onTransport: (report) => reports.push(report) })).rejects.toBe(reason);
    expect(reports).toHaveLength(1);
    expect(reports[0]).not.toHaveProperty("connectionCode");
    expect(reports[0]).not.toHaveProperty("headersAt");
  });

  it("reports nothing for a request that never left", async () => {
    let sent = 0;
    stubFetch(() => {
      sent++;
      return answer();
    });
    const reports: TransportReport[] = [];
    const broken = systemOneDecision("clef", { baseURL: "https://decide.example.test/v1", apiKey: "bad\nkey" });
    await expect(broken.decide({ state: "x", questions, onTransport: (report) => reports.push(report) })).rejects.toThrow("Authorization");
    await expect(router().decide({ state: "x", questions, signal: AbortSignal.abort(), onTransport: (report) => reports.push(report) })).rejects.toThrow();
    expect(sent).toBe(0);
    expect(reports).toEqual([]);
  });

  it("ignores an error thrown by the hook", async () => {
    stubFetch(answer);
    const result = await router().decide({ state: "x", questions, onTransport: () => { throw new Error("telemetry down"); } });
    expect(result.answers.urgent.probability).toBe(0.8);
  });

  it("reports for the Cloudflare preset too", async () => {
    stubFetch(() => Response.json({ success: true, result: { answers: { urgent: { type: "noul", noul: 0.2 } } } }));
    const reports: TransportReport[] = [];
    await cloudflareDecision("clef", { accountId: "acct", apiToken: "token" }).decide({ state: "x", questions, onTransport: (report) => reports.push(report) });
    expect(reports).toEqual([expect.objectContaining({ status: 200, firstByteAt: expect.any(Number) })]);
  });
});
