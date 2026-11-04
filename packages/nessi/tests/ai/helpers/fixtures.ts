export function fixtureText(path: string): Promise<string> {
  return Bun.file(new URL(path, import.meta.url)).text();
}

export function fixtureJson<T>(path: string): Promise<T> {
  return Bun.file(new URL(path, import.meta.url)).json() as Promise<T>;
}

export function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

export function textResponse(body: string, contentType: string): Response {
  return new Response(body, {
    status: 200,
    headers: { "Content-Type": contentType },
  });
}

type FetchHandler = (input: string | URL | Request, init?: RequestInit) => Response | Promise<Response>;

/** Replace the global fetch for one test. Restore the original in `afterEach`. */
export function stubFetch(handler: FetchHandler): void {
  globalThis.fetch = Object.assign(async (input: string | URL | Request, init?: RequestInit) => handler(input, init), {
    preconnect: () => {},
  });
}

/**
 * A Chat Completions body as the equivalent SSE stream: one delta per message field, tool calls
 * with their index, then the finish reason, usage, and `[DONE]`. A body without choices (an
 * error envelope, for example) becomes a single record.
 */
export function completionStream(body: unknown): Response {
  const records: unknown[] = [];
  const payload = typeof body === "object" && body !== null ? body as Record<string, unknown> : {};
  const choices = Array.isArray(payload.choices) ? payload.choices as Array<Record<string, unknown>> : undefined;
  if (!choices) {
    records.push(body);
  } else {
    for (const choice of choices) {
      const message = (choice.message ?? {}) as Record<string, unknown>;
      const { tool_calls: toolCalls, role: _role, ...fields } = message;
      for (const [key, value] of Object.entries(fields)) {
        if (value !== null && value !== undefined) records.push({ choices: [{ index: 0, delta: { [key]: value }, finish_reason: null }] });
      }
      if (Array.isArray(toolCalls)) {
        toolCalls.forEach((call, index) => records.push({ choices: [{ index: 0, delta: { tool_calls: [{ index, ...call }] }, finish_reason: null }] }));
      }
      records.push({ choices: [{ index: 0, delta: {}, finish_reason: choice.finish_reason ?? "stop" }] });
    }
    if (payload.usage) records.push({ choices: [], usage: payload.usage });
  }
  const sse = records.map((record) => `data: ${JSON.stringify(record)}\n\n`).join("") + "data: [DONE]\n\n";
  return new Response(sse, { status: 200, headers: { "Content-Type": "text/event-stream" } });
}
