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
