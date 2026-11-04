import { describe, it, expect } from "bun:test";
import { z } from "zod";
import { nessi } from "../src/nessi.js";
import { memoryStore } from "../src/stores.js";
import { defineTool, toolToJsonSchema, toolToSpec } from "../src/tools.js";
import { mockProviderMultiTurn } from "./mock-provider.js";

describe("defineTool", () => {
  const echoTool = defineTool({
    name: "echo",
    description: "Echoes input",
    inputSchema: z.object({ text: z.string() }),
    outputSchema: z.object({ echoed: z.string() }),
  });

  it("creates a tool definition with correct properties", () => {
    expect(echoTool.name).toBe("echo");
    expect(echoTool.description).toBe("Echoes input");
    expect(echoTool.needsApproval).toBe(false);
  });

  it("creates a server tool", () => {
    const server = echoTool.server(async (input) => ({ echoed: input.text }));
    expect(server.kind).toBe("server");
    expect(server.def).toBe(echoTool);
  });

  it("creates a client tool", () => {
    const client = echoTool.client((input) => ({ echoed: input.text }));
    expect(client.kind).toBe("client");
    expect(client.def).toBe(echoTool);
  });

  it("server tool executes correctly", async () => {
    const server = echoTool.server(async (input) => ({ echoed: input.text }));
    const result = await server.execute(
      { text: "hello" },
      {
        signal: new AbortController().signal,
        requestApproval: async () => true,
        requestClientTool: () => Promise.reject(new Error("not used")),
      },
    );
    expect(result).toEqual({ echoed: "hello" });
  });

  it("client tool executes correctly", async () => {
    const client = echoTool.client((input) => ({ echoed: input.text }));
    const result = await client.execute({ text: "hello" });
    expect(result).toEqual({ echoed: "hello" });
  });

  it("respects needsApproval flag", () => {
    const tool = defineTool({
      name: "danger",
      description: "Dangerous",
      inputSchema: z.object({}),
      needsApproval: true,
    });
    expect(tool.needsApproval).toBe(true);
  });
});

describe("toolToJsonSchema", () => {
  it("converts tool to JSON Schema", () => {
    const tool = defineTool({
      name: "greet",
      description: "Greets someone",
      inputSchema: z.object({
        name: z.string().describe("Person's name"),
        formal: z.boolean().optional(),
      }),
    });

    const schema = toolToJsonSchema(tool.server(async () => undefined));

    expect(schema.name).toBe("greet");
    expect(schema.description).toBe("Greets someone");
    expect(schema.parameters).toBeDefined();
    expect((schema.parameters as any).type).toBe("object");
    expect((schema.parameters as any).properties.name).toBeDefined();
    expect((schema.parameters as any).properties.formal).toBeDefined();
  });
});

describe("client tool approval", () => {
  it("rejects needsApproval on client tools", () => {
    const def = defineTool({
      name: "confirm",
      description: "Needs approval",
      inputSchema: z.object({}),
      needsApproval: true,
    });

    expect(() => def.client(() => ({}))).toThrow('Tool "confirm": needsApproval is only supported for server tools');
    expect(def.server(async () => ({})).kind).toBe("server");
  });

  it("rejects hand-built client tools with needsApproval in nessi()", async () => {
    const { nessi } = await import("../src/nessi.js");
    const { memoryStore } = await import("../src/stores.js");
    const { mockProvider } = await import("./mock-provider.js");
    const base = defineTool({ name: "confirm", description: "x", inputSchema: z.object({}) });
    const tool = { kind: "client" as const, def: { ...base, needsApproval: true }, execute: () => ({}) };

    expect(() => nessi({ provider: mockProvider([]), store: memoryStore(), systemPrompt: "sys", input: "go", tools: [tool] }))
      .toThrow("needsApproval is only supported for server tools");
  });
});

describe("tool JSON schemas", () => {
  it("convert each Zod schema once per loop and share the frozen result", async () => {
    const inputSchema = z.object({ query: z.string(), filters: z.object({ tags: z.array(z.string()) }) });
    const sent: unknown[] = [];
    const provider = mockProviderMultiTurn((request, index) => {
      sent.push(request.tools?.[0]?.inputSchema);
      return index === 0
        ? [{ type: "tool_call", callId: "c1", name: "search", args: { query: "x", filters: { tags: [] } } }, { type: "usage", usage: { input: 1, output: 1, total: 2 }, finishReason: "tool_use" }]
        : [{ type: "text", delta: "Done." }, { type: "usage", usage: { input: 1, output: 1, total: 2 }, finishReason: "stop" }];
    });
    // The resolver builds new tool objects every turn around the same schema.
    const tools = async () => [defineTool({ name: "search", description: "Search", inputSchema }).server(async () => "ok")];
    for await (const _event of nessi({ provider, store: memoryStore(), systemPrompt: "test", input: "Search", tools })) { /* drain */ }

    expect(sent).toHaveLength(2);
    expect(sent[1]).toBe(sent[0]);
    expect(Object.isFrozen(sent[0])).toBe(true);
    expect(() => { (sent[0] as { type?: string }).type = "array"; }).toThrow(TypeError);
  });

  it("keep public conversions fresh and mutable", () => {
    const tool = defineTool({ name: "search", description: "Search", inputSchema: z.object({ query: z.string() }) }).server(async () => "ok");
    const first = toolToSpec(tool).inputSchema;
    expect(toolToSpec(tool).inputSchema).not.toBe(first);
    expect(Object.isFrozen(first)).toBe(false);
  });
});
