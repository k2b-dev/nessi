import { describe, it, expect } from "bun:test";
import { z } from "zod";
import { defineTool, toolToJsonSchema } from "../src/tools.js";

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
