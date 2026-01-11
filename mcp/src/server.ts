import http from "node:http";
import { z } from "zod";

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";

const FASTAPI_BASE_URL = process.env.FASTAPI_BASE_URL ?? "http://127.0.0.1:8000";
const MCP_PORT = Number(process.env.MCP_PORT ?? "8787");

// Pass-through object validation (FastAPI validates deeply with Pydantic)
const AnyObject = z.record(z.any());

async function callFastApi(path: string, body: unknown) {
  const url = `${FASTAPI_BASE_URL}${path}`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });

  const text = await res.text();
  let json: any;
  try {
    json = JSON.parse(text);
  } catch {
    throw new Error(`Backend returned non-JSON: ${text.slice(0, 300)}`);
  }

  if (!res.ok) {
    throw new Error(`Backend error ${res.status}: ${JSON.stringify(json).slice(0, 800)}`);
  }

  // Your FastAPI returns { tool, input, output }
  return json?.output ?? json;
}

function toolForwarder(endpointPath: string) {
  return async (input: unknown) => {
    const parsed = AnyObject.safeParse((input ?? {}) as any);
    if (!parsed.success) {
      return {
        content: [{ type: "text", text: "Invalid tool input: expected an object." }],
        isError: true,
      };
    }

    const out = await callFastApi(endpointPath, parsed.data);

    return {
      content: [{ type: "json", json: out }],
    };
  };
}

async function main() {
  const server = new McpServer({ name: "rv-confidence", version: "0.7.0" });

  server.tool(
    "manufacturer_intelligence",
    "Neutral manufacturer/model intelligence. Warranty and construction are factual only when source-backed; otherwise returns explicitly unverified educational guidance.",
    AnyObject,
    toolForwarder("/tools/manufacturer_intelligence")
  );

  server.tool(
    "rv_compare",
    "Compare two RVs using source-backed data when available; otherwise flags unknowns without guessing.",
    AnyObject,
    toolForwarder("/tools/rv_compare")
  );

  server.tool(
    "cost_depreciation_estimate",
    "Depreciation and ownership cost ranges with assumptions and disclosures. Estimates only; not financial advice.",
    AnyObject,
    toolForwarder("/tools/cost_depreciation_estimate")
  );

  server.tool(
    "deal_risk_scan",
    "Traffic-light risk scan for an RV deal quote. Educational; does not accuse dealers; provides clarifying questions and scripts.",
    AnyObject,
    toolForwarder("/tools/deal_risk_scan")
  );

  server.tool(
    "maintenance_repair_triage",
    "Post-purchase triage. Safety-first STOP gating for dangerous conditions (propane smell, CO alarm, etc.).",
    AnyObject,
    toolForwarder("/tools/maintenance_repair_triage")
  );

  // Official docs use Streamable HTTP transport for remote servers. :contentReference[oaicite:1]{index=1}
  const transport = new StreamableHTTPServerTransport({
    sessionIdGenerator: undefined,
    enableJsonResponse: true,
  });

  await server.connect(transport);

  const httpServer = http.createServer(async (req, res) => {
    if (req.method === "GET" && req.url === "/health") {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ status: "ok", service: "mcp", fastapi: FASTAPI_BASE_URL }));
      return;
    }

    await transport.handleRequest(req, res);
  });

  httpServer.listen(MCP_PORT, () => {
    console.log(`MCP server running on http://127.0.0.1:${MCP_PORT}`);
    console.log(`Forwarding tools to FastAPI at ${FASTAPI_BASE_URL}`);
  });
}

main().catch((err) => {
  console.error("MCP server failed:", err);
  process.exit(1);
});
