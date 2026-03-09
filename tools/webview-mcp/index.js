#!/usr/bin/env node
/**
 * darktable WebView MCP Server
 *
 * Enables Claude Code to automate and test darktable's Nova UI by executing
 * JavaScript in the embedded WebView via the frame server's /test/eval endpoint.
 *
 * Communication: HTTP POST to localhost:PORT/test/eval (frame server)
 * Port discovery: reads /tmp/darktable_test_port written by darktable at startup
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import http from "node:http";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execSync } from "node:child_process";

// ── Port Discovery ──────────────────────────────────────────────────

function discoverPort() {
  // 1. Check environment variable
  if (process.env.DT_TEST_PORT) {
    return parseInt(process.env.DT_TEST_PORT, 10);
  }

  // 2. Read from discovery file
  const portFile = path.join(os.tmpdir(), "darktable_test_port");
  try {
    const port = parseInt(fs.readFileSync(portFile, "utf-8").trim(), 10);
    if (port > 0 && port < 65536) return port;
  } catch {
    // File doesn't exist yet
  }

  return null;
}

// ── HTTP Client ─────────────────────────────────────────────────────

function postEval(port, code) {
  return new Promise((resolve, reject) => {
    const req = http.request(
      {
        hostname: "127.0.0.1",
        port,
        path: "/test/eval",
        method: "POST",
        headers: {
          "Content-Type": "text/plain",
          "Content-Length": Buffer.byteLength(code),
        },
        timeout: 15000,
      },
      (res) => {
        let data = "";
        res.on("data", (chunk) => (data += chunk));
        res.on("end", () => {
          try {
            resolve(JSON.parse(data));
          } catch {
            resolve({ ok: false, error: `Invalid JSON response: ${data}` });
          }
        });
      },
    );
    req.on("error", (err) => reject(err));
    req.on("timeout", () => {
      req.destroy();
      reject(new Error("Request timeout"));
    });
    req.write(code);
    req.end();
  });
}

// ── JS Snippets for Tools ───────────────────────────────────────────

const JS_CLICK = (selector) => `
(() => {
  const el = document.querySelector(${JSON.stringify(selector)});
  if (!el) return { ok: false, error: 'Element not found: ${selector}' };
  const rect = el.getBoundingClientRect();
  const x = rect.left + rect.width / 2;
  const y = rect.top + rect.height / 2;
  ['pointerdown', 'pointerup', 'click'].forEach(type => {
    el.dispatchEvent(new PointerEvent(type, { bubbles: true, clientX: x, clientY: y }));
  });
  return { ok: true, tag: el.tagName, text: el.textContent?.slice(0, 100) };
})()
`;

const JS_TYPE = (selector, text) => `
(() => {
  const el = document.querySelector(${JSON.stringify(selector)});
  if (!el) return { ok: false, error: 'Element not found: ${selector}' };
  el.focus();
  el.value = ${JSON.stringify(text)};
  el.dispatchEvent(new Event('input', { bubbles: true }));
  el.dispatchEvent(new Event('change', { bubbles: true }));
  return { ok: true };
})()
`;

const JS_DRAG = (selector, dx, dy, steps) => `
(() => {
  const el = document.querySelector(${JSON.stringify(selector)});
  if (!el) return { ok: false, error: 'Element not found: ${selector}' };
  const rect = el.getBoundingClientRect();
  const startX = rect.left + rect.width / 2;
  const startY = rect.top + rect.height / 2;
  const endX = startX + ${dx};
  const endY = startY + ${dy};
  const numSteps = ${steps || 10};
  el.dispatchEvent(new PointerEvent('pointerdown', {
    bubbles: true, clientX: startX, clientY: startY, pointerId: 1
  }));
  for (let i = 1; i <= numSteps; i++) {
    const t = i / numSteps;
    el.dispatchEvent(new PointerEvent('pointermove', {
      bubbles: true,
      clientX: startX + (endX - startX) * t,
      clientY: startY + (endY - startY) * t,
      pointerId: 1
    }));
  }
  el.dispatchEvent(new PointerEvent('pointerup', {
    bubbles: true, clientX: endX, clientY: endY, pointerId: 1
  }));
  return { ok: true, from: [startX, startY], to: [endX, endY] };
})()
`;

const JS_QUERY = (selector) => `
(() => {
  const els = document.querySelectorAll(${JSON.stringify(selector)});
  if (els.length === 0) return { ok: true, count: 0, elements: [] };
  const results = Array.from(els).slice(0, 20).map(el => {
    const rect = el.getBoundingClientRect();
    return {
      tag: el.tagName,
      text: el.textContent?.slice(0, 200)?.trim(),
      value: el.value,
      className: el.className?.slice?.(0, 200),
      visible: rect.width > 0 && rect.height > 0,
      bounds: { x: rect.x, y: rect.y, w: rect.width, h: rect.height },
      attrs: Object.fromEntries(
        Array.from(el.attributes || []).slice(0, 10).map(a => [a.name, a.value?.slice(0, 100)])
      ),
    };
  });
  return { ok: true, count: els.length, elements: results };
})()
`;

// ── MCP Server ──────────────────────────────────────────────────────

const server = new McpServer({
  name: "darktable-webview",
  version: "0.1.0",
});

/** Resolve port, throw if darktable not running */
function getPort() {
  const port = discoverPort();
  if (!port) {
    throw new Error(
      "darktable not running or port not found. " +
        "Start darktable and check /tmp/darktable_test_port exists. " +
        "You can also set DT_TEST_PORT environment variable.",
    );
  }
  return port;
}

server.tool(
  "evaluate",
  "Execute JavaScript code in darktable's WebView and return the result. " +
    "Use this as the foundation for all UI automation — click, type, query DOM, etc.",
  { code: z.string().describe("JavaScript code to evaluate in the WebView") },
  async ({ code }) => {
    const port = getPort();
    const result = await postEval(port, code);
    return {
      content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
    };
  },
);

server.tool(
  "click",
  "Click an element matching a CSS selector. Dispatches pointer and click events.",
  { selector: z.string().describe("CSS selector for the element to click") },
  async ({ selector }) => {
    const port = getPort();
    const result = await postEval(port, JS_CLICK(selector));
    return {
      content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
    };
  },
);

server.tool(
  "type",
  "Type text into an input element matching a CSS selector.",
  {
    selector: z.string().describe("CSS selector for the input element"),
    text: z.string().describe("Text to type into the element"),
  },
  async ({ selector, text }) => {
    const port = getPort();
    const result = await postEval(port, JS_TYPE(selector, text));
    return {
      content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
    };
  },
);

server.tool(
  "drag",
  "Drag an element by a pixel offset. Useful for sliders and interactive controls.",
  {
    selector: z.string().describe("CSS selector for the element to drag"),
    dx: z.number().describe("Horizontal pixel offset"),
    dy: z.number().describe("Vertical pixel offset"),
    steps: z
      .number()
      .optional()
      .default(10)
      .describe("Number of intermediate move events (default: 10)"),
  },
  async ({ selector, dx, dy, steps }) => {
    const port = getPort();
    const result = await postEval(port, JS_DRAG(selector, dx, dy, steps));
    return {
      content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
    };
  },
);

server.tool(
  "query",
  "Query DOM elements matching a CSS selector. Returns tag, text, value, " +
    "visibility, bounding box, and attributes for up to 20 matches.",
  {
    selector: z.string().describe("CSS selector to query"),
  },
  async ({ selector }) => {
    const port = getPort();
    const result = await postEval(port, JS_QUERY(selector));
    return {
      content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
    };
  },
);

server.tool(
  "wait_for",
  "Wait for an element matching a CSS selector to appear in the DOM. " +
    "Polls every 200ms until found or timeout.",
  {
    selector: z.string().describe("CSS selector to wait for"),
    timeout: z
      .number()
      .optional()
      .default(5000)
      .describe("Timeout in milliseconds (default: 5000)"),
  },
  async ({ selector, timeout }) => {
    const port = getPort();
    const deadline = Date.now() + timeout;
    while (Date.now() < deadline) {
      const result = await postEval(
        port,
        `(() => {
        const el = document.querySelector(${JSON.stringify(selector)});
        if (!el) return { found: false };
        const rect = el.getBoundingClientRect();
        return { found: true, tag: el.tagName, visible: rect.width > 0 && rect.height > 0 };
      })()`,
      );
      if (result.ok && result.value?.found) {
        return {
          content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
        };
      }
      await new Promise((r) => setTimeout(r, 200));
    }
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            ok: false,
            error: `Timeout: element "${selector}" not found within ${timeout}ms`,
          }),
        },
      ],
    };
  },
);

server.tool(
  "screenshot",
  "Take a screenshot of the darktable window. " +
    "Uses macOS screencapture to capture the active window. " +
    "Returns the file path to the saved PNG.",
  {
    output_path: z
      .string()
      .optional()
      .describe("Output file path (default: /tmp/dt-screenshot.png)"),
  },
  async ({ output_path }) => {
    const outPath = output_path || "/tmp/dt-screenshot.png";
    try {
      // macOS: capture the frontmost window
      execSync(`screencapture -l $(osascript -e 'tell app "darktable" to id of window 1') "${outPath}"`, {
        timeout: 5000,
      });
    } catch {
      // Fallback: capture the entire screen to a region
      try {
        execSync(`screencapture -x "${outPath}"`, { timeout: 5000 });
      } catch (e) {
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                ok: false,
                error: `Screenshot failed: ${e.message}`,
              }),
            },
          ],
        };
      }
    }
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({ ok: true, path: outPath }),
        },
      ],
    };
  },
);

// ── Start Server ────────────────────────────────────────────────────

const transport = new StdioServerTransport();
await server.connect(transport);
