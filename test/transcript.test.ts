import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { estimateTokens, redact, renderEntry, serializeTranscript, TIERS } from "../extensions/pi-counsel/transcript.ts";

const opts = { skipToolCalls: [] as string[], maxToolArgsChars: 1_500, maxToolResultChars: 3_000, maxTextChars: 12_000, headFraction: 0.25 };

type Entry = Parameters<typeof renderEntry>[0];
const user = (text: string): Entry => ({ type: "message", message: { role: "user", content: text } });
const assistant = (text: string, calls: { name: string; args: unknown }[] = []): Entry => ({
	type: "message",
	message: {
		role: "assistant",
		content: [
			{ type: "thinking", thinking: "private" },
			{ type: "text", text },
			...calls.map((c, i) => ({ type: "toolCall", id: `c${i}`, name: c.name, arguments: c.args })),
		],
	},
});
const toolResult = (name: string, text: string, isError = false): Entry => ({
	type: "message",
	message: { role: "toolResult", toolName: name, isError, content: [{ type: "text", text }] },
});

describe("renderEntry", () => {
	it("labels user, assistant, tool call and tool result blocks", () => {
		assert.deepEqual(renderEntry(user("hi"), opts), ["[user]\nhi"]);
		assert.deepEqual(renderEntry(assistant("plan", [{ name: "read", args: { path: "a.ts" } }]), opts), [
			"[assistant]\nplan",
			'[tool call: read]\n{"path":"a.ts"}',
		]);
		assert.deepEqual(renderEntry(toolResult("bash", "ok"), opts), ["[tool result: bash]\nok"]);
		assert.deepEqual(renderEntry(toolResult("bash", "boom", true), opts), ["[tool result: bash, error]\nboom"]);
	});

	it("drops thinking and keeps images as placeholders", () => {
		const entry = {
			type: "message",
			message: { role: "user", content: [{ type: "image", mimeType: "image/png", data: "A".repeat(4096) }, { type: "text", text: "see" }] },
		};
		assert.deepEqual(renderEntry(entry, opts), ["[user]\n[image image/png, ~3 KB]\nsee"]);
		assert.ok(!renderEntry(assistant("x"), opts).join().includes("private"));
	});

	it("renders compaction summaries and skips bookkeeping entries", () => {
		assert.deepEqual(renderEntry({ type: "compaction", summary: "earlier" }, opts), ["[summary of earlier conversation]\nearlier"]);
		assert.deepEqual(renderEntry({ type: "model_change" }, opts), []);
		assert.deepEqual(renderEntry({ type: "custom", customType: "counsel" }, opts), []);
	});

	it("clips oversized tool bodies but keeps head and tail", () => {
		const big = `START${"x".repeat(10_000)}END`;
		const [block] = renderEntry(toolResult("bash", big), opts);
		assert.ok(block.length < 3_200, `got ${block.length}`);
		assert.ok(block.includes("START") && block.includes("END") && block.includes("chars elided"));
	});
});

describe("redact", () => {
	it("masks known token shapes and assignments, keeps the surrounding text", () => {
		const cases: [string, string][] = [
			["key sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789", "key [redacted]"],
			["ghp_abcdefghijklmnopqrstuvwxyz0123456789 pushed", "[redacted] pushed"],
			["xoxb-1234567890-abcdefghij token", "[redacted] token"],
			["Authorization: Bearer abcdefghijklmnopqrstuvwxyz", "Authorization: Bearer [redacted]"],
			["AI_GATEWAY_API_KEY=vck_supersecretvalue123", "AI_GATEWAY_API_KEY=[redacted]"],
			['"DATABASE_PASSWORD": "hunter22!"', '"DATABASE_PASSWORD": "[redacted]"'],
			["postgres://user:s3cr3t@host/db", "postgres://user:[redacted]@host/db"],
			["https://me:tok3n123@example.com/x", "https://me:[redacted]@example.com/x"],
		];
		for (const [input, expected] of cases) assert.equal(redact(input), expected, input);
	});

	it("leaves ordinary text alone", () => {
		const text = "const token = await getToken(); // KEY_COUNT = 3; api_key_id: abc";
		assert.equal(redact(text), text);
	});
});

describe("serializeTranscript", () => {
	it("returns one message per rendered entry under budget", () => {
		const entries = [user("task"), assistant("ok", [{ name: "read", args: {} }]), toolResult("read", "file"), { type: "model_change" }];
		const out = serializeTranscript(entries, { budgetTokens: 10_000 });
		assert.equal(out.messages.length, 3);
		assert.equal(out.elidedEntries, 0);
		assert.equal(out.totalEntries, 3);
		assert.ok(out.messages.every((m) => m.role === "user"));
		assert.equal(out.tokens, out.messages.reduce((a, m) => a + estimateTokens(m.content), 0));
	});

	it("is byte-stable for the shared prefix across consults", () => {
		const base = [user("task"), assistant("first"), toolResult("bash", "out")];
		const a = serializeTranscript(base, { budgetTokens: 10_000 });
		const b = serializeTranscript([...base, assistant("second")], { budgetTokens: 10_000 });
		assert.deepEqual(b.messages.slice(0, a.messages.length), a.messages);
	});

	it("elides the middle and keeps opening and tail with a marker", () => {
		const entries: Entry[] = [user("the task")];
		for (let i = 0; i < 60; i++) entries.push(toolResult("bash", `${i}:${"y".repeat(400)}`));
		entries.push(assistant("latest state"));
		const out = serializeTranscript(entries, { budgetTokens: 2_000 });
		assert.ok(out.elidedEntries > 0);
		assert.ok(out.tokens <= 2_000, `tokens ${out.tokens}`);
		assert.equal(out.messages[0].content, "[user]\nthe task");
		assert.equal(out.messages.at(-1)?.content, "[assistant]\nlatest state");
		const marker = out.messages.find((m) => m.content.startsWith("[..."));
		assert.ok(marker && marker.content.includes(`${out.elidedEntries} entries`));
	});

	it("skips listed tool calls but keeps their results", () => {
		const entries = [assistant("asking", [{ name: "counsel", args: { question: "q" } }, { name: "read", args: {} }]), toolResult("counsel", "earlier advice")];
		const out = serializeTranscript(entries, { budgetTokens: 1_000, skipToolCalls: ["counsel"] });
		assert.deepEqual(
			out.messages.map((m) => m.content),
			["[assistant]\nasking\n\n[tool call: read]\n{}", "[tool result: counsel]\nearlier advice"],
		);
	});

	it("reads large tool results whole when the budget allows", () => {
		const big = `START${"x".repeat(40_000)}END`;
		const out = serializeTranscript([user("task"), toolResult("read", big)], { budgetTokens: 50_000 });
		assert.equal(out.maxToolResultChars, TIERS[0].maxToolResultChars);
		assert.ok(out.messages[1].content.includes(big));
	});

	it("steps down clip tiers before eliding entries", () => {
		const entries: Entry[] = [user("task")];
		for (let i = 0; i < 10; i++) entries.push(toolResult("read", `${i}:${"z".repeat(30_000)}`));
		// Tier 0 needs ~75K tokens, tier 1 (20K chars) ~50K, tier 2 (8K chars) ~20K.
		const out = serializeTranscript(entries, { budgetTokens: 25_000 });
		assert.equal(out.elidedEntries, 0);
		assert.equal(out.maxToolResultChars, 8_000);
		assert.ok(out.tokens <= 25_000);
	});

	it("pins one tier when a cap is passed", () => {
		const out = serializeTranscript([toolResult("read", "w".repeat(10_000))], { budgetTokens: 50_000, maxToolResultChars: 3_000 });
		assert.equal(out.maxToolResultChars, 3_000);
		assert.ok(out.messages[0].content.includes("chars elided"));
	});

	it("groups entries into chunks and marks the last complete chunk and the last message", () => {
		const entries: Entry[] = Array.from({ length: 37 }, (_, i) => toolResult("bash", `out ${i}`));
		const out = serializeTranscript(entries, { budgetTokens: 10_000, chunkEntries: 16 });
		assert.equal(out.messages.length, 3);
		assert.equal(out.totalEntries, 37);
		assert.ok(out.messages[0].content.startsWith("[tool result: bash]\nout 0\n\n[tool result: bash]\nout 1"));
		assert.deepEqual(out.breakpoints, [1, 2]);
		assert.equal(serializeTranscript(entries.slice(0, 32), { budgetTokens: 10_000, chunkEntries: 16 }).breakpoints.join(), "1");
		assert.equal(serializeTranscript(entries.slice(0, 5), { budgetTokens: 10_000, chunkEntries: 16 }).breakpoints.join(), "0");
	});

	it("keeps complete chunks byte-identical as the session grows", () => {
		const entries: Entry[] = Array.from({ length: 80 }, (_, i) => toolResult("bash", `out ${i}`));
		const a = serializeTranscript(entries.slice(0, 40), { budgetTokens: 10_000, chunkEntries: 16 });
		const b = serializeTranscript(entries, { budgetTokens: 10_000, chunkEntries: 16 });
		const stable = a.breakpoints[0];
		assert.deepEqual(b.messages.slice(0, stable + 1), a.messages.slice(0, stable + 1));
	});

	it("puts the stable breakpoint in the head when eliding", () => {
		const entries: Entry[] = [user("the task")];
		for (let i = 0; i < 200; i++) entries.push(toolResult("bash", `${i}:${"y".repeat(400)}`));
		const out = serializeTranscript(entries, { budgetTokens: 20_000, chunkEntries: 4 });
		assert.ok(out.elidedEntries > 0);
		const marker = out.messages.findIndex((m) => m.content.startsWith("[..."));
		assert.ok(out.breakpoints[0] < marker, `breakpoints ${out.breakpoints} marker ${marker}`);
		assert.equal(out.breakpoints.at(-1), out.messages.length - 1);
	});

	it("redacts inside entries", () => {
		const out = serializeTranscript([toolResult("bash", "TOKEN=abcdef123456")], { budgetTokens: 1_000 });
		assert.equal(out.messages[0].content, "[tool result: bash]\nTOKEN=[redacted]");
	});
});
