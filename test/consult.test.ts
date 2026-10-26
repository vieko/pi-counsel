import assert from "node:assert/strict";
import { describe, it } from "node:test";
import type { Api, Model } from "@earendil-works/pi-ai";
import { addCacheBreakpoints, pinGatewayRouting, resolveCounselModel, transcriptBudget } from "../extensions/pi-counsel/consult.ts";
import { DEFAULT_SETTINGS } from "../extensions/pi-counsel/settings.ts";
import { CounselError } from "../extensions/pi-counsel/consult.ts";

const model = (over: Partial<Model<Api>> = {}): Model<Api> =>
	({
		id: "anthropic/claude-fable-5.1",
		name: "fable",
		api: "anthropic-messages",
		provider: "vercel-ai-gateway",
		baseUrl: "https://ai-gateway.vercel.sh",
		input: ["text", "image"],
		cost: { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 },
		reasoning: true,
		contextWindow: 1_000_000,
		maxTokens: 64_000,
		...over,
	}) as Model<Api>;

describe("pinGatewayRouting", () => {
	it("injects providerOptions.gateway from compat on gateway models", () => {
		const m = model({ compat: { vercelGatewayRouting: { only: ["anthropic"], order: ["anthropic"] } } as never });
		const out = pinGatewayRouting({ model: "x", messages: [] }, m) as Record<string, unknown>;
		assert.deepEqual(out.providerOptions, { gateway: { only: ["anthropic"], order: ["anthropic"] } });
		assert.equal(out.model, "x");
	});

	it("leaves the payload alone without a pin, on other providers, or when already set", () => {
		assert.equal(pinGatewayRouting({ a: 1 }, model()), undefined);
		assert.equal(pinGatewayRouting({ a: 1 }, model({ provider: "anthropic", compat: { vercelGatewayRouting: { only: ["anthropic"] } } as never })), undefined);
		const m = model({ compat: { vercelGatewayRouting: { only: ["anthropic"] } } as never });
		assert.equal(pinGatewayRouting({ providerOptions: { gateway: { only: ["bedrock"] } } }, m), undefined);
	});
});

describe("transcriptBudget", () => {
	it("leaves room for output, margin and the fixed prompt text", () => {
		const b = transcriptBudget(model(), DEFAULT_SETTINGS, 40_000);
		assert.equal(b, 1_000_000 - 16_000 - 8_000 - 10_000);
	});

	it("caps output at the model max and never goes below the floor", () => {
		assert.equal(transcriptBudget(model({ maxTokens: 4_000 }), DEFAULT_SETTINGS, 0), 1_000_000 - 4_000 - 8_000);
		assert.equal(transcriptBudget(model({ contextWindow: 8_000 }), DEFAULT_SETTINGS, 0), 4_000);
	});
});

describe("resolveCounselModel", () => {
	const registry = (found: Model<Api> | undefined, auth: boolean) =>
		({ modelRegistry: { find: () => found, hasConfiguredAuth: () => auth } }) as never;

	it("throws typed errors for missing model and missing auth", () => {
		assert.throws(() => resolveCounselModel(registry(undefined, true), DEFAULT_SETTINGS), (e: unknown) => e instanceof CounselError && e.kind === "no-model");
		assert.throws(() => resolveCounselModel(registry(model(), false), DEFAULT_SETTINGS), (e: unknown) => e instanceof CounselError && e.kind === "no-auth");
	});

	it("returns the model when found and authenticated", () => {
		assert.equal(resolveCounselModel(registry(model(), true), DEFAULT_SETTINGS).id, "anthropic/claude-fable-5.1");
	});
});

describe("addCacheBreakpoints", () => {
	const cc = { type: "ephemeral", ttl: "1h" };
	const text = (t: string, mark = false) => ({ type: "text", text: t, ...(mark ? { cache_control: cc } : {}) });
	const payload = (n: number) => ({
		system: [text("sys", true)],
		messages: [...Array.from({ length: n - 1 }, (_, i) => ({ role: "user", content: [text(`m${i}`)] })), { role: "user", content: [text("request", true)] }],
	});
	type Out = { messages: { content: Record<string, unknown>[] }[] };
	const marked = (out: unknown) => (out as Out).messages.flatMap((m, i) => (m.content.at(-1)?.cache_control ? [i] : []));

	it("copies pi-ai's breakpoint onto the given messages", () => {
		const out = addCacheBreakpoints(payload(6), model(), [2, 4], 6);
		assert.deepEqual(marked(out), [2, 4, 5]);
		assert.deepEqual((out as Out).messages[2].content[0].cache_control, cc);
	});

	it("stays within four breakpoints, stable index first", () => {
		const p = payload(6);
		(p.messages[0].content[0] as Record<string, unknown>).cache_control = cc;
		assert.deepEqual(marked(addCacheBreakpoints(p, model(), [2, 4], 6)), [0, 2, 5]);
	});

	it("counts user messages only when pi-ai appends effort markers", () => {
		const p = payload(6) as { messages: unknown[] };
		p.messages.push({ role: "system", content: [] });
		assert.deepEqual(marked(addCacheBreakpoints(p, model(), [2, 4], 6)), [2, 4, 5]);
	});

	it("leaves payloads it did not build alone", () => {
		assert.equal(addCacheBreakpoints(payload(6), model(), [2], 7), undefined);
		assert.equal(addCacheBreakpoints(payload(6), model({ api: "openai-responses" as never }), [2], 6), undefined);
		const noCache = { messages: [{ role: "user", content: [text("a")] }, { role: "user", content: [text("b")] }] };
		assert.equal(addCacheBreakpoints(noCache, model(), [0], 2), undefined);
	});
});
