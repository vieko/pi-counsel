import assert from "node:assert/strict";
import { describe, it } from "node:test";
import type { Api, Model } from "@earendil-works/pi-ai";
import { pinGatewayRouting, resolveCounselModel, transcriptBudget } from "../extensions/pi-counsel/consult.ts";
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
