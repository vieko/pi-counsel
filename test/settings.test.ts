import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { DEFAULT_SETTINGS, modelFamily, parseEnvOverride, parseModelRef, parseSettings, sessionNote, withModelOverride } from "../extensions/pi-counsel/settings.ts";

describe("parseSettings", () => {
	it("falls back per field", () => {
		const s = parseSettings({ provider: "anthropic", effort: "loud", maxTokens: 10, maxSpendPerSession: -1, enabled: "yes" });
		assert.equal(s.provider, "anthropic");
		assert.equal(s.model, DEFAULT_SETTINGS.model);
		assert.equal(s.effort, DEFAULT_SETTINGS.effort);
		assert.equal(s.maxTokens, DEFAULT_SETTINGS.maxTokens);
		assert.equal(s.maxSpendPerSession, DEFAULT_SETTINGS.maxSpendPerSession);
		assert.equal(s.enabled, DEFAULT_SETTINGS.enabled);
	});

	it("returns defaults for garbage", () => {
		assert.deepEqual(parseSettings(null), DEFAULT_SETTINGS);
		assert.deepEqual(parseSettings("x"), DEFAULT_SETTINGS);
	});

	it("accepts zero budgets (tool always declines)", () => {
		assert.equal(parseSettings({ maxConsultsPerSession: 0 }).maxConsultsPerSession, 0);
	});
});

describe("parseModelRef", () => {
	it("splits provider from a slashed model id and an effort suffix", () => {
		assert.deepEqual(parseModelRef("vercel-ai-gateway/anthropic/claude-fable-5.1:high"), {
			provider: "vercel-ai-gateway",
			model: "anthropic/claude-fable-5.1",
			effort: "high",
		});
		assert.deepEqual(parseModelRef("anthropic/claude-opus-5.5"), { provider: "anthropic", model: "claude-opus-5.5", effort: undefined });
	});

	it("keeps colons that are not effort levels", () => {
		assert.deepEqual(parseModelRef("ollama/llama3:70b"), { provider: "ollama", model: "llama3:70b", effort: undefined });
	});

	it("rejects refs without a provider", () => {
		assert.equal(parseModelRef("claude-opus-5.5"), undefined);
		assert.equal(parseModelRef("/x"), undefined);
		assert.equal(parseModelRef("x/"), undefined);
	});
});

describe("parseEnvOverride", () => {
	it("maps off/on/model", () => {
		assert.deepEqual(parseEnvOverride("off"), { kind: "off" });
		assert.deepEqual(parseEnvOverride("0"), { kind: "off" });
		assert.equal(parseEnvOverride("on"), undefined);
		assert.equal(parseEnvOverride(undefined), undefined);
		assert.equal(parseEnvOverride(""), undefined);
		assert.deepEqual(parseEnvOverride("anthropic/claude-opus-5.5:max"), { kind: "model", provider: "anthropic", model: "claude-opus-5.5", effort: "max" });
		assert.equal(parseEnvOverride("nonsense"), undefined);
	});
});

describe("modelFamily", () => {
	it("reads the vendor from gateway ids and direct providers", () => {
		assert.equal(modelFamily("vercel-ai-gateway", "anthropic/claude-fable-5.1"), "anthropic");
		assert.equal(modelFamily("anthropic", "claude-opus-5.5"), "anthropic");
		assert.equal(modelFamily("vercel-ai-gateway", "openai/gpt-6.1-sol"), "openai");
		assert.equal(modelFamily("openai", "o4-mini"), "openai");
		assert.equal(modelFamily("google-vertex", "claude-opus-5.5"), "anthropic");
		assert.equal(modelFamily("openrouter", "google/gemini-3-pro"), "google");
	});

	it("returns undefined when nothing matches", () => {
		assert.equal(modelFamily("local", "my-finetune"), undefined);
	});
});

describe("sessionNote", () => {
	const fable = { provider: "vercel-ai-gateway", model: "anthropic/claude-fable-5.1" };

	it("flags the same model", () => {
		assert.match(sessionNote({ provider: fable.provider, id: fable.model }, fable) ?? "", /same model/);
	});

	it("flags the same family", () => {
		assert.match(sessionNote({ provider: "vercel-ai-gateway", id: "anthropic/claude-opus-5.5" }, fable) ?? "", /same family as the session \(anthropic\)/);
	});

	it("says nothing across families or without a session model", () => {
		assert.equal(sessionNote({ provider: "vercel-ai-gateway", id: "openai/gpt-6.1-sol" }, fable), undefined);
		assert.equal(sessionNote(undefined, fable), undefined);
		assert.equal(sessionNote({ provider: "local", id: "my-finetune" }, { provider: "local", model: "other" }), undefined);
	});
});

describe("withModelOverride", () => {
	it("keeps the settings when no model is given", () => {
		assert.equal(withModelOverride(DEFAULT_SETTINGS, undefined), DEFAULT_SETTINGS);
		assert.equal(withModelOverride(DEFAULT_SETTINGS, "  "), DEFAULT_SETTINGS);
	});

	it("swaps the model and keeps the configured effort unless given", () => {
		const s = withModelOverride({ ...DEFAULT_SETTINGS, effort: "medium" }, "vercel-ai-gateway/openai/gpt-6.1-sol");
		assert.ok(!("error" in s));
		assert.equal(s.provider, "vercel-ai-gateway");
		assert.equal(s.model, "openai/gpt-6.1-sol");
		assert.equal(s.effort, "medium");
		assert.equal(s.maxTokens, DEFAULT_SETTINGS.maxTokens);
		const t = withModelOverride(DEFAULT_SETTINGS, "openai/gpt-6.1-sol:xhigh");
		assert.ok(!("error" in t));
		assert.equal(t.effort, "xhigh");
	});

	it("rejects a ref without a provider", () => {
		assert.ok("error" in withModelOverride(DEFAULT_SETTINGS, "gpt-6.1-sol"));
	});
});
