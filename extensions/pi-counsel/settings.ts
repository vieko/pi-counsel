/**
 * Counsel settings: which model to consult, how hard it thinks, and how much a
 * session may spend on it.
 *
 * Lives at `~/.pi/agent/counsel.json`, outside the extension directory, and
 * every field falls back on its own so a missing, corrupt or half-edited file
 * cannot take the main session down. Writes go to a temp file and are renamed
 * into place.
 *
 * `PI_COUNSEL` overrides the file for one process: `off` disables the tool
 * (familiars and print-mode runs use this), `provider/model[:effort]` swaps
 * the counsel model without touching the file.
 */

import { mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { randomUUID } from "node:crypto";
import type { ThinkingLevel } from "@earendil-works/pi-ai";

export type Effort = "off" | ThinkingLevel;

export interface CounselSettings {
	enabled: boolean;
	provider: string;
	model: string;
	effort: Effort;
	/** Output cap per consult, thinking plus text. */
	maxTokens: number;
	/** Consults allowed per session before the tool asks for `/counsel more`. */
	maxConsultsPerSession: number;
	/** Dollars allowed per session before the tool asks for `/counsel more`. */
	maxSpendPerSession: number;
}

export const EFFORTS: readonly Effort[] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];

export const DEFAULT_SETTINGS: CounselSettings = {
	enabled: true,
	provider: "vercel-ai-gateway",
	model: "anthropic/claude-fable-5.1",
	effort: "high",
	maxTokens: 16_000,
	maxConsultsPerSession: 5,
	maxSpendPerSession: 10,
};

export const SETTINGS_PATH = join(process.env.PI_CODING_AGENT_DIR ?? join(homedir(), ".pi", "agent"), "counsel.json");

const isEffort = (value: unknown): value is Effort => typeof value === "string" && (EFFORTS as readonly string[]).includes(value);

const positive = (value: unknown, fallback: number, min: number): number =>
	typeof value === "number" && Number.isFinite(value) && value >= min ? value : fallback;

export function parseSettings(value: unknown): CounselSettings {
	if (!value || typeof value !== "object") return { ...DEFAULT_SETTINGS };
	const raw = value as Record<string, unknown>;
	return {
		enabled: typeof raw.enabled === "boolean" ? raw.enabled : DEFAULT_SETTINGS.enabled,
		provider: typeof raw.provider === "string" && raw.provider ? raw.provider : DEFAULT_SETTINGS.provider,
		model: typeof raw.model === "string" && raw.model ? raw.model : DEFAULT_SETTINGS.model,
		effort: isEffort(raw.effort) ? raw.effort : DEFAULT_SETTINGS.effort,
		maxTokens: positive(raw.maxTokens, DEFAULT_SETTINGS.maxTokens, 1000),
		maxConsultsPerSession: positive(raw.maxConsultsPerSession, DEFAULT_SETTINGS.maxConsultsPerSession, 0),
		maxSpendPerSession: positive(raw.maxSpendPerSession, DEFAULT_SETTINGS.maxSpendPerSession, 0),
	};
}

/**
 * Parse `provider/model[:effort]`. The model id may itself contain slashes
 * (`vercel-ai-gateway/anthropic/claude-fable-5.1`), so the provider is the
 * first segment and the effort is the last `:` suffix when it names a level.
 */
export function parseModelRef(ref: string): { provider: string; model: string; effort?: Effort } | undefined {
	const trimmed = ref.trim();
	const slash = trimmed.indexOf("/");
	if (slash <= 0 || slash === trimmed.length - 1) return undefined;
	let rest = trimmed.slice(slash + 1);
	let effort: Effort | undefined;
	const colon = rest.lastIndexOf(":");
	if (colon > 0 && isEffort(rest.slice(colon + 1))) {
		effort = rest.slice(colon + 1) as Effort;
		rest = rest.slice(0, colon);
	}
	if (!rest) return undefined;
	return { provider: trimmed.slice(0, slash), model: rest, effort };
}

export type EnvOverride = { kind: "off" } | { kind: "model"; provider: string; model: string; effort?: Effort } | undefined;

export function parseEnvOverride(value: string | undefined): EnvOverride {
	if (value === undefined) return undefined;
	const v = value.trim();
	if (!v || v === "on" || v === "1" || v === "true") return undefined;
	if (v === "off" || v === "0" || v === "false") return { kind: "off" };
	const ref = parseModelRef(v);
	return ref ? { kind: "model", ...ref } : undefined;
}

/** What the file says, before any `PI_COUNSEL` override. Edits start from this. */
export function loadFileSettings(): CounselSettings {
	try {
		return parseSettings(JSON.parse(readFileSync(SETTINGS_PATH, "utf8")));
	} catch {
		return { ...DEFAULT_SETTINGS };
	}
}

/** Effective settings for this process: the file with `PI_COUNSEL` applied. */
export function loadSettings(env: NodeJS.ProcessEnv = process.env): CounselSettings {
	const settings = loadFileSettings();
	const override = parseEnvOverride(env.PI_COUNSEL);
	if (override?.kind === "off") settings.enabled = false;
	if (override?.kind === "model") {
		settings.provider = override.provider;
		settings.model = override.model;
		if (override.effort) settings.effort = override.effort;
	}
	return settings;
}

export function saveSettings(settings: CounselSettings): boolean {
	try {
		mkdirSync(dirname(SETTINGS_PATH), { recursive: true });
		const tmp = `${SETTINGS_PATH}.${process.pid}.${randomUUID()}.tmp`;
		writeFileSync(tmp, `${JSON.stringify(settings, null, 2)}\n`);
		renameSync(tmp, SETTINGS_PATH);
		return true;
	} catch {
		return false;
	}
}

export const modelKey = (s: Pick<CounselSettings, "provider" | "model">): string => `${s.provider}/${s.model}`;

const FAMILY_HINTS: readonly [RegExp, string][] = [
	[/claude|anthropic/, "anthropic"],
	[/(^|[/.])(gpt|o\d|codex)|openai/, "openai"],
	[/gemini|gemma|google/, "google"],
	[/grok|xai/, "xai"],
	[/llama|meta/, "meta"],
	[/mistral|codestral|devstral/, "mistral"],
	[/deepseek/, "deepseek"],
	[/qwen|alibaba/, "alibaba"],
	[/kimi|moonshot/, "moonshot"],
	[/glm|zai/, "zai"],
];

/**
 * The lab behind a model, guessed from its id and provider. Gateways and
 * routers carry no vendor in the provider name, so the id is checked first.
 * Undefined when nothing matches: no note beats a wrong one.
 */
export function modelFamily(provider: string, model: string): string | undefined {
	for (const text of [model.toLowerCase(), provider.toLowerCase()]) {
		for (const [pattern, family] of FAMILY_HINTS) if (pattern.test(text)) return family;
	}
	return undefined;
}

/** How counsel relates to the session model, when that limits the review. */
export function sessionNote(session: { provider: string; id: string } | undefined, counsel: Pick<CounselSettings, "provider" | "model">): string | undefined {
	if (!session) return undefined;
	if (session.provider === counsel.provider && session.id === counsel.model) return "same model as the session: a fresh read, not a stronger one";
	const family = modelFamily(counsel.provider, counsel.model);
	if (family && family === modelFamily(session.provider, session.id)) return `same family as the session (${family}): blind spots overlap; another family gives a more independent read`;
	return undefined;
}
