/**
 * pi-counsel: a `counsel` tool the agent calls to consult a stronger model
 * on the work so far, plus `/counsel` to configure it.
 *
 * The executor keeps the work. Counsel reads a plain-text transcript of the
 * session and returns a verdict, risks, next steps, and what to verify. It
 * runs only when the agent (or the user via `/counsel ask`) asks; there are
 * no automatic gates.
 *
 * Budget follows the branch: consults and dollars are summed from the
 * session's own `counsel` tool results, so `/tree` and resume see the
 * right numbers without in-memory state. `/counsel more` appends a
 * `counsel_budget` entry that raises the allowance for this session.
 */

import type { AgentToolResult } from "@earendil-works/pi-agent-core";
import type { ExtensionAPI, ExtensionCommandContext, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { DynamicBorder, getMarkdownTheme } from "@earendil-works/pi-coding-agent";
import { getSupportedThinkingLevels, type Api, type Model } from "@earendil-works/pi-ai";
import { Container, Markdown, matchesKey, Text } from "@earendil-works/pi-tui";
import { Type } from "typebox";
import { consult, CounselError, resolveCounselModel, TOOL_NAME, type ConsultProgress } from "./consult.ts";
import { PROMPT_GUIDELINES, PROMPT_SNIPPET, TOOL_DESCRIPTION } from "./prompt.ts";
import { type CounselSettings, type Effort, EFFORTS, loadFileSettings, loadSettings, modelKey, parseEnvOverride, parseModelRef, saveSettings, SETTINGS_PATH } from "./settings.ts";

const BUDGET_ENTRY = "counsel_budget";
const RECORD_ENTRY = "counsel";

export interface CounselDetails {
	model: string;
	effort: Effort;
	question?: string;
	durationMs: number;
	cost: number;
	tokens: { input: number; cacheRead: number; cacheWrite: number; output: number };
	transcriptTokens: number;
	elidedEntries: number;
	totalEntries: number;
	truncated: boolean;
	sameModelAsSession: boolean;
	budget: Budget;
	/** Streaming only. */
	phase?: ConsultProgress["phase"];
	error?: string;
}

interface Budget {
	consults: number;
	maxConsults: number;
	spend: number;
	maxSpend: number;
}

interface BudgetGrant {
	consults?: number;
	spend?: number;
}

const money = (n: number): string => `$${n.toFixed(n >= 10 ? 0 : n >= 1 ? 2 : 3)}`;
const seconds = (ms: number): string => `${Math.round(ms / 1000)}s`;
const kilo = (n: number): string => (n >= 1000 ? `${(n / 1000).toFixed(n >= 10_000 ? 0 : 1)}K` : String(n));
const firstLine = (text: string): string => text.split("\n").find((l) => l.trim())?.replace(/^\*\*Verdict:\*\*\s*/i, "").trim() ?? "";

/** Sum this session's consults from the branch, plus any grants. */
function readBudget(ctx: Pick<ExtensionContext, "sessionManager">, settings: CounselSettings): Budget {
	let consults = 0;
	let spend = 0;
	let extraConsults = 0;
	let extraSpend = 0;
	for (const entry of ctx.sessionManager.getBranch()) {
		if (entry.type === "message" && entry.message.role === "toolResult" && entry.message.toolName === TOOL_NAME && !entry.message.isError) {
			consults += 1;
			const cost = (entry.message.details as Partial<CounselDetails> | undefined)?.cost;
			if (typeof cost === "number") spend += cost;
		} else if (entry.type === "custom" && entry.customType === BUDGET_ENTRY) {
			const grant = entry.data as BudgetGrant | undefined;
			extraConsults += grant?.consults ?? 0;
			extraSpend += grant?.spend ?? 0;
		}
	}
	return { consults, spend, maxConsults: settings.maxConsultsPerSession + extraConsults, maxSpend: settings.maxSpendPerSession + extraSpend };
}

const overBudget = (b: Budget): string | undefined => {
	if (b.consults >= b.maxConsults) return `counsel budget reached: ${b.consults}/${b.maxConsults} consults this session`;
	if (b.spend >= b.maxSpend) return `counsel budget reached: ${money(b.spend)} of ${money(b.maxSpend)} this session`;
	return undefined;
};

export default function counselExtension(pi: ExtensionAPI) {
	const envOverride = parseEnvOverride(process.env.PI_COUNSEL);
	if (envOverride?.kind === "off") return;

	let settings = loadSettings();

	const syncActive = (enabled: boolean) => {
		const active = pi.getActiveTools();
		const has = active.includes(TOOL_NAME);
		if (enabled && !has) pi.setActiveTools([...active, TOOL_NAME]);
		if (!enabled && has) pi.setActiveTools(active.filter((t) => t !== TOOL_NAME));
	};

	pi.registerTool<typeof Params, CounselDetails>({
		name: TOOL_NAME,
		label: "counsel",
		description: TOOL_DESCRIPTION,
		promptSnippet: PROMPT_SNIPPET,
		promptGuidelines: PROMPT_GUIDELINES,
		parameters: Params,
		exposure: "model-only",
		defaultActive: settings.enabled,
		annotations: { readOnlyHint: true, openWorldHint: true },
		async execute(_id, params, signal, onUpdate, ctx) {
			const question = params.question?.trim() || undefined;
			const budget = readBudget(ctx, settings);
			const base: CounselDetails = {
				model: modelKey(settings),
				effort: settings.effort,
				question,
				durationMs: 0,
				cost: 0,
				tokens: { input: 0, cacheRead: 0, cacheWrite: 0, output: 0 },
				transcriptTokens: 0,
				elidedEntries: 0,
				totalEntries: 0,
				truncated: false,
				sameModelAsSession: false,
				budget,
			};
			const fail = (message: string): AgentToolResult<CounselDetails> => ({
				content: [{ type: "text", text: message }],
				details: { ...base, error: message },
				isError: true,
			});

			if (!settings.enabled) return fail("counsel is off. Ask the user to run /counsel on if a consultation is wanted.");
			const blocked = overBudget(budget);
			if (blocked) return fail(`${blocked}. Continue with your own judgement, or ask the user to run /counsel more.`);

			let outcome;
			try {
				outcome = await consult(ctx, settings, question, signal, (p) => {
					onUpdate?.({
						content: [{ type: "text", text: `consulting ${modelKey(settings)}… ${seconds(p.elapsedMs)}` }],
						details: { ...base, durationMs: p.elapsedMs, phase: p.phase },
					});
				});
			} catch (err) {
				if (err instanceof CounselError) return fail(err.message);
				return fail(`counsel failed: ${err instanceof Error ? err.message : String(err)}`);
			}

			const usage = outcome.usage;
			const details: CounselDetails = {
				...base,
				model: outcome.model,
				effort: outcome.effort,
				durationMs: outcome.durationMs,
				cost: usage?.cost.total ?? 0,
				tokens: { input: usage?.input ?? 0, cacheRead: usage?.cacheRead ?? 0, cacheWrite: usage?.cacheWrite ?? 0, output: usage?.output ?? 0 },
				transcriptTokens: outcome.transcriptTokens,
				elidedEntries: outcome.elidedEntries,
				totalEntries: outcome.totalEntries,
				truncated: outcome.truncated,
				sameModelAsSession: outcome.sameModelAsSession,
				budget: { ...budget, consults: budget.consults + 1, spend: budget.spend + (usage?.cost.total ?? 0) },
			};
			pi.appendEntry(RECORD_ENTRY, {
				model: details.model,
				effort: details.effort,
				question,
				cost: details.cost,
				tokens: details.tokens,
				durationMs: details.durationMs,
				transcriptTokens: details.transcriptTokens,
				elidedEntries: details.elidedEntries,
				truncated: details.truncated,
				verdict: firstLine(outcome.advice),
			});

			const notes: string[] = [];
			if (outcome.truncated) notes.push("[counsel hit its output cap; the advice above may be incomplete]");
			if (outcome.elidedEntries > 0) notes.push(`[${outcome.elidedEntries} of ${outcome.totalEntries} transcript entries were elided from the middle to fit counsel's context]`);
			const text = notes.length ? `${outcome.advice}\n\n${notes.join("\n")}` : outcome.advice;

			return { content: [{ type: "text", text }], details, usage };
		},
		renderCall(args, theme) {
			let line = theme.fg("toolTitle", theme.bold("counsel ")) + theme.fg("muted", modelKey(settings));
			if (args.question) line += ` ${theme.fg("dim", `"${args.question}"`)}`;
			return new Text(line, 0, 0);
		},
		renderResult(result, { expanded, isPartial }, theme) {
			const d = result.details;
			if (isPartial) {
				const phase = d?.phase === "answering" ? "answering" : d?.phase === "thinking" ? "thinking" : "preparing";
				return new Text(theme.fg("dim", `counsel ${phase}… ${seconds(d?.durationMs ?? 0)}`), 0, 0);
			}
			const advice = result.content.find((c): c is { type: "text"; text: string } => c.type === "text")?.text ?? "";
			if (result.isError || d?.error) return new Text(theme.fg("error", d?.error ?? advice), 0, 0);
			const header = [
				d.model.split("/").slice(-1)[0],
				d.effort,
				seconds(d.durationMs),
				money(d.cost),
				`${kilo(d.tokens.input + d.tokens.cacheRead + d.tokens.cacheWrite)}→${kilo(d.tokens.output)}`,
				d.tokens.cacheRead > 0 ? `${Math.round((100 * d.tokens.cacheRead) / Math.max(1, d.tokens.input + d.tokens.cacheRead + d.tokens.cacheWrite))}% cached` : undefined,
				`${d.budget.consults}/${d.budget.maxConsults} · ${money(d.budget.spend)}/${money(d.budget.maxSpend)}`,
			]
				.filter(Boolean)
				.join(" · ");
			const container = new Container();
			container.addChild(new Text(theme.fg("dim", header), 0, 0));
			if (expanded) {
				container.addChild(new Markdown(advice, 0, 0, getMarkdownTheme()));
			} else {
				container.addChild(new Text(firstLine(advice) + theme.fg("dim", "  (expand to read)"), 0, 0));
			}
			if (d.truncated) container.addChild(new Text(theme.fg("warning", "[advice truncated at the output cap]"), 0, 0));
			return container;
		},
	});

	pi.on("session_start", () => {
		settings = loadSettings();
		syncActive(settings.enabled);
	});

	const setModel = async (ctx: ExtensionCommandContext, ref: { provider: string; model: string; effort?: Effort }) => {
		const model = ctx.modelRegistry.find(ref.provider, ref.model);
		if (!model) {
			ctx.ui.notify(`${ref.provider}/${ref.model} is not an available model`, "warning");
			return;
		}
		const effort = ref.effort ?? (await pickEffort(ctx, model)) ?? settings.effort;
		const next: CounselSettings = { ...loadFileSettings(), provider: ref.provider, model: ref.model, effort, enabled: true };
		persist(ctx, next, `counsel: ${modelKey(next)} · ${next.effort}`);
	};

	// Edits are applied to the file's settings, never to the env-overridden
	// effective ones, so a familiar running with PI_COUNSEL=<model> cannot
	// write that model into the shared config.
	const persist = (ctx: ExtensionCommandContext, next: CounselSettings, message: string) => {
		if (envOverride?.kind === "model") {
			ctx.ui.notify("PI_COUNSEL is set for this process; the file was updated but the env override still wins here", "warning");
		}
		if (!saveSettings(next)) {
			ctx.ui.notify(`could not write ${SETTINGS_PATH}; nothing changed`, "error");
			return;
		}
		settings = loadSettings();
		syncActive(settings.enabled);
		ctx.ui.notify(message, "info");
	};

	const pickEffort = async (ctx: ExtensionCommandContext, model: Model<Api>): Promise<Effort | undefined> => {
		if (!ctx.hasUI) return undefined;
		const levels = model.reasoning ? getSupportedThinkingLevels(model) : (["off"] as Effort[]);
		if (levels.length <= 1) return levels[0];
		const picked = await ctx.ui.select(`Effort for ${model.provider}/${model.id}`, levels);
		return picked && (EFFORTS as readonly string[]).includes(picked) ? (picked as Effort) : undefined;
	};

	const status = (ctx: ExtensionContext): string => {
		const b = readBudget(ctx, settings);
		const parts = [settings.enabled ? "on" : "off", `${modelKey(settings)} · ${settings.effort}`, `${b.consults}/${b.maxConsults} consults · ${money(b.spend)}/${money(b.maxSpend)} this session`];
		if (settings.enabled) {
			try {
				resolveCounselModel(ctx, settings);
			} catch (err) {
				parts.push(err instanceof Error ? err.message : String(err));
			}
			if (ctx.model && ctx.model.provider === settings.provider && ctx.model.id === settings.model) parts.push("same model as the session: a fresh read, not a stronger one");
		}
		if (envOverride?.kind === "model") parts.push("model set by PI_COUNSEL for this process");
		return `counsel: ${parts.join(" · ")}`;
	};

	const showLast = async (ctx: ExtensionCommandContext) => {
		const branch = ctx.sessionManager.getBranch();
		for (let i = branch.length - 1; i >= 0; i--) {
			const entry = branch[i];
			if (entry.type !== "message" || entry.message.role !== "toolResult" || entry.message.toolName !== TOOL_NAME || entry.message.isError) continue;
			const text = entry.message.content.find((c): c is { type: "text"; text: string } => c.type === "text")?.text ?? "";
			const d = entry.message.details as Partial<CounselDetails> | undefined;
			if (ctx.mode !== "tui") {
				ctx.ui.notify(text, "info");
				return;
			}
			await ctx.ui.custom((_tui, theme, _kb, done) => {
				const container = new Container();
				const border = new DynamicBorder((s: string) => theme.fg("accent", s));
				container.addChild(border);
				container.addChild(new Text(theme.fg("accent", theme.bold(`counsel · ${d?.model ?? ""} · ${d ? money(d.cost ?? 0) : ""}`)), 1, 0));
				container.addChild(new Markdown(text, 1, 1, getMarkdownTheme()));
				container.addChild(new Text(theme.fg("dim", "Enter or Esc to close"), 1, 0));
				container.addChild(border);
				return {
					render: (width: number) => container.render(width),
					invalidate: () => container.invalidate(),
					handleInput: (data: string) => {
						if (matchesKey(data, "enter") || matchesKey(data, "escape")) done(undefined);
					},
				};
			});
			return;
		}
		ctx.ui.notify("no consultation on this branch yet", "info");
	};

	pi.registerCommand("counsel", {
		description: "Configure counsel: on|off|status|pick|more|ask <question>|last|<provider/model[:effort]>",
		getArgumentCompletions: (prefix) => {
			const words = ["status", "on", "off", "pick", "more", "ask ", "last"];
			const hits = words.filter((w) => w.startsWith(prefix)).map((w) => ({ value: w, label: w.trim() }));
			return hits.length ? hits : null;
		},
		handler: async (args, ctx) => {
			const [verb, ...rest] = args.trim().split(/\s+/);
			switch (verb) {
				case "":
				case "status":
					ctx.ui.notify(status(ctx), "info");
					return;
				case "on":
					persist(ctx, { ...loadFileSettings(), enabled: true }, `counsel on: ${modelKey(settings)} · ${settings.effort}`);
					return;
				case "off":
					persist(ctx, { ...loadFileSettings(), enabled: false }, "counsel off");
					return;
				case "more": {
					const grant: BudgetGrant = { consults: settings.maxConsultsPerSession, spend: settings.maxSpendPerSession };
					pi.appendEntry(BUDGET_ENTRY, grant);
					const b = readBudget(ctx, settings);
					ctx.ui.notify(`counsel budget raised: ${b.consults}/${b.maxConsults} consults · ${money(b.spend)}/${money(b.maxSpend)}`, "info");
					return;
				}
				case "ask": {
					const question = rest.join(" ").trim();
					if (!question) {
						ctx.ui.notify("usage: /counsel ask <question>", "warning");
						return;
					}
					if (!settings.enabled) {
						ctx.ui.notify("counsel is off; run /counsel on first", "warning");
						return;
					}
					pi.sendUserMessage(`Call the counsel tool now with this question, then act on the advice: ${question}`);
					return;
				}
				case "last":
					await showLast(ctx);
					return;
				case "pick": {
					if (!ctx.hasUI) {
						ctx.ui.notify("pick needs an interactive session; use /counsel <provider/model[:effort]>", "warning");
						return;
					}
					const pool = ctx.scopedModels.length ? ctx.scopedModels.map((s) => s.model) : ctx.modelRegistry.getAvailable();
					const options = pool.filter((m) => ctx.modelRegistry.hasConfiguredAuth(m)).map((m) => `${m.provider}/${m.id}`);
					if (!options.length) {
						ctx.ui.notify("no models with credentials to pick from", "warning");
						return;
					}
					const picked = await ctx.ui.select("Counsel model", options);
					if (!picked) return;
					const ref = parseModelRef(picked);
					if (ref) await setModel(ctx, ref);
					return;
				}
				default: {
					const ref = parseModelRef(args);
					if (!ref) {
						ctx.ui.notify(`"${args}" is not a subcommand or a provider/model[:effort]`, "warning");
						return;
					}
					await setModel(ctx, ref);
				}
			}
		},
	});
}

const Params = Type.Object({
	question: Type.Optional(Type.String({ description: "Optional focus for the reading. Leave empty for a general review of the approach and current state." })),
});
