/**
 * One consultation: resolve the counsel model, size and serialise the
 * transcript, stream the answer, and hand back advice plus usage.
 *
 * The request goes through `ctx.modelRegistry.streamSimple()`, which is
 * outside pi's agent loop. Two consequences the code handles here:
 *
 * - `before_provider_request` does not fire for it (pi wires that hook into
 *   the loop's `onPayload`, not the runtime), so a gateway routing pin in
 *   the model's `compat.vercelGatewayRouting` would be dropped. We pass our
 *   own `onPayload` and set `providerOptions.gateway` from the same field.
 * - Nothing accounts for the call unless the tool result carries `usage`.
 *   The caller puts `usage` on the result; pi then rolls it into session
 *   totals under "Tools/summaries".
 * - pi-ai caches only up to the last block, which is our per-consult
 *   request, so the same `onPayload` adds breakpoints inside the
 *   transcript (see transcript.ts).
 */

import type { Api, AssistantMessage, Model, Usage, UserMessage } from "@earendil-works/pi-ai";
import { clampThinkingLevel } from "@earendil-works/pi-ai";
import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import { executorSystemPromptBlock, requestBlock, SYSTEM_PROMPT } from "./prompt.ts";
import type { CounselSettings, Effort } from "./settings.ts";
import { modelKey } from "./settings.ts";
import { estimateTokens, serializeTranscript } from "./transcript.ts";

/** Context and output headroom kept back from the counsel model's window. */
const CONTEXT_MARGIN_TOKENS = 8_000;
const EXECUTOR_SYSTEM_PROMPT_MAX_CHARS = 8_000;
/** Transcript entries per message; see transcript.ts for why. 20-block lookback x 16 covers 320 new entries between consults. */
const CHUNK_ENTRIES = 16;
/** Anthropic rejects requests with more cache breakpoints than this. */
const MAX_BREAKPOINTS = 4;
export const TOOL_NAME = "counsel";

export type CounselErrorKind = "disabled" | "no-model" | "no-auth" | "empty" | "aborted" | "provider";

export class CounselError extends Error {
	readonly kind: CounselErrorKind;
	constructor(message: string, kind: CounselErrorKind) {
		super(message);
		this.name = "CounselError";
		this.kind = kind;
	}
}

export interface ConsultOutcome {
	advice: string;
	usage: Usage | undefined;
	stopReason: AssistantMessage["stopReason"];
	truncated: boolean;
	durationMs: number;
	model: string;
	effort: Effort;
	transcriptTokens: number;
	elidedEntries: number;
	totalEntries: number;
	/** Per-tool-result clip the transcript used, in characters. */
	clipChars: number;
	sameModelAsSession: boolean;
}

export interface ConsultProgress {
	elapsedMs: number;
	phase: "preparing" | "thinking" | "answering";
	chars: number;
}

interface GatewayRouting {
	only?: string[];
	order?: string[];
}

/** Mirror of the gateway-routing extension, applied to our own request. */
export function pinGatewayRouting(payload: unknown, model: Model<Api>): unknown {
	if (model.provider !== "vercel-ai-gateway" || !payload || typeof payload !== "object") return undefined;
	const routing = (model.compat as { vercelGatewayRouting?: GatewayRouting } | undefined)?.vercelGatewayRouting;
	if (!routing || (!routing.only && !routing.order)) return undefined;
	const body = payload as Record<string, unknown>;
	const existing = body.providerOptions as { gateway?: unknown } | undefined;
	if (existing?.gateway) return undefined;
	const gateway: GatewayRouting = {};
	if (routing.only) gateway.only = routing.only;
	if (routing.order) gateway.order = routing.order;
	return { ...body, providerOptions: { ...(existing ?? {}), gateway } };
}

type Block = Record<string, unknown>;
type WireMessage = { role?: string; content?: string | Block[] };

const countBreakpoints = (blocks: unknown): number =>
	Array.isArray(blocks) ? blocks.filter((b) => b && typeof b === "object" && "cache_control" in b).length : 0;

/**
 * Copy the breakpoint pi-ai put on our last message onto the given user
 * messages (stable ones first), without passing Anthropic's limit. Indices
 * count user messages only: pi-ai may append non-user messages (effort
 * markers) after ours. Returns undefined when the payload is not the shape
 * we built.
 */
export function addCacheBreakpoints(payload: unknown, model: Model<Api>, indices: number[], expectedMessages: number): unknown {
	if (model.api !== "anthropic-messages" || !payload || typeof payload !== "object") return undefined;
	const body = payload as { system?: unknown; tools?: unknown; messages?: WireMessage[] };
	const messages = body.messages;
	if (!Array.isArray(messages)) return undefined;
	const userAt = messages.flatMap((m, i) => (m.role === "user" ? [i] : []));
	if (userAt.length !== expectedMessages) return undefined;
	const last = messages[userAt[userAt.length - 1]].content;
	const template = Array.isArray(last) ? (last.at(-1)?.cache_control as Block | undefined) : undefined;
	if (!template) return undefined;
	let budget = MAX_BREAKPOINTS - countBreakpoints(body.system) - countBreakpoints(body.tools) - messages.reduce((n, m) => n + countBreakpoints(m.content), 0);
	const next = [...messages];
	for (const u of indices) {
		if (budget <= 0) break;
		const i = userAt[u];
		const msg = i === undefined ? undefined : next[i];
		if (!msg) continue;
		const blocks: Block[] = typeof msg.content === "string" ? [{ type: "text", text: msg.content }] : [...(msg.content ?? [])];
		const tail = blocks.at(-1);
		if (!tail || tail.type !== "text" || "cache_control" in tail) continue;
		blocks[blocks.length - 1] = { ...tail, cache_control: { ...template } };
		next[i] = { ...msg, content: blocks };
		budget--;
	}
	return { ...body, messages: next };
}

export function resolveCounselModel(ctx: Pick<ExtensionContext, "modelRegistry">, settings: CounselSettings): Model<Api> {
	const model = ctx.modelRegistry.find(settings.provider, settings.model);
	if (!model) throw new CounselError(`counsel model ${modelKey(settings)} is not in the model registry; run /counsel <provider/model> to pick another`, "no-model");
	if (!ctx.modelRegistry.hasConfiguredAuth(model)) throw new CounselError(`no credentials for ${modelKey(settings)}; run /login ${settings.provider} or /counsel <provider/model>`, "no-auth");
	return model;
}

export function transcriptBudget(model: Model<Api>, settings: CounselSettings, fixedChars: number): number {
	const window = model.contextWindow > 0 ? model.contextWindow : 200_000;
	return Math.max(4_000, window - Math.min(settings.maxTokens, model.maxTokens || settings.maxTokens) - CONTEXT_MARGIN_TOKENS - estimateTokens("x".repeat(fixedChars)));
}

const textOf = (message: AssistantMessage): string =>
	message.content
		.filter((c): c is { type: "text"; text: string } => c.type === "text")
		.map((c) => c.text)
		.join("\n")
		.trim();

export async function consult(
	ctx: ExtensionContext,
	settings: CounselSettings,
	question: string | undefined,
	signal: AbortSignal | undefined,
	onProgress?: (p: ConsultProgress) => void,
): Promise<ConsultOutcome> {
	if (!settings.enabled) throw new CounselError("counsel is off; run /counsel on", "disabled");
	const model = resolveCounselModel(ctx, settings);
	const startedAt = Date.now();
	onProgress?.({ elapsedMs: 0, phase: "preparing", chars: 0 });

	const executorPrompt = executorSystemPromptBlock(ctx.getSystemPrompt(), EXECUTOR_SYSTEM_PROMPT_MAX_CHARS);
	const request = requestBlock(question);
	const budget = transcriptBudget(model, settings, SYSTEM_PROMPT.length + executorPrompt.length + request.length);
	const transcript = serializeTranscript(ctx.sessionManager.buildContextEntries(), { budgetTokens: budget, skipToolCalls: [TOOL_NAME], chunkEntries: CHUNK_ENTRIES });
	if (transcript.totalEntries === 0) throw new CounselError("nothing to consult on yet: the session has no conversation", "empty");

	const now = Date.now();
	const messages: UserMessage[] = [
		{ role: "user", content: [{ type: "text", text: executorPrompt }], timestamp: now },
		...transcript.messages.map((m) => ({ role: "user" as const, content: [{ type: "text" as const, text: m.content }], timestamp: now })),
		{ role: "user", content: [{ type: "text", text: request }], timestamp: now },
	];

	const level = settings.effort === "off" ? "off" : clampThinkingLevel(model, settings.effort);
	const reasoning = !model.reasoning || level === "off" ? undefined : level;
	const stream = ctx.modelRegistry.streamSimple(
		model,
		{ systemPrompt: SYSTEM_PROMPT, messages },
		{
			reasoning,
			maxTokens: Math.min(settings.maxTokens, model.maxTokens || settings.maxTokens),
			signal,
			cacheRetention: "short",
			sessionId: `counsel:${ctx.sessionManager.getSessionId()}`,
			// Message 0 is the executor prompt, so transcript index i is wire index i + 1.
			onPayload: (payload, m) => {
				const cached = addCacheBreakpoints(payload, m, transcript.breakpoints.map((i) => i + 1), messages.length);
				return pinGatewayRouting(cached ?? payload, m) ?? cached;
			},
			timeoutMs: 10 * 60_000,
		},
	);

	let chars = 0;
	let phase: ConsultProgress["phase"] = "thinking";
	for await (const event of stream) {
		if (event.type === "text_delta") {
			chars += event.delta.length;
			phase = "answering";
		} else if (event.type === "thinking_delta") {
			phase = "thinking";
		}
		if (event.type === "text_delta" || event.type === "thinking_delta" || event.type === "text_start") {
			onProgress?.({ elapsedMs: Date.now() - startedAt, phase, chars });
		}
	}
	const message = await stream.result();
	const durationMs = Date.now() - startedAt;

	if (message.stopReason === "aborted") throw new CounselError("consultation aborted", "aborted");
	if (message.stopReason === "error") throw new CounselError(`counsel model failed: ${message.errorMessage ?? "unknown error"}`, "provider");

	const advice = textOf(message);
	return {
		advice: advice || "(counsel returned no text)",
		usage: message.usage,
		stopReason: message.stopReason,
		truncated: message.stopReason === "length",
		durationMs,
		model: `${model.provider}/${model.id}`,
		effort: level,
		transcriptTokens: transcript.tokens,
		elidedEntries: transcript.elidedEntries,
		totalEntries: transcript.totalEntries,
		clipChars: transcript.maxToolResultChars,
		sameModelAsSession: ctx.model?.provider === model.provider && ctx.model?.id === model.id,
	};
}
