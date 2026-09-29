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
	const transcript = serializeTranscript(ctx.sessionManager.buildContextEntries(), { budgetTokens: budget, skipToolCalls: [TOOL_NAME] });
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
			onPayload: pinGatewayRouting,
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
		sameModelAsSession: ctx.model?.provider === model.provider && ctx.model?.id === model.id,
	};
}
