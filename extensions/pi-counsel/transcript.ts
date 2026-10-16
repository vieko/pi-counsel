/**
 * Turn the executor's session into what counsel reads.
 *
 * Plain text, not the provider messages: raw forwarding carries block types
 * (thinking signatures, tool_use pairs, images the counsel model may not
 * accept) that another provider rejects. Each session entry becomes one
 * user message with a fixed label, so the request prefix is byte-stable
 * across consults and a provider prompt cache can reuse it.
 *
 * Sizing is model-aware: the caller passes a token budget derived from the
 * counsel model's context window. The serialiser picks the loosest clip tier
 * whose total fits, so a session that fits is read nearly whole; only when
 * the tightest tier does not fit does it elide whole entries from the
 * middle, keeping the opening (the task) and the tail (where the decision
 * is). A fixed 3,000-char clip let counsel see ~5K tokens of sessions whose
 * reads ran to tens of thousands, and reviews then called the executor's
 * code claims unverified. The tier changes only when the session outgrows
 * it, so consecutive consults mostly keep a byte-identical prefix.
 *
 * This module imports nothing from pi so it can be lifted elsewhere.
 */

export interface TranscriptMessage {
	role: "user";
	content: string;
}

export interface SerializeOptions {
	/** Tokens the transcript may occupy. Estimated at 4 chars/token. */
	budgetTokens: number;
	/** Tool calls to leave out, e.g. the in-flight counsel call itself. Their results are still rendered. */
	skipToolCalls?: string[];
	/** Per-entry caps, in characters. Setting any of them pins one tier instead of searching TIERS. */
	maxToolArgsChars?: number;
	maxToolResultChars?: number;
	maxTextChars?: number;
	/** Fraction of the budget reserved for the head when eliding. */
	headFraction?: number;
}

export interface SerializeResult {
	messages: TranscriptMessage[];
	/** Estimated tokens of everything sent. */
	tokens: number;
	/** Entries dropped from the middle. */
	elidedEntries: number;
	/** Entries considered before eliding. */
	totalEntries: number;
	/** Per-result clip that was used, in characters. */
	maxToolResultChars: number;
}

// Loose shapes of what pi's session manager returns. Kept structural so the
// module has no pi import; the extension passes `SessionEntry[]` straight in.
interface ContentPart {
	type?: string;
	text?: string;
	name?: string;
	arguments?: unknown;
	mimeType?: string;
	data?: string;
	thinking?: string;
}
export interface EntryLike {
	type?: string;
	message?: {
		role?: string;
		content?: unknown;
		toolName?: string;
		isError?: boolean;
	};
	summary?: string;
	customType?: string;
}

export const estimateTokens = (text: string): number => Math.ceil(text.length / 4);

interface Caps {
	maxToolArgsChars: number;
	maxToolResultChars: number;
	maxTextChars: number;
}

/**
 * Clip tiers, loosest first. The top tool-result cap matches pi's own 50 KB
 * tool output limit, so in practice it does not clip. Arguments carry edit
 * and write bodies, which a review of changes needs.
 */
export const TIERS: readonly Caps[] = [
	{ maxToolResultChars: 50_000, maxToolArgsChars: 8_000, maxTextChars: 24_000 },
	{ maxToolResultChars: 20_000, maxToolArgsChars: 4_000, maxTextChars: 16_000 },
	{ maxToolResultChars: 8_000, maxToolArgsChars: 2_000, maxTextChars: 12_000 },
	{ maxToolResultChars: 3_000, maxToolArgsChars: 1_500, maxTextChars: 12_000 },
];

const DEFAULTS = {
	skipToolCalls: [] as string[],
	...TIERS[TIERS.length - 1],
	headFraction: 0.25,
};

/** Best-effort. Counsel goes to the same provider as the session; this is hygiene, not a boundary. */
const SECRET_PATTERNS: { re: RegExp; to: string }[] = [
	{ re: /\bsk-ant-[A-Za-z0-9_-]{16,}\b/g, to: "[redacted]" },
	{ re: /\b(?:sk|rk|pk)-[A-Za-z0-9_-]{16,}\b/g, to: "[redacted]" },
	{ re: /\bgh[pousr]_[A-Za-z0-9]{20,}\b/g, to: "[redacted]" },
	{ re: /\bgithub_pat_[A-Za-z0-9_]{20,}\b/g, to: "[redacted]" },
	{ re: /\bxox[abposr]-[A-Za-z0-9-]{10,}\b/g, to: "[redacted]" },
	{ re: /\bAKIA[0-9A-Z]{16}\b/g, to: "[redacted]" },
	{ re: /\bvcp_[A-Za-z0-9]{20,}\b/g, to: "[redacted]" },
	{ re: /\blin_api_[A-Za-z0-9]{20,}\b/g, to: "[redacted]" },
	{ re: /\bAIza[0-9A-Za-z_-]{30,}\b/g, to: "[redacted]" },
	{ re: /\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b/g, to: "[redacted]" },
	{ re: /(\b(?:bearer|basic)\s+)[A-Za-z0-9._~+/=-]{16,}/gi, to: "$1[redacted]" },
	{
		re: /((?:^|[\s"'`])[A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?)[A-Z0-9_]*["']?\s*[=:]\s*["']?)[^\s"'`]{6,}/g,
		to: "$1[redacted]",
	},
	{ re: /((?:postgres(?:ql)?|mysql|redis|mongodb(?:\+srv)?|https?):\/\/[^:\s/@]+:)[^@\s/]+(@)/gi, to: "$1[redacted]$2" },
];

export function redact(text: string): string {
	let out = text;
	for (const { re, to } of SECRET_PATTERNS) out = out.replace(re, to);
	return out;
}

const clip = (text: string, max: number): string => {
	if (text.length <= max) return text;
	const keep = Math.max(0, max - 40);
	const head = Math.floor(keep * 0.7);
	const tail = keep - head;
	return `${text.slice(0, head)}\n[... ${text.length - keep} chars elided ...]\n${text.slice(text.length - tail)}`;
};

const partsOf = (content: unknown): ContentPart[] => {
	if (typeof content === "string") return [{ type: "text", text: content }];
	return Array.isArray(content) ? (content.filter((p) => p && typeof p === "object") as ContentPart[]) : [];
};

const imagePlaceholder = (part: ContentPart): string => {
	const kb = part.data ? Math.round((part.data.length * 3) / 4 / 1024) : 0;
	return `[image ${part.mimeType ?? "unknown"}${kb ? `, ~${kb} KB` : ""}]`;
};

const textOf = (content: unknown, maxChars: number): string =>
	partsOf(content)
		.map((p) => {
			if (p.type === "text" && typeof p.text === "string") return p.text;
			if (p.type === "image") return imagePlaceholder(p);
			return "";
		})
		.filter(Boolean)
		.join("\n")
		.trim()
		.slice(0, Math.max(0, maxChars) || undefined);

/** Render one session entry as labelled blocks. Empty when there is nothing counsel should read. */
export function renderEntry(entry: EntryLike, opts: Required<Omit<SerializeOptions, "budgetTokens">>): string[] {
	const blocks: string[] = [];
	if (entry.type === "compaction" && typeof entry.summary === "string") {
		blocks.push(`[summary of earlier conversation]\n${clip(entry.summary, opts.maxTextChars)}`);
		return blocks;
	}
	if (entry.type === "branch_summary" && typeof entry.summary === "string") {
		blocks.push(`[summary of an earlier branch]\n${clip(entry.summary, opts.maxTextChars)}`);
		return blocks;
	}
	if (entry.type === "custom_message" && entry.message) {
		const text = textOf(entry.message.content, Infinity);
		if (text) blocks.push(`[note]\n${clip(text, opts.maxTextChars)}`);
		return blocks;
	}
	if (entry.type !== "message" || !entry.message) return blocks;
	const { role, content } = entry.message;
	if (role === "user") {
		const text = textOf(content, Infinity);
		if (text) blocks.push(`[user]\n${clip(text, opts.maxTextChars)}`);
		return blocks;
	}
	if (role === "assistant") {
		const texts: string[] = [];
		for (const p of partsOf(content)) {
			if (p.type === "text" && typeof p.text === "string" && p.text.trim()) texts.push(p.text.trim());
			if (p.type === "toolCall" && typeof p.name === "string" && !opts.skipToolCalls.includes(p.name)) {
				let args: string;
				try {
					args = JSON.stringify(p.arguments ?? {});
				} catch {
					args = "[unserializable arguments]";
				}
				blocks.push(`[tool call: ${p.name}]\n${clip(args, opts.maxToolArgsChars)}`);
			}
		}
		if (texts.length) blocks.unshift(`[assistant]\n${clip(texts.join("\n"), opts.maxTextChars)}`);
		return blocks;
	}
	if (role === "toolResult") {
		const text = textOf(content, Infinity);
		const label = `[tool result: ${entry.message.toolName ?? "unknown"}${entry.message.isError ? ", error" : ""}]`;
		blocks.push(`${label}\n${clip(text || "(empty)", opts.maxToolResultChars)}`);
		return blocks;
	}
	return blocks;
}

/**
 * Serialise entries into per-entry user messages under a token budget.
 * Eliding removes whole entries from the middle so the head (task framing)
 * and the tail (current state) survive; a marker says how much went.
 */
export function serializeTranscript(entries: EntryLike[], options: SerializeOptions): SerializeResult {
	const pinned = options.maxToolArgsChars !== undefined || options.maxToolResultChars !== undefined || options.maxTextChars !== undefined;
	const base = { ...DEFAULTS, ...stripUndefined(options), budgetTokens: options.budgetTokens };
	const tiers: Caps[] = pinned ? [{ maxToolArgsChars: base.maxToolArgsChars, maxToolResultChars: base.maxToolResultChars, maxTextChars: base.maxTextChars }] : [...TIERS];
	let opts = { ...base, ...tiers[0] };
	let rendered: string[] = [];
	let sizes: number[] = [];
	let total = 0;
	for (const tier of tiers) {
		opts = { ...base, ...tier };
		rendered = entries.map((entry) => renderEntry(entry, opts).map(redact).join("\n\n")).filter((text) => text.length > 0);
		sizes = rendered.map(estimateTokens);
		total = sizes.reduce((a, b) => a + b, 0);
		if (total <= opts.budgetTokens) {
			return {
				messages: rendered.map((content) => ({ role: "user", content })),
				tokens: total,
				elidedEntries: 0,
				totalEntries: rendered.length,
				maxToolResultChars: opts.maxToolResultChars,
			};
		}
	}

	const headBudget = Math.floor(opts.budgetTokens * opts.headFraction);
	const tailBudget = opts.budgetTokens - headBudget - 40; // room for the marker
	const head: number[] = [];
	let used = 0;
	for (let i = 0; i < rendered.length; i++) {
		if (used + sizes[i] > headBudget) break;
		head.push(i);
		used += sizes[i];
	}
	const tail: number[] = [];
	used = 0;
	for (let i = rendered.length - 1; i > (head.at(-1) ?? -1); i--) {
		if (used + sizes[i] > tailBudget) break;
		tail.unshift(i);
		used += sizes[i];
	}
	const firstTail = tail[0] ?? rendered.length;
	const lastHead = head.at(-1) ?? -1;
	const elided = firstTail - lastHead - 1;
	const elidedTokens = sizes.slice(lastHead + 1, firstTail).reduce((a, b) => a + b, 0);

	const messages: TranscriptMessage[] = [];
	for (const i of head) messages.push({ role: "user", content: rendered[i] });
	if (elided > 0) {
		messages.push({
			role: "user",
			content: `[... ${elided} entries (~${elidedTokens} tokens) elided to fit the counsel model's context; the opening and the most recent work are kept ...]`,
		});
	}
	for (const i of tail) messages.push({ role: "user", content: rendered[i] });

	return {
		messages,
		tokens: messages.reduce((a, m) => a + estimateTokens(m.content), 0),
		elidedEntries: Math.max(0, elided),
		totalEntries: rendered.length,
		maxToolResultChars: opts.maxToolResultChars,
	};
}

function stripUndefined<T extends object>(o: T): Partial<T> {
	return Object.fromEntries(Object.entries(o).filter(([, v]) => v !== undefined)) as Partial<T>;
}
