# pi-counsel

Give your [Pi](https://pi.dev) agent a `counsel` tool: a stronger model it
consults before committing to an approach, when stuck, or before declaring
done. The executor keeps the work. Counsel reads a plain-text transcript of
the session and returns a verdict, the risks it sees, numbered next steps,
and what to verify.

This is the shape of Anthropic's server-side
[advisor tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/advisor-tool),
built client-side so it works with any model Pi can reach (including through
a gateway) and shows up in Pi's own cost accounting.

## Install

```bash
pi install npm:pi-counsel
```

Requires Pi 0.99 or later.

## What happens

The agent gets one tool, `counsel`, with an optional `question`. Its
description tells the model when to call it: after orientation and before
committing to an approach on non-trivial work, when the same error recurs or
an approach has stalled after two attempts, before declaring multi-step work
done, or when the user asks. Not for routine steps.

A consult serialises the session (compaction summary, user messages,
assistant text, tool calls with clipped arguments, tool results with clipped
bodies, images as placeholders) into one message per entry, redacts obvious
secrets, sizes it to the counsel model's context window (eliding the middle,
keeping the opening and the recent tail), and streams the answer. The tool
result carries the counsel model's usage, so `/session` and the footer
include it; a `counsel` session entry records model, cost, duration and the
verdict line for later analysis.

Counsel has no tools and takes no actions. Its advice comes back as the tool
result and the executor decides what to adopt.

## Use

| Command | Effect |
|---|---|
| `/counsel` or `/counsel status` | Model, effort, budget used this session, and warnings |
| `/counsel <provider/model[:effort]>` | Set the counsel model (asks for effort in the TUI when omitted) |
| `/counsel pick` | Pick from your scoped models |
| `/counsel on` / `/counsel off` | Add or remove the tool from the agent's active set |
| `/counsel ask <question>` | Ask the agent to consult now with that question |
| `/counsel last` | Reopen the last advice on this branch |
| `/counsel more` | Raise this session's budget by one more allowance |

In a prompt, "consult counsel before you continue" works the way it does with
any tool. The tool also takes an optional `model` (`provider/model[:effort]`)
for one consult, so "have counsel review with an OpenAI model" works without
changing the configured counsel. That consult counts against the same budget.
If the model does not resolve, the error lists available models from the
same family.

## Configure

`~/.pi/agent/counsel.json`, every field optional:

```json
{
  "enabled": true,
  "provider": "vercel-ai-gateway",
  "model": "anthropic/claude-fable-5.1",
  "effort": "high",
  "maxTokens": 16000,
  "maxConsultsPerSession": 5,
  "maxSpendPerSession": 10
}
```

`maxTokens` caps output per consult (thinking plus text). The two budgets
are per session and follow the branch: when either is reached the tool
declines and tells the agent to carry on with its own judgement, and
`/counsel more` grants another allowance for that session.

`PI_COUNSEL` overrides the file for one process without writing to it:

- `PI_COUNSEL=off` skips registration entirely. Use it for worker sessions
  (subagents, familiars, batch jobs) that should not spend on counsel.
- `PI_COUNSEL=provider/model[:effort]` swaps the counsel model.

## Notes

- **Cost.** Each consult reads the transcript at the counsel model's price;
  at 200K tokens on a frontier model that is a few dollars. The result line
  shows tokens, dollars, cache hit rate, and budget used. Consecutive
  consults share a byte-identical prefix, so a provider prompt cache can
  reuse most of it.
- **Privacy.** The transcript goes to the counsel model's provider. When
  that is the same provider as the session, nothing new is exposed. Redaction
  (API-key shapes, bearer tokens, `KEY=`/`TOKEN=` assignments, credentials
  in URLs) is hygiene, not a boundary.
- **Same model.** Counsel can be the session model. `/counsel status` says
  so; the value is a fresh read without the executor's sunk cost, not a
  stronger one.
- **Same family.** A counsel from the session model's lab (Fable reviewing
  Opus) shares its training and its blind spots, and models tend to rate
  their own family's work higher. A comparably strong model from another
  lab is more likely to disagree where the executor is wrong. `/counsel
  status` notes a same-family pair; pass `model` for one cross-family consult. The cost is privacy: the transcript
  then goes to a second provider. A weaker cross-family model adds noise,
  not independence.
- **Routing.** Requests go through `modelRegistry.streamSimple()`, outside
  Pi's agent loop, so `before_provider_request` handlers do not see them.
  For `vercel-ai-gateway` models with `compat.vercelGatewayRouting` in
  `models.json`, the extension applies the pin itself.
- **Time.** A frontier model at high effort on a long transcript can take
  minutes. Esc aborts; the tool description asks the agent to make
  deliverables durable before a completion check.

## Develop

```bash
npm install
npm run gate          # typecheck + tests
pi -e ./extensions/pi-counsel/index.ts   # try it in a session
```

`PI_COUNSEL=vercel-ai-gateway/anthropic/claude-haiku-4.5:low` is a cheap
counsel for exercising the flow.

## License

MIT
