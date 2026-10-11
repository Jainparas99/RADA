# Team setup: run Ratchet on the Atlas sandbox with Groq

Everything on `main` runs against the event's Atlas sandbox with free Groq models. Allen
sends the three secrets (two MongoDB URIs, one Groq key) privately. **Never commit them.**

## 1. Get the code

```bash
git checkout main && git pull
uv sync                                   # installs everything, including the openai SDK used for Groq
cp .env.team.example .env                 # .env is gitignored
```

Fill in `MONGODB_URI_RW`, `MONGODB_URI_RO` and `GROQ_API_KEY` in `.env`. Set `RATCHET_NS` to your
own letter (`a`, `b` or `c`) so you don't overwrite anyone else's data. Only the final team run uses `main`.

## 2. Check the connection

```bash
uv run ratchet info --ping
```

You want `ping rw: ok`, `ping ro: ok` and `GROQ_API_KEY set`. If not:

| Symptom | Fix |
|---|---|
| ping times out, or only one of the three Atlas nodes answers | Your public IP isn't on the Atlas IP Access List. The event Wi-Fi has more than one egress IP; compare `curl https://api.ipify.org` and `curl https://api64.ipify.org` and ask Allen to add both. |
| `Authentication failed` | The URI was pasted with the `<` `>` of the `<db_password>` template, or with a typo. |
| `groq:... not callable` | `GROQ_API_KEY` empty, or `uv sync` not run. |
| HTTP 429 from Groq | Shared account limits (~30 requests/min). Wait a minute; don't run eval/evolve on two machines at once. |

## 3. Try it (your namespace)

```bash
uv run ratchet init                        # state DB ratchet_<ns>, gen-0 harness, vector indexes
uv run ratchet seed                        # workloads + scratch copies (~170 MB, ~30 s, server-side)
uv run ratchet measure mflix_w01           # baseline explain table + detector findings (no LLM)
uv run ratchet attack                      # 13 seeded cheats, all must be REJECTED (no LLM)
uv run ratchet tune mflix_w01              # one real agent run: ~20 s, ~4k Groq tokens
uv run ratchet eval --split train          # gen 0 on all 8 train workloads: ~35k tokens
```

What you should see (checked on the sandbox, ns `b`):

- `measure mflix_w01`: workload cost 3,612,860, with COLLSCAN on q2/q4/q5 and in-memory SORTs.
- `attack`: "all 13 cheats rejected".
- `tune mflix_w01`: a plan, before/after explain, a score, `planner=groq:openai/gpt-oss-120b`.
  Gen 0 is deliberately weak (no detector findings in context, critic off), so a small or zero
  improvement is expected. Evolution is what fixes that.

## 4. The team run (one person, `RATCHET_NS=main`)

```bash
RATCHET_NS=main uv run ratchet init
RATCHET_NS=main uv run ratchet seed
RATCHET_NS=main uv run ratchet evolve --gens 2      # each generation: 15 tunes + proposer, ~60-100k tokens
RATCHET_NS=main uv run ratchet history
RATCHET_NS=main uv run ratchet chart                # docs/chart.png
```

Free Groq quotas are per model per day, so 2 generations fit; 8 probably don't. Every LLM call is
cached in `llm_cache`, so `ratchet replay` re-runs a recorded generation for the demo without
spending tokens or hitting rate limits.

## Models

| Role | Model | Why |
|---|---|---|
| planner | `groq:openai/gpt-oss-120b` | got the ESR index order right in a live check (`gpt-oss-20b` did not) |
| critic | `groq:qwen/qwen3.8-27b` | a different model family from the planner, and a separate quota |
| proposer | `groq:openai/gpt-oss-120b` | few calls, long context |

Anthropic models still work: set `ANTHROPIC_API_KEY` and `RATCHET_*_MODEL=anthropic:...`. The
gen-0 allowlist is pinned when you run `init`, so re-run `init` on a fresh namespace after changing models.
