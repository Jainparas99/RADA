# Demo script (≈ 3 minutes)

Two terminal panes, large font, `RATCHET_NS=main`, the full evolution run already done.
Left pane: commands. Right pane: `uv run ratchet watch` (started before you begin).

## Before going on stage

```bash
uv run ratchet info --ping              # both URIs ok, keys set
uv run ratchet history                  # the full run is there
uv run ratchet chart                    # docs/chart.png exists and looks right
uv run ratchet attack                   # every row REJECTED
uv run ratchet tune mflix_w02           # warms llm_cache, so the live tune is instant
```

Open `docs/chart.png` in an image viewer, ready to alt-tab.

## Script

| Time | Say | Run / show |
|---|---|---|
| 0:00–0:25 **Problem** | "Apps and agents fire queries at MongoDB all day. This one returns 10 documents and reads 23,000 (use the real numbers from the table). Our detectors say why: a collection scan and an in-memory sort." | `uv run ratchet measure mflix_w02` |
| 0:25–1:00 **Inside** | "The agent plans a fix. A critic, a different model, reviews it, here it catches a wrong key order. We apply on a scratch copy and re-measure: collection scan to index scan, cost down N-fold (read the real factor off the table), results identical." | `uv run ratchet tune mflix_w02` |
| 1:00–1:50 **Outside** | "That's one workload. The product is the harness: rules, context, memory, model routing. It patches itself from train traces, and a deterministic gate promotes a version only if workloads it has never seen improve." Point at the right pane as one generation runs. "Here's the full run: holdout climbs, rejected candidates carry their reason code." Show one accepted diff and its diagnosis. | `uv run ratchet evolve --gens 1` (right pane shows it live) → `uv run ratchet history` → alt-tab to `docs/chart.png` → `uv run ratchet diff <v-1> <v>` |
| 1:50–2:30 **Can't cheat** | "Faster isn't enough. These cheats are seeded on purpose. Narrow a filter: result mismatch. Index everything: over budget. `$out` the answers: forbidden. And at the harness level: touch the budget, leak a holdout name, overfit train, every one rejected with its code." | `uv run ratchet attack` |
| 2:30–2:50 **Model wrangling** | "A harness evolved with one model, tested on others: does the learning transfer?" Read one row. | `uv run ratchet transfer` (or `cat docs/transfer.md` if already run) |
| 2:50–3:00 **Close** | "Everything ran on Atlas: versions, traces, vector memory, the LLM cache, change streams. All built today." | — |

If asked "Doesn't Atlas Performance Advisor do this?": it suggests indexes from fixed heuristics.
Ratchet also rewrites pipelines, learns across workloads, measurably improves on unseen ones, and
its own improvement loop is gated against cheating.

## If the live run fails

| Symptom | Fallback |
|---|---|
| Provider down / rate-limited | `RATCHET_LLM_CACHE_ONLY=1 uv run ratchet tune mflix_w02` (served from `llm_cache`) |
| `evolve` too slow or erroring | Skip it; show `ratchet history` + the chart from the full run, then `uv run ratchet replay --gen <n>` (no provider calls) |
| Atlas unreachable | `uv run ratchet attack --fake` (offline, same cheats, same codes) and the recorded 60-second video |
| `watch` shows nothing | `uv run ratchet watch --poll` |
| Anything else | Play the 60-second video; narrate over it |

## 60-second video

| Time | Show | Say |
|---|---|---|
| 0–10 s | `ratchet measure mflix_w02` | "Returns 10, reads 23,000." |
| 10–25 s | `ratchet tune mflix_w02` | "Agent plans, a second model critiques, re-measured on a scratch copy: N-fold less work (real number), same results." |
| 25–40 s | `docs/chart.png` + `ratchet history` | "The harness improves itself, and is promoted only when unseen workloads improve." |
| 40–55 s | `ratchet attack` | "Seeded cheats at both levels, every one rejected with a reason code." |
| 55–60 s | `ratchet transfer` table | "Learned with one model, tested on others. All on Atlas, built today." |
