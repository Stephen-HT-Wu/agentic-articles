---
name: agentic-pipeline-observability
description: >-
  Add per-node timing, token usage, cost estimates, wall-clock totals, and
  per-revision draft comparisons to agentic-articles LangGraph stage pipelines.
  Use when implementing or extending practice/stage*/graph.py, adding metrics,
  run summaries, or draft version diffs.
---

# Agentic Pipeline Observability

## When to apply

Use this skill when working on `practice/stage*/graph.py` in the agentic-articles repo, especially when:

- Adding a new stage (stage 4+)
- User asks for per-agent timing, token counts, total runtime, or revision comparison
- Refactoring `call_llm()` or node wrappers

Reference: `PLAN.md` →「可觀測性（Observability）」

## Required instrumentation

### 1. Global trackers

```python
usage_log: list = []      # each LLM call: node, model, input, output tokens
node_times: dict = {}     # per-node accumulated seconds
revision_events: list = []  # stage2+: write/review cycle events
_current_node = "?"
```

Reset all trackers at the start of each `run_pipeline()` / `__main__`.

### 2. `instrument(name, fn)`

Wrap every graph node:

- Set `_current_node = name` before execution
- Accumulate `time.perf_counter()` delta into `node_times[name]`
- Return node result unchanged

### 3. `call_llm(...)`

Every LLM entry point must append to `usage_log`:

```python
usage_log.append({
    "node": _current_node,
    "model": model,
    "input": response.usage.input_tokens,
    "output": response.usage.output_tokens,
})
```

Extract text from all `type == "text"` blocks (thinking models may return `ThinkingBlock` first).

### 4. End-of-run summary

Print a table with columns:

`節點 | 節點時間 | LLM呼叫 | 輸入tokens | 輸出tokens | 成本USD`

Also print:

- `合計(節點)` — sum of node times and tokens
- `合計(牆鐘)` — wall-clock seconds for entire run

### 5. Persist artifacts to `outputs/` (gitignored)

| File | Content |
|------|---------|
| `run_{slug}_summary.json` | Full report: nodes, tokens, cost, memory_hits, revision_events |
| `draft_{slug}_v0.md`, `v1.md`, … | Each draft revision |
| `draft_{slug}_diff.txt` | Unified diff between adjacent versions |
| `draft_{slug}_revision_comparison.md` | Human-readable version delta summary |

## Revision loop tracking (stage 2+)

When `write` and `chief_editor` form a retry loop:

**In `write`**, after producing draft:

```python
revision_events.append({
    "event": "write",
    "version": len(versions) - 1,
    "retry_count": retry_count,
    "chars": len(draft),
})
```

**In `chief_editor`**, before each `return Command(...)`:

```python
revision_events.append({
    "event": "review",
    "decision": "approve" | "revise" | "revise_max_retry",
    "retry_count": retry_count,
    "draft_version": ...,
})
```

**After run**, print draft version table:

```
版本    字元數    相對上一版
v0        312
v1       1473         +1161
```

## Stage-specific notes

| Stage | Extra observability |
|-------|---------------------|
| 1 | Baseline: first metrics table |
| 2 | `draft_versions` + diff; `dedup` uses LLM |
| 3 | `dedup_embed` should show **0 LLM calls**; `memory_hits` in JSON report |
| 6 | Compare reports across stages to quantify savings |

## PR checklist

- [ ] All nodes wrapped with `instrument()`
- [ ] `call_llm()` logs tokens with correct `node`
- [ ] `print_run_summary(wall_seconds)` at end
- [ ] `save_run_artifacts()` writes JSON + draft versions
- [ ] Stage 2+: revision events and version diff
- [ ] `note.md` updated if outputs change
