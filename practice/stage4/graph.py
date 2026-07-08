"""
階段 4：平行化與壓縮

目標（PLAN.md）：
1. 多來源平行爬蟲：3 個來源（HN、GitHub、arXiv）改用 ThreadPoolExecutor 平行抓取，
   並記錄「平行牆鐘時間」vs「序列時間加總」，量出實際加速倍數
2. 壓縮節點：借鑑 open_deep_research 的 compress_research，把 authority 篩選後
   的完整摘要（尤其 arXiv 摘要很長）壓縮成精簡筆記，再送進 synthesize，
   並用官方 count_tokens 端點量出壓縮前後的 token 數差異

流程：
    crawl -> dedup_embed -> authority -> compress -> recall_memory -> synthesize
         -> write -> chief_editor（可退回 write）

執行範例：
    python stage4/graph.py "AI agents" 1              # 平行爬蟲（預設）
    python stage4/graph.py "AI agents" 1 --sequential # 序列爬蟲，跟上面比較耗時
"""

import warnings

import langchain_core  # noqa: F401
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

import argparse
import difflib
import hashlib
import json
import math
import os
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, TypedDict

import sys

import anthropic
import requests
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

# 共用基礎設施（只有經過驗證完全一樣的 instrument/cost_of/PRICING，見 _common.py 開頭說明）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import PRICING, cost_of, current_node, instrument, node_times, reset_metrics

# ---------------------------------------------------------------------------
# 環境設定
# ---------------------------------------------------------------------------

_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _value = _line.partition("=")
            os.environ.setdefault(_key.strip(), _value.strip())

CHEAP_MODEL = "claude-haiku-4-5-20251001"
SMART_MODEL = "claude-sonnet-5"

client = anthropic.Anthropic()

MEMORY_DIR = Path(__file__).resolve().parent / "memory"
MEMORY_FILE = MEMORY_DIR / "topic_memory.json"

DEDUP_SIMILARITY_THRESHOLD = 0.80
MEMORY_SIMILARITY_THRESHOLD = 0.35
EMBED_DIM = 256

usage_log: list = []
revision_events: list = []
editor_reviews: list = []


def print_run_summary(total_wall_s: float) -> None:
    print(f"\n{'=' * 72}")
    print("執行總結")
    print(f"{'-' * 72}")
    print(
        f"{'節點':<16}{'節點時間':>10}{'LLM呼叫':>8}"
        f"{'輸入tokens':>12}{'輸出tokens':>12}{'成本USD':>12}"
    )
    total_cost = 0.0
    for name in node_times:
        calls = [entry for entry in usage_log if entry["node"] == name]
        tokens_in = sum(entry["input"] for entry in calls)
        tokens_out = sum(entry["output"] for entry in calls)
        cost = sum(cost_of(entry) for entry in calls)
        total_cost += cost
        print(
            f"{name:<16}{node_times[name]:>9.1f}s{len(calls):>8}"
            f"{tokens_in:>12,}{tokens_out:>12,}{cost:>12.4f}"
        )
    total_node_s = sum(node_times.values())
    total_in = sum(entry["input"] for entry in usage_log)
    total_out = sum(entry["output"] for entry in usage_log)
    print(f"{'-' * 72}")
    print(
        f"{'合計(節點)':<16}{total_node_s:>9.1f}s{len(usage_log):>8}"
        f"{total_in:>12,}{total_out:>12,}{total_cost:>12.4f}"
    )
    print(f"{'合計(牆鐘)':<16}{total_wall_s:>9.1f}s")


def print_stage4_highlights(result: dict) -> None:
    """階段 4 特有的兩個觀察重點：平行加速倍數、壓縮省下的 token 比例。"""
    print(f"\n{'=' * 72}")
    print("階段 4 觀察重點")
    print(f"{'-' * 72}")
    parallelism = result.get("crawl_parallelism") or {}
    if parallelism:
        print(
            f"爬蟲：{parallelism['mode']}｜各來源耗時 {parallelism['source_seconds']}"
        )
        print(
            f"      序列預估 {parallelism['sequential_estimate_s']}s vs "
            f"實際牆鐘 {parallelism['wall_s']}s（加速 {parallelism['speedup_x']}x）"
        )
    stats = result.get("compression_stats") or {}
    if stats:
        print(
            f"壓縮：原始 {stats['raw_tokens']:,} tokens -> "
            f"壓縮後 {stats['compressed_tokens']:,} tokens"
            f"（省 {stats['saved_tokens']:,} tokens，{stats['saved_pct']}%）"
        )


def build_run_report(result: dict, total_wall_s: float) -> dict:
    nodes = []
    total_cost = 0.0
    for name in node_times:
        calls = [entry for entry in usage_log if entry["node"] == name]
        tokens_in = sum(entry["input"] for entry in calls)
        tokens_out = sum(entry["output"] for entry in calls)
        cost = sum(cost_of(entry) for entry in calls)
        total_cost += cost
        nodes.append(
            {
                "node": name,
                "seconds": round(node_times[name], 3),
                "llm_calls": len(calls),
                "input_tokens": tokens_in,
                "output_tokens": tokens_out,
                "cost_usd": round(cost, 6),
            }
        )

    versions = result.get("draft_versions") or []
    version_stats = [
        {"version": i, "chars": len(text), "file": f"draft_v{i}.md"}
        for i, text in enumerate(versions)
    ]
    for i in range(1, len(versions)):
        version_stats[i]["delta_chars"] = len(versions[i]) - len(versions[i - 1])

    return {
        "topic": result.get("topic"),
        "wall_seconds": round(total_wall_s, 3),
        "node_seconds_total": round(sum(node_times.values()), 3),
        "llm_calls_total": len(usage_log),
        "input_tokens_total": sum(entry["input"] for entry in usage_log),
        "output_tokens_total": sum(entry["output"] for entry in usage_log),
        "cost_usd_total": round(total_cost, 6),
        "nodes": nodes,
        "draft_versions": version_stats,
        "revision_events": revision_events,
        "editor_reviews": editor_reviews,
        "memory_hits": result.get("memory_hits", []),
        "dedup_method": "embedding",
        "crawl_parallelism": result.get("crawl_parallelism", {}),
        "compression_stats": result.get("compression_stats", {}),
    }


def print_draft_version_summary(versions: list) -> None:
    if not versions:
        return
    print(f"\n{'=' * 72}")
    print("草稿版本比較")
    print(f"{'-' * 72}")
    print(f"{'版本':<8}{'字元數':>10}{'相對上一版':>14}")
    for i, text in enumerate(versions):
        delta = ""
        if i > 0:
            diff = len(text) - len(versions[i - 1])
            delta = f"{diff:+d}"
        print(f"v{i:<7}{len(text):>10}{delta:>14}")


def build_revision_log(slug: str, versions: list, reviews: list) -> str:
    lines = [f"# 草稿修訂記錄：{slug.replace('_', ' ')}\n"]
    review_by_version = {r["draft_version"]: r for r in reviews}
    for i, draft in enumerate(versions):
        lines.append(f"## v{i} 草稿")
        lines.append(f"- 字元數：{len(draft)}")
        lines.append(f"- 檔案：`draft_{slug}_v{i}.md`\n")
        review = review_by_version.get(i)
        if review:
            lines.append(f"### 主編審核（第 {review.get('retry_count', 0)} 輪）")
            lines.append(f"- 決定：**{review['decision']}**")
            feedback = (review.get("feedback") or "").strip()
            lines.append(f"- 意見：\n\n{feedback or '（無）'}\n")
    if not versions:
        lines.append("（本次無草稿版本）\n")
    return "\n".join(lines)


def save_run_artifacts(result: dict, slug: str, total_wall_s: float, output_dir: Path) -> dict:
    output_dir.mkdir(exist_ok=True)
    paths: dict = {}

    report = build_run_report(result, total_wall_s)
    summary_path = output_dir / f"run_{slug}_summary.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    paths["summary"] = str(summary_path)

    versions = result.get("draft_versions") or []
    version_paths = []
    for i, draft in enumerate(versions):
        vf = output_dir / f"draft_{slug}_v{i}.md"
        vf.write_text(draft)
        version_paths.append(str(vf))
    paths["draft_versions"] = version_paths

    feedback_paths = []
    for review in editor_reviews:
        v = review["draft_version"]
        body = (
            f"# 主編審核 — v{v}\n\n"
            f"- 決定：{review['decision']}\n"
            f"- 輪次（retry_count）：{review.get('retry_count', 0)}\n\n"
            f"## 意見\n\n{(review.get('feedback') or '').strip() or '（無）'}\n"
        )
        ef = output_dir / f"editor_{slug}_v{v}.md"
        ef.write_text(body)
        feedback_paths.append(str(ef))
    if feedback_paths:
        paths["editor_feedback"] = feedback_paths

    if versions or editor_reviews:
        log_path = output_dir / f"draft_{slug}_revision_log.md"
        log_path.write_text(build_revision_log(slug, versions, editor_reviews))
        paths["revision_log"] = str(log_path)

    if len(versions) > 1:
        diff_lines = []
        comparison_lines = ["# 草稿版本差異摘要\n"]
        for i in range(len(versions) - 1):
            a = versions[i].splitlines(keepends=True)
            b = versions[i + 1].splitlines(keepends=True)
            diff_lines.extend(
                difflib.unified_diff(
                    a, b,
                    fromfile=f"draft_{slug}_v{i}.md",
                    tofile=f"draft_{slug}_v{i+1}.md",
                )
            )
            diff_lines.append("\n")
            comparison_lines.append(
                f"## v{i} → v{i+1}\n"
                f"- v{i}: {len(versions[i])} 字元\n"
                f"- v{i+1}: {len(versions[i+1])} 字元\n"
                f"- 變化: {len(versions[i+1]) - len(versions[i]):+d} 字元\n"
            )
        diff_path = output_dir / f"draft_{slug}_diff.txt"
        diff_path.write_text("".join(diff_lines))
        paths["diff"] = str(diff_path)

        comparison_path = output_dir / f"draft_{slug}_revision_comparison.md"
        comparison_path.write_text("\n".join(comparison_lines))
        paths["revision_comparison"] = str(comparison_path)

    if not versions and result.get("draft"):
        fallback = output_dir / f"draft_{slug}.md"
        fallback.write_text(result["draft"])
        paths["draft"] = str(fallback)

    export_memory_library(output_dir / "memory_library.md")
    paths["memory_library"] = str(output_dir / "memory_library.md")
    paths["memory_json"] = str(MEMORY_FILE)

    return paths


def call_llm(model: str, system: str, user: str, max_tokens: int = 2000) -> str:
    def _create(tokens: int):
        return client.messages.create(
            model=model,
            max_tokens=tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )

    for attempt in range(2):
        tokens = max_tokens if attempt == 0 else min(max_tokens * 2, 8000)
        response = _create(tokens)
        usage_log.append(
            {
                "node": current_node(),
                "model": model,
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
            }
        )
        # 被截斷就直接重試（加大 max_tokens），不管有沒有部分文字——
        # 否則截斷但非空的回覆會被下面 `if text_parts` 提前 return，永遠輪不到重試。
        if getattr(response, "stop_reason", None) == "max_tokens" and attempt == 0:
            continue
        text_parts = [
            block.text
            for block in getattr(response, "content", []) or []
            if getattr(block, "type", None) == "text" and hasattr(block, "text")
        ]
        if text_parts:
            return "".join(text_parts)
        output_text = getattr(response, "output_text", "") or ""
        if output_text.strip():
            return output_text
        raise ValueError("模型回覆中沒有可用文字輸出。")
    raise ValueError("模型回覆中沒有可用文字輸出（已重試）。")


def count_tokens(text: str, model: str = CHEAP_MODEL) -> int:
    """
    量測一段文字的 token 數，用來比較壓縮前後的差異。
    這是官方 count_tokens 端點，不產生任何生成內容，所以不計入 usage_log/成本。
    """
    if not text:
        return 0
    result = client.messages.count_tokens(model=model, messages=[{"role": "user", "content": text}])
    return result.input_tokens


def extract_json(text: str):
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
    else:
        start = min(
            (i for i in (text.find("["), text.find("{")) if i != -1), default=-1
        )
        if start == -1:
            raise ValueError(f"回覆中找不到 JSON：{text[:200]}")
        candidate = text[start:].strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    obj = re.search(r"\{[\s\S]*\}", candidate)
    if obj:
        return json.loads(obj.group(0))
    arr = re.search(r"\[[\s\S]*\]", candidate)
    if arr:
        return json.loads(arr.group(0))
    raise ValueError(f"回覆 JSON 解析失敗：{candidate[:200]}")


def _preview_update(node_name: str, update: dict) -> None:
    print(f"\n{'=' * 72}")
    print(f"節點：{node_name}")
    for key, value in update.items():
        if key == "memory_hits":
            print(f"  {key}：{len(value)} 筆")
            for hit in value[:3]:
                print(
                    f"    - {hit.get('run_date')} | {hit.get('topic')} "
                    f"(相似度 {hit.get('similarity', 0):.2f})"
                )
            continue
        if key in ("crawl_parallelism", "compression_stats"):
            print(f"  {key}：{json.dumps(value, ensure_ascii=False)}")
            continue
        if isinstance(value, list):
            print(f"  {key}：{len(value)} 筆")
            for item in value[:3]:
                print(f"    - {json.dumps(item, ensure_ascii=False)}")
            if len(value) > 3:
                print(f"    ... 還有 {len(value) - 3} 筆")
        elif isinstance(value, str) and len(value) > 400:
            print(f"  {key}：({len(value)} 字元)")
            print(f"    {value[:400]}...")
        else:
            print(f"  {key}：{value}")


# ---------------------------------------------------------------------------
# Embedding 工具（跟 stage3 相同：純 Python feature hashing）
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    text = text.lower()
    return re.findall(r"[a-z0-9一-鿿]+", text)


def embed_text(text: str, dim: int = EMBED_DIM) -> list[float]:
    vec = [0.0] * dim
    for token in _tokenize(text):
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        idx = int(digest, 16) % dim
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def dedup_by_embedding(items: list, threshold: float = DEDUP_SIMILARITY_THRESHOLD) -> list:
    kept: list = []
    kept_vecs: list[list[float]] = []
    for item in items:
        text = f"{item.get('title', '')} {item.get('snippet', '')}"
        vec = embed_text(text)
        if any(cosine_similarity(vec, prev) >= threshold for prev in kept_vecs):
            continue
        kept.append(item)
        kept_vecs.append(vec)
    return kept


# ---------------------------------------------------------------------------
# 跨日記憶（跟 stage3 相同，但存到 stage4 自己的 memory/ 目錄，跟 stage3 隔離）
# ---------------------------------------------------------------------------

def load_memory() -> list:
    if not MEMORY_FILE.exists():
        return []
    return json.loads(MEMORY_FILE.read_text())


def memory_entry_for_display(entry: dict) -> dict:
    return {
        "run_date": entry.get("run_date"),
        "topic": entry.get("topic"),
        "insights": entry.get("insights"),
        "draft_excerpt": entry.get("draft_excerpt"),
    }


def export_memory_library(dest: Optional[Path] = None) -> str:
    memories = load_memory()
    lines = [
        "# 跨日記憶庫",
        "",
        f"來源 JSON：`{MEMORY_FILE}`",
        f"共 {len(memories)} 筆",
        "",
    ]
    if not memories:
        lines.append("（記憶庫為空）")
    for i, mem in enumerate(memories):
        display = memory_entry_for_display(mem)
        lines.append(f"## [{i}] {display['run_date']} — {display['topic']}")
        lines.append("")
        lines.append("### 洞見")
        lines.append("")
        lines.append(display.get("insights") or "（無）")
        lines.append("")
        lines.append("### 草稿摘要")
        lines.append("")
        lines.append(display.get("draft_excerpt") or "（無）")
        lines.append("")
        lines.append("---")
        lines.append("")
    text = "\n".join(lines)
    if dest is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
    return text


def print_memory_library(max_chars: int = 8000) -> None:
    text = export_memory_library(MEMORY_DIR / "memory_library.md")
    print(f"\n{'=' * 72}")
    print("跨日記憶庫內容")
    print(f"{'-' * 72}")
    if len(text) <= max_chars:
        print(text)
    else:
        print(text[:max_chars])
        print(f"\n...（其餘 {len(text) - max_chars} 字元，完整內容見 memory/memory_library.md）")
    print(f"\nJSON 原始檔：{MEMORY_FILE}")


def save_memory_entry(topic: str, insights: str, draft: str) -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    memories = load_memory()
    entry = {
        "run_date": date.today().isoformat(),
        "topic": topic,
        "insights": insights,
        "draft_excerpt": draft[:500],
        "topic_embedding": embed_text(topic),
        "embedding": embed_text(f"{topic}\n{insights}"),
    }
    memories.append(entry)
    MEMORY_FILE.write_text(json.dumps(memories, ensure_ascii=False, indent=2))
    export_memory_library(MEMORY_DIR / "memory_library.md")


def recall_similar_memories(topic: str, top_k: int = 3) -> list:
    memories = load_memory()
    if not memories:
        return []
    query_vec = embed_text(topic)
    scored = []
    for mem in memories:
        mem_vec = mem.get("topic_embedding") or embed_text(mem.get("topic", ""))
        sim = cosine_similarity(query_vec, mem_vec)
        if sim >= MEMORY_SIMILARITY_THRESHOLD:
            scored.append({**mem, "similarity": round(sim, 3)})
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_k]


def seed_yesterday_memory() -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    topic = "AI agents"
    insights = (
        "# 昨日洞見（種子資料）\n\n"
        "- **通用 agent 敘事降溫**：市場從 AutoGPT 式萬能自主，轉向 OpenCode、browser-use 等垂直工具。\n"
        "- **框架路線分化**：LangChain（程式碼優先）與 Dify/Langflow（低代碼）長期並存。\n"
        "- **評測可信度成隱憂**：benchmark 可能被針對性優化，與真實任務能力有落差。"
    )
    entry = {
        "run_date": yesterday,
        "topic": topic,
        "insights": insights,
        "draft_excerpt": "（昨日草稿摘要）垂直化 agent 比通用 agent 更容易落地。",
        "topic_embedding": embed_text(topic),
        "embedding": embed_text(f"{topic}\n{insights}"),
    }
    MEMORY_FILE.write_text(json.dumps([entry], ensure_ascii=False, indent=2))
    export_memory_library(MEMORY_DIR / "memory_library.md")
    print(f"已寫入種子記憶：{MEMORY_FILE}")
    print(f"  run_date={yesterday}, topic={topic}")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class PipelineState(TypedDict):
    topic: str
    sequential_crawl: bool
    raw_items: list
    crawl_parallelism: dict
    dedup_items: list
    scored_items: list
    compressed_context: str
    compression_stats: dict
    memory_hits: list
    memory_context: str
    insights: str
    draft: str
    draft_versions: list
    editor_feedback: str
    retry_count: int
    max_retry: int
    force_bad_first_draft: bool


# ---------------------------------------------------------------------------
# 爬蟲來源（每個回傳 (items, 花費秒數)，方便算平行加速比）
# ---------------------------------------------------------------------------

def _fetch_hackernews(topic: str) -> tuple[list, float]:
    start = time.perf_counter()
    items: list = []
    try:
        response = requests.get(
            "https://hn.algolia.com/api/v1/search",
            params={"query": topic, "tags": "story", "hitsPerPage": 15},
            timeout=10,
        )
        for hit in response.json().get("hits", []):
            items.append(
                {
                    "source": "hackernews",
                    "title": hit.get("title") or "",
                    "url": hit.get("url")
                    or f"https://news.ycombinator.com/item?id={hit['objectID']}",
                    "score": hit.get("points") or 0,
                    "snippet": (hit.get("story_text") or "")[:1200],
                }
            )
    except Exception as error:
        print(f"  [crawl] HN 來源失敗（先跳過）：{error}")
    return items, time.perf_counter() - start


def _fetch_github(topic: str) -> tuple[list, float]:
    start = time.perf_counter()
    items: list = []
    try:
        response = requests.get(
            "https://api.github.com/search/repositories",
            params={"q": topic, "sort": "stars", "per_page": 15},
            headers={"User-Agent": "agentic-practice/0.1"},
            timeout=10,
        )
        for repo in response.json().get("items", []):
            items.append(
                {
                    "source": "github",
                    "title": repo.get("full_name") or "",
                    "url": repo.get("html_url") or "",
                    "score": repo.get("stargazers_count") or 0,
                    "snippet": (repo.get("description") or "")[:1200],
                }
            )
    except Exception as error:
        print(f"  [crawl] GitHub 來源失敗（先跳過）：{error}")
    return items, time.perf_counter() - start


def _fetch_arxiv(topic: str) -> tuple[list, float]:
    """
    免 key 的第三個來源。刻意選 arXiv 是因為論文摘要通常有 800-1500 字元，
    比 HN/GitHub 的短摘要長很多，這樣壓縮節點的「壓縮前後差異」才有感。
    arXiv 沒有熱度分數，score 固定填 0（不是抓取失敗）。
    """
    start = time.perf_counter()
    items: list = []
    try:
        response = requests.get(
            "https://export.arxiv.org/api/query",
            params={"search_query": f"all:{topic}", "start": 0, "max_results": 10},
            timeout=10,
        )
        root = ET.fromstring(response.text)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("a:entry", ns):
            title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
            summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
            link = (entry.findtext("a:id", default="", namespaces=ns) or "").strip()
            items.append(
                {
                    "source": "arxiv",
                    "title": re.sub(r"\s+", " ", title),
                    "url": link,
                    "score": 0,
                    "snippet": re.sub(r"\s+", " ", summary)[:1200],
                }
            )
    except Exception as error:
        print(f"  [crawl] arXiv 來源失敗（先跳過）：{error}")
    return items, time.perf_counter() - start


SOURCES = [
    ("hackernews", _fetch_hackernews),
    ("github", _fetch_github),
    ("arxiv", _fetch_arxiv),
]


# ---------------------------------------------------------------------------
# 節點
# ---------------------------------------------------------------------------

def crawl(state: PipelineState) -> dict:
    """
    3 個來源預設用 ThreadPoolExecutor 平行抓取（I/O bound，用執行緒就夠，
    不需要把整個 graph 改成 async）。傳 sequential_crawl=True 時改成逐一
    序列呼叫，方便跟平行版本直接比較牆鐘時間。
    """
    topic = state["topic"]
    sequential = state.get("sequential_crawl", False)
    items: list = []
    source_seconds: dict = {}

    wall_start = time.perf_counter()
    if sequential:
        for name, fn in SOURCES:
            fetched, seconds = fn(topic)
            items.extend(fetched)
            source_seconds[name] = round(seconds, 2)
    else:
        with ThreadPoolExecutor(max_workers=len(SOURCES)) as pool:
            future_to_name = {pool.submit(fn, topic): name for name, fn in SOURCES}
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                fetched, seconds = future.result()
                items.extend(fetched)
                source_seconds[name] = round(seconds, 2)
    wall_s = time.perf_counter() - wall_start

    sequential_estimate = round(sum(source_seconds.values()), 2)
    speedup = round(sequential_estimate / wall_s, 2) if wall_s else 1.0
    mode = "序列" if sequential else "平行"

    print(f"  [crawl] 共抓到 {len(items)} 筆（{mode}，{len(SOURCES)} 來源）")
    print(f"    各來源耗時：{source_seconds}")
    print(f"    序列預估 {sequential_estimate}s vs 實際牆鐘 {wall_s:.2f}s（加速 {speedup}x）")

    return {
        "raw_items": items,
        "crawl_parallelism": {
            "mode": mode,
            "source_seconds": source_seconds,
            "sequential_estimate_s": sequential_estimate,
            "wall_s": round(wall_s, 2),
            "speedup_x": speedup,
        },
    }


def dedup_embed(state: PipelineState) -> dict:
    items = state["raw_items"]
    if not items:
        return {"dedup_items": []}
    dedup_items = dedup_by_embedding(items)
    print(
        f"  [dedup_embed] {len(items)} 筆 -> {len(dedup_items)} 筆 "
        f"(embedding 門檻 {DEDUP_SIMILARITY_THRESHOLD})"
    )
    return {"dedup_items": dedup_items}


def authority(state: PipelineState) -> dict:
    items = state["dedup_items"]
    if not items:
        return {"scored_items": []}

    listing = "\n".join(
        f"{i}. [{item['source']}] {item['title']}（社群分數 {item['score']}）"
        for i, item in enumerate(items)
    )
    reply = call_llm(
        model=CHEAP_MODEL,
        system="你是資訊來源評估員，評估每筆資料的權威性與可信度。",
        user=(
            f"針對話題「{state['topic']}」，為以下每筆資料打 1-5 分"
            "（考量：來源類型、社群熱度、標題是否像一手資訊而非農場文；"
            "沒有社群分數的學術論文來源不代表不重要）：\n"
            f"{listing}\n\n"
            '請回傳 JSON 陣列：[{"index": 0, "authority": 4, "reason": "簡短理由"}, ...]。只回傳 JSON。'
        ),
        # 3 個來源合併後常有 30-40+ 筆，預設 2000 太緊繃、容易被截斷，調高留緩衝。
        max_tokens=3000,
    )
    try:
        scores = {entry["index"]: entry for entry in extract_json(reply)}
    except Exception as error:
        # 就算加大 max_tokens 後還是解析失敗，也不要讓整條 pipeline 崩掉——
        # 保底改用社群分數排序，並在 reason 標明這是降級結果，方便事後追查。
        print(f"  [authority] ⚠️ JSON 解析失敗，改用社群分數排序保底：{error}")
        fallback_items = sorted(items, key=lambda item: item.get("score", 0), reverse=True)[:8]
        scored_items = [
            {**item, "authority": 3, "reason": "（JSON 解析失敗，改用社群分數排序保底）"}
            for item in fallback_items
        ]
        print(f"  [authority] {len(items)} 筆 -> 留下 {len(scored_items)} 筆（保底模式）")
        return {"scored_items": scored_items}

    scored_items = []
    for i, item in enumerate(items):
        entry = scores.get(i)
        if entry and entry["authority"] >= 3:
            scored_items.append(
                {**item, "authority": entry["authority"], "reason": entry["reason"]}
            )
    scored_items.sort(key=lambda item: item["authority"], reverse=True)
    scored_items = scored_items[:8]
    print(f"  [authority] {len(items)} 筆 -> 留下 {len(scored_items)} 筆")
    return {"scored_items": scored_items}


def _build_raw_context(items: list) -> str:
    """壓縮節點的「壓縮前」內容：把每筆資料的完整摘要原封不動列出來。"""
    blocks = []
    for item in items:
        header = f"### [{item['source']}] {item['title']}（權威性 {item.get('authority', '?')}/5）"
        if item.get("reason"):
            header += f"\n判斷理由：{item['reason']}"
        blocks.append(f"{header}\n{item.get('snippet') or '（無摘要）'}")
    return "\n\n".join(blocks)


def compress(state: PipelineState) -> dict:
    """
    借鑑 open_deep_research 的 compress_research：authority 篩完的資料還是
    帶著完整摘要（尤其 arXiv 論文摘要很長），這裡用便宜模型壓縮成一段精簡筆記，
    讓下游 synthesize（貴模型）只需要處理濃縮後的內容。
    """
    items = state["scored_items"]
    if not items:
        return {
            "compressed_context": "",
            "compression_stats": {
                "raw_tokens": 0, "compressed_tokens": 0,
                "saved_tokens": 0, "saved_pct": 0.0,
            },
        }

    raw_context = _build_raw_context(items)
    raw_tokens = count_tokens(raw_context)

    compressed = call_llm(
        model=CHEAP_MODEL,
        system=(
            "你負責壓縮研究資料。把多筆原始摘要濃縮成一段精簡的重點筆記，"
            "只保留跟話題直接相關的事實，去掉冗長描述與重複資訊。"
        ),
        user=(
            f"話題：「{state['topic']}」\n\n原始資料：\n\n{raw_context}\n\n"
            "請輸出壓縮後的重點筆記（條列式，繁體中文，保留每筆的來源標記）。"
        ),
        max_tokens=1000,
    )
    compressed_tokens = count_tokens(compressed)
    saved_pct = round((1 - compressed_tokens / raw_tokens) * 100, 1) if raw_tokens else 0.0

    stats = {
        "raw_tokens": raw_tokens,
        "compressed_tokens": compressed_tokens,
        "saved_tokens": raw_tokens - compressed_tokens,
        "saved_pct": saved_pct,
    }
    print(f"  [compress] 原始 {raw_tokens:,} tokens -> 壓縮後 {compressed_tokens:,} tokens（省 {saved_pct}%）")
    return {"compressed_context": compressed, "compression_stats": stats}


def recall_memory(state: PipelineState) -> dict:
    hits = recall_similar_memories(state["topic"])
    if not hits:
        print("  [recall_memory] 沒有找到歷史類似話題")
        return {"memory_hits": [], "memory_context": ""}

    lines = ["以下是系統過去處理過的類似話題與洞見，請在歸納時主動對照、延續或修正："]
    for hit in hits:
        lines.append(
            f"\n### {hit['run_date']}｜{hit['topic']}（相似度 {hit['similarity']:.2f}）\n"
            f"{hit.get('insights', '')}"
        )
    context = "\n".join(lines)
    print(f"  [recall_memory] 命中 {len(hits)} 筆歷史記憶")
    return {"memory_hits": hits, "memory_context": context}


def synthesize(state: PipelineState) -> dict:
    """
    跟 stage1-3 的差異：這裡吃的是 compress 節點吐出的壓縮筆記，
    不是自己重新把 scored_items 攤開組 listing——這才是壓縮真正省到 token 的地方。
    """
    compressed = (state.get("compressed_context") or "").strip()
    if not compressed:
        return {"insights": "（沒有足夠的資料可以形成洞見）"}

    memory_block = state.get("memory_context") or ""
    user = f"話題：「{state['topic']}」\n\n"
    if memory_block:
        user += f"【跨日記憶 RAG】\n{memory_block}\n\n"
    user += (
        f"【今日壓縮後的研究筆記】\n{compressed}\n\n"
        "請歸納出 2-3 個洞見。若有歷史記憶，請明確指出「延續昨日觀點」或「與昨日不同之處」。"
        "用繁體中文、markdown 條列。"
    )
    insights = call_llm(
        model=SMART_MODEL,
        system="你是趨勢分析師，擅長跨時間對照趨勢變化。",
        user=user,
    )
    print(f"  [synthesize] 洞見產出 {len(insights)} 字元")
    return {"insights": insights}


def write(state: PipelineState) -> dict:
    sources = "\n".join(f"- {item['title']}：{item['url']}" for item in state["scored_items"])
    feedback = (state.get("editor_feedback") or "").strip()
    if state.get("force_bad_first_draft", False) and state.get("retry_count", 0) == 0:
        system = "你是寫作新手，請刻意寫得很粗糙、鬆散、觀點不明確。"
        user = f"話題：{state['topic']}\n洞見：\n{state['insights']}\n\n"
        user += "請寫一篇不到 200 字、非常粗糙的短文（含標題）。"
    else:
        system = "你是科技專欄編輯，文風精煉、觀點清晰。"
        user = "根據以下洞見，寫一篇約 500 字的繁體中文短文（含標題）：\n\n"
        user += f"{state['insights']}\n\n"
        if state.get("memory_context"):
            user += "若有引用跨日記憶，請在文中明確提到「相較昨日/過去」的變化。\n\n"
        if feedback:
            user += f"主編退回意見（請務必修正）：\n{feedback}\n\n"
        user += f"文末附上參考來源清單:\n{sources}"
    draft = call_llm(model=SMART_MODEL, system=system, user=user)
    retry_count = int(state.get("retry_count", 0))
    print(f"  [write] 草稿產出 {len(draft)} 字元（第 {retry_count} 次）")
    versions = list(state.get("draft_versions", []))
    versions.append(draft)
    revision_events.append(
        {
            "event": "write",
            "version": len(versions) - 1,
            "retry_count": retry_count,
            "chars": len(draft),
        }
    )
    return {"draft": draft, "draft_versions": versions}


def _record_editor_review(state: PipelineState, decision: str, feedback: str, retry_count: int) -> int:
    draft_version = max(len(state.get("draft_versions", [])) - 1, 0)
    editor_reviews.append(
        {
            "draft_version": draft_version,
            "retry_count": retry_count,
            "decision": decision,
            "feedback": feedback,
        }
    )
    return draft_version


def chief_editor(state: PipelineState) -> Command:
    max_retry = int(state.get("max_retry", 1))
    retry_count = int(state.get("retry_count", 0))

    def _judge_once(prompt: str, max_tokens: int) -> dict:
        reply = call_llm(
            model=SMART_MODEL,
            system="你是主編，嚴格審稿，會提出可執行的修改建議。",
            user=prompt,
            max_tokens=max_tokens,
        )
        return extract_json(reply)

    prompt = (
        f"話題：{state['topic']}\n\n"
        "請只回傳合法 JSON：\n"
        '{ "decision": "approve" 或 "revise", "feedback": ["建議1", "建議2"] }\n\n'
        "feedback 最多列 4 條、每條不超過 80 字——意見要具體可執行，但不要長篇大論，"
        "以免超出輸出長度限制。\n\n"
        f"草稿：\n{state['draft']}"
    )

    # 若模型回傳非合法 JSON（markdown 圍欄、截斷、多餘說明），不再呼叫 LLM repair：
    # 先前 repair prompt 內含 approve 範例，模型易照抄而誤放行；改為本地保守退回。
    # max_tokens 600 實測幾乎每次都不夠（feedback 稍微詳細就會截斷），調到 1500
    # 留緩衝；上面同時限制 feedback 條數/長度，兩者一起才不會又把 token 往上推。
    try:
        data = _judge_once(prompt, max_tokens=1500)
    except Exception:
        print("  [chief_editor] ⚠️ JSON 解析失敗，保守退回")
        data = {
            "decision": "revise",
            "feedback": ["主編審核回覆格式錯誤，請依洞見與結構重新檢查草稿"],
        }

    decision = (data.get("decision") or "").strip().lower()
    feedback_raw = data.get("feedback") or []
    if isinstance(feedback_raw, list):
        feedback = "\n".join(f"- {s}" for s in feedback_raw if str(s).strip())
    else:
        feedback = str(feedback_raw).strip()

    if decision == "approve":
        print("  [chief_editor] ✅ 通過")
        draft_version = _record_editor_review(state, "approve", feedback, retry_count)
        revision_events.append(
            {"event": "review", "decision": "approve", "retry_count": retry_count,
             "draft_version": draft_version, "feedback": feedback}
        )
        return Command(update={"editor_feedback": feedback}, goto=END)

    if retry_count >= max_retry:
        print("  [chief_editor] ⚠️ 已達 max_retry，停止重寫")
        draft_version = _record_editor_review(state, "revise_max_retry", feedback, retry_count)
        revision_events.append(
            {"event": "review", "decision": "revise_max_retry", "retry_count": retry_count,
             "draft_version": draft_version, "feedback": feedback}
        )
        return Command(update={"editor_feedback": feedback}, goto=END)

    print(f"  [chief_editor] ❌ 退回重寫（{retry_count + 1}/{max_retry}）")
    draft_version = _record_editor_review(state, "revise", feedback, retry_count)
    revision_events.append(
        {"event": "review", "decision": "revise", "retry_count": retry_count,
         "draft_version": draft_version, "feedback": feedback}
    )
    return Command(
        update={
            "editor_feedback": feedback or "請加強論點、結構與具體例子。",
            "retry_count": retry_count + 1,
        },
        goto="write",
    )


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

builder = StateGraph(PipelineState)
builder.add_node("crawl", instrument("crawl", crawl))
builder.add_node("dedup_embed", instrument("dedup_embed", dedup_embed))
builder.add_node("authority", instrument("authority", authority))
builder.add_node("compress", instrument("compress", compress))
builder.add_node("recall_memory", instrument("recall_memory", recall_memory))
builder.add_node("synthesize", instrument("synthesize", synthesize))
builder.add_node("write", instrument("write", write))
builder.add_node("chief_editor", instrument("chief_editor", chief_editor))

builder.add_edge(START, "crawl")
builder.add_edge("crawl", "dedup_embed")
builder.add_edge("dedup_embed", "authority")
builder.add_edge("authority", "compress")
builder.add_edge("compress", "recall_memory")
builder.add_edge("recall_memory", "synthesize")
builder.add_edge("synthesize", "write")
builder.add_edge("write", "chief_editor")
builder.add_edge("chief_editor", END)

graph = builder.compile()


def run_pipeline(
    topic: str,
    max_retry: int = 1,
    force_bad_first_draft: bool = False,
    sequential_crawl: bool = False,
) -> dict:
    global usage_log, revision_events, editor_reviews
    usage_log = []
    reset_metrics()
    revision_events = []
    editor_reviews = []

    wall_start = time.perf_counter()
    initial_state: PipelineState = {
        "topic": topic,
        "sequential_crawl": sequential_crawl,
        "raw_items": [],
        "crawl_parallelism": {},
        "dedup_items": [],
        "scored_items": [],
        "compressed_context": "",
        "compression_stats": {},
        "memory_hits": [],
        "memory_context": "",
        "insights": "",
        "draft": "",
        "draft_versions": [],
        "editor_feedback": "",
        "retry_count": 0,
        "max_retry": max_retry,
        "force_bad_first_draft": force_bad_first_draft,
    }

    result: dict = {"topic": topic}
    for chunk in graph.stream(initial_state, stream_mode="updates"):
        for node_name, update in chunk.items():
            _preview_update(node_name, update)
            result.update(update)

    if result.get("insights"):
        save_memory_entry(topic, result["insights"], result.get("draft", ""))
        print(f"\n已寫入跨日記憶：{MEMORY_FILE}")

    print("\n" + "=" * 72)
    print("歷史記憶命中：\n")
    for hit in result.get("memory_hits", []):
        print(f"- {hit['run_date']} | {hit['topic']} (相似度 {hit['similarity']:.2f})")
    print("\n" + "=" * 72)
    print("洞見：\n")
    print(result.get("insights", ""))
    print("\n" + "=" * 72)
    print("草稿：\n")
    print(result.get("draft", ""))
    print("\n" + "=" * 72)
    print("主編回饋：\n")
    print(result.get("editor_feedback", ""))

    slug = re.sub(r"[^一-鿿a-zA-Z0-9]+", "_", topic)
    output_dir = Path(__file__).resolve().parent / "outputs"
    total_wall_s = time.perf_counter() - wall_start

    print_draft_version_summary(result.get("draft_versions") or [])
    artifact_paths = save_run_artifacts(result, slug, total_wall_s, output_dir)

    versions = result.get("draft_versions") or []
    if versions:
        print(f"\n草稿版本數：{len(versions)}（已輸出到 {output_dir}）")
    if editor_reviews:
        print(f"主編審核次數：{len(editor_reviews)}")
        for review in editor_reviews:
            v = review["draft_version"]
            print(f"  - v{v}：{review['decision']} → editor_{slug}_v{v}.md")
    if artifact_paths.get("revision_log"):
        print(f"修訂時間軸：{artifact_paths['revision_log']}")
    if artifact_paths.get("diff"):
        print(f"版本差異檔：{artifact_paths['diff']}")

    print_memory_library()

    if artifact_paths:
        print(f"\n輸出檔案：")
        for key, path in artifact_paths.items():
            if isinstance(path, list):
                for p in path:
                    print(f"  - {p}")
            else:
                print(f"  - {key}: {path}")

    print_stage4_highlights(result)
    print_run_summary(total_wall_s)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4: 平行爬蟲 + 壓縮節點")
    parser.add_argument("topic", nargs="?", default="AI agents", help="話題")
    parser.add_argument("max_retry", nargs="?", type=int, default=1, help="主編最多退回次數")
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="爬蟲改序列執行（跟預設的平行版本比較耗時用）",
    )
    parser.add_argument(
        "--seed-yesterday",
        action="store_true",
        help="種一筆昨天的記憶（驗收跨日引用，不必真的等一天）",
    )
    parser.add_argument(
        "--bad-first",
        action="store_true",
        help="第一次草稿刻意寫差（測試主編退回）",
    )
    parser.add_argument(
        "--show-memory",
        action="store_true",
        help="只查看跨日記憶庫內容，不跑 pipeline",
    )
    args = parser.parse_args()

    if args.show_memory:
        print_memory_library(max_chars=20000)
    elif args.seed_yesterday:
        seed_yesterday_memory()
    else:
        print(f"話題：{args.topic}")
        print(f"max_retry：{args.max_retry}")
        print(f"爬蟲模式：{'序列' if args.sequential else '平行'}")
        print(f"記憶庫：{MEMORY_FILE}")
        print()
        run_pipeline(
            args.topic,
            args.max_retry,
            force_bad_first_draft=args.bad_first,
            sequential_crawl=args.sequential,
        )
