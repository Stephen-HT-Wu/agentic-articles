"""
階段 3：導入 RAG 記憶

目標（PLAN.md）：
1. 比對去重改用 embedding 相似度（不再用 LLM 兩兩比較）
2. 跨日話題記憶：第二天能看到系統引用「昨天提過類似話題」

流程：
    crawl -> dedup_embed -> authority -> recall_memory -> synthesize
         -> write -> chief_editor（可退回 write）

執行範例：
    # 一般執行（會讀取/寫入 memory/topic_memory.json）
    python stage3/graph.py "AI agents"

    # 先種一筆「昨天」的記憶，再跑今天（驗收跨日引用）
    python stage3/graph.py --seed-yesterday
    python stage3/graph.py "AI agents"
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
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, TypedDict

import anthropic
import requests
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

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

# embedding 去重：標題+摘要 cosine >= 門檻視為同一則
DEDUP_SIMILARITY_THRESHOLD = 0.80
# 跨日記憶：以「話題」embedding 比對（不用整段洞見，避免稀釋相似度）
MEMORY_SIMILARITY_THRESHOLD = 0.35
EMBED_DIM = 256

usage_log: list = []
node_times: dict = {}
revision_events: list = []
editor_reviews: list = []
_current_node = "?"

PRICING = {
    CHEAP_MODEL: (1.00, 5.00),
    SMART_MODEL: (3.00, 15.00),
}

# 計算成本
def cost_of(entry: dict) -> float:
    price_in, price_out = PRICING.get(entry["model"], (0.0, 0.0))
    return entry["input"] / 1e6 * price_in + entry["output"] / 1e6 * price_out

# 印出執行總結
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

# 組出可存檔的執行報告（節點時間、token、成本、版本、記憶命中）
def build_run_report(result: dict, total_wall_s: float) -> dict:
    """組出可存檔的執行報告（節點時間、token、成本、版本、記憶命中）。"""
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
    }

# 印出每個草稿版本的字數與相鄰版本差異
def print_draft_version_summary(versions: list) -> None:
    """印出每個草稿版本的字數與相鄰版本差異。"""
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

# 組出人類可讀的「草稿 + 主編意見」時間軸
def build_revision_log(slug: str, versions: list, reviews: list) -> str:
    """組出人類可讀的「草稿 + 主編意見」時間軸。"""
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

# 把執行報告、草稿版本、編輯意見、版本 diff 寫入 outputs/
def save_run_artifacts(result: dict, slug: str, total_wall_s: float, output_dir: Path) -> dict:
    """把執行報告、草稿版本、編輯意見、版本 diff 寫入 outputs/。"""
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
                    a,
                    b,
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

# 呼叫 LLM
def call_llm(model: str, system: str, user: str, max_tokens: int = 2000) -> str:
    def _create(tokens: int):
        return client.messages.create(
            model=model,
            max_tokens=tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )

    for attempt in range(2):
        tokens = max_tokens if attempt == 0 else min(max_tokens * 2, 8000) # 第一次呼叫 max_tokens，第二次呼叫 max_tokens * 2，但最多8000 tokens
        response = _create(tokens)
        usage_log.append(
            {
                "node": _current_node,
                "model": model,
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
            }
        )
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
        if getattr(response, "stop_reason", None) == "max_tokens" and attempt == 0:
            continue
        raise ValueError("模型回覆中沒有可用文字輸出。")
    raise ValueError("模型回覆中沒有可用文字輸出（已重試）。")

# 記錄節點時間
def instrument(name: str, fn):
    def wrapped(state):
        global _current_node
        _current_node = name
        start = time.perf_counter()
        result = fn(state)
        node_times[name] = node_times.get(name, 0.0) + (time.perf_counter() - start)
        return result

    return wrapped

# 提取 JSON
def extract_json(text: str):
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL) # 提取 JSON 字串 使用正規表示式 提取 ```json 和 ``` 之間的內容
    # 如果提取到 JSON 字串，則返回 JSON 字串
    if fence:
        candidate = fence.group(1).strip()
    # 如果沒有提取到 JSON 字串，則使用正規表示式 提取 [ 和 ] 之間的內容
    else:
        start = min(
            (i for i in (text.find("["), text.find("{")) if i != -1), default=-1
        ) # 找到 [ 或 { 第一次出現的位置
        if start == -1:
            raise ValueError(f"回覆中找不到 JSON：{text[:200]}")
        candidate = text[start:].strip() # 提取 [ 或 { 第一次出現的位置到最後一個字符之間的內容
    # 嘗試解析 JSON 字串
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # 如果解析 JSON 字串失敗，則使用正規表示式 提取 { 和 } 之間的內容
    obj = re.search(r"\{[\s\S]*\}", candidate) # 使用正規表示式 提取 { 和 } 之間的內容  
    if obj:
        return json.loads(obj.group(0))
    arr = re.search(r"\[[\s\S]*\]", candidate) # 使用正規表示式 提取 [ 和 ] 之間的內容
    if arr:
        return json.loads(arr.group(0))
    raise ValueError(f"回覆 JSON 解析失敗：{candidate[:200]}") # 如果解析 JSON 字串失敗，則抛出錯誤

# 預覽更新
def _preview_update(node_name: str, update: dict) -> None:
    print(f"\n{'=' * 72}") # 打印分隔線
    print(f"節點：{node_name}")
    for key, value in update.items(): # 遍歷更新字典
        if key == "memory_hits":
            print(f"  {key}：{len(value)} 筆") # 打印記憶庫命中次數
            for hit in value[:3]:
                print( # 打印記憶庫命中次數
                    f"    - {hit.get('run_date')} | {hit.get('topic')} "
                    f"(相似度 {hit.get('similarity', 0):.2f})"
                )
            continue
        if isinstance(value, list): # 如果 value 是列表
            print(f"  {key}：{len(value)} 筆") # 打印列表長度
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
# Embedding 工具（純 Python feature hashing，免 API、免額外依賴）
# ---------------------------------------------------------------------------

# 分詞
def _tokenize(text: str) -> list[str]:
    text = text.lower() # 將文本轉換為小寫
    return re.findall(r"[a-z0-9\u4e00-\u9fff]+", text)

# 將文本轉換為固定維度向量
def embed_text(text: str, dim: int = EMBED_DIM) -> list[float]:
    """把文字轉成固定維度向量，用於 cosine 相似度（hash 必須跨執行穩定）。"""
    vec = [0.0] * dim # 初始化向量
    for token in _tokenize(text):
        digest = hashlib.md5(token.encode("utf-8")).hexdigest() # 計算 token 的 MD5 哈希值
        idx = int(digest, 16) % dim # 計算 token 的索引
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0 # 計算向量的模長 模長的意思是向量的長度
    return [v / norm for v in vec] # 返回正規化後的向量

# 計算余弦相似度
def cosine_similarity(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b)) # 計算兩個向量的點積 點積的意思是兩個向量的對應元素相乘後相加

# 用 embedding 相似度去重
def dedup_by_embedding(items: list, threshold: float = DEDUP_SIMILARITY_THRESHOLD) -> list:
    kept: list = [] # 保留的項目
    kept_vecs: list[list[float]] = [] # 保留的向量
    for item in items:
        text = f"{item.get('title', '')} {item.get('snippet', '')}" # 計算文本的向量
        vec = embed_text(text)
        if any(cosine_similarity(vec, prev) >= threshold for prev in kept_vecs):
            continue # 如果向量相似度大於門檻，則跳過   
        kept.append(item)
        kept_vecs.append(vec)
    return kept # 返回保留的項目


# ---------------------------------------------------------------------------
# 跨日記憶（本機 JSON 持久化）
# ---------------------------------------------------------------------------

# 加載記憶庫
def load_memory() -> list:
    if not MEMORY_FILE.exists(): # 如果記憶庫文件不存在，則返回空列表
        return []
    return json.loads(MEMORY_FILE.read_text()) # 加載記憶庫文件

# 記憶庫項目顯示
def memory_entry_for_display(entry: dict) -> dict:
    """去掉 embedding 向量，方便印出或存檔回顧。"""
    return { # 返回記憶庫項目
        "run_date": entry.get("run_date"),
        "topic": entry.get("topic"),
        "insights": entry.get("insights"),
        "draft_excerpt": entry.get("draft_excerpt"),
    }

# 導出記憶庫    
def export_memory_library(dest: Optional[Path] = None) -> str:
    """把 topic_memory.json 轉成人類可讀的 markdown。"""
    memories = load_memory()
    lines = [
        "# 跨日記憶庫",
        "",
        f"來源 JSON：`{MEMORY_FILE}`",
        f"共 {len(memories)} 筆",
        "",
    ]
    # 如果記憶庫為空，則添加提示信息
    if not memories:
        lines.append("（記憶庫為空）")
    # 遍歷記憶庫項目
    for i, mem in enumerate(memories):
        display = memory_entry_for_display(mem) # 記憶庫項目顯示
        lines.append(f"## [{i}] {display['run_date']} — {display['topic']}") # 添加記憶庫項目
        lines.append("") # 添加空行
        lines.append("### 洞見")
        lines.append("") # 添加空行
        lines.append(display.get("insights") or "（無）")
        lines.append("")
        lines.append("### 草稿摘要")
        lines.append("")
        lines.append(display.get("draft_excerpt") or "（無）")
        lines.append("")
        lines.append("---")
        lines.append("")
    text = "\n".join(lines)
    if dest is not None: # 如果目標文件存在，則創建目標文件的父目錄
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text) # 寫入文本到目標文件
    return text # 返回文本

# 打印記憶庫內容
def print_memory_library(max_chars: int = 8000) -> None:
    """在終端印出記憶庫內容（過長則截斷並提示完整檔案）。"""
    text = export_memory_library(MEMORY_DIR / "memory_library.md")
    print(f"\n{'=' * 72}")
    print("跨日記憶庫內容")
    print(f"{'-' * 72}")
    # 如果文本長度小於最大字符數，則打印文本
    if len(text) <= max_chars:
        print(text) # 打印文本
    else:
        print(text[:max_chars]) # 打印文本的前 max_chars 個字符
        print(f"\n...（其餘 {len(text) - max_chars} 字元，完整內容見 memory/memory_library.md）") # 打印提示信息
    print(f"\nJSON 原始檔：{MEMORY_FILE}") # 打印 JSON 原始檔

# 保存記憶庫項目
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


# 召回相似記憶  
def recall_similar_memories(topic: str, top_k: int = 3) -> list:
    memories = load_memory() # 加載記憶庫
    memories = load_memory()
    if not memories: # 如果記憶庫為空，則返回空列表
        return []
    query_vec = embed_text(topic) # 計算文本的向量
    scored = []
    for mem in memories: # 遍歷記憶庫項目
        mem_vec = mem.get("topic_embedding") or embed_text(mem.get("topic", "")) # 計算文本的向量
        sim = cosine_similarity(query_vec, mem_vec) # 計算余弦相似度
        if sim >= MEMORY_SIMILARITY_THRESHOLD:
            scored.append({**mem, "similarity": round(sim, 3)}) # 添加記憶庫項目
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_k] # 返回相似記憶

# 種子記憶
def seed_yesterday_memory() -> None:
    """種一筆「昨天」的記憶，方便驗收跨日引用（不必真的等一天）。"""
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

# 管道狀態
class PipelineState(TypedDict):
    topic: str
    raw_items: list
    dedup_items: list
    scored_items: list
    memory_hits: list # 記憶庫命中次數  
    memory_context: str
    insights: str
    draft: str
    draft_versions: list
    editor_feedback: str
    retry_count: int
    max_retry: int
    force_bad_first_draft: bool


# ---------------------------------------------------------------------------
# 節點
# ---------------------------------------------------------------------------

# 爬取資料
def crawl(state: PipelineState) -> dict:
    topic = state["topic"]
    items = []
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
                    "snippet": (hit.get("story_text") or "")[:200],
                }
            )
    except Exception as error:
        print(f"  [crawl] HN 來源失敗（先跳過）：{error}")

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
                    "snippet": (repo.get("description") or "")[:200],
                }
            )
    except Exception as error:
        print(f"  [crawl] GitHub 來源失敗（先跳過）：{error}")

    print(f"  [crawl] 共抓到 {len(items)} 筆")
    return {"raw_items": items}

# 用 embedding 相似度去重
def dedup_embed(state: PipelineState) -> dict:
    """用 embedding 相似度去重：O(N) 貪婪保留，不吃 LLM token。"""
    items = state["raw_items"]
    if not items:
        return {"dedup_items": []}
    dedup_items = dedup_by_embedding(items)
    print(
        f"  [dedup_embed] {len(items)} 筆 -> {len(dedup_items)} 筆 "
        f"(embedding 門檻 {DEDUP_SIMILARITY_THRESHOLD})"
    )
    return {"dedup_items": dedup_items}

# 權威性評估
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
            f"針對話題「{state['topic']}」，為以下每筆資料打 1-5 分：\n"
            f"{listing}\n\n"
            '請回傳 JSON 陣列：[{"index": 0, "authority": 4, "reason": "簡短理由"}, ...]。只回傳 JSON。'
        ),
    )
    scores = {entry["index"]: entry for entry in extract_json(reply)}
    scored_items = []
    for i, item in enumerate(items):
        entry = scores.get(i) # 獲取評分
        if entry and entry["authority"] >= 3: # 如果評分大於 3，則添加評分項目
            scored_items.append(
                {
                    **item,
                    "authority": entry["authority"],
                    "reason": entry["reason"],
                }
            )
    scored_items.sort(key=lambda item: item["authority"], reverse=True) # 按權威性排序
    scored_items = scored_items[:8] # 保留前 8 筆
    print(f"  [authority] {len(items)} 筆 -> 留下 {len(scored_items)} 筆")
    return {"scored_items": scored_items} # 返回評估結果    

# 召回記憶
def recall_memory(state: PipelineState) -> dict:
    """RAG：從本機記憶庫檢索過去類似話題，供 synthesize 引用。"""
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

# 歸納洞見
def synthesize(state: PipelineState) -> dict:
    items = state["scored_items"]
    if not items: # 如果沒有足夠的資料，則返回空洞見
        return {"insights": "（沒有足夠的資料可以形成洞見）"}

    listing = "\n".join(
        f"- [{item['source']}] {item['title']}（權威性 {item['authority']}/5：{item['reason']}）"
        for item in items # 遍歷資料
    )
    memory_block = state.get("memory_context") or "" # 獲取記憶庫上下文
    user = f"話題：「{state['topic']}」\n\n"
    if memory_block: # 如果記憶庫上下文存在，則添加記憶庫上下文
        user += f"【跨日記憶 RAG】\n{memory_block}\n\n"
    user += (
        f"今日經過去重與權威性過濾的資料：\n{listing}\n\n"
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

# 寫作
def write(state: PipelineState) -> dict:
    sources = "\n".join(
        f"- {item['title']}：{item['url']}" for item in state["scored_items"]
    )
    feedback = (state.get("editor_feedback") or "").strip() # 獲取編輯反饋
    # 如果需要強制寫作很粗糙，提示訊息是寫作新手，刻意寫得很粗糙、鬆散、觀點不明確。
    if state.get("force_bad_first_draft", False) and state.get("retry_count", 0) == 0: # 如果需要強制寫作很粗糙，則添加提示信息
        system = "你是寫作新手，請刻意寫得很粗糙、鬆散、觀點不明確。" 
        user = f"話題：{state['topic']}\n洞見：\n{state['insights']}\n\n" 
        user += "請寫一篇不到 200 字、非常粗糙的短文（含標題）。" 
    else: # 如果不需要強制寫作很粗糙，提示訊息是科技專欄編輯，文風精煉、觀點清晰。
        system = "你是科技專欄編輯，文風精煉、觀點清晰。" 
        user = "根據以下洞見，寫一篇約 500 字的繁體中文短文（含標題）：\n\n" 
        user += f"{state['insights']}\n\n" 
        # 如果記憶庫上下文存在，提示訊息是若有引用跨日記憶，請在文中明確提到「相較昨日/過去」的變化。
        if state.get("memory_context"): # 如果記憶庫上下文存在，則添加記憶庫上下文
            user += "若有引用跨日記憶，請在文中明確提到「相較昨日/過去」的變化。\n\n" 
        if feedback: # 如果編輯反饋存在，則添加編輯反饋
            user += f"主編退回意見（請務必修正）：\n{feedback}\n\n"
        user += f"文末附上參考來源清單:\n{sources}" 
    draft = call_llm(model=SMART_MODEL, system=system, user=user) # 調用 LLM 生成草稿
    retry_count = int(state.get("retry_count", 0))
    print(f"  [write] 草稿產出 {len(draft)} 字元（第 {retry_count} 次）")
    versions = list(state.get("draft_versions", [])) # 獲取草稿版本
    versions.append(draft)
    # 添加草稿版本
    revision_events.append(
        {
            "event": "write",
            "version": len(versions) - 1,
            "retry_count": retry_count,
            "chars": len(draft),
        }
    )
    return {"draft": draft, "draft_versions": versions}


# 記錄編輯反饋
def _record_editor_review(
    state: PipelineState, decision: str, feedback: str, retry_count: int
) -> int:
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

# 主編審稿
def chief_editor(state: PipelineState) -> Command:
    max_retry = int(state.get("max_retry", 1)) # 獲取最大重試次數
    retry_count = int(state.get("retry_count", 0))
    # 調用 LLM 生成草稿
    def _judge_once(prompt: str, max_tokens: int) -> dict:
        reply = call_llm(
            model=SMART_MODEL, 
            system="你是主編，嚴格審稿，會提出可執行的修改建議。",
            user=prompt,
            max_tokens=max_tokens,
        )
        return extract_json(reply)
    # 主審稿：帶話題與完整草稿，請模型真的判斷可否發布。
    prompt = (
        f"話題：{state['topic']}\n\n"
        "請只回傳合法 JSON：\n"
        '{ "decision": "approve" 或 "revise", "feedback": ["建議1", "建議2"] }\n\n'
        f"草稿：\n{state['draft']}"
    )

    # 若模型回傳非合法 JSON（常見：markdown 圍欄、截斷、多餘說明），
    # 改送精簡 prompt 再試一次，避免整條 pipeline 因 parse 失敗中斷。
    # 取捨：repair 不帶草稿，修復力較弱，但 token 少、較不易再截斷。
    try:
        data = _judge_once(prompt, max_tokens=600) # 調用 LLM 生成草稿
    except Exception:
        data = _judge_once( "請只回傳合法 JSON：" '{ "decision": "approve", "feedback": [] }', max_tokens=200) # 調用 LLM 生成草稿

    decision = (data.get("decision") or "").strip().lower()
    feedback_raw = data.get("feedback") or []
    if isinstance(feedback_raw, list):
        feedback = "\n".join(f"- {s}" for s in feedback_raw if str(s).strip())
    else:
        feedback = str(feedback_raw).strip()
    # 如果決策是通過，則記錄編輯反饋
    if decision == "approve":
        print("  [chief_editor] ✅ 通過")
        draft_version = _record_editor_review(state, "approve", feedback, retry_count)
        revision_events.append(
            {
                "event": "review",
                "decision": "approve",
                "retry_count": retry_count,
                "draft_version": draft_version,
                "feedback": feedback,
            }
        )
        return Command(update={"editor_feedback": feedback}, goto=END)
    # 如果重試次數大於最大重試次數，則記錄編輯反饋
    if retry_count >= max_retry:
        print("  [chief_editor] ⚠️ 已達 max_retry，停止重寫")
        draft_version = _record_editor_review(
            state, "revise_max_retry", feedback, retry_count
        )
        revision_events.append(
            {
                "event": "review",
                "decision": "revise_max_retry",
                "retry_count": retry_count,
                "draft_version": draft_version,
                "feedback": feedback,
            }
        )
        return Command(update={"editor_feedback": feedback}, goto=END)
    # 如果決策是退回，則記錄編輯反饋
    print(f"  [chief_editor] ❌ 退回重寫（{retry_count + 1}/{max_retry}）")
    draft_version = _record_editor_review(state, "revise", feedback, retry_count)
    revision_events.append(
        {
            "event": "review",
            "decision": "revise",
            "retry_count": retry_count,
            "draft_version": draft_version,
            "feedback": feedback,
        }
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
# 建立狀態圖
builder = StateGraph(PipelineState)
builder.add_node("crawl", instrument("crawl", crawl))
builder.add_node("dedup_embed", instrument("dedup_embed", dedup_embed))
builder.add_node("authority", instrument("authority", authority))
builder.add_node("recall_memory", instrument("recall_memory", recall_memory))
builder.add_node("synthesize", instrument("synthesize", synthesize))
builder.add_node("write", instrument("write", write))
builder.add_node("chief_editor", instrument("chief_editor", chief_editor))

builder.add_edge(START, "crawl")
builder.add_edge("crawl", "dedup_embed")
builder.add_edge("dedup_embed", "authority")
builder.add_edge("authority", "recall_memory")
builder.add_edge("recall_memory", "synthesize")
builder.add_edge("synthesize", "write")
builder.add_edge("write", "chief_editor")
builder.add_edge("chief_editor", END)

graph = builder.compile()

# 運行 pipeline
def run_pipeline(topic: str, max_retry: int = 1, force_bad_first_draft: bool = False) -> dict:
    global usage_log, node_times, revision_events, editor_reviews
    usage_log = []
    node_times = {}
    revision_events = []
    editor_reviews = []

    wall_start = time.perf_counter()
    initial_state: PipelineState = {
        "topic": topic,
        "raw_items": [],
        "dedup_items": [],
        "scored_items": [],
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

    # 跑完後寫入跨日記憶
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
    # 生成 slug
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
    # 打印記憶庫內容
    print_memory_library()

    if artifact_paths:
        print(f"\n輸出檔案：")
        for key, path in artifact_paths.items():
            if isinstance(path, list):
                for p in path:
                    print(f"  - {p}")
            else:
                print(f"  - {key}: {path}")

    print_run_summary(total_wall_s)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: RAG memory + embedding dedup")
    parser.add_argument("topic", nargs="?", default="AI agents", help="話題")
    parser.add_argument("max_retry", nargs="?", type=int, default=1, help="主編最多退回次數")
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
        print(f"記憶庫：{MEMORY_FILE}")
        print()
        run_pipeline(args.topic, args.max_retry, force_bad_first_draft=args.bad_first)
