"""
階段 2：條件邊退回重做（方案一 + 主編審核迴圈）

目標：在 stage1 的固定流水線後面加上「主編審核」節點，
用 `Command(goto=...)` 做退回重寫迴圈，並設 `max_retry` 避免無限迴圈。

驗收標準（PLAN.md）：
- 故意讓草稿寫差，觀察真的被退回重寫
- 不會無限迴圈（max_retry 生效）

執行前需要設定 ANTHROPIC_API_KEY（可放在 practice/.env）：
    ANTHROPIC_API_KEY=sk-ant-...
"""

import warnings

import langchain_core  # noqa: F401 — 先載入，讓 langchain 註冊完 warning 規則
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

import json
import os
import re
import time
import difflib
from pathlib import Path
from typing import TypedDict

import sys

import anthropic
import requests
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

# 共用基礎設施（只有經過驗證完全一樣的 instrument/cost_of/PRICING，見 _common.py 開頭說明）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import PRICING, cost_of, current_node, instrument, node_times

# ---------------------------------------------------------------------------
# 環境設定
# ---------------------------------------------------------------------------

# 簡易 .env 載入：只認 KEY=VALUE 格式，避免多裝一個 python-dotenv 依賴。
_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _value = _line.partition("=")
            os.environ.setdefault(_key.strip(), _value.strip())

# 模型分級：便宜模型做機械性任務，貴模型做需要判斷力的任務。
CHEAP_MODEL = "claude-haiku-4-5-20251001"  # 去重、權威性評分
SMART_MODEL = "claude-sonnet-5"  # 洞見合成、編輯撰稿、主編審核

client = anthropic.Anthropic()  # 自動讀取 ANTHROPIC_API_KEY

# usage_log 收集每次 LLM 呼叫的 token 數（掛在哪個節點下）。
# node_times（每個節點的執行秒數）現在由 _common.instrument() 維護。
usage_log: list = []


def print_run_summary(total_wall_s: float) -> None:
    """執行結束後印出每個節點的時間、token 與成本總表。"""
    print(f"\n{'=' * 72}")
    print("執行總結")
    print(f"{'-' * 72}")
    print(
        f"{'節點':<14}{'節點時間':>10}{'LLM呼叫':>8}"
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
            f"{name:<14}{node_times[name]:>9.1f}s{len(calls):>8}"
            f"{tokens_in:>12,}{tokens_out:>12,}{cost:>12.4f}"
        )

    total_node_s = sum(node_times.values())
    total_in = sum(entry["input"] for entry in usage_log)
    total_out = sum(entry["output"] for entry in usage_log)
    print(f"{'-' * 72}")
    print(
        f"{'合計(節點)':<14}{total_node_s:>9.1f}s{len(usage_log):>8}"
        f"{total_in:>12,}{total_out:>12,}{total_cost:>12.4f}"
    )
    print(f"{'合計(牆鐘)':<14}{total_wall_s:>9.1f}s")


def call_llm(model: str, system: str, user: str, max_tokens: int = 2000) -> str:
    """所有節點共用的 LLM 呼叫入口，順便把 token 用量記進 usage_log。"""
    def _create(tokens: int):
        return client.messages.create(
            model=model,
            max_tokens=tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )

    # 某些情況下（例如輸出被截斷、或只回 thinking block），content 可能沒有 text block。
    # 這裡先嘗試正常取 text；若沒有，再用 SDK 的 output_text（若存在）或加大 max_tokens 重試一次。
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

        stop_reason = getattr(response, "stop_reason", None)
        # 被截斷就直接重試（加大 max_tokens），不管有沒有部分文字——
        # 否則截斷但非空的回覆會被下面 `if text_parts` 提前 return，永遠輪不到重試。
        if stop_reason == "max_tokens" and attempt == 0:
            continue

        text_parts = []
        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", None) == "text" and hasattr(block, "text"):
                text_parts.append(block.text)

        if text_parts:
            return "".join(text_parts)

        output_text = getattr(response, "output_text", "") or ""
        if output_text.strip():
            return output_text

        types = [getattr(b, "type", type(b).__name__) for b in (getattr(response, "content", []) or [])]
        raise ValueError(
            "模型回覆中沒有可用文字輸出。"
            f" stop_reason={stop_reason!r}, content_types={types!r}"
        )

    raise ValueError("模型回覆中沒有可用文字輸出（已重試）。")


def extract_json(text: str):
    """
    從 LLM 回覆中撈出 JSON。
    模型有時會在 JSON 前後多講幾句話或包 ```json 圍欄，
    所以先找圍欄、再退而求其次找第一個 [ 或 { 開頭的區塊。
    """
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

    # 先直接 parse；失敗時再嘗試用「最大括號區塊」補救（模型偶爾會在 JSON 外多講話）。
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # 優先嘗試 object，再嘗試 array
    obj = re.search(r"\{[\s\S]*\}", candidate)
    if obj:
        return json.loads(obj.group(0))
    arr = re.search(r"\[[\s\S]*\]", candidate)
    if arr:
        return json.loads(arr.group(0))

    raise ValueError(f"回覆 JSON 解析失敗：{candidate[:200]}")


def _preview_update(node_name: str, update: dict) -> None:
    """印出單一節點回傳的 state 更新；長清單只預覽前幾筆。"""
    print(f"\n{'=' * 72}")
    print(f"節點：{node_name}")
    for key, value in update.items():
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
# State：共享黑板（整條流水線共用的資料容器）
# ---------------------------------------------------------------------------


class PipelineState(TypedDict):
    topic: str
    raw_items: list
    dedup_items: list
    scored_items: list
    insights: str
    draft: str
    draft_versions: list

    # stage2 新增：主編審核與重寫控制
    editor_feedback: str
    retry_count: int
    max_retry: int
    force_bad_first_draft: bool


# ---------------------------------------------------------------------------
# 節點 1：爬蟲（不用 LLM）
# ---------------------------------------------------------------------------


def crawl(state: PipelineState) -> dict:
    topic = state["topic"]
    items = []

    # 來源 1：Hacker News（Algolia 搜尋 API）
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

    # 來源 2：GitHub 搜尋 API（免 key）
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


# ---------------------------------------------------------------------------
# 節點 2：去重（便宜模型）
# ---------------------------------------------------------------------------


def dedup(state: PipelineState) -> dict:
    items = state["raw_items"]
    if not items:
        return {"dedup_items": []}

    titles = "\n".join(
        f"{i}. [{item['source']}] {item['title']}" for i, item in enumerate(items)
    )
    reply = call_llm(
        model=CHEAP_MODEL,
        system="你是新聞去重助手。多筆標題若在講同一件事，只保留資訊最完整的一筆。",
        user=(
            f"以下是關於「{state['topic']}」的標題清單：\n{titles}\n\n"
            "請回傳要『保留』的編號，JSON 陣列格式，例如 [0, 2, 5]。只回傳 JSON。"
        ),
    )
    keep = set(extract_json(reply))
    dedup_items = [item for i, item in enumerate(items) if i in keep]
    print(f"  [dedup] {len(items)} 筆 -> {len(dedup_items)} 筆")
    return {"dedup_items": dedup_items}


# ---------------------------------------------------------------------------
# 節點 3：權威性判斷（便宜模型）
# ---------------------------------------------------------------------------


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
            "（考量：來源類型、社群熱度、標題是否像一手資訊而非農場文）：\n"
            f"{listing}\n\n"
            '請回傳 JSON 陣列：[{"index": 0, "authority": 4, "reason": "簡短理由"}, ...]。只回傳 JSON。'
        ),
    )
    scores = {entry["index"]: entry for entry in extract_json(reply)}

    scored_items = []
    for i, item in enumerate(items):
        entry = scores.get(i)
        if entry and entry["authority"] >= 3:
            scored_items.append(
                {
                    **item,
                    "authority": entry["authority"],
                    "reason": entry["reason"],
                }
            )

    scored_items.sort(key=lambda item: item["authority"], reverse=True)
    scored_items = scored_items[:8]

    print(
        f"  [authority] {len(items)} 筆 -> 留下 {len(scored_items)} 筆（3 分以上、最多 8 筆）"
    )
    return {"scored_items": scored_items}


# ---------------------------------------------------------------------------
# 節點 4：洞見合成（貴模型）
# ---------------------------------------------------------------------------


def synthesize(state: PipelineState) -> dict:
    items = state["scored_items"]
    if not items:
        return {"insights": "（沒有足夠的資料可以形成洞見）"}

    listing = "\n".join(
        f"- [{item['source']}] {item['title']}（權威性 {item['authority']}/5：{item['reason']}）"
        for item in items
    )
    insights = call_llm(
        model=SMART_MODEL,
        system=(
            "你是趨勢分析師。從多筆熱門資料中歸納出「洞見」——"
            "不是逐筆摘要，而是找出底層的共同趨勢、矛盾點、或大家還沒注意到的角度。"
        ),
        user=(
            f"話題：「{state['topic']}」\n經過去重與權威性過濾的資料：\n{listing}\n\n"
            "請歸納出 2-3 個洞見，每個洞見一行標題加 2-3 句說明，用繁體中文、markdown 條列。"
        ),
    )
    print(f"  [synthesize] 洞見產出 {len(insights)} 字元")
    return {"insights": insights}


# ---------------------------------------------------------------------------
# 節點 5：編輯撰稿（貴模型）
# ---------------------------------------------------------------------------


def write(state: PipelineState) -> dict:
    sources = "\n".join(
        f"- {item['title']}：{item['url']}" for item in state["scored_items"]
    )
    feedback = (state.get("editor_feedback") or "").strip()

    # 故意讓第一次草稿寫差：方便看 stage2 的「退回重寫」真的會發生。
    if state.get("force_bad_first_draft", False) and state.get("retry_count", 0) == 0:
        system = "你是寫作新手，請刻意寫得很粗糙、鬆散、觀點不明確。"
        user = (
            f"話題：{state['topic']}\n洞見：\n{state['insights']}\n\n"
            "請寫一篇不到 200 字、非常粗糙的短文（含標題），內容要顯得空泛。"
        )
    else:
        system = "你是科技專欄編輯，文風精煉、觀點清晰，避免空話與流水帳。"
        user = (
            "根據以下洞見，寫一篇約 500 字的繁體中文短文（含標題）：\n\n"
            f"{state['insights']}\n\n"
        )
        if feedback:
            user += f"主編退回意見（請務必修正）：\n{feedback}\n\n"
        user += f"文末附上參考來源清單:\n{sources}"

    draft = call_llm(model=SMART_MODEL, system=system, user=user)
    print(f"  [write] 草稿產出 {len(draft)} 字元（第 {state.get('retry_count', 0)} 次）")
    versions = list(state.get("draft_versions", []))
    versions.append(draft)
    return {"draft": draft, "draft_versions": versions}


# ---------------------------------------------------------------------------
# 節點 6：主編審核（貴模型）— 用 Command(goto=...) 決定下一步
# ---------------------------------------------------------------------------


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
        "以下是文章草稿，請審核是否可發布。\n"
        "請只回傳『合法 JSON』且務必精簡，格式如下：\n"
        '{ "decision": "approve" 或 "revise", "feedback": ["建議1", "建議2", "建議3"] }\n'
        "- feedback 最多 5 條\n"
        "- 每條建議 <= 120 字\n"
        "- 不要輸出任何額外文字、不要 markdown 圍欄\n\n"
        f"草稿：\n{state['draft']}"
    )

    try:
        data = _judge_once(prompt, max_tokens=600)
    except Exception:
        # 模型偶爾會輸出破損/截斷的 JSON；這裡強制它「修正成合法 JSON」再試一次。
        repair_prompt = (
            "上一則輸出不是合法 JSON。請你現在只回傳合法 JSON（不要任何解釋）。\n"
            "格式：\n"
            '{ "decision": "approve" 或 "revise", "feedback": ["建議1", "建議2", "建議3"] }\n'
            "- feedback 最多 5 條\n"
            "- 每條建議 <= 120 字\n\n"
            f"草稿：\n{state['draft']}"
        )
        data = _judge_once(repair_prompt, max_tokens=400)

    decision = (data.get("decision") or "").strip().lower() # 決定是否通過
    feedback_raw = data.get("feedback") or [] # 回饋意見
    if isinstance(feedback_raw, list): # 如果回饋意見是列表
        feedback = "\n".join(f"- {s}" for s in feedback_raw if str(s).strip())
    else: # 如果回饋意見不是列表    
        feedback = str(feedback_raw).strip()

    if decision == "approve": # 如果決定是通過
        print("  [chief_editor] ✅ 通過")
        return Command(update={"editor_feedback": feedback}, goto=END)

    # decision == revise（或任何非 approve） 或 已達重試上限
    if retry_count >= max_retry:
        print("  [chief_editor] ⚠️ 已達 max_retry，停止重寫並直接結束（保留最後草稿）")
        return Command(
            update={
                "editor_feedback": feedback or "（主編退回，但已達重試上限）",
            },
            goto=END,
        )
    
    print(f"  [chief_editor] ❌ 退回重寫（{retry_count + 1}/{max_retry}）") 
    return Command(
        update={
            "editor_feedback": feedback or "請加強論點、結構與具體例子。",
            "retry_count": retry_count + 1,
        },
        goto="write",
    )


# ---------------------------------------------------------------------------
# 組裝 graph：固定順序 + 主編審核可退回 write
# ---------------------------------------------------------------------------


builder = StateGraph(PipelineState)

builder.add_node("crawl", instrument("crawl", crawl))
builder.add_node("dedup", instrument("dedup", dedup))
builder.add_node("authority", instrument("authority", authority))
builder.add_node("synthesize", instrument("synthesize", synthesize))
builder.add_node("write", instrument("write", write))
builder.add_node("chief_editor", instrument("chief_editor", chief_editor))

builder.add_edge(START, "crawl")
builder.add_edge("crawl", "dedup")
builder.add_edge("dedup", "authority")
builder.add_edge("authority", "synthesize")
builder.add_edge("synthesize", "write")
builder.add_edge("write", "chief_editor")

# chief_editor 預設走向 END；若退回重寫，會用 Command(goto="write") 覆蓋
builder.add_edge("chief_editor", END)

graph = builder.compile()


if __name__ == "__main__":
    import sys

    topic = sys.argv[1] if len(sys.argv) > 1 else "AI agents"
    max_retry = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    print(f"話題：{topic}")
    print(f"max_retry：{max_retry}")
    print()

    wall_start = time.perf_counter()

    initial_state: PipelineState = {
        "topic": topic,
        "raw_items": [],
        "dedup_items": [],
        "scored_items": [],
        "insights": "",
        "draft": "",
        "draft_versions": [],
        "editor_feedback": "",
        "retry_count": 0,
        "max_retry": max_retry,
        "force_bad_first_draft": True,
    }

    result: dict = {"topic": topic}
    for chunk in graph.stream(initial_state, stream_mode="updates"):
        for node_name, update in chunk.items():
            _preview_update(node_name, update)
            result.update(update)

    print("\n" + "=" * 72)
    print("洞見：\n")
    print(result.get("insights", ""))
    print("\n" + "=" * 72)
    print("草稿：\n")
    print(result.get("draft", ""))
    print("\n" + "=" * 72)
    print("主編回饋：\n")
    print(result.get("editor_feedback", ""))

    # 把每一版草稿存檔，並輸出版本差異（方便比較「退回重做前後」）
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    slug = re.sub(r"[^一-鿿a-zA-Z0-9]+", "_", topic)
    versions = result.get("draft_versions") or []
    if versions:
        for i, draft in enumerate(versions):
            vf = output_dir / f"draft_{slug}_v{i}.md"
            vf.write_text(draft)
        print(f"\n草稿版本數：{len(versions)}（已輸出到 {output_dir}）")

        # 版本差異：只做相鄰版本的 unified diff（v0->v1, v1->v2, ...）
        diff_lines = []
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
        diff_file = output_dir / f"draft_{slug}_diff.txt"
        diff_file.write_text("".join(diff_lines))
        print(f"版本差異檔：{diff_file}")
    else:
        output_file = output_dir / f"draft_{slug}.md"
        output_file.write_text(result.get("draft", ""))
        print(f"\n草稿已存到 {output_file}")

    total_wall_s = time.perf_counter() - wall_start
    print_run_summary(total_wall_s)

