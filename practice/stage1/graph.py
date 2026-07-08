"""
階段 1：固定流水線（方案一）

目標：把 5 個節點用 add_edge「硬串」成固定順序，不加任何條件邊：

    爬蟲 -> 比對去重 -> 權威性判斷 -> 洞見合成 -> 編輯撰稿

對應 PLAN.md「階段 1」的驗收標準：輸入一個話題，能一路跑出一份文字草稿。

這一階段刻意示範 PLAN.md 裡兩個「省 token」手法：
1. 模型分級 —— 機械性任務（去重、評分）用便宜模型 CHEAP_MODEL，
   需要判斷力的任務（洞見、撰稿）才用較貴的 SMART_MODEL。
2. 結構化輸出 —— 節點之間傳遞的是精簡的 dict/list（標題+摘要+分數），
   不是整頁 HTML 或長篇散文；摘要也會先截斷再餵給模型。

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
import sys
from pathlib import Path
from typing import TypedDict

import anthropic
import requests
from langgraph.graph import END, START, StateGraph

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
# 之後階段 6 觀察成本時，就是回來調整這兩個常數。
CHEAP_MODEL = "claude-haiku-4-5-20251001"   # 去重、權威性評分
SMART_MODEL = "claude-sonnet-5"             # 洞見合成、編輯撰稿

client = anthropic.Anthropic()  # 自動讀取 ANTHROPIC_API_KEY

# usage_log 收集每次 LLM 呼叫的 token 數（掛在哪個節點下）。
# node_times（每個節點的執行秒數）現在由 _common.instrument() 維護。
# 這就是 PLAN.md 階段 6「觀察與優化」的雛形。
usage_log: list = []


def call_llm(model: str, system: str, user: str, max_tokens: int = 2000) -> str:
    """所有節點共用的 LLM 呼叫入口，順便把 token 用量記進 usage_log。"""
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    usage_log.append({
        "node": current_node(),
        "model": model,
        "input": response.usage.input_tokens,
        "output": response.usage.output_tokens,
    })
    # Sonnet 5 等 thinking 模型會先回 ThinkingBlock，不能假設 content[0] 就是文字。
    text_parts = [block.text for block in response.content if block.type == "text"]
    if not text_parts:
        raise ValueError(f"模型回覆中沒有 text block：{response.content!r}")
    return "".join(text_parts)


def print_run_summary() -> None:
    """執行結束後印出每個節點的時間、token 與成本總表。"""
    print(f"\n{'=' * 72}")
    print("執行總結")
    print(f"{'-' * 72}")
    print(f"{'節點':<12}{'時間':>8}{'LLM呼叫':>8}{'輸入tokens':>12}{'輸出tokens':>12}{'成本USD':>12}")
    total_cost = 0.0
    for name in node_times:
        calls = [entry for entry in usage_log if entry["node"] == name]
        tokens_in = sum(entry["input"] for entry in calls)
        tokens_out = sum(entry["output"] for entry in calls)
        cost = sum(cost_of(entry) for entry in calls)
        total_cost += cost
        print(
            f"{name:<12}{node_times[name]:>7.1f}s{len(calls):>8}"
            f"{tokens_in:>12,}{tokens_out:>12,}{cost:>12.4f}"
        )
    total_time = sum(node_times.values())
    total_in = sum(entry['input'] for entry in usage_log)
    total_out = sum(entry['output'] for entry in usage_log)
    print(f"{'-' * 72}")
    print(
        f"{'合計':<12}{total_time:>7.1f}s{len(usage_log):>8}"
        f"{total_in:>12,}{total_out:>12,}{total_cost:>12.4f}"
    )


def extract_json(text: str):
    """
    從 LLM 回覆中撈出 JSON。
    模型有時會在 JSON 前後多講幾句話或包 ```json 圍欄，
    所以先找圍欄、再退而求其次找第一個 [ 或 { 開頭的區塊。
    """
    # 先找圍欄，再找第一個 [ 或 { 開頭的區塊，如果都沒有，則拋出錯誤
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    # 如果找到圍欄，則取出圍欄內的文字，並去除前後的空白
    if fence:
        text = fence.group(1).strip()
    else: # 如果沒有找到圍欄，則找第一個 [ 或 { 開頭的區塊
        start = min((i for i in (text.find("["), text.find("{")) if i != -1), default=-1)
        # 如果都沒有找到，則拋出錯誤
        if start == -1:
            raise ValueError(f"回覆中找不到 JSON：{text[:200]}")
        # 如果找到，則取出區塊內的文字，並去除前後的空白
        text = text[start:].strip()

    return json.loads(text)


# ---------------------------------------------------------------------------
# State：整條流水線共用的資料容器
# ---------------------------------------------------------------------------

class PipelineState(TypedDict):
    topic: str          # 輸入：想追的話題（例如 "AI agents"）
    raw_items: list     # 爬蟲產出：[{source, title, url, score, snippet}, ...]
    dedup_items: list   # 去重後留下的 items
    scored_items: list  # 加上權威性分數的 items：[{..., authority, reason}, ...]
    insights: str       # 洞見合成的產出（markdown 條列）
    draft: str          # 最終文字草稿


# ---------------------------------------------------------------------------
# 節點 1：爬蟲
# ---------------------------------------------------------------------------

def crawl(state: PipelineState) -> dict:
    """
    從 2 個免費、不用 API key 的來源抓資料（PLAN.md：先固定 2-3 個來源就好）。
    - Hacker News：Algolia 提供的官方搜尋 API
    - Reddit：公開的 search.json 端點
    這個節點完全不用 LLM —— 能用普通程式解決的事就不要花 token。
    """
    topic = state["topic"]
    items = []

    # 來源 1：Hacker News（Algolia 搜尋 API）
    try:
        response = requests.get(
            "https://hn.algolia.com/api/v1/search",
            params={"query": topic, "tags": "story", "hitsPerPage": 15}, # 前15筆
            timeout=10,
        )
        for hit in response.json().get("hits", []):
            items.append({
                "source": "hackernews",
                "title": hit.get("title") or "",
                "url": hit.get("url") or f"https://news.ycombinator.com/item?id={hit['objectID']}",
                "score": hit.get("points") or 0,
                # 摘要截斷到 200 字元：控制之後餵給 LLM 的 token 量
                "snippet": (hit.get("story_text") or "")[:200],
            })
    except Exception as error:
        print(f"  [crawl] HN 來源失敗（先跳過）：{error}")

    # 來源 2：GitHub 搜尋 API（免 key、回 JSON；未登入時限流 10 次/分鐘，練習夠用）。
    # 註：原本想用 Reddit，但它的公開 JSON 端點現在一律回 403（要走 OAuth）——
    # 這正是 PLAN.md「爬蟲 ToS 風險」的實例：平台隨時可能收緊政策，來源要可抽換。
    try:
        response = requests.get(
            "https://api.github.com/search/repositories",
            params={"q": topic, "sort": "stars", "per_page": 15}, # 前15筆
            headers={"User-Agent": "agentic-practice/0.1"},
            timeout=10,
        )
        for repo in response.json().get("items", []):
            items.append({
                "source": "github",
                "title": repo.get("full_name") or "",
                "url": repo.get("html_url") or "",
                "score": repo.get("stargazers_count") or 0,
                "snippet": (repo.get("description") or "")[:200],
            })
    except Exception as error:
        print(f"  [crawl] GitHub 來源失敗（先跳過）：{error}")

    print(f"  [crawl] 共抓到 {len(items)} 筆")
    return {"raw_items": items}


# ---------------------------------------------------------------------------
# 節點 2：比對去重（便宜模型）
# ---------------------------------------------------------------------------

def dedup(state: PipelineState) -> dict:
    """
    讓便宜模型看「編號 + 標題」的清單，挑出同一件事只留一筆。
    只給標題、不給全文 —— 去重不需要全文，這是省 token 的前置過濾。
    （階段 3 會把這個節點換成 embedding 相似度，完全不吃 LLM token。）
    """
    items = state["raw_items"]
    if not items:
        return {"dedup_items": []}

    titles = "\n".join(f"{i}. [{item['source']}] {item['title']}" for i, item in enumerate(items))
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
    """
    給每筆資料打權威性分數 1-5（來源可信度、社群熱度、是否一手資訊），
    並只保留 3 分以上、最多 8 筆 —— 又一層前置過濾，
    讓下一步昂貴的「洞見合成」只需要處理過濾後的精華。
    """
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
            scored_items.append({**item, "authority": entry["authority"], "reason": entry["reason"]})

    # 依權威性排序、最多取 8 筆，控制下游 token 用量
    scored_items.sort(key=lambda item: item["authority"], reverse=True)
    scored_items = scored_items[:8]

    print(f"  [authority] {len(items)} 筆 -> 留下 {len(scored_items)} 筆（3 分以上、最多 8 筆）")
    return {"scored_items": scored_items}


# ---------------------------------------------------------------------------
# 節點 4：洞見合成（貴模型）
# ---------------------------------------------------------------------------

def synthesize(state: PipelineState) -> dict:
    """
    到這裡資料已經被前面的便宜節點濾成精華，才輪到貴模型上場。
    輸入是精簡的條列（標題+分數+理由），不是原始全文。
    """
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
    """
    把洞見寫成一篇短文草稿。
    注意輸入只有「洞見 + 來源清單」，不需要再回頭看原始資料 ——
    每一棒只接前一棒的濃縮輸出，這就是固定流水線省 token 的核心。
    """
    sources = "\n".join(f"- {item['title']}：{item['url']}" for item in state["scored_items"])
    draft = call_llm(
        model=SMART_MODEL,
        system="你是科技專欄編輯，文風精煉、觀點清晰，避免空話與流水帳。",
        user=(
            f"根據以下洞見，寫一篇約 500 字的繁體中文短文（含標題）：\n\n"
            f"{state['insights']}\n\n文末附上參考來源清單:\n{sources}"
        ),
    )

    print(f"  [write] 草稿產出 {len(draft)} 字元")
    return {"draft": draft}


# ---------------------------------------------------------------------------
# 組裝 graph：全部用 add_edge 固定順序（方案一），階段 2 才會加條件邊
# ---------------------------------------------------------------------------

builder = StateGraph(PipelineState)

# 每個節點都用 instrument() 包起來：計時 + token 歸屬，節點本身的程式碼不用動
builder.add_node("crawl", instrument("crawl", crawl))
builder.add_node("dedup", instrument("dedup", dedup))
builder.add_node("authority", instrument("authority", authority))
builder.add_node("synthesize", instrument("synthesize", synthesize))
builder.add_node("write", instrument("write", write))

builder.add_edge(START, "crawl")
builder.add_edge("crawl", "dedup")
builder.add_edge("dedup", "authority")
builder.add_edge("authority", "synthesize")
builder.add_edge("synthesize", "write")
builder.add_edge("write", END)

graph = builder.compile()


def _describe_item(item) -> str:
    """把一筆資料濃縮成一行人類看得懂的描述，取代整包 JSON。"""
    if isinstance(item, dict) and "title" in item:
        line = f"[{item.get('source', '?')}] {item['title'][:50]}"
        if "authority" in item:
            line += f"　權威性 {item['authority']}/5（{item.get('reason', '')[:30]}）"
        elif "score" in item:
            line += f"　社群分數 {item['score']:,}"
        return line
    return json.dumps(item, ensure_ascii=False)[:80]


def _preview_update(node_name: str, update: dict, seconds: float) -> None:
    """印出單一節點回傳的 state 更新；長清單只預覽前幾筆。"""
    print(f"\n{'=' * 60}")
    print(f"節點：{node_name}（{seconds:.1f} 秒）")
    for key, value in update.items():
        if isinstance(value, list):
            print(f"  {key}：{len(value)} 筆")
            for item in value[:3]:
                print(f"    - {_describe_item(item)}")
            if len(value) > 3:
                print(f"    ... 還有 {len(value) - 3} 筆")
        elif isinstance(value, str) and len(value) > 400:
            print(f"  {key}：({len(value)} 字元)")
            print(f"    {value[:400]}...")
        else:
            print(f"  {key}：{value}")


if __name__ == "__main__":
    import sys

    # 話題可以從命令列帶入：python stage1/graph.py "AI agents"
    topic = sys.argv[1] if len(sys.argv) > 1 else "AI agents"
    print(f"話題：{topic}\n")

    # stream(stream_mode="updates") 每跑完一個節點，就吐出該節點回傳的 dict。
    result: dict = {"topic": topic}
    for chunk in graph.stream({"topic": topic}, stream_mode="updates"):
        for node_name, update in chunk.items():
            _preview_update(node_name, update, node_times.get(node_name, 0.0))
            result.update(update)

    print("\n" + "=" * 60)
    print("洞見：\n")
    print(result["insights"])
    print("\n" + "=" * 60)
    print("草稿：\n")
    print(result["draft"])

    print_run_summary()

    # 把草稿存檔，方便階段 2 之後比較「退回重做前後」的版本差異
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"draft_{re.sub(r'[^一-鿿a-zA-Z0-9]+', '_', topic)}.md"
    output_file.write_text(result["draft"])
    print(f"\n草稿已存到 {output_file}")
