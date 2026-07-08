"""
跨階段共用的基礎設施。

只放進經過逐行 diff 驗證、stage1-4 完全一樣（或功能上等價）的部分：
instrument / cost_of / PRICING。

call_llm、extract_json、embedding 工具等還在演化中的邏輯刻意不放這裡——
它們在 stage1 -> stage2 之間就已經加過重試/修復邏輯，代表還會繼續變動，
抽成共用模組只會讓「改一個階段、牽動全部階段」的風險提早發生。
"""

import time

PRICING = {
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-5": (3.00, 15.00),
}

node_times: dict = {}
_current_node = "?"


def reset_metrics() -> None:
    """每次 run_pipeline() 開始前呼叫，清空上一輪殘留的計時資料。"""
    node_times.clear()
    global _current_node
    _current_node = "?"


def current_node() -> str:
    """給各階段的 call_llm 用，取得 instrument() 目前設定的節點名稱。"""
    return _current_node


def cost_of(entry: dict) -> float:
    """單筆 LLM 呼叫的成本（USD）。"""
    price_in, price_out = PRICING.get(entry["model"], (0.0, 0.0))
    return entry["input"] / 1e6 * price_in + entry["output"] / 1e6 * price_out


def instrument(name: str, fn):
    """包住節點函式：記錄執行時間，並讓 call_llm 知道現在跑到哪個節點。"""

    def wrapped(state):
        global _current_node
        _current_node = name
        start = time.perf_counter()
        result = fn(state)
        node_times[name] = node_times.get(name, 0.0) + (time.perf_counter() - start)
        return result

    return wrapped
