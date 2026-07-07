"""
階段 0：環境與骨架

目標：先不呼叫任何 LLM，純粹確認 LangGraph 能跑起來，
並看懂 state 如何在節點之間被讀取、更新、傳遞下去。
對應 PLAN.md 裡「階段 0」的驗收標準：graph.invoke() 能跑、看懂 state 怎麼傳遞。
"""
import warnings

import langchain_core  # noqa: F401 — 先載入，讓 langchain 註冊完 warning 規則
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


# state 是所有節點共用的「資料容器」。
# 每個節點函式收到目前的 state，回傳一個 dict（要更新的欄位），
# LangGraph 會自動把回傳值合併回 state，再交給下一個節點。
class GreetingState(TypedDict):
    name: str       # 輸入欄位：使用者名字
    message: str    # 節點之間傳遞/累積的訊息


def greet(state: GreetingState) -> dict:
    """第一個節點：讀取 name，寫入一句問候語到 message。"""
    name = state["name"]
    return {"message": f"哈囉，{name}！"}


def shout(state: GreetingState) -> dict:
    """
    第二個節點：讀取上一個節點寫入的 message，加工後再寫回去。
    這裡刻意示範「下游節點可以讀到上游節點寫入的欄位」，
    這就是之後 8 個 agent 之間傳資料的基本機制。
    """
    message = state["message"]
    return {"message": message.upper() + " 🎉"}


# 建立 graph 時要告訴 StateGraph 這張圖的 state 長什麼樣子（GreetingState）。
# 之後用 add_node 加進來的函式，都要符合「吃 state、回傳 dict」這個介面。
builder = StateGraph(GreetingState)

builder.add_node("greet", greet)
builder.add_node("shout", shout)

# START / END 是 LangGraph 內建的特殊節點，代表整張圖的入口與出口。
# add_edge 決定節點之間「固定」的執行順序，
# 對應 PLAN.md「方案一：固定流水線」的做法（之後階段 2 才會換成條件邊）。
builder.add_edge(START, "greet")
builder.add_edge("greet", "shout")
builder.add_edge("shout", END)

# compile() 把節點與邊組裝成一個真正可以執行的 graph 物件。
graph = builder.compile()


if __name__ == "__main__":
    initial_state: GreetingState = {"name": "Stephen", "message": ""}
    print("初始 state：", initial_state)
    print()

    # stream(stream_mode="values") 會在每個節點跑完後，吐出「合併後的完整 state」。
    # 比 invoke() 只回傳最終結果更適合觀察 state 怎麼一步步變化。
    for i, state in enumerate(graph.stream(initial_state, stream_mode="values")):
        if i == 0:
            print("（進入圖之前，尚未執行任何節點）")
        else:
            print(f"（第 {i} 個節點執行完）")
        print("  state：", state)
        print()

    print("最終 state：", state)
