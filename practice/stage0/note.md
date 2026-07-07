# Stage 0 Note — 環境與骨架

## 目標（對照 `PLAN.md`）

- **不呼叫 LLM**，先確認 LangGraph 的 `StateGraph` 能跑
- **看懂 state 如何在節點之間被讀取 / 更新 / 合併 / 傳遞**

對應程式：`practice/stage0/graph.py`

## 你會學到什麼

- **一張 graph 共享一份 state**：用 `TypedDict` 定義 state 的欄位（例：`name`、`message`）
- **每個節點都吃 state、吐「部分更新」**（`dict`），LangGraph 會把更新 merge 回同一份 state
- 用 `stream(stream_mode="values")` 看到 state 在每一步的變化（比 `invoke()` 只看最後結果更直覺）

## 怎麼跑

在 `practice/` 目錄：

```bash
source .venv/bin/activate
python stage0/graph.py
```

## 觀察重點：state 的變化過程

這個階段沒有「洞見」產出（因為完全沒用 LLM），你應該觀察的是：

- **進入圖之前**：`{"name": "...", "message": ""}`
- **跑完 `greet` 後**：`message` 被寫入「哈囉，...！」
- **跑完 `shout` 後**：`message` 被加工成大寫 + 🎉

