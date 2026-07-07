# Stage 2 Note — 條件邊退回重做（Chief Editor Loop）

## 目標（對照 `PLAN.md`）

在 stage1 的固定流水線後面加上 **主編審核** 節點，並用 `Command(goto=...)` 實作退回重寫迴圈：

- 故意讓草稿寫差，觀察真的會被退回重寫
- 設 `max_retry`，確保不會無限迴圈

對應程式：`practice/stage2/graph.py`

## 怎麼跑

在 `practice/` 目錄：

```bash
source .venv/bin/activate
python stage2/graph.py "AI agents" 1
```

第二個參數是 `max_retry`（最多重寫幾次）。`stage2` 預設會讓第 0 次草稿刻意寫差，方便看到主編真的退回。

## 你會看到什麼（動態路由）

- 固定順序跑到 `write`
- `chief_editor` 會產生 `decision` 與 `feedback`
  - 若 **approve**：`goto=END`
  - 若 **revise** 且還沒達上限：更新 `editor_feedback`、`retry_count += 1`，並 `goto="write"`
  - 若 **revise** 但已達 `max_retry`：直接結束（避免無限迴圈）

## 本次洞見（從輸出檔摘錄）

來源：`practice/stage2/outputs/draft_AI_agents.md`

### 洞見

- **垂直化（窄任務）是務實方向**：任務邊界收斂後，可控性與成功率通常高於通用型 agent
- **編排層分工不是混亂，而是需求分層**：LangChain（程式碼優先）、Dify（企業低代碼）、Langflow（視覺化原型）各吃不同族群
- **個人開發者的影響力上升**：Karpathy、obra 等個人專案能獲得與機構專案相近的關注度
- **走向可能是互操作而非單一標準**：更像透過協議互通，而不是等某個框架吃下全市場

（完整內容請看輸出檔內文。）

