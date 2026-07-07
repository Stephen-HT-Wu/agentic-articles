# Stage 1 Note — 固定流水線（Pipeline-DAG）

## 目標（對照 `PLAN.md`）

用固定順序把 5 個節點串起來：

**crawl → dedup → authority → synthesize → write**

驗收：輸入一個話題，能一路跑出一份文字草稿。

對應程式：`practice/stage1/graph.py`

## 怎麼跑

在 `practice/` 目錄：

```bash
source .venv/bin/activate
python stage1/graph.py "AI agents"
```

## 你會看到什麼（每節點抓到什麼）

`stage1` 已改成會印出每個節點更新 state 的內容（清單只預覽前幾筆）：

- **crawl**：抓到 `raw_items`（HN + GitHub）
- **dedup**：輸出 `dedup_items`（去重後的清單）
- **authority**：輸出 `scored_items`（加上權威性分數、再過濾）
- **synthesize**：輸出 `insights`（洞見文字）
- **write**：輸出 `draft`（草稿）

## 本次洞見（從輸出檔摘錄）

來源：`practice/stage1/outputs/draft_AI_agents.md`

### 洞見

- **agent 工具鏈從「萬能」轉向「垂直專用」**：AutoGPT 式的通用敘事降溫，browser-use、gemini-cli 這類專注單一任務的工具更容易落地
- **開發模式二分**：低代碼/視覺化（Langflow、Dify）與程式碼優先（LangChain）看起來會長期並存
- **主導權仍在開源社群**：即使大廠下場，話語權並未自然轉移，草根專案仍能主導方向

（完整內容請看輸出檔內文。）

