# Agentic 熱點洞見管線 — 練習計劃

## 目標

用 agentic AI 的概念，自動蒐集熱門話題、形成洞見，最終產出文章／視覺化／短影音。
主要目的是**理解 agentic AI 的可能性**（多代理協作、動態路由、RAG、token 節省），
不追求一開始就做到完美或商用等級。

## 角色（8 個 agent）

1. 爬蟲代理群 — 多來源蒐集熱門話題
2. 權威性判斷 — 評估來源可信度
3. 比對去重 — 找出重複/矛盾話題
4. 洞見合成（思考） — 綜合出有價值的觀點
5. 編輯撰稿 — 產出初稿
6. 主編審核 — 品質把關，決定通過或退回重做
7. 視覺化生成 — 把洞見轉成圖表
8. 短影音製作 — 腳本 → TTS → 影片合成

## 兩種架構方案（已比較）

### 方案一：固定流水線（Pipeline-DAG）
程式碼決定順序，8 個角色照固定順序執行，每階段只接收前一階段的濃縮輸出。
- 優點：token 成本可預測、容易除錯、適合當 MVP
- 缺點：單向通過，前面的錯誤不會被下游抓到，"agentic" 程度較低

### 方案二：共享黑板 + 動態路由（Blackboard）
一個「主編路由」agent 看著共享狀態動態決定下一步，並可以把品質不夠的產出「退回重做」。
- 優點：有交叉檢查/自我修正能力，更接近真正的自主協作
- 缺點：token 成本較難預估（需設 retry 上限），除錯較難

**結論**：先做方案一打底跑通，再只在「主編審核」這一關疊加方案二的退回重做迴圈（設最多重試 1-2 次）。

## 技術選型：LangGraph + RAG

- **LangGraph** 的 `StateGraph` 同時能承載兩種方案：`add_edge` 做固定順序；
  `add_conditional_edges`（或更新寫法 `Command(goto=...)`）做動態退回重做。
  `checkpointer` 免費拿到「黑板」的持久化能力。
- **RAG（向量資料庫）** 插入點：
  1. 比對去重 → 用 embedding 相似度分群，取代 LLM 兩兩比較（O(N²) → 近乎免費）
  2. 權威性判斷 → 檢索既有可信來源做交叉查證
  3. 洞見合成 → 檢索歷史話題/過去結論，做成跨日的「趨勢記憶」
  4. 編輯撰稿 → 檢索過去核准的風格範例做 few-shot

## 省 token 的共通手法

- 模型分級：機械性任務（爬蟲摘要/去重/分類）用便宜模型，只有洞見合成、主編審核用貴模型
- 結構化輸出取代散文：階段間傳遞 JSON/短欄位
- 外部狀態儲存：中間結果寫檔案/資料庫，下游只讀需要的欄位
- 前置過濾：去重/權威性判斷放便宜階段，先濾掉大部分雜訊
- Prompt caching：固定的 system prompt/tool 定義盡量重用
- Map-reduce + 壓縮節點：多來源平行處理後，先壓縮成摘要再往下傳（借鑑 open_deep_research 的 `compress_research`）

## 可借鑑的 GitHub 專案

| 階段 | 專案 | 借鑑重點 | License / 維護狀態 |
|---|---|---|---|
| 整體流程藍本 | [gpt-newspaper](https://github.com/rotemweiss57/gpt-newspaper) | Search→Curator→Writer→Critique→Designer，用舊版 `add_conditional_edges` 做退回重做 | MIT，已停更約 2 年（僅供讀邏輯參考） |
| 主編/審核團隊模式 | [gpt-researcher](https://github.com/assafelovic/gpt-researcher) | Chief Editor/Researcher/Editor/Reviewer/Revisor 分工 | Apache-2.0，活躍維護 |
| 官方最新寫法（子圖巢狀、平行研究、壓縮節點） | [langchain-ai/open_deep_research](https://github.com/langchain-ai/open_deep_research) | `Command(goto=...)`、supervisor 子圖、`asyncio.gather` 平行 researcher 子圖、`compress_research` 壓縮節點、模型分級寫在 config | MIT，活躍維護 |
| 洞見要多視角 | [stanford-oval/storm](https://github.com/stanford-oval/storm) | 多視角提問＋模擬對話，避免洞見角度單一 | MIT，活躍維護 |
| 動態路由最小元件 | [langgraph-supervisor-py](https://github.com/langchain-ai/langgraph-supervisor-py)、[langgraph-reflection](https://github.com/langchain-ai/langgraph-reflection) | 官方拆出的可重用 supervisor / reflection 套件 | 官方維護 |
| 熱點蒐集（中文平台） | [TrendRadar](https://github.com/SANSAN0/TrendRadar) | 聚合抖音/知乎/B站/微博等熱點 + AI 篩選摘要 | **GPL-3.0**（直接複製程式碼進商業/閉源專案要小心，僅參考架構、自己重寫沒問題），活躍維護 |
| 短影音製作 | [AI-Faceless-Video-Generator](https://github.com/SamurAIGPT/AI-Faceless-Video-Generator)、[AI-Youtube-Shorts-Generator](https://github.com/SaarD00/AI-Youtube-Shorts-Generator) | 腳本→TTS→素材/講者臉→FFmpeg 合成鏈路 | MIT |

**原則**：只借「graph 怎麼設計」這個架構模式，自己重新寫程式碼，不直接複製貼上——這樣完全沒有授權疑慮，也是最好的練習方式。

## 已知風險（練習階段皆為低風險，正式產出前要注意）

- **License**：TrendRadar 是 GPL-3.0，若直接複製其程式碼進商業/閉源專案會有 copyleft 風險；只參考架構沒問題
- **過時依賴**：gpt-newspaper 已停更 2 年，直接跑可能會撞 LangGraph 版本不相容，建議只讀邏輯、自己用新版重寫
- **爬蟲 ToS**：抖音/知乎/微博等平台通常禁止自動化爬取，練習階段小量測試風險低，正式大量高頻爬取有 ToS 與封鎖風險，優先用官方 API/RSS
- **Secrets**：API key 用有額度上限的測試 key，跑第三方小型專案前先看一下依賴清單，環境建議隔離（容器/虛擬環境）
- **內容風險**：AI 生成的洞見/文章未經主編審核就發布有錯誤/幻覺風險；短影音牽涉原內容著作權，且社群平台對「AI 生成大量內容」有偵測與限流機制，練習階段不要自動發布到正式帳號

## 分階段練習路線圖

| 階段 | 目標 | 產出 / 驗收標準 |
|---|---|---|
| 0. 環境與骨架 | 跑通最小的 LangGraph 單節點 | `graph.invoke()` 能跑，看懂 state 怎麼傳遞 |
| 1. 固定流水線（方案一） | 5 個節點（爬蟲→去重→權威判斷→洞見→編輯）循序跑完 | 輸入一個話題，能一路跑出一份文字草稿 |
| 2. 條件邊退回重做 | 加入主編審核節點，用 `Command(goto=...)` 做重做迴圈 | 故意讓草稿寫差，觀察真的被退回重寫，且不會無限迴圈（設 max_retry） |
| 3. 導入 RAG 記憶 | embedding 去重 + 跨日話題記憶 | 跑兩天，第二天能看到系統引用「昨天提過類似話題」 |
| 4. 平行化與壓縮 | 多來源平行爬蟲 + 壓縮節點（可選：子圖巢狀） | 比較平行前後跑完時間，以及壓縮前後傳給洞見合成的 token 數差異 |
| 5. 視覺化與短影音 | 加上圖表生成與短影音腳本/影片合成 | 完整跑一次「話題輸入→短影片輸出」全程 |
| 6. 觀察與優化 | 記錄每個節點的 token/成本，調整模型分級 | 能講出具體數字，例如「去重換 embedding 後這階段 token 少了 N%」 |

**建議**：每個階段結束開一個 git commit/tag（例如 `stage-1-linear-pipeline`），方便回頭比較架構演進的差異。
