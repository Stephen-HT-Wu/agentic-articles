# Stage 3 記憶與 JSON 運作說明

本文件說明 `practice/stage3/graph.py` 裡**跨日話題記憶**如何持久化、檢索，以及相關 JSON 檔案的用途與格式。

程式入口：`practice/stage3/graph.py`  
流程概覽：`practice/stage3/note.md`

---

## 一句話總結

Stage 3 把每次跑完的**洞見**存進本機 JSON；下次跑**相似話題**時，在 `synthesize` 之前先**檢索**歷史洞見，當成 RAG 上下文注入 LLM，讓分析能「對照昨天說了什麼」。

這不是向量資料庫，而是**單一 JSON 檔 + 自製 embedding** 的簡化版 RAG，適合練習與本機實驗。

---

## 檔案一覽

| 路徑 | 是否入 git | 角色 |
|------|------------|------|
| `memory/topic_memory.json` | 否（`.gitignore`） | **主記憶庫**：所有歷史條目的原始資料 |
| `memory/memory_library.md` | 否 | JSON 的**人類可讀版**（不含 embedding 向量） |
| `outputs/run_{topic}_summary.json` | 否 | **單次執行報告**（含 memory_hits、token、revision） |
| `outputs/memory_library.md` | 否 | 該次執行結束時的記憶庫**快照** |

查看記憶庫（不跑 pipeline、不吃 API）：

```bash
cd practice
python stage3/graph.py --show-memory
```

---

## 資料流（何時讀、何時寫）

```mermaid
flowchart LR
    subgraph 執行開始
        A[話題 topic] --> B[recall_memory]
        B --> C{topic_memory.json<br/>有相似條目？}
        C -->|是| D[memory_hits + memory_context]
        C -->|否| E[空 context]
    end

    D --> F[synthesize<br/>今日資料 + 歷史洞見]
    E --> F
    F --> G[write → chief_editor]
    G --> H[跑完後 save_memory_entry]
    H --> I[append 到 topic_memory.json]
    I --> J[更新 memory_library.md]
```

**讀取時機**：`authority` 之後、`synthesize` 之前的 `recall_memory` 節點。  
**寫入時機**：整條 pipeline 跑完後（`run_pipeline` 結尾），且本次有產出 `insights` 才寫入。

也就是說：記憶庫存的是**洞見（insights）**，不是完整草稿；草稿只留前 500 字當 `draft_excerpt` 方便人類回顧。

---

## `topic_memory.json` 結構

檔案是一個 **JSON 陣列**，每個元素是一筆「某天處理某話題」的記錄：

```json
[
  {
    "run_date": "2026-07-08",
    "topic": "AI agents",
    "insights": "# AI agents 趨勢歸納\n\n## 1. ...",
    "draft_excerpt": "# 草稿標題\n\n正文前 500 字...",
    "topic_embedding": [0.0625, 0.0, ...],
    "embedding": [0.0312, 0.125, ...]
  }
]
```

### 欄位說明

| 欄位 | 型別 | 用途 |
|------|------|------|
| `run_date` | string (`YYYY-MM-DD`) | 這筆記憶是哪一天寫入的 |
| `topic` | string | 當次輸入的話題 |
| `insights` | string | **RAG 檢索後注入 LLM 的主要內容**（synthesize 的歷史參考） |
| `draft_excerpt` | string | 最終草稿前 500 字，僅供人類回顧，**不參與檢索** |
| `topic_embedding` | `float[256]` | 只對 `topic` 做 embedding，**用來找相似話題** |
| `embedding` | `float[256]` | 對 `topic + insights` 做 embedding，預留欄位，目前檢索未使用 |

### 為什麼有兩種 embedding？

- **檢索**用 `topic_embedding`：比對「今天輸入的話題」和「過去哪幾天的話題像不像」。
- 若用整段 `insights` 做向量，長文會稀釋話題相似度，例如「AI agents」和「AI coding tools」可能被拉得很近或很遠，不穩定。
- `embedding`（topic + insights）先存著，之後若要改成「按洞見語意搜尋」可以再用。

---

## 檢索邏輯（`recall_similar_memories`）

1. 讀取 `topic_memory.json`（不存在則當空陣列）。
2. 對**本次話題**算 `query_vec = embed_text(topic)`。
3. 對每筆歷史記憶，用 `topic_embedding`（沒有則 fallback 重算 `embed_text(mem["topic"])`）。
4. 算 **cosine 相似度**；`>= MEMORY_SIMILARITY_THRESHOLD`（預設 **0.35**）才算命中。
5. 依相似度排序，取前 **3** 筆（`top_k=3`）。
6. 組成 `memory_context` 字串，寫入 state，供 `synthesize` 使用。

命中後，`synthesize` 的 prompt 會多一段：

```
【跨日記憶 RAG】
以下是系統過去處理過的類似話題與洞見，請在歸納時主動對照、延續或修正：

### 2026-07-06｜AI agents（相似度 1.00）
（該日的 insights 全文）
```

`memory_hits` 會留在 state 裡，最後也會寫進 `run_*_summary.json`。

---

## Embedding 怎麼算（與 JSON 的關係）

記憶庫裡的 `topic_embedding` / `embedding` 不是呼叫外部 API，而是程式內的 **feature hashing**：

1. 把文字切成 token（英文數字、中文）。
2. 每個 token 用 **MD5** 映射到 256 維向量的某一格（`EMBED_DIM = 256`）。
3. L2 正規化後得到單位向量。
4. 兩向量做內積 = cosine 相似度。

**重點**：必須用 MD5 等穩定 hash，不能用 Python `hash()`（每次執行結果不同，舊 JSON 裡的向量就失效）。

同一套 `embed_text()` 也用於 `dedup_embed`（爬蟲去重），但那是另一條路，門檻是 **0.80**，與記憶檢索無關。

---

## 寫入邏輯（`save_memory_entry`）

每次 pipeline 成功產出 `insights` 後：

```python
entry = {
    "run_date": date.today().isoformat(),
    "topic": topic,
    "insights": insights,
    "draft_excerpt": draft[:500],
    "topic_embedding": embed_text(topic),
    "embedding": embed_text(f"{topic}\n{insights}"),
}
memories.append(entry)  # 整檔讀出 → append → 整檔寫回
```

- **追加（append）**，不覆蓋舊資料；同一天跑兩次同一話題會有兩筆。
- 寫完 JSON 後會同步更新 `memory/memory_library.md`。
- 主編是否通過、草稿有幾版，**不影響**寫入條件；只要有 `insights` 就寫。

### 種子資料（`--seed-yesterday`）

```bash
python stage3/graph.py --seed-yesterday
```

會**覆寫**整個 `topic_memory.json` 為一筆「昨天」的假資料，方便立刻測跨日引用，不必真的等一天。  
正式累積記憶請用一般執行，不要用 seed 覆蓋已有資料。

---

## `run_{topic}_summary.json`（執行報告 JSON）

這是**單次 run 的觀測紀錄**，與跨日記憶庫分開存放，每次執行覆寫/新增一份在 `outputs/`。

與記憶相關的欄位：

```json
{
  "topic": "AI agents",
  "memory_hits": [
    {
      "run_date": "2026-07-06",
      "topic": "AI agents",
      "insights": "...",
      "draft_excerpt": "...",
      "topic_embedding": [...],
      "embedding": [...],
      "similarity": 1.0
    }
  ],
  "revision_events": [...],
  "editor_reviews": [...],
  "nodes": [...],
  "cost_usd_total": 0.0546
}
```

| 欄位 | 說明 |
|------|------|
| `memory_hits` | 本次 `recall_memory` 命中的條目（含相似度），是當次 RAG 的依據 |
| `revision_events` / `editor_reviews` | 主編退回循環紀錄，與記憶庫無直接關係 |
| `nodes` | 各節點時間、token、成本 |

**差異**：

- `topic_memory.json` = 長期累積的「知識庫」
- `run_*_summary.json` = 某一次跑的「黑盒子紀錄」

---

## LangGraph state 裡的記憶欄位

執行過程中，記憶相關 state 只有兩個：

| State 鍵 | 誰寫入 | 誰讀取 |
|----------|--------|--------|
| `memory_hits` | `recall_memory` | 終端輸出、`run_*_summary.json` |
| `memory_context` | `recall_memory` | `synthesize`（必要時 `write` 也會提到跨日對照） |

JSON 檔本身**不在 state 裡**；節點透過 `load_memory()` / `save_memory_entry()` 讀寫磁碟。

---

## 常見操作

| 目的 | 指令 |
|------|------|
| 查看記憶庫 | `python stage3/graph.py --show-memory` |
| 種昨天資料後測 RAG | `python stage3/graph.py --seed-yesterday` 再跑話題 |
| 一般執行（會追加一筆記憶） | `python stage3/graph.py "AI agents"` |
| 清空記憶 | 刪除 `memory/topic_memory.json`（或改成 `[]`） |

---

## 設計取捨與限制

1. **單檔 JSON**：實作簡單，但條目變多後整檔讀寫會變慢；正式產品可換 SQLite / 向量庫。
2. **相似度門檻 0.35**：故意設得較低，讓「AI agents」「AI coding」這類相關話題有機會命中；可調 `MEMORY_SIMILARITY_THRESHOLD` 實驗。
3. **不 dedupe 記憶條目**：同一天同話題跑多次會有多筆，檢索時可能都出現。
4. **`embedding` 欄位尚未用於檢索**：目前 RAG 只看歷史 `insights` 文字，檢索鍵只有話題相似度。
5. **記憶不入 git**：每人本機累積不同；要分享可手動複製 `topic_memory.json`。

---

## 相關程式位置（`graph.py`）

| 函式 | 行為 |
|------|------|
| `load_memory()` | 讀 `topic_memory.json` → `list[dict]` |
| `save_memory_entry()` | 追加一筆並寫檔 |
| `recall_similar_memories()` | 話題相似度檢索 |
| `recall_memory()` | LangGraph 節點，產出 `memory_hits` / `memory_context` |
| `export_memory_library()` | JSON → Markdown |
| `seed_yesterday_memory()` | 寫入測試用種子資料 |

常數：`MEMORY_FILE`、`MEMORY_SIMILARITY_THRESHOLD`（0.35）、`EMBED_DIM`（256）。
