# Stage 3 Note — 導入 RAG 記憶

## 目標（對照 `PLAN.md`）

1. **embedding 去重**：`dedup_embed` 用 cosine 相似度取代 stage1/2 的 LLM 去重（省 token）
2. **跨日話題記憶**：本機 `memory/topic_memory.json` 持久化，第二天能引用「昨天提過類似話題」

對應程式：`practice/stage3/graph.py`

**記憶與 JSON 詳細說明**：見 [`memory.md`](memory.md)

## 流程

```
crawl → dedup_embed → authority → recall_memory → synthesize → write → chief_editor
```

- **dedup_embed**：純 Python feature hashing embedding，門檻 `0.80`
- **recall_memory**：從記憶庫檢索相似話題（門檻 `0.35`），注入 `synthesize`
- **跑完後**：自動把本次 `topic + insights` 寫回記憶庫

## 怎麼跑

### 驗收「跨日引用」（推薦兩步）

```bash
cd practice
source .venv/bin/activate

# 步驟 1：種一筆「昨天」的記憶
python stage3/graph.py --seed-yesterday

# 步驟 2：今天再跑同一話題，觀察 recall_memory 命中 + synthesize 引用
python stage3/graph.py "AI agents"
```

看輸出裡的：
- `[recall_memory] 命中 N 筆歷史記憶`
- 洞見裡是否出現「延續昨日觀點」或「與昨日不同」

### 一般執行

```bash
python stage3/graph.py "AI agents" 1

# 只查看記憶庫（不跑 pipeline、不吃 API）
python stage3/graph.py --show-memory
```

第二個參數是 `max_retry`（主編退回次數上限）。

## 執行觀測（與 stage1/2 相同）

跑完後終端會印：

- **執行總結表**：每個 agent 的時間、LLM 呼叫數、input/output tokens、成本 USD
- **合計(牆鐘)**：整次執行總秒數
- **草稿版本比較**：v0、v1… 各版字元數與相鄰差異

並寫入 `practice/stage3/outputs/`（不入版控）：

| 檔案 | 內容 |
|------|------|
| `run_{topic}_summary.json` | 完整執行報告（含 memory_hits、revision_events、editor_reviews） |
| `draft_{topic}_v0.md` … | 每輪草稿 |
| `editor_{topic}_v0.md` … | 對應版本的主編審核意見 |
| `draft_{topic}_revision_log.md` | 草稿 + 編輯意見時間軸（方便回顧） |
| `draft_{topic}_diff.txt` | 相鄰版本 unified diff |
| `draft_{topic}_revision_comparison.md` | 版本變化摘要 |
| `outputs/memory_library.md` | 本次執行時記憶庫快照（人類可讀） |
| `memory/memory_library.md` | 記憶庫即時可讀版（與 JSON 同步更新） |

**stage3 特別觀察**：`dedup_embed` 的 LLM 呼叫數應為 0（embedding 去重省 token）。

## 跟 stage2 的差異

| 項目 | stage2 | stage3 |
|------|--------|--------|
| 去重 | LLM（Haiku） | embedding（免 API） |
| 記憶 | 無 | `memory/topic_memory.json` |
| synthesize 輸入 | 只有今日資料 | 今日資料 + 歷史洞見 RAG |

## 設計筆記

- embedding 用 **feature hashing + MD5**（跨執行穩定，記憶庫可持久化），教學夠用；之後可換成真正的 embedding model
- 記憶檢索用 **`topic_embedding`**（只比對話題，不被長洞見稀釋相似度）
- 記憶庫在 `practice/stage3/memory/`，已加入 `.gitignore`（本機持久化，不入版控）
- 真的跑兩天也可以：第一天正常跑完會自動寫入記憶，第二天再跑同話題即可
