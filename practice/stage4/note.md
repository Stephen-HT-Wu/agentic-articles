# Stage 4 Note — 平行化與壓縮

## 目標（對照 `PLAN.md`）

1. **多來源平行爬蟲**：3 個來源（HN、GitHub、arXiv）用 `ThreadPoolExecutor` 平行抓取，量出加速倍數
2. **壓縮節點**：借鑑 open_deep_research 的 `compress_research`，把 authority 篩選後的完整摘要
   壓縮成精簡筆記再送進 `synthesize`，用官方 `count_tokens` 端點量出壓縮前後的 token 差異

對應程式：`practice/stage4/graph.py`

## 流程

```
crawl → dedup_embed → authority → compress → recall_memory → synthesize → write → chief_editor
```

- **crawl**：3 個來源（`_fetch_hackernews`、`_fetch_github`、`_fetch_arxiv`），每個回傳 `(items, 花費秒數)`
  - 預設用 `ThreadPoolExecutor` 平行跑；加 `--sequential` 改成逐一序列呼叫
  - 特意加入 **arXiv**（免 key、論文摘要 800-1500 字元）當第三個來源，讓壓縮節點的「前後差異」有感——HN/GitHub 的摘要通常很短，壓不出什麼東西
- **compress**：把 8 筆 `scored_items` 的完整摘要串起來，量 token 數，丟給 Haiku 壓縮成一段筆記，再量一次 token 數
- **synthesize**：吃 `compress` 吐出的筆記，不是自己重新組 `scored_items` 的完整內容——這才是省 token 真正發生的地方

## 怎麼跑

```bash
cd practice
source .venv/bin/activate

# 平行爬蟲（預設）
python stage4/graph.py "AI agents" 1

# 序列爬蟲，跟上面比較耗時
python stage4/graph.py "AI agents" 1 --sequential

# 其餘旗標跟 stage3 相同
python stage4/graph.py "AI agents" 1 --bad-first     # 測試主編退回
python stage4/graph.py --seed-yesterday               # 種一筆昨天的記憶
python stage4/graph.py --show-memory                  # 只看記憶庫，不吃 API
```

## 實測數字（2026-07-08，話題 "AI agents"）

| 項目 | 平行 | 序列 |
|------|------|------|
| 各來源耗時 | arxiv 0.22s / github 0.27s / hackernews 1.23s | hackernews 1.86s / github 0.26s / arxiv 0.15s |
| 序列預估 | 1.72s | 2.27s |
| 實際牆鐘 | **1.23s** | 2.27s |
| 加速倍數 | **1.4x** | 1.0x（基準） |

> 三個來源都很快，加速幅度不算誇張；來源數量越多、越慢（例如換成更多家慢速 API），平行化的效益會更明顯。

| 項目 | 數值 |
|------|------|
| 壓縮前（原始摘要串接） | 1,918 tokens |
| 壓縮後（Haiku 濃縮筆記） | 543 tokens |
| 省下 | 1,375 tokens（**71.7%**） |
| 壓縮節點本身花費 | 1 次 Haiku 呼叫，約 $0.005 |

換算下來：壓縮節點自己花一點小錢（Haiku 很便宜），換到 synthesize（Sonnet，貴很多）少處理 1,375 個 input tokens——這筆交易划算，且來源摘要越長（例如接入更多論文/長文），效益會更明顯。

## 執行觀測（與 stage1-3 相同 + 兩個新增區塊）

跑完除了 stage1-3 就有的執行總結表、草稿版本比較，新增：

```
階段 4 觀察重點
------------------------------------------------------------------------
爬蟲：平行｜各來源耗時 {...}
      序列預估 1.72s vs 實際牆鐘 1.23s（加速 1.4x）
壓縮：原始 1,918 tokens -> 壓縮後 543 tokens（省 1,375 tokens，71.7%）
```

`run_{topic}_summary.json` 新增兩個欄位：`crawl_parallelism`（各來源耗時、序列預估、實際牆鐘、加速倍數）、
`compression_stats`（壓縮前後 token 數、省下比例）。其餘欄位（`memory_hits`、`revision_events`、
`editor_reviews`、`dedup_method`）跟 stage3 相同。

**驗證重點**：`dedup_embed` 的 `llm_calls` 仍是 0（跟 stage3 一樣用 embedding，不吃 LLM token）。

## 跟 stage3 的差異

| 項目 | stage3 | stage4 |
|------|--------|--------|
| 爬蟲 | 2 來源、序列 | 3 來源、平行（可切回序列比較） |
| 資料流向 synthesize | 直接組 `scored_items` 的 listing | 先經過 `compress` 壓縮 |
| 記憶庫 | `stage3/memory/` | `stage4/memory/`（各階段隔離，互不影響） |
| 新增觀測 | — | `crawl_parallelism`、`compression_stats` |

## 尚未做的部分（PLAN.md 標記為「可選」）

- **子圖巢狀**：把 `crawl + dedup_embed + authority` 包成一個 subgraph 節點（open_deep_research 的
  `research_supervisor` 模式）。目前這個平面結構已經夠用，先不做，等真的需要重用這段邏輯或想練習
  subgraph 語法時再加。
