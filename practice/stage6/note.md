# Stage 6 Note — 深度研究＋三幕式長影片

## 目標（對照使用者需求：「做得更有深度更有價值」）

在 stage5 的短影音（~50 秒）之外，另外做一個功能：輸入同一個話題，產出一支最長 5 分鐘、內容有深度洞見的影片，而不是把短稿硬拉長。外部研究確認的方向：深度不是把現有輸出拉長，而是重做「研究/洞見」這一段讓內容本身更豐富，再用結構化敘事（三幕式）組織出來。

對應程式：`practice/stage6/graph.py`（stage5 的完整獨立副本再擴充，不 import stage5）

## 流程

```
crawl → dedup_embed → authority → compress → recall_memory
     → perspective_research → write → chief_editor（可退回 write）
     → visualize
     → plan_chapters → write_long_narration → segment_chapter_scenes
     → synthesize_narration_audio → generate_scene_images → compose_video
```

## 跟 stage5 的核心差異

### `perspective_research`（取代 `synthesize`）
- 借鑑 stanford-oval/STORM 的多視角研究手法（`PLAN.md` 早就點名這個技巧，但一路沒實作）：設計 3-4 個觀點明顯不同的分析者角色，各自提問、各自根據**原始資料**（不是壓縮摘要）回答，再綜合成整合洞見
- 關鍵設計：吃 `_build_raw_context(scored_items)`（壓縮前的完整素材）當主要依據，不是只吃 `compressed_context`——後者是為 stage5 單一便宜呼叫設計的精簡摘要，4 個角色都吃同一份摘要只會收斂成大同小異的空泛答案
- 雙重防禦：`_parse_and_repair_perspectives` 驗證角色數/欄位完整性，失敗**不做 LLM repair**（沿用 `chief_editor` 的教訓），改用 3 個固定角色的 `_fallback_perspectives`
- **踩到的坑**：`max_tokens=4000` 實測幾乎必定被截斷（4 個角色的具體引用回答＋4-6 段整合洞見，篇幅比預期大），觸發 `call_llm` 內建的截斷重試多花一次 API 呼叫。跟 `chief_editor`/`authority` 一樣的模式，直接把 `max_tokens` 開到 8000，不依賴重試機制兜底（已修正，見下方實測數字是修正前的版本）

### `plan_chapters`（新）
- 把定稿文章＋多視角洞見規劃成三幕式大綱：hook（固定） → 2-4 個 body（逐步遞進） → resolution（固定）
- 雙重防禦：**角色用陣列位置強制覆寫**（index 0 一定是 hook、index -1 一定是 resolution，不管模型寫的字串是什麼——位置比標籤字串可信）；`chapter_id` 一律用位置重新編號；`target_chars` 總和超出容忍帶就等比例縮放；結構整個壞掉才退回固定 5 章模板

### `write_long_narration`（新）
- 依章節大綱寫出全片旁白（目標 1500-1700 字），明確要求忠實依據多視角洞見延伸、不要憑空杜撰
- 雙重防禦是**覆蓋率不變量**，不是逐字不變量（這步是生成不是轉錄，沒有原文可逐字比對）：照大綱順序走訪每個 `chapter_id`，缺哪章就用該章的 `beat` 生成確定性 placeholder，絕不讓某一章整個消失

### `segment_chapter_scenes`（取代 `segment_scenes`）
- 逐章呼叫（4-6 次小呼叫，一章失敗不影響其他章），每章拆 2-3 段畫面——比 stage5 的 8-10 秒/鏡頭更粗的顆粒度（20-40 秒/鏡頭），符合長片的節奏
- 雙重防禦跟 stage5 `segment_scenes` 完全同一套模式（narration_text 加總＝該章原文，失敗退回機械式切分），只是不變量範圍縮小到單一章節
- `IMAGE_STYLE_SUFFIX` 延續 stage5 剛做完的教訓：guardrail（尺寸/不要文字）放程式碼，具體畫風交給模型依內容決定——實測 18 張圖確實呈現明顯不同的視覺處理（伺服器機房光流、警示紅光、天平隱喻、辦公室場景、拼圖等），不是套同一個模板

### `compose_video` 的必要 bug 修正
- stage5 寫死 `z='min(zoom+0.0012,1.3)'`：24fps 下約 10.4 秒就撞到 1.3 倍縮放上限。stage5 的鏡頭只有 8-10 秒剛好沒撞到，stage6 的鏡頭拉長到 15-40 秒**一定會**撞頂、後半段畫面凍結
- 修正：新增 `_zoompan_rate(duration_s) = ZOOM_RANGE / (duration_s * VIDEO_FPS)`，讓縮放速率跟鏡頭長度成反比，縮放剛好在片尾到頂。單元測試驗證 `_zoompan_rate(8.9)`（stage5 舊尺度）跟 `_zoompan_rate(35.0)`（stage6 新尺度）都精確落在 `ZOOM_RANGE`

### `--no-zoompan`：資訊圖表模式的靜態定格選項
- 換 `gpt-image-2` 之後場景圖是文字豐富的資訊圖表，觀眾要一邊聽一邊讀畫面上的字——Ken Burns 推近在這種畫面上反而干擾閱讀（字一直在動、邊緣還會被放大裁掉）
- 加 `--no-zoompan` 參數（`run_pipeline`/`rerender_images` 都吃 `use_zoompan`，走 `PipelineState` 傳進 `compose_video`，跟 `sequential_crawl` 同一種傳法）：場景改成靜態定格，只保留段落間的 xfade 轉場
- 驗證方式：靜態模式抽 t=0.2s 跟 t=4.5s 兩幀逐像素比對，diff 為 0（真的完全不動）；同一張圖用 zoompan 模式抽同樣兩幀 diff 為 255（確認兩種模式真的不同）
- 預設仍是 zoompan 開啟——stage5 那種無文字的情境插畫還是適合推近，這個參數是給資訊圖表內容用的

### 場景密度決策：章節級粗粒度，不是線性放大 stage5 的密度
以實測的 1835 字旁白為例：線性延用 stage5 的 ~55字/scene 密度會產生約 33 張圖；改成章節級（6 章 × 平均 3 張）只需 18 張圖，圖片成本從估算的 ~$0.13 降到實測 $0.06，且鏡頭節奏更符合長片該有的沉穩感。

## 實測數字（2026-07-09，話題 "AI agents"，真實跑測，ElevenLabs + OpenAI 圖片皆已開通）

```
多視角研究：4 個角色（CISO／跨組織互操作性架構師／治理學者／開源生態工程師）
章節大綱：6 章（hook + 4 body + resolution），三幕式，逐章遞進不是並列條列
全片旁白：1835 字元（目標 1500-1700，實際略超出但在容忍帶內）
場景：18 段（橫跨 6 章），全部有真實 OpenAI 生成的情境圖，畫風彼此明顯不同
影片：video_AI_agents.mp4（1080x1920，音訊/影片皆 340.4 秒 ≈ 5.7 分鐘）
字幕：video_AI_agents.srt（字元級對時分句，獨立檔案）
配音供應商：elevenlabs
```

| 節點 | 時間 | LLM/服務呼叫 | 成本 |
|------|------|----------|------|
| crawl~recall_memory | ~22s | 2（authority+compress） | $0.014 |
| perspective_research | 139.0s | 2（截斷重試，已修正為預期 1 次） | $0.173 |
| write | 59.3s | 2（chief_editor 退回一次） | $0.068 |
| chief_editor | 28.3s | 2 | $0.038 |
| visualize | 0.3s | 0 | $0 |
| plan_chapters | 8.9s | 1 | $0.008 |
| write_long_narration | 40.4s | 1 | $0.049 |
| segment_chapter_scenes | 20.1s | 6 | $0.019 |
| synthesize_narration_audio | 107.4s | 0（ElevenLabs，1835 字元） | **$0.551** |
| generate_scene_images | 289.0s | 0（OpenAI，18 張） | **$0.061** |
| compose_video | 176.3s | 0 | $0 |
| **合計** | **891s（~14.9 分鐘）** | | **$0.980** |

跟計劃階段的估算（$0.62-0.80）比，實際成本偏高，主要落在兩處：
1. `perspective_research` 的截斷重試（已修正，修正後預期會落在單次呼叫、成本減半）
2. 前段 `write`/`chief_editor` 因為 `chief_editor` 判定 `revise_max_retry`（觸發了一次重寫），比計劃估算的「1-2 次」偏向上限，且 `write` 目標字數提高到 800-1000 字後單次呼叫本身也比 stage5 貴

ElevenLabs 配音成本（$0.55）完全跟字數成正比，跟章節/圖片密度無關——這是這支影片最大的單筆花費，符合計劃階段的判斷。

## 跟 stage5 的差異總表

| 項目 | stage5 | stage6 |
|------|--------|--------|
| 產出長度 | ~50 秒 | ~5-6 分鐘 |
| 洞見來源 | 單一視角（`synthesize`，1 次呼叫） | 多視角研究（`perspective_research`，3-4 個角色各自提問+根據原始資料回答） |
| 敘事結構 | 攤平單層（3-5 個 scene，無章節概念） | 三幕式章節（hook/body×N/resolution），先規劃大綱再逐章寫稿 |
| 旁白字數 | 150-220 字 | 1500-1700 字（目標帶，允許 ±15%） |
| 場景密度 | ~55 字/scene（8-10 秒/鏡頭） | 章節級粗粒度（2-3 張/章，20-40 秒/鏡頭） |
| zoompan | 固定速率（>10s 鏡頭會撞頂凍結，stage5 沒踩到但是潛在 bug） | 依鏡頭長度動態算速率（`_zoompan_rate`） |
| 新增雙重防禦 | narration_text 逐字不變量 | 角色位置強制覆寫、章節覆蓋率不變量（都是不同形式的「不能只信 prompt」） |
| 單支預估成本 | ~$0.23（含前 4 階段） | ~$0.98（實測，含前段研究/寫稿/審稿） |

## 圖片模型後續升級：`gpt-image-1-mini` → `gpt-image-2`

上面「實測數字」那次真實跑測用的是 `gpt-image-1-mini`——這個模型畫文字幾乎必糊，所以當時 `IMAGE_STYLE_SUFFIX` 明文禁止出現任何文字。之後換成 `gpt-image-2`（2026-04 發布），單獨測過一張「大標題+百分比數字+多組圖示標籤」的資訊圖表，全部清晰可讀——這是文字渲染品質的世代差異，不是換個牌子而已。因此：

- `IMAGE_STYLE_SUFFIX` 拿掉「不要文字」的禁令，改成鼓勵資訊圖表風格（大數字/關鍵詞/圖示標籤），`segment_chapter_scenes` 的 image_prompt 指示也同步放寬
- 成本計價方式也不一樣：`gpt-image-2` 的 `usage` 物件把 token 拆成 `input_tokens_details.{text_tokens,image_tokens}`／`output_tokens_details.{text_tokens,image_tokens}` 三個桶分別計價（$5／$8／$30 per 1M tokens），不是 `gpt-image-1-mini` 時代單純的 `input_tokens`/`output_tokens` 兩桶——沒注意到這點算出來的成本會是 0
- **後續拿完整 pipeline 跑了一次 `gpt-image-2` 版本，確認資訊圖表效果，但發現並修正了一個真實 bug**：`OPENAI_IMAGE_SIZE="1024x1536"`（2:3）跟畫布 9:16 長寬比差很多，`_cover_crop` 裁掉約 100px/邊（以 1024 寬計），資訊圖表常把文字放在版面邊緣（多欄位/流程圖左右兩側常有標籤），實測真的看到文字被裁掉一半的畫面。改成 `1024x1792`（0.571，非常接近 9:16 的 0.5625）後 crop 只需裁掉約 8px/邊，重新測一張刻意左右都放文字欄位的複雜流程圖，完全沒有被裁切。同時在 `IMAGE_STYLE_SUFFIX` 加了一句「重要文字與版面元素請集中在畫面中央，避免緊貼左右邊緣」當第二層保險（尺寸修正是主要解法，這句是輔助）
- **完整 pipeline 重跑後的真實數字**（2026-07-09 第二次跑測，話題同樣是 "AI agents"）：`perspective_research` 這次只呼叫 1 次（先前 `max_tokens=4000→8000` 的修正生效，沒有再觸發截斷重試），6 章、18 段場景、旁白 1661 字元、影片 309.5 秒（≈5.16 分，比第一次的 5.7 分更接近 5 分鐘目標）、總成本 $0.92（含圖片 $0.094，比第一次的 $0.061 略高是因為換了 `gpt-image-2` 定價更貴，但文字終於清晰可讀）

## 尚未做的部分

- **`perspective_research` 的 `max_tokens` 修正還沒重新真實跑測驗證**：已改成 8000，語法檢查通過、邏輯上應該一次到位，但還沒花真錢再跑一次確認不再觸發截斷重試
- **字幕燒錄進畫面**：跟 stage5 一樣的限制，這裡裝的 ffmpeg 沒有 libass，字幕維持獨立 `.srt`
- **v1 沒有加深爬蟲來源**：`perspective_research` 吃的原始素材還是跟 stage5 一樣的 `authority` 篩選結果（最多 8 筆、每筆約 1200 字元），沒有額外抓取完整文章 HTML——目前這個量對 3-4 個角色的提問/回答已經足夠，如果之後想要更深的研究依據，這是下一個可以擴充的方向
- **影片實際長度略超過 5 分鐘目標**（340.4 秒 ≈ 5.7 分）：`CHARS_PER_MINUTE_PACE=330` 是根據 stage5 一支 274 字元的短稿量出來的，長稿的 ElevenLabs 實際語速跟短稿不完全一樣，這個常數目前只當 sanity-check 參考，沒有拿來反向控制旁白字數上限——如果要更精準卡在 5 分鐘內，可以考慮把 `TARGET_SCRIPT_CHARS_MAX` 依實測語速往下修
