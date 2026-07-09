# Stage 5 Note — 視覺化與短影音

## 目標（對照 `PLAN.md`）

完整跑一次「話題輸入 → 短影片輸出」全程。在 stage4 的流水線後面加：

1. **visualize**：把 `authority` 篩選出的來源畫成長條圖（權威性評分），**0 LLM 呼叫**
2. **write_narration → segment_scenes → synthesize_narration_audio → generate_scene_images → compose_video**：定稿文章 → 濃縮成口語旁白稿 → 拆成 3-5 段場景 → 每段各自配音+生圖+動畫 → xfade 轉場串接成一支短影音

對應程式：`practice/stage5/graph.py`

## 流程

```
crawl → dedup_embed → authority → compress → recall_memory → synthesize
     → write → chief_editor（可退回 write）→ visualize
     → write_narration → segment_scenes → synthesize_narration_audio
     → generate_scene_images → compose_video
```

`chief_editor` 的 `approve` 跟 `revise_max_retry` 兩條路徑現在都走向 `visualize`（原本是直接 `END`）；只有 `revise` 還是繞回 `write`。也就是說：不管文章有沒有被核准，只要跑完重寫迴圈，都會產出視覺化跟短影音——練習用途不需要卡在「必須核准才產出」。

## 為什麼從單一 `short_video` 節點拆成 5 個

第一版做完自己看了才發現的問題：「靜態圖 + 旁白配音」看起來像有聲音的照片，不像影片；語音也是免費的 edge-tts，不夠自然。使用者確認願意付費換自然語音跟真的會變化的畫面，兩個決策點：

- **配音**：改用 **ElevenLabs**、指定中文「Anna Su」音色（比 edge-tts 自然，`with-timestamps` 端點一次拿到音檔+字元級對時，不用再另外呼叫轉錄服務）
- **動畫**：選「分段情境圖片＋轉場」而不是真的文字轉影片 API（Kling/Veo 一支影片約 $3-9、還要處理生成延遲跟失敗重試）——旁白拆成 3-5 段，每段配一張 OpenAI 生成的情境圖跟各自的 Ken Burns 效果，段落間用 `xfade` 淡入淡出

拆成細粒度節點（跟 `dedup_embed`/`compress`/`recall_memory` 一樣的做法）是為了在「執行總結」表裡個別看到每個外部服務（LLM/TTS/圖片）各自的時間跟成本，而不是全部混在一個節點裡看不出來哪一步貴、哪一步慢。

## 怎麼跑

```bash
cd practice
source .venv/bin/activate

# 完整跑一次（爬蟲 → 寫稿 → 配音 → 生圖 → 合成）
python stage5/graph.py "AI agents" 1

# 只重生場景圖 + 重合成影片（讀上次 run_*_summary.json）
# 適合：OpenAI billing limit 調好後，不必重跑整條 pipeline / 不必再付 ElevenLabs
python stage5/graph.py "AI agents" --rerender-images
```

## 環境需求（跟前 4 個階段不同，這階段要裝系統套件 + 兩個付費 API）

```bash
brew install ffmpeg              # 影片合成
pip install matplotlib edge-tts openai  # 圖表 + 備援語音 + 圖片生成
```

`.env` 需要：
```
OPENAI_API_KEY=...        # 場景圖片生成（gpt-image-2）
ELEVENLABS_API_KEY=...    # 配音（Anna Su 中文語音），沒設或呼叫失敗會自動降級回 edge-tts
```

**已知限制**：這裡裝的 homebrew ffmpeg 沒有帶 `libass`（沒有 `subtitles` 濾鏡），所以字幕沒有燒錄進畫面，是輸出成獨立的 `.srt` 檔——這其實是 YouTube 等平台接受字幕的標準做法之一，不是功能缺失，只是跟「字幕燒進畫面」是不同交付形式。如果要燒錄，需要 `brew reinstall ffmpeg --with-libass` 或用有帶 libass 的 build。

## 節點設計

### `visualize`
- 輸入 `scored_items`（`authority` 篩出的最多 8 筆），畫成深色背景的直式（9:16，1080x1920）長條圖
- 標籤直接畫在 bar 內部，不用 y-tick——CJK 字混英文寬度不一，靠邊界的 tick label 很容易被裁掉
- 顏色照來源區分（github 紫／hackernews 橘／arxiv 紅），圖表下方有色塊圖例
- 完全不用 LLM，跟 `dedup_embed` 一樣是「能用程式解決就不花 token」的示範
- 圖表尺寸刻意做成跟短影音一樣的 9:16，`compose_video` 沒有場景圖時直接拿來當背景保底

### `write_narration`
- 用 Haiku 把定稿文章濃縮成 150-220 字的口語旁白稿（不是逐字唸文章，是重新改寫成適合聽的版本）
- **踩到一個熟悉的坑**：模型一開始會在旁白稿前面加一行 `# 短影音旁白稿` 標題，字幕檔真的把這行讀出來了。跟 stage4 的 `chief_editor`/`authority` 一樣的模式——prompt 加強（明確說「不要用#開頭」）+ 程式碼防禦性清理（`re.sub` 砍掉開頭的 markdown 標題行）雙管齊下，不能只靠 prompt

### `segment_scenes`
- 用 Haiku 把旁白稿拆成 3-5 段場景，每段配一句畫面提示詞（`image_prompt`）
- **雙重防禦是這個節點的核心**：prompt 要求「narration_text 依序接起來必須逐字等於原始旁白稿」，但不能只信 prompt——程式碼驗證 `"".join(narration_text) == script`，驗證失敗就退回 `_mechanical_split`（照中文句尾標點`。！？`切句、貪婪塞進 3-5 個 bucket）。這個不變量非常關鍵，後面 `synthesize_narration_audio` 按字數比例分配時間軸完全依賴它才會正確
- `image_prompt` 在程式碼裡（不是靠 prompt）強制加上固定風格後綴 `IMAGE_STYLE_SUFFIX`（深色系抽象科技插畫、不要文字、直式構圖），避免模型忘記

### `synthesize_narration_audio`
- 優先用 ElevenLabs 的 `with-timestamps` 端點：先呼叫 `GET /v2/voices?search=Anna Su` 動態解析 `voice_id`（不硬編），再呼叫 `POST /v1/text-to-speech/{voice_id}/with-timestamps` 一次拿到音檔 + 字元級對時（`characters`/`character_start_times_seconds`/`character_end_times_seconds` 三個等長陣列）
- 字元級對時依中文句尾標點分組成 `.srt`——**取代了原本規劃要用的 Whisper 轉錄步驟**，少一次 API 呼叫
- 沒設 `ELEVENLABS_API_KEY`，或呼叫失敗（例如額度用完），會自動降級回 edge-tts，不會讓 pipeline 掛掉
- 場景時間軸用 `_assign_scene_times`：按每個 scene 的字數佔全文字數比例分配 `audio_seconds`，不是用 Whisper 對照原文重建邊界（那個更脆弱，中文標點斷句常跟原文對不齊）——字數比例在語速穩定時已經夠準

### `generate_scene_images`
- 每個 scene 呼叫 `gpt-image-2`（`size=1024x1536`、`quality=low`），存下 `b64_json`，用 `usage.input_tokens_details`/`output_tokens_details` 三個桶（文字輸入/圖片輸入/圖片輸出）精確算成本（不用估）
- 圖片是 2:3，畫布是 9:16，用 ffmpeg 一行 `scale=...:force_original_aspect_ratio=increase,crop=...` cover-crop 成 1080x1920，不用額外裝 Pillow
- 單一 scene 生成失敗只跳過該 scene（`image_path=""`），不中斷整個節點；完全沒有 `OPENAI_API_KEY` 或全部失敗，所有 scene 都留空，交給 `compose_video` 退回背景圖保底
- **原本用 `gpt-image-1-mini` 時 `IMAGE_STYLE_SUFFIX` 明文禁止出現文字**，因為那個模型畫文字幾乎必糊。換成 `gpt-image-2`（2026-04 發布）後拿掉這條禁令，改成鼓勵資訊圖表風格（大數字/關鍵詞/圖示標籤）——實測中文標題、百分比數字、多組圖示標籤都清晰可讀，是文字渲染品質的世代差異，不是換個牌子而已

### `compose_video`
- 每個 scene 先各自 render 成一段帶 zoompan 的靜音短片（沿用舊版的 Ken Burns 語法，範圍縮小到該 scene 的時長 + `XFADE_DURATION` 緩衝），再用 `xfade` 依序串接，最後疊上完整旁白音軌
- 分段做比直接寫一個吃 N 張圖的巨大 `filter_complex` 更容易在出錯時定位問題——跟這個檔案原本 debug zoompan/`-shortest` 的方式一致
- **`xfade` 串接的 offset 公式**（下一段轉場的 `offset` = 目前已串接長度 − 轉場秒數）是標準寫法，但不能盡信——最後仍然用明確的 `-t {audio_seconds}` 鎖定總長度當保險，跟原本 zoompan+`-shortest` 的教訓一致。實測：3 段測試片（3s/2.5s/4s + 各自轉場緩衝）串接後，`ffprobe` 量出的長度跟 `-t` 鎖定的秒數完全對齊；抽幀比對三個時間點的畫面顏色分別是紅/綠/藍，證實真的是三段不同畫面而不是同一張圖在推近
- 降級順序（絕不讓整條 pipeline 崩掉）：完全沒有場景時間資訊 → 退回舊版單一背景 zoompan；部分 scene 沒圖 → 用 `chart_path` 或純色圖墊背，時間軸照舊保留；單一 scene 短片合成失敗 → 從串接列表拿掉；最終 xfade 合成失敗 → 保留已產出的語音/字幕，`video_assets` 不含 `"video"`

## 成本追蹤

新增 `media_usage_log`（跟既有的 `usage_log` 分開存，但一起算成本），每筆付費外部服務呼叫（`elevenlabs_tts`/`openai_image`）都記錄 `node`/`service`/`cost_usd`。`print_run_summary`/`print_stage5_highlights`/`build_run_report` 都已經把這筆成本併進「執行總結」的每節點成本欄，跟原本 LLM token 成本用同一套機制呈現，不是另外開一個獨立報表。

## 踩到的坑（這次升級新增的）

- **ElevenLabs API key 的權限範圍（scope）**：一開始拿到的 key 呼叫 `GET /v2/voices` 回 `401 missing_permissions`（缺 `voices_read`）——建立 key 時選了「Restricted」而非「Full access」，需要回 ElevenLabs 後台把 Voices Read / Text to Speech 權限加開
- **「Anna Su」要先加入帳號才搜得到**：`GET /v2/voices?search=` 只會列出「已加入你帳號」的語音，不是整個 Voice Library 市集。帳號一開始只有 21 個內建語音，搜不到 Anna Su；要先到 `elevenlabs.io/app/voice-library` 搜尋並「Add to my voices」，加入後 API 才找得到（`voice_id: 9lHjugDhwqoxA5MhX0az`，语言標記 `zh` / `taiwan mandarin`）
- **OpenAI 帳號的 billing hard limit**：圖片生成呼叫回 `400 billing_hard_limit_reached`——這跟「有沒有設 key」無關，是帳號在 OpenAI 後台設定的花費上限被打到，需要去 Billing → Limits 調高。這個限制目前還沒解除，所以目前的正式跑測（見下方實測數字）圖片生成全部走降級路徑（退回背景圖），還沒驗證過真的帶場景圖片的版本
- **`_parse_and_repair_scenes` 的雙重防禦在實測中就派上用場**：分場 prompt 第一次就照句子邊界正確切分、逐字對齊原文，沒有觸發機械式切分保底——但這個保底路徑本身有先用假資料單獨測過（刻意塞一個跟原文對不上的假回覆），確認觸發後機械式切分一樣能維持「narration_text 加總 = 原文」的不變量
- **`gpt-image-1-mini` 的文字渲染幾乎必糊**：實測連圖示裡的短標籤（2-3 個字）都常出現亂碼，這是模型本身的限制，不是 prompt 沒寫好——換成 `gpt-image-2` 後直接測了一張「大標題+百分比數字+四組圖示標籤」的資訊圖表，全部清晰可讀，換模型比調 prompt 有效得多
- **`gpt-image-2` 的計價結構跟 `gpt-image-1-mini` 不一樣**：`usage` 物件把 token 拆成 `input_tokens_details.{text_tokens,image_tokens}` 跟 `output_tokens_details.{text_tokens,image_tokens}`，不是單純的 `input_tokens`/`output_tokens` 兩桶——沒注意到這點會導致算出來的成本是 0（因為新版 usage 物件上已經沒有頂層 `input_tokens` 屬性）

## 實測數字（2026-07-08，話題 "AI agents"，ElevenLabs 語音已生效，OpenAI 圖片仍卡在 billing limit）

```
圖表：chart_AI_agents.png（8 筆來源）
旁白稿：274 字元
場景：5 段（0.0s-8.9s / 8.9s-16.7s / 16.7s-26.2s / 26.2s-42.9s / 42.9s-49.8s），全部退回背景圖（OpenAI 圖片被 billing limit 擋下）
影片：video_AI_agents.mp4（1080x1920，音訊 49.8 秒，5 段場景 xfade 串接）
字幕：video_AI_agents.srt（獨立檔案，字元級對時分句，非燒錄）
配音供應商：elevenlabs
```

| 節點 | 時間 | LLM 呼叫 | 成本 |
|------|------|----------|------|
| visualize | 0.3s | 0 | $0 |
| write_narration | 4.7s | 1 | $0.0025 |
| segment_scenes | 4.0s | 1 | $0.0033 |
| synthesize_narration_audio | 8.7s | 0 | $0.0822（ElevenLabs，274 字元） |
| generate_scene_images | 3.2s | 0 | $0（billing limit 擋下，全部降級） |
| compose_video | 14.6s | 0 | $0 |

整條 pipeline（含前 4 階段 + 這 6 個新節點）合計約 164 秒、$0.229——新增的視覺化+短影音部分約佔 $0.091，其中 $0.082 是 ElevenLabs 配音（單價比原本估算的免費 edge-tts 貴不少，但語音明顯更自然）。等 OpenAI billing limit 調高、真的生成場景圖片後，還需要重跑一次補上圖片生成那筆真實成本（估算約 4-5 張圖 $0.02-0.03）。

## 跟 stage4 的差異

| 項目 | stage4 | stage5 |
|------|--------|--------|
| 產出 | 文字草稿 | 文字草稿 + 圖表 + 分段短影音（mp4+mp3+srt） |
| chief_editor 出口 | approve/max_retry → END | approve/max_retry → visualize → write_narration → ... → compose_video |
| 新增依賴 | 無 | ffmpeg（系統）、matplotlib、edge-tts、openai |
| 新增外部付費服務 | 無 | ElevenLabs（配音，可降級 edge-tts）、OpenAI 圖片生成（可降級純背景） |
| 新增 LLM 呼叫 | 無 | write_narration（旁白稿濃縮）+ segment_scenes（場景分鏡），各 1 次 Haiku |
| 動畫方式 | 單張圖表 zoompan 推近 | 每段場景各自 zoompan + xfade 轉場串接 |

## 尚未做的部分

- **字幕燒錄進畫面**：目前 ffmpeg build 沒有 libass（也沒有 `drawtext` 需要的 libfreetype），字幕是獨立 `.srt`。要燒錄需換一個有 libass 的 ffmpeg build
- **帶真實 `gpt-image-2` 場景圖片的完整 stage5 影片還沒重跑**：billing limit 已解除、`generate_scene_images` 也已經單獨測過真的會生出清晰文字的資訊圖表（見上方踩坑記錄），但還沒把整條 stage5 pipeline 重新跑一次拿到含真實圖片的實測數字（stage6 已經跑過一次完整版，見 `stage6/note.md`）
- **子圖巢狀**：跟 stage4 一樣，PLAN.md 標記為可選，目前平面結構夠用
