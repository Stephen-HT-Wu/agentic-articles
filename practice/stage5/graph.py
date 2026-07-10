"""
階段 5：視覺化與短影音

目標（PLAN.md）：完整跑一次「話題輸入 -> 短影片輸出」全程。

在 stage4 的流水線後面加：
1. visualize：把 authority 篩選出的來源畫成長條圖（權威性評分），0 LLM 呼叫
2. write_narration：把定稿文章濃縮成口語旁白稿（Haiku）
3. segment_scenes：把旁白稿拆成 3-5 段場景，每段配一句畫面提示詞（Haiku + 雙重防禦）
4. synthesize_narration_audio：ElevenLabs（Anna Su 中文語音）配音 + 字元級對時產生字幕，
   沒設 key 或呼叫失敗時降級回免費的 edge-tts
5. generate_scene_images：每段場景各生成一張情境圖（OpenAI gpt-image-1-mini）+ 裁成 9:16
6. compose_video：每段場景各自 zoompan 動畫 + xfade 轉場串接，疊上完整旁白音軌

流程：
    crawl -> dedup_embed -> authority -> compress -> recall_memory -> synthesize
         -> write -> chief_editor（可退回 write）-> visualize
         -> write_narration -> segment_scenes -> synthesize_narration_audio
         -> generate_scene_images -> compose_video

環境需求：
    brew install ffmpeg          # 影片合成（這裡用的 build 沒有 libass，
                                  # 所以字幕是輸出成獨立 .srt，不是燒錄進畫面）
    pip install matplotlib edge-tts openai
    .env 需要 OPENAI_API_KEY（圖片生成）、ELEVENLABS_API_KEY（配音，未設就降級用 edge-tts）

執行範例：
    python stage5/graph.py "AI agents" 1
    python stage5/graph.py "AI agents" --rerender-images   # 只重生圖 + 重合成影片
"""

import warnings

import langchain_core  # noqa: F401
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

import argparse
import base64
import difflib
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, TypedDict

import sys

import anthropic
import edge_tts
import matplotlib
import requests
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

matplotlib.use("Agg")  # 不開視窗、純存檔用，跑在沒有畫面的環境也不會報錯
import matplotlib.pyplot as plt  # noqa: E402 — 一定要在 matplotlib.use() 之後才 import

# 共用基礎設施（只有經過驗證完全一樣的 instrument/cost_of/PRICING，見 _common.py 開頭說明）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import PRICING, cost_of, current_node, instrument, node_times, reset_metrics

# ---------------------------------------------------------------------------
# 環境設定
# ---------------------------------------------------------------------------

_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _value = _line.partition("=")
            os.environ.setdefault(_key.strip(), _value.strip())

CHEAP_MODEL = "claude-haiku-4-5-20251001"
SMART_MODEL = "claude-sonnet-5"

client = anthropic.Anthropic()

# 模型的訓練截止日期在實際執行日之前，本身不知道「今天」是幾號——沒有這行，
# 會把爬到的近期日期／記憶庫的 run_date 誤判成「未來、不合理」而在洞見/草稿/審稿意見裡瞎質疑。
# 有判斷時間合理性需求的節點（synthesize/write/chief_editor）都要把這行接到 system prompt 裡。
TODAY_STR = date.today().isoformat()
DATE_GROUNDING = f"今天的實際日期是 {TODAY_STR}。內容中提到接近這個日期的時間點都是正常最新資訊，不是未來或異常日期，不要因為日期「看起來太新」就質疑真實性。"

# 付費配音/圖片供應商——沒設對應 key 時整個降級成免費的 edge-tts / 純色背景，不會讓 pipeline 掛掉。
try:
    from openai import OpenAI
    openai_client = OpenAI() if os.environ.get("OPENAI_API_KEY") else None
except Exception:
    openai_client = None

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")

MEMORY_DIR = Path(__file__).resolve().parent / "memory"
MEMORY_FILE = MEMORY_DIR / "topic_memory.json"

DEDUP_SIMILARITY_THRESHOLD = 0.80
MEMORY_SIMILARITY_THRESHOLD = 0.35
EMBED_DIM = 256

# 短影音設定：直式（9:16），跟主流短影音平台一致
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
VIDEO_FPS = 24
VOICE = "zh-TW-HsiaoyuNeural"  # edge-tts 備援語音（沒設 ELEVENLABS_API_KEY 時使用）
CJK_FONT = "Heiti TC"  # matplotlib 預設字型不含中文字形，macOS 上這個字型有涵蓋
SOURCE_COLORS = {
    "github": "#6e5494",
    "hackernews": "#ff6600",
    "arxiv": "#b31b1b",
}

# ElevenLabs：配音 + 逐字對時（一次呼叫拿到兩者，不用另外呼叫轉錄服務）
ELEVENLABS_VOICE_NAME = "Anna Su"
ELEVENLABS_MODEL = "eleven_multilingual_v2"
# 免費額度用完後的超額費率，依實際方案可能不同——只當估算，不是精確帳單。
ELEVENLABS_OVERAGE_PER_1K_CHARS = 0.30

# OpenAI 圖片：每個 scene 一張情境圖，當作該段的動畫背景
# gpt-image-1-mini 的文字渲染常常是亂碼（實測連小圖示裡的短標籤都會糊），
# 換成 gpt-image-2（2026-04 發布）——文字準確度大幅提升，實測中文標題/大數字/
# 圖示標籤都清晰可讀，才有辦法做真正的資訊圖表風格，不只是裝飾性插畫。
OPENAI_IMAGE_MODEL = "gpt-image-2"
# 1024x1536（2:3=0.667）跟畫布 1080x1920（9:16=0.5625）長寬比差很多，cover-crop
# 會裁掉約 100px/邊（以 1024 寬計），資訊圖表常把文字放在版面邊緣，實測真的會被切字。
# 1024x1792（0.571）非常接近 9:16，crop 只需裁掉約 8px/邊，幾乎不影響版面完整性。
OPENAI_IMAGE_SIZE = "1024x1792"
OPENAI_IMAGE_QUALITY = "low"
# 只放不能妥協的硬限制（尺寸、人物肖像的肖像權風險、邊緣留白給 cover-crop 當保險）；
# 拿掉「不要文字」的舊限制——那是為文字容易糊掉的 gpt-image-1-mini 設的防禦，
# 換成 gpt-image-2 後改成鼓勵資訊圖表風格（大數字/關鍵詞/簡短標籤）。
IMAGE_STYLE_SUFFIX = (
    "，深色系資訊圖表風格，可包含簡短的中文文字標籤或數字重點，字要大而清楚，"
    "重要文字與版面元素請集中在畫面中央，避免緊貼左右邊緣，不要出現人物肖像，直式構圖"
)
# gpt-image-2 依 usage.input_tokens_details/output_tokens_details 分別計價
# （文字輸入/圖片輸入/圖片輸出三個桶），比 gpt-image-1-mini 時代單一 input/output
# 兩桶更細——這裡對照官方定價頁的估算值，實際帳單校正見 note.md。
OPENAI_IMAGE_PRICE_PER_1M_TEXT_INPUT = 5.00
OPENAI_IMAGE_PRICE_PER_1M_IMAGE_INPUT = 8.00
OPENAI_IMAGE_PRICE_PER_1M_IMAGE_OUTPUT = 30.00

SCENE_COUNT_MIN = 3
SCENE_COUNT_MAX = 5
XFADE_DURATION = 0.5  # 段落間轉場秒數

usage_log: list = []
media_usage_log: list = []  # TTS/圖片等付費外部服務的呼叫紀錄，跟 usage_log 分開存但一起算成本
revision_events: list = []
editor_reviews: list = []


def print_run_summary(total_wall_s: float) -> None:
    print(f"\n{'=' * 72}")
    print("執行總結")
    print(f"{'-' * 72}")
    print(
        f"{'節點':<16}{'節點時間':>10}{'LLM呼叫':>8}"
        f"{'輸入tokens':>12}{'輸出tokens':>12}{'成本USD':>12}"
    )
    total_cost = 0.0
    for name in node_times:
        calls = [entry for entry in usage_log if entry["node"] == name]
        tokens_in = sum(entry["input"] for entry in calls)
        tokens_out = sum(entry["output"] for entry in calls)
        cost = sum(cost_of(entry) for entry in calls)
        media_cost = sum(entry["cost_usd"] for entry in media_usage_log if entry["node"] == name)
        total_cost += cost + media_cost
        print(
            f"{name:<16}{node_times[name]:>9.1f}s{len(calls):>8}"
            f"{tokens_in:>12,}{tokens_out:>12,}{cost + media_cost:>12.4f}"
        )
    total_node_s = sum(node_times.values())
    total_in = sum(entry["input"] for entry in usage_log)
    total_out = sum(entry["output"] for entry in usage_log)
    print(f"{'-' * 72}")
    print(
        f"{'合計(節點)':<16}{total_node_s:>9.1f}s{len(usage_log):>8}"
        f"{total_in:>12,}{total_out:>12,}{total_cost:>12.4f}"
    )
    if media_usage_log:
        media_total = sum(entry["cost_usd"] for entry in media_usage_log)
        print(f"{'  含媒體成本':<16}{'':>10}{'':>8}{'':>12}{'':>12}{media_total:>12.4f}")
    print(f"{'合計(牆鐘)':<16}{total_wall_s:>9.1f}s")


def print_stage4_highlights(result: dict) -> None:
    """階段 4 特有的兩個觀察重點：平行加速倍數、壓縮省下的 token 比例。"""
    print(f"\n{'=' * 72}")
    print("階段 4 觀察重點")
    print(f"{'-' * 72}")
    parallelism = result.get("crawl_parallelism") or {}
    if parallelism:
        print(
            f"爬蟲：{parallelism['mode']}｜各來源耗時 {parallelism['source_seconds']}"
        )
        print(
            f"      序列預估 {parallelism['sequential_estimate_s']}s vs "
            f"實際牆鐘 {parallelism['wall_s']}s（加速 {parallelism['speedup_x']}x）"
        )
    stats = result.get("compression_stats") or {}
    if stats:
        print(
            f"壓縮：原始 {stats['raw_tokens']:,} tokens -> "
            f"壓縮後 {stats['compressed_tokens']:,} tokens"
            f"（省 {stats['saved_tokens']:,} tokens，{stats['saved_pct']}%）"
        )


def print_stage5_highlights(result: dict) -> None:
    """階段 5 特有的觀察重點：圖表、場景分鏡、短影音產出、媒體服務成本明細。"""
    print(f"\n{'=' * 72}")
    print("階段 5 觀察重點")
    print(f"{'-' * 72}")
    chart_path = result.get("chart_path") or ""
    if chart_path:
        print(f"圖表：{chart_path}")
    script = result.get("video_script") or ""
    if script:
        print(f"旁白稿（{len(script)} 字元）：{script[:80]}...")
    scenes = result.get("video_scenes") or []
    if scenes:
        print(f"場景分鏡（{len(scenes)} 段）：")
        for i, scene in enumerate(scenes):
            has_image = "有圖" if scene.get("image_path") else "無圖(退回背景)"
            print(
                f"  - scene {i}：{scene.get('start_s', '?')}s-{scene.get('end_s', '?')}s"
                f"｜{has_image}｜{scene.get('narration_text', '')[:24]}..."
            )
    assets = result.get("video_assets") or {}
    if assets.get("video"):
        print(
            f"影片：{assets['video']}"
            f"（{assets.get('scene_count', '?')} 段場景，音訊 {assets.get('audio_seconds', '?')} 秒）"
        )
        print(f"字幕：{assets.get('srt')}（獨立檔案，未燒錄進畫面——見檔頭說明）")
    elif assets.get("audio"):
        print(f"語音：{assets['audio']}（ffmpeg 合成影片失敗或未安裝，只留語音+字幕）")
    if assets.get("tts_provider"):
        print(f"配音供應商：{assets['tts_provider']}")
    if media_usage_log:
        print("媒體服務成本明細：")
        for entry in media_usage_log:
            extra = entry.get("chars")
            if extra is None:
                extra = entry.get("scene")
            suffix = f"（{extra}）" if extra is not None else ""
            print(f"  - {entry['service']}｜node={entry['node']}｜${entry['cost_usd']:.4f}{suffix}")


def build_run_report(result: dict, total_wall_s: float) -> dict:
    nodes = []
    total_cost = 0.0
    for name in node_times:
        calls = [entry for entry in usage_log if entry["node"] == name]
        tokens_in = sum(entry["input"] for entry in calls)
        tokens_out = sum(entry["output"] for entry in calls)
        cost = sum(cost_of(entry) for entry in calls)
        media_calls = [entry for entry in media_usage_log if entry["node"] == name]
        media_cost = sum(entry["cost_usd"] for entry in media_calls)
        total_cost += cost + media_cost
        nodes.append(
            {
                "node": name,
                "seconds": round(node_times[name], 3),
                "llm_calls": len(calls),
                "input_tokens": tokens_in,
                "output_tokens": tokens_out,
                "cost_usd": round(cost, 6),
                "media_cost_usd": round(media_cost, 6),
            }
        )

    versions = result.get("draft_versions") or []
    version_stats = [
        {"version": i, "chars": len(text), "file": f"draft_v{i}.md"}
        for i, text in enumerate(versions)
    ]
    for i in range(1, len(versions)):
        version_stats[i]["delta_chars"] = len(versions[i]) - len(versions[i - 1])

    return {
        "topic": result.get("topic"),
        "wall_seconds": round(total_wall_s, 3),
        "node_seconds_total": round(sum(node_times.values()), 3),
        "llm_calls_total": len(usage_log),
        "input_tokens_total": sum(entry["input"] for entry in usage_log),
        "output_tokens_total": sum(entry["output"] for entry in usage_log),
        "cost_usd_total": round(total_cost, 6),
        "media_cost_usd_total": round(sum(entry["cost_usd"] for entry in media_usage_log), 6),
        "media_usage_log": media_usage_log,
        "nodes": nodes,
        "draft_versions": version_stats,
        "revision_events": revision_events,
        "editor_reviews": editor_reviews,
        "memory_hits": result.get("memory_hits", []),
        "dedup_method": "embedding",
        "crawl_parallelism": result.get("crawl_parallelism", {}),
        "compression_stats": result.get("compression_stats", {}),
        "chart_path": result.get("chart_path", ""),
        "chart_stats": result.get("chart_stats", {}),
        "video_script": result.get("video_script", ""),
        "video_scenes": result.get("video_scenes", []),
        "video_assets": result.get("video_assets", {}),
    }


def print_draft_version_summary(versions: list) -> None:
    if not versions:
        return
    print(f"\n{'=' * 72}")
    print("草稿版本比較")
    print(f"{'-' * 72}")
    print(f"{'版本':<8}{'字元數':>10}{'相對上一版':>14}")
    for i, text in enumerate(versions):
        delta = ""
        if i > 0:
            diff = len(text) - len(versions[i - 1])
            delta = f"{diff:+d}"
        print(f"v{i:<7}{len(text):>10}{delta:>14}")


def build_revision_log(slug: str, versions: list, reviews: list) -> str:
    lines = [f"# 草稿修訂記錄：{slug.replace('_', ' ')}\n"]
    review_by_version = {r["draft_version"]: r for r in reviews}
    for i, draft in enumerate(versions):
        lines.append(f"## v{i} 草稿")
        lines.append(f"- 字元數：{len(draft)}")
        lines.append(f"- 檔案：`draft_{slug}_v{i}.md`\n")
        review = review_by_version.get(i)
        if review:
            lines.append(f"### 主編審核（第 {review.get('retry_count', 0)} 輪）")
            lines.append(f"- 決定：**{review['decision']}**")
            feedback = (review.get("feedback") or "").strip()
            lines.append(f"- 意見：\n\n{feedback or '（無）'}\n")
    if not versions:
        lines.append("（本次無草稿版本）\n")
    return "\n".join(lines)


def save_run_artifacts(result: dict, slug: str, total_wall_s: float, output_dir: Path) -> dict:
    output_dir.mkdir(exist_ok=True)
    paths: dict = {}

    report = build_run_report(result, total_wall_s)
    summary_path = output_dir / f"run_{slug}_summary.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    paths["summary"] = str(summary_path)

    versions = result.get("draft_versions") or []
    version_paths = []
    for i, draft in enumerate(versions):
        vf = output_dir / f"draft_{slug}_v{i}.md"
        vf.write_text(draft)
        version_paths.append(str(vf))
    paths["draft_versions"] = version_paths

    feedback_paths = []
    for review in editor_reviews:
        v = review["draft_version"]
        body = (
            f"# 主編審核 — v{v}\n\n"
            f"- 決定：{review['decision']}\n"
            f"- 輪次（retry_count）：{review.get('retry_count', 0)}\n\n"
            f"## 意見\n\n{(review.get('feedback') or '').strip() or '（無）'}\n"
        )
        ef = output_dir / f"editor_{slug}_v{v}.md"
        ef.write_text(body)
        feedback_paths.append(str(ef))
    if feedback_paths:
        paths["editor_feedback"] = feedback_paths

    if versions or editor_reviews:
        log_path = output_dir / f"draft_{slug}_revision_log.md"
        log_path.write_text(build_revision_log(slug, versions, editor_reviews))
        paths["revision_log"] = str(log_path)

    if len(versions) > 1:
        diff_lines = []
        comparison_lines = ["# 草稿版本差異摘要\n"]
        for i in range(len(versions) - 1):
            a = versions[i].splitlines(keepends=True)
            b = versions[i + 1].splitlines(keepends=True)
            diff_lines.extend(
                difflib.unified_diff(
                    a, b,
                    fromfile=f"draft_{slug}_v{i}.md",
                    tofile=f"draft_{slug}_v{i+1}.md",
                )
            )
            diff_lines.append("\n")
            comparison_lines.append(
                f"## v{i} → v{i+1}\n"
                f"- v{i}: {len(versions[i])} 字元\n"
                f"- v{i+1}: {len(versions[i+1])} 字元\n"
                f"- 變化: {len(versions[i+1]) - len(versions[i]):+d} 字元\n"
            )
        diff_path = output_dir / f"draft_{slug}_diff.txt"
        diff_path.write_text("".join(diff_lines))
        paths["diff"] = str(diff_path)

        comparison_path = output_dir / f"draft_{slug}_revision_comparison.md"
        comparison_path.write_text("\n".join(comparison_lines))
        paths["revision_comparison"] = str(comparison_path)

    if not versions and result.get("draft"):
        fallback = output_dir / f"draft_{slug}.md"
        fallback.write_text(result["draft"])
        paths["draft"] = str(fallback)

    export_memory_library(output_dir / "memory_library.md")
    paths["memory_library"] = str(output_dir / "memory_library.md")
    paths["memory_json"] = str(MEMORY_FILE)

    return paths


def call_llm(model: str, system: str, user: str, max_tokens: int = 2000) -> str:
    def _create(tokens: int):
        return client.messages.create(
            model=model,
            max_tokens=tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )

    for attempt in range(2):
        tokens = max_tokens if attempt == 0 else min(max_tokens * 2, 8000)
        response = _create(tokens)
        usage_log.append(
            {
                "node": current_node(),
                "model": model,
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
            }
        )
        # 被截斷就直接重試（加大 max_tokens），不管有沒有部分文字——
        # 否則截斷但非空的回覆會被下面 `if text_parts` 提前 return，永遠輪不到重試。
        if getattr(response, "stop_reason", None) == "max_tokens" and attempt == 0:
            continue
        text_parts = [
            block.text
            for block in getattr(response, "content", []) or []
            if getattr(block, "type", None) == "text" and hasattr(block, "text")
        ]
        if text_parts:
            return "".join(text_parts)
        output_text = getattr(response, "output_text", "") or ""
        if output_text.strip():
            return output_text
        raise ValueError("模型回覆中沒有可用文字輸出。")
    raise ValueError("模型回覆中沒有可用文字輸出（已重試）。")


def count_tokens(text: str, model: str = CHEAP_MODEL) -> int:
    """
    量測一段文字的 token 數，用來比較壓縮前後的差異。
    這是官方 count_tokens 端點，不產生任何生成內容，所以不計入 usage_log/成本。
    """
    if not text:
        return 0
    result = client.messages.count_tokens(model=model, messages=[{"role": "user", "content": text}])
    return result.input_tokens


def _slugify(topic: str) -> str:
    return re.sub(r"[^一-鿿a-zA-Z0-9]+", "_", topic)


def extract_json(text: str):
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
    else:
        start = min(
            (i for i in (text.find("["), text.find("{")) if i != -1), default=-1
        )
        if start == -1:
            raise ValueError(f"回覆中找不到 JSON：{text[:200]}")
        candidate = text[start:].strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    obj = re.search(r"\{[\s\S]*\}", candidate)
    if obj:
        return json.loads(obj.group(0))
    arr = re.search(r"\[[\s\S]*\]", candidate)
    if arr:
        return json.loads(arr.group(0))
    raise ValueError(f"回覆 JSON 解析失敗：{candidate[:200]}")


def _preview_update(node_name: str, update: dict) -> None:
    print(f"\n{'=' * 72}")
    print(f"節點：{node_name}")
    for key, value in update.items():
        if key == "memory_hits":
            print(f"  {key}：{len(value)} 筆")
            for hit in value[:3]:
                print(
                    f"    - {hit.get('run_date')} | {hit.get('topic')} "
                    f"(相似度 {hit.get('similarity', 0):.2f})"
                )
            continue
        if key in ("crawl_parallelism", "compression_stats", "chart_stats", "video_assets"):
            print(f"  {key}：{json.dumps(value, ensure_ascii=False)}")
            continue
        if isinstance(value, list):
            print(f"  {key}：{len(value)} 筆")
            for item in value[:3]:
                print(f"    - {json.dumps(item, ensure_ascii=False)}")
            if len(value) > 3:
                print(f"    ... 還有 {len(value) - 3} 筆")
        elif isinstance(value, str) and len(value) > 400:
            print(f"  {key}：({len(value)} 字元)")
            print(f"    {value[:400]}...")
        else:
            print(f"  {key}：{value}")


# ---------------------------------------------------------------------------
# Embedding 工具（跟 stage3 相同：純 Python feature hashing）
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    text = text.lower()
    return re.findall(r"[a-z0-9一-鿿]+", text)


def embed_text(text: str, dim: int = EMBED_DIM) -> list[float]:
    vec = [0.0] * dim
    for token in _tokenize(text):
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        idx = int(digest, 16) % dim
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def dedup_by_embedding(items: list, threshold: float = DEDUP_SIMILARITY_THRESHOLD) -> list:
    kept: list = []
    kept_vecs: list[list[float]] = []
    for item in items:
        text = f"{item.get('title', '')} {item.get('snippet', '')}"
        vec = embed_text(text)
        if any(cosine_similarity(vec, prev) >= threshold for prev in kept_vecs):
            continue
        kept.append(item)
        kept_vecs.append(vec)
    return kept


# ---------------------------------------------------------------------------
# 跨日記憶（跟 stage3 相同，但存到 stage4 自己的 memory/ 目錄，跟 stage3 隔離）
# ---------------------------------------------------------------------------

def load_memory() -> list:
    if not MEMORY_FILE.exists():
        return []
    return json.loads(MEMORY_FILE.read_text())


def memory_entry_for_display(entry: dict) -> dict:
    return {
        "run_date": entry.get("run_date"),
        "topic": entry.get("topic"),
        "insights": entry.get("insights"),
        "draft_excerpt": entry.get("draft_excerpt"),
    }


def export_memory_library(dest: Optional[Path] = None) -> str:
    memories = load_memory()
    lines = [
        "# 跨日記憶庫",
        "",
        f"來源 JSON：`{MEMORY_FILE}`",
        f"共 {len(memories)} 筆",
        "",
    ]
    if not memories:
        lines.append("（記憶庫為空）")
    for i, mem in enumerate(memories):
        display = memory_entry_for_display(mem)
        lines.append(f"## [{i}] {display['run_date']} — {display['topic']}")
        lines.append("")
        lines.append("### 洞見")
        lines.append("")
        lines.append(display.get("insights") or "（無）")
        lines.append("")
        lines.append("### 草稿摘要")
        lines.append("")
        lines.append(display.get("draft_excerpt") or "（無）")
        lines.append("")
        lines.append("---")
        lines.append("")
    text = "\n".join(lines)
    if dest is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
    return text


def print_memory_library(max_chars: int = 8000) -> None:
    text = export_memory_library(MEMORY_DIR / "memory_library.md")
    print(f"\n{'=' * 72}")
    print("跨日記憶庫內容")
    print(f"{'-' * 72}")
    if len(text) <= max_chars:
        print(text)
    else:
        print(text[:max_chars])
        print(f"\n...（其餘 {len(text) - max_chars} 字元，完整內容見 memory/memory_library.md）")
    print(f"\nJSON 原始檔：{MEMORY_FILE}")


def save_memory_entry(topic: str, insights: str, draft: str) -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    memories = load_memory()
    entry = {
        "run_date": date.today().isoformat(),
        "topic": topic,
        "insights": insights,
        "draft_excerpt": draft[:500],
        "topic_embedding": embed_text(topic),
        "embedding": embed_text(f"{topic}\n{insights}"),
    }
    memories.append(entry)
    MEMORY_FILE.write_text(json.dumps(memories, ensure_ascii=False, indent=2))
    export_memory_library(MEMORY_DIR / "memory_library.md")


def recall_similar_memories(topic: str, top_k: int = 3) -> list:
    memories = load_memory()
    if not memories:
        return []
    query_vec = embed_text(topic)
    scored = []
    for mem in memories:
        mem_vec = mem.get("topic_embedding") or embed_text(mem.get("topic", ""))
        sim = cosine_similarity(query_vec, mem_vec)
        if sim >= MEMORY_SIMILARITY_THRESHOLD:
            scored.append({**mem, "similarity": round(sim, 3)})
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_k]


def seed_yesterday_memory() -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    topic = "AI agents"
    insights = (
        "# 昨日洞見（種子資料）\n\n"
        "- **通用 agent 敘事降溫**：市場從 AutoGPT 式萬能自主，轉向 OpenCode、browser-use 等垂直工具。\n"
        "- **框架路線分化**：LangChain（程式碼優先）與 Dify/Langflow（低代碼）長期並存。\n"
        "- **評測可信度成隱憂**：benchmark 可能被針對性優化，與真實任務能力有落差。"
    )
    entry = {
        "run_date": yesterday,
        "topic": topic,
        "insights": insights,
        "draft_excerpt": "（昨日草稿摘要）垂直化 agent 比通用 agent 更容易落地。",
        "topic_embedding": embed_text(topic),
        "embedding": embed_text(f"{topic}\n{insights}"),
    }
    MEMORY_FILE.write_text(json.dumps([entry], ensure_ascii=False, indent=2))
    export_memory_library(MEMORY_DIR / "memory_library.md")
    print(f"已寫入種子記憶：{MEMORY_FILE}")
    print(f"  run_date={yesterday}, topic={topic}")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class PipelineState(TypedDict):
    topic: str
    sequential_crawl: bool
    raw_items: list
    crawl_parallelism: dict
    dedup_items: list
    scored_items: list
    compressed_context: str
    compression_stats: dict
    memory_hits: list
    memory_context: str
    insights: str
    draft: str
    draft_versions: list
    editor_feedback: str
    retry_count: int
    max_retry: int
    force_bad_first_draft: bool
    chart_path: str
    chart_stats: dict
    video_script: str
    video_scenes: list
    video_assets: dict


# ---------------------------------------------------------------------------
# 爬蟲來源（每個回傳 (items, 花費秒數)，方便算平行加速比）
# ---------------------------------------------------------------------------

def _fetch_hackernews(topic: str) -> tuple[list, float]:
    start = time.perf_counter()
    items: list = []
    try:
        response = requests.get(
            "https://hn.algolia.com/api/v1/search",
            params={"query": topic, "tags": "story", "hitsPerPage": 15},
            timeout=10,
        )
        for hit in response.json().get("hits", []):
            items.append(
                {
                    "source": "hackernews",
                    "title": hit.get("title") or "",
                    "url": hit.get("url")
                    or f"https://news.ycombinator.com/item?id={hit['objectID']}",
                    "score": hit.get("points") or 0,
                    "snippet": (hit.get("story_text") or "")[:1200],
                }
            )
    except Exception as error:
        print(f"  [crawl] HN 來源失敗（先跳過）：{error}")
    return items, time.perf_counter() - start


def _fetch_github(topic: str) -> tuple[list, float]:
    start = time.perf_counter()
    items: list = []
    try:
        response = requests.get(
            "https://api.github.com/search/repositories",
            params={"q": topic, "sort": "stars", "per_page": 15},
            headers={"User-Agent": "agentic-practice/0.1"},
            timeout=10,
        )
        for repo in response.json().get("items", []):
            items.append(
                {
                    "source": "github",
                    "title": repo.get("full_name") or "",
                    "url": repo.get("html_url") or "",
                    "score": repo.get("stargazers_count") or 0,
                    "snippet": (repo.get("description") or "")[:1200],
                }
            )
    except Exception as error:
        print(f"  [crawl] GitHub 來源失敗（先跳過）：{error}")
    return items, time.perf_counter() - start


def _fetch_arxiv(topic: str) -> tuple[list, float]:
    """
    免 key 的第三個來源。刻意選 arXiv 是因為論文摘要通常有 800-1500 字元，
    比 HN/GitHub 的短摘要長很多，這樣壓縮節點的「壓縮前後差異」才有感。
    arXiv 沒有熱度分數，score 固定填 0（不是抓取失敗）。
    """
    start = time.perf_counter()
    items: list = []
    try:
        response = requests.get(
            "https://export.arxiv.org/api/query",
            params={"search_query": f"all:{topic}", "start": 0, "max_results": 10},
            timeout=10,
        )
        root = ET.fromstring(response.text)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("a:entry", ns):
            title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
            summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
            link = (entry.findtext("a:id", default="", namespaces=ns) or "").strip()
            items.append(
                {
                    "source": "arxiv",
                    "title": re.sub(r"\s+", " ", title),
                    "url": link,
                    "score": 0,
                    "snippet": re.sub(r"\s+", " ", summary)[:1200],
                }
            )
    except Exception as error:
        print(f"  [crawl] arXiv 來源失敗（先跳過）：{error}")
    return items, time.perf_counter() - start


SOURCES = [
    ("hackernews", _fetch_hackernews),
    ("github", _fetch_github),
    ("arxiv", _fetch_arxiv),
]


# ---------------------------------------------------------------------------
# 節點
# ---------------------------------------------------------------------------

def crawl(state: PipelineState) -> dict:
    """
    3 個來源預設用 ThreadPoolExecutor 平行抓取（I/O bound，用執行緒就夠，
    不需要把整個 graph 改成 async）。傳 sequential_crawl=True 時改成逐一
    序列呼叫，方便跟平行版本直接比較牆鐘時間。
    """
    topic = state["topic"]
    sequential = state.get("sequential_crawl", False)
    items: list = []
    source_seconds: dict = {}

    wall_start = time.perf_counter()
    if sequential:
        for name, fn in SOURCES:
            fetched, seconds = fn(topic)
            items.extend(fetched)
            source_seconds[name] = round(seconds, 2)
    else:
        with ThreadPoolExecutor(max_workers=len(SOURCES)) as pool:
            future_to_name = {pool.submit(fn, topic): name for name, fn in SOURCES}
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                fetched, seconds = future.result()
                items.extend(fetched)
                source_seconds[name] = round(seconds, 2)
    wall_s = time.perf_counter() - wall_start

    sequential_estimate = round(sum(source_seconds.values()), 2)
    speedup = round(sequential_estimate / wall_s, 2) if wall_s else 1.0
    mode = "序列" if sequential else "平行"

    print(f"  [crawl] 共抓到 {len(items)} 筆（{mode}，{len(SOURCES)} 來源）")
    print(f"    各來源耗時：{source_seconds}")
    print(f"    序列預估 {sequential_estimate}s vs 實際牆鐘 {wall_s:.2f}s（加速 {speedup}x）")

    return {
        "raw_items": items,
        "crawl_parallelism": {
            "mode": mode,
            "source_seconds": source_seconds,
            "sequential_estimate_s": sequential_estimate,
            "wall_s": round(wall_s, 2),
            "speedup_x": speedup,
        },
    }


def dedup_embed(state: PipelineState) -> dict:
    items = state["raw_items"]
    if not items:
        return {"dedup_items": []}
    dedup_items = dedup_by_embedding(items)
    print(
        f"  [dedup_embed] {len(items)} 筆 -> {len(dedup_items)} 筆 "
        f"(embedding 門檻 {DEDUP_SIMILARITY_THRESHOLD})"
    )
    return {"dedup_items": dedup_items}


def authority(state: PipelineState) -> dict:
    items = state["dedup_items"]
    if not items:
        return {"scored_items": []}

    listing = "\n".join(
        f"{i}. [{item['source']}] {item['title']}（社群分數 {item['score']}）"
        for i, item in enumerate(items)
    )
    reply = call_llm(
        model=CHEAP_MODEL,
        system="你是資訊來源評估員，評估每筆資料的權威性與可信度。",
        user=(
            f"針對話題「{state['topic']}」，為以下每筆資料打 1-5 分"
            "（考量：來源類型、社群熱度、標題是否像一手資訊而非農場文；"
            "沒有社群分數的學術論文來源不代表不重要）：\n"
            f"{listing}\n\n"
            '請回傳 JSON 陣列：[{"index": 0, "authority": 4, "reason": "簡短理由"}, ...]。只回傳 JSON。'
        ),
        # 3 個來源合併後常有 30-40+ 筆，預設 2000 太緊繃、容易被截斷，調高留緩衝。
        max_tokens=3000,
    )
    try:
        scores = {entry["index"]: entry for entry in extract_json(reply)}
    except Exception as error:
        # 就算加大 max_tokens 後還是解析失敗，也不要讓整條 pipeline 崩掉——
        # 保底改用社群分數排序，並在 reason 標明這是降級結果，方便事後追查。
        print(f"  [authority] ⚠️ JSON 解析失敗，改用社群分數排序保底：{error}")
        fallback_items = sorted(items, key=lambda item: item.get("score", 0), reverse=True)[:8]
        scored_items = [
            {**item, "authority": 3, "reason": "（JSON 解析失敗，改用社群分數排序保底）"}
            for item in fallback_items
        ]
        print(f"  [authority] {len(items)} 筆 -> 留下 {len(scored_items)} 筆（保底模式）")
        return {"scored_items": scored_items}

    scored_items = []
    for i, item in enumerate(items):
        entry = scores.get(i)
        if entry and entry["authority"] >= 3:
            scored_items.append(
                {**item, "authority": entry["authority"], "reason": entry["reason"]}
            )
    scored_items.sort(key=lambda item: item["authority"], reverse=True)
    scored_items = scored_items[:8]
    print(f"  [authority] {len(items)} 筆 -> 留下 {len(scored_items)} 筆")
    return {"scored_items": scored_items}


def _build_raw_context(items: list) -> str:
    """壓縮節點的「壓縮前」內容：把每筆資料的完整摘要原封不動列出來。"""
    blocks = []
    for item in items:
        header = f"### [{item['source']}] {item['title']}（權威性 {item.get('authority', '?')}/5）"
        if item.get("reason"):
            header += f"\n判斷理由：{item['reason']}"
        blocks.append(f"{header}\n{item.get('snippet') or '（無摘要）'}")
    return "\n\n".join(blocks)


def compress(state: PipelineState) -> dict:
    """
    借鑑 open_deep_research 的 compress_research：authority 篩完的資料還是
    帶著完整摘要（尤其 arXiv 論文摘要很長），這裡用便宜模型壓縮成一段精簡筆記，
    讓下游 synthesize（貴模型）只需要處理濃縮後的內容。
    """
    items = state["scored_items"]
    if not items:
        return {
            "compressed_context": "",
            "compression_stats": {
                "raw_tokens": 0, "compressed_tokens": 0,
                "saved_tokens": 0, "saved_pct": 0.0,
            },
        }

    raw_context = _build_raw_context(items)
    raw_tokens = count_tokens(raw_context)

    compressed = call_llm(
        model=CHEAP_MODEL,
        system=(
            "你負責壓縮研究資料。把多筆原始摘要濃縮成一段精簡的重點筆記，"
            "只保留跟話題直接相關的事實，去掉冗長描述與重複資訊。"
        ),
        user=(
            f"話題：「{state['topic']}」\n\n原始資料：\n\n{raw_context}\n\n"
            "請輸出壓縮後的重點筆記（條列式，繁體中文，保留每筆的來源標記）。"
        ),
        max_tokens=1000,
    )
    compressed_tokens = count_tokens(compressed)
    saved_pct = round((1 - compressed_tokens / raw_tokens) * 100, 1) if raw_tokens else 0.0

    stats = {
        "raw_tokens": raw_tokens,
        "compressed_tokens": compressed_tokens,
        "saved_tokens": raw_tokens - compressed_tokens,
        "saved_pct": saved_pct,
    }
    print(f"  [compress] 原始 {raw_tokens:,} tokens -> 壓縮後 {compressed_tokens:,} tokens（省 {saved_pct}%）")
    return {"compressed_context": compressed, "compression_stats": stats}


def recall_memory(state: PipelineState) -> dict:
    hits = recall_similar_memories(state["topic"])
    if not hits:
        print("  [recall_memory] 沒有找到歷史類似話題")
        return {"memory_hits": [], "memory_context": ""}

    lines = ["以下是系統過去處理過的類似話題與洞見，請在歸納時主動對照、延續或修正："]
    for hit in hits:
        lines.append(
            f"\n### {hit['run_date']}｜{hit['topic']}（相似度 {hit['similarity']:.2f}）\n"
            f"{hit.get('insights', '')}"
        )
    context = "\n".join(lines)
    print(f"  [recall_memory] 命中 {len(hits)} 筆歷史記憶")
    return {"memory_hits": hits, "memory_context": context}


def synthesize(state: PipelineState) -> dict:
    """
    跟 stage1-3 的差異：這裡吃的是 compress 節點吐出的壓縮筆記，
    不是自己重新把 scored_items 攤開組 listing——這才是壓縮真正省到 token 的地方。
    """
    compressed = (state.get("compressed_context") or "").strip()
    if not compressed:
        return {"insights": "（沒有足夠的資料可以形成洞見）"}

    memory_block = state.get("memory_context") or ""
    user = f"話題：「{state['topic']}」\n\n"
    if memory_block:
        user += f"【跨日記憶 RAG】\n{memory_block}\n\n"
    user += (
        f"【今日壓縮後的研究筆記】\n{compressed}\n\n"
        "請歸納出 2-3 個洞見。若有歷史記憶，請明確指出「延續昨日觀點」或「與昨日不同之處」。"
        "用繁體中文、markdown 條列。"
    )
    insights = call_llm(
        model=SMART_MODEL,
        system=f"你是趨勢分析師，擅長跨時間對照趨勢變化。{DATE_GROUNDING}",
        user=user,
    )
    print(f"  [synthesize] 洞見產出 {len(insights)} 字元")
    return {"insights": insights}


def write(state: PipelineState) -> dict:
    sources = "\n".join(f"- {item['title']}：{item['url']}" for item in state["scored_items"])
    feedback = (state.get("editor_feedback") or "").strip()
    if state.get("force_bad_first_draft", False) and state.get("retry_count", 0) == 0:
        system = "你是寫作新手，請刻意寫得很粗糙、鬆散、觀點不明確。"
        user = f"話題：{state['topic']}\n洞見：\n{state['insights']}\n\n"
        user += "請寫一篇不到 200 字、非常粗糙的短文（含標題）。"
    else:
        system = f"你是科技專欄編輯，文風精煉、觀點清晰。{DATE_GROUNDING}"
        user = "根據以下洞見，寫一篇約 500 字的繁體中文短文（含標題）：\n\n"
        user += f"{state['insights']}\n\n"
        if state.get("memory_context"):
            user += "若有引用跨日記憶，請在文中明確提到「相較昨日/過去」的變化。\n\n"
        if feedback:
            user += f"主編退回意見（請務必修正）：\n{feedback}\n\n"
        user += f"文末附上參考來源清單:\n{sources}"
    draft = call_llm(model=SMART_MODEL, system=system, user=user)
    retry_count = int(state.get("retry_count", 0))
    print(f"  [write] 草稿產出 {len(draft)} 字元（第 {retry_count} 次）")
    versions = list(state.get("draft_versions", []))
    versions.append(draft)
    revision_events.append(
        {
            "event": "write",
            "version": len(versions) - 1,
            "retry_count": retry_count,
            "chars": len(draft),
        }
    )
    return {"draft": draft, "draft_versions": versions}


def _record_editor_review(state: PipelineState, decision: str, feedback: str, retry_count: int) -> int:
    draft_version = max(len(state.get("draft_versions", [])) - 1, 0)
    editor_reviews.append(
        {
            "draft_version": draft_version,
            "retry_count": retry_count,
            "decision": decision,
            "feedback": feedback,
        }
    )
    return draft_version


def chief_editor(state: PipelineState) -> Command:
    max_retry = int(state.get("max_retry", 1))
    retry_count = int(state.get("retry_count", 0))

    def _judge_once(prompt: str, max_tokens: int) -> dict:
        reply = call_llm(
            model=SMART_MODEL,
            system=f"你是主編，嚴格審稿，會提出可執行的修改建議。{DATE_GROUNDING}",
            user=prompt,
            max_tokens=max_tokens,
        )
        return extract_json(reply)

    prompt = (
        f"話題：{state['topic']}\n\n"
        "請只回傳合法 JSON：\n"
        '{ "decision": "approve" 或 "revise", "feedback": ["建議1", "建議2"] }\n\n'
        "feedback 最多列 4 條、每條不超過 80 字——意見要具體可執行，但不要長篇大論，"
        "以免超出輸出長度限制。\n\n"
        f"草稿：\n{state['draft']}"
    )

    # 若模型回傳非合法 JSON（markdown 圍欄、截斷、多餘說明），不再呼叫 LLM repair：
    # 先前 repair prompt 內含 approve 範例，模型易照抄而誤放行；改為本地保守退回。
    # max_tokens 600 實測幾乎每次都不夠（feedback 稍微詳細就會截斷），調到 1500
    # 留緩衝；上面同時限制 feedback 條數/長度，兩者一起才不會又把 token 往上推。
    try:
        data = _judge_once(prompt, max_tokens=1500)
    except Exception:
        print("  [chief_editor] ⚠️ JSON 解析失敗，保守退回")
        data = {
            "decision": "revise",
            "feedback": ["主編審核回覆格式錯誤，請依洞見與結構重新檢查草稿"],
        }

    decision = (data.get("decision") or "").strip().lower()
    feedback_raw = data.get("feedback") or []
    if isinstance(feedback_raw, list):
        feedback = "\n".join(f"- {s}" for s in feedback_raw if str(s).strip())
    else:
        feedback = str(feedback_raw).strip()

    if decision == "approve":
        print("  [chief_editor] ✅ 通過")
        draft_version = _record_editor_review(state, "approve", feedback, retry_count)
        revision_events.append(
            {"event": "review", "decision": "approve", "retry_count": retry_count,
             "draft_version": draft_version, "feedback": feedback}
        )
        return Command(update={"editor_feedback": feedback}, goto="visualize")

    if retry_count >= max_retry:
        print("  [chief_editor] ⚠️ 已達 max_retry，停止重寫")
        draft_version = _record_editor_review(state, "revise_max_retry", feedback, retry_count)
        revision_events.append(
            {"event": "review", "decision": "revise_max_retry", "retry_count": retry_count,
             "draft_version": draft_version, "feedback": feedback}
        )
        # 即使沒被核准，定稿的草稿還是拿去做視覺化/短影音——練習用途不必卡在這裡。
        return Command(update={"editor_feedback": feedback}, goto="visualize")

    print(f"  [chief_editor] ❌ 退回重寫（{retry_count + 1}/{max_retry}）")
    draft_version = _record_editor_review(state, "revise", feedback, retry_count)
    revision_events.append(
        {"event": "review", "decision": "revise", "retry_count": retry_count,
         "draft_version": draft_version, "feedback": feedback}
    )
    return Command(
        update={
            "editor_feedback": feedback or "請加強論點、結構與具體例子。",
            "retry_count": retry_count + 1,
        },
        goto="write",
    )


def visualize(state: PipelineState) -> dict:
    """
    把 authority 篩選出的來源畫成長條圖（權威性評分 x 來源），完全不用 LLM——
    跟 dedup_embed 一樣，能用程式解決的事就不花 token。
    圖表刻意做成 9:16 直式，之後 short_video 節點直接拿來當影片背景，不用再處理一次尺寸。
    """
    items = state.get("scored_items") or []
    if not items:
        print("  [visualize] 沒有資料可畫圖")
        return {"chart_path": "", "chart_stats": {}}

    slug = _slugify(state["topic"])
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    chart_path = output_dir / f"chart_{slug}.png"

    matplotlib.rcParams["font.family"] = CJK_FONT
    matplotlib.rcParams["axes.unicode_minus"] = False

    plot_items = list(reversed(items))  # barh 由下往上畫，反過來讓權威性最高的排最上面
    labels = [it["title"][:20] for it in plot_items]
    scores = [it.get("authority", 0) for it in plot_items]
    colors = [SOURCE_COLORS.get(it["source"], "#888888") for it in plot_items]

    fig = plt.figure(figsize=(VIDEO_WIDTH / 100, VIDEO_HEIGHT / 100), dpi=100)
    fig.patch.set_facecolor("#0f172a")

    title_ax = fig.add_axes((0.05, 0.85, 0.9, 0.12))
    title_ax.axis("off")
    title_ax.text(0.5, 0.5, state["topic"], ha="center", va="center",
                  fontsize=46, weight="bold", color="white")

    # 標籤直接畫在 bar 內部（bar 起點右側），不用 y-tick——CJK 字混英文寬度不一，
    # 靠左邊界的 tick label 常常被裁掉，畫在 bar 裡面就不會有這個問題。
    chart_ax = fig.add_axes((0.06, 0.16, 0.88, 0.6))
    bars = chart_ax.barh(range(len(labels)), scores, color=colors, height=0.65)
    for bar, label in zip(bars, labels):
        chart_ax.text(
            0.12, bar.get_y() + bar.get_height() / 2, label,
            ha="left", va="center", fontsize=17, color="white",
        )
    chart_ax.set_xlim(0, 5)
    chart_ax.set_yticks([])
    chart_ax.set_facecolor("#0f172a")
    chart_ax.tick_params(axis="x", colors="white", labelsize=15)
    chart_ax.set_xlabel("權威性評分", fontsize=18, color="white")
    for spine in chart_ax.spines.values():
        spine.set_color("#334155")

    legend_ax = fig.add_axes((0.05, 0.09, 0.9, 0.04))
    legend_ax.axis("off")
    x = 0.0
    for source, color in SOURCE_COLORS.items():
        legend_ax.add_patch(plt.Rectangle((x, 0.3), 0.03, 0.4, color=color, transform=legend_ax.transAxes))
        legend_ax.text(x + 0.04, 0.5, source, transform=legend_ax.transAxes,
                        ha="left", va="center", fontsize=14, color="#cbd5e1")
        x += 0.03 + 0.02 * len(source) + 0.08

    footer_ax = fig.add_axes((0.05, 0.02, 0.9, 0.05))
    footer_ax.axis("off")
    footer_ax.text(0.5, 0.5, "本次採用來源一覽 · agentic pipeline 練習",
                   ha="center", va="center", fontsize=16, color="#94a3b8")

    fig.savefig(chart_path, facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"  [visualize] 圖表已存到 {chart_path}（{len(items)} 筆來源）")
    return {"chart_path": str(chart_path), "chart_stats": {"items_plotted": len(items)}}


def _synthesize_speech(text: str, mp3_path: Path, srt_path: Path) -> None:
    """
    用 edge-tts 把文字轉成語音（mp3）跟字幕（srt）。免費、不用 API key，
    但技術上不是微軟官方支援的用法（條款寫的是給 Edge 瀏覽器朗讀功能用），
    練習用途風險低，正式量產前應改用有官方合約的 TTS 服務。
    """
    communicate = edge_tts.Communicate(text, voice=VOICE)
    submaker = edge_tts.SubMaker()
    with open(mp3_path, "wb") as f:
        for chunk in communicate.stream_sync():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                submaker.feed(chunk)
    srt_path.write_text(submaker.get_srt(), encoding="utf-8")


def _probe_duration_seconds(media_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(media_path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def write_narration(state: PipelineState) -> dict:
    """把定稿文章濃縮成口語旁白稿（Haiku），是後面分鏡/配音/圖片節點的共同輸入。"""
    draft = (state.get("draft") or "").strip()
    if not draft:
        print("  [write_narration] 沒有定稿草稿，跳過")
        return {"video_script": ""}

    script = call_llm(
        model=CHEAP_MODEL,
        system="你是短影音腳本編輯，把文章改寫成口語化的旁白稿，適合語音朗讀。",
        user=(
            "請把以下文章濃縮改寫成一段 150-220 字的繁體中文口語旁白稿，"
            "適合 45-60 秒的短影音朗讀。直接輸出旁白內容本身，第一個字就是旁白的第一句話——"
            "不要任何標題行、不要用「#」開頭、不要條列、不要參考來源、不要前言或說明文字，"
            "開頭第一句話要能吸引注意力。\n\n"
            f"{draft}"
        ),
        max_tokens=500,
    )
    # 防禦性清理：模型偶爾還是會加一行 markdown 標題，沒濾掉會被 TTS 整句唸出來。
    script = re.sub(r"^\s*#.*\n+", "", script).strip()
    print(f"  [write_narration] 旁白稿產出 {len(script)} 字元")
    return {"video_script": script}


def _mechanical_split(script: str) -> list:
    """
    LLM 分場失敗（格式錯誤或跟原文對不齊）時的保底方案：照中文句尾標點切句，
    再貪婪塞進 3-5 個 bucket。純機械式操作，narration_text 加總必定逐字等於原文。
    """
    sentences = [s for s in re.findall(r"[^。！？]*[。！？]|[^。！？]+$", script) if s]
    if not sentences:
        return [{"narration_text": script, "image_prompt": "抽象科技概念視覺化" + IMAGE_STYLE_SUFFIX}]

    scene_count = max(SCENE_COUNT_MIN, min(SCENE_COUNT_MAX, len(sentences)))
    buckets: list[list[str]] = [[] for _ in range(scene_count)]
    for i, sentence in enumerate(sentences):
        buckets[i * scene_count // len(sentences)].append(sentence)

    scenes = []
    for bucket in buckets:
        if not bucket:
            continue
        text = "".join(bucket)
        scenes.append({"narration_text": text, "image_prompt": text[:40] + IMAGE_STYLE_SUFFIX})
    return scenes


def _parse_and_repair_scenes(reply: str, script: str) -> list:
    """narration_text 加總必須逐字等於原始旁白稿——這個不變量後面按字數比例分配時間軸完全靠它。"""
    try:
        scenes = extract_json(reply)
        assert isinstance(scenes, list)
        assert SCENE_COUNT_MIN <= len(scenes) <= SCENE_COUNT_MAX
        joined = "".join(scene["narration_text"] for scene in scenes)
        assert joined.strip() == script.strip()
        for scene in scenes:
            scene["image_prompt"] = scene["image_prompt"].strip() + IMAGE_STYLE_SUFFIX
        return scenes
    except Exception:
        print("  [segment_scenes] ⚠️ narration_text 對不齊原文或格式錯誤，改用機械式切分保底")
        return _mechanical_split(script)


def segment_scenes(state: PipelineState) -> dict:
    """
    把旁白稿拆成 3-5 個場景，每段配一句畫面提示詞，之後每段各自生成一張情境圖+各自的
    Ken Burns 動畫，讓短影音真的「跟著內容變化」，不再是整支影片只有一張圖推近。
    """
    script = (state.get("video_script") or "").strip()
    if not script:
        print("  [segment_scenes] 沒有旁白稿，跳過")
        return {"video_scenes": []}

    reply = call_llm(
        model=CHEAP_MODEL,
        system="你是短影音分鏡師，把一段旁白稿拆成幾個場景，每段配一句畫面提示詞。",
        user=(
            f"請把以下旁白稿拆成 {SCENE_COUNT_MIN}-{SCENE_COUNT_MAX} 段場景，"
            "只能在句子邊界切，不能改寫、增刪、調整任何一個字——"
            "所有片段的 narration_text 依序接起來，必須逐字等於原始旁白稿。"
            "每段另外配一句 image_prompt（30 字內，具體描述這段內容適合的資訊圖表畫面——"
            "可以指定要出現的大數字、關鍵詞、圖示標籤等具體文字內容，也可以是純視覺化的"
            "抽象或具體場景插畫，依內容決定，不要每段都套同一種畫風，讓不同場景之間有"
            "明顯視覺差異；不要提到人物肖像）。\n\n"
            '輸出 JSON 陣列，格式：[{"narration_text": "...", "image_prompt": "..."}, ...]\n\n'
            f"旁白稿：\n{script}"
        ),
        max_tokens=800,
    )
    scenes = _parse_and_repair_scenes(reply, script)
    print(f"  [segment_scenes] 拆成 {len(scenes)} 個場景")
    return {"video_scenes": scenes}


def _assign_scene_times(scenes: list, full_script: str, audio_seconds: float) -> list:
    """
    按每個 scene 的字數佔全文字數比例分配時間——TTS 是逐字唸出全文，字數比例在語速穩定時
    是簡單、確定性、夠準的估算，不需要靠 Whisper 對照原文重建場景邊界（那個更脆弱）。
    """
    total_chars = len(full_script) or 1
    cursor = 0.0
    for scene in scenes:
        share = len(scene["narration_text"]) / total_chars
        scene["start_s"] = round(cursor, 3)
        scene["duration_s"] = round(audio_seconds * share, 3)
        cursor += scene["duration_s"]
        scene["end_s"] = round(cursor, 3)
    if scenes:
        scenes[-1]["end_s"] = round(audio_seconds, 3)
    return scenes


def _resolve_elevenlabs_voice_id(name: str) -> Optional[str]:
    """用搜尋 API 動態解析音色名稱 -> voice_id，不硬編一個猜的 ID。"""
    response = requests.get(
        "https://api.elevenlabs.io/v2/voices",
        params={"search": name},
        headers={"xi-api-key": ELEVENLABS_API_KEY},
        timeout=10,
    )
    response.raise_for_status()
    voices = response.json().get("voices", [])
    for voice in voices:
        if voice.get("name", "").strip().lower() == name.strip().lower():
            return voice["voice_id"]
    return voices[0]["voice_id"] if voices else None


def _elevenlabs_tts_with_timestamps(text: str, voice_id: str) -> tuple:
    response = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps",
        headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"},
        json={"text": text, "model_id": ELEVENLABS_MODEL},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    return base64.b64decode(payload["audio_base64"]), payload["alignment"]


def _alignment_to_srt(alignment: dict, srt_path: Path) -> None:
    """
    ElevenLabs with-timestamps 回傳字元級對時（characters/character_start_times_seconds/
    character_end_times_seconds 三個等長陣列），依中文句尾標點分組成字幕行，
    取代整個 Whisper 轉錄步驟。
    """
    chars = alignment.get("characters", [])
    starts = alignment.get("character_start_times_seconds", [])
    ends = alignment.get("character_end_times_seconds", [])

    def _fmt(t: float) -> str:
        ms = int(round(t * 1000))
        h, ms = divmod(ms, 3_600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines, buf, buf_start = [], "", None
    for char, start, end in zip(chars, starts, ends):
        if buf_start is None:
            buf_start = start
        buf += char
        if char in "。！？":
            lines.append((buf_start, end, buf))
            buf, buf_start = "", None
    if buf.strip():
        lines.append((buf_start or 0.0, ends[-1] if ends else 0.0, buf))

    srt = "\n".join(
        f"{i}\n{_fmt(start)} --> {_fmt(end)}\n{text.strip()}\n"
        for i, (start, end, text) in enumerate(lines, start=1)
    )
    srt_path.write_text(srt, encoding="utf-8")


def synthesize_narration_audio(state: PipelineState) -> dict:
    """
    旁白稿 -> 語音。優先用 ElevenLabs（Anna Su 中文語音，with-timestamps 端點一次拿到
    音檔+字元級對時，不用再多呼叫一次轉錄服務），沒設 key／呼叫失敗就降級回 edge-tts
    （免費、能力較陽春，但確保沒有 ElevenLabs 額度時 pipeline 還是能跑完）。
    """
    script = (state.get("video_script") or "").strip()
    scenes = state.get("video_scenes") or []
    if not script:
        print("  [synthesize_narration_audio] 沒有旁白稿，跳過")
        return {"video_assets": dict(state.get("video_assets") or {})}

    slug = _slugify(state["topic"])
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    audio_path = output_dir / f"video_{slug}.mp3"
    srt_path = output_dir / f"video_{slug}.srt"

    used_elevenlabs = False
    if ELEVENLABS_API_KEY:
        try:
            voice_id = _resolve_elevenlabs_voice_id(ELEVENLABS_VOICE_NAME)
            if not voice_id:
                raise RuntimeError(f"帳號裡找不到語音「{ELEVENLABS_VOICE_NAME}」")
            audio_bytes, alignment = _elevenlabs_tts_with_timestamps(script, voice_id)
            audio_path.write_bytes(audio_bytes)
            _alignment_to_srt(alignment, srt_path)
            cost = len(script) / 1000 * ELEVENLABS_OVERAGE_PER_1K_CHARS
            media_usage_log.append(
                {"node": current_node(), "service": "elevenlabs_tts", "chars": len(script), "cost_usd": cost}
            )
            used_elevenlabs = True
            print(
                f"  [synthesize_narration_audio] ElevenLabs 語音已產出"
                f"（{len(script)} 字元，估計成本 ${cost:.4f}，是否落在免費額度內需對照帳單）"
            )
        except Exception as error:
            print(f"  [synthesize_narration_audio] ⚠️ ElevenLabs 失敗，改用 edge-tts 備援：{error}")

    if not used_elevenlabs:
        _synthesize_speech(script, audio_path, srt_path)
        print("  [synthesize_narration_audio] edge-tts 備援語音已產出")

    audio_seconds = _probe_duration_seconds(audio_path)
    if scenes:
        scenes = _assign_scene_times(scenes, script, audio_seconds)

    return {
        "video_scenes": scenes,
        "video_assets": {
            **(state.get("video_assets") or {}),
            "audio": str(audio_path),
            "srt": str(srt_path),
            "audio_seconds": round(audio_seconds, 1),
            "tts_provider": "elevenlabs" if used_elevenlabs else "edge-tts",
        },
    }


def _cover_crop(src: Path, dest: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(src),
            "-vf",
            f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={VIDEO_WIDTH}:{VIDEO_HEIGHT}",
            str(dest),
        ],
        capture_output=True, text=True, check=True,
    )


def _generate_scene_image(prompt: str, dest: Path) -> dict:
    """成功回傳 usage dict 給成本計算用；呼叫端要能容忍單張圖失敗（丟例外）。"""
    response = openai_client.images.generate(
        model=OPENAI_IMAGE_MODEL, prompt=prompt,
        size=OPENAI_IMAGE_SIZE, quality=OPENAI_IMAGE_QUALITY, n=1,
    )
    raw = dest.with_suffix(".raw.png")
    raw.write_bytes(base64.b64decode(response.data[0].b64_json))
    _cover_crop(raw, dest)
    raw.unlink(missing_ok=True)
    usage = getattr(response, "usage", None)
    in_details = getattr(usage, "input_tokens_details", None) if usage else None
    out_details = getattr(usage, "output_tokens_details", None) if usage else None
    return {
        "text_tokens_in": getattr(in_details, "text_tokens", 0) if in_details else 0,
        "image_tokens_in": getattr(in_details, "image_tokens", 0) if in_details else 0,
        "image_tokens_out": getattr(out_details, "image_tokens", 0) if out_details else 0,
    }


def generate_scene_images(state: PipelineState) -> dict:
    """
    每個 scene 各生成一張情境圖（OpenAI gpt-image-2）+ ffmpeg cover-crop 成 9:16，
    讓不同段落的畫面真的不一樣（不再是整支影片只推近同一張圖）。
    單一 scene 生成失敗只跳過該 scene（image_path=""），不中斷整個節點；
    完全沒有 OPENAI_API_KEY 或全部失敗，就讓所有 scene 的 image_path 留空，
    compose_video 會整段退回舊版單圖 zoompan 安全網。
    """
    scenes = state.get("video_scenes") or []
    if not scenes:
        print("  [generate_scene_images] 沒有場景，跳過")
        return {"video_scenes": scenes}

    if not openai_client:
        print("  [generate_scene_images] ⚠️ 沒有 OPENAI_API_KEY，略過圖片生成（退回單圖背景）")
        return {"video_scenes": scenes}

    slug = _slugify(state["topic"])
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    for i, scene in enumerate(scenes):
        dest = output_dir / f"scene_{slug}_{i}.png"
        try:
            usage = _generate_scene_image(scene["image_prompt"], dest)
            cost = (
                usage["text_tokens_in"] / 1_000_000 * OPENAI_IMAGE_PRICE_PER_1M_TEXT_INPUT
                + usage["image_tokens_in"] / 1_000_000 * OPENAI_IMAGE_PRICE_PER_1M_IMAGE_INPUT
                + usage["image_tokens_out"] / 1_000_000 * OPENAI_IMAGE_PRICE_PER_1M_IMAGE_OUTPUT
            )
            media_usage_log.append(
                {"node": current_node(), "service": "openai_image", "scene": i, "cost_usd": cost}
            )
            scene["image_path"] = str(dest)
            print(f"  [generate_scene_images] scene {i} 圖片已存到 {dest}（${cost:.4f}）")
        except Exception as error:
            scene["image_path"] = ""
            print(f"  [generate_scene_images] ⚠️ scene {i} 生成失敗，這段沒有圖：{error}")

    return {"video_scenes": scenes}


def _render_scene_clip(image_path: str, duration_s: float, clip_path: Path) -> None:
    frame_count = int(duration_s * VIDEO_FPS) + VIDEO_FPS
    vf = (
        f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={VIDEO_WIDTH}:{VIDEO_HEIGHT},"
        f"zoompan=z='min(zoom+0.0012,1.3)':d={frame_count}:s={VIDEO_WIDTH}x{VIDEO_HEIGHT}:fps={VIDEO_FPS}"
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-loop", "1", "-i", image_path, "-vf", vf,
            "-t", f"{duration_s:.3f}", "-pix_fmt", "yuv420p", "-an", str(clip_path),
        ],
        capture_output=True, text=True, check=True,
    )


def _xfade_concat(clip_paths: list, audio_path: Path, audio_seconds: float, video_path: Path) -> None:
    """
    依序把每段場景短片用 xfade 接起來，最後疊上完整旁白音軌。offset 公式（下一段轉場的
    offset = 目前已串接長度 - 轉場秒數）是標準寫法，但跟這個檔案原本 zoompan/-shortest
    的教訓一樣不能盡信公式——每段 clip 都多墊了 XFADE_DURATION 秒，最後仍然用 -t 明確
    鎖定總長度對齊音訊當保險。
    """
    durations = [_probe_duration_seconds(p) for p in clip_paths]
    inputs = []
    for p in clip_paths:
        inputs += ["-i", str(p)]
    inputs += ["-i", str(audio_path)]

    filters = []
    merged_len = durations[0]
    last_label = "0:v"
    for i in range(1, len(clip_paths)):
        offset = merged_len - XFADE_DURATION
        out_label = f"v{i}"
        filters.append(
            f"[{last_label}][{i}:v]xfade=transition=fade:duration={XFADE_DURATION}:"
            f"offset={offset:.3f}[{out_label}]"
        )
        merged_len += durations[i] - XFADE_DURATION
        last_label = out_label

    cmd = ["ffmpeg", "-y", *inputs]
    if filters:
        cmd += ["-filter_complex", ";".join(filters), "-map", f"[{last_label}]"]
    else:
        cmd += ["-map", "0:v"]
    cmd += [
        "-map", f"{len(clip_paths)}:a",
        "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p",
        "-t", f"{audio_seconds:.3f}", "-shortest", str(video_path),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)


def compose_video(state: PipelineState) -> dict:
    """
    每個 scene 各自 render 一段帶 zoompan 的靜音短片，再用 xfade 依序串接，最後疊上完整
    旁白音軌。比起吃 N 張圖的巨大 filter_complex，分段做更容易在出錯時定位問題——
    跟這個檔案原本 debug zoompan/-shortest 的方式一致。

    降級順序（絕不讓整條 pipeline 崩掉）：
    1. 完全沒有可用場景時間資訊 -> 退回舊版單一背景 zoompan（沿用 chart_path 或純色背景）
    2. 部分 scene 沒圖 -> 用 chart_path 或純色圖墊背，時間軸照舊保留
    3. 單一 scene 的短片合成失敗 -> 該 clip 從串接列表拿掉，繼續其餘的
    4. 最終 xfade 合成失敗 -> 保留已產出的語音/字幕/場景圖，video_assets 不含 "video"
    """
    assets = dict(state.get("video_assets") or {})
    audio_path_str = assets.get("audio")
    if not audio_path_str:
        print("  [compose_video] 沒有語音檔，跳過")
        return {"video_assets": assets}

    audio_path = Path(audio_path_str)
    audio_seconds = assets.get("audio_seconds") or _probe_duration_seconds(audio_path)
    scenes = state.get("video_scenes") or []

    slug = _slugify(state["topic"])
    output_dir = Path(__file__).resolve().parent / "outputs"
    video_path = output_dir / f"video_{slug}.mp4"

    if not shutil.which("ffmpeg"):
        print("  [compose_video] ⚠️ 找不到 ffmpeg，只保留語音+字幕")
        return {"video_assets": assets}

    chart_path = state.get("chart_path") or ""
    fallback_bg = chart_path if chart_path and Path(chart_path).exists() else None
    if fallback_bg is None:
        # 沒有圖表（例如 scored_items 是空的）就用純色畫布頂著，影片還是要能生出來。
        fallback_bg = str(output_dir / f"video_bg_{slug}.png")
        fig = plt.figure(figsize=(VIDEO_WIDTH / 100, VIDEO_HEIGHT / 100), dpi=100)
        fig.patch.set_facecolor("#0f172a")
        fig.savefig(fallback_bg, facecolor=fig.get_facecolor())
        plt.close(fig)

    scene_specs = [s for s in scenes if s.get("duration_s", 0) > 0]
    if not scene_specs:
        scene_specs = [{"image_path": "", "duration_s": audio_seconds}]

    clip_paths = []
    for i, scene in enumerate(scene_specs):
        image_path = scene.get("image_path") or fallback_bg
        clip_path = output_dir / f"clip_{slug}_{i}.mp4"
        try:
            _render_scene_clip(image_path, scene["duration_s"] + XFADE_DURATION, clip_path)
            clip_paths.append(clip_path)
        except subprocess.CalledProcessError as error:
            print(f"  [compose_video] ⚠️ scene {i} 短片合成失敗，跳過這段：{error.stderr[-200:]}")

    if not clip_paths:
        print("  [compose_video] ⚠️ 所有場景短片都失敗，只保留語音+字幕")
        return {"video_assets": assets}

    try:
        _xfade_concat(clip_paths, audio_path, audio_seconds, video_path)
    except subprocess.CalledProcessError as error:
        print(f"  [compose_video] ⚠️ xfade 合成失敗，只保留語音+字幕：{error.stderr[-300:]}")
        return {"video_assets": assets}

    print(
        f"  [compose_video] 影片已存到 {video_path}"
        f"（{len(clip_paths)} 段場景，音訊 {audio_seconds:.1f} 秒，字幕另存 {Path(assets['srt']).name}）"
    )
    return {
        "video_assets": {**assets, "video": str(video_path), "scene_count": len(clip_paths)},
    }


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

builder = StateGraph(PipelineState)
builder.add_node("crawl", instrument("crawl", crawl))
builder.add_node("dedup_embed", instrument("dedup_embed", dedup_embed))
builder.add_node("authority", instrument("authority", authority))
builder.add_node("compress", instrument("compress", compress))
builder.add_node("recall_memory", instrument("recall_memory", recall_memory))
builder.add_node("synthesize", instrument("synthesize", synthesize))
builder.add_node("write", instrument("write", write))
builder.add_node("chief_editor", instrument("chief_editor", chief_editor))
builder.add_node("visualize", instrument("visualize", visualize))
builder.add_node("write_narration", instrument("write_narration", write_narration))
builder.add_node("segment_scenes", instrument("segment_scenes", segment_scenes))
builder.add_node(
    "synthesize_narration_audio",
    instrument("synthesize_narration_audio", synthesize_narration_audio),
)
builder.add_node(
    "generate_scene_images", instrument("generate_scene_images", generate_scene_images)
)
builder.add_node("compose_video", instrument("compose_video", compose_video))

builder.add_edge(START, "crawl")
builder.add_edge("crawl", "dedup_embed")
builder.add_edge("dedup_embed", "authority")
builder.add_edge("authority", "compress")
builder.add_edge("compress", "recall_memory")
builder.add_edge("recall_memory", "synthesize")
builder.add_edge("synthesize", "write")
builder.add_edge("write", "chief_editor")
# chief_editor 沒有靜態邊指到 write/visualize——它一律用 Command(goto=...) 動態決定，
# 通過或撞到 max_retry 都會走到 visualize，只有 revise 才會繞回 write。
builder.add_edge("visualize", "write_narration")
builder.add_edge("write_narration", "segment_scenes")
builder.add_edge("segment_scenes", "synthesize_narration_audio")
builder.add_edge("synthesize_narration_audio", "generate_scene_images")
builder.add_edge("generate_scene_images", "compose_video")
builder.add_edge("compose_video", END)

graph = builder.compile()


def rerender_images(topic: str) -> dict:
    """
    只重跑 generate_scene_images → compose_video。
    從上次 run_{slug}_summary.json 讀分鏡、音檔、圖表路徑，不重跑爬蟲/寫稿/配音。
    適合：OpenAI billing limit 調好後，補生場景圖並重合成影片。
    """
    global media_usage_log
    media_usage_log = []
    reset_metrics()

    slug = _slugify(topic)
    output_dir = Path(__file__).resolve().parent / "outputs"
    summary_path = output_dir / f"run_{slug}_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"找不到上次執行報告：{summary_path}\n"
            f"請先完整跑一次：python stage5/graph.py \"{topic}\""
        )

    report = json.loads(summary_path.read_text())
    scenes = report.get("video_scenes") or []
    assets = dict(report.get("video_assets") or {})
    chart_path = report.get("chart_path") or str(output_dir / f"chart_{slug}.png")
    audio_path = assets.get("audio") or str(output_dir / f"video_{slug}.mp3")

    if not scenes:
        raise ValueError(f"{summary_path} 沒有 video_scenes，無法重生圖")
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"找不到語音檔：{audio_path}")

    # 清掉上次失敗留下的空 image_path，讓這次重新生成
    for scene in scenes:
        scene["image_path"] = ""

    print(f"話題：{topic}")
    print(f"模式：只重生圖 + 重合成影片（讀取 {summary_path.name}）")
    print(f"場景數：{len(scenes)}｜語音：{Path(audio_path).name}")
    print()

    state: PipelineState = {
        "topic": topic,
        "video_scenes": scenes,
        "video_assets": {**assets, "audio": audio_path},
        "chart_path": chart_path if Path(chart_path).exists() else "",
        "video_script": report.get("video_script") or "",
    }

    wall_start = time.perf_counter()
    update = generate_scene_images(state)
    state.update(update)
    update = compose_video(state)
    state.update(update)
    total_wall_s = time.perf_counter() - wall_start

    # 把新的 image_path / video 寫回 summary，方便下次再 rerender
    report["video_scenes"] = state.get("video_scenes") or scenes
    report["video_assets"] = state.get("video_assets") or assets
    report["media_usage_log"] = list(media_usage_log)
    report["media_cost_usd_total"] = round(
        sum(entry.get("cost_usd", 0.0) for entry in media_usage_log), 6
    )
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print_stage5_highlights(state)
    print(f"\n已更新：{summary_path}")
    print(f"牆鐘：{total_wall_s:.1f}s｜圖片成本：${report['media_cost_usd_total']:.4f}")
    return state


def run_pipeline(
    topic: str,
    max_retry: int = 1,
    force_bad_first_draft: bool = False,
    sequential_crawl: bool = False,
) -> dict:
    global usage_log, revision_events, editor_reviews, media_usage_log
    usage_log = []
    reset_metrics()
    revision_events = []
    editor_reviews = []
    media_usage_log = []

    wall_start = time.perf_counter()
    initial_state: PipelineState = {
        "topic": topic,
        "sequential_crawl": sequential_crawl,
        "raw_items": [],
        "crawl_parallelism": {},
        "dedup_items": [],
        "scored_items": [],
        "compressed_context": "",
        "compression_stats": {},
        "memory_hits": [],
        "memory_context": "",
        "insights": "",
        "draft": "",
        "draft_versions": [],
        "editor_feedback": "",
        "retry_count": 0,
        "max_retry": max_retry,
        "force_bad_first_draft": force_bad_first_draft,
        "chart_path": "",
        "chart_stats": {},
        "video_script": "",
        "video_scenes": [],
        "video_assets": {},
    }

    result: dict = {"topic": topic}
    for chunk in graph.stream(initial_state, stream_mode="updates"):
        for node_name, update in chunk.items():
            _preview_update(node_name, update)
            result.update(update)

    if result.get("insights"):
        save_memory_entry(topic, result["insights"], result.get("draft", ""))
        print(f"\n已寫入跨日記憶：{MEMORY_FILE}")

    print("\n" + "=" * 72)
    print("歷史記憶命中：\n")
    for hit in result.get("memory_hits", []):
        print(f"- {hit['run_date']} | {hit['topic']} (相似度 {hit['similarity']:.2f})")
    print("\n" + "=" * 72)
    print("洞見：\n")
    print(result.get("insights", ""))
    print("\n" + "=" * 72)
    print("草稿：\n")
    print(result.get("draft", ""))
    print("\n" + "=" * 72)
    print("主編回饋：\n")
    print(result.get("editor_feedback", ""))

    slug = _slugify(topic)
    output_dir = Path(__file__).resolve().parent / "outputs"
    total_wall_s = time.perf_counter() - wall_start

    print_draft_version_summary(result.get("draft_versions") or [])
    artifact_paths = save_run_artifacts(result, slug, total_wall_s, output_dir)

    versions = result.get("draft_versions") or []
    if versions:
        print(f"\n草稿版本數：{len(versions)}（已輸出到 {output_dir}）")
    if editor_reviews:
        print(f"主編審核次數：{len(editor_reviews)}")
        for review in editor_reviews:
            v = review["draft_version"]
            print(f"  - v{v}：{review['decision']} → editor_{slug}_v{v}.md")
    if artifact_paths.get("revision_log"):
        print(f"修訂時間軸：{artifact_paths['revision_log']}")
    if artifact_paths.get("diff"):
        print(f"版本差異檔：{artifact_paths['diff']}")

    print_memory_library()

    if artifact_paths:
        print(f"\n輸出檔案：")
        for key, path in artifact_paths.items():
            if isinstance(path, list):
                for p in path:
                    print(f"  - {p}")
            else:
                print(f"  - {key}: {path}")

    print_stage4_highlights(result)
    print_stage5_highlights(result)
    print_run_summary(total_wall_s)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 5: 視覺化 + 短影音")
    parser.add_argument("topic", nargs="?", default="AI agents", help="話題")
    parser.add_argument("max_retry", nargs="?", type=int, default=1, help="主編最多退回次數")
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="爬蟲改序列執行（跟預設的平行版本比較耗時用）",
    )
    parser.add_argument(
        "--seed-yesterday",
        action="store_true",
        help="種一筆昨天的記憶（驗收跨日引用，不必真的等一天）",
    )
    parser.add_argument(
        "--bad-first",
        action="store_true",
        help="第一次草稿刻意寫差（測試主編退回）",
    )
    parser.add_argument(
        "--show-memory",
        action="store_true",
        help="只查看跨日記憶庫內容，不跑 pipeline",
    )
    parser.add_argument(
        "--rerender-images",
        action="store_true",
        help="只重生場景圖並重合成影片（讀取上次 run_*_summary.json，不重跑爬蟲/寫稿/配音）",
    )
    args = parser.parse_args()

    if args.show_memory:
        print_memory_library(max_chars=20000)
    elif args.seed_yesterday:
        seed_yesterday_memory()
    elif args.rerender_images:
        rerender_images(args.topic)
    else:
        print(f"話題：{args.topic}")
        print(f"max_retry：{args.max_retry}")
        print(f"爬蟲模式：{'序列' if args.sequential else '平行'}")
        print(f"記憶庫：{MEMORY_FILE}")
        print()
        run_pipeline(
            args.topic,
            args.max_retry,
            force_bad_first_draft=args.bad_first,
            sequential_crawl=args.sequential,
        )
