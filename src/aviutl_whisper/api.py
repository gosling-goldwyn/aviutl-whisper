"""Python↔JS ブリッジAPI - pywebviewのJSから呼び出されるAPI"""

import json
import logging
import math
import os
import re
import tempfile
import threading
from pathlib import Path

import webview

from . import (
    audio,
    diarizer,
    elevenlabs_tts,
    exporter,
    models,
    settings,
    transcriber,
)

PROJECT_VERSION = 1

logger = logging.getLogger(__name__)

_SPEAKER_PREFIX_RE = re.compile(
    r"^\[\s*speaker\s*(?!0+\s*\])(\d+)\s*\]\s*(?:[:：]\s*)?",
    re.IGNORECASE,
)


def segments_from_text(
    text: str,
    ms_per_char: int = 100,
) -> list[transcriber.TranscriptionSegment]:
    """改行区切りテキストを話者prefix対応の連続セグメントへ変換する。"""
    if isinstance(ms_per_char, bool) or not isinstance(ms_per_char, (int, float)):
        raise ValueError("1文字あたりの時間は正の整数で指定してください")
    if (
        not math.isfinite(ms_per_char)
        or ms_per_char <= 0
        or not float(ms_per_char).is_integer()
    ):
        raise ValueError("1文字あたりの時間は正の整数で指定してください")

    char_ms = int(ms_per_char)
    current_ms = 0
    current_speaker = "Speaker 1"
    segments = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        prefix_match = _SPEAKER_PREFIX_RE.match(line)
        if prefix_match:
            current_speaker = f"Speaker {int(prefix_match.group(1))}"
            line = line[prefix_match.end():].strip()
        if not line:
            continue

        end_ms = current_ms + len(line) * char_ms
        segments.append(
            transcriber.TranscriptionSegment(
                start=current_ms / 1000.0,
                end=end_ms / 1000.0,
                text=line,
                speaker=current_speaker,
            )
        )
        current_ms = end_ms
    if not segments:
        raise ValueError("有効なテキスト行がありません")
    return segments


def _get_system_fonts() -> list[str]:
    """Windowsのシステムフォント名一覧を取得する。
    
    HKEY_LOCAL_MACHINE (全ユーザー共通) と
    HKEY_CURRENT_USER (ユーザー個別インストール) の両方を参照する。
    """
    fonts = set()
    font_key_path = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts"
    try:
        import winreg
        for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            try:
                key = winreg.OpenKey(hive, font_key_path)
                i = 0
                while True:
                    try:
                        name, _, _ = winreg.EnumValue(key, i)
                        # "MS Gothic (TrueType)" → "MS Gothic"
                        font_name = name.split(" (")[0].strip()
                        if font_name:
                            fonts.add(font_name)
                        i += 1
                    except OSError:
                        break
                winreg.CloseKey(key)
            except OSError:
                pass
    except Exception:
        pass

    if not fonts:
        fonts = {
            "MS UI Gothic", "MS Gothic", "MS Mincho",
            "Yu Gothic UI", "Yu Gothic", "Yu Mincho",
            "Meiryo", "Meiryo UI", "BIZ UDGothic",
            "Arial", "Segoe UI", "Consolas",
        }

    return sorted(fonts)


def _resolve_font_path(font_name: str) -> str | None:
    """フォント名からフォントファイルパスを解決する。

    Windowsレジストリのフォント登録情報を参照し、フォント名に一致する
    ファイルパスを返す。見つからない場合は None。
    """
    fonts_dir = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    font_key_path = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts"

    try:
        import winreg
    except ImportError:
        return None

    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        try:
            key = winreg.OpenKey(hive, font_key_path)
            i = 0
            while True:
                try:
                    reg_name, reg_value, _ = winreg.EnumValue(key, i)
                    # "MS UI Gothic & MS UI PGothic & MS UI Gothic (TrueType)"
                    # → 括弧前を取得し、& で分割して各名前を確認
                    base_name = reg_name.split(" (")[0].strip()
                    names = [n.strip() for n in base_name.split("&")]
                    if font_name in names:
                        # reg_value がフルパスか、ファイル名のみか
                        if os.path.isabs(reg_value):
                            path = reg_value
                        else:
                            path = os.path.join(fonts_dir, reg_value)
                        if os.path.exists(path):
                            winreg.CloseKey(key)
                            return path
                    i += 1
                except OSError:
                    break
            winreg.CloseKey(key)
        except OSError:
            pass

    return None


def _render_subtitle_image(
    text: str,
    font_name: str = "MS UI Gothic",
    font_size: int = 34,
    bold: bool = False,
    italic: bool = False,
    text_color: str = "ffffff",
    edge_color: str = "000000",
    soft_edge: bool = True,
    align: int = 4,
    pos_x: float = 0,
    pos_y: float = 0,
    spacing_y: int = 0,
    max_chars_per_line: int = 0,
    width: int = 1920,
    height: int = 1080,
) -> bytes:
    """字幕テキストを透過PNGとしてレンダリングする。

    Returns:
        PNG画像のバイト列
    """
    from io import BytesIO

    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # テキスト折り返し
    if max_chars_per_line > 0:
        wrapped_lines = []
        for line in text.split("\n"):
            if not line:
                wrapped_lines.append("")
                continue
            for i in range(0, len(line), max_chars_per_line):
                wrapped_lines.append(line[i:i + max_chars_per_line])
        lines = wrapped_lines
    else:
        lines = text.split("\n")

    # フォント解決
    font_path = _resolve_font_path(font_name)
    if not font_path:
        # フォールバック: msgothic.ttc
        fallback = os.path.join(
            os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "msgothic.ttc"
        )
        if os.path.exists(fallback):
            font_path = fallback

    try:
        font = ImageFont.truetype(font_path or "", font_size)
    except (OSError, AttributeError):
        font = ImageFont.load_default()

    # 色パース
    def parse_hex(h: str) -> tuple[int, int, int, int]:
        h = h.lstrip("#")
        if len(h) != 6:
            h = "ffffff"
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255)

    fill_color = parse_hex(text_color)
    stroke_color = parse_hex(edge_color)
    stroke_width = max(1, font_size // 12) if soft_edge else 0

    line_height = font_size + spacing_y
    total_text_height = len(lines) * line_height

    # align: 3x3グリッド (0-8)
    col_align = align % 3   # 0=左, 1=中, 2=右
    row_align = align // 3  # 0=上, 1=中, 2=下

    # AviUtl座標系: 画面中央が(0,0)。pos_x/pos_yはアンカーポイント。
    anchor_x = width / 2 + pos_x
    anchor_y = height / 2 + pos_y

    # Y基準位置（アライメントでアンカーからの配置方向を決定）
    if row_align == 0:
        base_y = int(anchor_y)
    elif row_align == 1:
        base_y = int(anchor_y - total_text_height / 2)
    else:
        base_y = int(anchor_y - total_text_height)

    for i, line in enumerate(lines):
        if not line:
            continue
        y = base_y + i * line_height

        # X位置計算
        bbox = font.getbbox(line)
        text_w = bbox[2] - bbox[0]

        if col_align == 0:
            x = int(anchor_x)
        elif col_align == 1:
            x = int(anchor_x - text_w / 2)
        else:
            x = int(anchor_x - text_w)

        draw.text(
            (x, y), line, font=font, fill=fill_color,
            stroke_width=stroke_width, stroke_fill=stroke_color,
        )

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _render_preview_frame(
    text: str,
    speaker_index: int,
    settings: dict,
    num_speakers: int = 2,
    width: int = 1920,
    height: int = 1080,
) -> bytes:
    """背景+立ち絵+字幕を合成した1枚のJPEG画像を返す。

    Args:
        text: 字幕テキスト
        speaker_index: アクティブ話者のインデックス (0-based)
        settings: exo設定辞書 (collectExoSettings()相当)
        num_speakers: 話者数
        width, height: 出力画像サイズ
    Returns:
        JPEG画像のバイト列
    """
    from io import BytesIO

    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 255))

    # --- Layer 1: 背景画像 ---
    bg_path = settings.get("background_image", "")
    if bg_path and os.path.exists(bg_path):
        try:
            bg_img = Image.open(bg_path).convert("RGBA")
            # cover: アスペクト比を維持しつつフルカバー
            scale = max(width / bg_img.width, height / bg_img.height)
            new_w = int(bg_img.width * scale)
            new_h = int(bg_img.height * scale)
            bg_img = bg_img.resize((new_w, new_h), Image.LANCZOS)
            dx = (width - new_w) // 2
            dy = (height - new_h) // 2
            canvas.paste(bg_img, (dx, dy))
        except Exception:
            pass

    # --- Layer 2: 立ち絵 ---
    speaker_images = settings.get("speaker_images", [])
    for i in range(num_speakers):
        if i >= len(speaker_images):
            continue
        si = speaker_images[i]
        file_path = si.get("file", "")
        if not file_path or not os.path.exists(file_path):
            continue

        try:
            tachie_img = Image.open(file_path).convert("RGBA")
            scale_pct = si.get("scale", 100) / 100.0
            new_w = int(tachie_img.width * scale_pct)
            new_h = int(tachie_img.height * scale_pct)
            if new_w > 0 and new_h > 0:
                tachie_img = tachie_img.resize((new_w, new_h), Image.LANCZOS)

            # AviUtl座標系: 画面中央が(0,0)
            offset_x = si.get("x", 0)
            offset_y = si.get("y", 0)
            dx = int((width / 2) + offset_x - new_w / 2)
            dy = int((height / 2) + offset_y - new_h / 2)

            is_active = (i == speaker_index)
            if not is_active:
                # グレースケール変換
                tachie_img = tachie_img.convert("LA").convert("RGBA")

            # アルファ合成
            temp = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            temp.paste(tachie_img, (dx, dy))
            canvas = Image.alpha_composite(canvas, temp)
        except Exception:
            pass

    # --- Layer 3: 字幕テキスト ---
    colors = settings.get("speaker_colors", [])
    if not colors:
        colors = ["ffffff", "00ffff", "00ff00", "ff00ff",
                  "ffff00", "ff8000", "8080ff", "80ff80"]
    edge_colors = settings.get("speaker_edge_colors", [])
    if not edge_colors:
        edge_colors = ["000000"]
    text_color = colors[speaker_index % len(colors)]
    edge_color = edge_colors[speaker_index % len(edge_colors)]

    subtitle_png = _render_subtitle_image(
        text=text,
        font_name=settings.get("font", "MS UI Gothic"),
        font_size=settings.get("font_size", 34),
        bold=settings.get("bold", False),
        italic=settings.get("italic", False),
        text_color=text_color,
        edge_color=edge_color,
        soft_edge=settings.get("soft_edge", True),
        align=settings.get("align", 4),
        pos_x=settings.get("pos_x", 0),
        pos_y=settings.get("pos_y", 0),
        spacing_y=settings.get("spacing_y", 0),
        max_chars_per_line=settings.get("max_chars_per_line", 0),
        width=width,
        height=height,
    )
    subtitle_layer = Image.open(BytesIO(subtitle_png))
    canvas = Image.alpha_composite(canvas, subtitle_layer)

    # JPEG出力
    rgb = canvas.convert("RGB")
    buf = BytesIO()
    rgb.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _apply_speaker_mapping(
    segments: list[transcriber.TranscriptionSegment],
    mapping: dict[str, int] | None,
) -> list[transcriber.TranscriptionSegment]:
    """話者マッピングを適用してセグメントの話者ラベルを入れ替える。

    mapping: {"Speaker 1": 0, "Speaker 2": 1} のような辞書。
    スロット番号は0始まり。スロット番号をもとに "Speaker N" ラベルに変換する。
    mapping が None またはデフォルト(各話者がそのままのスロット)の場合は元のセグメントを返す。
    """
    if not mapping:
        return segments

    # 元の話者名リスト (ソート済み)
    original_speakers = sorted(set(s.speaker or "Speaker 1" for s in segments))

    # デフォルトマッピングか確認 (変更なしならスキップ)
    is_default = all(
        mapping.get(spk, i) == i for i, spk in enumerate(original_speakers)
    )
    if is_default:
        return segments

    # スロット番号 → 新しい話者名
    slot_to_name = {i: spk for i, spk in enumerate(original_speakers)}
    # 元の話者名 → 新しい話者名 (スロットが指す話者名)
    name_remap = {}
    for spk in original_speakers:
        slot = mapping.get(spk)
        if slot is not None and slot in slot_to_name:
            name_remap[spk] = slot_to_name[slot]
        else:
            name_remap[spk] = spk

    return [
        transcriber.TranscriptionSegment(
            start=seg.start,
            end=seg.end,
            text=seg.text,
            speaker=name_remap.get(seg.speaker or "Speaker 1", seg.speaker),
        )
        for seg in segments
    ]


class Api:
    """pywebview JS APIクラス。

    フロントエンドのJSから `pywebview.api.*` で呼び出される。
    """

    def __init__(self, window: webview.Window | None = None):
        self.window = window
        self._progress = {"progress": 0.0, "message": ""}
        self._cancelled = False
        self._last_result: transcriber.TranscriptionResult | None = None
        self._last_segments: list[transcriber.TranscriptionSegment] | None = None
        self._last_wav_path: str | None = None
        self._audio_mode: str = "none"
        self._silent_wav_duration_ms: int = 0
        self._last_output_format: str = "exo"
        self._speaker_mapping: dict[str, int] | None = None
        self._exo_settings: exporter.ExoSettings | None = None
        self._system_fonts: list[str] | None = None
        self._is_dirty: bool = False
        self._current_project_path: str | None = None
        self._skip_close_dialog: bool = False
        self._tts_progress = {"progress": 0.0, "message": ""}
        self._tts_cancelled = False

    def set_window(self, window: webview.Window):
        self.window = window

    # --- プロジェクト状態管理 ---

    def mark_dirty(self):
        """プロジェクトに未保存の変更があることをマークする。"""
        self._is_dirty = True
        self._update_window_title()

    def is_project_dirty(self) -> bool:
        """プロジェクトに未保存の変更があるか返す。"""
        return self._is_dirty

    def force_close(self):
        """確認ダイアログをスキップしてウィンドウを閉じる。"""
        self._skip_close_dialog = True
        if self.window:
            self.window.destroy()

    def _update_window_title(self):
        """現在のプロジェクト状態に基づいてウィンドウタイトルを更新する。"""
        if not self.window:
            return
        base = "aviutl-whisper"
        if self._current_project_path:
            project_name = Path(self._current_project_path).stem
            title = f"{base} - {project_name}"
        else:
            title = base
        if self._is_dirty:
            title = f"*{title}"
        self.window.set_title(title)

    def select_file(self):
        """ファイル選択ダイアログを開く。"""
        file_types = (
            "音声ファイル (*.m4a;*.mp3;*.wav;*.flac;*.ogg;*.aac;*.wma)",
            "すべてのファイル (*.*)",
        )
        result = self.window.create_file_dialog(
            webview.FileDialog.OPEN,
            file_types=file_types,
        )
        if result and len(result) > 0:
            path = result[0]
            stat = os.stat(path)
            return {
                "path": path,
                "name": Path(path).name,
                "extension": Path(path).suffix,
                "size": stat.st_size,
            }
        return None

    def transcribe(self, file_path: str, settings_dict: dict):
        """文字起こし＋話者分離を実行する。"""
        self._cancelled = False
        self._speaker_mapping = None
        # 前回のWAVファイルを削除
        self._cleanup_wav()
        # exo設定を保存
        if settings_dict.get("exo_settings"):
            self._exo_settings = exporter.ExoSettings.from_dict(settings_dict["exo_settings"])
        try:
            return self._run_transcription(file_path, settings_dict)
        except Exception as e:
            logger.exception("文字起こしエラー")
            return {"success": False, "error": str(e)}

    def import_text(self, text: str, ms_per_char: int, settings_dict: dict):
        """改行区切りテキストを文字起こし結果として取り込む。"""
        try:
            segments = segments_from_text(text, ms_per_char)
            exo_settings = self._exo_settings
            if settings_dict.get("exo_settings"):
                exo_settings = exporter.ExoSettings.from_dict(
                    settings_dict["exo_settings"]
                )
            text_output = exporter.export_exo(
                segments,
                settings=exo_settings,
            )
            duration_ms = math.ceil(segments[-1].end * 1000)
            silent_wav_path = audio.create_silent_wav(duration_ms)

            self._cleanup_wav()
            self._speaker_mapping = None
            self._last_output_format = "exo"
            self._exo_settings = exo_settings
            result = transcriber.TranscriptionResult(
                segments=list(segments),
                language="text",
                language_probability=1.0,
            )
            self._last_result = result
            self._last_segments = segments
            self._audio_mode = "silence"
            self._last_wav_path = silent_wav_path
            self._silent_wav_duration_ms = duration_ms

            speaker_info = self._build_speaker_info(segments)
            return {
                "success": True,
                "text": text_output,
                "num_segments": len(segments),
                "num_speakers": len(speaker_info),
                "language": result.language,
                "input_type": "text",
                "speakers": speaker_info,
            }
        except Exception as e:
            logger.exception("テキスト取り込みエラー")
            return {"success": False, "error": str(e)}

    def _run_transcription(self, file_path: str, settings_dict: dict):
        model_size = settings_dict.get("model_size", "medium")
        language = settings_dict.get("language")
        num_speakers = settings_dict.get("num_speakers")
        output_format = settings_dict.get("output_format", "text")
        diarization_method = settings_dict.get("diarization_method", "speechbrain")
        hf_token = settings_dict.get("hf_token", "")
        self._last_output_format = output_format

        # 1. 音声変換
        self._update_progress(0.05, "音声ファイルを変換中...")
        wav_path = audio.convert_to_wav(file_path)

        if self._cancelled:
            return {"success": False, "error": "キャンセルされました"}

        # 2. Whisperモデル読み込み
        self._update_progress(0.1, f"Whisperモデル({model_size})を読み込み中...")
        whisper_model = models.load_whisper_model(
            model_size=model_size,
            progress_callback=self._progress_callback,
        )

        if self._cancelled:
            return {"success": False, "error": "キャンセルされました"}

        # 3. 文字起こし
        self._update_progress(0.2, "文字起こし中...")
        result = transcriber.transcribe(
            model=whisper_model,
            audio_path=wav_path,
            language=language,
            progress_callback=self._progress_callback,
        )

        # WhisperモデルをGPUから解放 (話者分離のためにVRAMを確保)
        del whisper_model
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        if self._cancelled:
            return {"success": False, "error": "キャンセルされました"}

        # 4-5. 話者分離 (方式に応じて分岐)
        if diarization_method == "pyannote":
            segments = self._run_pyannote_diarization(
                wav_path, result.segments, num_speakers, hf_token,
            )
        else:
            segments = self._run_speechbrain_diarization(
                wav_path, result.segments, num_speakers,
            )

        # 6. 結果フォーマット
        self._update_progress(0.95, "結果を生成中...")
        _, _, export_fn = exporter.EXPORTERS[output_format]
        if output_format == "exo":
            text = exporter.export_exo(segments, settings=self._exo_settings)
        else:
            text = export_fn(segments)

        self._last_result = result
        self._last_segments = segments
        self._last_wav_path = wav_path  # 話者サンプル再生用に保持
        self._audio_mode = "source"
        self._silent_wav_duration_ms = 0

        # 話者情報を構築
        speaker_info = self._build_speaker_info(segments)
        self._update_progress(1.0, "完了！")

        return {
            "success": True,
            "text": text,
            "num_segments": len(segments),
            "num_speakers": len(speaker_info),
            "language": result.language,
            "speakers": speaker_info,
        }

    def _run_speechbrain_diarization(self, wav_path, segments, num_speakers):
        """speechbrainによる話者分離。"""
        self._update_progress(0.6, "話者分離モデル(speechbrain)を読み込み中...")
        speaker_model = models.load_speechbrain_model(
            progress_callback=self._progress_callback,
        )
        self._update_progress(0.7, "話者分離中...")
        return diarizer.assign_speakers(
            model=speaker_model,
            audio_path=wav_path,
            segments=segments,
            num_speakers=num_speakers,
            progress_callback=self._progress_callback,
        )

    def _run_pyannote_diarization(self, wav_path, segments, num_speakers, hf_token):
        """pyannote.audioによる話者分離。"""
        # 保存済みトークンからのフォールバック
        if not hf_token:
            saved = settings.load_settings()
            encrypted = saved.get("hf_token_encrypted", "")
            if encrypted:
                hf_token = settings.decrypt_token(encrypted)
        if not hf_token:
            raise ValueError(
                "pyannoteを使用するにはHuggingFaceトークンが必要です。\n"
                "設定画面でトークンを入力してください。"
            )

        self._update_progress(0.6, "話者分離モデル(pyannote)を読み込み中...")
        try:
            pipeline = models.load_pyannote_pipeline(
                hf_token=hf_token,
                progress_callback=self._progress_callback,
            )
        except ImportError as e:
            raise ImportError(str(e))

        self._update_progress(0.7, "pyannote話者分離中...")
        result = diarizer.assign_speakers_pyannote(
            pipeline=pipeline,
            audio_path=wav_path,
            segments=segments,
            num_speakers=num_speakers,
            progress_callback=self._progress_callback,
        )

        # パイプラインをGPUから解放
        del pipeline
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        return result

    def save_result(self, format_type: str, exo_settings: dict | None = None,
                    speaker_mapping: dict | None = None):
        """結果をファイルに保存する。"""
        if not self._last_segments:
            return {"success": False, "error": "保存する結果がありません"}

        # exo設定を更新（保存時に最新の設定を反映）
        if exo_settings:
            self._exo_settings = exporter.ExoSettings.from_dict(exo_settings)

        # フロントエンドから渡された値を優先し、省略時はAPI内部の状態を使う。
        # プロジェクト再読込後など、UIがまだマッピングを再送していない場合も
        # 入れ替え状態を失わずにexoへ反映する。
        if speaker_mapping is not None:
            self._speaker_mapping = speaker_mapping
        segments = _apply_speaker_mapping(
            self._last_segments, self._speaker_mapping
        )

        label, ext, _ = exporter.EXPORTERS.get(format_type, ("テキスト", ".txt", None))
        file_types = (f"{label}ファイル (*{ext})",)

        result = self.window.create_file_dialog(
            webview.FileDialog.SAVE,
            file_types=file_types,
        )

        if result:
            path = result if isinstance(result, str) else result[0]
            saved_path = exporter.export_to_file(
                segments, path, format_type,
                exo_settings=self._exo_settings if format_type == "exo" else None,
            )
            return {"success": True, "path": saved_path}
        return None

    def get_progress(self):
        """現在の進捗を返す。"""
        return dict(self._progress)

    def cancel(self):
        """処理をキャンセルする。"""
        self._cancelled = True

    def get_device_info(self):
        """実行デバイス情報を返す。"""
        try:
            import ctranslate2
            cuda_types = ctranslate2.get_supported_compute_types("cuda")
            if cuda_types:
                return {"device": "GPU (CUDA)", "detail": "ctranslate2 CUDA対応"}
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                return {"device": "GPU (CUDA)", "detail": name}
        except ImportError:
            pass
        return {"device": "CPU", "detail": "GPUは利用できません"}

    def get_system_fonts(self):
        """システムフォント一覧を返す (キャッシュ付き)。"""
        if self._system_fonts is None:
            self._system_fonts = _get_system_fonts()
        return self._system_fonts

    def get_exo_align_options(self):
        """exoの寄せ方向オプション一覧を返す。"""
        return [{"label": k, "value": v} for k, v in exporter.EXO_ALIGN.items()]

    def get_exo_defaults(self):
        """exo設定のデフォルト値を返す。"""
        defaults = exporter.ExoSettings()
        return {
            "font": defaults.font,
            "font_size": defaults.font_size,
            "spacing_x": defaults.spacing_x,
            "spacing_y": defaults.spacing_y,
            "display_speed": defaults.display_speed,
            "align": defaults.align,
            "bold": defaults.bold,
            "italic": defaults.italic,
            "soft_edge": defaults.soft_edge,
            "pos_x": defaults.pos_x,
            "pos_y": defaults.pos_y,
            "speaker_colors": list(exporter.DEFAULT_SPEAKER_COLORS),
            "default_edge_color": exporter.DEFAULT_EDGE_COLOR,
            "speaker_images": [],
            "background_image": "",
            "max_chars_per_line": 20,
        }

    def select_image_file(self):
        """画像ファイル選択ダイアログを開く。"""
        file_types = (
            "画像ファイル (*.png;*.jpg;*.jpeg;*.bmp;*.gif)",
            "すべてのファイル (*.*)",
        )
        result = self.window.create_file_dialog(
            webview.FileDialog.OPEN,
            file_types=file_types,
        )
        if result and len(result) > 0:
            return result[0]
        return None

    def remap_speakers(self, mapping: dict, format_type: str | None = None,
                       exo_settings: dict | None = None):
        """話者マッピングを変更して出力を再生成する。

        Args:
            mapping: {"Speaker 1": 0, "Speaker 2": 1} — 設定スロット番号
            format_type: 出力形式 (省略時は前回と同じ)
            exo_settings: exo設定 (省略時は前回と同じ)
        """
        if not self._last_segments:
            return {"success": False, "error": "結果がありません"}

        self._speaker_mapping = mapping
        fmt = format_type or self._last_output_format
        if exo_settings:
            self._exo_settings = exporter.ExoSettings.from_dict(exo_settings)

        segments = _apply_speaker_mapping(self._last_segments, mapping)
        _, _, export_fn = exporter.EXPORTERS[fmt]
        if fmt == "exo":
            text = exporter.export_exo(segments, settings=self._exo_settings)
        else:
            text = export_fn(segments)

        return {"success": True, "text": text}

    def play_speaker_sample(self, speaker_name: str):
        """話者の最初の発声区間を再生する。"""
        if not self._last_segments or not self._last_wav_path:
            return {"success": False, "error": "再生データがありません"}

        if not os.path.exists(self._last_wav_path):
            return {"success": False, "error": "音声ファイルが見つかりません"}

        # 該当話者の最初のセグメントを取得
        seg = next(
            (s for s in self._last_segments if s.speaker == speaker_name),
            None,
        )
        if seg is None:
            return {"success": False, "error": f"話者 {speaker_name} が見つかりません"}

        try:
            import sounddevice as sd
            import soundfile as sf

            data, sr = sf.read(self._last_wav_path)
            start_sample = int(seg.start * sr)
            end_sample = min(int(seg.end * sr), len(data))
            segment_data = data[start_sample:end_sample]
            # 非同期再生（前の再生を停止してから）
            sd.stop()
            sd.play(segment_data, sr)
            return {"success": True}
        except Exception as e:
            logger.exception("音声再生エラー")
            return {"success": False, "error": str(e)}

    def play_segment_audio(self, index: int):
        """指定インデックスのセグメント区間の音声を再生する。"""
        if not self._last_segments or not self._last_wav_path:
            return {"success": False, "error": "再生データがありません"}

        if not os.path.exists(self._last_wav_path):
            return {"success": False, "error": "音声ファイルが見つかりません"}

        if index < 0 or index >= len(self._last_segments):
            return {"success": False, "error": "無効なインデックス"}

        seg = self._last_segments[index]

        try:
            import sounddevice as sd
            import soundfile as sf

            data, sr = sf.read(self._last_wav_path)
            start_sample = int(seg.start * sr)
            end_sample = min(int(seg.end * sr), len(data))
            segment_data = data[start_sample:end_sample]
            sd.stop()
            sd.play(segment_data, sr)
            return {"success": True}
        except Exception as e:
            logger.exception("セグメント音声再生エラー")
            return {"success": False, "error": str(e)}

    def stop_audio(self):
        """音声再生を停止する。"""
        try:
            import sounddevice as sd
            sd.stop()
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _build_speaker_info(
        self, segments: list[transcriber.TranscriptionSegment],
    ) -> list[dict]:
        """検出された話者の情報リストを構築する。"""
        speakers_seen: dict[str, dict] = {}
        for seg in segments:
            name = seg.speaker or "Speaker 1"
            if name not in speakers_seen:
                speakers_seen[name] = {
                    "name": name,
                    "sample_text": seg.text[:80],
                    "segment_count": 0,
                    "first_start": seg.start,
                    "first_end": seg.end,
                }
            speakers_seen[name]["segment_count"] += 1
        return sorted(speakers_seen.values(), key=lambda x: x["name"])

    def _cleanup_wav(self):
        """保持しているWAVファイルを削除する。"""
        if self._last_wav_path:
            try:
                os.unlink(self._last_wav_path)
            except OSError:
                pass
            self._last_wav_path = None
        self._silent_wav_duration_ms = 0

    def _sync_silent_wav(self):
        """テキスト入力のタイムライン長に合わせて無音WAVを同期する。"""
        if self._audio_mode != "silence" or not self._last_segments:
            return
        duration_ms = max(1, math.ceil(
            max(segment.end for segment in self._last_segments) * 1000
        ))
        if (
            self._last_wav_path
            and os.path.exists(self._last_wav_path)
            and self._silent_wav_duration_ms == duration_ms
        ):
            return
        self._cleanup_wav()
        self._last_wav_path = audio.create_silent_wav(duration_ms)
        self._silent_wav_duration_ms = duration_ms

    def load_settings(self):
        """保存された設定を読み込む。HFトークンは復号して返す。"""
        data = settings.load_settings()
        # 暗号化されたトークンを復号してフロントエンドに渡す
        encrypted = data.get("hf_token_encrypted", "")
        if encrypted:
            data["hf_token_decrypted"] = settings.decrypt_token(encrypted)
        tts_encrypted = data.get("elevenlabs_api_key_encrypted", "")
        if tts_encrypted:
            data["elevenlabs_api_key_decrypted"] = settings.decrypt_token(
                tts_encrypted
            )
        return data

    def save_settings(self, data: dict):
        """設定を保存する。HFトークンは暗号化して保存。"""
        existing = settings.load_settings()
        # フロントエンドから渡されたhf_tokenを暗号化
        hf_token = data.pop("hf_token", "")
        if hf_token:
            data["hf_token_encrypted"] = settings.encrypt_token(hf_token)
        elif "hf_token_encrypted" not in data:
            data["hf_token_encrypted"] = ""
        # 通常設定の自動保存で、独立したTTS設定を消さない。
        for key in (
            "elevenlabs_api_key_encrypted",
            "elevenlabs_model_id",
            "elevenlabs_output_format",
            "elevenlabs_speaker_voice_ids",
            "elevenlabs_output_dir",
            "elevenlabs_voice_settings",
            "elevenlabs_apply_language_text_normalization",
            "elevenlabs_apply_text_normalization",
            "elevenlabs_generation_mode",
            "elevenlabs_combined_wav_silence_ms",
            "elevenlabs_chunk_settings",
        ):
            if key not in data and key in existing:
                data[key] = existing[key]
        settings.save_settings(data)
        return {"success": True}

    def export_settings(self):
        """アプリ設定をJSONファイルに書き出す。復号済みの秘密情報は含めない。"""
        result = self.window.create_file_dialog(
            webview.FileDialog.SAVE,
            file_types=("設定ファイル (*.json)",),
            save_filename="aviutl-whisper-settings.json",
        )
        if not result:
            return {"success": False, "error": "キャンセルされました"}

        path = result[0] if isinstance(result, (list, tuple)) else result
        try:
            payload = {
                "app": "aviutl-whisper",
                "version": PROJECT_VERSION,
                "settings": settings.load_settings(),
            }
            Path(path).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return {"success": True, "path": str(path)}
        except Exception as e:
            logger.exception("設定書き出しエラー")
            return {"success": False, "error": str(e)}

    def import_settings(self):
        """JSONファイルからアプリ設定を読み込み、既存設定へマージして保存する。"""
        result = self.window.create_file_dialog(
            webview.FileDialog.OPEN,
            file_types=("設定ファイル (*.json)",),
        )
        if not result:
            return {"success": False, "error": "キャンセルされました"}

        path = result[0] if isinstance(result, (list, tuple)) else result
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            imported = None
            if isinstance(payload, dict) and isinstance(payload.get("settings"), dict):
                imported = payload["settings"]
            elif isinstance(payload, dict) and any(
                key in settings.DEFAULT_SETTINGS for key in payload
            ):
                imported = payload
            if not isinstance(imported, dict):
                return {"success": False, "error": "設定ファイルの形式が不正です"}

            imported.pop("hf_token", None)
            imported.pop("hf_token_decrypted", None)
            imported.pop("elevenlabs_api_key_decrypted", None)

            merged = settings._deep_merge(settings.load_settings(), imported)
            settings.save_settings(merged)
            return {"success": True, "settings": self.load_settings(), "path": str(path)}
        except json.JSONDecodeError:
            return {"success": False, "error": "JSONの読み込みに失敗しました"}
        except Exception as e:
            logger.exception("設定読み込みエラー")
            return {"success": False, "error": str(e)}
    # --- ElevenLabs TTS（プロジェクト・Undo/Redoから独立） ---

    def _tts_segments(self):
        if not self._last_segments:
            return []
        return _apply_speaker_mapping(self._last_segments, self._speaker_mapping)

    def _save_tts_settings(self, tts_settings: dict):
        saved = settings.load_settings()
        if "api_key" in tts_settings:
            saved["elevenlabs_api_key_encrypted"] = settings.encrypt_token(
                str(tts_settings.get("api_key") or "")
            )
        saved["elevenlabs_model_id"] = str(
            tts_settings.get("model_id")
            or elevenlabs_tts.DEFAULT_MODEL_ID
        )
        saved["elevenlabs_output_format"] = str(
            tts_settings.get("output_format")
            or elevenlabs_tts.DEFAULT_OUTPUT_FORMAT
        )
        voice_ids = tts_settings.get("speaker_voice_ids", {})
        saved["elevenlabs_speaker_voice_ids"] = (
            dict(voice_ids) if isinstance(voice_ids, dict) else {}
        )
        saved["elevenlabs_output_dir"] = str(
            tts_settings.get("output_dir") or ""
        )
        saved["elevenlabs_voice_settings"] = (
            elevenlabs_tts.normalize_voice_settings(tts_settings)
        )
        saved["elevenlabs_apply_language_text_normalization"] = False
        saved["elevenlabs_apply_text_normalization"] = (
            elevenlabs_tts.normalize_apply_text_normalization(
                tts_settings.get("apply_text_normalization", "auto")
            )
        )
        saved["elevenlabs_generation_mode"] = (
            elevenlabs_tts.resolve_generation_mode(tts_settings)
        )
        saved["elevenlabs_combined_wav_silence_ms"] = (
            elevenlabs_tts.normalize_combined_wav_silence_ms(tts_settings)
        )
        saved["elevenlabs_chunk_settings"] = (
            elevenlabs_tts.normalize_chunk_settings(tts_settings)
        )
        settings.save_settings(saved)

    def save_tts_settings(self, tts_settings: dict):
        """TTS設定だけを既存settings.jsonへマージして保存する。"""
        self._save_tts_settings(tts_settings)
        return {"success": True}

    def _elevenlabs_api_key(self, api_key: str | None = None) -> str:
        if api_key and api_key.strip():
            return api_key.strip()
        encrypted = settings.load_settings().get(
            "elevenlabs_api_key_encrypted", ""
        )
        return settings.decrypt_token(encrypted) if encrypted else ""

    def list_elevenlabs_voices(self, api_key: str | None = None):
        """利用可能なElevenLabsボイスを返す。"""
        try:
            voices = elevenlabs_tts.list_voices(
                self._elevenlabs_api_key(api_key)
            )
            return {"success": True, "voices": voices}
        except Exception as exc:
            logger.warning("ElevenLabsボイス一覧取得エラー: %s", exc)
            return {"success": False, "error": str(exc)}

    def list_elevenlabs_models(self, api_key: str | None = None):
        """TTS対応ElevenLabsモデルを返す。"""
        try:
            models_list = elevenlabs_tts.list_models(
                self._elevenlabs_api_key(api_key)
            )
            return {"success": True, "models": models_list}
        except Exception as exc:
            logger.warning("ElevenLabsモデル一覧取得エラー: %s", exc)
            return {"success": False, "error": str(exc)}

    def select_tts_output_dir(self):
        """TTS音声の出力先フォルダを選択する。"""
        result = self.window.create_file_dialog(webview.FileDialog.FOLDER)
        if not result:
            return None
        return result if isinstance(result, str) else result[0]

    def save_tts_script(self, format_type: str, tts_settings: dict):
        """現在のセグメントをTTS用TXT/CSV台本として保存する。"""
        segments = self._tts_segments()
        if not segments:
            return {"success": False, "error": "保存するセグメントがありません"}
        fmt = str(format_type).lower()
        if fmt not in ("txt", "csv"):
            return {"success": False, "error": "未対応の台本形式です"}
        self._save_tts_settings(tts_settings)
        result = self.window.create_file_dialog(
            webview.FileDialog.SAVE,
            file_types=(
                "テキストファイル (*.txt)"
                if fmt == "txt"
                else "CSVファイル (*.csv)",
            ),
            save_filename=f"tts_script.{fmt}",
        )
        if not result:
            return {"success": False, "error": "キャンセルされました"}
        path = Path(result if isinstance(result, str) else result[0])
        try:
            if fmt == "txt":
                content = elevenlabs_tts.export_script_txt(segments)
                path.write_text(content, encoding="utf-8")
            else:
                content = elevenlabs_tts.export_script_csv(
                    segments, tts_settings
                )
                path.write_text(content, encoding="utf-8-sig", newline="")
            return {"success": True, "path": str(path)}
        except Exception as exc:
            logger.exception("TTS台本保存エラー")
            return {"success": False, "error": str(exc)}

    def save_combined_tts_wav(self, tts_settings: dict):
        """生成済みTTS音声を現在順で結合し、WAVとして保存する。"""
        segments = self._tts_segments()
        if not segments:
            return {"success": False, "error": "保存するセグメントがありません"}
        self._save_tts_settings(tts_settings)
        available, skipped = elevenlabs_tts.get_combined_audio_items(
            segments, tts_settings
        )
        if not available:
            return {
                "success": False,
                "error": "結合できる生成済みTTS音声がありません",
                "skipped": skipped,
            }
        result = self.window.create_file_dialog(
            webview.FileDialog.SAVE,
            file_types=("WAVファイル (*.wav)",),
            save_filename="tts_combined.wav",
        )
        if not result:
            return {"success": False, "error": "キャンセルされました"}
        path = Path(result if isinstance(result, str) else result[0])
        if path.suffix.lower() != ".wav":
            path = path.with_suffix(".wav")
        self._tts_progress = {
            "progress": 0.0,
            "message": "生成済みTTS音声を結合中...",
        }
        try:
            exported = elevenlabs_tts.export_combined_wav(
                segments, tts_settings, path
            )
            self._tts_progress = {
                "progress": 1.0,
                "message": "結合WAVを保存しました",
            }
            return {"success": True, **exported}
        except Exception as exc:
            logger.warning("TTS結合WAV保存エラー: %s", exc)
            self._tts_progress = {"progress": -1.0, "message": str(exc)}
            return {"success": False, "error": str(exc)}

    def generate_tts_for_segment(
        self, index: int, tts_settings: dict, force: bool = True
    ):
        """指定セグメントのTTSを生成する。"""
        segments = self._tts_segments()
        if not segments:
            return {"success": False, "error": "生成するセグメントがありません"}
        self._save_tts_settings(tts_settings)
        self._tts_cancelled = False
        self._tts_progress = {
            "progress": 0.0,
            "message": f"セグメント {index + 1} を生成中...",
        }
        try:
            mode = elevenlabs_tts.resolve_generation_mode(tts_settings)
            if mode == elevenlabs_tts.CHUNK_V3:
                chunks = elevenlabs_tts.build_chunks(
                    segments, tts_settings
                )
                owning_chunk = next(
                    (
                        chunk for chunk in chunks
                        if index in chunk["segment_indices"]
                    ),
                    None,
                )
                if owning_chunk is None:
                    raise elevenlabs_tts.ElevenLabsTtsError(
                        "対象チャンクが見つかりません"
                    )
                self._tts_progress["message"] = (
                    f"チャンク {owning_chunk['chunk_index'] + 1} を生成中..."
                )
                generated = elevenlabs_tts.generate_chunk(
                    segments,
                    owning_chunk["chunk_index"],
                    tts_settings,
                    force=force,
                )
            else:
                generated = elevenlabs_tts.generate_segment(
                    segments, index, tts_settings, force=force
                )
            self._tts_progress = {"progress": 1.0, "message": "完了"}
            return {"success": True, **generated}
        except Exception as exc:
            logger.warning("TTSセグメント生成エラー: %s", exc)
            self._tts_progress = {"progress": -1.0, "message": str(exc)}
            return {"success": False, "error": str(exc)}

    def generate_tts_for_all(self, tts_settings: dict, force: bool = False):
        """全セグメントを順番に生成し、個別エラー後も続行する。"""
        segments = self._tts_segments()
        if not segments:
            return {"success": False, "error": "生成するセグメントがありません"}
        voice_ids = tts_settings.get("speaker_voice_ids", {})
        speakers = sorted({seg.speaker or "Speaker 1" for seg in segments})
        missing = [speaker for speaker in speakers if not voice_ids.get(speaker)]
        if missing:
            return {
                "success": False,
                "error": "voice_id未設定: " + ", ".join(missing),
            }
        self._save_tts_settings(tts_settings)
        self._tts_cancelled = False
        counts = {"generated": 0, "skipped": 0, "errors": 0}
        errors = []
        mode = elevenlabs_tts.resolve_generation_mode(tts_settings)
        chunks = (
            elevenlabs_tts.build_chunks(segments, tts_settings)
            if mode == elevenlabs_tts.CHUNK_V3
            else []
        )
        total = len(chunks) if chunks else len(segments)
        for index in range(total):
            if self._tts_cancelled:
                self._tts_progress = {
                    "progress": index / total,
                    "message": "TTS生成を中止しました",
                }
                return {
                    "success": False,
                    "cancelled": True,
                    **counts,
                    "errors_detail": errors,
                }
            self._tts_progress = {
                "progress": index / total,
                "message": (
                    f"チャンク {index + 1} / {total} を生成中..."
                    if mode == elevenlabs_tts.CHUNK_V3
                    else f"{index + 1} / {total} を生成中..."
                ),
            }
            try:
                if mode == elevenlabs_tts.CHUNK_V3:
                    result = elevenlabs_tts.generate_chunk(
                        segments, index, tts_settings, force=force
                    )
                else:
                    result = elevenlabs_tts.generate_segment(
                        segments, index, tts_settings, force=force
                    )
                counts[result["status"]] += 1
            except Exception as exc:
                logger.warning("TTS一括生成の個別エラー index=%d: %s", index, exc)
                counts["errors"] += 1
                errors.append({"index": index, "error": str(exc)})
        self._tts_progress = {
            "progress": 1.0,
            "message": (
                f"完了: 生成 {counts['generated']} / "
                f"スキップ {counts['skipped']} / エラー {counts['errors']}"
            ),
        }
        return {
            "success": counts["errors"] == 0,
            **counts,
            "errors_detail": errors,
        }

    def get_tts_status(self, tts_settings: dict):
        """manifestと現在のセグメントからTTS状態を返す。"""
        segments = self._tts_segments()
        if not segments:
            return {"success": False, "error": "セグメントがありません"}
        try:
            statuses = elevenlabs_tts.get_segment_statuses(
                segments, tts_settings
            )
            return {
                "success": True,
                "segments": statuses,
                "speakers": sorted({item["speaker"] for item in statuses}),
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def get_tts_progress(self):
        return dict(self._tts_progress)

    def cancel_tts(self):
        self._tts_cancelled = True
        return {"success": True}

    def play_tts_segment(self, index: int, tts_settings: dict):
        """生成済みTTS音声を再生する。元音声再生には影響しない。"""
        segments = self._tts_segments()
        if not segments:
            return {"success": False, "error": "セグメントがありません"}
        try:
            statuses = elevenlabs_tts.get_segment_statuses(
                segments, tts_settings
            )
            if index < 0 or index >= len(statuses):
                return {"success": False, "error": "無効なインデックス"}
            path = statuses[index].get("audio_path", "")
            if not path or not os.path.exists(path):
                return {"success": False, "error": "TTS音声が生成されていません"}
            import sounddevice as sd
            import soundfile as sf

            temp_path = None
            try:
                if Path(path).suffix.lower() == ".wav":
                    wav_path = path
                else:
                    from pydub import AudioSegment

                    fd, temp_path = tempfile.mkstemp(suffix=".wav")
                    os.close(fd)
                    AudioSegment.from_file(path).export(
                        temp_path, format="wav"
                    )
                    wav_path = temp_path
                data, sample_rate = sf.read(wav_path)
            finally:
                if temp_path:
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass
            sd.stop()
            sd.play(data, sample_rate)
            return {"success": True}
        except Exception as exc:
            logger.warning("TTS音声再生エラー: %s", exc)
            return {"success": False, "error": str(exc)}

    # --- プロジェクト保存・読み込み ---

    def _save_project_to_file(self, data: dict, path: str) -> dict:
        """プロジェクトデータを指定パスに書き出す内部ヘルパー。

        ダイアログを開かず、渡されたパスに直接保存する。
        保存成功時は _is_dirty をリセットし、ウィンドウタイトルを更新する。
        """
        if not data.get("segments"):
            return {"success": False, "error": "保存するセグメントがありません"}
        try:
            Path(path).write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._is_dirty = False
            self._current_project_path = path
            self._update_window_title()
            return {"success": True, "path": path}
        except PermissionError as e:
            logger.exception("プロジェクト保存エラー (アクセス拒否)")
            return {"success": False, "error": f"ファイルへの書き込み権限がありません: {e}"}
        except OSError as e:
            logger.exception("プロジェクト保存エラー (OS)")
            return {"success": False, "error": f"ファイルの書き込みに失敗しました: {e}"}
        except Exception as e:
            logger.exception("プロジェクト保存エラー")
            return {"success": False, "error": str(e)}

    def _build_project_data(self, project_data: dict) -> dict | None:
        """フロントエンドから受け取ったデータとバックエンド状態からプロジェクトデータを構築する。"""
        if not self._last_segments:
            return None
        mapping = self._speaker_mapping
        # セグメントは元の話者ラベルのまま保存し、マッピングを別フィールドに
        # 保存する。読み込み時にマッピングを一度だけ適用できるようにする。
        segments = self._last_segments
        return {
            "version": PROJECT_VERSION,
            "source_file": project_data.get("source_file", ""),
            "audio_mode": self._audio_mode,
            "language": (
                self._last_result.language if self._last_result else ""
            ),
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "speaker": s.speaker or "Speaker 1",
                }
                for s in segments
            ],
            "speaker_mapping": mapping,
            "exo_settings": project_data.get("exo_settings", {}),
            "preview_index": project_data.get("preview_index", 0),
        }

    def save_project(self, project_data: dict):
        """プロジェクトファイル (.awproj) に保存する。

        保存先パスが既知の場合は上書き保存する。
        未設定の場合はダイアログを開いて保存先を選択する。
        """
        if self._current_project_path:
            data = self._build_project_data(project_data)
            if data is None:
                return {"success": False, "error": "保存するセグメントがありません"}
            return self._save_project_to_file(data, self._current_project_path)

        return self.save_project_as(project_data)

    def save_project_as(self, project_data: dict):
        """プロジェクトファイル (.awproj) を別名で保存する。

        常にダイアログを開いて保存先を選択する。
        """
        result = self.window.create_file_dialog(
            webview.FileDialog.SAVE,
            file_types=("プロジェクトファイル (*.awproj)",),
            save_filename="project.awproj",
        )
        if not result:
            return {"success": False, "error": "キャンセルされました"}

        path = result if isinstance(result, str) else result[0]

        data = self._build_project_data(project_data)
        if data is None:
            return {"success": False, "error": "保存するセグメントがありません"}

        return self._save_project_to_file(data, path)

    def load_project(self):
        """プロジェクトファイル (.awproj) を読み込み、内部状態を復元する。"""
        result = self.window.create_file_dialog(
            webview.FileDialog.OPEN,
            file_types=("プロジェクトファイル (*.awproj)",),
        )
        if not result or len(result) == 0:
            return {"success": False, "error": "キャンセルされました"}

        path = result[0]
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as e:
            logger.exception("プロジェクトファイル読み込みエラー")
            return {"success": False, "error": f"ファイルの読み込みに失敗: {e}"}

        if not isinstance(data, dict) or "segments" not in data:
            return {"success": False, "error": "無効なプロジェクトファイルです"}

        segments_raw = data.get("segments", [])
        segments = [
            transcriber.TranscriptionSegment(
                start=s["start"],
                end=s["end"],
                text=s["text"],
                speaker=s.get("speaker"),
            )
            for s in segments_raw
        ]

        if not segments:
            return {"success": False, "error": "セグメントが空です"}

        self._last_segments = segments
        self._speaker_mapping = data.get("speaker_mapping")
        self._current_project_path = path
        self._is_dirty = False
        self._update_window_title()

        language = data.get("language", "")
        self._last_result = transcriber.TranscriptionResult(
            segments=list(segments),
            language=language,
            language_probability=0.0,
        )

        source_file = data.get("source_file", "")
        audio_mode = data.get("audio_mode")
        self._cleanup_wav()
        if audio_mode == "silence":
            self._audio_mode = "silence"
            self._sync_silent_wav()
        elif source_file and os.path.exists(source_file):
            try:
                wav_path = audio.convert_to_wav(source_file)
                self._last_wav_path = wav_path
                self._audio_mode = "source"
            except Exception:
                self._audio_mode = "none"
                logger.warning("WAV変換に失敗: %s", source_file)
        else:
            self._audio_mode = "none"

        speaker_info = self._build_speaker_info(segments)

        return {
            "success": True,
            "path": path,
            "source_file": source_file,
            "audio_mode": self._audio_mode,
            "language": language,
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "speaker": s.speaker or "Speaker 1",
                }
                for s in segments
            ],
            "speakers": speaker_info,
            "num_segments": len(segments),
            "num_speakers": len(speaker_info),
            "exo_settings": data.get("exo_settings", {}),
            "preview_index": data.get("preview_index", 0),
            "speaker_mapping": self._speaker_mapping,
        }

    # --- セグメント編集 ---

    def _bake_mapping(self):
        """現在のマッピングを _last_segments に焼き込み、マッピングをリセットする。

        セグメント編集操作の前に呼ぶことで、エディタが表示通りの話者名で
        直接操作できるようにする。
        """
        if not self._speaker_mapping or not self._last_segments:
            return
        self._last_segments = _apply_speaker_mapping(
            self._last_segments, self._speaker_mapping
        )
        self._speaker_mapping = None

    def _regenerate_output(self):
        """現在のセグメントから出力テキストを再生成して返す。"""
        segments = _apply_speaker_mapping(
            self._last_segments, self._speaker_mapping
        )
        fmt = self._last_output_format
        if fmt == "exo":
            text = exporter.export_exo(segments, settings=self._exo_settings)
        else:
            _, _, export_fn = exporter.EXPORTERS[fmt]
            text = export_fn(segments)
        return text

    def _segments_response(self):
        """セグメント編集後の共通レスポンスを構築する。"""
        text = self._regenerate_output()
        segments = _apply_speaker_mapping(
            self._last_segments, self._speaker_mapping
        )
        speakers = sorted(set(s.speaker or "Speaker 1" for s in self._last_segments))
        return {
            "success": True,
            "text": text,
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "speaker": s.speaker or "Speaker 1",
                }
                for s in segments
            ],
            "speakers": speakers,
        }

    def update_segment(self, index: int, speaker: str | None = None,
                       text: str | None = None,
                       start: float | None = None, end: float | None = None):
        """セグメントの話者・テキスト・時刻を更新する。"""
        if not self._last_segments:
            return {"success": False, "error": "結果がありません"}
        self._bake_mapping()
        if index < 0 or index >= len(self._last_segments):
            return {"success": False, "error": "無効なインデックス"}

        seg = self._last_segments[index]
        try:
            next_start = float(start) if start is not None else seg.start
            next_end = float(end) if end is not None else seg.end
        except (TypeError, ValueError):
            return {"success": False, "error": "開始時刻と終了時刻には数値を指定してください"}
        time_error = self._validate_segment_times(next_start, next_end)
        if time_error:
            return {"success": False, "error": time_error}

        updated = transcriber.TranscriptionSegment(
            start=next_start,
            end=next_end,
            text=text if text is not None else seg.text,
            speaker=speaker if speaker is not None else seg.speaker,
        )
        self._last_segments[index] = updated
        self._last_segments.sort(key=lambda item: (item.start, item.end))
        self._sync_silent_wav()
        self._is_dirty = True
        resp = self._segments_response()
        resp["updated_index"] = next(
            i for i, item in enumerate(self._last_segments) if item is updated
        )
        return resp

    @staticmethod
    def _validate_segment_times(start: float, end: float):
        """セグメント時刻の共通バリデーション。"""
        if not math.isfinite(start) or not math.isfinite(end):
            return "開始時刻と終了時刻には有限の数値を指定してください"
        if start < 0:
            return "開始時刻は0秒以上にしてください"
        if end <= start:
            return "開始時刻は終了時刻より前にしてください"
        if end - start < 0.05:
            return "セグメントの長さは0.05秒以上にしてください"
        return None

    def add_segment(self, start: float, end: float, text: str,
                    speaker: str = "Speaker 1"):
        """新しいセグメントを時刻順に挿入する。"""
        if not self._last_segments:
            return {"success": False, "error": "結果がありません"}
        try:
            start = float(start)
            end = float(end)
        except (TypeError, ValueError):
            return {"success": False, "error": "開始時刻と終了時刻には数値を指定してください"}
        time_error = self._validate_segment_times(start, end)
        if time_error:
            return {"success": False, "error": time_error}
        self._bake_mapping()

        new_seg = transcriber.TranscriptionSegment(
            start=start, end=end, text=text, speaker=speaker,
        )
        # 時刻順に挿入位置を探す
        insert_idx = 0
        for i, seg in enumerate(self._last_segments):
            if seg.start > start:
                break
            insert_idx = i + 1
        self._last_segments.insert(insert_idx, new_seg)

        self._sync_silent_wav()
        self._is_dirty = True
        resp = self._segments_response()
        resp["inserted_index"] = insert_idx
        return resp

    def duplicate_segment(self, index: int):
        """セグメントを同じ内容・長さで元セグメントの直後に複製する。"""
        if not self._last_segments:
            return {"success": False, "error": "結果がありません"}
        self._bake_mapping()
        if index < 0 or index >= len(self._last_segments):
            return {"success": False, "error": "無効なインデックス"}

        source = self._last_segments[index]
        duration = source.end - source.start
        duplicated = transcriber.TranscriptionSegment(
            start=source.end,
            end=source.end + duration,
            text=source.text,
            speaker=source.speaker,
        )
        self._last_segments.append(duplicated)
        self._last_segments.sort(key=lambda item: (item.start, item.end))

        self._sync_silent_wav()
        self._is_dirty = True
        resp = self._segments_response()
        resp["duplicated_index"] = next(
            i for i, item in enumerate(self._last_segments) if item is duplicated
        )
        return resp

    def delete_segment(self, index: int):
        """セグメントを削除する。"""
        if not self._last_segments:
            return {"success": False, "error": "結果がありません"}
        self._bake_mapping()
        if index < 0 or index >= len(self._last_segments):
            return {"success": False, "error": "無効なインデックス"}
        if len(self._last_segments) <= 1:
            return {"success": False, "error": "最後のセグメントは削除できません"}

        self._last_segments.pop(index)
        self._sync_silent_wav()
        self._is_dirty = True
        return self._segments_response()

    def restore_segments(self, segments_data: list):
        """スナップショットからセグメントを復元する（Undo/Redo 用）。

        Args:
            segments_data: [{"start": float, "end": float, "text": str, "speaker": str}] の配列
        """
        if segments_data is None:
            return {"success": False, "error": "セグメントデータがありません"}

        try:
            self._last_segments = [
                transcriber.TranscriptionSegment(
                    start=s["start"],
                    end=s["end"],
                    text=s["text"],
                    speaker=s.get("speaker", "Speaker 1"),
                )
                for s in segments_data
            ]
        except Exception as e:
            logger.exception("セグメント復元エラー")
            return {"success": False, "error": str(e)}

        # スナップショットはマッピング適用済みのため、マッピングをリセット
        self._speaker_mapping = None
        self._sync_silent_wav()
        self._is_dirty = True
        return self._segments_response()

    def merge_segments(self, index: int):
        """隣接する2セグメントを結合する。

        index が前側セグメント、index+1 が後側セグメント。
        両者の話者が同一の場合のみ結合を許可する。
        結合後: start=前.start, end=後.end, text=前.text+"\n"+後.text
        """
        if not self._last_segments:
            return {"success": False, "error": "結果がありません"}
        self._bake_mapping()
        if index < 0 or index + 1 >= len(self._last_segments):
            return {"success": False, "error": "無効なインデックス"}

        seg_a = self._last_segments[index]
        seg_b = self._last_segments[index + 1]
        if (seg_a.speaker or "Speaker 1") != (seg_b.speaker or "Speaker 1"):
            return {"success": False, "error": "話者が異なるセグメントは結合できません"}

        merged = transcriber.TranscriptionSegment(
            start=seg_a.start,
            end=seg_b.end,
            text=seg_a.text + "\n" + seg_b.text,
            speaker=seg_a.speaker,
        )
        self._last_segments[index] = merged
        self._last_segments.pop(index + 1)
        self._sync_silent_wav()
        self._is_dirty = True
        resp = self._segments_response()
        resp["merged_index"] = index
        return resp

    def get_preview_segments(self, speaker_mapping: dict | None = None):
        """プレビュー用のセグメント一覧を返す（マッピング適用済み）。"""
        if not self._last_segments:
            return {"success": False, "error": "結果がありません"}

        mapping = speaker_mapping or self._speaker_mapping
        segments = _apply_speaker_mapping(self._last_segments, mapping)

        return {
            "success": True,
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "speaker": s.speaker or "Speaker 1",
                }
                for s in segments
            ],
        }

    def get_image_base64(self, file_path: str):
        """ローカル画像ファイルをbase64エンコードして返す。"""
        import base64
        import mimetypes

        if not file_path or not os.path.exists(file_path):
            return {"success": False, "error": "ファイルが見つかりません"}

        try:
            mime, _ = mimetypes.guess_type(file_path)
            if mime is None:
                mime = "image/png"
            with open(file_path, "rb") as f:
                data = base64.b64encode(f.read()).decode("ascii")
            return {
                "success": True,
                "data_url": f"data:{mime};base64,{data}",
            }
        except Exception as e:
            logger.exception("画像読み込みエラー")
            return {"success": False, "error": str(e)}

    def render_subtitle_image(self, text: str, speaker_index: int = 0,
                              settings_dict: dict | None = None):
        """字幕テキストをPillow で透過PNGにレンダリングし、base64で返す。"""
        import base64

        s = settings_dict or {}
        colors = s.get("speaker_colors", [])
        if not colors:
            colors = ["ffffff", "00ffff", "00ff00", "ff00ff",
                      "ffff00", "ff8000", "8080ff", "80ff80"]
        edge_colors = s.get("speaker_edge_colors", [])
        if not edge_colors:
            edge_colors = ["000000"]

        text_color = colors[speaker_index % len(colors)]
        edge_color = edge_colors[speaker_index % len(edge_colors)]

        try:
            png_bytes = _render_subtitle_image(
                text=text,
                font_name=s.get("font", "MS UI Gothic"),
                font_size=s.get("font_size", 34),
                bold=s.get("bold", False),
                italic=s.get("italic", False),
                text_color=text_color,
                edge_color=edge_color,
                soft_edge=s.get("soft_edge", True),
                align=s.get("align", 4),
                pos_x=s.get("pos_x", 0),
                pos_y=s.get("pos_y", 0),
                spacing_y=s.get("spacing_y", 0),
                max_chars_per_line=s.get("max_chars_per_line", 0),
            )
            data = base64.b64encode(png_bytes).decode("ascii")
            return {
                "success": True,
                "data_url": f"data:image/png;base64,{data}",
            }
        except Exception as e:
            logger.exception("字幕レンダリングエラー")
            return {"success": False, "error": str(e)}

    def render_preview_frame(self, index: int, settings_dict: dict | None = None):
        """指定セグメントの完全プレビュー画像（背景+立ち絵+字幕）を返す。

        全レイヤーをPillow側で合成した1枚のJPEGをbase64で返す。
        """
        import base64

        if not self._last_segments:
            return {"success": False, "error": "結果がありません"}

        mapping = self._speaker_mapping
        segments = _apply_speaker_mapping(self._last_segments, mapping)

        if index < 0 or index >= len(segments):
            return {"success": False, "error": "無効なインデックス"}

        seg = segments[index]
        s = settings_dict or {}

        # 話者インデックスを算出
        import re
        match = re.match(r"Speaker (\d+)", seg.speaker or "Speaker 1")
        speaker_idx = (int(match.group(1)) - 1) if match else 0

        # 話者数
        num_speakers = len(set(
            seg.speaker or "Speaker 1" for seg in segments
        ))

        try:
            jpeg_bytes = _render_preview_frame(
                text=seg.text,
                speaker_index=speaker_idx,
                settings=s,
                num_speakers=num_speakers,
            )
            data = base64.b64encode(jpeg_bytes).decode("ascii")
            return {
                "success": True,
                "data_url": f"data:image/jpeg;base64,{data}",
            }
        except Exception as e:
            logger.exception("プレビューレンダリングエラー")
            return {"success": False, "error": str(e)}

    def _update_progress(self, progress: float, message: str):
        self._progress = {"progress": progress, "message": message}

    def _progress_callback(self, progress: float, message: str):
        self._update_progress(progress, message)
