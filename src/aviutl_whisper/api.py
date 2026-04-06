"""Python↔JS ブリッジAPI - pywebviewのJSから呼び出されるAPI"""

import logging
import os
import threading
from pathlib import Path

import webview

from . import audio, diarizer, exporter, models, transcriber

logger = logging.getLogger(__name__)


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

    def set_window(self, window: webview.Window):
        self.window = window

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

    def transcribe(self, file_path: str, settings: dict):
        """文字起こし＋話者分離を実行する。"""
        self._cancelled = False
        try:
            return self._run_transcription(file_path, settings)
        except Exception as e:
            logger.exception("文字起こしエラー")
            return {"success": False, "error": str(e)}

    def _run_transcription(self, file_path: str, settings: dict):
        model_size = settings.get("model_size", "medium")
        language = settings.get("language")
        num_speakers = settings.get("num_speakers")
        output_format = settings.get("output_format", "text")

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

        if self._cancelled:
            return {"success": False, "error": "キャンセルされました"}

        # 4. 話者分離モデル読み込み
        self._update_progress(0.6, "話者分離モデルを読み込み中...")
        speaker_model = models.load_speechbrain_model(
            progress_callback=self._progress_callback,
        )

        if self._cancelled:
            return {"success": False, "error": "キャンセルされました"}

        # 5. 話者分離
        self._update_progress(0.7, "話者分離中...")
        segments = diarizer.assign_speakers(
            model=speaker_model,
            audio_path=wav_path,
            segments=result.segments,
            num_speakers=num_speakers,
            progress_callback=self._progress_callback,
        )

        # 6. 結果フォーマット
        self._update_progress(0.95, "結果を生成中...")
        _, _, export_fn = exporter.EXPORTERS[output_format]
        text = export_fn(segments)

        self._last_result = result
        self._last_segments = segments

        # 一時ファイル削除
        try:
            os.unlink(wav_path)
        except OSError:
            pass

        speakers = set(s.speaker for s in segments if s.speaker)
        self._update_progress(1.0, "完了！")

        return {
            "success": True,
            "text": text,
            "num_segments": len(segments),
            "num_speakers": len(speakers),
            "language": result.language,
        }

    def save_result(self, format_type: str):
        """結果をファイルに保存する。"""
        if not self._last_segments:
            return {"success": False, "error": "保存する結果がありません"}

        _, ext, _ = exporter.EXPORTERS.get(format_type, ("", ".txt", None))
        file_types = (f"*{ext}",)

        result = self.window.create_file_dialog(
            webview.FileDialog.SAVE,
            file_types=file_types,
        )

        if result:
            path = result if isinstance(result, str) else result[0]
            saved_path = exporter.export_to_file(
                self._last_segments, path, format_type
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
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                return {"device": "GPU (CUDA)", "detail": name}
        except ImportError:
            pass
        return {"device": "CPU", "detail": "GPUは利用できません"}

    def _update_progress(self, progress: float, message: str):
        self._progress = {"progress": progress, "message": message}

    def _progress_callback(self, progress: float, message: str):
        self._update_progress(progress, message)
