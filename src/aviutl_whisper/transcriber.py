"""文字起こしモジュール - faster-whisperによる音声認識"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionSegment:
    """文字起こしセグメント"""
    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclass
class TranscriptionResult:
    """文字起こし結果"""
    segments: list[TranscriptionSegment]
    language: str
    language_probability: float


def transcribe(
    model,
    audio_path: str,
    language: str | None = None,
    progress_callback=None,
) -> TranscriptionResult:
    """faster-whisperで音声を文字起こしする。

    Args:
        model: faster-whisperモデル
        audio_path: WAVファイルパス
        language: 言語コード (Noneで自動検出)
        progress_callback: 進捗コールバック (progress: float, message: str)

    Returns:
        TranscriptionResult
    """
    if progress_callback:
        progress_callback(0.0, "文字起こし開始...")

    segments_iter, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
        ),
        word_timestamps=True,
    )

    logger.info(
        "言語検出: %s (確率: %.2f%%)",
        info.language,
        info.language_probability * 100,
    )

    segments = []
    for segment in segments_iter:
        segments.append(
            TranscriptionSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text.strip(),
            )
        )
        if progress_callback:
            progress_callback(-1.0, f"文字起こし中... ({len(segments)}セグメント)")

    if progress_callback:
        progress_callback(1.0, f"文字起こし完了 ({len(segments)}セグメント)")

    logger.info("文字起こし完了: %dセグメント", len(segments))

    return TranscriptionResult(
        segments=segments,
        language=info.language,
        language_probability=info.language_probability,
    )
