"""出力モジュール - 各種フォーマットでの結果出力"""

import csv
import io
from pathlib import Path

from .transcriber import TranscriptionSegment


def format_timestamp_srt(seconds: float) -> str:
    """秒数をSRT形式のタイムスタンプに変換する。"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_timestamp_plain(seconds: float) -> str:
    """秒数を簡易タイムスタンプに変換する。"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def format_timestamp_csv(seconds: float) -> str:
    """秒数をCSV用タイムスタンプに変換する。"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def export_srt(segments: list[TranscriptionSegment]) -> str:
    """SRT形式で出力する。"""
    lines = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp_srt(seg.start)
        end = format_timestamp_srt(seg.end)
        speaker_prefix = f"[{seg.speaker}] " if seg.speaker else ""
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(f"{speaker_prefix}{seg.text}")
        lines.append("")
    return "\n".join(lines)


def export_csv(
    segments: list[TranscriptionSegment],
    delimiter: str = ",",
) -> str:
    """CSV/TSV形式で出力する。"""
    output = io.StringIO()
    writer = csv.writer(output, delimiter=delimiter)
    writer.writerow(["start", "end", "speaker", "text"])
    for seg in segments:
        writer.writerow([
            format_timestamp_csv(seg.start),
            format_timestamp_csv(seg.end),
            seg.speaker or "",
            seg.text,
        ])
    return output.getvalue()


def export_tsv(segments: list[TranscriptionSegment]) -> str:
    """TSV形式で出力する。"""
    return export_csv(segments, delimiter="\t")


def export_text(segments: list[TranscriptionSegment]) -> str:
    """プレーンテキスト形式で出力する。"""
    lines = []
    for seg in segments:
        start = format_timestamp_plain(seg.start)
        end = format_timestamp_plain(seg.end)
        speaker = seg.speaker or "Unknown"
        lines.append(f"[{start} - {end}] {speaker}: {seg.text}")
    return "\n".join(lines)


EXPORTERS = {
    "srt": ("SRT字幕", ".srt", export_srt),
    "csv": ("CSV", ".csv", export_csv),
    "tsv": ("TSV", ".tsv", export_tsv),
    "text": ("テキスト", ".txt", export_text),
}


def export_to_file(
    segments: list[TranscriptionSegment],
    output_path: str,
    format_type: str = "text",
) -> str:
    """指定フォーマットでファイルに出力する。

    Returns:
        出力ファイルパス
    """
    if format_type not in EXPORTERS:
        raise ValueError(
            f"未対応の出力形式: {format_type}\n"
            f"対応形式: {', '.join(EXPORTERS.keys())}"
        )

    _, ext, export_fn = EXPORTERS[format_type]
    path = Path(output_path)

    if path.suffix.lower() != ext:
        path = path.with_suffix(ext)

    content = export_fn(segments)
    path.write_text(content, encoding="utf-8")
    return str(path)
