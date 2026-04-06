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


def _encode_exo_text(text: str) -> str:
    """テキストをAviUtl exo形式のhexエンコードに変換する。

    UTF-16LEでエンコードし、hex文字列にして4096文字に0埋めする。
    """
    hex_str = text.encode("utf-16-le").hex()
    return hex_str.ljust(4096, "0")


def export_exo(
    segments: list[TranscriptionSegment],
    fps: float = 30.0,
    width: int = 1920,
    height: int = 1080,
    font: str = "MS UI Gothic",
    font_size: int = 34,
) -> str:
    """AviUtl拡張編集のexo形式で出力する。

    各セグメントをテキストオブジェクトとしてタイムライン上に配置する。
    話者ごとにレイヤーを分け、色を変える。
    """
    if not segments:
        return ""

    # 話者→レイヤー/色マッピング
    speakers = sorted(set(s.speaker or "Speaker 1" for s in segments))
    speaker_colors = [
        "ffffff",  # 白
        "00ffff",  # シアン (BGR: ffff00 → exoはBGR)
        "00ff00",  # 緑
        "ff00ff",  # マゼンタ
        "ffff00",  # 黄
        "ff8000",  # オレンジ
        "8080ff",  # 薄青
        "80ff80",  # 薄緑
    ]
    speaker_layer = {spk: i + 1 for i, spk in enumerate(speakers)}
    speaker_color = {
        spk: speaker_colors[i % len(speaker_colors)]
        for i, spk in enumerate(speakers)
    }

    # 全体の長さ (フレーム)
    max_end = max(s.end for s in segments)
    total_frames = int(max_end * fps) + 1

    lines = []

    # [exedit] ヘッダー
    lines.append("[exedit]")
    lines.append(f"width={width}")
    lines.append(f"height={height}")
    lines.append(f"rate={int(fps)}")
    lines.append("scale=1")
    lines.append(f"length={total_frames}")
    lines.append("audio_rate=44100")
    lines.append("audio_ch=2")

    for idx, seg in enumerate(segments):
        speaker = seg.speaker or "Speaker 1"
        start_frame = max(1, int(seg.start * fps))
        end_frame = int(seg.end * fps)
        if end_frame <= start_frame:
            end_frame = start_frame + 1

        layer = speaker_layer[speaker]
        color = speaker_color[speaker]
        display_text = seg.text
        if len(speakers) > 1:
            display_text = f"[{speaker}] {seg.text}"

        hex_text = _encode_exo_text(display_text)

        # [N] オブジェクトヘッダー
        lines.append(f"[{idx}]")
        lines.append(f"start={start_frame}")
        lines.append(f"end={end_frame}")
        lines.append(f"layer={layer}")
        lines.append("overlay=1")
        lines.append("camera=0")

        # [N.0] テキストオブジェクト
        lines.append(f"[{idx}.0]")
        lines.append("_name=テキスト")
        lines.append(f"サイズ={font_size}")
        lines.append("表示速度=0.0")
        lines.append("文字毎に個別オブジェクト=0")
        lines.append("移動座標上に表示する=0")
        lines.append("自動スクロール=0")
        lines.append("B=0")
        lines.append("I=0")
        lines.append("type=0")
        lines.append("autoadjust=0")
        lines.append("soft=1")
        lines.append("monospace=0")
        lines.append("align=4")
        lines.append("spacing_x=0")
        lines.append("spacing_y=0")
        lines.append("precision=1")
        lines.append(f"color={color}")
        lines.append("color2=000000")
        lines.append(f"font={font}")
        lines.append(f"text={hex_text}")

        # [N.1] 標準描画
        lines.append(f"[{idx}.1]")
        lines.append("_name=標準描画")
        lines.append("X=0.0")
        lines.append("Y=0.0")
        lines.append("Z=0.0")
        lines.append("拡大率=100.00")
        lines.append("透明度=0.0")
        lines.append("回転=0.00")
        lines.append("blend=0")

    return "\r\n".join(lines) + "\r\n"


EXPORTERS = {
    "srt": ("SRT字幕", ".srt", export_srt),
    "csv": ("CSV", ".csv", export_csv),
    "tsv": ("TSV", ".tsv", export_tsv),
    "text": ("テキスト", ".txt", export_text),
    "exo": ("AviUtl exo", ".exo", export_exo),
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

    # exo は CP932 (Shift_JIS) エンコーディング
    encoding = "cp932" if format_type == "exo" else "utf-8"
    path.write_text(content, encoding=encoding)
    return str(path)
