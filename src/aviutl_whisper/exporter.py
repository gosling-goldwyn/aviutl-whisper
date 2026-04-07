"""出力モジュール - 各種フォーマットでの結果出力"""

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path

from .transcriber import TranscriptionSegment


# AviUtl exo の align 値
EXO_ALIGN = {
    "左上": 0, "上中央": 1, "右上": 2,
    "左中": 3, "中央": 4, "右中": 5,
    "左下": 6, "下中央": 7, "右下": 8,
}

DEFAULT_SPEAKER_COLORS = [
    "ffffff",  # 白
    "00ffff",  # シアン
    "00ff00",  # 緑
    "ff00ff",  # マゼンタ
    "ffff00",  # 黄
    "ff8000",  # オレンジ
    "8080ff",  # 薄青
    "80ff80",  # 薄緑
]

DEFAULT_EDGE_COLOR = "000000"


@dataclass
class ExoSettings:
    """AviUtl exo出力の詳細設定。"""
    fps: float = 30.0
    width: int = 1920
    height: int = 1080
    font: str = "MS UI Gothic"
    font_size: int = 34
    spacing_x: int = 0
    spacing_y: int = 0
    display_speed: float = 0.0
    align: int = 4
    bold: bool = False
    italic: bool = False
    soft_edge: bool = True
    pos_x: float = 0.0
    pos_y: float = 0.0
    # 話者ごとの文字色 (hex, BGRではなくRGB入力 → exo出力時にBGR変換)
    speaker_colors: list[str] = field(default_factory=lambda: list(DEFAULT_SPEAKER_COLORS))
    # 話者ごとの縁色
    speaker_edge_colors: list[str] = field(default_factory=lambda: [])

    def get_speaker_color(self, index: int) -> str:
        """話者インデックスに対応する文字色 (RGB hex) を返す。"""
        colors = self.speaker_colors or DEFAULT_SPEAKER_COLORS
        return colors[index % len(colors)]

    def get_speaker_edge_color(self, index: int) -> str:
        """話者インデックスに対応する縁色 (RGB hex) を返す。"""
        if self.speaker_edge_colors:
            return self.speaker_edge_colors[index % len(self.speaker_edge_colors)]
        return DEFAULT_EDGE_COLOR

    @classmethod
    def from_dict(cls, d: dict) -> "ExoSettings":
        """フロントエンドから渡された辞書からExoSettingsを生成する。"""
        if not d:
            return cls()
        return cls(
            fps=float(d.get("fps", 30.0)),
            width=int(d.get("width", 1920)),
            height=int(d.get("height", 1080)),
            font=d.get("font", "MS UI Gothic"),
            font_size=int(d.get("font_size", 34)),
            spacing_x=int(d.get("spacing_x", 0)),
            spacing_y=int(d.get("spacing_y", 0)),
            display_speed=float(d.get("display_speed", 0.0)),
            align=int(d.get("align", 4)),
            bold=bool(d.get("bold", False)),
            italic=bool(d.get("italic", False)),
            soft_edge=bool(d.get("soft_edge", True)),
            pos_x=float(d.get("pos_x", 0.0)),
            pos_y=float(d.get("pos_y", 0.0)),
            speaker_colors=d.get("speaker_colors", list(DEFAULT_SPEAKER_COLORS)),
            speaker_edge_colors=d.get("speaker_edge_colors", []),
        )




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
    settings: ExoSettings | None = None,
    **kwargs,
) -> str:
    """AviUtl拡張編集のexo形式で出力する。

    各セグメントをテキストオブジェクトとしてタイムライン上に配置する。
    話者ごとにレイヤーを分け、色を変える。
    """
    if not segments:
        return ""

    if settings is None:
        settings = ExoSettings(**kwargs) if kwargs else ExoSettings()

    # 話者→レイヤー/色マッピング
    speakers = sorted(set(s.speaker or "Speaker 1" for s in segments))
    speaker_layer = {spk: i + 1 for i, spk in enumerate(speakers)}
    speaker_color = {
        spk: settings.get_speaker_color(i)
        for i, spk in enumerate(speakers)
    }
    speaker_edge = {
        spk: settings.get_speaker_edge_color(i)
        for i, spk in enumerate(speakers)
    }

    # 全体の長さ (フレーム)
    max_end = max(s.end for s in segments)
    total_frames = int(max_end * settings.fps) + 1

    lines = []

    # [exedit] ヘッダー
    lines.append("[exedit]")
    lines.append(f"width={settings.width}")
    lines.append(f"height={settings.height}")
    lines.append(f"rate={int(settings.fps)}")
    lines.append("scale=1")
    lines.append(f"length={total_frames}")
    lines.append("audio_rate=44100")
    lines.append("audio_ch=2")

    for idx, seg in enumerate(segments):
        speaker = seg.speaker or "Speaker 1"
        start_frame = max(1, int(seg.start * settings.fps))
        end_frame = int(seg.end * settings.fps)
        if end_frame <= start_frame:
            end_frame = start_frame + 1

        layer = speaker_layer[speaker]
        color = speaker_color[speaker]
        edge_color = speaker_edge[speaker]
        display_text = seg.text

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
        lines.append(f"サイズ={settings.font_size}")
        lines.append(f"表示速度={settings.display_speed:.1f}")
        lines.append("文字毎に個別オブジェクト=0")
        lines.append("移動座標上に表示する=0")
        lines.append("自動スクロール=0")
        lines.append(f"B={1 if settings.bold else 0}")
        lines.append(f"I={1 if settings.italic else 0}")
        lines.append("type=0")
        lines.append("autoadjust=0")
        lines.append(f"soft={1 if settings.soft_edge else 0}")
        lines.append("monospace=0")
        lines.append(f"align={settings.align}")
        lines.append(f"spacing_x={settings.spacing_x}")
        lines.append(f"spacing_y={settings.spacing_y}")
        lines.append("precision=1")
        lines.append(f"color={color}")
        lines.append(f"color2={edge_color}")
        lines.append(f"font={settings.font}")
        lines.append(f"text={hex_text}")

        # [N.1] 標準描画
        lines.append(f"[{idx}.1]")
        lines.append("_name=標準描画")
        lines.append(f"X={settings.pos_x:.1f}")
        lines.append(f"Y={settings.pos_y:.1f}")
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
    exo_settings: ExoSettings | None = None,
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

    if format_type == "exo":
        content = export_exo(segments, settings=exo_settings)
    else:
        content = export_fn(segments)

    # exo は CP932 (Shift_JIS) エンコーディング
    encoding = "cp932" if format_type == "exo" else "utf-8"
    path.write_text(content, encoding=encoding)
    return str(path)
