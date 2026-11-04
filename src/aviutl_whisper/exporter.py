"""出力モジュール - 各種フォーマットでの結果出力"""

import csv
import io
import math
import random
from dataclasses import dataclass, field, replace
from fractions import Fraction
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
class SpeakerImageSettings:
    """話者ごとの立ち絵画像設定。"""
    file: str = ""
    x: float = 0.0
    y: float = 0.0
    scale: float = 100.0

    @classmethod
    def from_dict(cls, d: dict) -> "SpeakerImageSettings":
        if not d:
            return cls()
        return cls(
            file=d.get("file", ""),
            x=float(d.get("x", 0.0)),
            y=float(d.get("y", 0.0)),
            scale=float(d.get("scale", 100.0)),
        )


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
    # 話者ごとの文字色 (hex RGB)
    speaker_colors: list[str] = field(default_factory=lambda: list(DEFAULT_SPEAKER_COLORS))
    # 話者ごとの縁色
    speaker_edge_colors: list[str] = field(default_factory=lambda: [])
    # 話者ごとの立ち絵画像設定
    speaker_images: list[SpeakerImageSettings] = field(default_factory=lambda: [])
    # 背景画像パス
    background_image: str = ""
    # 1行の最大文字数 (0で無効)
    max_chars_per_line: int = 20
    audio_start_seconds: float = 1.5
    bgm_start_file: str = ""
    bgm_end_file: str = ""
    bgm_files: list[str] = field(default_factory=list)
    bgm_volume: float = 20.0
    outro_seconds: float = 1.5
    closing_jingle: str = ""
    closing_image: str = ""
    closing_seconds: float = 15.0

    def __post_init__(self):
        for name, minimum, maximum in (
            ("fps", 0.01, 1000),
            ("audio_start_seconds", 0, None),
            ("bgm_volume", 0, 100),
            ("outro_seconds", 0, None),
            ("closing_seconds", 0.01, None),
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < minimum or (
                maximum is not None and value > maximum
            ):
                raise ValueError(f"{name} の値が範囲外です")
        if not isinstance(self.bgm_files, list) or any(
            not isinstance(path, str) or not path.strip() for path in self.bgm_files
        ):
            raise ValueError("BGMの曲リストが不正です")

    def get_speaker_color(self, index: int) -> str:
        """話者インデックスに対応する文字色 (RGB hex) を返す。"""
        colors = self.speaker_colors or DEFAULT_SPEAKER_COLORS
        return colors[index % len(colors)]

    def get_speaker_edge_color(self, index: int) -> str:
        """話者インデックスに対応する縁色 (RGB hex) を返す。"""
        if self.speaker_edge_colors:
            return self.speaker_edge_colors[index % len(self.speaker_edge_colors)]
        return DEFAULT_EDGE_COLOR

    def get_speaker_image(self, index: int) -> SpeakerImageSettings | None:
        """話者インデックスに対応する立ち絵設定を返す。無い場合はNone。"""
        if index < len(self.speaker_images):
            img = self.speaker_images[index]
            if img.file:
                return img
        return None

    @classmethod
    def from_dict(cls, d: dict) -> "ExoSettings":
        """フロントエンドから渡された辞書からExoSettingsを生成する。"""
        if not d:
            return cls()
        speaker_images = [
            SpeakerImageSettings.from_dict(img)
            for img in d.get("speaker_images", [])
        ]
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
            speaker_images=speaker_images,
            background_image=d.get("background_image", ""),
            max_chars_per_line=int(d.get("max_chars_per_line", 20)),
            audio_start_seconds=float(d.get("audio_start_seconds", 1.5)),
            bgm_start_file=d.get("bgm_start_file", ""),
            bgm_end_file=d.get("bgm_end_file", ""),
            bgm_files=d.get("bgm_files", []),
            bgm_volume=float(d.get("bgm_volume", 20)),
            outro_seconds=float(d.get("outro_seconds", 1.5)),
            closing_jingle=d.get("closing_jingle", ""),
            closing_image=d.get("closing_image", ""),
            closing_seconds=float(d.get("closing_seconds", 15)),
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


def _build_speaker_intervals(
    segments: list[TranscriptionSegment],
    speaker: str,
    fps: float,
    total_frames: int,
) -> list[tuple[int, int, bool]]:
    """話者の話中/非話中の区間リストを生成する。

    Returns:
        (start_frame, end_frame, is_talking) のリスト。
        隣接する同種の区間はマージされている。
    """
    # 該当話者のセグメントを時系列順に取得
    speaker_segs = sorted(
        [s for s in segments if (s.speaker or "Speaker 1") == speaker],
        key=lambda s: s.start,
    )

    if not speaker_segs:
        # 話者のセグメントが無い場合は全体が非話中
        return [(1, total_frames, False)]

    # 話中フレーム区間を構築
    talking_frames: list[tuple[int, int]] = []
    for seg in speaker_segs:
        start = max(1, int(seg.start * fps) + 1)
        end = min(total_frames, max(start, _frame_count(seg.end, fps)))
        if start > total_frames:
            continue
        talking_frames.append((start, end))

    # 重複するフレーム区間をマージ
    merged: list[tuple[int, int]] = []
    for start, end in sorted(talking_frames):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    # 話中/非話中の区間に分割
    intervals: list[tuple[int, int, bool]] = []
    pos = 1
    for talk_start, talk_end in merged:
        if pos < talk_start:
            intervals.append((pos, talk_start - 1, False))
        intervals.append((talk_start, talk_end, True))
        pos = talk_end + 1
    if pos <= total_frames:
        intervals.append((pos, total_frames, False))

    return intervals


def _emit_image_objects(
    lines: list[str],
    intervals: list[tuple[int, int, bool]],
    image_settings: SpeakerImageSettings,
    layer: int,
    obj_idx_start: int,
) -> int:
    """画像オブジェクト群をexo行リストに追加する。

    Returns:
        次のオブジェクトインデックス。
    """
    idx = obj_idx_start
    for start_frame, end_frame, is_talking in intervals:
        saturation = 100.0 if is_talking else 0.0

        # [N] オブジェクトヘッダー
        lines.append(f"[{idx}]")
        lines.append(f"start={start_frame}")
        lines.append(f"end={end_frame}")
        lines.append(f"layer={layer}")
        lines.append("overlay=1")
        lines.append("camera=0")

        # [N.0] 画像ファイル
        lines.append(f"[{idx}.0]")
        lines.append("_name=画像ファイル")
        lines.append(f"file={image_settings.file}")

        # [N.1] 標準描画
        lines.append(f"[{idx}.1]")
        lines.append("_name=標準描画")
        lines.append(f"X={image_settings.x:.1f}")
        lines.append(f"Y={image_settings.y:.1f}")
        lines.append("Z=0.0")
        lines.append(f"拡大率={image_settings.scale:.2f}")
        lines.append("透明度=0.0")
        lines.append("回転=0.00")
        lines.append("blend=0")

        # [N.2] 色調補正
        lines.append(f"[{idx}.2]")
        lines.append("_name=色調補正")
        lines.append("明るさ=100.0")
        lines.append("コントラスト=100.0")
        lines.append("色相=0.0")
        lines.append("輝度=100.0")
        lines.append(f"彩度={saturation:.1f}")
        lines.append("飽和する=0")

        idx += 1

    return idx


def _wrap_text(text: str, max_chars: int) -> str:
    """テキストを指定文字数で改行する。max_chars <= 0 の場合はそのまま返す。"""
    if max_chars <= 0:
        return text
    lines = []
    for line in text.split("\n"):
        while len(line) > max_chars:
            lines.append(line[:max_chars])
            line = line[max_chars:]
        lines.append(line)
    return "\n".join(lines)


def _encode_exo_text(text: str) -> str:
    """テキストをAviUtl exo形式のhexエンコードに変換する。

    UTF-16LEでエンコードし、hex文字列にして4096文字に0埋めする。
    """
    hex_str = text.encode("utf-16-le").hex()
    return hex_str.ljust(4096, "0")


def _frame_count(seconds: float, fps: float) -> int:
    """秒数を、末尾を切り捨てないフレーム数にする。"""
    return max(1, math.ceil(seconds * fps - 1e-9))


def _emit_audio_object(lines, idx, file, start, end, layer, volume,
                       fade_out=0.0, loop=False):
    lines.extend([
        f"[{idx}]", f"start={start}", f"end={end}", f"layer={layer}",
        "overlay=1", "audio=1",
        f"[{idx}.0]", "_name=音声ファイル", "再生位置=0.00",
        "再生速度=100.0", f"ループ再生={int(loop)}", "動画ファイルと連携=0",
        f"file={file}",
    ])
    effect = 1
    if fade_out > 0:
        lines.extend([
            f"[{idx}.1]", "_name=音量フェード", "イン=0.00",
            f"アウト={fade_out:.4f}",
        ])
        effect += 1
    lines.extend([
        f"[{idx}.{effect}]", "_name=標準再生", f"音量={volume:.1f}", "左右=0.0",
    ])
    return idx + 1


def _bgm_playlist(settings):
    """開始曲を一度再生し、その後は曲リストをシャッフルして繰り返す。"""
    previous = None
    if settings.bgm_start_file:
        previous = settings.bgm_start_file
        yield previous
    tracks = list(dict.fromkeys(settings.bgm_files))
    if not tracks and settings.bgm_start_file:
        tracks = [settings.bgm_start_file]
    while tracks:
        bag = list(tracks)
        random.shuffle(bag)
        if len(bag) > 1 and bag[0] == previous:
            bag[0], bag[1] = bag[1], bag[0]
        for path in bag:
            previous = path
            yield path


def _emit_bgm(lines, idx, settings, main_frames, scene_frames, layer):
    from .audio import get_audio_duration

    # 終了曲は本編終了の30秒前から。短い本編では先頭から再生する。
    ending_start = max(0, main_frames - round(30 * settings.fps))
    cutoff = ending_start if settings.bgm_end_file else scene_frames
    fade_frames = min(cutoff, round(
        (1.5 if settings.bgm_end_file else
         (settings.outro_seconds if scene_frames > main_frames else 0)) * settings.fps
    ))
    cursor = 0
    for path in _bgm_playlist(settings):
        if cursor >= cutoff:
            break
        track_frames = _frame_count(get_audio_duration(path), settings.fps)
        end = min(cutoff, cursor + track_frames)
        # フェード中は現在の曲を保ち、曲境界で音量が戻らないようにする。
        if end >= cutoff - fade_frames:
            end = cutoff
        fade = fade_frames / settings.fps if end == cutoff else 0
        idx = _emit_audio_object(
            lines, idx, path, cursor + 1, end, layer, settings.bgm_volume,
            fade_out=fade, loop=end - cursor > track_frames,
        )
        cursor = end
    if settings.bgm_end_file:
        duration = get_audio_duration(settings.bgm_end_file)
        fade = (scene_frames - main_frames) / settings.fps
        idx = _emit_audio_object(
            lines, idx, settings.bgm_end_file, ending_start + 1,
            scene_frames, layer, settings.bgm_volume, fade_out=fade,
            loop=_frame_count(duration, settings.fps) < scene_frames - ending_start,
        )
    return idx


def export_exo(
    segments: list[TranscriptionSegment],
    settings: ExoSettings | None = None,
    *,
    audio_file: str = "",
    **kwargs,
) -> str:
    """AviUtl拡張編集のexo形式で出力する。

    各セグメントをテキストオブジェクトとしてタイムライン上に配置する。
    話者ごとにレイヤーを分け、色を変える。
    レイヤー順: 背景画像 → 立ち絵 → 字幕テキスト (連番)
    """
    if not segments:
        return ""

    if settings is None:
        settings = ExoSettings(**kwargs) if kwargs else ExoSettings()

    # 話者リスト
    speakers = sorted(set(s.speaker or "Speaker 1" for s in segments))
    speaker_color = {
        spk: settings.get_speaker_color(i)
        for i, spk in enumerate(speakers)
    }
    speaker_edge = {
        spk: settings.get_speaker_edge_color(i)
        for i, spk in enumerate(speakers)
    }

    # 立ち絵が設定されている話者を把握（発話のない設定スロットも表示）。
    tachie_speakers = []
    for i in range(len(settings.speaker_images)):
        spk = speakers[i] if i < len(speakers) else f"__silent_{i}"
        img = settings.get_speaker_image(i)
        if img is not None:
            tachie_speakers.append((i, spk, img))

    # レイヤー割り当て (連番: 背景→立ち絵→字幕)
    next_layer = 1
    bg_layer = None
    if settings.background_image:
        bg_layer = next_layer
        next_layer += 1

    tachie_layer = {}
    for i, spk, _ in tachie_speakers:
        tachie_layer[spk] = next_layer
        next_layer += 1

    speaker_text_layer = {}
    for spk in speakers:
        speaker_text_layer[spk] = next_layer
        next_layer += 1

    # 本編は入力音声と字幕のうち長い方。EXOだけをオフセットし、元データは保つ。
    max_end = max(s.end for s in segments)
    source_frames = 0
    if audio_file:
        from .audio import get_audio_duration
        source_frames = _frame_count(get_audio_duration(audio_file), settings.fps)
    intro_frames = round(settings.audio_start_seconds * settings.fps)
    body_frames = max(_frame_count(max_end, settings.fps), source_frames)
    main_frames = intro_frames + body_frames
    has_closing = bool(settings.closing_image or settings.closing_jingle)
    outro_frames = round(settings.outro_seconds * settings.fps) if has_closing else 0
    scene_frames = main_frames + outro_frames
    closing_frames = _frame_count(settings.closing_seconds, settings.fps) if has_closing else 0
    total_frames = scene_frames + closing_frames
    fps_ratio = Fraction(str(settings.fps)).limit_denominator(100000)

    lines = []
    obj_idx = 0

    # [exedit] ヘッダー
    lines.append("[exedit]")
    lines.append(f"width={settings.width}")
    lines.append(f"height={settings.height}")
    lines.append(f"rate={fps_ratio.numerator}")
    lines.append(f"scale={fps_ratio.denominator}")
    lines.append(f"length={total_frames}")
    lines.append("audio_rate=44100")
    lines.append("audio_ch=2")

    # 背景画像オブジェクト
    if settings.background_image and bg_layer is not None:
        lines.append(f"[{obj_idx}]")
        lines.append("start=1")
        lines.append(f"end={scene_frames}")
        lines.append(f"layer={bg_layer}")
        lines.append("overlay=1")
        lines.append("camera=0")

        lines.append(f"[{obj_idx}.0]")
        lines.append("_name=画像ファイル")
        lines.append(f"file={settings.background_image}")

        lines.append(f"[{obj_idx}.1]")
        lines.append("_name=標準描画")
        lines.append("X=0.0")
        lines.append("Y=0.0")
        lines.append("Z=0.0")
        lines.append("拡大率=100.00")
        lines.append("透明度=0.0")
        lines.append("回転=0.00")
        lines.append("blend=0")
        obj_idx += 1

    # 立ち絵画像オブジェクト
    for i, spk, img_settings in tachie_speakers:
        layer = tachie_layer[spk]
        body_intervals = _build_speaker_intervals(segments, spk, settings.fps, body_frames)
        intervals = [(a + intro_frames, b + intro_frames, talking)
                     for a, b, talking in body_intervals]
        if intro_frames:
            intervals.insert(0, (1, intro_frames, True))
        if outro_frames:
            intervals.append((main_frames + 1, scene_frames, True))
        obj_idx = _emit_image_objects(lines, intervals, img_settings, layer, obj_idx)

    # テキストオブジェクト
    for seg in segments:
        speaker = seg.speaker or "Speaker 1"
        start_frame = intro_frames + max(1, int(seg.start * settings.fps) + 1)
        end_frame = max(start_frame, intro_frames + _frame_count(seg.end, settings.fps))

        layer = speaker_text_layer[speaker]
        color = speaker_color[speaker]
        edge_color = speaker_edge[speaker]

        wrapped_text = _wrap_text(seg.text, settings.max_chars_per_line)
        hex_text = _encode_exo_text(wrapped_text)

        # [N] オブジェクトヘッダー
        lines.append(f"[{obj_idx}]")
        lines.append(f"start={start_frame}")
        lines.append(f"end={end_frame}")
        lines.append(f"layer={layer}")
        lines.append("overlay=1")
        lines.append("camera=0")

        # [N.0] テキストオブジェクト
        lines.append(f"[{obj_idx}.0]")
        lines.append("_name=テキスト")
        lines.append(f"サイズ={settings.font_size}")
        lines.append(f"表示速度={settings.display_speed:.1f}")
        lines.append("文字毎に個別オブジェクト=0")
        lines.append("移動座標上に表示する=0")
        lines.append("自動スクロール=0")
        lines.append(f"B={1 if settings.bold else 0}")
        lines.append(f"I={1 if settings.italic else 0}")
        lines.append(f"type={3 if settings.soft_edge else 0}")
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
        lines.append(f"[{obj_idx}.1]")
        lines.append("_name=標準描画")
        lines.append(f"X={settings.pos_x:.1f}")
        lines.append(f"Y={settings.pos_y:.1f}")
        lines.append("Z=0.0")
        lines.append("拡大率=100.00")
        lines.append("透明度=0.0")
        lines.append("回転=0.00")
        lines.append("blend=0")
        obj_idx += 1

    # クロージング画像は最前面。背景・立ち絵はここより前に終了させる。
    if settings.closing_image:
        obj_idx = _emit_image_objects(
            lines, [(scene_frames + 1, total_frames, True)],
            SpeakerImageSettings(file=settings.closing_image), next_layer, obj_idx,
        )
        next_layer += 1
    if audio_file:
        obj_idx = _emit_audio_object(
            lines, obj_idx, audio_file, intro_frames + 1,
            intro_frames + source_frames, next_layer, 100,
        )
        next_layer += 1
    obj_idx = _emit_bgm(lines, obj_idx, settings, main_frames, scene_frames, next_layer)
    next_layer += 1
    if settings.closing_jingle:
        from .audio import get_audio_duration
        jingle_frames = _frame_count(get_audio_duration(settings.closing_jingle), settings.fps)
        _emit_audio_object(
            lines, obj_idx, settings.closing_jingle, scene_frames + 1,
            min(total_frames, scene_frames + jingle_frames), next_layer, 100,
        )
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
    audio_file: str = "",
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
        # 圧縮音声は解析用の一時WAVとは別に永続WAVを書き出す。
        prepared = {}

        def prepare_audio(source: str, suffix: str) -> str:
            if not source:
                return ""
            audio_path = Path(source).resolve()
            key = str(audio_path)
            if key in prepared:
                return prepared[key]
            if audio_path.suffix.lower() != ".wav":
                from pydub import AudioSegment

                wav_path = path.with_name(f"{path.stem}_{suffix}.wav").resolve()
                number = 2
                # 既存の素材を、EXOに付随するファイルで上書きしない。
                while wav_path.exists():
                    wav_path = path.with_name(f"{path.stem}_{suffix}_{number}.wav").resolve()
                    number += 1
                AudioSegment.from_file(str(audio_path)).export(str(wav_path), format="wav")
                audio_path = wav_path
            prepared[key] = str(audio_path)
            return prepared[key]

        original = exo_settings or ExoSettings()
        output_settings = replace(
            original,
            bgm_start_file=prepare_audio(original.bgm_start_file, "bgm_start"),
            bgm_end_file=prepare_audio(original.bgm_end_file, "bgm_end"),
            bgm_files=[prepare_audio(file, f"bgm_{i + 1}")
                       for i, file in enumerate(original.bgm_files)],
            closing_jingle=prepare_audio(original.closing_jingle, "jingle"),
        )
        content = export_exo(
            segments, settings=output_settings, audio_file=prepare_audio(audio_file, "audio"),
        )
    else:
        content = export_fn(segments)

    # exo は CP932 (Shift_JIS) エンコーディング
    encoding = "cp932" if format_type == "exo" else "utf-8"
    path.write_text(content, encoding=encoding, newline="")
    return str(path)
