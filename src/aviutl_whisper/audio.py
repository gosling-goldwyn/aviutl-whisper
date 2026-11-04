"""音声前処理モジュール - m4aファイルのwav変換"""

import tempfile
import math
from functools import lru_cache
from pathlib import Path

from pydub import AudioSegment


SUPPORTED_EXTENSIONS = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".wma", ".aac"}


def validate_audio_file(file_path: str) -> Path:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"未対応の形式です: {path.suffix}\n"
            f"対応形式: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return path


def convert_to_wav(
    file_path: str,
    sample_rate: int = 16000,
    channels: int = 1,
    output_path: str | None = None,
) -> str:
    """音声ファイルをWAV(16kHz, mono)に変換する。

    Args:
        file_path: 入力音声ファイルパス
        sample_rate: サンプリングレート (デフォルト: 16000)
        channels: チャンネル数 (デフォルト: 1=モノラル)
        output_path: 出力先パス (Noneの場合は一時ファイル)

    Returns:
        変換後のWAVファイルパス
    """
    path = validate_audio_file(file_path)

    audio = AudioSegment.from_file(str(path))
    audio = audio.set_frame_rate(sample_rate).set_channels(channels)

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = tmp.name
        tmp.close()

    audio.export(output_path, format="wav")
    return output_path


def get_audio_duration(file_path: str) -> float:
    """音声ファイルの長さ（秒）を取得する。"""
    path = Path(file_path).resolve()
    stat = path.stat()
    return _cached_audio_duration(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=128)
def _cached_audio_duration(file_path: str, mtime_ns: int, size: int) -> float:
    # WAV等はヘッダーだけで取得する。曲の再出力ごとの全デコードを避ける。
    import soundfile as sf

    try:
        duration = float(sf.info(file_path).duration)
    except (RuntimeError, sf.LibsndfileError):
        duration = len(AudioSegment.from_file(file_path)) / 1000.0
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"音声ファイルの長さを取得できません: {file_path}")
    return duration


def create_silent_wav(
    duration_ms: int,
    sample_rate: int = 16000,
    channels: int = 1,
    output_path: str | None = None,
) -> str:
    """指定時間の無音WAVを作成する。"""
    if duration_ms <= 0:
        raise ValueError("無音音声の長さは1ms以上にしてください")

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = tmp.name
        tmp.close()

    silent = AudioSegment.silent(
        duration=duration_ms,
        frame_rate=sample_rate,
    ).set_channels(channels)
    silent.export(output_path, format="wav")
    return output_path
