"""音声前処理モジュール - m4aファイルのwav変換"""

import tempfile
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
    audio = AudioSegment.from_file(file_path)
    return len(audio) / 1000.0
