"""設定の永続化モジュール - settings.json の読み書き"""

import json
import logging
import os
import platform
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_SETTINGS = {
    "model_size": "medium",
    "language": "ja",
    "num_speakers": "auto",
    "output_format": "text",
    "exo": {
        "font": "MS UI Gothic",
        "font_size": 34,
        "spacing_x": 0,
        "spacing_y": 0,
        "display_speed": 0.0,
        "align": 4,
        "bold": False,
        "italic": False,
        "soft_edge": True,
        "pos_x": 0.0,
        "pos_y": 0.0,
        "speaker_colors": ["ffffff", "00ffff", "00ff00", "ff00ff",
                           "ffff00", "ff8000", "8080ff", "80ff80"],
        "speaker_edge_colors": [],
        "speaker_images": [],
        "background_image": "",
        "max_chars_per_line": 20,
    },
}


def _get_settings_path() -> Path:
    """設定ファイルのパスを取得する。"""
    if platform.system() == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "aviutl-whisper" / "settings.json"


def load_settings() -> dict:
    """設定ファイルを読み込む。存在しない場合はデフォルト値を返す。"""
    path = _get_settings_path()
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            # デフォルト値でマージ（新しいキーの追加に対応）
            merged = _deep_merge(DEFAULT_SETTINGS, data)
            return merged
    except Exception:
        logger.warning("設定ファイルの読み込みに失敗: %s", path, exc_info=True)
    return dict(DEFAULT_SETTINGS)


def save_settings(settings: dict) -> None:
    """設定ファイルに書き込む。"""
    path = _get_settings_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(settings, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        logger.warning("設定ファイルの保存に失敗: %s", path, exc_info=True)


def _deep_merge(defaults: dict, override: dict) -> dict:
    """デフォルト辞書にオーバーライド辞書をマージする。"""
    result = dict(defaults)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
