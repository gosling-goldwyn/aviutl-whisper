"""モデル管理モジュール - ダウンロードとキャッシュ"""

import logging
import os
import platform
from pathlib import Path
from typing import Callable

# Windows シムリンク警告を抑制
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def get_cache_dir() -> Path:
    """モデルキャッシュディレクトリを取得する。"""
    if platform.system() == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    cache_dir = base / "aviutl-whisper"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_whisper_model_dir() -> Path:
    """faster-whisperモデルのキャッシュディレクトリ。"""
    d = get_cache_dir() / "whisper"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_speechbrain_model_dir() -> Path:
    """speechbrainモデルのキャッシュディレクトリ。"""
    d = get_cache_dir() / "speechbrain"
    d.mkdir(parents=True, exist_ok=True)
    return d


WHISPER_MODELS = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large-v3": "large-v3",
}

WHISPER_MODEL_SIZES = {
    "tiny": "~75 MB",
    "base": "~140 MB",
    "small": "~460 MB",
    "medium": "~1.5 GB",
    "large-v3": "~3.0 GB",
}


def load_whisper_model(
    model_size: str = "medium",
    device: str = "auto",
    progress_callback: ProgressCallback | None = None,
):
    """faster-whisperモデルを読み込む。

    初回はダウンロードが行われる。
    """
    from faster_whisper import WhisperModel

    if model_size not in WHISPER_MODELS:
        raise ValueError(
            f"未対応のモデルサイズ: {model_size}\n"
            f"対応モデル: {', '.join(WHISPER_MODELS.keys())}"
        )

    if progress_callback:
        progress_callback(0.0, f"Whisperモデル({model_size})を準備中...")

    if device == "auto":
        device, compute_type = _detect_device()
    else:
        compute_type = "float16" if device == "cuda" else "int8"

    logger.info("Whisperモデル読み込み: size=%s, device=%s, compute=%s", model_size, device, compute_type)

    model = WhisperModel(
        WHISPER_MODELS[model_size],
        device=device,
        compute_type=compute_type,
        download_root=str(get_whisper_model_dir()),
    )

    if progress_callback:
        progress_callback(1.0, "Whisperモデル準備完了")

    return model


def load_speechbrain_model(progress_callback: ProgressCallback | None = None):
    """speechbrainの話者埋め込みモデルを読み込む。"""
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import FetchConfig, LocalStrategy

    if progress_callback:
        progress_callback(0.0, "話者分離モデルを準備中...")

    # Windows ではシムリンクに管理者権限が必要なためコピー戦略を使用
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(get_speechbrain_model_dir()),
        fetch_config=FetchConfig(local_strategy=LocalStrategy.COPY),
    )

    if progress_callback:
        progress_callback(1.0, "話者分離モデル準備完了")

    return model


def _detect_device() -> tuple[str, str]:
    """GPU/CPUを自動検出する。"""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("CUDA GPU検出: %s", torch.cuda.get_device_name(0))
            return "cuda", "float16"
    except ImportError:
        pass
    logger.info("CPUモードで実行")
    return "cpu", "int8"
