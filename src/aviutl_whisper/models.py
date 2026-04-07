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
        device, compute_type = _detect_device(model_size)
    else:
        compute_type = "float16" if device == "cuda" else "int8"

    logger.info("Whisperモデル読み込み: size=%s, device=%s, compute=%s", model_size, device, compute_type)

    try:
        model = WhisperModel(
            WHISPER_MODELS[model_size],
            device=device,
            compute_type=compute_type,
            download_root=str(get_whisper_model_dir()),
        )
    except Exception as e:
        if device == "cuda":
            logger.warning("CUDA読み込み失敗、CPUにフォールバック: %s", e)
            if progress_callback:
                progress_callback(0.0, f"GPU失敗、CPUで再試行中...")
            model = WhisperModel(
                WHISPER_MODELS[model_size],
                device="cpu",
                compute_type="int8",
                download_root=str(get_whisper_model_dir()),
            )
        else:
            raise

    if progress_callback:
        progress_callback(1.0, "Whisperモデル準備完了")

    return model


def _patch_speechbrain_fetch():
    """Windows でシムリンクエラーを回避するため link_with_strategy を直接パッチ。

    fetch() だけでなく parameter_transfer.py の Pretrainer.collect_files() も
    内部で link_with_strategy を呼ぶため、根本のリンク関数をパッチする。
    """
    if platform.system() != "Windows":
        return

    import speechbrain.utils.fetching as sb_fetching
    from speechbrain.utils.fetching import LocalStrategy

    if getattr(sb_fetching, "_patched_for_windows", False):
        return

    original_link = sb_fetching.link_with_strategy

    def _patched_link(src, dst, local_strategy):
        if local_strategy == LocalStrategy.SYMLINK:
            local_strategy = LocalStrategy.COPY
        return original_link(src, dst, local_strategy)

    sb_fetching.link_with_strategy = _patched_link
    sb_fetching._patched_for_windows = True

    # 既存のシムリンクをキャッシュから削除（次回COPYで再作成される）
    sb_dir = get_speechbrain_model_dir()
    for f in sb_dir.iterdir():
        if f.is_symlink():
            logger.info("シムリンク削除: %s", f.name)
            f.unlink()


def load_speechbrain_model(progress_callback: ProgressCallback | None = None):
    """speechbrainの話者埋め込みモデルを読み込む。"""
    from speechbrain.inference.speaker import EncoderClassifier

    _patch_speechbrain_fetch()

    if progress_callback:
        progress_callback(0.0, "話者分離モデルを準備中...")

    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(get_speechbrain_model_dir()),
    )

    if progress_callback:
        progress_callback(1.0, "話者分離モデル準備完了")

    return model


def _detect_device(model_size: str = "medium") -> tuple[str, str]:
    """GPU/CPUを自動検出する。ctranslate2 → torch の順でCUDAを確認。

    大きなモデル (large-v3) は int8_float16 を使いVRAMを節約する。
    """
    is_large = model_size.startswith("large")

    # ctranslate2 (faster-whisper のバックエンド) のCUDAサポートを優先確認
    try:
        import ctranslate2
        cuda_types = ctranslate2.get_supported_compute_types("cuda")
        if cuda_types:
            if is_large and "int8_float16" in cuda_types:
                compute = "int8_float16"
            elif "float16" in cuda_types:
                compute = "float16"
            else:
                compute = "int8"
            logger.info("CUDA GPU検出 (ctranslate2): compute=%s (model=%s)", compute, model_size)
            return "cuda", compute
    except Exception:
        pass

    # フォールバック: torch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            compute = "int8_float16" if is_large else "float16"
            logger.info("CUDA GPU検出 (torch): %s", torch.cuda.get_device_name(0))
            return "cuda", compute
    except ImportError:
        pass

    logger.info("CPUモードで実行")
    return "cpu", "int8"
