"""話者分離モジュール - speechbrainによる話者識別"""

import logging

import numpy as np
import soundfile as sf
import torch
from sklearn.cluster import AgglomerativeClustering

from .transcriber import TranscriptionSegment

logger = logging.getLogger(__name__)

# 話者分離時のセグメント最小長（秒）。短すぎる区間は埋め込み精度が低い。
MIN_SEGMENT_DURATION = 0.5


def assign_speakers(
    model,
    audio_path: str,
    segments: list[TranscriptionSegment],
    num_speakers: int | None = None,
    distance_threshold: float = 1.0,
    progress_callback=None,
) -> list[TranscriptionSegment]:
    """文字起こしセグメントに話者ラベルを割り当てる。

    Args:
        model: speechbrainの話者埋め込みモデル
        audio_path: WAVファイルパス (16kHz, mono)
        segments: 文字起こしセグメントのリスト
        num_speakers: 話者数 (Noneの場合は自動推定)
        distance_threshold: クラスタリング距離閾値 (num_speakers=Noneの場合に使用)
        progress_callback: 進捗コールバック

    Returns:
        話者ラベルが割り当てられたセグメントリスト
    """
    if not segments:
        return segments

    if progress_callback:
        progress_callback(0.0, "話者分離開始...")

    # soundfile で読み込み (torchaudio/torchcodec の FFmpeg 依存を回避)
    data, sample_rate = sf.read(audio_path, dtype="float32")

    # mono化
    if data.ndim > 1:
        data = data.mean(axis=1)

    # 16kHz リサンプル (簡易線形補間)
    if sample_rate != 16000:
        import scipy.signal
        num_samples = int(len(data) * 16000 / sample_rate)
        data = scipy.signal.resample(data, num_samples)
        sample_rate = 16000

    # torch tensor に変換 (1, num_samples)
    waveform = torch.from_numpy(data).unsqueeze(0)

    embeddings = _extract_embeddings(model, waveform, sample_rate, segments, progress_callback)

    if len(embeddings) == 0:
        logger.warning("有効な埋め込みが取得できませんでした")
        return segments

    valid_indices, valid_embeddings = zip(*embeddings)
    embedding_matrix = np.vstack(valid_embeddings)

    labels = _cluster_speakers(
        embedding_matrix,
        num_speakers=num_speakers,
        distance_threshold=distance_threshold,
    )

    result_segments = list(segments)
    for idx, label in zip(valid_indices, labels):
        result_segments[idx] = TranscriptionSegment(
            start=segments[idx].start,
            end=segments[idx].end,
            text=segments[idx].text,
            speaker=f"Speaker {label + 1}",
        )

    # 埋め込みが取れなかった短いセグメントには前後の話者を割り当て
    _fill_missing_speakers(result_segments)

    if progress_callback:
        n_speakers = len(set(labels))
        progress_callback(1.0, f"話者分離完了 ({n_speakers}人検出)")

    logger.info("話者分離完了: %d人検出", len(set(labels)))
    return result_segments


def _extract_embeddings(
    model,
    waveform: torch.Tensor,
    sample_rate: int,
    segments: list[TranscriptionSegment],
    progress_callback=None,
) -> list[tuple[int, np.ndarray]]:
    """各セグメントの話者埋め込みを抽出する。"""
    embeddings = []
    total = len(segments)

    for i, seg in enumerate(segments):
        duration = seg.end - seg.start
        if duration < MIN_SEGMENT_DURATION:
            continue

        start_sample = int(seg.start * sample_rate)
        end_sample = int(seg.end * sample_rate)
        segment_audio = waveform[:, start_sample:end_sample]

        if segment_audio.shape[1] == 0:
            continue

        with torch.no_grad():
            embedding = model.encode_batch(segment_audio)
            embeddings.append((i, embedding.squeeze().cpu().numpy()))

        if progress_callback and (i + 1) % 10 == 0:
            progress_callback(
                (i + 1) / total * 0.8,
                f"話者埋め込み抽出中... ({i + 1}/{total})",
            )

    return embeddings


def _cluster_speakers(
    embeddings: np.ndarray,
    num_speakers: int | None = None,
    distance_threshold: float = 1.0,
) -> list[int]:
    """埋め込みベクトルをクラスタリングして話者を分類する。"""
    if len(embeddings) == 1:
        return [0]

    if num_speakers is not None:
        clustering = AgglomerativeClustering(
            n_clusters=num_speakers,
            metric="cosine",
            linkage="average",
        )
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average",
        )

    labels = clustering.fit_predict(embeddings)
    return labels.tolist()


def _fill_missing_speakers(segments: list[TranscriptionSegment]) -> None:
    """話者が割り当てられていないセグメントに前後の話者を伝播する。"""
    # 前方から埋める
    last_speaker = None
    for seg in segments:
        if seg.speaker is not None:
            last_speaker = seg.speaker
        elif last_speaker is not None:
            seg.speaker = last_speaker

    # 後方から埋める（先頭の未割当を処理）
    last_speaker = None
    for seg in reversed(segments):
        if seg.speaker is not None:
            last_speaker = seg.speaker
        elif last_speaker is not None:
            seg.speaker = last_speaker

    # それでも残っていれば Speaker 1 を割り当て
    for seg in segments:
        if seg.speaker is None:
            seg.speaker = "Speaker 1"
