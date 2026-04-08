"""話者分離モジュール - speechbrainによる話者識別"""

import logging

import numpy as np
import soundfile as sf
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

from .transcriber import TranscriptionSegment

logger = logging.getLogger(__name__)

# 話者分離時のセグメント最小長（秒）。短すぎる区間は埋め込み精度が低い。
MIN_SEGMENT_DURATION = 0.5

# 埋め込み抽出時のサブウィンドウ最大長（秒）。これより長いセグメントは分割する。
MAX_SEGMENT_FOR_EMBED = 5.0

# サブウィンドウの長さ（秒）
SUB_WINDOW_DURATION = 2.5

# デフォルトのクラスタリング距離閾値（コサイン距離）
# speechbrain ECAPA-TDNN の場合、同一話者間距離は ~0.1-0.3、異話者間は ~0.7-1.0
DEFAULT_DISTANCE_THRESHOLD = 0.45


def assign_speakers(
    model,
    audio_path: str,
    segments: list[TranscriptionSegment],
    num_speakers: int | None = None,
    distance_threshold: float | None = None,
    progress_callback=None,
) -> list[TranscriptionSegment]:
    """文字起こしセグメントに話者ラベルを割り当てる。

    Args:
        model: speechbrainの話者埋め込みモデル
        audio_path: WAVファイルパス (16kHz, mono)
        segments: 文字起こしセグメントのリスト
        num_speakers: 話者数 (Noneの場合は自動推定)
        distance_threshold: クラスタリング距離閾値 (Noneの場合は自動推定)
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

    # 16kHz リサンプル
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

    # 埋め込みからセグメントインデックスとベクトルを取り出す
    sub_indices = [e[0] for e in embeddings]
    embedding_matrix = np.vstack([e[1] for e in embeddings])

    # L2正規化でコサイン距離の精度を向上
    embedding_matrix = normalize(embedding_matrix, norm="l2")

    # 閾値の自動推定 or デフォルト
    if distance_threshold is None and num_speakers is None:
        distance_threshold = _estimate_threshold(embedding_matrix)
        logger.info("自動推定閾値: %.4f", distance_threshold)

    labels = _cluster_speakers(
        embedding_matrix,
        num_speakers=num_speakers,
        distance_threshold=distance_threshold or DEFAULT_DISTANCE_THRESHOLD,
    )

    # サブセグメント→元セグメントの話者ラベル割り当て (多数決)
    from collections import Counter
    seg_label_votes: dict[int, list[int]] = {}
    for sub_idx, label in zip(sub_indices, labels):
        seg_label_votes.setdefault(sub_idx, []).append(label)

    result_segments = list(segments)
    for seg_idx, votes in seg_label_votes.items():
        most_common = Counter(votes).most_common(1)[0][0]
        result_segments[seg_idx] = TranscriptionSegment(
            start=segments[seg_idx].start,
            end=segments[seg_idx].end,
            text=segments[seg_idx].text,
            speaker=f"Speaker {most_common + 1}",
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
    """各セグメントの話者埋め込みを抽出する。

    長いセグメント (> MAX_SEGMENT_FOR_EMBED秒) はサブウィンドウに分割して
    個別に埋め込みを抽出する。これにより、複数話者が含まれる長セグメントでの
    クラスタリング精度が向上する。
    """
    embeddings = []
    total = len(segments)

    for i, seg in enumerate(segments):
        duration = seg.end - seg.start
        if duration < MIN_SEGMENT_DURATION:
            continue

        # 長いセグメントはサブウィンドウに分割
        if duration > MAX_SEGMENT_FOR_EMBED:
            windows = _split_into_windows(seg.start, seg.end, SUB_WINDOW_DURATION)
        else:
            windows = [(seg.start, seg.end)]

        for win_start, win_end in windows:
            start_sample = int(win_start * sample_rate)
            end_sample = int(win_end * sample_rate)
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


def _split_into_windows(
    start: float, end: float, window_duration: float
) -> list[tuple[float, float]]:
    """時間区間をサブウィンドウに分割する。"""
    windows = []
    t = start
    while t < end:
        win_end = min(t + window_duration, end)
        if win_end - t >= MIN_SEGMENT_DURATION:
            windows.append((t, win_end))
        t += window_duration
    return windows


def _cluster_speakers(
    embeddings: np.ndarray,
    num_speakers: int | None = None,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
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


def _estimate_threshold(embeddings: np.ndarray) -> float:
    """ペアワイズ距離の分布から最適な閾値を自動推定する。

    距離のヒストグラムで最大のギャップ（谷）を探し、
    同一話者クラスタと異話者クラスタの境界を見つける。
    """
    from sklearn.metrics.pairwise import cosine_distances

    if len(embeddings) < 3:
        return DEFAULT_DISTANCE_THRESHOLD

    dist_matrix = cosine_distances(embeddings)
    # 上三角の距離値を取得
    triu_indices = np.triu_indices_from(dist_matrix, k=1)
    distances = dist_matrix[triu_indices]

    if len(distances) < 2:
        return DEFAULT_DISTANCE_THRESHOLD

    # ソートして隣接差分の最大ギャップを探す
    sorted_dists = np.sort(distances)
    gaps = np.diff(sorted_dists)

    if len(gaps) == 0:
        return DEFAULT_DISTANCE_THRESHOLD

    max_gap_idx = np.argmax(gaps)
    threshold = (sorted_dists[max_gap_idx] + sorted_dists[max_gap_idx + 1]) / 2

    # 妥当な範囲にクランプ (0.2 ~ 0.8)
    threshold = np.clip(threshold, 0.2, 0.8)

    return float(threshold)


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


def assign_speakers_pyannote(
    pipeline,
    audio_path: str,
    segments: list[TranscriptionSegment],
    num_speakers: int | None = None,
    progress_callback=None,
) -> list[TranscriptionSegment]:
    """pyannote.audioパイプラインを使って話者ラベルを割り当てる。

    Args:
        pipeline: pyannote.audio Pipeline (事前ロード済み)
        audio_path: WAVファイルパス
        segments: 文字起こしセグメントのリスト
        num_speakers: 話者数 (Noneの場合はpyannoteが自動推定)
        progress_callback: 進捗コールバック

    Returns:
        話者ラベルが割り当てられたセグメントリスト
    """
    if not segments:
        return segments

    if progress_callback:
        progress_callback(0.0, "pyannote話者分離開始...")

    # pyannoteでダイアライゼーション実行
    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers

    logger.info("pyannote diarization開始: audio=%s, num_speakers=%s", audio_path, num_speakers)
    diarization = pipeline(audio_path, **kwargs)

    if progress_callback:
        progress_callback(0.5, "話者ラベル割り当て中...")

    # pyannote結果からタイムラインを取得
    # pyannote 4.x: DiarizeOutput dataclass → .speaker_diarization が Annotation
    # pyannote 3.x: 直接 Annotation を返す
    annotation = getattr(diarization, "speaker_diarization", diarization)

    pyannote_segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        pyannote_segments.append((turn.start, turn.end, speaker))

    # pyannoteの話者ラベルを正規化 (SPEAKER_00 → Speaker 1)
    unique_speakers = sorted(set(s[2] for s in pyannote_segments))
    speaker_map = {name: f"Speaker {i + 1}" for i, name in enumerate(unique_speakers)}

    # whisperセグメントとpyannoteセグメントの重複マッチング
    result_segments = _match_speakers_by_overlap(segments, pyannote_segments, speaker_map)

    # 未割当のセグメントを前後から埋める
    _fill_missing_speakers(result_segments)

    if progress_callback:
        n_speakers = len(unique_speakers)
        progress_callback(1.0, f"話者分離完了 ({n_speakers}人検出)")

    logger.info("pyannote話者分離完了: %d人検出", len(unique_speakers))
    return result_segments


def _match_speakers_by_overlap(
    segments: list[TranscriptionSegment],
    pyannote_segments: list[tuple[float, float, str]],
    speaker_map: dict[str, str],
) -> list[TranscriptionSegment]:
    """whisperセグメントとpyannoteセグメントを時間重複でマッチングする。

    各whisperセグメントに対して、最も重複時間が長いpyannoteセグメントの話者を割り当てる。
    """
    result = []
    for seg in segments:
        best_speaker = None
        best_overlap = 0.0

        for pa_start, pa_end, pa_speaker in pyannote_segments:
            # 重複区間の計算
            overlap_start = max(seg.start, pa_start)
            overlap_end = min(seg.end, pa_end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker_map.get(pa_speaker, pa_speaker)

        result.append(TranscriptionSegment(
            start=seg.start,
            end=seg.end,
            text=seg.text,
            speaker=best_speaker,
        ))

    return result
