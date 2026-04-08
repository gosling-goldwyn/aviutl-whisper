"""aviutl-whisper バックエンド単体テスト

各モジュールの基本機能を検証する。
音声ファイルを使うテストはダミーWAVを生成して実行する。
"""

import os
import struct
import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest


# --- ヘルパー ---

def create_dummy_wav(path: str, duration: float = 3.0, sample_rate: int = 16000, num_channels: int = 1):
    """テスト用のダミーWAVファイルを生成する。サイン波を書き込む。"""
    n_samples = int(duration * sample_rate)
    # 440Hz のサイン波
    t = np.linspace(0, duration, n_samples, endpoint=False)
    samples = (np.sin(2 * np.pi * 440 * t) * 32767 * 0.5).astype(np.int16)

    with wave.open(path, "w") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())

    return path


@pytest.fixture
def dummy_wav(tmp_path):
    """3秒のダミーWAVファイルを返すフィクスチャ。"""
    path = str(tmp_path / "test.wav")
    create_dummy_wav(path, duration=3.0)
    return path


@pytest.fixture
def long_dummy_wav(tmp_path):
    """10秒のダミーWAVファイルを返すフィクスチャ（話者分離テスト用）。"""
    path = str(tmp_path / "test_long.wav")
    create_dummy_wav(path, duration=10.0)
    return path


# ============================================================
# models.py テスト
# ============================================================

class TestModels:
    """models モジュールのテスト。"""

    def test_cache_dir(self):
        """キャッシュディレクトリが作成される。"""
        from aviutl_whisper.models import get_cache_dir
        d = get_cache_dir()
        assert d.exists()
        assert d.is_dir()

    def test_whisper_model_dir(self):
        """Whisperモデルディレクトリが作成される。"""
        from aviutl_whisper.models import get_whisper_model_dir
        d = get_whisper_model_dir()
        assert d.exists()
        assert "whisper" in str(d)

    def test_speechbrain_model_dir(self):
        """speechbrainモデルディレクトリが作成される。"""
        from aviutl_whisper.models import get_speechbrain_model_dir
        d = get_speechbrain_model_dir()
        assert d.exists()
        assert "speechbrain" in str(d)

    def test_detect_device(self):
        """デバイス検出が有効な値を返す。"""
        from aviutl_whisper.models import _detect_device
        device, compute_type = _detect_device()
        assert device in ("cuda", "cpu")
        assert compute_type in ("float16", "int8", "int8_float16", "float32")

    def test_whisper_model_sizes(self):
        """モデルサイズ定義が正しい。"""
        from aviutl_whisper.models import WHISPER_MODELS, WHISPER_MODEL_SIZES
        assert set(WHISPER_MODELS.keys()) == set(WHISPER_MODEL_SIZES.keys())
        assert "medium" in WHISPER_MODELS

    def test_invalid_model_size(self):
        """不正なモデルサイズでValueErrorが発生する。"""
        from aviutl_whisper.models import load_whisper_model
        with pytest.raises(ValueError, match="未対応のモデルサイズ"):
            load_whisper_model(model_size="nonexistent")

    def test_patch_speechbrain_fetch(self):
        """speechbrainのfetchパッチが適用される。"""
        from aviutl_whisper.models import _patch_speechbrain_fetch
        import speechbrain.utils.fetching as sb_fetching

        # パッチ前の状態をリセット
        if hasattr(sb_fetching, "_patched_for_windows"):
            delattr(sb_fetching, "_patched_for_windows")

        _patch_speechbrain_fetch()

        import platform
        if platform.system() == "Windows":
            assert getattr(sb_fetching, "_patched_for_windows", False)


# ============================================================
# audio.py テスト
# ============================================================

class TestAudio:
    """audio モジュールのテスト。"""

    def test_convert_wav_to_wav(self, dummy_wav):
        """WAVファイルをWAVに変換（パススルーではなく16kHz mono化）。"""
        from aviutl_whisper.audio import convert_to_wav
        result = convert_to_wav(dummy_wav)
        assert os.path.exists(result)
        # 変換結果のWAVを検証
        with wave.open(result, "r") as wf:
            assert wf.getframerate() == 16000
            assert wf.getnchannels() == 1
        # 一時ファイル削除
        if result != dummy_wav:
            os.unlink(result)

    def test_convert_nonexistent_file(self):
        """存在しないファイルでエラーが出る。"""
        from aviutl_whisper.audio import convert_to_wav
        with pytest.raises(Exception):
            convert_to_wav("/nonexistent/file.m4a")


# ============================================================
# transcriber.py テスト
# ============================================================

class TestTranscriber:
    """transcriber モジュールのテスト。"""

    def test_transcription_segment_dataclass(self):
        """TranscriptionSegmentが正しく作成される。"""
        from aviutl_whisper.transcriber import TranscriptionSegment
        seg = TranscriptionSegment(start=0.0, end=1.5, text="テスト")
        assert seg.start == 0.0
        assert seg.end == 1.5
        assert seg.text == "テスト"
        assert seg.speaker is None

    def test_transcription_segment_with_speaker(self):
        """話者付きセグメントが作成される。"""
        from aviutl_whisper.transcriber import TranscriptionSegment
        seg = TranscriptionSegment(start=0.0, end=1.5, text="テスト", speaker="Speaker 1")
        assert seg.speaker == "Speaker 1"

    def test_transcription_result_dataclass(self):
        """TranscriptionResultが正しく作成される。"""
        from aviutl_whisper.transcriber import TranscriptionResult, TranscriptionSegment
        segments = [TranscriptionSegment(start=0.0, end=1.0, text="hello")]
        result = TranscriptionResult(segments=segments, language="ja", language_probability=0.99)
        assert result.language == "ja"
        assert len(result.segments) == 1


# ============================================================
# diarizer.py テスト
# ============================================================

class TestDiarizer:
    """diarizer モジュールのテスト。"""

    def test_audio_loading_with_soundfile(self, dummy_wav):
        """soundfileでWAVファイルが正常に読み込める。"""
        import soundfile as sf
        import torch

        data, sr = sf.read(dummy_wav, dtype="float32")
        assert sr == 16000
        assert len(data) > 0

        # torch tensor に変換
        waveform = torch.from_numpy(data).unsqueeze(0)
        assert waveform.shape[0] == 1
        assert waveform.shape[1] > 0

    def test_empty_segments(self, dummy_wav):
        """空セグメントリストで話者分離が正常に返る。"""
        from aviutl_whisper.diarizer import assign_speakers
        result = assign_speakers(model=None, audio_path=dummy_wav, segments=[])
        assert result == []

    def test_cluster_speakers_single(self):
        """1セグメントのクラスタリングは話者0を返す。"""
        from aviutl_whisper.diarizer import _cluster_speakers
        embeddings = np.random.randn(1, 192)
        labels = _cluster_speakers(embeddings)
        assert labels == [0]

    def test_cluster_speakers_multiple(self):
        """複数セグメントのクラスタリングが動作する。"""
        from aviutl_whisper.diarizer import _cluster_speakers
        # 2つの明確に異なるクラスタ
        cluster1 = np.random.randn(3, 192) + 5.0
        cluster2 = np.random.randn(3, 192) - 5.0
        embeddings = np.vstack([cluster1, cluster2])
        labels = _cluster_speakers(embeddings, num_speakers=2)
        assert len(labels) == 6
        assert len(set(labels)) == 2

    def test_cluster_with_auto_threshold(self):
        """自動閾値推定で2話者が分離される。"""
        from aviutl_whisper.diarizer import _cluster_speakers, _estimate_threshold
        from sklearn.preprocessing import normalize
        # 明確に分離できる2クラスタを生成
        np.random.seed(42)
        cluster1 = np.random.randn(5, 192) + 3.0
        cluster2 = np.random.randn(5, 192) - 3.0
        embeddings = normalize(np.vstack([cluster1, cluster2]), norm="l2")
        threshold = _estimate_threshold(embeddings)
        labels = _cluster_speakers(embeddings, distance_threshold=threshold)
        assert len(set(labels)) == 2, f"Expected 2 speakers, got {len(set(labels))} (threshold={threshold:.4f})"

    def test_estimate_threshold(self):
        """閾値推定が妥当な範囲の値を返す。"""
        from aviutl_whisper.diarizer import _estimate_threshold
        from sklearn.preprocessing import normalize
        np.random.seed(42)
        cluster1 = np.random.randn(3, 192) + 3.0
        cluster2 = np.random.randn(3, 192) - 3.0
        embeddings = normalize(np.vstack([cluster1, cluster2]), norm="l2")
        threshold = _estimate_threshold(embeddings)
        assert 0.2 <= threshold <= 0.8, f"Threshold {threshold} out of range"

    def test_fill_missing_speakers(self):
        """未割当セグメントに話者が伝播される。"""
        from aviutl_whisper.diarizer import _fill_missing_speakers
        from aviutl_whisper.transcriber import TranscriptionSegment

        segments = [
            TranscriptionSegment(start=0.0, end=1.0, text="a", speaker="Speaker 1"),
            TranscriptionSegment(start=1.0, end=2.0, text="b", speaker=None),
            TranscriptionSegment(start=2.0, end=3.0, text="c", speaker="Speaker 2"),
        ]
        _fill_missing_speakers(segments)
        assert segments[1].speaker == "Speaker 1"  # 前方伝播

    def test_split_into_windows(self):
        """_split_into_windowsが正しくサブウィンドウに分割する。"""
        from aviutl_whisper.diarizer import _split_into_windows
        windows = _split_into_windows(0.0, 10.0, 2.5)
        assert len(windows) == 4
        assert windows[0] == (0.0, 2.5)
        assert windows[1] == (2.5, 5.0)
        assert windows[2] == (5.0, 7.5)
        assert windows[3] == (7.5, 10.0)

    def test_split_into_windows_short(self):
        """短いセグメントは分割されない。"""
        from aviutl_whisper.diarizer import _split_into_windows
        windows = _split_into_windows(0.0, 2.0, 2.5)
        assert len(windows) == 1
        assert windows[0] == (0.0, 2.0)

    def test_split_into_windows_remainder(self):
        """端数のウィンドウがMIN_SEGMENT_DURATION未満なら除外される。"""
        from aviutl_whisper.diarizer import _split_into_windows, MIN_SEGMENT_DURATION
        # 5.3秒を2.5秒ウィンドウで分割: [0-2.5], [2.5-5.0], [5.0-5.3]
        # 残り0.3秒 < MIN_SEGMENT_DURATION(0.5) → 除外
        windows = _split_into_windows(0.0, 5.3, 2.5)
        assert len(windows) == 2
        assert windows[-1] == (2.5, 5.0)

    def test_match_speakers_by_overlap(self):
        """pyannoteセグメントとの重複マッチングが正しく動作する。"""
        from aviutl_whisper.diarizer import _match_speakers_by_overlap
        from aviutl_whisper.transcriber import TranscriptionSegment

        segments = [
            TranscriptionSegment(start=0.0, end=2.0, text="aaa"),
            TranscriptionSegment(start=2.0, end=4.0, text="bbb"),
            TranscriptionSegment(start=4.0, end=6.0, text="ccc"),
        ]
        # pyannote: speaker A talks 0-3s, speaker B talks 3-6s
        pyannote_segs = [
            (0.0, 3.0, "SPEAKER_00"),
            (3.0, 6.0, "SPEAKER_01"),
        ]
        speaker_map = {"SPEAKER_00": "Speaker 1", "SPEAKER_01": "Speaker 2"}
        result = _match_speakers_by_overlap(segments, pyannote_segs, speaker_map)
        assert result[0].speaker == "Speaker 1"  # 0-2s: fully in SPEAKER_00
        assert result[1].speaker == "Speaker 1"  # 2-4s: 1s overlap with A, 1s with B → tie → first wins
        assert result[2].speaker == "Speaker 2"  # 4-6s: fully in SPEAKER_01

    def test_match_speakers_by_overlap_no_pyannote(self):
        """pyannoteセグメントが空の場合、話者はNoneになる。"""
        from aviutl_whisper.diarizer import _match_speakers_by_overlap
        from aviutl_whisper.transcriber import TranscriptionSegment

        segments = [
            TranscriptionSegment(start=0.0, end=2.0, text="test"),
        ]
        result = _match_speakers_by_overlap(segments, [], {})
        assert result[0].speaker is None

    def test_assign_speakers_pyannote_empty(self):
        """空セグメントリストでpyannote話者分離が正常に返る。"""
        from aviutl_whisper.diarizer import assign_speakers_pyannote
        result = assign_speakers_pyannote(pipeline=None, audio_path="", segments=[])
        assert result == []

    def test_match_speakers_overlap_partial(self):
        """部分的な重複でも正しい話者が割り当てられる。"""
        from aviutl_whisper.diarizer import _match_speakers_by_overlap
        from aviutl_whisper.transcriber import TranscriptionSegment

        segments = [
            TranscriptionSegment(start=1.0, end=3.0, text="test"),
        ]
        # SPEAKER_00: 0-1.5s (0.5s overlap), SPEAKER_01: 1.5-4.0s (1.5s overlap)
        pyannote_segs = [
            (0.0, 1.5, "SPEAKER_00"),
            (1.5, 4.0, "SPEAKER_01"),
        ]
        speaker_map = {"SPEAKER_00": "Speaker 1", "SPEAKER_01": "Speaker 2"}
        result = _match_speakers_by_overlap(segments, pyannote_segs, speaker_map)
        assert result[0].speaker == "Speaker 2"  # more overlap with SPEAKER_01


# ============================================================
# exporter.py テスト
# ============================================================

class TestExporter:
    """exporter モジュールのテスト。"""

    @pytest.fixture
    def sample_segments(self):
        from aviutl_whisper.transcriber import TranscriptionSegment
        return [
            TranscriptionSegment(start=0.0, end=2.5, text="こんにちは", speaker="Speaker 1"),
            TranscriptionSegment(start=3.0, end=6.0, text="お元気ですか", speaker="Speaker 2"),
        ]

    def test_srt_export(self, sample_segments):
        """SRT形式が正しく出力される。"""
        from aviutl_whisper.exporter import export_srt
        text = export_srt(sample_segments)
        assert "1\n" in text
        assert "00:00:00,000 --> 00:00:02,500" in text
        assert "[Speaker 1]" in text
        assert "こんにちは" in text

    def test_csv_export(self, sample_segments):
        """CSV形式が正しく出力される。"""
        from aviutl_whisper.exporter import export_csv
        text = export_csv(sample_segments)
        lines = text.strip().splitlines()
        assert lines[0] == "start,end,speaker,text"
        assert len(lines) == 3  # header + 2 segments

    def test_tsv_export(self, sample_segments):
        """TSV形式が正しく出力される。"""
        from aviutl_whisper.exporter import export_tsv
        text = export_tsv(sample_segments)
        lines = text.strip().split("\n")
        assert "\t" in lines[1]

    def test_text_export(self, sample_segments):
        """テキスト形式が正しく出力される。"""
        from aviutl_whisper.exporter import export_text
        text = export_text(sample_segments)
        assert "Speaker 1:" in text
        assert "こんにちは" in text

    def test_exporters_dict(self):
        """EXPORTERS辞書に全形式が登録されている。"""
        from aviutl_whisper.exporter import EXPORTERS
        assert "srt" in EXPORTERS
        assert "csv" in EXPORTERS
        assert "tsv" in EXPORTERS
        assert "text" in EXPORTERS
        assert "exo" in EXPORTERS

    def test_exo_export(self, sample_segments):
        """AviUtl exo形式が正しく出力される。"""
        from aviutl_whisper.exporter import export_exo
        text = export_exo(sample_segments)
        assert "[exedit]" in text
        assert "width=1920" in text
        assert "height=1080" in text
        assert "[0]" in text
        assert "[0.0]" in text
        assert "_name=テキスト" in text
        assert "[1]" in text  # 2つ目のセグメント

    def test_exo_hex_encoding(self):
        """exoのテキストhexエンコードが正しい。"""
        from aviutl_whisper.exporter import _encode_exo_text
        result = _encode_exo_text("AB")
        # "AB" → UTF-16LE: 0x41 0x00 0x42 0x00
        assert result.startswith("4100420")
        assert len(result) == 4096

    def test_exo_speaker_layers(self, sample_segments):
        """話者ごとに異なるレイヤーが割り当てられる。"""
        from aviutl_whisper.exporter import export_exo
        text = export_exo(sample_segments)
        # Speaker 1 → layer=1, Speaker 2 → layer=2
        assert "layer=1" in text
        assert "layer=2" in text

    def test_exo_export_to_file(self, sample_segments, tmp_path):
        """exoファイルがCP932で保存される。"""
        from aviutl_whisper.exporter import export_to_file
        path = str(tmp_path / "output.exo")
        result = export_to_file(sample_segments, path, "exo")
        assert os.path.exists(result)
        # CP932で読み込めることを確認
        content = open(result, encoding="cp932").read()
        assert "[exedit]" in content

    def test_exo_settings_dataclass(self):
        """ExoSettingsのデフォルト値が正しい。"""
        from aviutl_whisper.exporter import ExoSettings
        s = ExoSettings()
        assert s.font == "MS UI Gothic"
        assert s.font_size == 34
        assert s.spacing_x == 0
        assert s.spacing_y == 0
        assert s.display_speed == 0.0
        assert s.align == 4
        assert s.bold is False
        assert s.italic is False
        assert s.soft_edge is True

    def test_exo_settings_from_dict(self):
        """ExoSettings.from_dict()がフロントエンドの辞書を正しくパースする。"""
        from aviutl_whisper.exporter import ExoSettings
        d = {
            "font": "Meiryo",
            "font_size": 48,
            "spacing_x": 5,
            "spacing_y": 10,
            "display_speed": 1.5,
            "align": 7,
            "bold": True,
            "italic": True,
            "soft_edge": False,
            "speaker_colors": ["ff0000", "0000ff"],
            "speaker_edge_colors": ["111111", "222222"],
        }
        s = ExoSettings.from_dict(d)
        assert s.font == "Meiryo"
        assert s.font_size == 48
        assert s.spacing_x == 5
        assert s.spacing_y == 10
        assert s.display_speed == 1.5
        assert s.align == 7
        assert s.bold is True
        assert s.italic is True
        assert s.soft_edge is False
        assert s.speaker_colors == ["ff0000", "0000ff"]
        assert s.speaker_edge_colors == ["111111", "222222"]

    def test_exo_settings_from_empty_dict(self):
        """ExoSettings.from_dict(None)はデフォルト値を返す。"""
        from aviutl_whisper.exporter import ExoSettings
        s = ExoSettings.from_dict(None)
        assert s.font_size == 34

    def test_exo_settings_with_speaker_images(self):
        """ExoSettings.from_dict()がspeaker_imagesを正しくパースする。"""
        from aviutl_whisper.exporter import ExoSettings, SpeakerImageSettings
        d = {
            "speaker_images": [
                {"file": "C:\\img\\speaker1.png", "x": 100.0, "y": -50.0, "scale": 80.0},
                {"file": "C:\\img\\speaker2.png", "x": -100.0, "y": -50.0, "scale": 120.0},
            ],
        }
        s = ExoSettings.from_dict(d)
        assert len(s.speaker_images) == 2
        assert s.speaker_images[0].file == "C:\\img\\speaker1.png"
        assert s.speaker_images[0].x == 100.0
        assert s.speaker_images[0].y == -50.0
        assert s.speaker_images[0].scale == 80.0
        assert s.speaker_images[1].file == "C:\\img\\speaker2.png"

    def test_exo_get_speaker_image(self):
        """get_speaker_image()が正しい設定を返す。"""
        from aviutl_whisper.exporter import ExoSettings, SpeakerImageSettings
        s = ExoSettings(speaker_images=[
            SpeakerImageSettings(file="a.png", x=10, y=20, scale=50),
            SpeakerImageSettings(file="", x=0, y=0, scale=100),
        ])
        img0 = s.get_speaker_image(0)
        assert img0 is not None
        assert img0.file == "a.png"
        # file が空の場合はNone
        assert s.get_speaker_image(1) is None
        # 範囲外もNone
        assert s.get_speaker_image(5) is None

    def test_build_speaker_intervals_basic(self, sample_segments):
        """_build_speaker_intervals が話中/非話中を正しく分割する。"""
        from aviutl_whisper.exporter import _build_speaker_intervals
        fps = 30.0
        total_frames = int(5.0 * fps) + 1  # 5秒分
        intervals = _build_speaker_intervals(sample_segments, "Speaker 1", fps, total_frames)
        # 少なくとも1つの話中区間がある
        talking = [iv for iv in intervals if iv[2] is True]
        assert len(talking) >= 1
        # 全区間がタイムライン全体をカバーする
        assert intervals[0][0] == 1
        assert intervals[-1][1] == total_frames

    def test_build_speaker_intervals_no_segments(self):
        """話者のセグメントが無い場合は全体が非話中。"""
        from aviutl_whisper.exporter import _build_speaker_intervals
        from aviutl_whisper.transcriber import TranscriptionSegment
        segs = [TranscriptionSegment(start=1.0, end=2.0, text="hello", speaker="Speaker 1")]
        intervals = _build_speaker_intervals(segs, "Speaker 2", 30.0, 100)
        assert len(intervals) == 1
        assert intervals[0] == (1, 100, False)

    def test_build_speaker_intervals_coverage(self):
        """区間がタイムライン全体をカバーし、隙間がない。"""
        from aviutl_whisper.exporter import _build_speaker_intervals
        from aviutl_whisper.transcriber import TranscriptionSegment
        segs = [
            TranscriptionSegment(start=1.0, end=2.0, text="a", speaker="S1"),
            TranscriptionSegment(start=3.0, end=4.0, text="b", speaker="S1"),
        ]
        intervals = _build_speaker_intervals(segs, "S1", 30.0, 150)
        # 隙間なくカバーしているか
        for i in range(len(intervals) - 1):
            assert intervals[i][1] + 1 == intervals[i + 1][0]
        assert intervals[0][0] == 1
        assert intervals[-1][1] == 150

    def test_emit_image_objects(self):
        """_emit_image_objects が正しいexo行を出力する。"""
        from aviutl_whisper.exporter import _emit_image_objects, SpeakerImageSettings
        img = SpeakerImageSettings(file="C:\\img\\test.png", x=100.0, y=-50.0, scale=80.0)
        intervals = [(1, 30, True), (31, 60, False)]
        lines = []
        next_idx = _emit_image_objects(lines, intervals, img, layer=101, obj_idx_start=5)
        assert next_idx == 7
        content = "\n".join(lines)
        assert "[5]" in content
        assert "[6]" in content
        assert "layer=101" in content
        assert "_name=画像ファイル" in content
        assert "file=C:\\img\\test.png" in content
        assert "_name=標準描画" in content
        assert "X=100.0" in content
        assert "Y=-50.0" in content
        assert "拡大率=80.00" in content
        assert "_name=色調補正" in content
        assert "彩度=100.0" in content  # 話中
        assert "彩度=0.0" in content    # 非話中

    def test_exo_with_tachie(self, sample_segments):
        """立ち絵付きexo出力が画像オブジェクトを含む。"""
        from aviutl_whisper.exporter import ExoSettings, SpeakerImageSettings, export_exo
        settings = ExoSettings(
            speaker_images=[
                SpeakerImageSettings(file="C:\\img\\s1.png", x=200, y=-100, scale=80),
                SpeakerImageSettings(file="C:\\img\\s2.png", x=-200, y=-100, scale=80),
            ],
        )
        text = export_exo(sample_segments, settings=settings)
        assert "画像ファイル" in text
        assert "file=C:\\img\\s1.png" in text
        assert "file=C:\\img\\s2.png" in text
        assert "色調補正" in text
        assert "layer=101" in text or "layer=1" in text  # tachie layers
        assert "layer=102" in text or "layer=2" in text
        # New layer structure: tachie 1,2 → text 3,4 (consecutive, no gaps)
        assert "layer=1\r\n" in text
        assert "layer=2\r\n" in text
        assert "layer=3\r\n" in text
        assert "layer=4\r\n" in text

    def test_exo_without_tachie(self, sample_segments):
        """立ち絵なしのexo出力に画像オブジェクトが含まれない。"""
        from aviutl_whisper.exporter import ExoSettings, export_exo
        settings = ExoSettings(speaker_images=[])
        text = export_exo(sample_segments, settings=settings)
        assert "画像ファイル" not in text
        assert "色調補正" not in text

    def test_exo_rgb_to_bgr(self):
        """exoは色をRGBそのまま使用する (BGR変換なし)。"""
        from aviutl_whisper.exporter import ExoSettings
        s = ExoSettings(speaker_colors=["ff0000"])
        # RGB ff0000 がそのまま返される
        assert s.get_speaker_color(0) == "ff0000"

    def test_exo_custom_settings(self, sample_segments):
        """カスタムExoSettingsでexo出力が正しく反映される。"""
        from aviutl_whisper.exporter import ExoSettings, export_exo
        settings = ExoSettings(
            font="Meiryo",
            font_size=48,
            spacing_x=5,
            spacing_y=10,
            display_speed=1.5,
            align=7,
            bold=True,
            italic=True,
            soft_edge=False,
            pos_x=100.0,
            pos_y=-50.5,
            speaker_colors=["ff0000", "0000ff"],
            speaker_edge_colors=["aaaaaa", "bbbbbb"],
        )
        text = export_exo(sample_segments, settings=settings)
        assert "font=Meiryo" in text
        assert "サイズ=48" in text
        assert "spacing_x=5" in text
        assert "spacing_y=10" in text
        assert "表示速度=1.5" in text
        assert "align=7" in text
        assert "B=1" in text
        assert "I=1" in text
        assert "soft=0" in text
        assert "X=100.0" in text
        assert "Y=-50.5" in text
        # RGB がそのまま使われる
        assert "color=ff0000" in text
        assert "color2=aaaaaa" in text

    def test_exo_per_speaker_colors(self, sample_segments):
        """話者ごとの色が正しく割り当てられる。"""
        from aviutl_whisper.exporter import ExoSettings, export_exo
        settings = ExoSettings(
            speaker_colors=["ff0000", "00ff00"],
            speaker_edge_colors=["111111", "222222"],
        )
        text = export_exo(sample_segments, settings=settings)
        # RGB がそのまま exo に出力される
        assert "color=ff0000" in text
        assert "color=00ff00" in text
        assert "color2=111111" in text
        assert "color2=222222" in text

    def test_exo_export_to_file_with_settings(self, sample_segments, tmp_path):
        """ExoSettingsを使ったファイル出力が正しい。"""
        from aviutl_whisper.exporter import ExoSettings, export_to_file
        settings = ExoSettings(font="Arial", font_size=60)
        path = str(tmp_path / "custom.exo")
        result = export_to_file(sample_segments, path, "exo", exo_settings=settings)
        assert os.path.exists(result)
        content = open(result, encoding="cp932").read()
        assert "font=Arial" in content
        assert "サイズ=60" in content

    def test_export_to_file(self, sample_segments, tmp_path):
        """ファイルへのエクスポートが動作する。"""
        from aviutl_whisper.exporter import export_to_file
        path = str(tmp_path / "output.srt")
        result = export_to_file(sample_segments, path, "srt")
        assert os.path.exists(result)
        content = open(result, encoding="utf-8").read()
        assert "こんにちは" in content

    def test_exo_background_image(self, sample_segments):
        """背景画像が設定されている場合、layer=1に画像ファイルオブジェクトが出力される。"""
        from aviutl_whisper.exporter import ExoSettings, export_exo
        settings = ExoSettings(background_image="C:\\bg\\background.png")
        text = export_exo(sample_segments, settings=settings)
        # 背景画像がexoに含まれる
        assert "file=C:\\bg\\background.png" in text
        # 背景画像は最初のオブジェクト ([0])
        lines = text.split("\r\n")
        obj0_idx = lines.index("[0]")
        # 背景は layer=1
        assert "layer=1" in lines[obj0_idx + 3]
        # start=1, end=total_frames
        assert lines[obj0_idx + 1] == "start=1"
        # テキストは layer=2, 3 (2話者、背景あり、立ち絵なし)
        assert "layer=2\r\n" in text
        assert "layer=3\r\n" in text

    def test_exo_background_image_empty(self, sample_segments):
        """背景画像が空の場合、画像ファイルオブジェクトが出力されない。"""
        from aviutl_whisper.exporter import ExoSettings, export_exo
        settings = ExoSettings(background_image="")
        text = export_exo(sample_segments, settings=settings)
        # 背景画像関連は含まれない (ただし立ち絵の画像ファイルとは区別)
        # テキストのlayerが1から始まる
        assert "layer=1\r\n" in text

    def test_exo_layer_order_bg_tachie_text(self, sample_segments):
        """レイヤー順が 背景→立ち絵→テキスト の連番になる。"""
        from aviutl_whisper.exporter import ExoSettings, SpeakerImageSettings, export_exo
        settings = ExoSettings(
            background_image="C:\\bg\\bg.png",
            speaker_images=[
                SpeakerImageSettings(file="C:\\img\\s1.png", x=0, y=0, scale=100),
                SpeakerImageSettings(file="C:\\img\\s2.png", x=0, y=0, scale=100),
            ],
        )
        text = export_exo(sample_segments, settings=settings)
        import re
        layers = [int(m.group(1)) for m in re.finditer(r"layer=(\d+)", text)]
        unique_layers = sorted(set(layers))
        # 背景=1, 立ち絵=2,3, テキスト=4,5 → 連番
        assert unique_layers == [1, 2, 3, 4, 5]

    def test_exo_layer_order_tachie_text_no_bg(self, sample_segments):
        """背景なし: 立ち絵→テキスト の連番。"""
        from aviutl_whisper.exporter import ExoSettings, SpeakerImageSettings, export_exo
        settings = ExoSettings(
            speaker_images=[
                SpeakerImageSettings(file="C:\\img\\s1.png", x=0, y=0, scale=100),
                SpeakerImageSettings(file="C:\\img\\s2.png", x=0, y=0, scale=100),
            ],
        )
        text = export_exo(sample_segments, settings=settings)
        import re
        layers = [int(m.group(1)) for m in re.finditer(r"layer=(\d+)", text)]
        unique_layers = sorted(set(layers))
        # 立ち絵=1,2, テキスト=3,4 → 連番
        assert unique_layers == [1, 2, 3, 4]

    def test_exo_layer_order_text_only(self, sample_segments):
        """背景も立ち絵もなし: テキストのみ layer=1,2。"""
        from aviutl_whisper.exporter import ExoSettings, export_exo
        settings = ExoSettings()
        text = export_exo(sample_segments, settings=settings)
        import re
        layers = [int(m.group(1)) for m in re.finditer(r"layer=(\d+)", text)]
        unique_layers = sorted(set(layers))
        assert unique_layers == [1, 2]

    def test_exo_settings_background_image(self):
        """ExoSettingsのbackground_imageフィールドが正しく動作する。"""
        from aviutl_whisper.exporter import ExoSettings
        s = ExoSettings(background_image="C:\\bg\\test.png")
        assert s.background_image == "C:\\bg\\test.png"

    def test_exo_settings_from_dict_background(self):
        """from_dictでbackground_imageが正しく読み込まれる。"""
        from aviutl_whisper.exporter import ExoSettings
        s = ExoSettings.from_dict({"background_image": "C:\\bg\\test.png"})
        assert s.background_image == "C:\\bg\\test.png"
        # 空の場合
        s2 = ExoSettings.from_dict({})
        assert s2.background_image == ""

    def test_wrap_text_basic(self):
        """_wrap_textが指定文字数で改行する。"""
        from aviutl_whisper.exporter import _wrap_text
        assert _wrap_text("あいうえおかきくけこ", 5) == "あいうえお\nかきくけこ"

    def test_wrap_text_short(self):
        """_wrap_textが短いテキストをそのまま返す。"""
        from aviutl_whisper.exporter import _wrap_text
        assert _wrap_text("abc", 10) == "abc"

    def test_wrap_text_zero(self):
        """_wrap_textが0の場合テキストをそのまま返す。"""
        from aviutl_whisper.exporter import _wrap_text
        text = "あいうえおかきくけこさしすせそ"
        assert _wrap_text(text, 0) == text

    def test_wrap_text_preserves_existing_newlines(self):
        """_wrap_textが既存の改行を保持する。"""
        from aviutl_whisper.exporter import _wrap_text
        assert _wrap_text("あいう\nえおか", 5) == "あいう\nえおか"

    def test_wrap_text_long(self):
        """_wrap_textが長いテキストを複数行に分割する。"""
        from aviutl_whisper.exporter import _wrap_text
        result = _wrap_text("123456789012345", 5)
        assert result == "12345\n67890\n12345"

    def test_exo_text_wrapping(self, sample_segments):
        """exo出力でテキストが改行される。"""
        from aviutl_whisper.exporter import ExoSettings, export_exo, _encode_exo_text
        settings = ExoSettings(max_chars_per_line=5)
        text = export_exo(sample_segments, settings=settings)
        # "こんにちは" は5文字なので改行なし
        hello_hex = _encode_exo_text("こんにちは")
        assert hello_hex in text

    def test_exo_settings_max_chars(self):
        """ExoSettingsのmax_chars_per_lineフィールド。"""
        from aviutl_whisper.exporter import ExoSettings
        s = ExoSettings()
        assert s.max_chars_per_line == 20
        s2 = ExoSettings.from_dict({"max_chars_per_line": 30})
        assert s2.max_chars_per_line == 30

    def test_detect_device_large_model(self):
        """large-v3モデルの場合、GPU時はint8_float16が選択される。"""
        from aviutl_whisper.models import _detect_device
        device, compute_type = _detect_device("large-v3")
        assert device in ("cuda", "cpu")
        if device == "cuda":
            assert compute_type == "int8_float16"
        else:
            assert compute_type == "int8"

    def test_detect_device_medium_model(self):
        """mediumモデルの場合、GPU時はfloat16が選択される。"""
        from aviutl_whisper.models import _detect_device
        device, compute_type = _detect_device("medium")
        assert device in ("cuda", "cpu")
        if device == "cuda":
            assert compute_type == "float16"
        else:
            assert compute_type == "int8"

    def test_exo_edge_type(self, sample_segments):
        """soft_edge=Trueの場合、type=3(縁取り)が出力される。"""
        from aviutl_whisper.exporter import ExoSettings, export_exo
        settings = ExoSettings(soft_edge=True)
        text = export_exo(sample_segments, settings=settings)
        assert "type=3" in text
        assert "soft=1" in text

    def test_exo_no_edge_type(self, sample_segments):
        """soft_edge=Falseの場合、type=0(標準)が出力される。"""
        from aviutl_whisper.exporter import ExoSettings, export_exo
        settings = ExoSettings(soft_edge=False)
        text = export_exo(sample_segments, settings=settings)
        assert "type=0" in text
        assert "soft=0" in text


# ============================================================
# api.py テスト
# ============================================================

class TestApi:
    """api モジュールのテスト。"""

    def test_get_device_info(self):
        """デバイス情報が返る。"""
        from aviutl_whisper.api import Api
        api = Api()
        info = api.get_device_info()
        assert "device" in info
        assert "detail" in info
        assert info["device"] in ("GPU (CUDA)", "CPU")

    def test_get_progress(self):
        """進捗が辞書で返る。"""
        from aviutl_whisper.api import Api
        api = Api()
        progress = api.get_progress()
        assert "progress" in progress
        assert "message" in progress

    def test_get_preview_segments_no_result(self):
        """結果がない場合はエラーを返す。"""
        from aviutl_whisper.api import Api
        api = Api()
        res = api.get_preview_segments()
        assert res["success"] is False

    def test_get_preview_segments_with_data(self):
        """セグメントデータがある場合は正しく返す。"""
        from aviutl_whisper.api import Api
        from aviutl_whisper.transcriber import TranscriptionSegment
        api = Api()
        api._last_segments = [
            TranscriptionSegment(start=0.0, end=1.5, text="こんにちは", speaker="Speaker 1"),
            TranscriptionSegment(start=1.5, end=3.0, text="元気ですか", speaker="Speaker 2"),
        ]
        res = api.get_preview_segments()
        assert res["success"] is True
        assert len(res["segments"]) == 2
        assert res["segments"][0]["text"] == "こんにちは"
        assert res["segments"][0]["speaker"] == "Speaker 1"
        assert res["segments"][1]["start"] == 1.5

    def test_get_image_base64_missing(self):
        """存在しないファイルはエラーを返す。"""
        from aviutl_whisper.api import Api
        api = Api()
        res = api.get_image_base64("nonexistent.png")
        assert res["success"] is False

    def test_get_image_base64_valid(self, tmp_path):
        """画像ファイルをbase64で返す。"""
        from aviutl_whisper.api import Api
        # 1x1 PNG
        import base64
        png_data = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        img_path = tmp_path / "test.png"
        img_path.write_bytes(png_data)
        api = Api()
        res = api.get_image_base64(str(img_path))
        assert res["success"] is True
        assert res["data_url"].startswith("data:image/png;base64,")


class TestSegmentEditing:
    """セグメント編集APIのテスト。"""

    def _make_api_with_segments(self):
        from aviutl_whisper.api import Api
        from aviutl_whisper.transcriber import TranscriptionSegment
        api = Api()
        api._last_segments = [
            TranscriptionSegment(start=0.0, end=1.5, text="こんにちは", speaker="Speaker 1"),
            TranscriptionSegment(start=1.5, end=3.0, text="元気ですか", speaker="Speaker 2"),
            TranscriptionSegment(start=3.0, end=5.0, text="はい元気です", speaker="Speaker 1"),
        ]
        api._last_output_format = "text"
        return api

    def test_update_segment_text(self):
        api = self._make_api_with_segments()
        res = api.update_segment(0, text="さようなら")
        assert res["success"] is True
        assert api._last_segments[0].text == "さようなら"
        assert api._last_segments[0].speaker == "Speaker 1"  # 変更なし

    def test_update_segment_speaker(self):
        api = self._make_api_with_segments()
        res = api.update_segment(1, speaker="Speaker 1")
        assert res["success"] is True
        assert api._last_segments[1].speaker == "Speaker 1"
        assert api._last_segments[1].text == "元気ですか"  # 変更なし

    def test_update_segment_time(self):
        api = self._make_api_with_segments()
        res = api.update_segment(0, start=0.5, end=2.0)
        assert res["success"] is True
        assert api._last_segments[0].start == 0.5
        assert api._last_segments[0].end == 2.0

    def test_update_segment_invalid_index(self):
        api = self._make_api_with_segments()
        res = api.update_segment(99, text="x")
        assert res["success"] is False

    def test_update_segment_no_data(self):
        from aviutl_whisper.api import Api
        api = Api()
        res = api.update_segment(0, text="x")
        assert res["success"] is False

    def test_add_segment(self):
        api = self._make_api_with_segments()
        res = api.add_segment(2.0, 2.5, "新テキスト", "Speaker 2")
        assert res["success"] is True
        assert len(api._last_segments) == 4
        assert res["inserted_index"] == 2
        assert api._last_segments[2].text == "新テキスト"
        assert api._last_segments[2].start == 2.0

    def test_add_segment_at_end(self):
        api = self._make_api_with_segments()
        res = api.add_segment(10.0, 12.0, "末尾", "Speaker 1")
        assert res["success"] is True
        assert len(api._last_segments) == 4
        assert res["inserted_index"] == 3

    def test_add_segment_invalid_time(self):
        api = self._make_api_with_segments()
        res = api.add_segment(5.0, 3.0, "逆転", "Speaker 1")
        assert res["success"] is False

    def test_delete_segment(self):
        api = self._make_api_with_segments()
        res = api.delete_segment(1)
        assert res["success"] is True
        assert len(api._last_segments) == 2
        assert api._last_segments[0].text == "こんにちは"
        assert api._last_segments[1].text == "はい元気です"

    def test_delete_last_segment_fails(self):
        from aviutl_whisper.api import Api
        from aviutl_whisper.transcriber import TranscriptionSegment
        api = Api()
        api._last_segments = [
            TranscriptionSegment(start=0.0, end=1.0, text="唯一", speaker="Speaker 1"),
        ]
        api._last_output_format = "text"
        res = api.delete_segment(0)
        assert res["success"] is False

    def test_delete_invalid_index(self):
        api = self._make_api_with_segments()
        res = api.delete_segment(99)
        assert res["success"] is False

    def test_edit_returns_segments_and_text(self):
        api = self._make_api_with_segments()
        res = api.update_segment(0, text="変更後")
        assert "segments" in res
        assert "text" in res
        assert len(res["segments"]) == 3
        assert res["segments"][0]["text"] == "変更後"

    def test_play_segment_audio_no_data(self):
        from aviutl_whisper.api import Api
        api = Api()
        res = api.play_segment_audio(0)
        assert res["success"] is False

    def test_play_segment_audio_invalid_index(self):
        api = self._make_api_with_segments()
        api._last_wav_path = "nonexistent.wav"
        res = api.play_segment_audio(99)
        assert res["success"] is False

    def test_play_segment_audio_no_wav(self):
        api = self._make_api_with_segments()
        api._last_wav_path = "nonexistent.wav"
        res = api.play_segment_audio(0)
        assert res["success"] is False

    def test_stop_audio(self):
        """stop_audio は例外なく呼べる。"""
        from aviutl_whisper.api import Api
        api = Api()
        res = api.stop_audio()
        # sounddevice が入っていれば success=True, なければ error
        assert "success" in res

    def test_bake_mapping_on_update(self):
        """update_segment実行時にマッピングが焼き込まれる。"""
        api = self._make_api_with_segments()
        # Speaker 1 → slot 1, Speaker 2 → slot 0 (入れ替え)
        api._speaker_mapping = {"Speaker 1": 1, "Speaker 2": 0}
        # マッピング後: seg[0] Speaker 1 → Speaker 2, seg[1] Speaker 2 → Speaker 1
        res = api.update_segment(0, text="変更テスト")
        assert res["success"] is True
        # ベイクイン後、マッピングはNone
        assert api._speaker_mapping is None
        # seg[0] は元 Speaker 1 だが、ベイク後 Speaker 2 になっているはず
        assert api._last_segments[0].speaker == "Speaker 2"
        assert api._last_segments[0].text == "変更テスト"

    def test_bake_mapping_on_add(self):
        """add_segment実行時にマッピングが焼き込まれる。"""
        api = self._make_api_with_segments()
        api._speaker_mapping = {"Speaker 1": 1, "Speaker 2": 0}
        res = api.add_segment(2.0, 2.5, "追加", "Speaker 1")
        assert res["success"] is True
        assert api._speaker_mapping is None

    def test_bake_mapping_on_delete(self):
        """delete_segment実行時にマッピングが焼き込まれる。"""
        api = self._make_api_with_segments()
        api._speaker_mapping = {"Speaker 1": 1, "Speaker 2": 0}
        res = api.delete_segment(0)
        assert res["success"] is True
        assert api._speaker_mapping is None
        assert len(api._last_segments) == 2

    def test_bake_mapping_no_mapping(self):
        """マッピングがNoneの場合はベイクインが何もしない。"""
        api = self._make_api_with_segments()
        api._speaker_mapping = None
        original_speakers = [s.speaker for s in api._last_segments]
        res = api.update_segment(0, text="テスト")
        assert res["success"] is True
        assert [s.speaker for s in api._last_segments if s.text != "テスト"] == original_speakers[1:]


class TestSubtitleRendering:
    """字幕画像レンダリングのテスト。"""

    def test_render_subtitle_image_basic(self):
        from aviutl_whisper.api import _render_subtitle_image
        png_bytes = _render_subtitle_image("こんにちは")
        assert len(png_bytes) > 0
        # PNG magic bytes
        assert png_bytes[:4] == b"\x89PNG"

    def test_render_subtitle_image_custom_settings(self):
        from aviutl_whisper.api import _render_subtitle_image
        png_bytes = _render_subtitle_image(
            "テスト字幕",
            font_size=48,
            text_color="ff0000",
            edge_color="0000ff",
            align=7,  # 下中央
            soft_edge=True,
        )
        assert len(png_bytes) > 0
        assert png_bytes[:4] == b"\x89PNG"

    def test_render_subtitle_image_multiline(self):
        from aviutl_whisper.api import _render_subtitle_image
        png_bytes = _render_subtitle_image(
            "一行目\n二行目\n三行目",
            max_chars_per_line=0,
        )
        assert len(png_bytes) > 0

    def test_render_subtitle_image_wrap(self):
        from aviutl_whisper.api import _render_subtitle_image
        png_bytes = _render_subtitle_image(
            "あいうえおかきくけこさしすせそ",
            max_chars_per_line=5,
        )
        assert len(png_bytes) > 0

    def test_render_subtitle_api(self):
        from aviutl_whisper.api import Api
        api = Api()
        res = api.render_subtitle_image("テスト", 0, {
            "font": "MS UI Gothic",
            "font_size": 34,
        })
        assert res["success"] is True
        assert res["data_url"].startswith("data:image/png;base64,")

    def test_render_subtitle_api_with_colors(self):
        from aviutl_whisper.api import Api
        api = Api()
        res = api.render_subtitle_image("テスト", 1, {
            "speaker_colors": ["ffffff", "ff0000"],
            "speaker_edge_colors": ["000000", "00ff00"],
        })
        assert res["success"] is True

    def test_resolve_font_path(self):
        from aviutl_whisper.api import _resolve_font_path
        # MS Gothic should be resolvable on Windows
        path = _resolve_font_path("MS Gothic")
        if path:  # may be None on non-Windows / CI
            assert os.path.exists(path)
            assert path.lower().endswith((".ttf", ".ttc", ".otf"))


# ============================================================
# speaker mapping テスト
# ============================================================

class TestSpeakerMapping:
    """話者マッピング機能のテスト。"""

    @pytest.fixture
    def two_speaker_segments(self):
        from aviutl_whisper.transcriber import TranscriptionSegment
        return [
            TranscriptionSegment(start=0.0, end=1.0, text="おはよう", speaker="Speaker 1"),
            TranscriptionSegment(start=1.0, end=2.0, text="こんにちは", speaker="Speaker 2"),
            TranscriptionSegment(start=2.0, end=3.0, text="元気？", speaker="Speaker 1"),
            TranscriptionSegment(start=3.0, end=4.0, text="うん", speaker="Speaker 2"),
        ]

    def test_apply_mapping_none(self, two_speaker_segments):
        """mapping=Noneの場合はセグメントがそのまま返る。"""
        from aviutl_whisper.api import _apply_speaker_mapping
        result = _apply_speaker_mapping(two_speaker_segments, None)
        assert result is two_speaker_segments

    def test_apply_mapping_default(self, two_speaker_segments):
        """デフォルトマッピング(変更なし)の場合はそのまま返る。"""
        from aviutl_whisper.api import _apply_speaker_mapping
        mapping = {"Speaker 1": 0, "Speaker 2": 1}
        result = _apply_speaker_mapping(two_speaker_segments, mapping)
        assert result is two_speaker_segments

    def test_apply_mapping_swap(self, two_speaker_segments):
        """話者を入れ替えるマッピングが正しく動作する。"""
        from aviutl_whisper.api import _apply_speaker_mapping
        mapping = {"Speaker 1": 1, "Speaker 2": 0}
        result = _apply_speaker_mapping(two_speaker_segments, mapping)
        # Speaker 1 のセグメント(index 0,2) が Speaker 2 に、逆も同様
        assert result[0].speaker == "Speaker 2"
        assert result[0].text == "おはよう"
        assert result[1].speaker == "Speaker 1"
        assert result[1].text == "こんにちは"
        assert result[2].speaker == "Speaker 2"
        assert result[3].speaker == "Speaker 1"

    def test_apply_mapping_swap_exo_colors(self, two_speaker_segments):
        """マッピング入れ替え後のexo出力で色が正しく割り当てられる。"""
        from aviutl_whisper.api import _apply_speaker_mapping
        from aviutl_whisper.exporter import ExoSettings, export_exo
        mapping = {"Speaker 1": 1, "Speaker 2": 0}
        swapped = _apply_speaker_mapping(two_speaker_segments, mapping)
        settings = ExoSettings(speaker_colors=["ff0000", "00ff00"])
        text = export_exo(swapped, settings=settings)
        # "おはよう" は元Speaker1だがswap後はSpeaker2 → layer 2, color 00ff00
        # "こんにちは" は元Speaker2だがswap後はSpeaker1 → layer 1, color ff0000
        assert "color=ff0000" in text
        assert "color=00ff00" in text

    def test_build_speaker_info(self):
        """_build_speaker_infoが話者情報を正しく返す。"""
        from aviutl_whisper.api import Api
        from aviutl_whisper.transcriber import TranscriptionSegment
        api = Api()
        segs = [
            TranscriptionSegment(start=0.0, end=1.0, text="hello world", speaker="Speaker 1"),
            TranscriptionSegment(start=1.0, end=2.0, text="hi there", speaker="Speaker 2"),
            TranscriptionSegment(start=2.0, end=3.0, text="ok", speaker="Speaker 1"),
        ]
        info = api._build_speaker_info(segs)
        assert len(info) == 2
        assert info[0]["name"] == "Speaker 1"
        assert info[0]["segment_count"] == 2
        assert info[0]["sample_text"] == "hello world"
        assert info[0]["first_start"] == 0.0
        assert info[1]["name"] == "Speaker 2"
        assert info[1]["segment_count"] == 1

    def test_apply_mapping_preserves_text(self, two_speaker_segments):
        """マッピング適用後もテキストとタイムスタンプが保持される。"""
        from aviutl_whisper.api import _apply_speaker_mapping
        mapping = {"Speaker 1": 1, "Speaker 2": 0}
        result = _apply_speaker_mapping(two_speaker_segments, mapping)
        for orig, mapped in zip(two_speaker_segments, result):
            assert orig.start == mapped.start
            assert orig.end == mapped.end
            assert orig.text == mapped.text


# ============================================================
# settings.py テスト
# ============================================================

class TestSettings:
    """settings モジュールのテスト。"""

    def test_load_default_settings(self, monkeypatch, tmp_path):
        """設定ファイルが無い場合デフォルト値を返す。"""
        from aviutl_whisper import settings
        monkeypatch.setattr(settings, "_get_settings_path",
                            lambda: tmp_path / "nonexistent" / "settings.json")
        result = settings.load_settings()
        assert result["model_size"] == "medium"
        assert result["exo"]["font"] == "MS UI Gothic"

    def test_save_and_load_settings(self, monkeypatch, tmp_path):
        """設定を保存して再読み込みできる。"""
        from aviutl_whisper import settings
        path = tmp_path / "settings.json"
        monkeypatch.setattr(settings, "_get_settings_path", lambda: path)

        data = {"model_size": "large-v3", "language": "en",
                "exo": {"font": "Arial", "font_size": 60}}
        settings.save_settings(data)
        assert path.exists()

        loaded = settings.load_settings()
        assert loaded["model_size"] == "large-v3"
        assert loaded["exo"]["font"] == "Arial"
        assert loaded["exo"]["font_size"] == 60
        # デフォルト値がマージされている
        assert loaded["exo"]["bold"] is False

    def test_deep_merge(self):
        """_deep_mergeがネストした辞書を正しくマージする。"""
        from aviutl_whisper.settings import _deep_merge
        defaults = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 99}, "e": 5}
        result = _deep_merge(defaults, override)
        assert result["a"] == 1
        assert result["b"]["c"] == 99
        assert result["b"]["d"] == 3
        assert result["e"] == 5

    def test_default_settings_has_diarization_method(self):
        """デフォルト設定にdiarization_methodが含まれる。"""
        from aviutl_whisper.settings import DEFAULT_SETTINGS
        assert "diarization_method" in DEFAULT_SETTINGS
        assert DEFAULT_SETTINGS["diarization_method"] == "speechbrain"

    def test_default_settings_has_hf_token_encrypted(self):
        """デフォルト設定にhf_token_encryptedが含まれる。"""
        from aviutl_whisper.settings import DEFAULT_SETTINGS
        assert "hf_token_encrypted" in DEFAULT_SETTINGS
        assert DEFAULT_SETTINGS["hf_token_encrypted"] == ""

    def test_encrypt_decrypt_token_empty(self):
        """空トークンの暗号化/復号。"""
        from aviutl_whisper.settings import encrypt_token, decrypt_token
        assert encrypt_token("") == ""
        assert decrypt_token("") == ""

    def test_encrypt_decrypt_token_roundtrip(self):
        """トークンの暗号化→復号ラウンドトリップ。"""
        from aviutl_whisper.settings import encrypt_token, decrypt_token
        token = "hf_test_token_12345"
        encrypted = encrypt_token(token)
        assert encrypted != ""
        assert encrypted != token  # 暗号化されている
        decrypted = decrypt_token(encrypted)
        assert decrypted == token

    def test_encrypt_token_b64_fallback(self, monkeypatch):
        """DPAPI不使用時のbase64フォールバック。"""
        import platform
        from aviutl_whisper import settings
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        token = "hf_fallback_test"
        encrypted = settings.encrypt_token(token)
        assert encrypted.startswith("b64:")
        decrypted = settings.decrypt_token(encrypted)
        assert decrypted == token


# ============================================================
# pyannote models テスト
# ============================================================

class TestPyannoteModels:
    """pyannote関連のモデルロードテスト。"""

    def test_load_pyannote_missing_token(self):
        """トークンなしでpyannoteロードするとValueError。"""
        pytest.importorskip("pyannote.audio")
        from aviutl_whisper.models import load_pyannote_pipeline
        with pytest.raises(ValueError, match="HuggingFaceトークン"):
            load_pyannote_pipeline(hf_token="")

    def test_load_pyannote_not_installed(self, monkeypatch):
        """pyannote未インストール時にImportError。"""
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pyannote.audio" or name.startswith("pyannote"):
                raise ImportError("No module named 'pyannote'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        from aviutl_whisper.models import load_pyannote_pipeline
        with pytest.raises(ImportError, match="pyannote.audioがインストールされていません"):
            load_pyannote_pipeline(hf_token="hf_test_token")
