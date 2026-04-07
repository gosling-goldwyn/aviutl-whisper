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

    def test_exo_rgb_to_bgr(self):
        """RGB→BGR変換が正しい。"""
        from aviutl_whisper.exporter import _rgb_to_bgr
        assert _rgb_to_bgr("ff0000") == "0000ff"  # 赤 → BGR
        assert _rgb_to_bgr("00ff00") == "00ff00"  # 緑 → そのまま
        assert _rgb_to_bgr("0000ff") == "ff0000"  # 青 → BGR
        assert _rgb_to_bgr("ffffff") == "ffffff"   # 白 → そのまま

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
        # ff0000 → BGR: 0000ff
        assert "color=0000ff" in text
        # aaaaaa → BGR: aaaaaa
        assert "color2=aaaaaa" in text

    def test_exo_per_speaker_colors(self, sample_segments):
        """話者ごとの色が正しく割り当てられる。"""
        from aviutl_whisper.exporter import ExoSettings, export_exo
        settings = ExoSettings(
            speaker_colors=["ff0000", "00ff00"],
            speaker_edge_colors=["111111", "222222"],
        )
        text = export_exo(sample_segments, settings=settings)
        # Speaker 1 → ff0000 → BGR 0000ff, Speaker 2 → 00ff00 → BGR 00ff00
        assert "color=0000ff" in text
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
