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
        assert "layer=101" in text
        assert "layer=102" in text

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
