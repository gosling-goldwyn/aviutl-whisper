"""EXOの音声・BGM・クロージングを実際のフレーム区間で検証する。"""

import configparser
import json
import wave
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aviutl_whisper import exporter, settings
from aviutl_whisper.api import Api
from aviutl_whisper.exporter import ExoSettings, SpeakerImageSettings
from aviutl_whisper.transcriber import TranscriptionSegment


def wav(tmp_path, name, seconds):
    path = tmp_path / name
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(8000)
        stream.writeframes(b"\0\0" * round(seconds * 8000))
    return str(path)


def parse_exo(text):
    ini = configparser.ConfigParser(interpolation=None)
    ini.optionxform = str
    ini.read_string(text)
    objects = []
    for name in ini.sections():
        if not name.isdigit():
            continue
        obj = dict(ini[name])
        obj.update({key: int(obj[key]) for key in ("start", "end", "layer")})
        obj["effects"] = [dict(ini[section]) for section in ini.sections()
                          if section.startswith(name + ".")]
        obj["file"] = obj["effects"][0].get("file")
        obj["kind"] = obj["effects"][0]["_name"]
        objects.append(obj)
    return ini["exedit"], objects


def effect(obj, name):
    return next(item for item in obj["effects"] if item["_name"] == name)


def test_source_audio_uses_full_duration_and_offsets_subtitles_without_mutation(tmp_path):
    source = wav(tmp_path, "input.wav", 10)
    segments = [TranscriptionSegment(0, 2, "字幕", "Speaker 1")]
    exo = ExoSettings(background_image="bg.png", speaker_images=[
        SpeakerImageSettings(file="s1.png"), SpeakerImageSettings(file="s2.png"),
    ])
    header, objects = parse_exo(exporter.export_exo(segments, exo, audio_file=source))

    assert int(header["length"]) == 345  # 1.5 + 10秒、字幕終了で切らない
    audio = next(obj for obj in objects if obj["file"] == source)
    assert (audio["start"], audio["end"]) == (46, 345)
    assert audio["audio"] == "1"
    assert effect(audio, "標準再生")["音量"] == "100.0"
    subtitle = next(obj for obj in objects if obj["kind"] == "テキスト")
    assert (subtitle["start"], subtitle["end"]) == (46, 105)
    assert (segments[0].start, segments[0].end) == (0, 2)
    for name in ("s1.png", "s2.png"):
        intro = next(obj for obj in objects if obj["file"] == name and obj["start"] == 1)
        assert intro["end"] == 45
        assert effect(intro, "色調補正")["彩度"] == "100.0"


def test_ending_song_outro_and_closing_have_exact_boundaries(tmp_path):
    source = wav(tmp_path, "input.wav", 60)
    opening = wav(tmp_path, "opening.wav", 100)
    ending = wav(tmp_path, "ending.wav", 5)
    jingle = wav(tmp_path, "jingle.wav", 3)
    exo = ExoSettings(
        background_image="bg.png", speaker_images=[SpeakerImageSettings(file="s.png")],
        bgm_start_file=opening, bgm_end_file=ending, closing_jingle=jingle,
        closing_image="closing.png",
    )
    header, objects = parse_exo(exporter.export_exo(
        [TranscriptionSegment(0, 55, "本編", "Speaker 1")], exo, audio_file=source,
    ))
    assert int(header["length"]) == 2340  # 1.5 + 60 + 1.5 + 15
    start_music = next(obj for obj in objects if obj["file"] == opening)
    end_music = next(obj for obj in objects if obj["file"] == ending)
    assert (start_music["start"], start_music["end"]) == (1, 945)
    assert effect(start_music, "音量フェード")["アウト"] == "1.5000"
    assert (end_music["start"], end_music["end"]) == (946, 1890)
    assert effect(end_music, "音声ファイル")["ループ再生"] == "1"
    assert effect(end_music, "音量フェード")["アウト"] == "1.5000"
    assert effect(end_music, "標準再生")["音量"] == "20.0"

    closing = next(obj for obj in objects if obj["file"] == "closing.png")
    assert (closing["start"], closing["end"]) == (1891, 2340)
    visuals = [obj for obj in objects if obj["kind"] != "音声ファイル" and obj is not closing]
    assert all(obj["end"] < closing["start"] for obj in visuals)
    assert all(obj["layer"] < closing["layer"] for obj in visuals)
    outro = next(obj for obj in objects if obj["file"] == "s.png" and obj["start"] == 1846)
    assert outro["end"] == 1890
    assert effect(outro, "色調補正")["彩度"] == "100.0"
    sound = next(obj for obj in objects if obj["file"] == jingle)
    assert (sound["start"], sound["end"]) == (1891, 1980)
    assert effect(sound, "標準再生")["音量"] == "100.0"


def test_other_songs_are_shuffled_and_fill_main_without_gaps(tmp_path, monkeypatch):
    opening = wav(tmp_path, "opening.wav", 1)
    others = [wav(tmp_path, f"other{i}.wav", 2) for i in range(3)]
    shuffles = []

    def reverse_bag(bag):
        shuffles.append(list(bag))
        bag.reverse()

    monkeypatch.setattr(exporter.random, "shuffle", reverse_bag)
    exo = ExoSettings(audio_start_seconds=0, bgm_start_file=opening, bgm_files=others)
    header, objects = parse_exo(exporter.export_exo([TranscriptionSegment(0, 17, "本編")], exo))
    music = [obj for obj in objects if obj["kind"] == "音声ファイル"]
    assert music[0]["file"] == opening
    assert [obj["file"] for obj in music[1:4]] == list(reversed(others))
    assert len(shuffles) >= 2
    assert music[0]["start"] == 1 and music[-1]["end"] == int(header["length"])
    assert len({obj["layer"] for obj in music}) == 1
    assert all(left["end"] + 1 == right["start"] for left, right in zip(music, music[1:]))
    assert all(left["file"] != right["file"] for left, right in zip(music, music[1:]))
    assert all(obj["file"] in others for obj in music[1:])


def test_short_main_starts_ending_song_at_first_frame(tmp_path):
    ending = wav(tmp_path, "ending.wav", 2)
    _, objects = parse_exo(exporter.export_exo(
        [TranscriptionSegment(0, 4, "短い本編")], ExoSettings(bgm_end_file=ending),
    ))
    music = [obj for obj in objects if obj["kind"] == "音声ファイル"]
    assert len(music) == 1
    assert (music[0]["start"], music[0]["end"]) == (1, 165)


def test_final_fade_is_continuous_even_when_song_shorter_than_fade(tmp_path):
    track = wav(tmp_path, "short.wav", 0.4)
    _, objects = parse_exo(exporter.export_exo(
        [TranscriptionSegment(0, 4, "本編")],
        ExoSettings(audio_start_seconds=0, bgm_start_file=track, closing_image="end.png"),
    ))
    music = [obj for obj in objects if obj["kind"] == "音声ファイル"]
    final = music[-1]
    assert final["start"] <= 121
    assert final["end"] == 165
    assert effect(final, "音量フェード")["アウト"] == "1.5000"
    assert effect(final, "音声ファイル")["ループ再生"] == "1"


def test_custom_times_zero_volume_and_long_jingle_are_respected(tmp_path):
    track = wav(tmp_path, "music.wav", 20)
    jingle = wav(tmp_path, "jingle.wav", 10)
    header, objects = parse_exo(exporter.export_exo(
        [TranscriptionSegment(0, 4, "本編")], ExoSettings(
            audio_start_seconds=0, bgm_start_file=track, bgm_volume=0,
            outro_seconds=0, closing_seconds=2, closing_jingle=jingle,
        ),
    ))
    assert int(header["length"]) == 180
    music = next(obj for obj in objects if obj["file"] == track)
    assert music["end"] == 120
    assert effect(music, "標準再生")["音量"] == "0.0"
    sound = next(obj for obj in objects if obj["file"] == jingle)
    assert (sound["start"], sound["end"]) == (121, 180)
    assert effect(sound, "標準再生")["音量"] == "100.0"


def test_subframe_and_fractional_fps_keep_valid_objects():
    header, objects = parse_exo(exporter.export_exo(
        [TranscriptionSegment(0, 0.001, "短い字幕")], ExoSettings(fps=29.97),
    ))
    assert int(header["rate"]) / int(header["scale"]) == pytest.approx(29.97)
    assert all(1 <= obj["start"] <= obj["end"] <= int(header["length"]) for obj in objects)


@pytest.mark.parametrize("field,value", [
    ("audio_start_seconds", -1), ("audio_start_seconds", float("inf")),
    ("bgm_volume", -1), ("bgm_volume", 101), ("bgm_volume", float("nan")),
    ("outro_seconds", -0.1), ("closing_seconds", 0), ("closing_seconds", float("nan")),
    ("bgm_files", "not-a-list"),
])
def test_invalid_media_settings_are_rejected(field, value):
    with pytest.raises(ValueError):
        ExoSettings.from_dict({field: value})


def test_settings_json_roundtrip_keeps_every_media_field_and_old_files_get_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("AVIUTL_WHISPER_SETTINGS_PATH", str(tmp_path / "settings.json"))
    media = dict(
        audio_start_seconds=0, bgm_start_file="start.wav", bgm_end_file="end.wav",
        bgm_files=["one.mp3", "two.wav"], bgm_volume=0, outro_seconds=2.5,
        closing_jingle="jingle.wav", closing_image="end.png", closing_seconds=12,
    )
    api = Api(MagicMock())
    api.save_settings({"exo": media})
    exported = tmp_path / "aviutl-whisper-settings.json"
    api.window.create_file_dialog.return_value = [str(exported)]
    assert api.export_settings()["success"]
    assert all(json.loads(exported.read_text("utf-8"))["settings"]["exo"][k] == v
               for k, v in media.items())
    settings.save_settings({"exo": {"font": "Arial"}})
    assert settings.load_settings()["exo"]["audio_start_seconds"] == 1.5
    assert settings.load_settings()["exo"]["closing_seconds"] == 15
    result = api.import_settings()
    assert result["success"]
    restored = asdict(ExoSettings.from_dict(result["settings"]["exo"]))
    assert all(restored[key] == value for key, value in media.items())
    assert restored["font"] == "MS UI Gothic"


def test_reloaded_project_exports_original_audio_after_temporary_wav_cleanup(tmp_path):
    source = wav(tmp_path, "source.wav", 4)
    project = tmp_path / "project.awproj"
    project.write_text(json.dumps({
        "source_file": source, "audio_mode": "source",
        "segments": [{"start": 0, "end": 1, "text": "字幕"}],
        "exo_settings": {"audio_start_seconds": 2},
    }), encoding="utf-8")
    api = Api(MagicMock())
    api.window.create_file_dialog.return_value = [str(project)]
    assert api.load_project()["success"]
    temporary = api._last_wav_path
    api._cleanup_wav()
    assert not Path(temporary).exists()
    assert Path(source).exists()
    result = tmp_path / "movie.exo"
    api.window.create_file_dialog.return_value = [str(result)]
    assert api.save_result("exo")["success"]
    raw = result.read_bytes()
    assert b"\r\r\n" not in raw and b"\r\n" in raw
    header, objects = parse_exo(raw.decode("cp932"))
    sound = next(obj for obj in objects if obj["kind"] == "音声ファイル")
    assert sound["file"] == str(Path(source).resolve())
    assert (sound["start"], sound["end"]) == (61, 180)
    assert api._build_project_data({"source_file": "wrong.wav"})["source_file"] == source


def test_compressed_input_gets_persistent_full_quality_wav(tmp_path, monkeypatch):
    from pydub import AudioSegment
    import soundfile as sf

    source = tmp_path / "source.m4a"
    source.write_bytes(b"compressed source")
    decoded = AudioSegment.silent(duration=2000, frame_rate=48000).set_channels(2)
    monkeypatch.setattr(AudioSegment, "from_file", lambda path: decoded)
    output = tmp_path / "movie.exo"
    exporter.export_to_file(
        [TranscriptionSegment(0, 1, "字幕")], str(output), "exo", audio_file=str(source),
    )
    _, objects = parse_exo(output.read_text("cp932"))
    sound = next(obj for obj in objects if obj["kind"] == "音声ファイル")
    saved_wav = Path(sound["file"])
    assert saved_wav == tmp_path / "movie_audio.wav"
    assert sf.info(str(saved_wav)).channels == 2
    assert sf.info(str(saved_wav)).samplerate == 48000
    assert (sound["start"], sound["end"]) == (46, 105)


def test_missing_bgm_is_reported_instead_of_silently_dropping_it(tmp_path):
    with pytest.raises(FileNotFoundError):
        exporter.export_exo(
            [TranscriptionSegment(0, 2, "字幕")],
            ExoSettings(bgm_start_file=str(tmp_path / "missing.wav")),
        )


def test_music_conversion_deduplicates_assets_and_preserves_existing_files(tmp_path, monkeypatch):
    from pydub import AudioSegment

    source = tmp_path / "music.mp3"
    source.write_bytes(b"compressed music")
    existing = tmp_path / "movie_bgm_start.wav"
    existing.write_bytes(b"existing material")
    decoded = AudioSegment.silent(duration=2000, frame_rate=44100)
    calls = []

    def decode(path):
        calls.append(path)
        return decoded

    monkeypatch.setattr(AudioSegment, "from_file", decode)
    exo = ExoSettings(
        bgm_start_file=str(source), bgm_end_file=str(source),
        bgm_files=[str(source)], closing_jingle=str(source),
    )
    output = tmp_path / "movie.exo"
    exporter.export_to_file([TranscriptionSegment(0, 40, "本編")], str(output), "exo", exo)
    _, objects = parse_exo(output.read_text("cp932"))
    music = [obj for obj in objects if obj["kind"] == "音声ファイル"]
    assert {obj["file"] for obj in music} == {str(tmp_path / "movie_bgm_start_2.wav")}
    assert calls == [str(source)]
    assert existing.read_bytes() == b"existing material"
    assert exo.bgm_start_file == exo.bgm_end_file == exo.closing_jingle == str(source)
    assert exo.bgm_files == [str(source)]


def test_transcription_remembers_source_and_text_import_clears_it(tmp_path, monkeypatch):
    from aviutl_whisper.transcriber import TranscriptionResult

    source = wav(tmp_path, "input.wav", 4)
    segments = [TranscriptionSegment(0, 1, "字幕", "Speaker 1")]
    api = Api()
    monkeypatch.setattr("aviutl_whisper.api.models.load_whisper_model", lambda **kw: object())
    monkeypatch.setattr("aviutl_whisper.api.transcriber.transcribe",
                        lambda **kw: TranscriptionResult(segments, "ja", 1.0))
    monkeypatch.setattr(api, "_run_speechbrain_diarization", lambda *args: segments)
    try:
        result = api.transcribe(source, {"output_format": "exo"})
        assert result["success"]
        assert api._source_audio_path == str(Path(source).resolve())
        _, objects = parse_exo(api._regenerate_output())
        sound = next(obj for obj in objects if obj["kind"] == "音声ファイル")
        assert (sound["start"], sound["end"]) == (46, 165)
        result = api.import_text("無音のテキスト", 100, {})
        assert result["success"]
        assert api._source_audio_path == ""
        assert "_name=音声ファイル" not in api._regenerate_output()
        assert Path(source).exists()
    finally:
        api._cleanup_wav()
