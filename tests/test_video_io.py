import base64
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import Mock

import pytest

from aviutl_whisper.api import Api
from aviutl_whisper.transcriber import TranscriptionSegment
from aviutl_whisper.video_io import MAX_CHUNK_BYTES, VideoIO
from tests.test_backend import create_dummy_wav


@pytest.fixture
def video_io():
    io = VideoIO()
    yield io
    io.close()


def encoded(data):
    return base64.b64encode(data).decode('ascii')


def test_atomic_commit_only_replaces_destination_when_complete(video_io, tmp_path):
    target = tmp_path / '日本語.mp4'
    target.write_bytes(b'old movie')
    selected = video_io.reserve(str(target))
    video_io.begin(selected['token'])
    video_io.write(selected['token'], 0, encoded(b'first'))
    video_io.write(selected['token'], 5, encoded(b' second'))
    assert target.read_bytes() == b'old movie'
    assert Path(video_io.finish(selected['token'])).read_bytes() == b'first second'
    assert not list(tmp_path.glob('*.part'))


def test_cancel_preserves_existing_destination_and_removes_partial(video_io, tmp_path):
    target = tmp_path / 'movie.mp4'
    target.write_bytes(b'keep')
    selected = video_io.reserve(str(target))
    video_io.begin(selected['token'])
    video_io.write(selected['token'], 0, encoded(b'partial'))
    video_io.abort(selected['token'])
    video_io.abort(selected['token'])  # Idempotent cleanup.
    assert target.read_bytes() == b'keep'
    assert not list(tmp_path.glob('*.part'))
    assert not video_io.busy


@pytest.mark.parametrize('position,payload', [(1, 'YQ=='), (-1, 'YQ=='), (True, 'YQ=='), (0, '!'), (0, 'a' * (MAX_CHUNK_BYTES * 2))], ids=['order', 'negative', 'bool', 'base64', 'oversize'])
def test_rejects_invalid_offsets_and_chunks(video_io, tmp_path, position, payload):
    token = video_io.reserve(str(tmp_path / 'movie.mp4'))['token']
    video_io.begin(token)
    with pytest.raises((ValueError, TypeError)):
        video_io.write(token, position, payload)


def test_rejects_paths_tokens_replays_and_empty_files(video_io, tmp_path):
    token = video_io.reserve(str(tmp_path / 'movie.mp4'))['token']
    with pytest.raises(ValueError):
        video_io.begin(str(tmp_path / 'elsewhere.mp4'))
    video_io.begin(token)
    with pytest.raises(ValueError):
        video_io.write('forged', 0, 'YQ==')
    with pytest.raises(ValueError):
        video_io.abort('forged')
    with pytest.raises(ValueError):
        video_io.finish(token)
    with pytest.raises(ValueError):
        video_io.reserve(str(tmp_path / 'second.mp4'))
    video_io.abort(token)
    with pytest.raises(ValueError):
        video_io.begin(token)


def test_close_cleans_active_output(video_io, tmp_path):
    token = video_io.reserve(str(tmp_path / 'movie.mp4'))['token']
    video_io.begin(token)
    video_io.close()
    assert not list(tmp_path.iterdir())


def test_range_reads_are_bounded_and_only_registered_files_are_visible(video_io, tmp_path):
    source = tmp_path / 'audio.wav'
    source.write_bytes(bytes(range(256)) * 4096)
    url = video_io.media_url(str(source))
    assert video_io.media_url(str(source)) == url
    request = urllib.request.Request(url, headers={'Range': 'bytes=8-15'})
    with urllib.request.urlopen(request) as response:
        assert response.status == 206
        assert response.headers['Content-Range'] == f'bytes 8-15/{source.stat().st_size}'
        assert response.read() == bytes(range(8, 16))
    request = urllib.request.Request(url, headers={'Range': 'bytes=-4'})
    with urllib.request.urlopen(request) as response:
        assert response.read() == bytes(range(252, 256))
    for suffix in ['../audio.wav', 'forged']:
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(url.rsplit('/', 1)[0] + '/' + suffix)
        assert error.value.code == 404
    request = urllib.request.Request(url, headers={'Range': 'bytes=999999999-'})
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request)
    assert error.value.code == 416


def test_dialog_is_only_authority_for_output_path(tmp_path):
    api = Api()
    api.window = Mock()
    api.window.create_file_dialog.return_value = [str(tmp_path / 'selected.mp4')]
    selection = api.select_video_output()
    assert selection['success']
    assert api.begin_video_output(str(tmp_path / 'other.mp4'))['success'] is False
    assert api.begin_video_output(selection['token'])['success']
    api._video_io.close()


def test_audio_sources_and_tts_missing_data_are_explicit(tmp_path, monkeypatch):
    api = Api()
    try:
        assert api.get_video_audio('silence')['clips'] == []
        assert not api.get_video_audio('source')['success']
        source = create_dummy_wav(str(tmp_path / 'source.wav'), 0.5)
        api._source_audio_path, api._audio_mode = source, 'source'
        result = api.get_video_audio('source')
        assert result['success'] and result['clips'][0]['start'] == 0
        assert result['clips'][0]['url'].startswith('http://127.0.0.1:')
        api._last_segments = [TranscriptionSegment(start=0, end=1, text='字幕', speaker='Speaker 1')]
        monkeypatch.setattr('aviutl_whisper.elevenlabs_tts.get_combined_audio_items', lambda *args: ([], [{'segment_indices': [0]}]))
        assert '不足' in api.get_video_audio('tts', {})['error']
        monkeypatch.setattr('aviutl_whisper.elevenlabs_tts.get_combined_audio_items', lambda *args: ([{'audio_path': source, 'segment_indices': [0]}], []))
        assert api.get_video_audio('tts', {})['clips'][0]['start'] == 0
    finally:
        api._video_io.close()
