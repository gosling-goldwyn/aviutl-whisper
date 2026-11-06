"""Actual Windows WebView2 H.264/AAC exports, verified with ffprobe/FFmpeg."""
import base64
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageChops, ImageDraw

from aviutl_whisper import elevenlabs_tts
from aviutl_whisper.video_io import VideoIO
from tests.e2e.conftest import PROJECT_ROOT, _stop_process

pytestmark = pytest.mark.e2e
PORT = 9223
SEGMENTS = [
    {'start': 0, 'end': 1, 'text': '日本語の字幕\n明示改行', 'speaker': 'Speaker 1'},
    {'start': 0.5, 'end': 1.5, 'text': '重複する発話', 'speaker': 'Speaker 2'},
    {'start': 2, 'end': 2.5, 'text': '最後の字幕', 'speaker': 'Speaker 1'},
]
TEST_FONT = 'Noto Sans JP'


@pytest.fixture(scope='module')
def japanese_font_css():
    """Serve the lockfile-pinned test font without depending on Windows fonts."""
    package = PROJECT_ROOT / 'node_modules' / '@fontsource' / 'noto-sans-jp'
    stylesheet = package / '400.css'
    if not stylesheet.is_file():
        pytest.fail('MP4 E2E用フォントがありません。先に npm ci を実行してください')
    font_io = VideoIO()
    try:
        css = re.sub(
            r'url\(([^)]+)\)',
            lambda match: f'url("{font_io.media_url(str(package / match[1]))}")',
            stylesheet.read_text(encoding='utf-8'),
        )
        yield css
    finally:
        font_io.close()


def write_wav(path, seconds, rate=16000, frequency=440):
    # Write one second at a time; long-fixture construction itself stays bounded.
    with wave.open(str(path), 'wb') as output:
        output.setparams((1, 2, rate, 0, 'NONE', 'not compressed'))
        for offset in range(int(np.ceil(seconds))):
            count = min(rate, round((seconds - offset) * rate))
            t = np.arange(count) / rate + offset
            # Distinct pulses allow testing that audio begins at t=0 without EXO's intro.
            gate = ((t >= .4) & (t < 1.1)) | (t >= 2)
            samples = (np.sin(2 * np.pi * frequency * t) * 12000 * gate).astype('<i2')
            output.writeframes(samples.tobytes())


@pytest.fixture(scope='module')
def video_environment(_pw, japanese_font_css):
    if not shutil.which('ffprobe') or not shutil.which('ffmpeg'):
        pytest.skip('ffprobe and ffmpeg are required for actual media verification')
    directory = PROJECT_ROOT / 'test-results' / 'video'
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / 'source.wav'
    write_wav(source, 3)
    background = directory / 'background.png'
    Image.new('RGB', (384, 216), (15, 30, 65)).save(background)
    for name, color in [('speaker1.png', (240, 40, 30)), ('speaker2.png', (30, 220, 60))]:
        image = Image.new('RGBA', (220, 360))
        ImageDraw.Draw(image).rounded_rectangle((0, 0, 219, 359), 35, fill=(*color, 255))
        image.save(directory / name)
    settings = {
        'audio_start_seconds': 0,
        'width': 1920, 'height': 1080, 'fps': 30, 'font': TEST_FONT, 'font_size': 54,
        'max_chars_per_line': 8, 'align': 4, 'pos_y': 300,
        'background_image': str(background),
        'speaker_images': [
            {'file': str(directory / 'speaker1.png'), 'x': -550, 'y': 0, 'scale': 100},
            {'file': str(directory / 'speaker2.png'), 'x': 550, 'y': 0, 'scale': 100},
        ],
        'speaker_colors': ['ffffff', '00ffff'], 'speaker_edge_colors': ['000000'],
    }
    project = directory / 'legacy.awproj'
    project.write_text(json.dumps({
        'version': 1, 'source_file': str(source), 'segments': SEGMENTS,
        'exo_settings': settings, 'preview_index': 0,
    }), encoding='utf-8')
    control = directory / 'dialogs.json'
    control.write_text(json.dumps({'save': str(directory / 'source.mp4'), 'open': str(project)}), encoding='utf-8')
    env = os.environ.copy()
    env.update({
        'AVIUTL_WHISPER_REMOTE_DEBUGGING_PORT': str(PORT), 'AVIUTL_WHISPER_HIDDEN': '1',
        'AVIUTL_WHISPER_SETTINGS_PATH': str(directory / 'settings.json'),
        'AVIUTL_TEST_VIDEO_CONTROL': str(control),
    })
    env.pop('WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS', None)
    with (directory / 'app.log').open('w', encoding='utf-8') as log:
        proc = subprocess.Popen([str(PROJECT_ROOT / '.venv/Scripts/python.exe'), str(Path(__file__).with_name('video_app.py'))],
                                cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 40
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    pytest.fail((directory / 'app.log').read_text(encoding='utf-8'))
                try:
                    urllib.request.urlopen(f'http://localhost:{PORT}/json/version', timeout=1).close()
                    break
                except OSError:
                    time.sleep(.25)
            browser = _pw.chromium.connect_over_cdp(f'http://localhost:{PORT}')
            page = browser.contexts[0].pages[0]
            page.wait_for_function('window.__aviutlWhisperReady === true', timeout=60000)
            page.add_style_tag(content=japanese_font_css)
            # Load every Unicode subset before measuring or drawing any text.
            page.evaluate('''async family => {
                const faces = [...document.fonts].filter(face => face.family.replace(/["']/g, '') === family);
                if (!faces.length) throw new Error('Test font faces were not registered');
                await Promise.all(faces.map(face => face.load()));
            }''', TEST_FONT)
            yield {'page': page, 'directory': directory, 'control': control, 'project': project,
                   'settings': settings, 'browser': browser}
            browser.close()
        finally:
            _stop_process(proc)


@pytest.fixture
def video_page(video_environment):
    context = video_environment
    page = context['page']
    page.evaluate("document.querySelector('#video-modal').classList.add('hidden')")
    page.evaluate('async () => { clearDirty(); await loadProject(); }')
    diagnostics = page.evaluate('''() => ({
        ready: canvasPreview?.ready === true,
        error: document.querySelector('#preview-error').textContent,
        font: document.querySelector('#exo-font').value,
        segments: previewSegments.length,
    })''')
    (context['directory'] / 'preview-diagnostics.json').write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    assert diagnostics['ready'] and not diagnostics['error'], diagnostics
    return context


def export_ui(context, name, source):
    output = context['directory'] / name
    context['control'].write_text(json.dumps({'save': str(output), 'open': str(context['project'])}), encoding='utf-8')
    page = context['page']
    page.evaluate("document.querySelector('#video-modal').classList.add('hidden')")
    page.locator('#btn-export-video').click()
    page.wait_for_function("document.querySelector('#video-audio').value === 'source'")
    page.locator('#video-audio').select_option(source)
    page.locator('#btn-video-start').click()
    page.wait_for_function('videoAbortController === null', timeout=120000)
    assert page.locator('#video-error').inner_text() == ''
    assert page.locator('#video-percent').inner_text() == '100%'
    assert '完了:' in page.locator('#video-status').inner_text()
    return output


def probe(path):
    return json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)]))


def frame(path, timepoint):
    data = subprocess.check_output(['ffmpeg', '-v', 'error', '-ss', str(timepoint), '-i', str(path), '-frames:v', '1', '-f', 'image2pipe', '-vcodec', 'png', '-threads', '1', '-'])
    from io import BytesIO
    return Image.open(BytesIO(data)).convert('RGB')


def test_source_mp4_from_legacy_project_and_preview_match(video_page):
    context, page = video_page, video_page['page']
    output = export_ui(context, 'source.mp4', 'source')
    data = probe(output)
    video = next(stream for stream in data['streams'] if stream['codec_type'] == 'video')
    sound = next(stream for stream in data['streams'] if stream['codec_type'] == 'audio')
    assert (video['codec_name'], video['width'], video['height'], video['r_frame_rate']) == ('h264', 1920, 1080, '30/1')
    assert sound['codec_name'] == 'aac'
    assert abs(float(data['format']['duration']) - 3) < .06
    page.evaluate("async () => { const video=document.querySelector('#video-result'); video.muted=true; await video.play(); }")
    page.wait_for_function("document.querySelector('#video-result').currentTime > .1")
    assert page.evaluate("document.querySelector('#video-result').videoWidth") == 1920
    page.evaluate("document.querySelector('#video-result').pause()")
    comparisons = []
    for timepoint, active in [(.25, (True, False)), (.75, (True, True)), (1.25, (False, True)), (1.75, (False, False)), (2.25, (True, False)), (2.75, (False, False))]:
        encoded_frame = frame(output, timepoint)
        encoded_frame.save(context['directory'] / f'frame-{timepoint}.png')
        for x, colored in zip([410, 1510], active):
            pixel = encoded_frame.getpixel((x, 540))
            assert (max(pixel) - min(pixel) > 30) is colored
        preview_url = page.evaluate('async time => { await renderPreviewImage(time); return document.querySelector("#preview-image").toDataURL(); }', timepoint)
        from io import BytesIO
        preview = Image.open(BytesIO(base64.b64decode(preview_url.split(',')[1]))).convert('RGB')
        preview.save(context['directory'] / f'preview-{timepoint}.png')
        error = np.array(ImageChops.difference(preview, encoded_frame)).mean()
        comparisons.append(float(error))
        assert error < 3  # Lossy H.264, plus .25s rounds to the next stored frame.
    pcm = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(output), '-vn', '-f', 'f32le', '-ac', '1', '-ar', '16000', '-'])
    samples = np.frombuffer(pcm, dtype='<f4')
    rms = lambda begin, end: float(np.sqrt(np.mean(samples[int(begin * 16000):int(end * 16000)] ** 2)))
    assert rms(.1, .3) < .01 and rms(.5, .9) > .1 and rms(1.3, 1.6) < .01
    (context['directory'] / 'source-verification.json').write_text(json.dumps({'probe': data, 'frame_mean_absolute_errors': comparisons, 'fileSystemAccess': page.evaluate("typeof showSaveFilePicker === 'function'")}, indent=2), encoding='utf-8')


def test_silent_single_speaker_subtitles(video_page):
    context, page = video_page, video_page['page']
    page.evaluate('''async () => {
        await pywebview.api.restore_segments([{start:0,end:0.7,text:'日本語字幕のみ',speaker:'Speaker 1'}]);
        applyExoSettingsToUI({...collectExoSettings(), background_image:'', speaker_images:[], width:640,height:360,font_size:32,pos_y:0});
        backgroundImage = ''; tachieData = [];
        await initExoPreview();
    }''')
    output = export_ui(context, 'silent.mp4', 'silence')
    data = probe(output)
    assert [stream['codec_name'] for stream in data['streams']] == ['h264']
    assert abs(float(data['format']['duration']) - .7) < .04
    assert np.asarray(frame(output, .2)).max() > 180


def test_existing_tts_audio_is_mixed_at_segment_timestamps(video_page):
    context, page = video_page, video_page['page']
    tts_dir = context['directory'] / 'tts'
    tts_dir.mkdir(exist_ok=True)
    settings = page.evaluate('''directory => {
        document.querySelector('#tts-output-dir').value = directory;
        document.querySelector('#tts-model-id').value = 'eleven_multilingual_v2';
        syncTtsGenerationMode();
        return collectTtsSettings();
    }''', str(tts_dir))
    manifest = {'version': 1, 'segments': {}}
    for index, (segment, duration) in enumerate(zip(SEGMENTS, [.7, 1.4, 1.3])):
        audio = tts_dir / f'{index}.wav'
        write_wav(audio, duration, frequency=440 + 220 * index)
        manifest['segments'][str(index)] = {
            'status': 'generated', 'audio_path': audio.name,
            'text_hash': elevenlabs_tts.compute_text_hash(segment, '', settings['model_id'], settings['output_format'], elevenlabs_tts.normalize_voice_settings(settings), False, 'auto'),
        }
    elevenlabs_tts.save_manifest(tts_dir, manifest)
    output = export_ui(context, 'tts.mp4', 'tts')
    data = probe(output)
    assert {stream['codec_name'] for stream in data['streams']} == {'h264', 'aac'}
    assert abs(float(data['format']['duration']) - 3.3) < .06
    pcm = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(output), '-vn', '-f', 'f32le', '-ac', '1', '-ar', '16000', '-'])
    samples = np.frombuffer(pcm, dtype='<f4')
    # Last generated clip starts at 2s and has its pulse at local .4s.
    assert np.sqrt(np.mean(samples[int(2.5 * 16000):int(2.8 * 16000)] ** 2)) > .1


def test_cancel_and_encoder_error_never_publish_partial_files(video_page):
    context, page = video_page, video_page['page']
    output = context['directory'] / 'cancel.mp4'
    output.write_bytes(b'previous movie')
    context['control'].write_text(json.dumps({'save': str(output), 'open': str(context['project'])}), encoding='utf-8')
    page.evaluate('''async () => {
        await pywebview.api.restore_segments([{start:0,end:60,text:'キャンセル',speaker:'Speaker 1'}]);
        await initExoPreview();
    }''')
    page.locator('#btn-export-video').click()
    page.locator('#video-audio').select_option('silence')
    page.locator('#btn-video-start').click()
    page.wait_for_function("document.querySelector('#video-progress').value > 0")
    page.locator('#btn-video-cancel').click()
    page.wait_for_function('videoAbortController === null', timeout=30000)
    assert 'キャンセル' in page.locator('#video-status').inner_text()
    assert output.read_bytes() == b'previous movie'
    assert not list(context['directory'].glob('*.mp4.part'))
    # Real codec configuration rejection (odd dimensions), before opening a file.
    page.locator('#video-width').fill('1919')
    page.locator('#btn-video-start').click()
    page.wait_for_function('videoAbortController === null')
    assert '偶数' in page.locator('#video-error').inner_text()
    assert page.locator('#video-percent').inner_text() != '100%'


def test_missing_fonts_images_and_tts_are_visible(video_page):
    context, page = video_page, video_page['page']
    page.evaluate("async () => { document.querySelector('#exo-font').add(new Option('missing-font-xyz', 'missing-font-xyz')); document.querySelector('#exo-font').value = 'missing-font-xyz'; await renderPreviewImage(); }")
    assert 'フォント' in page.locator('#preview-error').inner_text()
    page.evaluate('''async family => {
        document.querySelector('#exo-font').value=family;
        backgroundImage = 'C:/nonexistent/missing-image.png';
        await renderPreviewImage();
    }''', TEST_FONT)
    assert page.locator('#preview-error').inner_text()
    page.evaluate("backgroundImage = ''; document.querySelector('#tts-output-dir').value = 'C:/nonexistent/tts'")
    page.locator('#btn-export-video').click()
    page.locator('#video-audio').select_option('tts')
    page.locator('#btn-video-start').click()
    page.wait_for_function('videoAbortController === null')
    assert '不足' in page.locator('#video-error').inner_text()


def test_encoder_runtime_failure_reports_error_and_cleans_output(video_page):
    context, page = video_page, video_page['page']
    output = context['directory'] / 'encoder-failure.mp4'
    output.write_bytes(b'previous movie')
    context['control'].write_text(json.dumps({'save': str(output), 'open': str(context['project'])}), encoding='utf-8')
    page.evaluate('''() => {
        window.__savedVideoEncoder = window.VideoEncoder;
        window.VideoEncoder = class extends window.__savedVideoEncoder {
            encode() { throw new Error('test encoder rejected frame'); }
        };
    }''')
    try:
        page.locator('#btn-export-video').click()
        page.wait_for_function("document.querySelector('#video-audio').value === 'source'")
        page.locator('#video-audio').select_option('silence')
        page.locator('#btn-video-start').click()
        page.wait_for_function('videoAbortController === null', timeout=30000)
        assert page.locator('#video-error').inner_text()
        assert '失敗' in page.locator('#video-status').inner_text()
        assert page.locator('#video-percent').inner_text() != '100%'
        assert output.read_bytes() == b'previous movie'
        assert not list(context['directory'].glob('*.mp4.part'))
    finally:
        page.evaluate('() => { window.VideoEncoder = window.__savedVideoEncoder; }')


def test_bgm_closing_preview_export_and_project_roundtrip(video_page):
    context, page = video_page, video_page['page']
    directory = context['directory']
    music = {}
    for name, frequency, seconds in [('opening', 220, .6), ('one', 660, .4),
                                     ('two', 880, .7), ('ending', 1100, .8), ('jingle', 1320, 2)]:
        path = directory / f'{name}.wav'
        with wave.open(str(path), 'wb') as stream:
            stream.setparams((1, 2, 48000, 0, 'NONE', 'not compressed'))
            samples = np.sin(2 * np.pi * frequency * np.arange(round(seconds * 48000)) / 48000) * .3
            stream.writeframes((samples * 32767).astype('<i2').tobytes())
        music[name] = str(path)
    closing = directory / 'closing.png'
    Image.new('RGB', (640, 360), (230, 180, 30)).save(closing)
    config = {**context['settings'], 'width': 640, 'height': 360, 'font_size': 24,
              'speaker_images': [], 'pos_y': 0, 'audio_start_seconds': .5,
              'bgm_start_file': music['opening'], 'bgm_files': [music['one'], music['two']],
              'bgm_volume': 25, 'closing_image': str(closing), 'closing_jingle': music['jingle'],
              'outro_seconds': .5, 'closing_seconds': 1}
    page.evaluate('async config => { applyExoSettingsToUI(config); await initExoPreview(); }', config)
    timing = page.evaluate('previewPlan.timing')
    assert (timing['mainEnd'], timing['closingStart'], timing['duration']) == (3.5, 4, 5)
    output = export_ui(context, 'with-bgm-closing.mp4', 'source')
    assert abs(float(probe(output)['format']['duration']) - 5) < .06
    pixels = frame(output, 4.2)
    assert np.max(np.abs(np.asarray(pixels).astype(float) - [230, 180, 30])) < 5
    preview_url = page.evaluate('async () => { await renderPreviewImage(4.2); return document.querySelector("#preview-image").toDataURL(); }')
    from io import BytesIO
    preview = Image.open(BytesIO(base64.b64decode(preview_url.split(',')[1]))).convert('RGB')
    assert np.array(ImageChops.difference(preview, pixels)).mean() < 2
    pcm = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(output), '-vn', '-f', 'f32le', '-af', 'pan=mono|c0=c0', '-ar', '48000', '-'])
    decoded = np.frombuffer(pcm, dtype='<f4')

    def amplitude(samples, frequency, rate=48000):
        t = np.arange(len(samples)) / rate
        return float(2 * abs(np.sum(samples * np.exp(-2j * np.pi * frequency * t))) / len(samples))

    # BGM starts before speech and follows the 25% volume. Speech's original
    # .4s pulse is shifted by the .5s intro. Jingle runs at 100% after the fade.
    assert .06 < amplitude(decoded[4800:14400], 220) < .09
    assert amplitude(decoded[4800:14400], 440) < .01
    assert amplitude(decoded[48000:57600], 440) > .3
    assert amplitude(decoded[201600:220800], 1320) > .27
    assert amplitude(decoded[201600:220800], 440) < .01
    rms = lambda begin, end: float(np.sqrt(np.mean(decoded[round(begin * 48000):round(end * 48000)] ** 2)))
    assert rms(3.6, 3.7) > 3 * rms(3.91, 3.98)
    preview_pcm = page.evaluate('''async () => {
        await renderPreviewImage(.1);
        await toggleVideoPreview();
        const player = previewPlayer;
        const samples = Array.from([...player.sources][0].buffer.getChannelData(0));
        const state = player.context.state;
        await stopVideoPreview();
        return { samples, state, stopped: previewPlayer === null };
    }''')
    assert preview_pcm['state'] == 'running' and preview_pcm['stopped']
    assert .07 < amplitude(np.array(preview_pcm['samples']), 220) < .08
    ending_pcm = page.evaluate('''async () => {
        await renderPreviewImage(4.1);
        await toggleVideoPreview();
        const samples = Array.from([...previewPlayer.sources][0].buffer.getChannelData(0));
        await stopVideoPreview();
        return samples;
    }''')
    assert amplitude(np.array(ending_pcm), 1320) > .29
    assert not page.locator('#preview-error').inner_text()
    # Native-dialog-backed .awproj save/load restores every production media
    # field; importing preferences in between must not overwrite those values.
    saved = directory / 'media-roundtrip.awproj'
    context['control'].write_text(json.dumps({'save': str(saved), 'open': str(context['project'])}), encoding='utf-8')
    page.evaluate('''async () => {
        const alert = window.alert; window.alert = () => {};
        try { await saveProjectAs(); } finally { window.alert = alert; }
    }''')
    assert json.loads(saved.read_text(encoding='utf-8'))['exo_settings']['closing_image'] == str(closing)
    context['control'].write_text(json.dumps({'save': str(saved), 'open': str(saved)}), encoding='utf-8')
    page.evaluate('async () => { applyExoMediaToUI(); clearDirty(); await loadProject(); }')
    restored = page.evaluate('collectExoSettings()')
    for key in ['bgm_start_file', 'bgm_files', 'bgm_volume', 'closing_image', 'closing_jingle', 'outro_seconds', 'closing_seconds']:
        assert restored[key] == config[key]
    page.evaluate("applySettingsToUi({exo:{closing_image:'ignored.png',bgm_volume:99}})")
    assert page.evaluate('collectProjectMediaSettings()') == {key: restored[key] for key in [
        'bgm_start_file', 'bgm_end_file', 'bgm_files', 'bgm_volume', 'outro_seconds',
        'closing_jingle', 'closing_image', 'closing_seconds']}
    preferences = page.evaluate('''async () => {
        await pywebview.api.save_settings(collectAllSettings());
        return await pywebview.api.load_settings();
    }''')
    assert 'closing_image' not in preferences['exo'] and 'bgm_files' not in preferences['exo']
    # A selected ending song replaces the ordinary playlist from max(0, main-30).
    page.evaluate('async path => { exoMediaFiles.bgm_end_file = path; await renderPreviewImage(.1); }', music['ending'])
    output = export_ui(context, 'with-ending-song.mp4', 'silence')
    pcm = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(output), '-vn', '-f', 'f32le', '-af', 'pan=mono|c0=c0', '-ar', '48000', '-'])
    samples = np.frombuffer(pcm, dtype='<f4')
    assert amplitude(samples[4800:14400], 1100) > .06
    assert amplitude(samples[4800:14400], 220) < .01
    context['control'].write_text(json.dumps({'save': str(output), 'open': str(context['project'])}), encoding='utf-8')


@pytest.mark.slow
def test_ten_minute_audio_video_memory_is_bounded(video_page):
    context, page = video_page, video_page['page']
    audio = context['directory'] / 'ten-minute.wav'
    write_wav(audio, 600, rate=8000)
    output = context['directory'] / 'ten-minute.mp4'
    project = context['directory'] / 'ten-minute.awproj'
    project.write_text(json.dumps({
        'version': 1, 'source_file': str(audio),
        'segments': [{'start': 0, 'end': 600, 'text': '10分間の計測', 'speaker': 'Speaker 1'}],
        'exo_settings': {'audio_start_seconds': 0, 'font': TEST_FONT, 'width': 160, 'height': 90, 'fps': 30, 'font_size': 14, 'max_chars_per_line': 10, 'speaker_images': []},
    }), encoding='utf-8')
    context['control'].write_text(json.dumps({'save': str(output), 'open': str(project)}), encoding='utf-8')
    page.evaluate('async () => { clearDirty(); await loadProject(); }')
    page.evaluate('''async () => {
        window.__longDone = false; window.__longError = null; window.__longFrame = 0;
        const destination = await pywebview.api.select_video_output();
        const source = await pywebview.api.get_video_audio('source', {});
        window.__longRun = VideoTools.exportVideo({
            bridge: pywebview.api, token:destination.token,
            segments:structuredClone(previewSegments),settings:collectExoSettings(),assets:{},clips:source.clips,
            onProgress: (value) => { window.__longFrame = Math.min(18000, Math.round(value / .98 * 18000)); },
        }).then(result => { window.__longResult = result; }, error => { window.__longError = String(error); })
          .finally(() => { window.__longDone = true; });
    }''')
    session = context['browser'].contexts[0].new_cdp_session(page)
    session.send('Performance.enable')
    samples = []
    next_checkpoint = 1800
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        status = page.evaluate('({done:window.__longDone,frame:window.__longFrame})')
        if status['frame'] >= next_checkpoint or status['done']:
            session.send('HeapProfiler.collectGarbage')
            metrics = {metric['name']: metric['value'] for metric in session.send('Performance.getMetrics')['metrics']}
            samples.append({'frame': status['frame'], 'heap': metrics['JSHeapUsedSize']})
            next_checkpoint = status['frame'] + 1800
        if status['done']:
            break
        page.wait_for_timeout(250)
    assert page.evaluate('window.__longDone'), '10-minute export timed out'
    assert page.evaluate('window.__longError') is None
    result = page.evaluate('window.__longResult')
    assert result['frames'] == 18000
    assert result['stats']['maxChunk'] <= 256 * 1024
    data = probe(output)
    assert abs(float(data['format']['duration']) - 600) < .08
    assert {stream['codec_name'] for stream in data['streams']} == {'h264', 'aac'}
    # First minute is warm-up. GC-normalized heap must not grow with total frames.
    measured = [sample['heap'] for sample in samples]
    assert len(measured) >= 3
    assert max(measured) - min(measured) < 32 * 1024 * 1024
    (context['directory'] / 'memory-verification.json').write_text(json.dumps({'samples': samples, 'result': result, 'probe': data}, indent=2), encoding='utf-8')
    session.detach()
