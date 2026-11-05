import {
    Output, Mp4OutputFormat, StreamTarget, CanvasSource, AudioSampleSource,
    AudioSample, AudioSampleSink, Input, UrlSource, ALL_FORMATS, Quality,
    canEncodeVideo, canEncodeAudio,
} from 'mediabunny';
import { CanvasRenderer, normalizeSettings, timelineDuration } from './canvas-renderer.js';

const RATE = 48000;
const CHANNELS = 2;
export const CHUNK_SIZE = 256 * 1024;

function check(signal) { signal?.throwIfAborted(); }

export function requireSuccess(result) {
    if (!result?.success) throw new Error(result?.error || 'Pythonブリッジとの通信に失敗しました');
    return result;
}

export async function checkCapabilities(settings, withAudio) {
    const s = normalizeSettings(settings);
    if (s.width % 2 || s.height % 2) throw new Error('H.264の解像度は幅・高さとも偶数で指定してください');
    if (!globalThis.VideoEncoder || (withAudio && !globalThis.AudioEncoder)) {
        throw new Error('このWebView2では必要なWebCodecsを使用できません。WebView2 Runtimeを更新してください');
    }
    const quality = new Quality({ bitrate: s.video_bitrate ?? Math.round(8e6 * s.width * s.height / (1920 * 1080) * s.fps / 30) });
    if (!await canEncodeVideo('avc', { width: s.width, height: s.height, frameRate: s.fps, quality })) {
        throw new Error(`H.264エンコードに対応していません（${s.width}×${s.height} / ${s.fps}fps）。WebView2とGPUドライバーを確認してください`);
    }
    const audioQuality = new Quality({ bitrate: 192000 });
    if (withAudio && !await canEncodeAudio('aac', { sampleRate: RATE, numberOfChannels: CHANNELS, quality: audioQuality })) {
        throw new Error('AACエンコード（48kHz・ステレオ）に対応していません。WebView2 Runtimeを更新してください');
    }
    return { quality, audioQuality, fileSystemAccess: typeof globalThis.showSaveFilePicker === 'function' };
}

function openInput(url) {
    return new Input({ source: new UrlSource(url, {
        maxCacheSize: 1024 * 1024, parallelism: 1,
        getRetryDelay: () => null, // A missing local file must fail instead of retrying forever.
    }), formats: ALL_FORMATS });
}

export async function inspectAudio(clips, signal) {
    const result = [];
    for (const clip of clips) {
        check(signal);
        if (!Number.isFinite(clip.start) || clip.start < 0) throw new Error('音声の開始時刻が不正です');
        const input = openInput(clip.url);
        const abort = () => input.dispose();
        signal?.addEventListener('abort', abort, { once: true });
        try {
            const track = await input.getPrimaryAudioTrack();
            if (!track || !await track.canDecode()) throw new Error('選択した音声をWebView2でデコードできません。WAVに変換して読み込み直してください');
            const origin = await track.getFirstTimestamp();
            const duration = await track.computeDuration() - origin;
            if (!Number.isFinite(duration) || duration <= 0) throw new Error('音声尺を取得できません');
            result.push({ ...clip, origin, duration });
        } finally {
            signal?.removeEventListener('abort', abort);
            input.dispose();
        }
    }
    return result.sort((a, b) => a.start - b.start);
}

// Streams decoded samples into fixed-size stereo PCM blocks. Inputs are opened only
// while audible and disposed at EOF. Memory depends on simultaneous clips, not length.
export class AudioMixer {
    constructor(clips, signal) {
        this.clips = clips;
        this.signal = signal;
        this.cursor = 0;
        this.active = new Set();
        this.disposed = false;
    }

    async open(clip) {
        if (this.active.size >= 32) throw new Error('同時に重なるTTS音声が32件を超えています');
        const input = openInput(clip.url);
        const state = { clip, input, iterator: null, sample: null, planes: null };
        this.active.add(state);
        const track = await input.getPrimaryAudioTrack();
        state.iterator = new AudioSampleSink(track).samples();
        await this.next(state);
        return state;
    }

    async next(state) {
        state.sample?.close();
        state.sample = null;
        state.planes = null;
        check(this.signal);
        const value = await state.iterator.next();
        if (this.disposed || this.signal?.aborted) {
            value.value?.close();
            check(this.signal);
            throw new Error('音声処理は終了しています');
        }
        check(this.signal);
        if (value.done) {
            state.input.dispose();
            this.active.delete(state);
            return;
        }
        state.sample = value.value;
        state.start = state.clip.start + state.sample.timestamp - state.clip.origin;
        state.end = state.start + state.sample.duration;
        state.planes = Array.from({ length: Math.min(2, state.sample.numberOfChannels) }, (_, channel) => {
            const plane = new Float32Array(state.sample.numberOfFrames);
            state.sample.copyTo(plane, { planeIndex: channel, format: 'f32-planar' });
            return plane;
        });
    }

    async block(offset, count) {
        check(this.signal);
        if (this.disposed) throw new Error('音声処理は終了しています');
        const start = offset / RATE, end = (offset + count) / RATE;
        const data = new Float32Array(count * CHANNELS);
        while (this.cursor < this.clips.length && this.clips[this.cursor].start < end) {
            await this.open(this.clips[this.cursor++]);
        }
        for (const state of [...this.active]) {
            while (state.sample && state.start < end) {
                const from = Math.max(0, Math.ceil((state.start - start) * RATE - 1e-7));
                const to = Math.min(count, Math.ceil((state.end - start) * RATE - 1e-7));
                for (let i = from; i < to; i++) {
                    // Linear resampling has bounded state and supports arbitrary input rates.
                    const position = Math.max(0, (start + i / RATE - state.start) * state.sample.sampleRate);
                    const left = Math.min(state.sample.numberOfFrames - 1, Math.floor(position));
                    const right = Math.min(state.sample.numberOfFrames - 1, left + 1);
                    const fraction = position - Math.floor(position);
                    for (let channel = 0; channel < CHANNELS; channel++) {
                        const plane = state.planes[Math.min(channel, state.planes.length - 1)];
                        data[channel * count + i] += plane[left] + (plane[right] - plane[left]) * fraction;
                    }
                }
                if (state.end <= end + 1e-9) await this.next(state); else break;
            }
        }
        for (let i = 0; i < data.length; i++) data[i] = Math.max(-1, Math.min(1, data[i]));
        return new AudioSample({ data, format: 'f32-planar', sampleRate: RATE, numberOfChannels: CHANNELS, timestamp: start });
    }

    async dispose() {
        this.disposed = true;
        const states = [...this.active];
        this.active.clear();
        for (const state of states) {
            state.sample?.close();
            state.input.dispose();
        }
        await Promise.allSettled(states.map(state => state.iterator?.return()));
    }
}

function base64(bytes) {
    let binary = '';
    for (let i = 0; i < bytes.length; i += 8192) binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
    return btoa(binary);
}

export function bridgeWritable(bridge, token, signal, stats) {
    return new WritableStream({
        async write(chunk) {
            for (let i = 0; i < chunk.data.length; i += CHUNK_SIZE) {
                check(signal);
                const part = chunk.data.subarray(i, i + CHUNK_SIZE);
                requireSuccess(await bridge.write_video_chunk(token, chunk.position + i, base64(part)));
                stats.bytes += part.length;
                stats.maxChunk = Math.max(stats.maxChunk, part.length);
            }
        },
    }, { highWaterMark: 1 });
}

export async function exportVideo({ bridge, token, segments, settings, assets, clips = [], signal, onProgress = () => {} }) {
    let output, mixer, renderer, committed = false, cancellation;
    const stats = { bytes: 0, maxChunk: 0, frames: 0 };
    const abort = () => {
        cancellation = output?.cancel().catch(() => {});
        // Disposing Inputs also interrupts pending HTTP reads and decoder operations.
        void mixer?.dispose();
    };
    signal?.addEventListener('abort', abort, { once: true });
    try {
        check(signal);
        onProgress(0, 'WebCodecsを確認中…');
        const config = await checkCapabilities(settings, clips.length > 0);
        check(signal);
        onProgress(0, '画像・フォントを読み込み中…');
        renderer = new CanvasRenderer({ canvas: document.createElement('canvas'), segments, settings, assets });
        await renderer.initialize();
        check(signal);
        onProgress(0, '音声を確認中…');
        const audio = await inspectAudio(clips, signal);
        const audioEnd = audio.reduce((end, clip) => Math.max(end, clip.start + clip.duration), 0);
        const s = renderer.settings;
        const timing = timelineDuration(segments, audioEnd, s.fps);
        requireSuccess(await bridge.begin_video_output(token));
        check(signal);
        // Fragmented MP4 bounds muxer memory and is strictly append-only.
        output = new Output({
            format: new Mp4OutputFormat({ fastStart: 'fragmented', minimumFragmentDuration: 1 }),
            target: new StreamTarget(bridgeWritable(bridge, token, signal, stats), { chunked: true, chunkSize: CHUNK_SIZE }),
        });
        const video = new CanvasSource(renderer.canvas, { codec: 'avc', quality: config.quality, keyFrameInterval: 2 });
        output.addVideoTrack(video, { frameRate: s.fps });
        let sound;
        if (audio.length) {
            sound = new AudioSampleSource({ codec: 'aac', quality: config.audioQuality });
            output.addAudioTrack(sound);
            mixer = new AudioMixer(audio, signal);
        }
        await output.start();
        let audioOffset = 0, lastUpdate = -Infinity;
        for (let i = 0; i < timing.frames; i++) {
            check(signal);
            renderer.renderFrame(i / s.fps);
            await video.add(i / s.fps, 1 / s.fps);
            if (sound) {
                const end = Math.round((i + 1) / s.fps * RATE);
                const sample = await mixer.block(audioOffset, end - audioOffset);
                try { await sound.add(sample); } finally { sample.close(); }
                audioOffset = end;
            }
            stats.frames++;
            if (performance.now() - lastUpdate >= 150 || i + 1 === timing.frames) {
                onProgress(0.98 * (i + 1) / timing.frames, `エンコード中 ${i + 1} / ${timing.frames} フレーム`);
                lastUpdate = performance.now();
            }
            if (i % 5 === 0) await new Promise(resolve => setTimeout(resolve, 0));
        }
        video.close();
        sound?.close();
        check(signal);
        onProgress(0.99, 'MP4を確定中…');
        await output.finalize();
        check(signal);
        const result = requireSuccess(await bridge.finish_video_output(token));
        committed = true;
        onProgress(1, `完了: ${result.path}`);
        return { ...result, ...timing, stats, fileSystemAccess: config.fileSystemAccess };
    } finally {
        signal?.removeEventListener('abort', abort);
        if (!committed) {
            if (output && output.state !== 'finalized') await output.cancel().catch(() => {});
            await cancellation;
        }
        await mixer?.dispose();
        renderer?.dispose();
        if (!committed) requireSuccess(await bridge.abort_video_output(token));
    }
}
