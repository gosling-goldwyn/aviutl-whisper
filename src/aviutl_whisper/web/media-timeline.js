import { normalizeSettings, sceneTiming } from './canvas-renderer.js';

// One immutable plan is shared by preview and export. Only metadata is expanded;
// decoded audio remains streamed by AudioMixer.
export function buildMediaTimeline(segments, settings, speech, media, random = Math.random) {
    const s = normalizeSettings(settings);
    const audioEnd = speech.reduce((end, clip) => Math.max(end, clip.start + clip.duration), 0);
    const timing = sceneTiming(segments, audioEnd, s);
    const clips = speech.map(clip => ({ ...clip, start: clip.start + timing.intro,
        end: clip.start + timing.intro + clip.duration, gain: 1 }));
    const gain = s.bgm_volume / 100;
    const get = path => {
        const clip = media[path];
        if (!clip || !Number.isFinite(clip.duration) || clip.duration <= 0) throw new Error(`音声尺を取得できません: ${path}`);
        return clip;
    };
    const add = (path, start, end, fadeOut = 0) => {
        const source = get(path);
        const fadeStart = Math.max(start, end - fadeOut);
        // Repeat short music without caching decoded PCM. Fade uses absolute
        // scene time, so crossing a loop boundary never resets the envelope.
        for (let cursor = start; cursor < end - 1e-9; cursor += source.duration) {
            clips.push({ ...source, start: cursor, end: Math.min(end, cursor + source.duration),
                gain, fadeStart, fadeEnd: end });
        }
    };
    const endingStart = Math.max(0, timing.mainEnd - 30);
    const cutoff = s.bgm_end_file ? endingStart : timing.closingStart;
    const fade = Math.min(cutoff, s.bgm_end_file ? 1.5 : timing.hasClosing ? s.outro_seconds : 0);
    const tracks = [...new Set(s.bgm_files || [])];
    if (!tracks.length && s.bgm_start_file) tracks.push(s.bgm_start_file);
    let cursor = 0, previous = null, bag = [], first = s.bgm_start_file;
    while (cursor < cutoff - 1e-9 && (first || tracks.length)) {
        if (!first && !bag.length) {
            bag = [...tracks];
            for (let i = bag.length - 1; i > 0; i--) {
                const j = Math.floor(random() * (i + 1));
                [bag[i], bag[j]] = [bag[j], bag[i]];
            }
            if (bag.length > 1 && bag[0] === previous) [bag[0], bag[1]] = [bag[1], bag[0]];
        }
        const path = first || bag.shift();
        first = ''; previous = path;
        const end = Math.min(cutoff, cursor + get(path).duration);
        const finalEnd = end >= cutoff - fade ? cutoff : end;
        add(path, cursor, finalEnd, finalEnd === cutoff ? fade : 0);
        cursor = finalEnd;
    }
    if (s.bgm_end_file) add(s.bgm_end_file, endingStart, timing.closingStart,
        timing.closingStart - timing.mainEnd);
    if (s.closing_jingle) {
        const source = get(s.closing_jingle);
        clips.push({ ...source, start: timing.closingStart,
            end: Math.min(timing.duration, timing.closingStart + source.duration), gain: 1 });
    }
    return { timing, clips: clips.sort((a, b) => a.start - b.start) };
}

export function clipGain(clip, time) {
    if (time < clip.start || time >= clip.end) return 0;
    const fade = clip.fadeEnd > clip.fadeStart
        ? Math.max(0, Math.min(1, (clip.fadeEnd - time) / (clip.fadeEnd - clip.fadeStart))) : 1;
    return (clip.gain ?? 1) * fade;
}
