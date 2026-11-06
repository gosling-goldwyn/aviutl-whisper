import test from 'node:test';
import assert from 'node:assert/strict';
import { buildMediaTimeline, clipGain } from '../../src/aviutl_whisper/web/media-timeline.js';

const music = Object.fromEntries(['opening', 'one', 'two', 'ending', 'jingle'].map(path =>
    [path, { url: path, origin: 0, duration: path === 'jingle' ? 20 : 4 }]));

test('scene timing offsets speech, bases ending song on main content, then closes', () => {
    const plan = buildMediaTimeline([{ start: 0, end: 40 }], {
        audio_start_seconds: 1.5, bgm_start_file: 'opening', bgm_files: ['one', 'two'],
        bgm_end_file: 'ending', bgm_volume: 25, closing_image: 'end.png', closing_jingle: 'jingle',
        outro_seconds: 1.5, closing_seconds: 15,
    }, [{ url: 'speech', origin: 0, start: 2, duration: 42 }], music, () => .99);
    assert.equal(plan.timing.mainEnd, 45.5);
    assert.equal(plan.timing.closingStart, 47);
    assert.equal(plan.timing.duration, 62);
    const speech = plan.clips.find(c => c.url === 'speech');
    assert.deepEqual([speech.start, speech.end, speech.gain], [3.5, 45.5, 1]);
    assert.deepEqual(plan.clips.filter(c => ['opening', 'one', 'two'].includes(c.url)).map(c => c.url),
        ['opening', 'one', 'two', 'one']);
    const lastBgm = plan.clips.find(c => c.start === 12);
    assert.equal(lastBgm.end, 15.5);
    assert.equal(clipGain(lastBgm, 14), .25);
    assert.ok(Math.abs(clipGain(lastBgm, 14.75) - .125) < 1e-8);
    const ending = plan.clips.filter(c => c.url === 'ending');
    assert.equal(ending[0].start, 15.5);
    assert.equal(ending.at(-1).end, 47);
    const jingle = plan.clips.find(c => c.url === 'jingle');
    assert.deepEqual([jingle.start, jingle.end, jingle.gain], [47, 62, 1]);
});

test('looping opening fades continuously across loops and silent main retains music', () => {
    const { clips, timing } = buildMediaTimeline([{ start: 0, end: 10 }], {
        audio_start_seconds: 0, bgm_start_file: 'opening', closing_image: 'end.png',
        outro_seconds: 3, closing_seconds: 2,
    }, [], music);
    assert.equal(timing.duration, 15);
    assert.deepEqual(clips.map(c => [c.start, c.end]), [[0, 4], [4, 8], [8, 12], [12, 13]]);
    assert.ok(clipGain(clips[2], 11.9) > clipGain(clips[3], 12));
    assert.equal(clipGain(clips[3], 13), 0);
    assert.ok(Math.abs(clipGain(clips[3], 12.5) - .2 / 6) < 1e-8);
});

test('short main starts ending song at zero and jingle is trimmed without looping', () => {
    const { clips, timing } = buildMediaTimeline([{ start: 0, end: .1 }], {
        audio_start_seconds: 0, bgm_start_file: 'opening', bgm_end_file: 'ending',
        closing_jingle: 'jingle', outro_seconds: 0, closing_seconds: .5, bgm_volume: 0,
    }, [], music);
    assert.equal(timing.frames, 18);
    assert.deepEqual(clips.map(c => c.url), ['ending', 'jingle']);
    assert.equal(clips[0].start, 0);
    assert.equal(clipGain(clips[0], .05), 0);
    assert.equal(clips[1].end, .6);
    assert.throws(() => buildMediaTimeline([{ start: 0, end: 1 }], { bgm_start_file: 'absent' }, [], music), /absent/);
});
