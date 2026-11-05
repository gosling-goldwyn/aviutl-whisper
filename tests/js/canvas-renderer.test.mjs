import test from 'node:test';
import assert from 'node:assert/strict';
import { activeSegments, CanvasRenderer, normalizeSettings, speakerIndex, timelineDuration, wrapText } from '../../src/aviutl_whisper/web/canvas-renderer.js';

const segments = [
    { start: 0, end: 1, speaker: 'Speaker 1', text: '日本語字幕\n二行目' },
    { start: 0.5, end: 1.5, speaker: 'Speaker 2', text: '同時発話' },
    { start: 2, end: 2.01, speaker: 'Speaker 1', text: '短い' },
];

test('half-open intervals preserve overlap, gaps, exact endpoints and short segments', () => {
    assert.equal(activeSegments(segments, 0).length, 1);
    assert.equal(activeSegments(segments, 0.5).length, 2);
    assert.deepEqual(activeSegments(segments, 1).map(s => s.speaker), ['Speaker 2']);
    assert.equal(activeSegments(segments, 1.5).length, 0);
    assert.equal(activeSegments(segments, 2).length, 1);
    assert.equal(activeSegments(segments, 2.01).length, 0);
});

test('duration rounds upwards once, preserving audio tails and final frame', () => {
    assert.deepEqual(timelineDuration([{ start: 0, end: 0.1 }], 0, 30), { frames: 3, duration: 0.1, contentDuration: 0.1 });
    assert.equal(timelineDuration(segments, 5.01, 30).frames, 151);
    assert.equal(timelineDuration(segments, 1, 30).frames, 61);
    assert.equal(timelineDuration([], 0, 30).frames, 1);
    assert.throws(() => timelineDuration([{ start: 1, end: 1 }]), /時刻/);
    assert.throws(() => timelineDuration([], NaN), /尺/);
});

test('wrapping preserves explicit and empty lines and Unicode code points', () => {
    assert.deepEqual(wrapText('あいう\n\nえ😀お', 2), ['あい', 'う', '', 'え😀', 'お']);
    assert.deepEqual(wrapText('あいう\nえお', 0), ['あいう', 'えお']);
    assert.equal(speakerIndex('Speaker 10'), 9);
    assert.equal(speakerIndex('Speaker 0'), 0);
});

test('invalid resolution, non-finite coordinates and animation settings fail explicitly', () => {
    for (const settings of [{ width: -1 }, { fps: 0 }, { pos_x: Infinity }, { display_speed: -1 }, { align: 9 }]) {
        assert.throws(() => normalizeSettings(settings));
    }
});

function fixture(settings = {}, inputSegments = segments) {
    const operations = [];
    const ctx = {
        resetTransform() {}, fillRect() {},
        drawImage(image, ...rect) { operations.push({ image: image.name, filter: this.filter, rect }); },
        measureText(char) { return { width: char.length * (this.font.includes('MS UI Gothic') ? 10 : this.font.includes('serif') ? 6 : 8) }; },
        strokeText(char, x, y) { operations.push({ edge: this.strokeStyle, char, x, y }); },
        fillText(char, x, y) { operations.push({ color: this.fillStyle, char, x, y }); },
    };
    const canvas = { getContext: () => ctx };
    const renderer = new CanvasRenderer({
        canvas, segments: inputSegments, settings: { audio_start_seconds: 0, width: 320, height: 180, font_size: 20, speaker_images: [
            { file: 'a.png', x: -50 }, { file: 'b.png', x: 50 },
        ], ...settings }, assets: { 'a.png': 'a.png', 'b.png': 'b.png', 'end.png': 'end.png' },
        imageLoader: async name => ({ name, width: 40, height: 80 }),
    });
    return { renderer, operations, ctx };
}

globalThis.document = { fonts: { load: async () => [] } };

test('renderer switches every speaker independently and seeking is deterministic', async () => {
    const { renderer, operations } = fixture();
    await renderer.initialize();
    renderer.renderFrame(0.75);
    assert.deepEqual(operations.filter(o => o.image).map(o => o.filter), ['none', 'none']);
    operations.length = 0;
    renderer.renderFrame(1.75);
    assert.deepEqual(operations.filter(o => o.image).map(o => o.filter), ['grayscale(1)', 'grayscale(1)']);
    operations.length = 0;
    renderer.renderFrame(0.25);
    const first = structuredClone(operations);
    operations.length = 0;
    renderer.renderFrame(0.25);
    assert.deepEqual(operations, first);
    assert.deepEqual(operations.filter(o => o.image).map(o => o.filter), ['none', 'grayscale(1)']);
    renderer.dispose();
});

test('subtitle color, edge, alignment, tracking and line spacing reach drawing operations', async () => {
    const { renderer, operations } = fixture({ align: 0, pos_x: -100, pos_y: -50,
        spacing_x: 3, spacing_y: 7, max_chars_per_line: 3,
        speaker_colors: ['123456'], speaker_edge_colors: ['abcdef'] });
    await renderer.initialize();
    renderer.renderFrame(0.1);
    const text = operations.filter(o => o.color);
    assert.equal(text[0].color, '#123456');
    assert.equal(text[0].x, 60);
    assert.equal(text[0].y, 40);
    assert.equal(text[1].x - text[0].x, 13);
    assert.equal(text[3].y - text[0].y, 27);
    assert.equal(operations.find(o => o.edge).edge, '#abcdef');
});

test('display speed reveals letters without shifting their full-text layout', async () => {
    const { renderer, operations } = fixture({ display_speed: 4 });
    await renderer.initialize();
    renderer.renderFrame(0);
    assert.equal(operations.filter(o => o.color).length, 0);
    operations.length = 0;
    renderer.renderFrame(0.25);
    const first = operations.find(o => o.color);
    assert.equal(operations.filter(o => o.color).length, 1);
    operations.length = 0;
    renderer.renderFrame(0.49);
    assert.deepEqual(operations.find(o => o.color), first);
});

test('missing images and failed font loads reject initialization', async () => {
    const { renderer } = fixture({ background_image: 'missing.png' });
    await assert.rejects(renderer.initialize(), /画像/);
    const second = fixture().renderer;
    document.fonts.load = async () => { throw new Error('font failed'); };
    await assert.rejects(second.initialize(), /font failed/);
    document.fonts.load = async () => [];
});

test('speaker slots match EXO sorting for sparse numbers and custom names', async () => {
    const { renderer, operations } = fixture({ speaker_colors: ['123456', 'abcdef'] }, [
        { start: 0, end: 1, speaker: 'Speaker 3', text: '疎な番号' },
        { start: 2, end: 3, speaker: 'Alice', text: '手動の話者名' },
    ]);
    await renderer.initialize();
    const state = renderer.renderFrame(.25);
    assert.deepEqual([...state.activeSpeakers], [1]);
    assert.deepEqual(operations.filter(o => o.image).map(o => o.filter), ['grayscale(1)', 'none']);
    assert.equal(operations.find(o => o.color).color, '#abcdef');
    operations.length = 0;
    renderer.renderFrame(2.25);
    assert.equal(operations.find(o => o.color).color, '#123456');
});

test('intro, main, outro and ending obey shared boundaries even after seeking backwards', async () => {
    const { renderer, operations } = fixture({ audio_start_seconds: .5,
        closing_image: 'end.png', outro_seconds: .5, closing_seconds: 1 },
    [{ start: 0, end: 1, text: '字幕', speaker: 'Speaker 1' }]);
    await renderer.initialize();
    renderer.renderFrame(.25);
    assert.equal(operations.filter(o => o.color).length, 0);
    assert.deepEqual(operations.filter(o => o.image).map(o => o.filter), ['none', 'none']);
    operations.length = 0;
    renderer.renderFrame(.5);
    assert.ok(operations.some(o => o.color));
    operations.length = 0;
    renderer.renderFrame(1.5);
    assert.equal(operations.filter(o => o.color).length, 0);
    assert.deepEqual(operations.filter(o => o.image).map(o => o.filter), ['none', 'none']);
    operations.length = 0;
    renderer.renderFrame(2);
    assert.deepEqual(operations.map(o => o.image), ['end.png']);
    operations.length = 0;
    renderer.renderFrame(.5);
    assert.ok(operations.some(o => o.color));
    assert.ok(!operations.some(o => o.image === 'end.png'));
});
