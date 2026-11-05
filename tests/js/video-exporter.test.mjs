import test from 'node:test';
import assert from 'node:assert/strict';
import { bridgeWritable, CHUNK_SIZE, requireSuccess, checkCapabilities } from '../../src/aviutl_whisper/web/video-exporter.js';

test('stream writes are split, awaited and retain correct append offsets', async () => {
    const calls = [];
    let running = 0, maximum = 0;
    const bridge = {
        async write_video_chunk(token, offset, data) {
            maximum = Math.max(maximum, ++running);
            await new Promise(resolve => setTimeout(resolve, 2));
            calls.push({ token, offset, length: Buffer.from(data, 'base64').length });
            running--;
            return { success: true };
        },
    };
    const stats = { bytes: 0, maxChunk: 0 };
    const stream = bridgeWritable(bridge, 'selected', null, stats);
    const writer = stream.getWriter();
    await writer.write({ position: 0, data: new Uint8Array(CHUNK_SIZE * 2 + 31) });
    await writer.close();
    assert.deepEqual(calls.map(call => call.offset), [0, CHUNK_SIZE, CHUNK_SIZE * 2]);
    assert.equal(maximum, 1);
    assert.equal(stats.bytes, CHUNK_SIZE * 2 + 31);
    assert.equal(stats.maxChunk, CHUNK_SIZE);
});

test('cancellation stops further bridge writes, while I/O errors propagate', async () => {
    const controller = new AbortController();
    let calls = 0;
    const writer = bridgeWritable({ async write_video_chunk() {
        calls++; controller.abort(); return { success: true };
    } }, 'selected', controller.signal, { bytes: 0, maxChunk: 0 }).getWriter();
    await assert.rejects(writer.write({ position: 0, data: new Uint8Array(CHUNK_SIZE + 1) }), { name: 'AbortError' });
    assert.equal(calls, 1);
    assert.throws(() => requireSuccess({ success: false, error: 'disk full' }), /disk full/);
});

test('unsupported codec environment fails before a file can be opened', async () => {
    await assert.rejects(checkCapabilities({ width: 1919 }, false), /偶数/);
    await assert.rejects(checkCapabilities({}, true), /WebCodecs/);
});
