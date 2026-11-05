import { AudioMixer } from './video-exporter.js';

// Schedule a small PCM window ahead of the AudioContext clock. The same mixer,
// clip gains and scene times are used by MP4; no full-length decoded buffer.
export class PreviewPlayer {
    constructor(plan, onFrame, onEnded, onError) {
        this.plan = plan;
        this.onFrame = onFrame;
        this.onEnded = onEnded;
        this.onError = onError;
        this.sources = new Set();
        this.controller = new AbortController();
    }

    async start(time) {
        this.startTime = time;
        this.context = new AudioContext({ sampleRate: 48000 });
        await this.context.resume();
        this.controller.signal.throwIfAborted();
        this.mixer = new AudioMixer(this.plan.clips, this.controller.signal, time);
        this.offset = Math.round(time * 48000);
        this.end = Math.round(this.plan.timing.duration * 48000);
        // Decode the first block before starting the clock.
        await this.queue(true);
        this.tick();
        void this.pump();
    }

    async queue(first = false) {
        if (this.offset >= this.end) return;
        const offset = this.offset, count = Math.min(4800, this.end - offset);
        const sample = await this.mixer.block(offset, count);
        try {
            this.controller.signal.throwIfAborted();
            const buffer = this.context.createBuffer(2, count, 48000);
            for (let channel = 0; channel < 2; channel++) {
                sample.copyTo(buffer.getChannelData(channel), { planeIndex: channel, format: 'f32-planar' });
            }
            if (first) this.epoch = this.context.currentTime + .08 - offset / 48000;
            const when = this.epoch + offset / 48000;
            if (when < this.context.currentTime - .02) throw new Error('プレビュー音声の処理が再生に追いつきません。停止して再生し直してください');
            const source = this.context.createBufferSource();
            source.buffer = buffer;
            source.connect(this.context.destination);
            this.sources.add(source);
            source.onended = () => { this.sources.delete(source); source.disconnect(); };
            source.start(Math.max(this.context.currentTime, when));
            this.offset += count;
        } finally { sample.close(); }
    }

    tick() {
        if (this.controller.signal.aborted) return;
        const time = Math.max(this.startTime, Math.min(this.plan.timing.duration, this.context.currentTime - this.epoch));
        this.onFrame(time);
        if (time >= this.plan.timing.duration) { this.onEnded(); return; }
        this.frame = requestAnimationFrame(() => this.tick());
    }

    async pump() {
        try {
            while (!this.controller.signal.aborted && this.offset < this.end) {
                if (this.epoch + this.offset / 48000 < this.context.currentTime + .3) await this.queue();
                else await new Promise(resolve => setTimeout(resolve, 20));
            }
        } catch (error) { if (!this.controller.signal.aborted) this.onError(error); }
    }

    async stop() {
        this.controller.abort();
        cancelAnimationFrame(this.frame);
        for (const source of this.sources) { source.stop(); source.disconnect(); }
        this.sources.clear();
        await Promise.allSettled([this.mixer?.dispose(), this.context?.close()]);
    }
}
