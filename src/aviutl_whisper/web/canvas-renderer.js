// Coordinates and font sizes always use the output's native resolution.
export const DEFAULT_SETTINGS = Object.freeze({
    width: 1920, height: 1080, fps: 30, font: 'MS UI Gothic', font_size: 34,
    spacing_x: 0, spacing_y: 0, display_speed: 0, align: 4,
    bold: false, italic: false, soft_edge: true, pos_x: 0, pos_y: 0,
    max_chars_per_line: 20, speaker_images: [], background_image: '',
    speaker_colors: ['ffffff', '00ffff', '00ff00', 'ff00ff', 'ffff00', 'ff8000', '8080ff', '80ff80'],
    speaker_edge_colors: ['000000'],
});

export function normalizeSettings(settings = {}) {
    const s = { ...DEFAULT_SETTINGS, ...settings };
    for (const key of ['width', 'height', 'fps', 'font_size', 'spacing_x', 'spacing_y',
        'display_speed', 'align', 'pos_x', 'pos_y', 'max_chars_per_line']) {
        if (!Number.isFinite(s[key])) throw new Error(`${key} が不正です`);
    }
    if (!Number.isInteger(s.width) || !Number.isInteger(s.height) || s.width < 2 || s.height < 2
        || s.width > 8192 || s.height > 8192 || s.fps <= 0 || s.fps > 240
        || s.font_size < 1 || s.font_size > 2048 || s.align < 0 || s.align > 8
        || !Number.isInteger(s.align) || s.display_speed < 0
        || !Number.isInteger(s.max_chars_per_line) || s.max_chars_per_line < 0) {
        throw new Error('解像度・fps・字幕設定の値が範囲外です');
    }
    return s;
}

export function speakerIndex(name) {
    const match = /^Speaker (\d+)$/.exec(name || 'Speaker 1');
    return match ? Math.max(0, Number(match[1]) - 1) : 0;
}

export function activeSegments(segments, time) {
    return segments.filter(segment => segment.start <= time && time < segment.end);
}

export function wrapText(text, maxChars) {
    return String(text).replace(/\r\n?/g, '\n').split('\n').flatMap(line => {
        const chars = Array.from(line);
        if (!maxChars || !chars.length) return [line];
        const result = [];
        for (let i = 0; i < chars.length; i += maxChars) result.push(chars.slice(i, i + maxChars).join(''));
        return result;
    });
}

export function timelineDuration(segments, audioDuration = 0, fps = 30) {
    if (!Number.isFinite(audioDuration) || audioDuration < 0 || !Number.isFinite(fps) || fps <= 0) {
        throw new Error('動画尺が不正です');
    }
    let end = audioDuration;
    for (const segment of segments) {
        if (!Number.isFinite(segment.start) || !Number.isFinite(segment.end)
            || segment.start < 0 || segment.end <= segment.start) throw new Error('セグメント時刻が不正です');
        end = Math.max(end, segment.end);
    }
    // Avoid an extra frame caused by floating-point multiplication at exact boundaries.
    const frames = Math.max(1, Math.ceil(end * fps - 1e-8));
    return { frames, duration: frames / fps, contentDuration: end };
}

function color(values, index, fallback) {
    const value = values?.length ? values[index % values.length] : fallback;
    return /^#?[0-9a-f]{6}$/i.test(value) ? `#${value.replace('#', '')}` : `#${fallback}`;
}

function fontSpec(s, family = JSON.stringify(s.font)) {
    return `${s.italic ? 'italic ' : ''}${s.bold ? 'bold ' : ''}${s.font_size}px ${family}`;
}

async function loadImage(url, path) {
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error('image request failed');
        // Blob decoding selects the first animation frame and freezes it. An
        // HTMLImageElement could advance GIF/APNG frames with wall-clock time.
        return await createImageBitmap(await response.blob());
    }
    catch { throw new Error(`画像を読み込めません: ${path}`); }
}

export class CanvasRenderer {
    constructor({ canvas, segments, settings, assets = {}, imageLoader = loadImage }) {
        this.canvas = canvas;
        this.settings = normalizeSettings(settings);
        this.segments = segments.map(segment => ({ ...segment }));
        timelineDuration(this.segments, 0, this.settings.fps);
        // Match export_exo: mapped speaker names are sorted into setting slots.
        // Numeric suffixes can be sparse, and manually edited names are valid.
        this.speakerSlots = new Map([...new Set(this.segments.map(segment => segment.speaker || 'Speaker 1'))]
            .sort().map((name, index) => [name, index]));
        this.assets = assets;
        this.imageLoader = imageLoader;
        this.images = new Map();
        this.events = this.segments.flatMap((s, i) => [
            { time: s.start, index: i, add: true }, { time: s.end, index: i, add: false },
        ]).sort((a, b) => a.time - b.time || Number(a.add) - Number(b.add) || a.index - b.index);
        this.active = new Set();
        this.cursor = 0;
        this.time = -Infinity;
        this.ready = false;
    }

    async initialize() {
        const s = this.settings;
        this.canvas.width = s.width;
        this.canvas.height = s.height;
        this.ctx = this.canvas.getContext('2d', { alpha: false });
        if (!this.ctx) throw new Error('Canvas 2Dを使用できません');
        await document.fonts.load(fontSpec(s), '日本語字幕ABC');
        // FontFaceSet.check alone accepts missing system fonts by silently substituting.
        const sample = '日本語字幕漢字 012345 WMij';
        const widths = ['monospace', 'serif'].map(fallback => {
            this.ctx.font = fontSpec(s, fallback);
            const base = this.ctx.measureText(sample).width;
            this.ctx.font = fontSpec(s, `${JSON.stringify(s.font)}, ${fallback}`);
            return Math.abs(this.ctx.measureText(sample).width - base);
        });
        if (widths.every(diff => diff < 0.01)) throw new Error(`フォントが見つかりません: ${s.font}`);
        const paths = [...new Set([s.background_image, ...s.speaker_images.map(image => image.file)].filter(Boolean))];
        try {
            for (const path of paths) {
                if (!this.assets[path]) throw new Error(`画像が見つかりません: ${path}`);
                this.images.set(path, await this.imageLoader(this.assets[path], path));
            }
            this.ready = true;
        } catch (error) { this.dispose(); throw error; }
    }

    renderFrame(time) {
        if (!this.ready || !Number.isFinite(time)) throw new Error('レンダラーが未初期化、または時刻が不正です');
        const s = this.settings, ctx = this.ctx;
        if (time < this.time) { this.active.clear(); this.cursor = 0; }
        while (this.cursor < this.events.length && this.events[this.cursor].time <= time) {
            const event = this.events[this.cursor++];
            if (event.add) this.active.add(event.index); else this.active.delete(event.index);
        }
        this.time = time;
        ctx.resetTransform();
        ctx.filter = 'none';
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, s.width, s.height);
        const background = this.images.get(s.background_image);
        if (background) {
            const scale = Math.max(s.width / background.width, s.height / background.height);
            const w = background.width * scale, h = background.height * scale;
            ctx.drawImage(background, (s.width - w) / 2, (s.height - h) / 2, w, h);
        }
        const visible = [...this.active].sort((a, b) => a - b).map(index => this.segments[index]);
        const speakers = new Set(visible.map(segment => this.speakerSlots.get(segment.speaker || 'Speaker 1')));
        s.speaker_images.forEach((image, index) => {
            const bitmap = this.images.get(image.file);
            if (!bitmap) return;
            const scale = Number(image.scale ?? 100) / 100;
            const x = Number(image.x ?? 0), y = Number(image.y ?? 0);
            if (![scale, x, y].every(Number.isFinite) || scale <= 0) throw new Error('立ち絵の座標・拡大率が不正です');
            const w = bitmap.width * scale, h = bitmap.height * scale;
            ctx.filter = speakers.has(index) ? 'none' : 'grayscale(1)';
            ctx.drawImage(bitmap, s.width / 2 + x - w / 2, s.height / 2 + y - h / 2, w, h);
        });
        ctx.filter = 'none';
        for (const segment of visible) this.drawSubtitle(segment, time);
        return { segments: visible, activeSpeakers: speakers };
    }

    drawSubtitle(segment, time) {
        const s = this.settings, ctx = this.ctx, index = this.speakerSlots.get(segment.speaker || 'Speaker 1');
        const lines = wrapText(segment.text, s.max_chars_per_line);
        ctx.font = fontSpec(s);
        ctx.textBaseline = 'top';
        ctx.textAlign = 'left';
        ctx.lineJoin = 'round';
        const spacing = s.spacing_x;
        const widths = lines.map(line => {
            const chars = Array.from(line);
            return Math.max(0, chars.reduce((width, char) => width + ctx.measureText(char).width, 0) + Math.max(0, chars.length - 1) * spacing);
        });
        const height = s.font_size + (lines.length - 1) * Math.max(1, s.font_size + s.spacing_y);
        const ax = s.width / 2 + s.pos_x, ay = s.height / 2 + s.pos_y;
        const col = s.align % 3, row = Math.floor(s.align / 3);
        const top = ay - row * height / 2;
        // AviUtl display_speed: characters per second; 0 displays the whole subtitle.
        // Layout uses the complete text so typewriter animation does not shift it.
        let remaining = s.display_speed > 0 ? Math.floor(Math.max(0, time - segment.start) * s.display_speed + 1e-8) : Infinity;
        ctx.fillStyle = color(s.speaker_colors, index, 'ffffff');
        ctx.strokeStyle = color(s.speaker_edge_colors, index, '000000');
        ctx.lineWidth = Math.max(2, s.font_size / 12);
        lines.forEach((line, lineIndex) => {
            let x = ax - col * widths[lineIndex] / 2;
            const y = top + lineIndex * Math.max(1, s.font_size + s.spacing_y);
            for (const char of Array.from(line)) {
                if (remaining-- > 0) {
                    if (s.soft_edge) ctx.strokeText(char, x, y);
                    ctx.fillText(char, x, y);
                }
                x += ctx.measureText(char).width + spacing;
            }
        });
    }

    dispose() {
        for (const image of this.images.values()) image.close?.();
        this.images.clear();
        this.ready = false;
    }
}
