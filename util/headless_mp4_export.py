#!/usr/bin/env python3
"""Headless MP4 export for aviutl-whisper projects (.awproj).

The app's built-in MP4 export runs inside the Windows GUI (WebView2 +
WebCodecs). This utility reproduces the same export on machines where the
GUI cannot run (Linux servers, CI, ...), following:

- spec/mp4-export.md  (Canvas renderer / encoding parameters)
- spec/exo-export.md   (BGM / closing timeline rules)
- src/aviutl_whisper/web/canvas-renderer.js (layout, timing, subtitles)

It renders with Pillow and encodes with ffmpeg (H.264 / AAC). The picture
is a faithful re-implementation, not a pixel-identical copy of the Canvas
output. Only projects whose audio_mode is "source" are supported (the
project's own audio file is used as the main track).

Usage:
    python3 headless_mp4_export.py PROJECT.awproj \
        --media-dir DIR_WITH_MEDIA [...] \
        --out OUTPUT.mp4 [--font FONT.ttf] [--seed N] [--keep-work]

Media files referenced by the project (Windows paths inside .awproj /
exo_settings) are resolved by file name inside the media directories.
"""

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SR = 48000  # spec: AAC 48 kHz stereo


# ---------------------------------------------------------------- media lookup

def build_media_index(media_dirs):
    index = {}
    for d in media_dirs:
        for root, _dirs, files in os.walk(d):
            for name in files:
                index.setdefault(name, os.path.join(root, name))
    return index


def resolve(index, win_path, required=True):
    if not win_path:
        return None
    name = str(win_path).replace("\\", "/").rstrip("/").split("/")[-1]
    path = index.get(name)
    if path is None and required:
        raise SystemExit(f"media file not found in --media-dir: {name} (from {win_path})")
    return path


# ---------------------------------------------------------------- fonts

def find_font_by_family(family, extra_dirs):
    """Scan font directories for a font whose family name matches."""
    from fontTools.ttLib import TTFont, TTCollection

    candidates = []
    search_dirs = list(extra_dirs) + [
        os.path.expanduser("~/.fonts"),
        os.path.expanduser("~/.local/share/fonts"),
        "/usr/share/fonts",
        "/usr/local/share/fonts",
    ]
    for d in search_dirs:
        if not d or not os.path.isdir(d):
            continue
        for root, _dirs, files in os.walk(d):
            for name in files:
                if name.lower().endswith((".ttf", ".otf", ".ttc")):
                    candidates.append(os.path.join(root, name))
    for path in candidates:
        try:
            if path.lower().endswith(".ttc"):
                coll = TTCollection(path, lazy=True)
                fonts = coll.fonts
            else:
                fonts = [TTFont(path, lazy=True)]
            for tt in fonts:
                names = {n.toUnicode() for n in tt["name"].names if n.nameID in (1, 16)}
                if family in names:
                    return path
        except Exception:
            continue
    return None


class FontBook:
    def __init__(self, main_path, fallback_path):
        self.main_path = main_path
        self.fallback_path = fallback_path
        self._fonts = {}
        self._cmap = None
        if main_path:
            from fontTools.ttLib import TTFont

            tt = TTFont(main_path, fontNumber=0, lazy=True)
            self._cmap = tt.getBestCmap()
            tt.close()

    def _load(self, path, size):
        key = (path, size)
        if key not in self._fonts:
            self._fonts[key] = ImageFont.truetype(path, size, index=0)
        return self._fonts[key]

    def font_for(self, ch, size):
        if self.main_path and self._cmap is not None and ord(ch) in self._cmap:
            return self._load(self.main_path, size)
        if self.fallback_path:
            return self._load(self.fallback_path, size)
        return ImageFont.load_default()


# ---------------------------------------------------------------- timing
# Port of canvas-renderer.js timelineDuration() / sceneTiming().

def timeline_duration(segments, audio_duration, fps):
    end = audio_duration
    for seg in segments:
        end = max(end, seg["end"])
    frames = max(1, math.ceil(end * fps - 1e-8))
    return frames / fps, end


def scene_timing(segments, audio_duration, s):
    body, content_end = timeline_duration(segments, audio_duration, s["fps"])
    intro = round(s["audio_start_seconds"] * s["fps"]) / s["fps"]
    main_end = intro + body
    has_closing = bool(s.get("closing_image") or s.get("closing_jingle"))
    closing_start = main_end + (round(s["outro_seconds"] * s["fps"]) / s["fps"] if has_closing else 0)
    duration = closing_start + (math.ceil(s["closing_seconds"] * s["fps"] - 1e-8) / s["fps"] if has_closing else 0)
    return {
        "intro": intro, "main_end": main_end, "closing_start": closing_start,
        "has_closing": has_closing, "duration": duration,
        "frames": round(duration * s["fps"]), "content_end": content_end,
    }


# ---------------------------------------------------------------- audio

def decode_audio(path):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy()


def probe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, check=True, text=True).stdout
    return float(out.strip())


def mix_audio(s, timing, voice_path, bgm_start, bgm_files, bgm_end, jingle_path, out_wav, seed=0):
    """BGM / closing rules from spec/exo-export.md, mixed at 48 kHz stereo."""
    n = timing["frames"] * (SR // s["fps"])
    mix = np.zeros((n, 2), dtype=np.float32)
    voice = decode_audio(voice_path)
    off = int(timing["intro"] * SR)
    ln = min(len(voice), n - off)
    mix[off:off + ln] += voice[:ln]

    if bgm_start or bgm_files:
        bgm = np.zeros((n, 2), dtype=np.float32)
        switch_i = int((timing["main_end"] - 30.0) * SR) if bgm_end else int(timing["main_end"] * SR)
        switch_i = max(0, min(switch_i, n))
        rng = random.Random(seed)

        def shuffled_cycle(prev_last=None):
            cyc = list(bgm_files)
            rng.shuffle(cyc)
            if prev_last is not None and len(cyc) > 1 and cyc[0] == prev_last:
                cyc[0], cyc[1] = cyc[1], cyc[0]
            return cyc

        # Region A: start song, then shuffled list (looping).
        seq = ([bgm_start] if bgm_start else []) + shuffled_cycle()
        pos = idx = 0
        while pos < switch_i:
            if idx >= len(seq):
                if not bgm_files:
                    seq.append(bgm_start)  # start song only: repeat it
                else:
                    seq.extend(shuffled_cycle(prev_last=seq[-1]))
            track = decode_audio(seq[idx])
            take = min(len(track), switch_i - pos)
            bgm[pos:pos + take] += track[:take]
            pos += take
            idx += 1
            if take < len(track):
                break
        if bgm_end:
            # Previous track fades out over the 1.5 s before the switch.
            fade = min(int(1.5 * SR), switch_i)
            ramp = np.linspace(1.0, 0.0, fade, dtype=np.float32)
            bgm[switch_i - fade:switch_i] *= ramp[:, None]
            # Region B: end song until the closing section, outro fade-out.
            closing_i = int(timing["closing_start"] * SR)
            end_track = decode_audio(bgm_end)
            pos = switch_i
            while pos < closing_i:
                take = min(len(end_track), closing_i - pos)
                bgm[pos:pos + take] += end_track[:take]
                pos += take
            main_end_i = int(timing["main_end"] * SR)
            out_fade = closing_i - main_end_i
            if out_fade > 0:
                ramp = np.linspace(1.0, 0.0, out_fade, dtype=np.float32)
                bgm[main_end_i:closing_i] *= ramp[:, None]
        mix += bgm * (s["bgm_volume"] / 100.0)

    if jingle_path:
        closing_i = int(timing["closing_start"] * SR)
        jingle = decode_audio(jingle_path)
        ln = min(len(jingle), n - closing_i)
        mix[closing_i:closing_i + ln] += jingle[:ln]

    np.clip(mix, -1.0, 1.0, out=mix)
    pcm = (mix * 32767).astype(np.int16)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ac", "2", "-ar", str(SR),
         "-i", "-", out_wav], input=pcm.tobytes(), check=True)


# ---------------------------------------------------------------- video

def wrap_text(text, max_chars):
    out = []
    for line in str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        chars = list(line)
        if not max_chars or not chars:
            out.append(line)
        else:
            for i in range(0, len(chars), max_chars):
                out.append("".join(chars[i:i + max_chars]))
    return out


def hexcol(v):
    v = str(v).lstrip("#")
    return tuple(int(v[i:i + 2], 16) for i in (0, 2, 4))


class StateRenderer:
    def __init__(self, s, segments, slots, paths, fontbook):
        self.s = s
        self.segs = segments
        self.slots = slots  # speaker name -> slot index
        self.W, self.H = s["width"], s["height"]
        self.fontbook = fontbook
        bg = Image.open(paths["background"]).convert("RGB") if paths.get("background") else None
        if bg is None:
            self.bg = Image.new("RGB", (self.W, self.H), (0, 0, 0))
        else:
            sc = max(self.W / bg.width, self.H / bg.height)
            if sc != 1.0:
                bg = bg.resize((round(bg.width * sc), round(bg.height * sc)), Image.LANCZOS)
                x = (bg.width - self.W) // 2
                y = (bg.height - self.H) // 2
                bg = bg.crop((x, y, x + self.W, y + self.H))
            self.bg = bg
        self.closing_img = None
        if paths.get("closing_image"):
            self.closing_img = Image.new("RGB", (self.W, self.H), (0, 0, 0))
            eye = Image.open(paths["closing_image"]).convert("RGB")
            self.closing_img.paste(eye, ((self.W - eye.width) // 2, (self.H - eye.height) // 2))
        self.chars = []
        for img_cfg, path in zip(s["speaker_images"], paths["speaker_images"]):
            if not path:
                self.chars.append(None)
                continue
            im = Image.open(path).convert("RGBA")
            scale = img_cfg.get("scale", 100) / 100.0
            w, h = im.width * scale, im.height * scale
            resized = im.resize((max(1, round(w)), max(1, round(h))), Image.LANCZOS)
            gray_rgb = resized.convert("RGB").convert("L").convert("RGB")
            gray = Image.merge("RGBA", (*gray_rgb.split(), resized.split()[3]))
            x0 = round(self.W / 2 + img_cfg.get("x", 0) - w / 2)
            y0 = round(self.H / 2 + img_cfg.get("y", 0) - h / 2)
            self.chars.append((resized, gray, x0, y0))
        self._sub_cache = {}

    def subtitle_image(self, idx):
        if idx in self._sub_cache:
            return self._sub_cache[idx]
        s = self.s
        seg = self.segs[idx]
        slot = self.slots[seg.get("speaker") or "Speaker 1"]
        size = s["font_size"]
        lines = wrap_text(seg["text"], s["max_chars_per_line"])
        probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
        widths = [sum(probe.textlength(ch, font=self.fontbook.font_for(ch, size)) for ch in line)
                  for line in lines]
        step = max(1, size + s["spacing_y"])
        block_h = size + (len(lines) - 1) * step
        ax = self.W / 2 + s["pos_x"]
        ay = self.H / 2 + s["pos_y"]
        col, row = s["align"] % 3, s["align"] // 3
        # Canvas: line x starts at ax - col*lineWidth/2, block top at ay - row*blockH/2.
        starts = [ax - col * w / 2 for w in widths]
        top = ay - row * block_h / 2
        pad = 12
        left = min(starts) - pad
        right = max(st + w for st, w in zip(starts, widths)) + pad
        img = Image.new("RGBA", (max(1, math.ceil(right - left)), block_h + 2 * pad), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        colors = s["speaker_colors"] or ["ffffff"]
        edges = s["speaker_edge_colors"] or ["000000"]
        fill = hexcol(colors[slot % len(colors)]) + (255,)
        edge = hexcol(edges[slot % len(edges)]) + (255,)
        stroke_w = max(2, round(size / 12))
        for li, line in enumerate(lines):
            cx = starts[li] - left
            y = pad + li * step
            for ch in line:
                f = self.fontbook.font_for(ch, size)
                if s["soft_edge"]:
                    d.text((cx, y), ch, font=f, fill=fill, stroke_width=stroke_w, stroke_fill=edge)
                else:
                    d.text((cx, y), ch, font=f, fill=fill)
                cx += probe.textlength(ch, font=f) + s["spacing_x"]
        result = (img, (round(left), round(top - pad)))
        self._sub_cache[idx] = result
        return result

    def compose(self, key, timing):
        kind, active, colors = key
        if kind == "closing":
            return (self.closing_img or Image.new("RGB", (self.W, self.H), (0, 0, 0))).copy()
        img = self.bg.copy()
        for slot, entry in enumerate(self.chars):
            if entry is None:
                continue
            color_im, gray_im, x0, y0 = entry
            use = color_im if colors[slot] else gray_im
            img.paste(use, (x0, y0), use)
        for idx in active:
            simg, pos = self.subtitle_image(idx)
            img.paste(simg, pos, simg)
        return img

    def state_at(self, t, timing):
        s = self.s
        if timing["has_closing"] and t >= timing["closing_start"]:
            return ("closing", (), ())
        ct = t - timing["intro"]
        active = tuple(i for i, g in enumerate(self.segs) if g["start"] <= ct < g["end"])
        if t < timing["intro"] or t >= timing["main_end"]:
            n = max(len(self.chars), 1)
            colors = tuple(True for _ in range(n))
        else:
            speaking = {self.slots[g.get("speaker") or "Speaker 1"] for g in
                        (self.segs[i] for i in active)}
            colors = tuple(slot in speaking for slot in range(len(self.chars)))
        return ("main", active, colors)

    def render_states(self, timing, states_dir):
        os.makedirs(states_dir, exist_ok=True)
        fps = self.s["fps"]
        total = timing["frames"]
        runs = []
        cur, start = None, 0
        for f in range(total):
            key = self.state_at(f / fps, timing)
            if key != cur:
                if cur is not None:
                    runs.append((cur, f - start))
                cur, start = key, f
        runs.append((cur, total - start))
        rendered = {}
        lines = []
        for key, frames in runs:
            if key not in rendered:
                path = os.path.join(states_dir, f"state_{len(rendered):03d}.png")
                self.compose(key, timing).save(path)
                rendered[key] = path
            lines.append(f"file '{rendered[key]}'")
            lines.append(f"duration {frames / fps:.6f}")
        lines.append(f"file '{rendered[runs[-1][0]]}'")
        list_path = os.path.join(states_dir, "..", "states.txt")
        with open(list_path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        return list_path, len(runs), len(rendered)


# ---------------------------------------------------------------- main

DEFAULTS = {
    "width": 1920, "height": 1080, "fps": 30, "font": "", "font_size": 80,
    "spacing_x": 0, "spacing_y": 0, "display_speed": 0, "align": 7,
    "bold": False, "italic": False, "soft_edge": True, "pos_x": 0, "pos_y": 480,
    "max_chars_per_line": 20, "speaker_colors": [], "speaker_edge_colors": [],
    "speaker_images": [], "background_image": "",
    "bgm_start_file": "", "bgm_end_file": "", "bgm_files": [],
    "closing_jingle": "", "closing_image": "",
    "audio_start_seconds": 1.5, "bgm_volume": 20.0,
    "outro_seconds": 1.5, "closing_seconds": 15,
}


def main():
    ap = argparse.ArgumentParser(description="Headless MP4 export for aviutl-whisper .awproj projects")
    ap.add_argument("project", help=".awproj file")
    ap.add_argument("--media-dir", action="append", required=True,
                    help="directory containing the referenced media (repeatable)")
    ap.add_argument("--audio", help="main audio file (default: project source_file, resolved by name)")
    ap.add_argument("--font", help="subtitle font file (default: look up the settings font family)")
    ap.add_argument("--out", required=True, help="output .mp4")
    ap.add_argument("--seed", type=int, default=20261106, help="BGM shuffle seed (default: fixed)")
    ap.add_argument("--bitrate", default="8M", help="video bitrate (default: 8M, per spec for 1080p30)")
    ap.add_argument("--keep-work", action="store_true", help="keep intermediate files")
    ap.add_argument("--no-encode", action="store_true", help="stop after audio mix + state rendering")
    args = ap.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise SystemExit(f"{tool} is required")

    with open(args.project, encoding="utf-8") as fh:
        project = json.load(fh)
    if project.get("audio_mode", "source") != "source":
        raise SystemExit("only audio_mode='source' projects are supported")
    segments = project["segments"]
    s = {**DEFAULTS, **project.get("exo_settings", {})}
    if s["display_speed"]:
        raise SystemExit("display_speed > 0 (typewriter subtitles) is not supported yet")

    index = build_media_index(args.media_dir)
    voice = args.audio or resolve(index, project.get("source_file", ""))
    paths = {
        "background": resolve(index, s["background_image"], required=False),
        "closing_image": resolve(index, s["closing_image"], required=False),
        "speaker_images": [resolve(index, im.get("file", ""), required=False)
                           for im in s["speaker_images"]],
    }
    bgm_start = resolve(index, s["bgm_start_file"], required=False)
    bgm_end = resolve(index, s["bgm_end_file"], required=False)
    bgm_files = [resolve(index, p) for p in s["bgm_files"]]
    jingle = resolve(index, s["closing_jingle"], required=False)

    slots = {name: i for i, name in enumerate(sorted({g.get("speaker") or "Speaker 1" for g in segments}))}

    audio_dur = probe_duration(voice)
    timing = scene_timing(segments, audio_dur, s)
    print(f"audio {audio_dur:.3f}s / total {timing['duration']:.3f}s / {timing['frames']} frames "
          f"@ {s['fps']}fps, {len(segments)} segments")

    font_path = args.font or (find_font_by_family(s["font"], []) if s["font"] else None)
    fallback = find_font_by_family("Noto Sans CJK JP", [])
    fontbook = FontBook(font_path, fallback)
    if s["font"] and not font_path:
        print(f"warning: font '{s['font']}' not found; using fallback for all glyphs", file=sys.stderr)

    work = tempfile.mkdtemp(prefix="awproj-mp4-")
    try:
        mixed = os.path.join(work, "mixed.wav")
        mix_audio(s, timing, voice, bgm_start, bgm_files, bgm_end, jingle, mixed, seed=args.seed)
        renderer = StateRenderer(s, segments, slots, paths, fontbook)
        list_path, runs, distinct = renderer.render_states(timing, os.path.join(work, "states"))
        print(f"states: {runs} runs, {distinct} distinct images")
        if args.no_encode:
            print(f"work dir kept (--no-encode): {work}")
            return
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        # NOTE: use the fps filter (not -r) to expand the concat timestamps
        # to CFR; -r on a concat input mis-counts frames. Cap the frame
        # count so the total length is exact.
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
             "-i", mixed, "-map", "0:v:0", "-map", "1:a:0",
             "-vf", f"fps={s['fps']}", "-vsync", "cfr", "-frames:v", str(timing["frames"]),
             "-c:v", "libx264", "-preset", "veryfast", "-b:v", args.bitrate,
             "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "192k", "-ar", str(SR),
             "-movflags", "+faststart", args.out], check=True)
        print(f"wrote {args.out}")
    finally:
        if not args.keep_work and not args.no_encode:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
