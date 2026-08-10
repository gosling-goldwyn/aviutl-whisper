"""ElevenLabs TTS generation and external manifest management."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

DEFAULT_MODEL_ID = "eleven_multilingual_v2"
DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"
MANIFEST_FILENAME = "tts_manifest.json"
API_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
VOICES_URL = "https://api.elevenlabs.io/v2/voices"
MODELS_URL = "https://api.elevenlabs.io/v1/models"
DEFAULT_VOICE_SETTINGS = {
    "stability": 0.5,
    "similarity_boost": 0.75,
    "style": 0.0,
    "use_speaker_boost": True,
    "speed": 1.0,
}
PER_SEGMENT_CONTEXT = "per_segment_context"
CHUNK_V3 = "chunk_v3"
CONTEXT_MODELS = {"eleven_multilingual_v2", "eleven_flash_v2_5"}
DEFAULT_CHUNK_SETTINGS = {
    "max_chars_per_chunk": 1200,
    "max_segments_per_chunk": 8,
    "split_on_speaker_change": False,
}
DEFAULT_COMBINED_WAV_SILENCE_MS = 0
MODEL_UNSUPPORTED_FIELDS = {
    "eleven_v3": {
        "previous_text",
        "next_text",
        "previous_request_ids",
        "next_request_ids",
        "apply_language_text_normalization",
        "language_code",
    },
    "eleven_multilingual_v2": {
        "apply_language_text_normalization",
        "language_code",
    },
}
OPTIONAL_COMPAT_FIELDS = {
    "previous_text",
    "next_text",
    "previous_request_ids",
    "next_request_ids",
    "apply_language_text_normalization",
    "language_code",
    "apply_text_normalization",
}
COMPAT_ERROR_MARKERS = {
    "apply_language_text_normalization",
    "language_text_normalization_not_supported",
    "language_code",
    "previous_text",
    "next_text",
    "unsupported_model",
}


class ElevenLabsTtsError(RuntimeError):
    """User-facing TTS validation or generation error."""


def normalize_apply_text_normalization(value: Any) -> str | None:
    if value is None:
        return "auto"
    normalized = str(value).strip().lower()
    if normalized in {"auto", "on", "off"}:
        return normalized
    raise ElevenLabsTtsError(
        "apply_text_normalization はauto/on/offで指定してください"
    )


def build_elevenlabs_payload(
    *,
    text: str,
    model_id: str,
    previous_text: str | None = None,
    next_text: str | None = None,
    previous_request_ids: list[str] | None = None,
    next_request_ids: list[str] | None = None,
    voice_settings: dict | None = None,
    seed: int | None = None,
    apply_text_normalization: str | None = None,
    apply_language_text_normalization: bool = False,
    language_code: str | None = None,
) -> dict:
    """モデル互換性に従ってElevenLabsのbodyを構築する。"""
    payload: dict[str, Any] = {
        "text": text,
        "model_id": model_id,
    }
    if voice_settings:
        payload["voice_settings"] = voice_settings
    if seed is not None:
        payload["seed"] = int(seed)
    normalized_text_setting = normalize_apply_text_normalization(
        apply_text_normalization
    )
    if normalized_text_setting is not None:
        payload["apply_text_normalization"] = normalized_text_setting

    if model_id == "eleven_v3":
        return payload

    if previous_request_ids:
        payload["previous_request_ids"] = list(previous_request_ids[:3])
    elif previous_text:
        payload["previous_text"] = previous_text
    if next_request_ids:
        payload["next_request_ids"] = list(next_request_ids[:3])
    elif next_text:
        payload["next_text"] = next_text

    # 初回修正では全モデルでlanguage text normalizationを送らない。
    # 引数は設定互換性のため受け取るがpayloadには追加しない。
    _ = apply_language_text_normalization, language_code
    return payload


def sanitize_payload_for_model(payload: dict, model_id: str) -> dict:
    result = dict(payload)
    for key in MODEL_UNSUPPORTED_FIELDS.get(model_id, set()):
        result.pop(key, None)
    return result


def _value(segment: Any, name: str, default: Any = None) -> Any:
    if isinstance(segment, dict):
        return segment.get(name, default)
    return getattr(segment, name, default)


def _speaker(segment: Any) -> str:
    return str(_value(segment, "speaker") or "Speaker 1")


def _text(segment: Any) -> str:
    return str(_value(segment, "text", "") or "")


def _voice_ids(tts_settings: dict) -> dict[str, str]:
    raw = tts_settings.get("speaker_voice_ids", {})
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value).strip() for key, value in raw.items()}


def normalize_voice_settings(tts_settings: dict) -> dict:
    raw = tts_settings.get("voice_settings", {})
    if not isinstance(raw, dict):
        raw = {}
    result = dict(DEFAULT_VOICE_SETTINGS)
    for key in ("stability", "similarity_boost", "style", "speed"):
        if key in raw:
            try:
                result[key] = float(raw[key])
            except (TypeError, ValueError) as exc:
                raise ElevenLabsTtsError(
                    f"voice_settings.{key} は数値で指定してください"
                ) from exc
    for key in ("stability", "similarity_boost"):
        if not 0.0 <= result[key] <= 1.0:
            raise ElevenLabsTtsError(
                f"voice_settings.{key} は0.0〜1.0で指定してください"
            )
    if result["speed"] <= 0:
        raise ElevenLabsTtsError(
            "voice_settings.speed は0より大きい値を指定してください"
        )
    result["use_speaker_boost"] = bool(
        raw.get(
            "use_speaker_boost",
            DEFAULT_VOICE_SETTINGS["use_speaker_boost"],
        )
    )
    return result


def resolve_generation_mode(tts_settings: dict) -> str:
    """モデルに対して安全な生成方式を強制する。"""
    model_id = str(
        tts_settings.get("model_id") or DEFAULT_MODEL_ID
    ).strip()
    return CHUNK_V3 if model_id == "eleven_v3" else PER_SEGMENT_CONTEXT


def normalize_chunk_settings(tts_settings: dict) -> dict:
    raw = tts_settings.get("chunk_settings", {})
    if not isinstance(raw, dict):
        raw = {}
    try:
        max_chars = int(
            raw.get(
                "max_chars_per_chunk",
                DEFAULT_CHUNK_SETTINGS["max_chars_per_chunk"],
            )
        )
        max_segments = int(
            raw.get(
                "max_segments_per_chunk",
                DEFAULT_CHUNK_SETTINGS["max_segments_per_chunk"],
            )
        )
    except (TypeError, ValueError) as exc:
        raise ElevenLabsTtsError(
            "チャンク上限は整数で指定してください"
        ) from exc
    if max_chars <= 0 or max_segments <= 0:
        raise ElevenLabsTtsError(
            "チャンク上限は1以上で指定してください"
        )
    return {
        "max_chars_per_chunk": max_chars,
        "max_segments_per_chunk": max_segments,
        "split_on_speaker_change": bool(
            raw.get(
                "split_on_speaker_change",
                DEFAULT_CHUNK_SETTINGS["split_on_speaker_change"],
            )
        ),
    }


def normalize_combined_wav_silence_ms(tts_settings: dict) -> int:
    """Return a non-negative integer gap for combined WAV output."""
    try:
        value = int(tts_settings.get("combined_wav_silence_ms", 0))
    except (AttributeError, TypeError, ValueError):
        return DEFAULT_COMBINED_WAV_SILENCE_MS
    return max(0, value)


def compute_text_hash(
    segment: Any,
    voice_id: str,
    model_id: str,
    output_format: str,
    voice_settings: dict | None = None,
    apply_language_text_normalization: bool = True,
    apply_text_normalization: str | None = "auto",
) -> str:
    """Hash the fields that determine whether a segment must be regenerated."""
    payload = {
        "text": _text(segment),
        "speaker": _speaker(segment),
        "voice_id": voice_id,
        "model_id": model_id,
        "output_format": output_format,
        "voice_settings": voice_settings or dict(DEFAULT_VOICE_SETTINGS),
        "apply_language_text_normalization": bool(
            apply_language_text_normalization
        ),
        "apply_text_normalization": normalize_apply_text_normalization(
            apply_text_normalization
        ),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _chunk_script(segments: list[Any], indices: list[int]) -> str:
    return "\n".join(
        f"{_speaker(segments[index])}: {_text(segments[index])}"
        for index in indices
    )


def compute_chunk_hash(
    segments: list[Any],
    indices: list[int],
    tts_settings: dict,
) -> str:
    voice_ids = _voice_ids(tts_settings)
    payload = {
        "generation_mode": CHUNK_V3,
        "segment_indices": indices,
        "segments": [
            {
                "text": _text(segments[index]),
                "speaker": _speaker(segments[index]),
                "voice_id": voice_ids.get(_speaker(segments[index]), ""),
            }
            for index in indices
        ],
        "model_id": str(
            tts_settings.get("model_id") or DEFAULT_MODEL_ID
        ),
        "output_format": str(
            tts_settings.get("output_format") or DEFAULT_OUTPUT_FORMAT
        ),
        "voice_settings": normalize_voice_settings(tts_settings),
        "apply_language_text_normalization": False,
        "apply_text_normalization": normalize_apply_text_normalization(
            tts_settings.get("apply_text_normalization", "auto")
        ),
        "chunk_settings": normalize_chunk_settings(tts_settings),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_chunks(segments: list[Any], tts_settings: dict) -> list[dict]:
    """現在順を維持し、通常TTS用の単一話者チャンクを構築する。"""
    chunk_settings = normalize_chunk_settings(tts_settings)
    voice_ids = _voice_ids(tts_settings)
    max_chars = chunk_settings["max_chars_per_chunk"]
    max_segments = chunk_settings["max_segments_per_chunk"]
    grouped: list[list[int]] = []
    current: list[int] = []

    for index, segment in enumerate(segments):
        line = f"{_speaker(segment)}: {_text(segment)}"
        current_text = _chunk_script(segments, current) if current else ""
        exceeds_chars = bool(
            current
            and len(current_text) + 1 + len(line) > max_chars
        )
        exceeds_segments = len(current) >= max_segments
        # Text to Speech APIはURL上1 voice_idのため常に話者変更で分割する。
        changes_speaker = bool(
            current
            and _speaker(segments[current[-1]]) != _speaker(segment)
        )
        if current and (
            exceeds_chars or exceeds_segments or changes_speaker
        ):
            grouped.append(current)
            current = []
        current.append(index)
    if current:
        grouped.append(current)

    chunks = []
    for chunk_index, indices in enumerate(grouped):
        speakers = sorted({_speaker(segments[index]) for index in indices})
        chunks.append({
            "chunk_index": chunk_index,
            "segment_indices": list(indices),
            "text": _chunk_script(segments, indices),
            "speaker": speakers[0],
            "voice_ids": {
                speaker: voice_ids.get(speaker, "")
                for speaker in speakers
            },
            "text_hash": compute_chunk_hash(
                segments, indices, tts_settings
            ),
        })
    return chunks


def _safe_speaker_name(speaker: str) -> str:
    name = re.sub(r"[^0-9A-Za-z_-]+", "_", speaker.strip()).strip("_")
    return name or "Speaker"


def _audio_extension(output_format: str) -> str:
    prefix = output_format.split("_", 1)[0].lower()
    return {
        "mp3": ".mp3", "opus": ".opus", "pcm": ".pcm",
        "ulaw": ".ulaw", "alaw": ".alaw",
    }.get(prefix, ".bin")


def audio_filename(index: int, speaker: str, output_format: str) -> str:
    return (
        f"{index + 1:04d}_{_safe_speaker_name(speaker)}"
        f"{_audio_extension(output_format)}"
    )


def load_manifest(output_dir: str | os.PathLike[str]) -> dict:
    path = Path(output_dir) / MANIFEST_FILENAME
    if not path.exists():
        return {"version": 1, "segments": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("segments", {}), dict):
            raise ValueError("invalid manifest")
        data.setdefault("version", 1)
        data.setdefault("segments", {})
        return data
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ElevenLabsTtsError(
            f"TTS manifestの読み込みに失敗しました: {exc}"
        ) from exc


def save_manifest(output_dir: str | os.PathLike[str], manifest: dict) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST_FILENAME
    fd, temp_name = tempfile.mkstemp(
        prefix="tts_manifest_", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    return path


def _base_manifest(tts_settings: dict, existing: dict | None = None) -> dict:
    manifest = existing or {"version": 1, "segments": {}}
    manifest["version"] = 2
    manifest["model_id"] = str(
        tts_settings.get("model_id") or DEFAULT_MODEL_ID
    )
    manifest["output_format"] = str(
        tts_settings.get("output_format") or DEFAULT_OUTPUT_FORMAT
    )
    manifest["voice_settings"] = normalize_voice_settings(tts_settings)
    manifest["apply_language_text_normalization"] = False
    manifest["apply_text_normalization"] = (
        normalize_apply_text_normalization(
            tts_settings.get("apply_text_normalization", "auto")
        )
    )
    manifest.setdefault("segments", {})
    manifest.setdefault("chunks", {})
    return manifest


def _resolved_audio_path(output_dir: Path, entry: dict | None) -> Path | None:
    if not entry or not entry.get("audio_path"):
        return None
    path = Path(str(entry["audio_path"]))
    return path if path.is_absolute() else output_dir / path


def get_chunk_statuses(segments: list[Any], tts_settings: dict) -> list[dict]:
    output_value = str(tts_settings.get("output_dir") or "").strip()
    manifest = load_manifest(output_value) if output_value else {"chunks": {}}
    output_dir = Path(output_value) if output_value else Path()
    statuses = []
    for chunk in build_chunks(segments, tts_settings):
        entry = manifest.get("chunks", {}).get(str(chunk["chunk_index"]))
        audio_path = _resolved_audio_path(output_dir, entry)
        hash_matches = bool(
            entry and entry.get("text_hash") == chunk["text_hash"]
        )
        if entry and entry.get("status") == "error" and hash_matches:
            status = "error"
        elif not entry:
            status = "not_generated"
        elif (
            hash_matches
            and entry.get("status") == "generated"
            and audio_path is not None
            and audio_path.exists()
        ):
            status = "generated"
        else:
            status = "needs_regeneration"
        statuses.append({
            **chunk,
            "audio_path": str(audio_path) if audio_path else "",
            "status": status,
            "error": str(entry.get("error", "")) if entry else "",
        })
    return statuses


def get_segment_statuses(segments: list[Any], tts_settings: dict) -> list[dict]:
    if resolve_generation_mode(tts_settings) == CHUNK_V3:
        voice_ids = _voice_ids(tts_settings)
        segment_statuses: list[dict | None] = [None] * len(segments)
        for chunk in get_chunk_statuses(segments, tts_settings):
            for index in chunk["segment_indices"]:
                speaker = _speaker(segments[index])
                segment_statuses[index] = {
                    "index": index,
                    "speaker": speaker,
                    "text": _text(segments[index]),
                    "voice_id": voice_ids.get(speaker, ""),
                    "audio_path": chunk["audio_path"],
                    "status": chunk["status"],
                    "error": chunk["error"],
                    "chunk_index": chunk["chunk_index"],
                    "segment_indices": chunk["segment_indices"],
                }
        return [item for item in segment_statuses if item is not None]

    output_value = str(tts_settings.get("output_dir") or "").strip()
    manifest = load_manifest(output_value) if output_value else {"segments": {}}
    output_dir = Path(output_value) if output_value else Path()
    model_id = str(tts_settings.get("model_id") or DEFAULT_MODEL_ID)
    output_format = str(
        tts_settings.get("output_format") or DEFAULT_OUTPUT_FORMAT
    )
    voice_ids = _voice_ids(tts_settings)
    voice_settings = normalize_voice_settings(tts_settings)
    apply_normalization = False
    apply_text_normalization = normalize_apply_text_normalization(
        tts_settings.get("apply_text_normalization", "auto")
    )
    statuses: list[dict] = []

    for index, segment in enumerate(segments):
        speaker = _speaker(segment)
        voice_id = voice_ids.get(speaker, "")
        entry = manifest.get("segments", {}).get(str(index))
        expected_hash = compute_text_hash(
            segment,
            voice_id,
            model_id,
            output_format,
            voice_settings,
            apply_normalization,
            apply_text_normalization,
        )
        audio_path = _resolved_audio_path(output_dir, entry)
        if entry and entry.get("status") == "error":
            status = "error"
        elif not entry:
            status = "not_generated"
        elif (
            entry.get("text_hash") == expected_hash
            and entry.get("status") == "generated"
            and audio_path is not None
            and audio_path.exists()
        ):
            status = "generated"
        else:
            status = "needs_regeneration"
        statuses.append({
            "index": index,
            "speaker": speaker,
            "text": _text(segment),
            "voice_id": voice_id,
            "audio_path": str(audio_path) if audio_path else "",
            "status": status,
            "error": str(entry.get("error", "")) if entry else "",
        })
    return statuses


def get_combined_audio_items(
    segments: list[Any], tts_settings: dict
) -> tuple[list[dict], list[dict]]:
    """Return current generated audio in playback order and skipped items."""
    if resolve_generation_mode(tts_settings) == CHUNK_V3:
        statuses = get_chunk_statuses(segments, tts_settings)
        kind = "chunk"
    else:
        statuses = get_segment_statuses(segments, tts_settings)
        kind = "segment"

    available: list[dict] = []
    skipped: list[dict] = []
    for item in statuses:
        index = int(
            item["chunk_index"] if kind == "chunk" else item["index"]
        )
        audio_path = str(item.get("audio_path") or "")
        if (
            item.get("status") == "generated"
            and audio_path
            and Path(audio_path).is_file()
        ):
            available.append({
                "kind": kind,
                "index": index,
                "audio_path": audio_path,
                "segment_indices": list(
                    item.get("segment_indices", [index])
                ),
            })
        else:
            skipped.append({
                "kind": kind,
                "index": index,
                "status": str(item.get("status") or "not_generated"),
                "segment_indices": list(
                    item.get("segment_indices", [index])
                ),
            })
    return available, skipped


def export_combined_wav(
    segments: list[Any], tts_settings: dict, output_path: str | Path
) -> dict:
    """Concatenate current generated TTS audio and atomically save a WAV."""
    available, skipped = get_combined_audio_items(segments, tts_settings)
    if not available:
        raise ElevenLabsTtsError(
            "結合できる生成済みTTS音声がありません"
        )

    try:
        from pydub import AudioSegment

        combined = None
        frame_rate = channels = sample_width = None
        silence_ms = normalize_combined_wav_silence_ms(tts_settings)
        for item in available:
            audio = AudioSegment.from_file(item["audio_path"])
            if combined is None:
                frame_rate = audio.frame_rate
                channels = audio.channels
                sample_width = audio.sample_width
                combined = audio
                continue
            audio = audio.set_frame_rate(frame_rate)
            audio = audio.set_channels(channels)
            audio = audio.set_sample_width(sample_width)
            if silence_ms:
                silence = AudioSegment.silent(
                    duration=silence_ms, frame_rate=frame_rate
                )
                silence = silence.set_channels(channels)
                silence = silence.set_sample_width(sample_width)
                combined += silence
            combined += audio

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{destination.stem}.",
            suffix=".wav",
            dir=destination.parent,
        )
        os.close(fd)
        try:
            combined.export(temp_name, format="wav")
            os.replace(temp_name, destination)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
    except ElevenLabsTtsError:
        raise
    except Exception as exc:
        raise ElevenLabsTtsError(
            f"TTS結合WAVの保存に失敗しました: {exc}"
        ) from exc

    return {
        "path": str(destination),
        "included": len(available),
        "skipped": skipped,
    }


def _get_json(
    url: str,
    api_key: str,
    opener: Callable[..., Any] | None = None,
    timeout: float = 30.0,
) -> Any:
    if not api_key.strip():
        raise ElevenLabsTtsError("ElevenLabs API Keyを入力してください")
    request = urllib.request.Request(
        url,
        headers={"xi-api-key": api_key},
        method="GET",
    )
    open_fn = opener or urllib.request.urlopen
    try:
        response = open_fn(request, timeout=timeout)
        if hasattr(response, "__enter__"):
            with response as opened:
                payload = opened.read()
        else:
            payload = response.read()
        return json.loads(payload.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ElevenLabsTtsError(
            f"ElevenLabs APIエラー ({exc.code})"
        ) from exc
    except urllib.error.URLError as exc:
        raise ElevenLabsTtsError(
            f"ElevenLabsへの接続に失敗しました: {exc.reason}"
        ) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ElevenLabsTtsError(
            "ElevenLabs APIの応答を解析できませんでした"
        ) from exc


def list_voices(
    api_key: str,
    opener: Callable[..., Any] | None = None,
) -> list[dict]:
    """Return normalized ElevenLabs voices for GUI selection."""
    raw_voices = []
    url = VOICES_URL
    seen_tokens = set()
    while True:
        payload = _get_json(url, api_key, opener=opener)
        if not isinstance(payload, dict):
            break
        page_voices = payload.get("voices", [])
        if isinstance(page_voices, list):
            raw_voices.extend(page_voices)
        token = payload.get("next_page_token")
        if not payload.get("has_more") or not token or token in seen_tokens:
            break
        seen_tokens.add(token)
        url = VOICES_URL + "?" + urllib.parse.urlencode({
            "next_page_token": str(token)
        })
    voices = []
    for item in raw_voices:
        if not isinstance(item, dict) or not item.get("voice_id"):
            continue
        voices.append({
            "voice_id": str(item["voice_id"]),
            "name": str(item.get("name") or item["voice_id"]),
            "preview_url": str(item.get("preview_url") or ""),
        })
    return sorted(voices, key=lambda voice: voice["name"].casefold())


def list_models(
    api_key: str,
    opener: Callable[..., Any] | None = None,
) -> list[dict]:
    """Return only models that support text-to-speech."""
    payload = _get_json(MODELS_URL, api_key, opener=opener)
    raw_models = (
        payload.get("models", [])
        if isinstance(payload, dict)
        else payload if isinstance(payload, list) else []
    )
    models = []
    for item in raw_models:
        if (
            not isinstance(item, dict)
            or not item.get("model_id")
            or item.get("can_do_text_to_speech") is not True
        ):
            continue
        models.append({
            "model_id": str(item["model_id"]),
            "name": str(item.get("name") or item["model_id"]),
        })
    return sorted(models, key=lambda model: model["name"].casefold())


def _request_audio(
    text: str,
    voice_id: str,
    model_id: str,
    output_format: str,
    api_key: str,
    previous_text: str | None,
    next_text: str | None,
    voice_settings: dict,
    apply_language_text_normalization: bool,
    apply_text_normalization: str | None = "auto",
    include_context: bool = True,
    opener: Callable[..., Any] | None = None,
    timeout: float = 120.0,
) -> bytes:
    url = API_URL.format(voice_id=urllib.parse.quote(voice_id, safe=""))
    url += "?" + urllib.parse.urlencode({"output_format": output_format})
    payload = build_elevenlabs_payload(
        text=text,
        model_id=model_id,
        previous_text=previous_text if include_context else None,
        next_text=next_text if include_context else None,
        voice_settings=voice_settings,
        apply_text_normalization=apply_text_normalization,
        apply_language_text_normalization=(
            apply_language_text_normalization
        ),
    )
    payload = sanitize_payload_for_model(payload, model_id)
    open_fn = opener or urllib.request.urlopen

    def send(current_payload: dict) -> bytes:
        request = urllib.request.Request(
            url,
            data=json.dumps(
                current_payload, ensure_ascii=False
            ).encode("utf-8"),
            headers={
                "xi-api-key": api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        response = open_fn(request, timeout=timeout)
        if hasattr(response, "__enter__"):
            with response as opened:
                return opened.read()
        return response.read()

    def http_error_detail(exc: urllib.error.HTTPError) -> str:
        try:
            return exc.read().decode(
                "utf-8", errors="replace"
            )[:500]
        except Exception:
            return ""

    try:
        return send(payload)
    except urllib.error.HTTPError as exc:
        detail = http_error_detail(exc)
        lower_detail = detail.lower()
        should_retry = (
            exc.code == 400
            and any(
                marker in lower_detail
                for marker in COMPAT_ERROR_MARKERS
            )
        )
        if should_retry:
            retry_payload = {
                key: value
                for key, value in payload.items()
                if key not in OPTIONAL_COMPAT_FIELDS
            }
            try:
                return send(retry_payload)
            except urllib.error.HTTPError as retry_exc:
                retry_detail = http_error_detail(retry_exc)
                message = (
                    f"ElevenLabs APIエラー ({retry_exc.code})"
                )
                if retry_detail:
                    message += f": {retry_detail}"
                raise ElevenLabsTtsError(message) from retry_exc
            except urllib.error.URLError as retry_exc:
                raise ElevenLabsTtsError(
                    "ElevenLabsへの接続に失敗しました: "
                    f"{retry_exc.reason}"
                ) from retry_exc
        message = f"ElevenLabs APIエラー ({exc.code})"
        if detail:
            message += f": {detail}"
        raise ElevenLabsTtsError(message) from exc
    except urllib.error.URLError as exc:
        raise ElevenLabsTtsError(
            f"ElevenLabsへの接続に失敗しました: {exc.reason}"
        ) from exc


def generate_segment(
    segments: list[Any],
    index: int,
    tts_settings: dict,
    force: bool = False,
    opener: Callable[..., Any] | None = None,
) -> dict:
    if index < 0 or index >= len(segments):
        raise ElevenLabsTtsError("無効なセグメントindexです")
    output_value = str(tts_settings.get("output_dir") or "").strip()
    if not output_value:
        raise ElevenLabsTtsError("TTS出力フォルダを選択してください")
    output_dir = Path(output_value)
    output_dir.mkdir(parents=True, exist_ok=True)

    segment = segments[index]
    speaker = _speaker(segment)
    voice_id = _voice_ids(tts_settings).get(speaker, "")
    if not voice_id:
        raise ElevenLabsTtsError(f"{speaker} のvoice_idが設定されていません")
    model_id = str(
        tts_settings.get("model_id") or DEFAULT_MODEL_ID
    ).strip()
    if resolve_generation_mode(tts_settings) == CHUNK_V3:
        raise ElevenLabsTtsError(
            "eleven_v3はチャンク単位で生成してください"
        )
    output_format = str(
        tts_settings.get("output_format") or DEFAULT_OUTPUT_FORMAT
    ).strip()
    voice_settings = normalize_voice_settings(tts_settings)
    apply_normalization = False
    apply_text_normalization = normalize_apply_text_normalization(
        tts_settings.get("apply_text_normalization", "auto")
    )
    manifest = _base_manifest(tts_settings, load_manifest(output_dir))
    text_hash = compute_text_hash(
        segment,
        voice_id,
        model_id,
        output_format,
        voice_settings,
        apply_normalization,
        apply_text_normalization,
    )
    entry = manifest["segments"].get(str(index))
    existing_path = _resolved_audio_path(output_dir, entry)
    if (
        not force
        and entry
        and entry.get("status") == "generated"
        and entry.get("text_hash") == text_hash
        and existing_path is not None
        and existing_path.exists()
    ):
        return {
            "index": index, "status": "skipped",
            "audio_path": str(existing_path),
        }

    api_key = str(tts_settings.get("api_key") or "").strip()
    if not api_key:
        raise ElevenLabsTtsError("ElevenLabs API Keyを入力してください")
    current_text = _text(segment)
    if not current_text.strip():
        raise ElevenLabsTtsError("空のセグメントはTTS生成できません")
    file_name = audio_filename(index, speaker, output_format)
    audio_path = output_dir / file_name
    previous_text = _text(segments[index - 1]) if index > 0 else None
    next_text = _text(segments[index + 1]) if index + 1 < len(segments) else None
    include_context = model_id in CONTEXT_MODELS

    try:
        audio_bytes = _request_audio(
            current_text, voice_id, model_id, output_format, api_key,
            previous_text, next_text, voice_settings, apply_normalization,
            apply_text_normalization=apply_text_normalization,
            include_context=include_context,
            opener=opener,
        )
        if not audio_bytes:
            raise ElevenLabsTtsError("ElevenLabsから空の音声が返されました")
        fd, temp_name = tempfile.mkstemp(
            prefix=f"tts_{index + 1:04d}_", suffix=".tmp", dir=output_dir
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(audio_bytes)
            os.replace(temp_name, audio_path)
        except Exception:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise
    except Exception as exc:
        manifest["segments"][str(index)] = {
            "text_hash": text_hash,
            "speaker": speaker,
            "voice_id": voice_id,
            "audio_path": file_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "error",
            "error": str(exc)[:500],
        }
        save_manifest(output_dir, manifest)
        if isinstance(exc, ElevenLabsTtsError):
            raise
        raise ElevenLabsTtsError(
            f"TTS音声の保存に失敗しました: {exc}"
        ) from exc

    manifest["segments"][str(index)] = {
        "text_hash": text_hash,
        "speaker": speaker,
        "voice_id": voice_id,
        "audio_path": file_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "generated",
    }
    save_manifest(output_dir, manifest)
    return {"index": index, "status": "generated", "audio_path": str(audio_path)}


def generate_chunk(
    segments: list[Any],
    chunk_index: int,
    tts_settings: dict,
    force: bool = False,
    opener: Callable[..., Any] | None = None,
) -> dict:
    """eleven_v3用に指定チャンクを文脈キーなしで生成する。"""
    if resolve_generation_mode(tts_settings) != CHUNK_V3:
        raise ElevenLabsTtsError(
            "チャンク生成はeleven_v3でのみ使用できます"
        )
    chunks = build_chunks(segments, tts_settings)
    if chunk_index < 0 or chunk_index >= len(chunks):
        raise ElevenLabsTtsError("無効なチャンクindexです")
    output_value = str(tts_settings.get("output_dir") or "").strip()
    if not output_value:
        raise ElevenLabsTtsError("TTS出力フォルダを選択してください")
    output_dir = Path(output_value)
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    chunk = chunks[chunk_index]
    speaker = chunk["speaker"]
    voice_id = chunk["voice_ids"].get(speaker, "")
    if not voice_id:
        raise ElevenLabsTtsError(
            f"{speaker} のvoice_idが設定されていません"
        )
    model_id = str(tts_settings.get("model_id") or DEFAULT_MODEL_ID).strip()
    output_format = str(
        tts_settings.get("output_format") or DEFAULT_OUTPUT_FORMAT
    ).strip()
    voice_settings = normalize_voice_settings(tts_settings)
    apply_normalization = False
    apply_text_normalization = normalize_apply_text_normalization(
        tts_settings.get("apply_text_normalization", "auto")
    )
    manifest = _base_manifest(tts_settings, load_manifest(output_dir))
    entry = manifest["chunks"].get(str(chunk_index))
    existing_path = _resolved_audio_path(output_dir, entry)
    if (
        not force
        and entry
        and entry.get("status") == "generated"
        and entry.get("text_hash") == chunk["text_hash"]
        and existing_path is not None
        and existing_path.exists()
    ):
        return {
            "chunk_index": chunk_index,
            "segment_indices": chunk["segment_indices"],
            "status": "skipped",
            "audio_path": str(existing_path),
        }

    api_key = str(tts_settings.get("api_key") or "").strip()
    if not api_key:
        raise ElevenLabsTtsError("ElevenLabs API Keyを入力してください")
    extension = _audio_extension(output_format)
    file_name = f"chunk_{chunk_index + 1:04d}{extension}"
    relative_path = (Path("chunks") / file_name).as_posix()
    audio_path = chunks_dir / file_name

    try:
        audio_bytes = _request_audio(
            chunk["text"],
            voice_id,
            model_id,
            output_format,
            api_key,
            None,
            None,
            voice_settings,
            apply_normalization,
            apply_text_normalization=apply_text_normalization,
            include_context=False,
            opener=opener,
        )
        if not audio_bytes:
            raise ElevenLabsTtsError(
                "ElevenLabsから空の音声が返されました"
            )
        fd, temp_name = tempfile.mkstemp(
            prefix=f"chunk_{chunk_index + 1:04d}_",
            suffix=".tmp",
            dir=chunks_dir,
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(audio_bytes)
            os.replace(temp_name, audio_path)
        except Exception:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise
    except Exception as exc:
        manifest["chunks"][str(chunk_index)] = {
            "chunk_index": chunk_index,
            "segment_indices": chunk["segment_indices"],
            "text_hash": chunk["text_hash"],
            "model_id": model_id,
            "voice_ids": chunk["voice_ids"],
            "audio_path": relative_path,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "error",
            "error": str(exc)[:500],
        }
        save_manifest(output_dir, manifest)
        if isinstance(exc, ElevenLabsTtsError):
            raise
        raise ElevenLabsTtsError(
            f"TTSチャンク音声の保存に失敗しました: {exc}"
        ) from exc

    manifest["chunks"][str(chunk_index)] = {
        "chunk_index": chunk_index,
        "segment_indices": chunk["segment_indices"],
        "text_hash": chunk["text_hash"],
        "model_id": model_id,
        "voice_ids": chunk["voice_ids"],
        "audio_path": relative_path,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "generated",
    }
    save_manifest(output_dir, manifest)
    return {
        "chunk_index": chunk_index,
        "segment_indices": chunk["segment_indices"],
        "status": "generated",
        "audio_path": str(audio_path),
    }


def format_timestamp(seconds: float) -> str:
    total_ms = max(0, round(float(seconds) * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def export_script_txt(segments: list[Any]) -> str:
    blocks = []
    for index, segment in enumerate(segments, 1):
        start = format_timestamp(float(_value(segment, "start", 0.0)))
        end = format_timestamp(float(_value(segment, "end", 0.0)))
        blocks.append(
            f"{index:03d} [{start} - {end}] {_speaker(segment)}:\n"
            f"{_text(segment)}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def export_script_csv(segments: list[Any], tts_settings: dict) -> str:
    statuses = get_segment_statuses(segments, tts_settings)
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow([
        "index", "start", "end", "speaker", "text",
        "voice_id", "audio_path", "tts_status",
    ])
    for index, (segment, status) in enumerate(zip(segments, statuses), 1):
        writer.writerow([
            index,
            format_timestamp(float(_value(segment, "start", 0.0))),
            format_timestamp(float(_value(segment, "end", 0.0))),
            _speaker(segment),
            _text(segment),
            status["voice_id"],
            status["audio_path"],
            status["status"],
        ])
    return output.getvalue()
