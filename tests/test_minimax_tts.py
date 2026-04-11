#!/usr/bin/env python3
"""
MiniMax TTS pipeline E2E tests.

Covers the MiniMax speech bug fix:
  1. Default 16kHz → resampled to 24kHz by plugin
  2. 24kHz native → no resample needed (was chipmunk bug before WS URL fix)
  3. _is_pcm_provider() check for MiniMax vs Telnyx voices
  4. synthesize() single-shot path
  5. Short text → resampler flush works
  6. Multi-segment stream → audio not truncated

Usage:
    TELNYX_API_KEY=... uv run python tests/test_minimax_tts.py

    # Defaults to dev gateway; override with:
    TELNYX_API_BASE_URL=wss://api.telnyx.com/v2 uv run python tests/test_minimax_tts.py
"""

import asyncio
import io
import os
import wave
from pathlib import Path

from dotenv import load_dotenv
from livekit.plugins import telnyx

load_dotenv()

API_KEY = os.environ["TELNYX_API_KEY"]
BASE_URL = os.environ.get("TELNYX_API_BASE_URL", "wss://apidev.telnyx.com/v2")
MINIMAX_VOICE = "Minimax.speech-02-turbo.Calm_Woman"
MINIMAX_TEXT = "The quick brown fox jumps over the lazy dog."
PIPELINE_SAMPLE_RATE = 24000
OUTPUT_DIR = Path(__file__).parent / "output"


def _save_wav(data: bytes, sample_rate: int, filename: str) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / filename
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(data)
    with open(path, "wb") as f:
        f.write(buf.getvalue())
    return path


async def _run_stream(
    sample_rate: int = 16000,
    voice: str = MINIMAX_VOICE,
    text: str = MINIMAX_TEXT,
    flush: bool = True,
    segments: list[str] | None = None,
) -> tuple[int, int, bytes]:
    """Run MiniMax TTS through the stream() pipeline.

    Returns (frame_rate, num_chunks, raw_audio).
    """
    tts_inst = telnyx.TTS(voice=voice, api_key=API_KEY, sample_rate=sample_rate)

    audio_chunks = []
    frame_sample_rate = None

    try:
        stream = tts_inst.stream()

        if segments:
            for seg in segments:
                stream.push_text(seg)
                stream.flush()
        else:
            stream.push_text(text)
            if flush:
                stream.flush()

        stream.end_input()

        async for event in stream:
            if hasattr(event, "frame") and event.frame and hasattr(event.frame, "data"):
                audio_chunks.append(event.frame.data.tobytes())
                if frame_sample_rate is None and hasattr(event.frame, "sample_rate"):
                    frame_sample_rate = event.frame.sample_rate
    finally:
        await tts_inst.aclose()

    raw_audio = b"".join(audio_chunks)
    return frame_sample_rate or 0, len(audio_chunks), raw_audio


async def _run_synthesize(
    sample_rate: int = 16000,
    voice: str = MINIMAX_VOICE,
    text: str = MINIMAX_TEXT,
) -> tuple[int, int, bytes]:
    """Run MiniMax TTS through the synthesize() single-shot path.

    Returns (frame_rate, num_chunks, raw_audio).
    """
    tts_inst = telnyx.TTS(voice=voice, api_key=API_KEY, sample_rate=sample_rate)

    audio_chunks = []
    frame_sample_rate = None

    try:
        result = tts_inst.synthesize(text)

        async for event in result:
            if hasattr(event, "frame") and event.frame and hasattr(event.frame, "data"):
                audio_chunks.append(event.frame.data.tobytes())
                if frame_sample_rate is None and hasattr(event.frame, "sample_rate"):
                    frame_sample_rate = event.frame.sample_rate
    finally:
        await tts_inst.aclose()

    raw_audio = b"".join(audio_chunks)
    return frame_sample_rate or 0, len(audio_chunks), raw_audio


# ── Test Cases ────────────────────────────────────────────────────────

async def test_pcm_default_resampled() -> dict:
    """Test 1: PCM 16kHz (default) → resampled to 24kHz by plugin.

    MiniMax returns 16kHz PCM. The plugin resamples to 24kHz for the
    LiveKit pipeline. Frames should arrive at 24kHz.
    """
    frame_rate, chunks, raw = await _run_stream(sample_rate=16000)
    duration = len(raw) / (frame_rate * 2) if frame_rate else 0
    wav = str(_save_wav(raw, frame_rate, "minimax_pcm_default.wav")) if raw else None

    return {
        "name": "PCM default (16kHz → resampled to 24kHz)",
        "frame_sample_rate": frame_rate,
        "audio_bytes": len(raw),
        "chunks": chunks,
        "duration_s": round(duration, 2),
        "resampled_to_pipeline": frame_rate == PIPELINE_SAMPLE_RATE,
        "has_audio": len(raw) > 0,
        "wav": wav,
    }


async def test_pcm_24k_native() -> dict:
    """Test 2: PCM 24kHz → native output, no resampler.

    With the WS URL fix, the plugin sends sample_rate=24000 to the gateway,
    which forwards it to MiniMax. MiniMax returns native 24kHz PCM.
    No resampler is needed — audio comes through at the pipeline rate.
    Before the fix, MiniMax returned 16kHz (default) causing chipmunk audio.
    """
    frame_rate, chunks, raw = await _run_stream(sample_rate=24000)
    duration = len(raw) / (frame_rate * 2) if frame_rate else 0
    wav = str(_save_wav(raw, frame_rate, "minimax_pcm_24k.wav")) if raw else None

    return {
        "name": "PCM 24kHz (native, no resample)",
        "frame_sample_rate": frame_rate,
        "audio_bytes": len(raw),
        "chunks": chunks,
        "duration_s": round(duration, 2),
        "at_pipeline_rate": frame_rate == PIPELINE_SAMPLE_RATE,
        "has_audio": len(raw) > 0,
        "wav": wav,
    }


async def test_is_pcm_provider() -> dict:
    """Test 3: _is_pcm_provider() correctly identifies MiniMax vs Telnyx voices.

    MiniMax voices should use PCM path (no MP3 decoder).
    Telnyx voices should use MP3 path (with decoder).
    """
    minimax_tts = telnyx.TTS(voice=MINIMAX_VOICE, api_key=API_KEY)
    telnyx_tts = telnyx.TTS(voice="Telnyx.NaturalHD.astra", api_key=API_KEY)

    minimax_stream = minimax_tts.stream()
    telnyx_stream = telnyx_tts.stream()

    minimax_is_pcm = minimax_stream._is_pcm_provider()
    telnyx_is_pcm = telnyx_stream._is_pcm_provider()

    await minimax_tts.aclose()
    await telnyx_tts.aclose()

    return {
        "name": "_is_pcm_provider() check",
        "minimax_is_pcm": minimax_is_pcm,
        "telnyx_is_pcm": telnyx_is_pcm,
        "correct": minimax_is_pcm is True and telnyx_is_pcm is False,
    }


async def test_synthesize_single_shot() -> dict:
    """Test 4: synthesize() single-shot path works for MiniMax.

    The synthesize() API is a convenience wrapper around the streaming API.
    It should produce the same quality audio as stream() but in a single
    request-response cycle.
    """
    frame_rate, chunks, raw = await _run_synthesize(sample_rate=16000)
    duration = len(raw) / (frame_rate * 2) if frame_rate else 0
    wav = str(_save_wav(raw, frame_rate, "minimax_synthesize.wav")) if raw else None

    return {
        "name": "synthesize() single-shot",
        "frame_sample_rate": frame_rate,
        "audio_bytes": len(raw),
        "chunks": chunks,
        "duration_s": round(duration, 2),
        "resampled_to_pipeline": frame_rate == PIPELINE_SAMPLE_RATE,
        "has_audio": len(raw) > 0,
        "wav": wav,
    }


async def test_short_text_flush() -> dict:
    """Test 5: Short text → resampler flush works.

    With very short text, the PCM byte stream may have leftover samples
    when the WebSocket closes. The resampler flush path must emit those
    remaining frames. Without it, the last few ms of audio gets dropped.
    """
    frame_rate, chunks, raw = await _run_stream(
        sample_rate=16000, text="Hi.", flush=True
    )
    duration = len(raw) / (frame_rate * 2) if frame_rate else 0
    wav = str(_save_wav(raw, frame_rate, "minimax_short_text.wav")) if raw else None

    return {
        "name": "Short text flush",
        "frame_sample_rate": frame_rate,
        "audio_bytes": len(raw),
        "chunks": chunks,
        "duration_s": round(duration, 2),
        "resampled_to_pipeline": frame_rate == PIPELINE_SAMPLE_RATE,
        "has_audio": len(raw) > 0,
        "wav": wav,
    }


async def test_multi_segment_stream() -> dict:
    """Test 6: Multi-segment stream → audio not truncated.

    Push multiple text segments through the stream with flush between each.
    Each segment produces its own WebSocket request, and the resulting audio
    should be the concatenation of all segments — no truncation between them.
    """
    segments = [
        "The first sentence.",
        "The second sentence.",
        "The third and final sentence.",
    ]
    frame_rate, chunks, raw = await _run_stream(
        sample_rate=16000, segments=segments
    )
    duration = len(raw) / (frame_rate * 2) if frame_rate else 0
    wav = str(_save_wav(raw, frame_rate, "minimax_multi_segment.wav")) if raw else None

    # For comparison, synthesize the full text as one shot
    full_text = " ".join(segments)
    _, _, raw_single = await _run_stream(sample_rate=16000, text=full_text, flush=True)

    # Multi-segment audio should be in the same ballpark as single-shot
    # (not exactly equal due to TTS variance, but same order of magnitude)
    ratio = len(raw) / len(raw_single) if raw_single else 0

    return {
        "name": "Multi-segment stream",
        "frame_sample_rate": frame_rate,
        "audio_bytes": len(raw),
        "single_shot_bytes": len(raw_single),
        "ratio_vs_single": round(ratio, 2),
        "chunks": chunks,
        "duration_s": round(duration, 2),
        "resampled_to_pipeline": frame_rate == PIPELINE_SAMPLE_RATE,
        "has_audio": len(raw) > 0,
        "not_truncated": 0.5 < ratio < 2.0,
        "wav": wav,
    }


# ── Runner ────────────────────────────────────────────────────────────

async def main():
    print("🧪 MiniMax TTS Pipeline Tests")
    print(f"   Base URL: {BASE_URL}")
    print(f"   Voice: {MINIMAX_VOICE}")
    print("=" * 60)

    # Clean output dir
    if OUTPUT_DIR.exists():
        for f in OUTPUT_DIR.iterdir():
            if f.is_file() and f.suffix in (".wav", ".mp3"):
                f.unlink()
    OUTPUT_DIR.mkdir(exist_ok=True)

    all_passed = True

    # Test 1: PCM default (resampled)
    print("\n1. PCM default (16kHz → resampled to 24kHz)")
    try:
        r = await test_pcm_default_resampled()
        print(f"   Frame rate: {r['frame_sample_rate']} Hz | Bytes: {r['audio_bytes']} | Duration: {r['duration_s']}s")
        if r["wav"]:
            print(f"   WAV: {r['wav']}")
        if r["resampled_to_pipeline"] and r["has_audio"]:
            print(f"   ✅ Audio at pipeline rate ({PIPELINE_SAMPLE_RATE} Hz) with resampling")
        else:
            print(f"   ❌ Expected {PIPELINE_SAMPLE_RATE} Hz with audio, got {r['frame_sample_rate']} Hz / {r['audio_bytes']} bytes")
            all_passed = False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        all_passed = False

    # Test 2: PCM 24kHz (native, no resample)
    print("\n2. PCM 24kHz (native, no resample)")
    try:
        r = await test_pcm_24k_native()
        print(f"   Frame rate: {r['frame_sample_rate']} Hz | Bytes: {r['audio_bytes']} | Duration: {r['duration_s']}s")
        if r["wav"]:
            print(f"   WAV: {r['wav']}")
        if r["at_pipeline_rate"] and r["has_audio"]:
            print(f"   ✅ Native 24kHz from MiniMax (WS URL fix working)")
        else:
            print(f"   ❌ Expected {PIPELINE_SAMPLE_RATE} Hz with audio, got {r['frame_sample_rate']} Hz / {r['audio_bytes']} bytes")
            print(f"   ⚠️  If frame_rate=0 or no audio: plugin WS URL may not be sending sample_rate to gateway")
            all_passed = False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        all_passed = False

    # Test 3: _is_pcm_provider()
    print("\n3. _is_pcm_provider() check")
    try:
        r = await test_is_pcm_provider()
        if r["correct"]:
            print(f"   ✅ MiniMax voice → PCM ({r['minimax_is_pcm']}), Telnyx voice → MP3 ({r['telnyx_is_pcm']})")
        else:
            print(f"   ❌ MiniMax={r['minimax_is_pcm']}, Telnyx={r['telnyx_is_pcm']}")
            all_passed = False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        all_passed = False

    # Test 4: synthesize() single-shot
    print("\n4. synthesize() single-shot")
    try:
        r = await test_synthesize_single_shot()
        print(f"   Frame rate: {r['frame_sample_rate']} Hz | Bytes: {r['audio_bytes']} | Duration: {r['duration_s']}s")
        if r["wav"]:
            print(f"   WAV: {r['wav']}")
        if r["has_audio"] and r["resampled_to_pipeline"]:
            print(f"   ✅ synthesize() produces audio at pipeline rate")
        else:
            print(f"   ❌ No audio or wrong rate: {r['frame_sample_rate']} Hz / {r['audio_bytes']} bytes")
            all_passed = False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        all_passed = False

    # Test 5: Short text flush
    print("\n5. Short text flush")
    try:
        r = await test_short_text_flush()
        print(f"   Frame rate: {r['frame_sample_rate']} Hz | Bytes: {r['audio_bytes']} | Duration: {r['duration_s']}s")
        if r["wav"]:
            print(f"   WAV: {r['wav']}")
        if r["has_audio"] and r["resampled_to_pipeline"]:
            print(f"   ✅ Short text flush produces audio at pipeline rate")
        else:
            print(f"   ❌ No audio or wrong rate: {r['frame_sample_rate']} Hz / {r['audio_bytes']} bytes")
            all_passed = False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        all_passed = False

    # Test 6: Multi-segment stream
    print("\n6. Multi-segment stream")
    try:
        r = await test_multi_segment_stream()
        print(f"   Frame rate: {r['frame_sample_rate']} Hz | Bytes: {r['audio_bytes']} | Single-shot: {r['single_shot_bytes']} | Ratio: {r['ratio_vs_single']}")
        if r["wav"]:
            print(f"   WAV: {r['wav']}")
        if r["has_audio"] and r["not_truncated"]:
            print(f"   ✅ Multi-segment audio not truncated (ratio vs single-shot: {r['ratio_vs_single']})")
        else:
            print(f"   ❌ Multi-segment audio may be truncated: {r['audio_bytes']} bytes vs {r['single_shot_bytes']} single-shot")
            all_passed = False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        all_passed = False

    # Summary
    print(f"\n{'=' * 60}")
    if all_passed:
        print("✅ All MiniMax pipeline tests passed")
    else:
        print("⚠️  Some tests failed")


if __name__ == "__main__":
    asyncio.run(main())
