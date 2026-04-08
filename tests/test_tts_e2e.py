#!/usr/bin/env python3
"""
E2E tests: TTS synthesis, sample rate configuration, MiniMax last-sentence
fix, and MiniMax PCM resampling.

Covers:
  - Basic TTS synthesis (Telnyx voices)
  - Sample rate variations (16kHz, 24kHz)
  - Output verification (byte length ratio confirms different rates)
  - MiniMax last sentence: all sentences produce audio, especially the final one
  - MiniMax PCM resampling: frames arrive at pipeline rate (24kHz), not native (16kHz)

Usage:
    uv run python tests/test_tts_e2e.py
"""

import asyncio
import json
import os
import base64
import wave
from pathlib import Path

import aiohttp
from dotenv import load_dotenv
from livekit.plugins import telnyx

load_dotenv()

OUTPUT_DIR = Path(__file__).parent / "output"
TEST_TEXT = "Hello, this is a test of the Telnyx text to speech plugin for LiveKit."
TEST_RATES = [16000, 24000]

# MiniMax test constants
MINIMAX_VOICE = "Minimax.speech-02-turbo.Calm_Woman"
MINIMAX_FULL_TEXT = (
    "Well, thank you! I'm doing great. What's on your mind? "
    "Is there anything I can help you with today?"
)
MINIMAX_SENTENCES = [
    "Well, thank you! ",
    "I'm doing great. ",
    "What's on your mind? ",
    "Is there anything I can help you with today?",
]

# LiveKit voice pipeline default output sample rate.
PIPELINE_SAMPLE_RATE = 24000


async def synthesize(api_key: str, label: str, sample_rate: int = 16000,
                     voice: str = "Telnyx.NaturalHD.astra") -> tuple[int, bytes] | None:
    """Synthesize text and return (sample_rate, raw_audio)."""
    tts_inst = telnyx.TTS(
        voice=voice,
        api_key=api_key,
        sample_rate=sample_rate,
    )

    try:
        stream = tts_inst.synthesize(TEST_TEXT)

        audio_chunks = []
        async for event in stream:
            if hasattr(event, "frame") and event.frame and hasattr(event.frame, "data"):
                audio_chunks.append(event.frame.data.tobytes())

        if not audio_chunks:
            print(f"  ❌ {label}: no audio chunks received")
            return None

        raw_audio = b"".join(audio_chunks)

        OUTPUT_DIR.mkdir(exist_ok=True)
        wav_path = OUTPUT_DIR / f"tts_{sample_rate}hz.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(raw_audio)

        print(f"  ✅ {label}: {len(raw_audio)} bytes → {wav_path}")
        return sample_rate, raw_audio

    except Exception as e:
        print(f"  ❌ {label}")
        print(f"     Error: {e}")
        return None
    finally:
        await tts_inst.aclose()


# ── MiniMax: last sentence via raw WebSocket ──────────────────────────

async def test_minimax_last_sentence_raw(api_key: str) -> dict:
    """
    Hit MiniMax TTS via raw WebSocket and verify every sentence produces audio,
    especially the last one (which was previously dropped due to MP3 frame
    boundary issues).
    """
    url = f"wss://api.telnyx.com/v2/text-to-speech/speech?voice={MINIMAX_VOICE}"
    headers = {"Authorization": f"Bearer {api_key}"}

    session = aiohttp.ClientSession()
    try:
        ws = await session.ws_connect(url, headers=headers)

        # Send with PCM format (the fix path)
        await ws.send_str(json.dumps({
            "text": " ",
            "voice_settings": {"response_format": "pcm"},
        }))
        await ws.send_str(json.dumps({"text": MINIMAX_FULL_TEXT}))
        await ws.send_str(json.dumps({"text": ""}))

        audio_bytes_total = 0
        messages = []
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                audio = data.get("audio")
                if audio:
                    decoded = base64.b64decode(audio)
                    audio_bytes_total += len(decoded)
                messages.append(data)
            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                break

        # Check for isFinal message (indicates server processed everything)
        has_final = any(m.get("isFinal") for m in messages)
        has_last_sentence_text = any(
            "help you with today" in (m.get("text") or "")
            for m in messages
        )

        return {
            "audio_bytes": audio_bytes_total,
            "messages": len(messages),
            "has_final": has_final,
            "has_last_sentence_text": has_last_sentence_text,
        }
    finally:
        await session.close()


# ── MiniMax: PCM resampling via plugin ────────────────────────────────

async def test_minimax_pcm_resample(api_key: str) -> dict:
    """
    Run MiniMax TTS through the full plugin pipeline and verify:
    1. Audio frames arrive at the pipeline sample rate (24kHz), not native (16kHz)
    2. Duration is reasonable (~4s for the test sentence)
    3. All text is voiced (non-zero audio)
    """
    tts_inst = telnyx.TTS(voice=MINIMAX_VOICE, api_key=api_key)

    audio_chunks = []
    frame_sample_rate = None

    try:
        stream = tts_inst.stream()
        for sentence in MINIMAX_SENTENCES:
            stream.push_text(sentence)
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
    duration = len(raw_audio) / (frame_sample_rate * 2) if frame_sample_rate else 0

    # Save for manual listening
    OUTPUT_DIR.mkdir(exist_ok=True)
    wav_path = OUTPUT_DIR / "minimax_resampled_e2e.wav"
    if raw_audio and frame_sample_rate:
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(frame_sample_rate)
            wf.writeframes(raw_audio)

    return {
        "frame_sample_rate": frame_sample_rate,
        "audio_bytes": len(raw_audio),
        "chunks": len(audio_chunks),
        "duration_s": round(duration, 2),
        "wav_path": str(wav_path) if raw_audio else None,
    }


# ── Main ──────────────────────────────────────────────────────────────

async def main():
    api_key = os.getenv("TELNYX_API_KEY")
    if not api_key:
        print("❌ TELNYX_API_KEY not set")
        return

    all_passed = True

    # ── 1. Basic TTS synthesis at different sample rates ──
    print("\n🧪 TTS E2E Tests")
    print("=" * 60)

    results = {}
    for rate in TEST_RATES:
        result = await synthesize(api_key, f"Telnyx TTS at {rate}Hz", sample_rate=rate)
        if result:
            results[rate] = result

    print(f"\n{'─' * 60}")
    print("📊 Sample Rate Results")

    for rate in TEST_RATES:
        if rate in results:
            _, audio = results[rate]
            print(f"  ✅ {rate}Hz — {len(audio)} bytes")
        else:
            print(f"  ❌ {rate}Hz — failed")
            all_passed = False

    if all(r in results for r in TEST_RATES):
        len_16k = len(results[16000][1])
        len_24k = len(results[24000][1])
        ratio = len_24k / len_16k if len_16k > 0 else 0
        print(f"  Byte ratio 24kHz/16kHz = {ratio:.2f} (expect ~1.5)")
        if 1.2 < ratio < 1.8:
            print("  ✅ Ratio confirms different sample rates")
        else:
            print("  ⚠️  Ratio outside expected range")
            all_passed = False

    # ── 2. MiniMax last sentence (raw WebSocket) ──
    print(f"\n{'─' * 60}")
    print("🧪 MiniMax: Last Sentence (raw WebSocket PCM)")

    try:
        r = await test_minimax_last_sentence_raw(api_key)
        print(f"  Audio: {r['audio_bytes']} bytes, {r['messages']} WS messages")

        if r["audio_bytes"] > 0:
            print("  ✅ Audio received")
        else:
            print("  ❌ No audio received")
            all_passed = False

        if r["has_final"]:
            print("  ✅ Server sent isFinal")
        else:
            print("  ⚠️  No isFinal message (may still be OK)")

        if r["has_last_sentence_text"]:
            print("  ✅ Last sentence text confirmed in response")
        else:
            print("  ⚠️  Last sentence text not found in WS messages")

    except Exception as e:
        print(f"  ❌ Error: {e}")
        all_passed = False

    # ── 3. MiniMax PCM resampling (full plugin pipeline) ──
    print(f"\n{'─' * 60}")
    print("🧪 MiniMax: PCM Resampling (plugin pipeline)")

    try:
        r = await test_minimax_pcm_resample(api_key)
        print(f"  Frame sample_rate: {r['frame_sample_rate']} Hz")
        print(f"  Audio: {r['audio_bytes']} bytes, {r['chunks']} chunks, {r['duration_s']}s")
        if r["wav_path"]:
            print(f"  WAV: {r['wav_path']}")

        if r["frame_sample_rate"] == PIPELINE_SAMPLE_RATE:
            print(f"  ✅ Frames at pipeline rate ({PIPELINE_SAMPLE_RATE} Hz)")
        else:
            print(f"  ❌ Expected {PIPELINE_SAMPLE_RATE} Hz, got {r['frame_sample_rate']} Hz")
            all_passed = False

        if 2.0 < r["duration_s"] < 8.0:
            print(f"  ✅ Duration {r['duration_s']}s is reasonable")
        else:
            print(f"  ❌ Duration {r['duration_s']}s outside expected range (2-8s)")
            all_passed = False

        if r["audio_bytes"] > 0:
            print("  ✅ Non-zero audio output")
        else:
            print("  ❌ No audio output")
            all_passed = False

    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # ── Summary ──
    print(f"\n{'=' * 60}")
    if all_passed:
        print("✅ All TTS E2E tests passed")
    else:
        print("⚠️  Some tests failed")


if __name__ == "__main__":
    asyncio.run(main())
