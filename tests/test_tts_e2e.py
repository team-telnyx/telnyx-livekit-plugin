#!/usr/bin/env python3
"""
E2E tests: TTS synthesis and sample rate configuration.

Covers:
  - Basic TTS synthesis
  - Sample rate variations (16kHz, 24kHz)
  - Output verification (byte length ratio confirms different rates)

Usage:
    uv run python tests/test_tts_e2e.py
"""

import asyncio
import os
import wave
from pathlib import Path

from dotenv import load_dotenv
from livekit.plugins import telnyx

load_dotenv()

OUTPUT_DIR = Path(__file__).parent / "output"
TEST_TEXT = "Hello, this is a test of the Telnyx text to speech plugin for LiveKit."
TEST_RATES = [16000, 24000]


async def synthesize(api_key: str, label: str, sample_rate: int = 16000,
                     voice: str = "Telnyx.NaturalHD.astra") -> tuple[int, bytes] | None:
    """Synthesize text and return (sample_rate, raw_audio)."""
    tts = telnyx.TTS(
        voice=voice,
        api_key=api_key,
        sample_rate=sample_rate,
    )

    try:
        # Verify instance reports correct sample rate
        assert tts.sample_rate == sample_rate, (
            f"Expected sample_rate={sample_rate}, got {tts.sample_rate}"
        )

        stream = tts.synthesize(TEST_TEXT)

        audio_chunks = []
        async for event in stream:
            if hasattr(event, "frame") and event.frame and hasattr(event.frame, "data"):
                audio_chunks.append(event.frame.data.tobytes())

        if not audio_chunks:
            print(f"  ❌ {label}: no audio chunks received")
            return None

        raw_audio = b"".join(audio_chunks)

        # Save WAV for inspection
        OUTPUT_DIR.mkdir(exist_ok=True)
        wav_path = OUTPUT_DIR / f"tts_{sample_rate}hz.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(sample_rate)
            wf.writeframes(raw_audio)

        print(f"  ✅ {label}: {len(raw_audio)} bytes → {wav_path}")
        return sample_rate, raw_audio

    except Exception as e:
        print(f"  ❌ {label}")
        print(f"     Error: {e}")
        return None
    finally:
        await tts.aclose()


async def main():
    api_key = os.getenv("TELNYX_API_KEY")
    if not api_key:
        print("❌ TELNYX_API_KEY not set")
        return

    results = {}

    # --- Basic synthesis at each sample rate ---
    print("\n🧪 TTS E2E Tests")
    print("=" * 55)

    for rate in TEST_RATES:
        result = await synthesize(api_key, f"TTS at {rate}Hz", sample_rate=rate)
        if result:
            results[rate] = result

    # --- Sample rate verification ---
    print(f"\n{'=' * 55}")
    print("📊 RESULTS")
    print(f"{'=' * 55}")

    all_passed = True
    for rate in TEST_RATES:
        if rate in results:
            _, audio = results[rate]
            print(f"  ✅ {rate}Hz — {len(audio)} bytes")
        else:
            print(f"  ❌ {rate}Hz — failed")
            all_passed = False

    # Verify different rates produce different byte lengths
    if all(r in results for r in TEST_RATES):
        len_16k = len(results[16000][1])
        len_24k = len(results[24000][1])
        ratio = len_24k / len_16k if len_16k > 0 else 0
        print(f"\n  Byte ratio 24kHz/16kHz = {ratio:.2f} (expect ~1.5)")
        if 1.2 < ratio < 1.8:
            print("  ✅ Ratio confirms different sample rates")
        else:
            print("  ⚠️  Ratio outside expected range — may need investigation")
            all_passed = False

    print()
    if all_passed:
        print("✅ All TTS E2E tests passed")
    else:
        print("⚠️  Some tests failed")


if __name__ == "__main__":
    asyncio.run(main())
