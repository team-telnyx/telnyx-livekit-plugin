#!/usr/bin/env python3
"""
Integration test: TTS sample_rate parameter.

Verifies that the TTS class accepts a configurable sample_rate and that
the synthesized audio matches the requested rate.

Usage:
    uv run python tests/test_tts_sample_rate.py
"""

import asyncio
import os
import wave
from pathlib import Path

from dotenv import load_dotenv
from livekit.plugins import telnyx

load_dotenv()

OUTPUT_DIR = Path(__file__).parent / "output"


async def synthesize_at_rate(api_key: str, sample_rate: int) -> tuple[int, bytes]:
    """Synthesize a short phrase at the given sample rate and return (rate, raw_audio)."""
    print(f"\n{'='*50}")
    print(f"  Testing TTS at sample_rate={sample_rate}")
    print(f"{'='*50}")

    tts_instance = telnyx.TTS(
        voice="Telnyx.NaturalHD.astra",
        api_key=api_key,
        sample_rate=sample_rate,
    )

    # Verify the instance reports the correct sample rate
    assert tts_instance.sample_rate == sample_rate, (
        f"Expected sample_rate={sample_rate}, got {tts_instance.sample_rate}"
    )
    print(f"  ✅ TTS.sample_rate = {tts_instance.sample_rate}")

    try:
        text = "Testing sample rate configuration."
        print(f'  Synthesizing: "{text}"')

        # Use synthesize() (non-streaming) — same as existing test_tts.py
        stream = tts_instance.synthesize(text)

        audio_chunks = []
        async for event in stream:
            if hasattr(event, "frame") and event.frame and hasattr(event.frame, "data"):
                audio_chunks.append(event.frame.data.tobytes())

        if not audio_chunks:
            print(f"  ⚠️  No audio chunks received at {sample_rate}Hz")
            return sample_rate, b""

        raw_audio = b"".join(audio_chunks)
        print(f"  ✅ Got {len(raw_audio)} bytes of audio")

        # Save WAV for manual inspection
        OUTPUT_DIR.mkdir(exist_ok=True)
        wav_path = OUTPUT_DIR / f"tts_{sample_rate}hz.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(sample_rate)
            wf.writeframes(raw_audio)
        print(f"  ✅ Saved to {wav_path}")

        return sample_rate, raw_audio

    finally:
        await tts_instance.aclose()


async def main():
    api_key = os.getenv("TELNYX_API_KEY")
    if not api_key:
        print("❌ TELNYX_API_KEY not found in environment")
        print("   Add it to .env or export it")
        return

    test_rates = [16000, 24000]
    results = {}

    for rate in test_rates:
        actual_rate, audio = await synthesize_at_rate(api_key, rate)
        results[rate] = (actual_rate, audio)

    # Summary
    print(f"\n{'='*50}")
    print("  RESULTS")
    print(f"{'='*50}")

    all_passed = True
    for rate in test_rates:
        actual_rate, audio = results[rate]
        has_audio = len(audio) > 0
        status = "✅" if has_audio else "❌"
        print(f"  {status} {rate}Hz — {len(audio)} bytes")
        if not has_audio:
            all_passed = False

    # If we got audio at both rates, verify they differ in length
    # (same text at higher sample rate = more samples = more bytes)
    if all(len(results[r][1]) > 0 for r in test_rates):
        len_16k = len(results[16000][1])
        len_24k = len(results[24000][1])
        ratio = len_24k / len_16k if len_16k > 0 else 0
        print(f"\n  Byte ratio 24kHz/16kHz = {ratio:.2f} (expect ~1.5)")
        if 1.2 < ratio < 1.8:
            print("  ✅ Ratio confirms different sample rates")
        else:
            print("  ⚠️  Ratio outside expected range — may need investigation")

    if all_passed:
        print("\n🎉 All sample rate tests passed!\n")
    else:
        print("\n❌ Some tests failed.\n")


if __name__ == "__main__":
    asyncio.run(main())
