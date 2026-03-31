#!/usr/bin/env python3
"""
Test: Deepgram keywords (weighted boosting) on Nova-2 via Telnyx STT.

Validates that the `keywords` parameter is forwarded through the Telnyx
backend to Deepgram and affects transcription output.

Usage:
    uv run python tests/test_keywords_nova2.py
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from livekit import rtc
from livekit.plugins import telnyx

load_dotenv()

ASSETS = Path(__file__).parent / "assets"


async def transcribe(api_key: str, audio_path: Path, label: str, **stt_kwargs) -> str:
    print(f"\n{'='*60}")
    print(f"Test: {label}")
    print(f"Params: {stt_kwargs}")
    print(f"{'='*60}")

    stt = telnyx.STT(
        language="en",
        transcription_engine="deepgram",
        api_key=api_key,
        **stt_kwargs,
    )

    stream = stt.stream()

    with open(audio_path, "rb") as f:
        pcm = f.read()[44:]  # skip WAV header

    frame = rtc.AudioFrame(
        data=pcm,
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=len(pcm) // 2,
    )
    stream.push_frame(frame)
    stream.end_input()

    finals = []
    async for event in stream:
        if hasattr(event, "alternatives") and event.alternatives:
            finals.append(event.alternatives[0].text)

    await stt.aclose()
    text = " ".join(finals)
    print(f"  Result: {text}")
    return text


async def main():
    api_key = os.getenv("TELNYX_API_KEY")
    if not api_key:
        print("❌ TELNYX_API_KEY not set")
        return

    audio = ASSETS / "keyterm_test.wav"
    if not audio.exists():
        print(f"❌ Audio not found: {audio}")
        return

    print("🧪 Keywords (Nova-2) — Weighted Boosting Test")
    print(f"   Audio: {audio}")

    # 1. Baseline
    baseline = await transcribe(api_key, audio, "Nova-2 baseline", model="nova-2")

    # 2. Boost a competing word
    boosted = await transcribe(
        api_key, audio,
        "Nova-2 keywords=[Classics:5.0]",
        model="nova-2",
        keywords=["Classics:5.0"],
    )

    # 3. Negative weight
    suppressed = await transcribe(
        api_key, audio,
        "Nova-2 keywords=[Quasics:-10.0, Classics:5.0]",
        model="nova-2",
        keywords=["Quasics:-10.0", "Classics:5.0"],
    )

    # Summary
    print(f"\n{'='*60}")
    print("📊 RESULTS")
    print(f"{'='*60}")
    print(f"  Baseline:    {baseline}")
    print(f"  Boosted:     {boosted}")
    print(f"  Suppressed:  {suppressed}")

    if baseline != boosted:
        print("\n  ✅ keywords changed the transcription — params forwarded!")
    else:
        print("\n  ⚠️  No difference — keywords may not be forwarded, or boost too weak")


if __name__ == "__main__":
    asyncio.run(main())
