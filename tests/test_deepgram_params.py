#!/usr/bin/env python3
"""
Phase 1 validation: Test that Deepgram parameters (smart_format, keyterm)
are actually forwarded by the Telnyx backend.

Usage:
    uv run python tests/test_deepgram_params.py
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from livekit.plugins import telnyx

load_dotenv()


async def transcribe_with_params(api_key: str, audio_path: Path, label: str, **stt_kwargs):
    """Transcribe audio with given STT params and return the transcript."""
    print(f"\n{'='*60}")
    print(f"Test: {label}")
    print(f"Params: {stt_kwargs}")
    print(f"{'='*60}")

    stt = telnyx.STT(
        language="en",
        transcription_engine="deepgram",
        interim_results=True,
        api_key=api_key,
        **stt_kwargs,
    )

    try:
        stream = stt.stream()

        # Read and send audio
        import struct
        with open(audio_path, "rb") as f:
            audio_data = f.read()

        # Skip WAV header (44 bytes), send as raw PCM frames
        from livekit import rtc
        pcm_data = audio_data[44:]
        
        # Create AudioFrame from PCM data
        frame = rtc.AudioFrame(
            data=pcm_data,
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=len(pcm_data) // 2,  # 16-bit = 2 bytes per sample
        )
        stream.push_frame(frame)
        stream.end_input()

        # Collect results
        finals = []
        interims = []
        async for event in stream:
            from livekit.agents import stt as stt_types
            if event.type == stt_types.SpeechEventType.FINAL_TRANSCRIPT:
                if event.alternatives:
                    text = event.alternatives[0].text
                    finals.append(text)
                    print(f"  FINAL: {text}")
            elif event.type == stt_types.SpeechEventType.INTERIM_TRANSCRIPT:
                if event.alternatives:
                    interims.append(event.alternatives[0].text)

        full_transcript = " ".join(finals)
        print(f"\n  Full transcript: {full_transcript}")
        print(f"  Interim count: {len(interims)}")
        return full_transcript

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None
    finally:
        await stt.aclose()


async def main():
    api_key = os.getenv("TELNYX_API_KEY")
    if not api_key:
        print("❌ TELNYX_API_KEY not set")
        return

    audio_path = Path(__file__).parent / "assets" / "sample.wav"
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        return

    print("🧪 Phase 1 Validation: Deepgram Parameters via Telnyx STT")
    print(f"   Audio: {audio_path}")

    # Test 1: Baseline — no Deepgram params
    baseline = await transcribe_with_params(
        api_key, audio_path,
        "Baseline (no Deepgram params)",
    )

    # Test 2: smart_format=True (should format numbers, dates, punctuation)
    smart = await transcribe_with_params(
        api_key, audio_path,
        "smart_format=True",
        smart_format=True,
        model="nova-3",
    )

    # Test 3: keyterm (should boost recognition of specified terms)
    keyterm = await transcribe_with_params(
        api_key, audio_path,
        "keyterm=['Telnyx', 'LiveKit']",
        keyterm=["Telnyx", "LiveKit"],
        model="nova-3",
    )

    # Test 4: All Phase 1 params together
    combined = await transcribe_with_params(
        api_key, audio_path,
        "Combined (smart_format + keyterm + model)",
        smart_format=True,
        keyterm=["Telnyx", "LiveKit"],
        model="nova-3",
    )

    # Summary
    print(f"\n{'='*60}")
    print("📊 RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Baseline:     {baseline}")
    print(f"  smart_format: {smart}")
    print(f"  keyterm:      {keyterm}")
    print(f"  combined:     {combined}")
    print()

    # Check if smart_format made a difference
    if baseline and smart and baseline != smart:
        print("  ✅ smart_format produced DIFFERENT output — params are being forwarded!")
    elif baseline and smart and baseline == smart:
        print("  ⚠️  smart_format produced SAME output — may not be forwarded, or audio doesn't trigger formatting differences")
    
    if all([baseline, smart, keyterm, combined]):
        print("  ✅ All tests completed successfully — Telnyx backend accepted the params")
    else:
        print("  ❌ Some tests failed")


if __name__ == "__main__":
    asyncio.run(main())
