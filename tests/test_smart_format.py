#!/usr/bin/env python3
"""
Phase 1 Test: smart_format

Validates that smart_format=True actually changes Deepgram's output.
Uses audio containing numbers, currency, dates — things smart_format
should reformat (e.g., "fifteen hundred" → "$1,500").

Expected behavior:
  - WITHOUT smart_format: numbers spelled out or raw
  - WITH smart_format: numbers formatted ($1,500, dates, etc.)

Usage:
    uv run python tests/test_smart_format.py
"""

import asyncio
import os
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import stt as stt_types
from livekit.plugins import telnyx

load_dotenv()

# Text designed to trigger smart_format differences
# Numbers, currency, dates, phone numbers, URLs
TTS_TEXT = (
    "The invoice total is fifteen hundred dollars and was due on january third twenty twenty six. "
    "Please call us at eight hundred five five five twelve thirty four or visit our website at w w w dot example dot com. "
    "Your account number is one two three four five six seven eight nine."
)


async def generate_test_audio(api_key: str, output_path: Path) -> bool:
    """Generate test audio using Telnyx TTS via a simple WebSocket call."""
    print(f"🔊 Generating test audio with TTS...")
    print(f"   Text: {TTS_TEXT[:80]}...")

    # Use say command as a quick fallback to generate WAV with the target text
    # (We need controlled audio content, not just any audio)
    try:
        proc = subprocess.run(
            ["say", "-o", str(output_path), "--data-format=LEI16@16000", TTS_TEXT],
            capture_output=True, timeout=30,
        )
        if proc.returncode == 0:
            print(f"   ✅ Generated: {output_path}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    print("   ❌ Could not generate audio (need macOS 'say' command)")
    return False


async def transcribe(api_key: str, audio_path: Path, label: str, **stt_kwargs) -> str | None:
    """Transcribe audio and return the transcript."""
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")

    stt = telnyx.STT(
        language="en",
        transcription_engine="deepgram",
        interim_results=True,
        api_key=api_key,
        **stt_kwargs,
    )

    try:
        stream = stt.stream()

        with open(audio_path, "rb") as f:
            audio_data = f.read()

        pcm_data = audio_data[44:]  # Skip WAV header
        frame = rtc.AudioFrame(
            data=pcm_data,
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=len(pcm_data) // 2,
        )
        stream.push_frame(frame)
        stream.end_input()

        finals = []
        async for event in stream:
            if event.type == stt_types.SpeechEventType.FINAL_TRANSCRIPT:
                if event.alternatives:
                    finals.append(event.alternatives[0].text)

        transcript = " ".join(finals)
        print(f"  Result: {transcript}")
        return transcript

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

    audio_path = Path(__file__).parent / "assets" / "smart_format_test.wav"

    # Generate audio with numbers/currency/dates
    if not audio_path.exists():
        if not await generate_test_audio(api_key, audio_path):
            return
    else:
        print(f"🔊 Using existing audio: {audio_path}")

    print("\n🧪 smart_format Test")
    print("="*60)

    # Test A: WITHOUT smart_format
    without = await transcribe(
        api_key, audio_path,
        "A) Baseline — no smart_format",
        model="nova-3",
    )

    # Test B: WITH smart_format=True
    with_sf = await transcribe(
        api_key, audio_path,
        "B) smart_format=True",
        model="nova-3",
        smart_format=True,
    )

    # Compare
    print(f"\n{'='*60}")
    print("📊 COMPARISON")
    print(f"{'='*60}")
    print(f"  Without: {without}")
    print(f"  With:    {with_sf}")
    print()

    if without and with_sf and without != with_sf:
        print("  ✅ smart_format CONFIRMED — output differs!")
        print("  The Telnyx backend IS forwarding smart_format to Deepgram.")
    elif without and with_sf and without == with_sf:
        print("  ⚠️  Output is identical.")
        print("  Either smart_format isn't being forwarded, or the default")
        print("  engine already applies similar formatting.")
    else:
        print("  ❌ One or both tests failed.")


if __name__ == "__main__":
    asyncio.run(main())
