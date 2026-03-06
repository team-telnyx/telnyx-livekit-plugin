#!/usr/bin/env python3
"""
E2E test: smart_format with realistic speech input.

Validates that Deepgram smart_format correctly handles:
- Currency (spoken dollars → $X,XXX.XX)
- Dates (spoken date → MM/DD/YYYY)
- Numbers (spoken digits → numeric)

Usage:
    uv run python tests/test_smart_format_e2e.py
"""

import asyncio
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import stt as stt_types
from livekit.plugins import telnyx

load_dotenv()

TESTS = [
    {
        "name": "Currency",
        "speech": "The total is twenty three hundred and fifty dollars and sixty seven cents",
        "check": lambda t: "$" in t and "2,350" in t,
    },
    {
        "name": "Date",
        "speech": "The due date is february fourteenth twenty twenty six",
        "check": lambda t: "2026" in t and ("02/14" in t or "2/14" in t or "February 14" in t),
    },
    {
        "name": "Digits",
        "speech": "My account number is one two three four five six seven eight nine",
        "check": lambda t: "123456789" in t.replace(" ", ""),
    },
]


async def transcribe(api_key: str, audio_path: Path, **kwargs) -> str | None:
    stt_inst = telnyx.STT(
        language="en", transcription_engine="deepgram", api_key=api_key, **kwargs,
    )
    try:
        stream = stt_inst.stream()
        with open(audio_path, "rb") as f:
            pcm = f.read()[44:]
        frame = rtc.AudioFrame(
            data=pcm, sample_rate=16000, num_channels=1,
            samples_per_channel=len(pcm) // 2,
        )
        stream.push_frame(frame)
        stream.end_input()

        finals = []
        async for event in stream:
            if event.type == stt_types.SpeechEventType.FINAL_TRANSCRIPT and event.alternatives:
                finals.append(event.alternatives[0].text)
        return " ".join(finals)
    except Exception as e:
        return None
    finally:
        await stt_inst.aclose()


def generate_audio(speech: str, path: Path) -> bool:
    if path.exists():
        return True
    try:
        subprocess.run(
            ["say", "-o", str(path), "--data-format=LEI16@16000", speech],
            capture_output=True, timeout=30, check=True,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return False


async def main():
    api_key = os.getenv("TELNYX_API_KEY")
    if not api_key:
        print("❌ TELNYX_API_KEY not set")
        return

    assets = Path(__file__).parent / "assets"
    assets.mkdir(exist_ok=True)

    print("🧪 Smart Format E2E Tests")
    print("=" * 60)

    all_pass = True
    for test in TESTS:
        audio_path = assets / f"sf_{test['name'].lower()}.wav"
        if not generate_audio(test["speech"], audio_path):
            print(f"  ❌ {test['name']}: could not generate audio")
            all_pass = False
            continue

        transcript = await transcribe(api_key, audio_path, model="nova-3")
        if not transcript:
            print(f"  ❌ {test['name']}: transcription failed")
            all_pass = False
            continue

        passed = test["check"](transcript)
        status = "✅" if passed else "❌"
        print(f"  {status} {test['name']}")
        print(f"     Input:  \"{test['speech']}\"")
        print(f"     Output: \"{transcript}\"")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("✅ All smart format checks passed")
    else:
        print("⚠️  Some checks failed — review output above")


if __name__ == "__main__":
    asyncio.run(main())
