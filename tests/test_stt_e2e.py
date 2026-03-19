#!/usr/bin/env python3
"""
E2E tests: STT across all supported Deepgram models via Telnyx.

Each test uses the exact constructor from the README usage examples to
verify documented examples actually work against the live API.

Uses tests/assets/flux_test.wav as the test audio for all models.

Usage:
    TELNYX_API_KEY=... .venv/bin/python tests/test_stt_e2e.py
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import stt as stt_types
from livekit.plugins import telnyx

load_dotenv()

ASSETS = Path(__file__).parent / "assets"
TEST_AUDIO = ASSETS / "flux_test.wav"
STREAM_TIMEOUT = 60


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

async def run_stt(stt_inst: telnyx.deepgram.STT, label: str,
                  timeout: int = STREAM_TIMEOUT) -> str | None:
    """Push test audio through an STT instance and return the transcript."""
    try:
        stream = stt_inst.stream()
        with open(TEST_AUDIO, "rb") as f:
            pcm = f.read()[44:]  # skip WAV header
        frame = rtc.AudioFrame(
            data=pcm, sample_rate=16000, num_channels=1,
            samples_per_channel=len(pcm) // 2,
        )
        stream.push_frame(frame)
        stream.end_input()

        finals = []
        async def collect():
            async for event in stream:
                if (event.type == stt_types.SpeechEventType.FINAL_TRANSCRIPT
                        and event.alternatives):
                    finals.append(event.alternatives[0].text)

        try:
            await asyncio.wait_for(collect(), timeout=timeout)
        except asyncio.TimeoutError:
            pass

        transcript = " ".join(finals) if finals else None
        status = "✅" if transcript else "❌"
        print(f"  {status} {label}")
        if transcript:
            print(f'     → "{transcript}"')
        else:
            print(f"     → no transcript")
        return transcript

    except Exception as e:
        print(f"  ❌ {label}")
        print(f"     Error: {e}")
        return None
    finally:
        await stt_inst.aclose()


# ---------------------------------------------------------------------------
# Tests — constructors mirror README usage examples exactly
# ---------------------------------------------------------------------------

def make_nova3_full(api_key: str) -> telnyx.deepgram.STT:
    """README Nova-3 example — all documented params."""
    return telnyx.deepgram.STT(
        model="nova-3",
        language="en",
        interim_results=True,
        api_key=api_key,
        # Always enabled on Telnyx — no need to set them
        smart_format=True,
        numerals=True,
        # Keyword boosting (Nova-3)
        keyterm=["Telnyx", "LiveKit"],
        # Deepgram defaults — changing them not supported on Telnyx, yet
        punctuate=True,
        no_delay=True,
        endpointing=25,
        filler_words=True,
        profanity_filter=False,
        vad_events=True,
        diarize=False,
    )


def make_nova3_minimal(api_key: str) -> telnyx.deepgram.STT:
    """Nova-3 with just required params."""
    return telnyx.deepgram.STT(
        model="nova-3",
        language="en",
        api_key=api_key,
    )


def make_nova2_full(api_key: str) -> telnyx.deepgram.STT:
    """README Nova-2 example — all documented params."""
    return telnyx.deepgram.STT(
        model="nova-2",
        language="en",
        interim_results=True,
        api_key=api_key,
        # Always enabled on Telnyx — no need to set them
        smart_format=True,
        numerals=True,
        # Keyword boosting (Nova-2 — weighted)
        keywords=["Telnyx:2.0", "LiveKit:1.5"],
        # Deepgram defaults — changing them not supported on Telnyx, yet
        punctuate=True,
        no_delay=True,
        endpointing=25,
        filler_words=True,
        profanity_filter=False,
        vad_events=True,
        diarize=False,
    )


def make_nova2_minimal(api_key: str) -> telnyx.deepgram.STT:
    """Nova-2 with just required params."""
    return telnyx.deepgram.STT(
        model="nova-2",
        language="en",
        api_key=api_key,
    )


def make_flux_full(api_key: str) -> telnyx.deepgram.STT:
    """README Flux example — all documented params."""
    return telnyx.deepgram.STT(
        model="flux",
        language="en",
        interim_results=True,
        api_key=api_key,
        # Keyword boosting (Flux)
        keyterm=["Telnyx", "LiveKit"],
        # Flux end-of-turn detection
        eot_threshold=0.5,
        eot_timeout_ms=3000,
        eager_eot_threshold=0.3,
        # Deepgram defaults — changing them not supported on Telnyx, yet
        punctuate=True,
        no_delay=True,
        endpointing=25,
        filler_words=True,
        profanity_filter=False,
        vad_events=True,
        diarize=False,
    )


def make_flux_minimal(api_key: str) -> telnyx.deepgram.STT:
    """Flux with just required params."""
    return telnyx.deepgram.STT(
        model="flux",
        language="en",
        api_key=api_key,
    )


def make_flux_eot_only(api_key: str) -> telnyx.deepgram.STT:
    """Flux with only EOT params."""
    return telnyx.deepgram.STT(
        model="flux",
        language="en",
        api_key=api_key,
        eot_threshold=0.8,
        eot_timeout_ms=2000,
        eager_eot_threshold=0.5,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    api_key = os.getenv("TELNYX_API_KEY")
    if not api_key:
        print("❌ TELNYX_API_KEY not set")
        return

    if not TEST_AUDIO.exists():
        print(f"❌ Test audio not found: {TEST_AUDIO}")
        return

    tests = [
        # (section, label, constructor)
        ("Nova-3", "full README example", make_nova3_full),
        ("Nova-3", "minimal", make_nova3_minimal),
        ("Nova-2", "full README example", make_nova2_full),
        ("Nova-2", "minimal", make_nova2_minimal),
        ("Flux",   "full README example", make_flux_full),
        ("Flux",   "minimal", make_flux_minimal),
        ("Flux",   "EOT params only", make_flux_eot_only),
    ]

    results = {}
    current_section = None

    for section, label, make_stt in tests:
        if section != current_section:
            current_section = section
            print(f"\n🧪 {section} Tests")
            print("=" * 55)
            results[section] = [0, 0]

        stt_inst = make_stt(api_key)
        transcript = await run_stt(stt_inst, f"{section} {label}")
        results[section][1] += 1
        if transcript:
            results[section][0] += 1

    # --- Summary ---
    print(f"\n{'=' * 55}")
    print("📊 RESULTS")
    print(f"{'=' * 55}")
    total_passed = 0
    total_tests = 0
    for section, (p, t) in results.items():
        status = "✅" if p == t else "⚠️"
        print(f"  {status} {section}: {p}/{t}")
        total_passed += p
        total_tests += t

    print(f"\n  Total: {total_passed}/{total_tests}")
    if total_passed == total_tests:
        print("✅ All STT E2E tests passed")
    else:
        print(f"⚠️  {total_tests - total_passed} test(s) failed")


if __name__ == "__main__":
    asyncio.run(main())
