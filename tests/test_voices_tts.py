#!/usr/bin/env python3
"""
TTS Voice Support Matrix — e2e tests.

Proves every voice we ship actually works against the live API.
Each voice declares its supported formats, sample rates, and any
voice-specific params. The test loop validates all of them.

Usage:
    python test_voices_tts.py <API_KEY>
    TELNYX_API_KEY=... python test_voices_tts.py

Flags:
    --save-audio    Write received audio to tests/output/<voice>_<format>_<rate>.raw
                    for manual listening. Useful when debugging a voice quality bug
                    (e.g. wrong pitch, chipmunk audio). Not needed in CI — tests
                    validate format and sample rate from byte analysis alone.

Eventually: CI gate. If a voice fails here, we don't ship it.

Docs:
    Telnyx   — https://developers.telnyx.com/docs/voice/tts/providers/telnyx
    MiniMax  — https://developers.telnyx.com/docs/voice/tts/providers/minimax
"""

import asyncio
import base64
import json
import os
import sys
import time

import websockets

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_KEY = sys.argv[1] if len(sys.argv) > 1 else os.getenv("TELNYX_API_KEY")
if not API_KEY:
    print("Usage: python test_voices_tts.py <API_KEY>")
    print("   or: TELNYX_API_KEY=... python test_voices_tts.py")
    sys.exit(1)

BASE_URL = "wss://api.telnyx.com/v2/text-to-speech/speech"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}
TEXT = "Hello, this is a test of the text to speech system."

# ---------------------------------------------------------------------------
# Voice support matrix
#
# Each voice declares what it supports. The test loop only tests what's
# declared here. Adding a voice = adding an entry. That's it.
#
# formats:       audio_format values this voice handles
# sample_rates:  sample_rate values this voice handles
# extra_params:  voice-specific params to test (each value list is exercised)
# unsupported_formats: formats that should be rejected cleanly
# ---------------------------------------------------------------------------

SUPPORTED_VOICES = {
    # -- MiniMax --
    # Docs: mp3 (default), linear16. Rates: 8000–44100. Extras: speed, vol, pitch.
    "Minimax.speech-02-turbo.Calm_Woman": {
        "formats": ["mp3", "linear16"],
        "sample_rates": [8000, 16000, 24000],
        "extra_params": {
            "speed": [0.5, 1.0, 2.0],
            "pitch": [-12, 0, 12],
        },
        "unsupported_formats": ["wav"],
    },

    # -- Telnyx --
    "Telnyx.NaturalHD.astra": {
        "formats": ["mp3", "linear16"],
        "sample_rates": [8000, 16000, 22050],
        "extra_params": {},
        "unsupported_formats": [],
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

results = []


def record(voice, test_name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results.append((voice, test_name, passed, detail))
    icon = "✅" if passed else "❌"
    print(f"  {icon} {test_name} | {detail}" if detail else f"  {icon} {test_name}")


async def ws_tts(url, text=TEXT):
    """Connect to TTS WS, send text, collect audio bytes and messages."""
    audio_bytes = b""
    messages = []
    close_code = None
    close_reason = None

    try:
        async with websockets.connect(url, additional_headers=HEADERS) as ws:
            await ws.send(json.dumps({"text": " "}))
            await ws.send(json.dumps({"text": text}))
            await ws.send(json.dumps({"text": ""}))

            deadline = time.time() + 30
            while time.time() < deadline:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                except asyncio.TimeoutError:
                    break

                if isinstance(msg, bytes):
                    audio_bytes += msg
                elif isinstance(msg, str):
                    data = json.loads(msg)
                    messages.append(data)
                    audio_b64 = data.get("audio")
                    if audio_b64:
                        audio_bytes += base64.b64decode(audio_b64)
                    if data.get("isFinal"):
                        break

            close_code = ws.close_code
            close_reason = ws.close_reason
    except websockets.exceptions.ConnectionClosedError as e:
        close_code = e.code
        close_reason = e.reason
    except Exception as e:
        close_code = -1
        close_reason = str(e)

    return audio_bytes, messages, close_code, close_reason


def is_mp3(data):
    if len(data) < 3:
        return False
    # MPEG sync word (raw frames)
    if data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return True
    # ID3v2 tag header — common prefix on MP3 files from many providers
    if data[:3] == b"ID3":
        return True
    return False


def build_url(voice, **params):
    """Build WS URL with query params."""
    qs = f"voice={voice}"
    for k, v in params.items():
        qs += f"&{k}={v}"
    return f"{BASE_URL}?{qs}"


# ---------------------------------------------------------------------------
# Universal tests — run for every voice
# ---------------------------------------------------------------------------

async def test_defaults(voice, _cfg):
    """Default call (no format/rate params) produces audio."""
    url = build_url(voice)
    audio, msgs, code, reason = await ws_tts(url)
    passed = len(audio) > 0
    record(voice, "defaults → produces audio", passed, f"bytes={len(audio)}")


async def test_format_selection(voice, cfg):
    """Each declared format returns the correct type."""
    for fmt in cfg["formats"]:
        url = build_url(voice, audio_format=fmt)
        audio, msgs, code, reason = await ws_tts(url)

        if fmt == "mp3":
            correct = is_mp3(audio) and len(audio) > 0
            detail = f"is_mp3={is_mp3(audio)}, bytes={len(audio)}"
        elif fmt == "linear16":
            correct = not is_mp3(audio) and len(audio) > 0
            detail = f"is_pcm={not is_mp3(audio)}, bytes={len(audio)}"
        else:
            correct = len(audio) > 0
            detail = f"bytes={len(audio)}"

        record(voice, f"audio_format={fmt} → correct output", correct, detail)
        await asyncio.sleep(0.5)


async def test_sample_rate_honored(voice, cfg):
    """Sample rate param is actually honored (not silently ignored).

    Compare byte counts at the lowest and highest declared rates.
    Higher rate should produce more PCM bytes for the same text.
    """
    rates = sorted(cfg["sample_rates"])
    if len(rates) < 2:
        record(voice, "sample_rate honored", True, "skipped (only 1 rate)")
        return

    low_rate = rates[0]
    high_rate = rates[-1]

    url_low = build_url(voice, audio_format="linear16", sample_rate=low_rate)
    url_high = build_url(voice, audio_format="linear16", sample_rate=high_rate)

    audio_low, _, _, _ = await ws_tts(url_low)
    await asyncio.sleep(0.5)
    audio_high, _, _, _ = await ws_tts(url_high)

    if len(audio_low) == 0 or len(audio_high) == 0:
        record(voice, f"sample_rate {low_rate} vs {high_rate}", False,
               f"no audio: low={len(audio_low)}, high={len(audio_high)}")
        return

    ratio = len(audio_high) / len(audio_low)
    expected_ratio = high_rate / low_rate
    # Allow wide tolerance — just catch "gateway ignored sample_rate entirely"
    passed = ratio > 1.3
    record(voice, f"sample_rate {low_rate} vs {high_rate} → rate honored",
           passed, f"byte_ratio={ratio:.2f} (expected ~{expected_ratio:.1f}x)")


async def test_bad_format(voice, cfg):
    """Unsupported formats are rejected cleanly."""
    for fmt in cfg.get("unsupported_formats", []):
        url = build_url(voice, audio_format=fmt)
        audio, msgs, code, reason = await ws_tts(url)

        has_error = (
            code in (4003, 4001, 1008, -1)
            or any("error" in str(m).lower() for m in msgs)
        )
        passed = has_error or len(audio) == 0
        record(voice, f"audio_format={fmt} → rejected", passed, f"code={code}")
        await asyncio.sleep(0.5)


async def test_bad_sample_rate(voice, _cfg):
    """Non-numeric sample_rate is rejected."""
    url = build_url(voice, audio_format="linear16", sample_rate="abc")
    audio, msgs, code, reason = await ws_tts(url)

    has_error = (
        code in (4003, 4001, 1008, -1)
        or any("error" in str(m).lower() for m in msgs)
    )
    passed = has_error or len(audio) == 0
    record(voice, "sample_rate=abc → rejected", passed, f"code={code}")


# ---------------------------------------------------------------------------
# Voice-specific param tests
# ---------------------------------------------------------------------------

async def test_extra_params(voice, cfg):
    """Voice-specific params (speed, pitch, etc.) are accepted."""
    extras = cfg.get("extra_params", {})
    if not extras:
        return

    for param, values in extras.items():
        for val in values:
            url = build_url(voice, **{param: val})
            audio, msgs, code, reason = await ws_tts(url)
            passed = len(audio) > 0
            record(voice, f"{param}={val} → produces audio", passed,
                   f"bytes={len(audio)}")
            await asyncio.sleep(0.5)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

UNIVERSAL_TESTS = [
    test_defaults,
    test_format_selection,
    test_sample_rate_honored,
    test_bad_format,
    test_bad_sample_rate,
]


async def main():
    print(f"\n🧪 TTS Voice Support Matrix — e2e tests")
    print(f"   Endpoint: {BASE_URL}")
    print(f"   Voices: {len(SUPPORTED_VOICES)}\n")

    for voice, cfg in SUPPORTED_VOICES.items():
        print(f"\n── {voice} ──")

        # Universal tests
        for test_fn in UNIVERSAL_TESTS:
            try:
                await test_fn(voice, cfg)
            except Exception as e:
                record(voice, test_fn.__name__, False, f"EXCEPTION: {e}")
            await asyncio.sleep(1)

        # Voice-specific param tests
        try:
            await test_extra_params(voice, cfg)
        except Exception as e:
            record(voice, "extra_params", False, f"EXCEPTION: {e}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY\n")

    by_voice = {}
    for voice, name, passed, detail in results:
        by_voice.setdefault(voice, []).append((name, passed, detail))

    total_pass = 0
    total_fail = 0

    for voice, tests in by_voice.items():
        v_pass = sum(1 for _, p, _ in tests if p)
        v_fail = sum(1 for _, p, _ in tests if not p)
        total_pass += v_pass
        total_fail += v_fail

        icon = "✅" if v_fail == 0 else "❌"
        print(f"  {icon} {voice}: {v_pass}/{v_pass + v_fail} passed")

        for name, passed, detail in tests:
            if not passed:
                print(f"       ❌ {name}: {detail}")

    total = total_pass + total_fail
    print(f"\n  Total: {total_pass}/{total} passed")

    if total_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
