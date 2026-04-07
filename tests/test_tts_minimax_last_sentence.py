#!/usr/bin/env python3
"""
Repro test: MiniMax TTS drops the last sentence.

Customer report: When using MiniMax TTS (speech-02-turbo), the last sentence
of the LLM inference is consistently not voiced. Rime does not exhibit this.

This test sends a multi-sentence text through streaming TTS and verifies
that ALL sentences produce audio output — especially the last one.

Usage:
    uv run python tests/test_tts_minimax_last_sentence.py
"""

import asyncio
import json
import os
import wave
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# The exact repro from the customer report
FULL_TEXT = (
    "Well, thank you! I'm doing great. What's on your mind? "
    "Is there anything I can help you with today?"
)

# Split into sentences to simulate LLM streaming token-by-token
SENTENCES = [
    "Well, thank you! ",
    "I'm doing great. ",
    "What's on your mind? ",
    "Is there anything I can help you with today?",
]

# Voices to test — MiniMax (broken) vs Rime (works) vs Telnyx (baseline)
VOICES = {
    "minimax": "Minimax.speech-02-turbo.Calm_Woman",
    # Uncomment to compare against working providers:
    # "rime": "Rime.mist.luna",
    # "telnyx": "Telnyx.NaturalHD.astra",
}

WS_BASE = "wss://api.telnyx.com/v2/text-to-speech/speech"
TTS_ENDPOINT = WS_BASE
SAMPLE_RATE = 24000


async def test_full_text_single_segment(api_key: str, label: str, voice: str) -> dict:
    """
    Send the full text as a single segment (like the customer's TelnyxTTSFactory).
    This is how the LiveKit plugin works: collect text, open WS, send all, close.
    """
    url = f"{WS_BASE}?voice={voice}"
    headers = {"Authorization": f"Bearer {api_key}"}

    session = aiohttp.ClientSession()
    audio_chunks: list[bytes] = []
    messages_received: list[dict] = []

    try:
        ws = await session.ws_connect(url, headers=headers)

        # Handshake
        await ws.send_str(json.dumps({"text": " "}))

        # Send full text as one message (this is what the plugin does)
        await ws.send_str(json.dumps({"text": FULL_TEXT}))

        # EOS
        await ws.send_str(json.dumps({"text": ""}))

        # Collect all responses
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                messages_received.append(data)

                if data.get("error"):
                    print(f"  ❌ {label} (single): Error: {data['error']}")
                    break

                audio_b64 = data.get("audio")
                if audio_b64:
                    import base64
                    audio_chunks.append(base64.b64decode(audio_b64))

                if data.get("isFinal"):
                    break

            elif msg.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSING,
            ):
                break

        await ws.close()
    finally:
        await session.close()

    raw_audio = b"".join(audio_chunks)

    # MiniMax returns MP3 — save as .mp3 (not WAV)
    mp3_path = OUTPUT_DIR / f"minimax_single_{label}.mp3"
    if raw_audio:
        mp3_path.write_bytes(raw_audio)

    return {
        "label": label,
        "mode": "single_segment",
        "audio_bytes": len(raw_audio),
        "chunks": len(audio_chunks),
        "messages": len(messages_received),
        "wav_path": str(mp3_path) if raw_audio else None,
        "raw_audio": raw_audio,
    }


async def test_streamed_sentences(api_key: str, label: str, voice: str) -> dict:
    """
    Send sentences one at a time with flush, simulating how the LiveKit agent
    pipeline would stream LLM output sentence-by-sentence.
    """
    url = f"{WS_BASE}?voice={voice}"
    headers = {"Authorization": f"Bearer {api_key}"}

    session = aiohttp.ClientSession()
    audio_chunks: list[bytes] = []
    messages_received: list[dict] = []

    try:
        ws = await session.ws_connect(url, headers=headers)

        # Handshake
        await ws.send_str(json.dumps({"text": " "}))

        # Send each sentence individually (simulating LLM token streaming)
        for sentence in SENTENCES:
            await ws.send_str(json.dumps({"text": sentence}))
            # Small delay to simulate LLM token arrival
            await asyncio.sleep(0.05)

        # EOS
        await ws.send_str(json.dumps({"text": ""}))

        # Collect all responses
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                messages_received.append(data)

                if data.get("error"):
                    print(f"  ❌ {label} (streamed): Error: {data['error']}")
                    break

                audio_b64 = data.get("audio")
                if audio_b64:
                    import base64
                    audio_chunks.append(base64.b64decode(audio_b64))

                if data.get("isFinal"):
                    break

            elif msg.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSING,
            ):
                break

        await ws.close()
    finally:
        await session.close()

    raw_audio = b"".join(audio_chunks)

    # MiniMax returns MP3 — save as .mp3
    mp3_path = OUTPUT_DIR / f"minimax_streamed_{label}.mp3"
    if raw_audio:
        mp3_path.write_bytes(raw_audio)

    return {
        "label": label,
        "mode": "streamed_sentences",
        "audio_bytes": len(raw_audio),
        "chunks": len(audio_chunks),
        "messages": len(messages_received),
        "wav_path": str(mp3_path) if raw_audio else None,
        "raw_audio": raw_audio,
    }


async def test_via_plugin_raw(api_key: str, label: str, voice: str) -> dict:
    """
    Bypass the plugin's MP3 decoder — use the plugin's WebSocket path but
    capture raw MP3 bytes to isolate whether the issue is in WS recv or decoding.
    """
    import base64 as b64mod

    url = f"{TTS_ENDPOINT}?voice={voice}"
    headers = {"Authorization": f"Bearer {api_key}"}

    session = aiohttp.ClientSession()
    raw_mp3_chunks: list[bytes] = []
    ws_messages: list[dict] = []

    try:
        ws = await session.ws_connect(url, headers=headers)

        # Same flow as plugin's _run_ws: handshake, text, EOS
        await ws.send_str(json.dumps({"text": " "}))
        await ws.send_str(json.dumps({"text": FULL_TEXT}))
        await ws.send_str(json.dumps({"text": ""}))

        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                ws_messages.append({
                    "has_audio": data.get("audio") is not None,
                    "audio_len": len(data.get("audio", "") or ""),
                    "isFinal": data.get("isFinal", False),
                    "cached": data.get("cached"),
                    "text": data.get("text"),
                })
                audio_b64 = data.get("audio")
                if audio_b64:
                    raw_mp3_chunks.append(b64mod.b64decode(audio_b64))
                if data.get("isFinal"):
                    break
            elif msg.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSING,
            ):
                break

        await ws.close()
    finally:
        await session.close()

    raw_mp3 = b"".join(raw_mp3_chunks)

    # Save raw MP3 (bypassing decoder entirely)
    mp3_path = OUTPUT_DIR / f"minimax_plugin_raw_{label}.mp3"
    if raw_mp3:
        mp3_path.write_bytes(raw_mp3)

    # Log all WS messages for diagnosis
    print(f"      WS messages received: {len(ws_messages)}")
    for i, m in enumerate(ws_messages):
        print(f"        [{i}] audio={m['has_audio']} audio_b64_len={m['audio_len']} "
              f"isFinal={m['isFinal']} cached={m['cached']} text={m['text']!r:.40}")

    return {
        "label": label,
        "mode": "plugin_raw",
        "audio_bytes": len(raw_mp3),
        "chunks": len(raw_mp3_chunks),
        "messages": len(ws_messages),
        "wav_path": str(mp3_path) if raw_mp3 else None,
        "raw_audio": raw_mp3,
    }


async def test_via_plugin(api_key: str, label: str, voice: str) -> dict:
    """
    Use the actual TelnyxTTS plugin (same path as the customer).
    This tests the full plugin pipeline including sentence segmentation.
    """
    from livekit.plugins import telnyx

    # MiniMax PCM returns 16kHz — match it to avoid sample rate mismatch
    plugin_sample_rate = 16000 if "minimax" in voice.lower() else SAMPLE_RATE
    tts_inst = telnyx.TTS(
        voice=voice,
        api_key=api_key,
        sample_rate=plugin_sample_rate,
    )

    audio_chunks: list[bytes] = []

    try:
        # Use streaming mode (what the agent pipeline uses)
        stream = tts_inst.stream()

        # Push sentences like LLM would
        for sentence in SENTENCES:
            stream.push_text(sentence)
        stream.flush()
        stream.end_input()

        async for event in stream:
            if hasattr(event, "frame") and event.frame and hasattr(event.frame, "data"):
                audio_chunks.append(event.frame.data.tobytes())

    except Exception as e:
        import traceback
        print(f"  ❌ {label} (plugin): {e}")
        traceback.print_exc()
        raw_audio = b"".join(audio_chunks)
        wav_path = OUTPUT_DIR / f"minimax_plugin_{label}.wav"
        if raw_audio:
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(raw_audio)
        return {
            "label": label,
            "mode": "plugin",
            "error": str(e),
            "audio_bytes": len(raw_audio),
            "chunks": len(audio_chunks),
            "wav_path": str(wav_path) if raw_audio else None,
        }
    finally:
        await tts_inst.aclose()

    raw_audio = b"".join(audio_chunks)

    wav_path = OUTPUT_DIR / f"minimax_plugin_{label}.wav"
    if raw_audio:
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(plugin_sample_rate)
            wf.writeframes(raw_audio)

    return {
        "label": label,
        "mode": "plugin",
        "audio_bytes": len(raw_audio),
        "chunks": len(audio_chunks),
        "wav_path": str(wav_path) if raw_audio else None,
        "raw_audio": raw_audio,
        "sample_rate": plugin_sample_rate,
    }


async def main():
    api_key = os.getenv("TELNYX_API_KEY")
    if not api_key:
        print("❌ TELNYX_API_KEY not set")
        return

    print("\n🧪 MiniMax TTS Last Sentence Drop — Repro Test")
    print("=" * 60)
    print(f"Full text: {FULL_TEXT!r}")
    print(f"Sentences: {len(SENTENCES)}")
    print()

    all_results = []

    for provider_label, voice in VOICES.items():
        print(f"\n--- {provider_label.upper()} ({voice}) ---\n")

        # Test 1: Single segment (full text at once)
        print("  [1] Single segment (full text)...")
        r1 = await test_full_text_single_segment(api_key, provider_label, voice)
        all_results.append(r1)
        print(f"      → {r1['audio_bytes']} bytes, {r1['chunks']} chunks")
        if r1.get("wav_path"):
            print(f"      → {r1['wav_path']}")

        # Test 2: Streamed sentences (simulating LLM)
        print("  [2] Streamed sentences...")
        r2 = await test_streamed_sentences(api_key, provider_label, voice)
        all_results.append(r2)
        print(f"      → {r2['audio_bytes']} bytes, {r2['chunks']} chunks")
        if r2.get("wav_path"):
            print(f"      → {r2['wav_path']}")

        # Test 3a: Raw WS (same path as plugin but bypass decoder)
        print("  [3a] Plugin WS path — raw MP3 (no decoder)...")
        r3a = await test_via_plugin_raw(api_key, provider_label, voice)
        all_results.append(r3a)
        print(f"      → {r3a['audio_bytes']} bytes, {r3a['chunks']} chunks")
        if r3a.get("wav_path"):
            print(f"      → {r3a['wav_path']}")

        # Test 3b: Via plugin (actual customer path)
        print("  [3b] Via TelnyxTTS plugin (stream mode)...")
        r3 = await test_via_plugin(api_key, provider_label, voice)
        all_results.append(r3)
        print(f"      → {r3['audio_bytes']} bytes, {r3['chunks']} chunks")
        if r3.get("wav_path"):
            print(f"      → {r3['wav_path']}")

    # Summary
    print(f"\n{'=' * 60}")
    print("📊 SUMMARY")
    print(f"{'=' * 60}")
    print(f"\nText: {FULL_TEXT!r}\n")
    print("Listen to each file and check if the last sentence is present:")
    print('  Expected last sentence: "Is there anything I can help you with today?"')
    print()

    for r in all_results:
        status = "✅" if r["audio_bytes"] > 0 else "❌"
        duration_info = ""
        # For plugin mode (PCM in WAV), we can calculate duration
        if r["mode"] == "plugin" and r["audio_bytes"] > 0:
            # 16-bit mono PCM
            duration_s = r["audio_bytes"] / (SAMPLE_RATE * 2)
            duration_info = f" | ~{duration_s:.1f}s"
        print(f"  {status} {r['label']:10s} | {r['mode']:20s} | {r['audio_bytes']:>8} bytes | {r['chunks']} chunks{duration_info}")
        if r.get("wav_path"):
            print(f"     → Listen: {r['wav_path']}")

    print()
    print("🔍 KEY QUESTION: Does the plugin WAV contain the last sentence?")
    print("   If not → bug confirmed in tts-gateway's EOS/flush handling for MiniMax.")
    print("   Compare single vs streamed MP3s — if those have all sentences,")
    print("   the issue is in the Worker's handle_connection_close timing.\n")


if __name__ == "__main__":
    asyncio.run(main())
