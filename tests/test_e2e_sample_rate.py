#!/usr/bin/env python3
"""
E2E Test: Verify TTS sample_rate works through a LiveKit room.

1. Starts a test agent subprocess with TTS sample_rate=24000
2. Connects a test client to a LiveKit Cloud room
3. Client publishes test audio (48kHz WAV)
4. Agent: STT transcribes → TTS responds at 24kHz → publishes audio back
5. Client verifies it received audio frames from the agent

Pass criteria:
  - Agent joins and subscribes to client audio
  - Client receives >10 audio frames back from agent
  - Agent reports TTS_COMPLETE with sample_rate=24000

Usage:
    uv run python tests/test_e2e_sample_rate.py
"""

import asyncio
import os
import signal
import subprocess
import sys
import uuid
import wave

from dotenv import load_dotenv
from livekit import api, rtc

load_dotenv()

# Config — uses same LiveKit Cloud project as the original e2e tests
LIVEKIT_URL = os.getenv('LIVEKIT_URL', 'wss://jonrestaurantvoiceagent-pru12ky9.livekit.cloud')
LIVEKIT_API_KEY = os.getenv('LIVEKIT_API_KEY', 'APIF65qYcCPsjE6')
LIVEKIT_API_SECRET = os.getenv('LIVEKIT_API_SECRET', 'ZxpE18cEq0e0o6BunG8OSgybsMJCAI49ePeXtxBUSnQA')
TELNYX_API_KEY = os.getenv('TELNYX_API_KEY')
TTS_SAMPLE_RATE = int(os.getenv('TTS_SAMPLE_RATE', '24000'))

TEST_AUDIO_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'test_speech_48k.wav')
TEST_ROOM_NAME = f"e2e-sample-rate-{uuid.uuid4().hex[:8]}"


def create_token(room_name: str, identity: str) -> str:
    token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    token.with_identity(identity)
    token.with_grants(api.VideoGrants(
        room_join=True,
        room=room_name,
        can_publish=True,
        can_subscribe=True,
    ))
    return token.to_jwt()


async def run_e2e_test():
    print("=" * 60)
    print(f"E2E Test: TTS sample_rate={TTS_SAMPLE_RATE} through LiveKit")
    print("=" * 60)

    if not os.path.exists(TEST_AUDIO_PATH):
        print(f"❌ Test audio not found: {TEST_AUDIO_PATH}")
        return False

    if not TELNYX_API_KEY:
        print("❌ TELNYX_API_KEY not set")
        return False

    print(f"  LiveKit: {LIVEKIT_URL}")
    print(f"  Room: {TEST_ROOM_NAME}")
    print(f"  TTS sample_rate: {TTS_SAMPLE_RATE}")
    print()

    agent_process = None

    try:
        # === Step 1: Start agent ===
        print("📡 Starting test agent...")

        agent_script = os.path.join(
            os.path.dirname(__file__), 'agents', 'test_agent_sample_rate.py'
        )

        agent_env = os.environ.copy()
        agent_env.update({
            'LIVEKIT_URL': LIVEKIT_URL,
            'LIVEKIT_API_KEY': LIVEKIT_API_KEY,
            'LIVEKIT_API_SECRET': LIVEKIT_API_SECRET,
            'TELNYX_API_KEY': TELNYX_API_KEY,
            'TEST_ROOM_NAME': TEST_ROOM_NAME,
            'TTS_SAMPLE_RATE': str(TTS_SAMPLE_RATE),
        })

        agent_log = open('/tmp/agent_sample_rate_e2e.log', 'w')
        agent_process = subprocess.Popen(
            [sys.executable, '-u', agent_script],
            env=agent_env,
            stdout=agent_log,
            stderr=subprocess.STDOUT,
        )

        await asyncio.sleep(5)

        if agent_process.poll() is not None:
            agent_log.flush()
            with open('/tmp/agent_sample_rate_e2e.log') as f:
                print(f"❌ Agent crashed:\n{f.read()[:1000]}")
            return False

        print("  ✓ Agent started")
        print()

        # === Step 2: Connect client ===
        print("🔗 Connecting test client...")

        token = create_token(TEST_ROOM_NAME, "e2e-test-client")
        room = rtc.Room()

        received_audio = []
        received_data = []
        tts_complete_info = None

        @room.on("track_subscribed")
        def on_track_subscribed(track: rtc.Track, publication, participant):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                print(f"  ✓ Subscribed to audio from [{participant.identity}]")

                async def collect_audio():
                    audio_stream = rtc.AudioStream(track)
                    async for event in audio_stream:
                        if hasattr(event, 'frame'):
                            received_audio.append(event.frame)
                            if len(received_audio) % 20 == 0:
                                print(f"  ... {len(received_audio)} audio frames received")

                asyncio.create_task(collect_audio())

        @room.on("data_received")
        def on_data(data_packet):
            nonlocal tts_complete_info
            try:
                text = data_packet.data.decode('utf-8') if isinstance(data_packet.data, bytes) else str(data_packet.data)
                received_data.append(text)
                print(f"  ✓ Data: {text[:100]}")

                if text.startswith("TTS_COMPLETE:"):
                    tts_complete_info = text
            except Exception:
                pass

        await room.connect(LIVEKIT_URL, token)
        print(f"  ✓ Connected to room")

        # Wait for agent to join
        await asyncio.sleep(8)
        participants = [p.identity for p in room.remote_participants.values()]
        print(f"  Participants: {participants}")
        print()

        # === Step 3: Publish audio ===
        print("🎤 Publishing test audio...")

        with wave.open(TEST_AUDIO_PATH, 'rb') as w:
            sample_rate = w.getframerate()
            num_channels = w.getnchannels()
            audio_data = w.readframes(w.getnframes())

        source = rtc.AudioSource(sample_rate, num_channels)
        track = rtc.LocalAudioTrack.create_audio_track("test-audio", source)

        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_MICROPHONE
        await room.local_participant.publish_track(track, options)

        chunk_samples = sample_rate // 20  # 50ms
        chunk_bytes = chunk_samples * num_channels * 2

        for i in range(0, len(audio_data), chunk_bytes):
            chunk = audio_data[i:i + chunk_bytes]
            if len(chunk) < chunk_bytes:
                chunk = chunk + bytes(chunk_bytes - len(chunk))

            frame = rtc.AudioFrame(
                data=chunk,
                sample_rate=sample_rate,
                num_channels=num_channels,
                samples_per_channel=chunk_samples,
            )
            await source.capture_frame(frame)
            await asyncio.sleep(0.05)

        print("  ✓ Audio sent")
        print()

        # === Step 4: Wait for response ===
        print("⏳ Waiting for agent response (up to 30s)...")

        for _ in range(60):
            await asyncio.sleep(0.5)
            if tts_complete_info and len(received_audio) > 10:
                break

        print()

        # === Step 5: Results ===
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)

        got_audio = len(received_audio) > 10
        got_completion = tts_complete_info is not None
        correct_rate = f"sample_rate={TTS_SAMPLE_RATE}" in (tts_complete_info or "")

        print(f"  Audio frames received: {len(received_audio)} {'✅' if got_audio else '❌'}")
        print(f"  TTS completion signal: {tts_complete_info or 'none'} {'✅' if got_completion else '❌'}")
        print(f"  Correct sample rate:   {'✅' if correct_rate else '❌'}")

        if received_data:
            print(f"  Data messages:")
            for d in received_data:
                print(f"    → {d}")

        passed = got_audio and correct_rate
        print()
        if passed:
            print(f"🎉 E2E PASSED — TTS at {TTS_SAMPLE_RATE}Hz works end-to-end!")
        else:
            print("❌ E2E FAILED")

        await room.disconnect()
        return passed

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if agent_process:
            print("\n🧹 Stopping agent...")
            agent_process.send_signal(signal.SIGTERM)
            try:
                agent_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                agent_process.kill()

            try:
                with open('/tmp/agent_sample_rate_e2e.log') as f:
                    logs = f.read()
                    print(f"\n📋 Agent logs:\n{'='*40}")
                    print(logs[-2000:] if len(logs) > 2000 else logs)
            except Exception:
                pass


if __name__ == "__main__":
    success = asyncio.run(run_e2e_test())
    sys.exit(0 if success else 1)
