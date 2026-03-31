"""
Full E2E Test: LiveKit Room + Telnyx STT/TTS

What it does:

1. Starts a test agent as a subprocess — this agent uses Telnyx STT/TTS only (no LLM)

2. Connects a test client to LiveKit Cloud room using the SDK

3. Agent joins the room and subscribes to audio

4. Client publishes test audio — a recorded wav file, sent in 50ms chunks to simulate real-time speech

5. Agent receives audio → STT transcribes it → Agent responds → TTS generates audio

6. Client subscribes to agent's audio track and collects incoming frames

7. Test passes if: we received more than 10 audio frames back from the agent

Why this validates the fixes:
- If TTS crashes, end_segment() still gets called (finally block)
- If STT crashes, stream.aclose() still gets called (finally block)
- Both plugins survive errors and clean up properly

To run:
    cd ~/Code/agents-telnyx
    TELNYX_API_KEY="your-key" uv run python tests/test_e2e_livekit.py
"""

import asyncio
import os
import sys
import subprocess
import signal
import wave
import uuid

from livekit import api, rtc

# Configuration
LIVEKIT_URL = os.getenv('LIVEKIT_URL', 'wss://jonrestaurantvoiceagent-pru12ky9.livekit.cloud')
LIVEKIT_API_KEY = os.getenv('LIVEKIT_API_KEY', 'APIF65qYcCPsjE6')
LIVEKIT_API_SECRET = os.getenv('LIVEKIT_API_SECRET', 'ZxpE18cEq0e0o6BunG8OSgybsMJCAI49ePeXtxBUSnQA')
TELNYX_API_KEY = os.getenv('TELNYX_API_KEY')

TEST_AUDIO_PATH = os.path.join(os.path.dirname(__file__), "assets", "test_speech_48k.wav")  # "Hello, this is a test" (48kHz for LiveKit)
# Unique room name to avoid interference from other agents in the LiveKit Cloud project
TEST_ROOM_NAME = f"e2e-telnyx-test-{uuid.uuid4().hex[:8]}"


def create_token(room_name: str, identity: str) -> str:
    """Create a LiveKit access token."""
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
    print("=" * 70)
    print("E2E Test: LiveKit Room + Telnyx STT/TTS")
    print("=" * 70)
    
    # Check prerequisites
    if not os.path.exists(TEST_AUDIO_PATH):
        print(f"❌ Test audio not found: {TEST_AUDIO_PATH}")
        print("   Please provide the audio file first")
        return False
    
    if not TELNYX_API_KEY:
        print("❌ Missing API key: TELNYX_API_KEY")
        return False
    
    print(f"✓ LiveKit URL: {LIVEKIT_URL}")
    print(f"✓ Test room: {TEST_ROOM_NAME}")
    print(f"✓ Test audio: {TEST_AUDIO_PATH}")
    print()
    
    agent_process = None
    
    try:
        # ========== STEP 1: Start Agent ==========
        print("📡 Step 1: Starting test agent...")
        
        agent_script = os.path.join(
            os.path.dirname(__file__), 'agents', 'test_agent.py'
        )
        
        agent_env = os.environ.copy()
        agent_env['LIVEKIT_URL'] = LIVEKIT_URL
        agent_env['LIVEKIT_API_KEY'] = LIVEKIT_API_KEY
        agent_env['LIVEKIT_API_SECRET'] = LIVEKIT_API_SECRET
        agent_env['TELNYX_API_KEY'] = TELNYX_API_KEY
        agent_env['TEST_ROOM_NAME'] = TEST_ROOM_NAME
        
        # Start simple test agent that joins room directly
        # Use -u for unbuffered output so we can see prints in real-time
        agent_log_file = open('/tmp/agent_e2e.log', 'w')
        agent_process = subprocess.Popen(
            [sys.executable, '-u', agent_script],
            env=agent_env,
            stdout=agent_log_file,
            stderr=subprocess.STDOUT,
            cwd=os.path.dirname(agent_script),
        )
        
        # Wait for agent to register
        print("   Waiting for agent to register...")
        await asyncio.sleep(5)
        
        # Show agent logs so far
        agent_log_file.flush()
        with open('/tmp/agent_e2e.log', 'r') as f:
            log_content = f.read()
            if log_content:
                print(f"   Agent logs:\n{log_content[:500]}")
        
        if agent_process.poll() is not None:
            output = agent_process.stdout.read().decode() if agent_process.stdout else ""
            print(f"❌ Agent failed to start:\n{output[:500]}")
            return False
        
        print("   ✓ Agent started")
        print()
        
        # ========== STEP 2: Connect Test Client ==========
        print("🔗 Step 2: Connecting test client (this triggers agent dispatch)...")
        
        token = create_token(TEST_ROOM_NAME, "e2e-test-client")
        room = rtc.Room()
        
        # Track received audio and transcripts
        received_audio = []
        received_transcripts = []
        
        @room.on("participant_connected")
        def on_participant_connected(participant):
            print(f"   ✓ Participant joined: {participant.identity}")
        
        @room.on("track_published")
        def on_track_published(publication, participant):
            print(f"   ✓ Track published by {participant.identity}: {publication.kind}")
        
        @room.on("track_subscribed")
        def on_track_subscribed(track: rtc.Track, publication, participant):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                print(f"   ✓ Subscribed to audio track from [{participant.identity}]")
                
                async def receive_audio():
                    audio_stream = rtc.AudioStream(track)
                    async for event in audio_stream:
                        if hasattr(event, 'frame'):
                            received_audio.append(event.frame)
                            if len(received_audio) % 10 == 0:
                                print(f"   ... received {len(received_audio)} audio frames")
                
                asyncio.create_task(receive_audio())
        
        @room.on("data_received")  
        def on_data(data_packet):
            # Agent might send transcript as data
            try:
                data = data_packet.data if hasattr(data_packet, 'data') else data_packet
                text = data.decode('utf-8') if isinstance(data, bytes) else str(data)
                participant = data_packet.participant if hasattr(data_packet, 'participant') else None
                speaker = participant.identity if participant else "test-agent"
                received_transcripts.append(f"[{speaker}] {text}")
                print(f"   ✓ [{speaker}] Data: {text[:100]}...")
            except Exception as e:
                print(f"   ⚠ Data received but couldn't parse: {e}")
        
        @room.on("transcription_received")
        def on_transcription(segments, participant, publication):
            for seg in segments:
                text = seg.text if hasattr(seg, 'text') else str(seg)
                speaker = participant.identity if participant else "unknown"
                received_transcripts.append(f"[{speaker}] {text}")
                print(f"   ✓ [{speaker}] Transcript: '{text}'")
        
        await room.connect(LIVEKIT_URL, token)
        print(f"   ✓ Connected to room: {TEST_ROOM_NAME}")
        print()
        
        # Wait for agent to be dispatched and join
        print("   Waiting for agent to join room (dispatch)...")
        await asyncio.sleep(8)  # Agent dispatch can take a few seconds
        
        # Debug: show participants in room
        print(f"   Participants in room: {[p.identity for p in room.remote_participants.values()]}")
        
        # ========== STEP 3: Publish Test Audio ==========
        print("🎤 Step 3: Publishing test audio...")
        
        # Load test audio
        with wave.open(TEST_AUDIO_PATH, 'rb') as w:
            sample_rate = w.getframerate()
            num_channels = w.getnchannels()
            audio_data = w.readframes(w.getnframes())
        
        print(f"   Audio: {len(audio_data)} bytes, {sample_rate}Hz")
        
        # Create audio source and track
        source = rtc.AudioSource(sample_rate, num_channels)
        track = rtc.LocalAudioTrack.create_audio_track("test-audio", source)
        
        # Publish track
        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_MICROPHONE
        publication = await room.local_participant.publish_track(track, options)
        print(f"   ✓ Published audio track")
        
        # Send audio in chunks (simulating real-time)
        chunk_samples = sample_rate // 20  # 50ms chunks
        chunk_bytes = chunk_samples * num_channels * 2  # 16-bit
        
        print(f"   [e2e-test-client] Sending audio: '{TEST_AUDIO_PATH}'...")
        for i in range(0, len(audio_data), chunk_bytes):
            chunk = audio_data[i:i + chunk_bytes]
            if len(chunk) < chunk_bytes:
                # Pad last chunk
                chunk = chunk + bytes(chunk_bytes - len(chunk))
            
            frame = rtc.AudioFrame(
                data=chunk,
                sample_rate=sample_rate,
                num_channels=num_channels,
                samples_per_channel=chunk_samples,
            )
            await source.capture_frame(frame)
            await asyncio.sleep(0.05)  # 50ms between chunks
        
        print("   ✓ Audio sent")
        print()
        
        # ========== STEP 4: Wait for Response ==========
        print("⏳ Step 4: Waiting for agent response...")
        
        # Wait up to 30 seconds for response (STT + TTS pipeline takes time)
        for i in range(60):
            await asyncio.sleep(0.5)
            if len(received_audio) > 10:  # Got significant audio back
                print(f"   ✓ Received {len(received_audio)} audio frames from agent")
                break
        else:
            print(f"   ⚠ Only received {len(received_audio)} audio frames")
        
        print()
        
        # ========== STEP 5: Verify Results ==========
        print("📊 Step 5: Verifying results...")
        
        # Check transcripts (if available via data channel)
        stt_verified = False
        if received_transcripts:
            print("   Transcripts received:")
            for t in received_transcripts[-5:]:  # Show last 5 transcripts
                print(f"      {t}")
            full_transcript = " ".join(received_transcripts).lower()
            
            if "hello this is a test" in full_transcript.replace(".", " ").replace(",", " ").replace("  ", " "):
                print("   ✓ STT verified: contains 'hello this is a test'")
                stt_verified = True
        else:
            print("   ⚠ No transcripts received via data channel")
            print("   (STT might still be working - agent processes internally)")
        
        # Check TTS (received audio)
        tts_verified = len(received_audio) > 10
        if tts_verified:
            total_bytes = sum(len(f.data) if hasattr(f, 'data') else 0 for f in received_audio)
            print(f"   ✓ TTS verified: received {len(received_audio)} frames ({total_bytes} bytes)")
        else:
            print(f"   ❌ TTS not verified: only {len(received_audio)} frames received")
        
        print()
        
        # ========== RESULT ==========
        await room.disconnect()
        
        if tts_verified and stt_verified:
            print("=" * 70)
            print("✅ E2E TEST PASSED!")
            print("   - Agent connected and processed audio")
            print(f"   - Received {len(received_audio)} audio frames from agent")
            print("   - STT transcript verified")
            print("=" * 70)
            return True
        elif tts_verified and not stt_verified:
            print("=" * 70)
            print("❌ E2E TEST FAILED!")
            print("   - TTS works: received audio frames")
            print("   - STT failed: transcript was empty or didn't match expected text")
            print("=" * 70)
            return False
        else:
            print("=" * 70)
            print("❌ E2E TEST FAILED!")
            print("   - Did not receive sufficient audio response from agent")
            print("=" * 70)
            return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup: stop agent
        if agent_process:
            print("\n🧹 Cleaning up...")
            agent_process.send_signal(signal.SIGTERM)
            try:
                agent_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                agent_process.kill()
            print("   ✓ Agent stopped")
            
            # Show final agent logs
            try:
                with open('/tmp/agent_e2e.log', 'r') as f:
                    log_content = f.read()
                    print(f"\n📋 Agent logs:\n{'='*40}")
                    print(log_content[-2000:] if len(log_content) > 2000 else log_content)
            except:
                pass


if __name__ == "__main__":
    success = asyncio.run(run_e2e_test())
    sys.exit(0 if success else 1)
