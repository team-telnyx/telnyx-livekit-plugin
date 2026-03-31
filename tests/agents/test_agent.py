"""
Simple test agent that joins a room and processes audio.
Used for E2E testing without relying on agent dispatch.

Flow:
1. Connects to room as "test-agent"
2. Waits for audio from other participants
3. Runs STT on received audio
4. Sends transcript to data channel
5. Generates TTS response ("I heard you say: ...")
6. Publishes TTS audio back to room
"""

import asyncio
import os
import sys

from livekit import api, rtc
from livekit.agents import stt as stt_module

# Add plugin to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'telnyx-livekit-plugin'))

LIVEKIT_URL = os.getenv('LIVEKIT_URL')
LIVEKIT_API_KEY = os.getenv('LIVEKIT_API_KEY')
LIVEKIT_API_SECRET = os.getenv('LIVEKIT_API_SECRET')
TELNYX_API_KEY = os.getenv('TELNYX_API_KEY')
ROOM_NAME = os.getenv('TEST_ROOM_NAME', 'e2e-test-room')


def create_token(room_name: str, identity: str) -> str:
    token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    token.with_identity(identity)
    token.with_grants(api.VideoGrants(
        room_join=True,
        room=room_name,
        can_publish=True,
        can_publish_data=True,
        can_subscribe=True,
    ))
    return token.to_jwt()


async def run_agent():
    from livekit.plugins.telnyx import STT, TTS
    
    print(f"[test-agent] Starting for room: {ROOM_NAME}")
    
    # Create STT and TTS
    stt = STT(api_key=TELNYX_API_KEY)
    tts = TTS(api_key=TELNYX_API_KEY)
    
    # Connect to room
    token = create_token(ROOM_NAME, "test-agent")
    room = rtc.Room()
    
    # Track for publishing audio response
    audio_source = rtc.AudioSource(16000, 1)
    audio_track = rtc.LocalAudioTrack.create_audio_track("agent-audio", audio_source)
    
    # State
    incoming_audio = []
    processing = False
    audio_published = False
    
    # Resampler: 48kHz -> 16kHz (Telnyx STT expects 16kHz)
    resampler = rtc.AudioResampler(input_rate=48000, output_rate=16000, num_channels=1)
    
    async def publish_audio_track():
        """Publish audio track when we have something to say"""
        nonlocal audio_published
        if not audio_published:
            options = rtc.TrackPublishOptions()
            options.source = rtc.TrackSource.SOURCE_MICROPHONE
            await room.local_participant.publish_track(audio_track, options)
            audio_published = True
            print("[test-agent] Published audio track")
    
    async def send_transcript(text: str):
        """Send transcript to data channel so test client can see it"""
        try:
            await room.local_participant.publish_data(
                f"[test-agent] Heard: {text}".encode('utf-8'),
                reliable=True
            )
            print(f"[test-agent] Sent transcript to data channel")
        except Exception as e:
            print(f"[test-agent] Failed to send transcript: {e}")
    
    @room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication, participant):
        # Only listen to e2e-test-client, ignore other agents (like the restaurant agent)
        if track.kind == rtc.TrackKind.KIND_AUDIO and participant.identity == "e2e-test-client":
            print(f"[test-agent] Subscribed to audio from [{participant.identity}]")
            
            async def receive_and_process():
                nonlocal processing
                audio_stream = rtc.AudioStream(track)
                
                async for event in audio_stream:
                    if hasattr(event, 'frame'):
                        incoming_audio.append(event.frame)
                        
                        # Process after receiving enough audio (~4 seconds at 400 frames = 400*10ms)
                        # 48kHz with 480 samples per frame = 10ms per frame
                        if len(incoming_audio) >= 400 and not processing:
                            processing = True
                            print(f"[test-agent] Received {len(incoming_audio)} audio frames, processing...")
                            await process_audio()
            
            asyncio.create_task(receive_and_process())
    
    async def process_audio():
        """Process received audio: STT -> publish transcript -> TTS -> publish audio"""
        try:
            # Debug: show frame info
            if incoming_audio:
                f = incoming_audio[0]
                print(f"[test-agent] Frame info: sample_rate={f.sample_rate}, channels={f.num_channels}, samples={f.samples_per_channel}, data_len={len(f.data)}")
            
            # Resample audio from 48kHz to 16kHz for Telnyx STT
            print("[test-agent] Resampling audio 48kHz -> 16kHz...")
            resampled_frames = []
            for frame in incoming_audio:
                for resampled in resampler.push(frame):
                    resampled_frames.append(resampled)
            
            # Flush remaining samples
            for resampled in resampler.flush():
                resampled_frames.append(resampled)
            
            print(f"[test-agent] Resampled to {len(resampled_frames)} frames")
            if resampled_frames:
                rf = resampled_frames[0]
                print(f"[test-agent] Resampled frame info: sample_rate={rf.sample_rate}, channels={rf.num_channels}, samples={rf.samples_per_channel}, data_len={len(rf.data)}")
            
            # Run STT on resampled audio
            print("[test-agent] Running STT...")
            stt_stream = stt.stream()
            for frame in resampled_frames:
                stt_stream.push_frame(frame)
            
            # Signal end of input immediately after pushing all frames
            stt_stream.end_input()
            
            transcript = ""
            async for event in stt_stream:
                if event.type == stt_module.SpeechEventType.FINAL_TRANSCRIPT:
                    if event.alternatives:
                        text = event.alternatives[0].text
                        transcript += text
                        print(f"[test-agent] STT: '{text}'")
            
            await stt_stream.aclose()
            
            if not transcript:
                print("[test-agent] No transcript from STT")
                transcript = "(empty)"
            
            print(f"[test-agent] Full transcript: '{transcript}'")
            
            # Send transcript to data channel
            await send_transcript(transcript)
            
            # Publish audio track now that we have something to say
            await publish_audio_track()
            
            # Generate response with TTS
            response_text = f"I heard you say: {transcript}"
            print(f"[test-agent] Generating TTS: '{response_text}'")
            
            tts_stream = tts.stream()
            tts_stream.push_text(response_text)
            tts_stream.end_input()
            
            frame_count = 0
            async for event in tts_stream:
                if hasattr(event, 'frame') and event.frame:
                    await audio_source.capture_frame(event.frame)
                    frame_count += 1
            
            await tts_stream.aclose()
            print(f"[test-agent] TTS complete: sent {frame_count} audio frames")
            
        except Exception as e:
            print(f"[test-agent] Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Connect to room
    await room.connect(LIVEKIT_URL, token)
    print(f"[test-agent] Connected to room: {ROOM_NAME}")
    
    # Keep running
    print("[test-agent] Waiting for audio...")
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await room.disconnect()
        await stt.aclose()
        await tts.aclose()
        print("[test-agent] Disconnected")


if __name__ == "__main__":
    asyncio.run(run_agent())
