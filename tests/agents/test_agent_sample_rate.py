"""
Test agent for sample_rate E2E test.

Joins a LiveKit room, listens for audio, runs STT, responds with TTS
at the configured sample rate, and publishes audio back.

Environment:
    LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, TELNYX_API_KEY,
    TEST_ROOM_NAME, TTS_SAMPLE_RATE (default 24000)
"""

import asyncio
import os
import sys

from livekit import api, rtc
from livekit.agents import stt as stt_module

# Ensure local plugin is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'telnyx-livekit-plugin'))

LIVEKIT_URL = os.environ['LIVEKIT_URL']
LIVEKIT_API_KEY = os.environ['LIVEKIT_API_KEY']
LIVEKIT_API_SECRET = os.environ['LIVEKIT_API_SECRET']
TELNYX_API_KEY = os.environ['TELNYX_API_KEY']
ROOM_NAME = os.environ.get('TEST_ROOM_NAME', 'e2e-sample-rate-test')
TTS_SAMPLE_RATE = int(os.environ.get('TTS_SAMPLE_RATE', '24000'))


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

    print(f"[test-agent] Room: {ROOM_NAME}, TTS sample_rate: {TTS_SAMPLE_RATE}")

    stt = STT(api_key=TELNYX_API_KEY)
    tts = TTS(api_key=TELNYX_API_KEY, sample_rate=TTS_SAMPLE_RATE)

    print(f"[test-agent] TTS.sample_rate = {tts.sample_rate}")

    room = rtc.Room()

    audio_source = rtc.AudioSource(TTS_SAMPLE_RATE, 1)
    audio_track = rtc.LocalAudioTrack.create_audio_track("agent-audio", audio_source)

    incoming_audio = []
    processing = False
    audio_published = False

    resampler = rtc.AudioResampler(input_rate=48000, output_rate=16000, num_channels=1)

    async def publish_audio_track():
        nonlocal audio_published
        if not audio_published:
            options = rtc.TrackPublishOptions()
            options.source = rtc.TrackSource.SOURCE_MICROPHONE
            await room.local_participant.publish_track(audio_track, options)
            audio_published = True
            print("[test-agent] Published audio track")

    async def send_transcript(text: str):
        try:
            await room.local_participant.publish_data(
                f"[test-agent] Heard: {text}".encode('utf-8'),
                reliable=True
            )
        except Exception as e:
            print(f"[test-agent] Failed to send transcript: {e}")

    @room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO and participant.identity == "e2e-test-client":
            print(f"[test-agent] Subscribed to audio from [{participant.identity}]")

            async def receive_and_process():
                nonlocal processing
                audio_stream = rtc.AudioStream(track)

                async for event in audio_stream:
                    if hasattr(event, 'frame'):
                        incoming_audio.append(event.frame)
                        if len(incoming_audio) >= 400 and not processing:
                            processing = True
                            print(f"[test-agent] Got {len(incoming_audio)} frames, processing...")
                            await process_audio()

            asyncio.create_task(receive_and_process())

    async def process_audio():
        try:
            # Resample 48kHz -> 16kHz for STT
            resampled_frames = []
            for frame in incoming_audio:
                for resampled in resampler.push(frame):
                    resampled_frames.append(resampled)
            for resampled in resampler.flush():
                resampled_frames.append(resampled)

            print(f"[test-agent] Resampled to {len(resampled_frames)} frames for STT")

            # STT
            stt_stream = stt.stream()
            for frame in resampled_frames:
                stt_stream.push_frame(frame)
            stt_stream.end_input()

            transcript = ""
            async for event in stt_stream:
                if event.type == stt_module.SpeechEventType.FINAL_TRANSCRIPT:
                    if event.alternatives:
                        transcript += event.alternatives[0].text

            await stt_stream.aclose()
            print(f"[test-agent] STT transcript: '{transcript}'")

            await send_transcript(transcript)
            await publish_audio_track()

            # TTS at configured sample rate
            response_text = f"I heard you say: {transcript}" if transcript else "I did not hear anything."
            print(f"[test-agent] TTS at {TTS_SAMPLE_RATE}Hz: '{response_text}'")

            tts_stream = tts.stream()
            tts_stream.push_text(response_text)
            tts_stream.flush()
            tts_stream.end_input()

            frame_count = 0
            async for event in tts_stream:
                if hasattr(event, 'frame') and event.frame:
                    await audio_source.capture_frame(event.frame)
                    frame_count += 1

            await tts_stream.aclose()
            print(f"[test-agent] TTS done: {frame_count} frames at {TTS_SAMPLE_RATE}Hz")

            # Signal completion via data channel
            await room.local_participant.publish_data(
                f"TTS_COMPLETE:sample_rate={TTS_SAMPLE_RATE},frames={frame_count}".encode('utf-8'),
                reliable=True
            )

        except Exception as e:
            print(f"[test-agent] Error: {e}")
            import traceback
            traceback.print_exc()

    await room.connect(LIVEKIT_URL, create_token(ROOM_NAME, "test-agent"))
    print(f"[test-agent] Connected to room")

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await room.disconnect()
        await stt.aclose()
        await tts.aclose()


if __name__ == "__main__":
    asyncio.run(run_agent())
