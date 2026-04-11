"""
*   Telnyx TTS API documentation:
    <https://developers.telnyx.com/docs/voice/programmable-voice/tts-standalone>.
"""

from __future__ import annotations

import asyncio
import base64
import json
import weakref
from dataclasses import dataclass

import aiohttp

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from .common import NUM_CHANNELS, SAMPLE_RATE, TTS_ENDPOINT, SessionManager, get_api_key
from .log import logger

# LiveKit voice pipeline default output sample rate.
_PIPELINE_SAMPLE_RATE = 24000

# MiniMax supported PCM sample rates. 24kHz is NOT supported natively —
# MiniMax maps 24000 → 22050.  Requesting 32kHz (the PCM default) and
# downsampling to 24kHz gives better quality than upsampling from 16kHz.
_MINIMAX_PCM_SAMPLE_RATE = 32000


@dataclass
class _TTSOptions:
    api_key: str
    voice: str
    base_url: str
    sample_rate: int


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: str = "Telnyx.NaturalHD.astra",
        api_key: str | None = None,
        base_url: str = TTS_ENDPOINT,
        sample_rate: int = SAMPLE_RATE,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        self._opts = _TTSOptions(
            voice=voice,
            api_key=get_api_key(api_key),
            base_url=base_url,
            sample_rate=sample_rate,
        )
        self._session_manager = SessionManager(http_session)
        self._streams = weakref.WeakSet[SynthesizeStream]()

    @property
    def model(self) -> str:
        return self._opts.voice

    @property
    def provider(self) -> str:
        return "telnyx"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return self._synthesize_with_stream(text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await self._session_manager.close()


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self, *, tts: TTS, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        self._segments_ch = utils.aio.Chan[str]()
        request_id = utils.shortuuid()
        # When using PCM with resampling, tell the emitter the output
        # rate so frame metadata matches what we actually push.
        emitter_rate = (
            _PIPELINE_SAMPLE_RATE
            if self._is_pcm_provider()
            else self._tts._opts.sample_rate
        )
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=emitter_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )

        async def _collect_segments() -> None:
            segment_text = ""
            async for input_data in self._input_ch:
                if isinstance(input_data, str):
                    segment_text += input_data
                elif isinstance(input_data, self._FlushSentinel):
                    if segment_text:
                        self._segments_ch.send_nowait(segment_text)
                        segment_text = ""
            # Flush any remaining text that wasn't followed by a FlushSentinel.
            # This happens when end_input() is called after the last text push
            # without an intervening flush — e.g. the final sentence from an LLM stream.
            if segment_text:
                self._segments_ch.send_nowait(segment_text)
            self._segments_ch.close()

        async def _run_segments() -> None:
            async for text in self._segments_ch:
                await self._run_ws(text, output_emitter)

        tasks = [
            asyncio.create_task(_collect_segments()),
            asyncio.create_task(_run_segments()),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=request_id, body=None
            ) from None
        except APIConnectionError:
            raise
        except APIStatusError:
            raise
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)

    def _is_pcm_provider(self) -> bool:
        """Check if the voice uses a provider that should request PCM format.

        MiniMax returns MP3 chunks whose frame boundaries don't align with
        WebSocket message boundaries, causing PyAV decode errors and audio
        truncation.  Requesting raw PCM avoids the MP3 decoder entirely.
        """
        voice = self._tts._opts.voice.lower()
        return voice.startswith("minimax.")

    def _build_ws_url(self) -> str:
        """Build the WebSocket URL with appropriate query params.

        For PCM providers (MiniMax), include audio_format and sample_rate
        so the gateway can forward them to the upstream TTS service.
        Non-PCM providers (Telnyx voices returning MP3) don't need these.

        MiniMax does not natively support 24kHz PCM (maps to 22050 instead).
        Requesting 32kHz and downsampling gives higher quality than
        upsampling from 16kHz.
        """
        base = f"{self._tts._opts.base_url}?voice={self._tts._opts.voice}"

        if self._is_pcm_provider():
            return f"{base}&audio_format=linear16&sample_rate={_MINIMAX_PCM_SAMPLE_RATE}"

        return base

    async def _run_ws(self, text: str, output_emitter: tts.AudioEmitter) -> None:
        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)

        use_pcm = self._is_pcm_provider()
        url = self._build_ws_url()
        headers = {"Authorization": f"Bearer {self._tts._opts.api_key}"}

        decoder = None
        if not use_pcm:
            decoder = utils.codecs.AudioStreamDecoder(
                sample_rate=self._tts._opts.sample_rate,
                num_channels=NUM_CHANNELS,
                format="audio/mp3",
            )

        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            handshake: dict = {"text": " "}
            if use_pcm:
                handshake["voice_settings"] = {"response_format": "pcm"}
            await ws.send_str(json.dumps(handshake))
            self._mark_started()
            await ws.send_str(json.dumps({"text": text}))
            await ws.send_str(json.dumps({"text": ""}))

        # When using raw PCM from MiniMax, resample from the provider's
        # native rate (32 kHz) down to the LiveKit pipeline rate (24 kHz).
        # MiniMax doesn't support 24kHz natively (maps to 22050), so we
        # request 32kHz and downsample for better quality than upsampling
        # from 16kHz.
        resampler: rtc.AudioResampler | None = None
        pcm_byte_stream: utils.audio.AudioByteStream | None = None
        if use_pcm:
            resampler = rtc.AudioResampler(
                input_rate=_MINIMAX_PCM_SAMPLE_RATE,
                output_rate=_PIPELINE_SAMPLE_RATE,
            )
            pcm_byte_stream = utils.audio.AudioByteStream(
                sample_rate=_MINIMAX_PCM_SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
            )

        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        audio_data = data.get("audio")
                        if audio_data:
                            audio_bytes = base64.b64decode(audio_data)
                            if audio_bytes:
                                if use_pcm:
                                    if resampler and pcm_byte_stream:
                                        for frame in pcm_byte_stream.push(audio_bytes):
                                            for resampled in resampler.push(frame):
                                                output_emitter.push(resampled.data.tobytes())
                                    else:
                                        output_emitter.push(audio_bytes)
                                else:
                                    decoder.push(audio_bytes)
                    except json.JSONDecodeError:
                        logger.warning("Telnyx TTS: Received invalid JSON")

                elif msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error("Telnyx TTS WebSocket error: %s", ws.exception())
                    break

            # Flush any remaining PCM through the resampler
            if resampler and pcm_byte_stream:
                for frame in pcm_byte_stream.flush():
                    for resampled in resampler.push(frame):
                        output_emitter.push(resampled.data.tobytes())
                for resampled in resampler.flush():
                    output_emitter.push(resampled.data.tobytes())

            if decoder:
                decoder.end_input()

        async def decode_task() -> None:
            if not decoder:
                return
            async for frame in decoder:
                output_emitter.push(frame.data.tobytes())

        try:
            ws = await asyncio.wait_for(
                self._tts._session_manager.ensure_session().ws_connect(url, headers=headers),
                self._conn_options.timeout,
            )
            async with ws:
                tasks = [
                    asyncio.create_task(send_task(ws)),
                    asyncio.create_task(recv_task(ws)),
                    asyncio.create_task(decode_task()),
                ]
                try:
                    await asyncio.gather(*tasks)
                finally:
                    await utils.aio.gracefully_cancel(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except (APIConnectionError, APIStatusError, APITimeoutError):
            raise
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            if decoder:
                await decoder.aclose()
            output_emitter.end_segment()
