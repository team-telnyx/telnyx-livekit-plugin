"""
Telnyx STT plugin for LiveKit Agents — with full Deepgram parameter support.

Fixes the URL construction bug in the upstream plugin and exposes all
Deepgram STT parameters as first-class constructor arguments.

Usage:
    from livekit.plugins.telnyx import STT

    stt = STT(
        transcription_engine="deepgram",
        model="nova-3",
        smart_format=True,
        keyterm=["Telnyx", "LiveKit"],
    )
"""

from __future__ import annotations

import asyncio
import json
import struct
import weakref
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import aiohttp

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    stt,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from .common import NUM_CHANNELS, SAMPLE_RATE, STT_ENDPOINT, SessionManager, get_api_key
from .log import logger

TranscriptionEngine = Literal["telnyx", "google", "deepgram", "azure"]


@dataclass
class _STTOptions:
    api_key: str
    language: str
    transcription_engine: TranscriptionEngine
    interim_results: bool
    base_url: str
    sample_rate: int
    # Deepgram parameters (only non-None values are sent as query params)
    deepgram_params: dict[str, Any] = field(default_factory=dict)


class STT(stt.STT):
    """Telnyx STT with full Deepgram parameter support.

    All standard Telnyx STT parameters are supported, plus any Deepgram
    parameter can be passed directly in the constructor:

        stt = STT(
            transcription_engine="deepgram",
            # Deepgram model
            model="nova-3",
            # Formatting
            smart_format=True,
            numerals=True,
            punctuate=True,
            # Recognition boosting
            keyterm=["Telnyx", "LiveKit", "custom-brand"],
            keywords=["Telnyx:2.0"],  # Nova-2 only
            # Latency / behavior
            no_delay=True,
            endpointing=300,
            filler_words=False,
            profanity_filter=False,
            # Speaker
            diarize=False,
            # Any other Deepgram param not listed above
            tag="my-app",
        )

    Parameters are passed as query string params on the WebSocket URL.
    The Telnyx backend forwards them to the underlying Deepgram engine.

    Model compatibility:
        - smart_format, numerals: Nova only (not Flux streaming)
        - keyterm: Nova-3 and Flux
        - keywords: Nova-2 only (use keyterm for Nova-3/Flux)
    """

    def __init__(
        self,
        *,
        # Standard Telnyx STT params
        language: str = "en",
        transcription_engine: TranscriptionEngine = "telnyx",
        interim_results: bool = True,
        api_key: str | None = None,
        base_url: str = STT_ENDPOINT,
        sample_rate: int = SAMPLE_RATE,
        http_session: aiohttp.ClientSession | None = None,
        # --- Deepgram parameters (Phase 1) ---
        model: str | None = None,
        smart_format: bool | None = None,
        numerals: bool | None = None,
        punctuate: bool | None = None,
        no_delay: bool | None = None,
        filler_words: bool | None = None,
        profanity_filter: bool | None = None,
        endpointing: int | bool | None = None,
        diarize: bool | None = None,
        vad_events: bool | None = None,
        # Recognition boosting
        keyterm: str | list[str] | None = None,
        keywords: str | list[str] | None = None,
        # --- Catch-all for any Deepgram param not listed above ---
        **extra_deepgram_params: Any,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=interim_results,
            )
        )

        # Build deepgram params dict (only non-None scalar values)
        deepgram_params: dict[str, Any] = {}

        scalar_opts = {
            "model": model,
            "smart_format": smart_format,
            "numerals": numerals,
            "punctuate": punctuate,
            "no_delay": no_delay,
            "filler_words": filler_words,
            "profanity_filter": profanity_filter,
            "endpointing": endpointing,
            "diarize": diarize,
            "vad_events": vad_events,
        }

        for k, v in scalar_opts.items():
            if v is not None:
                deepgram_params[k] = str(v).lower() if isinstance(v, bool) else str(v)

        # List params (Deepgram expects repeated keys: keyterm=X&keyterm=Y)
        if keyterm is not None:
            deepgram_params["keyterm"] = [keyterm] if isinstance(keyterm, str) else keyterm
        if keywords is not None:
            deepgram_params["keywords"] = [keywords] if isinstance(keywords, str) else keywords

        # Extra params the user passes via **kwargs
        for k, v in extra_deepgram_params.items():
            if isinstance(v, bool):
                deepgram_params[k] = str(v).lower()
            elif isinstance(v, list):
                deepgram_params[k] = v
            else:
                deepgram_params[k] = str(v)

        self._opts = _STTOptions(
            api_key=get_api_key(api_key),
            language=language,
            transcription_engine=transcription_engine,
            interim_results=interim_results,
            base_url=base_url,
            sample_rate=sample_rate,
            deepgram_params=deepgram_params,
        )
        self._session_manager = SessionManager(http_session)
        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def model(self) -> str:
        return self._opts.deepgram_params.get("model", self._opts.transcription_engine)

    @property
    def provider(self) -> str:
        return "telnyx"

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        resolved_language = language if is_given(language) else self._opts.language

        stream = self.stream(language=language, conn_options=conn_options)
        try:
            frames = buffer if isinstance(buffer, list) else [buffer]
            for frame in frames:
                stream.push_frame(frame)
            stream.end_input()

            final_text = ""
            async for event in stream:
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    if event.alternatives:
                        final_text += event.alternatives[0].text

            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        language=resolved_language,
                        text=final_text,
                    )
                ],
            )
        finally:
            await stream.aclose()

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        resolved_language = language if is_given(language) else self._opts.language
        stream = SpeechStream(
            stt=self,
            conn_options=conn_options,
            language=resolved_language,
        )
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await self._session_manager.close()


def _create_streaming_wav_header(sample_rate: int, num_channels: int) -> bytes:
    """Create a WAV header for streaming with maximum possible size."""
    bytes_per_sample = 2
    byte_rate = sample_rate * num_channels * bytes_per_sample
    block_align = num_channels * bytes_per_sample
    data_size = 0x7FFFFFFF
    file_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        file_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        16,
        b"data",
        data_size,
    )
    return header


def _build_ws_url(base_url: str, params: dict[str, Any]) -> str:
    """Build WebSocket URL by properly merging base_url query params with additional params.

    Handles:
    - Base URLs with existing query params (no double '?' bug)
    - List params as repeated keys (keyterm=X&keyterm=Y)
    - Scalar params as simple key=value
    """
    parsed = urlparse(base_url)

    # Extract existing query params from base_url
    existing = {k: v[0] for k, v in parse_qs(parsed.query).items()}

    # Separate scalar and list params
    scalar_params = dict(existing)
    list_parts: list[str] = []

    for k, v in params.items():
        if isinstance(v, list):
            for item in v:
                list_parts.append(f"{k}={item}")
        else:
            scalar_params[k] = v

    query = urlencode(scalar_params)
    if list_parts:
        if query:
            query += "&"
        query += "&".join(list_parts)

    return urlunparse(parsed._replace(query=query))


class SpeechStream(stt.RecognizeStream):
    def __init__(
        self,
        *,
        stt: STT,
        conn_options: APIConnectOptions,
        language: str,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=stt._opts.sample_rate)
        self._stt: STT = stt
        self._language = language
        self._speaking = False

    async def _run(self) -> None:
        closing_ws = False

        @utils.log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws

            wav_header = _create_streaming_wav_header(self._stt._opts.sample_rate, NUM_CHANNELS)
            await ws.send_bytes(wav_header)

            samples_per_chunk = self._stt._opts.sample_rate // 20
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._stt._opts.sample_rate,
                num_channels=NUM_CHANNELS,
                samples_per_channel=samples_per_chunk,
            )

            async for data in self._input_ch:
                if isinstance(data, rtc.AudioFrame):
                    for frame in audio_bstream.write(data.data.tobytes()):
                        await ws.send_bytes(frame.data.tobytes())
                elif isinstance(data, self._FlushSentinel):
                    for frame in audio_bstream.flush():
                        await ws.send_bytes(frame.data.tobytes())

            for frame in audio_bstream.flush():
                await ws.send_bytes(frame.data.tobytes())

            closing_ws = True

        @utils.log_exceptions(logger=logger)
        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws:
                        return
                    raise APIStatusError(message="Telnyx STT WebSocket closed unexpectedly")

                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        logger.debug(
                            "Telnyx STT received: is_final=%s, has_transcript=%s",
                            data.get("is_final"),
                            bool(data.get("transcript")),
                        )
                        self._process_stream_event(data)
                    except Exception:
                        logger.exception("Failed to process Telnyx STT message")
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error("Telnyx STT WebSocket error: %s", ws.exception())

        ws: aiohttp.ClientWebSocketResponse | None = None
        try:
            ws = await self._connect_ws()
            tasks = [
                asyncio.create_task(send_task(ws)),
                asyncio.create_task(recv_task(ws)),
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                await utils.aio.gracefully_cancel(*tasks)
        finally:
            if ws is not None:
                await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        opts = self._stt._opts

        # Base params (always sent)
        params: dict[str, Any] = {
            "transcription_engine": opts.transcription_engine,
            "language": self._language,
            "input_format": "wav",
        }

        # Merge Deepgram params
        params.update(opts.deepgram_params)

        # Build URL with proper query string merging (no double '?' bug)
        url = _build_ws_url(opts.base_url, params)
        headers = {"Authorization": f"Bearer {opts.api_key}"}

        logger.debug("Connecting to Telnyx STT: %s", url.split("?")[0] + "?...")

        try:
            ws = await asyncio.wait_for(
                self._stt._session_manager.ensure_session().ws_connect(url, headers=headers),
                self._conn_options.timeout,
            )
            logger.debug("Established Telnyx STT WebSocket connection")
            return ws
        except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
            raise APIConnectionError("Failed to connect to Telnyx STT") from e

    def _process_stream_event(self, data: dict) -> None:
        transcript = data.get("transcript", "")
        is_final = data.get("is_final", False)

        if not transcript:
            return

        if not self._speaking:
            self._speaking = True
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))

        alternatives = [
            stt.SpeechData(
                language=self._language,
                text=transcript,
                confidence=data.get("confidence", 0.0),
            )
        ]

        if is_final:
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=alternatives,
                )
            )
            self._speaking = False
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))
        else:
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    alternatives=alternatives,
                )
            )
