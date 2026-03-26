"""Deepgram STT via Telnyx.

Provides a Deepgram-native constructor that handles all the translation
to Telnyx's WebSocket API. Developers configure Deepgram params directly;
the plugin maps them to what the backend expects.

Usage:
    from livekit.plugins import telnyx

    stt = telnyx.deepgram.STT(
        model="nova-3",
        keyterm=["YourBrand", "custom-term"],
        endpointing=300,
    )
"""

from __future__ import annotations

from typing import Any, Literal

import aiohttp

from ..common import SAMPLE_RATE, STT_ENDPOINT
from ..stt import STT as BaseTelnyxSTT


class STT(BaseTelnyxSTT):
    """Deepgram STT through Telnyx.

    Configure Deepgram the way Deepgram's docs describe it. The plugin
    handles connecting to Telnyx, setting the right engine, and mapping
    parameters to what the backend expects.

    Warnings are emitted for parameters that are hardcoded or not yet
    supported on the Telnyx backend, so developers know exactly what
    works without guessing.
    """

    def __init__(
        self,
        *,
        model: Literal["nova-3", "nova-2", "flux"] = "nova-3",
        language: str = "en",
        interim_results: bool = True,
        api_key: str | None = None,
        base_url: str = STT_ENDPOINT,
        sample_rate: int = SAMPLE_RATE,
        http_session: aiohttp.ClientSession | None = None,
        # Formatting
        smart_format: bool | None = None,
        numerals: bool | None = None,
        punctuate: bool | None = None,
        # Recognition boosting
        keyterm: str | list[str] | None = None,
        keywords: str | list[str] | None = None,
        # Behavior
        no_delay: bool | None = None,
        filler_words: bool | None = None,
        profanity_filter: bool | None = None,
        endpointing: int | bool | None = None,
        diarize: bool | None = None,
        vad_events: bool | None = None,
        # Catch-all for any Deepgram param
        **extra_deepgram_params: Any,
    ) -> None:
        super().__init__(
            language=language,
            transcription_engine="Deepgram",
            interim_results=interim_results,
            api_key=api_key,
            base_url=base_url,
            sample_rate=sample_rate,
            http_session=http_session,
            model=model,
            smart_format=smart_format,
            numerals=numerals,
            punctuate=punctuate,
            keyterm=keyterm,
            keywords=keywords,
            no_delay=no_delay,
            filler_words=filler_words,
            profanity_filter=profanity_filter,
            endpointing=endpointing,
            diarize=diarize,
            vad_events=vad_events,
            **extra_deepgram_params,
        )
