"""Deepgram services via Telnyx.

Usage:
    from livekit.plugins import telnyx

    stt = telnyx.deepgram.STT(model="nova-3", keyterm=["YourBrand"])
"""

from .stt import STT

__all__ = ["STT"]
