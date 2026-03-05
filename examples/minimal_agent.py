#!/usr/bin/env python3
"""
Minimal LiveKit Agent using Telnyx STT + TTS.

This example uses:
- Telnyx for Speech-to-Text
- Telnyx for Text-to-Speech
- OpenAI for LLM (you can use any LLM provider)

Usage:
    uv run python examples/minimal_agent.py
"""

from dotenv import load_dotenv
from livekit.agents import AgentSession, AutoSubscribe, WorkerOptions, cli, llm
from livekit.plugins import openai, telnyx

load_dotenv()


async def entrypoint(ctx):
    """Agent entrypoint."""
    
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    session = AgentSession(
        # Speech-to-Text: Telnyx
        stt=telnyx.STT(
            transcription_engine="deepgram",
            language="en",
            interim_results=True
        ),
        
        # Text-to-Speech: Telnyx
        tts=telnyx.TTS(
            voice="Telnyx.NaturalHD.astra"
        ),
        
        # LLM: OpenAI (or use any other provider)
        llm=openai.LLM(
            model="gpt-4o-mini"
        ),
    )
    
    await session.start(
        room=ctx.room,
        agent=llm.ChatContext().append(
            role="system",
            text="You are a helpful voice assistant. Keep your responses concise and conversational.",
        ),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
