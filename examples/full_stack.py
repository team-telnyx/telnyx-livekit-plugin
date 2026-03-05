#!/usr/bin/env python3
"""
Full Stack LiveKit Agent using Telnyx for everything.

This example uses:
- Telnyx for Speech-to-Text
- Telnyx for Text-to-Speech
- Telnyx for LLM Inference

Usage:
    uv run python examples/full_stack.py
"""

from dotenv import load_dotenv
from livekit.agents import AgentSession, AutoSubscribe, WorkerOptions, cli, llm
from livekit.plugins import openai, telnyx

load_dotenv()


async def entrypoint(ctx):
    """Agent entrypoint with full Telnyx stack."""
    
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    session = AgentSession(
        # Speech-to-Text: Telnyx with Deepgram engine
        stt=telnyx.STT(
            transcription_engine="deepgram",
            base_url="wss://api.telnyx.com/v2/speech-to-text/transcription?transcription_model=deepgram/nova-3",
            language="en",
            interim_results=True
        ),
        
        # Text-to-Speech: Telnyx MiniMax voice
        tts=telnyx.TTS(
            voice="Minimax.speech-02-hd.Wise_Woman"
        ),
        
        # LLM: Telnyx inference via OpenAI plugin
        llm=openai.LLM.with_telnyx(
            model="Qwen/Qwen3-235B-A22B",
            temperature=0.7
        ),
    )
    
    await session.start(
        room=ctx.room,
        agent=llm.ChatContext().append(
            role="system",
            text="You are a helpful voice assistant powered by Telnyx. Keep your responses concise and conversational.",
        ),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
