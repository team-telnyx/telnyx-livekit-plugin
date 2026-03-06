"""
Full Stack Telnyx Agent — STT + TTS + LLM all on Telnyx.

Usage:
    uv run python examples/full_stack.py
"""

from livekit.agents import AgentSession, AutoSubscribe, WorkerOptions, cli, llm
from livekit.plugins import openai, telnyx


async def entrypoint(ctx):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    session = AgentSession(
        stt=telnyx.STT(
            transcription_engine="deepgram",
            model="nova-3",
            smart_format=True,
            keyterm=["Telnyx", "LiveKit"],
        ),
        tts=telnyx.TTS(voice="Minimax.speech-02-hd.Wise_Woman"),
        llm=openai.LLM.with_telnyx(model="Qwen/Qwen3-235B-A22B"),
    )

    await session.start(
        room=ctx.room,
        agent=llm.ChatContext().append(
            role="system",
            text="You are a helpful voice assistant powered by Telnyx.",
        ),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
