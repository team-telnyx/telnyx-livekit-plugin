<h1><a href="https://portal.telnyx.com"><img src="assets/telnyx-mark.svg" width="40" height="42" alt="Telnyx" align="top"></a> Telnyx plugin for LiveKit</h1>

Get ultra-low latency STT and TTS for your LiveKit Cloud agents. This plugin connects Telnyx's voice AI directly to your LiveKit agents with the fastest response times.

## Installation

```bash
pip install "telnyx-livekit-plugin @ git+https://github.com/team-telnyx/telnyx-livekit-plugin.git#subdirectory=telnyx-livekit-plugin"
```

## Pre-requisites

You'll need a Telnyx API key. Set it as an environment variable: `TELNYX_API_KEY`

In the [Telnyx Portal](https://portal.telnyx.com), search for "API keys" once logged in and create one.

## Usage

### Speech-to-Text (STT)

```python
from livekit.plugins import telnyx

stt = telnyx.deepgram.STT(
    model="nova-2",
    language="en",
    interim_results=True,
    # Behavior
    no_delay=True,
    filler_words=False,
    profanity_filter=False,
    endpointing=300,
    diarize=False,
    vad_events=False,
)
```

### Text-to-Speech (TTS)

```python
from livekit.plugins import telnyx

tts = telnyx.TTS(
    voice="Rime.ArcanaV3.astra",
    api_key=None,              # optional - defaults to TELNYX_API_KEY env var
    base_url="wss://api.telnyx.com/v2/text-to-speech",  # optional
    sample_rate=24000,         # optional
)
```

### LLM

```python
from livekit.plugins import openai

llm = openai.LLM.with_telnyx(model="kimi")
```

## Full Example

```python
from dotenv import load_dotenv
from livekit.agents import AgentSession, AutoSubscribe, WorkerOptions, cli, llm
from livekit.plugins import openai, telnyx

load_dotenv()


async def entrypoint(ctx):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    session = AgentSession(
        stt=telnyx.deepgram.STT(model="nova-2", language="en"),
        tts=telnyx.TTS(voice="Rime.ArcanaV3.astra"),
        llm=openai.LLM.with_telnyx(model="meta-llama/Meta-Llama-3.1-70B-Instruct"),
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
```

## Resources

- [Telnyx STT docs](https://developers.telnyx.com/docs/voice/programmable-voice/stt-standalone)
- [Telnyx TTS docs](https://developers.telnyx.com/docs/voice/programmable-voice/tts-standalone)
- [Telnyx TTS voices](https://developers.telnyx.com/docs/tts-stt/tts-available-voices/index)
