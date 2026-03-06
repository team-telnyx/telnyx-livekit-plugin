# Telnyx plugin for LiveKit Agents

Support for [Telnyx](https://telnyx.com/)'s voice AI services in LiveKit Agents, including STT (powered by Deepgram), TTS, and LLM inference.

## Installation

```bash
pip install telnyx-livekit-plugin
```

## Pre-requisites

You'll need an API key from Telnyx. It can be set as an environment variable: `TELNYX_API_KEY`

Get your API key from the [Telnyx Portal](https://portal.telnyx.com/#/app/api-keys).

## Usage

### Speech-to-Text (STT)

```python
from livekit.plugins import telnyx

stt = telnyx.STT(
    transcription_engine="deepgram",
    model="nova-3",
)
```

Deepgram parameters can be passed directly:

```python
stt = telnyx.STT(
    transcription_engine="deepgram",
    model="nova-3",
    smart_format=True,
    keyterm=["YourBrand", "custom-term"],
    endpointing=300,
)
```

### Text-to-Speech (TTS)

```python
from livekit.plugins import telnyx

tts = telnyx.TTS(voice="Telnyx.NaturalHD.astra")
```

### LLM

Telnyx LLM inference is available via the OpenAI plugin:

```python
from livekit.plugins import openai

llm = openai.LLM.with_telnyx(model="Qwen/Qwen3-235B-A22B")
```

## Resources

- [Telnyx STT docs](https://developers.telnyx.com/docs/voice/programmable-voice/stt-standalone)
- [Telnyx TTS docs](https://developers.telnyx.com/docs/voice/programmable-voice/tts-standalone)
- [Telnyx TTS voices](https://developers.telnyx.com/api/call-control/list-text-to-speech-voices)
- [LiveKit Agents docs](https://docs.livekit.io/agents/)
