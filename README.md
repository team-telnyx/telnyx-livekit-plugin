# Telnyx LiveKit Plugins

Documentation and examples for integrating Telnyx's LiveKit plugins: Speech-to-Text (STT), Text-to-Speech (TTS), and LLM inference.

## Installation

Install dependencies with `uv`:

```bash
uv sync
```

Or with pip:

```bash
pip install livekit-agents livekit-plugins-telnyx livekit-plugins-openai
```

## Setup

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Add your Telnyx API key:
   ```
   TELNYX_API_KEY=your_api_key_here
   ```

Get your API key from the [Telnyx Portal](https://portal.telnyx.com/#/app/api-keys).

---

## 1. STT Plugin (Speech-to-Text)

Telnyx provides streaming speech-to-text with support for multiple transcription engines.

### Basic Usage

```python
from livekit.plugins import telnyx

stt = telnyx.STT(
    language="en",
    transcription_engine="deepgram",  # Options: telnyx, google, deepgram, azure
    interim_results=True
)
```

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `language` | `str` | `"en"` | Language code (e.g., "en", "es", "fr") |
| `transcription_engine` | `str` | `"telnyx"` | Engine: `telnyx`, `google`, `deepgram`, `azure` |
| `interim_results` | `bool` | `True` | Return partial transcription results |
| `sample_rate` | `int` | `16000` | Audio sample rate in Hz |
| `base_url` | `str` | `wss://api.telnyx.com/v2/speech-to-text/transcription` | WebSocket endpoint |

### Test It

Run the standalone STT test:

```bash
uv run python tests/test_stt.py
```

This will transcribe `tests/assets/sample.wav` and print the result.

---

## 2. TTS Plugin (Text-to-Speech)

Telnyx provides high-quality streaming text-to-speech with NaturalHD voices and MiniMax voices.

### Basic Usage

```python
from livekit.plugins import telnyx

tts = telnyx.TTS(
    voice="Telnyx.NaturalHD.astra"
)
```

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `voice` | `str` | `"Telnyx.NaturalHD.astra"` | Voice identifier (see [Telnyx Voices API](https://developers.telnyx.com/api/call-control/list-text-to-speech-voices)) |
| `base_url` | `str` | `wss://api.telnyx.com/v2/text-to-speech/speech` | WebSocket endpoint |

### Popular Voices

- **NaturalHD**: `Telnyx.NaturalHD.astra`, `Telnyx.NaturalHD.orion`, `Telnyx.NaturalHD.nova`
- **MiniMax**: `Minimax.speech-02-hd.Wise_Woman`, `Minimax.speech-02-hd.Confident_Young_Man`

### Test It

Run the standalone TTS test:

```bash
uv run python tests/test_tts.py
```

This will generate audio from "Hello, this is a test" and save it to `tests/output.wav`.

---

## 3. LLM Integration

Telnyx LLM inference is available via the OpenAI plugin using the `with_telnyx()` helper.

### Basic Usage

```python
from livekit.plugins import openai

llm = openai.LLM.with_telnyx(
    model="Qwen/Qwen3-235B-A22B"
)
```

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | Required | Model identifier (see [Telnyx Models API](https://developers.telnyx.com/api-reference/chat/get-available-models)) |
| `temperature` | `float` | `0.7` | Randomness (0.0-2.0). Higher = more random. |
| `parallel_tool_calls` | `bool` | `True` | Allow multiple tool calls in parallel |
| `tool_choice` | `str` | `"auto"` | `"auto"`, `"required"`, or `"none"` |

### Popular Models

- **Qwen**: `Qwen/Qwen3-235B-A22B`
- **Llama**: `meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo`

### Test It

Run the standalone LLM test:

```bash
uv run python tests/test_llm.py
```

This will send a test prompt and print the response.

---

## Examples

### Minimal Agent (STT + TTS)

Use Telnyx for speech transcription and synthesis with any LLM provider.

```bash
uv run python examples/minimal_agent.py
```

**Code:** [`examples/minimal_agent.py`](examples/minimal_agent.py)

```python
from livekit.agents import AgentSession, AutoSubscribe, WorkerOptions, cli, llm
from livekit.plugins import openai, telnyx

async def entrypoint(ctx):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    session = AgentSession(
        stt=telnyx.STT(
            transcription_engine="deepgram",
            language="en"
        ),
        tts=telnyx.TTS(voice="Telnyx.NaturalHD.astra"),
        llm=openai.LLM(model="gpt-4o-mini"),  # Use any LLM provider
    )
    
    await session.start(
        room=ctx.room,
        agent=llm.ChatContext().append(
            role="system",
            text="You are a helpful voice assistant.",
        ),
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

### Full Stack (STT + TTS + LLM)

Use Telnyx for the entire voice AI stack.

```bash
uv run python examples/full_stack.py
```

**Code:** [`examples/full_stack.py`](examples/full_stack.py)

```python
from livekit.agents import AgentSession, AutoSubscribe, WorkerOptions, cli, llm
from livekit.plugins import openai, telnyx

async def entrypoint(ctx):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    session = AgentSession(
        stt=telnyx.STT(
            transcription_engine="deepgram",
            base_url="wss://api.telnyx.com/v2/speech-to-text/transcription?transcription_model=deepgram/nova-3",
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
```

---

## Resources

### Official Documentation

- **Telnyx STT Plugin**: [GitHub](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-telnyx)
- **Telnyx TTS Plugin**: [GitHub](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-telnyx)
- **Telnyx LLM Integration**: [LiveKit Docs](https://docs.livekit.io/agents/models/llm/telnyx/)
- **LiveKit Agents Framework**: [docs.livekit.io/agents](https://docs.livekit.io/agents/)

### Telnyx APIs

- **Speech-to-Text**: [developers.telnyx.com/docs/voice/programmable-voice/stt-standalone](https://developers.telnyx.com/docs/voice/programmable-voice/stt-standalone)
- **Text-to-Speech**: [developers.telnyx.com/docs/voice/programmable-voice/tts-standalone](https://developers.telnyx.com/docs/voice/programmable-voice/tts-standalone)
- **List TTS Voices**: [developers.telnyx.com/api/call-control/list-text-to-speech-voices](https://developers.telnyx.com/api/call-control/list-text-to-speech-voices)
- **Chat Completions (LLM)**: [developers.telnyx.com/api-reference/chat/create-a-chat-completion](https://developers.telnyx.com/api-reference/chat/create-a-chat-completion)
- **Get Available Models**: [developers.telnyx.com/api-reference/chat/get-available-models](https://developers.telnyx.com/api-reference/chat/get-available-models)

---

## Support

- **Telnyx Support**: [support.telnyx.com](https://support.telnyx.com)
- **LiveKit Discord**: [livekit.io/discord](https://livekit.io/discord)
