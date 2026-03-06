# Telnyx LiveKit Plugins

Telnyx plugins for [LiveKit Agents](https://docs.livekit.io/agents/) — Speech-to-Text (STT), Text-to-Speech (TTS), and LLM inference.

This repo contains the plugin source code and is published independently, so you get fixes and features without waiting for upstream approvals.

## Installation

Install from this repo:

```bash
pip install git+https://github.com/team-telnyx/livekit-plugins.git#subdirectory=livekit-plugins-telnyx
```

Or for development:

```bash
git clone https://github.com/team-telnyx/livekit-plugins.git
cd livekit-plugins
uv sync
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

Telnyx provides streaming speech-to-text with support for multiple transcription engines, including Deepgram with full parameter support.

### Basic Usage

```python
from livekit.plugins import telnyx

stt = telnyx.STT(
    transcription_engine="deepgram",
    language="en",
)
```

### Deepgram Parameters

When using `transcription_engine="deepgram"`, you can pass Deepgram-specific parameters directly in the constructor:

```python
stt = telnyx.STT(
    transcription_engine="deepgram",
    model="nova-3",
    smart_format=True,
    numerals=True,
    punctuate=True,
    keyterm=["Telnyx", "LiveKit", "your-brand"],
    keywords=["custom-word:2.0"],       # Nova-2 only
    no_delay=True,
    endpointing=300,
    filler_words=False,
    profanity_filter=False,
    diarize=False,
)
```

Any Deepgram parameter not explicitly listed above can be passed as a keyword argument:

```python
stt = telnyx.STT(
    transcription_engine="deepgram",
    model="nova-3",
    tag="my-app",
    multichannel=True,
)
```

#### Deepgram Parameter Reference

**Formatting:**

- `smart_format` (bool) — Auto-format numbers, dates, currency, etc. **Enabled by default on Telnyx.**
- `numerals` (bool) — Format spoken numbers as digits. **Enabled by default on Telnyx.**
- `punctuate` (bool) — Add punctuation to transcript.

**Recognition Boosting:**

- `keyterm` (str or list[str]) — Boost recognition of specific terms. Works with Nova-3 and Flux. Up to 100 terms.
- `keywords` (str or list[str]) — Boost keywords with optional intensity scores (e.g., `"Telnyx:2.0"`). Nova-2 only.

**Model Selection:**

- `model` (str) — Deepgram model: `"nova-3"`, `"nova-2"`, or `"nova-3-flux"`.

**Behavior:**

- `no_delay` (bool) — Reduce latency for real-time applications.
- `endpointing` (int or bool) — Milliseconds of silence before end-of-speech (0–5000ms).
- `filler_words` (bool) — Include filler words ("um", "uh") in transcript.
- `profanity_filter` (bool) — Censor profanity in transcript.
- `diarize` (bool) — Enable speaker identification.
- `vad_events` (bool) — Voice activity detection events.

#### Model Compatibility

- **Nova-3**: `smart_format`, `numerals`, `keyterm`, `punctuate`, `endpointing`, `filler_words`
- **Nova-2**: `smart_format`, `numerals`, `keywords`, `punctuate`, `endpointing`, `filler_words`
- **Flux**: `keyterm`, `punctuate`, `endpointing`, `filler_words` (no `smart_format` in streaming)

### All STT Parameters

- `language` (str, default `"en"`) — Language code
- `transcription_engine` (str, default `"telnyx"`) — Engine: `telnyx`, `google`, `deepgram`, `azure`
- `interim_results` (bool, default `True`) — Return partial transcription results
- `sample_rate` (int, default `16000`) — Audio sample rate in Hz
- `base_url` (str) — WebSocket endpoint override
- `api_key` (str) — Telnyx API key (or set `TELNYX_API_KEY` env var)

### Test It

```bash
uv run python tests/test_stt.py
uv run python tests/test_deepgram_params.py
```

---

## 2. TTS Plugin (Text-to-Speech)

Telnyx provides high-quality streaming text-to-speech with NaturalHD and MiniMax voices.

### Basic Usage

```python
from livekit.plugins import telnyx

tts = telnyx.TTS(
    voice="Telnyx.NaturalHD.astra"
)
```

### Popular Voices

- **NaturalHD**: `Telnyx.NaturalHD.astra`, `Telnyx.NaturalHD.orion`, `Telnyx.NaturalHD.nova`
- **MiniMax**: `Minimax.speech-02-hd.Wise_Woman`, `Minimax.speech-02-hd.Confident_Young_Man`

### Parameters

- `voice` (str, default `"Telnyx.NaturalHD.astra"`) — Voice identifier ([browse voices](https://developers.telnyx.com/api/call-control/list-text-to-speech-voices))
- `base_url` (str) — WebSocket endpoint override

### Test It

```bash
uv run python tests/test_tts.py
```

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

### Popular Models

- **Qwen**: `Qwen/Qwen3-235B-A22B`
- **Llama**: `meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo`

### Test It

```bash
uv run python tests/test_llm.py
```

---

## Examples

### Minimal Agent (STT + TTS)

```bash
uv run python examples/minimal_agent.py
```

```python
from livekit.agents import AgentSession, AutoSubscribe, WorkerOptions, cli, llm
from livekit.plugins import openai, telnyx

async def entrypoint(ctx):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    session = AgentSession(
        stt=telnyx.STT(
            transcription_engine="deepgram",
            model="nova-3",
            keyterm=["your-brand"],
        ),
        tts=telnyx.TTS(voice="Telnyx.NaturalHD.astra"),
        llm=openai.LLM(model="gpt-4o-mini"),
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

```bash
uv run python examples/full_stack.py
```

```python
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
```

---

## Resources

- **LiveKit Agents Framework**: [docs.livekit.io/agents](https://docs.livekit.io/agents/)
- **Telnyx STT Docs**: [developers.telnyx.com/docs/voice/programmable-voice/stt-standalone](https://developers.telnyx.com/docs/voice/programmable-voice/stt-standalone)
- **Telnyx TTS Docs**: [developers.telnyx.com/docs/voice/programmable-voice/tts-standalone](https://developers.telnyx.com/docs/voice/programmable-voice/tts-standalone)
- **Deepgram Parameters**: [developers.deepgram.com/docs/features](https://developers.deepgram.com/docs/features)
- **Telnyx Support**: [support.telnyx.com](https://support.telnyx.com)
