from livekit.plugins import telnyx
from livekit.plugins.telnyx.stt import _build_ws_url


FAKE_API_KEY = "KEY123"


def test_deepgram_stt_accepts_flux_multi_model() -> None:
    stt = telnyx.deepgram.STT(model="flux-multi", api_key=FAKE_API_KEY)

    assert stt.model == "flux-multi"
    assert stt._opts.deepgram_params["model"] == "flux-multi"


def test_deepgram_stt_accepts_deepgram_flux_multi_model() -> None:
    stt = telnyx.deepgram.STT(model="deepgram/flux-multi", api_key=FAKE_API_KEY)

    assert stt.model == "deepgram/flux-multi"
    assert stt._opts.deepgram_params["model"] == "deepgram/flux-multi"


def test_deepgram_stt_forwards_language_hint() -> None:
    stt = telnyx.deepgram.STT(
        model="deepgram/flux-multi",
        language_hint=["en", "es"],
        api_key=FAKE_API_KEY,
    )

    assert stt._opts.deepgram_params["language_hint"] == ["en", "es"]


def test_deepgram_stt_forwards_single_language_hint_as_repeated_param_list() -> None:
    stt = telnyx.deepgram.STT(
        model="deepgram/flux-multi",
        language_hint="pt-BR",
        api_key=FAKE_API_KEY,
    )

    assert stt._opts.deepgram_params["language_hint"] == ["pt-BR"]


def test_build_ws_url_repeats_language_hint_params() -> None:
    url = _build_ws_url(
        "wss://api.telnyx.com/v2/speech-to-text/transcription",
        {
            "transcription_engine": "Deepgram",
            "model": "deepgram/flux-multi",
            "language_hint": ["en", "es"],
        },
    )

    assert "model=deepgram%2Fflux-multi" in url
    assert "language_hint=en" in url
    assert "language_hint=es" in url


def test_existing_flux_model_still_uses_english_flux_alias() -> None:
    stt = telnyx.deepgram.STT(model="flux", api_key=FAKE_API_KEY)

    assert stt.model == "flux"
    assert stt._opts.deepgram_params["model"] == "flux"
    assert "language_hint" not in stt._opts.deepgram_params
