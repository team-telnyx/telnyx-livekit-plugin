#!/usr/bin/env python3
"""
Test Telnyx STT plugin standalone.

Usage:
    uv run python tests/test_stt.py
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents.utils import AudioBuffer
from livekit.plugins import telnyx

# Load environment variables
load_dotenv()

async def test_stt():
    """Test STT with a sample audio file."""
    
    # Check for API key
    api_key = os.getenv("TELNYX_API_KEY")
    if not api_key:
        print("❌ TELNYX_API_KEY not found in environment")
        print("   Add it to .env or export it:")
        print("   export TELNYX_API_KEY=your_api_key_here")
        return
    
    # Check for sample audio file
    sample_path = Path(__file__).parent / "assets" / "sample.wav"
    if not sample_path.exists():
        print(f"❌ Sample audio not found: {sample_path}")
        print("   Add a WAV file to tests/assets/sample.wav")
        return
    
    print("🎙️  Testing Telnyx STT...")
    print(f"   Audio file: {sample_path}")
    
    # Initialize STT
    stt = telnyx.STT(
        language="en",
        transcription_engine="deepgram",  # Using Deepgram engine
        interim_results=True,
        api_key=api_key
    )
    
    try:
        # Read audio file
        with open(sample_path, "rb") as f:
            audio_data = f.read()
        
        # Create audio buffer (skip WAV header, typically 44 bytes)
        audio_buffer = AudioBuffer(data=audio_data[44:], sample_rate=16000, num_channels=1)
        
        # Transcribe
        print("   Transcribing...")
        result = await stt._recognize_impl(audio_buffer)
        
        if result.alternatives:
            transcript = result.alternatives[0].text
            print(f"\n✅ Transcript: {transcript}\n")
        else:
            print("\n⚠️  No transcription results\n")
    
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
    
    finally:
        await stt.aclose()

if __name__ == "__main__":
    asyncio.run(test_stt())
