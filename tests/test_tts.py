#!/usr/bin/env python3
"""
Test Telnyx TTS plugin standalone.

Usage:
    uv run python tests/test_tts.py
"""

import asyncio
import os
import wave
from pathlib import Path

from dotenv import load_dotenv
from livekit.plugins import telnyx

# Load environment variables
load_dotenv()

async def test_tts():
    """Test TTS with a sample text."""
    
    # Check for API key
    api_key = os.getenv("TELNYX_API_KEY")
    if not api_key:
        print("❌ TELNYX_API_KEY not found in environment")
        print("   Add it to .env or export it:")
        print("   export TELNYX_API_KEY=your_api_key_here")
        return
    
    print("🔊 Testing Telnyx TTS...")
    
    # Initialize TTS
    tts = telnyx.TTS(
        voice="Telnyx.NaturalHD.astra",
        api_key=api_key
    )
    
    try:
        # Test text
        test_text = "Hello, this is a test of the Telnyx text to speech plugin for LiveKit."
        print(f"   Text: {test_text}")
        print(f"   Voice: Telnyx.NaturalHD.astra")
        
        # Synthesize
        print("   Synthesizing...")
        stream = tts.synthesize(test_text)
        
        # Collect audio chunks
        audio_chunks = []
        async for event in stream:
            if event.type == "audio":
                audio_chunks.append(event.frame.data.tobytes())
        
        if audio_chunks:
            # Save to file
            output_path = Path(__file__).parent / "output.wav"
            with wave.open(str(output_path), "wb") as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)   # 16-bit
                wav_file.setframerate(16000)
                wav_file.writeframes(b"".join(audio_chunks))
            
            print(f"\n✅ Audio saved to: {output_path}\n")
        else:
            print("\n⚠️  No audio generated\n")
    
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
    
    finally:
        await tts.aclose()

if __name__ == "__main__":
    asyncio.run(test_tts())
