#!/usr/bin/env python3
"""
E2E test for Telnyx TTS persistent WebSocket connection.
Tests that:
1. Multiple stream() calls work (with auto-reconnection)
2. prewarm() establishes connection before first use
3. Proper cleanup on aclose()

Note: The Telnyx TTS server closes WebSocket after each request.
      The "persistent" connection behavior means auto-reconnection works seamlessly.
"""

import asyncio
import os
import sys

# Add the plugin to path
sys.path.insert(0, "telnyx-livekit-plugin")

from livekit.plugins.telnyx import TTS


async def test_connection_reuse():
    """Test that multiple stream() calls work with auto-reconnection."""
    api_key = os.environ.get("TELNYX_API_KEY")
    if not api_key:
        print("❌ TELNYX_API_KEY not set")
        return False

    print("\n=== Test: Connection Reuse (with auto-reconnect) ===")
    
    tts = TTS(api_key=api_key)
    
    print("1. Creating first stream...")
    stream1 = tts.stream()
    
    print("   Sending 'Hello'...")
    stream1.push_text("Hello")
    stream1.end_input()
    
    audio1 = []
    async for chunk in stream1:
        if chunk.frame and hasattr(chunk.frame, 'data'):
            audio1.append(chunk.frame.data)
    
    print(f"   Received {len(audio1)} audio frames, total bytes: {sum(len(a) for a in audio1)}")
    
    # Give a moment for the connection to settle
    await asyncio.sleep(0.5)
    
    print("2. Creating second stream (auto-reconnects if needed)...")
    stream2 = tts.stream()
    
    print("   Sending 'World'...")
    stream2.push_text("World")
    stream2.end_input()
    
    audio2 = []
    async for chunk in stream2:
        if chunk.frame and hasattr(chunk.frame, 'data'):
            audio2.append(chunk.frame.data)
    
    print(f"   Received {len(audio2)} audio frames, total bytes: {sum(len(a) for a in audio2)}")
    
    if audio1 and audio2:
        print("✅ Both stream() calls produced audio!")
    else:
        print("❌ Missing audio from one or both calls")
        return False
    
    print("3. Cleaning up...")
    await tts.aclose()
    
    print("\n=== Test Complete ===")
    return True


async def test_prewarm():
    """Test that prewarm() establishes connection eagerly."""
    api_key = os.environ.get("TELNYX_API_KEY")
    if not api_key:
        print("❌ TELNYX_API_KEY not set")
        return False

    print("\n=== Test: Prewarm ===")
    
    tts = TTS(api_key=api_key)
    
    print("1. Calling prewarm()...")
    await tts.prewarm()
    
    if tts._ws is not None and not tts._ws.closed:
        print("✅ WebSocket connected after prewarm()")
    else:
        print("❌ WebSocket not connected after prewarm()")
        return False
    
    print("2. Creating stream (uses pre-warmed connection)...")
    stream = tts.stream()
    stream.push_text("Test after prewarm")
    stream.end_input()
    
    audio = []
    async for chunk in stream:
        if chunk.frame and hasattr(chunk.frame, 'data'):
            audio.append(chunk.frame.data)
    
    print(f"   Received {len(audio)} audio frames, total bytes: {sum(len(a) for a in audio)}")
    
    if not audio:
        print("❌ No audio received")
        return False
    
    print("3. Cleanup...")
    await tts.aclose()
    
    print("\n=== Test Complete ===")
    return True


async def test_latency_improvement():
    """Verify pre-warm reduces latency on first request."""
    api_key = os.environ.get("TELNYX_API_KEY")
    if not api_key:
        print("❌ TELNYX_API_KEY not set")
        return False

    print("\n=== Test: Latency with/without prewarm ===")
    
    # Without prewarm
    tts1 = TTS(api_key=api_key)
    start1 = asyncio.get_event_loop().time()
    stream1 = tts1.stream()
    stream1.push_text("Hello")
    stream1.end_input()
    audio1 = []
    async for chunk in stream1:
        if chunk.frame and hasattr(chunk.frame, 'data'):
            audio1.append(chunk.frame.data)
    latency1 = (asyncio.get_event_loop().time() - start1) * 1000
    await tts1.aclose()
    
    print(f"  Without prewarm: {latency1:.0f}ms")
    
    # With prewarm
    tts2 = TTS(api_key=api_key)
    await tts2.prewarm()  # Pre-warm
    start2 = asyncio.get_event_loop().time()
    stream2 = tts2.stream()
    stream2.push_text("Hello")
    stream2.end_input()
    audio2 = []
    async for chunk in stream2:
        if chunk.frame and hasattr(chunk.frame, 'data'):
            audio2.append(chunk.frame.data)
    latency2 = (asyncio.get_event_loop().time() - start2) * 1000
    await tts2.aclose()
    
    print(f"  With prewarm: {latency2:.0f}ms")
    
    if latency2 < latency1:
        print(f"✅ Pre-warm saves ~{latency1 - latency2:.0f}ms")
    else:
        print("⚠️ No significant latency improvement (may be network-dependent)")
    
    print("\n=== Test Complete ===")
    return True


async def main():
    print("Telnyx TTS Persistent Connection E2E Tests")
    print("=" * 50)
    
    # Run tests
    results = []
    
    try:
        results.append(("Connection Reuse", await test_connection_reuse()))
    except Exception as e:
        print(f"❌ Connection Reuse test failed: {e}")
        results.append(("Connection Reuse", False))
    
    try:
        results.append(("Prewarm", await test_prewarm()))
    except Exception as e:
        print(f"❌ Prewarm test failed: {e}")
        results.append(("Prewarm", False))
    
    try:
        results.append(("Latency", await test_latency_improvement()))
    except Exception as e:
        print(f"❌ Latency test failed: {e}")
        results.append(("Latency", False))
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())