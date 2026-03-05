#!/usr/bin/env python3
"""
Test Telnyx LLM integration standalone.

Usage:
    uv run python tests/test_llm.py
"""

import asyncio
import os

from dotenv import load_dotenv
from livekit.agents import llm
from livekit.plugins import openai

# Load environment variables
load_dotenv()

async def test_llm():
    """Test LLM with a sample prompt."""
    
    # Check for API key
    api_key = os.getenv("TELNYX_API_KEY")
    if not api_key:
        print("❌ TELNYX_API_KEY not found in environment")
        print("   Add it to .env or export it:")
        print("   export TELNYX_API_KEY=your_api_key_here")
        return
    
    print("🤖 Testing Telnyx LLM...")
    
    # Initialize LLM with Telnyx
    model = openai.LLM.with_telnyx(
        model="Qwen/Qwen3-235B-A22B",
        temperature=0.7
    )
    
    try:
        # Test prompt
        test_prompt = "What is the capital of France? Reply in one sentence."
        print(f"   Prompt: {test_prompt}")
        print(f"   Model: Qwen/Qwen3-235B-A22B")
        
        # Create chat context
        chat_context = llm.ChatContext()
        chat_context.append(role="user", text=test_prompt)
        
        # Get response
        print("   Generating response...")
        stream = model.chat(chat_ctx=chat_context)
        
        response_text = ""
        async for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    response_text += delta
        
        if response_text:
            print(f"\n✅ Response: {response_text}\n")
        else:
            print("\n⚠️  No response generated\n")
    
    except Exception as e:
        print(f"\n❌ Error: {e}\n")

if __name__ == "__main__":
    asyncio.run(test_llm())
