"""
Test script comparing Bidflow agent performance:
- raysurfer SDK (with caching)
- claude_agent_sdk (direct, no caching)

Measures: response time, caching behavior, output quality
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Ensure the bidflow agent can be imported
sys.path.insert(0, str(Path(__file__).parent.parent / "yc_w26_agents" / "bidflow"))

from dotenv import load_dotenv

# Load .env from parent directory (raysurfer)
load_dotenv(Path(__file__).parent.parent / ".env")

# Set RAYSURFER_API_KEY for testing if not already set
# The backend has API_KEYS_ENABLED=false so any key should work
if not os.environ.get("RAYSURFER_API_KEY"):
    os.environ["RAYSURFER_API_KEY"] = "rs_test_bidflow_comparison"

# Task related to Bidflow's domain (electrical estimating)
TEST_TASK = """
Calculate a rough material cost estimate for a small commercial office space
(2,000 sq ft) that needs:
- 20 LED recessed lights (4-inch)
- 15 duplex receptacles
- 3 GFCI receptacles
- 200A main panel
- Basic lighting circuit wiring

Just provide a quick summary breakdown of estimated material costs.
Do not use any tools - just respond with your knowledge.
"""

# Paths to prompts
PROMPTS_DIR = Path(__file__).parent.parent / "yc_w26_agents" / "bidflow" / "bidflow_agent" / "prompts"


def load_prompt(filename: str) -> str:
    """Load a prompt from the prompts directory."""
    prompt_path = PROMPTS_DIR / filename
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


async def test_with_raysurfer():
    """Test using raysurfer SDK (with caching)."""
    from raysurfer import AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, ResultMessage

    lead_agent_prompt = load_prompt("lead_agent.txt")

    options = ClaudeAgentOptions(
        permission_mode="bypassPermissions",
        system_prompt=lead_agent_prompt,
        allowed_tools=["Read", "Write", "Bash", "Task"],
        model="claude-opus-4-5-20250514",
    )

    response_text = []
    cache_info = {"enabled": bool(os.environ.get("RAYSURFER_API_KEY"))}

    start_time = time.time()

    async with ClaudeSDKClient(options=options) as client:
        await client.query(TEST_TASK)
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if hasattr(block, "text"):
                        response_text.append(block.text)
            elif isinstance(msg, ResultMessage):
                cache_info["success"] = msg.subtype == "success"

    end_time = time.time()

    return {
        "sdk": "raysurfer",
        "response": "".join(response_text),
        "time": end_time - start_time,
        "cache_enabled": cache_info.get("enabled", False),
        "success": cache_info.get("success", False),
    }


async def test_with_claude_agent_sdk():
    """Test using claude_agent_sdk directly (no caching)."""
    from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, ResultMessage

    lead_agent_prompt = load_prompt("lead_agent.txt")

    options = ClaudeAgentOptions(
        permission_mode="bypassPermissions",
        system_prompt=lead_agent_prompt,
        allowed_tools=["Read", "Write", "Bash", "Task"],
        model="claude-opus-4-5-20250514",
    )

    response_text = []
    result_info = {}

    start_time = time.time()

    async with ClaudeSDKClient(options=options) as client:
        await client.query(TEST_TASK)
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if hasattr(block, "text"):
                        response_text.append(block.text)
            elif isinstance(msg, ResultMessage):
                result_info["success"] = msg.subtype == "success"

    end_time = time.time()

    return {
        "sdk": "claude_agent_sdk",
        "response": "".join(response_text),
        "time": end_time - start_time,
        "cache_enabled": False,
        "success": result_info.get("success", False),
    }


def compare_results(raysurfer_result, claude_sdk_result):
    """Compare and display results."""
    print("\n" + "=" * 70)
    print("  BIDFLOW AGENT COMPARISON: RAYSURFER vs CLAUDE-AGENT-SDK")
    print("=" * 70)

    print("\nTest Task: Calculate material costs for 2,000 sq ft office space")
    print("-" * 70)

    # Timing comparison
    print("\n## TIMING COMPARISON")
    print(f"  Raysurfer:        {raysurfer_result['time']:.2f} seconds")
    print(f"  Claude-Agent-SDK: {claude_sdk_result['time']:.2f} seconds")

    time_diff = claude_sdk_result["time"] - raysurfer_result["time"]
    if time_diff > 0:
        print(
            f"  --> Raysurfer was {time_diff:.2f}s FASTER ({(time_diff/claude_sdk_result['time'])*100:.1f}% improvement)"
        )
    elif time_diff < 0:
        print(f"  --> Claude-Agent-SDK was {-time_diff:.2f}s FASTER")
    else:
        print("  --> Both took the same time")

    # Caching info
    print("\n## CACHING STATUS")
    print(f"  Raysurfer caching enabled: {raysurfer_result['cache_enabled']}")
    print(f"  RAYSURFER_API_KEY set: {bool(os.environ.get('RAYSURFER_API_KEY'))}")

    # Success status
    print("\n## TASK COMPLETION")
    print(f"  Raysurfer success:        {raysurfer_result['success']}")
    print(f"  Claude-Agent-SDK success: {claude_sdk_result['success']}")

    # Response length comparison
    print("\n## RESPONSE QUALITY")
    print(f"  Raysurfer response length:        {len(raysurfer_result['response'])} chars")
    print(f"  Claude-Agent-SDK response length: {len(claude_sdk_result['response'])} chars")

    # Show responses
    print("\n" + "-" * 70)
    print("## RAYSURFER RESPONSE:")
    print("-" * 70)
    print(raysurfer_result["response"][:2000] + ("..." if len(raysurfer_result["response"]) > 2000 else ""))

    print("\n" + "-" * 70)
    print("## CLAUDE-AGENT-SDK RESPONSE:")
    print("-" * 70)
    print(claude_sdk_result["response"][:2000] + ("..." if len(claude_sdk_result["response"]) > 2000 else ""))

    # Summary
    print("\n" + "=" * 70)
    print("## SUMMARY")
    print("=" * 70)

    winner = "Raysurfer" if raysurfer_result["time"] < claude_sdk_result["time"] else "Claude-Agent-SDK"
    print(f"  Faster SDK: {winner}")
    print(f"  Speed difference: {abs(time_diff):.2f} seconds")

    if raysurfer_result["cache_enabled"]:
        print("  Raysurfer caching: ACTIVE (may include cached code in prompt)")
    else:
        print("  Raysurfer caching: NOT ACTIVE (RAYSURFER_API_KEY not set)")

    return {
        "winner": winner,
        "time_difference": time_diff,
        "raysurfer_time": raysurfer_result["time"],
        "claude_sdk_time": claude_sdk_result["time"],
    }


async def main():
    """Run the comparison tests."""
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set in environment")
        return

    print("Starting Bidflow agent comparison test...")
    print(f"ANTHROPIC_API_KEY: {'set' if os.environ.get('ANTHROPIC_API_KEY') else 'NOT SET'}")
    print(f"RAYSURFER_API_KEY: {'set' if os.environ.get('RAYSURFER_API_KEY') else 'NOT SET'}")

    errors = []

    # Test with raysurfer
    print("\n[1/2] Testing with RAYSURFER SDK...")
    try:
        raysurfer_result = await test_with_raysurfer()
        print(f"      Completed in {raysurfer_result['time']:.2f}s")
    except Exception as e:
        print(f"      ERROR: {e}")
        errors.append(("raysurfer", str(e)))
        raysurfer_result = None

    # Test with claude_agent_sdk
    print("\n[2/2] Testing with CLAUDE-AGENT-SDK...")
    try:
        claude_sdk_result = await test_with_claude_agent_sdk()
        print(f"      Completed in {claude_sdk_result['time']:.2f}s")
    except Exception as e:
        print(f"      ERROR: {e}")
        errors.append(("claude_agent_sdk", str(e)))
        claude_sdk_result = None

    # Compare if both succeeded
    if raysurfer_result and claude_sdk_result:
        compare_results(raysurfer_result, claude_sdk_result)
    else:
        print("\n" + "=" * 70)
        print("  COMPARISON FAILED - ERRORS ENCOUNTERED")
        print("=" * 70)
        for sdk, error in errors:
            print(f"\n{sdk}: {error}")


if __name__ == "__main__":
    asyncio.run(main())
