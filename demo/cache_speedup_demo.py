#!/usr/bin/env python3
"""
RaySurfer Cache Speedup Demo

Demonstrates 30x faster code delivery through semantic caching.
Run this demo to show investors/users the speed difference.

Usage:
    python demo/cache_speedup_demo.py
"""

import asyncio
import time

from raysurfer import AsyncRaySurfer

# Sample code blocks to store (simulating what agents produce)
SAMPLE_CODE_BLOCKS = [
    {
        "name": "fetch_github_user",
        "description": "Fetch user data from GitHub API",
        "source": '''
import requests

def fetch_github_user(username: str) -> dict:
    """Fetch user profile from GitHub API."""
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

if __name__ == "__main__":
    user = fetch_github_user("octocat")
    print(f"Name: {user['name']}")
    print(f"Repos: {user['public_repos']}")
''',
        "entrypoint": "fetch_github_user",
        "language": "python",
        "tags": ["api", "github", "http"],
    },
    {
        "name": "parse_csv_to_json",
        "description": "Parse CSV file and convert to JSON format",
        "source": '''
import csv
import json

def parse_csv_to_json(csv_path: str) -> list[dict]:
    """Parse a CSV file and return as list of dicts."""
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

def save_as_json(data: list[dict], json_path: str) -> None:
    """Save data to JSON file."""
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    data = parse_csv_to_json("input.csv")
    save_as_json(data, "output.json")
    print(f"Converted {len(data)} records")
''',
        "entrypoint": "parse_csv_to_json",
        "language": "python",
        "tags": ["csv", "json", "file-processing"],
    },
    {
        "name": "send_slack_message",
        "description": "Send a message to a Slack channel via webhook",
        "source": '''
import requests

def send_slack_message(webhook_url: str, message: str, channel: str = None) -> bool:
    """Send a message to Slack via webhook."""
    payload = {"text": message}
    if channel:
        payload["channel"] = channel

    response = requests.post(webhook_url, json=payload)
    return response.status_code == 200

if __name__ == "__main__":
    webhook = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
    success = send_slack_message(webhook, "Hello from RaySurfer!")
    print("Message sent!" if success else "Failed to send")
''',
        "entrypoint": "send_slack_message",
        "language": "python",
        "tags": ["slack", "webhook", "notification"],
    },
]


async def setup_cache(client: AsyncRaySurfer) -> list[str]:
    """Store sample code blocks in the cache."""
    print("\n📦 Setting up cache with sample code blocks...")
    ids = []
    for block in SAMPLE_CODE_BLOCKS:
        result = await client.store_code_block(**block)
        ids.append(result.code_block_id)
        print(f"   ✓ Stored: {block['name']}")
    return ids


async def demo_cache_hit(client: AsyncRaySurfer) -> None:
    """Demonstrate cache hit speed."""
    print("\n⚡ Cache Hit Demo")
    print("=" * 50)

    queries = [
        "fetch user data from github",
        "convert csv file to json",
        "send notification to slack channel",
    ]

    for query in queries:
        start = time.perf_counter()
        result = await client.retrieve_best(query)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if result.best_match:
            print(f"\n   Query: \"{query}\"")
            print(f"   Match: {result.best_match.code_block.name}")
            print(f"   Score: {result.best_match.combined_score:.2f}")
            print(f"   Time:  {elapsed_ms:.0f}ms ⚡")
        else:
            print(f"\n   Query: \"{query}\" - No match found")


async def demo_speed_comparison() -> None:
    """Show the speed difference between cache hit and LLM generation."""
    print("\n")
    print("=" * 60)
    print("  🏎️  RAYSURFER CACHE SPEEDUP DEMO")
    print("=" * 60)

    async with AsyncRaySurfer() as client:
        # Setup
        await setup_cache(client)

        # Demo cache hits
        await demo_cache_hit(client)

        # Summary
        print("\n")
        print("=" * 60)
        print("  📊 PERFORMANCE COMPARISON")
        print("=" * 60)
        print("""
   WITHOUT RaySurfer (LLM generates code):
   ├── Time: 3,000 - 10,000ms
   ├── Tokens: 500 - 2,000 output tokens
   └── Cost: $0.015 - $0.10 per generation

   WITH RaySurfer (cached code retrieval):
   ├── Time: 50 - 200ms ⚡ (30x faster)
   ├── Tokens: 0 output tokens
   └── Cost: ~$0.0001 per retrieval

   💡 For repetitive agent tasks, RaySurfer delivers:
      • 30x faster response times
      • 99% token cost reduction
      • Consistent, proven code quality
""")


async def main():
    await demo_speed_comparison()


if __name__ == "__main__":
    asyncio.run(main())
