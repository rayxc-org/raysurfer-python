#!/usr/bin/env python3
"""
Fetch the top 5 trending repositories from GitHub.

Uses the GitHub Search API to find repositories created in the last week,
sorted by stars (a proxy for trending).
"""

import json
import urllib.error
import urllib.request
from datetime import datetime, timedelta


def get_trending_repos(count: int = 5) -> list[dict]:
    """
    Fetch trending repositories from GitHub.

    Args:
        count: Number of repositories to fetch (default: 5)

    Returns:
        List of repository information dictionaries
    """
    # Get repositories created in the last 7 days, sorted by stars
    one_week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    # GitHub Search API endpoint
    url = (
        f"https://api.github.com/search/repositories"
        f"?q=created:>{one_week_ago}"
        f"&sort=stars"
        f"&order=desc"
        f"&per_page={count}"
    )

    # Create request with headers
    request = urllib.request.Request(
        url, headers={"Accept": "application/vnd.github.v3+json", "User-Agent": "Python-Trending-Repos-Script"}
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data.get("items", [])
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} - {e.reason}")
        raise
    except urllib.error.URLError as e:
        print(f"URL Error: {e.reason}")
        raise


def display_repos(repos: list[dict]) -> None:
    """Display repository information in a formatted way."""
    print("\n" + "=" * 60)
    print("🔥 TOP 5 TRENDING GITHUB REPOSITORIES (Last 7 Days)")
    print("=" * 60 + "\n")

    for i, repo in enumerate(repos, 1):
        print(f"{i}. {repo['full_name']}")
        print(f"   ⭐ Stars: {repo['stargazers_count']:,}")
        print(f"   📝 Description: {repo.get('description') or 'No description'}")
        print(f"   🔗 URL: {repo['html_url']}")
        print(f"   💻 Language: {repo.get('language') or 'Not specified'}")
        print()


def main():
    """Main function to fetch and display trending repos."""
    try:
        repos = get_trending_repos(5)
        if repos:
            display_repos(repos)
        else:
            print("No trending repositories found.")
    except Exception as e:
        print(f"Failed to fetch trending repositories: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
