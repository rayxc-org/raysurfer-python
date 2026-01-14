#!/usr/bin/env python3
"""
Test variety of code blocks - store and retrieve to verify Pinecone is working.
"""

import asyncio

from raysurfer import AsyncRaySurfer

# Diverse code blocks to test semantic matching
CODE_BLOCKS = [
    {
        "name": "send_email_smtp",
        "description": "Send an email using SMTP with Python",
        "source": '''
import smtplib
from email.mime.text import MIMEText

def send_email(to: str, subject: str, body: str, smtp_server: str = "smtp.gmail.com"):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["To"] = to

    with smtplib.SMTP(smtp_server, 587) as server:
        server.starttls()
        server.login("user@example.com", "password")
        server.send_message(msg)
    return True
''',
        "entrypoint": "send_email",
        "language": "python",
        "tags": ["email", "smtp", "notification"],
    },
    {
        "name": "download_file_url",
        "description": "Download a file from a URL and save to disk",
        "source": '''
import requests

def download_file(url: str, output_path: str) -> str:
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return output_path
''',
        "entrypoint": "download_file",
        "language": "python",
        "tags": ["http", "download", "file"],
    },
    {
        "name": "parse_json_file",
        "description": "Read and parse a JSON file",
        "source": '''
import json

def parse_json(filepath: str) -> dict:
    with open(filepath, 'r') as f:
        return json.load(f)

def write_json(data: dict, filepath: str) -> None:
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
''',
        "entrypoint": "parse_json",
        "language": "python",
        "tags": ["json", "file", "parsing"],
    },
    {
        "name": "scrape_webpage",
        "description": "Scrape text content from a webpage using BeautifulSoup",
        "source": '''
import requests
from bs4 import BeautifulSoup

def scrape_page(url: str) -> dict:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    return {
        "title": soup.title.string if soup.title else None,
        "text": soup.get_text(separator=' ', strip=True),
        "links": [a.get('href') for a in soup.find_all('a', href=True)]
    }
''',
        "entrypoint": "scrape_page",
        "language": "python",
        "tags": ["web", "scraping", "beautifulsoup"],
    },
    {
        "name": "create_sqlite_db",
        "description": "Create and query a SQLite database",
        "source": '''
import sqlite3

def create_database(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    return conn

def execute_query(conn: sqlite3.Connection, query: str, params: tuple = ()) -> list:
    cursor = conn.cursor()
    cursor.execute(query, params)
    return cursor.fetchall()

def create_table(conn: sqlite3.Connection, table_name: str, columns: dict) -> None:
    cols = ", ".join([f"{k} {v}" for k, v in columns.items()])
    conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({cols})")
    conn.commit()
''',
        "entrypoint": "create_database",
        "language": "python",
        "tags": ["database", "sqlite", "sql"],
    },
]

# Queries to test semantic retrieval
TEST_QUERIES = [
    ("send an email notification", "send_email_smtp"),
    ("download file from URL", "download_file_url"),
    ("read JSON data from file", "parse_json_file"),
    ("scrape website content", "scrape_webpage"),
    ("create SQLite database", "create_sqlite_db"),
    # Semantic variations
    ("notify user via email", "send_email_smtp"),
    ("fetch remote file", "download_file_url"),
    ("parse JSON configuration", "parse_json_file"),
    ("extract text from webpage", "scrape_webpage"),
    ("set up database connection", "create_sqlite_db"),
]


async def main():
    print("\n" + "=" * 70)
    print("  RAYSURFER VARIETY TEST - Store & Retrieve Multiple Code Blocks")
    print("=" * 70)

    async with AsyncRaySurfer() as rs:
        # Step 1: Store all code blocks
        print("\n📦 STORING CODE BLOCKS")
        print("-" * 50)
        stored_ids = {}
        for block in CODE_BLOCKS:
            result = await rs.store_code_block(**block)
            stored_ids[block["name"]] = result.code_block_id
            print(f"   ✓ {block['name']}: {result.code_block_id}")

        # Step 2: Test retrieval with various queries
        print("\n🔍 TESTING SEMANTIC RETRIEVAL")
        print("-" * 50)

        correct = 0
        total = len(TEST_QUERIES)

        for query, expected_name in TEST_QUERIES:
            result = await rs.retrieve_best(query)

            if result.best_match:
                matched_name = result.best_match.code_block.name
                is_correct = matched_name == expected_name
                status = "✓" if is_correct else "✗"
                correct += 1 if is_correct else 0

                print(f"   {status} Query: \"{query}\"")
                print(f"      Expected: {expected_name}")
                print(f"      Got:      {matched_name} (score: {result.best_match.combined_score:.0f})")
            else:
                print(f"   ✗ Query: \"{query}\" - No match found")

        # Step 3: Summary
        print("\n" + "=" * 70)
        print("  RESULTS SUMMARY")
        print("=" * 70)
        print(f"\n   Accuracy: {correct}/{total} ({100*correct/total:.0f}%)")
        print(f"   Code blocks stored: {len(stored_ids)}")

        if correct == total:
            print("\n   ✅ All semantic matches correct! Pinecone storage working properly.")
        elif correct >= total * 0.7:
            print("\n   ⚠️  Most matches correct. Some semantic variations may need tuning.")
        else:
            print("\n   ❌ Low accuracy. Check embedding model or stored descriptions.")

        # Step 4: Verify we can retrieve by direct ID
        print("\n🔧 VERIFYING DIRECT RETRIEVAL")
        print("-" * 50)

        # Test retrieve (not retrieve_best) to see raw results
        result = await rs.retrieve("database sql query", top_k=3)
        print("   Query: \"database sql query\"")
        print(f"   Found {result.total_found} matches:")
        for i, match in enumerate(result.code_blocks[:3], 1):
            print(f"      {i}. {match.code_block.name} (score: {match.score:.0f})")


if __name__ == "__main__":
    asyncio.run(main())
