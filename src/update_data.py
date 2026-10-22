"""GDELT headline exploration -> local SQLite (educational example only)."""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from article_security_classifier import classify_topic

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "demo_articles.db"
API = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_RECORDS = 50


def fetch_articles():
    """Fetch a small public metadata sample. Network/API errors are surfaced."""
    params = {
        "query": '(cyberattack OR "energy security" OR military OR diplomacy)',
        "mode": "artlist",
        "format": "json",
        "maxrecords": MAX_RECORDS,
        "timespan": "1week",
    }
    response = requests.get(
        API, params=params, timeout=30,
        headers={"User-Agent": "EuropeanSecurityMonitor-EducationalDemo/1.0"},
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Unexpected GDELT API payload")
    return payload.get("articles", []) or []


def normalize(article):
    url = str(article.get("url") or "").strip()
    title = str(article.get("title") or "").strip()
    if not url.startswith(("http://", "https://")) or not title:
        return None
    return (
        url,
        title[:1000],
        str(article.get("seendate") or "")[:32],
        str(article.get("sourcecountry") or "")[:100],
        urlsplit(url).hostname or "",
        classify_topic(title),
        datetime.now(timezone.utc).isoformat(),
    )


def save_articles(articles):
    DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS articles (
            url TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            seen_date TEXT,
            source_country TEXT,
            source_domain TEXT,
            topic TEXT,
            fetched_at TEXT
        )""")
        records = [row for item in articles if (row := normalize(item))]
        db.executemany(
            """INSERT INTO articles
            (url,title,seen_date,source_country,source_domain,topic,fetched_at)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(url) DO UPDATE SET
                title=excluded.title,
                seen_date=excluded.seen_date,
                source_country=excluded.source_country,
                source_domain=excluded.source_domain,
                topic=excluded.topic,
                fetched_at=excluded.fetched_at""",
            records,
        )
    return len(records)


def main():
    items = fetch_articles()
    count = save_articles(items)
    print(f"Processed {count} public article metadata records into {DB}")


if __name__ == "__main__":
    main()
