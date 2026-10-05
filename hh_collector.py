#!/usr/bin/env python3
"""
Asynchronous HeadHunter (hh.ru) API Vacancy Collector.
Course: Machine Learning Algorithms - Midterm Project
Author: Galymzhan Turemuratov

Collects IT vacancies from https://api.hh.ru/vacancies with detailed endpoints
to capture key_skills, full descriptions, salary ranges, experience, and schedules.
Saves raw snapshot to both CSV and SQLite formats.
"""

import asyncio
import csv
import json
import logging
import sqlite3
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import aiohttp
except ImportError:
    aiohttp = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("HH_Collector")

BASE_URL = "https://api.hh.ru/vacancies"
USER_AGENT = "HH-Grade-Classification-StudentProject/1.0 (academic.research@edu.university)"

# Target skills to monitor
TARGET_SKILLS = ["python", "go", "golang", "fastapi", "postgresql", "docker"]


async def fetch_page(
    session: aiohttp.ClientSession,
    text: str,
    page: int,
    per_page: int = 100,
    area: int = 113  # 113 = Russia, or 40 = Kazakhstan / 1 = Moscow
) -> List[Dict[str, Any]]:
    """Fetch a single page of vacancy search results."""
    params = {
        "text": text,
        "page": page,
        "per_page": per_page,
        "area": area,
        "order_by": "publication_time"
    }
    headers = {"User-Agent": USER_AGENT}
    try:
        async with session.get(BASE_URL, params=params, headers=headers, timeout=15) as resp:
            if resp.status == 200:
                data = await resp.json()
                items = data.get("items", [])
                logger.info(f"Page {page} fetched: {len(items)} vacancies found.")
                return items
            elif resp.status == 429:
                logger.warning("Rate limit hit (HTTP 429). Sleeping for 5 seconds...")
                await asyncio.sleep(5)
                return []
            else:
                logger.error(f"Error fetching page {page}: HTTP {resp.status}")
                return []
    except Exception as e:
        logger.error(f"Exception during page {page} fetch: {e}")
        return []


async def fetch_vacancy_detail(
    session: aiohttp.ClientSession,
    vacancy_id: str,
    semaphore: asyncio.Semaphore
) -> Optional[Dict[str, Any]]:
    """Fetch full vacancy details (including key_skills and raw HTML description)."""
    headers = {"User-Agent": USER_AGENT}
    url = f"{BASE_URL}/{vacancy_id}"
    async with semaphore:
        for attempt in range(3):
            try:
                async with session.get(url, headers=headers, timeout=12) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 429:
                        await asyncio.sleep(2 * (attempt + 1))
                    elif resp.status == 404:
                        return None
                    else:
                        logger.debug(f"Failed to fetch detail for {vacancy_id}: HTTP {resp.status}")
            except Exception as e:
                await asyncio.sleep(1)
        return None


def parse_vacancy_payload(detail: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and normalize all required fields according to course specifications."""
    vac_id = detail.get("id")
    name = detail.get("name", "")
    
    # Experience parsing
    exp_dict = detail.get("experience") or {}
    experience_id = exp_dict.get("id", "")
    experience_name = exp_dict.get("name", "")
    
    # Salary parsing
    salary_dict = detail.get("salary") or {}
    salary_from = salary_dict.get("from")
    salary_to = salary_dict.get("to")
    salary_currency = salary_dict.get("currency")
    
    # Schedule
    schedule_dict = detail.get("schedule") or {}
    schedule_name = schedule_dict.get("name", "") or schedule_dict.get("id", "")
    
    # Key skills extraction
    raw_skills = detail.get("key_skills", [])
    skill_names = [s.get("name", "").strip() for s in raw_skills if s.get("name")]
    
    # Description (raw HTML preserved for snapshot)
    description = detail.get("description", "")
    
    return {
        "id": vac_id,
        "name": name,
        "experience": experience_name,
        "experience_id": experience_id,
        "salary_from": salary_from,
        "salary_to": salary_to,
        "salary_currency": salary_currency,
        "schedule": schedule_name,
        "key_skills": json.dumps(skill_names, ensure_ascii=False),
        "description": description
    }


def save_raw_sqlite(records: List[Dict[str, Any]], db_path: str):
    """Save raw snapshot records to SQLite database."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS raw_vacancies (
            id TEXT PRIMARY KEY,
            name TEXT,
            experience TEXT,
            experience_id TEXT,
            salary_from REAL,
            salary_to REAL,
            salary_currency TEXT,
            schedule TEXT,
            key_skills TEXT,
            description TEXT,
            collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    for r in records:
        cur.execute("""
            INSERT OR REPLACE INTO raw_vacancies 
            (id, name, experience, experience_id, salary_from, salary_to, salary_currency, schedule, key_skills, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(r["id"]), r["name"], r["experience"], r["experience_id"],
            r["salary_from"], r["salary_to"], r["salary_currency"],
            r["schedule"], r["key_skills"], r["description"]
        ))
    conn.commit()
    conn.close()
    logger.info(f"Successfully saved {len(records)} records into SQLite: {db_path}")


def save_raw_csv(records: List[Dict[str, Any]], csv_path: str):
    """Save raw snapshot records to CSV."""
    if not records:
        return
    fieldnames = list(records[0].keys())
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    logger.info(f"Successfully saved {len(records)} records into CSV: {csv_path}")


async def run_collector(
    query_text: str = "Python OR Go OR Backend OR Developer",
    target_count: int = 1100,
    concurrency_limit: int = 8,
    output_csv: str = "hh_vacancies_raw_snapshot.csv",
    output_db: str = "hh_vacancies_raw.db"
):
    """Orchestrate asynchronous fetching of list and detailed vacancies."""
    if aiohttp is None:
        raise RuntimeError("aiohttp is required for async parsing. Install via 'pip install aiohttp'.")
        
    logger.info(f"Starting async collection for query: '{query_text}'. Target: >= {target_count} vacancies.")
    
    connector = aiohttp.TCPConnector(limit=concurrency_limit, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Step 1: Collect vacancy IDs from search pages
        vacancy_ids = []
        page = 0
        max_pages = 20  # HH allows up to 2000 vacancies (20 pages * 100)
        
        while len(vacancy_ids) < target_count and page < max_pages:
            items = await fetch_page(session, text=query_text, page=page, per_page=100)
            if not items:
                break
            for item in items:
                v_id = str(item.get("id"))
                if v_id and v_id not in vacancy_ids:
                    vacancy_ids.append(v_id)
            page += 1
            await asyncio.sleep(0.3)  # Gentle delay between page requests
            
        logger.info(f"Discovered {len(vacancy_ids)} unique vacancy IDs. Fetching full details concurrently...")
        
        # Step 2: Fetch detailed items concurrently
        semaphore = asyncio.Semaphore(concurrency_limit)
        tasks = [fetch_vacancy_detail(session, v_id, semaphore) for v_id in vacancy_ids]
        raw_results = await asyncio.gather(*tasks)
        
        # Step 3: Parse and filter valid payloads
        parsed_records = []
        for res in raw_results:
            if res:
                parsed = parse_vacancy_payload(res)
                parsed_records.append(parsed)
                
        logger.info(f"Successfully parsed {len(parsed_records)} detailed vacancy payloads.")
        
        # Step 4: Persist raw snapshots
        save_raw_csv(parsed_records, output_csv)
        save_raw_sqlite(parsed_records, output_db)
        logger.info("Data collection finished successfully.")
        return parsed_records


if __name__ == "__main__":
    parser = ArgumentParser(description="Asynchronous HH.ru IT Vacancy Collector")
    parser.add_argument("--query", type=str, default="Python OR Go OR Backend OR Developer", help="Search query")
    parser.add_argument("--count", type=int, default=1100, help="Target minimum count")
    parser.add_argument("--concurrency", type=int, default=8, help="Async concurrency limit")
    parser.add_argument("--csv", type=str, default="hh_vacancies_raw_snapshot.csv", help="CSV destination")
    parser.add_argument("--db", type=str, default="hh_vacancies_raw.db", help="SQLite destination")
    args = parser.parse_args()
    
    asyncio.run(run_collector(
        query_text=args.query,
        target_count=args.count,
        concurrency_limit=args.concurrency,
        output_csv=args.csv,
        output_db=args.db
    ))
