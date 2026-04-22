"""
Scrape NCSL AI legislation tables (2023/2024), fetch bill HTML from StateNet,
extract text, reorganize 2025 bills, and produce a unified 2023-2025 CSV.

Usage:
    python scripts/scrape_ncsl_2023_2024.py [--step STEP]

Steps (run all by default):
    1  Scrape metadata tables from NCSL 2023/2024 pages
    2  Fetch bill HTML from statenet.com for 2023/2024
    3  Move existing bills/ to bills_2025/
    4  Extract text from 2023/2024 bill HTML
    5  Build unified 2023-2025 text CSV
"""

import argparse
import json
import os
import re
import shutil
import time
import unicodedata
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ncsl"

NCSL_URLS = {
    2023: "https://www.ncsl.org/technology-and-communication/artificial-intelligence-2023-legislation",
    2024: "https://www.ncsl.org/technology-and-communication/artificial-intelligence-2024-legislation",
}

META_COLUMNS = [
    "state", "bill_id", "bill_url", "title", "status",
    "date_of_last_action", "author", "topics", "summary", "history",
]

REQUEST_TIMEOUT = 30
FETCH_DELAY_SECONDS = 1
STATENET_BASE = "http://custom.statenet.com"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_state_abbr_from_url(url: str) -> str:
    """Extract 2-letter state abbreviation from statenet bill URL.

    URL pattern: ...id=ID:bill:{STATE_ABBR}{YEAR}000{BILL}...
    Example: ID:bill:AZ2023000H2482 -> AZ
    """
    match = re.search(r"ID:bill:([A-Z]{2})\d", url)
    if match:
        return match.group(1)
    return ""


def _build_bill_id(state_abbr: str, raw_bill_number: str) -> str:
    """Build bill_id in the format used by the existing 2025 meta CSV.

    Existing format: 'AZ H  2482' — state abbr + space + chamber/type + padded number.
    The raw bill number from the table is e.g. 'H 2482'.
    We need to reconstruct the padded format: split into alpha prefix + numeric suffix,
    then left-pad the alpha part to 4 chars and the number to 5 chars within a total
    field width that matches the existing data.

    The existing 2025 data uses the format produced by StateNet's linkList, which is:
    '{STATE} {TYPE}{SPACES}{NUMBER}' where TYPE is left-justified in a 4-char field.
    Example: 'AL H    169' -> 'AL' + ' ' + 'H   ' + ' 169'
    """
    raw = raw_bill_number.strip()
    # Split into chamber/type prefix and number
    match = re.match(r"([A-Za-z]+)\s*(\d+)", raw)
    if not match:
        return f"{state_abbr} {raw}"

    chamber = match.group(1).upper()
    number = match.group(2)
    # Pad to match existing format: chamber left-justified in ~4 chars, number right-justified in ~5 chars
    bill_id = f"{state_abbr} {chamber:<4s}{number:>4s}"
    return bill_id


# ── Step 1: Scrape metadata tables ──────────────────────────────────────────

def scrape_table(url: str, year: int) -> pd.DataFrame:
    """Fetch an NCSL legislation page and parse the HTML table into a DataFrame."""
    print(f"\n{'='*60}")
    print(f"Step 1: Scraping {year} metadata from {url}")
    print(f"{'='*60}")

    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # 2024 table has id="ai2024table"; 2023 has no id — use first table
    table = soup.find("table", id="ai2024table") if year == 2024 else soup.find("table")
    if not table:
        raise RuntimeError(f"No <table> found on {year} page")

    rows = table.find_all("tr")
    print(f"  Found {len(rows)} rows (including header)")

    bills = []
    for row in rows[1:]:  # skip header row
        cells = row.find_all("td")
        if len(cells) < 6:
            continue

        state_name = cells[0].get_text(strip=True)
        bill_number_text = cells[1].get_text(strip=True)

        # Skip states with no legislation or unavailable sessions
        skip_patterns = ("none", "not available", "no 20")
        if not bill_number_text or any(bill_number_text.lower().startswith(p) for p in skip_patterns):
            continue

        # Extract bill link
        link = cells[1].find("a")
        bill_url = link.get("href", "") if link else ""
        state_abbr = _extract_state_abbr_from_url(bill_url)
        bill_id = _build_bill_id(state_abbr, bill_number_text) if state_abbr else bill_number_text

        bills.append({
            "state": state_name,
            "bill_id": bill_id,
            "bill_url": bill_url,
            "title": cells[2].get_text(strip=True),
            "status": cells[3].get_text(strip=True),
            "date_of_last_action": "",
            "author": "",
            "topics": cells[5].get_text(strip=True),
            "summary": cells[4].get_text(strip=True),
            "history": "",
        })

    df = pd.DataFrame(bills, columns=META_COLUMNS)
    df["year"] = year
    print(f"  Extracted {len(df)} bills")

    # Save
    out_path = DATA_DIR / f"us_ai_legislation_ncsl_meta_{year}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  Saved to {out_path}")
    return df


# ── Step 2: Fetch bill HTML ─────────────────────────────────────────────────

VERSION_LISTING_MARKER = "FULL TEXT OF ALL AVAILABLE VERSIONS"


def _resolve_version_listing(html: str) -> str:
    """If html is a version listing page, follow the last version link to get actual text.

    StateNet URLs without `&mode=current_text` return a page listing all versions
    (Introduced, Engrossed, Enrolled, Enacted, etc.) with relative links. We pick
    the last version (most recent / final) and fetch its text page.
    """
    if VERSION_LISTING_MARKER not in html:
        return html

    soup = BeautifulSoup(html, "html.parser")
    version_links = [
        a for a in soup.find_all("a")
        if a.get("href") and "mode=show_text" in a.get("href", "")
    ]
    if not version_links:
        return html

    # Last link is typically the most advanced version (Enacted > Enrolled > Engrossed > Introduced)
    last_link = version_links[-1]
    version_url = STATENET_BASE + last_link["href"]

    resp = requests.get(version_url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.text


def fetch_bills(df: pd.DataFrame, year: int) -> None:
    """Fetch bill HTML from statenet.com for each bill in the metadata DataFrame."""
    bills_dir = DATA_DIR / f"bills_{year}"
    bills_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = DATA_DIR / f"ncsl_bills_{year}.jsonl"

    print(f"\n{'='*60}")
    print(f"Step 2: Fetching {year} bill HTML ({len(df)} bills)")
    print(f"  Output dir: {bills_dir}")
    print(f"  JSONL: {jsonl_path}")
    print(f"{'='*60}")

    skipped = 0
    fetched = 0
    resolved = 0
    errors = 0

    with open(jsonl_path, "w", encoding="utf-8") as f_jsonl:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Fetching {year} bills"):
            bill_id = row["bill_id"]
            bill_url = row["bill_url"]
            html_path = bills_dir / f"{bill_id}.html"

            # Idempotency: skip if already fetched
            if html_path.exists():
                html = html_path.read_text(encoding="utf-8")
                record = {"bill_id": bill_id, "html": html}
                f_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                skipped += 1
                continue

            try:
                resp = requests.get(bill_url, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                html = resp.text

                # If page is a version listing, follow the last version link
                if VERSION_LISTING_MARKER in html:
                    html = _resolve_version_listing(html)
                    resolved += 1
                    time.sleep(FETCH_DELAY_SECONDS)

                fetched += 1
            except Exception as e:
                html = f"ERROR: {e}"
                errors += 1

            html_path.write_text(html, encoding="utf-8")
            record = {"bill_id": bill_id, "html": html}
            f_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

            time.sleep(FETCH_DELAY_SECONDS)

    print(f"  Done: fetched={fetched}, resolved={resolved}, skipped={skipped}, errors={errors}")


# ── Step 3: Move 2025 bills ─────────────────────────────────────────────────

def move_2025_bills() -> None:
    """Move data/ncsl/bills/ contents to data/ncsl/bills_2025/."""
    src_dir = DATA_DIR / "bills"
    dst_dir = DATA_DIR / "bills_2025"

    print(f"\n{'='*60}")
    print(f"Step 3: Moving 2025 bills from {src_dir} to {dst_dir}")
    print(f"{'='*60}")

    if not src_dir.exists():
        print("  Source directory does not exist — nothing to move.")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)

    files = list(src_dir.glob("*.html"))
    print(f"  Found {len(files)} HTML files to move")

    moved = 0
    for f in tqdm(files, desc="Moving files"):
        dst_path = dst_dir / f.name
        if not dst_path.exists():
            shutil.move(str(f), str(dst_path))
            moved += 1

    print(f"  Moved {moved} files (skipped {len(files) - moved} already present)")

    # Remove source dir if empty
    remaining = list(src_dir.iterdir())
    if not remaining:
        src_dir.rmdir()
        print(f"  Removed empty directory {src_dir}")


# ── Step 4: Extract text ────────────────────────────────────────────────────

NO_TEXT_MSG = "no text versions currently associated"


def extract_text(html: str) -> tuple:
    """Extract bill text from StateNet HTML. Returns (text, status)."""
    if html.startswith("ERROR:"):
        return None, "fetch_error"
    if NO_TEXT_MSG in html.lower():
        return None, "empty"

    soup = BeautifulSoup(html, "html.parser")
    div_text = soup.find("div", class_="text")
    if not div_text:
        return None, "no_div_text"

    raw = div_text.get_text(separator="\n", strip=True)
    raw = unicodedata.normalize("NFKC", raw)
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip(), "ok"


def extract_texts(jsonl_path: Path) -> pd.DataFrame:
    """Load JSONL and extract text from each bill's HTML."""
    print(f"\n  Extracting text from {jsonl_path}")

    with open(jsonl_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    print(f"  Loaded {len(records)} records")

    results = []
    for rec in tqdm(records, desc="Extracting text"):
        text, status = extract_text(rec["html"])
        results.append({
            "bill_id": rec["bill_id"],
            "text": text,
            "extraction_status": status,
        })

    df = pd.DataFrame(results)
    print(f"  Extraction status:\n{df['extraction_status'].value_counts().to_string()}")
    return df


# ── Step 5: Build unified CSV ───────────────────────────────────────────────

def build_unified_csv() -> None:
    """Merge 2023, 2024, and 2025 metadata+text into a single CSV."""
    print(f"\n{'='*60}")
    print("Step 5: Building unified 2023-2025 text CSV")
    print(f"{'='*60}")

    year_configs = [
        (2023, DATA_DIR / "us_ai_legislation_ncsl_meta_2023.csv", DATA_DIR / "ncsl_bills_2023.jsonl"),
        (2024, DATA_DIR / "us_ai_legislation_ncsl_meta_2024.csv", DATA_DIR / "ncsl_bills_2024.jsonl"),
        (2025, DATA_DIR / "us_ai_legislation_ncsl_meta_2025.csv",  DATA_DIR / "ncsl_bills_2025.jsonl"),
    ]

    dfs = []
    for year, meta_path, jsonl_path in year_configs:
        print(f"\n  Processing {year}...")

        if not meta_path.exists():
            print(f"    Meta CSV not found: {meta_path} — skipping")
            continue

        df_meta = pd.read_csv(meta_path)
        df_meta["year"] = year
        print(f"    Meta: {len(df_meta)} rows")

        if jsonl_path.exists():
            df_text = extract_texts(jsonl_path)
            df_merged = df_meta.merge(df_text, on="bill_id", how="left")
        else:
            print(f"    JSONL not found: {jsonl_path} — text columns will be empty")
            df_merged = df_meta.copy()
            df_merged["text"] = None
            df_merged["extraction_status"] = "no_jsonl"

        # Compute text metrics
        df_merged["text_len"] = df_merged["text"].apply(
            lambda t: len(t) if isinstance(t, str) else 0
        )
        df_merged["word_count"] = df_merged["text"].apply(
            lambda t: len(t.split()) if isinstance(t, str) else 0
        )

        dfs.append(df_merged)
        print(f"    Merged: {len(df_merged)} rows")

    df_all = pd.concat(dfs, ignore_index=True)

    # Column ordering: meta columns + year + text columns
    final_columns = META_COLUMNS + ["year", "text", "text_len", "word_count", "extraction_status"]
    for col in final_columns:
        if col not in df_all.columns:
            df_all[col] = ""
    df_all = df_all[final_columns]

    out_csv = DATA_DIR / "us_ai_legislation_ncsl_text.csv"
    df_all.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # Also generate merged text JSONL
    out_jsonl = DATA_DIR / "us_ai_legislation_ncsl_text.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for _, row in df_all.iterrows():
            rec = {}
            for col in df_all.columns:
                val = row[col]
                if pd.isna(val):
                    rec[col] = None if col == "text" else ""
                else:
                    rec[col] = int(val) if col in ("year", "text_len", "word_count") else str(val)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Also generate merged meta CSV
    out_meta = DATA_DIR / "us_ai_legislation_ncsl_meta.csv"
    meta_dfs = []
    for year, meta_path, _ in year_configs:
        if meta_path.exists():
            m = pd.read_csv(meta_path)
            if "year" not in m.columns:
                m["year"] = year
            meta_dfs.append(m)
    if meta_dfs:
        pd.concat(meta_dfs, ignore_index=True).to_csv(out_meta, index=False, encoding="utf-8-sig")

    # Also generate merged bills JSONL
    out_bills = DATA_DIR / "ncsl_bills.jsonl"
    with open(out_bills, "w", encoding="utf-8") as f:
        for year, _, jsonl_path in year_configs:
            if jsonl_path.exists():
                with open(jsonl_path, encoding="utf-8") as src:
                    for line in src:
                        rec = json.loads(line)
                        rec["year"] = year
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n  Unified CSV saved to {out_csv}")
    print(f"  Unified JSONL saved to {out_jsonl}")
    print(f"  Merged meta saved to {out_meta}")
    print(f"  Merged bills JSONL saved to {out_bills}")
    print(f"  Total rows: {len(df_all)}")
    print(f"  By year:\n{df_all['year'].value_counts().sort_index().to_string()}")
    print(f"  Extraction status:\n{df_all['extraction_status'].value_counts().to_string()}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NCSL 2023-2025 AI legislation scraping pipeline")
    parser.add_argument(
        "--step", type=int, choices=[1, 2, 3, 4, 5], default=None,
        help="Run a specific step only (default: run all steps 1-5)",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    steps = [args.step] if args.step else [1, 2, 3, 4, 5]

    meta_2023 = None
    meta_2024 = None

    if 1 in steps:
        meta_2023 = scrape_table(NCSL_URLS[2023], 2023)
        meta_2024 = scrape_table(NCSL_URLS[2024], 2024)

    if 2 in steps:
        if meta_2023 is None:
            meta_path = DATA_DIR / "us_ai_legislation_ncsl_meta_2023.csv"
            if meta_path.exists():
                meta_2023 = pd.read_csv(meta_path)
            else:
                raise FileNotFoundError(f"Run step 1 first: {meta_path} not found")
        if meta_2024 is None:
            meta_path = DATA_DIR / "us_ai_legislation_ncsl_meta_2024.csv"
            if meta_path.exists():
                meta_2024 = pd.read_csv(meta_path)
            else:
                raise FileNotFoundError(f"Run step 1 first: {meta_path} not found")

        fetch_bills(meta_2023, 2023)
        fetch_bills(meta_2024, 2024)

    if 3 in steps:
        move_2025_bills()

    if 4 in steps:
        # Step 4 is integrated into step 5 (extract_texts called during build_unified_csv)
        print("\n  Step 4 extraction is performed as part of step 5.")

    if 5 in steps:
        build_unified_csv()

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
