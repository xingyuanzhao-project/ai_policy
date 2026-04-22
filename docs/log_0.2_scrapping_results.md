# Log 0.2 — NCSL AI Legislation Scraping (2023–2025)

**Date**: 2025-04-15
**Agent session**: [NCSL scraping pipeline](a8125f95-a4c6-430f-a5b1-bfd52af28f5e)

---

## What the agent did

1. **Explored existing codebase** — read `op1`–`op6` notebooks, `data/ncsl/` directory structure, existing CSV/JSONL formats, and `requirements.txt` to understand the established pipeline.
2. **Inspected NCSL webpage HTML** — fetched 2023 and 2024 NCSL AI legislation pages, confirmed they use `<table>` elements (not div-based layouts), identified column structure and bill hyperlink format pointing to `custom.statenet.com`.
3. **Created `scripts/scrape_ncsl_2023_2024.py`** — a 5-step CLI pipeline:
   - Step 1: `scrape_table()` — parse NCSL HTML tables into per-year metadata CSVs, extracting state, bill number, title, status, summary, category, and bill URL. Filters out rows with "None", "Not available", or "No 20XX legislative session".
   - Step 2: `fetch_bills()` — fetch raw HTML for each bill from StateNet, save to `data/ncsl/bills_{year}/`, append to per-year JSONL. Uses `tqdm` + `time.sleep` for rate limiting. Skips already-fetched files.
   - Step 3: `move_2025_bills()` — relocate existing `data/ncsl/bills/` files to `data/ncsl/bills_2025/`.
   - Step 4: `extract_texts()` — reuses `extract_text()` logic from `op4_create_text.ipynb` to pull plain text from `<div class="text">` in saved HTML.
   - Step 5: `build_unified_csv()` — merge metadata + text across all years into unified output files.
4. **Ran the scraping pipeline** — fetched 135 bills for 2023, 480 for 2024. Moved 1,262 existing 2025 bills.
5. **Debugged 2024 `no_div_text` failures** — discovered that 2024 bill URLs lack `&mode=current_text`, causing StateNet to return a version-listing page instead of the bill text. Implemented `_resolve_version_listing()` to detect these pages, find the last version link (Enacted > Enrolled > Engrossed > Introduced), and follow it to get actual text. Re-fetched affected files.
6. **Debugged metadata duplicates** — found that some NCSL table rows with "Not available" or "No 20XX legislative session" were being parsed as bills. Added explicit skip logic.
7. **Reorganized `data/ncsl/` file naming** — renamed 2025-only unsuffixed files to `_2025` suffix, created new unsuffixed files as merged 2023–2025 data. Convention: unsuffixed = merged, `_YYYY` = per-year.
8. **Added `year` column/field** to all CSVs and JSONLs (both per-year and merged).
9. **Updated `scrape_ncsl_2023_2024.py`** to use the new naming convention (`_2025` suffixed sources, unsuffixed merged outputs).
10. **Verified code references** — confirmed all consumer code in `src/`, `scripts/`, and `settings/` reads unsuffixed paths that now resolve to merged 2023–2025 data. Confirmed old `data/ncsl/bills/` directory no longer exists.
11. **Added `beautifulsoup4>=4.12.0`** to `requirements.txt`.

---

## Final data layout

### `data/ncsl/` directory

| File / Directory | Content | Rows / Files | `year` field |
|---|---|---|---|
| `us_ai_legislation_ncsl_text.csv` | Merged 2023–2025 text + metadata | 1,879 | Yes |
| `us_ai_legislation_ncsl_text.jsonl` | Merged 2023–2025 text + metadata | 1,879 | Yes |
| `us_ai_legislation_ncsl_meta.csv` | Merged 2023–2025 metadata only | 1,877 | Yes |
| `ncsl_bills.jsonl` | Merged 2023–2025 raw bill HTML | 1,877 | Yes |
| `us_ai_legislation_ncsl_meta_2023.csv` | 2023 metadata | 135 | Yes |
| `us_ai_legislation_ncsl_meta_2024.csv` | 2024 metadata | 480 | Yes |
| `us_ai_legislation_ncsl_meta_2025.csv` | 2025 metadata | 1,262 | Yes |
| `ncsl_bills_2023.jsonl` | 2023 raw bill HTML | 135 | Yes |
| `ncsl_bills_2024.jsonl` | 2024 raw bill HTML | 480 | Yes |
| `ncsl_bills_2025.jsonl` | 2025 raw bill HTML | 1,262 | Yes |
| `us_ai_legislation_ncsl_text_2025.csv` | 2025-only text + metadata | 1,262 | Yes |
| `us_ai_legislation_ncsl_text_2025.jsonl` | 2025-only text + metadata | 1,262 | Yes |
| `bills_2023/` | HTML files | 135 | — |
| `bills_2024/` | HTML files | 480 | — |
| `bills_2025/` | HTML files | 1,262 | — |

### Unified CSV columns

`state`, `bill_id`, `bill_url`, `title`, `status`, `date_of_last_action`, `author`, `topics`, `summary`, `history`, `year`, `text`, `text_len`, `word_count`, `extraction_status`

### Row count note

Meta CSV has 1,877 rows; text CSV has 1,879. The 2-row difference comes from `TX H 1` being listed twice in the NCSL 2023 table (duplicate in source data), producing 2 extra rows after the left merge with text.

---

## Extraction results

- **Total**: 1,879 bills
- **Extracted successfully (`ok`)**: 1,826 (97.2%)
- **Failed**: 53 (2.8%)

### Failures by status

| Status | Count | Meaning |
|---|---|---|
| `empty` | 37 | `<div class="text">` found but contained no extractable text |
| `no_div_text` | 15 | Page exists but no `<div class="text">` element found |
| `fetch_error` | 1 | No URL available on NCSL page |

### Failures by year

| Year | `empty` | `no_div_text` | `fetch_error` | Total |
|---|---|---|---|---|
| 2023 | 5 | 0 | 1 | 6 |
| 2024 | 0 | 15 | 0 | 15 |
| 2025 | 32 | 0 | 0 | 32 |

### Failures by jurisdiction

| Jurisdiction | Count |
|---|---|
| Hawaii | 22 |
| Puerto Rico | 21 |
| Massachusetts | 4 |
| Guam | 3 |
| Iowa | 1 |
| North Carolina | 1 |
| U.S. Virgin Islands | 1 |

Pattern: failures concentrate in U.S. territories (Puerto Rico, Guam, U.S. Virgin Islands) and Hawaii (resolutions: HCR, HR, SCR, SR). These jurisdictions' bill texts are either not hosted on StateNet or the pages return empty text containers. Massachusetts failures are HD/SD docket-prefixed bills.

---

## All bills with empty or missing text

### `fetch_error` (1)

| Year | Bill ID | URL |
|---|---|---|
| 2023 | H 259 | *(no link on NCSL page)* |

### `no_div_text` (15)

| Year | Bill ID | URL |
|---|---|---|
| 2024 | HI HCR 65 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:HI2023000HCR65&cuiq=93d84396-c63b-526a-b152-38b7f79b4cfd&client_md=e4f6fea4-27b4-5d41-b7d3-766fe52569f0 |
| 2024 | HI HCR 71 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:HI2023000HCR71&cuiq=93d84396-c63b-526a-b152-38b7f79b4cfd&client_md=e4f6fea4-27b4-5d41-b7d3-766fe52569f0 |
| 2024 | HI HR 48 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:HI2023000HR48&cuiq=93d84396-c63b-526a-b152-38b7f79b4cfd&client_md=e4f6fea4-27b4-5d41-b7d3-766fe52569f0 |
| 2024 | HI SCR 95 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:HI2023000SCR95&cuiq=93d84396-c63b-526a-b152-38b7f79b4cfd&client_md=e4f6fea4-27b4-5d41-b7d3-766fe52569f0 |
| 2024 | HI SR 81 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:HI2023000SR81&cuiq=93d84396-c63b-526a-b152-38b7f79b4cfd&client_md=e4f6fea4-27b4-5d41-b7d3-766fe52569f0 |
| 2024 | PR H 1961 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:PR2021000H1961&cuiq=93d84396-c63b-526a-b152-38b7f79b4cfd&client_md=e4f6fea4-27b4-5d41-b7d3-766fe52569f0 |
| 2024 | PR H 1962 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:PR2021000H1962&cuiq=93d84396-c63b-526a-b152-38b7f79b4cfd&client_md=e4f6fea4-27b4-5d41-b7d3-766fe52569f0 |
| 2024 | PR H 2027 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:PR2021000H2027&cuiq=93d84396-c63b-526a-b152-38b7f79b4cfd&client_md=e4f6fea4-27b4-5d41-b7d3-766fe52569f0 |
| 2024 | PR H 2111 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:PR2021000H2111&cuiq=93d84396-c63b-526a-b152-38b7f79b4cfd&client_md=e4f6fea4-27b4-5d41-b7d3-766fe52569f0 |
| 2024 | PR HR 1097 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:PR2021000HR1097&cuiq=93d84396-c63b-526a-b152-38b7f79b4cfd&client_md=e4f6fea4-27b4-5d41-b7d3-766fe52569f0 |
| 2024 | PR S 1179 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:PR2021000S1179&cuiq=93d84396-c63b-526a-b152-38b7f79b4cfd&client_md=e4f6fea4-27b4-5d41-b7d3-766fe52569f0 |
| 2024 | PR S 1440 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:PR2021000S1440&cuiq=93d84396-c63b-526a-b152-38b7f79b4cfd&client_md=e4f6fea4-27b4-5d41-b7d3-766fe52569f0 |
| 2024 | PR S 1463 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:PR2021000S1463&cuiq=93d84396-c63b-526a-b152-38b7f79b4cfd&client_md=e4f6fea4-27b4-5d41-b7d3-766fe52569f0 |
| 2024 | PR SJR 412 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:PR2021000SJR412&cuiq=93d84396-c63b-526a-b152-38b7f79b4cfd&client_md=e4f6fea4-27b4-5d41-b7d3-766fe52569f0 |
| 2024 | VI B 131 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:VI2023000B131&cuiq=93d84396-c63b-526a-b152-38b7f79b4cfd&client_md=e4f6fea4-27b4-5d41-b7d3-766fe52569f0 |

### `empty` (37)

| Year | Bill ID | URL |
|---|---|---|
| 2023 | HI SCR 179 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:HI2023000SCR179&ciq=ncsl&client_md=3d892e51a8c29791a39d2783a1ec71da&mode=current_text |
| 2023 | HI SR 123 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:HI2023000SR123&ciq=ncsl&client_md=9505f0415a56b8239964a45d4c1e6ad5&mode=current_text |
| 2023 | PR S 1179 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:PR2021000S1179&ciq=ncsl&client_md=068a5627ecd9d21ce5e633a876281c0a&mode=current_text |
| 2023 | PR SJR 412 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:PR2021000SJR412&ciq=ncsl&client_md=16f516d905640c076ed892fdf0110a19&mode=current_text |
| 2023 | PR SR 684 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:PR2021000SR684&ciq=ncsl&client_md=2010c5cead5366a5469d77f1afdfc9c5&mode=current_text |
| 2025 | GU B 171 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:GU2025000B171&ciq=urn:user:PA196471263&client_md=5e240636eeda2a9f490cd3c33f9d3062&mode=current_text |
| 2025 | GU B 209 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:GU2025000B209&ciq=urn:user:PA196471263&client_md=af5d6f5358ba840ee93b8942278eb258&mode=current_text |
| 2025 | GU B 64 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:GU2025000B64&ciq=urn:user:PA196471263&client_md=34008bbf606876b3d67e8d82fffe4a5e&mode=current_text |
| 2025 | HI HCR 116 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:HI2025000HCR116&ciq=urn:user:PA196471263&client_md=757812a55dd848ef45fb1fb11621eca3&mode=current_text |
| 2025 | HI HCR 156 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:HI2025000HCR156&ciq=urn:user:PA196471263&client_md=2c47b21a0d4e142c08bfa75146b0b716&mode=current_text |
| 2025 | HI HCR 175 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:HI2025000HCR175&ciq=urn:user:PA196471263&client_md=e5cfa223cdac3e47356f7d16d44065cf&mode=current_text |
| 2025 | HI HCR 202 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:HI2025000HCR202&ciq=urn:user:PA196471263&client_md=addde64e248ea83b4d318861386c2e39&mode=current_text |
| 2025 | HI HR 112 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:HI2025000HR112&ciq=urn:user:PA196471263&client_md=8025632a77a7db1712bf0cc150e3d7b5&mode=current_text |
| 2025 | HI HR 151 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:HI2025000HR151&ciq=urn:user:PA196471263&client_md=d5c4d1c6d439e820cc6523e33497b45e&mode=current_text |
| 2025 | HI HR 171 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:HI2025000HR171&ciq=urn:user:PA196471263&client_md=d8ca63a4522903e651b9cdc31511c8b2&mode=current_text |
| 2025 | HI HR 194 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:HI2025000HR194&ciq=urn:user:PA196471263&client_md=b8aabf2b36e2577b19c6fb2f4bcf6017&mode=current_text |
| 2025 | HI SCR 35 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:HI2025000SCR35&ciq=urn:user:PA196471263&client_md=31e432ec5c5cfc14679613e57f6c7648&mode=current_text |
| 2025 | HI SCR 43 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:HI2025000SCR43&ciq=urn:user:PA196471263&client_md=156d5f5ba16e69dfdb2582cde6048000&mode=current_text |
| 2025 | HI SCR 45 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:HI2025000SCR45&ciq=urn:user:PA196471263&client_md=ba69c7890e55fca4bbac9f3caf5b08e8&mode=current_text |
| 2025 | HI SCR 78 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:HI2025000SCR78&ciq=urn:user:PA196471263&client_md=cfbfa3ea24e059a33867cd5a18dfcb5f&mode=current_text |
| 2025 | HI SR 26 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:HI2025000SR26&ciq=urn:user:PA196471263&client_md=3224fb3a1d08ab1370eec27463e8e221&mode=current_text |
| 2025 | HI SR 28 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:HI2025000SR28&ciq=urn:user:PA196471263&client_md=7a48e9fb187b825420a5ee7f3e61f498&mode=current_text |
| 2025 | HI SR 61 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:HI2025000SR61&ciq=urn:user:PA196471263&client_md=185d5b6ba4dc94a7211f7089798ae1c0&mode=current_text |
| 2025 | IA D 1398 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:IA2025000D1398&ciq=urn:user:PA196471263&client_md=70fcde5c8cd3842520e0cea8e400a8fd&mode=current_text |
| 2025 | MA HD 3373 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:MA2025000HD3373&ciq=urn:user:PA196471263&client_md=58317a6bb8ead9bb943d7f3987a6200d&mode=current_text |
| 2025 | MA HD 396 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:MA2025000HD396&ciq=urn:user:PA196471263&client_md=03367e05363d7081a8c1f5804277bd7d&mode=current_text |
| 2025 | MA HD 3986 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:MA2025000HD3986&ciq=urn:user:PA196471263&client_md=6fe42dfc40bef677a178dcb0d19814f2&mode=current_text |
| 2025 | MA SD 873 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:MA2025000SD873&ciq=urn:user:PA196471263&client_md=84b62d3a3cc1b613f088a72d9063bfdc&mode=current_text |
| 2025 | PR H 427 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:PR2025000H427&ciq=urn:user:PA196471263&client_md=dadb225cc59f6d06e3c092f959014ffd&mode=current_text |
| 2025 | PR HJR 68 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:PR2025000HJR68&ciq=urn:user:PA196471263&client_md=78160e41d7020097924f25a9408c6768&mode=current_text |
| 2025 | PR HR 424 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:PR2025000HR424&ciq=urn:user:PA196471263&client_md=fc71187fd37139f9862ff09d10cac948&mode=current_text |
| 2025 | PR S 348 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:PR2025000S348&ciq=urn:user:PA196471263&client_md=7507608a270222b79c0eeb25e542856a&mode=current_text |
| 2025 | PR S 622 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:PR2025000S622&ciq=urn:user:PA196471263&client_md=767e35297b4e4a6668bbddf7d7cee70e&mode=current_text |
| 2025 | PR S 68 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:PR2025000S68&ciq=urn:user:PA196471263&client_md=0da7236e69b577ab8dc58e542008bcb0&mode=current_text |
| 2025 | PR S 731 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:PR2025000S731&ciq=urn:user:PA196471263&client_md=85ae1b481a8ca54c37a309ffe24671f7&mode=current_text |
| 2025 | PR SJR 1 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:PR2025000SJR1&ciq=urn:user:PA196471263&client_md=1d01962c66f8e13ee949a7ae87c5021c&mode=current_text |
| 2025 | PR SR 133 | http://custom.statenet.com/public/resources.cgi?id=ID:bill:PR2025000SR133&ciq=urn:user:PA196471263&client_md=0383ed9d905d5ea538bd06c00b2fd047&mode=current_text |

---

## Known data quality issues

1. **`TX H 1` duplicate** — listed twice in the NCSL 2023 table (source data issue). Produces 2 extra rows in text CSV after merge (1,879 text rows vs 1,877 meta rows).
2. **Cross-year bill overlap** — 71 bills appear on both 2023 and 2024 NCSL pages (bills that carry over between legislative sessions). These are kept as separate rows, one per year.
3. **2024 URL format** — 2024 NCSL page uses URLs without `&mode=current_text`, leading to version-listing pages on StateNet. The scraper resolves this by following the last version link (`_resolve_version_listing()`), but Hawaii resolutions and Puerto Rico bills often have no versions available, resulting in `no_div_text`.

---

## Files created or modified

| File | Action |
|---|---|
| `scripts/scrape_ncsl_2023_2024.py` | Created (scraping pipeline) |
| `requirements.txt` | Added `beautifulsoup4>=4.12.0` |
| `data/ncsl/bills_2023/` | Created (135 HTML files) |
| `data/ncsl/bills_2024/` | Created (480 HTML files) |
| `data/ncsl/bills_2025/` | Created (moved from `bills/`, 1,262 files) |
| `data/ncsl/us_ai_legislation_ncsl_meta_2023.csv` | Created |
| `data/ncsl/us_ai_legislation_ncsl_meta_2024.csv` | Created |
| `data/ncsl/ncsl_bills_2023.jsonl` | Created |
| `data/ncsl/ncsl_bills_2024.jsonl` | Created |
| `data/ncsl/ncsl_bills.jsonl` | Renamed 2025-only to `_2025`, recreated as merged 2023–2025 |
| `data/ncsl/us_ai_legislation_ncsl_meta.csv` | Renamed 2025-only to `_2025`, recreated as merged 2023–2025 |
| `data/ncsl/us_ai_legislation_ncsl_text.csv` | Renamed 2025-only to `_2025`, recreated as merged 2023–2025 |
| `data/ncsl/us_ai_legislation_ncsl_text.jsonl` | Renamed 2025-only to `_2025`, recreated as merged 2023–2025 |
| `data/ncsl/us_ai_legislation_ncsl_text_2023_2025.csv` | Created as intermediate, then deleted |
| `data/ncsl/bills/` | Deleted (contents moved to `bills_2025/`) |
