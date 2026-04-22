"""One-shot corpus inspector for ground-truth authoring.

Dumps a compact, human-readable summary of the NCSL corpus to stdout so the
author can see bill IDs, titles, statuses, topics, and a 1-line summary per
bill without opening the 100+ MB JSONL. Not part of the production code
path; safe to delete afterwards.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    corpus_path = Path("data/ncsl/us_ai_legislation_ncsl_text.jsonl")
    bills = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()]

    mode = sys.argv[1] if len(sys.argv) > 1 else "overview"

    if mode == "overview":
        print(f"Total bills: {len(bills)}")
        print(f"Years: {Counter(b['year'] for b in bills)}")
        print("Statuses (top 15):")
        for s, n in Counter(b.get("status", "") for b in bills).most_common(15):
            print(f"  {n:4d}  {s}")
        states = sorted({b["state"] for b in bills})
        print(f"Distinct states: {len(states)}")
        enacted = [b for b in bills if b.get("status", "").lower().startswith("enacted")]
        print(f"Enacted total: {len(enacted)}")
        print(f"Enacted 2025: {sum(1 for b in enacted if b['year']==2025)}")
        print(f"Enacted 2024: {sum(1 for b in enacted if b['year']==2024)}")
        print(f"Enacted 2023: {sum(1 for b in enacted if b['year']==2023)}")
        print("Enacted 2025 by state:")
        for s, n in Counter(b["state"] for b in enacted if b["year"] == 2025).most_common(30):
            print(f"  {n:3d}  {s}")

    elif mode == "topics":
        # Topic is a comma/slash-separated string per bill. Split and count.
        tc = Counter()
        for b in bills:
            raw = b.get("topics", "") or ""
            for t in [x.strip() for x in raw.replace("/", ",").split(",")]:
                if t:
                    tc[t] += 1
        for t, n in tc.most_common(40):
            print(f"  {n:4d}  {t}")
        print(f"  (distinct topics: {len(tc)})")

    elif mode == "state":
        target_state = sys.argv[2]
        want_year = int(sys.argv[3]) if len(sys.argv) > 3 else None
        enacted_only = len(sys.argv) > 4 and sys.argv[4] == "enacted"
        for b in bills:
            if b["state"].lower() != target_state.lower():
                continue
            if want_year and b["year"] != want_year:
                continue
            if enacted_only and not b.get("status", "").lower().startswith("enacted"):
                continue
            summary_snippet = (b.get("summary", "") or "")[:140].replace("\n", " ")
            print(
                f"{b['year']}  {b['bill_id']:20s}  {b.get('status','')[:30]:30s}  "
                f"[{b.get('topics','')[:40]}]  {b.get('title','')[:70]}"
            )
            if summary_snippet:
                print(f"    summary: {summary_snippet}")

    elif mode == "bill":
        # Look up one specific bill by (year, state, bill_id substring).
        year = int(sys.argv[2])
        state = sys.argv[3]
        bill_key = sys.argv[4]
        for b in bills:
            if b["year"] != year:
                continue
            if b["state"].lower() != state.lower():
                continue
            if bill_key.replace(" ", "") not in b["bill_id"].replace(" ", ""):
                continue
            print(json.dumps({
                "year": b["year"],
                "state": b["state"],
                "bill_id": b["bill_id"],
                "status": b.get("status"),
                "topics": b.get("topics"),
                "title": b.get("title"),
                "summary": (b.get("summary") or "")[:800],
            }, indent=2))
            print("---")

    elif mode == "topic":
        # Bills whose comma-separated topics contain the given substring.
        topic_key = sys.argv[2].lower()
        year_filter = int(sys.argv[3]) if len(sys.argv) > 3 else None
        enacted_only = len(sys.argv) > 4 and sys.argv[4] == "enacted"
        for b in bills:
            if year_filter and b["year"] != year_filter:
                continue
            if enacted_only and not b.get("status", "").lower().startswith("enacted"):
                continue
            if topic_key not in (b.get("topics", "") or "").lower():
                continue
            print(
                f"{b['year']}  {b['state'][:15]:15s}  {b['bill_id']:20s}  "
                f"{b.get('status','')[:25]:25s}  [{b.get('topics','')[:45]}]  {b.get('title','')[:60]}"
            )

    elif mode == "titlesearch":
        # Full substring search on title + summary for questions.
        needle = sys.argv[2].lower()
        year_filter = int(sys.argv[3]) if len(sys.argv) > 3 else None
        enacted_only = len(sys.argv) > 4 and sys.argv[4] == "enacted"
        for b in bills:
            if year_filter and b["year"] != year_filter:
                continue
            if enacted_only and not b.get("status", "").lower().startswith("enacted"):
                continue
            hay = f"{b.get('title','')} || {b.get('summary','')}".lower()
            if needle not in hay:
                continue
            print(
                f"{b['year']}  {b['state'][:15]:15s}  {b['bill_id']:20s}  "
                f"{b.get('status','')[:25]:25s}  {b.get('title','')[:70]}"
            )

    else:
        print(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
