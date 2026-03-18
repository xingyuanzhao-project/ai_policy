"""Corpus loading for canonical raw bill records.

- Loads the real bill corpus from CSV or JSONL.
- Normalizes heterogeneous row payloads into canonical `BillRecord` objects.
- Does not derive chunks, call the LLM, or persist downstream artifacts.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ..schemas.artifacts import BillRecord
from ..schemas.validation import validate_bill_record


class CorpusStore:
    """Load raw bill metadata and bill text from the project corpus.

    The real project corpus exists in both CSV and JSONL forms. The store reads
    whichever format the runtime config points to and normalizes both into the
    same canonical `BillRecord` contract.
    """

    def __init__(self, corpus_path: Path) -> None:
        """Initialize the corpus store.

        Args:
            corpus_path (Path): Filesystem path to the bill corpus in CSV or
                JSONL format.
        """

        self._corpus_path = corpus_path
        self._records_by_bill_id: dict[str, BillRecord] = {}

    def load(self) -> list[BillRecord]:
        """Load the configured corpus file into canonical ``BillRecord`` objects.

        Returns:
            list[BillRecord]: Ordered list of parsed bill records from the
                configured corpus.
        """

        if self._corpus_path.suffix.lower() == ".jsonl":
            records = self._load_jsonl()
        else:
            records = self._load_csv()

        self._records_by_bill_id = {record.bill_id: record for record in records}
        return records

    def get_bill(self, bill_id: str) -> BillRecord:
        """Return one loaded bill by id.

        Args:
            bill_id (str): Bill identifier to retrieve from the loaded corpus.

        Returns:
            BillRecord: Matching canonical bill record.

        Raises:
            KeyError: If the requested bill id is not present in the loaded
                corpus.
        """

        if bill_id not in self._records_by_bill_id:
            raise KeyError(f"Bill '{bill_id}' is not loaded in the corpus store")
        return self._records_by_bill_id[bill_id]

    def _load_csv(self) -> list[BillRecord]:
        """Load the corpus from CSV.

        Returns:
            list[BillRecord]: Ordered bill records parsed from the CSV corpus.
        """

        records: list[BillRecord] = []
        with open(self._corpus_path, encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                records.append(_row_to_bill_record(row))
        return records

    def _load_jsonl(self) -> list[BillRecord]:
        """Load the corpus from JSONL.

        Returns:
            list[BillRecord]: Ordered bill records parsed from the JSONL
                corpus.

        Raises:
            ValueError: If a JSONL row does not decode to a JSON object.
        """

        records: list[BillRecord] = []
        with open(self._corpus_path, encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"JSONL line {line_number} must decode to a JSON object"
                    )
                records.append(_row_to_bill_record(payload))
        return records


def _row_to_bill_record(row: dict[str, object]) -> BillRecord:
    """Normalize one CSV or JSONL payload into the canonical bill schema.

    Args:
        row (dict[str, object]): Source row decoded from CSV or JSONL.

    Returns:
        BillRecord: Canonical ``BillRecord`` produced from the source row.
    """

    record = BillRecord(
        bill_id=_normalize_text(row.get("bill_id")),
        state=_normalize_text(row.get("state")),
        text=_normalize_text(
            row.get("text"),
            preserve_none_as_empty=True,
            strip_whitespace=False,
        ),
        bill_url=_normalize_text(row.get("bill_url")),
        title=_normalize_text(row.get("title")),
        status=_normalize_text(row.get("status")),
        date_of_last_action=_normalize_text(row.get("date_of_last_action")),
        author=_normalize_text(row.get("author")),
        topics=_normalize_text(row.get("topics")),
        summary=_normalize_text(row.get("summary")),
        history=_normalize_text(
            row.get("history"),
            preserve_none_as_empty=True,
            strip_whitespace=False,
        ),
    )
    validate_bill_record(record)
    return record


def _normalize_text(
    value: object,
    preserve_none_as_empty: bool = False,
    strip_whitespace: bool = True,
) -> str:
    """Normalize CSV/JSONL field values into safe string payloads.

    Args:
        value (object): Raw field value from the source corpus.
        preserve_none_as_empty (bool): Whether ``None`` should become the empty
            string.
        strip_whitespace (bool): Whether string values should be trimmed.

    Returns:
        str: Normalized string payload safe to place in the canonical bill
            schema.
    """

    if value is None:
        return "" if preserve_none_as_empty else ""
    if isinstance(value, str):
        return value.strip() if strip_whitespace else value
    return str(value)

