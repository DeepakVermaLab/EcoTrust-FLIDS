#!/usr/bin/env python3
"""Verify selected journals against current Scopus and WoS source lists."""

from __future__ import annotations

import csv
import re
import tempfile
import urllib.request
from pathlib import Path

from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "literature_manifest.csv"
OUTPUT = ROOT / "indexing_verification.csv"

SCOPUS_INFORMATION_URL = "https://www.elsevier.com/products/scopus/content"
SCOPUS_SOURCE_URL = (
    "https://downloads.ctfassets.net/o78em1y1w4i4/7xtaTxNiNcWRTeZkV86eNy/"
    "e6b17b29beaf8d18175591a4e8dee0bf/ext_list_Sep_2026.xlsx"
)
WOS_OFFICIAL_URL = "https://mjl.clarivate.com/collection-list-downloads"
WOS_SNAPSHOT_BASE = "https://raw.githubusercontent.com/wosaide/wosaide-journal-lists/main/WOS"


def normalized_issn(value: object) -> str:
    return re.sub(r"[^0-9X]", "", str(value or "").upper())


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "EcoTrust-literature-audit/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        destination.write_bytes(response.read())


def year_is_covered(year: int, coverage: str) -> bool:
    for item in re.split(r"[;,]", coverage or ""):
        item = item.strip()
        if not item:
            continue
        match = re.fullmatch(r"(\d{4})-(\d{4})", item)
        if match and int(match.group(1)) <= year <= int(match.group(2)):
            return True
        if item.isdigit() and int(item) == year:
            return True
    return False


def main() -> None:
    manifest = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    if len(manifest) != 48:
        raise RuntimeError(f"expected 48 manifest rows, found {len(manifest)}")

    with tempfile.TemporaryDirectory(prefix="ecotrust-indexing-") as temp_name:
        temp = Path(temp_name)
        scopus_file = temp / "ext_list_Sep_2026.xlsx"
        download(SCOPUS_SOURCE_URL, scopus_file)
        scopus_book = load_workbook(scopus_file, read_only=True, data_only=True)
        scopus_sheet = scopus_book["Scopus Sources Sep. 2026"]
        scopus_by_issn: dict[str, list[tuple]] = {}
        for source in scopus_sheet.iter_rows(min_row=2, values_only=True):
            for position in (2, 3):
                key = normalized_issn(source[position])
                if key:
                    scopus_by_issn.setdefault(key, []).append(source)

        wos_by_issn: dict[str, list[tuple[str, dict[str, str]]]] = {}
        for collection in ("SCIE", "ESCI"):
            target = temp / f"{collection}.csv"
            download(f"{WOS_SNAPSHOT_BASE}/{collection}.csv", target)
            with target.open(encoding="utf-8-sig", newline="") as handle:
                for source in csv.DictReader(handle):
                    for field in ("ISSN", "eISSN"):
                        key = normalized_issn(source.get(field))
                        if key:
                            wos_by_issn.setdefault(key, []).append((collection, source))

        output_rows = []
        for paper in manifest:
            issns = [normalized_issn(value) for value in paper["issn"].split(";") if value]
            scopus_hits: dict[object, tuple] = {}
            for issn in issns:
                for hit in scopus_by_issn.get(issn, []):
                    scopus_hits[hit[0]] = hit
            if not scopus_hits:
                raise RuntimeError(f"no Scopus source match for {paper['id']} {paper['journal']}")
            scopus = next(iter(scopus_hits.values()))
            covered = year_is_covered(int(paper["year"]), str(scopus[5] or ""))
            if not covered:
                raise RuntimeError(
                    f"Scopus coverage does not include article year for {paper['id']}: "
                    f"{paper['year']} not in {scopus[5]}"
                )

            wos_hits: list[tuple[str, dict[str, str]]] = []
            for issn in issns:
                wos_hits.extend(wos_by_issn.get(issn, []))
            collections = sorted({collection for collection, _ in wos_hits})
            wos_title = wos_hits[0][1]["Journal title"] if wos_hits else ""

            output_rows.append(
                {
                    "id": paper["id"],
                    "doi": paper["doi"],
                    "article_year": paper["year"],
                    "journal": paper["journal"],
                    "issn": paper["issn"],
                    "scopus_verified": "yes",
                    "scopus_source_id": scopus[0],
                    "scopus_source_title": scopus[1],
                    "scopus_status_sep_2026": scopus[4],
                    "scopus_coverage": scopus[5],
                    "article_year_in_scopus_coverage": "yes",
                    "wos_collection_snapshot": "+".join(collections) if collections else "not_matched",
                    "wos_source_title": wos_title,
                    "scopus_information_url": SCOPUS_INFORMATION_URL,
                    "scopus_source_list_url": SCOPUS_SOURCE_URL,
                    "wos_official_collection_url": WOS_OFFICIAL_URL,
                    "wos_snapshot_provenance": f"{WOS_SNAPSHOT_BASE}/README.md",
                }
            )

    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Scopus verified: {sum(row['scopus_verified'] == 'yes' for row in output_rows)}/48")
    print(
        "WoS snapshot: "
        + ", ".join(
            f"{name}={sum(row['wos_collection_snapshot'] == name for row in output_rows)}"
            for name in ("SCIE", "ESCI", "not_matched")
        )
    )


if __name__ == "__main__":
    main()
