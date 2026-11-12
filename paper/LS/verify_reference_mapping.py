#!/usr/bin/env python3
"""Verify PDF-to-reference provenance and write reference_mapping.csv.

The local audit checks every BibTeX field against the saved Crossref record,
the manifest, and the downloaded PDF checksum. ``--live`` additionally checks
the current Crossref record and the doi.org redirect for every DOI.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MAPPING = ROOT / "reference_mapping.csv"
USER_AGENT = "EcoTrust-FLIDS-reference-verifier/1.0"


def clean(value: str) -> str:
    value = html.unescape(value or "")
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalized(value: str) -> str:
    value = unicodedata.normalize("NFKD", clean(value))
    value = value.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def parse_bibtex_entry(block: str) -> tuple[str, dict[str, str]]:
    head, body = block.split(",", 1)
    key = head.split("{", 1)[1].strip()
    body = body.rsplit("}", 1)[0]
    fields: dict[str, str] = {}
    index = 0
    while index < len(body):
        while index < len(body) and (body[index].isspace() or body[index] == ","):
            index += 1
        match = re.match(r"([A-Za-z][A-Za-z0-9_-]*)\s*=\s*", body[index:])
        if not match:
            break
        name = match.group(1).lower()
        index += match.end()
        if index < len(body) and body[index] == "{":
            depth = 1
            end = index + 1
            while end < len(body) and depth:
                if body[end] == "{":
                    depth += 1
                elif body[end] == "}":
                    depth -= 1
                end += 1
            value = body[index + 1 : end - 1]
            index = end
        elif index < len(body) and body[index] == '"':
            end = index + 1
            while end < len(body) and body[end] != '"':
                end += 1
            value = body[index + 1 : end]
            index = end + 1
        else:
            end = index
            while end < len(body) and body[end] != ",":
                end += 1
            value = body[index:end].strip()
            index = end
        fields[name] = value.strip()
    return key, fields


def bibtex_by_doi() -> dict[str, tuple[str, dict[str, str]]]:
    text = (ROOT / "ref.bib").read_text(encoding="utf-8").strip()
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    result: dict[str, tuple[str, dict[str, str]]] = {}
    for block in blocks:
        key, fields = parse_bibtex_entry(block)
        doi = fields.get("doi", "").lower()
        if not doi or doi in result:
            raise RuntimeError(f"missing or duplicate BibTeX DOI: {doi!r}")
        result[doi] = (key, fields)
    return result


def final_year(record: dict) -> int | None:
    for key in ("published-print", "published", "issued", "published-online"):
        parts = (record.get(key) or {}).get("date-parts") or []
        if parts and parts[0]:
            return int(parts[0][0])
    return None


def authors_text(record: dict) -> str:
    values = []
    for author in record.get("author") or []:
        family = clean(author.get("family", ""))
        given = clean(author.get("given", ""))
        values.append(", ".join(value for value in (family, given) if value))
    return "; ".join(values)


def publisher_url(record: dict, doi: str) -> str:
    resource = (record.get("resource") or {}).get("primary") or {}
    return resource.get("URL") or record.get("URL") or f"https://doi.org/{doi}"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


NO_REDIRECT = urllib.request.build_opener(NoRedirect)


def request(url: str, *, opener=None, attempts: int = 6) -> tuple[int, object, bytes]:
    last: Exception | None = None
    for attempt in range(attempts):
        req = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json, text/html"},
        )
        try:
            if opener is None:
                with urllib.request.urlopen(req, timeout=45) as response:
                    return response.status, response.headers, response.read()
            with opener.open(req, timeout=45) as response:
                return response.status, response.headers, response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in {301, 302, 303, 307, 308}:
                return exc.code, exc.headers, exc.read()
            last = exc
            if exc.code != 429 and exc.code < 500:
                raise
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else min(2**attempt, 20)
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
            time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"request failed after {attempts} attempts: {url}: {last}")


def live_record(doi: str) -> tuple[dict, int, str]:
    encoded = urllib.parse.quote(doi, safe="")
    status, _headers, payload = request(f"https://api.crossref.org/works/{encoded}")
    if status != 200:
        raise RuntimeError(f"Crossref returned HTTP {status} for {doi}")
    record = json.loads(payload.decode("utf-8"))["message"]
    redirect_status, redirect_headers, _ = request(f"https://doi.org/{doi}", opener=NO_REDIRECT)
    redirect_target = redirect_headers.get("Location", "")
    if redirect_status not in {301, 302, 303, 307, 308} or not redirect_target:
        raise RuntimeError(f"DOI did not resolve through doi.org: {doi}: HTTP {redirect_status}")
    return record, redirect_status, redirect_target


def verify(*, live: bool) -> None:
    with (ROOT / "literature_manifest.csv").open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    with (ROOT / "selected_dois.tsv").open(encoding="utf-8", newline="") as handle:
        selected = list(csv.DictReader(handle, delimiter="\t"))
    bib = bibtex_by_doi()
    selected_dois = {row["doi"].lower() for row in selected}
    manifest_dois = {row["doi"].lower() for row in manifest}
    if len(manifest) != 48 or len(selected) != 48 or len(bib) != 48:
        raise RuntimeError("expected exactly 48 manifest, selection, and BibTeX records")
    if selected_dois != manifest_dois or selected_dois != set(bib):
        raise RuntimeError("DOI sets differ across selection, manifest, and BibTeX")

    output: list[dict[str, str]] = []
    for row in manifest:
        doi = row["doi"].lower()
        metadata_path = ROOT / "metadata" / f"{row['id']}_{row['year']}.json"
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        saved = payload["crossref"]
        current = saved
        doi_status = "not_checked"
        redirect_target = ""
        if live:
            current, status, redirect_target = live_record(doi)
            doi_status = str(status)
            time.sleep(0.15)

        key, fields = bib[doi]
        title = clean((current.get("title") or [""])[0])
        journal = clean((current.get("container-title") or [""])[0])
        year = final_year(current)
        locator = clean(current.get("page", "") or current.get("article-number", ""))
        pdf_path = ROOT / row["pdf_file"]
        actual_sha = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        expected_url = f"https://doi.org/{doi}"

        checks = {
            "type": current.get("type") == "journal-article",
            "doi": current.get("DOI", "").lower() == doi == fields.get("doi", "").lower(),
            "title": normalized(title) == normalized(row["title"]) == normalized(fields.get("title", "")),
            "journal": normalized(journal) == normalized(row["journal"]) == normalized(fields.get("journal", "")),
            "year": str(year) == row["year"] == fields.get("year", ""),
            "authors": bool(current.get("author"))
            and all(
                normalized(author.get("family", "")) in normalized(fields.get("author", ""))
                for author in current.get("author") or []
                if author.get("family")
            ),
            "volume": not row["volume"] or fields.get("volume", "") == row["volume"],
            "issue": not row["issue"] or fields.get("number", "") == row["issue"],
            "locator": not locator or normalized(fields.get("pages", "")) == normalized(locator),
            "doi_url": fields.get("url", "").lower() == expected_url,
            "pdf": pdf_path.is_file() and actual_sha == row["pdf_sha256"],
            "saved_live_title": normalized((saved.get("title") or [""])[0]) == normalized(title),
            "saved_live_authors": [a.get("family", "") for a in saved.get("author") or []]
            == [a.get("family", "") for a in current.get("author") or []],
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(f"reference mismatch for {doi}: {', '.join(failed)}")

        output.append(
            {
                "id": row["id"],
                "pdf_file": row["pdf_file"],
                "bibtex_key": key,
                "year": str(year),
                "title": title,
                "authors": authors_text(current),
                "journal": journal,
                "volume": clean(current.get("volume", "")),
                "issue": clean(current.get("issue", "")),
                "pages_or_article": locator,
                "doi": doi,
                "doi_url": expected_url,
                "publisher_article_url": publisher_url(current, doi),
                "pdf_source_url": row["pdf_source_url"],
                "crossref_metadata_url": f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='')}",
                "bibtex_source_url": f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='')}/transform/application/x-bibtex",
                "metadata_json": f"metadata/{metadata_path.name}",
                "pdf_sha256": actual_sha,
                "doi_resolution_http": doi_status,
                "doi_redirect_target": redirect_target,
                "verification_status": "verified_live" if live else "verified_from_saved_source",
            }
        )

    with MAPPING.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"REFERENCE_MAPPING_STATUS: verified")
    print(f"records: {len(output)}")
    print(f"live: {str(live).lower()}")
    print(f"mapping: {MAPPING}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="check current Crossref records and doi.org redirects")
    args = parser.parse_args()
    verify(live=args.live)
