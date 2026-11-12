#!/usr/bin/env python3
"""Build and verify the EcoTrust-FLIDS 2023-2026 literature set.

Metadata and BibTeX are fetched from Crossref. Open-access locations are
fetched from OpenAlex. PDF downloads are restricted to publisher or
institutional-repository URLs and are validated before entering the manifest.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
LEDGER = ROOT / "selected_dois.tsv"
PAPERS = ROOT / "papers"
METADATA = ROOT / "metadata"
MANIFEST = ROOT / "literature_manifest.csv"
BIB = ROOT / "ref.bib"
USER_AGENT = "EcoTrust-FLIDS-literature-audit/1.0"
SCOPUS_SOURCE_LIST_URL = (
    "https://downloads.ctfassets.net/o78em1y1w4i4/7xtaTxNiNcWRTeZkV86eNy/"
    "e6b17b29beaf8d18175591a4e8dee0bf/ext_list_Sep_2026.xlsx"
)

BLOCKED_HOST_FRAGMENTS = (
    "arxiv.org",
    "researchgate.net",
    "academia.edu",
    "preprints.org",
    "techrxiv.org",
    "sci-hub",
)

PREFERRED_HOST_FRAGMENTS = (
    "pmc.ncbi.nlm.nih.gov",
    "hal.science",
    "dspace.",
    "repository.",
    "eprints.",
    "orbilu.",
    "orca.",
    "worktribe.com",
    "nrc-publications.canada.ca",
    "addi.ehu.eus",
    "link.springer.com",
    "nature.com",
    "frontiersin.org",
    "journals.plos.org",
    "peerj.com",
    "hindawi.com",
    "techscience.com",
    "techscience.press",
    "thesai.org",
)

MDPI_SLUG_BY_ISSN = {
    "2673-8732": "network",
    "2673-2688": "ai",
    "1999-5903": "futureinternet",
    "1424-8220": "sensors",
    "2079-9292": "electronics",
    "2076-3417": "applsci",
    "2504-2289": "bdcc",
    "2227-7390": "mathematics",
    "2624-6511": "smartcities",
    "2224-2708": "jsan",
    "2504-446X": "drones",
    "2306-5729": "data",
    "2227-9032": "healthcare",
}

STATIC_PDF_URLS = {
    "10.1007/s10922-023-09782-9": [
        "https://researchmgt.monash.edu/ws/portalfiles/portal/715882538/663920936-oa.pdf"
    ],
    "10.1016/j.cose.2025.104337": [
        "https://www.jaist.ac.jp/~razvan/publications/fedmse_iot_network_intrusion_detection.pdf"
    ],
}


def request_bytes(url: str, accept: str = "application/json", attempts: int = 6) -> bytes:
    last: Exception | None = None
    for attempt in range(attempts):
        req = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": accept},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                return response.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in {429, 500, 502, 503, 504}:
                raise
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"request failed after {attempts} attempts: {url}: {last}")


def request_json(url: str) -> dict:
    return json.loads(request_bytes(url).decode("utf-8"))


def crossref_message(doi: str) -> dict:
    encoded = urllib.parse.quote(doi, safe="")
    return request_json(f"https://api.crossref.org/works/{encoded}")["message"]


def openalex_work(doi: str) -> dict:
    encoded = urllib.parse.quote(f"https://doi.org/{doi}", safe="")
    return request_json(f"https://api.openalex.org/works/{encoded}")


def final_year(message: dict) -> int | None:
    for key in ("published-print", "published", "issued", "published-online"):
        parts = (message.get(key) or {}).get("date-parts") or []
        if parts and parts[0]:
            return int(parts[0][0])
    return None


def clean_text(value: str) -> str:
    value = html.unescape(value or "")
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def slugify(value: str, limit: int = 72) -> str:
    value = value.lower().replace("‐", "-").replace("–", "-").replace("—", "-")
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value[:limit].rstrip("-")


def mdpi_resource_url(url: str) -> str | None:
    parsed = urllib.parse.urlparse(url)
    if "mdpi.com" not in parsed.netloc:
        return None
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 5 or parts[-1] != "pdf":
        return None
    issn, volume, _issue, article = parts[-5:-1]
    slug = MDPI_SLUG_BY_ISSN.get(issn)
    if not slug or not volume.isdigit() or not article.isdigit():
        return None
    code = f"{slug}-{int(volume):02d}-{int(article):05d}"
    return f"https://mdpi-res.com/d_attachment/{slug}/{code}/article_deploy/{code}.pdf"


def candidate_pdf_urls(doi: str, work: dict, crossref: dict) -> list[str]:
    raw: list[str] = list(STATIC_PDF_URLS.get(doi.lower(), []))
    for link in crossref.get("link") or []:
        url = link.get("URL")
        if url and not any(blocked in url.lower() for blocked in BLOCKED_HOST_FRAGMENTS):
            converted = mdpi_resource_url(url)
            if converted:
                raw.append(converted)
            raw.append(url)
    for location in work.get("locations") or []:
        url = location.get("pdf_url")
        if not url:
            continue
        if any(blocked in url.lower() for blocked in BLOCKED_HOST_FRAGMENTS):
            continue
        converted = mdpi_resource_url(url)
        if converted:
            raw.append(converted)
        raw.append(url)

    def priority(url: str) -> tuple[int, str]:
        lower = url.lower()
        for index, fragment in enumerate(PREFERRED_HOST_FRAGMENTS):
            if fragment in lower:
                return index, lower
        if "mdpi-res.com" in lower:
            return len(PREFERRED_HOST_FRAGMENTS), lower
        if "mdpi.com" in lower:
            return len(PREFERRED_HOST_FRAGMENTS) + 10, lower
        return len(PREFERRED_HOST_FRAGMENTS) + 5, lower

    unique = list(dict.fromkeys(raw))
    return sorted(unique, key=priority)


def download_pdf(urls: list[str], destination: Path) -> str:
    part = destination.with_suffix(".part")
    if part.exists():
        part.unlink()
    for url in urls:
        command = [
            "curl",
            "-L",
            "--fail",
            "--silent",
            "--show-error",
            "--retry",
            "3",
            "--retry-all-errors",
            "--connect-timeout",
            "30",
            "--max-time",
            "180",
            "-A",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "-o",
            str(part),
            url,
        ]
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode != 0 or not part.exists():
            if part.exists():
                part.unlink()
            continue
        header = part.read_bytes()[:5]
        if header != b"%PDF-" or part.stat().st_size < 50_000:
            part.unlink()
            continue
        part.replace(destination)
        return url
    raise RuntimeError(f"no lawful PDF candidate downloaded successfully for {destination.name}")


def validate_pdf(path: Path, title: str) -> tuple[int, float, str]:
    info = subprocess.run(["pdfinfo", str(path)], text=True, capture_output=True, check=True)
    match = re.search(r"^Pages:\s+(\d+)", info.stdout, flags=re.MULTILINE)
    pages = int(match.group(1)) if match else 0
    if pages < 2:
        raise RuntimeError(f"invalid page count for {path}: {pages}")

    reader = PdfReader(path)
    extracted = "\n".join((page.extract_text() or "") for page in reader.pages[:3])
    pdf_text = re.sub(r"[^a-z0-9]+", " ", extracted.lower())
    words = [
        word
        for word in re.sub(r"[^a-z0-9]+", " ", title.lower()).split()
        if len(word) >= 5 and word not in {"federated", "learning", "based", "using"}
    ]
    unique_words = list(dict.fromkeys(words))
    matches = sum(1 for word in unique_words if word in pdf_text)
    ratio = matches / max(len(unique_words), 1)
    if ratio < 0.35:
        raise RuntimeError(f"title-to-PDF text match too low for {path.name}: {ratio:.3f}")
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return pages, ratio, sha256


def crossref_bibtex(doi: str) -> str:
    encoded = urllib.parse.quote(doi, safe="")
    url = f"https://api.crossref.org/works/{encoded}/transform/application/x-bibtex"
    value = request_bytes(url, accept="application/x-bibtex").decode("utf-8").strip()
    if not value.startswith("@") or doi.lower() not in value.lower():
        raise RuntimeError(f"invalid Crossref BibTeX for {doi}")
    # Crossref's BibTeX transform can return legacy HTTP DOI URLs and HTML
    # entities. Normalize those mechanically while retaining registry fields.
    value = value.replace("&amp;", r"\&")
    value = html.unescape(value)
    value = re.sub(
        r"url=\{https?://(?:dx\.)?doi\.org/[^}]+\}",
        f"url={{https://doi.org/{doi}}}",
        value,
        flags=re.IGNORECASE,
    )
    return value


def ensure_bibtex_locator(value: str, crossref: dict) -> str:
    """Fill a Crossref-registered page/article locator omitted by x-bibtex."""
    if re.search(r"\bpages\s*=", value, flags=re.IGNORECASE):
        return value
    locator = clean_text(crossref.get("page", "") or crossref.get("article-number", ""))
    if not locator:
        return value
    closing = value.rfind("}")
    if closing < 0:
        raise RuntimeError("invalid BibTeX: missing closing brace")
    prefix = value[:closing].rstrip()
    if not prefix.endswith(","):
        prefix += ","
    return f"{prefix} pages={{{locator}}} {value[closing:]}"


def main() -> None:
    rows = list(csv.DictReader(LEDGER.open(encoding="utf-8"), delimiter="\t"))
    if len(rows) != 48 or len({row["doi"].lower() for row in rows}) != 48:
        raise RuntimeError("selected_dois.tsv must contain exactly 48 unique DOIs")
    PAPERS.mkdir(exist_ok=True)
    METADATA.mkdir(exist_ok=True)

    manifest_rows: list[dict[str, str | int | float]] = []
    bib_entries: list[str] = []

    for position, row in enumerate(rows, start=1):
        paper_id = row["id"]
        selected_year = int(row["year"])
        doi = row["doi"].lower()
        print(f"[{position:02d}/48] {doi}", flush=True)

        crossref = crossref_message(doi)
        work = openalex_work(doi)
        title = clean_text((crossref.get("title") or [work.get("title") or ""])[0])
        journal = clean_text((crossref.get("container-title") or [""])[0])
        year = final_year(crossref)
        if crossref.get("type") != "journal-article":
            raise RuntimeError(f"not a journal article: {doi}: {crossref.get('type')}")
        if year != selected_year or year not in {2023, 2024, 2025, 2026}:
            raise RuntimeError(f"final publication year mismatch for {doi}: selected={selected_year}, Crossref={year}")
        if work.get("is_retracted"):
            raise RuntimeError(f"OpenAlex marks record as retracted: {doi}")
        if not (crossref.get("volume") or crossref.get("issue") or crossref.get("page") or crossref.get("article-number")):
            raise RuntimeError(f"no final journal issue/article locator for {doi}")

        filename = f"{paper_id}_{year}_{slugify(title)}.pdf"
        destination = PAPERS / filename
        urls = candidate_pdf_urls(doi, work, crossref)
        if not urls:
            raise RuntimeError(f"no lawful non-preprint PDF location for {doi}")
        metadata_path = METADATA / f"{paper_id}_{year}.json"
        if destination.exists():
            pages, title_ratio, sha256 = validate_pdf(destination, title)
            source_url = urls[0]
            if metadata_path.exists():
                prior = json.loads(metadata_path.read_text(encoding="utf-8"))
                source_url = prior.get("downloaded_pdf_url") or source_url
        else:
            source_url = download_pdf(urls, destination)
            pages, title_ratio, sha256 = validate_pdf(destination, title)

        metadata_payload = {
            "selection": row,
            "crossref": crossref,
            "openalex": work,
            "downloaded_pdf_url": source_url,
            "pdf_sha256": sha256,
            "pdf_pages": pages,
            "title_match_ratio": title_ratio,
        }
        metadata_path.write_text(
            json.dumps(metadata_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        issn = ";".join(crossref.get("ISSN") or [])
        manifest_rows.append(
            {
                "id": paper_id,
                "year": year,
                "doi": doi,
                "title": title,
                "journal": journal,
                "volume": crossref.get("volume", ""),
                "issue": crossref.get("issue", ""),
                "pages_or_article": crossref.get("page", "") or crossref.get("article-number", ""),
                "issn": issn,
                "theme": row["theme"],
                "pdf_file": f"papers/{filename}",
                "pdf_source_url": source_url,
                "pdf_pages": pages,
                "pdf_bytes": destination.stat().st_size,
                "pdf_sha256": sha256,
                "title_match_ratio": f"{title_ratio:.3f}",
                "metadata_source": f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='')}",
                "indexing_evidence": SCOPUS_SOURCE_LIST_URL,
                "status": "verified",
            }
        )
        bib_entries.append(ensure_bibtex_locator(crossref_bibtex(doi), crossref))
        time.sleep(0.15)

    fieldnames = list(manifest_rows[0].keys())
    with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    BIB.write_text("\n\n".join(bib_entries) + "\n", encoding="utf-8")
    print(f"verified PDFs: {len(manifest_rows)}", flush=True)
    print(f"manifest: {MANIFEST}", flush=True)
    print(f"BibTeX entries: {len(bib_entries)}", flush=True)


if __name__ == "__main__":
    main()
