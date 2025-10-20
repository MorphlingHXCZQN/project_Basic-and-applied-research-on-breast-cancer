"""Utilities for querying PubMed for breast cancer literature."""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Iterable

import xml.etree.ElementTree as ET

import requests

LOGGER = logging.getLogger(__name__)


def _year_bounds(years: int) -> tuple[str, str]:
    today = _dt.date.today()
    start_year = max(today.year - years + 1, 1900)
    return f"{start_year}/01/01", today.strftime("%Y/%m/%d")


def search_pubmed(
    query: str,
    *,
    years: int = 5,
    retmax: int = 50,
    email: str | None = None,
    api_key: str | None = None,
) -> list[str]:
    """Return PubMed IDs for the given query."""

    mindate, maxdate = _year_bounds(years)
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": retmax,
        "sort": "cited",
        "mindate": mindate,
        "maxdate": maxdate,
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key

    LOGGER.info("Searching PubMed for query=%s", query)
    response = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params=params, timeout=30
    )
    response.raise_for_status()
    data = response.json()
    return data.get("esearchresult", {}).get("idlist", [])


def fetch_summaries(
    pmids: Iterable[str], *, email: str | None = None, api_key: str | None = None
) -> list[dict]:
    """Retrieve summary information for each PMID."""

    pmid_list = list(pmids)
    if not pmid_list:
        return []

    params = {
        "db": "pubmed",
        "retmode": "json",
        "id": ",".join(pmid_list),
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key

    LOGGER.info("Fetching summaries for %d PubMed IDs", len(pmid_list))
    response = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi", params=params, timeout=30
    )
    response.raise_for_status()
    data = response.json()
    summaries = data.get("result", {})
    return [summaries[pmid] for pmid in pmid_list if pmid in summaries]


def fetch_details(
    pmids: Iterable[str], *, email: str | None = None, api_key: str | None = None
) -> dict[str, dict]:
    """Retrieve detailed article metadata including abstracts and MeSH terms."""

    pmid_list = list(pmids)
    if not pmid_list:
        return {}

    params = {
        "db": "pubmed",
        "retmode": "xml",
        "id": ",".join(pmid_list),
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key

    LOGGER.info("Fetching detailed records for %d PubMed IDs", len(pmid_list))
    response = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", params=params, timeout=30
    )
    response.raise_for_status()
    details: dict[str, dict] = {}
    root = ET.fromstring(response.text)
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID")
        if not pmid:
            continue
        abstract_text = " ".join(
            (section.text or "").strip() for section in article.findall(".//Abstract/AbstractText")
        ).strip()
        mesh_terms = [
            descriptor.text.strip()
            for descriptor in article.findall(".//MeshHeading/DescriptorName")
            if descriptor.text
        ]
        publication_types = [
            pub_type.text.strip()
            for pub_type in article.findall(".//PublicationType")
            if pub_type.text
        ]
        details[pmid] = {
            "abstract": abstract_text,
            "mesh_terms": mesh_terms,
            "publication_types": publication_types,
        }
    return details


APPLICATION_KEYWORDS = {
    "translational",
    "clinical",
    "applied",
    "therapeutic",
    "diagnostic",
    "biomarker",
    "precision",
    "trial",
    "imaging",
    "intervention",
}


def filter_applied_research(articles: Iterable[dict]) -> list[dict]:
    """Keep only articles whose title or keywords suggest an applied focus."""

    filtered: list[dict] = []
    for article in articles:
        title = article.get("title", "").lower()
        keywords = " ".join(article.get("keywords", [])).lower()
        if any(keyword in title or keyword in keywords for keyword in APPLICATION_KEYWORDS):
            filtered.append(article)
    return filtered


def summarize_articles(articles: Iterable[dict], details: dict[str, dict] | None = None) -> list[dict]:
    """Normalize article payloads to a consistent structure."""

    normalized: list[dict] = []
    for article in articles:
        pmid = article.get("uid", "")
        detail = (details or {}).get(pmid, {})
        normalized.append(
            {
                "pmid": pmid,
                "title": article.get("title", ""),
                "authors": [
                    f"{author.get('name')}" for author in article.get("authors", []) if author.get("name")
                ],
                "pubdate": article.get("pubdate", ""),
                "journal": article.get("fulljournalname", ""),
                "summary": article.get("elocationid", ""),
                "abstract": detail.get("abstract", ""),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{article.get('uid', '')}/",
                "keywords": article.get("keywords", []) or detail.get("mesh_terms", []),
                "mesh_terms": detail.get("mesh_terms", []),
                "publication_types": detail.get("publication_types", []),
                "cited_by": article.get("citedbycount", 0),
            }
        )
    return normalized
