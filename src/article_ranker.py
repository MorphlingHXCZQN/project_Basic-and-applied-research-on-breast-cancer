"""Article ranking utilities to prioritize impactful translational studies."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from .config import PipelineConfig

TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+")


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def _parse_year(date_str: str) -> int | None:
    if not date_str:
        return None
    for fmt in ("%Y", "%Y %b", "%Y %m %d", "%Y/%m/%d", "%Y %b %d"):
        try:
            return datetime.strptime(date_str[: len(fmt)], fmt).year
        except ValueError:
            continue
    for chunk in re.findall(r"(19|20)\d{2}", date_str):
        try:
            return int(chunk)
        except ValueError:
            continue
    return None


@dataclass
class ArticleScore:
    article: dict
    total: float
    components: dict[str, float]


class ArticleRanker:
    """Score and rank articles using heuristic signals."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    def _keyword_component(self, article: dict) -> float:
        title = article.get("title", "").lower()
        keywords = " ".join(article.get("keywords", [])).lower()
        focus = self._config.keyword_focus
        if not focus:
            return 0.0
        matches = sum(1 for keyword in focus if keyword.lower() in title or keyword.lower() in keywords)
        return matches / len(focus)

    def _background_component(self, article: dict, background_tokens: Counter[str]) -> float:
        if not background_tokens:
            return 0.0
        tokens = Counter(_tokenize(article.get("title", "") + " " + article.get("abstract", "")))
        if not tokens:
            return 0.0
        overlap = sum(min(tokens[token], count) for token, count in background_tokens.items())
        total = sum(background_tokens.values())
        if total == 0:
            return 0.0
        return overlap / total

    def _recency_component(self, article: dict) -> float:
        pubdate = article.get("pubdate") or article.get("sortpubdate")
        year = _parse_year(str(pubdate)) if pubdate else None
        if year is None:
            return 0.0
        current_year = datetime.now().year
        age = max(current_year - year, 0)
        return max(0.0, 1.0 - age / max(self._config.years, 1))

    def _citation_component(self, article: dict) -> float:
        cited = article.get("cited_by") or article.get("citedbycount") or article.get("pmcrefcount")
        try:
            cited_value = float(cited)
        except (TypeError, ValueError):
            return 0.0
        if cited_value <= 0:
            return 0.0
        return math.log1p(cited_value) / math.log(100 + 1)

    def score(self, articles: Iterable[dict], background: str) -> list[ArticleScore]:
        background_tokens = Counter(_tokenize(background))
        scored: list[ArticleScore] = []
        for article in articles:
            components = {
                "keyword": self._keyword_component(article),
                "background": self._background_component(article, background_tokens),
                "recency": self._recency_component(article),
                "citation": self._citation_component(article),
            }
            total = (
                components["keyword"] * self._config.keyword_weight
                + components["background"] * self._config.background_weight
                + components["recency"] * self._config.recency_weight
                + components["citation"] * self._config.citation_weight
            )
            scored.append(ArticleScore(article=article, total=total, components=components))
        scored.sort(key=lambda score: score.total, reverse=True)
        return scored

    def rank(self, articles: Iterable[dict], background: str) -> list[dict]:
        scored = self.score(articles, background)
        ranked: list[dict] = []
        for entry in scored:
            enriched = dict(entry.article)
            enriched["score"] = round(entry.total, 4)
            enriched["score_breakdown"] = {
                key: round(value, 4) for key, value in entry.components.items()
            }
            ranked.append(enriched)
        return ranked
