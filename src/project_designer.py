"""Generate project directions based on PubMed literature."""

from __future__ import annotations

import json
from typing import Iterable

from .gpt_client import GPTClient


class ProjectDesigner:
    """Analyse literature and produce project outlines."""

    def __init__(self, gpt: GPTClient) -> None:
        self._gpt = gpt

    def design_direction(self, articles: Iterable[dict], background: str) -> str:
        """Return a candidate research direction summary."""

        article_payload = list(articles)
        if not article_payload:
            raise ValueError("No articles were provided for project design.")
        return self._gpt.summarize_articles(article_payload, background)

    def create_outline(self, direction_summary: str) -> str:
        """Structure the GPT direction summary into an outline."""

        sections = [
            "一、研究背景",
            "二、研究目标",
            "三、研究内容与技术路线",
            "四、可行性分析",
            "五、研究基础与团队优势",
            "六、预期成果与考核指标",
            "七、实施计划与进度安排",
        ]
        outline = {section: direction_summary for section in sections}
        return json.dumps(outline, ensure_ascii=False, indent=2)
