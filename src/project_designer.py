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

        system_prompt = (
            "You are an experienced grant writer."
            " Convert the research direction summary into a structured JSON outline "
            "with Chinese section titles and concise bullet points."
        )
        sections = [
            "一、研究背景",
            "二、研究目标",
            "三、研究内容与技术路线",
            "四、可行性分析",
            "五、研究基础与团队优势",
            "六、预期成果与考核指标",
            "七、实施计划与进度安排",
        ]
        user_prompt = (
            "请根据以下研究方向总结，生成一个JSON对象，键为固定章节标题，值为要点数组。"
            " 每个章节至少给出3条要点。\n\n"
            f"章节列表：{json.dumps(sections, ensure_ascii=False)}\n"
            f"研究方向总结：\n{direction_summary}"
        )
        try:
            completion = self._gpt.chat_completion(
                system_prompt,
                [{"role": "user", "content": user_prompt}],
            )
        except Exception:  # pragma: no cover - network fallback
            fallback = {
                section: [f"无法访问GPT服务，保留原始方向概要：{direction_summary[:200]}"]
                for section in sections
            }
            return json.dumps(fallback, ensure_ascii=False, indent=2)
        outline = self._parse_completion(completion, sections)
        return json.dumps(outline, ensure_ascii=False, indent=2)

    def _parse_completion(self, completion: str, sections: list[str]) -> dict[str, list[str]]:
        text = completion.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[len("json"):]
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = {}
        outline: dict[str, list[str]] = {}
        for section in sections:
            value = data.get(section)
            if isinstance(value, list):
                outline[section] = [str(item).strip() for item in value if str(item).strip()]
            elif isinstance(value, str):
                outline[section] = [value.strip()]
            else:
                outline[section] = []
        return outline
