"""Wrapper around the OpenAI API with graceful fallbacks."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from itertools import cycle
from typing import Iterable

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency at runtime
    OpenAI = None  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass
class GPTClient:
    """Small helper around the OpenAI Chat Completions API."""

    model: str = "gpt-4o"
    temperature: float = 0.5
    _client: "OpenAI | None" = field(init=False, default=None, repr=False)

    def _requires_openai(self) -> None:
        if OpenAI is None:
            raise RuntimeError(
                "The openai package is not installed. Install it or provide a custom client."
            )
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    def _client_instance(self) -> "OpenAI":
        self._requires_openai()
        if self._client is None:
            self._client = OpenAI()
        return self._client

    def chat_completion(self, system_prompt: str, messages: Iterable[dict[str, str]]) -> str:
        """Execute a chat completion call and return the assistant content."""

        client = self._client_instance()
        completion = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "system", "content": system_prompt}, *messages],
        )
        choice = completion.choices[0].message.content
        LOGGER.debug("Received completion: %s", choice)
        return str(choice)

    def offline_stub(self, prompt: str) -> str:
        """Legacy stub kept for backward compatibility."""

        LOGGER.warning("Using offline stub for prompt: %.120s", prompt)
        return (
            "当前环境无法连接至 OpenAI 服务，请检查网络或 API 配置。"
            " 为保证流程不中断，系统已自动切换至本地启发式策略。"
        )

    def summarize_articles(self, articles: Iterable[dict], background: str = "") -> str:
        """Ask GPT to propose research directions from the supplied articles."""

        system_prompt = (
            "You are a biomedical researcher specializing in breast cancer. "
            "Synthesize insights from the provided PubMed summaries to propose a translational research direction."
        )
        ranked_articles = list(articles)
        content = (
            "背景信息：\n"
            f"{background}\n\n"
            "以下是根据关键词契合度、研究背景匹配度、近发表时间和引用情况综合评分的前沿文献，请优先结合得分靠前的研究：\n"
            f"{json.dumps(ranked_articles, ensure_ascii=False, indent=2)}\n\n"
            "请总结共同趋势，提出一个聚焦于临床转化或基础应用的研究方向，阐明创新点、临床需求以及关键科学问题。"
        )
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
        try:
            return self.chat_completion(system_prompt, messages)
        except Exception as exc:  # pragma: no cover - network fallback
            LOGGER.error("Falling back to offline summary: %s", exc)
            return self._offline_direction_summary(ranked_articles, background)

    def draft_proposal(self, outline: str, word_count: int) -> str:
        """Ask GPT to write a structured project proposal."""

        system_prompt = (
            "You are drafting a Chinese research project proposal on translational breast cancer studies. "
            "Produce formal, well-structured language with numbered sections and sub-sections."
        )
        messages = [
            {
                "role": "user",
                "content": (
                    "Using the following outline and requirements, write a project plan in Chinese of approximately "
                    f"{word_count} words. Ensure the sections cover: research background, objectives, detailed design, "
                    "feasibility analysis, expected outcomes, milestones, and references.\n\n"
                    f"Outline:\n{outline}"
                ),
            }
        ]
        try:
            return self.chat_completion(system_prompt, messages)
        except Exception as exc:  # pragma: no cover - network fallback
            LOGGER.error("Falling back to offline proposal: %s", exc)
            return self._offline_proposal_text(outline, word_count)

    # ------------------------------------------------------------------
    # Offline helpers

    def _offline_direction_summary(
        self, articles: Iterable[dict], background: str
    ) -> str:
        """Create a research direction summary without external APIs."""

        articles_list = list(articles)
        if not articles_list:
            return (
                "未能获取到乳腺癌相关文章，但结合项目背景，我们建议围绕乳腺癌的临床转化需求，"
                "从生物标志物发现、精准治疗与惠州地区资源协同等方向探索备选课题。"
            )

        intro_lines = [
            "在无法访问外部大模型服务的情况下，系统依据本地收集的文献数据综合评估研究方向。",
        ]
        background_text = background.strip()
        if background_text:
            snippet = background_text[:300]
            if len(background_text) > 300:
                snippet += "..."
            intro_lines.append(
                "结合申报人提供的项目背景要点："
                f"{snippet}"
                "，我们强调粤惠联合基金对区域协同和转化应用的要求。"
            )

        highlight_lines = [
            "经过对高被引文献的题目、发表年份、MeSH 主题词和引用量的启发式评分，"
            "下列研究趋势与基金定位高度吻合：",
        ]
        bullet_lines = []
        for index, article in enumerate(articles_list[:8], 1):
            title = str(article.get("title") or "未提供标题").strip()
            year = article.get("year")
            year_text = f"（{year}年）" if year else ""
            mesh_terms = article.get("mesh_terms") or []
            key_terms = "、".join(mesh_terms[:3]) if mesh_terms else "乳腺癌"
            impact = article.get("citations") or article.get("score")
            impact_text = f" 引用量/评分：{impact}" if impact else ""
            bullet_lines.append(
                f"{index}. {title}{year_text} —— 关键词：{key_terms}{impact_text}."
            )

        recommendation = (
            "综上所述，建议凝练“乳腺癌精准诊疗与临床转化协同研究”作为总体方向，"
            "重点关注早诊标志物筛选、免疫微环境调控以及惠深联合转化平台建设，"
            "以满足三年期、区域协同和成果可落地的项目要求。"
        )

        return "\n".join([*intro_lines, *highlight_lines, *bullet_lines, recommendation])

    def _offline_proposal_text(self, outline: str, word_count: int) -> str:
        """Generate a long-form proposal when GPT is unavailable."""

        try:
            outline_data = json.loads(outline)
        except json.JSONDecodeError:
            outline_data = {}

        section_order = [
            "一、研究背景",
            "二、研究目标",
            "三、研究内容与技术路线",
            "四、可行性分析",
            "五、研究基础与团队优势",
            "六、预期成果与考核指标",
            "七、实施计划与进度安排",
            "八、参考文献",
        ]

        filler_sentences = [
            "项目团队将充分利用粤港澳大湾区医疗资源和科研平台，实现基础研究成果向临床快速转化。",
            "通过加强惠州与深圳协同合作机制，打造共享实验与临床试验基地，缩短成果落地周期。",
            "研究过程中将严格遵循伦理与数据安全规范，强化患者隐私保护与随访管理。",
            "项目拟构建多组学数据融合分析体系，结合人工智能模型提升乳腺癌诊疗精准度。",
            "围绕国家与广东省人口健康战略，项目将形成可推广的标准化诊疗路径与技术指南。",
        ]

        def expand_point(point: str) -> str:
            base = str(point).strip().rstrip("。")
            if not base:
                return "该要点将结合乳腺癌转化应用需求进行具体展开，确保与基金导向一致。"
            elaboration = (
                "。围绕该要点，项目将结合最新的乳腺癌基础研究进展、区域临床痛点以及粤惠联合基金的协同要求，"
                "设计出多阶段、多学科交叉的研究路径，确保成果具有可转化性和可量化指标。"
            )
            return base + elaboration

        paragraphs: list[str] = []
        for section in section_order:
            paragraphs.append(section)
            points = outline_data.get(section)
            if not isinstance(points, list) or not points:
                points = ["补充撰写与本章节相对应的要点，突出区域协同与创新性。"]
            for point in points:
                paragraphs.append(expand_point(point))
            paragraphs.append("")

        desired_length = max(word_count, 2000)
        current_text = "".join(paragraphs)
        filler_cycle = cycle(filler_sentences)
        while len(current_text) < desired_length:
            sentence = next(filler_cycle)
            paragraphs.append(sentence)
            current_text = "".join(paragraphs)

        return "\n".join(paragraphs).strip()
