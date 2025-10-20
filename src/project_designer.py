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
            return json.dumps(
                self._offline_outline(direction_summary, sections),
                ensure_ascii=False,
                indent=2,
            )
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

    def _offline_outline(
        self, direction_summary: str, sections: list[str]
    ) -> dict[str, list[str]]:
        """Fallback outline when GPT is not available."""

        summary = direction_summary.strip()
        if not summary:
            summary = "乳腺癌精准诊疗方向，强调基础与临床协同。"

        base_points = [
            "针对乳腺癌早诊与耐药机制提出系统研究思路。",
            "构建跨区域的样本与数据共享平台，强化粤惠联合特色。",
            "通过生物标志物、免疫微环境和临床验证形成闭环。",
            "设置分阶段里程碑和量化考核指标保障项目落地。",
        ]

        outline: dict[str, list[str]] = {}
        for index, section in enumerate(sections):
            if index == 0:
                outline[section] = [summary[:200]] + base_points[:2]
            elif index == 1:
                outline[section] = [
                    "总体目标聚焦乳腺癌精准诊疗技术体系构建。",
                    "分解为关键科学问题、技术攻关与临床转化三个层级。",
                ]
            elif index == 2:
                outline[section] = [
                    "模块一：多组学筛查乳腺癌早诊标志物并建立预测模型。",
                    "模块二：解析免疫微环境与耐药关联，提出干预策略。",
                    "模块三：依托惠州—深圳联合平台开展临床验证与推广。",
                ]
            elif index == 3:
                outline[section] = [
                    "已有区域合作基础、临床病例资源及实验平台支撑研究实施。",
                    "技术路线成熟，数据与伦理管理制度完善，风险可控。",
                ]
            elif index == 4:
                outline[section] = [
                    "团队涵盖基础研究、临床肿瘤与数据科学专家，形成互补优势。",
                    "近年承担相关省市课题并形成原创成果，为本项目奠定基础。",
                ]
            elif index == 5:
                outline[section] = [
                    "预期发表高水平论文、申请发明专利并形成诊疗指南。",
                    "量化指标包括标志物筛选准确率、转化成果推广数量等。",
                ]
            elif index == 6:
                outline[section] = [
                    "第一年完成样本与数据平台建设，完成多组学测序。",
                    "第二年聚焦机制研究与临床小试，优化关键技术指标。",
                    "第三年开展多中心验证与成果转化落地，形成示范应用。",
                ]
            else:
                outline[section] = base_points
        return outline
