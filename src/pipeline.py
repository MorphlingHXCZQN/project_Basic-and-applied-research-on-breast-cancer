"""End-to-end automation pipeline for breast cancer project ideation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from textwrap import dedent
from typing import Any

from .article_ranker import ArticleRanker
from .config import DEFAULT_OUTPUT_DIR, PipelineConfig
from .github_uploader import GitHubAsset, upload_assets
from .gpt_client import GPTClient
from .project_designer import ProjectDesigner
from .proposal_writer import ProposalWriter
from .pubmed_scraper import (
    fetch_details,
    fetch_summaries,
    filter_applied_research,
    search_pubmed,
    summarize_articles,
)
from .offline_articles import OFFLINE_ARTICLES

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._gpt = GPTClient(model=config.openai_model, temperature=config.openai_temperature)
        self._designer = ProjectDesigner(self._gpt)
        self._ranker = ArticleRanker(config)
        self._writer = ProposalWriter(self._gpt)
        self._data_source = "pubmed"

    def _build_query(self) -> str:
        base_query = self._config.query
        applied_focus = (
            "(breast cancer[Title/Abstract]) AND (translational OR applied OR clinical)"
        )
        if base_query:
            return f"({base_query}) AND {applied_focus}"
        return applied_focus

    def gather_articles(self) -> list[dict[str, Any]]:
        query = self._build_query()
        try:
            pmids = search_pubmed(
                query,
                years=self._config.years,
                retmax=self._config.retmax,
                email=self._config.email,
                api_key=self._config.api_key,
            )
            summaries = fetch_summaries(
                pmids, email=self._config.email, api_key=self._config.api_key
            )
            try:
                details = fetch_details(
                    pmids, email=self._config.email, api_key=self._config.api_key
                )
            except Exception as error:  # pragma: no cover - network fallback
                LOGGER.warning("Failed to fetch article details: %s", error)
                details = {}
            normalized = summarize_articles(summaries, details)
            filtered = filter_applied_research(normalized)
            LOGGER.info("Retrieved %d applied research articles", len(filtered))
            self._data_source = "pubmed"
            return filtered
        except Exception as error:  # pragma: no cover - network fallback
            LOGGER.error("PubMed retrieval failed, using offline cache: %s", error)
            return self._load_offline_articles(error)

    def design_project(self, articles: list[dict[str, Any]], background: str) -> str:
        return self._designer.design_direction(articles, background)

    def produce_outline(self, direction_summary: str) -> str:
        return self._designer.create_outline(direction_summary)

    def write_proposal(self, outline: str, output_dir: Path) -> Path:
        draft = self._writer.draft_proposal(outline, self._config.proposal_word_count)
        doc_path = output_dir / "乳腺癌基础与应用研究项目书.docx"
        return self._writer.save_to_word(draft, doc_path)

    def run(self, background: str) -> dict[str, Any]:
        output_dir = self._config.ensure_output_dir()
        articles = self.gather_articles()
        ranked_articles = self._ranker.rank(articles, background)
        top_articles = ranked_articles[: self._config.top_article_count]
        direction_summary = self.design_project(top_articles, background)
        outline = self.produce_outline(direction_summary)
        proposal_path = self.write_proposal(outline, output_dir)
        try:
            outline_data = json.loads(outline)
        except json.JSONDecodeError:
            LOGGER.warning("Outline was not valid JSON; falling back to empty structure")
            outline_data = {}
        payload = {
            "query": self._build_query(),
            "articles": ranked_articles,
            "top_articles": top_articles,
            "direction_summary": direction_summary,
            "outline": outline_data,
            "proposal_path": str(proposal_path),
            "data_source": self._data_source,
        }
        metadata_path = output_dir / "pipeline_result.json"
        metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("Pipeline complete. Results saved to %s", metadata_path)
        if self._gpt.offline_engaged():
            manual_prompt_path = self._emit_manual_prompt(
                output_dir,
                background,
                top_articles,
                direction_summary,
                outline_data,
            )
            payload["manual_prompt_path"] = str(manual_prompt_path)
            metadata_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        if self._config.github_enabled():
            LOGGER.info(
                "Uploading generated assets to GitHub repository %s/%s",
                self._config.github_owner,
                self._config.github_repo,
            )
            upload_assets(
                token=self._config.github_token or "",
                owner=self._config.github_owner or "",
                repo=self._config.github_repo or "",
                branch=self._config.github_branch,
                base_directory=self._config.github_directory,
                commit_message=self._config.github_commit_message,
                assets=[
                    GitHubAsset(metadata_path),
                    GitHubAsset(proposal_path),
                ],
            )
        return payload

    def _load_offline_articles(self, error: Exception) -> list[dict[str, Any]]:
        """Return the built-in offline article cache when PubMed is unreachable."""

        self._data_source = "offline_cache"
        note = (
            "PubMed 检索失败，已启用内置的乳腺癌转化研究文献缓存。"
            f" 错误信息: {error}"
        )
        self._gpt.add_offline_note("pubmed_offline", note)
        LOGGER.warning(
            "Using offline article cache with %d entries", len(OFFLINE_ARTICLES)
        )
        return [dict(article) for article in OFFLINE_ARTICLES]

    def _emit_manual_prompt(
        self,
        output_dir: Path,
        background: str,
        articles: list[dict[str, Any]],
        direction_summary: str,
        outline_data: dict[str, Any],
    ) -> Path:
        """Create a manual GPT prompt file for human-in-the-loop drafting."""

        prompt_path = output_dir / "GPT手动提示词.txt"
        article_blocks: list[str] = []
        for index, article in enumerate(articles[:20], 1):
            title = str(article.get("title") or "未提供标题").strip()
            journal = str(article.get("journal") or "").strip()
            pubdate = str(article.get("pubdate") or "").strip()
            pmid = str(article.get("pmid") or "").strip()
            citations = article.get("cited_by") or article.get("citations")
            abstract = str(article.get("abstract") or "摘要缺失").strip()
            keywords = article.get("keywords") or article.get("mesh_terms") or []
            keyword_text = "、".join(str(keyword) for keyword in keywords[:6]) or "乳腺癌"
            article_blocks.append(
                "\n".join(
                    [
                        f"{index}. 标题：{title}",
                        f"   期刊/年份：{journal} | {pubdate}",
                        f"   PMID：{pmid} | 引用信息：{citations if citations is not None else '未知'}",
                        f"   关键词：{keyword_text}",
                        "   摘要：" + abstract,
                    ]
                )
            )

        outline_lines = []
        for section, points in outline_data.items():
            outline_lines.append(f"{section}:")
            if isinstance(points, list):
                for point in points:
                    outline_lines.append(f"  - {point}")
            else:
                outline_lines.append(f"  - {points}")

        offline_notes = "\n".join(f"- {note}" for note in self._gpt.offline_notes())
        article_section = "\n".join(article_blocks)
        outline_section = "\n".join(outline_lines)

        prompt_text = dedent(
            f"""
            使用说明：
            1. 当前环境无法直接访问 OpenAI 接口，系统已生成线下兜底方案。
            2. 打开任意可用的 GPT 对话窗口（如浏览器中的 ChatGPT）。
            3. 复制“提示词主体”部分的全部文本并粘贴到聊天窗口，等待生成研究计划。
            4. 若 GPT 支持附件，可同时上传本目录下的 pipeline_result.json 与 Word 项目书草稿以供参考。

            离线触发记录：
            {offline_notes or '- 无额外说明'}

            -------------------- 提示词主体（复制以下全部内容） --------------------
            你是一名熟悉粤惠联合基金要求的乳腺癌转化医学专家。请基于以下背景和文献资料，撰写不少于 5000 字的中文科研项目书，结构需包含：研究背景、研究目的、研究内容与技术路线、可行性分析、研究基础与团队优势、预期成果与考核指标、实施计划与进度安排、参考文献。语言需正式、准确，突出惠州与深圳协同、临床转化价值及创新点。

            【项目背景】
            {background.strip() or '（待补充背景信息）'}

            【系统建议的研究方向概述】
            {direction_summary.strip()}

            【候选文献（近五年乳腺癌基础与应用研究，高引用优先）】
            {article_section or '暂无可用文献，请结合实际情况补充'}

            【建议提纲】
            {outline_section or '暂无提纲'}

            请严格按照上述材料完成项目书撰写，确保与粤惠联合基金（青年基金、重点项目等）对区域协同、经费额度与三年周期的要求匹配，可在文末给出拟采用的核心参考文献列表。
            ----------------------------------------------------------------------
            """
        ).strip()

        prompt_path.write_text(prompt_text, encoding="utf-8")
        LOGGER.info("Offline manual GPT prompt saved to %s", prompt_path)
        return prompt_path


def run_pipeline(background: str, **config_kwargs: Any) -> dict[str, Any]:
    config = PipelineConfig(**{key: value for key, value in config_kwargs.items() if value is not None})
    pipeline = Pipeline(config)
    return pipeline.run(background)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("background", help="Project background text in Chinese.")
    parser.add_argument("--query", default="breast cancer")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--retmax", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--github-owner")
    parser.add_argument("--github-repo")
    parser.add_argument("--github-branch", default="main")
    parser.add_argument("--github-directory", default="")
    parser.add_argument("--github-token")
    parser.add_argument("--github-commit-message", default="Add breast cancer automation outputs")
    args = parser.parse_args()

    try:
        payload = run_pipeline(
            args.background,
            query=args.query,
            years=args.years,
            retmax=args.retmax,
            output_dir=args.output_dir,
            github_owner=args.github_owner,
            github_repo=args.github_repo,
            github_branch=args.github_branch,
            github_directory=args.github_directory,
            github_token=args.github_token,
            github_commit_message=args.github_commit_message,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception as exc:  # pragma: no cover - CLI convenience
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)
