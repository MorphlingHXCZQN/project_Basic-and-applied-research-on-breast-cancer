"""End-to-end automation pipeline for breast cancer project ideation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
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

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._gpt = GPTClient(model=config.openai_model, temperature=config.openai_temperature)
        self._designer = ProjectDesigner(self._gpt)
        self._ranker = ArticleRanker(config)
        self._writer = ProposalWriter(self._gpt)

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
        pmids = search_pubmed(
            query,
            years=self._config.years,
            retmax=self._config.retmax,
            email=self._config.email,
            api_key=self._config.api_key,
        )
        summaries = fetch_summaries(pmids, email=self._config.email, api_key=self._config.api_key)
        try:
            details = fetch_details(pmids, email=self._config.email, api_key=self._config.api_key)
        except Exception as error:  # pragma: no cover - network fallback
            LOGGER.warning("Failed to fetch article details: %s", error)
            details = {}
        normalized = summarize_articles(summaries, details)
        filtered = filter_applied_research(normalized)
        LOGGER.info("Retrieved %d applied research articles", len(filtered))
        return filtered

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
        payload = {
            "query": self._build_query(),
            "articles": ranked_articles,
            "top_articles": top_articles,
            "direction_summary": direction_summary,
            "outline": json.loads(outline),
            "proposal_path": str(proposal_path),
        }
        metadata_path = output_dir / "pipeline_result.json"
        metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("Pipeline complete. Results saved to %s", metadata_path)
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
