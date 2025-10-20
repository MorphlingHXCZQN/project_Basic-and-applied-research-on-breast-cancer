"""End-to-end automation pipeline for breast cancer project ideation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .config import DEFAULT_OUTPUT_DIR, PipelineConfig
from .gpt_client import GPTClient
from .project_designer import ProjectDesigner
from .proposal_writer import ProposalWriter
from .pubmed_scraper import (
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
        normalized = summarize_articles(summaries)
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
        direction_summary = self.design_project(articles, background)
        outline = self.produce_outline(direction_summary)
        proposal_path = self.write_proposal(outline, output_dir)
        payload = {
            "query": self._build_query(),
            "articles": articles,
            "direction_summary": direction_summary,
            "outline": json.loads(outline),
            "proposal_path": str(proposal_path),
        }
        metadata_path = output_dir / "pipeline_result.json"
        metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("Pipeline complete. Results saved to %s", metadata_path)
        return payload


def run_pipeline(background: str, **config_kwargs: Any) -> dict[str, Any]:
    config = PipelineConfig(**config_kwargs)
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
    args = parser.parse_args()

    try:
        payload = run_pipeline(
            args.background,
            query=args.query,
            years=args.years,
            retmax=args.retmax,
            output_dir=args.output_dir,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception as exc:  # pragma: no cover - CLI convenience
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)
