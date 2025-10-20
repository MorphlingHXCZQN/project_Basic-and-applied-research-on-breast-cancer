"""Configuration helpers for the breast cancer research pipeline."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path("D:/基础work")


@dataclasses.dataclass
class PipelineConfig:
    """Runtime configuration for the automation pipeline."""

    query: str = "breast cancer"
    years: int = 5
    retmax: int = 50
    output_dir: Path = dataclasses.field(default_factory=lambda: DEFAULT_OUTPUT_DIR)
    email: str | None = None
    api_key: str | None = dataclasses.field(default_factory=lambda: os.getenv("NCBI_API_KEY"))
    openai_model: str = "gpt-4o"
    openai_temperature: float = 0.5
    proposal_word_count: int = 5000
    github_owner: str | None = None
    github_repo: str | None = None
    github_branch: str = "main"
    github_directory: str = ""
    github_token: str | None = dataclasses.field(
        default_factory=lambda: os.getenv("GITHUB_TOKEN")
    )
    github_commit_message: str = "Add breast cancer automation outputs"

    def ensure_output_dir(self) -> Path:
        """Create the output directory if it does not exist."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir

    def github_enabled(self) -> bool:
        """Return ``True`` when GitHub upload metadata is fully configured."""

        return bool(self.github_owner and self.github_repo and self.github_token)
