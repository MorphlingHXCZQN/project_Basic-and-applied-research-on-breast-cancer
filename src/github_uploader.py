"""Helpers for uploading generated assets to a GitHub repository."""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import requests

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GitHubAsset:
    """Mapping between a local file and its target path in the repository."""

    local_path: Path
    repo_path: str | None = None

    def resolved_repo_path(self, base_directory: str = "") -> str:
        """Return the POSIX path that should be used inside the repository."""

        relative = self.repo_path or self.local_path.name
        relative = relative.replace("\\", "/")
        if base_directory:
            prefix = base_directory.strip("/")
            return f"{prefix}/{relative.lstrip('/')}"
        return relative.lstrip("/")


def _headers(token: str) -> Mapping[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _retrieve_sha(url: str, token: str, branch: str) -> str | None:
    response = requests.get(url, headers=_headers(token), params={"ref": branch}, timeout=15)
    if response.status_code == 200:
        data = response.json()
        return data.get("sha")
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return None


def _put_content(
    url: str,
    token: str,
    branch: str,
    commit_message: str,
    file_bytes: bytes,
    sha: str | None = None,
) -> None:
    payload = {
        "message": commit_message,
        "content": base64.b64encode(file_bytes).decode("ascii"),
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha
    response = requests.put(url, headers=_headers(token), json=payload, timeout=15)
    if response.status_code not in {200, 201}:
        response.raise_for_status()


def upload_assets(
    *,
    token: str,
    owner: str,
    repo: str,
    branch: str,
    base_directory: str,
    assets: Iterable[GitHubAsset],
    commit_message: str,
) -> None:
    """Upload the provided assets to the configured GitHub repository."""

    base_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
    for asset in assets:
        repo_path = asset.resolved_repo_path(base_directory)
        url = f"{base_url}/{repo_path}"
        file_bytes = asset.local_path.read_bytes()
        LOGGER.info("Uploading %s to GitHub repository %s/%s (branch %s)", repo_path, owner, repo, branch)
        sha = _retrieve_sha(url, token, branch)
        _put_content(url, token, branch, commit_message, file_bytes, sha)
