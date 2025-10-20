# Breast Cancer Research Automation Pipeline

This repository contains a Python pipeline that automates background research for
breast cancer translational projects. It performs the following steps:

1. Queries PubMed for highly cited breast cancer publications from the last five
   years with an applied or translational focus.
2. Summarises the articles with GPT to propose a feasible project direction in
   line with regional funding programmes such as the 粤惠联合基金.
3. Generates a structured outline and requests GPT to draft a ~5000 word project
   proposal in Chinese covering background, objectives, research design,
   feasibility, expected outcomes, and timelines.
4. Saves the integrated plan to a Word document alongside a JSON metadata file
   detailing the query and selected literature.

> **Note:** The scripts rely on external services (NCBI E-utilities and the
> OpenAI API). When the network or API key is unavailable, the pipeline falls
> back to a local heuristic mode that assembles readable direction summaries,
> outlines, and proposal drafts from the collected literature.

## Project structure

```
src/
├── config.py              # Dataclass storing pipeline configuration
├── article_ranker.py      # Heuristic scoring of PubMed articles
├── gpt_client.py          # Thin wrapper around the OpenAI ChatCompletion API
├── github_uploader.py     # Optional helpers to push outputs to GitHub
├── pipeline.py            # Orchestrates the end-to-end workflow
├── project_designer.py    # Derives project directions and outlines
├── proposal_writer.py     # Generates proposals and exports to Word
└── pubmed_scraper.py      # Queries and filters PubMed literature
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Set the following environment variables before executing the pipeline:

- `NCBI_API_KEY` (optional but recommended for higher PubMed rate limits).
- `OPENAI_API_KEY` (required to call the OpenAI API).
- `GITHUB_TOKEN` (optional, required only when you want the pipeline to upload
  generated files to GitHub automatically).

## Running the pipeline

默认检索词为“breast cancer”，会自动与转化/应用相关关键词组合进行筛选；如需自定义请使用 `--query` 选项覆盖。

流水线会对检索到的文献计算综合评分，指标包括：

- 与“转化/应用”相关关键词的覆盖度；
- 与输入项目背景文本的语义重合度（基于关键词匹配）；
- 发表的时间（越新越好）；
- 引用次数（对数缩放）。

可以在 `PipelineConfig` 中调整 `keyword_weight`、`background_weight`、`recency_weight`、`citation_weight` 和 `top_article_count` 等参数，以定制评分策略和传递给 GPT 的文献数量。

```bash
python -m src.pipeline "<项目背景文本>" \
  --years 5 \
  --retmax 50 \
  --output-dir "D:/基础work"
```

The script writes two files to the output directory:

- `乳腺癌基础与应用研究项目书.docx` – generated Word proposal.
- `pipeline_result.json` – metadata containing the selected literature and GPT
  outputs.

You can also import the module and call `run_pipeline` directly from Python
code:

```python
from src.pipeline import run_pipeline

background = "...粤惠联合基金等项目背景描述..."
result = run_pipeline(background, output_dir="D:/基础work")
print(result["proposal_path"])
```

### Automatically uploading outputs to GitHub

If you would like the generated Word document and JSON metadata to appear in a
GitHub repository immediately after the pipeline finishes, supply the GitHub
options via CLI flags or keyword arguments. The token must have `repo`
permissions for private repositories or `public_repo` for public repositories.

```bash
python -m src.pipeline "<项目背景文本>" \
  --github-owner <GitHub用户名> \
  --github-repo <仓库名> \
  --github-branch main \
  --github-directory automation-results \
  --github-token $GITHUB_TOKEN
```

When these values are provided (either through CLI flags or the corresponding
`PipelineConfig` keyword arguments), the pipeline pushes `pipeline_result.json`
and the generated Word proposal to the specified repository path using the
GitHub REST API.
