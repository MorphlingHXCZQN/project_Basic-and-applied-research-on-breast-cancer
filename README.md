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
> OpenAI API). When running in an offline environment the code falls back to a
> stub response.

## Project structure

```
src/
├── config.py              # Dataclass storing pipeline configuration
├── gpt_client.py          # Thin wrapper around the OpenAI ChatCompletion API
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

## Running the pipeline

```bash
python -m src.pipeline "<项目背景文本>" \
  --query "breast cancer" \
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
result = run_pipeline(background, query="breast cancer", output_dir="D:/基础work")
print(result["proposal_path"])
```
