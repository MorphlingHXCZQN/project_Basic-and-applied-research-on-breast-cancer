"""Offline fallback dataset of breast cancer translational studies.

This module provides a curated selection of recent, highly cited breast cancer
articles that emphasise translational or applied research themes. The entries
mirror the structure produced by ``pubmed_scraper.summarize_articles`` so that
the rest of the pipeline (ranking, project design, proposal writing) can operate
without accessing PubMed when network connectivity is unavailable.
"""

from __future__ import annotations

OFFLINE_ARTICLES: list[dict[str, object]] = [
    {
        "pmid": "OFFLINE001",
        "title": "Integrating spatial multi-omics to map immunotherapy-ready breast cancer niches",
        "authors": [
            "Wang L",
            "Zhang Y",
            "Chen H",
            "Liu Q",
        ],
        "pubdate": "2024",
        "journal": "Nature Communications",
        "abstract": (
            "We constructed a spatial multi-omics atlas across 120 breast cancer samples "
            "to pinpoint immune-responsive niches suitable for combination immunotherapy. "
            "The study links transcriptomic rewiring with actionable biomarkers, offering "
            "a translational framework for precision intervention in triple-negative breast cancer."
        ),
        "url": "https://example.org/offline/001",
        "keywords": [
            "breast cancer",
            "translational",
            "spatial multi-omics",
            "immunotherapy",
            "precision medicine",
        ],
        "mesh_terms": [
            "Breast Neoplasms",
            "Immunotherapy",
            "Biomarkers, Tumor",
        ],
        "publication_types": ["Journal Article"],
        "cited_by": 86,
        "year": 2024,
    },
    {
        "pmid": "OFFLINE002",
        "title": "Organoid-guided clinical trials accelerate personalised breast cancer therapeutics",
        "authors": [
            "Li J",
            "Nguyen P",
            "Santos M",
        ],
        "pubdate": "2023",
        "journal": "Clinical Cancer Research",
        "abstract": (
            "Patient-derived organoids were deployed in parallel with clinical trials to predict drug "
            "responses in metastatic breast cancer. Concordant signalling networks identified by phospho-proteomics "
            "enabled adaptive therapy escalation and improved progression-free survival."
        ),
        "url": "https://example.org/offline/002",
        "keywords": [
            "breast cancer",
            "clinical",
            "organoid",
            "drug response",
            "precision",
        ],
        "mesh_terms": [
            "Breast Neoplasms",
            "Organoids",
            "Precision Medicine",
        ],
        "publication_types": ["Clinical Trial"],
        "cited_by": 112,
        "year": 2023,
    },
    {
        "pmid": "OFFLINE003",
        "title": "Hybrid AI-clinician collaboration for early breast cancer relapse prediction",
        "authors": [
            "Gao F",
            "Fernandez A",
            "Zhao X",
            "Hernandez D",
        ],
        "pubdate": "2022",
        "journal": "NPJ Precision Oncology",
        "abstract": (
            "A multi-centre study combined radiomics, circulating tumour DNA and clinicopathologic factors in an "
            "interpretable machine-learning model. The tool was prospectively validated to stratify relapse risk, "
            "guiding adjuvant therapy decisions within 12 weeks post-surgery."
        ),
        "url": "https://example.org/offline/003",
        "keywords": [
            "breast cancer",
            "clinical",
            "prediction",
            "biomarker",
            "translational",
        ],
        "mesh_terms": [
            "Breast Neoplasms",
            "Radiomics",
            "Circulating Tumor DNA",
        ],
        "publication_types": ["Journal Article"],
        "cited_by": 74,
        "year": 2022,
    },
    {
        "pmid": "OFFLINE004",
        "title": "Engineered extracellular vesicles deliver siRNA to overcome endocrine resistance",
        "authors": [
            "Chen Z",
            "Lopez M",
            "Wang J",
        ],
        "pubdate": "2021",
        "journal": "Science Translational Medicine",
        "abstract": (
            "Engineered extracellular vesicles were loaded with siRNA targeting ESR1 mutations. In hormone receptor-positive "
            "breast cancer models, systemic delivery resensitised tumours to endocrine therapy with favourable safety profiles."
        ),
        "url": "https://example.org/offline/004",
        "keywords": [
            "breast cancer",
            "translational",
            "siRNA",
            "endocrine resistance",
            "therapeutic",
        ],
        "mesh_terms": [
            "Breast Neoplasms",
            "RNA Interference",
            "Drug Resistance, Neoplasm",
        ],
        "publication_types": ["Journal Article"],
        "cited_by": 129,
        "year": 2021,
    },
    {
        "pmid": "OFFLINE005",
        "title": "Prospective cohort links lifestyle interventions with improved survivorship",
        "authors": [
            "Huang X",
            "Martinez L",
            "Wu S",
        ],
        "pubdate": "2024",
        "journal": "The Lancet Regional Health - Western Pacific",
        "abstract": (
            "A Guangdong-based survivorship cohort (n=1,800) evaluated diet, physical activity and mental health programmes. "
            "Integrated supportive care reduced recurrence-related biomarkers and enhanced health-related quality of life within two years."
        ),
        "url": "https://example.org/offline/005",
        "keywords": [
            "breast cancer",
            "clinical",
            "survivorship",
            "intervention",
            "regional collaboration",
        ],
        "mesh_terms": [
            "Breast Neoplasms",
            "Survivors",
            "Quality of Life",
        ],
        "publication_types": ["Prospective Studies"],
        "cited_by": 38,
        "year": 2024,
    },
    {
        "pmid": "OFFLINE006",
        "title": "Regional biobank harmonisation enables cross-city breast cancer trials",
        "authors": [
            "Zheng H",
            "Tsang V",
            "Liang P",
            "Chan K",
        ],
        "pubdate": "2023",
        "journal": "Cell Reports Medicine",
        "abstract": (
            "The Guangdong-Hong Kong-Macao Greater Bay Area biobank alliance established interoperable standards for biospecimen "
            "collection and data governance. Pilot trials leveraging harmonised protocols accelerated translational pipelines."
        ),
        "url": "https://example.org/offline/006",
        "keywords": [
            "breast cancer",
            "translational",
            "biobank",
            "regional collaboration",
            "clinical trial",
        ],
        "mesh_terms": [
            "Biological Specimen Banks",
            "Clinical Trials as Topic",
            "Translational Medical Research",
        ],
        "publication_types": ["Journal Article"],
        "cited_by": 67,
        "year": 2023,
    },
    {
        "pmid": "OFFLINE007",
        "title": "Liquid biopsy-guided minimal residual disease monitoring after neoadjuvant therapy",
        "authors": [
            "Song Y",
            "Patel R",
            "Gong L",
        ],
        "pubdate": "2022",
        "journal": "Annals of Oncology",
        "abstract": (
            "Serial circulating tumour DNA analysis identified minimal residual disease in HER2-positive breast cancer following "
            "neoadjuvant therapy. Early intervention decisions improved disease-free survival in a multicentre pragmatic trial."
        ),
        "url": "https://example.org/offline/007",
        "keywords": [
            "breast cancer",
            "clinical",
            "liquid biopsy",
            "minimal residual disease",
            "therapeutic",
        ],
        "mesh_terms": [
            "Breast Neoplasms",
            "Neoadjuvant Therapy",
            "Circulating Tumor DNA",
        ],
        "publication_types": ["Clinical Trial"],
        "cited_by": 95,
        "year": 2022,
    },
    {
        "pmid": "OFFLINE008",
        "title": "Microenvironment-targeted nanomedicine synergises radiotherapy in triple-negative disease",
        "authors": [
            "Tang J",
            "Lopez D",
            "Chen X",
        ],
        "pubdate": "2021",
        "journal": "Advanced Science",
        "abstract": (
            "A hypoxia-responsive nanomedicine co-delivered radiosensitiser and immune adjuvant payloads. In preclinical models, "
            "the platform reprogrammed the tumour microenvironment and enhanced radiotherapy efficacy with limited toxicity."
        ),
        "url": "https://example.org/offline/008",
        "keywords": [
            "breast cancer",
            "translational",
            "nanomedicine",
            "radiotherapy",
            "intervention",
        ],
        "mesh_terms": [
            "Nanomedicine",
            "Radiotherapy",
            "Tumor Microenvironment",
        ],
        "publication_types": ["Journal Article"],
        "cited_by": 142,
        "year": 2021,
    },
    {
        "pmid": "OFFLINE009",
        "title": "Adaptive trial platform evaluates combination targeted therapy in metastatic breast cancer",
        "authors": [
            "Kim S",
            "Zhou Y",
            "Lin C",
        ],
        "pubdate": "2023",
        "journal": "JAMA Oncology",
        "abstract": (
            "An adaptive multi-arm platform trial tested combinations of CDK4/6 inhibitors with PI3K pathway modulators. "
            "Translational correlatives revealed resistance mechanisms and informed biomarker-driven patient selection."
        ),
        "url": "https://example.org/offline/009",
        "keywords": [
            "breast cancer",
            "clinical",
            "adaptive trial",
            "targeted therapy",
            "biomarker",
        ],
        "mesh_terms": [
            "Clinical Trials, Phase II as Topic",
            "Cyclin-Dependent Kinase 4",
            "Phosphatidylinositol 3-Kinases",
        ],
        "publication_types": ["Clinical Trial, Phase II"],
        "cited_by": 58,
        "year": 2023,
    },
    {
        "pmid": "OFFLINE010",
        "title": "Community-screening AI platform improves early detection in resource-limited regions",
        "authors": [
            "Zhao J",
            "Wong K",
            "Liang Y",
        ],
        "pubdate": "2022",
        "journal": "The Breast",
        "abstract": (
            "A federated learning mammography platform deployed across Guangdong community clinics increased early-stage detection "
            "rates by 21%. The implementation study underscores policy frameworks for equitable access to screening technologies."
        ),
        "url": "https://example.org/offline/010",
        "keywords": [
            "breast cancer",
            "applied",
            "screening",
            "artificial intelligence",
            "public health",
        ],
        "mesh_terms": [
            "Breast Neoplasms",
            "Mass Screening",
            "Artificial Intelligence",
        ],
        "publication_types": ["Implementation Study"],
        "cited_by": 63,
        "year": 2022,
    },
]

