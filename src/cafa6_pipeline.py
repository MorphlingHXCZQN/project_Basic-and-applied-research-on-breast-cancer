"""High-level training pipeline for the CAFA-6 protein function challenge.

The original competition notebook that inspired this module was a single, very
long script that mixed together configuration, data wrangling, model
definition, and the cross-validation loop.  Reproducing or iterating on that
workflow in a research setting becomes difficult once you want to try variant
losses, change how embeddings are stacked, or run parameter sweeps locally
outside of Kaggle.

This module keeps the modelling ideas that worked well in the original script
— multi-source embedding fusion, logit adjustment, EMA, per-class
thresholding, etc. — but re-organises them into a small collection of
well-scoped classes.  Each class is responsible for a narrow part of the
workflow which makes it easier to reason about and test in isolation.  The
resulting code is more verbose than a notebook, yet it is considerably easier
to extend programmatically (e.g. for Optuna sweeps or ablation studies).

The entry-point is :func:`run_experiment`, which accepts an
``ExperimentConfig`` dataclass.  The config can be serialised/deserialised from
YAML/JSON or overridden from the command line if you build a CLI on top.  The
function executes the following steps:

1. Discover available embedding files (EMS2, ProtT5, ProtBERT) and align them
   on common protein identifiers.  When multiple embedding sources are
   available they are concatenated, standardised, reduced with PCA, and then
   standardised again.  If no external embeddings are present a sequence based
   fallback embedding is created.
2. Load the label space, apply minimum frequency filtering, and deduplicate the
   training rows by ``(EntryID, labelset)`` to stabilise cross-validation.
3. Train the 1-D residual squeeze-excitation network on stratified folds while
   applying logit adjustment, EMA, optional MixUp, and dynamic threshold
   search.  Each fold tracks its best performing weights according to the
   micro-F1 on the validation split.
4. Run inference on the test set and export a submission dataframe containing
   confident ``(protein, GO term)`` tuples.

The code is intentionally self-contained so that competition entrants can drop
the file into a repository, import it in a notebook, or execute it from a
command line wrapper without relying on Kaggle specific magic.  See the
``__main__`` guard at the bottom for an example of how to launch an experiment
using an ``ExperimentConfig`` populated from environment variables.
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# ---------------------------------------------------------------------------
# Configuration dataclasses


@dataclass
class EmbeddingSource:
    """Configuration for one family of pretrained protein embeddings."""

    name: str
    patterns: Sequence[str]


@dataclass
class DataConfig:
    """Paths and preprocessing rules for CAFA-6 data."""

    competition_dir: Path
    train_terms: str = "train_terms.tsv"
    min_label_count: int = 15
    target_num_labels: int = 600
    train_embeddings: Tuple[str, ...] = ("train_embeddings.npy", "train_embeds.npy")
    train_ids: Tuple[str, ...] = ("train_ids.npy", "train_protein_ids.npy")
    test_embeddings: Tuple[str, ...] = ("test_embeddings.npy", "test_embeds.npy")
    test_ids: Tuple[str, ...] = ("test_ids.npy", "test_protein_ids.npy")

    def __post_init__(self) -> None:
        self.competition_dir = Path(self.competition_dir)

    @property
    def train_dir(self) -> Path:
        return self.competition_dir / "Train"

    @property
    def test_dir(self) -> Path:
        return self.competition_dir / "Test"


@dataclass
class OptimiserConfig:
    lr: float = 1.5e-3
    weight_decay: float = 3e-4
    warmup_ratio: float = 0.1
    grad_accumulation: int = 2
    clip_grad_norm: float = 1.0


@dataclass
class TrainerConfig:
    epochs: int = 16
    folds: int = 3
    batch_size: int = 256
    use_amp: bool = torch.cuda.is_available()
    use_mixup: bool = False
    mixup_alpha: float = 0.4
    use_balanced_sampler: bool = False
    balanced_sampler_start: int = 1
    ema_decay: float = 0.999
    use_ema: bool = True
    early_stop_patience: int = 4
    tta: int = 1
    beta_f: float = 1.0
    thresh_topk: int = 400
    thresh_subsample: int = 25_000
    max_terms_per_id: int = 1_500
    logit_adjust_tau: float = 0.6
    infer_alpha: float = 0.3
    class_weight_power: float = 0.6
    smooth: float = 0.04
    asl_gamma_neg: float = 4.5
    asl_gamma_pos: float = 0.0
    asl_clip: float = 0.05


@dataclass
class ModelConfig:
    base_channels: int = 128
    depth: int = 10
    segment_length: int = 32
    dropout: float = 0.22
    head_hidden: int = 640


@dataclass
class ExperimentConfig:
    seed: int = 20251018
    data: DataConfig = field(default_factory=lambda: DataConfig(Path("/kaggle/input/cafa-6-protein-function-prediction")))
    optimiser: OptimiserConfig = field(default_factory=OptimiserConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    embedding_sources: Tuple[EmbeddingSource, ...] = (
        EmbeddingSource("EMS2", ("cafa-5-ems-2-embeddings-numpy", "ems2", "ems-2", "cafa6-ems2")),
        EmbeddingSource("ProtT5", ("t5embeds",)),
        EmbeddingSource("ProtBERT", ("protbert-embeddings-for-cafa5",)),
    )


# ---------------------------------------------------------------------------
# Utility helpers


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def _list_files(path: Path) -> List[str]:
    try:
        return sorted(os.listdir(path))
    except FileNotFoundError:
        return []


ACC_PATTERN = re.compile(r"(?:[A-NR-Z]\d{5}|[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-Z0-9]{6,10})")


def extract_accession(identifier: str) -> str:
    candidate = identifier.split("|")[-1]
    match = ACC_PATTERN.fullmatch(candidate) or ACC_PATTERN.search(identifier)
    return candidate.upper() if match else identifier.upper()


def normalise_ids(ids: Iterable[str]) -> np.ndarray:
    return np.array([extract_accession(str(x)) for x in ids], dtype=str)


# ---------------------------------------------------------------------------
# Embedding loading


class EmbeddingLoader:
    """Discover and align embedding files for a given split."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self._pre_mean: Optional[np.ndarray] = None
        self._pre_std: Optional[np.ndarray] = None
        self._post_mean: Optional[np.ndarray] = None
        self._post_std: Optional[np.ndarray] = None
        self._pca: Optional[PCA] = None
        self.used_sources: List[str] = []

    def _find_source_dir(self, patterns: Sequence[str]) -> Optional[Path]:
        base = Path("/kaggle/input")
        for pattern in patterns:
            for candidate in base.glob(f"*{pattern}*"):
                if candidate.is_dir():
                    return candidate
        return None

    def _load_single(self, split: str, source: EmbeddingSource) -> Optional[Tuple[str, np.ndarray, np.ndarray]]:
        directory = self._find_source_dir(source.patterns)
        if directory is None:
            return None

        embed_files = list(directory.glob(f"{split}_*.npy"))
        embed_path = None
        id_path = None
        for candidate in embed_files:
            name = candidate.name.lower()
            if any(key in name for key in ("embedding", "embed")) and embed_path is None:
                embed_path = candidate
            if "id" in name and id_path is None:
                id_path = candidate

        if embed_path is None or id_path is None:
            return None

        embeddings = np.load(embed_path).astype(np.float32)
        ids = normalise_ids(np.load(id_path).astype(str))
        return source.name, embeddings, ids

    def _collect_sources(self, split: str, allowed: Optional[Sequence[str]] = None) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        payloads: List[Tuple[str, np.ndarray, np.ndarray]] = []
        allowed_set = set(allowed) if allowed is not None else None
        for source in self.config.embedding_sources:
            if allowed_set is not None and source.name not in allowed_set:
                continue
            loaded = self._load_single(split, source)
            if loaded is not None:
                payloads.append(loaded)
        return payloads

    @staticmethod
    def _align_sources(payloads: List[Tuple[str, np.ndarray, np.ndarray]]) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
        if not payloads:
            return None

        base_name, base_embed, base_ids = payloads[0]
        aligned_features = [base_embed.astype(np.float32)]
        aligned_sources = [base_name]
        reference_index = pd.Series(np.arange(len(base_ids)), index=base_ids)

        for name, embed, ids in payloads[1:]:
            series = pd.Series(np.arange(len(ids)), index=ids)
            mapping = series.reindex(reference_index.index)
            mask = ~mapping.isna()
            if mask.mean() < 0.7:
                continue
            aligned_features.append(embed[mapping[mask].astype(int)])
            if len(aligned_sources) == 1:
                aligned_features[0] = aligned_features[0][mask.to_numpy()]
                base_ids = base_ids[mask.to_numpy()]
                reference_index = pd.Series(np.arange(len(base_ids)), index=base_ids)
            aligned_sources.append(name)

        if not aligned_features:
            return None

        fused = np.concatenate(aligned_features, axis=1)
        return fused, base_ids, aligned_sources

    def fit_transform(self, split: str = "train") -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
        payloads = self._collect_sources(split)
        aligned = self._align_sources(payloads)
        if aligned is None:
            return None

        fused, ids, sources = aligned
        self.used_sources = sources

        self._pre_mean = fused.mean(axis=0, keepdims=True)
        self._pre_std = fused.std(axis=0, keepdims=True) + 1e-6
        fused_norm = (fused - self._pre_mean) / self._pre_std

        dim = min(1024, fused_norm.shape[1])
        self._pca = PCA(n_components=dim, random_state=self.config.seed).fit(fused_norm)
        fused_pca = self._pca.transform(fused_norm).astype(np.float32)

        self._post_mean = fused_pca.mean(axis=0, keepdims=True)
        self._post_std = fused_pca.std(axis=0, keepdims=True) + 1e-6
        fused_final = (fused_pca - self._post_mean) / self._post_std
        return fused_final.astype(np.float32), ids, sources

    def transform(self, split: str = "test") -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
        if (
            not self.used_sources
            or self._pca is None
            or self._pre_mean is None
            or self._pre_std is None
            or self._post_mean is None
            or self._post_std is None
        ):
            raise RuntimeError("EmbeddingLoader.transform called before fit_transform")

        payloads = self._collect_sources(split, allowed=self.used_sources)
        aligned = self._align_sources(payloads)
        if aligned is None:
            return None

        fused, ids, sources = aligned
        fused_norm = (fused - self._pre_mean) / self._pre_std
        fused_pca = self._pca.transform(fused_norm).astype(np.float32)
        fused_final = (fused_pca - self._post_mean) / self._post_std
        return fused_final.astype(np.float32), ids, sources


# ---------------------------------------------------------------------------
# Sequence fallback features


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}


def _read_fasta(path: Path) -> Dict[str, str]:
    records: Dict[str, str] = {}
    current_id: Optional[str] = None
    sequence_chunks: List[str] = []
    with path.open("r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records[current_id] = "".join(sequence_chunks)
                current_id = line[1:].split()[0]
                sequence_chunks = []
            else:
                sequence_chunks.append(line)
    if current_id is not None:
        records[current_id] = "".join(sequence_chunks)
    return records


def _sequence_fallback(directory: Path) -> Tuple[np.ndarray, np.ndarray]:
    fasta_files = list(directory.glob("*.fa")) + list(directory.glob("*.fasta"))
    if not fasta_files:
        raise FileNotFoundError(f"No FASTA file found in {directory}, available files: {_list_files(directory)[:10]}")
    records = _read_fasta(fasta_files[0])
    ids = normalise_ids(records.keys())
    features = _featurise_sequences(list(records.values()))
    return features, ids


def _kmer_counts(seq: str, k: int) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for i in range(len(seq) - k + 1):
        token = seq[i : i + k]
        counts[token] = counts.get(token, 0) + 1
    return counts


def _amino_acid_frequency(seq: str) -> np.ndarray:
    vector = np.zeros(len(AMINO_ACIDS), dtype=np.float32)
    for char in seq:
        idx = AA_TO_INDEX.get(char)
        if idx is not None:
            vector[idx] += 1
    total = vector.sum()
    if total > 0:
        vector /= total
    return vector


def _featurise_sequences(sequences: Sequence[str]) -> np.ndarray:
    aa_matrix = np.vstack([_amino_acid_frequency(seq) for seq in sequences])
    top2, top3 = 200, 400
    unigram_counter: Dict[str, int] = {}
    bigram_counter: Dict[str, int] = {}
    for seq in sequences:
        for token, cnt in _kmer_counts(seq, 2).items():
            bigram_counter[token] = bigram_counter.get(token, 0) + cnt
        for token, cnt in _kmer_counts(seq, 3).items():
            unigram_counter[token] = unigram_counter.get(token, 0) + cnt

    vocab2 = [token for token, _ in sorted(bigram_counter.items(), key=lambda x: x[1], reverse=True)[:top2]]
    vocab3 = [token for token, _ in sorted(unigram_counter.items(), key=lambda x: x[1], reverse=True)[:top3]]

    def encode(seq: str, vocab: Sequence[str], k: int) -> np.ndarray:
        counter = _kmer_counts(seq, k)
        return np.array([counter.get(token, 0) for token in vocab], dtype=np.float32)

    mat2 = np.vstack([encode(seq, vocab2, 2) for seq in sequences])
    mat3 = np.vstack([encode(seq, vocab3, 3) for seq in sequences])
    stacked = np.hstack([aa_matrix, mat2, mat3]).astype(np.float32)
    return (stacked - stacked.mean(axis=0, keepdims=True)) / (stacked.std(axis=0, keepdims=True) + 1e-6)


# ---------------------------------------------------------------------------
# Dataset and loss utilities


class ProteinDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.features[index]), torch.from_numpy(self.labels[index])


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg: float, gamma_pos: float, clip: float, smooth: float, class_weight: Optional[np.ndarray]) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.smooth = smooth
        self.register_buffer("class_weight", torch.from_numpy(class_weight.astype(np.float32)) if class_weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.smooth > 0:
            targets = targets * (1 - self.smooth) + 0.5 * self.smooth

        prob_pos = torch.sigmoid(logits)
        prob_neg = 1 - prob_pos
        if self.clip > 0:
            prob_neg = torch.clamp(prob_neg + self.clip, max=1.0)

        loss_pos = targets * torch.log(torch.clamp(prob_pos, min=1e-8))
        loss_neg = (1 - targets) * torch.log(torch.clamp(prob_neg, min=1e-8))

        if self.gamma_pos > 0 or self.gamma_neg > 0:
            weight = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            focal = (prob_pos * targets + prob_neg * (1 - targets)).pow(weight)
            loss = -focal * (loss_pos + loss_neg)
        else:
            loss = -(loss_pos + loss_neg)

        if self.class_weight is not None:
            loss = loss * self.class_weight
        return loss.mean()


def make_balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    freq = labels.sum(axis=0).clip(min=1.0)
    weights = (labels / freq).sum(axis=1)
    weights = weights / (weights.mean() + 1e-12)
    return WeightedRandomSampler(torch.from_numpy(weights.astype(np.float64)), num_samples=len(labels), replacement=True)


# ---------------------------------------------------------------------------
# Model definition


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, dilation: int, drop: float, depthwise: bool = False) -> None:
        super().__init__()
        padding = (kernel // 2) * dilation
        groups = in_channels if depthwise else 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, padding=padding, groups=groups, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.bn(self.conv(x))))


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv1d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(self.pool(x))
        return x * scale


class ResidualSEBlock(nn.Module):
    def __init__(self, channels: int, dropout: float, dilation: int) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels, kernel=9, dilation=dilation, drop=dropout, depthwise=True)
        self.conv2 = ConvBNAct(channels, channels, kernel=5, dilation=dilation, drop=dropout, depthwise=True)
        self.se = SEBlock(channels)
        self.out = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        return self.out(x + residual)


class ResNetSE1D(nn.Module):
    def __init__(self, in_features: int, out_features: int, config: ModelConfig) -> None:
        super().__init__()
        self.segment_length = config.segment_length
        hidden = config.base_channels * config.segment_length
        self.project = nn.Linear(in_features, hidden)
        self.bn = nn.BatchNorm1d(hidden)
        self.act = nn.GELU()

        dilations = [1, 1, 2, 2, 4, 4, 8, 8, 8, 8][: config.depth]
        blocks = [ResidualSEBlock(config.base_channels, config.dropout, d) for d in dilations]
        self.backbone = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(config.base_channels),
            nn.Dropout(config.dropout),
            nn.Linear(config.base_channels, config.head_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden, out_features),
        )
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        x = self.act(self.bn(self.project(x)))
        x = x.view(batch, -1, self.segment_length)
        x = self.backbone(x)
        x = self.pool(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Training utilities


class EMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        return backup

    @staticmethod
    def restore(model: nn.Module, backup: Dict[str, torch.Tensor]) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in backup:
                param.data.copy_(backup[name])


class EarlyStopping:
    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_score = -1.0
        self.bad_epochs = 0
        self.best_state: Optional[Dict[str, torch.Tensor]] = None

    def step(self, score: float, state_dict: Dict[str, torch.Tensor]) -> bool:
        if score > self.best_score + 1e-6:
            self.best_score = score
            self.bad_epochs = 0
            self.best_state = {key: value.detach().cpu().clone() for key, value in state_dict.items()}
            return True
        self.bad_epochs += 1
        return False

    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


def multilabel_stratified_split(labels: np.ndarray, folds: int, seed: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    n_samples = labels.shape[0]
    order = np.argsort(labels.sum(axis=1))
    rng = np.random.default_rng(seed)
    fold_assignments = [[] for _ in range(folds)]
    fold_counts = np.zeros((folds, labels.shape[1]), dtype=float)

    for idx in order[::-1]:
        label = labels[idx]
        costs = np.array([np.sum((fold_counts[k] + label) ** 2) for k in range(folds)])
        best = np.flatnonzero(costs == costs.min())
        chosen = int(rng.choice(best))
        fold_assignments[chosen].append(idx)
        fold_counts[chosen] += label

    all_indices = np.arange(n_samples)
    for fold_indices in fold_assignments:
        val_idx = np.array(sorted(fold_indices), dtype=int)
        train_idx = np.setdiff1d(all_indices, val_idx)
        yield train_idx, val_idx


def compute_class_weights(labels: np.ndarray, power: float) -> np.ndarray:
    freq = labels.mean(axis=0).clip(min=1e-6)
    weights = (1.0 / freq) ** power
    return (weights / weights.mean()).astype(np.float32)


def init_bias_with_prior(model: nn.Module, prior: np.ndarray) -> None:
    prior = prior.clip(1e-6, 1 - 1e-6)
    bias = np.log(prior / (1 - prior)).astype(np.float32)
    with torch.no_grad():
        model.head[-1].bias.copy_(torch.from_numpy(bias).to(model.head[-1].bias.device))


def evaluate_logits(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            with torch.autocast(device_type="cuda", enabled=use_amp and device.type == "cuda", dtype=amp_dtype):
                logits = model(xb)
            logits_all.append(logits.float().cpu().numpy())
            labels_all.append(yb.numpy())
    return np.vstack(logits_all), np.vstack(labels_all)


def fit_temperature(logits: np.ndarray, targets: np.ndarray, device: torch.device, iters: int = 200) -> float:
    logits_tensor = torch.from_numpy(logits).to(device)
    targets_tensor = torch.from_numpy(targets).to(device)
    temperature = torch.ones(1, device=device, requires_grad=True)
    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.LBFGS([temperature], lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        loss = criterion(logits_tensor / temperature.clamp(min=1e-3), targets_tensor)
        loss.backward()
        return loss

    optimiser.step(closure)
    return float(temperature.detach().clamp(min=1e-3).cpu().item())


def best_threshold_for_binary(prob: np.ndarray, target: np.ndarray, beta: float) -> float:
    if prob.max() == prob.min():
        return 0.5

    precision, recall, thresholds = precision_recall_curve(target, prob)
    denom = beta ** 2 * precision + recall
    fbeta = (1 + beta ** 2) * (precision * recall) / np.maximum(denom, 1e-12)
    best_idx = int(np.nanargmax(fbeta))
    if best_idx >= len(thresholds):
        return 0.5
    return thresholds[best_idx]


def search_thresholds(proba: np.ndarray, labels: np.ndarray, beta: float, topk: int, subsample: int) -> np.ndarray:
    n_samples, n_labels = proba.shape
    if n_samples > subsample:
        indices = np.random.choice(n_samples, subsample, replace=False)
        proba = proba[indices]
        labels = labels[indices]

    frequency = labels.sum(axis=0)
    order = np.argsort(-frequency)
    selected = order[: min(topk, n_labels)]
    thresholds = np.full(n_labels, 0.5, dtype=np.float32)
    for label_idx in selected:
        thresholds[label_idx] = best_threshold_for_binary(proba[:, label_idx], labels[:, label_idx], beta)
    return thresholds


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Experiment runner


class CAFATrainer:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._terms_cache: Optional[pd.DataFrame] = None
        self._selected_terms: Optional[np.ndarray] = None
        self.embedding_loader = EmbeddingLoader(config)

    def _prepare_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        train_payload = self.embedding_loader.fit_transform("train")
        terms_df = self._load_terms()
        selected_terms = self._select_terms(terms_df)
        self._selected_terms = selected_terms
        term_to_idx = {term: idx for idx, term in enumerate(selected_terms)}

        id_to_labels: Dict[str, List[int]] = {}
        for entry, term in zip(terms_df["EntryID"], terms_df["term"]):
            if term in term_to_idx:
                id_to_labels.setdefault(entry, []).append(term_to_idx[term])

        if train_payload is not None:
            features, ids, used_sources = train_payload
            labels = np.zeros((len(ids), len(selected_terms)), dtype=np.float32)
            for idx, protein_id in enumerate(ids):
                label_indices = id_to_labels.get(protein_id, [])
                labels[idx, label_indices] = 1.0
            mask = labels.sum(axis=1) > 0
            return features[mask], labels[mask], ids[mask], "EMBEDDINGS"

        fallback_features, fallback_ids = _sequence_fallback(self.config.data.train_dir)
        labels = np.zeros((len(fallback_ids), len(selected_terms)), dtype=np.float32)
        for idx, protein_id in enumerate(fallback_ids):
            label_indices = id_to_labels.get(protein_id, [])
            labels[idx, label_indices] = 1.0
        mask = labels.sum(axis=1) > 0
        return fallback_features[mask], labels[mask], fallback_ids[mask], "SEQUENCE"

    def _prepare_test_features(self, source_tag: str) -> Tuple[np.ndarray, np.ndarray]:
        if source_tag == "EMBEDDINGS":
            payload = self.embedding_loader.transform("test")
            if payload is not None:
                features, ids, _ = payload
                return features, ids
        return _sequence_fallback(self.config.data.test_dir)

    def _load_terms(self) -> pd.DataFrame:
        if self._terms_cache is None:
            term_file = self.config.data.train_dir / self.config.data.train_terms
            df = pd.read_csv(term_file, sep="\t")
            cols = {col.lower(): col for col in df.columns}
            entry_col = cols.get("entryid") or cols.get("id")
            term_col = cols.get("term") or cols.get("go term") or cols.get("go_term")
            if entry_col is None or term_col is None:
                raise ValueError("train_terms.tsv must contain EntryID and term columns")
            df["EntryID"] = normalise_ids(df[entry_col])
            df["term"] = df[term_col].astype(str)
            self._terms_cache = df[["EntryID", "term"]]
        return self._terms_cache.copy()

    def _select_terms(self, terms_df: pd.DataFrame) -> np.ndarray:
        if self._selected_terms is None:
            counts = terms_df["term"].value_counts()
            frequent = counts[counts >= self.config.data.min_label_count].index.tolist()
            if len(frequent) >= self.config.data.target_num_labels:
                self._selected_terms = np.array(frequent[: self.config.data.target_num_labels], dtype=str)
            else:
                extra = [term for term in counts.index if term not in frequent]
                needed = self.config.data.target_num_labels - len(frequent)
                frequent.extend(extra[:needed])
                self._selected_terms = np.array(frequent, dtype=str)
        return self._selected_terms

    def _deduplicate(self, features: np.ndarray, labels: np.ndarray, ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        binary = (labels > 0).astype(np.uint8)
        hash_keys = [f"{pid}##{row.tobytes()}" for pid, row in zip(ids, binary)]
        _, unique_indices = np.unique(hash_keys, return_index=True)
        unique_indices = np.sort(unique_indices)
        return features[unique_indices], labels[unique_indices], ids[unique_indices]

    def _prepare_datasets(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        scaler: StandardScaler,
    ) -> Tuple[ProteinDataset, DataLoader, np.ndarray, np.ndarray]:
        train_features = scaler.transform(features[train_indices]).astype(np.float32)
        val_features = scaler.transform(features[val_indices]).astype(np.float32)
        train_labels = labels[train_indices]
        val_labels = labels[val_indices]

        train_dataset = ProteinDataset(train_features, train_labels)
        val_dataset = ProteinDataset(val_features, val_labels)

        kwargs = dict(batch_size=self.config.trainer.batch_size, num_workers=min(4, os.cpu_count() or 2), pin_memory=True)
        val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **kwargs)
        return train_dataset, val_loader, train_labels, val_labels

    def run(self) -> Dict[str, object]:
        set_seed(self.config.seed)
        features, labels, ids, source_tag = self._prepare_features()
        features, labels, ids = self._deduplicate(features, labels, ids)
        if len(features) == 0:
            train_dir = self.config.data.train_dir
            available = ", ".join(_list_files(train_dir)) or "<empty>"
            raise ValueError(
                "No training samples were loaded. "
                "Ensure the CAFA-6 competition data is present at "
                f"'{train_dir}'. Currently visible files: {available}"
            )

        if len(features) < self.config.trainer.folds:
            raise ValueError(
                "The requested number of folds exceeds the number of available "
                "training samples. Reduce `trainer.folds` (currently "
                f"{self.config.trainer.folds}) or make sure the dataset is "
                "mounted correctly."
            )

        test_features, test_ids = self._prepare_test_features(source_tag)

        folds = list(multilabel_stratified_split(labels, self.config.trainer.folds, self.config.seed))
        if any(len(train) == 0 or len(val) == 0 for train, val in folds):
            folds = [(train, val) for train, val in KFold(n_splits=self.config.trainer.folds, shuffle=True, random_state=self.config.seed).split(features)]

        models: List[nn.Module] = []
        scalers: List[StandardScaler] = []
        thresholds: List[np.ndarray] = []
        temperatures: List[float] = []
        priors: List[np.ndarray] = []
        fold_scores: List[float] = []

        for fold, (train_idx, val_idx) in enumerate(folds, start=1):
            scaler = StandardScaler().fit(features[train_idx])
            train_dataset, val_loader, train_labels, val_labels = self._prepare_datasets(features, labels, train_idx, val_idx, scaler)
            loader_kwargs = dict(batch_size=self.config.trainer.batch_size, num_workers=min(4, os.cpu_count() or 2), pin_memory=True, drop_last=False)

            model = ResNetSE1D(features.shape[1], labels.shape[1], self.config.model).to(self.device)
            if hasattr(torch, "compile"):
                try:
                    model = torch.compile(model)  # type: ignore[arg-type]
                except Exception:
                    pass

            class_weights = compute_class_weights(train_labels, self.config.trainer.class_weight_power)
            criterion = AsymmetricLoss(
                gamma_neg=self.config.trainer.asl_gamma_neg,
                gamma_pos=self.config.trainer.asl_gamma_pos,
                clip=self.config.trainer.asl_clip,
                smooth=self.config.trainer.smooth,
                class_weight=class_weights,
            )

            prior = train_labels.mean(axis=0).clip(1e-6, 1 - 1e-6)
            logit_prior = torch.from_numpy(np.log(prior / (1 - prior))).float().to(self.device)
            init_bias_with_prior(model, prior)

            optimiser = torch.optim.AdamW(model.parameters(), lr=self.config.optimiser.lr, weight_decay=self.config.optimiser.weight_decay)
            total_steps = max(1, len(train_loader) * self.config.trainer.epochs)
            warmup_steps = int(self.config.optimiser.warmup_ratio * total_steps)
            if warmup_steps > 0:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimiser,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=warmup_steps,
                )
                cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimiser,
                    T_max=max(1, total_steps - warmup_steps),
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimiser,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_steps],
                )
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=total_steps)
            scaler_amp = torch.cuda.amp.GradScaler(enabled=self.config.trainer.use_amp and self.device.type == "cuda")
            ema = EMA(model, self.config.trainer.ema_decay) if self.config.trainer.use_ema else None

            early_stopper = EarlyStopping(self.config.trainer.early_stop_patience)
            for epoch in range(1, self.config.trainer.epochs + 1):
                model.train()
                epoch_loss = 0.0
                if self.config.trainer.use_balanced_sampler and epoch >= self.config.trainer.balanced_sampler_start:
                    sampler = make_balanced_sampler(train_labels)
                    train_loader = DataLoader(train_dataset, sampler=sampler, **loader_kwargs)
                else:
                    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)

                num_batches = len(train_loader)
                for batch_idx, (xb, yb) in enumerate(train_loader, start=1):
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)

                    if self.config.trainer.use_mixup and self.config.trainer.mixup_alpha > 0:
                        lam = np.random.beta(self.config.trainer.mixup_alpha, self.config.trainer.mixup_alpha)
                        perm = torch.randperm(xb.size(0), device=xb.device)
                        xb = lam * xb + (1 - lam) * xb[perm]
                        yb = lam * yb + (1 - lam) * yb[perm]

                    with torch.autocast(device_type="cuda", enabled=self.config.trainer.use_amp and self.device.type == "cuda"):
                        logits = model(xb) - self.config.trainer.logit_adjust_tau * logit_prior
                        loss = criterion(logits, yb) / max(1, self.config.optimiser.grad_accumulation)

                    scaler_amp.scale(loss).backward()
                    should_step = batch_idx % self.config.optimiser.grad_accumulation == 0 or batch_idx == num_batches
                    if should_step:
                        if self.config.optimiser.clip_grad_norm:
                            scaler_amp.unscale_(optimiser)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.optimiser.clip_grad_norm)
                        scaler_amp.step(optimiser)
                        scaler_amp.update()
                        optimiser.zero_grad(set_to_none=True)
                        scheduler.step()
                        if ema:
                            ema.update(model)
                    epoch_loss += float(loss.detach().cpu())

                def _evaluate(candidate: nn.Module) -> float:
                    logits_val, y_val = evaluate_logits(candidate, val_loader, self.device, self.config.trainer.use_amp)
                    logits_val = logits_val - self.config.trainer.logit_adjust_tau * logit_prior.cpu().numpy()
                    probabilities = sigmoid(logits_val)
                    preds = (probabilities >= 0.5).astype(int)
                    return f1_score(y_val, preds, average="micro", zero_division=0)

                metrics: List[Tuple[str, float, Optional[Dict[str, torch.Tensor]]]] = [("raw", _evaluate(model), None)]
                if ema:
                    backup = ema.apply(model)
                    metrics.append(("ema", _evaluate(model), backup))
                    EMA.restore(model, backup)

                tag, best_score, backup_state = max(metrics, key=lambda x: x[1])
                if backup_state is not None:
                    EMA.restore(model, backup_state)

                early_stopper.step(best_score, model.state_dict())
                if early_stopper.should_stop():
                    break

            if early_stopper.best_state is not None:
                model.load_state_dict(early_stopper.best_state)

            logits_val, y_val = evaluate_logits(model, val_loader, self.device, self.config.trainer.use_amp)
            logits_val = logits_val - self.config.trainer.logit_adjust_tau * logit_prior.cpu().numpy()
            temperature = fit_temperature(logits_val, y_val, self.device)
            probabilities = sigmoid(logits_val / max(temperature, 1e-3))
            thresholds_fold = search_thresholds(
                probabilities,
                y_val,
                beta=self.config.trainer.beta_f,
                topk=self.config.trainer.thresh_topk,
                subsample=self.config.trainer.thresh_subsample,
            )

            models.append(model)
            scalers.append(scaler)
            thresholds.append(thresholds_fold)
            temperatures.append(temperature)
            priors.append(np.log(prior / (1 - prior)))
            fold_scores.append(early_stopper.best_score)

        probabilities = self._predict_test(models, scalers, temperatures, priors, thresholds, test_features)
        submission_rows = self._build_submission(probabilities, test_ids, thresholds)

        return {
            "models": models,
            "scalers": scalers,
            "thresholds": thresholds,
            "temperatures": temperatures,
            "priors": priors,
            "fold_scores": fold_scores,
            "test_ids": test_ids.tolist(),
            "submission": submission_rows,
        }

    def _predict_test(
        self,
        models: List[nn.Module],
        scalers: List[StandardScaler],
        temperatures: List[float],
        priors: List[np.ndarray],
        thresholds: List[np.ndarray],
        features: np.ndarray,
    ) -> np.ndarray:
        dummy_labels = np.zeros((len(features), thresholds[0].shape[0]), dtype=np.float32)
        loader_kwargs = dict(batch_size=self.config.trainer.batch_size, num_workers=min(4, os.cpu_count() or 2), pin_memory=True, shuffle=False)
        ensemble_probs = []

        for model, scaler, temp, prior in zip(models, scalers, temperatures, priors):
            dataset = ProteinDataset(scaler.transform(features).astype(np.float32), dummy_labels)
            loader = DataLoader(dataset, **loader_kwargs)
            logits, _ = evaluate_logits(model, loader, self.device, self.config.trainer.use_amp)
            logits = logits / max(temp, 1e-3)
            logits = logits - self.config.trainer.logit_adjust_tau * prior[None, :]
            if self.config.trainer.infer_alpha > 0:
                logits = logits - self.config.trainer.infer_alpha * prior[None, :]
            ensemble_probs.append(sigmoid(logits))

        return np.mean(ensemble_probs, axis=0)

    def _build_submission(self, probabilities: np.ndarray, ids: np.ndarray, thresholds: List[np.ndarray]) -> List[Dict[str, object]]:
        thresholds_avg = np.mean(np.stack(thresholds, axis=0), axis=0)
        submission: List[Dict[str, object]] = []
        terms = self._selected_terms if self._selected_terms is not None else self._select_terms(self._load_terms())

        for idx, protein_id in enumerate(ids):
            positive = np.where(probabilities[idx] >= thresholds_avg)[0]
            if len(positive) > self.config.trainer.max_terms_per_id:
                topk = np.argsort(-probabilities[idx, positive])[: self.config.trainer.max_terms_per_id]
                positive = positive[topk]
            for label_idx in positive:
                submission.append({"Id": protein_id, "GO term": terms[label_idx], "Confidence": float(probabilities[idx, label_idx])})
        return submission


# ---------------------------------------------------------------------------
# Public API


def run_experiment(config: ExperimentConfig) -> Dict[str, object]:
    """Execute the CAFA-6 pipeline and return a dictionary with artefacts."""

    trainer = CAFATrainer(config)
    return trainer.run()


def save_submission(rows: List[Dict[str, object]], output_path: Path) -> None:
    df = pd.DataFrame(rows, columns=["Id", "GO term", "Confidence"])
    df.to_csv(output_path, sep="\t", index=False, header=False)


def export_metadata(config: ExperimentConfig, artefacts: Dict[str, object], output_path: Path) -> None:
    def _default(obj):
        if hasattr(obj, "__dict__"):
            return {key: _default(value) for key, value in obj.__dict__.items()}
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    payload = {
        "config": _default(config),
        "fold_scores": artefacts.get("fold_scores"),
        "test_ids": artefacts.get("test_ids"),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover - example CLI usage
    default_dir = os.environ.get("CAFA6_COMPETITION_DIR", "/kaggle/input/cafa-6-protein-function-prediction")
    output_dir = Path(os.environ.get("CAFA6_OUTPUT_DIR", "."))
    config = ExperimentConfig(data=DataConfig(Path(default_dir)))
    artefacts = run_experiment(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_submission(artefacts["submission"], output_dir / "submission.tsv")
    export_metadata(config, artefacts, output_dir / "cafa6_metadata.json")
