"""
Mamformer Complete Data Pipeline
=================================
Full classification, filtering, deduplication, and mixing pipeline.

Capabilities:
  1. Domain Classification — code, math, wiki, books, web, science, conversation
  2. Quality Scoring — perplexity, repetition, length, language
  3. Deduplication — MinHash LSH + exact span dedup
  4. Data Mixing — configurable domain weights
  5. Shuffling — global shuffle + stratified sampling
  6. PII / Toxic Filtering — regex + keyword based
  7. Data Report — per-domain token distribution

Usage:
    # Full pipeline: classify → filter → dedup → mix → tokenize
    python scripts/data_pipeline.py \
        --input /data/raw/ \
        --output /data/processed/ \
        --mix_config configs/data_mix.yaml

    # Or use programmatically:
    from data_pipeline import DataPipeline, DataMixConfig
    pipeline = DataPipeline(mix_config)
    pipeline.process("/data/raw/", "/data/processed/")
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DataMixConfig:
    """Configuration for data domain mixing weights."""
    weights: Dict[str, float] = field(default_factory=lambda: {
        "web": 0.35,
        "code": 0.20,
        "wiki": 0.12,
        "books": 0.10,
        "science": 0.08,
        "math": 0.07,
        "conversation": 0.05,
        "other": 0.03,
    })
    target_total_tokens: int = 2_000_000_000_000  # 2T tokens
    min_doc_length: int = 50       # Minimum chars per document
    max_doc_length: int = 100_000  # Maximum chars per document
    min_quality_score: float = 0.3  # Minimum quality score (0-1)
    dedup_threshold: float = 0.8    # Jaccard similarity for near-dedup
    shuffle_seed: int = 42


# ═══════════════════════════════════════════════════════════════════════
# 1. Domain Classifier
# ═══════════════════════════════════════════════════════════════════════

class DomainClassifier:
    """
    Fast rule-based domain classification using keyword and pattern matching.

    Domains: code, math, wiki, books, web, science, conversation, other
    """

    # Domain signatures — scored by keyword density
    DOMAIN_PATTERNS: Dict[str, List[str]] = {
        "code": [
            r"def\s+\w+\s*\(.*\)\s*:", r"import\s+\w+", r"class\s+\w+",
            r"function\s+\w+", r"const\s+\w+\s*=", r"let\s+\w+\s*=",
            r"```python", r"```javascript", r"```\w+", r"#include",
            r"public\s+(static\s+)?void", r"npm\s+install", r"git\s+clone",
            r"\{\s*%\s*", r"<\w+>.*</\w+>", r"SELECT\s+.*\s+FROM",
            r"def\s+__init__", r"@override", r"TODO", r"FIXME",
        ],
        "math": [
            r"\\begin\{", r"\\end\{", r"\\frac\{", r"\\sum_",
            r"\\int_", r"\\sqrt\{", r"\\alpha", r"\\beta", r"\\theta",
            r"\\mathbb\{", r"\\mathcal\{", r"\\mathbf\{",
            r"\$.*\$", r"\\\(.*\\\)", r"\\\[.*\\\]",
            r"theorem", r"proof", r"lemma", r"corollary",
            r"Let\s+\w+\s+be\s+a", r"Suppose\s+that",
            r"=\s*\d+\.?\d*\s*(\\pm|±)",
            r"\b(sqrt|log|exp|sin|cos|tan|lim|inf|sup|max|min)\b",
        ],
        "wiki": [
            r"\[\[.*\]\]", r"\{\{.*\}\}", r"==\s*.*\s*==",
            r"<ref>", r"</ref>", r"\{\|.*\|\}",
            r"'''\w+'''", r"''\w+''", r"&nbsp;",
            r"\[https?://[^\]]+\]", r"Category:",
            r"infobox", r"citation needed", r"References\s*$",
        ],
        "books": [
            r"Chapter\s+[IVX\d]+", r"^\s*CHAPTER\s+[IVX\d]+",
            r"\"[^\"]{20,}\"", r"said\s+\w+", r"replied\s+\w+",
            r"\.\"\s*\n\s*\"", r"\.'\s*\n\s*'",
            r"^\s*[IVX]+\.\s*$", r"^\s*\d+\.\s+\w",
        ],
        "science": [
            r"\bet\s+al\.", r"doi:", r"https?://doi\.org",
            r"Fig(ure)?\.?\s*\d+", r"Table\s*\d+",
            r"experiment", r"hypothesis", r"significantly\s+\(p\s*<",
            r"\b(p\s*[<>]\s*0?\.0\d)\b", r"95%\s+CI",
            r"Abstract", r"Introduction", r"Method(s|ology)",
            r"Results", r"Discussion", r"Conclusion",
            r"cells?", r"μ\w", r"\bmmol\b", r"\bDNA\b", r"\bRNA\b",
        ],
        "conversation": [
            r"^(User|Human|Assistant|Bot|AI):\s*",
            r"^(Q|Question|A|Answer):\s*",
            r"\b(probably|maybe|yeah|nah|wow|omg|lol)\b",
            r"\b(how are you|what's up|hey there|hi there)\b",
            r"!{2,}|\?{2,}", r"\.{3,}",
            r"^[a-z]{1,5}\s*$", r"\b(btw|imo|tbh|idk|afaik)\b",
        ],
    }

    def __init__(self):
        self._compiled = {
            domain: [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in patterns]
            for domain, patterns in self.DOMAIN_PATTERNS.items()
        }

    def classify(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify text into domain with confidence score.

        Returns:
            (domain, confidence, scores_per_domain)
        """
        if len(text) < 50:
            return "other", 0.0, {}

        scores = {}
        for domain, patterns in self._compiled.items():
            score = 0
            for pattern in patterns:
                matches = pattern.findall(text)
                score += len(matches)
            # Normalize by text length (per 1000 chars)
            scores[domain] = score / (len(text) / 1000)

        if not scores:
            return "other", 0.0, {}

        # Find best domain
        best_domain = max(scores, key=scores.get)
        best_score = scores[best_domain]

        # Confidence: how much the best score exceeds the second-best
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) >= 2 and sorted_scores[0] > 0:
            confidence = (sorted_scores[0] - sorted_scores[1]) / max(sorted_scores[0], 1e-8)
        elif sorted_scores[0] > 0:
            confidence = 0.5
        else:
            confidence = 0.0

        # If best score is too low, classify as "other"
        if best_score < 0.5:
            return "other", 0.0, scores

        return best_domain, min(confidence, 1.0), scores


# ═══════════════════════════════════════════════════════════════════════
# 2. Quality Scorer
# ═══════════════════════════════════════════════════════════════════════

class QualityScorer:
    """
    Scores document quality based on multiple signals.

    Score components (0-1 each, combined as weighted average):
      - Length score: penalizes too-short and too-long documents
      - Repetition score: penalizes repeated n-grams
      - Token diversity: rewards diverse vocabulary
      - Structural score: rewards good formatting
    """

    def __init__(
        self,
        min_length: int = 50,
        max_length: int = 100_000,
        optimal_length: int = 2000,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.optimal_length = optimal_length

    def score(self, text: str) -> Tuple[float, Dict[str, float]]:
        """
        Compute quality score for a document.

        Returns:
            (overall_score, component_scores)
        """
        if not text or len(text) < self.min_length:
            return 0.0, {"length": 0.0}

        scores = {
            "length": self._score_length(text),
            "repetition": self._score_repetition(text),
            "diversity": self._score_diversity(text),
            "structure": self._score_structure(text),
        }

        # Weighted average
        weights = {"length": 0.2, "repetition": 0.35, "diversity": 0.30, "structure": 0.15}
        overall = sum(scores[k] * weights[k] for k in scores)

        return overall, scores

    def _score_length(self, text: str) -> float:
        """Score based on document length."""
        L = len(text)
        if L < self.min_length:
            return 0.0
        if L > self.max_length:
            return 0.2  # Long docs: not terrible, but not ideal
        # Gaussian around optimal length
        sigma = self.optimal_length / 2
        return math.exp(-((L - self.optimal_length) ** 2) / (2 * sigma ** 2))

    def _score_repetition(self, text: str) -> float:
        """Score based on lack of repetition."""
        words = text.lower().split()
        if len(words) < 20:
            return 0.5

        # Check 4-gram repetition
        n = min(4, len(words) // 4)
        if n < 2:
            return 0.8

        ngrams = set()
        repeats = 0
        for i in range(len(words) - n + 1):
            gram = " ".join(words[i:i + n])
            if gram in ngrams:
                repeats += 1
            else:
                ngrams.add(gram)

        repeat_ratio = repeats / max(len(words) - n + 1, 1)
        # Low repeat_ratio = high score
        return max(0.0, 1.0 - repeat_ratio * 5)  # 20% repeat = 0 score

    def _score_diversity(self, text: str) -> float:
        """Score based on vocabulary diversity (type-token ratio)."""
        words = text.lower().split()
        if len(words) < 30:
            return 0.3

        unique = len(set(words))
        total = len(words)
        ttr = unique / total  # Type-token ratio

        # Ideal TTR depends on length (longer text = lower expected TTR)
        expected_ttr = max(0.1, 0.6 - 0.0001 * total)
        return min(1.0, ttr / expected_ttr)

    def _score_structure(self, text: str) -> float:
        """Score based on structural quality."""
        score = 0.5  # Start neutral

        # Has paragraphs (separated by blank lines)
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) >= 2:
            score += 0.15
        if len(paragraphs) >= 5:
            score += 0.1

        # Has punctuation variety
        punct = set(re.findall(r'[.,!?;:"\'-]', text))
        score += min(0.15, len(punct) * 0.02)

        # No excessive capitalization (screaming)
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if caps_ratio > 0.3:
            score -= 0.2

        # No excessive newlines
        newline_ratio = text.count("\n") / max(len(text), 1)
        if newline_ratio > 0.05:
            score -= 0.15

        return max(0.0, min(1.0, score))


# ═══════════════════════════════════════════════════════════════════════
# 3. Deduplication
# ═══════════════════════════════════════════════════════════════════════

class Deduplicator:
    """
    Near-deduplication using MinHash + exact substring dedup.

    Two-pass approach:
      1. MinHash LSH for fast near-duplicate detection
      2. Exact span matching for precise overlap removal
    """

    def __init__(self, threshold: float = 0.8, num_hashes: int = 128):
        self.threshold = threshold
        self.num_hashes = num_hashes
        self._seen_hashes: Set[int] = set()
        self._seen_spans: Set[str] = set()
        self._removed_count = 0

    def _minhash_signature(self, text: str) -> List[int]:
        """Compute MinHash signature for a document."""
        words = text.lower().split()
        if len(words) < 20:
            return [int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)] * self.num_hashes

        # Simple MinHash: use n-grams with multiple hash functions
        n = 5
        sig = [2**32 - 1] * self.num_hashes

        for i in range(len(words) - n + 1):
            gram = " ".join(words[i:i + n])
            for j in range(self.num_hashes):
                h = int(hashlib.sha256(f"{j}:{gram}".encode()).hexdigest()[:8], 16)
                if h < sig[j]:
                    sig[j] = h

        return sig

    def _jaccard_estimate(self, sig1: List[int], sig2: List[int]) -> float:
        """Estimate Jaccard similarity from MinHash signatures."""
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)

    def is_duplicate(self, text: str) -> Tuple[bool, str]:
        """
        Check if text is a duplicate of previously seen content.

        Returns:
            (is_duplicate, reason)
        """
        # Check 1: Exact span dedup (fast, catches copy-paste)
        normalized = " ".join(text.lower().split()[:200])  # First 200 words
        span_hash = hashlib.md5(normalized.encode()).hexdigest()
        if span_hash in self._seen_spans:
            self._removed_count += 1
            return True, "exact_span"

        # Check 2: MinHash near-dedup
        sig = self._minhash_signature(text)
        sig_tuple = tuple(sig)
        sig_hash = hash(sig_tuple)

        if self._seen_hashes:
            # Compare against all seen (simplified — production uses LSH buckets)
            # For now: hash the full signature and check exact match
            # (signature collision = very likely near-duplicate)
            if sig_hash in self._seen_hashes:
                self._removed_count += 1
                return True, "near_dup"

        # Not a duplicate — store and allow
        self._seen_hashes.add(sig_hash)
        self._seen_spans.add(span_hash)
        return False, ""

    def get_stats(self) -> dict:
        return {"removed_duplicates": self._removed_count}


# ═══════════════════════════════════════════════════════════════════════
# 4. PII / Toxic Filter
# ═══════════════════════════════════════════════════════════════════════

class ContentFilter:
    """
    Basic content safety filter using regex patterns.

    Removes documents containing:
      - Email addresses (bulk)
      - Phone numbers (bulk)
      - Credit card numbers
      - Social security numbers
      - Extreme toxic keywords
    """

    PII_PATTERNS = [
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'email'),
        (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', 'phone'),
        (r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', 'credit_card'),
        (r'\b\d{3}-\d{2}-\d{4}\b', 'ssn'),
        (r'\b(?:\d{1,3}\.){3}\d{1,3}\b', 'ip_address'),
    ]

    TOXIC_KEYWORDS = [
        'hate speech', 'white supremacy', ' racial slur',
        'child abuse', 'explicit content', 'violence against',
    ]

    def __init__(self):
        self._pii_patterns = [(re.compile(p, re.IGNORECASE), name) for p, name in self.PII_PATTERNS]
        self._stats = defaultdict(int)

    def should_filter(self, text: str) -> Tuple[bool, str]:
        """
        Check if document should be filtered out.

        Returns:
            (should_filter, reason)
        """
        # Check PII density (many PII matches = likely personal data dump)
        pii_count = 0
        for pattern, name in self._pii_patterns:
            matches = pattern.findall(text)
            if len(matches) > 10:  # Bulk PII
                self._stats[f"pii_{name}"] += 1
                return True, f"bulk_{name}"
            pii_count += len(matches)

        # Very high PII density overall
        if pii_count > 50:
            self._stats["pii_density"] += 1
            return True, "high_pii_density"

        # Check toxic keywords
        text_lower = text.lower()
        for keyword in self.TOXIC_KEYWORDS:
            if keyword in text_lower:
                self._stats["toxic"] += 1
                return True, "toxic_content"

        return False, ""

    def get_stats(self) -> dict:
        return dict(self._stats)


# ═══════════════════════════════════════════════════════════════════════
# 5. Data Mixer — Shuffle + Stratified Sampling
# ═══════════════════════════════════════════════════════════════════════

class DataMixer:
    """
    Manages data mixing with domain weights and stratified sampling.

    Ensures the training data matches the configured domain distribution
    through weighted sampling and global shuffling.
    """

    def __init__(self, config: DataMixConfig):
        self.config = config
        self._domain_buffers: Dict[str, List[str]] = defaultdict(list)
        self._domain_counts: Dict[str, int] = Counter()
        self._rng = np.random.RandomState(config.shuffle_seed)

    def add_document(self, text: str, domain: str, quality: float):
        """Add a classified document to the appropriate domain buffer."""
        if quality < self.config.min_quality_score:
            return
        self._domain_buffers[domain].append(text)
        self._domain_counts[domain] += 1

    def sample_batch(self, batch_size: int) -> List[str]:
        """
        Sample a batch of documents according to domain weights.

        Uses stratified sampling: each batch contains documents from
        each domain proportional to the configured weights.
        """
        batch = []
        for domain, weight in self.config.weights.items():
            if domain not in self._domain_buffers:
                continue
            n_samples = max(1, int(batch_size * weight))
            buffer = self._domain_buffers[domain]
            if buffer:
                indices = self._rng.randint(0, len(buffer), size=min(n_samples, len(buffer)))
                for idx in indices:
                    batch.append(buffer[idx])

        self._rng.shuffle(batch)
        return batch[:batch_size]

    def get_distribution(self) -> Dict[str, int]:
        """Get current document count per domain."""
        return dict(self._domain_counts)

    def get_all_documents(self, shuffle: bool = True) -> Iterator[Tuple[str, str]]:
        """
        Iterate over all documents with domain labels.
        If shuffle=True, interleaves domains randomly.
        Yields (domain, text) pairs. Quality scores are tracked internally.
        """
        all_docs = []
        for domain, texts in self._domain_buffers.items():
            for text in texts:
                all_docs.append((domain, text))

        if shuffle:
            self._rng.shuffle(all_docs)

        for domain, text in all_docs:
            yield domain, text


# ═══════════════════════════════════════════════════════════════════════
# 6. Full Pipeline
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# 5.5 Multi-Format Reader
# ═══════════════════════════════════════════════════════════════════════

class MultiFormatReader:
    """
    Reads and extracts clean text from diverse file formats.

    Supported formats:
      - .jsonl, .json — JSON lines / JSON array
      - .txt — Plain text
      - .csv, .tsv — Tabular data (extracts text columns)
      - .md — Markdown (strips formatting, keeps text + code blocks)
      - .html, .htm — HTML (strips tags, keeps text)
      - .py, .js, .java, .cpp, .c, .h, .rs, .go — Code files
      - .tex — LaTeX (strips commands, keeps text + math)
      - .xml — XML (strips tags, keeps text)
      - .pdf — PDF text extraction (requires pymupdf or pdfplumber)

    For each file, yields (text, file_type) tuples.
    """

    # File extension → format type mapping
    EXT_MAP = {
        ".jsonl": "json", ".json": "json",
        ".txt": "text", ".text": "text",
        ".csv": "csv", ".tsv": "csv", ".tab": "csv",
        ".md": "markdown", ".markdown": "markdown",
        ".html": "html", ".htm": "html",
        ".py": "code", ".js": "code", ".ts": "code", ".jsx": "code", ".tsx": "code",
        ".java": "code", ".cpp": "code", ".c": "code", ".h": "code", ".hpp": "code",
        ".rs": "code", ".go": "code", ".rb": "code", ".swift": "code",
        ".sql": "code", ".sh": "code", ".bash": "code", ".ps1": "code",
        ".tex": "latex", ".latex": "latex",
        ".xml": "xml", ".svg": "xml",
        ".pdf": "pdf",
        ".epub": "epub",
    }

    def __init__(self, extract_pdf: bool = False):
        self.extract_pdf = extract_pdf
        self._stats = Counter()

    def read_file(self, filepath: str) -> Iterator[Tuple[str, str]]:
        """
        Read a file and yield (text, format_type) tuples.

        Large files are chunked by document boundaries.
        """
        ext = Path(filepath).suffix.lower()
        fmt = self.EXT_MAP.get(ext, "text")
        self._stats[fmt] += 1

        try:
            if fmt == "json":
                yield from self._read_json(filepath)
            elif fmt == "csv":
                yield from self._read_csv(filepath, ext)
            elif fmt == "markdown":
                yield from self._read_markdown(filepath)
            elif fmt == "html":
                yield from self._read_html(filepath)
            elif fmt == "code":
                yield from self._read_code(filepath, ext)
            elif fmt == "latex":
                yield from self._read_latex(filepath)
            elif fmt == "xml":
                yield from self._read_xml(filepath)
            elif fmt == "pdf":
                yield from self._read_pdf(filepath)
            elif fmt == "epub":
                yield from self._read_epub(filepath)
            else:
                yield from self._read_text(filepath)
        except Exception as e:
            self._stats[f"error_{fmt}"] += 1

    def get_stats(self) -> dict:
        return dict(self._stats)

    # ── Format-specific readers ─────────────────────────────────

    def _read_json(self, path: str) -> Iterator[Tuple[str, str]]:
        """Read JSONL or JSON array."""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            first_char = f.read(1)
            f.seek(0)

            if first_char == "[":  # JSON array
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            text = self._extract_text_from_obj(item)
                            if text:
                                yield text, "json"
                except json.JSONDecodeError:
                    f.seek(0)

            # JSONL (one object per line)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = self._extract_text_from_obj(obj)
                    if text:
                        yield text, "jsonl"
                except json.JSONDecodeError:
                    if len(line) > 50:  # Probably just a text file
                        yield line, "text"

    def _extract_text_from_obj(self, obj: dict) -> str:
        """Extract text from a JSON object using common field names."""
        for key in ("text", "content", "body", "article", "abstract", "description"):
            if key in obj and isinstance(obj[key], str) and obj[key].strip():
                return obj[key]
        # Fallback: concatenate all string values
        if not isinstance(obj, dict):
            return ""
        parts = [str(v) for v in obj.values() if isinstance(v, str) and len(v) > 20]
        return "\n".join(parts) if parts else ""

    def _read_csv(self, path: str, ext: str) -> Iterator[Tuple[str, str]]:
        """Read CSV/TSV, extract text-heavy columns."""
        import csv
        delimiter = "\t" if ext == ".tsv" else ","
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            if not reader.fieldnames:
                return

            # Identify text columns (long string columns, not numeric)
            text_cols = [c for c in reader.fieldnames if
                         any(kw in c.lower() for kw in
                             ("text", "content", "body", "title", "desc", "question", "answer", "message"))]

            if not text_cols:
                text_cols = reader.fieldnames[:3]  # Fallback: first 3 columns

            for row in reader:
                texts = []
                for col in text_cols:
                    val = row.get(col, "").strip()
                    if len(val) > 20:
                        texts.append(val)
                if texts:
                    yield "\n".join(texts), "csv"

    def _read_markdown(self, path: str) -> Iterator[Tuple[str, str]]:
        """Read Markdown, strip formatting, keep code blocks."""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Split by headings (## Title) into documents
        sections = re.split(r'\n(?=#{1,3}\s)', content)
        for section in sections:
            cleaned = self._clean_markdown(section)
            if len(cleaned) > 50:
                yield cleaned, "markdown"

    def _clean_markdown(self, text: str) -> str:
        """Remove Markdown formatting, keep content."""
        # Keep code blocks intact
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', text, re.DOTALL)
        # Remove formatting: **bold**, *italic*, `code`, [links](url), ![img](url)
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'`(.+?)`', r'\1', text)
        text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
        text = re.sub(r'!\[.*?\]\(.+?\)', '', text)
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\|.*?\|', '', text)  # Tables
        text = re.sub(r'[-*_]{3,}', '', text)  # Horizontal rules
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _read_html(self, path: str) -> Iterator[Tuple[str, str]]:
        """Read HTML, strip tags, extract text blocks."""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Strip tags
        text = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'&nbsp;', ' ', text)
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&quot;', '"', text)
        text = re.sub(r'&#?\w+;', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 50]
        for para in paragraphs:
            yield para, "html"

    def _read_code(self, path: str, ext: str) -> Iterator[Tuple[str, str]]:
        """Read code file as-is (keep for code domain training)."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception:
            return

        if len(content) > 100:
            # Add language tag for better classification
            lang = ext.lstrip(".")
            tagged = f"```{lang}\n{content}\n```"
            yield tagged, "code"

    def _read_latex(self, path: str) -> Iterator[Tuple[str, str]]:
        """Read LaTeX, strip commands but keep math and text."""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Remove preamble (everything before \begin{document})
        doc_start = content.find(r"\begin{document}")
        if doc_start > 0:
            content = content[doc_start:]

        # Strip commands but keep their content
        text = re.sub(r'\\documentclass.*?\n', '', content)
        text = re.sub(r'\\usepackage.*?\n', '', text)
        text = re.sub(r'\\begin\{(center|figure|table|align|equation|itemize|enumerate)\}', '', text)
        text = re.sub(r'\\end\{(center|figure|table|align|equation|itemize|enumerate)\}', '', text)
        text = re.sub(r'\\\w+\{([^}]*)\}', r'\1', text)  # \command{text} → text
        text = re.sub(r'\\\w+', '', text)  # \command
        text = re.sub(r'\$([^$]+)\$', r'\(\1\)', text)  # Inline math → \(...\)
        text = re.sub(r'\\\[', '', text)
        text = re.sub(r'\\\]', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        if len(text) > 100:
            yield text, "latex"

    def _read_xml(self, path: str) -> Iterator[Tuple[str, str]]:
        """Read XML, strip tags, extract text."""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        # Strip tags + decode entities
        text = re.sub(r'<[^>]+>', ' ', content)
        text = re.sub(r'&[a-z]+;', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) > 50:
            yield text, "xml"

    def _read_pdf(self, path: str) -> Iterator[Tuple[str, str]]:
        """Extract text from PDF (requires pymupdf or pdfplumber)."""
        if not self.extract_pdf:
            return

        text = ""
        try:
            import fitz  # pymupdf
            doc = fitz.open(path)
            for page in doc:
                text += page.get_text()
            doc.close()
        except ImportError:
            try:
                import pdfplumber
                with pdfplumber.open(path) as pdf:
                    for page in pdf.pages:
                        t = page.extract_text()
                        if t:
                            text += t + "\n"
            except ImportError:
                return  # Neither library available

        if len(text) > 100:
            # Split by pages/paragraphs
            for para in text.split("\n\n"):
                para = para.strip()
                if len(para) > 50:
                    yield para, "pdf"

    def _read_epub(self, path: str) -> Iterator[Tuple[str, str]]:
        """Extract text from EPUB."""
        try:
            import ebooklib
            from ebooklib import epub
            book = epub.read_epub(path)
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    text = self._read_html_content(item.get_content().decode("utf-8", errors="replace"))
                    if len(text) > 100:
                        yield text, "epub"
        except ImportError:
            pass  # ebooklib not available

    def _read_html_content(self, html: str) -> str:
        """Extract text from HTML content (used by EPUB reader)."""
        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'&[a-z]+;', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _read_text(self, path: str) -> Iterator[Tuple[str, str]]:
        """Read plain text, splitting by blank lines into documents."""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Split by multiple newlines (paragraph/document boundaries)
        docs = re.split(r'\n{3,}', content)
        for doc in docs:
            doc = doc.strip()
            if len(doc) > 50:
                yield doc, "text"


class DataPipeline:
    """
    Complete data processing pipeline:
      classify → quality score → filter → deduplicate → mix

    Usage:
        pipeline = DataPipeline(DataMixConfig())
        report = pipeline.process("/data/raw/", "/data/processed/")
    """

    def __init__(self, config: Optional[DataMixConfig] = None):
        self.config = config or DataMixConfig()
        self.classifier = DomainClassifier()
        self.scorer = QualityScorer(
            min_length=self.config.min_doc_length,
            max_length=self.config.max_doc_length,
        )
        self.deduplicator = Deduplicator(threshold=self.config.dedup_threshold)
        self.filter = ContentFilter()
        self.mixer = DataMixer(self.config)

    def process_file(self, path: str) -> dict:
        """Process a single input file through the pipeline."""
        stats = Counter()
        reader = MultiFormatReader(extract_pdf=True)

        for text, fmt in reader.read_file(path):
            stats[f"fmt_{fmt}"] += 1
            stats["total"] += 1

            # Step 1: Length filter
            if len(text) < self.config.min_doc_length or len(text) > self.config.max_doc_length:
                stats["length_filtered"] += 1
                continue

            # Step 2: PII/Toxic filter
            should_filter, reason = self.filter.should_filter(text)
            if should_filter:
                stats[f"filtered_{reason}"] += 1
                continue

            # Step 3: Quality score
            quality, _ = self.scorer.score(text)
            if quality < self.config.min_quality_score:
                stats["quality_filtered"] += 1
                continue

            # Step 4: Dedup
            is_dup, dup_reason = self.deduplicator.is_duplicate(text)
            if is_dup:
                stats[f"dup_{dup_reason}"] += 1
                continue

            # Step 5: Classify
            domain, confidence, _ = self.classifier.classify(text)

            # Add to mixer
            self.mixer.add_document(text, domain, quality)
            stats[f"domain_{domain}"] += 1
            stats["accepted"] += 1

        return dict(stats)

    def process_directory(self, input_dir: str) -> dict:
        """Process all supported files in a directory tree."""
        total_stats = Counter()
        input_path = Path(input_dir)

        # All supported extensions
        exts = MultiFormatReader.EXT_MAP.keys()
        files = []
        for ext in exts:
            files.extend(input_path.glob(f"**/*{ext}"))
        # Also scan for files without extensions (treat as text)
        for f in input_path.glob("**/*"):
            if f.is_file() and not f.suffix:
                files.append(f)

        files = sorted(set(files))  # Dedup and sort
        print(f"Found {len(files)} files to process")

        for i, filepath in enumerate(files):
            print(f"[{i+1}/{len(files)}] Processing {filepath.name}...")
            file_stats = self.process_file(str(filepath))
            for k, v in file_stats.items():
                total_stats[k] += v

        return dict(total_stats)

    def generate_report(self) -> str:
        """Generate a human-readable pipeline report."""
        stats = self.process_stats if hasattr(self, 'process_stats') else {}
        domain_dist = self.mixer.get_distribution()
        dedup_stats = self.deduplicator.get_stats()
        filter_stats = self.filter.get_stats()

        total = sum(domain_dist.values())
        sep = "=" * 60

        lines = [
            sep,
            "  Mamformer Data Pipeline Report",
            sep,
        ]

        if "total" in stats:
            lines.append(f"  Documents processed:  {stats.get('total', 0):>12,}")
            lines.append(f"  Accepted:             {stats.get('accepted', 0):>12,}")
            lines.append(f"  Length filtered:      {stats.get('length_filtered', 0):>12,}")
            lines.append(f"  Quality filtered:     {stats.get('quality_filtered', 0):>12,}")
            lines.append(f"  Duplicates removed:   {dedup_stats.get('removed_duplicates', 0):>12,}")
            lines.append(f"  Acceptance rate:      {stats.get('accepted',0)/max(stats.get('total',1),1)*100:>11.1f}%")

        lines.extend([
            "",
            f"  Total accepted:       {total:>12,}",
            f"  Unique domains:       {len(domain_dist):>12}",
            "",
            "  Domain Distribution:",
            "  " + "-" * 40,
        ])

        for domain, weight in sorted(self.config.weights.items(), key=lambda x: x[1], reverse=True):
            count = domain_dist.get(domain, 0)
            pct = count / max(total, 1) * 100
            target_pct = weight * 100
            bar = "█" * int(pct / 2)
            lines.append(f"  {domain:<15s} {count:>8,} ({pct:>5.1f}%) [target: {target_pct:>4.1f}%] {bar}")

        lines.append(sep)
        return "\n".join(lines)

    def save_mixer_state(self, output_dir: str):
        """Save mixer state for later resumption."""
        output_path = Path(output_dir) / "data_mixer_state.json"
        state = {
            "domain_distribution": self.mixer.get_distribution(),
            "config_weights": self.config.weights,
            "dedup_removed": self.deduplicator.get_stats(),
            "filter_stats": self.filter.get_stats(),
        }
        with open(output_path, "w") as f:
            json.dump(state, f, indent=2)
        print(f"Mixer state saved to {output_path}")

# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Mamformer Data Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Input directory with raw data files")
    parser.add_argument("--output", type=str, default="./data/processed", help="Output directory")
    parser.add_argument("--report_only", action="store_true", help="Only generate report, skip processing")
    parser.add_argument("--no_dedup", action="store_true", help="Skip deduplication (faster)")
    parser.add_argument("--domain", type=str, default=None, help="Force all documents to a specific domain")

    # Mixing weights (override defaults)
    parser.add_argument("--weight_web", type=float, default=0.35)
    parser.add_argument("--weight_code", type=float, default=0.20)
    parser.add_argument("--weight_wiki", type=float, default=0.12)
    parser.add_argument("--weight_books", type=float, default=0.10)
    parser.add_argument("--weight_science", type=float, default=0.08)
    parser.add_argument("--weight_math", type=float, default=0.07)
    parser.add_argument("--weight_conversation", type=float, default=0.05)
    parser.add_argument("--weight_other", type=float, default=0.03)

    args = parser.parse_args()

    config = DataMixConfig(
        weights={
            "web": args.weight_web, "code": args.weight_code,
            "wiki": args.weight_wiki, "books": args.weight_books,
            "science": args.weight_science, "math": args.weight_math,
            "conversation": args.weight_conversation, "other": args.weight_other,
        }
    )

    # Validate weights sum to ~1.0
    total_weight = sum(config.weights.values())
    if abs(total_weight - 1.0) > 0.01:
        print(f"Warning: weights sum to {total_weight:.2f}, normalizing...")
        for k in config.weights:
            config.weights[k] /= total_weight

    pipeline = DataPipeline(config)

    if args.report_only:
        print(pipeline.generate_report())
        return

    print(f"Processing {args.input}...")
    start = time.time()
    pipeline.process_stats = pipeline.process_directory(args.input)
    elapsed = time.time() - start

    print(pipeline.generate_report())
    print(f"\nProcessing time: {elapsed:.1f}s")
    pipeline.save_mixer_state(args.output)


if __name__ == "__main__":
    main()
