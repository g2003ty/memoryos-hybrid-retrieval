"""MemoryOS end-to-end reliability selection evaluation.

This is an independent evaluation entry point.  It reuses the existing
MemoryOS evaluation components but does not modify MemoryOS core modules,
the feature extractor, the dataset construction code, or predictor training.

The default ``all`` mode performs the following leakage-safe protocol:

1. Load the existing MemoryOS snapshot for every LoCoMo sample.
2. Split queries into validation and untouched test partitions with seed 42.
3. Extract the initial candidate queue and the three MemoryVerifier features
   once per query.
4. Search a hard-filter threshold on validation queries only.
5. Lock the selected ``tau_val`` and evaluate test queries with:
   baseline, HF-0.5 (fixed comparator), HF-tau_val (final policy), and RR.

The fixed HF-0.5 and RR conditions are pre-registered comparators.  They are
reported on the test split but are never used to choose the final test policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# The script is intentionally executable from either the repository root or
# the eval directory.  Existing eval modules use local absolute imports.
EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

FEATURE_ORDER = ["relevance_score", "faithfulness_score", "conflict_score"]
RANDOM_SEED = 42
DEFAULT_DATASET = EVAL_DIR / "locomo10.json"
DEFAULT_MODEL = EVAL_DIR / "memory_quality_model.pkl"
DEFAULT_OUTPUT_DIR = EVAL_DIR / "e2e_results"
DEFAULT_STORAGE_ROOT = EVAL_DIR / "e2e_storage"
DEFAULT_THRESHOLD_GRID = tuple(round(value / 10.0, 1) for value in range(1, 10))
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.inferera.com/v1")


def _json_default(value: Any) -> Any:
    """Convert common NumPy/scalar values without hiding serialization errors."""
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    """Atomically persist JSON so an interrupted run cannot corrupt a checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def safe_path_component(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def install_retrieval_embedding_cache() -> Dict[str, Any]:
    """Install a process-local, lazy singleton cache for retrieval embeddings.

    The existing MemoryOS evaluation modules import ``get_embedding`` directly.
    Patching before those modules are loaded preserves the embedding algorithm
    while preventing SentenceTransformer from being constructed for every text.
    One model instance is retained for each distinct model name, and identical
    text vectors are reused for the lifetime of the evaluation process.
    """
    import utils as eval_utils

    model_cache: Dict[str, Any] = {}
    vector_cache: Dict[Tuple[str, str], Any] = {}

    def cached_get_embedding(text: Any, model_name: str = "all-MiniLM-L6-v2") -> Any:
        text_value = str(text)
        cache_key = (model_name, text_value)
        if cache_key in vector_cache:
            return vector_cache[cache_key]
        model = model_cache.get(model_name)
        if model is None:
            from sentence_transformers import SentenceTransformer

            print(f"Embedding cache: lazy-loading {model_name} (once per process).")
            model = SentenceTransformer(model_name)
            model_cache[model_name] = model
        vector = model.encode([text_value], convert_to_numpy=True)[0]
        vector_cache[cache_key] = vector
        return vector

    eval_utils.get_embedding = cached_get_embedding
    # Standalone execution imports these modules after this function.  The
    # assignments also cover interactive/reused Python processes safely.
    for module_name in ("mid_term_memory", "long_term_memory"):
        module = sys.modules.get(module_name)
        if module is not None:
            setattr(module, "get_embedding", cached_get_embedding)
    return model_cache


class EvaluationProgress:
    """One overall progress meter with conversation/query/strategy context."""

    def __init__(self, total: int, initial: int = 0):
        self.total = max(0, int(total))
        self.completed = max(0, min(int(initial), self.total))
        self.initial = self.completed
        self.started_at = time.monotonic()
        self._bar = None
        self._last_fallback_percent = -1
        try:
            from tqdm.auto import tqdm

            self._bar = tqdm(
                total=self.total,
                initial=self.completed,
                unit="step",
                dynamic_ncols=True,
                bar_format=(
                    "{desc}\nOverall progress: {bar} {percentage:6.2f}% "
                    "| Elapsed: {elapsed} | Estimated remaining: {remaining}"
                ),
            )
        except Exception:
            print("tqdm is unavailable; using the built-in progress display.")

    @staticmethod
    def _description(query: Mapping[str, Any], strategy: str) -> str:
        return (
            f"Conversation: {query['conversation_position']}/{query['conversation_total']} | "
            f"Current strategy: {strategy} | "
            f"Query: {query['query_position_in_conversation']}/{query['queries_in_conversation']}"
        )

    @staticmethod
    def _duration(seconds: Optional[float]) -> str:
        if seconds is None or not math.isfinite(seconds):
            return "--:--:--"
        value = max(0, int(seconds))
        hours, remainder = divmod(value, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def begin(self, query: Mapping[str, Any], strategy: str) -> None:
        description = self._description(query, strategy)
        if self._bar is not None:
            self._bar.set_description_str(description, refresh=True)
            return
        percent = int(100 * self.completed / max(1, self.total))
        if percent != self._last_fallback_percent:
            elapsed = time.monotonic() - self.started_at
            session_completed = self.completed - self.initial
            rate = session_completed / elapsed if elapsed > 0 else 0.0
            remaining = (self.total - self.completed) / rate if rate > 0 else None
            filled = int(round(20 * self.completed / max(1, self.total)))
            bar = "#" * filled + "-" * (20 - filled)
            print(
                f"{description} | Overall: [{bar}] {percent}% | "
                f"Elapsed: {self._duration(elapsed)} | ETA: {self._duration(remaining)}"
            )
            self._last_fallback_percent = percent

    def complete(self) -> None:
        if self.completed >= self.total:
            return
        self.completed += 1
        if self._bar is not None:
            self._bar.update(1)

    def credit_resumed(self, count: int, label: str) -> None:
        credit = max(0, min(int(count), self.total - self.completed))
        if not credit:
            return
        self.completed += credit
        self.initial += credit
        if self._bar is not None:
            self._bar.set_description_str(f"Resuming completed checkpoints: {label}", refresh=False)
            self._bar.n = self.completed
            self._bar.last_print_n = self.completed
            self._bar.refresh()
        else:
            print(f"Resumed {credit} completed work units from {label} checkpoints.")

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
        elif self.completed >= self.total:
            elapsed = time.monotonic() - self.started_at
            print(
                f"Overall progress: 100% | Elapsed: {self._duration(elapsed)} | "
                "Estimated remaining: 00:00:00"
            )


class ConversationCheckpoint:
    """Query-batched, conversation-partitioned, resumable JSON checkpoints."""

    VERSION = 1

    def __init__(self, output_dir: Path, directory: str, strategy: str,
                 run_fingerprint: str, expected_queries: Sequence[Mapping[str, Any]],
                 batch_size: int, parameters: Optional[Mapping[str, Any]] = None):
        self.directory = output_dir / directory
        self.strategy = strategy
        self.run_fingerprint = run_fingerprint
        self.batch_size = max(1, int(batch_size))
        self.parameters = dict(parameters or {})
        self.query_by_id = {str(query["query_id"]): query for query in expected_queries}
        self.expected_by_sample: Dict[str, List[str]] = defaultdict(list)
        for query in expected_queries:
            self.expected_by_sample[str(query["sample_id"])].append(str(query["query_id"]))
        self.records: Dict[str, Dict[str, Any]] = {}
        self.pending_by_sample: Dict[str, int] = defaultdict(int)
        self._load()

    def _path(self, sample_id: str) -> Path:
        query_id = self.expected_by_sample[sample_id][0]
        query = self.query_by_id[query_id]
        return self.directory / (
            f"conv_{int(query['conversation_position']):02d}_"
            f"{safe_path_component(sample_id)}.json"
        )

    def _load(self) -> None:
        for sample_id in self.expected_by_sample:
            path = self._path(sample_id)
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise RuntimeError(f"Invalid checkpoint JSON: {path}") from exc
            if payload.get("checkpoint_version") != self.VERSION:
                raise RuntimeError(f"Checkpoint version mismatch: {path}")
            if payload.get("run_fingerprint") != self.run_fingerprint:
                raise RuntimeError(
                    f"Checkpoint configuration mismatch: {path}. "
                    "Use a new --output-dir for a different experiment configuration."
                )
            if payload.get("strategy") != self.strategy or payload.get("parameters", {}) != self.parameters:
                raise RuntimeError(f"Checkpoint strategy metadata mismatch: {path}")
            expected_ids = set(self.expected_by_sample[sample_id])
            for record in payload.get("results", []):
                query_id = str(record.get("query_id", ""))
                if query_id not in expected_ids:
                    raise RuntimeError(f"Unexpected query_id {query_id!r} in {path}")
                self.records[query_id] = record

    def get(self, query_id: str) -> Optional[Dict[str, Any]]:
        return self.records.get(str(query_id))

    def add(self, query: Mapping[str, Any], record: Mapping[str, Any]) -> None:
        query_id = str(query["query_id"])
        sample_id = str(query["sample_id"])
        if query_id not in self.query_by_id:
            raise KeyError(f"Query {query_id!r} does not belong to checkpoint {self.strategy}")
        stored = dict(record)
        stored["query_id"] = query_id
        self.records[query_id] = stored
        self.pending_by_sample[sample_id] += 1
        expected_ids = self.expected_by_sample[sample_id]
        sample_complete = all(item in self.records for item in expected_ids)
        if self.pending_by_sample[sample_id] >= self.batch_size or sample_complete:
            self.flush_sample(sample_id)

    def flush_sample(self, sample_id: str) -> None:
        expected_ids = self.expected_by_sample.get(sample_id, [])
        if not expected_ids:
            return
        results = [self.records[query_id] for query_id in expected_ids if query_id in self.records]
        write_json(self._path(sample_id), {
            "checkpoint_version": self.VERSION,
            "run_fingerprint": self.run_fingerprint,
            "strategy": self.strategy,
            "parameters": self.parameters,
            "sample_id": sample_id,
            "conversation_position": self.query_by_id[expected_ids[0]]["conversation_position"],
            "completed_query_count": len(results),
            "expected_query_count": len(expected_ids),
            "complete": len(results) == len(expected_ids),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "results": results,
        })
        self.pending_by_sample[sample_id] = 0

    def flush_all(self) -> None:
        for sample_id in self.expected_by_sample:
            if self.pending_by_sample[sample_id]:
                self.flush_sample(sample_id)

    def ordered_records(self, queries: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        return [self.records[str(query["query_id"])] for query in queries
                if str(query["query_id"]) in self.records]

    def __len__(self) -> int:
        return len(self.records)


class ConfiguredLLMClient:
    """Small adapter matching the client interface used by existing eval code."""

    def __init__(self, api_key: Optional[str], base_url: str):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError("The 'openai' package is required for E2E evaluation") from exc
        if not api_key:
            raise RuntimeError(
                "No API key supplied. Set OPENAI_API_KEY or pass --api-key. "
                "The script never writes the key to experiment artifacts."
            )
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def chat_completion(self, model: str, messages: Sequence[Mapping[str, str]],
                        temperature: float = 0.7, max_tokens: int = 2000) -> str:
        response = self._client.chat.completions.create(
            model=model,
            messages=list(messages),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        return "" if content is None else str(content).strip()


class DisabledLLMClient:
    """Guard used by snapshot-only diagnostics, which must never call an LLM."""

    def chat_completion(self, *args: Any, **kwargs: Any) -> str:
        raise RuntimeError("LLM calls are disabled during --debug-snapshot-only")


def parse_float_list(value: str) -> List[float]:
    values = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        number = float(item)
        if not 0.0 <= number <= 1.0:
            raise argparse.ArgumentTypeError("Thresholds must be in [0, 1]")
        values.append(number)
    if not values:
        raise argparse.ArgumentTypeError("At least one threshold is required")
    return sorted(set(round(number, 6) for number in values))


def load_locomo(path: Path) -> List[Dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("LoCoMo dataset must be a non-empty JSON list")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or "conversation" not in row or "qa" not in row:
            raise ValueError(f"LoCoMo row {index} is missing conversation or qa")
    return rows


def load_predictor(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        artifact = pickle.load(handle)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError("Predictor artifact must contain a 'model' entry")
    if artifact.get("feature_order") != FEATURE_ORDER:
        raise ValueError(
            f"Predictor feature order mismatch: {artifact.get('feature_order')!r}; "
            f"expected {FEATURE_ORDER!r}"
        )
    model = artifact["model"]
    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise ValueError("Predictor model must expose predict_proba() and classes_")
    if 1 not in list(model.classes_):
        raise ValueError("Predictor model has no reliable class (label=1)")
    return artifact


def build_source_map(conversation_data: Mapping[str, Any]) -> Dict[Tuple[str, str], List[str]]:
    """Map a constructed QA page back to LoCoMo evidence dialogue ids."""
    source_map: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    speaker_a = conversation_data["speaker_a"]
    session_keys = [
        key for key in conversation_data
        if key.startswith("session_") and not key.endswith("_date_time")
    ]

    def session_number(key: str) -> int:
        match = re.match(r"session_(\d+)$", key)
        return int(match.group(1)) if match else 10**9

    for session_key in sorted(session_keys, key=session_number):
        dialogs = conversation_data[session_key]
        for index, dialog in enumerate(dialogs):
            if dialog.get("speaker") != speaker_a:
                continue
            user_text = dialog.get("text", "")
            if dialog.get("blip_caption"):
                user_text = f"{user_text} (image description: {dialog['blip_caption']})"
            answer_text = ""
            ids = []
            if dialog.get("dia_id"):
                ids.append(str(dialog["dia_id"]))
            if index + 1 < len(dialogs) and dialogs[index + 1].get("speaker") != speaker_a:
                answer_text = dialogs[index + 1].get("text", "")
                if dialogs[index + 1].get("blip_caption"):
                    answer_text = f"{answer_text} (image description: {dialogs[index + 1]['blip_caption']})"
                if dialogs[index + 1].get("dia_id"):
                    ids.append(str(dialogs[index + 1]["dia_id"]))
            source_map[(user_text, answer_text)].extend(ids)
    return dict(source_map)


def normalize_evidence(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)]


def flatten_queries(dataset: Sequence[Mapping[str, Any]], validation_ratio: float,
                    seed: int, limit: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    queries: List[Dict[str, Any]] = []
    for sample_index, sample in enumerate(dataset):
        sample_id = str(sample.get("sample_id", f"sample_{sample_index}"))
        for qa_index, qa in enumerate(sample.get("qa", [])):
            answer = qa.get("answer", "")
            if answer in (None, ""):
                answer = qa.get("adversarial_answer", "")
            queries.append({
                "order": len(queries),
                "sample_index": sample_index,
                "sample_id": sample_id,
                "qa_index": qa_index,
                "query_id": f"{sample_id}:{qa_index}",
                "query": str(qa.get("question", "")),
                "reference_answer": str(answer),
                "category": qa.get("category"),
                "evidence": normalize_evidence(qa.get("evidence")),
            })
    if limit is not None:
        queries = queries[:limit]
    if len(queries) < 2:
        raise ValueError("At least two queries are required for validation/test evaluation")
    active_sample_ids = list(dict.fromkeys(str(query["sample_id"]) for query in queries))
    conversation_positions = {
        sample_id: index + 1 for index, sample_id in enumerate(active_sample_ids)
    }
    conversation_query_counts: Dict[str, int] = defaultdict(int)
    for query in queries:
        conversation_query_counts[str(query["sample_id"])] += 1
    conversation_seen: Dict[str, int] = defaultdict(int)
    for query in queries:
        sample_id = str(query["sample_id"])
        conversation_seen[sample_id] += 1
        query["conversation_position"] = conversation_positions[sample_id]
        query["conversation_total"] = len(active_sample_ids)
        query["query_position_in_conversation"] = conversation_seen[sample_id]
        query["queries_in_conversation"] = conversation_query_counts[sample_id]
    validation_count = int(round(len(queries) * validation_ratio))
    validation_count = max(1, min(len(queries) - 1, validation_count))
    indices = list(range(len(queries)))
    random.Random(seed).shuffle(indices)
    validation_indices = set(indices[:validation_count])
    for index, query in enumerate(queries):
        query["split"] = "validation" if index in validation_indices else "test"
    split_meta = {
        "split_unit": "query",
        "limit": limit,
        "validation_ratio": validation_ratio,
        "validation_count": validation_count,
        "test_count": len(queries) - validation_count,
        "split_seed": seed,
        "validation_query_ids": [q["query_id"] for q in queries if q["split"] == "validation"],
        "test_query_ids": [q["query_id"] for q in queries if q["split"] == "test"],
    }
    validation_samples = {q["sample_id"] for q in queries if q["split"] == "validation"}
    test_samples = {q["sample_id"] for q in queries if q["split"] == "test"}
    shared_samples = sorted(validation_samples & test_samples)
    split_meta["conversation_leakage_check"] = {
        "checked_by": "sample_id",
        "validation_sample_count": len(validation_samples),
        "test_sample_count": len(test_samples),
        "shared_sample_ids": shared_samples,
        "has_conversation_leakage": bool(shared_samples),
        "interpretation": "query-level split; validation and test queries can share a conversation memory state",
    }
    split_meta["query_order_hash"] = stable_hash([q["query_id"] for q in queries])
    split_meta["split_hash"] = stable_hash(split_meta)
    return queries, split_meta


@dataclass
class SampleContext:
    sample_id: str
    speaker_a: str
    speaker_b: str
    short_memory: ShortTermMemory
    mid_memory: MidTermMemory
    long_memory: LongTermMemory
    retrieval: RetrievalAndAnswer
    source_map: Dict[Tuple[str, str], List[str]]


def _memory_paths(storage_root: Path, sample_id: str) -> Dict[str, Path]:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)
    sample_dir = storage_root / safe_id
    return {
        "dir": sample_dir,
        "short": sample_dir / "short_term.json",
        "mid": sample_dir / "mid_term.json",
        "long": sample_dir / "long_term.json",
    }


def snapshot_inventory(storage_root: Path, dataset: Sequence[Mapping[str, Any]],
                       required_sample_ids: set[str]) -> Dict[str, Any]:
    inventory: Dict[str, Any] = {}
    for sample in dataset:
        sample_id = str(sample.get("sample_id", "unknown_sample"))
        if sample_id not in required_sample_ids:
            continue
        paths = _memory_paths(storage_root, sample_id)
        files = {
            "short_term": paths["short"],
            "mid_term": paths["mid"],
            "long_term": paths["long"],
            "manifest": paths["dir"] / "snapshot_manifest.json",
        }
        missing = [str(path) for path in files.values() if not path.is_file()]
        if missing:
            raise RuntimeError(
                f"Complete prebuilt snapshot required for {sample_id}; missing: {missing}. "
                "This evaluator never constructs MemoryOS state."
            )
        inventory[sample_id] = {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in files.items()
        }
    missing_samples = sorted(required_sample_ids - set(inventory))
    if missing_samples:
        raise RuntimeError(f"Dataset samples have no matching snapshot directory: {missing_samples}")
    return inventory


def experiment_fingerprint(args: argparse.Namespace, dataset_path: Path, model_path: Path,
                           split_meta: Mapping[str, Any], snapshot_hash: str) -> str:
    identity = {
        "protocol": "v2-validation-threshold-hf-rerank",
        "mode": args.mode,
        "dataset_sha256": sha256_file(dataset_path),
        "model_sha256": (
            sha256_file(model_path) if args.mode != "baseline" and model_path.exists() else None
        ),
        "snapshot_sha256": snapshot_hash,
        "feature_order": FEATURE_ORDER,
        "split_hash": split_meta["split_hash"],
        "threshold_grid": list(args.threshold_grid),
        "tie_tolerance": args.tie_tolerance,
        "llm_model": args.llm_model,
        "base_url": args.base_url,
        "retrieval": {
            "queue_capacity": args.retrieval_queue_capacity,
            "segment_threshold": args.segment_threshold,
            "page_threshold": args.page_threshold,
            "knowledge_threshold": args.knowledge_threshold,
        },
        "memory": {
            "short_term_capacity": args.short_term_capacity,
            "mid_term_capacity": args.mid_term_capacity,
            "topic_similarity_threshold": args.topic_similarity_threshold,
        },
        "generation": {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        },
        "retrieval_embedding_model": "all-MiniLM-L6-v2",
        "verifier_embedding_model": args.embedding_model,
    }
    return stable_hash(identity)


def prepare_sample(sample: Mapping[str, Any], storage_root: Path, client: ConfiguredLLMClient,
                   args: argparse.Namespace) -> SampleContext:
    """Load a complete MemoryOS snapshot without constructing or mutating it."""
    # Lazy imports keep ``--help`` and static inspection usable in an
    # environment where optional retrieval dependencies are not installed.
    from dynamic_update import DynamicUpdate
    from long_term_memory import LongTermMemory
    from mid_term_memory import MidTermMemory
    from retrieval_and_answer import RetrievalAndAnswer
    from short_term_memory import ShortTermMemory

    sample_id = str(sample.get("sample_id", "unknown_sample"))
    conversation = sample["conversation"]
    paths = _memory_paths(storage_root, sample_id)
    required_paths = [
        paths["short"],
        paths["mid"],
        paths["long"],
        paths["dir"] / "snapshot_manifest.json",
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"Complete prebuilt snapshot required for {sample_id}; missing: {missing}. "
            "This evaluator never constructs MemoryOS state."
        )
    decoded: Dict[Path, Any] = {}
    for path in required_paths:
        try:
            decoded[path] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Snapshot file is not valid JSON: {path}") from exc
    manifest_path = paths["dir"] / "snapshot_manifest.json"
    manifest = decoded[manifest_path]
    if manifest.get("sample_id") not in (None, sample_id):
        raise RuntimeError(
            f"Snapshot manifest sample_id mismatch for {sample_id}: {manifest.get('sample_id')!r}"
        )
    manifest_parameters = manifest.get("memory_parameters", {})
    expected_parameters = {
        "short_term_capacity": args.short_term_capacity,
        "mid_term_capacity": args.mid_term_capacity,
        "topic_similarity_threshold": args.topic_similarity_threshold,
    }
    mismatches = {
        name: {"snapshot": manifest_parameters[name], "requested": expected}
        for name, expected in expected_parameters.items()
        if name in manifest_parameters and manifest_parameters[name] != expected
    }
    if mismatches:
        raise RuntimeError(f"Snapshot memory parameter mismatch for {sample_id}: {mismatches}")

    short_memory = ShortTermMemory(max_capacity=args.short_term_capacity, file_path=str(paths["short"]))
    mid_memory = MidTermMemory(max_capacity=args.mid_term_capacity, file_path=str(paths["mid"]))
    long_memory = LongTermMemory(file_path=str(paths["long"]))
    dynamic = DynamicUpdate(
        short_memory,
        mid_memory,
        long_memory,
        topic_similarity_threshold=args.topic_similarity_threshold,
        client=client,
    )
    retrieval = RetrievalAndAnswer(
        short_memory,
        mid_memory,
        long_memory,
        dynamic,
        queue_capacity=args.retrieval_queue_capacity,
    )

    return SampleContext(
        sample_id=sample_id,
        speaker_a=conversation["speaker_a"],
        speaker_b=conversation["speaker_b"],
        short_memory=short_memory,
        mid_memory=mid_memory,
        long_memory=long_memory,
        retrieval=retrieval,
        source_map=build_source_map(conversation),
    )


def _embedding_inventory(mid_sessions: Mapping[str, Any],
                         long_entries: Sequence[Mapping[str, Any]],
                         assistant_entries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    summary_vectors = [
        session.get("summary_embedding")
        for session in mid_sessions.values()
        if isinstance(session, Mapping) and session.get("summary_embedding") is not None
    ]
    page_vectors = [
        page.get("page_embedding")
        for session in mid_sessions.values()
        if isinstance(session, Mapping)
        for page in session.get("details", [])
        if isinstance(page, Mapping) and page.get("page_embedding") is not None
    ]
    knowledge_vectors = [
        entry.get("knowledge_embedding")
        for entry in list(long_entries) + list(assistant_entries)
        if isinstance(entry, Mapping) and entry.get("knowledge_embedding") is not None
    ]

    def describe(vectors: Sequence[Any]) -> Dict[str, Any]:
        dimensions = sorted({len(vector) for vector in vectors if isinstance(vector, (list, tuple))})
        return {
            "count": len(vectors),
            "dimensions": dimensions,
            "all_vectors_are_sequences": all(isinstance(vector, (list, tuple)) for vector in vectors),
        }

    return {
        "mid_term_summary_embeddings": describe(summary_vectors),
        "mid_term_page_embeddings": describe(page_vectors),
        "long_term_knowledge_embeddings": describe(knowledge_vectors),
    }


def run_snapshot_debug_check(dataset: Sequence[Mapping[str, Any]],
                             required_sample_ids: set[str],
                             contexts: Mapping[str, SampleContext],
                             storage_root: Path, output_dir: Path,
                             seed: int) -> Dict[str, Any]:
    """Compare persisted post-build state with the objects loaded from it.

    A fresh LLM-based reconstruction is intentionally not used as the
    reference: it can create different topic summaries and session groupings.
    The JSON snapshot is the state written directly by the original build and
    is therefore the correct source for testing serialization fidelity.
    """
    ordered_sample_ids = [
        str(sample.get("sample_id", "unknown_sample"))
        for sample in dataset
        if str(sample.get("sample_id", "unknown_sample")) in required_sample_ids
    ]
    if not ordered_sample_ids:
        raise RuntimeError("Snapshot debug check has no eligible conversations")
    sample_id = random.Random(seed).choice(ordered_sample_ids)
    context = contexts[sample_id]
    paths = _memory_paths(storage_root, sample_id)
    raw_short = json.loads(paths["short"].read_text(encoding="utf-8"))
    raw_mid = json.loads(paths["mid"].read_text(encoding="utf-8"))
    raw_long = json.loads(paths["long"].read_text(encoding="utf-8"))

    persisted_state = {
        "short_term": raw_short,
        "mid_term": {
            "sessions": raw_mid.get("sessions", {}),
            "access_frequency": raw_mid.get("access_frequency", {}),
        },
        "long_term": {
            "user_profiles": raw_long.get("user_profiles", {}),
            "knowledge_base": raw_long.get("knowledge_base", []),
            "assistant_knowledge": raw_long.get("assistant_knowledge", []),
        },
    }
    loaded_state = {
        "short_term": context.short_memory.get_all(),
        "mid_term": {
            "sessions": context.mid_memory.sessions,
            "access_frequency": dict(context.mid_memory.access_frequency),
        },
        "long_term": {
            "user_profiles": context.long_memory.user_profiles,
            "knowledge_base": context.long_memory.knowledge_base,
            "assistant_knowledge": context.long_memory.assistant_knowledge,
        },
    }

    def memory_counts(state: Mapping[str, Any]) -> Dict[str, int]:
        sessions = state["mid_term"]["sessions"]
        return {
            "short_term_qa": len(state["short_term"]),
            "mid_term_sessions": len(sessions),
            "mid_term_pages": sum(
                len(session.get("details", []))
                for session in sessions.values() if isinstance(session, Mapping)
            ),
            "mid_term_access_frequency_entries": len(state["mid_term"]["access_frequency"]),
            "long_term_user_profiles": len(state["long_term"]["user_profiles"]),
            "long_term_knowledge": len(state["long_term"]["knowledge_base"]),
            "long_term_assistant_knowledge": len(state["long_term"]["assistant_knowledge"]),
        }

    persisted_counts = memory_counts(persisted_state)
    loaded_counts = memory_counts(loaded_state)
    count_comparisons = {
        name: {
            "post_build_persisted": persisted_counts[name],
            "snapshot_loaded": loaded_counts[name],
            "match": persisted_counts[name] == loaded_counts[name],
        }
        for name in persisted_counts
    }
    persisted_embeddings = _embedding_inventory(
        persisted_state["mid_term"]["sessions"],
        persisted_state["long_term"]["knowledge_base"],
        persisted_state["long_term"]["assistant_knowledge"],
    )
    loaded_embeddings = _embedding_inventory(
        loaded_state["mid_term"]["sessions"],
        loaded_state["long_term"]["knowledge_base"],
        loaded_state["long_term"]["assistant_knowledge"],
    )
    persisted_hash = stable_hash(persisted_state)
    loaded_hash = stable_hash(loaded_state)
    heap_session_ids = {str(item[1]) for item in context.mid_memory.heap}
    expected_session_ids = {str(item) for item in context.mid_memory.sessions}
    checks = {
        "all_memory_counts_match": all(item["match"] for item in count_comparisons.values()),
        "persistent_state_exact_match": persisted_hash == loaded_hash,
        "embedding_inventory_match": persisted_embeddings == loaded_embeddings,
        "mid_term_heap_rebuilt_for_all_sessions": heap_session_ids == expected_session_ids,
        "retrieval_queue_starts_empty": len(context.retrieval.retrieval_queue) == 0,
    }
    report = {
        "debug_check": "snapshot_serialization_fidelity",
        "seed": seed,
        "selected_conversation": sample_id,
        "reference_definition": (
            "post-build state serialized by the original MemoryOS save methods; "
            "no nondeterministic LLM reconstruction is performed"
        ),
        "counts": count_comparisons,
        "persistent_state_sha256": {
            "post_build_persisted": persisted_hash,
            "snapshot_loaded": loaded_hash,
        },
        "embedding_inventory": {
            "post_build_persisted": persisted_embeddings,
            "snapshot_loaded": loaded_embeddings,
        },
        "index_state": {
            "faiss_index_persisted": False,
            "behavior": (
                "Mid-term and long-term IndexFlatIP indices are rebuilt per search "
                "from the restored embedding vectors, matching the original retrieval code."
            ),
        },
        "transient_state": {
            "retrieval_queue": "fresh empty queue; candidates are query-specific",
            "dynamic_update_last_evicted_page": "fresh runtime field; unused by evaluation retrieval",
            "mid_term_heap": "rebuilt deterministically from restored H_segment values",
        },
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path = output_dir / "snapshot_debug_report.json"
    write_json(report_path, report)
    print(
        f"Snapshot debug check: conversation={sample_id}, "
        f"all_checks_passed={report['all_checks_passed']}, report={report_path}"
    )
    if not report["all_checks_passed"]:
        raise RuntimeError(f"Snapshot fidelity check failed; see {report_path}")
    return report


def feature_vector(feature_row: Mapping[str, Any]) -> List[float]:
    values = [float(feature_row[name]) for name in FEATURE_ORDER]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"Invalid MemoryVerifier feature vector: {values!r}")
    return values


def reliable_probability(model: Any, vector: Sequence[float]) -> float:
    probabilities = model.predict_proba([list(vector)])[0]
    classes = list(model.classes_)
    probability = float(probabilities[classes.index(1)])
    return max(0.0, min(1.0, probability))


def source_ids_for_page(page: Mapping[str, Any], source_map: Mapping[Tuple[str, str], List[str]]) -> List[str]:
    key = (str(page.get("user_input", "")), str(page.get("agent_response", "")))
    return list(source_map.get(key, []))


def build_candidate_bundle(question: str, pages: Sequence[Mapping[str, Any]],
                           source_map: Mapping[Tuple[str, str], List[str]],
                           verifier: MemoryVerifier, model: Any) -> List[Dict[str, Any]]:
    verification = verifier.verify(question, pages)
    by_id = {}
    for row in verification.get("features", []):
        page_id = row.get("page_id")
        if page_id is not None:
            by_id[str(page_id)] = row

    bundle: List[Dict[str, Any]] = []
    for rank, page in enumerate(pages):
        page_id = str(page.get("page_id", f"candidate_{rank}"))
        row = by_id.get(page_id)
        vector: Optional[List[float]] = None
        probability: Optional[float] = None
        if row is not None:
            vector = feature_vector(row)
            probability = reliable_probability(model, vector)
        bundle.append({
            "page_id": page_id,
            "page": dict(page),
            "initial_rank": rank,
            "features": vector,
            "probability": probability,
            "source_dialogue_ids": source_ids_for_page(page, source_map),
        })
    return bundle


def hard_filter(bundle: Sequence[Mapping[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    return [
        dict(candidate)
        for candidate in bundle
        if candidate.get("probability") is not None and float(candidate["probability"]) >= threshold
    ]


def reliability_rerank(bundle: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    valid = [candidate for candidate in bundle if candidate.get("probability") is not None]
    missing = [candidate for candidate in bundle if candidate.get("probability") is None]
    ranked = sorted(valid, key=lambda candidate: (-float(candidate["probability"]), int(candidate["initial_rank"])))
    return [dict(candidate) for candidate in ranked + missing]


def simple_tokenize(text: Any) -> List[str]:
    if text is None:
        return []
    return re.findall(r"\b\w+\b", str(text).lower())


def answer_f1(prediction: Any, reference: Any) -> float:
    predicted = set(simple_tokenize(prediction))
    expected = set(simple_tokenize(reference))
    if not predicted or not expected:
        return 0.0
    common = predicted & expected
    precision = len(common) / len(predicted)
    recall = len(common) / len(expected)
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def prompt_token_counter_name(model: str) -> str:
    """Return the tokenizer used for logging without changing prompt content."""
    try:
        import tiktoken
        try:
            tiktoken.encoding_for_model(model)
            return f"tiktoken:{model}"
        except Exception:
            tiktoken.get_encoding("cl100k_base")
            return "tiktoken:cl100k_base_fallback"
    except Exception:
        return "whitespace_fallback"


def count_prompt_tokens(messages: Sequence[Mapping[str, str]], model: str) -> int:
    """Count input prompt tokens for reporting only; generation is unchanged."""
    try:
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model(model)
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
        return int(sum(len(encoding.encode(str(message.get("content", "")))) for message in messages))
    except Exception:
        # A deterministic fallback keeps the experiment runnable when the
        # optional tokenizer package is unavailable.  Metadata records this.
        return int(sum(len(re.findall(r"\S+", str(message.get("content", "")))) for message in messages))


def evidence_metrics(record: Mapping[str, Any]) -> Dict[str, float]:
    gold = set(record.get("evidence", []))
    retrieved = set(record.get("retrieved_source_dialogue_ids", []))
    selected = set(record.get("selected_source_dialogue_ids", []))
    recall = len(selected & gold) / len(gold) if gold else 0.0
    precision = len(selected & gold) / len(selected) if selected else 0.0
    initial_recall = len(retrieved & gold) / len(gold) if gold else 0.0
    return {
        "initial_evidence_recall": initial_recall,
        "selected_evidence_recall": recall,
        "selected_evidence_precision": precision,
        "evidence_retention": recall / initial_recall if initial_recall else 0.0,
    }


def make_result(query: Mapping[str, Any], bundle: Sequence[Mapping[str, Any]], selected: Sequence[Mapping[str, Any]],
                answer: str, policy: str, threshold: Optional[float],
                long_term_knowledge: Sequence[Any], prompt_tokens_before: Optional[int] = None,
                prompt_tokens_after: Optional[int] = None) -> Dict[str, Any]:
    retrieved_ids = [str(candidate["page_id"]) for candidate in bundle]
    selected_ids = [str(candidate["page_id"]) for candidate in selected]
    initial_ranks = {
        str(candidate["page_id"]): index for index, candidate in enumerate(bundle)
    }
    rank_displacements = []
    if policy == "rerank":
        rank_displacements = [
            initial_ranks[page_id] - selected_index
            for selected_index, page_id in enumerate(selected_ids)
            if page_id in initial_ranks
        ]
    gold_evidence = set(query.get("evidence", []))
    initial_evidence_positions = [
        index for index, candidate in enumerate(bundle)
        if gold_evidence.intersection(candidate.get("source_dialogue_ids", []))
    ]
    selected_evidence_positions = [
        index for index, candidate in enumerate(selected)
        if gold_evidence.intersection(candidate.get("source_dialogue_ids", []))
    ]
    removed_memory_count = max(0, len(bundle) - len(selected))
    before_tokens = int(prompt_tokens_before or 0)
    after_tokens = int(prompt_tokens_after or 0)
    token_reduction_ratio = (
        float(max(0, before_tokens - after_tokens) / before_tokens)
        if before_tokens > 0 else 0.0
    )
    record: Dict[str, Any] = {
        "query_id": query["query_id"],
        "sample_id": query["sample_id"],
        "qa_index": query["qa_index"],
        "split": query["split"],
        "query": query["query"],
        "category": query.get("category"),
        "evidence": query.get("evidence", []),
        "reference_answer": query["reference_answer"],
        "retrieved_memory_ids": retrieved_ids,
        "candidate_memory_count": len(bundle),
        "memory_features": [] if policy == "baseline" else [candidate.get("features") for candidate in bundle],
        "reliability_probability": [] if policy == "baseline" else [candidate.get("probability") for candidate in bundle],
        "retrieved_source_dialogue_ids": sorted({
            source_id for candidate in bundle for source_id in candidate.get("source_dialogue_ids", [])
        }),
        "selected_memory_ids": selected_ids,
        "selected_memory_count": len(selected),
        "removed_memory_count": removed_memory_count,
        "selected_source_dialogue_ids": sorted({
            source_id for candidate in selected for source_id in candidate.get("source_dialogue_ids", [])
        }),
        "selection_policy": policy,
        "selection_threshold": threshold,
        "rank_displacement_mean": float(statistics.mean(rank_displacements)) if rank_displacements else 0.0,
        "first_evidence_rank_initial": min(initial_evidence_positions) if initial_evidence_positions else None,
        "first_evidence_rank_selected": min(selected_evidence_positions) if selected_evidence_positions else None,
        "low_probability_selected_count": sum(
            int(candidate.get("probability") is not None and float(candidate["probability"]) < 0.5)
            for candidate in selected
        ),
        "prompt_tokens_before": before_tokens,
        "prompt_tokens_after": after_tokens,
        "token_reduction_ratio": token_reduction_ratio,
        "answer": answer,
        "answer_f1": answer_f1(answer, query["reference_answer"]),
        "long_term_knowledge_count": len(long_term_knowledge or []),
    }
    record.update(evidence_metrics(record))
    return record


def generate_system_response_with_meta(query: str, short_mem: ShortTermMemory,
                                       long_mem: LongTermMemory, retrieval_queue: Sequence[Mapping[str, Any]],
                                       long_knowledge: Sequence[Mapping[str, Any]], client: ConfiguredLLMClient,
                                       sample_id: str, speaker_a: str, speaker_b: str,
                                       llm_model: str, temperature: float, max_tokens: int,
                                       return_prompt_tokens: bool = False) -> Any:
    """The prompt template is kept aligned with eval/main_loco_parse.py."""
    history = short_mem.get_all()
    history_text = "\n".join([
        f"{speaker_a}: {qa.get('user_input', '')}\n{speaker_b}: {qa.get('agent_response', '')}\n"
        f"Time: ({qa.get('timestamp', '')})"
        for qa in history
    ])
    retrieval_text = "\n".join([
        f"【Historical Memory】 {speaker_a}: {page.get('user_input', '')}\n"
        f"{speaker_b}: {page.get('agent_response', '')}\n"
        f"Time:({page.get('timestamp', '')})\n"
        f"Conversation chain overview:({page.get('meta_info', '')})\n"
        for page in retrieval_queue
    ])
    profile_obj = long_mem.get_user_profile(sample_id)
    profile_text = str(profile_obj.get("data", "None")) if profile_obj else "None"
    background = f"【User Profile】\n{profile_text}\n\n"
    for knowledge in long_knowledge or []:
        if isinstance(knowledge, dict) and knowledge.get("knowledge") not in (None, "", "None"):
            background += f"{knowledge['knowledge']}\n"
    background = re.sub(r"(?i)\buser\b", speaker_a, background)
    background = re.sub(r"(?i)\bassistant\b", speaker_b, background)
    assistant_knowledge_text = "【Assistant Knowledge】\n"
    for knowledge in long_mem.get_assistant_knowledge():
        assistant_knowledge_text += f"- {knowledge['knowledge']} ({knowledge['timestamp']})\n"
    assistant_knowledge_text = re.sub(r"\bI\b", speaker_b, assistant_knowledge_text)
    system_prompt = (
        f"You are role-playing as {speaker_b} in a conversation with the user is playing is  {speaker_a}. "
        f"Here are some of your character traits and knowledge:\n{assistant_knowledge_text}\n"
        f"Any content referring to 'User' in the prompt refers to {speaker_a}'s content, and any content referring to 'AI'or 'assiant' refers to {speaker_b}'s content."
        f"Your task is to answer questions about {speaker_a} or {speaker_b} in an extremely concise manner.\n"
        f"When the question is: \"What did the charity race raise awareness for?\", you should not answer in the form of: \"The charity race raised awareness for mental health.\" Instead, it should be: \"mental health\", as this is more concise."
    )
    user_prompt = (
        f"<CONTEXT>\n"
        f"Recent conversation between {speaker_a} and {speaker_b}:\n"
        f"{history_text}\n\n"
        f"<MEMORY>\n"
        f"Relevant past conversations:\n"
        f"{retrieval_text}\n\n"
        f"<CHARACTER TRAITS>\n"
        f"Characteristics of {speaker_a}:\n"
        f"{background}\n\n"
        f"the question is: {query}\n"
        f"Your task is to answer questions about {speaker_a} or {speaker_b} in an extremely concise manner.\n"
        f"Please only provide the content of the answer, without including 'answer:'\n"
        f"For questions that require answering a date or time, strictly follow the format \"15 July 2023\" and provide a specific date whenever possible. For example, if you need to answer \"last year,\" give the specific year of last year rather than just saying \"last year.\" Only provide one year, date, or time, without any extra responses.\n"
        f"If the question is about the duration, answer in the form of several years, months, or days.\n"
        f"Generate answers primarily composed of concrete entities, such as Mentoring program, school speech, etc"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = client.chat_completion(
        model=llm_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if return_prompt_tokens:
        return response, count_prompt_tokens(messages, llm_model)
    return response


def summarize_records(records: Sequence[Mapping[str, Any]], include_categories: bool = True) -> Dict[str, Any]:
    if not records:
        return {"count": 0}
    def mean(key: str) -> float:
        values = [float(record.get(key, 0.0)) for record in records]
        return float(statistics.mean(values)) if values else 0.0

    categories: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    if include_categories:
        for record in records:
            categories[str(record.get("category", "unknown"))].append(record)
    result: Dict[str, Any] = {
        "count": len(records),
        "answer_f1_mean": mean("answer_f1"),
        "initial_evidence_recall_mean": mean("initial_evidence_recall"),
        "selected_evidence_recall_mean": mean("selected_evidence_recall"),
        "selected_evidence_precision_mean": mean("selected_evidence_precision"),
        "average_candidate_count": mean("candidate_memory_count"),
        "average_selected_count": mean("selected_memory_count"),
        "rank_displacement_mean": mean("rank_displacement_mean"),
        "low_probability_selected_count_mean": mean("low_probability_selected_count"),
        "average_removed_memory_count": mean("removed_memory_count"),
        "prompt_tokens_before_mean": mean("prompt_tokens_before"),
        "prompt_tokens_after_mean": mean("prompt_tokens_after"),
        "token_reduction_ratio_mean": mean("token_reduction_ratio"),
        "average_removed_count": float(statistics.mean(
            int(record.get("removed_memory_count", max(
                0, int(record["candidate_memory_count"]) - int(record["selected_memory_count"])
            )))
            for record in records
        )),
        "empty_selection_rate": float(statistics.mean(
            int(record["selected_memory_count"] == 0) for record in records
        )),
        "filter_ratio": float(sum(
            int(record.get("removed_memory_count", max(
                0, int(record["candidate_memory_count"]) - int(record["selected_memory_count"])
            )))
            for record in records
        ) / max(1, sum(int(record["candidate_memory_count"]) for record in records))),
        "categories": {},
    }
    for category, category_records in sorted(categories.items()):
        result["categories"][category] = summarize_records(category_records, include_categories=False)
    return result


def choose_validation_threshold(records_by_threshold: Mapping[str, Sequence[Mapping[str, Any]]],
                                tie_tolerance: float) -> Dict[str, Any]:
    if not records_by_threshold:
        raise ValueError("No validation threshold results were produced")
    candidates = []
    for text_threshold, records in records_by_threshold.items():
        threshold = float(text_threshold)
        summary = summarize_records(records)
        candidates.append({
            "threshold": threshold,
            "answer_f1": summary.get("answer_f1_mean", 0.0),
            "evidence_recall": summary.get("selected_evidence_recall_mean", 0.0),
            "average_selected_count": summary.get("average_selected_count", 0.0),
            "summary": summary,
        })
    best_f1 = max(candidate["answer_f1"] for candidate in candidates)
    eligible = [candidate for candidate in candidates if candidate["answer_f1"] >= best_f1 - tie_tolerance]
    eligible.sort(key=lambda item: (-item["evidence_recall"], item["average_selected_count"], item["threshold"]))
    selected = eligible[0]
    return {
        "selected_threshold": selected["threshold"],
        "best_tau": selected["threshold"],
        "selection_objective": "validation_answer_f1",
        "tie_tolerance": tie_tolerance,
        "tie_break": ["evidence_recall_desc", "average_selected_count_asc", "threshold_asc"],
        "candidates": sorted(candidates, key=lambda item: item["threshold"]),
    }


def candidate_cache_record(query: Mapping[str, Any], bundle: Sequence[Mapping[str, Any]],
                          long_term_knowledge: Sequence[Any]) -> Dict[str, Any]:
    return {
        "query_id": query["query_id"],
        "sample_id": query["sample_id"],
        "split": query["split"],
        "query": query["query"],
        "retrieved_memory_ids": [candidate["page_id"] for candidate in bundle],
        "candidate_memory_count": len(bundle),
        "memory_features": [candidate.get("features") for candidate in bundle],
        "reliability_probability": [candidate.get("probability") for candidate in bundle],
        "source_dialogue_ids": [candidate.get("source_dialogue_ids", []) for candidate in bundle],
        "long_term_knowledge_count": len(long_term_knowledge or []),
    }


def save_condition(output_dir: Path, condition: str, metadata: Mapping[str, Any],
                   records: Sequence[Mapping[str, Any]]) -> None:
    payload = {
        "metadata": dict(metadata),
        "condition": condition,
        "summary": summarize_records(records),
        "results": list(records),
    }
    write_json(output_dir / f"{condition}_results.json", payload)


def build_metadata(args: argparse.Namespace, dataset_path: Path, model_path: Path,
                   split_meta: Mapping[str, Any], selected_threshold: Optional[float],
                   run_fingerprint: str, snapshot_hash: str,
                   snapshot_count: int) -> Dict[str, Any]:
    return {
        "experiment": "MemoryOS_E2E_Reliability_Selection",
        "protocol_version": "v2-validation-threshold-hf-rerank",
        "implementation_path": str(Path(__file__).resolve()),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": sha256_file(dataset_path),
        "model_path": str(model_path.resolve()),
        "model_sha256": sha256_file(model_path) if model_path.exists() else None,
        "feature_order": FEATURE_ORDER,
        "random_seed": args.seed,
        "limit": args.limit,
        "validation_split": dict(split_meta),
        "final_test_policy": "hard_filter_validation_threshold",
        "selected_threshold": selected_threshold,
        "best_tau": selected_threshold,
        "fixed_comparator_threshold": 0.5,
        "selection_policies": ["baseline", "hard_filter", "rerank"],
        "llm_model": args.llm_model,
        # RetrievalAndAnswer currently obtains embeddings through the existing
        # eval/utils.py default; the verifier model is independently recorded.
        # Both experiment conditions use exactly the same retrieval model.
        "embedding_model": "all-MiniLM-L6-v2",
        "retrieval_embedding_model": "all-MiniLM-L6-v2",
        "verifier_embedding_model": args.embedding_model,
        "base_url": args.base_url,
        "retrieval_parameters": {
            "queue_capacity": args.retrieval_queue_capacity,
            "segment_threshold": args.segment_threshold,
            "page_threshold": args.page_threshold,
            "knowledge_threshold": args.knowledge_threshold,
        },
        "memory_parameters": {
            "short_term_capacity": args.short_term_capacity,
            "mid_term_capacity": args.mid_term_capacity,
            "topic_similarity_threshold": args.topic_similarity_threshold,
        },
        "generation_parameters": {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        },
        "prompt_token_counter": prompt_token_counter_name(args.llm_model),
        "storage_root": str(Path(args.storage_root).resolve()),
        "output_dir": str(Path(args.output_dir).resolve()),
        "snapshot_loading_mode": "existing_snapshots_only",
        "snapshot_conversation_count": snapshot_count,
        "snapshot_fingerprint": snapshot_hash,
        "snapshot_debug_check": {
            "enabled": bool(args.debug_snapshot_check),
            "seed": args.debug_snapshot_seed,
            "report_path": (
                str((Path(args.output_dir).resolve() / "snapshot_debug_report.json"))
                if args.debug_snapshot_check else None
            ),
        },
        "run_fingerprint": run_fingerprint,
        "checkpoint_batch_size": args.checkpoint_batch_size,
        "checkpoint_layout": "conversation_partitioned",
        "retrieval_embedding_cache": "process_local_lazy_singleton_and_text_vector_cache",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_key_recorded": bool(args.api_key),
        "no_model_training": True,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    install_retrieval_embedding_cache()
    from memory_verifier import MemoryVerifier

    dataset_path = Path(args.dataset).resolve()
    model_path = Path(args.model).resolve()
    output_dir = Path(args.output_dir).resolve()
    storage_root = Path(args.storage_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not storage_root.is_dir():
        raise RuntimeError(
            f"Snapshot root does not exist: {storage_root}. "
            "This evaluator only loads prebuilt MemoryOS snapshots."
        )
    dataset = load_locomo(dataset_path)
    queries, split_meta = flatten_queries(dataset, args.validation_ratio, args.seed, args.limit)
    validation_queries = [query for query in queries if query["split"] == "validation"]
    test_queries = [query for query in queries if query["split"] == "test"]
    selected_threshold: Optional[float] = None
    client: Any = (
        DisabledLLMClient()
        if args.debug_snapshot_only
        else ConfiguredLLMClient(args.api_key, args.base_url)
    )
    contexts: Dict[str, SampleContext] = {}
    required_sample_ids = {query["sample_id"] for query in queries}
    print(f"Validating {len(required_sample_ids)} existing MemoryOS conversation snapshots...")
    snapshots = snapshot_inventory(storage_root, dataset, required_sample_ids)
    snapshot_hash = stable_hash({
        sample_id: {
            name: details["sha256"] for name, details in files.items()
        }
        for sample_id, files in snapshots.items()
    })
    for sample in dataset:
        sample_id = str(sample.get("sample_id", "unknown_sample"))
        if sample_id not in required_sample_ids:
            continue
        contexts[sample_id] = prepare_sample(sample, storage_root, client, args)
    if set(contexts) != required_sample_ids:
        raise RuntimeError("Not every required LoCoMo conversation has a loaded snapshot")

    snapshot_debug_report: Optional[Dict[str, Any]] = None
    if args.debug_snapshot_check or args.debug_snapshot_only:
        snapshot_debug_report = run_snapshot_debug_check(
            dataset=dataset,
            required_sample_ids=required_sample_ids,
            contexts=contexts,
            storage_root=storage_root,
            output_dir=output_dir,
            seed=args.debug_snapshot_seed,
        )
    if args.debug_snapshot_only:
        print("Snapshot debug-only mode completed; retrieval and answer generation were not run.")
        return {"snapshot_debug": snapshot_debug_report}

    predictor_artifact = None if args.mode == "baseline" else load_predictor(model_path)
    verifier = MemoryVerifier(embedding_model=args.embedding_model)
    model = predictor_artifact["model"] if predictor_artifact is not None else None

    run_fingerprint = experiment_fingerprint(
        args, dataset_path, model_path, split_meta, snapshot_hash
    )
    checkpoints: List[ConversationCheckpoint] = []

    def checkpoint(directory: str, strategy: str,
                   expected: Sequence[Mapping[str, Any]],
                   parameters: Optional[Mapping[str, Any]] = None) -> ConversationCheckpoint:
        store = ConversationCheckpoint(
            output_dir=output_dir,
            directory=directory,
            strategy=strategy,
            run_fingerprint=run_fingerprint,
            expected_queries=expected,
            batch_size=args.checkpoint_batch_size,
            parameters=parameters,
        )
        checkpoints.append(store)
        return store

    candidate_checkpoint = checkpoint(
        "candidate_cache", "candidate_retrieval_and_features", queries
    )
    baseline_checkpoint = checkpoint("baseline", "baseline", queries)
    threshold_checkpoints: Dict[str, ConversationCheckpoint] = {}
    rerank_checkpoint: Optional[ConversationCheckpoint] = None
    hf05_checkpoint: Optional[ConversationCheckpoint] = None
    if args.mode != "baseline":
        rerank_checkpoint = checkpoint("rerank", "rerank", queries)
        hf05_checkpoint = checkpoint(
            "hf_0.5", "hf_0.5", test_queries, {"threshold": 0.5}
        )
        for threshold in args.threshold_grid:
            threshold_key = f"{threshold:.6f}"
            threshold_directory = f"validation_thresholds/tau_{threshold_key.replace('.', '_')}"
            threshold_checkpoints[threshold_key] = checkpoint(
                threshold_directory,
                "validation_hard_filter",
                validation_queries,
                {"threshold": float(threshold)},
            )

    if args.mode == "baseline":
        total_work = len(queries) * 2
    else:
        total_work = (
            len(queries)
            + len(validation_queries) * (2 + len(args.threshold_grid))
            + len(test_queries) * 4
        )
    resumed_work = sum(len(store) for store in checkpoints)
    print(
        f"Snapshot-only evaluation ready: {len(contexts)} conversations, "
        f"{len(queries)} queries, {resumed_work}/{total_work} work units resumed."
    )
    progress = EvaluationProgress(total_work, resumed_work)

    # Retrieval and feature extraction remain exactly once per query.  The
    # complete bundle is checkpointed because the public cache summary alone
    # is insufficient to resume prompt construction.
    prepared: List[Tuple[Dict[str, Any], SampleContext, List[Dict[str, Any]], List[Any]]] = []

    def generate_for_selection(query: Mapping[str, Any], context: SampleContext,
                               bundle: Sequence[Mapping[str, Any]], selected: Sequence[Mapping[str, Any]],
                               long_term_knowledge: Sequence[Any], policy: str,
                               threshold: Optional[float],
                               prompt_tokens_before: Optional[int] = None) -> Dict[str, Any]:
        answer, prompt_tokens_after = generate_system_response_with_meta(
            query["query"], context.short_memory, context.long_memory,
            [candidate["page"] for candidate in selected], long_term_knowledge, client,
            context.sample_id, context.speaker_a, context.speaker_b,
            args.llm_model, args.temperature, args.max_tokens,
            return_prompt_tokens=True,
        )
        if prompt_tokens_before is None:
            prompt_tokens_before = prompt_tokens_after
        return make_result(
            query, bundle, selected, answer, policy, threshold, long_term_knowledge,
            prompt_tokens_before=prompt_tokens_before,
            prompt_tokens_after=prompt_tokens_after,
        )

    try:
        for query in queries:
            context = contexts[query["sample_id"]]
            cached = candidate_checkpoint.get(query["query_id"])
            if cached is None:
                progress.begin(query, "retrieval + feature extraction")
                retrieval_result = context.retrieval.retrieve(
                    query["query"],
                    segment_threshold=args.segment_threshold,
                    page_threshold=args.page_threshold,
                    knowledge_threshold=args.knowledge_threshold,
                    client=client,
                )
                pages = retrieval_result.get("retrieval_queue", [])
                long_term_knowledge = list(retrieval_result.get("long_term_knowledge", []))
                if args.mode == "baseline":
                    bundle = [{
                        "page_id": str(page.get("page_id", f"candidate_{index}")),
                        "page": dict(page),
                        "initial_rank": index,
                        "features": None,
                        "probability": None,
                        "source_dialogue_ids": source_ids_for_page(page, context.source_map),
                    } for index, page in enumerate(pages)]
                else:
                    bundle = build_candidate_bundle(
                        query["query"], pages, context.source_map, verifier, model
                    )
                cached = {
                    "query_id": query["query_id"],
                    "bundle": bundle,
                    "long_term_knowledge": long_term_knowledge,
                }
                candidate_checkpoint.add(query, cached)
                progress.complete()
            bundle = [dict(candidate) for candidate in cached.get("bundle", [])]
            long_term_knowledge = list(cached.get("long_term_knowledge", []))
            prepared.append((query, context, bundle, long_term_knowledge))
        candidate_checkpoint.flush_all()

        if args.mode == "baseline":
            for query, context, bundle, long_term_knowledge in prepared:
                if baseline_checkpoint.get(query["query_id"]) is not None:
                    continue
                progress.begin(query, "baseline")
                record = generate_for_selection(
                    query, context, bundle, bundle, long_term_knowledge, "baseline", None
                )
                baseline_checkpoint.add(query, record)
                progress.complete()
            baseline_checkpoint.flush_all()

            baseline_records = baseline_checkpoint.ordered_records(queries)
            metadata = build_metadata(
                args, dataset_path, model_path, split_meta, None,
                run_fingerprint, snapshot_hash, len(contexts),
            )
            metadata["final_test_policy"] = "not_applicable_baseline_only"
            save_condition(output_dir, "baseline", metadata, baseline_records)
            write_json(output_dir / "e2e_candidate_cache.json", {
                "metadata": metadata,
                "results": [
                    candidate_cache_record(query, bundle, long_term_knowledge)
                    for query, _, bundle, long_term_knowledge in prepared
                ],
            })
            write_json(output_dir / "e2e_experiment_config.json", metadata)
            return {"metadata": metadata, "baseline": summarize_records(baseline_records)}

        if rerank_checkpoint is None or hf05_checkpoint is None:
            raise RuntimeError("Reliability checkpoints were not initialized")

        # Validation order is unchanged: baseline, every threshold, then RR
        # for each validation query.  Test generation cannot start here.
        validation_sample: Optional[str] = None
        for query, context, bundle, long_term_knowledge in prepared:
            if query["split"] != "validation":
                continue
            current_sample = str(query["sample_id"])
            if validation_sample is not None and current_sample != validation_sample:
                baseline_checkpoint.flush_sample(validation_sample)
                rerank_checkpoint.flush_sample(validation_sample)
                for threshold_checkpoint in threshold_checkpoints.values():
                    threshold_checkpoint.flush_sample(validation_sample)
            validation_sample = current_sample
            baseline_record = baseline_checkpoint.get(query["query_id"])
            if baseline_record is None:
                progress.begin(query, "baseline (validation)")
                baseline_record = generate_for_selection(
                    query, context, bundle, bundle, long_term_knowledge, "baseline", None
                )
                baseline_checkpoint.add(query, baseline_record)
                progress.complete()
            for threshold in args.threshold_grid:
                threshold_key = f"{threshold:.6f}"
                threshold_checkpoint = threshold_checkpoints[threshold_key]
                if threshold_checkpoint.get(query["query_id"]) is not None:
                    continue
                progress.begin(query, f"HF validation tau={threshold:.1f}")
                selected = hard_filter(bundle, threshold)
                record = generate_for_selection(
                    query, context, bundle, selected, long_term_knowledge,
                    "hard_filter", threshold, baseline_record["prompt_tokens_before"],
                )
                threshold_checkpoint.add(query, record)
                progress.complete()
            if rerank_checkpoint.get(query["query_id"]) is None:
                progress.begin(query, "rerank (validation)")
                ranked = reliability_rerank(bundle)
                record = generate_for_selection(
                    query, context, bundle, ranked, long_term_knowledge, "rerank", None,
                    baseline_record["prompt_tokens_before"],
                )
                rerank_checkpoint.add(query, record)
                progress.complete()

        if validation_sample is not None:
            baseline_checkpoint.flush_sample(validation_sample)
            rerank_checkpoint.flush_sample(validation_sample)
            for threshold_checkpoint in threshold_checkpoints.values():
                threshold_checkpoint.flush_sample(validation_sample)
        baseline_checkpoint.flush_all()
        rerank_checkpoint.flush_all()
        for threshold_checkpoint in threshold_checkpoints.values():
            threshold_checkpoint.flush_all()

        validation_by_threshold = {
            threshold: store.ordered_records(validation_queries)
            for threshold, store in threshold_checkpoints.items()
        }
        validation_baseline = baseline_checkpoint.ordered_records(validation_queries)
        validation_rr = rerank_checkpoint.ordered_records(validation_queries)

        # Lock tau_val before entering the test generation phase.
        threshold_selection = choose_validation_threshold(validation_by_threshold, args.tie_tolerance)
        selected_threshold = float(threshold_selection["selected_threshold"])
        write_json(output_dir / "best_tau.json", {
            "best_tau": selected_threshold,
            "source": "validation",
            "selection_objective": threshold_selection["selection_objective"],
            "test_policy": "hard_filter",
            "run_fingerprint": run_fingerprint,
        })
        write_json(output_dir / "validation_threshold_selection_checkpoint.json", {
            "run_fingerprint": run_fingerprint,
            "selection": threshold_selection,
            "baseline_validation_summary": summarize_records(validation_baseline),
            "rerank_validation_summary": summarize_records(validation_rr),
        })

        hf_tau_checkpoint = checkpoint(
            "hf_tau", "hf_tau", test_queries, {"threshold": selected_threshold}
        )
        progress.credit_resumed(len(hf_tau_checkpoint), "hf_tau")

        test_sample: Optional[str] = None
        for query, context, bundle, long_term_knowledge in prepared:
            if query["split"] != "test":
                continue
            current_sample = str(query["sample_id"])
            if test_sample is not None and current_sample != test_sample:
                for store in (baseline_checkpoint, hf05_checkpoint, hf_tau_checkpoint, rerank_checkpoint):
                    store.flush_sample(test_sample)
            test_sample = current_sample
            baseline_record = baseline_checkpoint.get(query["query_id"])
            if baseline_record is None:
                progress.begin(query, "baseline (test)")
                baseline_record = generate_for_selection(
                    query, context, bundle, bundle, long_term_knowledge, "baseline", None
                )
                baseline_checkpoint.add(query, baseline_record)
                progress.complete()
            if hf05_checkpoint.get(query["query_id"]) is None:
                progress.begin(query, "HF-0.5 (test)")
                record = generate_for_selection(
                    query, context, bundle, hard_filter(bundle, 0.5), long_term_knowledge,
                    "hard_filter", 0.5, baseline_record["prompt_tokens_before"],
                )
                hf05_checkpoint.add(query, record)
                progress.complete()
            if hf_tau_checkpoint.get(query["query_id"]) is None:
                progress.begin(query, f"HF-tau={selected_threshold:.1f} (test)")
                record = generate_for_selection(
                    query, context, bundle, hard_filter(bundle, selected_threshold),
                    long_term_knowledge, "hard_filter", selected_threshold,
                    baseline_record["prompt_tokens_before"],
                )
                hf_tau_checkpoint.add(query, record)
                progress.complete()
            if rerank_checkpoint.get(query["query_id"]) is None:
                progress.begin(query, "rerank (test)")
                record = generate_for_selection(
                    query, context, bundle, reliability_rerank(bundle), long_term_knowledge,
                    "rerank", None, baseline_record["prompt_tokens_before"],
                )
                rerank_checkpoint.add(query, record)
                progress.complete()

        if test_sample is not None:
            for store in (baseline_checkpoint, hf05_checkpoint, hf_tau_checkpoint, rerank_checkpoint):
                store.flush_sample(test_sample)
        for store in (baseline_checkpoint, hf05_checkpoint, hf_tau_checkpoint, rerank_checkpoint):
            store.flush_all()

        test_baseline = baseline_checkpoint.ordered_records(test_queries)
        test_hf05 = hf05_checkpoint.ordered_records(test_queries)
        test_hf_val = hf_tau_checkpoint.ordered_records(test_queries)
        test_rr = rerank_checkpoint.ordered_records(test_queries)
        metadata = build_metadata(
            args, dataset_path, model_path, split_meta, selected_threshold,
            run_fingerprint, snapshot_hash, len(contexts),
        )
        metadata["threshold_grid"] = args.threshold_grid
        metadata["threshold_selection"] = threshold_selection
        metadata["validation_baseline_summary"] = summarize_records(validation_baseline)
        metadata["validation_rerank_summary"] = summarize_records(validation_rr)
        metadata["test_policy_is_validation_locked"] = True
        write_json(output_dir / "e2e_experiment_config.json", metadata)
        write_json(output_dir / "best_tau.json", {
            "best_tau": selected_threshold,
            "source": "validation",
            "selection_objective": threshold_selection["selection_objective"],
            "test_policy": "hard_filter",
            "run_fingerprint": run_fingerprint,
        })
        write_json(output_dir / "validation_threshold_selection.json", {
            "metadata": metadata,
            "selection": threshold_selection,
            "baseline_validation_summary": summarize_records(validation_baseline),
            "rerank_validation_summary": summarize_records(validation_rr),
            "threshold_results": {
                threshold: {
                    "summary": summarize_records(records),
                    "results": records,
                }
                for threshold, records in sorted(validation_by_threshold.items())
            },
        })
        write_json(output_dir / "e2e_candidate_cache.json", {
            "metadata": metadata,
            "results": [
                candidate_cache_record(query, bundle, long_term_knowledge)
                for query, _, bundle, long_term_knowledge in prepared
            ],
        })
        save_condition(output_dir, "baseline", metadata, validation_baseline + test_baseline)
        save_condition(output_dir, "hf_0_5", metadata, test_hf05)
        save_condition(output_dir, "hf_tau_val", metadata, test_hf_val)
        save_condition(output_dir, "rerank", metadata, validation_rr + test_rr)
        # Paper-facing aliases keep the final policy and comparator names explicit.
        save_condition(output_dir, "reliability", metadata, test_hf_val)
        save_condition(output_dir, "reranking", metadata, validation_rr + test_rr)
        write_json(output_dir / "threshold_sensitivity_results.json", {
            "metadata": metadata,
            "selected_threshold": selected_threshold,
            "best_tau": selected_threshold,
            "thresholds": {
                threshold: {
                    "summary": summarize_records(records),
                    "results": records,
                }
                for threshold, records in sorted(validation_by_threshold.items())
            },
        })
        combined = {
            "metadata": metadata,
            "validation": {
                "baseline": summarize_records(validation_baseline),
                "rerank": summarize_records(validation_rr),
                "thresholds": {
                    threshold: summarize_records(records)
                    for threshold, records in sorted(validation_by_threshold.items())
                },
            },
            "test": {
                "baseline": summarize_records(test_baseline),
                "hf_0_5": summarize_records(test_hf05),
                "hf_tau_val": summarize_records(test_hf_val),
                "rerank": summarize_records(test_rr),
            },
        }
        write_json(output_dir / "e2e_experiment_results.json", combined)
        print("MemoryOS E2E reliability evaluation completed")
        print(f"  validation queries: {split_meta['validation_count']}")
        print(f"  test queries: {split_meta['test_count']}")
        print(f"  selected tau_val: {selected_threshold:.6f}")
        print(f"  final test policy: hard_filter(tau_val={selected_threshold:.6f})")
        for condition, records in (
            ("baseline", test_baseline),
            ("hf_0_5", test_hf05),
            ("hf_tau_val", test_hf_val),
            ("rerank", test_rr),
        ):
            print(f"  {condition} test F1: {summarize_records(records).get('answer_f1_mean', 0.0):.6f}")
        return combined
    finally:
        for store in checkpoints:
            store.flush_all()
        progress.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Leakage-safe MemoryOS reliability E2E evaluation")
    parser.add_argument("--mode", choices=("all", "baseline", "reliability"), default="all",
                        help="all/reliability runs validation search plus test comparators; baseline skips predictor")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--storage-root", default=str(DEFAULT_STORAGE_ROOT))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--embedding-model", default="paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N queries after preserving LoCoMo query order (smoke test)")
    parser.add_argument("--checkpoint-batch-size", type=int, default=10,
                        help="Persist each strategy after this many completed queries (default: 10)")
    parser.add_argument("--debug-snapshot-check", action="store_true",
                        help="Validate one seeded-random snapshot against its loaded memory objects")
    parser.add_argument("--debug-snapshot-only", action="store_true",
                        help="Run the snapshot check and exit before retrieval or answer generation")
    parser.add_argument("--debug-snapshot-seed", type=int, default=RANDOM_SEED,
                        help="Seed used to select the debug conversation (default: 42)")
    parser.add_argument("--validation-ratio", type=float, default=0.20)
    parser.add_argument("--threshold-grid", type=parse_float_list,
                        default=list(DEFAULT_THRESHOLD_GRID),
                        help="Comma-separated HF validation search thresholds")
    parser.add_argument("--tie-tolerance", type=float, default=0.001)
    parser.add_argument("--short-term-capacity", type=int, default=1)
    parser.add_argument("--mid-term-capacity", type=int, default=2000)
    parser.add_argument("--retrieval-queue-capacity", type=int, default=25)
    parser.add_argument("--topic-similarity-threshold", type=float, default=0.6)
    parser.add_argument("--segment-threshold", type=float, default=0.1)
    parser.add_argument("--page-threshold", type=float, default=0.1)
    parser.add_argument("--knowledge-threshold", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=2000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.debug_snapshot_only:
        args.debug_snapshot_check = True
    if not 0.0 < args.validation_ratio < 1.0:
        raise SystemExit("--validation-ratio must be in (0, 1)")
    if args.limit is not None and args.limit < 2:
        raise SystemExit("--limit must be at least 2 so validation and test splits both exist")
    if args.tie_tolerance < 0.0:
        raise SystemExit("--tie-tolerance must be non-negative")
    if args.checkpoint_batch_size < 1:
        raise SystemExit("--checkpoint-batch-size must be at least 1")
    if args.mode == "reliability":
        args.mode = "all"
    try:
        run(args)
    except Exception as exc:
        print(f"E2E evaluation failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
