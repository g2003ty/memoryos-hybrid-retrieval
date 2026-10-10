"""Retrieval-only page-level dense/BM25/RRF comparison.

The script indexes immutable snapshot pages directly.  It never calls an LLM,
constructs prompts, generates answers, or modifies snapshots and existing
experiment artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from tqdm import tqdm
except Exception as exc:  # pragma: no cover - exercised only in minimal envs
    raise RuntimeError("tqdm is required; install it with: pip install tqdm") from exc

try:
    from rank_bm25 import BM25Okapi
except Exception as exc:  # pragma: no cover - exercised only in minimal envs
    raise RuntimeError("rank_bm25 is required; install it with: pip install rank_bm25") from exc


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPT_DIR / "locomo10.json"
DEFAULT_STORAGE_ROOT = SCRIPT_DIR / "e2e_storage"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "page_level_hybrid_retrieval_100"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DENSE_TOP_K = 50
BM25_TOP_K = 50
CANDIDATE_BUDGET = 25
RRF_CONSTANT = 60
METRIC_KS = (1, 5, 10)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"Required file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    import hashlib

    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def mean(values: Sequence[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def import_query_helper() -> Any:
    sys.path.insert(0, str(SCRIPT_DIR))
    import run_v2_relevance_only_e2e as v2
    return v2


def dense_embedding_text(page: Mapping[str, Any]) -> str:
    """The exact text layout used by the original page-level dense runner."""
    return (
        f"User: {page.get('user_input', '')}\n"
        f"Assistant: {page.get('agent_response', '')}\n"
        f"Time: {page.get('timestamp', '')}"
    )


def build_source_map(conversation: Mapping[str, Any]) -> Dict[Tuple[str, str], List[str]]:
    source_map: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    speaker_a = conversation["speaker_a"]
    session_keys = [
        key for key in conversation
        if key.startswith("session_") and not key.endswith("_date_time")
    ]

    def session_number(key: str) -> int:
        match = re.match(r"session_(\d+)$", key)
        return int(match.group(1)) if match else 10**9

    for session_key in sorted(session_keys, key=session_number):
        dialogs = conversation[session_key]
        for index, dialog in enumerate(dialogs):
            if dialog.get("speaker") != speaker_a:
                continue
            user_text = dialog.get("text", "")
            if dialog.get("blip_caption"):
                user_text = f"{user_text} (image description: {dialog['blip_caption']})"
            answer_text = ""
            ids: List[str] = []
            if dialog.get("dia_id"):
                ids.append(str(dialog["dia_id"]))
            if index + 1 < len(dialogs) and dialogs[index + 1].get("speaker") != speaker_a:
                answer_text = dialogs[index + 1].get("text", "")
                if dialogs[index + 1].get("blip_caption"):
                    answer_text = f"{answer_text} (image description: {dialogs[index + 1]['blip_caption']})"
                if dialogs[index + 1].get("dia_id"):
                    ids.append(str(dialogs[index + 1]["dia_id"]))
            source_map[(str(user_text), str(answer_text))].extend(ids)
    return dict(source_map)


def source_ids_for_page(page: Mapping[str, Any], source_map: Mapping[Tuple[str, str], List[str]]) -> List[str]:
    key = (str(page.get("user_input", "")), str(page.get("agent_response", "")))
    return list(source_map.get(key, []))


def load_snapshot_pages(dataset: Sequence[Mapping[str, Any]], storage_root: Path,
                        sample_ids: Iterable[str]) -> Tuple[
                            Dict[str, List[Dict[str, Any]]],
                            Dict[str, Any],
                            Dict[str, Dict[str, int]],
                        ]:
    pages_by_sample: Dict[str, List[Dict[str, Any]]] = {}
    snapshot_files: Dict[str, Any] = {}
    page_counts: Dict[str, Dict[str, int]] = {}
    samples_by_id = {str(sample.get("sample_id")): sample for sample in dataset}
    for sample_id in dict.fromkeys(str(value) for value in sample_ids):
        sample = samples_by_id.get(sample_id)
        if sample is None:
            raise RuntimeError(f"Dataset has no sample {sample_id}")
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)
        sample_dir = storage_root / safe_id
        mid_path = sample_dir / "mid_term.json"
        manifest_path = sample_dir / "snapshot_manifest.json"
        if not mid_path.is_file() or not manifest_path.is_file():
            raise RuntimeError(
                f"Snapshot is missing for {sample_id}: expected {mid_path} and {manifest_path}"
            )
        snapshot_files[sample_id] = {
            "mid_term": {"path": str(mid_path.resolve()), "sha256": sha256_file(mid_path)},
            "manifest": {"path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path)},
        }
        payload = load_json(mid_path)
        sessions = payload.get("sessions", {}) if isinstance(payload, Mapping) else {}
        if not isinstance(sessions, Mapping):
            raise RuntimeError(f"Snapshot mid_term.sessions is invalid for {sample_id}")
        source_map = build_source_map(sample["conversation"])
        raw_records: List[Dict[str, Any]] = []
        initial_rank = 0
        for session_id, session in sessions.items():
            if not isinstance(session, Mapping):
                continue
            details = session.get("details", [])
            if not isinstance(details, list):
                continue
            for page in details:
                if not isinstance(page, Mapping):
                    continue
                page_copy = dict(page)
                page_id = str(page_copy.get("page_id", f"candidate_{initial_rank}"))
                page_copy["page_id"] = page_id
                raw_records.append({
                    "page_id": page_id,
                    "page": page_copy,
                    "initial_rank": initial_rank,
                    "source_dialogue_ids": source_ids_for_page(page_copy, source_map),
                    "session_id": str(session_id),
                    "session_summary": session.get("summary"),
                })
                initial_rank += 1
        if not raw_records:
            raise RuntimeError(f"Snapshot has no raw dialogue pages for {sample_id}")
        seen: set[str] = set()
        deduplicated: List[Dict[str, Any]] = []
        for record in raw_records:
            if record["page_id"] in seen:
                continue
            seen.add(record["page_id"])
            deduplicated.append(record)
        if len(seen) != len(deduplicated):
            raise RuntimeError(f"Duplicate page_id remains after deduplication for {sample_id}")
        page_counts[sample_id] = {
            "page_count_before_deduplication": len(raw_records),
            "page_count_after_deduplication": len(deduplicated),
            "duplicate_page_ids_removed": len(raw_records) - len(deduplicated),
        }
        pages_by_sample[sample_id] = deduplicated
    return pages_by_sample, snapshot_files, page_counts


def load_embedding_model(model_name: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError("sentence-transformers is required for dense retrieval") from exc
    return SentenceTransformer(model_name)


def encode_embedding_texts(model: Any, texts: Sequence[str]) -> Any:
    return model.encode(
        list(texts),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def whitespace_punctuation_tokenize(value: Any) -> List[str]:
    """Split on whitespace and punctuation while preserving timestamp tokens."""
    text = str(value or "").lower()
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def document_text(page: Mapping[str, Any]) -> str:
    return (
        f"{page.get('user_input', '')} "
        f"{page.get('agent_response', '')} "
        f"{page.get('timestamp', '')}"
    )


def snapshot_file_hashes(storage_root: Path, sample_ids: Iterable[str]) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for sample_id in sample_ids:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_id))
        sample_dir = storage_root / safe_id
        if not sample_dir.is_dir():
            raise RuntimeError(f"Snapshot directory does not exist: {sample_dir}")
        files = sorted(sample_dir.glob("*.json"))
        if not files:
            raise RuntimeError(f"Snapshot directory has no JSON files: {sample_dir}")
        for path in files:
            hashes[str(path.resolve())] = sha256_file(path)
    return hashes


def rank_indices(scores: Sequence[float], records: Sequence[Mapping[str, Any]], limit: int) -> List[int]:
    indexed = list(range(len(records)))
    ranked = sorted(
        indexed,
        key=lambda index: (
            -float(scores[index]),
            int(records[index]["initial_rank"]),
            str(records[index]["page_id"]),
        ),
    )
    return ranked[:limit]


def candidate_audit(record: Mapping[str, Any], dense_rank: Optional[int],
                    dense_score: Optional[float], bm25_rank: Optional[int],
                    bm25_score: Optional[float], rrf_score: Optional[float],
                    retrieval_source: str) -> Dict[str, Any]:
    page = dict(record["page"])
    return {
        "page_id": str(record["page_id"]),
        "user_input": page.get("user_input", ""),
        "agent_response": page.get("agent_response", ""),
        "timestamp": page.get("timestamp"),
        "source_dialogue_ids": [str(value) for value in record.get("source_dialogue_ids", [])],
        "initial_rank": int(record["initial_rank"]),
        "dense_rank": dense_rank,
        "dense_score": dense_score,
        "bm25_rank": bm25_rank,
        "bm25_score": bm25_score,
        "rrf_score": rrf_score,
        "retrieval_source": retrieval_source,
        "session_id": record.get("session_id"),
        "session_summary": record.get("session_summary"),
        "page": page,
    }


def build_condition_candidates(records: Sequence[Mapping[str, Any]],
                               dense_scores: Sequence[float],
                               bm25_scores: Sequence[float]) -> Dict[str, List[Dict[str, Any]]]:
    dense_indices = rank_indices(dense_scores, records, DENSE_TOP_K)
    bm25_indices = rank_indices(bm25_scores, records, BM25_TOP_K)
    dense_rank = {index: rank for rank, index in enumerate(dense_indices, 1)}
    bm25_rank = {index: rank for rank, index in enumerate(bm25_indices, 1)}
    dense_score_by_index = {index: float(dense_scores[index]) for index in dense_indices}
    bm25_score_by_index = {index: float(bm25_scores[index]) for index in bm25_indices}

    dense_candidates = [
        candidate_audit(
            records[index], dense_rank[index], dense_score_by_index[index],
            bm25_rank.get(index), bm25_score_by_index.get(index), None, "dense",
        )
        for index in dense_indices[:CANDIDATE_BUDGET]
    ]
    bm25_candidates = [
        candidate_audit(
            records[index], dense_rank.get(index), dense_score_by_index.get(index),
            bm25_rank[index], bm25_score_by_index[index], None, "bm25",
        )
        for index in bm25_indices[:CANDIDATE_BUDGET]
    ]

    union_indices = set(dense_indices) | set(bm25_indices)
    hybrid_rows: List[Tuple[float, int, str, int]] = []
    for index in union_indices:
        rrf = 0.0
        if index in dense_rank:
            rrf += 1.0 / (RRF_CONSTANT + dense_rank[index])
        if index in bm25_rank:
            rrf += 1.0 / (RRF_CONSTANT + bm25_rank[index])
        hybrid_rows.append((rrf, int(records[index]["initial_rank"]), str(records[index]["page_id"]), index))
    hybrid_rows.sort(key=lambda item: (-item[0], item[1], item[2]))
    hybrid_candidates = [
        candidate_audit(
            records[index], dense_rank.get(index), dense_score_by_index.get(index),
            bm25_rank.get(index), bm25_score_by_index.get(index), float(rrf),
            "both" if index in dense_rank and index in bm25_rank else (
                "dense" if index in dense_rank else "bm25"
            ),
        )
        for rrf, _, _, index in hybrid_rows[:CANDIDATE_BUDGET]
    ]
    for name, candidates in {
        "dense_only": dense_candidates,
        "bm25_only": bm25_candidates,
        "dense_bm25_hybrid": hybrid_candidates,
    }.items():
        ids = [str(candidate["page_id"]) for candidate in candidates]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"Duplicate page_id in {name} candidates")
    return {
        "dense_only": dense_candidates,
        "bm25_only": bm25_candidates,
        "dense_bm25_hybrid": hybrid_candidates,
    }


def source_ids(candidate: Mapping[str, Any]) -> List[str]:
    return [str(value) for value in candidate.get("source_dialogue_ids", [])]


def recall_at_k(candidates: Sequence[Mapping[str, Any]], gold: Sequence[str], k: int) -> float:
    gold_set = {str(value) for value in gold}
    if not gold_set:
        return 0.0
    retrieved = {source for candidate in candidates[:k] for source in source_ids(candidate)}
    return len(gold_set & retrieved) / len(gold_set)


def hit_at_k(candidates: Sequence[Mapping[str, Any]], gold: Sequence[str], k: int) -> int:
    return int(recall_at_k(candidates, gold, k) > 0.0)


def best_evidence_rank(candidates: Sequence[Mapping[str, Any]], gold: Sequence[str]) -> Optional[int]:
    gold_set = {str(value) for value in gold}
    if not gold_set:
        return None
    for position, candidate in enumerate(candidates, 1):
        if gold_set & set(source_ids(candidate)):
            return position
    return None


def condition_metrics(candidates: Sequence[Mapping[str, Any]], gold: Sequence[str]) -> Dict[str, Any]:
    return {
        **{
            f"recall_at_{k}": recall_at_k(candidates, gold, k)
            for k in METRIC_KS
        },
        **{
            f"hit_rate_at_{k}": hit_at_k(candidates, gold, k)
            for k in METRIC_KS
        },
        **{
            f"evidence_level_recall_at_{k}": recall_at_k(candidates, gold, k)
            for k in METRIC_KS
        },
        "best_evidence_rank": best_evidence_rank(candidates, gold),
        "candidate_count": len(candidates),
    }


def aggregate_condition(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {key: mean([float(row[key]) for row in rows]) for key in (
        *(f"recall_at_{k}" for k in METRIC_KS),
        *(f"hit_rate_at_{k}" for k in METRIC_KS),
        *(f"evidence_level_recall_at_{k}" for k in METRIC_KS),
    )}
    ranks = [row["best_evidence_rank"] for row in rows if row.get("best_evidence_rank") is not None]
    result.update({
        "query_count": len(rows),
        "avg_best_evidence_rank": mean([float(value) for value in ranks]),
        "best_evidence_rank_hit_count": len(ranks),
        "best_evidence_rank_miss_count": len(rows) - len(ranks),
        "candidate_count_avg": mean([float(row["candidate_count"]) for row in rows]),
    })
    return result


def write_condition_caches(output_dir: Path, queries: Sequence[Mapping[str, Any]],
                           result_rows: Sequence[Mapping[str, Any]],
                           metadata: Mapping[str, Any]) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    for condition in ("dense_only", "bm25_only", "dense_bm25_hybrid"):
        condition_dir = output_dir / "candidate_cache" / condition
        condition_dir.mkdir(parents=True, exist_ok=True)
        rows_by_sample: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for row in result_rows:
            rows_by_sample[str(row["sample_id"])].append(row)
        for sample_id, rows in rows_by_sample.items():
            safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)
            condition_rows = []
            for row in rows:
                condition_rows.append({
                    "query_id": row["query_id"],
                    "query": row["query"],
                    "sample_id": row["sample_id"],
                    "evidence": row["evidence"],
                    "bundle": row["conditions"][condition]["candidates"],
                })
            write_json(condition_dir / f"conv_{safe_id}.json", {
                "metadata": dict(metadata),
                "condition": condition,
                "results": condition_rows,
            })
        paths[condition] = str(condition_dir)
    return paths


def run(args: argparse.Namespace) -> Dict[str, Any]:
    v2 = import_query_helper()
    dataset_path = Path(args.dataset).resolve()
    storage_root = Path(args.storage_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not dataset_path.is_file():
        raise RuntimeError(f"Dataset does not exist: {dataset_path}")
    dataset = load_json(dataset_path)
    if not isinstance(dataset, list):
        raise RuntimeError(f"Dataset must be a JSON list: {dataset_path}")
    queries = v2.flatten_queries(dataset, args.limit)
    sample_ids = list(dict.fromkeys(str(query["sample_id"]) for query in queries))
    snapshot_before = snapshot_file_hashes(storage_root, sample_ids)
    pages_by_sample, snapshot_loader_files, page_counts = load_snapshot_pages(
        dataset, storage_root, sample_ids
    )

    model = load_embedding_model(args.embedding_model)
    records_by_sample: Dict[str, List[Dict[str, Any]]] = {}
    vectors_by_sample: Dict[str, Any] = {}
    bm25_by_sample: Dict[str, Any] = {}
    query_vectors_by_sample: Dict[str, Any] = {}
    query_groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for query in queries:
        query_groups[str(query["sample_id"])].append(query)

    for sample_id in sample_ids:
        records = pages_by_sample[sample_id]
        # Keep dense retrieval byte-for-byte aligned with the existing
        # page-level experiment. BM25 intentionally uses its own required
        # document text (user_input + agent_response + timestamp).
        dense_texts = [dense_embedding_text(record["page"]) for record in records]
        bm25_texts = [document_text(record["page"]) for record in records]
        vectors = encode_embedding_texts(model, dense_texts)
        if len(vectors) != len(records):
            raise RuntimeError(f"Dense embedding count mismatch for {sample_id}")
        records_by_sample[sample_id] = records
        vectors_by_sample[sample_id] = vectors
        tokenized_documents = [whitespace_punctuation_tokenize(text) for text in bm25_texts]
        bm25_by_sample[sample_id] = BM25Okapi(tokenized_documents, k1=1.5, b=0.75)
        query_vectors_by_sample[sample_id] = encode_embedding_texts(
            model, [query["query"] for query in query_groups[sample_id]]
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np
    np.savez_compressed(
        output_dir / "dense_embeddings.npz",
        **{sample_id: vectors_by_sample[sample_id] for sample_id in sample_ids},
    )
    write_json(output_dir / "dense_index.json", {
        "embedding_model": args.embedding_model,
        "embedding_normalized": True,
        "top_k": DENSE_TOP_K,
        "pages_by_sample": {
            sample_id: [
                {
                    "page_id": record["page_id"],
                    "initial_rank": record["initial_rank"],
                    "document_text": dense_embedding_text(record["page"]),
                }
                for record in records_by_sample[sample_id]
            ]
            for sample_id in sample_ids
        },
    })
    write_json(output_dir / "bm25_index.json", {
        "tokenizer": "whitespace_and_punctuation",
        "document_text": "user_input + agent_response + timestamp",
        "k1": 1.5,
        "b": 0.75,
        "top_k": BM25_TOP_K,
        "documents_by_sample": {
            sample_id: [
                {
                    "page_id": record["page_id"],
                    "initial_rank": record["initial_rank"],
                    "document_text": document_text(record["page"]),
                    "tokens": whitespace_punctuation_tokenize(document_text(record["page"])),
                }
                for record in records_by_sample[sample_id]
            ]
            for sample_id in sample_ids
        },
    })

    result_rows: List[Dict[str, Any]] = []
    query_positions: Dict[str, int] = defaultdict(int)
    total_queries = len(queries)
    started = time.monotonic()
    progress = tqdm(
        total=total_queries,
        desc="retrieval",
        unit="query",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )
    try:
        for query_index, query in enumerate(queries, 1):
            sample_id = str(query["sample_id"])
            conversation_queries = query_groups[sample_id]
            conversation_position = query_positions[sample_id] + 1
            query_positions[sample_id] = conversation_position
            records = records_by_sample[sample_id]
            dense_scores = vectors_by_sample[sample_id] @ query_vectors_by_sample[sample_id][conversation_position - 1]
            bm25_scores = bm25_by_sample[sample_id].get_scores(
                whitespace_punctuation_tokenize(query["query"])
            )
            conditions = build_condition_candidates(
                records,
                [float(value) for value in dense_scores],
                [float(value) for value in bm25_scores],
            )
            condition_results: Dict[str, Any] = {}
            for condition, candidates in conditions.items():
                condition_results[condition] = {
                    **condition_metrics(candidates, query["evidence"]),
                    "candidates": candidates,
                }
            result_rows.append({
                "query_id": query["query_id"],
                "sample_id": sample_id,
                "query": query["query"],
                "evidence": list(query["evidence"]),
                "category": query.get("category"),
                "conditions": condition_results,
                "page_count_before_deduplication": page_counts[sample_id]["page_count_before_deduplication"],
                "page_count_after_deduplication": page_counts[sample_id]["page_count_after_deduplication"],
                "duplicate_page_ids_removed": page_counts[sample_id]["duplicate_page_ids_removed"],
            })
            elapsed = time.monotonic() - started
            progress.set_postfix_str(
                f"conversation={sample_id} query={query_index}/{total_queries} "
                f"conversation_query={conversation_position}/{len(conversation_queries)} "
                f"elapsed={elapsed:.0f}s"
            )
            progress.update(1)
    finally:
        progress.close()

    snapshot_after = snapshot_file_hashes(storage_root, sample_ids)
    snapshot_unchanged = snapshot_before == snapshot_after
    condition_metric_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in result_rows:
        for condition in ("dense_only", "bm25_only", "dense_bm25_hybrid"):
            condition_metric_rows[condition].append(row["conditions"][condition])
    metrics = {
        condition: aggregate_condition(rows)
        for condition, rows in condition_metric_rows.items()
    }
    metadata = {
        "experiment": "page_level_dense_bm25_rrf",
        "dataset": str(dataset_path),
        "storage_root": str(storage_root),
        "limit": args.limit,
        "embedding_model": args.embedding_model,
        "embedding_normalized": True,
        "dense_document_text": "existing page-level build_embedding_text: User + Assistant + Time",
        "document_text": "user_input + agent_response + timestamp",
        "tokenizer": "whitespace_and_punctuation",
        "dense_top_k": DENSE_TOP_K,
        "bm25_top_k": BM25_TOP_K,
        "candidate_budget": CANDIDATE_BUDGET,
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "rrf_constant": RRF_CONSTANT,
        "rrf_rank_starts_at": 1,
        "session_summary_used_for_filtering": False,
        "llm_called": False,
        "snapshot_modified": False,
        "locomo_modified": False,
        "existing_experiment_json_modified": False,
        "query_order_hash": stable_hash([query["query_id"] for query in queries]),
        "snapshot_loader_files": snapshot_loader_files,
        "page_counts_by_sample": page_counts,
    }
    cache_paths = write_condition_caches(output_dir, queries, result_rows, metadata)
    comparison = {
        "experiment": "page_level_dense_bm25_rrf",
        "query_count": len(queries),
        "config": metadata,
        "metrics": metrics,
        "query_results": result_rows,
        "candidate_cache_paths": cache_paths,
        "deduplication_stats": {
            "page_count_before_deduplication": sum(value["page_count_before_deduplication"] for value in page_counts.values()),
            "page_count_after_deduplication": sum(value["page_count_after_deduplication"] for value in page_counts.values()),
            "duplicate_page_ids_removed": sum(value["duplicate_page_ids_removed"] for value in page_counts.values()),
            "by_sample": page_counts,
        },
        "validation": {
            "dense_only_metrics_present": "dense_only" in metrics,
            "bm25_only_metrics_present": "bm25_only" in metrics,
            "hybrid_metrics_present": "dense_bm25_hybrid" in metrics,
            "candidate_budget_all_25": all(
                row["conditions"][condition]["candidate_count"] <= CANDIDATE_BUDGET
                for row in result_rows
                for condition in ("dense_only", "bm25_only", "dense_bm25_hybrid")
            ),
            "page_id_unique_per_condition": all(
                len([candidate["page_id"] for candidate in row["conditions"][condition]["candidates"]])
                == len({candidate["page_id"] for candidate in row["conditions"][condition]["candidates"]})
                for row in result_rows
                for condition in ("dense_only", "bm25_only", "dense_bm25_hybrid")
            ),
            "audit_fields_complete": all(
                all(
                    all(field in candidate for field in (
                        "page_id", "user_input", "agent_response", "timestamp",
                        "source_dialogue_ids", "initial_rank", "dense_rank", "dense_score",
                        "bm25_rank", "bm25_score", "rrf_score", "retrieval_source",
                    ))
                    for candidate in row["conditions"][condition]["candidates"]
                )
                for row in result_rows
                for condition in ("dense_only", "bm25_only", "dense_bm25_hybrid")
            ),
            "snapshot_hash_unchanged": snapshot_unchanged,
            "snapshot_hash_before": stable_hash(snapshot_before),
            "snapshot_hash_after": stable_hash(snapshot_after),
            "llm_called": False,
            "prompt_generation_executed": False,
        },
    }
    write_json(output_dir / "retrieval_comparison.json", comparison)
    write_json(output_dir / "experiment_metadata.json", metadata)
    write_json(output_dir / "snapshot_integrity.json", {
        "snapshot_hash_before": snapshot_before,
        "snapshot_hash_after": snapshot_after,
        "unchanged": snapshot_unchanged,
    })
    print(f"Wrote retrieval comparison to {output_dir / 'retrieval_comparison.json'}")
    return comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Page-level dense/BM25/RRF retrieval-only experiment")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--storage-root", default=str(DEFAULT_STORAGE_ROOT))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--limit", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.output_dir is None:
        args.output_dir = str(SCRIPT_DIR / f"page_level_hybrid_retrieval_{args.limit}")
    try:
        run(args)
    except Exception as exc:
        print(f"Page-level hybrid retrieval failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
