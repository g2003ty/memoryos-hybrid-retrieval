"""Page-level candidate construction and Recall@10 evaluation.

This is the first retrieval-only experiment in the retrieval redesign.  Every
raw dialogue QA pair in the immutable snapshot is indexed independently;
session summaries are retained only as metadata and never used as a first
stage filter.  The script writes a new external embedding/index/cache tree and
never modifies snapshots, LoCoMo, or existing experiment artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPT_DIR / "locomo10.json"
DEFAULT_STORAGE_ROOT = SCRIPT_DIR / "e2e_storage"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "page_level_retrieval_100"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_QUEUE_CAPACITY = 25
TOP_K = 10


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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_evidence(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)]


def flatten_queries(dataset: Sequence[Mapping[str, Any]], limit: Optional[int]) -> List[Dict[str, Any]]:
    queries: List[Dict[str, Any]] = []
    for sample_index, sample in enumerate(dataset):
        sample_id = str(sample.get("sample_id", f"sample_{sample_index}"))
        for qa_index, qa in enumerate(sample.get("qa", [])):
            answer = qa.get("answer", "")
            if answer in (None, ""):
                answer = qa.get("adversarial_answer", "")
            queries.append({
                "query_id": f"{sample_id}:{qa_index}",
                "sample_id": sample_id,
                "qa_index": qa_index,
                "query": str(qa.get("question", "")),
                "reference_answer": str(answer),
                "evidence": normalize_evidence(qa.get("evidence")),
                "category": qa.get("category"),
            })
    if limit is not None:
        queries = queries[:limit]
    if not queries:
        raise RuntimeError("No queries selected")
    return queries


def safe_sample_dir(storage_root: Path, sample_id: str) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)
    return storage_root / safe_id


def build_source_map(conversation: Mapping[str, Any]) -> Dict[Tuple[str, str], List[str]]:
    """Mirror the existing evidence mapping without changing dataset data."""
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
    sample_id_set = {str(value) for value in sample_ids}
    samples_by_id = {str(sample.get("sample_id")): sample for sample in dataset}
    for sample_id in sample_id_set:
        sample = samples_by_id.get(sample_id)
        if sample is None:
            raise RuntimeError(f"Dataset has no sample {sample_id}")
        sample_dir = safe_sample_dir(storage_root, sample_id)
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
        mid_payload = load_json(mid_path)
        sessions = mid_payload.get("sessions", {}) if isinstance(mid_payload, Mapping) else {}
        if not isinstance(sessions, Mapping):
            raise RuntimeError(f"Snapshot mid_term.sessions is invalid for {sample_id}")
        source_map = build_source_map(sample["conversation"])
        page_records: List[Dict[str, Any]] = []
        initial_rank = 0
        for session_id, session in sessions.items():
            if not isinstance(session, Mapping):
                continue
            summary = session.get("summary")
            details = session.get("details", [])
            if not isinstance(details, list):
                continue
            for page in details:
                if not isinstance(page, Mapping):
                    continue
                page_copy = dict(page)
                page_id = str(page_copy.get("page_id", f"candidate_{initial_rank}"))
                page_copy["page_id"] = page_id
                page_records.append({
                    "page_id": page_id,
                    "page": page_copy,
                    "initial_rank": initial_rank,
                    "source_dialogue_ids": source_ids_for_page(page_copy, source_map),
                    "session_id": str(session_id),
                    "session_summary": summary,
                })
                initial_rank += 1
        if not page_records:
            raise RuntimeError(f"Snapshot has no raw dialogue pages for {sample_id}")
        page_count_before = len(page_records)
        seen_page_ids = set()
        deduplicated_records: List[Dict[str, Any]] = []
        for record in page_records:
            page_id = str(record["page_id"])
            if page_id in seen_page_ids:
                continue
            seen_page_ids.add(page_id)
            deduplicated_records.append(record)
        # The first occurrence is retained above; this check makes the
        # invariant explicit before any embeddings are generated.
        if len(seen_page_ids) != len(deduplicated_records):
            raise RuntimeError(f"Duplicate page_id remains after deduplication for {sample_id}")
        page_counts[sample_id] = {
            "page_count_before_deduplication": page_count_before,
            "page_count_after_deduplication": len(deduplicated_records),
            "duplicate_page_ids_removed": page_count_before - len(deduplicated_records),
        }
        pages_by_sample[sample_id] = deduplicated_records
    return pages_by_sample, snapshot_files, page_counts


def build_embedding_text(page: Mapping[str, Any]) -> str:
    return (
        f"User: {page.get('user_input', '')}\n"
        f"Assistant: {page.get('agent_response', '')}\n"
        f"Time: {page.get('timestamp', '')}"
    )


def load_sentence_transformer(model_name: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError(
            "sentence-transformers is required for page-level retrieval; "
            "install it before running this experiment"
        ) from exc
    return SentenceTransformer(model_name)


def encode_texts(model: Any, texts: Sequence[str]) -> Any:
    vectors = model.encode(
        list(texts),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors


def rank_page_candidates(page_records: Sequence[Mapping[str, Any]],
                         page_vectors: Any, query_vector: Any,
                         queue_capacity: int) -> List[Dict[str, Any]]:
    scores = page_vectors @ query_vector
    indexed = list(enumerate(page_records))
    ranked = sorted(
        indexed,
        key=lambda item: (-float(scores[item[0]]), int(item[1]["initial_rank"]), item[0]),
    )[:queue_capacity]
    bundle: List[Dict[str, Any]] = []
    for rank, (vector_index, record) in enumerate(ranked):
        candidate = {
            "page_id": str(record["page_id"]),
            "page": dict(record["page"]),
            "initial_rank": int(record["initial_rank"]),
            "retrieval_rank": rank + 1,
            "retrieval_score": float(scores[vector_index]),
            "features": {"embedding_score": float(scores[vector_index])},
            "source_dialogue_ids": list(record["source_dialogue_ids"]),
            "session_id": record.get("session_id"),
            "session_summary": record.get("session_summary"),
        }
        bundle.append(candidate)
    return bundle


def recall_at_k(bundle: Sequence[Mapping[str, Any]], gold: Sequence[str], k: int) -> float:
    gold_set = {str(item) for item in gold}
    if not gold_set:
        return 0.0
    retrieved = {
        source_id
        for candidate in bundle[:k]
        for source_id in candidate.get("source_dialogue_ids", [])
    }
    return len(gold_set & retrieved) / len(gold_set)


def hit_at_k(bundle: Sequence[Mapping[str, Any]], gold: Sequence[str], k: int) -> int:
    return int(recall_at_k(bundle, gold, k) > 0.0)


def write_candidate_cache(output_dir: Path, queries: Sequence[Mapping[str, Any]],
                          results_by_id: Mapping[str, Mapping[str, Any]], metadata: Mapping[str, Any]) -> Path:
    cache_dir = output_dir / "candidate_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    by_sample: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for query in queries:
        by_sample[str(query["sample_id"])].append(results_by_id[query["query_id"]])
    for sample_id, rows in by_sample.items():
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)
        write_json(cache_dir / f"conv_{safe_id}.json", {
            "metadata": dict(metadata),
            "results": list(rows),
        })
    write_json(output_dir / "page_level_candidate_cache.json", {
        "metadata": dict(metadata),
        "results": [results_by_id[query["query_id"]] for query in queries],
    })
    return cache_dir


def run(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_path = Path(args.dataset).resolve()
    storage_root = Path(args.storage_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not dataset_path.is_file():
        raise RuntimeError(f"Dataset does not exist: {dataset_path}")
    dataset = load_json(dataset_path)
    if not isinstance(dataset, list):
        raise RuntimeError(f"Dataset must be a JSON list: {dataset_path}")
    queries = flatten_queries(dataset, args.limit)
    sample_ids = list(dict.fromkeys(query["sample_id"] for query in queries))
    pages_by_sample, snapshot_files_before, page_counts_by_sample = load_snapshot_pages(
        dataset, storage_root, sample_ids
    )

    model = load_sentence_transformer(args.embedding_model)
    all_page_records: List[Dict[str, Any]] = []
    page_records_by_sample: Dict[str, List[Dict[str, Any]]] = {}
    page_vectors_by_sample: Dict[str, Any] = {}
    for sample_id in sample_ids:
        page_records = pages_by_sample[sample_id]
        texts = [build_embedding_text(record["page"]) for record in page_records]
        vectors = encode_texts(model, texts)
        if len(vectors) != len(page_records):
            raise RuntimeError(f"Embedding count mismatch for {sample_id}")
        page_records_by_sample[sample_id] = page_records
        page_vectors_by_sample[sample_id] = vectors
        for record, vector, text in zip(page_records, vectors, texts):
            all_page_records.append({
                "sample_id": sample_id,
                "page_id": record["page_id"],
                "embedding_text": text,
                "embedding_dimension": int(len(vector)),
                "initial_rank": record["initial_rank"],
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = output_dir / "page_level_embeddings.npz"
    import numpy as np
    concatenated_vectors = np.concatenate([page_vectors_by_sample[sample_id] for sample_id in sample_ids], axis=0)
    np.savez_compressed(embedding_path, embeddings=concatenated_vectors)
    write_json(output_dir / "page_level_index.json", {
        "embedding_model": args.embedding_model,
        "embedding_normalized": True,
        "page_count": len(all_page_records),
        "pages": all_page_records,
    })

    query_vectors_by_sample: Dict[str, Any] = {}
    for sample_id in sample_ids:
        sample_queries = [query for query in queries if query["sample_id"] == sample_id]
        query_vectors_by_sample[sample_id] = encode_texts(model, [query["query"] for query in sample_queries])

    results_by_id: Dict[str, Dict[str, Any]] = {}
    query_position_by_sample: Dict[str, int] = defaultdict(int)
    for query in queries:
        sample_id = query["sample_id"]
        position = query_position_by_sample[sample_id]
        query_position_by_sample[sample_id] += 1
        query_vector = query_vectors_by_sample[sample_id][position]
        page_records = page_records_by_sample[sample_id]
        page_vectors = page_vectors_by_sample[sample_id]
        bundle = rank_page_candidates(page_records, page_vectors, query_vector, args.queue_capacity)
        recall = recall_at_k(bundle, query["evidence"], TOP_K)
        hit = hit_at_k(bundle, query["evidence"], TOP_K)
        results_by_id[query["query_id"]] = {
            "query_id": query["query_id"],
            "sample_id": sample_id,
            "query": query["query"],
            "category": query.get("category"),
            "evidence": list(query["evidence"]),
            "candidate_count": len(bundle),
            "candidate_pool_size_before_budget": len(page_records),
            "page_count_before_deduplication": page_counts_by_sample[sample_id]["page_count_before_deduplication"],
            "page_count_after_deduplication": page_counts_by_sample[sample_id]["page_count_after_deduplication"],
            "duplicate_page_ids_removed": page_counts_by_sample[sample_id]["duplicate_page_ids_removed"],
            "retrieved_source_dialogue_ids": sorted({
                source_id
                for candidate in bundle[:TOP_K]
                for source_id in candidate["source_dialogue_ids"]
            }),
            "recall_at_10": recall,
            "hit_at_10": hit,
            "bundle": bundle,
            "session_summary_used_for_filtering": False,
        }

    metadata = {
        "experiment": "page_level_candidate_construction",
        "dataset": str(dataset_path),
        "storage_root": str(storage_root),
        "limit": args.limit,
        "embedding_model": args.embedding_model,
        "embedding_normalized": True,
        "retrieval_algorithm": "direct_page_level_cosine_similarity",
        "session_summary_used_for_filtering": False,
        "queue_capacity": args.queue_capacity,
        "recall_k": TOP_K,
        "candidate_pool_construction_changed": True,
        "retrieval_algorithm_changed": True,
        "snapshot_raw_conversations_unchanged": True,
        "locomo_dataset_unchanged": True,
        "generation_layer_unchanged": True,
        "snapshot_files_before": snapshot_files_before,
        "page_counts_by_sample": page_counts_by_sample,
        "query_order_hash": stable_hash([query["query_id"] for query in queries]),
    }
    cache_dir = write_candidate_cache(output_dir, queries, results_by_id, metadata)

    recalls = [float(results_by_id[query["query_id"]]["recall_at_10"]) for query in queries]
    hits = [int(results_by_id[query["query_id"]]["hit_at_10"]) for query in queries]
    by_sample: Dict[str, List[float]] = defaultdict(list)
    for query in queries:
        by_sample[query["sample_id"]].append(float(results_by_id[query["query_id"]]["recall_at_10"]))
    recall_report = {
        "experiment": "page_level_candidate_construction",
        "query_count": len(queries),
        "recall_at_10": float(statistics.mean(recalls)) if recalls else 0.0,
        "hit_rate_at_10": float(statistics.mean(hits)) if hits else 0.0,
        "recall_at_10_by_conversation": {
            sample_id: float(statistics.mean(values)) for sample_id, values in by_sample.items()
        },
        "candidate_count": {
            "avg_before_budget": mean([float(results_by_id[q["query_id"]]["candidate_pool_size_before_budget"]) for q in queries]),
            "avg_after_budget": mean([float(results_by_id[q["query_id"]]["candidate_count"]) for q in queries]),
        },
        "page_counts": {
            "page_count_before_deduplication": sum(
                value["page_count_before_deduplication"] for value in page_counts_by_sample.values()
            ),
            "page_count_after_deduplication": sum(
                value["page_count_after_deduplication"] for value in page_counts_by_sample.values()
            ),
            "duplicate_page_ids_removed": sum(
                value["duplicate_page_ids_removed"] for value in page_counts_by_sample.values()
            ),
            "by_sample": page_counts_by_sample,
        },
        "outputs": {
            "embedding_cache": str(embedding_path),
            "page_level_index": str(output_dir / "page_level_index.json"),
            "candidate_cache": str(cache_dir),
        },
        "protocol": {
            "retrieval_executed": True,
            "snapshot_modified": False,
            "locomo_modified": False,
            "session_summary_first_stage_filter": False,
            "page_level_candidates": True,
            "generation_executed": False,
        },
    }
    write_json(output_dir / "recall_at_10.json", recall_report)
    write_json(output_dir / "experiment_metadata.json", metadata)

    snapshot_files_after = {
        sample_id: {
            name: {**entry, "sha256_after": sha256_file(Path(entry["path"]))}
            for name, entry in files.items()
        }
        for sample_id, files in snapshot_files_before.items()
    }
    snapshot_unchanged = all(
        entry["sha256"] == entry["sha256_after"]
        for files in snapshot_files_after.values()
        for entry in files.values()
    )
    report = {
        "metadata": metadata,
        "metrics": recall_report,
        "validation": {
            "snapshot_modified": not snapshot_unchanged,
            "snapshot_unchanged": snapshot_unchanged,
            "session_summary_used_for_filtering": False,
            "page_level_candidate_construction": True,
            "page_counts": {
                "page_count_before_deduplication": sum(
                    value["page_count_before_deduplication"] for value in page_counts_by_sample.values()
                ),
                "page_count_after_deduplication": sum(
                    value["page_count_after_deduplication"] for value in page_counts_by_sample.values()
                ),
                "duplicate_page_ids_removed": sum(
                    value["duplicate_page_ids_removed"] for value in page_counts_by_sample.values()
                ),
                "by_sample": page_counts_by_sample,
            },
            "candidate_cache_written": cache_dir.is_dir(),
            "embedding_cache_written": embedding_path.is_file(),
        },
    }
    write_json(output_dir / "page_level_retrieval_report.json", report)
    print(f"Recall@10: {recall_report['recall_at_10']:.6f}")
    print(f"Hit rate@10: {recall_report['hit_rate_at_10']:.6f}")
    print(f"Wrote page-level retrieval artifacts to {output_dir}")
    return report


def mean(values: Sequence[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Page-level candidate construction Recall@10 experiment")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--storage-root", default=str(DEFAULT_STORAGE_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--queue-capacity", type=int, default=DEFAULT_QUEUE_CAPACITY)
    parser.add_argument("--limit", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.queue_capacity < TOP_K:
        raise SystemExit(f"--queue-capacity must be at least {TOP_K}")
    try:
        run(args)
    except Exception as exc:
        print(f"Page-level retrieval E2E failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
