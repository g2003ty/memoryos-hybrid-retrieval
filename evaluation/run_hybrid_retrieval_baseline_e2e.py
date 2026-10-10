"""End-to-end generation using the page-level dense/BM25 RRF cache.

This runner evaluates the hybrid retrieval candidate pool with the original
MemoryOS prompt. It reads baseline answers/F1 from the existing V2 artifact,
does not compute verifier features, and makes one gpt-4o-mini generation call
per query. Retrieval and snapshot state are read-only.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from tqdm import tqdm
except Exception as exc:  # pragma: no cover
    raise RuntimeError("tqdm is required; install it with: pip install tqdm") from exc


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULT = SCRIPT_DIR / "v2_relevance_only_e2e_500.json"
DEFAULT_CACHE = SCRIPT_DIR / "page_level_hybrid_retrieval_500_fixed" / "candidate_cache" / "dense_bm25_hybrid"
DEFAULT_DATASET = SCRIPT_DIR / "locomo10.json"
DEFAULT_STORAGE_ROOT = SCRIPT_DIR / "e2e_storage"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_TOKENS = 2000
CANDIDATE_BUDGET = 25


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"Required JSON does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: Sequence[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, remainder = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {remainder:02d}s"


def load_baseline(path: Path) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    payload = load_json(path)
    rows = payload.get("query_results") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        raise RuntimeError(f"Baseline artifact lacks query_results: {path}")
    by_id = {
        str(row.get("query_id")): dict(row)
        for row in rows
        if isinstance(row, Mapping) and row.get("query_id") not in (None, "")
    }
    if not by_id:
        raise RuntimeError(f"Baseline artifact contains no query results: {path}")
    return dict(payload), by_id


def load_helpers() -> Any:
    sys.path.insert(0, str(SCRIPT_DIR))
    import run_v2_relevance_only_e2e as v2
    return v2


def load_hybrid_cache(path: Path, v2: Any) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    metadata, records = v2.checkpoint_records(path)
    by_id = {
        str(record.get("query_id")): dict(record)
        for record in records
        if record.get("query_id") not in (None, "")
    }
    if not by_id:
        raise RuntimeError(f"Hybrid candidate cache contains no query records: {path}")
    return metadata, by_id


def rrf_ranked_candidates(record: Mapping[str, Any]) -> List[Dict[str, Any]]:
    bundle = record.get("bundle")
    if not isinstance(bundle, list) or not bundle:
        raise RuntimeError(f"Hybrid cache has no candidate bundle for {record.get('query_id')}")
    candidates: List[Dict[str, Any]] = []
    for index, raw in enumerate(bundle):
        if not isinstance(raw, Mapping) or not isinstance(raw.get("page"), Mapping):
            raise RuntimeError(f"Hybrid cache candidate is incomplete for {record.get('query_id')}")
        if raw.get("rrf_score") is None:
            raise RuntimeError(
                f"Hybrid cache candidate {raw.get('page_id')} lacks rrf_score; "
                "this runner does not recompute RRF"
            )
        candidate = dict(raw)
        candidate["page"] = dict(raw["page"])
        candidates.append(candidate)
    candidates.sort(
        key=lambda candidate: (
            -float(candidate["rrf_score"]),
            int(candidate.get("initial_rank", 10**9)),
            str(candidate.get("page_id", "")),
        )
    )
    selected = candidates[:CANDIDATE_BUDGET]
    ids = [str(candidate.get("page_id")) for candidate in selected]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Duplicate page_id remains in hybrid candidate pool for {record.get('query_id')}")
    return selected


def query_ids_from_dataset(dataset: Sequence[Mapping[str, Any]], v2: Any,
                           limit: Optional[int]) -> List[Dict[str, Any]]:
    return v2.flatten_queries(dataset, limit)


def checkpoint_load(path: Path, query_ids: Sequence[str], source_result: str,
                    cache_path: str, model: str, max_tokens: int) -> Dict[str, Dict[str, Any]]:
    if not path.is_file():
        return {}
    payload = load_json(path)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Invalid checkpoint: {path}")
    metadata = payload.get("metadata", {})
    expected = {
        "query_order": list(query_ids),
        "source_result": source_result,
        "candidate_cache": cache_path,
        "model": model,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    if isinstance(metadata, Mapping):
        for key, value in expected.items():
            if key in metadata and metadata[key] != value:
                raise RuntimeError(f"Checkpoint configuration mismatch for {key}")
    records = payload.get("query_results", {})
    if not isinstance(records, Mapping):
        raise RuntimeError(f"Invalid checkpoint query_results: {path}")
    return {
        str(key): dict(value)
        for key, value in records.items()
        if isinstance(value, Mapping)
    }


def checkpoint_save(path: Path, query_ids: Sequence[str], source_result: str,
                    cache_path: str, model: str, max_tokens: int,
                    records: Mapping[str, Mapping[str, Any]]) -> None:
    write_json(path, {
        "metadata": {
            "query_order": list(query_ids),
            "source_result": source_result,
            "candidate_cache": cache_path,
            "model": model,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        },
        "query_results": records,
    })


def run(args: argparse.Namespace) -> Dict[str, Any]:
    v2 = load_helpers()
    result_path = Path(args.baseline_result).resolve()
    cache_path = Path(args.candidate_cache).resolve()
    dataset_path = Path(args.dataset).resolve()
    storage_root = Path(args.storage_root).resolve()
    output_path = Path(args.output).resolve()
    if output_path == result_path:
        raise RuntimeError("Output must not overwrite the baseline V2 artifact")

    baseline_report, baseline_by_id = load_baseline(result_path)
    dataset = load_json(dataset_path)
    if not isinstance(dataset, list):
        raise RuntimeError(f"Dataset must be a JSON list: {dataset_path}")
    queries = query_ids_from_dataset(dataset, v2, args.limit)
    query_ids = [str(query["query_id"]) for query in queries]
    cache_metadata, cache_by_id = load_hybrid_cache(cache_path, v2)

    for query in queries:
        query_id = str(query["query_id"])
        if query_id not in baseline_by_id:
            raise RuntimeError(f"Baseline result is missing query {query_id}")
        if query_id not in cache_by_id:
            raise RuntimeError(f"Hybrid candidate cache is missing query {query_id}")

    source_config = baseline_report.get("config", {}) if isinstance(baseline_report, Mapping) else {}
    source_model = str(source_config.get("model", DEFAULT_MODEL))
    if source_model != DEFAULT_MODEL:
        raise RuntimeError(f"Baseline model is {source_model!r}; expected {DEFAULT_MODEL!r}")
    source_temperature = source_config.get("temperature")
    if source_temperature is not None and float(source_temperature) != 0.0:
        raise RuntimeError(f"Baseline temperature is not 0.0: {source_temperature}")
    source_max_tokens = int(source_config.get("max_tokens", args.max_tokens))
    if args.max_tokens != source_max_tokens:
        raise RuntimeError(
            f"max_tokens differs from baseline: source={source_max_tokens}, requested={args.max_tokens}"
        )
    if int(cache_metadata.get("candidate_budget", CANDIDATE_BUDGET)) != CANDIDATE_BUDGET:
        raise RuntimeError("Hybrid cache candidate budget is not 25")

    client = v2.client_from_env()
    loader_args = argparse.Namespace(
        short_term_capacity=1,
        mid_term_capacity=2000,
        topic_similarity_threshold=0.6,
        retrieval_queue_capacity=CANDIDATE_BUDGET,
    )
    contexts = v2.load_contexts(
        dataset,
        [query["sample_id"] for query in queries],
        storage_root,
        client,
        loader_args,
    )
    generation_client = v2.ExistingPromptLLMAdapter(client)

    checkpoint_path = output_path.with_suffix(".checkpoint.json")
    results_by_id = checkpoint_load(
        checkpoint_path,
        query_ids,
        str(result_path),
        str(cache_path),
        DEFAULT_MODEL,
        args.max_tokens,
    )
    started = time.monotonic()
    progress = tqdm(
        total=len(queries),
        initial=len(results_by_id),
        unit="query",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )
    try:
        for index, query in enumerate(queries, 1):
            query_id = str(query["query_id"])
            if query_id in results_by_id:
                elapsed = time.monotonic() - started
                progress.set_description(f"Query {index}/{len(queries)} resumed")
                progress.set_postfix_str(f"elapsed {format_duration(elapsed)} | ETA 0m 00s")
                continue
            record = cache_by_id[query_id]
            candidates = rrf_ranked_candidates(record)
            context = contexts[query["sample_id"]]
            long_knowledge = list(record.get("long_term_knowledge", []))
            messages, prompt_tokens = v2.build_prompt(
                query,
                context,
                candidates,
                long_knowledge,
                DEFAULT_MODEL,
            )
            answer = generation_client.chat_completion(
                model=DEFAULT_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=args.max_tokens,
            )
            baseline = baseline_by_id[query_id]
            gold = list(query.get("evidence", []))
            results_by_id[query_id] = {
                "query_id": query_id,
                "query": query["query"],
                "sample_id": query["sample_id"],
                "baseline_answer": baseline.get("baseline_answer"),
                "hybrid_answer": answer,
                "baseline_f1": baseline.get("baseline_f1"),
                "hybrid_f1": v2.answer_f1(answer, query["reference_answer"]),
                "baseline_em": baseline.get("baseline_em"),
                "hybrid_em": v2.exact_match(answer, query["reference_answer"]),
                "baseline_evidence_hit": baseline.get("baseline_evidence_hit"),
                "hybrid_evidence_hit": v2.evidence_hit(candidates, gold),
                "candidate_count": len(candidates),
                "candidate_ids": [str(candidate["page_id"]) for candidate in candidates],
                "candidate_pool_hash": baseline.get("candidate_pool_hash"),
                "hybrid_candidate_pool_hash": v2.candidate_pool_hash(candidates),
                "rrf_sorted": True,
                "candidate_audit": candidates,
                "prompt_tokens": prompt_tokens,
                "generation_call_count": 1,
            }
            checkpoint_save(
                checkpoint_path,
                query_ids,
                str(result_path),
                str(cache_path),
                DEFAULT_MODEL,
                args.max_tokens,
                results_by_id,
            )
            elapsed = time.monotonic() - started
            completed = len(results_by_id)
            rate = completed / elapsed if elapsed else 0.0
            eta = (len(queries) - completed) / rate if rate else 0.0
            progress.set_description(f"Query {index}/{len(queries)}")
            progress.set_postfix_str(
                f"elapsed {format_duration(elapsed)} | ETA {format_duration(eta)}"
            )
            progress.update(1)
    finally:
        progress.close()

    rows = [results_by_id[query_id] for query_id in query_ids]
    baseline_f1 = mean([float(row["baseline_f1"]) for row in rows])
    hybrid_f1 = mean([float(row["hybrid_f1"]) for row in rows])
    baseline_em = mean([int(bool(row["baseline_em"])) for row in rows])
    hybrid_em = mean([int(bool(row["hybrid_em"])) for row in rows])
    baseline_hit = mean([float(row["baseline_evidence_hit"]) for row in rows])
    hybrid_hit = mean([float(row["hybrid_evidence_hit"]) for row in rows])
    output = {
        "protocol": {
            "read_only": True,
            "retrieval_executed": False,
            "snapshot_modified": False,
            "locomo_modified": False,
            "v2_relevance_reranking_executed": False,
            "verifier_features_recomputed": False,
            "prompt_generation": "original baseline prompt",
            "single_generation_per_query": True,
            "llm_model": DEFAULT_MODEL,
            "temperature": 0.0,
        },
        "config": {
            "baseline_result": str(result_path),
            "candidate_cache": str(cache_path),
            "dataset": str(dataset_path),
            "storage_root": str(storage_root),
            "candidate_budget": CANDIDATE_BUDGET,
            "model": DEFAULT_MODEL,
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "candidate_order": "rrf_score descending, initial_rank ascending, page_id ascending",
            "cache_metadata": cache_metadata,
        },
        "metrics": {
            "query_count": len(rows),
            "baseline_f1": baseline_f1,
            "hybrid_f1": hybrid_f1,
            "delta_f1": hybrid_f1 - baseline_f1,
            "baseline_em": baseline_em,
            "hybrid_em": hybrid_em,
            "delta_em": hybrid_em - baseline_em,
            "baseline_evidence_hit_rate": baseline_hit,
            "hybrid_evidence_hit_rate": hybrid_hit,
            "delta_evidence_hit_rate": hybrid_hit - baseline_hit,
        },
        "query_results": rows,
        "validation": {
            "query_count_matches_limit": len(rows) == len(queries),
            "candidate_budget_all_25_or_less": all(row["candidate_count"] <= CANDIDATE_BUDGET for row in rows),
            "candidate_ids_unique": all(
                len(row["candidate_ids"]) == len(set(row["candidate_ids"])) for row in rows
            ),
            "rrf_sorted": all(row["rrf_sorted"] for row in rows),
            "generation_call_count": sum(int(row["generation_call_count"]) for row in rows),
            "expected_generation_call_count": len(rows),
            "retrieval_executed": False,
            "snapshot_modified": False,
            "verifier_features_recomputed": False,
        },
    }
    write_json(output_path, output)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(f"Wrote {output_path}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid retrieval baseline generation E2E")
    parser.add_argument("--baseline-result", default=str(DEFAULT_RESULT))
    parser.add_argument("--candidate-cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--storage-root", default=str(DEFAULT_STORAGE_ROOT))
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.output is None:
        args.output = str(SCRIPT_DIR / f"hybrid_retrieval_baseline_e2e_{args.limit}.json")
    try:
        run(args)
    except Exception as exc:
        print(f"Hybrid retrieval baseline E2E failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
