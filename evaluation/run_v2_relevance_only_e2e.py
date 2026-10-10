"""Paired E2E evaluation for relevance-only candidate reranking.

This script consumes the complete candidate-cache checkpoints produced by the
existing snapshot-only evaluator. It never calls retrieval and never mutates a
snapshot. The only experimental variable is the order of the cached page
candidates before the shared MemoryOS prompt builder is called.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPT_DIR / "locomo10.json"
DEFAULT_STORAGE_ROOT = SCRIPT_DIR / "e2e_storage"
DEFAULT_OUTPUT = SCRIPT_DIR / "v2_relevance_only_e2e_100.json"
DEFAULT_MODEL = "gpt-4o-mini"
TOP_K = 10


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def scalar_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def tokenize(text: Any) -> List[str]:
    return re.findall(r"\b\w+\b", str(text or "").lower())


def answer_f1(prediction: Any, reference: Any) -> float:
    predicted = set(tokenize(prediction))
    expected = set(tokenize(reference))
    if not predicted or not expected:
        return 0.0
    common = predicted & expected
    precision = len(common) / len(predicted)
    recall = len(common) / len(expected)
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def exact_match(prediction: Any, reference: Any) -> bool:
    normalize = lambda value: " ".join(tokenize(value))
    return normalize(prediction) == normalize(reference)


def prompt_tokens(messages: Sequence[Mapping[str, Any]], model: str) -> int:
    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model)
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
        return int(sum(len(encoding.encode(str(message.get("content", "")))) for message in messages))
    except Exception:
        return int(sum(len(re.findall(r"\S+", str(message.get("content", "")))) for message in messages))


def load_locomo(path: Path) -> List[Dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Dataset must be a non-empty JSON list: {path}")
    return rows


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
            })
    if limit is not None:
        queries = queries[:limit]
    if not queries:
        raise ValueError("No queries selected")
    return queries


def checkpoint_records(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load full candidate-cache records, rejecting summary-only cache files."""
    metadata: Dict[str, Any] = {}
    payloads: List[Mapping[str, Any]] = []
    if path.is_dir():
        files = sorted(path.glob("conv_*.json"))
        if not files:
            raise RuntimeError(f"No conv_*.json candidate checkpoints found in {path}")
        payloads = [json.loads(file.read_text(encoding="utf-8")) for file in files]
    elif path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            payloads = [{"results": payload}]
        elif isinstance(payload, Mapping):
            metadata = dict(payload.get("metadata", {}))
            if isinstance(payload.get("results"), list):
                payloads = [payload]
            elif isinstance(payload.get("records"), list):
                payloads = [{"results": payload["records"]}]
            else:
                raise RuntimeError(f"No results list found in candidate cache: {path}")
        else:
            raise RuntimeError(f"Unsupported candidate cache JSON: {path}")
    else:
        raise RuntimeError(f"Candidate cache path does not exist: {path}")

    records: List[Dict[str, Any]] = []
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        metadata.update(dict(payload.get("metadata", {})))
        for record in payload.get("results", []):
            if not isinstance(record, Mapping):
                continue
            row = dict(record)
            # Full checkpoints store the actual pages under bundle. A public
            # e2e_candidate_cache.json intentionally stores only a summary.
            bundle = row.get("bundle")
            if not isinstance(bundle, list) or any(
                not isinstance(candidate, Mapping) or not isinstance(candidate.get("page"), Mapping)
                for candidate in bundle
            ):
                raise RuntimeError(
                    "Candidate cache is summary-only and lacks complete candidate page text. "
                    "Pass the existing candidate_cache checkpoint directory (for example "
                    "e2e_results_full/candidate_cache), not e2e_candidate_cache.json."
                )
            records.append(row)
    if not records:
        raise RuntimeError(f"Candidate cache contains no complete records: {path}")
    return metadata, records


def candidate_id(candidate: Mapping[str, Any], index: int) -> str:
    value = candidate.get("page_id", candidate.get("candidate_id"))
    return str(value if value not in (None, "") else f"candidate_{index}")


def deduplicate_bundle(
    bundle: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Deduplicate a cached candidate bundle by page ID.

    Candidate checkpoints normally preserve retrieval order, but a page can
    occur more than once in a bundle.  Keep the occurrence with the smallest
    ``initial_rank`` (and the first occurrence when ranks tie), while keeping
    the position of the first page-ID occurrence in the returned bundle.  The
    latter makes the operation deterministic even when a malformed checkpoint
    contains out-of-order ranks.
    """
    first_order: List[str] = []
    best_by_id: Dict[str, Tuple[int, int, Dict[str, Any]]] = {}

    for index, raw_candidate in enumerate(bundle):
        candidate = dict(raw_candidate)
        page_id = candidate_id(candidate, index)
        rank = initial_rank(candidate, index)
        existing = best_by_id.get(page_id)
        if existing is None:
            first_order.append(page_id)
            best_by_id[page_id] = (rank, index, candidate)
            continue
        # Prefer the lowest initial rank; the original position is the stable
        # tie-breaker and therefore preserves the first occurrence on ties.
        if (rank, index) < (existing[0], existing[1]):
            best_by_id[page_id] = (rank, index, candidate)

    deduplicated = [best_by_id[page_id][2] for page_id in first_order]
    before = len(bundle)
    after = len(deduplicated)
    return deduplicated, {
        "candidate_count_before": before,
        "candidate_count_after": after,
        "duplicate_candidates_removed": before - after,
    }


def candidate_relevance(candidate: Mapping[str, Any]) -> Optional[float]:
    features = candidate.get("features")
    if isinstance(features, Mapping):
        for key in ("relevance_score", "relevance"):
            value = scalar_float(features.get(key))
            if value is not None:
                return value
    if isinstance(features, (list, tuple)) and features:
        return scalar_float(features[0])
    for key in ("relevance_score", "relevance"):
        value = scalar_float(candidate.get(key))
        if value is not None:
            return value
    return None


def initial_rank(candidate: Mapping[str, Any], index: int) -> int:
    value = candidate.get("initial_rank")
    try:
        return int(value)
    except (TypeError, ValueError):
        return index


def source_ids(candidate: Mapping[str, Any]) -> List[str]:
    value = candidate.get("source_dialogue_ids", [])
    if not isinstance(value, (list, tuple, set)):
        return []
    return [str(item) for item in value if item not in (None, "")]


def candidate_pool_hash(candidates: Sequence[Mapping[str, Any]]) -> str:
    return stable_hash([
        {
            "candidate_id": candidate_id(candidate, index),
            "initial_rank": initial_rank(candidate, index),
            "page": candidate.get("page", {}),
        }
        for index, candidate in enumerate(candidates)
    ])


def ensure_relevance_scores(records: Sequence[Mapping[str, Any]],
                            query_by_id: Optional[Mapping[str, Mapping[str, Any]]] = None) -> str:
    """Ensure scores exist using trace fields or verifier-only extraction."""
    all_present = all(
        candidate_relevance(candidate) is not None
        for record in records
        for candidate in record.get("bundle", [])
    )
    if all_present:
        return "candidate_cache.features.relevance_score"

    sys.path.insert(0, str(SCRIPT_DIR))
    from memory_verifier import MemoryVerifier

    verifier = MemoryVerifier()
    for record in records:
        candidates = record.get("bundle", [])
        pages = [dict(candidate["page"]) for candidate in candidates]
        query_id = str(record.get("query_id", ""))
        query_text = record.get("query")
        if query_text in (None, "") and query_by_id is not None:
            query_text = query_by_id.get(query_id, {}).get("query", "")
        result = verifier.verify(str(query_text or ""), pages)
        by_id = {
            str(row.get("page_id")): row
            for row in result.get("features", [])
            if isinstance(row, Mapping) and row.get("page_id") is not None
        }
        for index, candidate in enumerate(candidates):
            page = candidate.get("page", {})
            page_id = candidate_id(candidate, index)
            row = by_id.get(page_id)
            score = candidate_relevance(candidate)
            if score is None and row is not None:
                score = scalar_float(row.get("relevance_score"))
                if score is not None:
                    candidate["features"] = dict(candidate.get("features", {})) if isinstance(candidate.get("features"), Mapping) else {}
                    candidate["features"]["relevance_score"] = score
            if score is None:
                raise RuntimeError(f"Unable to obtain relevance_score for candidate {page_id!r}")
    return "MemoryVerifier.verify on cached candidate pages"


def evidence_recall(selected: Sequence[Mapping[str, Any]], gold: Sequence[str]) -> float:
    gold_set = {str(item) for item in gold}
    if not gold_set:
        return 0.0
    selected_sources = {source for candidate in selected for source in source_ids(candidate)}
    return len(gold_set & selected_sources) / len(gold_set)


def evidence_hit(selected: Sequence[Mapping[str, Any]], gold: Sequence[str]) -> int:
    gold_set = {str(item) for item in gold}
    return int(bool(gold_set & {source for candidate in selected for source in source_ids(candidate)}))


def best_evidence_rank(ranked: Sequence[Mapping[str, Any]], gold: Sequence[str]) -> Optional[int]:
    gold_set = {str(item) for item in gold}
    for position, candidate in enumerate(ranked, 1):
        if gold_set & set(source_ids(candidate)):
            return position
    return None


def select_budget(metadata: Mapping[str, Any], records: Sequence[Mapping[str, Any]], requested: Optional[int]) -> int:
    if requested is not None:
        if requested < 1:
            raise ValueError("--candidate-budget must be positive")
        return requested
    candidates = [
        metadata.get("candidate_budget"),
        metadata.get("retrieval_queue_capacity"),
        (metadata.get("retrieval") or {}).get("queue_capacity") if isinstance(metadata.get("retrieval"), Mapping) else None,
        (metadata.get("retrieval_parameters") or {}).get("queue_capacity") if isinstance(metadata.get("retrieval_parameters"), Mapping) else None,
    ]
    values = []
    for value in candidates:
        try:
            if value is not None:
                values.append(int(value))
        except (TypeError, ValueError):
            continue
    if not values:
        raise RuntimeError(
            "Cannot confirm the original candidate/prompt budget. "
            "Pass --candidate-budget explicitly or provide cache metadata containing "
            "candidate_budget/retrieval.queue_capacity."
        )
    budget = values[0]
    if any(value != budget for value in values):
        raise RuntimeError(f"Conflicting candidate budget metadata: {values}")
    if any(len(record.get("bundle", [])) < budget for record in records):
        # A queue may legitimately contain fewer candidates for one query; the
        # selected count is min(budget, pool size). This is not an error.
        pass
    return budget


def build_prompt(query: Mapping[str, Any], context: Any, selected: Sequence[Mapping[str, Any]],
                 long_knowledge: Sequence[Any], model: str) -> Tuple[List[Dict[str, str]], int]:
    """Reuse the existing MemoryOS prompt template without invoking retrieval."""
    history = context.short_memory.get_all()
    speaker_a, speaker_b = context.speaker_a, context.speaker_b
    history_text = "\n".join(
        f"{speaker_a}: {qa.get('user_input', '')}\n{speaker_b}: {qa.get('agent_response', '')}\n"
        f"Time: ({qa.get('timestamp', '')})"
        for qa in history
    )
    retrieval_text = "\n".join(
        f"【Historical Memory】 {speaker_a}: {candidate['page'].get('user_input', '')}\n"
        f"{speaker_b}: {candidate['page'].get('agent_response', '')}\n"
        f"Time:({candidate['page'].get('timestamp', '')})\n"
        f"Conversation chain overview:({candidate['page'].get('meta_info', '')})\n"
        for candidate in selected
    )
    profile_obj = context.long_memory.get_user_profile(context.sample_id)
    profile_text = str(profile_obj.get("data", "None")) if profile_obj else "None"
    background = f"【User Profile】\n{profile_text}\n\n"
    for knowledge in long_knowledge or []:
        if isinstance(knowledge, Mapping) and knowledge.get("knowledge") not in (None, "", "None"):
            background += f"{knowledge['knowledge']}\n"
    background = re.sub(r"(?i)\buser\b", speaker_a, background)
    background = re.sub(r"(?i)\bassistant\b", speaker_b, background)
    assistant_knowledge_text = "【Assistant Knowledge】\n"
    for knowledge in context.long_memory.get_assistant_knowledge():
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
        f"<CONTEXT>\nRecent conversation between {speaker_a} and {speaker_b}:\n{history_text}\n\n"
        f"<MEMORY>\nRelevant past conversations:\n{retrieval_text}\n\n"
        f"<CHARACTER TRAITS>\nCharacteristics of {speaker_a}:\n{background}\n\n"
        f"the question is: {query['query']}\n"
        f"Your task is to answer questions about {speaker_a} or {speaker_b} in an extremely concise manner.\n"
        f"Please only provide the content of the answer, without including 'answer:'\n"
        f"For questions that require answering a date or time, strictly follow the format \"15 July 2023\" and provide a specific date whenever possible. For example, if you need to answer \"last year,\" give the specific year of last year rather than just saying \"last year.\" Only provide one year, date, or time, without any extra responses.\n"
        f"If the question is about the duration, answer in the form of several years, months, or days.\n"
        f"Generate answers primarily composed of concrete entities, such as Mentoring program, school speech, etc"
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    return messages, prompt_tokens(messages, model)


def load_contexts(dataset: Sequence[Mapping[str, Any]], sample_ids: Iterable[str], storage_root: Path,
                  client: Any, args: argparse.Namespace) -> Dict[str, Any]:
    # Importing the existing loader keeps snapshot deserialization and prompt
    # state semantics aligned with the original evaluator. No retrieval call is
    # made by this script.
    sys.path.insert(0, str(SCRIPT_DIR))
    import run_reliability_memoryos_eval as existing

    contexts: Dict[str, Any] = {}
    by_id = {str(sample.get("sample_id", "")): sample for sample in dataset}
    for sample_id in dict.fromkeys(str(item) for item in sample_ids):
        if sample_id not in by_id:
            raise RuntimeError(f"Candidate cache references missing sample_id: {sample_id}")
        contexts[sample_id] = existing.prepare_sample(by_id[sample_id], storage_root, client, args)
    return contexts


def client_from_env() -> Any:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    if not base_url:
        raise RuntimeError("OPENAI_BASE_URL is required")
    return OpenAI(api_key=api_key, base_url=base_url)


class ExistingPromptLLMAdapter:
    """Adapter for the existing prompt builder's client interface."""

    def __init__(self, client: Any):
        self.client = client

    def chat_completion(self, model: str, messages: Sequence[Mapping[str, str]],
                        temperature: float = 0.0, max_tokens: int = 2000) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=list(messages),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        return "" if content is None else str(content).strip()


def query_result(query: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]],
                 baseline: Sequence[Mapping[str, Any]], v2: Sequence[Mapping[str, Any]],
                 context: Any, long_knowledge: Sequence[Any], client: Any,
                 max_tokens: int, existing: Any,
                 deduplication: Optional[Mapping[str, int]] = None) -> Dict[str, Any]:
    baseline_answer, baseline_tokens = existing.generate_system_response_with_meta(
        query["query"], context.short_memory, context.long_memory,
        [candidate["page"] for candidate in baseline], long_knowledge, client,
        context.sample_id, context.speaker_a, context.speaker_b,
        DEFAULT_MODEL, 0.0, max_tokens, return_prompt_tokens=True,
    )
    v2_answer, v2_tokens = existing.generate_system_response_with_meta(
        query["query"], context.short_memory, context.long_memory,
        [candidate["page"] for candidate in v2], long_knowledge, client,
        context.sample_id, context.speaker_a, context.speaker_b,
        DEFAULT_MODEL, 0.0, max_tokens, return_prompt_tokens=True,
    )
    gold = list(query.get("evidence", []))
    baseline_rank = best_evidence_rank(baseline, gold)
    v2_rank = best_evidence_rank(v2, gold)
    baseline_ids = [candidate_id(candidate, index) for index, candidate in enumerate(baseline[:TOP_K])]
    v2_ids = [candidate_id(candidate, index) for index, candidate in enumerate(v2[:TOP_K])]
    baseline_set, v2_set = set(baseline_ids), set(v2_ids)
    result = {
        "query_id": query["query_id"],
        "query": query["query"],
        "baseline_answer": baseline_answer,
        "v2_answer": v2_answer,
        "baseline_f1": answer_f1(baseline_answer, query["reference_answer"]),
        "v2_f1": answer_f1(v2_answer, query["reference_answer"]),
        "baseline_em": exact_match(baseline_answer, query["reference_answer"]),
        "v2_em": exact_match(v2_answer, query["reference_answer"]),
        "baseline_evidence_hit": evidence_hit(baseline, gold),
        "v2_evidence_hit": evidence_hit(v2, gold),
        "baseline_evidence_recall": evidence_recall(baseline, gold),
        "v2_evidence_recall": evidence_recall(v2, gold),
        "baseline_best_evidence_rank": baseline_rank,
        "v2_best_evidence_rank": v2_rank,
        "rank_improvement": (baseline_rank - v2_rank) if baseline_rank is not None and v2_rank is not None else None,
        "baseline_prompt_tokens": baseline_tokens,
        "v2_prompt_tokens": v2_tokens,
        "top_k_candidate_difference": {
            "k": TOP_K,
            "baseline_ids": baseline_ids,
            "v2_ids": v2_ids,
            "added": sorted(v2_set - baseline_set),
            "removed": sorted(baseline_set - v2_set),
            "difference_count": len(v2_set ^ baseline_set),
            "overlap_count": len(baseline_set & v2_set),
            "overlap_ratio": len(baseline_set & v2_set) / max(1, TOP_K),
        },
        "baseline_selected_candidate_ids": [candidate_id(candidate, index) for index, candidate in enumerate(baseline)],
        "v2_selected_candidate_ids": [candidate_id(candidate, index) for index, candidate in enumerate(v2)],
        "candidate_pool_hash": candidate_pool_hash(candidates),
        "baseline_candidate_pool_hash": candidate_pool_hash(candidates),
        "v2_candidate_pool_hash": candidate_pool_hash(candidates),
    }
    if deduplication is not None:
        result["deduplication_stats"] = dict(deduplication)
    return result


def aggregate(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def mean(key: str) -> float:
        return float(statistics.mean([float(item[key]) for item in results])) if results else 0.0

    def bool_mean(key: str) -> float:
        return float(statistics.mean([int(bool(item[key])) for item in results])) if results else 0.0

    return {
        "query_count": len(results),
        "baseline_f1": mean("baseline_f1"),
        "v2_f1": mean("v2_f1"),
        "delta_f1": mean("v2_f1") - mean("baseline_f1"),
        "baseline_em": bool_mean("baseline_em"),
        "v2_em": bool_mean("v2_em"),
        "delta_em": bool_mean("v2_em") - bool_mean("baseline_em"),
        "baseline_evidence_hit_rate": mean("baseline_evidence_hit"),
        "v2_evidence_hit_rate": mean("v2_evidence_hit"),
        "baseline_evidence_level_recall": mean("baseline_evidence_recall"),
        "v2_evidence_level_recall": mean("v2_evidence_recall"),
        "baseline_avg_best_evidence_rank": float(statistics.mean(
            [item["baseline_best_evidence_rank"] for item in results if item["baseline_best_evidence_rank"] is not None]
        )) if any(item["baseline_best_evidence_rank"] is not None for item in results) else None,
        "v2_avg_best_evidence_rank": float(statistics.mean(
            [item["v2_best_evidence_rank"] for item in results if item["v2_best_evidence_rank"] is not None]
        )) if any(item["v2_best_evidence_rank"] is not None for item in results) else None,
        "baseline_avg_prompt_tokens": mean("baseline_prompt_tokens"),
        "v2_avg_prompt_tokens": mean("v2_prompt_tokens"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paired relevance-only MemoryOS E2E evaluation")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--storage-root", default=str(DEFAULT_STORAGE_ROOT))
    parser.add_argument("--candidate-cache", required=True,
                        help="Complete candidate_cache checkpoint directory or full-cache JSON")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--candidate-budget", type=int, default=None,
                        help="Original prompt candidate budget; required if cache metadata omits it")
    parser.add_argument("--max-tokens", type=int, default=2000)
    return parser


def run(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_path = Path(args.dataset).resolve()
    storage_root = Path(args.storage_root).resolve()
    cache_path = Path(args.candidate_cache).resolve()
    output_path = Path(args.output).resolve()
    if not dataset_path.is_file():
        raise RuntimeError(f"Dataset does not exist: {dataset_path}")
    dataset = load_locomo(dataset_path)
    queries = flatten_queries(dataset, args.limit)
    query_by_id = {query["query_id"]: query for query in queries}
    metadata, cache_records = checkpoint_records(cache_path)

    # Normalize every loaded bundle before relevance extraction.  This avoids
    # asking the verifier to process the same page more than once and ensures
    # baseline/V2 start from exactly the same page-ID pool.  Keep the raw and
    # deduplicated counts on the in-memory record for the final audit report.
    for record in cache_records:
        deduplicated, stats = deduplicate_bundle(record.get("bundle", []))
        record["bundle"] = deduplicated
        record["_deduplication_stats"] = stats

    budget = select_budget(metadata, cache_records, args.candidate_budget)
    relevance_source = ensure_relevance_scores(cache_records, query_by_id)
    by_query = {str(record.get("query_id")): record for record in cache_records}
    missing = [query["query_id"] for query in queries if query["query_id"] not in by_query]
    if missing:
        raise RuntimeError(f"Candidate cache is missing selected queries, e.g. {missing[:5]}")
    if len(by_query) < len(queries):
        # Extra cached records are harmless, but query order is always driven by
        # the dataset so paired comparison remains auditable.
        pass

    deduplication_by_query: Dict[str, Dict[str, int]] = {}
    for query in queries:
        record = by_query[query["query_id"]]
        stats = record.get("_deduplication_stats")
        if not isinstance(stats, Mapping):
            # This fallback is only for compatibility with records assembled
            # by callers rather than checkpoint_records().
            _, stats = deduplicate_bundle(record.get("bundle", []))
        deduplication_by_query[query["query_id"]] = {
            "candidate_count_before": int(stats.get("candidate_count_before", 0)),
            "candidate_count_after": int(stats.get("candidate_count_after", 0)),
            "duplicate_candidates_removed": int(stats.get("duplicate_candidates_removed", 0)),
        }
    deduplication_totals = {
        key: sum(stats[key] for stats in deduplication_by_query.values())
        for key in (
            "candidate_count_before",
            "candidate_count_after",
            "duplicate_candidates_removed",
        )
    }

    sample_ids = [query["sample_id"] for query in queries]
    # The existing loader needs a client only to construct the in-memory
    # adapters; it does not call chat completion during snapshot loading.
    api_client = client_from_env()
    loader_args = argparse.Namespace(
        short_term_capacity=1,
        mid_term_capacity=2000,
        topic_similarity_threshold=0.6,
        retrieval_queue_capacity=budget,
    )
    sys.path.insert(0, str(SCRIPT_DIR))
    import run_reliability_memoryos_eval as existing

    contexts = load_contexts(dataset, sample_ids, storage_root, api_client, loader_args)
    generation_client = ExistingPromptLLMAdapter(api_client)
    checkpoint_path = output_path.with_suffix(".checkpoint.json")
    results_by_id: Dict[str, Dict[str, Any]] = {}
    if checkpoint_path.is_file():
        checkpoint_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        prior_ids = [str(item.get("query_id")) for item in checkpoint_payload.get("query_results", [])]
        expected_ids = [query["query_id"] for query in queries]
        if prior_ids and prior_ids != expected_ids[:len(prior_ids)]:
            raise RuntimeError("Existing checkpoint query order does not match the requested query set")
        results_by_id = {
            str(item["query_id"]): dict(item)
            for item in checkpoint_payload.get("query_results", [])
            if isinstance(item, Mapping) and item.get("query_id")
        }
    # A checkpoint created by an earlier run may not contain the new
    # per-query audit block.  Add it while preserving all previously computed
    # answer metrics.
    for query_id, result in results_by_id.items():
        stats = deduplication_by_query.get(query_id)
        if stats is not None:
            result["deduplication_stats"] = dict(stats)
        baseline_ids = set(result.get("baseline_selected_candidate_ids", []))
        v2_ids = set(result.get("v2_selected_candidate_ids", []))
        if baseline_ids != v2_ids:
            raise RuntimeError(
                f"Checkpoint has different baseline/V2 candidate ID sets after "
                f"deduplication for query {query_id}: "
                f"baseline_only={sorted(baseline_ids - v2_ids)}, "
                f"v2_only={sorted(v2_ids - baseline_ids)}"
            )
    results: List[Dict[str, Any]] = [results_by_id[query["query_id"]] for query in queries if query["query_id"] in results_by_id]
    started = time.monotonic()
    for index, query in enumerate(queries, 1):
        record = by_query[query["query_id"]]
        if query["query_id"] in results_by_id:
            print(f"Query {index}/{len(queries)} resumed", flush=True)
            continue
        candidates = [dict(candidate) for candidate in record["bundle"]]
        if not candidates:
            raise RuntimeError(f"Empty candidate pool for {query['query_id']}")
        scores = [candidate_relevance(candidate) for candidate in candidates]
        if any(score is None for score in scores):
            raise RuntimeError(f"Missing relevance_score for query {query['query_id']}")
        original_ids = [candidate_id(candidate, position) for position, candidate in enumerate(candidates)]
        if len(set(original_ids)) != len(original_ids):
            raise RuntimeError(f"Duplicate candidate IDs for query {query['query_id']}")
        indexed = list(enumerate(candidates))
        baseline = [candidate for _, candidate in sorted(
            indexed, key=lambda item: (initial_rank(item[1], item[0]), item[0])
        )[:budget]]
        v2_ranked = [candidate for _, candidate in sorted(
            indexed, key=lambda item: (-float(candidate_relevance(item[1]) or 0.0), initial_rank(item[1], item[0]), item[0])
        )]
        v2 = v2_ranked[:budget]
        baseline_id_set = {candidate_id(item, i) for i, item in enumerate(baseline)}
        v2_id_set = {candidate_id(item, i) for i, item in enumerate(v2)}
        if baseline_id_set != v2_id_set:
            raise RuntimeError(
                f"Baseline/V2 candidate ID sets differ after deduplication for "
                f"query {query['query_id']}: "
                f"baseline_only={sorted(baseline_id_set - v2_id_set)}, "
                f"v2_only={sorted(v2_id_set - baseline_id_set)}"
            )
        context = contexts[query["sample_id"]]
        long_knowledge = list(record.get("long_term_knowledge", []))
        result = query_result(
            query, candidates, baseline, v2, context, long_knowledge,
            generation_client, args.max_tokens, existing,
            deduplication=deduplication_by_query[query["query_id"]],
        )
        results.append(result)
        results_by_id[query["query_id"]] = result
        elapsed = time.monotonic() - started
        rate = index / elapsed if elapsed else 0.0
        eta = (len(queries) - index) / rate if rate else 0.0
        print(f"Query {index}/{len(queries)} | elapsed {elapsed:.0f}s | ETA {eta:.0f}s", flush=True)
        write_json(checkpoint_path, {
            "query_order": [item["query_id"] for item in queries],
            "query_results": results,
        })
    report = {
        "protocol": {
            "read_only": True,
            "retrieval_executed": False,
            "memory_rebuilt": False,
            "snapshot_modified": False,
            "model_trained": False,
            "llm_called_for_reranking": False,
            "answer_generation_enabled": True,
        },
        "config": {
            "dataset": str(dataset_path),
            "candidate_cache": str(cache_path),
            "storage_root": str(storage_root),
            "limit": args.limit,
            "temperature": 0.0,
            "model": DEFAULT_MODEL,
            "base_url_from_env": "OPENAI_BASE_URL",
            "api_key_from_env": "OPENAI_API_KEY",
            "candidate_budget": budget,
            "top_k_difference": TOP_K,
            "query_order_hash": stable_hash([query["query_id"] for query in queries]),
            "candidate_cache_sha256": sha256_file(cache_path) if cache_path.is_file() else None,
            "relevance_source": relevance_source,
        },
        "metrics": aggregate(results),
        "deduplication_stats": {
            **deduplication_totals,
            "per_query": deduplication_by_query,
        },
        "additional_analysis": {
            "evidence_rank_improved_f1_unchanged": sum(
                item["rank_improvement"] is not None and item["rank_improvement"] > 0 and item["v2_f1"] == item["baseline_f1"] for item in results
            ),
            "evidence_rank_improved_f1_improved": sum(
                item["rank_improvement"] is not None and item["rank_improvement"] > 0 and item["v2_f1"] > item["baseline_f1"] for item in results
            ),
            "evidence_rank_unchanged_f1_changed": sum(
                item["rank_improvement"] == 0 and item["v2_f1"] != item["baseline_f1"] for item in results
            ),
        },
        "query_results": results,
    }
    write_json(output_path, report)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(f"Wrote {output_path}")
    return report


def main() -> None:
    args = build_parser().parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    try:
        run(args)
    except Exception as exc:
        print(f"V2 relevance-only E2E evaluation failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
