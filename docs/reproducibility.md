# Reproducibility guide

## 1. Prepare external inputs

Obtain `locomo10.json` from the LoCoMo release and the matching prebuilt MemoryOS snapshot from the project artifacts. Keep both outside this repository. The snapshot must be complete for every selected sample and must not be regenerated or edited during evaluation.

## 2. Environment

Create a clean virtual environment and install `requirements.txt`. Set `OPENAI_API_KEY` (and, if needed, `OPENAI_BASE_URL`) in the shell. Set `PYTHONPATH` to `evaluation`, `memoryos_core`, and `retrieval`. Before publishing, remove the hard-coded credential currently present in `memoryos_core/utils.py`, rotate it, and replace it with environment-variable access.

## 3. Recommended order

1. Run `evaluation/run_page_level_retrieval_e2e.py` for a page-level dense-only check.
2. Run `retrieval/run_page_level_hybrid_retrieval.py` with the external dataset and snapshot. This produces dense-only, BM25-only, and dense+BM25/RRF retrieval artifacts and audit fields; it does not call an LLM.
3. Inspect the retrieval comparison and snapshot-integrity outputs before proceeding.
4. Run `evaluation/run_hybrid_retrieval_baseline_e2e.py` with the hybrid candidate cache and an existing baseline artifact. It performs one `gpt-4o-mini` generation per query with the original prompt.
5. Use `evaluation/run_v2_relevance_only_e2e.py` only when the original MemoryOS helper modules and candidate-cache layout are available.

Use a new output directory for each run. Record the dataset version, snapshot hash, embedding model, BM25 parameters, RRF constant, candidate budget, model, temperature, and max-token setting with the resulting artifacts.

## 4. Integrity and exclusions

Do not place JSON datasets, snapshot directories, candidate caches, API keys, or generated experiment results in this source tree. The scripts are intended to read those assets from paths supplied at runtime.
