"""MemoryOS memory feature extraction.

The verifier is deterministic apart from the optional multilingual embedding
model. The module does not estimate reliability or filter candidates; it only
extracts features for a downstream supervised reliability predictor. The
embedding model is lazy-loaded once and reused for all subsequent calls.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional


INVALID = {"", "none", "null", "n/a", "- none", "- none.", "无", "无内容", "无有效知识"}
_MODEL_CACHE = {}


class MemoryVerifier:
    def __init__(self, min_text_length: int = 5,
                 embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.min_text_length = min_text_length
        self.embedding_model_name = embedding_model
        self._model = None

    @staticmethod
    def _text(value: Any) -> str:
        return "" if value is None else str(value).strip()

    @staticmethod
    def _tokens(text: str) -> set[str]:
        text = text.lower()
        words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text)
        chars = re.findall(r"[\u4e00-\u9fff]", text)
        # Character bigrams make Chinese matching robust without requiring jieba.
        return set(words) | {"".join(chars[i:i + 2]) for i in range(len(chars) - 1)}

    def _get_model(self):
        if self._model is not None:
            return self._model
        if self.embedding_model_name in _MODEL_CACHE:
            self._model = _MODEL_CACHE[self.embedding_model_name]
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.embedding_model_name)
        except Exception:
            self._model = False
        _MODEL_CACHE[self.embedding_model_name] = self._model
        return self._model

    def _embedding_similarity(self, left: str, right: str) -> float:
        model = self._get_model()
        if not model:
            return 0.0
        try:
            vectors = model.encode([left, right], normalize_embeddings=True)
            return max(0.0, min(1.0, sum(float(a) * float(b) for a, b in zip(vectors[0], vectors[1]))))
        except Exception:
            return 0.0

    def _lexical_similarity(self, left: str, right: str) -> float:
        a, b = self._tokens(left), self._tokens(right)
        return len(a & b) / max(1, len(a | b))

    def _similarity(self, left: str, right: str) -> float:
        # Embedding receives most weight; lexical/entity overlap fixes common
        # Chinese paraphrase and named-entity failures.
        emb = self._embedding_similarity(left, right)
        lex = self._lexical_similarity(left, right)
        return round(0.65 * emb + 0.35 * lex if emb else lex, 4)

    def clean_memory(self, memory: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(memory, dict):
            return None
        user = self._text(memory.get("user_input"))
        answer = self._text(memory.get("agent_response"))
        if user.lower() in INVALID: user = ""
        if answer.lower() in INVALID: answer = ""
        if len(user + answer) < self.min_text_length:
            return None
        result = dict(memory)
        result.update(user_input=user, agent_response=answer)
        return result

    def relevance_check(self, question: str, memory: Dict[str, Any]) -> float:
        """Hybrid semantic + entity/lexical relevance, including Chinese."""
        memory_text = memory["user_input"] + " " + memory["agent_response"]
        embedding = self._embedding_similarity(question, memory_text)
        q_tokens, m_tokens = self._tokens(question), self._tokens(memory_text)
        overlap = len(q_tokens & m_tokens) / max(1, len(q_tokens))
        # Recall-oriented entity overlap is useful for questions such as
        # “我的研究方向是什么？” where the answer contains the key entity.
        entity_tokens = {t for t in m_tokens if len(t) > 1}
        entity_bonus = min(1.0, len(q_tokens & entity_tokens) / max(1, len(q_tokens)))
        lexical = max(self._lexical_similarity(question, memory_text), entity_bonus)
        attribute = self._attribute_similarity(question, memory["user_input"])
        hybrid = 0.55 * embedding + 0.25 * lexical + 0.20 * attribute if embedding else max(lexical, attribute)
        return round(max(hybrid, 0.85 * attribute, 0.80 * lexical), 4)

    def _extract_proposition(self, text: str) -> Dict[str, str]:
        """Extract a generic subject and attribute without domain keywords."""
        clean = re.sub(r"[？?。！!]", "", self._text(text).lower())
        match = re.match(r"^(我|你|他|她|它|我们|你们|他们|她们|[a-z][a-z0-9_-]*|[\u4e00-\u9fff]{2,8})(?:的)?(.*)$", clean)
        subject, attribute = (match.group(1), match.group(2)) if match else ("", clean)
        attribute = re.sub(r"^(是|为|有|在|目前|现在)", "", attribute)
        attribute = re.sub(r"(是什么|是哪里|在哪里|有哪些|多少|何时|什么时候)$", "", attribute)
        return {"subject": subject, "attribute": attribute.strip()}

    def _attribute_similarity(self, left: str, right: str) -> float:
        a, b = self._extract_proposition(left), self._extract_proposition(right)
        subject_match = 1.0 if not a["subject"] or not b["subject"] or a["subject"] == b["subject"] else self._similarity(a["subject"], b["subject"])
        return round(subject_match * self._similarity(a["attribute"], b["attribute"]), 4)

    def faithfulness_check(self, memory: Dict[str, Any], evidence_memories: Optional[Iterable[Dict[str, Any]]] = None) -> float:
        """Whether the agent response is supported by the user input/evidence."""
        answer = memory["agent_response"]
        evidence = list(evidence_memories or [])
        direct_support = self._similarity(answer, memory["user_input"])
        evidence_support = max(
            (self._similarity(answer, self._text(e.get("agent_response")) + " " + self._text(e.get("user_input")))
             for e in evidence if isinstance(e, dict)),
            default=0.0,
        )
        return round(max(direct_support, evidence_support), 4)

    def _answer_contradiction(self, first: Dict[str, Any], second: Dict[str, Any]) -> float:
        answer_similarity = self._similarity(first["agent_response"], second["agent_response"])
        if answer_similarity >= 0.65:
            return 0.0
        # For the same attribute, disjoint dates/numbers are strong evidence of conflict.
        n1 = set(re.findall(r"\b\d{1,4}\b", first["agent_response"]))
        n2 = set(re.findall(r"\b\d{1,4}\b", second["agent_response"]))
        numeric_conflict = bool(n1 and n2 and n1.isdisjoint(n2))
        return round(min(1.0, 0.7 * (1.0 - answer_similarity) + 0.3 * numeric_conflict), 4)

    def conflict_check(self, memory: Dict[str, Any], memories: List[Dict[str, Any]]) -> float:
        """Continuous conflict feature: attribute similarity × contradiction."""
        maximum = 0.0
        for other in memories:
            if other is memory or other.get("page_id") == memory.get("page_id"):
                continue
            # User-input similarity identifies whether both memories answer the
            # same attribute; no fixed domain vocabulary is used.
            attribute_similarity = self._attribute_similarity(memory["user_input"], other["user_input"])
            if attribute_similarity < 0.65:
                continue
            contradiction = self._answer_contradiction(memory, other)
            maximum = max(maximum, attribute_similarity * contradiction)
        return round(maximum, 4)

    def verify(self, question: str, memories: Iterable[Dict[str, Any]], evidence_memories=None) -> Dict[str, Any]:
        raw = list(memories or [])
        cleaned = [item for value in raw if (item := self.clean_memory(value))]
        features = []
        for memory in cleaned:
            relevance = self.relevance_check(question, memory)
            faithfulness = self.faithfulness_check(memory, evidence_memories)
            conflict = self.conflict_check(memory, cleaned)
            features.append({
                **memory,
                "relevance_score": relevance,
                "faithfulness_score": faithfulness,
                "conflict_score": conflict,
            })

        # `features` is the canonical feature-extractor output. Compatibility
        # aliases keep the existing MemoryOS caller operational without making
        # any trust decision: every cleaned candidate is returned unchanged.
        return {
            "features": features,
            "all_analysis": features,
            "verified_memories": features,
            "input_count": len(raw),
            "valid_count": len(features),
            "verified_count": len(features),
            "rejected_count": len(raw) - len(cleaned),
            "has_conflict": any(m["conflict_score"] > 0 for m in features),
        }
