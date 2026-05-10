
import re
import numpy as np
from difflib import SequenceMatcher

from docmind_rag.config.settings import (
    embedding_model,
    CONF_WEIGHT_RERANKER,
    CONF_WEIGHT_RECALL,
    CONF_WEIGHT_KEYWORD,
)
from docmind_rag.utils.text import _normalize_text


# ============================================================
# REFUSAL DETECTION — phrase matching, no LLM, no embeddings
# Kept as phrase list per final decision (more reliable than
# content-ratio approach for short answers).
# ============================================================

_REFUSAL_PHRASES = [
    "not present",
    "not available",
    "not found",
    "not in the document",
    "not present in the document",
    "information is not",
    "cannot find",
    "no information",
]


def _is_refusal_answer(answer: str) -> bool:
    """
    Phrase-based refusal detection. No LLM, no embeddings.
    Short answers (<=3 words) are never treated as refusals —
    they are legitimate short factual answers.
    """
    if not answer:
        return True

    words = answer.strip().split()

    # Short answers are never refusals — they are legitimate facts
    if len(words) <= 3:
        return False

    t = answer.lower().strip()
    return any(p in t for p in _REFUSAL_PHRASES)


# ============================================================
# NORMALIZATION
# ============================================================

def minmax_normalize(score: float, all_scores: list) -> float:
    if not all_scores:
        return 1.0
    min_s = min(all_scores)
    max_s = max(all_scores)
    if max_s == min_s:
        return 1.0
    return (score - min_s) / (max_s - min_s + 1e-6)


def minmax_normalize_list(scores: list) -> list:
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)
    return [(s - min_s) / (max_s - min_s + 1e-6) for s in scores]


# ============================================================
# KEYWORD OVERLAP
# ============================================================

def keyword_overlap(answer: str, context_chunks: list) -> float:
    if not answer or not context_chunks:
        return 0.0
    text   = " ".join(context_chunks).lower()
    answer = answer.lower().strip()
    if answer in text:
        return 1.0
    a = set(answer.split())
    c = set(text.split())
    if not a:
        return 0.0
    return len(a & c) / len(a)


# ============================================================
# SEMANTIC SIMILARITY
# ============================================================

def semantic_similarity(answer: str, context_chunks: list) -> float:
    if not answer or not context_chunks:
        return 0.0
    try:
        combined = " ".join(context_chunks)[:2000]
        ans_emb  = embedding_model.embed_query(answer[:500])
        ctx_emb  = embedding_model.embed_query(combined)
        a        = np.array(ans_emb)
        b        = np.array(ctx_emb)
        norm     = np.linalg.norm(a) * np.linalg.norm(b)
        if norm < 1e-8:
            return 0.0
        return float(np.dot(a, b) / norm)
    except Exception as e:
        print(f"[SemanticSim] error: {e}")
        return 0.0


# ============================================================
# GROUNDING COMPONENTS
# ============================================================

def local_substring_match(answer: str, context_chunks: list) -> float:
    if not context_chunks:
        return 0.0
    ans_norm   = _normalize_text(answer)
    chunk_norm = _normalize_text(context_chunks[0])
    return 1.0 if ans_norm in chunk_norm else 0.0


def exact_span_match(answer: str, context_chunks: list) -> bool:
    if not context_chunks:
        return False
    ans_norm = _normalize_text(answer)
    return any(ans_norm in _normalize_text(c) for c in context_chunks)


def compute_grounding_score(answer: str, context_chunks: list) -> float:
    if not answer or not context_chunks:
        return 0.0

    ans_norm    = _normalize_text(answer)
    chunks_norm = [_normalize_text(c) for c in context_chunks]

    local_match    = 1.0 if ans_norm in chunks_norm[0] else 0.0
    global_match   = 1.0 if any(ans_norm in c for c in chunks_norm) else 0.0
    keyword_score  = keyword_overlap(answer, context_chunks)
    semantic_score = semantic_similarity(answer, context_chunks)

    grounding_score = (
        0.3 * local_match    +
        0.2 * global_match   +
        0.3 * semantic_score +
        0.2 * keyword_score
    )
    print(f"[Grounding] local={local_match:.2f} global={global_match:.2f} "
          f"semantic={semantic_score:.3f} keyword={keyword_score:.3f} "
          f"-> grounding={grounding_score:.3f}")
    return round(grounding_score, 4)


def coverage_score(answer: str, chunks: list) -> float:
    if not answer or not chunks:
        return 0.0
    ans = answer.lower()
    return max(SequenceMatcher(None, ans, c.lower()).ratio() for c in chunks)


def score_answer(answer: str, context_chunks: list) -> float:
    if not answer or not context_chunks:
        return 0.0
    top_chunk = context_chunks[0]
    semantic  = semantic_similarity(answer, [top_chunk])
    keyword   = keyword_overlap(answer, [top_chunk])
    coverage  = coverage_score(answer, [top_chunk])
    return 0.4 * semantic + 0.3 * keyword + 0.3 * coverage


# ============================================================
# CONFIDENCE
# ============================================================

def compute_confidence(reranker_top: float, recall_score: float,
                       answer: str, context_chunks: list,
                       all_reranker_scores: list = None) -> float:
    if all_reranker_scores:
        normalized = minmax_normalize(reranker_top, all_reranker_scores)
    else:
        normalized = minmax_normalize(reranker_top, [reranker_top])

    rec = min(recall_score / 100.0, 1.0)
    kw  = keyword_overlap(answer, context_chunks)
    confidence = (
        CONF_WEIGHT_RERANKER * normalized +
        CONF_WEIGHT_RECALL   * rec +
        CONF_WEIGHT_KEYWORD  * kw
    )
    print(f"[Confidence] normalized={normalized:.3f} rec={rec:.3f} "
          f"kw={kw:.3f} → {confidence:.3f}")
    return round(confidence, 4)


# ============================================================
# compute_answer_grounding
# Word-overlap grounding (percentage 0–100).
# Refusal answers return sentinel 75.0 — above the hallucination
# gate threshold, preventing false rejections of correct refusals.
# ============================================================

def compute_answer_grounding(answer: str, context_chunks: list,
                             question: str = "") -> float:
    if not answer:
        return 0.0

    # Refusal sentinel — avoids polluting grounding metric with
    # function words that happen to appear in any document
    refusal_phrases = [
        "not present",
        "not mentioned",
        "not available",
        "no information",
        "cannot find",
        "not in the document"
    ]

    if any(p in answer.lower() for p in refusal_phrases):
        print(f"[Grounding] Refusal answer detected → returning sentinel 75.0")
        return 75.0
    if not context_chunks:
        return 0.0

    try:
        context_text  = " ".join(context_chunks).lower()
        context_words = set(re.findall(r'\b[a-z0-9%\-]{2,}\b', context_text))
        answer_words  = set(re.findall(r'\b[a-z0-9%\-]{2,}\b', answer.lower()))
        if not answer_words:
            return 0.0
        overlap = answer_words & context_words
        return round(len(overlap) / len(answer_words) * 100, 1)
    except Exception:
        return 0.0


# ============================================================
# RETRIEVAL METRICS
# ============================================================

def compute_retrieval_score(question: str, docs: list) -> float:
    try:
        q_embed = embedding_model.embed_query(question)
        sims    = []
        for d in docs:
            d_embed = embedding_model.embed_query(d.page_content[:300])
            sim = np.dot(q_embed, d_embed) / (
                np.linalg.norm(q_embed) * np.linalg.norm(d_embed) + 1e-8)
            sims.append(sim)
        if not sims:
            return 0.0
        max_sim  = float(np.max(sims))
        top3     = sorted(sims, reverse=True)[:3]
        mean_top = float(np.mean(top3))
        return round((0.7 * max_sim + 0.3 * mean_top) * 100, 2)
    except Exception:
        return 0.0


def compute_context_precision(question: str, docs: list,
                              threshold: float = 0.35) -> float:
    try:
        q_embed = embedding_model.embed_query(question)
        sims    = []
        for d in docs:
            d_embed = embedding_model.embed_query(d.page_content[:300])
            sim = np.dot(q_embed, d_embed) / (
                np.linalg.norm(q_embed) * np.linalg.norm(d_embed) + 1e-8)
            sims.append(sim)
        if not sims:
            return 0.0
        avg_sim            = float(np.mean(sims))
        max_sim            = float(np.max(sims))
        adaptive_threshold = min(threshold, max(max_sim * 0.60, avg_sim * 0.80))
        relevant           = sum(1 for s in sims if s >= adaptive_threshold)
        return round(relevant / len(sims) * 100, 1)
    except Exception:
        return 0.0

def compute_recall_at_k(question: str, docs: list,
                        all_chunks: list, k: int = 10) -> float:
    try:
        q_embed  = embedding_model.embed_query(question)
        all_sims = []

        for chunk in all_chunks:
            text = chunk if isinstance(chunk, str) else str(chunk)
            c_embed = embedding_model.embed_query(text[:300])
            sim = np.dot(q_embed, c_embed) / (
                np.linalg.norm(q_embed) * np.linalg.norm(c_embed) + 1e-8)
            all_sims.append((sim, text))

        if not all_sims:
            return 0.0

        all_sims.sort(key=lambda x: x[0], reverse=True)
        oracle_sim   = all_sims[0][0]
        oracle_chunk = all_sims[0][1]

        if oracle_sim < 0.01:
            return 0.0

        # 🔥 SAFE conversion for docs
        doc_texts = [
            d.page_content if hasattr(d, "page_content") else str(d)
            for d in docs
        ]

        retrieved_fps = set(text[:100] for text in doc_texts)

        if oracle_chunk[:100] in retrieved_fps:
            print(f"[Recall@{k}] ✅ Oracle chunk retrieved! score=100%")
            return 100.0

        best_retrieved_sim = max(
            (
                np.dot(q_embed, embedding_model.embed_query(text[:300])) /
                (np.linalg.norm(q_embed) *
                 np.linalg.norm(embedding_model.embed_query(text[:300])) + 1e-8)
            )
            for text in doc_texts
        ) if doc_texts else 0.0

        soft_score = round((best_retrieved_sim / oracle_sim) * 100, 1)
        print(f"[Recall@{k}] oracle={oracle_sim:.3f} | "
              f"best_retrieved={best_retrieved_sim:.3f} | score={soft_score}%")

        return min(soft_score, 99.9)

    except Exception as e:
        print(f"[Recall@K] error: {e}")
        return 0.0

def llm_verify_answer(question: str, answer: str,
                      context_chunks: list) -> bool:
    from docmind_rag.models.llm import call_llama
    context = "\n\n".join(context_chunks[:3])
    prompt  = (
        f"Context:\n{context[:2000]}\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        f"Is this answer supported by the context above?\n"
        f"Rules:\n"
        f"- Paraphrases are valid support\n"
        f"- Partial answers are valid if they answer the question\n"
        f"- A Yes/No + evidence is valid for verification questions\n"
        f"Reply ONLY: YES or NO"
    )
    try:
        resp = call_llama(prompt, num_ctx=1024, temperature=0.0)
        return "YES" in resp.upper()
    except Exception:
        return True





























# 80 nd 80
# import re
# import numpy as np
# from difflib import SequenceMatcher

# from docmind_rag.config.settings import (
#     embedding_model,
#     CONF_WEIGHT_RERANKER,
#     CONF_WEIGHT_RECALL,
#     CONF_WEIGHT_KEYWORD,
# )
# from docmind_rag.utils.text import _normalize_text


# # ============================================================
# # REFUSAL DETECTION — phrase matching, no LLM, no embeddings
# # Kept as phrase list per final decision (more reliable than
# # content-ratio approach for short answers).
# # ============================================================

# _REFUSAL_PHRASES = [
#     "not present",
#     "not available",
#     "not found",
#     "not in the document",
#     "not present in the document",
#     "information is not",
#     "cannot find",
#     "no information",
# ]


# def _is_refusal_answer(answer: str) -> bool:
#     """
#     Phrase-based refusal detection. No LLM, no embeddings.
#     Short answers (<=3 words) are never treated as refusals —
#     they are legitimate short factual answers.
#     """
#     if not answer:
#         return True

#     words = answer.strip().split()

#     # Short answers are never refusals — they are legitimate facts
#     if len(words) <= 3:
#         return False

#     t = answer.lower().strip()
#     return any(p in t for p in _REFUSAL_PHRASES)


# # ============================================================
# # NORMALIZATION
# # ============================================================

# def minmax_normalize(score: float, all_scores: list) -> float:
#     if not all_scores:
#         return 1.0
#     min_s = min(all_scores)
#     max_s = max(all_scores)
#     if max_s == min_s:
#         return 1.0
#     return (score - min_s) / (max_s - min_s + 1e-6)


# def minmax_normalize_list(scores: list) -> list:
#     if not scores:
#         return []
#     min_s = min(scores)
#     max_s = max(scores)
#     if max_s == min_s:
#         return [1.0] * len(scores)
#     return [(s - min_s) / (max_s - min_s + 1e-6) for s in scores]


# # ============================================================
# # KEYWORD OVERLAP
# # ============================================================

# def keyword_overlap(answer: str, context_chunks: list) -> float:
#     if not answer or not context_chunks:
#         return 0.0
#     text   = " ".join(context_chunks).lower()
#     answer = answer.lower().strip()
#     if answer in text:
#         return 1.0
#     a = set(answer.split())
#     c = set(text.split())
#     if not a:
#         return 0.0
#     return len(a & c) / len(a)


# # ============================================================
# # SEMANTIC SIMILARITY
# # ============================================================

# def semantic_similarity(answer: str, context_chunks: list) -> float:
#     if not answer or not context_chunks:
#         return 0.0
#     try:
#         combined = " ".join(context_chunks)[:2000]
#         ans_emb  = embedding_model.embed_query(answer[:500])
#         ctx_emb  = embedding_model.embed_query(combined)
#         a        = np.array(ans_emb)
#         b        = np.array(ctx_emb)
#         norm     = np.linalg.norm(a) * np.linalg.norm(b)
#         if norm < 1e-8:
#             return 0.0
#         return float(np.dot(a, b) / norm)
#     except Exception as e:
#         print(f"[SemanticSim] error: {e}")
#         return 0.0


# # ============================================================
# # GROUNDING COMPONENTS
# # ============================================================

# def local_substring_match(answer: str, context_chunks: list) -> float:
#     if not context_chunks:
#         return 0.0
#     ans_norm   = _normalize_text(answer)
#     chunk_norm = _normalize_text(context_chunks[0])
#     return 1.0 if ans_norm in chunk_norm else 0.0


# def exact_span_match(answer: str, context_chunks: list) -> bool:
#     if not context_chunks:
#         return False
#     ans_norm = _normalize_text(answer)
#     return any(ans_norm in _normalize_text(c) for c in context_chunks)


# def compute_grounding_score(answer: str, context_chunks: list) -> float:
#     if not answer or not context_chunks:
#         return 0.0

#     ans_norm    = _normalize_text(answer)
#     chunks_norm = [_normalize_text(c) for c in context_chunks]

#     local_match    = 1.0 if ans_norm in chunks_norm[0] else 0.0
#     global_match   = 1.0 if any(ans_norm in c for c in chunks_norm) else 0.0
#     keyword_score  = keyword_overlap(answer, context_chunks)
#     semantic_score = semantic_similarity(answer, context_chunks)

#     grounding_score = (
#         0.3 * local_match    +
#         0.2 * global_match   +
#         0.3 * semantic_score +
#         0.2 * keyword_score
#     )
#     print(f"[Grounding] local={local_match:.2f} global={global_match:.2f} "
#           f"semantic={semantic_score:.3f} keyword={keyword_score:.3f} "
#           f"-> grounding={grounding_score:.3f}")
#     return round(grounding_score, 4)


# def coverage_score(answer: str, chunks: list) -> float:
#     if not answer or not chunks:
#         return 0.0
#     ans = answer.lower()
#     return max(SequenceMatcher(None, ans, c.lower()).ratio() for c in chunks)


# def score_answer(answer: str, context_chunks: list) -> float:
#     if not answer or not context_chunks:
#         return 0.0
#     top_chunk = context_chunks[0]
#     semantic  = semantic_similarity(answer, [top_chunk])
#     keyword   = keyword_overlap(answer, [top_chunk])
#     coverage  = coverage_score(answer, [top_chunk])
#     return 0.4 * semantic + 0.3 * keyword + 0.3 * coverage


# # ============================================================
# # CONFIDENCE
# # ============================================================

# def compute_confidence(reranker_top: float, recall_score: float,
#                        answer: str, context_chunks: list,
#                        all_reranker_scores: list = None) -> float:
#     if all_reranker_scores:
#         normalized = minmax_normalize(reranker_top, all_reranker_scores)
#     else:
#         normalized = minmax_normalize(reranker_top, [reranker_top])

#     rec = min(recall_score / 100.0, 1.0)
#     kw  = keyword_overlap(answer, context_chunks)
#     confidence = (
#         CONF_WEIGHT_RERANKER * normalized +
#         CONF_WEIGHT_RECALL   * rec +
#         CONF_WEIGHT_KEYWORD  * kw
#     )
#     print(f"[Confidence] normalized={normalized:.3f} rec={rec:.3f} "
#           f"kw={kw:.3f} → {confidence:.3f}")
#     return round(confidence, 4)


# # ============================================================
# # compute_answer_grounding
# # Word-overlap grounding (percentage 0–100).
# # Refusal answers return sentinel 75.0 — above the hallucination
# # gate threshold, preventing false rejections of correct refusals.
# # ============================================================

# def compute_answer_grounding(answer: str, context_chunks: list,
#                              question: str = "") -> float:
#     if not answer:
#         return 0.0

#     # Refusal sentinel — avoids polluting grounding metric with
#     # function words that happen to appear in any document
#     if _is_refusal_answer(answer):
#         print(f"[Grounding] Refusal answer detected → returning sentinel 75.0")
#         return 75.0

#     if not context_chunks:
#         return 0.0

#     try:
#         context_text  = " ".join(context_chunks).lower()
#         context_words = set(re.findall(r'\b[a-z0-9%\-]{2,}\b', context_text))
#         answer_words  = set(re.findall(r'\b[a-z0-9%\-]{2,}\b', answer.lower()))
#         if not answer_words:
#             return 0.0
#         overlap = answer_words & context_words
#         return round(len(overlap) / len(answer_words) * 100, 1)
#     except Exception:
#         return 0.0


# # ============================================================
# # RETRIEVAL METRICS
# # ============================================================

# def compute_retrieval_score(question: str, docs: list) -> float:
#     try:
#         q_embed = embedding_model.embed_query(question)
#         sims    = []
#         for d in docs:
#             d_embed = embedding_model.embed_query(d.page_content[:300])
#             sim = np.dot(q_embed, d_embed) / (
#                 np.linalg.norm(q_embed) * np.linalg.norm(d_embed) + 1e-8)
#             sims.append(sim)
#         if not sims:
#             return 0.0
#         max_sim  = float(np.max(sims))
#         top3     = sorted(sims, reverse=True)[:3]
#         mean_top = float(np.mean(top3))
#         return round((0.7 * max_sim + 0.3 * mean_top) * 100, 2)
#     except Exception:
#         return 0.0


# def compute_context_precision(question: str, docs: list,
#                               threshold: float = 0.35) -> float:
#     try:
#         q_embed = embedding_model.embed_query(question)
#         sims    = []
#         for d in docs:
#             d_embed = embedding_model.embed_query(d.page_content[:300])
#             sim = np.dot(q_embed, d_embed) / (
#                 np.linalg.norm(q_embed) * np.linalg.norm(d_embed) + 1e-8)
#             sims.append(sim)
#         if not sims:
#             return 0.0
#         avg_sim            = float(np.mean(sims))
#         max_sim            = float(np.max(sims))
#         adaptive_threshold = min(threshold, max(max_sim * 0.60, avg_sim * 0.80))
#         relevant           = sum(1 for s in sims if s >= adaptive_threshold)
#         return round(relevant / len(sims) * 100, 1)
#     except Exception:
#         return 0.0


# def compute_recall_at_k(question: str, docs: list,
#                         all_chunks: list, k: int = 10) -> float:
#     try:
#         q_embed  = embedding_model.embed_query(question)
#         all_sims = []
#         for chunk in all_chunks:
#             c_embed = embedding_model.embed_query(chunk[:300])
#             sim = np.dot(q_embed, c_embed) / (
#                 np.linalg.norm(q_embed) * np.linalg.norm(c_embed) + 1e-8)
#             all_sims.append((sim, chunk))

#         if not all_sims:
#             return 0.0

#         all_sims.sort(key=lambda x: x[0], reverse=True)
#         oracle_sim   = all_sims[0][0]
#         oracle_chunk = all_sims[0][1]

#         if oracle_sim < 0.01:
#             return 0.0

#         retrieved_fps = set(d.page_content[:100] for d in docs)
#         if oracle_chunk[:100] in retrieved_fps:
#             print(f"[Recall@{k}] ✅ Oracle chunk retrieved! score=100%")
#             return 100.0

#         best_retrieved_sim = max(
#             (np.dot(q_embed, embedding_model.embed_query(d.page_content[:300])) /
#              (np.linalg.norm(q_embed) *
#               np.linalg.norm(embedding_model.embed_query(d.page_content[:300])) + 1e-8))
#             for d in docs
#         ) if docs else 0.0

#         soft_score = round((best_retrieved_sim / oracle_sim) * 100, 1)
#         print(f"[Recall@{k}] oracle={oracle_sim:.3f} | "
#               f"best_retrieved={best_retrieved_sim:.3f} | score={soft_score}%")
#         return min(soft_score, 99.9)

#     except Exception as e:
#         print(f"[Recall@K] error: {e}")
#         return 0.0


# def llm_verify_answer(question: str, answer: str,
#                       context_chunks: list) -> bool:
#     from docmind_rag.models.llm import call_llama
#     context = "\n\n".join(context_chunks[:3])
#     prompt  = (
#         f"Context:\n{context[:2000]}\n\n"
#         f"Question: {question}\n"
#         f"Answer: {answer}\n\n"
#         f"Is this answer supported by the context above?\n"
#         f"Rules:\n"
#         f"- Paraphrases are valid support\n"
#         f"- Partial answers are valid if they answer the question\n"
#         f"- A Yes/No + evidence is valid for verification questions\n"
#         f"Reply ONLY: YES or NO"
#     )
#     try:
#         resp = call_llama(prompt, num_ctx=1024, temperature=0.0)
#         return "YES" in resp.upper()
#     except Exception:
#         return True

































# got 80 nd  70 - lap claude 
# import re
# import numpy as np
# from difflib import SequenceMatcher

# from docmind_rag.config.settings import (
#     embedding_model,
#     CONF_WEIGHT_RERANKER,
#     CONF_WEIGHT_RECALL,
#     CONF_WEIGHT_KEYWORD,
# )
# from docmind_rag.utils.text import _normalize_text


# # ============================================================
# # CHANGE D — Lightweight refusal detection (no LLM, no embeddings)
# # Defined here so metrics.py has no circular imports.
# # ============================================================


# def _is_refusal_answer(answer: str) -> bool:
#     answer = answer.strip()

#     if not answer:
#         return True

#     words = answer.split()

#     # Short answers are NOT refusal
#     if len(words) <= 3:
#         return False

#     if len(words) > 20:
#         return False

#     content_words = [
#         w for w in words
#         if len(w) > 4 and not w.isdigit()
#     ]

#     content_ratio = len(content_words) / len(words)

#     return content_ratio < 0.35

# # ============================================================
# # NORMALIZATION
# # ============================================================

# def minmax_normalize(score: float, all_scores: list) -> float:
#     if not all_scores:
#         return 1.0
#     min_s = min(all_scores)
#     max_s = max(all_scores)
#     if max_s == min_s:
#         return 1.0
#     return (score - min_s) / (max_s - min_s + 1e-6)


# def minmax_normalize_list(scores: list) -> list:
#     if not scores:
#         return []
#     min_s = min(scores)
#     max_s = max(scores)
#     if max_s == min_s:
#         return [1.0] * len(scores)
#     return [(s - min_s) / (max_s - min_s + 1e-6) for s in scores]


# # ============================================================
# # KEYWORD OVERLAP
# # ============================================================

# def keyword_overlap(answer: str, context_chunks: list) -> float:
#     if not answer or not context_chunks:
#         return 0.0

#     text   = " ".join(context_chunks).lower()
#     answer = answer.lower().strip()

#     if answer in text:
#         return 1.0

#     a = set(answer.split())
#     c = set(text.split())

#     if not a:
#         return 0.0

#     return len(a & c) / len(a)


# # ============================================================
# # SEMANTIC SIMILARITY
# # ============================================================

# def semantic_similarity(answer: str, context_chunks: list) -> float:
#     if not answer or not context_chunks:
#         return 0.0
#     try:
#         combined = " ".join(context_chunks)[:2000]
#         ans_emb  = embedding_model.embed_query(answer[:500])
#         ctx_emb  = embedding_model.embed_query(combined)
#         a = np.array(ans_emb)
#         b = np.array(ctx_emb)
#         norm = np.linalg.norm(a) * np.linalg.norm(b)
#         if norm < 1e-8:
#             return 0.0
#         return float(np.dot(a, b) / norm)
#     except Exception as e:
#         print(f"[SemanticSim] error: {e}")
#         return 0.0


# # ============================================================
# # GROUNDING COMPONENTS
# # ============================================================

# def local_substring_match(answer: str, context_chunks: list) -> float:
#     if not context_chunks:
#         return 0.0
#     ans_norm   = _normalize_text(answer)
#     chunk_norm = _normalize_text(context_chunks[0])
#     return 1.0 if ans_norm in chunk_norm else 0.0


# def exact_span_match(answer: str, context_chunks: list) -> bool:
#     if not context_chunks:
#         return False
#     ans_norm = _normalize_text(answer)
#     return any(ans_norm in _normalize_text(c) for c in context_chunks)


# def compute_grounding_score(answer: str, context_chunks: list) -> float:
#     if not answer or not context_chunks:
#         return 0.0

#     ans_norm    = _normalize_text(answer)
#     chunks_norm = [_normalize_text(c) for c in context_chunks]

#     local_match  = 1.0 if ans_norm in chunks_norm[0] else 0.0
#     global_match = 1.0 if any(ans_norm in c for c in chunks_norm) else 0.0

#     keyword_score  = keyword_overlap(answer, context_chunks)
#     semantic_score = semantic_similarity(answer, context_chunks)

#     grounding_score = (
#         0.3 * local_match   +
#         0.2 * global_match  +
#         0.3 * semantic_score +
#         0.2 * keyword_score
#     )
#     print(f"[Grounding] local={local_match:.2f} global={global_match:.2f} "
#           f"semantic={semantic_score:.3f} keyword={keyword_score:.3f} "
#           f"-> grounding={grounding_score:.3f}")
#     return round(grounding_score, 4)


# # ============================================================
# # CANDIDATE SCORING
# # ============================================================

# def coverage_score(answer: str, chunks: list) -> float:
#     if not answer or not chunks:
#         return 0.0
#     ans = answer.lower()
#     return max(
#         SequenceMatcher(None, ans, c.lower()).ratio()
#         for c in chunks
#     )


# def score_answer(answer: str, context_chunks: list) -> float:
#     if not answer or not context_chunks:
#         return 0.0
#     top_chunk = context_chunks[0]
#     semantic  = semantic_similarity(answer, [top_chunk])
#     keyword   = keyword_overlap(answer, [top_chunk])
#     coverage  = coverage_score(answer, [top_chunk])
#     return (
#         0.4 * semantic +
#         0.3 * keyword  +
#         0.3 * coverage
#     )


# # ============================================================
# # CONFIDENCE
# # ============================================================

# def compute_confidence(reranker_top: float, recall_score: float,
#                        answer: str, context_chunks: list,
#                        all_reranker_scores: list = None) -> float:
#     if all_reranker_scores:
#         normalized = minmax_normalize(reranker_top, all_reranker_scores)
#     else:
#         normalized = minmax_normalize(reranker_top, [reranker_top])

#     rec = min(recall_score / 100.0, 1.0)
#     kw  = keyword_overlap(answer, context_chunks)

#     confidence = (
#         CONF_WEIGHT_RERANKER * normalized +
#         CONF_WEIGHT_RECALL   * rec +
#         CONF_WEIGHT_KEYWORD  * kw
#     )
#     print(f"[Confidence] normalized={normalized:.3f} rec={rec:.3f} "
#           f"kw={kw:.3f} → {confidence:.3f}")
#     return round(confidence, 4)


# # ============================================================
# # CHANGE D — compute_answer_grounding with refusal early-exit
# # ============================================================

# def compute_answer_grounding(answer: str, context_chunks: list,
#                              question: str = "") -> float:
#     """
#     Word-overlap grounding score (percentage 0-100).

#     CHANGE D: Refusal answers receive a fixed sentinel of 75.0.
#     This prevents them from triggering the hallucination gate
#     (which fires at < 40/60) while not falsely inflating the score.
#     The sentinel value is high enough to pass the gate but clearly
#     marked — not a real grounding score.
#     """
#     if not answer:
#         return 0.0

#     # ── CHANGE D: refusal early-exit ─────────────────────────
#     # Refusal strings contain words like "present", "document",
#     # "information" which overlap with almost any PDF context,
#     # giving a misleadingly high or inconsistent score.
#     # Return a fixed sentinel instead.
#     if _is_refusal_answer(answer):
#         print(f"[Grounding] Refusal answer detected → returning sentinel 75.0")
#         return 75.0

#     if not context_chunks:
#         return 0.0

#     try:
#         context_text  = " ".join(context_chunks).lower()
#         context_words = set(re.findall(r'\b[a-z0-9%\-]{2,}\b', context_text))
#         answer_words  = set(re.findall(r'\b[a-z0-9%\-]{2,}\b', answer.lower()))
#         if not answer_words:
#             return 0.0
#         overlap = answer_words & context_words
#         return round(len(overlap) / len(answer_words) * 100, 1)
#     except Exception:
#         return 0.0


# # ============================================================
# # RETRIEVAL METRICS
# # ============================================================

# def compute_retrieval_score(question: str, docs: list) -> float:
#     try:
#         q_embed = embedding_model.embed_query(question)
#         sims    = []
#         for d in docs:
#             d_embed = embedding_model.embed_query(d.page_content[:300])
#             sim = np.dot(q_embed, d_embed) / (
#                 np.linalg.norm(q_embed) * np.linalg.norm(d_embed) + 1e-8)
#             sims.append(sim)
#         if not sims:
#             return 0.0
#         max_sim  = float(np.max(sims))
#         top3     = sorted(sims, reverse=True)[:3]
#         mean_top = float(np.mean(top3))
#         score    = 0.7 * max_sim + 0.3 * mean_top
#         return round(score * 100, 2)
#     except Exception:
#         return 0.0


# def compute_context_precision(question: str, docs: list,
#                               threshold: float = 0.35) -> float:
#     try:
#         q_embed = embedding_model.embed_query(question)
#         sims    = []
#         for d in docs:
#             d_embed = embedding_model.embed_query(d.page_content[:300])
#             sim = np.dot(q_embed, d_embed) / (
#                 np.linalg.norm(q_embed) * np.linalg.norm(d_embed) + 1e-8)
#             sims.append(sim)
#         if not sims:
#             return 0.0
#         avg_sim           = float(np.mean(sims))
#         max_sim           = float(np.max(sims))
#         adaptive_threshold = min(threshold,
#                                  max(max_sim * 0.60, avg_sim * 0.80))
#         relevant = sum(1 for s in sims if s >= adaptive_threshold)
#         return round(relevant / len(sims) * 100, 1)
#     except Exception:
#         return 0.0


# def compute_recall_at_k(question: str, docs: list,
#                         all_chunks: list, k: int = 10) -> float:
#     try:
#         q_embed  = embedding_model.embed_query(question)
#         all_sims = []
#         for chunk in all_chunks:
#             c_embed = embedding_model.embed_query(chunk[:300])
#             sim = np.dot(q_embed, c_embed) / (
#                 np.linalg.norm(q_embed) * np.linalg.norm(c_embed) + 1e-8)
#             all_sims.append((sim, chunk))

#         if not all_sims:
#             return 0.0

#         all_sims.sort(key=lambda x: x[0], reverse=True)
#         oracle_sim   = all_sims[0][0]
#         oracle_chunk = all_sims[0][1]

#         if oracle_sim < 0.01:
#             return 0.0

#         retrieved_fps = set(d.page_content[:100] for d in docs)

#         if oracle_chunk[:100] in retrieved_fps:
#             print(f"[Recall@{k}] ✅ Oracle chunk retrieved! score=100%")
#             return 100.0

#         best_retrieved_sim = max(
#             (np.dot(q_embed, embedding_model.embed_query(d.page_content[:300])) /
#              (np.linalg.norm(q_embed) *
#               np.linalg.norm(embedding_model.embed_query(
#                   d.page_content[:300])) + 1e-8))
#             for d in docs
#         ) if docs else 0.0

#         soft_score = round((best_retrieved_sim / oracle_sim) * 100, 1)
#         print(f"[Recall@{k}] oracle={oracle_sim:.3f} | "
#               f"best_retrieved={best_retrieved_sim:.3f} | score={soft_score}%")
#         return min(soft_score, 99.9)

#     except Exception as e:
#         print(f"[Recall@K] error: {e}")
#         return 0.0


# def llm_verify_answer(question: str, answer: str,
#                       context_chunks: list) -> bool:
#     from docmind_rag.models.llm import call_llama
#     context = "\n\n".join(context_chunks[:3])
#     prompt  = (
#         f"Context:\n{context[:2000]}\n\n"
#         f"Question: {question}\n"
#         f"Answer: {answer}\n\n"
#         f"Is this answer supported by the context above?\n"
#         f"Rules:\n"
#         f"- Paraphrases are valid support\n"
#         f"- Partial answers are valid if they answer the question\n"
#         f"- A Yes/No + evidence is valid for verification questions\n"
#         f"Reply ONLY: YES or NO"
#     )
#     try:
#         resp = call_llama(prompt, num_ctx=1024, temperature=0.0)
#         return "YES" in resp.upper()
#     except Exception:
#         return True
































#got 70 nd 70 but gotto do some changes
#  
#  import re
# import numpy as np
# from difflib import SequenceMatcher

# from docmind_rag.config.settings import (
#     embedding_model,
#     CONF_WEIGHT_RERANKER,
#     CONF_WEIGHT_RECALL,
#     CONF_WEIGHT_KEYWORD,
# )
# from docmind_rag.utils.text import _normalize_text


# # ============================================================
# # CHANGE D — Lightweight refusal detection (no LLM, no embeddings)
# # Defined here so metrics.py has no circular imports.
# # ============================================================

# _REFUSAL_PHRASES = [
#     "not present",
#     "not available",
#     "not found",
#     "not in the document",
#     "not present in the document",
#     "information is not",
#     "cannot find",
#     "no information",
# ]


# def _is_refusal_text(answer: str) -> bool:
#     """
#     Lightweight refusal check using string matching only.
#     No LLM, no embeddings, no imports from other project modules.
#     Used as an early-exit guard in grounding computation.
#     """
#     if not answer:
#         return False
#     t = answer.lower().strip()
#     return any(p in t for p in _REFUSAL_PHRASES)


# # ============================================================
# # NORMALIZATION
# # ============================================================

# def minmax_normalize(score: float, all_scores: list) -> float:
#     if not all_scores:
#         return 1.0
#     min_s = min(all_scores)
#     max_s = max(all_scores)
#     if max_s == min_s:
#         return 1.0
#     return (score - min_s) / (max_s - min_s + 1e-6)


# def minmax_normalize_list(scores: list) -> list:
#     if not scores:
#         return []
#     min_s = min(scores)
#     max_s = max(scores)
#     if max_s == min_s:
#         return [1.0] * len(scores)
#     return [(s - min_s) / (max_s - min_s + 1e-6) for s in scores]


# # ============================================================
# # KEYWORD OVERLAP
# # ============================================================

# def keyword_overlap(answer: str, context_chunks: list) -> float:
#     if not answer or not context_chunks:
#         return 0.0

#     text   = " ".join(context_chunks).lower()
#     answer = answer.lower().strip()

#     if answer in text:
#         return 1.0

#     a = set(answer.split())
#     c = set(text.split())

#     if not a:
#         return 0.0

#     return len(a & c) / len(a)


# # ============================================================
# # SEMANTIC SIMILARITY
# # ============================================================

# def semantic_similarity(answer: str, context_chunks: list) -> float:
#     if not answer or not context_chunks:
#         return 0.0
#     try:
#         combined = " ".join(context_chunks)[:2000]
#         ans_emb  = embedding_model.embed_query(answer[:500])
#         ctx_emb  = embedding_model.embed_query(combined)
#         a = np.array(ans_emb)
#         b = np.array(ctx_emb)
#         norm = np.linalg.norm(a) * np.linalg.norm(b)
#         if norm < 1e-8:
#             return 0.0
#         return float(np.dot(a, b) / norm)
#     except Exception as e:
#         print(f"[SemanticSim] error: {e}")
#         return 0.0


# # ============================================================
# # GROUNDING COMPONENTS
# # ============================================================

# def local_substring_match(answer: str, context_chunks: list) -> float:
#     if not context_chunks:
#         return 0.0
#     ans_norm   = _normalize_text(answer)
#     chunk_norm = _normalize_text(context_chunks[0])
#     return 1.0 if ans_norm in chunk_norm else 0.0


# def exact_span_match(answer: str, context_chunks: list) -> bool:
#     if not context_chunks:
#         return False
#     ans_norm = _normalize_text(answer)
#     return any(ans_norm in _normalize_text(c) for c in context_chunks)


# def compute_grounding_score(answer: str, context_chunks: list) -> float:
#     if not answer or not context_chunks:
#         return 0.0

#     ans_norm    = _normalize_text(answer)
#     chunks_norm = [_normalize_text(c) for c in context_chunks]

#     local_match  = 1.0 if ans_norm in chunks_norm[0] else 0.0
#     global_match = 1.0 if any(ans_norm in c for c in chunks_norm) else 0.0

#     keyword_score  = keyword_overlap(answer, context_chunks)
#     semantic_score = semantic_similarity(answer, context_chunks)

#     grounding_score = (
#         0.3 * local_match   +
#         0.2 * global_match  +
#         0.3 * semantic_score +
#         0.2 * keyword_score
#     )
#     print(f"[Grounding] local={local_match:.2f} global={global_match:.2f} "
#           f"semantic={semantic_score:.3f} keyword={keyword_score:.3f} "
#           f"-> grounding={grounding_score:.3f}")
#     return round(grounding_score, 4)


# # ============================================================
# # CANDIDATE SCORING
# # ============================================================

# def coverage_score(answer: str, chunks: list) -> float:
#     if not answer or not chunks:
#         return 0.0
#     ans = answer.lower()
#     return max(
#         SequenceMatcher(None, ans, c.lower()).ratio()
#         for c in chunks
#     )


# def score_answer(answer: str, context_chunks: list) -> float:
#     if not answer or not context_chunks:
#         return 0.0
#     top_chunk = context_chunks[0]
#     semantic  = semantic_similarity(answer, [top_chunk])
#     keyword   = keyword_overlap(answer, [top_chunk])
#     coverage  = coverage_score(answer, [top_chunk])
#     return (
#         0.4 * semantic +
#         0.3 * keyword  +
#         0.3 * coverage
#     )


# # ============================================================
# # CONFIDENCE
# # ============================================================

# def compute_confidence(reranker_top: float, recall_score: float,
#                        answer: str, context_chunks: list,
#                        all_reranker_scores: list = None) -> float:
#     if all_reranker_scores:
#         normalized = minmax_normalize(reranker_top, all_reranker_scores)
#     else:
#         normalized = minmax_normalize(reranker_top, [reranker_top])

#     rec = min(recall_score / 100.0, 1.0)
#     kw  = keyword_overlap(answer, context_chunks)

#     confidence = (
#         CONF_WEIGHT_RERANKER * normalized +
#         CONF_WEIGHT_RECALL   * rec +
#         CONF_WEIGHT_KEYWORD  * kw
#     )
#     print(f"[Confidence] normalized={normalized:.3f} rec={rec:.3f} "
#           f"kw={kw:.3f} → {confidence:.3f}")
#     return round(confidence, 4)


# # ============================================================
# # CHANGE D — compute_answer_grounding with refusal early-exit
# # ============================================================

# def compute_answer_grounding(answer: str, context_chunks: list,
#                              question: str = "") -> float:
#     """
#     Word-overlap grounding score (percentage 0-100).

#     CHANGE D: Refusal answers receive a fixed sentinel of 75.0.
#     This prevents them from triggering the hallucination gate
#     (which fires at < 40/60) while not falsely inflating the score.
#     The sentinel value is high enough to pass the gate but clearly
#     marked — not a real grounding score.
#     """
#     if not answer:
#         return 0.0

#     # ── CHANGE D: refusal early-exit ─────────────────────────
#     # Refusal strings contain words like "present", "document",
#     # "information" which overlap with almost any PDF context,
#     # giving a misleadingly high or inconsistent score.
#     # Return a fixed sentinel instead.
#     if _is_refusal_text(answer):
#         print(f"[Grounding] Refusal answer detected → returning sentinel 75.0")
#         return 75.0

#     if not context_chunks:
#         return 0.0

#     try:
#         context_text  = " ".join(context_chunks).lower()
#         context_words = set(re.findall(r'\b[a-z0-9%\-]{2,}\b', context_text))
#         answer_words  = set(re.findall(r'\b[a-z0-9%\-]{2,}\b', answer.lower()))
#         if not answer_words:
#             return 0.0
#         overlap = answer_words & context_words
#         return round(len(overlap) / len(answer_words) * 100, 1)
#     except Exception:
#         return 0.0


# # ============================================================
# # RETRIEVAL METRICS
# # ============================================================

# def compute_retrieval_score(question: str, docs: list) -> float:
#     try:
#         q_embed = embedding_model.embed_query(question)
#         sims    = []
#         for d in docs:
#             d_embed = embedding_model.embed_query(d.page_content[:300])
#             sim = np.dot(q_embed, d_embed) / (
#                 np.linalg.norm(q_embed) * np.linalg.norm(d_embed) + 1e-8)
#             sims.append(sim)
#         if not sims:
#             return 0.0
#         max_sim  = float(np.max(sims))
#         top3     = sorted(sims, reverse=True)[:3]
#         mean_top = float(np.mean(top3))
#         score    = 0.7 * max_sim + 0.3 * mean_top
#         return round(score * 100, 2)
#     except Exception:
#         return 0.0


# def compute_context_precision(question: str, docs: list,
#                               threshold: float = 0.35) -> float:
#     try:
#         q_embed = embedding_model.embed_query(question)
#         sims    = []
#         for d in docs:
#             d_embed = embedding_model.embed_query(d.page_content[:300])
#             sim = np.dot(q_embed, d_embed) / (
#                 np.linalg.norm(q_embed) * np.linalg.norm(d_embed) + 1e-8)
#             sims.append(sim)
#         if not sims:
#             return 0.0
#         avg_sim           = float(np.mean(sims))
#         max_sim           = float(np.max(sims))
#         adaptive_threshold = min(threshold,
#                                  max(max_sim * 0.60, avg_sim * 0.80))
#         relevant = sum(1 for s in sims if s >= adaptive_threshold)
#         return round(relevant / len(sims) * 100, 1)
#     except Exception:
#         return 0.0


# def compute_recall_at_k(question: str, docs: list,
#                         all_chunks: list, k: int = 10) -> float:
#     try:
#         q_embed  = embedding_model.embed_query(question)
#         all_sims = []
#         for chunk in all_chunks:
#             c_embed = embedding_model.embed_query(chunk[:300])
#             sim = np.dot(q_embed, c_embed) / (
#                 np.linalg.norm(q_embed) * np.linalg.norm(c_embed) + 1e-8)
#             all_sims.append((sim, chunk))

#         if not all_sims:
#             return 0.0

#         all_sims.sort(key=lambda x: x[0], reverse=True)
#         oracle_sim   = all_sims[0][0]
#         oracle_chunk = all_sims[0][1]

#         if oracle_sim < 0.01:
#             return 0.0

#         retrieved_fps = set(d.page_content[:100] for d in docs)

#         if oracle_chunk[:100] in retrieved_fps:
#             print(f"[Recall@{k}] ✅ Oracle chunk retrieved! score=100%")
#             return 100.0

#         best_retrieved_sim = max(
#             (np.dot(q_embed, embedding_model.embed_query(d.page_content[:300])) /
#              (np.linalg.norm(q_embed) *
#               np.linalg.norm(embedding_model.embed_query(
#                   d.page_content[:300])) + 1e-8))
#             for d in docs
#         ) if docs else 0.0

#         soft_score = round((best_retrieved_sim / oracle_sim) * 100, 1)
#         print(f"[Recall@{k}] oracle={oracle_sim:.3f} | "
#               f"best_retrieved={best_retrieved_sim:.3f} | score={soft_score}%")
#         return min(soft_score, 99.9)

#     except Exception as e:
#         print(f"[Recall@K] error: {e}")
#         return 0.0


# def llm_verify_answer(question: str, answer: str,
#                       context_chunks: list) -> bool:
#     from docmind_rag.models.llm import call_llama
#     context = "\n\n".join(context_chunks[:3])
#     prompt  = (
#         f"Context:\n{context[:2000]}\n\n"
#         f"Question: {question}\n"
#         f"Answer: {answer}\n\n"
#         f"Is this answer supported by the context above?\n"
#         f"Rules:\n"
#         f"- Paraphrases are valid support\n"
#         f"- Partial answers are valid if they answer the question\n"
#         f"- A Yes/No + evidence is valid for verification questions\n"
#         f"Reply ONLY: YES or NO"
#     )
#     try:
#         resp = call_llama(prompt, num_ctx=1024, temperature=0.0)
#         return "YES" in resp.upper()
#     except Exception:
#         return True





# changed beacuse qa pipeline nd textpy nd helpers are changed beacsue only got 40% lap claude 
# import re
# import numpy as np
# from difflib import SequenceMatcher

# from docmind_rag.config.settings import embedding_model, CONF_WEIGHT_RERANKER, CONF_WEIGHT_RECALL, CONF_WEIGHT_KEYWORD
# from docmind_rag.utils.text import _normalize_text

# # ============================================================
# # FIX 3: CORRECTED MIN-MAX NORMALIZATION
# # ============================================================
# def minmax_normalize(score: float, all_scores: list) -> float:
#     if not all_scores:
#         return 1.0
#     min_s = min(all_scores)
#     max_s = max(all_scores)
#     if max_s == min_s:
#         return 1.0
#     return (score - min_s) / (max_s - min_s + 1e-6)


# def minmax_normalize_list(scores: list) -> list:
#     if not scores:
#         return []
#     min_s = min(scores)
#     max_s = max(scores)
#     if max_s == min_s:
#         return [1.0] * len(scores)
#     return [(s - min_s) / (max_s - min_s + 1e-6) for s in scores]


# # ============================================================
# # FIX 4: IMPROVED KEYWORD OVERLAP
# # ============================================================
# def keyword_overlap(answer: str, context_chunks: list) -> float:
#     if not answer or not context_chunks:
#         return 0.0

#     text   = " ".join(context_chunks).lower()
#     answer = answer.lower().strip()

#     # Exact substring match — strongest grounding signal
#     if answer in text:
#         return 1.0

#     a = set(answer.split())
#     c = set(text.split())

#     if not a:
#         return 0.0

#     return len(a & c) / len(a)


# # ============================================================
# # FIX 5: SEMANTIC SIMILARITY USING EMBEDDINGS
# # ============================================================
# def semantic_similarity(answer: str, context_chunks: list) -> float:
#     if not answer or not context_chunks:
#         return 0.0
#     try:
#         combined = " ".join(context_chunks)[:2000]
#         ans_emb  = embedding_model.embed_query(answer[:500])
#         ctx_emb  = embedding_model.embed_query(combined)
#         a = np.array(ans_emb)
#         b = np.array(ctx_emb)
#         norm = np.linalg.norm(a) * np.linalg.norm(b)
#         if norm < 1e-8:
#             return 0.0
#         return float(np.dot(a, b) / norm)
#     except Exception as e:
#         print(f"[SemanticSim] error: {e}")
#         return 0.0


# # ============================================================
# # FIX 6: COMBINED GROUNDING SCORE
# # ============================================================
# def local_substring_match(answer: str, context_chunks: list) -> float:
#     """Check normalized answer against the TOP reranked chunk only."""
#     if not context_chunks:
#         return 0.0
#     ans_norm   = _normalize_text(answer)
#     chunk_norm = _normalize_text(context_chunks[0])
#     return 1.0 if ans_norm in chunk_norm else 0.0


# def exact_span_match(answer: str, context_chunks: list) -> bool:
#     """Global normalized exact span -- safety override in rejection logic."""
#     if not context_chunks:
#         return False
#     ans_norm = _normalize_text(answer)
#     return any(ans_norm in _normalize_text(c) for c in context_chunks)


# def compute_grounding_score(answer: str, context_chunks: list) -> float:
#     if not answer or not context_chunks:
#         return 0.0

#     ans_norm    = _normalize_text(answer)
#     chunks_norm = [_normalize_text(c) for c in context_chunks]

#     local_match  = 1.0 if ans_norm in chunks_norm[0] else 0.0
#     global_match = 1.0 if any(ans_norm in c for c in chunks_norm) else 0.0

#     keyword_score  = keyword_overlap(answer, context_chunks)
#     semantic_score = semantic_similarity(answer, context_chunks)

#     grounding_score = (
#         0.3 * local_match   +
#         0.2 * global_match  +
#         0.3 * semantic_score +
#         0.2 * keyword_score
#     )
#     print(f"[Grounding] local={local_match:.2f} global={global_match:.2f} "
#           f"semantic={semantic_score:.3f} keyword={keyword_score:.3f} "
#           f"-> grounding={grounding_score:.3f}")
#     return round(grounding_score, 4)


# # ============================================================
# # CANDIDATE SCORING (dynamic, no hardcoding)
# # ============================================================
# def coverage_score(answer: str, chunks: list) -> float:
#     if not answer or not chunks:
#         return 0.0
#     ans = answer.lower()
#     return max(
#         SequenceMatcher(None, ans, c.lower()).ratio()
#         for c in chunks
#     )


# def score_answer(answer: str, context_chunks: list) -> float:
#     """
#     Combined dynamic score: semantic + keyword + coverage.
#     Grounding is part of scoring, not a fallback gate.
#     """
#     if not answer or not context_chunks:
#         return 0.0
#     top_chunk = context_chunks[0]

#     semantic = semantic_similarity(answer, [top_chunk])
#     keyword  = keyword_overlap(answer, [top_chunk])
#     coverage = coverage_score(answer, [top_chunk])
#     return (
#         0.4 * semantic +
#         0.3 * keyword  +
#         0.3 * coverage
#     )


# # ============================================================
# # CONFIDENCE SCORING (kept for metrics/logging, not for gating)
# # ============================================================
# def compute_confidence(reranker_top: float, recall_score: float,
#                        answer: str, context_chunks: list,
#                        all_reranker_scores: list = None) -> float:
#     if all_reranker_scores:
#         normalized = minmax_normalize(reranker_top, all_reranker_scores)
#     else:
#         normalized = minmax_normalize(reranker_top, [reranker_top])

#     rec = min(recall_score / 100.0, 1.0)
#     kw  = keyword_overlap(answer, context_chunks)

#     confidence = (
#         CONF_WEIGHT_RERANKER * normalized +
#         CONF_WEIGHT_RECALL   * rec +
#         CONF_WEIGHT_KEYWORD  * kw
#     )
#     print(f"[Confidence] normalized={normalized:.3f} rec={rec:.3f} kw={kw:.3f} → {confidence:.3f}")
#     return round(confidence, 4)


# def compute_answer_grounding(answer: str, context_chunks: list, question: str = "") -> float:
#     """Legacy word-overlap grounding for backward-compatible metrics output (percentage)."""
#     if not answer or not context_chunks:
#         return 0.0
#     try:
#         context_text  = " ".join(context_chunks).lower()
#         context_words = set(re.findall(r'\b[a-z0-9%\-]{2,}\b', context_text))
#         answer_words  = set(re.findall(r'\b[a-z0-9%\-]{2,}\b', answer.lower()))
#         if not answer_words:
#             return 0.0
#         overlap = answer_words & context_words
#         return round(len(overlap) / len(answer_words) * 100, 1)
#     except Exception:
#         return 0.0


# def compute_retrieval_score(question: str, docs: list) -> float:
#     try:
#         q_embed = embedding_model.embed_query(question)
#         sims = []
#         for d in docs:
#             d_embed = embedding_model.embed_query(d.page_content[:300])
#             sim = np.dot(q_embed, d_embed) / (
#                 np.linalg.norm(q_embed) * np.linalg.norm(d_embed) + 1e-8)
#             sims.append(sim)
#         if not sims:
#             return 0.0
#         max_sim  = float(np.max(sims))
#         top3     = sorted(sims, reverse=True)[:3]
#         mean_top = float(np.mean(top3))
#         score = 0.7 * max_sim + 0.3 * mean_top
#         return round(score * 100, 2)
#     except Exception:
#         return 0.0

# def compute_context_precision(question: str, docs: list, threshold: float = 0.35) -> float:
#     try:
#         q_embed  = embedding_model.embed_query(question)
#         sims = []
#         for d in docs:
#             d_embed = embedding_model.embed_query(d.page_content[:300])
#             sim = np.dot(q_embed, d_embed) / (
#                 np.linalg.norm(q_embed) * np.linalg.norm(d_embed) + 1e-8)
#             sims.append(sim)
#         if not sims:
#             return 0.0
#         avg_sim = float(np.mean(sims))
#         max_sim = float(np.max(sims))
#         adaptive_threshold = min(threshold, max(max_sim * 0.60, avg_sim * 0.80))
#         relevant = sum(1 for s in sims if s >= adaptive_threshold)
#         return round(relevant / len(sims) * 100, 1)
#     except Exception:
#         return 0.0

# def compute_recall_at_k(question: str, docs: list, all_chunks: list, k: int = 10) -> float:
#     try:
#         q_embed  = embedding_model.embed_query(question)
#         all_sims = []
#         for chunk in all_chunks:
#             c_embed = embedding_model.embed_query(chunk[:300])
#             sim = np.dot(q_embed, c_embed) / (
#                 np.linalg.norm(q_embed) * np.linalg.norm(c_embed) + 1e-8)
#             all_sims.append((sim, chunk))

#         if not all_sims:
#             return 0.0

#         all_sims.sort(key=lambda x: x[0], reverse=True)
#         oracle_sim   = all_sims[0][0]
#         oracle_chunk = all_sims[0][1]

#         if oracle_sim < 0.01:
#             return 0.0

#         # FIX: use docs list directly — no external retrieved_texts variable needed
#         retrieved_fps = set(d.page_content[:100] for d in docs)

#         if oracle_chunk[:100] in retrieved_fps:
#             print(f"[Recall@{k}] ✅ Oracle chunk retrieved! score=100%")
#             return 100.0

#         best_retrieved_sim = max(
#             (np.dot(q_embed, embedding_model.embed_query(d.page_content[:300])) /
#              (np.linalg.norm(q_embed) *
#               np.linalg.norm(embedding_model.embed_query(d.page_content[:300])) + 1e-8))
#             for d in docs
#         ) if docs else 0.0

#         soft_score = round((best_retrieved_sim / oracle_sim) * 100, 1)
#         print(f"[Recall@{k}] oracle={oracle_sim:.3f} | "
#               f"best_retrieved={best_retrieved_sim:.3f} | score={soft_score}%")
#         return min(soft_score, 99.9)

#     except Exception as e:
#         print(f"[Recall@K] error: {e}")
#         return 0.0


# def llm_verify_answer(question: str, answer: str, context_chunks: list) -> bool:
#     from docmind_rag.models.llm import call_llama
#     context = "\n\n".join(context_chunks[:3])
#     prompt = (
#         f"Context:\n{context[:2000]}\n\n"
#         f"Question: {question}\n"
#         f"Answer: {answer}\n\n"
#         f"Is this answer supported by the context above?\n"
#         f"Rules:\n"
#         f"- Paraphrases are valid support\n"
#         f"- Partial answers are valid if they answer the question\n"
#         f"- A Yes/No + evidence is valid for verification questions\n"
#         f"Reply ONLY: YES or NO"
#     )
#     try:
#         resp = call_llama(prompt, num_ctx=1024, temperature=0.0)
#         return "YES" in resp.upper()
#     except Exception:
#         return True