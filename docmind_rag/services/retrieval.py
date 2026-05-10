# import re
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
# from langchain.schema import Document

# from docmind_rag.models.llm import call_llama
# from docmind_rag.utils.helpers import _get_dynamic_stopwords


# def keyword_retrieve(question: str, all_chunks: list, k: int = 10) -> list:
#     if not all_chunks:
#         return []

#     try:
#         # 🔥 FIX — normalize input FIRST
#         all_chunks = [
#             c.page_content if hasattr(c, "page_content") else c
#             for c in all_chunks
#         ]

#         vectorizer = TfidfVectorizer(
#             stop_words="english",
#             ngram_range=(1, 2),
#             max_features=10000
#         )

#         tfidf_matrix = vectorizer.fit_transform(all_chunks)
#         q_vec = vectorizer.transform([question])
#         scores = sklearn_cosine(q_vec, tfidf_matrix).flatten()

#         top_indices = np.argsort(scores)[::-1][:k]

#         results = []

#         for idx in top_indices:
#             if scores[idx] > 0:
#                 text = all_chunks[idx]

#                 results.append(
#                     Document(
#                         page_content=str(text),
#                         metadata={
#                             "chunk_id": int(idx),
#                             "keyword_score": float(scores[idx])
#                         }
#                     )
#                 )

#         print(f"[KeywordSearch] '{question[:50]}' → {len(results)} chunks | "
#               f"top score={scores[top_indices[0]]:.3f}")

#         return results

#     except Exception as e:
#         print(f"[KeywordSearch] error: {e}")
#         return []
# def exact_match_retrieve(question: str, all_chunks: list) -> list:
#     if not all_chunks:
#         return []
#     all_chunks = [
#     c.page_content if hasattr(c, "page_content") else c
#     for c in all_chunks
# ]

#     number_patterns = re.findall(r'\b\d+\b', question)
#     stopwords = _get_dynamic_stopwords()
#     key_words = [w.lower() for w in question.split()
#                  if len(w) > 2 and w.lower() not in stopwords]

#     matched = []
#     seen    = set()

#     for i, chunk in enumerate(all_chunks):
#         chunk_lower = str(chunk).lower()
#         score = 0

#         for num in number_patterns:
#             if re.search(rf'\b{num}\b', chunk_lower):
#                 score += 3

#             # TITLE BOOST: number AND at least one content keyword
#             # both appear in the first line.
#             # Requiring co-occurrence prevents single-digit false positives
#             # (e.g. "3" matching page numbers, dates, list items everywhere).
#             text = chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
#             first_line = text.strip().split('\n')[0].lower()
#             num_in_first_line = bool(re.search(rf'\b{num}\b', first_line))
#             kw_in_first_line  = any(kw in first_line for kw in key_words)

#             if num_in_first_line and kw_in_first_line:
#                 score += 5
#                 print(f"[ExactMatch] Title boost: chunk {i} "
#                       f"(num={num} + keyword co-occur in first line)")

#         for word in key_words:
#             if word in chunk_lower:
#                 score += 1

#         if score > 0:
#             text = chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
#             fingerprint = text[:120].strip()
#             if fingerprint not in seen:
#                 seen.add(fingerprint)
#                 matched.append((score,Document(
#     page_content=text,
#     metadata={"chunk_id": i, "exact_score": score}
# )))

#     matched.sort(key=lambda x: x[0], reverse=True)

#     if not matched:
#         return []

#     top_score  = matched[0][0]
#     gap_cutoff = top_score * 0.5
#     filtered   = [(s, d) for s, d in matched if s >= gap_cutoff]
#     results    = [d for _, d in filtered[:8]]

#     if results:
#         print(f"[ExactMatch] '{question[:50]}' → {len(results)} exact chunks | "
#               f"top score={matched[0][0]}")
#     return results


# def rewrite_query_fallback(question: str) -> list:
#     prompt = (
#         f"Rewrite this question into 2 alternative search queries. "
#         f"Keep the meaning the same. No assumptions. "
#         f"Each on a new line. No numbering.\n\n"
#         f"Question: {question}\n\nAlternative queries:"
#     )
#     try:
#         raw      = call_llama(prompt, num_ctx=1024, temperature=0.0)
#         variants = [l.strip() for l in raw.split("\n")
#                     if l.strip() and len(l.strip()) > 5][:2]
#         return [question] + variants
#     except Exception:
#         return [question]


# def multi_query_retrieve(question: str, faiss_index, k: int = 12,
#                          all_chunks: list = None,
#                          query_type: str = "FACTUAL_QA",
#                          queries: list = None) -> list:
#     seen_weights: dict = {}
#     chunk_map:    dict = {}

#     all_queries = queries if queries else [question]
#     print(f"[Hybrid] {len(all_queries)} queries: {all_queries}")

#     # per_q_k = max(6, k // len(all_queries))
#     per_q_k = max(4, k // len(all_queries))
#     for q in all_queries:
#         try:
#             docs = faiss_index.similarity_search(q, k=per_q_k)
#             for d in docs:
#                 key = d.page_content[:120].strip()
#                 seen_weights[key] = max(seen_weights.get(key, 0), 0.6)
#                 chunk_map[key]    = d
#         except Exception as e:
#             print(f"[Hybrid] semantic error for '{q}': {e}")

#     print(f"[Hybrid] {len(seen_weights)} unique chunks after semantic")

#     if all_chunks:
#         try:
#             exact_docs = exact_match_retrieve(question, all_chunks)
#             for d in exact_docs:
#                 key = d.page_content[:120].strip()
#                 seen_weights[key] = max(seen_weights.get(key, 0), 0.6)
#                 chunk_map[key]    = d
#             if exact_docs:
#                 print(f"[Hybrid] {len(exact_docs)} exact-match chunks boosted to 0.6")
#         except Exception as e:
#             print(f"[Hybrid] exact match error: {e}")

#     if all_chunks:
#         try:
#             kw_docs = keyword_retrieve(question, all_chunks, k=8)
#             for d in kw_docs:
#                 key = d.page_content[:120].strip()
#                 seen_weights[key] = max(seen_weights.get(key, 0), 0.3)
#                 chunk_map[key]    = d
#         except Exception as e:
#             print(f"[Hybrid] keyword error: {e}")

#     print(f"[Hybrid] {len(seen_weights)} total unique chunks across all signals")

#     sorted_keys = sorted(seen_weights.keys(),
#                          key=lambda x: seen_weights[x],
#                          reverse=True)

#     final_docs = [chunk_map[key] for key in sorted_keys[:k]]

#     if len(final_docs) < k:
#         try:
#             seen_set = set(seen_weights.keys())
#             mmr_docs = faiss_index.max_marginal_relevance_search(
#                 question, k=k, fetch_k=k * 3)
#             for d in mmr_docs:
#                 key = d.page_content[:120].strip()
#                 if key not in seen_set:
#                     final_docs.append(d)
#                     seen_set.add(key)
#             print(f"[Hybrid] After MMR fallback: {len(final_docs)} chunks")
#         except Exception as e:
#             print(f"[Hybrid] MMR fallback error: {e}")

#     top_weight = seen_weights.get(sorted_keys[0], 0) if sorted_keys else 0

#     print(f"[Hybrid] Final pool: {len(final_docs)} chunks | "
#         f"top weight={top_weight:.1f}")

#     return final_docs[:k]


























































# single pdf 
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from langchain.schema import Document

from docmind_rag.models.llm import call_llama
from docmind_rag.utils.helpers import _get_dynamic_stopwords


def keyword_retrieve(question: str, all_chunks: list, k: int = 10) -> list:
    if not all_chunks:
        return []
    try:
        vectorizer   = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=10000
        )
        tfidf_matrix = vectorizer.fit_transform(all_chunks)
        q_vec        = vectorizer.transform([question])
        scores       = sklearn_cosine(q_vec, tfidf_matrix).flatten()
        top_indices  = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(Document(
                    page_content=all_chunks[idx],
                    metadata={"chunk_id": int(idx), "keyword_score": float(scores[idx])}
                ))
        print(f"[KeywordSearch] '{question[:50]}' → {len(results)} chunks | "
              f"top score={scores[top_indices[0]]:.3f}")
        return results
    except Exception as e:
        print(f"[KeywordSearch] error: {e}")
        return []


def exact_match_retrieve(question: str, all_chunks: list) -> list:
    if not all_chunks:
        return []

    number_patterns = re.findall(r'\b\d+\b', question)
    stopwords = _get_dynamic_stopwords()
    key_words = [w.lower() for w in question.split()
                 if len(w) > 2 and w.lower() not in stopwords]

    matched = []
    seen    = set()

    for i, chunk in enumerate(all_chunks):
        chunk_lower = chunk.lower()
        score = 0

        for num in number_patterns:
            if re.search(rf'\b{num}\b', chunk_lower):
                score += 3

            # TITLE BOOST: number AND at least one content keyword
            # both appear in the first line.
            # Requiring co-occurrence prevents single-digit false positives
            # (e.g. "3" matching page numbers, dates, list items everywhere).
            first_line = chunk.strip().split('\n')[0].lower()
            num_in_first_line = bool(re.search(rf'\b{num}\b', first_line))
            kw_in_first_line  = any(kw in first_line for kw in key_words)

            if num_in_first_line and kw_in_first_line:
                score += 5
                print(f"[ExactMatch] Title boost: chunk {i} "
                      f"(num={num} + keyword co-occur in first line)")

        for word in key_words:
            if word in chunk_lower:
                score += 1

        if score > 0:
            fingerprint = chunk[:120].strip()
            if fingerprint not in seen:
                seen.add(fingerprint)
                matched.append((score, Document(
                    page_content=chunk,
                    metadata={"chunk_id": i, "exact_score": score}
                )))

    matched.sort(key=lambda x: x[0], reverse=True)

    if not matched:
        return []

    top_score  = matched[0][0]
    gap_cutoff = top_score * 0.5
    filtered   = [(s, d) for s, d in matched if s >= gap_cutoff]
    results    = [d for _, d in filtered[:8]]

    if results:
        print(f"[ExactMatch] '{question[:50]}' → {len(results)} exact chunks | "
              f"top score={matched[0][0]}")
    return results


def rewrite_query_fallback(question: str) -> list:
    prompt = (
        f"Rewrite this question into 2 alternative search queries. "
        f"Keep the meaning the same. No assumptions. "
        f"Each on a new line. No numbering.\n\n"
        f"Question: {question}\n\nAlternative queries:"
    )
    try:
        raw      = call_llama(prompt, num_ctx=1024, temperature=0.0)
        variants = [l.strip() for l in raw.split("\n")
                    if l.strip() and len(l.strip()) > 5][:2]
        return [question] + variants
    except Exception:
        return [question]


def multi_query_retrieve(question: str, faiss_index, k: int = 12,
                         all_chunks: list = None,
                         query_type: str = "FACTUAL_QA",
                         queries: list = None) -> list:
    seen_weights: dict = {}
    chunk_map:    dict = {}

    all_queries = queries if queries else [question]
    print(f"[Hybrid] {len(all_queries)} queries: {all_queries}")

    # per_q_k = max(6, k // len(all_queries))
    per_q_k = max(4, k // len(all_queries))
    for q in all_queries:
        try:
            docs = faiss_index.similarity_search(q, k=per_q_k)
            for d in docs:
                key = d.page_content[:120].strip()
                seen_weights[key] = max(seen_weights.get(key, 0), 0.6)
                chunk_map[key]    = d
        except Exception as e:
            print(f"[Hybrid] semantic error for '{q}': {e}")

    print(f"[Hybrid] {len(seen_weights)} unique chunks after semantic")

    if all_chunks:
        try:
            exact_docs = exact_match_retrieve(question, all_chunks)
            for d in exact_docs:
                key = d.page_content[:120].strip()
                seen_weights[key] = max(seen_weights.get(key, 0), 0.6)
                chunk_map[key]    = d
            if exact_docs:
                print(f"[Hybrid] {len(exact_docs)} exact-match chunks boosted to 0.6")
        except Exception as e:
            print(f"[Hybrid] exact match error: {e}")

    if all_chunks:
        try:
            kw_docs = keyword_retrieve(question, all_chunks, k=8)
            for d in kw_docs:
                key = d.page_content[:120].strip()
                seen_weights[key] = max(seen_weights.get(key, 0), 0.3)
                chunk_map[key]    = d
        except Exception as e:
            print(f"[Hybrid] keyword error: {e}")

    print(f"[Hybrid] {len(seen_weights)} total unique chunks across all signals")

    sorted_keys = sorted(seen_weights.keys(),
                         key=lambda x: seen_weights[x],
                         reverse=True)

    final_docs = [chunk_map[key] for key in sorted_keys[:k]]

    if len(final_docs) < k:
        try:
            seen_set = set(seen_weights.keys())
            mmr_docs = faiss_index.max_marginal_relevance_search(
                question, k=k, fetch_k=k * 3)
            for d in mmr_docs:
                key = d.page_content[:120].strip()
                if key not in seen_set:
                    final_docs.append(d)
                    seen_set.add(key)
            print(f"[Hybrid] After MMR fallback: {len(final_docs)} chunks")
        except Exception as e:
            print(f"[Hybrid] MMR fallback error: {e}")

    top_weight = seen_weights.get(sorted_keys[0], 0) if sorted_keys else 0

    print(f"[Hybrid] Final pool: {len(final_docs)} chunks | "
        f"top weight={top_weight:.1f}")

    return final_docs[:k]






























# # got 100 for  ethihc  nd 57 for  crop  .  got overftted gonna chnage top_k now make it adaptive
# import re
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
# from langchain.schema import Document

# from docmind_rag.models.llm import call_llama
# from docmind_rag.utils.helpers import _get_dynamic_stopwords


# def keyword_retrieve(question: str, all_chunks: list, k: int = 10) -> list:
#     if not all_chunks:
#         return []
#     try:
#         vectorizer   = TfidfVectorizer(
#             stop_words="english",
#             ngram_range=(1, 2),
#             max_features=10000
#         )
#         tfidf_matrix = vectorizer.fit_transform(all_chunks)
#         q_vec        = vectorizer.transform([question])
#         scores       = sklearn_cosine(q_vec, tfidf_matrix).flatten()
#         top_indices  = np.argsort(scores)[::-1][:k]
#         results = []
#         for idx in top_indices:
#             if scores[idx] > 0:
#                 results.append(Document(
#                     page_content=all_chunks[idx],
#                     metadata={"chunk_id": int(idx), "keyword_score": float(scores[idx])}
#                 ))
#         print(f"[KeywordSearch] '{question[:50]}' → {len(results)} chunks | "
#               f"top score={scores[top_indices[0]]:.3f}")
#         return results
#     except Exception as e:
#         print(f"[KeywordSearch] error: {e}")
#         return []


# def exact_match_retrieve(question: str, all_chunks: list) -> list:
#     if not all_chunks:
#         return []

#     number_patterns = re.findall(r'\b\d+\b', question)
#     stopwords = _get_dynamic_stopwords()
#     key_words = [w.lower() for w in question.split()
#                  if len(w) > 2 and w.lower() not in stopwords]

#     matched = []
#     seen    = set()

#     for i, chunk in enumerate(all_chunks):
#         chunk_lower = chunk.lower()
#         score = 0

#         for num in number_patterns:
#             if re.search(rf'\b{num}\b', chunk_lower):
#                 score += 3

#             # TITLE BOOST: number AND at least one content keyword
#             # both appear in the first line.
#             # Requiring co-occurrence prevents single-digit false positives
#             # (e.g. "3" matching page numbers, dates, list items everywhere).
#             first_line = chunk.strip().split('\n')[0].lower()
#             num_in_first_line = bool(re.search(rf'\b{num}\b', first_line))
#             kw_in_first_line  = any(kw in first_line for kw in key_words)

#             if num_in_first_line and kw_in_first_line:
#                 score += 5
#                 print(f"[ExactMatch] Title boost: chunk {i} "
#                       f"(num={num} + keyword co-occur in first line)")

#         for word in key_words:
#             if word in chunk_lower:
#                 score += 1

#         if score > 0:
#             fingerprint = chunk[:120].strip()
#             if fingerprint not in seen:
#                 seen.add(fingerprint)
#                 matched.append((score, Document(
#                     page_content=chunk,
#                     metadata={"chunk_id": i, "exact_score": score}
#                 )))

#     matched.sort(key=lambda x: x[0], reverse=True)

#     if not matched:
#         return []

#     top_score  = matched[0][0]
#     gap_cutoff = top_score * 0.5
#     filtered   = [(s, d) for s, d in matched if s >= gap_cutoff]
#     results    = [d for _, d in filtered[:8]]

#     if results:
#         print(f"[ExactMatch] '{question[:50]}' → {len(results)} exact chunks | "
#               f"top score={matched[0][0]}")
#     return results


# def rewrite_query_fallback(question: str) -> list:
#     prompt = (
#         f"Rewrite this question into 2 alternative search queries. "
#         f"Keep the meaning the same. No assumptions. "
#         f"Each on a new line. No numbering.\n\n"
#         f"Question: {question}\n\nAlternative queries:"
#     )
#     try:
#         raw      = call_llama(prompt, num_ctx=1024, temperature=0.0)
#         variants = [l.strip() for l in raw.split("\n")
#                     if l.strip() and len(l.strip()) > 5][:2]
#         return [question] + variants
#     except Exception:
#         return [question]


# def multi_query_retrieve(question: str, faiss_index, k: int = 12,
#                          all_chunks: list = None,
#                          query_type: str = "FACTUAL_QA",
#                          queries: list = None) -> list:
#     seen_weights: dict = {}
#     chunk_map:    dict = {}

#     all_queries = queries if queries else [question]
#     print(f"[Hybrid] {len(all_queries)} queries: {all_queries}")

#     per_q_k = max(6, k // len(all_queries))
#     for q in all_queries:
#         try:
#             docs = faiss_index.similarity_search(q, k=per_q_k)
#             for d in docs:
#                 key = d.page_content[:120].strip()
#                 seen_weights[key] = max(seen_weights.get(key, 0), 0.6)
#                 chunk_map[key]    = d
#         except Exception as e:
#             print(f"[Hybrid] semantic error for '{q}': {e}")

#     print(f"[Hybrid] {len(seen_weights)} unique chunks after semantic")

#     if all_chunks:
#         try:
#             exact_docs = exact_match_retrieve(question, all_chunks)
#             for d in exact_docs:
#                 key = d.page_content[:120].strip()
#                 seen_weights[key] = max(seen_weights.get(key, 0), 0.9)
#                 chunk_map[key]    = d
#             if exact_docs:
#                 print(f"[Hybrid] {len(exact_docs)} exact-match chunks boosted to 0.9")
#         except Exception as e:
#             print(f"[Hybrid] exact match error: {e}")

#     if all_chunks:
#         try:
#             kw_docs = keyword_retrieve(question, all_chunks, k=8)
#             for d in kw_docs:
#                 key = d.page_content[:120].strip()
#                 seen_weights[key] = max(seen_weights.get(key, 0), 0.3)
#                 chunk_map[key]    = d
#         except Exception as e:
#             print(f"[Hybrid] keyword error: {e}")

#     print(f"[Hybrid] {len(seen_weights)} total unique chunks across all signals")

#     sorted_keys = sorted(seen_weights.keys(),
#                          key=lambda x: seen_weights[x],
#                          reverse=True)

#     final_docs = [chunk_map[key] for key in sorted_keys[:k]]

#     if len(final_docs) < k:
#         try:
#             seen_set = set(seen_weights.keys())
#             mmr_docs = faiss_index.max_marginal_relevance_search(
#                 question, k=k, fetch_k=k * 3)
#             for d in mmr_docs:
#                 key = d.page_content[:120].strip()
#                 if key not in seen_set:
#                     final_docs.append(d)
#                     seen_set.add(key)
#             print(f"[Hybrid] After MMR fallback: {len(final_docs)} chunks")
#         except Exception as e:
#             print(f"[Hybrid] MMR fallback error: {e}")

#     print(f"[Hybrid] Final pool: {len(final_docs)} chunks | "
#           f"top weight={seen_weights.get(sorted_keys[0], 0):.1f} "
#           f"if sorted_keys else 0")
#     return final_docs[:k]



















# import re
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
# from langchain.schema import Document

# from docmind_rag.models.llm import call_llama
# from docmind_rag.utils.helpers import _get_dynamic_stopwords

# # ============================================================
# # HYBRID RETRIEVAL: Keyword + Semantic + MMR
# # ============================================================
# def keyword_retrieve(question: str, all_chunks: list, k: int = 10) -> list:
#     if not all_chunks:
#         return []
#     try:
#         vectorizer   = TfidfVectorizer(
#             stop_words="english",
#             ngram_range=(1, 2),
#             max_features=10000
#         )
#         tfidf_matrix = vectorizer.fit_transform(all_chunks)
#         q_vec        = vectorizer.transform([question])
#         scores       = sklearn_cosine(q_vec, tfidf_matrix).flatten()
#         top_indices  = np.argsort(scores)[::-1][:k]
#         results = []
#         for idx in top_indices:
#             if scores[idx] > 0:
#                 results.append(Document(
#                     page_content=all_chunks[idx],
#                     metadata={"chunk_id": int(idx), "keyword_score": float(scores[idx])}
#                 ))
#         print(f"[KeywordSearch] '{question[:50]}' → {len(results)} chunks | "
#               f"top score={scores[top_indices[0]]:.3f}")
#         return results
#     except Exception as e:
#         print(f"[KeywordSearch] error: {e}")
#         return []


# # ============================================================
# # FIX: DYNAMIC exact_match_retrieve — no hardcoded stopwords list
# # ============================================================
# def exact_match_retrieve(question: str, all_chunks: list) -> list:
#     if not all_chunks:
#         return []

#     number_patterns = re.findall(r'\b\d+\b', question)
#     stopwords = _get_dynamic_stopwords()
#     key_words = [w.lower() for w in question.split()
#                  if len(w) > 2 and w.lower() not in stopwords]

#     matched = []
#     seen    = set()

#     for i, chunk in enumerate(all_chunks):
#         chunk_lower = chunk.lower()
#         score = 0
#         for num in number_patterns:
#             if re.search(rf'\b{num}\b', chunk_lower):
#                 score += 3
#         for word in key_words:
#             if word in chunk_lower:
#                 score += 1
#         if score > 0:
            
#             fingerprint = chunk[:120].strip()
#             if fingerprint not in seen:
#                 seen.add(fingerprint)
#                 matched.append((score, Document(
#                     page_content=chunk,
#                     metadata={"chunk_id": i, "exact_score": score}
#                 )))

#     matched.sort(key=lambda x: x[0], reverse=True)

#     # 🚨 NEW LOGIC (dynamic filtering instead of fixed slice)
#     if not matched:
#         return []

#     top_score  = matched[0][0]
#     gap_cutoff = top_score * 0.5   # keep chunks close to best

#     filtered   = [(s, d) for s, d in matched if s >= gap_cutoff]

#     results    = [d for _, d in filtered[:8]]  # still cap at 8 max
#     if results:
#         print(f"[ExactMatch] '{question[:50]}' → {len(results)} exact chunks | "
#               f"top score={matched[0][0]}")
#     return results


# def rewrite_query_fallback(question: str) -> list:
#     prompt = (
#         f"Rewrite this question into 2 alternative search queries. "
#         f"Keep the meaning the same. No assumptions. "
#         f"Each on a new line. No numbering.\n\n"
#         f"Question: {question}\n\nAlternative queries:"
#     )
#     try:
#         raw      = call_llama(prompt, num_ctx=1024, temperature=0.0)
#         variants = [l.strip() for l in raw.split("\n")
#                     if l.strip() and len(l.strip()) > 5][:2]
#         return [question] + variants
#     except Exception:
#         return [question]


# def multi_query_retrieve(question: str, faiss_index, k: int = 12,
#                          all_chunks: list = None,
#                          query_type: str = "FACTUAL_QA",
#                          queries: list = None) -> list:
#     seen_weights: dict = {}
#     chunk_map:    dict = {}

#     all_queries = queries if queries else [question]
#     print(f"[Hybrid] {len(all_queries)} queries: {all_queries}")

#     per_q_k = max(6, k // len(all_queries))
#     for q in all_queries:
#         try:
#             docs = faiss_index.similarity_search(q, k=per_q_k)
#             for d in docs:
#                 key = d.page_content[:120].strip()
#                 seen_weights[key] = max(seen_weights.get(key, 0), 0.6)
#                 chunk_map[key]    = d
#         except Exception as e:
#             print(f"[Hybrid] semantic error for '{q}': {e}")

#     print(f"[Hybrid] {len(seen_weights)} unique chunks after semantic")

#     if all_chunks:
#         try:
#             exact_docs = exact_match_retrieve(question, all_chunks)
#             for d in exact_docs:
#                 key = d.page_content[:120].strip()
#                 seen_weights[key] = max(seen_weights.get(key, 0), 0.9)
#                 chunk_map[key]    = d
#             if exact_docs:
#                 print(f"[Hybrid] {len(exact_docs)} exact-match chunks boosted to 0.9")
#         except Exception as e:
#             print(f"[Hybrid] exact match error: {e}")

#     if all_chunks:
#         try:
#             kw_docs = keyword_retrieve(question, all_chunks, k=8)
#             for d in kw_docs:
#                 key = d.page_content[:120].strip()
#                 seen_weights[key] = max(seen_weights.get(key, 0), 0.3)
#                 chunk_map[key]    = d
#         except Exception as e:
#             print(f"[Hybrid] keyword error: {e}")

#     print(f"[Hybrid] {len(seen_weights)} total unique chunks across all signals")

#     sorted_keys = sorted(seen_weights.keys(),
#                          key=lambda x: seen_weights[x],
#                          reverse=True)

#     final_docs = [chunk_map[key] for key in sorted_keys[:k]]

#     if len(final_docs) < k:
#         try:
#             seen_set = set(seen_weights.keys())
#             mmr_docs = faiss_index.max_marginal_relevance_search(
#                 question, k=k, fetch_k=k * 3)
#             for d in mmr_docs:
#                 key = d.page_content[:120].strip()
#                 if key not in seen_set:
#                     final_docs.append(d)
#                     seen_set.add(key)
#             print(f"[Hybrid] After MMR fallback: {len(final_docs)} chunks")
#         except Exception as e:
#             print(f"[Hybrid] MMR fallback error: {e}")

#     print(f"[Hybrid] Final pool: {len(final_docs)} chunks | "
#           f"top weight={seen_weights.get(sorted_keys[0], 0):.1f} "
#           f"if sorted_keys else 0")
#     return final_docs[:k]


























# import re
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
# from langchain.schema import Document

# from docmind_rag.models.llm import call_llama
# from docmind_rag.utils.helpers import _get_dynamic_stopwords

# # ============================================================
# # HYBRID RETRIEVAL: Keyword + Semantic + MMR
# # ============================================================
# def keyword_retrieve(question: str, all_chunks: list, k: int = 10) -> list:
#     if not all_chunks:
#         return []
#     try:
#         vectorizer   = TfidfVectorizer(
#             stop_words="english",
#             ngram_range=(1, 2),
#             max_features=10000
#         )
#         tfidf_matrix = vectorizer.fit_transform(all_chunks)
#         q_vec        = vectorizer.transform([question])
#         scores       = sklearn_cosine(q_vec, tfidf_matrix).flatten()
#         top_indices  = np.argsort(scores)[::-1][:k]
#         results = []
#         for idx in top_indices:
#             if scores[idx] > 0:
#                 results.append(Document(
#                     page_content=all_chunks[idx],
#                     metadata={"chunk_id": int(idx), "keyword_score": float(scores[idx])}
#                 ))
#         print(f"[KeywordSearch] '{question[:50]}' → {len(results)} chunks | "
#               f"top score={scores[top_indices[0]]:.3f}")
#         return results
#     except Exception as e:
#         print(f"[KeywordSearch] error: {e}")
#         return []


# # ============================================================
# # FIX: DYNAMIC exact_match_retrieve — no hardcoded stopwords list
# # ============================================================
# def exact_match_retrieve(question: str, all_chunks: list) -> list:
#     if not all_chunks:
#         return []

#     number_patterns = re.findall(r'\b\d+\b', question)
#     stopwords = _get_dynamic_stopwords()
#     key_words = [w.lower() for w in question.split()
#                  if len(w) > 2 and w.lower() not in stopwords]

#     matched = []
#     seen    = set()

#     for i, chunk in enumerate(all_chunks):
#         chunk_lower = chunk.lower()
#         score = 0
#         for num in number_patterns:
#             if re.search(rf'\b{num}\b', chunk_lower):
#                 score += 3
#         for word in key_words:
#             if word in chunk_lower:
#                 score += 1
#         if score >= 3:
#             fingerprint = chunk[:120].strip()
#             if fingerprint not in seen:
#                 seen.add(fingerprint)
#                 matched.append((score, Document(
#                     page_content=chunk,
#                     metadata={"chunk_id": i, "exact_score": score}
#                 )))

#     matched.sort(key=lambda x: x[0], reverse=True)
#     results = [d for _, d in matched[:8]]
#     if results:
#         print(f"[ExactMatch] '{question[:50]}' → {len(results)} exact chunks | "
#               f"top score={matched[0][0]}")
#     return results


# def rewrite_query_fallback(question: str) -> list:
#     prompt = (
#         f"Rewrite this question into 2 alternative search queries. "
#         f"Keep the meaning the same. No assumptions. "
#         f"Each on a new line. No numbering.\n\n"
#         f"Question: {question}\n\nAlternative queries:"
#     )
#     try:
#         raw      = call_llama(prompt, num_ctx=1024, temperature=0.0)
#         variants = [l.strip() for l in raw.split("\n")
#                     if l.strip() and len(l.strip()) > 5][:2]
#         return [question] + variants
#     except Exception:
#         return [question]


# def multi_query_retrieve(question: str, faiss_index, k: int = 12,
#                          all_chunks: list = None,
#                          query_type: str = "FACTUAL_QA",
#                          queries: list = None) -> list:
#     seen_weights: dict = {}
#     chunk_map:    dict = {}

#     all_queries = queries if queries else [question]
#     print(f"[Hybrid] {len(all_queries)} queries: {all_queries}")

#     per_q_k = max(6, k // len(all_queries))
#     for q in all_queries:
#         try:
#             docs = faiss_index.similarity_search(q, k=per_q_k)
#             for d in docs:
#                 key = d.page_content[:120].strip()
#                 seen_weights[key] = max(seen_weights.get(key, 0), 0.6)
#                 chunk_map[key]    = d
#         except Exception as e:
#             print(f"[Hybrid] semantic error for '{q}': {e}")

#     print(f"[Hybrid] {len(seen_weights)} unique chunks after semantic")

#     if all_chunks:
#         try:
#             exact_docs = exact_match_retrieve(question, all_chunks)
#             for d in exact_docs:
#                 key = d.page_content[:120].strip()
#                 seen_weights[key] = max(seen_weights.get(key, 0), 0.9)
#                 chunk_map[key]    = d
#             if exact_docs:
#                 print(f"[Hybrid] {len(exact_docs)} exact-match chunks boosted to 0.9")
#         except Exception as e:
#             print(f"[Hybrid] exact match error: {e}")

#     if all_chunks:
#         try:
#             kw_docs = keyword_retrieve(question, all_chunks, k=8)
#             for d in kw_docs:
#                 key = d.page_content[:120].strip()
#                 seen_weights[key] = max(seen_weights.get(key, 0), 0.3)
#                 chunk_map[key]    = d
#         except Exception as e:
#             print(f"[Hybrid] keyword error: {e}")

#     print(f"[Hybrid] {len(seen_weights)} total unique chunks across all signals")

#     sorted_keys = sorted(seen_weights.keys(),
#                          key=lambda x: seen_weights[x],
#                          reverse=True)

#     final_docs = [chunk_map[key] for key in sorted_keys[:k]]

#     if len(final_docs) < k:
#         try:
#             seen_set = set(seen_weights.keys())
#             mmr_docs = faiss_index.max_marginal_relevance_search(
#                 question, k=k, fetch_k=k * 3)
#             for d in mmr_docs:
#                 key = d.page_content[:120].strip()
#                 if key not in seen_set:
#                     final_docs.append(d)
#                     seen_set.add(key)
#             print(f"[Hybrid] After MMR fallback: {len(final_docs)} chunks")
#         except Exception as e:
#             print(f"[Hybrid] MMR fallback error: {e}")

#     print(f"[Hybrid] Final pool: {len(final_docs)} chunks | "
#           f"top weight={seen_weights.get(sorted_keys[0], 0):.1f} "
#           f"if sorted_keys else 0")
#     return final_docs[:k]