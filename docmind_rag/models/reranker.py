from docmind_rag.config.settings import reranker, RERANKER_PRUNE_MARGIN, FACTUAL_TOP_K
from docmind_rag.services.retrieval import exact_match_retrieve
# ============================================================
# RERANK WITH RELATIVE PRUNING + TOP-K CAP
# Returns (docs, top_score, all_scores)
# ============================================================
def rerank_docs(question: str, docs: list,
                top_k: int = FACTUAL_TOP_K,
                apply_pruning: bool = True) -> tuple:
    if not docs:
        return docs, -99.0, []

    try:
        pairs      = [(question, d.page_content[:512]) for d in docs]
        scores_arr = reranker.predict(pairs)
        all_scores = [float(s) for s in scores_arr]
        ranked     = sorted(zip(all_scores, docs), key=lambda x: x[0], reverse=True)
        top_score  = float(ranked[0][0])
        # ============================================================
        # 🔥 HARD REJECTION — NO RELEVANT CHUNKS
        # ============================================================
        # if top_score < 0:
        #     print(f"[Reranker] ❌ All scores negative → no relevant chunks")
        #     return [], top_score, all_scores
        if top_score < -5:
            print(f"[Reranker] ❌ Strong negative score → no relevant chunks")
            return [], top_score, all_scores

        # if apply_pruning:
        #     margin      = RERANKER_PRUNE_MARGIN if top_score >= 2.0 else 1.0
        #     pruned      = [(s, d) for s, d in ranked if s >= top_score - margin]
        #     # NO hard cap here — margin controls size
        #     result_docs = [d for _, d in pruned]
        if apply_pruning:
            margin = RERANKER_PRUNE_MARGIN if top_score >= 2.0 else 1.0
            pruned = [(s, d) for s, d in ranked if s >= top_score - margin]

            # ============================================================
            # 🔥 FIX: PREVENT COLLAPSING TO TOO FEW CHUNKS
            # ============================================================
            MIN_CHUNKS = min(5, top_k)

            if len(pruned) < MIN_CHUNKS:
                print(f"[Reranker] ⚠️ Too few chunks after pruning ({len(pruned)}) → expanding")

                # fallback to top-N ranked instead of margin pruning
                result_docs = [d for _, d in ranked[:MIN_CHUNKS]]
            else:
                result_docs = [d for _, d in pruned]

            # print(f"[Reranker] {len(docs)} → {len(result_docs)} chunks after pruning | "
            #     f"top={top_score:.3f} | margin={margin}") gotta remove duplicate print
            # print(f"[Reranker] {len(docs)} → {len(result_docs)} chunks after pruning | "
            #       f"top={top_score:.3f} | margin={margin} | "
            #       f"bottom kept={pruned[-1][0]:.3f}")
            bottom_score = pruned[-1][0] if pruned else -999

            print(f"[Reranker] {len(docs)} → {len(result_docs)} chunks after pruning | "
                f"top={top_score:.3f} | margin={margin} | "
                f"bottom kept={bottom_score:.3f}")
        else:
            result_docs = [d for _, d in ranked[:top_k]]
            print(f"[Reranker] {len(docs)} → {len(result_docs)} chunks (no pruning) | "
                  f"top={top_score:.3f}")

        return result_docs, top_score, all_scores

    except Exception as e:
        print(f"[Reranker] error: {e} — returning original order")
        return docs[:top_k], -99.0, []


def protect_exact_matches(question: str, reranked_docs: list,
                          all_chunks: list,
                          top_k: int = FACTUAL_TOP_K) -> list:
    if not all_chunks:
        return reranked_docs

    exact_docs = exact_match_retrieve(question, all_chunks)
    if not exact_docs:
        return reranked_docs

    reranked_fps = set(d.page_content[:120].strip() for d in reranked_docs)
    missing = [d for d in exact_docs
               if d.page_content[:120].strip() not in reranked_fps]

    if not missing:
        print(f"[Protect] All exact-match chunks already in reranked set ✅")
        return reranked_docs

    print(f"[Protect] Injecting {len(missing)} exact-match chunks dropped by reranker")
    # protected = missing + reranked_docs changed to fix the aspects of AI issue 
#     Why
# Keeps reranker’s best chunks at top
# Prevents noise from dominating context
# Works for ANY document type
    protected = reranked_docs + missing
    expanded_k = max(top_k, len(missing) + len(reranked_docs))
    expanded_k = min(expanded_k, FACTUAL_TOP_K + len(missing)) # this is very imp just by changinng this from this expanded_k = min(expanded_k, 5) i got crct retreival  nd the chat is in my chrome
#     ✅ Verdict: CORRECT + NECESSARY

# 👉 Why this works:

# Before → hard cap = 5 ❌
# Now → dynamic cap based on missing chunks ✅

# 👉 This directly fixes your issue:

# Hinton chunk was getting cut off
    print(f"[Protect] Context expanded: {top_k} → {expanded_k} chunks")
    return protected[:expanded_k]






































# did not work gonnna work on making generic nd make it work - lap claude 
# 
# from docmind_rag.config.settings import reranker, RERANKER_PRUNE_MARGIN, FACTUAL_TOP_K
# from docmind_rag.services.retrieval import exact_match_retrieve
# # ============================================================
# # RERANK WITH RELATIVE PRUNING + TOP-K CAP
# # Returns (docs, top_score, all_scores)
# # ============================================================
# def rerank_docs(question: str, docs: list,
#                 top_k: int = FACTUAL_TOP_K,
#                 apply_pruning: bool = True) -> tuple:
#     if not docs:
#         return docs, -99.0, []

#     try:
#         pairs      = [(question, d.page_content[:512]) for d in docs]
#         scores_arr = reranker.predict(pairs)
#         all_scores = [float(s) for s in scores_arr]
#         ranked     = sorted(zip(all_scores, docs), key=lambda x: x[0], reverse=True)
#         top_score  = float(ranked[0][0])
#         # ============================================================
#         # 🔥 HARD REJECTION — NO RELEVANT CHUNKS
#         # ============================================================
#         # if top_score < 0:
#         #     print(f"[Reranker] ❌ All scores negative → no relevant chunks")
#         #     return [], top_score, all_scores
#         if top_score < -5:
#             print(f"[Reranker] ❌ Strong negative score → no relevant chunks")
#             return [], top_score, all_scores

#         # if apply_pruning:
#         #     margin      = RERANKER_PRUNE_MARGIN if top_score >= 2.0 else 1.0
#         #     pruned      = [(s, d) for s, d in ranked if s >= top_score - margin]
#         #     # NO hard cap here — margin controls size
#         #     result_docs = [d for _, d in pruned]
#         if apply_pruning:
#             margin = RERANKER_PRUNE_MARGIN if top_score >= 2.0 else 1.0
#             pruned = [(s, d) for s, d in ranked if s >= top_score - margin]

#             # ============================================================
#             # 🔥 FIX: PREVENT COLLAPSING TO TOO FEW CHUNKS
#             # ============================================================
#             MIN_CHUNKS = min(5, top_k)

#             if len(pruned) < MIN_CHUNKS:
#                 print(f"[Reranker] ⚠️ Too few chunks after pruning ({len(pruned)}) → expanding")

#                 # fallback to top-N ranked instead of margin pruning
#                 result_docs = [d for _, d in ranked[:MIN_CHUNKS]]
#             else:
#                 result_docs = [d for _, d in pruned]

#             # print(f"[Reranker] {len(docs)} → {len(result_docs)} chunks after pruning | "
#             #     f"top={top_score:.3f} | margin={margin}") gotta remove duplicate print
#             # print(f"[Reranker] {len(docs)} → {len(result_docs)} chunks after pruning | "
#             #       f"top={top_score:.3f} | margin={margin} | "
#             #       f"bottom kept={pruned[-1][0]:.3f}")
#             bottom_score = pruned[-1][0] if pruned else -999

#             print(f"[Reranker] {len(docs)} → {len(result_docs)} chunks after pruning | "
#                 f"top={top_score:.3f} | margin={margin} | "
#                 f"bottom kept={bottom_score:.3f}")
#         else:
#             result_docs = [d for _, d in ranked[:top_k]]
#             print(f"[Reranker] {len(docs)} → {len(result_docs)} chunks (no pruning) | "
#                   f"top={top_score:.3f}")

#         return result_docs, top_score, all_scores

#     except Exception as e:
#         print(f"[Reranker] error: {e} — returning original order")
#         return docs[:top_k], -99.0, []


# def protect_exact_matches(question: str, reranked_docs: list,
#                           all_chunks: list,
#                           top_k: int = FACTUAL_TOP_K) -> list:
#     if not all_chunks:
#         return reranked_docs

#     exact_docs = exact_match_retrieve(question, all_chunks)
#     if not exact_docs:
#         return reranked_docs

#     reranked_fps = set(d.page_content[:120].strip() for d in reranked_docs)
#     missing = [d for d in exact_docs
#                if d.page_content[:120].strip() not in reranked_fps]

#     if not missing:
#         print(f"[Protect] All exact-match chunks already in reranked set ✅")
#         return reranked_docs

#     print(f"[Protect] Injecting {len(missing)} exact-match chunks dropped by reranker")
#     # protected = missing + reranked_docs changed to fix the aspects of AI issue 
# #     Why
# # Keeps reranker’s best chunks at top
# # Prevents noise from dominating context
# # Works for ANY document type
#     protected = reranked_docs + missing
#     expanded_k = max(top_k, len(missing) + len(reranked_docs))
#     expanded_k = min(expanded_k, FACTUAL_TOP_K + len(missing)) # this is very imp just by changinng this from this expanded_k = min(expanded_k, 5) i got crct retreival  nd the chat is in my chrome
# #     ✅ Verdict: CORRECT + NECESSARY

# # 👉 Why this works:

# # Before → hard cap = 5 ❌
# # Now → dynamic cap based on missing chunks ✅

# # 👉 This directly fixes your issue:

# # Hinton chunk was getting cut off
#     print(f"[Protect] Context expanded: {top_k} → {expanded_k} chunks")
#     return protected[:expanded_k]
























# #f wworked 100  fro ai nd 57 for crop overfits 
# from docmind_rag.config.settings import reranker, RERANKER_PRUNE_MARGIN, FACTUAL_TOP_K
# from docmind_rag.services.retrieval import exact_match_retrieve
# # ============================================================
# # RERANK WITH RELATIVE PRUNING + TOP-K CAP
# # Returns (docs, top_score, all_scores)
# # ============================================================
# def rerank_docs(question: str, docs: list,
#                 top_k: int = FACTUAL_TOP_K,
#                 apply_pruning: bool = True) -> tuple:
#     if not docs:
#         return docs, -99.0, []

#     try:
#         pairs      = [(question, d.page_content[:512]) for d in docs]
#         scores_arr = reranker.predict(pairs)
#         all_scores = [float(s) for s in scores_arr]
#         ranked     = sorted(zip(all_scores, docs), key=lambda x: x[0], reverse=True)
#         top_score  = float(ranked[0][0])
#         # ============================================================
#         # 🔥 HARD REJECTION — NO RELEVANT CHUNKS
#         # ============================================================
#         # if top_score < 0:
#         #     print(f"[Reranker] ❌ All scores negative → no relevant chunks")
#         #     return [], top_score, all_scores
#         if top_score < -5:
#             print(f"[Reranker] ❌ Strong negative score → no relevant chunks")
#             return [], top_score, all_scores

#         if apply_pruning:
#             margin      = RERANKER_PRUNE_MARGIN if top_score >= 2.0 else 1.0
#             pruned      = [(s, d) for s, d in ranked if s >= top_score - margin]
#             # NO hard cap here — margin controls size
#             result_docs = [d for _, d in pruned]
#             print(f"[Reranker] {len(docs)} → {len(result_docs)} chunks after pruning | "
#                   f"top={top_score:.3f} | margin={margin} | "
#                   f"bottom kept={pruned[-1][0]:.3f}")
#         else:
#             result_docs = [d for _, d in ranked[:top_k]]
#             print(f"[Reranker] {len(docs)} → {len(result_docs)} chunks (no pruning) | "
#                   f"top={top_score:.3f}")

#         return result_docs, top_score, all_scores

#     except Exception as e:
#         print(f"[Reranker] error: {e} — returning original order")
#         return docs[:top_k], -99.0, []


# def protect_exact_matches(question: str, reranked_docs: list,
#                           all_chunks: list,
#                           top_k: int = FACTUAL_TOP_K) -> list:
#     if not all_chunks:
#         return reranked_docs

#     exact_docs = exact_match_retrieve(question, all_chunks)
#     if not exact_docs:
#         return reranked_docs

#     reranked_fps = set(d.page_content[:120].strip() for d in reranked_docs)
#     missing = [d for d in exact_docs
#                if d.page_content[:120].strip() not in reranked_fps]

#     if not missing:
#         print(f"[Protect] All exact-match chunks already in reranked set ✅")
#         return reranked_docs

#     print(f"[Protect] Injecting {len(missing)} exact-match chunks dropped by reranker")
#     # protected = missing + reranked_docs changed to fix the aspects of AI issue 
# #     Why
# # Keeps reranker’s best chunks at top
# # Prevents noise from dominating context
# # Works for ANY document type
#     protected = reranked_docs + missing
#     expanded_k = max(top_k, len(missing) + len(reranked_docs))
#     expanded_k = min(expanded_k, FACTUAL_TOP_K + len(missing)) # this is very imp just by changinng this from this expanded_k = min(expanded_k, 5) i got crct retreival  nd the chat is in my chrome
# #     ✅ Verdict: CORRECT + NECESSARY

# # 👉 Why this works:

# # Before → hard cap = 5 ❌
# # Now → dynamic cap based on missing chunks ✅

# # 👉 This directly fixes your issue:

# # Hinton chunk was getting cut off
#     print(f"[Protect] Context expanded: {top_k} → {expanded_k} chunks")
#     return protected[:expanded_k]







# # got 9 pass nd one partial 


# from docmind_rag.config.settings import reranker, RERANKER_PRUNE_MARGIN, FACTUAL_TOP_K
# from docmind_rag.services.retrieval import exact_match_retrieve
# # ============================================================
# # RERANK WITH RELATIVE PRUNING + TOP-K CAP
# # Returns (docs, top_score, all_scores)
# # ============================================================
# def rerank_docs(question: str, docs: list,
#                 top_k: int = FACTUAL_TOP_K,
#                 apply_pruning: bool = True) -> tuple:
#     if not docs:
#         return docs, -99.0, []

#     try:
#         pairs      = [(question, d.page_content[:512]) for d in docs]
#         scores_arr = reranker.predict(pairs)
#         all_scores = [float(s) for s in scores_arr]
#         ranked     = sorted(zip(all_scores, docs), key=lambda x: x[0], reverse=True)
#         top_score  = float(ranked[0][0])

#         if apply_pruning:
#             margin      = RERANKER_PRUNE_MARGIN if top_score >= 2.0 else 1.0
#             pruned      = [(s, d) for s, d in ranked if s >= top_score - margin]
#             # NO hard cap here — margin controls size
#             result_docs = [d for _, d in pruned]
#             print(f"[Reranker] {len(docs)} → {len(result_docs)} chunks after pruning | "
#                   f"top={top_score:.3f} | margin={margin} | "
#                   f"bottom kept={pruned[-1][0]:.3f}")
#         else:
#             result_docs = [d for _, d in ranked[:top_k]]
#             print(f"[Reranker] {len(docs)} → {len(result_docs)} chunks (no pruning) | "
#                   f"top={top_score:.3f}")

#         return result_docs, top_score, all_scores

#     except Exception as e:
#         print(f"[Reranker] error: {e} — returning original order")
#         return docs[:top_k], -99.0, []


# def protect_exact_matches(question: str, reranked_docs: list,
#                           all_chunks: list,
#                           top_k: int = FACTUAL_TOP_K) -> list:
#     if not all_chunks:
#         return reranked_docs

#     exact_docs = exact_match_retrieve(question, all_chunks)
#     if not exact_docs:
#         return reranked_docs

#     reranked_fps = set(d.page_content[:120].strip() for d in reranked_docs)
#     missing = [d for d in exact_docs
#                if d.page_content[:120].strip() not in reranked_fps]

#     if not missing:
#         print(f"[Protect] All exact-match chunks already in reranked set ✅")
#         return reranked_docs

#     print(f"[Protect] Injecting {len(missing)} exact-match chunks dropped by reranker")
#     protected = missing + reranked_docs 
#     expanded_k = max(top_k, len(missing) + len(reranked_docs))
#     expanded_k = min(expanded_k, FACTUAL_TOP_K + len(missing)) # this is very imp just by changinng this from this expanded_k = min(expanded_k, 5) i got crct retreival  nd the chat is in my chrome
# #     ✅ Verdict: CORRECT + NECESSARY

# # 👉 Why this works:

# # Before → hard cap = 5 ❌
# # Now → dynamic cap based on missing chunks ✅

# # 👉 This directly fixes your issue:

# # Hinton chunk was getting cut off
#     print(f"[Protect] Context expanded: {top_k} → {expanded_k} chunks")
#     return protected[:expanded_k]



















# # got issues because  of def Fix 1 — rerank_docs
# Remove the hard cap inside apply_pruning. Margin controls the size, not top_k.

# from docmind_rag.config.settings import reranker, RERANKER_PRUNE_MARGIN, FACTUAL_TOP_K
# from docmind_rag.services.retrieval import exact_match_retrieve
# # ============================================================
# # RERANK WITH RELATIVE PRUNING + TOP-K CAP
# # Returns (docs, top_score, all_scores)
# # ============================================================
# def rerank_docs(question: str, docs: list,
#                 top_k: int = FACTUAL_TOP_K,
#                 apply_pruning: bool = True) -> tuple:
#     print(f"[DEBUG] apply_pruning={apply_pruning}")
#     if not docs:
#         return docs, -99.0, []

#     try:
#         pairs      = [(question, d.page_content[:512]) for d in docs]
#         scores_arr = reranker.predict(pairs)
#         all_scores = [float(s) for s in scores_arr]
#         ranked     = sorted(zip(all_scores, docs), key=lambda x: x[0], reverse=True)
#         top_score  = float(ranked[0][0])

#         if apply_pruning:
#             margin = RERANKER_PRUNE_MARGIN if top_score >= 2.0 else 1.0
#             pruned = [(s, d) for s, d in ranked if s >= top_score - margin]
#             pruned = pruned[:top_k]
#             result_docs = [d for _, d in pruned]
#             print(f"[Reranker] {len(docs)} → {len(result_docs)} chunks after pruning | "
#                   f"top={top_score:.3f} | margin={margin} | "
#                   f"bottom kept={pruned[-1][0]:.3f}")
#         else:
#             result_docs = [d for _, d in ranked[:top_k]]
#             print(f"[Reranker] {len(docs)} → {len(result_docs)} chunks (no pruning) | "
#                   f"top={top_score:.3f} | bottom kept={ranked[top_k-1][0]:.3f}")

#         return result_docs, top_score, all_scores

#     except Exception as e:
#         print(f"[Reranker] error: {e} — returning original order")
#         return docs[:top_k], -99.0, []



# def protect_exact_matches(question: str, reranked_docs: list,
#                           all_chunks: list,
#                           top_k: int = FACTUAL_TOP_K) -> list:
#     if not all_chunks:
#         return reranked_docs

#     exact_docs = exact_match_retrieve(question, all_chunks)
#     if not exact_docs:
#         return reranked_docs

#     reranked_fps = set(d.page_content[:120].strip() for d in reranked_docs)
#     missing = [d for d in exact_docs
#                if d.page_content[:120].strip() not in reranked_fps]

#     if not missing:
#         print(f"[Protect] All exact-match chunks already in reranked set ✅")
#         return reranked_docs

#     print(f"[Protect] Injecting {len(missing)} exact-match chunks dropped by reranker")
#     protected = missing + reranked_docs
#     expanded_k = max(top_k, len(missing) + len(reranked_docs))
#     expanded_k = min(expanded_k, FACTUAL_TOP_K + len(missing)) # this is very imp just by changinng this from this expanded_k = min(expanded_k, 5) i got crct retreival  nd the chat is in my chrome
# #     ✅ Verdict: CORRECT + NECESSARY

# # 👉 Why this works:

# # Before → hard cap = 5 ❌
# # Now → dynamic cap based on missing chunks ✅

# # 👉 This directly fixes your issue:

# # Hinton chunk was getting cut off
#     print(f"[Protect] Context expanded: {top_k} → {expanded_k} chunks")
#     return protected[:expanded_k]

















# Problem 1 — Retrieval
# The chunk containing "Geoff Hinton" and "survival strategy" and "AI mothers" is being dropped by protect_exact_matches cap of 5. Fix is what we already identified — the single line in protect_exact_matches.

# from docmind_rag.config.settings import reranker, RERANKER_PRUNE_MARGIN, FACTUAL_TOP_K
# from docmind_rag.services.retrieval import exact_match_retrieve
# # ============================================================
# # RERANK WITH RELATIVE PRUNING + TOP-K CAP
# # Returns (docs, top_score, all_scores)
# # ============================================================
# def rerank_docs(question: str, docs: list,
#                 top_k: int = FACTUAL_TOP_K,
#                 apply_pruning: bool = True) -> tuple:
#     print(f"[DEBUG] apply_pruning={apply_pruning}")
#     if not docs:
#         return docs, -99.0, []

#     try:
#         pairs      = [(question, d.page_content[:512]) for d in docs]
#         scores_arr = reranker.predict(pairs)
#         all_scores = [float(s) for s in scores_arr]
#         ranked     = sorted(zip(all_scores, docs), key=lambda x: x[0], reverse=True)
#         top_score  = float(ranked[0][0])

#         if apply_pruning:
#             margin = RERANKER_PRUNE_MARGIN if top_score >= 2.0 else 1.0
#             pruned = [(s, d) for s, d in ranked if s >= top_score - margin]
#             pruned = pruned[:top_k]
#             result_docs = [d for _, d in pruned]
#             print(f"[Reranker] {len(docs)} → {len(result_docs)} chunks after pruning | "
#                   f"top={top_score:.3f} | margin={margin} | "
#                   f"bottom kept={pruned[-1][0]:.3f}")
#         else:
#             result_docs = [d for _, d in ranked[:top_k]]
#             print(f"[Reranker] {len(docs)} → {len(result_docs)} chunks (no pruning) | "
#                   f"top={top_score:.3f} | bottom kept={ranked[top_k-1][0]:.3f}")

#         return result_docs, top_score, all_scores

#     except Exception as e:
#         print(f"[Reranker] error: {e} — returning original order")
#         return docs[:top_k], -99.0, []



# def protect_exact_matches(question: str, reranked_docs: list,
#                           all_chunks: list,
#                           top_k: int = FACTUAL_TOP_K) -> list:
#     if not all_chunks:
#         return reranked_docs

#     exact_docs = exact_match_retrieve(question, all_chunks)
#     if not exact_docs:
#         return reranked_docs

#     reranked_fps = set(d.page_content[:120].strip() for d in reranked_docs)
#     missing = [d for d in exact_docs
#                if d.page_content[:120].strip() not in reranked_fps]

#     if not missing:
#         print(f"[Protect] All exact-match chunks already in reranked set ✅")
#         return reranked_docs

#     print(f"[Protect] Injecting {len(missing)} exact-match chunks dropped by reranker")
#     protected = missing + reranked_docs
#     expanded_k = max(top_k, len(missing) + len(reranked_docs))
#     expanded_k = min(expanded_k, 5)
#     print(f"[Protect] Context expanded: {top_k} → {expanded_k} chunks")
#     return protected[:expanded_k]