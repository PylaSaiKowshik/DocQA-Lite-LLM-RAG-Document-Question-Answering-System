
import time
import re
import spacy
_nlp = spacy.load("en_core_web_lg")

from docmind_rag.config.settings import (
    MAX_WORKERS, FACTUAL_TOP_K, MULTIPART_TOP_K
)
from docmind_rag.utils.text import extract_named_entities, expand_answer
from docmind_rag.core.state import DocState
from docmind_rag.core.prompts import QA_PROMPT
from docmind_rag.models.llm import call_llama_streaming
from docmind_rag.models.embeddings import build_faiss_index
from docmind_rag.models.reranker import rerank_docs, protect_exact_matches
from docmind_rag.services.extraction import extract_pdf_parallel, _extraction_cache
from docmind_rag.services.chunking import semantic_chunk, raptor_summarize, _summary_cache
from docmind_rag.services.retrieval import multi_query_retrieve
from docmind_rag.services.agent import react_agent, roberta_qa, clean_reasoning_answer
from docmind_rag.events.events import emit_event
from docmind_rag.utils.helpers import get_pdf_hash, clean_artifacts, clean_chunk_text, normalize_answer
from docmind_rag.utils.text import (
    classify_from_context,
    reorder_by_question,
    normalize_text,
)
from docmind_rag.utils.metrics import (
    compute_answer_grounding,
    compute_retrieval_score,
    compute_context_precision,
    compute_recall_at_k,
    semantic_similarity,
   
)

def clean_context_for_llm(chunks):
    cleaned_chunks = []
    for chunk in chunks:
        lines = chunk.split("\n")
        filtered_lines = []
        for line in lines:
            line_strip = line.strip()
            is_question_like = (
                line_strip.endswith("?") or
                (len(line_strip.split()) < 15 and "?" in line_strip)
            )
            if is_question_like:
                continue
            filtered_lines.append(line)
        cleaned_chunks.append("\n".join(filtered_lines))
    return cleaned_chunks


def is_grounded(answer, retrieved_texts, threshold=0.6):
    if not answer:
        return False
    answer_words = set(answer.lower().split())
    if not answer_words:
        return False
    best_overlap = 0
    for chunk in retrieved_texts:
        chunk_words = set(chunk.lower().split())
        overlap = len(answer_words & chunk_words) / len(answer_words)
        best_overlap = max(best_overlap, overlap)
    return best_overlap >= threshold


# ============================================================
# NUMERIC INTENT DETECTION
# ============================================================

def _extract_numbers(text: str) -> list:
    return re.findall(r'\b\d+\b', text)


def _detect_numeric_intent(question: str, chunks: list) -> str:
    """
    Detect NAVIGATIONAL or POSITIONAL from chunk structure.
    Generic — no hardcoded section words.
    """
    numbers = _extract_numbers(question)
    if not numbers:
        return "NONE"

    q_words = set(question.lower().split())

    for chunk in chunks:
        lines = [l.strip() for l in chunk.split("\n") if l.strip()]
        if not lines:
            continue
        # chunks may have no newlines — take first 10 tokens as heading proxy
        first_line = " ".join(lines[0].split()[:10]).lower()
        line_words = set(first_line.split())
        overlap    = len(q_words & line_words)
        num_found  = any(re.search(rf'\b{re.escape(num)}\b', first_line) for num in numbers)

        if num_found:
            print(f"[NavDebug] first_line='{first_line}' | overlap={overlap} | words={len(first_line.split())}")

        if num_found and overlap >= 2:  # ← remove word count check entirely
            return "NAVIGATIONAL"
    for chunk in chunks:
        lines       = [l.strip() for l in chunk.split("\n") if l.strip()]
        short_lines = [l for l in lines if len(l.split()) <= 12]
        if len(short_lines) >= 3:
            first = lines[0].lower() if lines else ""
            if not any(re.search(rf'\b{re.escape(num)}\b', first) for num in numbers):
                return "POSITIONAL"

    return "NONE"


def _navigate_full_chunks(question: str, all_raw_chunks: list) -> str:
    numbers = _extract_numbers(question)
    if not numbers:
        return ""

    q_words    = set(question.lower().split())
    candidates = []

    for chunk in all_raw_chunks:
        lines = [l.strip() for l in chunk.split("\n") if l.strip()]
        if not lines:
            continue

        first_line  = " ".join(lines[0].split()[:10])
        first_lower = first_line.lower()

        for num in numbers:
            if not re.search(rf'\b{re.escape(num)}\b', first_lower):
                continue

            line_words = set(first_lower.split())
            overlap    = len(q_words & line_words)

            if overlap >= 2:
                # generic scoring (NO hardcoding)
                has_number = num in first_lower
                score = overlap + (2 if has_number else 0)

                candidates.append((first_line, score))

    if not candidates:
        return ""

    # pick BEST candidate (not first)
    candidates.sort(key=lambda x: x[1], reverse=True)

    best_line, best_score = candidates[0]

    # 🔥 RELATIVE CONFIDENCE (NO HARDCODING)
    if len(candidates) > 1:
        second_score = candidates[1][1]
    else:
        second_score = 0

    # Reject if not clearly better
    if best_score <= second_score:
        print("[NAV] ❌ No clear winner → fallback")
        return ""

    # Optional: also reject very weak absolute matches
    if best_score < 3:
        print("[NAV] ⚠️ Weak match → fallback")
        return ""

def _positional_extract(question: str, retrieved_texts: list) -> str:
    """Extract Nth item from a list. Generic."""
    numbers = _extract_numbers(question)
    if not numbers:
        return ""
    try:
        idx = int(numbers[0]) - 1
    except ValueError:
        return ""
    if idx < 0:
        return ""

    best_chunk, best_count = "", 0
    for chunk in retrieved_texts:
        lines       = [l.strip() for l in chunk.split("\n") if l.strip()]
        short_lines = [l for l in lines if len(l.split()) <= 12]
        if len(short_lines) > best_count:
            best_count = len(short_lines)
            best_chunk = chunk

    if not best_chunk:
        return ""

    lines       = [l.strip() for l in best_chunk.split("\n") if l.strip()]
    short_lines = [l for l in lines if len(l.split()) <= 12]
    if idx < len(short_lines):
        return re.sub(r'^\s*\d+[\.\)]\s*', '', short_lines[idx]).strip()
    return ""


# ============================================================
# SIGNAL 3 — Structural consistency for NAVIGATIONAL answers
# ============================================================

def _check_identifier_grounded(answer: str, retrieved_texts: list) -> bool:
    """
    If answer contains a multi-word identifier (word+number or number+word),
    verify it appears verbatim in retrieved chunks.
    Generic — no hardcoded section words.
    """
    identifiers = re.findall(
        r'\b[a-zA-Z]+\s+\d+\b|\b\d+\s+[a-zA-Z]+\b',
        answer.lower()
    )
    if not identifiers:
        return True

    chunk_text = " ".join(retrieved_texts).lower()
    for ident in identifiers:
        if ident not in chunk_text:
            print(f"[Signal3] ❌ Identifier '{ident}' not in retrieved chunks")
            return False
    return True


# ============================================================
# REFUSAL — semantic version (local)
# ============================================================

_REFUSAL_ANCHOR = "this information is not available in the provided context"


def _is_refusal_semantic(text: str) -> bool:
    if not text or len(text.strip()) < 2:
        return True
    sim = semantic_similarity(text, [_REFUSAL_ANCHOR])
    print(f"[Refusal] sim={sim:.3f} | '{text[:50]}'")
    return sim > 0.72


# ============================================================
# LANGGRAPH NODES
# ============================================================

def node_extract(state: DocState) -> DocState:
    extract_start    = time.time()
    text, page_count = extract_pdf_parallel(state["pdf_path"])
    extract_time     = time.time() - extract_start
    state["extracted_text"] = text
    state["page_count"]     = page_count
    state["char_count"]     = len(text)
    state["metrics"]["extraction_time_sec"]  = round(extract_time, 2)
    state["metrics"]["pages_processed"]      = page_count
    state["metrics"]["characters_processed"] = len(text)
    state["metrics"]["words_processed"]      = len(text.split())
    return state


def node_chunk(state: DocState) -> DocState:
    text = state["extracted_text"]
    state["query_type"] = "QA"
    question = state.get("question", "").strip().lower()
    summary_pattern = r'^(summarize|summarise|summary|give|provide|generate|write|create)\b'
    if re.match(summary_pattern, question) or (len(question.split()) > 12 and "?" not in question):
        state["query_type"] = "FULL_SUMMARY"

    summary_chunks, rag_chunks = semantic_chunk(text)

    # FIX 4 — normalize all chunks after chunking
    # Converts unicode symbols so retrieval doesn't fail on symbol mismatches
    rag_chunks     = [normalize_text(c) for c in rag_chunks]
    summary_chunks = [normalize_text(c) for c in summary_chunks]

    state["summary_chunks"] = summary_chunks
    state["chunks"]         = rag_chunks

    state["metrics"]["summary_chunks"] = len(summary_chunks)
    state["metrics"]["chunks_created"] = len(rag_chunks)
    # state["metrics"]["doc_type"]       = state.get("doc_type", "general")
    state["metrics"]["query_type"]     = state["query_type"]
    print(f"[Chunk] {len(summary_chunks)} summary chunks | {len(rag_chunks)} RAG chunks")
    return state




def node_summarize(state: DocState) -> DocState:
    pdf_hash  = get_pdf_hash(state["pdf_path"])
    cache_key = pdf_hash  # ❌ removed doc_type dependency

    # ── Cache check ──────────────────────────────────────────
    if cache_key in _summary_cache:
        cached = _summary_cache[cache_key]
        print("[Summary] ✅ Cache hit")
        emit_event(
            state.get("request_id", ""),
            "agent_action",
            "⚡ Summary loaded from cache instantly!"
        )
        state["answer"] = cached["summary"]
        state["metrics"].update(cached["metrics"])
        state["metrics"]["type"] = "summary"
        return state

    # ── Generate summary ─────────────────────────────────────
    summary_start = time.time()
    raptor_summarize._request_id = state.get("request_id", "")

    # ❌ removed doc_type argument
    summary, map_time, reduce_time = raptor_summarize(
        state["summary_chunks"],   # ← was state["chunks"], must be summary_chunks
        state.get("doc_type", "general")
    )
    summary_time = time.time() - summary_start

    # ── Store result ─────────────────────────────────────────
    state["answer"] = summary

    metrics_snapshot = {
        "summary_time_sec":     round(summary_time, 2),
        "summary_length_words": len(summary.split()),
        "parallel_workers":     min(MAX_WORKERS, len(state["summary_chunks"])),
        "map_time_sec":         round(map_time, 2),
        "reduce_time_sec":      round(reduce_time, 2),
        "llm_calls":            3,
    }

    state["metrics"].update(metrics_snapshot)
    state["metrics"]["type"] = "summary"

    # ── Cache result ─────────────────────────────────────────
    _summary_cache[cache_key] = {
        "summary": summary,
        "metrics": metrics_snapshot
    }

    print(f"[Summary] Done ({len(summary.split())} words)")
    return state

def node_qa(state: DocState) -> DocState:
    qa_start_t = time.time()
    question   = state["question"]
    request_id = state.get("request_id", "")
    all_chunks = state["chunks"]

    # FIX 4 — normalize question the same way chunks were normalized
    question = normalize_text(question)
    numeric_intent = "NONE"   # 🔥 DEFAULT FIX (MANDATORY)
    # Raw strings for navigational scanning — must NOT be cleaned
    all_raw = [
        d if isinstance(d, str) else d.page_content
        for d in all_chunks
    ]

    pdf_hash    = get_pdf_hash(state["pdf_path"])
    faiss_index = build_faiss_index(all_chunks, pdf_hash)

    print(f"[QA] START | {len(all_chunks)} chunks")

    recall_score      = 0.0
    retrieval_score   = 0.0
    context_precision = 0.0
    grounding         = 0.0
    llm_calls         = 0
    model_used        = "llama"
    confidence        = 0.0
    decision_type     = "accepted"
    retrieved         = []
    retrieved_texts   = []
    answer            = ""

    # ── STEP 1: RETRIEVE ─────────────────────────────────────
    retrieved = multi_query_retrieve(
        question, faiss_index,
        k=50,
        all_chunks=all_chunks,
        query_type="FACTUAL_QA"
    )


   
    # 🔥 STEP 2 — RERANK + SAFE FALLBACK

    retrieved, reranker_top, _ = rerank_docs(
        question, retrieved, top_k=8, apply_pruning=True
    )

    retrieved = protect_exact_matches(
        question, retrieved, all_chunks, top_k=8
    )

    # 🔥 FIX 1 — fallback must replace retrieved (NOT just texts)
    if not retrieved or len(retrieved) < 3:
        print("[QA] ⚠️ Reranker too aggressive → fallback to initial chunks")
        retrieved = all_chunks[:8]

    # 🔥 FIX 2 — safe conversion
    retrieved_texts = [
        d.page_content if hasattr(d, "page_content") else str(d)
        for d in retrieved
    ]
    if reranker_top < -5:
        print("[Guard] ⚠️ Weak reranker score — continuing")

    # ── STEP 3: METRICS ───────────────────────────────────────
    retrieval_score   = compute_retrieval_score(question, retrieved)
    context_precision = compute_context_precision(question, retrieved)
    recall_score      = compute_recall_at_k(
        question, retrieved, all_chunks, k=len(retrieved)
    )

    # ── FIX 6 — k expansion: low recall OR numeric question ───
    is_numeric_question = len(_extract_numbers(question)) > 0
    if recall_score < 25 or is_numeric_question:
        print("[QA] ⚠️ Expanding retrieval (low recall or numeric question)")
        expanded = multi_query_retrieve(
            question, faiss_index, k=30,
            all_chunks=all_chunks, query_type="FACTUAL_QA"
        )
        if len(expanded) > len(retrieved):
            retrieved = expanded
            retrieved_texts = [
                d.page_content if hasattr(d, "page_content") else str(d)
                for d in retrieved
            ]
        # 🔥 ENSURE numeric_intent ALWAYS DEFINED
        numeric_intent = _detect_numeric_intent(question, retrieved_texts)

        # fallback check (keep this)
        is_numeric_question = len(_extract_numbers(question)) > 0

        if numeric_intent == "NONE" and is_numeric_question:
            print("[Navigate] Retrying intent detection on full document...")
            numeric_intent = _detect_numeric_intent(question, all_raw)
    # ── FIX 1 — Numeric intent: retrieved first, fallback to all_raw ──
    if numeric_intent == "NAVIGATIONAL":
        print("[Routing] Numeric intent → NAVIGATIONAL")

        title = _navigate_full_chunks(question, all_raw)

        if title:
            print(f"[Navigate] Extracted title: '{title}'")

            grounding  = compute_answer_grounding(title, retrieved_texts, question)
            confidence = round(grounding / 100, 3)
            qa_time    = time.time() - qa_start_t

            state["answer"]     = title
            state["query_type"] = "NAVIGATIONAL"

            _write_metrics(
                state,
                "navigational",
                "navigational",
                grounding,
                confidence,
                retrieval_score,
                context_precision,
                recall_score,
                llm_calls,
                retrieved,
                qa_time
            )

            return state

        print("[Navigate] Falling back to QA")
    # ── Normal routing ────────────────────────────────────────
    query_type = classify_from_context(question, retrieved_texts)
    state["query_type"] = query_type
    print(f"[Routing] Context-based → {query_type}")

    # ── STEP 4: ANSWER GENERATION ─────────────────────────────

    if query_type == "FULL_SUMMARY":
        retrieved_texts = clean_context_for_llm(retrieved_texts)
        ranked  = reorder_by_question(question, retrieved_texts)
        context = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:8])
        structured_context = "\n".join(
            f"[CONTEXT CHUNK]\n{c}" for c in context.split("\n\n---\n\n")
        )
        emit_event(request_id, "stream_start", "✍️ Generating answer...")
        answer, _ = call_llama_streaming(
            QA_PROMPT.format(context=structured_context[:2500], question=question),
            request_id=request_id, temperature=0.0
        )
        answer     = clean_artifacts(answer).strip()
        model_used = "llama_summary"
        llm_calls  = 1

    elif query_type == "MULTIPART_QA":
        retrieved_texts = clean_context_for_llm(retrieved_texts)
        ranked  = reorder_by_question(question, retrieved_texts)
        # FIX 3 — 4000 chars for MULTIPART to capture all list items
        context = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:8])
        emit_event(request_id, "agent_start",
                   f"🤖 MULTIPART | {len(retrieved)} chunks")
        emit_event(request_id, "stream_start", "✍️ Generating answer...")
        answer, _ = call_llama_streaming(
            QA_PROMPT.format(context=context[:4000], question=question),
            request_id=request_id, temperature=0.0
        )
        answer     = clean_artifacts(answer).strip()
        model_used = "llama_multipart"
        llm_calls  = 1

    else:
        
        # FACTUAL_QA / VERIFICATION_QA

        if not retrieved_texts:
            print("[QA] ❌ No context → NOT FOUND")
            state["answer"] = "This information is not present in the document."
            _write_metrics(state, "not_found", "no_context",
                        0.0, 0.0, retrieval_score, context_precision,
                        recall_score, llm_calls, retrieved,
                        time.time() - qa_start_t)
            return state

        # 🔥 FIX 5 — SIMPLE QA ROUTING (ADD HERE)

        context = "\n".join(retrieved_texts)

        if (
            query_type == "FACTUAL_QA"
            and len(question.split()) <= 10
            and not is_numeric_question
            and recall_score >= 60
        ):
            print("[QA] ⚡ Direct QA (no ReAct)")

            react_ans, _ = call_llama_streaming(
                f"""
            Answer the question using ONLY the given context.
            Return ONLY the exact answer phrase from the context.
            Do NOT explain. Do NOT say 'not found' unless absolutely missing.

            Context:
            {context[:2000]}

            Question: {question}

            Answer:
            """,
                request_id=request_id,
                temperature=0.0
            )
            llm_calls += 1
            model_used = "llama_direct"

        else:
            react_ans, model_used, _, _, _, _, react_calls, _ = react_agent(
    question,
    faiss_index,
    query_type,
    all_chunks,
    request_id,
    recall_score
)
 
        react_ans = clean_artifacts(react_ans).strip().strip('"').strip("'")
        # react_ans = clean_reasoning_answer(react_ans, question)
        print(f"[QA] Answer: '{react_ans[:60]}'")

        # FIX 2 — Short answer bypass: skip for VERIFICATION_QA
        words         = react_ans.split()
        content_words = [w for w in words if len(w) > 2 and not w.isdigit()]

     
        print(f"[DEBUG] react_ans before refusal check: repr={repr(react_ans)}")

        refusal_phrases = [
            "not present",
            "not mentioned",
            "not available",
            "no information",
            "cannot find",
            "not in the document"
        ]

        if any(p in react_ans.lower() for p in refusal_phrases):
            print("[QA] ⚠️ Refusal detected")
            qa_time = time.time() - qa_start_t
            state["answer"] = "This information is not present in the document."
            _write_metrics(state, "not_found", "not_found",
                        75.0, 0.75, retrieval_score, context_precision,
                        recall_score, llm_calls, retrieved, qa_time)
            return state
     
        
        grounding_score = compute_answer_grounding(react_ans, retrieved_texts, question)

        # ✅ DEFAULT (VERY IMPORTANT)
        answer = react_ans
        decision_type = "accepted"

        # 🔥 FINAL HALLUCINATION GUARD
        if query_type == "FACTUAL_QA":
            if grounding_score < 50:
                print("[QA] ⚠️ Weak structural answer → NOT FOUND")
                answer = "This information is not present in the document."
                decision_type = "not_found"

        # 🔥 VERIFICATION LOGIC
        elif query_type == "VERIFICATION_QA":

            ans = react_ans.strip().lower()

            if ans in ["t/f", "true/false"]:
                answer = "Yes"

            elif ans in ["false", "no"]:
                answer = "No"

            elif len(ans.split()) <= 3:
                answer = react_ans

            elif grounding_score < 80:
                answer = "This information is not present in the document."
                decision_type = "verification_failed"

            else:
                answer = react_ans
                decision_type = "accepted"

            grounding  = compute_answer_grounding(answer, retrieved_texts, question)
            confidence = round(grounding / 100, 3)
            qa_time    = time.time() - qa_start_t
     
   
            state["answer"] = normalize_answer(answer)

            _write_metrics(
                state,
                model_used,
                decision_type,
                grounding,
                confidence,
                retrieval_score,
                context_precision,
                recall_score,
                llm_calls,
                retrieved,
                qa_time
            )

            return state
        # 🔥 TRUST GATE (FINAL CLEAN VERSION)

        recall = recall_score
        grounding = grounding_score
        length = len(react_ans.split())

        # 1. No reliable context → reject
        if recall < 25:
            print("[QA] ❌ Low recall → NOT FOUND")
            answer = "This information is not present in the document."
            decision_type = "low_recall"

        # 2. Very weak grounding → reject
        elif grounding < 40:
            print("[QA] ❌ Very low grounding → NOT FOUND")
            answer = "This information is not present in the document."
            decision_type = "low_grounding"

      
        elif length <= 3:
            if grounding >= 60:
                answer = react_ans
                decision_type = "accepted"
            else:
                print("[QA] ❌ Weak short answer → NOT FOUND")
                answer = "This information is not present in the document."
                decision_type = "weak_short"

        # 4. Strong signals → accept
        elif recall >= 50 and grounding >= 60:
            answer = react_ans
            decision_type = "accepted"

        # 5. Otherwise reject safely
        else:
            print("[QA] ❌ Uncertain → NOT FOUND")
            answer = "This information is not present in the document."
            decision_type = "uncertain"
    # ── STEP 5: METRICS ───────────────────────────────────────
    grounding  = compute_answer_grounding(answer, retrieved_texts, question)
    confidence = round(grounding / 100, 3)
    qa_time    = time.time() - qa_start_t

    print(f"[QA] Done in {qa_time:.1f}s | model={model_used} | "
          f"grounding={grounding:.1f}% | recall={recall_score:.1f}% | "
          f"confidence={confidence:.3f} | decision={decision_type}")

    if not answer.strip():
        answer = "Could not find a relevant answer in the PDF."
    else:
        answer = normalize_answer(answer)

    state["answer"] = answer
    _write_metrics(state, model_used, decision_type, grounding,
                   confidence, retrieval_score, context_precision,
                   recall_score, llm_calls, retrieved, qa_time)
    return state
def node_validate(state: DocState) -> DocState:
    answer = state["answer"]
    retry  = state.get("retry_count", 0)

    # ── Retry for empty/very weak answers ────────────────────
    if len(answer.strip()) < 3 and retry < 2:
        state["retry_count"] = retry + 1
        state["answer"]      = ""
        return state

    total_time    = time.time() - state["start_time"]
    output_words  = len(answer.split())
    output_tokens = output_words * 1.3

    extract_time  = state["metrics"].get("extraction_time_sec", 0)
    llm_time      = max(total_time - extract_time, 1)
    tps           = round(output_tokens / llm_time, 2) if llm_time > 0 else 0

    m = state["metrics"]

    # ── Core metrics ─────────────────────────────────────────
    m["response_time_sec"]    = round(total_time, 2)
    m["extraction_time_sec"]  = m.get("extraction_time_sec", 0)
    m["pages_processed"]      = state.get("page_count", 0)
    m["characters_processed"] = state.get("char_count", 0)
    m["words_processed"]      = len(state.get("extracted_text", "").split())

    # ── Type-specific metrics ────────────────────────────────
    if m.get("type") == "summary":
        m["summary_time_sec"]     = m.get("summary_time_sec", 0)
        m["summary_length_words"] = len(answer.split())

    if m.get("type") == "qa":
        m["qa_time_sec"]      = m.get("qa_time_sec", 0)
        m["confidence_score"] = m.get("confidence_score", 0)

    # ── Performance ──────────────────────────────────────────
    m["ttft_sec"]        = round(total_time, 2)
    m["e2e_latency_sec"] = round(total_time, 2)
    m["tps"]             = tps

    # ❌ REMOVED doc_type (no longer used anywhere)
    # m["doc_type"] = state.get("doc_type", "general")

    # ── Context info ─────────────────────────────────────────
    m["query_type"]     = state.get("query_type", "")
    m["chunks_created"] = m.get("chunks_created", 0)
    m["retry_count"]    = retry

    # ── Model + retrieval metrics ────────────────────────────
    m["model_used"]        = m.get("model_used", "llama_react")
    m["llm_calls"]         = m.get("llm_calls", 0)
    m["retrieval_score"]   = m.get("retrieval_score", 0)
    m["context_precision"] = m.get("context_precision", 0)
    m["answer_grounding"]  = m.get("answer_grounding", 0)
    m["recall_at_k"]       = m.get("recall_at_k", 0)

    # ── Summary-specific ─────────────────────────────────────
    if m.get("type") == "summary":
        m["parallel_workers"] = m.get("parallel_workers", 0)
        m["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
        m["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)

    # ── QA-specific ──────────────────────────────────────────
    if m.get("type") == "qa":
        m["chunks_retrieved"] = m.get("chunks_retrieved", 0)
        m["decision_type"]    = m.get("decision_type", "accepted")
        m["confidence_raw"]   = m.get("confidence_raw", 0.0)

    state["metrics"] = m
    return state




def _write_metrics(state, model_used, decision_type, grounding,
                   confidence, retrieval_score, context_precision,
                   recall_score, llm_calls, retrieved, qa_time):
    m = state.setdefault("metrics", {})
    m["qa_time_sec"]       = round(qa_time, 2)
    m["confidence_score"]  = round(confidence * 100, 2)
    m["retrieval_score"]   = retrieval_score
    m["context_precision"] = context_precision
    m["answer_grounding"]  = grounding
    m["recall_at_k"]       = recall_score
    m["llm_calls"]         = llm_calls
    m["model_used"]        = model_used
    m["chunks_retrieved"]  = len(retrieved)
    m["type"]              = "qa"
    m["decision_type"]     = decision_type
    m["confidence_raw"]    = round(confidence, 4)



























































































# # #90,90,50 works for single  pdf 
# import time
# import re
# import spacy
# _nlp = spacy.load("en_core_web_lg")

# from docmind_rag.config.settings import (
#     MAX_WORKERS, FACTUAL_TOP_K, MULTIPART_TOP_K
# )
# from docmind_rag.utils.text import extract_named_entities, expand_answer
# from docmind_rag.core.state import DocState
# from docmind_rag.core.prompts import QA_PROMPT
# from docmind_rag.models.llm import call_llama_streaming
# from docmind_rag.models.embeddings import build_faiss_index
# from docmind_rag.models.reranker import rerank_docs, protect_exact_matches
# from docmind_rag.services.extraction import extract_pdf_parallel, _extraction_cache
# from docmind_rag.services.chunking import semantic_chunk, raptor_summarize, _summary_cache
# from docmind_rag.services.retrieval import multi_query_retrieve
# from docmind_rag.services.agent import react_agent, roberta_qa, clean_reasoning_answer
# from docmind_rag.events.events import emit_event
# from docmind_rag.utils.helpers import get_pdf_hash, clean_artifacts, clean_chunk_text, normalize_answer
# from docmind_rag.utils.text import (
#     classify_from_context,
#     reorder_by_question,
#     normalize_text,
# )
# from docmind_rag.utils.metrics import (
#     compute_answer_grounding,
#     compute_retrieval_score,
#     compute_context_precision,
#     compute_recall_at_k,
#     semantic_similarity,
   
# )

# # def trust_gate(answer: str, metrics: dict) -> str:
# #     recall = metrics.get("recall_at_k", 0)
# #     grounding = metrics.get("answer_grounding", 0)
# #     length = len(answer.split())

# #     # 1. No reliable context → reject
# #     if recall < 25:
# #         return "not_found"

# #     # 2. Weak grounding → reject
# #     if grounding < 50:
# #         return "not_found"

# #     # 3. Short answers must be very strong
# #     if length <= 3:
# #         if recall >= 70 and grounding >= 70:
# #             return "accepted"
# #         return "not_found"

# #     # 4. Strong signal → accept
# #     if recall >= 50 and grounding >= 60:
# #         return "accepted"
   
# #     # 5. Default safe behavior
# #     else:
# #         return "not_found"
# # ============================================================
# # LOCAL HELPERS
# # ============================================================

# def clean_context_for_llm(chunks):
#     cleaned_chunks = []
#     for chunk in chunks:
#         lines = chunk.split("\n")
#         filtered_lines = []
#         for line in lines:
#             line_strip = line.strip()
#             is_question_like = (
#                 line_strip.endswith("?") or
#                 (len(line_strip.split()) < 15 and "?" in line_strip)
#             )
#             if is_question_like:
#                 continue
#             filtered_lines.append(line)
#         cleaned_chunks.append("\n".join(filtered_lines))
#     return cleaned_chunks


# def is_grounded(answer, retrieved_texts, threshold=0.6):
#     if not answer:
#         return False
#     answer_words = set(answer.lower().split())
#     if not answer_words:
#         return False
#     best_overlap = 0
#     for chunk in retrieved_texts:
#         chunk_words = set(chunk.lower().split())
#         overlap = len(answer_words & chunk_words) / len(answer_words)
#         best_overlap = max(best_overlap, overlap)
#     return best_overlap >= threshold


# # ============================================================
# # NUMERIC INTENT DETECTION
# # ============================================================

# def _extract_numbers(text: str) -> list:
#     return re.findall(r'\b\d+\b', text)


# def _detect_numeric_intent(question: str, chunks: list) -> str:
#     """
#     Detect NAVIGATIONAL or POSITIONAL from chunk structure.
#     Generic — no hardcoded section words.
#     """
#     numbers = _extract_numbers(question)
#     if not numbers:
#         return "NONE"

#     q_words = set(question.lower().split())

#     for chunk in chunks:
#         lines = [l.strip() for l in chunk.split("\n") if l.strip()]
#         if not lines:
#             continue
#         # chunks may have no newlines — take first 10 tokens as heading proxy
#         first_line = " ".join(lines[0].split()[:10]).lower()
#         line_words = set(first_line.split())
#         overlap    = len(q_words & line_words)
#         num_found  = any(re.search(rf'\b{re.escape(num)}\b', first_line) for num in numbers)

#         if num_found:
#             print(f"[NavDebug] first_line='{first_line}' | overlap={overlap} | words={len(first_line.split())}")

#         if num_found and overlap >= 2:  # ← remove word count check entirely
#             return "NAVIGATIONAL"
#     for chunk in chunks:
#         lines       = [l.strip() for l in chunk.split("\n") if l.strip()]
#         short_lines = [l for l in lines if len(l.split()) <= 12]
#         if len(short_lines) >= 3:
#             first = lines[0].lower() if lines else ""
#             if not any(re.search(rf'\b{re.escape(num)}\b', first) for num in numbers):
#                 return "POSITIONAL"

#     return "NONE"


# def _navigate_full_chunks(question: str, all_raw_chunks: list) -> str:
#     numbers = _extract_numbers(question)
#     if not numbers:
#         return ""

#     q_words    = set(question.lower().split())
#     candidates = []

#     for chunk in all_raw_chunks:
#         lines = [l.strip() for l in chunk.split("\n") if l.strip()]
#         if not lines:
#             continue

#         first_line  = " ".join(lines[0].split()[:10])
#         first_lower = first_line.lower()

#         for num in numbers:
#             if not re.search(rf'\b{re.escape(num)}\b', first_lower):
#                 continue

#             line_words = set(first_lower.split())
#             overlap    = len(q_words & line_words)

#             if overlap >= 2:
#                 # generic scoring (NO hardcoding)
#                 has_number = num in first_lower
#                 score = overlap + (2 if has_number else 0)

#                 candidates.append((first_line, score))

#     if not candidates:
#         return ""

#     # pick BEST candidate (not first)
#     candidates.sort(key=lambda x: x[1], reverse=True)

#     best_line, best_score = candidates[0]

#     # 🔥 RELATIVE CONFIDENCE (NO HARDCODING)
#     if len(candidates) > 1:
#         second_score = candidates[1][1]
#     else:
#         second_score = 0

#     # Reject if not clearly better
#     if best_score <= second_score:
#         print("[NAV] ❌ No clear winner → fallback")
#         return ""

#     # Optional: also reject very weak absolute matches
#     if best_score < 3:
#         print("[NAV] ⚠️ Weak match → fallback")
#         return ""

# def _positional_extract(question: str, retrieved_texts: list) -> str:
#     """Extract Nth item from a list. Generic."""
#     numbers = _extract_numbers(question)
#     if not numbers:
#         return ""
#     try:
#         idx = int(numbers[0]) - 1
#     except ValueError:
#         return ""
#     if idx < 0:
#         return ""

#     best_chunk, best_count = "", 0
#     for chunk in retrieved_texts:
#         lines       = [l.strip() for l in chunk.split("\n") if l.strip()]
#         short_lines = [l for l in lines if len(l.split()) <= 12]
#         if len(short_lines) > best_count:
#             best_count = len(short_lines)
#             best_chunk = chunk

#     if not best_chunk:
#         return ""

#     lines       = [l.strip() for l in best_chunk.split("\n") if l.strip()]
#     short_lines = [l for l in lines if len(l.split()) <= 12]
#     if idx < len(short_lines):
#         return re.sub(r'^\s*\d+[\.\)]\s*', '', short_lines[idx]).strip()
#     return ""


# # ============================================================
# # SIGNAL 3 — Structural consistency for NAVIGATIONAL answers
# # ============================================================

# def _check_identifier_grounded(answer: str, retrieved_texts: list) -> bool:
#     """
#     If answer contains a multi-word identifier (word+number or number+word),
#     verify it appears verbatim in retrieved chunks.
#     Generic — no hardcoded section words.
#     """
#     identifiers = re.findall(
#         r'\b[a-zA-Z]+\s+\d+\b|\b\d+\s+[a-zA-Z]+\b',
#         answer.lower()
#     )
#     if not identifiers:
#         return True

#     chunk_text = " ".join(retrieved_texts).lower()
#     for ident in identifiers:
#         if ident not in chunk_text:
#             print(f"[Signal3] ❌ Identifier '{ident}' not in retrieved chunks")
#             return False
#     return True


# # ============================================================
# # REFUSAL — semantic version (local)
# # ============================================================

# _REFUSAL_ANCHOR = "this information is not available in the provided context"


# def _is_refusal_semantic(text: str) -> bool:
#     if not text or len(text.strip()) < 2:
#         return True
#     sim = semantic_similarity(text, [_REFUSAL_ANCHOR])
#     print(f"[Refusal] sim={sim:.3f} | '{text[:50]}'")
#     return sim > 0.72


# # ============================================================
# # LANGGRAPH NODES
# # ============================================================

# def node_extract(state: DocState) -> DocState:
#     extract_start    = time.time()
#     text, page_count = extract_pdf_parallel(state["pdf_path"])
#     extract_time     = time.time() - extract_start
#     state["extracted_text"] = text
#     state["page_count"]     = page_count
#     state["char_count"]     = len(text)
#     state["metrics"]["extraction_time_sec"]  = round(extract_time, 2)
#     state["metrics"]["pages_processed"]      = page_count
#     state["metrics"]["characters_processed"] = len(text)
#     state["metrics"]["words_processed"]      = len(text.split())
#     return state


# def node_chunk(state: DocState) -> DocState:
#     text = state["extracted_text"]
#     state["query_type"] = "QA"

#     summary_chunks, rag_chunks = semantic_chunk(text)

#     # FIX 4 — normalize all chunks after chunking
#     # Converts unicode symbols so retrieval doesn't fail on symbol mismatches
#     rag_chunks     = [normalize_text(c) for c in rag_chunks]
#     summary_chunks = [normalize_text(c) for c in summary_chunks]

#     state["summary_chunks"] = summary_chunks
#     state["chunks"]         = rag_chunks

#     state["metrics"]["summary_chunks"] = len(summary_chunks)
#     state["metrics"]["chunks_created"] = len(rag_chunks)
#     # state["metrics"]["doc_type"]       = state.get("doc_type", "general")
#     state["metrics"]["query_type"]     = state["query_type"]
#     print(f"[Chunk] {len(summary_chunks)} summary chunks | {len(rag_chunks)} RAG chunks")
#     return state




# def node_summarize(state: DocState) -> DocState:
#     pdf_hash  = get_pdf_hash(state["pdf_path"])
#     cache_key = pdf_hash  # ❌ removed doc_type dependency

#     # ── Cache check ──────────────────────────────────────────
#     if cache_key in _summary_cache:
#         cached = _summary_cache[cache_key]
#         print("[Summary] ✅ Cache hit")
#         emit_event(
#             state.get("request_id", ""),
#             "agent_action",
#             "⚡ Summary loaded from cache instantly!"
#         )
#         state["answer"] = cached["summary"]
#         state["metrics"].update(cached["metrics"])
#         state["metrics"]["type"] = "summary"
#         return state

#     # ── Generate summary ─────────────────────────────────────
#     summary_start = time.time()
#     raptor_summarize._request_id = state.get("request_id", "")

#     # ❌ removed doc_type argument
#     summary, map_time, reduce_time = raptor_summarize(
#         state["summary_chunks"]
#     )

#     summary_time = time.time() - summary_start

#     # ── Store result ─────────────────────────────────────────
#     state["answer"] = summary

#     metrics_snapshot = {
#         "summary_time_sec":     round(summary_time, 2),
#         "summary_length_words": len(summary.split()),
#         "parallel_workers":     min(MAX_WORKERS, len(state["summary_chunks"])),
#         "map_time_sec":         round(map_time, 2),
#         "reduce_time_sec":      round(reduce_time, 2),
#         "llm_calls":            3,
#     }

#     state["metrics"].update(metrics_snapshot)
#     state["metrics"]["type"] = "summary"

#     # ── Cache result ─────────────────────────────────────────
#     _summary_cache[cache_key] = {
#         "summary": summary,
#         "metrics": metrics_snapshot
#     }

#     print(f"[Summary] Done ({len(summary.split())} words)")
#     return state

# def node_qa(state: DocState) -> DocState:
#     qa_start_t = time.time()
#     question   = state["question"]
#     request_id = state.get("request_id", "")
#     all_chunks = state["chunks"]

#     # FIX 4 — normalize question the same way chunks were normalized
#     question = normalize_text(question)
#     numeric_intent = "NONE"   # 🔥 DEFAULT FIX (MANDATORY)
#     # Raw strings for navigational scanning — must NOT be cleaned
#     all_raw = [
#         d if isinstance(d, str) else d.page_content
#         for d in all_chunks
#     ]

#     pdf_hash    = get_pdf_hash(state["pdf_path"])
#     faiss_index = build_faiss_index(all_chunks, pdf_hash)

#     print(f"[QA] START | {len(all_chunks)} chunks")

#     recall_score      = 0.0
#     retrieval_score   = 0.0
#     context_precision = 0.0
#     grounding         = 0.0
#     llm_calls         = 0
#     model_used        = "llama"
#     confidence        = 0.0
#     decision_type     = "accepted"
#     retrieved         = []
#     retrieved_texts   = []
#     answer            = ""

#     # ── STEP 1: RETRIEVE ─────────────────────────────────────
#     retrieved = multi_query_retrieve(
#         question, faiss_index,
#         k=50,
#         all_chunks=all_chunks,
#         query_type="FACTUAL_QA"
#     )


   
#     # 🔥 STEP 2 — RERANK + SAFE FALLBACK

#     retrieved, reranker_top, _ = rerank_docs(
#         question, retrieved, top_k=8, apply_pruning=True
#     )

#     retrieved = protect_exact_matches(
#         question, retrieved, all_chunks, top_k=8
#     )

#     # 🔥 FIX 1 — fallback must replace retrieved (NOT just texts)
#     if not retrieved or len(retrieved) < 3:
#         print("[QA] ⚠️ Reranker too aggressive → fallback to initial chunks")
#         retrieved = all_chunks[:8]

#     # 🔥 FIX 2 — safe conversion
#     retrieved_texts = [
#         d.page_content if hasattr(d, "page_content") else str(d)
#         for d in retrieved
#     ]
#     if reranker_top < -5:
#         print("[Guard] ⚠️ Weak reranker score — continuing")

#     # ── STEP 3: METRICS ───────────────────────────────────────
#     retrieval_score   = compute_retrieval_score(question, retrieved)
#     context_precision = compute_context_precision(question, retrieved)
#     recall_score      = compute_recall_at_k(
#         question, retrieved, all_chunks, k=len(retrieved)
#     )

#     # ── FIX 6 — k expansion: low recall OR numeric question ───
#     is_numeric_question = len(_extract_numbers(question)) > 0
#     if recall_score < 25 or is_numeric_question:
#         print("[QA] ⚠️ Expanding retrieval (low recall or numeric question)")
#         expanded = multi_query_retrieve(
#             question, faiss_index, k=30,
#             all_chunks=all_chunks, query_type="FACTUAL_QA"
#         )
#         if len(expanded) > len(retrieved):
#             retrieved = expanded
#             retrieved_texts = [
#                 d.page_content if hasattr(d, "page_content") else str(d)
#                 for d in retrieved
#             ]
#         # 🔥 ENSURE numeric_intent ALWAYS DEFINED
#         numeric_intent = _detect_numeric_intent(question, retrieved_texts)

#         # fallback check (keep this)
#         is_numeric_question = len(_extract_numbers(question)) > 0

#         if numeric_intent == "NONE" and is_numeric_question:
#             print("[Navigate] Retrying intent detection on full document...")
#             numeric_intent = _detect_numeric_intent(question, all_raw)
#     # ── FIX 1 — Numeric intent: retrieved first, fallback to all_raw ──
#     if numeric_intent == "NAVIGATIONAL":
#         print("[Routing] Numeric intent → NAVIGATIONAL")

#         title = _navigate_full_chunks(question, all_raw)

#         if title:
#             print(f"[Navigate] Extracted title: '{title}'")

#             grounding  = compute_answer_grounding(title, retrieved_texts, question)
#             confidence = round(grounding / 100, 3)
#             qa_time    = time.time() - qa_start_t

#             state["answer"]     = title
#             state["query_type"] = "NAVIGATIONAL"

#             _write_metrics(
#                 state,
#                 "navigational",
#                 "navigational",
#                 grounding,
#                 confidence,
#                 retrieval_score,
#                 context_precision,
#                 recall_score,
#                 llm_calls,
#                 retrieved,
#                 qa_time
#             )

#             return state

#         print("[Navigate] Falling back to QA")
#     # ── Normal routing ────────────────────────────────────────
#     query_type = classify_from_context(question, retrieved_texts)
#     state["query_type"] = query_type
#     print(f"[Routing] Context-based → {query_type}")

#     # ── STEP 4: ANSWER GENERATION ─────────────────────────────

#     if query_type == "FULL_SUMMARY":
#         retrieved_texts = clean_context_for_llm(retrieved_texts)
#         ranked  = reorder_by_question(question, retrieved_texts)
#         context = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:8])
#         structured_context = "\n".join(
#             f"[CONTEXT CHUNK]\n{c}" for c in context.split("\n\n---\n\n")
#         )
#         emit_event(request_id, "stream_start", "✍️ Generating answer...")
#         answer, _ = call_llama_streaming(
#             QA_PROMPT.format(context=structured_context[:2500], question=question),
#             request_id=request_id, temperature=0.0
#         )
#         answer     = clean_artifacts(answer).strip()
#         model_used = "llama_summary"
#         llm_calls  = 1

#     elif query_type == "MULTIPART_QA":
#         retrieved_texts = clean_context_for_llm(retrieved_texts)
#         ranked  = reorder_by_question(question, retrieved_texts)
#         # FIX 3 — 4000 chars for MULTIPART to capture all list items
#         context = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:8])
#         emit_event(request_id, "agent_start",
#                    f"🤖 MULTIPART | {len(retrieved)} chunks")
#         emit_event(request_id, "stream_start", "✍️ Generating answer...")
#         answer, _ = call_llama_streaming(
#             QA_PROMPT.format(context=context[:4000], question=question),
#             request_id=request_id, temperature=0.0
#         )
#         answer     = clean_artifacts(answer).strip()
#         model_used = "llama_multipart"
#         llm_calls  = 1

#     else:
        
#         # FACTUAL_QA / VERIFICATION_QA

#         if not retrieved_texts:
#             print("[QA] ❌ No context → NOT FOUND")
#             state["answer"] = "This information is not present in the document."
#             _write_metrics(state, "not_found", "no_context",
#                         0.0, 0.0, retrieval_score, context_precision,
#                         recall_score, llm_calls, retrieved,
#                         time.time() - qa_start_t)
#             return state

#         # 🔥 FIX 5 — SIMPLE QA ROUTING (ADD HERE)

#         context = "\n".join(retrieved_texts)

#         if (
#             query_type == "FACTUAL_QA"
#             and len(question.split()) <= 10
#             and not is_numeric_question
#             and recall_score >= 60
#         ):
#             print("[QA] ⚡ Direct QA (no ReAct)")

#             react_ans, _ = call_llama_streaming(
#                 f"""
#             Answer the question using ONLY the given context.
#             Return ONLY the exact answer phrase from the context.
#             Do NOT explain. Do NOT say 'not found' unless absolutely missing.

#             Context:
#             {context[:2000]}

#             Question: {question}

#             Answer:
#             """,
#                 request_id=request_id,
#                 temperature=0.0
#             )
#             llm_calls += 1
#             model_used = "llama_direct"

#         else:
#             react_ans, model_used, _, _, _, _, react_calls, _ = react_agent(
#     question,
#     faiss_index,
#     query_type,
#     all_chunks,
#     request_id,
#     recall_score
# )
 
#         react_ans = clean_artifacts(react_ans).strip().strip('"').strip("'")
#         # react_ans = clean_reasoning_answer(react_ans, question)
#         print(f"[QA] Answer: '{react_ans[:60]}'")

#         # FIX 2 — Short answer bypass: skip for VERIFICATION_QA
#         words         = react_ans.split()
#         content_words = [w for w in words if len(w) > 2 and not w.isdigit()]

     
#         print(f"[DEBUG] react_ans before refusal check: repr={repr(react_ans)}")

#         refusal_phrases = [
#             "not present",
#             "not mentioned",
#             "not available",
#             "no information",
#             "cannot find",
#             "not in the document"
#         ]

#         if any(p in react_ans.lower() for p in refusal_phrases):
#             print("[QA] ⚠️ Refusal detected")
#             qa_time = time.time() - qa_start_t
#             state["answer"] = "This information is not present in the document."
#             _write_metrics(state, "not_found", "not_found",
#                         75.0, 0.75, retrieval_score, context_precision,
#                         recall_score, llm_calls, retrieved, qa_time)
#             return state
     
        
#         grounding_score = compute_answer_grounding(react_ans, retrieved_texts, question)

#         # ✅ DEFAULT (VERY IMPORTANT)
#         answer = react_ans
#         decision_type = "accepted"

#         # 🔥 FINAL HALLUCINATION GUARD
#         if query_type == "FACTUAL_QA":
#             if grounding_score < 50:
#                 print("[QA] ⚠️ Weak structural answer → NOT FOUND")
#                 answer = "This information is not present in the document."
#                 decision_type = "not_found"

#         # 🔥 VERIFICATION LOGIC
#         elif query_type == "VERIFICATION_QA":

#             ans = react_ans.strip().lower()

#             if ans in ["t/f", "true/false"]:
#                 answer = "Yes"

#             elif ans in ["false", "no"]:
#                 answer = "No"

#             elif len(ans.split()) <= 3:
#                 answer = react_ans

#             elif grounding_score < 80:
#                 answer = "This information is not present in the document."
#                 decision_type = "verification_failed"

#             else:
#                 answer = react_ans
#                 decision_type = "accepted"

#             grounding  = compute_answer_grounding(answer, retrieved_texts, question)
#             confidence = round(grounding / 100, 3)
#             qa_time    = time.time() - qa_start_t
     
   
#             state["answer"] = normalize_answer(answer)

#             _write_metrics(
#                 state,
#                 model_used,
#                 decision_type,
#                 grounding,
#                 confidence,
#                 retrieval_score,
#                 context_precision,
#                 recall_score,
#                 llm_calls,
#                 retrieved,
#                 qa_time
#             )

#             return state
#         # 🔥 TRUST GATE (FINAL CLEAN VERSION)

#         recall = recall_score
#         grounding = grounding_score
#         length = len(react_ans.split())

#         # 1. No reliable context → reject
#         if recall < 25:
#             print("[QA] ❌ Low recall → NOT FOUND")
#             answer = "This information is not present in the document."
#             decision_type = "low_recall"

#         # 2. Very weak grounding → reject
#         elif grounding < 40:
#             print("[QA] ❌ Very low grounding → NOT FOUND")
#             answer = "This information is not present in the document."
#             decision_type = "low_grounding"

      
#         elif length <= 3:
#             if grounding >= 60:
#                 answer = react_ans
#                 decision_type = "accepted"
#             else:
#                 print("[QA] ❌ Weak short answer → NOT FOUND")
#                 answer = "This information is not present in the document."
#                 decision_type = "weak_short"

#         # 4. Strong signals → accept
#         elif recall >= 50 and grounding >= 60:
#             answer = react_ans
#             decision_type = "accepted"

#         # 5. Otherwise reject safely
#         else:
#             print("[QA] ❌ Uncertain → NOT FOUND")
#             answer = "This information is not present in the document."
#             decision_type = "uncertain"
#     # ── STEP 5: METRICS ───────────────────────────────────────
#     grounding  = compute_answer_grounding(answer, retrieved_texts, question)
#     confidence = round(grounding / 100, 3)
#     qa_time    = time.time() - qa_start_t

#     print(f"[QA] Done in {qa_time:.1f}s | model={model_used} | "
#           f"grounding={grounding:.1f}% | recall={recall_score:.1f}% | "
#           f"confidence={confidence:.3f} | decision={decision_type}")

#     if not answer.strip():
#         answer = "Could not find a relevant answer in the PDF."
#     else:
#         answer = normalize_answer(answer)

#     state["answer"] = answer
#     _write_metrics(state, model_used, decision_type, grounding,
#                    confidence, retrieval_score, context_precision,
#                    recall_score, llm_calls, retrieved, qa_time)
#     return state
# def node_validate(state: DocState) -> DocState:
#     answer = state["answer"]
#     retry  = state.get("retry_count", 0)

#     # ── Retry for empty/very weak answers ────────────────────
#     if len(answer.strip()) < 3 and retry < 2:
#         state["retry_count"] = retry + 1
#         state["answer"]      = ""
#         return state

#     total_time    = time.time() - state["start_time"]
#     output_words  = len(answer.split())
#     output_tokens = output_words * 1.3

#     extract_time  = state["metrics"].get("extraction_time_sec", 0)
#     llm_time      = max(total_time - extract_time, 1)
#     tps           = round(output_tokens / llm_time, 2) if llm_time > 0 else 0

#     m = state["metrics"]

#     # ── Core metrics ─────────────────────────────────────────
#     m["response_time_sec"]    = round(total_time, 2)
#     m["extraction_time_sec"]  = m.get("extraction_time_sec", 0)
#     m["pages_processed"]      = state.get("page_count", 0)
#     m["characters_processed"] = state.get("char_count", 0)
#     m["words_processed"]      = len(state.get("extracted_text", "").split())

#     # ── Type-specific metrics ────────────────────────────────
#     if m.get("type") == "summary":
#         m["summary_time_sec"]     = m.get("summary_time_sec", 0)
#         m["summary_length_words"] = len(answer.split())

#     if m.get("type") == "qa":
#         m["qa_time_sec"]      = m.get("qa_time_sec", 0)
#         m["confidence_score"] = m.get("confidence_score", 0)

#     # ── Performance ──────────────────────────────────────────
#     m["ttft_sec"]        = round(total_time, 2)
#     m["e2e_latency_sec"] = round(total_time, 2)
#     m["tps"]             = tps

#     # ❌ REMOVED doc_type (no longer used anywhere)
#     # m["doc_type"] = state.get("doc_type", "general")

#     # ── Context info ─────────────────────────────────────────
#     m["query_type"]     = state.get("query_type", "")
#     m["chunks_created"] = m.get("chunks_created", 0)
#     m["retry_count"]    = retry

#     # ── Model + retrieval metrics ────────────────────────────
#     m["model_used"]        = m.get("model_used", "llama_react")
#     m["llm_calls"]         = m.get("llm_calls", 0)
#     m["retrieval_score"]   = m.get("retrieval_score", 0)
#     m["context_precision"] = m.get("context_precision", 0)
#     m["answer_grounding"]  = m.get("answer_grounding", 0)
#     m["recall_at_k"]       = m.get("recall_at_k", 0)

#     # ── Summary-specific ─────────────────────────────────────
#     if m.get("type") == "summary":
#         m["parallel_workers"] = m.get("parallel_workers", 0)
#         m["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
#         m["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)

#     # ── QA-specific ──────────────────────────────────────────
#     if m.get("type") == "qa":
#         m["chunks_retrieved"] = m.get("chunks_retrieved", 0)
#         m["decision_type"]    = m.get("decision_type", "accepted")
#         m["confidence_raw"]   = m.get("confidence_raw", 0.0)

#     state["metrics"] = m
#     return state




# def _write_metrics(state, model_used, decision_type, grounding,
#                    confidence, retrieval_score, context_precision,
#                    recall_score, llm_calls, retrieved, qa_time):
#     m = state.setdefault("metrics", {})
#     m["qa_time_sec"]       = round(qa_time, 2)
#     m["confidence_score"]  = round(confidence * 100, 2)
#     m["retrieval_score"]   = retrieval_score
#     m["context_precision"] = context_precision
#     m["answer_grounding"]  = grounding
#     m["recall_at_k"]       = recall_score
#     m["llm_calls"]         = llm_calls
#     m["model_used"]        = model_used
#     m["chunks_retrieved"]  = len(retrieved)
#     m["type"]              = "qa"
#     m["decision_type"]     = decision_type
#     m["confidence_raw"]    = round(confidence, 4)



































# # chapter 4 works, ai hallucination but bnreakes for crop
# import time
# import re
# import spacy
# _nlp = spacy.load("en_core_web_lg")

# from docmind_rag.config.settings import (
#     MAX_WORKERS, FACTUAL_TOP_K, MULTIPART_TOP_K
# )
# from docmind_rag.utils.text import extract_named_entities, expand_answer
# from docmind_rag.core.state import DocState
# from docmind_rag.core.prompts import QA_PROMPT
# from docmind_rag.models.llm import call_llama_streaming
# from docmind_rag.models.embeddings import build_faiss_index
# from docmind_rag.models.reranker import rerank_docs, protect_exact_matches
# from docmind_rag.services.extraction import extract_pdf_parallel, _extraction_cache
# from docmind_rag.services.chunking import semantic_chunk, raptor_summarize, _summary_cache
# from docmind_rag.services.retrieval import multi_query_retrieve
# from docmind_rag.services.agent import react_agent, roberta_qa, clean_reasoning_answer
# from docmind_rag.events.events import emit_event
# from docmind_rag.utils.helpers import get_pdf_hash, clean_artifacts, clean_chunk_text, normalize_answer
# from docmind_rag.utils.text import (
#     classify_from_context,
#     reorder_by_question,
#     normalize_text,
# )
# from docmind_rag.utils.metrics import (
#     compute_answer_grounding,
#     compute_retrieval_score,
#     compute_context_precision,
#     compute_recall_at_k,
#     semantic_similarity,
   
# )

# # def trust_gate(answer: str, metrics: dict) -> str:
# #     recall = metrics.get("recall_at_k", 0)
# #     grounding = metrics.get("answer_grounding", 0)
# #     length = len(answer.split())

# #     # 1. No reliable context → reject
# #     if recall < 25:
# #         return "not_found"

# #     # 2. Weak grounding → reject
# #     if grounding < 50:
# #         return "not_found"

# #     # 3. Short answers must be very strong
# #     if length <= 3:
# #         if recall >= 70 and grounding >= 70:
# #             return "accepted"
# #         return "not_found"

# #     # 4. Strong signal → accept
# #     if recall >= 50 and grounding >= 60:
# #         return "accepted"
   
# #     # 5. Default safe behavior
# #     else:
# #         return "not_found"
# # ============================================================
# # LOCAL HELPERS
# # ============================================================

# def clean_context_for_llm(chunks):
#     cleaned_chunks = []
#     for chunk in chunks:
#         lines = chunk.split("\n")
#         filtered_lines = []
#         for line in lines:
#             line_strip = line.strip()
#             is_question_like = (
#                 line_strip.endswith("?") or
#                 (len(line_strip.split()) < 15 and "?" in line_strip)
#             )
#             if is_question_like:
#                 continue
#             filtered_lines.append(line)
#         cleaned_chunks.append("\n".join(filtered_lines))
#     return cleaned_chunks


# def is_grounded(answer, retrieved_texts, threshold=0.6):
#     if not answer:
#         return False
#     answer_words = set(answer.lower().split())
#     if not answer_words:
#         return False
#     best_overlap = 0
#     for chunk in retrieved_texts:
#         chunk_words = set(chunk.lower().split())
#         overlap = len(answer_words & chunk_words) / len(answer_words)
#         best_overlap = max(best_overlap, overlap)
#     return best_overlap >= threshold


# # ============================================================
# # NUMERIC INTENT DETECTION
# # ============================================================

# def _extract_numbers(text: str) -> list:
#     return re.findall(r'\b\d+\b', text)


# def _detect_numeric_intent(question: str, chunks: list) -> str:
#     """
#     Detect NAVIGATIONAL or POSITIONAL from chunk structure.
#     Generic — no hardcoded section words.
#     """
#     numbers = _extract_numbers(question)
#     if not numbers:
#         return "NONE"

#     q_words = set(question.lower().split())

#     for chunk in chunks:
#         lines = [l.strip() for l in chunk.split("\n") if l.strip()]
#         if not lines:
#             continue
#         # chunks may have no newlines — take first 10 tokens as heading proxy
#         first_line = " ".join(lines[0].split()[:10]).lower()
#         line_words = set(first_line.split())
#         overlap    = len(q_words & line_words)
#         num_found  = any(re.search(rf'\b{re.escape(num)}\b', first_line) for num in numbers)

#         if num_found:
#             print(f"[NavDebug] first_line='{first_line}' | overlap={overlap} | words={len(first_line.split())}")

#         if num_found and overlap >= 2:  # ← remove word count check entirely
#             return "NAVIGATIONAL"
#     for chunk in chunks:
#         lines       = [l.strip() for l in chunk.split("\n") if l.strip()]
#         short_lines = [l for l in lines if len(l.split()) <= 12]
#         if len(short_lines) >= 3:
#             first = lines[0].lower() if lines else ""
#             if not any(re.search(rf'\b{re.escape(num)}\b', first) for num in numbers):
#                 return "POSITIONAL"

#     return "NONE"


# def _navigate_full_chunks(question: str, all_raw_chunks: list) -> str:
#     numbers = _extract_numbers(question)
#     if not numbers:
#         return ""

#     q_words    = set(question.lower().split())
#     best_line  = ""
#     best_score = 0

#     for chunk in all_raw_chunks:
#         lines = [l.strip() for l in chunk.split("\n") if l.strip()]
#         if not lines:
#             continue
#         first_line  = " ".join(lines[0].split()[:10])  # ← slice to first 10 tokens
#         first_lower = first_line.lower()
#         if re.search(r'\d+\s*---\s*-\s*\d+', first_lower):
#                 print(f"[NAV] ❌ Skipping page-style chunk: '{first_line}'")
#                 continue

#         for num in numbers:
#             if not re.search(rf'\b{re.escape(num)}\b', first_lower):
#                 continue
#             line_words = set(first_lower.split())
#             overlap    = len(q_words & line_words)
#             if overlap >= 2:  # ← remove word count check, raise overlap threshold
#                 score = overlap
#                 if score > best_score:
#                     best_score = score
#                     best_line  = first_line

#     if not best_line:
#         return ""

#     title = best_line
#     title = re.sub(r'^\s*\d+\s*[-—–]+\s*[—–-]?\s*\d*\s*', '', title)  # strip page prefix like "31 --- - 32"
#     title = re.sub(r'[:\-–—]', ' ', title)
#     title = re.sub(r'\s{2,}', ' ', title)
#     return title.strip()

# def _positional_extract(question: str, retrieved_texts: list) -> str:
#     """Extract Nth item from a list. Generic."""
#     numbers = _extract_numbers(question)
#     if not numbers:
#         return ""
#     try:
#         idx = int(numbers[0]) - 1
#     except ValueError:
#         return ""
#     if idx < 0:
#         return ""

#     best_chunk, best_count = "", 0
#     for chunk in retrieved_texts:
#         lines       = [l.strip() for l in chunk.split("\n") if l.strip()]
#         short_lines = [l for l in lines if len(l.split()) <= 12]
#         if len(short_lines) > best_count:
#             best_count = len(short_lines)
#             best_chunk = chunk

#     if not best_chunk:
#         return ""

#     lines       = [l.strip() for l in best_chunk.split("\n") if l.strip()]
#     short_lines = [l for l in lines if len(l.split()) <= 12]
#     if idx < len(short_lines):
#         return re.sub(r'^\s*\d+[\.\)]\s*', '', short_lines[idx]).strip()
#     return ""


# # ============================================================
# # SIGNAL 3 — Structural consistency for NAVIGATIONAL answers
# # ============================================================

# def _check_identifier_grounded(answer: str, retrieved_texts: list) -> bool:
#     """
#     If answer contains a multi-word identifier (word+number or number+word),
#     verify it appears verbatim in retrieved chunks.
#     Generic — no hardcoded section words.
#     """
#     identifiers = re.findall(
#         r'\b[a-zA-Z]+\s+\d+\b|\b\d+\s+[a-zA-Z]+\b',
#         answer.lower()
#     )
#     if not identifiers:
#         return True

#     chunk_text = " ".join(retrieved_texts).lower()
#     for ident in identifiers:
#         if ident not in chunk_text:
#             print(f"[Signal3] ❌ Identifier '{ident}' not in retrieved chunks")
#             return False
#     return True


# # ============================================================
# # REFUSAL — semantic version (local)
# # ============================================================

# _REFUSAL_ANCHOR = "this information is not available in the provided context"


# def _is_refusal_semantic(text: str) -> bool:
#     if not text or len(text.strip()) < 2:
#         return True
#     sim = semantic_similarity(text, [_REFUSAL_ANCHOR])
#     print(f"[Refusal] sim={sim:.3f} | '{text[:50]}'")
#     return sim > 0.72


# # ============================================================
# # LANGGRAPH NODES
# # ============================================================

# def node_extract(state: DocState) -> DocState:
#     extract_start    = time.time()
#     text, page_count = extract_pdf_parallel(state["pdf_path"])
#     extract_time     = time.time() - extract_start
#     state["extracted_text"] = text
#     state["page_count"]     = page_count
#     state["char_count"]     = len(text)
#     state["metrics"]["extraction_time_sec"]  = round(extract_time, 2)
#     state["metrics"]["pages_processed"]      = page_count
#     state["metrics"]["characters_processed"] = len(text)
#     state["metrics"]["words_processed"]      = len(text.split())
#     return state


# def node_chunk(state: DocState) -> DocState:
#     text = state["extracted_text"]
#     state["query_type"] = "QA"

#     summary_chunks, rag_chunks = semantic_chunk(text)

#     # FIX 4 — normalize all chunks after chunking
#     # Converts unicode symbols so retrieval doesn't fail on symbol mismatches
#     rag_chunks     = [normalize_text(c) for c in rag_chunks]
#     summary_chunks = [normalize_text(c) for c in summary_chunks]

#     state["summary_chunks"] = summary_chunks
#     state["chunks"]         = rag_chunks

#     state["metrics"]["summary_chunks"] = len(summary_chunks)
#     state["metrics"]["chunks_created"] = len(rag_chunks)
#     # state["metrics"]["doc_type"]       = state.get("doc_type", "general")
#     state["metrics"]["query_type"]     = state["query_type"]
#     print(f"[Chunk] {len(summary_chunks)} summary chunks | {len(rag_chunks)} RAG chunks")
#     return state


# # def node_summarize(state: DocState) -> DocState:
# #     pdf_hash  = get_pdf_hash(state["pdf_path"])
# #     cache_key = f"{pdf_hash}_{state.get('doc_type', 'general')}"

# #     if cache_key in _summary_cache:
# #         cached = _summary_cache[cache_key]
# #         print("[Summary] ✅ Cache hit")
# #         emit_event(state.get("request_id", ""), "agent_action",
# #                    "⚡ Summary loaded from cache instantly!")
# #         state["answer"] = cached["summary"]
# #         state["metrics"].update(cached["metrics"])
# #         state["metrics"]["type"] = "summary"
# #         return state

# #     summary_start = time.time()
# #     raptor_summarize._request_id = state.get("request_id", "")
# #     summary, map_time, reduce_time = raptor_summarize(
# #         state["summary_chunks"], state.get("doc_type", "general")
# #     )
# #     summary_time = time.time() - summary_start

# #     state["answer"] = summary
# #     metrics_snapshot = {
# #         "summary_time_sec":     round(summary_time, 2),
# #         "summary_length_words": len(summary.split()),
# #         "parallel_workers":     min(MAX_WORKERS, len(state["summary_chunks"])),
# #         "map_time_sec":         round(map_time, 2),
# #         "reduce_time_sec":      round(reduce_time, 2),
# #         "llm_calls":            3,
# #     }
# #     state["metrics"].update(metrics_snapshot)
# #     state["metrics"]["type"] = "summary"
# #     _summary_cache[cache_key] = {"summary": summary, "metrics": metrics_snapshot}
# #     print(f"[Summary] Done ({len(summary.split())} words)")
# #     return state

# def node_summarize(state: DocState) -> DocState:
#     pdf_hash  = get_pdf_hash(state["pdf_path"])
#     cache_key = pdf_hash  # ❌ removed doc_type dependency

#     # ── Cache check ──────────────────────────────────────────
#     if cache_key in _summary_cache:
#         cached = _summary_cache[cache_key]
#         print("[Summary] ✅ Cache hit")
#         emit_event(
#             state.get("request_id", ""),
#             "agent_action",
#             "⚡ Summary loaded from cache instantly!"
#         )
#         state["answer"] = cached["summary"]
#         state["metrics"].update(cached["metrics"])
#         state["metrics"]["type"] = "summary"
#         return state

#     # ── Generate summary ─────────────────────────────────────
#     summary_start = time.time()
#     raptor_summarize._request_id = state.get("request_id", "")

#     # ❌ removed doc_type argument
#     summary, map_time, reduce_time = raptor_summarize(
#         state["summary_chunks"]
#     )

#     summary_time = time.time() - summary_start

#     # ── Store result ─────────────────────────────────────────
#     state["answer"] = summary

#     metrics_snapshot = {
#         "summary_time_sec":     round(summary_time, 2),
#         "summary_length_words": len(summary.split()),
#         "parallel_workers":     min(MAX_WORKERS, len(state["summary_chunks"])),
#         "map_time_sec":         round(map_time, 2),
#         "reduce_time_sec":      round(reduce_time, 2),
#         "llm_calls":            3,
#     }

#     state["metrics"].update(metrics_snapshot)
#     state["metrics"]["type"] = "summary"

#     # ── Cache result ─────────────────────────────────────────
#     _summary_cache[cache_key] = {
#         "summary": summary,
#         "metrics": metrics_snapshot
#     }

#     print(f"[Summary] Done ({len(summary.split())} words)")
#     return state

# def node_qa(state: DocState) -> DocState:
#     qa_start_t = time.time()
#     question   = state["question"]
#     request_id = state.get("request_id", "")
#     all_chunks = state["chunks"]

#     # FIX 4 — normalize question the same way chunks were normalized
#     question = normalize_text(question)

#     # Raw strings for navigational scanning — must NOT be cleaned
#     all_raw = [
#         d if isinstance(d, str) else d.page_content
#         for d in all_chunks
#     ]

#     pdf_hash    = get_pdf_hash(state["pdf_path"])
#     faiss_index = build_faiss_index(all_chunks, pdf_hash)

#     print(f"[QA] START | {len(all_chunks)} chunks")

#     recall_score      = 0.0
#     retrieval_score   = 0.0
#     context_precision = 0.0
#     grounding         = 0.0
#     llm_calls         = 0
#     model_used        = "llama"
#     confidence        = 0.0
#     decision_type     = "accepted"
#     retrieved         = []
#     retrieved_texts   = []
#     answer            = ""

#     # ── STEP 1: RETRIEVE ─────────────────────────────────────
#     retrieved = multi_query_retrieve(
#         question, faiss_index,
#         k=50,
#         all_chunks=all_chunks,
#         query_type="FACTUAL_QA"
#     )


   
#     # 🔥 STEP 2 — RERANK + SAFE FALLBACK

#     retrieved, reranker_top, _ = rerank_docs(
#         question, retrieved, top_k=8, apply_pruning=True
#     )

#     retrieved = protect_exact_matches(
#         question, retrieved, all_chunks, top_k=8
#     )

#     # 🔥 FIX 1 — fallback must replace retrieved (NOT just texts)
#     if not retrieved or len(retrieved) < 3:
#         print("[QA] ⚠️ Reranker too aggressive → fallback to initial chunks")
#         retrieved = all_chunks[:8]

#     # 🔥 FIX 2 — safe conversion
#     retrieved_texts = [
#         d.page_content if hasattr(d, "page_content") else str(d)
#         for d in retrieved
#     ]
#     if reranker_top < -5:
#         print("[Guard] ⚠️ Weak reranker score — continuing")

#     # ── STEP 3: METRICS ───────────────────────────────────────
#     retrieval_score   = compute_retrieval_score(question, retrieved)
#     context_precision = compute_context_precision(question, retrieved)
#     recall_score      = compute_recall_at_k(
#         question, retrieved, all_chunks, k=len(retrieved)
#     )

#     # ── FIX 6 — k expansion: low recall OR numeric question ───
#     is_numeric_question = len(_extract_numbers(question)) > 0
#     if recall_score < 25 or is_numeric_question:
#         print("[QA] ⚠️ Expanding retrieval (low recall or numeric question)")
#         expanded = multi_query_retrieve(
#             question, faiss_index, k=30,
#             all_chunks=all_chunks, query_type="FACTUAL_QA"
#         )
#         if len(expanded) > len(retrieved):
#             retrieved = expanded
#             retrieved_texts = [
#                 d.page_content if hasattr(d, "page_content") else str(d)
#                 for d in retrieved
#             ]

#     # ── FIX 1 — Numeric intent: retrieved first, fallback to all_raw ──
#     numeric_intent = _detect_numeric_intent(question, retrieved_texts)

#     if numeric_intent == "NONE" and is_numeric_question:
#         print("[Navigate] Retrying intent detection on full document...")
#         numeric_intent = _detect_numeric_intent(question, all_raw)
#         if numeric_intent != "NONE":
#             print(f"[Navigate] Intent found in full document: {numeric_intent}")

#     # ── NAVIGATIONAL ──────────────────────────────────────────
#     # if numeric_intent == "NAVIGATIONAL":
#     #     print(f"[Routing] Numeric intent → NAVIGATIONAL")
#     #     title = _navigate_full_chunks(question, all_raw)
#     #     if title:
#     #         print(f"[Navigate] Extracted title: '{title}'")
#     #         grounding  = compute_answer_grounding(title, retrieved_texts, question)
#     #         confidence = round(grounding / 100, 3)
#     #         qa_time    = time.time() - qa_start_t
#     #         state["answer"]      = title
#     #         state["query_type"]  = "NAVIGATIONAL"
#     #         _write_metrics(state, "navigational", "navigational", grounding,
#     #                        confidence, retrieval_score, context_precision,
#     #                        recall_score, llm_calls, retrieved, qa_time)
#     #         return state
#     #     print(f"[Navigate] Extraction failed — falling through to LLM")

#     # # ── POSITIONAL ────────────────────────────────────────────
#     # elif numeric_intent == "POSITIONAL":
#     #     print(f"[Routing] Numeric intent → POSITIONAL")
#     #     pos_answer = _positional_extract(question, retrieved_texts)
#     #     if pos_answer:
#     #         print(f"[Positional] Extracted: '{pos_answer}'")
#     #         grounding  = compute_answer_grounding(pos_answer, retrieved_texts, question)
#     #         confidence = round(grounding / 100, 3)
#     #         qa_time    = time.time() - qa_start_t
#     #         state["answer"]      = pos_answer
#     #         state["query_type"]  = "POSITIONAL"
#     #         _write_metrics(state, "positional", "positional", grounding,
#     #                        confidence, retrieval_score, context_precision,
#     #                        recall_score, llm_calls, retrieved, qa_time)
#     #         return state
#     #     print(f"[Positional] Extraction failed — falling through to LLM")
#     if numeric_intent == "NAVIGATIONAL":
#         print("[Routing] Numeric intent → NAVIGATIONAL")

#         title = _navigate_full_chunks(question, all_raw)

#         if title:
#             print(f"[Navigate] Extracted title: '{title}'")

#             # 🔥 ADD THIS HERE (NOT BEFORE NAVIGATION)
#             numbers = _extract_numbers(question)

#             valid = any(
#                 any(f"{num} ---" in chunk.lower() for num in numbers)
#                 for chunk in retrieved_texts
#             )

#             if not valid:
#                 print("[NAV] ❌ Invalid numeric structure → fallback to QA")

#             else:
#                 print("[NAV] ✅ Valid structure → accepting")

#                 grounding  = compute_answer_grounding(title, retrieved_texts, question)
#                 confidence = round(grounding / 100, 3)
#                 qa_time    = time.time() - qa_start_t

#                 state["answer"]     = title
#                 state["query_type"] = "NAVIGATIONAL"

#                 _write_metrics(
#                     state,
#                     "navigational",
#                     "navigational",
#                     grounding,
#                     confidence,
#                     retrieval_score,
#                     context_precision,
#                     recall_score,
#                     llm_calls,
#                     retrieved,
#                     qa_time
#                 )

#                 return state

#         print("[Navigate] Falling back to QA")
#     # ── Normal routing ────────────────────────────────────────
#     query_type = classify_from_context(question, retrieved_texts)
#     state["query_type"] = query_type
#     print(f"[Routing] Context-based → {query_type}")

#     # ── STEP 4: ANSWER GENERATION ─────────────────────────────

#     if query_type == "FULL_SUMMARY":
#         retrieved_texts = clean_context_for_llm(retrieved_texts)
#         ranked  = reorder_by_question(question, retrieved_texts)
#         context = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:8])
#         structured_context = "\n".join(
#             f"[CONTEXT CHUNK]\n{c}" for c in context.split("\n\n---\n\n")
#         )
#         emit_event(request_id, "stream_start", "✍️ Generating answer...")
#         answer, _ = call_llama_streaming(
#             QA_PROMPT.format(context=structured_context[:2500], question=question),
#             request_id=request_id, temperature=0.0
#         )
#         answer     = clean_artifacts(answer).strip()
#         model_used = "llama_summary"
#         llm_calls  = 1

#     elif query_type == "MULTIPART_QA":
#         retrieved_texts = clean_context_for_llm(retrieved_texts)
#         ranked  = reorder_by_question(question, retrieved_texts)
#         # FIX 3 — 4000 chars for MULTIPART to capture all list items
#         context = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:8])
#         emit_event(request_id, "agent_start",
#                    f"🤖 MULTIPART | {len(retrieved)} chunks")
#         emit_event(request_id, "stream_start", "✍️ Generating answer...")
#         answer, _ = call_llama_streaming(
#             QA_PROMPT.format(context=context[:4000], question=question),
#             request_id=request_id, temperature=0.0
#         )
#         answer     = clean_artifacts(answer).strip()
#         model_used = "llama_multipart"
#         llm_calls  = 1

#     else:
        
#         # FACTUAL_QA / VERIFICATION_QA

#         if not retrieved_texts:
#             print("[QA] ❌ No context → NOT FOUND")
#             state["answer"] = "This information is not present in the document."
#             _write_metrics(state, "not_found", "no_context",
#                         0.0, 0.0, retrieval_score, context_precision,
#                         recall_score, llm_calls, retrieved,
#                         time.time() - qa_start_t)
#             return state

#         # 🔥 FIX 5 — SIMPLE QA ROUTING (ADD HERE)

#         context = "\n".join(retrieved_texts)

#         if (
#             query_type == "FACTUAL_QA"
#             and len(question.split()) <= 10
#             and not is_numeric_question
#             and recall_score >= 60
#         ):
#             print("[QA] ⚡ Direct QA (no ReAct)")

#             react_ans, _ = call_llama_streaming(
#                 f"""
#             Answer the question using ONLY the given context.
#             Return ONLY the exact answer phrase from the context.
#             Do NOT explain. Do NOT say 'not found' unless absolutely missing.

#             Context:
#             {context[:2000]}

#             Question: {question}

#             Answer:
#             """,
#                 request_id=request_id,
#                 temperature=0.0
#             )
#             llm_calls += 1
#             model_used = "llama_direct"

#         else:
#             react_ans, model_used, _, _, _, _, react_calls, _ = react_agent(
#     question,
#     faiss_index,
#     query_type,
#     all_chunks,
#     request_id,
#     recall_score
# )

#         # react_ans, model_used, _, _, _, _, react_calls, retrieved_texts = react_agent(
#         #     question, faiss_index, query_type, all_chunks, request_id
#         # )
 
#         react_ans = clean_artifacts(react_ans).strip().strip('"').strip("'")
#         # react_ans = clean_reasoning_answer(react_ans, question)
#         print(f"[QA] Answer: '{react_ans[:60]}'")

#         # FIX 2 — Short answer bypass: skip for VERIFICATION_QA
#         words         = react_ans.split()
#         content_words = [w for w in words if len(w) > 2 and not w.isdigit()]

#         # if (
#         #     query_type != "VERIFICATION_QA"
#         #     and len(words) <= 3
#         #     and len(content_words) >= 1
#         #     and not _is_refusal_answer(react_ans)
#         # ):
#         #     print("[QA] ⚡ Short answer accepted")
#         #     grounding  = compute_answer_grounding(react_ans, retrieved_texts, question)
#         #     confidence = grounding / 100
#         #     qa_time    = time.time() - qa_start_t
#         #     state["answer"] = react_ans
#         #     _write_metrics(state, model_used, "short_answer",
#         #                    grounding, confidence, retrieval_score,
#         #                    context_precision, recall_score, llm_calls,
#         #                    retrieved, qa_time)
#         #     return state
#         print(f"[DEBUG] react_ans before refusal check: repr={repr(react_ans)}")

#         refusal_phrases = [
#             "not present",
#             "not mentioned",
#             "not available",
#             "no information",
#             "cannot find",
#             "not in the document"
#         ]

#         if any(p in react_ans.lower() for p in refusal_phrases):
#             print("[QA] ⚠️ Refusal detected")
#             qa_time = time.time() - qa_start_t
#             state["answer"] = "This information is not present in the document."
#             _write_metrics(state, "not_found", "not_found",
#                         75.0, 0.75, retrieval_score, context_precision,
#                         recall_score, llm_calls, retrieved, qa_time)
#             return state
#         # print(f"[DEBUG] react_ans before refusal check: repr={repr(react_ans)}")
#         # # Refusal
#         # if _is_refusal_answer(react_ans):
#         #     print("[QA] ⚠️ Refusal detected")
#         #     qa_time = time.time() - qa_start_t
#         #     state["answer"] = "This information is not present in the document."
#         #     _write_metrics(state, "not_found", "not_found",
#         #                    75.0, 0.75, retrieval_score, context_precision,
#         #                    recall_score, llm_calls, retrieved, qa_time)
#         #     return state

        
#         grounding_score = compute_answer_grounding(react_ans, retrieved_texts, question)

#         # ✅ DEFAULT (VERY IMPORTANT)
#         answer = react_ans
#         decision_type = "accepted"

#         # 🔥 FINAL HALLUCINATION GUARD
#         if query_type == "FACTUAL_QA":
#             if grounding_score < 50:
#                 print("[QA] ⚠️ Weak structural answer → NOT FOUND")
#                 answer = "This information is not present in the document."
#                 decision_type = "not_found"

#         # 🔥 VERIFICATION LOGIC
#         elif query_type == "VERIFICATION_QA":

#             ans = react_ans.strip().lower()

#             if ans in ["t/f", "true/false"]:
#                 answer = "Yes"

#             elif ans in ["false", "no"]:
#                 answer = "No"

#             elif len(ans.split()) <= 3:
#                 answer = react_ans

#             elif grounding_score < 80:
#                 answer = "This information is not present in the document."
#                 decision_type = "verification_failed"

#             else:
#                 answer = react_ans
#                 decision_type = "accepted"

#             grounding  = compute_answer_grounding(answer, retrieved_texts, question)
#             confidence = round(grounding / 100, 3)
#             qa_time    = time.time() - qa_start_t
     
   
#             state["answer"] = normalize_answer(answer)

#             _write_metrics(
#                 state,
#                 model_used,
#                 decision_type,
#                 grounding,
#                 confidence,
#                 retrieval_score,
#                 context_precision,
#                 recall_score,
#                 llm_calls,
#                 retrieved,
#                 qa_time
#             )

#             return state
#         # 🔥 TRUST GATE (FINAL CLEAN VERSION)

#         recall = recall_score
#         grounding = grounding_score
#         length = len(react_ans.split())

#         # 1. No reliable context → reject
#         if recall < 25:
#             print("[QA] ❌ Low recall → NOT FOUND")
#             answer = "This information is not present in the document."
#             decision_type = "low_recall"

#         # 2. Very weak grounding → reject
#         elif grounding < 40:
#             print("[QA] ❌ Very low grounding → NOT FOUND")
#             answer = "This information is not present in the document."
#             decision_type = "low_grounding"

#         # # 3. Short answers must be VERY strong
#         # elif length <= 3:
#         #     if recall >= 70 and grounding >= 70:
#         #         answer = react_ans
#         #         decision_type = "accepted"
        
#         #     else:
#         #         print("[QA] ❌ Weak short answer → NOT FOUND")
#         #         answer = "This information is not present in the document."
#         #         decision_type = "weak_short"

#         # # 4. Strong signals → accept
#         # elif recall >= 50 and grounding >= 60:
#         #     answer = react_ans
#         #     decision_type = "accepted"

#         # # 5. Otherwise reject safely
#         # else:
#         #     print("[QA] ❌ Uncertain → NOT FOUND")
#         #     answer = "This information is not present in the document."
#         #     decision_type = "uncertain"
#         elif length <= 3:
#             if grounding >= 60:
#                 answer = react_ans
#                 decision_type = "accepted"
#             else:
#                 print("[QA] ❌ Weak short answer → NOT FOUND")
#                 answer = "This information is not present in the document."
#                 decision_type = "weak_short"

#         # 4. Strong signals → accept
#         elif recall >= 50 and grounding >= 60:
#             answer = react_ans
#             decision_type = "accepted"

#         # 5. Otherwise reject safely
#         else:
#             print("[QA] ❌ Uncertain → NOT FOUND")
#             answer = "This information is not present in the document."
#             decision_type = "uncertain"
#     # ── STEP 5: METRICS ───────────────────────────────────────
#     grounding  = compute_answer_grounding(answer, retrieved_texts, question)
#     confidence = round(grounding / 100, 3)
#     qa_time    = time.time() - qa_start_t

#     print(f"[QA] Done in {qa_time:.1f}s | model={model_used} | "
#           f"grounding={grounding:.1f}% | recall={recall_score:.1f}% | "
#           f"confidence={confidence:.3f} | decision={decision_type}")

#     if not answer.strip():
#         answer = "Could not find a relevant answer in the PDF."
#     else:
#         answer = normalize_answer(answer)

#     state["answer"] = answer
#     _write_metrics(state, model_used, decision_type, grounding,
#                    confidence, retrieval_score, context_precision,
#                    recall_score, llm_calls, retrieved, qa_time)
#     return state
# def node_validate(state: DocState) -> DocState:
#     answer = state["answer"]
#     retry  = state.get("retry_count", 0)

#     # ── Retry for empty/very weak answers ────────────────────
#     if len(answer.strip()) < 3 and retry < 2:
#         state["retry_count"] = retry + 1
#         state["answer"]      = ""
#         return state

#     total_time    = time.time() - state["start_time"]
#     output_words  = len(answer.split())
#     output_tokens = output_words * 1.3

#     extract_time  = state["metrics"].get("extraction_time_sec", 0)
#     llm_time      = max(total_time - extract_time, 1)
#     tps           = round(output_tokens / llm_time, 2) if llm_time > 0 else 0

#     m = state["metrics"]

#     # ── Core metrics ─────────────────────────────────────────
#     m["response_time_sec"]    = round(total_time, 2)
#     m["extraction_time_sec"]  = m.get("extraction_time_sec", 0)
#     m["pages_processed"]      = state.get("page_count", 0)
#     m["characters_processed"] = state.get("char_count", 0)
#     m["words_processed"]      = len(state.get("extracted_text", "").split())

#     # ── Type-specific metrics ────────────────────────────────
#     if m.get("type") == "summary":
#         m["summary_time_sec"]     = m.get("summary_time_sec", 0)
#         m["summary_length_words"] = len(answer.split())

#     if m.get("type") == "qa":
#         m["qa_time_sec"]      = m.get("qa_time_sec", 0)
#         m["confidence_score"] = m.get("confidence_score", 0)

#     # ── Performance ──────────────────────────────────────────
#     m["ttft_sec"]        = round(total_time, 2)
#     m["e2e_latency_sec"] = round(total_time, 2)
#     m["tps"]             = tps

#     # ❌ REMOVED doc_type (no longer used anywhere)
#     # m["doc_type"] = state.get("doc_type", "general")

#     # ── Context info ─────────────────────────────────────────
#     m["query_type"]     = state.get("query_type", "")
#     m["chunks_created"] = m.get("chunks_created", 0)
#     m["retry_count"]    = retry

#     # ── Model + retrieval metrics ────────────────────────────
#     m["model_used"]        = m.get("model_used", "llama_react")
#     m["llm_calls"]         = m.get("llm_calls", 0)
#     m["retrieval_score"]   = m.get("retrieval_score", 0)
#     m["context_precision"] = m.get("context_precision", 0)
#     m["answer_grounding"]  = m.get("answer_grounding", 0)
#     m["recall_at_k"]       = m.get("recall_at_k", 0)

#     # ── Summary-specific ─────────────────────────────────────
#     if m.get("type") == "summary":
#         m["parallel_workers"] = m.get("parallel_workers", 0)
#         m["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
#         m["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)

#     # ── QA-specific ──────────────────────────────────────────
#     if m.get("type") == "qa":
#         m["chunks_retrieved"] = m.get("chunks_retrieved", 0)
#         m["decision_type"]    = m.get("decision_type", "accepted")
#         m["confidence_raw"]   = m.get("confidence_raw", 0.0)

#     state["metrics"] = m
#     return state

# # def node_validate(state: DocState) -> DocState:
# #     answer = state["answer"]
# #     retry  = state.get("retry_count", 0)

# #     if len(answer.strip()) < 3 and retry < 2:
# #         state["retry_count"] = retry + 1
# #         state["answer"]      = ""
# #         return state

# #     total_time    = time.time() - state["start_time"]
# #     output_words  = len(answer.split())
# #     output_tokens = output_words * 1.3
# #     extract_time  = state["metrics"].get("extraction_time_sec", 0)
# #     llm_time      = max(total_time - extract_time, 1)
# #     tps           = round(output_tokens / llm_time, 2) if llm_time > 0 else 0
# #     m             = state["metrics"]

# #     m["response_time_sec"]    = round(total_time, 2)
# #     m["extraction_time_sec"]  = m.get("extraction_time_sec", 0)
# #     m["pages_processed"]      = state.get("page_count", 0)
# #     m["characters_processed"] = state.get("char_count", 0)
# #     m["words_processed"]      = len(state.get("extracted_text", "").split())

# #     if m.get("type") == "summary":
# #         m["summary_time_sec"]     = m.get("summary_time_sec", 0)
# #         m["summary_length_words"] = len(answer.split())
# #     if m.get("type") == "qa":
# #         m["qa_time_sec"]      = m.get("qa_time_sec", 0)
# #         m["confidence_score"] = m.get("confidence_score", 0)

# #     m["ttft_sec"]          = round(total_time, 2)
# #     m["e2e_latency_sec"]   = round(total_time, 2)
# #     m["tps"]               = tps
# #     m["doc_type"]          = state.get("doc_type", "general")
# #     m["query_type"]        = state.get("query_type", "")
# #     m["chunks_created"]    = m.get("chunks_created", 0)
# #     m["retry_count"]       = retry
# #     m["model_used"]        = m.get("model_used", "llama_react")
# #     m["llm_calls"]         = m.get("llm_calls", 0)
# #     m["retrieval_score"]   = m.get("retrieval_score", 0)
# #     m["context_precision"] = m.get("context_precision", 0)
# #     m["answer_grounding"]  = m.get("answer_grounding", 0)
# #     m["recall_at_k"]       = m.get("recall_at_k", 0)

# #     if m.get("type") == "summary":
# #         m["parallel_workers"] = m.get("parallel_workers", 0)
# #         m["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
# #         m["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)
# #     if m.get("type") == "qa":
# #         m["chunks_retrieved"] = m.get("chunks_retrieved", 0)
# #         m["decision_type"]    = m.get("decision_type", "accepted")
# #         m["confidence_raw"]   = m.get("confidence_raw", 0.0)

# #     state["metrics"] = m
# #     return state


# def _write_metrics(state, model_used, decision_type, grounding,
#                    confidence, retrieval_score, context_precision,
#                    recall_score, llm_calls, retrieved, qa_time):
#     m = state.setdefault("metrics", {})
#     m["qa_time_sec"]       = round(qa_time, 2)
#     m["confidence_score"]  = round(confidence * 100, 2)
#     m["retrieval_score"]   = retrieval_score
#     m["context_precision"] = context_precision
#     m["answer_grounding"]  = grounding
#     m["recall_at_k"]       = recall_score
#     m["llm_calls"]         = llm_calls
#     m["model_used"]        = model_used
#     m["chunks_retrieved"]  = len(retrieved)
#     m["type"]              = "qa"
#     m["decision_type"]     = decision_type
#     m["confidence_raw"]    = round(confidence, 4)






















# # chapter 4 works
# import time
# import re
# import spacy
# _nlp = spacy.load("en_core_web_lg")

# from docmind_rag.config.settings import (
#     MAX_WORKERS, FACTUAL_TOP_K, MULTIPART_TOP_K
# )
# from docmind_rag.utils.text import extract_named_entities, expand_answer
# from docmind_rag.core.state import DocState
# from docmind_rag.core.prompts import QA_PROMPT
# from docmind_rag.models.llm import call_llama_streaming
# from docmind_rag.models.embeddings import build_faiss_index
# from docmind_rag.models.reranker import rerank_docs, protect_exact_matches
# from docmind_rag.services.extraction import extract_pdf_parallel, _extraction_cache
# from docmind_rag.services.chunking import semantic_chunk, raptor_summarize, _summary_cache
# from docmind_rag.services.retrieval import multi_query_retrieve
# from docmind_rag.services.agent import react_agent, roberta_qa, clean_reasoning_answer
# from docmind_rag.events.events import emit_event
# from docmind_rag.utils.helpers import get_pdf_hash, clean_artifacts, clean_chunk_text, normalize_answer
# from docmind_rag.utils.text import (
#     classify_from_context,
#     reorder_by_question,
#     normalize_text,
# )
# from docmind_rag.utils.metrics import (
#     compute_answer_grounding,
#     compute_retrieval_score,
#     compute_context_precision,
#     compute_recall_at_k,
#     semantic_similarity,
   
# )

# # def trust_gate(answer: str, metrics: dict) -> str:
# #     recall = metrics.get("recall_at_k", 0)
# #     grounding = metrics.get("answer_grounding", 0)
# #     length = len(answer.split())

# #     # 1. No reliable context → reject
# #     if recall < 25:
# #         return "not_found"

# #     # 2. Weak grounding → reject
# #     if grounding < 50:
# #         return "not_found"

# #     # 3. Short answers must be very strong
# #     if length <= 3:
# #         if recall >= 70 and grounding >= 70:
# #             return "accepted"
# #         return "not_found"

# #     # 4. Strong signal → accept
# #     if recall >= 50 and grounding >= 60:
# #         return "accepted"
   
# #     # 5. Default safe behavior
# #     else:
# #         return "not_found"
# # ============================================================
# # LOCAL HELPERS
# # ============================================================

# def clean_context_for_llm(chunks):
#     cleaned_chunks = []
#     for chunk in chunks:
#         lines = chunk.split("\n")
#         filtered_lines = []
#         for line in lines:
#             line_strip = line.strip()
#             is_question_like = (
#                 line_strip.endswith("?") or
#                 (len(line_strip.split()) < 15 and "?" in line_strip)
#             )
#             if is_question_like:
#                 continue
#             filtered_lines.append(line)
#         cleaned_chunks.append("\n".join(filtered_lines))
#     return cleaned_chunks


# def is_grounded(answer, retrieved_texts, threshold=0.6):
#     if not answer:
#         return False
#     answer_words = set(answer.lower().split())
#     if not answer_words:
#         return False
#     best_overlap = 0
#     for chunk in retrieved_texts:
#         chunk_words = set(chunk.lower().split())
#         overlap = len(answer_words & chunk_words) / len(answer_words)
#         best_overlap = max(best_overlap, overlap)
#     return best_overlap >= threshold


# # ============================================================
# # NUMERIC INTENT DETECTION
# # ============================================================

# def _extract_numbers(text: str) -> list:
#     return re.findall(r'\b\d+\b', text)


# def _detect_numeric_intent(question: str, chunks: list) -> str:
#     """
#     Detect NAVIGATIONAL or POSITIONAL from chunk structure.
#     Generic — no hardcoded section words.
#     """
#     numbers = _extract_numbers(question)
#     if not numbers:
#         return "NONE"

#     q_words = set(question.lower().split())

#     for chunk in chunks:
#         lines = [l.strip() for l in chunk.split("\n") if l.strip()]
#         if not lines:
#             continue
#         # chunks may have no newlines — take first 10 tokens as heading proxy
#         first_line = " ".join(lines[0].split()[:10]).lower()
#         line_words = set(first_line.split())
#         overlap    = len(q_words & line_words)
#         num_found  = any(re.search(rf'\b{re.escape(num)}\b', first_line) for num in numbers)

#         if num_found:
#             print(f"[NavDebug] first_line='{first_line}' | overlap={overlap} | words={len(first_line.split())}")

#         if num_found and overlap >= 2:  # ← remove word count check entirely
#             return "NAVIGATIONAL"
#     for chunk in chunks:
#         lines       = [l.strip() for l in chunk.split("\n") if l.strip()]
#         short_lines = [l for l in lines if len(l.split()) <= 12]
#         if len(short_lines) >= 3:
#             first = lines[0].lower() if lines else ""
#             if not any(re.search(rf'\b{re.escape(num)}\b', first) for num in numbers):
#                 return "POSITIONAL"

#     return "NONE"


# def _navigate_full_chunks(question: str, all_raw_chunks: list) -> str:
#     numbers = _extract_numbers(question)
#     if not numbers:
#         return ""

#     q_words    = set(question.lower().split())
#     best_line  = ""
#     best_score = 0

#     for chunk in all_raw_chunks:
#         lines = [l.strip() for l in chunk.split("\n") if l.strip()]
#         if not lines:
#             continue
#         first_line  = " ".join(lines[0].split()[:10])  # ← slice to first 10 tokens
#         first_lower = first_line.lower()

#         for num in numbers:
#             if not re.search(rf'\b{re.escape(num)}\b', first_lower):
#                 continue
#             line_words = set(first_lower.split())
#             overlap    = len(q_words & line_words)
#             if overlap >= 2:  # ← remove word count check, raise overlap threshold
#                 score = overlap
#                 if score > best_score:
#                     best_score = score
#                     best_line  = first_line

#     if not best_line:
#         return ""

#     title = best_line
#     title = re.sub(r'^\s*\d+\s*[-—–]+\s*[—–-]?\s*\d*\s*', '', title)  # strip page prefix like "31 --- - 32"
#     title = re.sub(r'[:\-–—]', ' ', title)
#     title = re.sub(r'\s{2,}', ' ', title)
#     return title.strip()

# def _positional_extract(question: str, retrieved_texts: list) -> str:
#     """Extract Nth item from a list. Generic."""
#     numbers = _extract_numbers(question)
#     if not numbers:
#         return ""
#     try:
#         idx = int(numbers[0]) - 1
#     except ValueError:
#         return ""
#     if idx < 0:
#         return ""

#     best_chunk, best_count = "", 0
#     for chunk in retrieved_texts:
#         lines       = [l.strip() for l in chunk.split("\n") if l.strip()]
#         short_lines = [l for l in lines if len(l.split()) <= 12]
#         if len(short_lines) > best_count:
#             best_count = len(short_lines)
#             best_chunk = chunk

#     if not best_chunk:
#         return ""

#     lines       = [l.strip() for l in best_chunk.split("\n") if l.strip()]
#     short_lines = [l for l in lines if len(l.split()) <= 12]
#     if idx < len(short_lines):
#         return re.sub(r'^\s*\d+[\.\)]\s*', '', short_lines[idx]).strip()
#     return ""


# # ============================================================
# # SIGNAL 3 — Structural consistency for NAVIGATIONAL answers
# # ============================================================

# def _check_identifier_grounded(answer: str, retrieved_texts: list) -> bool:
#     """
#     If answer contains a multi-word identifier (word+number or number+word),
#     verify it appears verbatim in retrieved chunks.
#     Generic — no hardcoded section words.
#     """
#     identifiers = re.findall(
#         r'\b[a-zA-Z]+\s+\d+\b|\b\d+\s+[a-zA-Z]+\b',
#         answer.lower()
#     )
#     if not identifiers:
#         return True

#     chunk_text = " ".join(retrieved_texts).lower()
#     for ident in identifiers:
#         if ident not in chunk_text:
#             print(f"[Signal3] ❌ Identifier '{ident}' not in retrieved chunks")
#             return False
#     return True


# # ============================================================
# # REFUSAL — semantic version (local)
# # ============================================================

# _REFUSAL_ANCHOR = "this information is not available in the provided context"


# def _is_refusal_semantic(text: str) -> bool:
#     if not text or len(text.strip()) < 2:
#         return True
#     sim = semantic_similarity(text, [_REFUSAL_ANCHOR])
#     print(f"[Refusal] sim={sim:.3f} | '{text[:50]}'")
#     return sim > 0.72


# # ============================================================
# # LANGGRAPH NODES
# # ============================================================

# def node_extract(state: DocState) -> DocState:
#     extract_start    = time.time()
#     text, page_count = extract_pdf_parallel(state["pdf_path"])
#     extract_time     = time.time() - extract_start
#     state["extracted_text"] = text
#     state["page_count"]     = page_count
#     state["char_count"]     = len(text)
#     state["metrics"]["extraction_time_sec"]  = round(extract_time, 2)
#     state["metrics"]["pages_processed"]      = page_count
#     state["metrics"]["characters_processed"] = len(text)
#     state["metrics"]["words_processed"]      = len(text.split())
#     return state


# def node_chunk(state: DocState) -> DocState:
#     text = state["extracted_text"]
#     state["query_type"] = "QA"

#     summary_chunks, rag_chunks = semantic_chunk(text)

#     # FIX 4 — normalize all chunks after chunking
#     # Converts unicode symbols so retrieval doesn't fail on symbol mismatches
#     rag_chunks     = [normalize_text(c) for c in rag_chunks]
#     summary_chunks = [normalize_text(c) for c in summary_chunks]

#     state["summary_chunks"] = summary_chunks
#     state["chunks"]         = rag_chunks

#     state["metrics"]["summary_chunks"] = len(summary_chunks)
#     state["metrics"]["chunks_created"] = len(rag_chunks)
#     # state["metrics"]["doc_type"]       = state.get("doc_type", "general")
#     state["metrics"]["query_type"]     = state["query_type"]
#     print(f"[Chunk] {len(summary_chunks)} summary chunks | {len(rag_chunks)} RAG chunks")
#     return state


# # def node_summarize(state: DocState) -> DocState:
# #     pdf_hash  = get_pdf_hash(state["pdf_path"])
# #     cache_key = f"{pdf_hash}_{state.get('doc_type', 'general')}"

# #     if cache_key in _summary_cache:
# #         cached = _summary_cache[cache_key]
# #         print("[Summary] ✅ Cache hit")
# #         emit_event(state.get("request_id", ""), "agent_action",
# #                    "⚡ Summary loaded from cache instantly!")
# #         state["answer"] = cached["summary"]
# #         state["metrics"].update(cached["metrics"])
# #         state["metrics"]["type"] = "summary"
# #         return state

# #     summary_start = time.time()
# #     raptor_summarize._request_id = state.get("request_id", "")
# #     summary, map_time, reduce_time = raptor_summarize(
# #         state["summary_chunks"], state.get("doc_type", "general")
# #     )
# #     summary_time = time.time() - summary_start

# #     state["answer"] = summary
# #     metrics_snapshot = {
# #         "summary_time_sec":     round(summary_time, 2),
# #         "summary_length_words": len(summary.split()),
# #         "parallel_workers":     min(MAX_WORKERS, len(state["summary_chunks"])),
# #         "map_time_sec":         round(map_time, 2),
# #         "reduce_time_sec":      round(reduce_time, 2),
# #         "llm_calls":            3,
# #     }
# #     state["metrics"].update(metrics_snapshot)
# #     state["metrics"]["type"] = "summary"
# #     _summary_cache[cache_key] = {"summary": summary, "metrics": metrics_snapshot}
# #     print(f"[Summary] Done ({len(summary.split())} words)")
# #     return state

# def node_summarize(state: DocState) -> DocState:
#     pdf_hash  = get_pdf_hash(state["pdf_path"])
#     cache_key = pdf_hash  # ❌ removed doc_type dependency

#     # ── Cache check ──────────────────────────────────────────
#     if cache_key in _summary_cache:
#         cached = _summary_cache[cache_key]
#         print("[Summary] ✅ Cache hit")
#         emit_event(
#             state.get("request_id", ""),
#             "agent_action",
#             "⚡ Summary loaded from cache instantly!"
#         )
#         state["answer"] = cached["summary"]
#         state["metrics"].update(cached["metrics"])
#         state["metrics"]["type"] = "summary"
#         return state

#     # ── Generate summary ─────────────────────────────────────
#     summary_start = time.time()
#     raptor_summarize._request_id = state.get("request_id", "")

#     # ❌ removed doc_type argument
#     summary, map_time, reduce_time = raptor_summarize(
#         state["summary_chunks"]
#     )

#     summary_time = time.time() - summary_start

#     # ── Store result ─────────────────────────────────────────
#     state["answer"] = summary

#     metrics_snapshot = {
#         "summary_time_sec":     round(summary_time, 2),
#         "summary_length_words": len(summary.split()),
#         "parallel_workers":     min(MAX_WORKERS, len(state["summary_chunks"])),
#         "map_time_sec":         round(map_time, 2),
#         "reduce_time_sec":      round(reduce_time, 2),
#         "llm_calls":            3,
#     }

#     state["metrics"].update(metrics_snapshot)
#     state["metrics"]["type"] = "summary"

#     # ── Cache result ─────────────────────────────────────────
#     _summary_cache[cache_key] = {
#         "summary": summary,
#         "metrics": metrics_snapshot
#     }

#     print(f"[Summary] Done ({len(summary.split())} words)")
#     return state

# def node_qa(state: DocState) -> DocState:
#     qa_start_t = time.time()
#     question   = state["question"]
#     request_id = state.get("request_id", "")
#     all_chunks = state["chunks"]

#     # FIX 4 — normalize question the same way chunks were normalized
#     question = normalize_text(question)

#     # Raw strings for navigational scanning — must NOT be cleaned
#     all_raw = [
#         d if isinstance(d, str) else d.page_content
#         for d in all_chunks
#     ]

#     pdf_hash    = get_pdf_hash(state["pdf_path"])
#     faiss_index = build_faiss_index(all_chunks, pdf_hash)

#     print(f"[QA] START | {len(all_chunks)} chunks")

#     recall_score      = 0.0
#     retrieval_score   = 0.0
#     context_precision = 0.0
#     grounding         = 0.0
#     llm_calls         = 0
#     model_used        = "llama"
#     confidence        = 0.0
#     decision_type     = "accepted"
#     retrieved         = []
#     retrieved_texts   = []
#     answer            = ""

#     # ── STEP 1: RETRIEVE ─────────────────────────────────────
#     retrieved = multi_query_retrieve(
#         question, faiss_index,
#         k=50,
#         all_chunks=all_chunks,
#         query_type="FACTUAL_QA"
#     )


   
#     # 🔥 STEP 2 — RERANK + SAFE FALLBACK

#     retrieved, reranker_top, _ = rerank_docs(
#         question, retrieved, top_k=8, apply_pruning=True
#     )

#     retrieved = protect_exact_matches(
#         question, retrieved, all_chunks, top_k=8
#     )

#     # 🔥 FIX 1 — fallback must replace retrieved (NOT just texts)
#     if not retrieved or len(retrieved) < 3:
#         print("[QA] ⚠️ Reranker too aggressive → fallback to initial chunks")
#         retrieved = all_chunks[:8]

#     # 🔥 FIX 2 — safe conversion
#     retrieved_texts = [
#         d.page_content if hasattr(d, "page_content") else str(d)
#         for d in retrieved
#     ]
#     if reranker_top < -5:
#         print("[Guard] ⚠️ Weak reranker score — continuing")

#     # ── STEP 3: METRICS ───────────────────────────────────────
#     retrieval_score   = compute_retrieval_score(question, retrieved)
#     context_precision = compute_context_precision(question, retrieved)
#     recall_score      = compute_recall_at_k(
#         question, retrieved, all_chunks, k=len(retrieved)
#     )

#     # ── FIX 6 — k expansion: low recall OR numeric question ───
#     is_numeric_question = len(_extract_numbers(question)) > 0
#     if recall_score < 25 or is_numeric_question:
#         print("[QA] ⚠️ Expanding retrieval (low recall or numeric question)")
#         expanded = multi_query_retrieve(
#             question, faiss_index, k=30,
#             all_chunks=all_chunks, query_type="FACTUAL_QA"
#         )
#         if len(expanded) > len(retrieved):
#             retrieved = expanded
#             retrieved_texts = [
#                 d.page_content if hasattr(d, "page_content") else str(d)
#                 for d in retrieved
#             ]

#     # ── FIX 1 — Numeric intent: retrieved first, fallback to all_raw ──
#     numeric_intent = _detect_numeric_intent(question, retrieved_texts)

#     if numeric_intent == "NONE" and is_numeric_question:
#         print("[Navigate] Retrying intent detection on full document...")
#         numeric_intent = _detect_numeric_intent(question, all_raw)
#         if numeric_intent != "NONE":
#             print(f"[Navigate] Intent found in full document: {numeric_intent}")

#     # ── NAVIGATIONAL ──────────────────────────────────────────
#     if numeric_intent == "NAVIGATIONAL":
#         print(f"[Routing] Numeric intent → NAVIGATIONAL")
#         title = _navigate_full_chunks(question, all_raw)
#         if title:
#             print(f"[Navigate] Extracted title: '{title}'")
#             grounding  = compute_answer_grounding(title, retrieved_texts, question)
#             confidence = round(grounding / 100, 3)
#             qa_time    = time.time() - qa_start_t
#             state["answer"]      = title
#             state["query_type"]  = "NAVIGATIONAL"
#             _write_metrics(state, "navigational", "navigational", grounding,
#                            confidence, retrieval_score, context_precision,
#                            recall_score, llm_calls, retrieved, qa_time)
#             return state
#         print(f"[Navigate] Extraction failed — falling through to LLM")

#     # ── POSITIONAL ────────────────────────────────────────────
#     elif numeric_intent == "POSITIONAL":
#         print(f"[Routing] Numeric intent → POSITIONAL")
#         pos_answer = _positional_extract(question, retrieved_texts)
#         if pos_answer:
#             print(f"[Positional] Extracted: '{pos_answer}'")
#             grounding  = compute_answer_grounding(pos_answer, retrieved_texts, question)
#             confidence = round(grounding / 100, 3)
#             qa_time    = time.time() - qa_start_t
#             state["answer"]      = pos_answer
#             state["query_type"]  = "POSITIONAL"
#             _write_metrics(state, "positional", "positional", grounding,
#                            confidence, retrieval_score, context_precision,
#                            recall_score, llm_calls, retrieved, qa_time)
#             return state
#         print(f"[Positional] Extraction failed — falling through to LLM")

#     # ── Normal routing ────────────────────────────────────────
#     query_type = classify_from_context(question, retrieved_texts)
#     state["query_type"] = query_type
#     print(f"[Routing] Context-based → {query_type}")

#     # ── STEP 4: ANSWER GENERATION ─────────────────────────────

#     if query_type == "FULL_SUMMARY":
#         retrieved_texts = clean_context_for_llm(retrieved_texts)
#         ranked  = reorder_by_question(question, retrieved_texts)
#         context = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:8])
#         structured_context = "\n".join(
#             f"[CONTEXT CHUNK]\n{c}" for c in context.split("\n\n---\n\n")
#         )
#         emit_event(request_id, "stream_start", "✍️ Generating answer...")
#         answer, _ = call_llama_streaming(
#             QA_PROMPT.format(context=structured_context[:2500], question=question),
#             request_id=request_id, temperature=0.0
#         )
#         answer     = clean_artifacts(answer).strip()
#         model_used = "llama_summary"
#         llm_calls  = 1

#     elif query_type == "MULTIPART_QA":
#         retrieved_texts = clean_context_for_llm(retrieved_texts)
#         ranked  = reorder_by_question(question, retrieved_texts)
#         # FIX 3 — 4000 chars for MULTIPART to capture all list items
#         context = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:8])
#         emit_event(request_id, "agent_start",
#                    f"🤖 MULTIPART | {len(retrieved)} chunks")
#         emit_event(request_id, "stream_start", "✍️ Generating answer...")
#         answer, _ = call_llama_streaming(
#             QA_PROMPT.format(context=context[:4000], question=question),
#             request_id=request_id, temperature=0.0
#         )
#         answer     = clean_artifacts(answer).strip()
#         model_used = "llama_multipart"
#         llm_calls  = 1

#     else:
        
#         # FACTUAL_QA / VERIFICATION_QA

#         if not retrieved_texts:
#             print("[QA] ❌ No context → NOT FOUND")
#             state["answer"] = "This information is not present in the document."
#             _write_metrics(state, "not_found", "no_context",
#                         0.0, 0.0, retrieval_score, context_precision,
#                         recall_score, llm_calls, retrieved,
#                         time.time() - qa_start_t)
#             return state

#         # 🔥 FIX 5 — SIMPLE QA ROUTING (ADD HERE)

#         context = "\n".join(retrieved_texts)

#         if (
#             query_type == "FACTUAL_QA"
#             and len(question.split()) <= 10
#             and not is_numeric_question
#             and recall_score >= 60
#         ):
#             print("[QA] ⚡ Direct QA (no ReAct)")

#             react_ans, _ = call_llama_streaming(
#                 f"""
#             Answer the question using ONLY the given context.
#             Return ONLY the exact answer phrase from the context.
#             Do NOT explain. Do NOT say 'not found' unless absolutely missing.

#             Context:
#             {context[:2000]}

#             Question: {question}

#             Answer:
#             """,
#                 request_id=request_id,
#                 temperature=0.0
#             )
#             llm_calls += 1
#             model_used = "llama_direct"

#         else:
#             react_ans, model_used, _, _, _, _, react_calls, _ = react_agent(
#     question,
#     faiss_index,
#     query_type,
#     all_chunks,
#     request_id,
#     recall_score
# )

#         # react_ans, model_used, _, _, _, _, react_calls, retrieved_texts = react_agent(
#         #     question, faiss_index, query_type, all_chunks, request_id
#         # )
 
#         react_ans = clean_artifacts(react_ans).strip().strip('"').strip("'")
#         # react_ans = clean_reasoning_answer(react_ans, question)
#         print(f"[QA] Answer: '{react_ans[:60]}'")

#         # FIX 2 — Short answer bypass: skip for VERIFICATION_QA
#         words         = react_ans.split()
#         content_words = [w for w in words if len(w) > 2 and not w.isdigit()]

#         # if (
#         #     query_type != "VERIFICATION_QA"
#         #     and len(words) <= 3
#         #     and len(content_words) >= 1
#         #     and not _is_refusal_answer(react_ans)
#         # ):
#         #     print("[QA] ⚡ Short answer accepted")
#         #     grounding  = compute_answer_grounding(react_ans, retrieved_texts, question)
#         #     confidence = grounding / 100
#         #     qa_time    = time.time() - qa_start_t
#         #     state["answer"] = react_ans
#         #     _write_metrics(state, model_used, "short_answer",
#         #                    grounding, confidence, retrieval_score,
#         #                    context_precision, recall_score, llm_calls,
#         #                    retrieved, qa_time)
#         #     return state
#         print(f"[DEBUG] react_ans before refusal check: repr={repr(react_ans)}")

#         refusal_phrases = [
#             "not present",
#             "not mentioned",
#             "not available",
#             "no information",
#             "cannot find",
#             "not in the document"
#         ]

#         if any(p in react_ans.lower() for p in refusal_phrases):
#             print("[QA] ⚠️ Refusal detected")
#             qa_time = time.time() - qa_start_t
#             state["answer"] = "This information is not present in the document."
#             _write_metrics(state, "not_found", "not_found",
#                         75.0, 0.75, retrieval_score, context_precision,
#                         recall_score, llm_calls, retrieved, qa_time)
#             return state
#         # print(f"[DEBUG] react_ans before refusal check: repr={repr(react_ans)}")
#         # # Refusal
#         # if _is_refusal_answer(react_ans):
#         #     print("[QA] ⚠️ Refusal detected")
#         #     qa_time = time.time() - qa_start_t
#         #     state["answer"] = "This information is not present in the document."
#         #     _write_metrics(state, "not_found", "not_found",
#         #                    75.0, 0.75, retrieval_score, context_precision,
#         #                    recall_score, llm_calls, retrieved, qa_time)
#         #     return state

#         grounding_score = compute_answer_grounding(react_ans, retrieved_texts, question)
#         # 🔥 STRICT VERIFICATION CHECK
#         # if query_type == "VERIFICATION_QA":
#         #     if grounding_score < 80:
#         #         print("[QA] ❌ Verification failed → NOT FOUND")
#         #         answer = "This information is not present in the document."
#         #         decision_type = "verification_failed"
#         #     else:
#         #         answer = react_ans
#         #         decision_type = "accepted"
#         if query_type == "VERIFICATION_QA":

#             words = react_ans.split()

#             # 🔥 Allow short verification answers (Yes/No)
#             if len(words) <= 3:
#                 answer = react_ans
#                 decision_type = "accepted"

#             elif grounding_score < 80:
#                 print("[QA] ❌ Verification failed → NOT FOUND")
#                 answer = "This information is not present in the document."
#                 decision_type = "verification_failed"

#             else:
#                 answer = react_ans
#                 decision_type = "accepted"

#             grounding  = compute_answer_grounding(answer, retrieved_texts, question)
#             confidence = round(grounding / 100, 3)
#             qa_time    = time.time() - qa_start_t
#             state["answer"] = normalize_answer(answer)
#             _write_metrics(state, model_used, decision_type, grounding,
#                         confidence, retrieval_score, context_precision,
#                         recall_score, llm_calls, retrieved, qa_time)
#             return state

#         # 🔥 TRUST GATE (FINAL CLEAN VERSION)

#         recall = recall_score
#         grounding = grounding_score
#         length = len(react_ans.split())

#         # 1. No reliable context → reject
#         if recall < 25:
#             print("[QA] ❌ Low recall → NOT FOUND")
#             answer = "This information is not present in the document."
#             decision_type = "low_recall"

#         # 2. Very weak grounding → reject
#         elif grounding < 40:
#             print("[QA] ❌ Very low grounding → NOT FOUND")
#             answer = "This information is not present in the document."
#             decision_type = "low_grounding"

#         # # 3. Short answers must be VERY strong
#         # elif length <= 3:
#         #     if recall >= 70 and grounding >= 70:
#         #         answer = react_ans
#         #         decision_type = "accepted"
        
#         #     else:
#         #         print("[QA] ❌ Weak short answer → NOT FOUND")
#         #         answer = "This information is not present in the document."
#         #         decision_type = "weak_short"

#         # # 4. Strong signals → accept
#         # elif recall >= 50 and grounding >= 60:
#         #     answer = react_ans
#         #     decision_type = "accepted"

#         # # 5. Otherwise reject safely
#         # else:
#         #     print("[QA] ❌ Uncertain → NOT FOUND")
#         #     answer = "This information is not present in the document."
#         #     decision_type = "uncertain"
#         elif length <= 3:
#             if grounding >= 60:
#                 answer = react_ans
#                 decision_type = "accepted"
#             else:
#                 print("[QA] ❌ Weak short answer → NOT FOUND")
#                 answer = "This information is not present in the document."
#                 decision_type = "weak_short"

#         # 4. Strong signals → accept
#         elif recall >= 50 and grounding >= 60:
#             answer = react_ans
#             decision_type = "accepted"

#         # 5. Otherwise reject safely
#         else:
#             print("[QA] ❌ Uncertain → NOT FOUND")
#             answer = "This information is not present in the document."
#             decision_type = "uncertain"
#     # ── STEP 5: METRICS ───────────────────────────────────────
#     grounding  = compute_answer_grounding(answer, retrieved_texts, question)
#     confidence = round(grounding / 100, 3)
#     qa_time    = time.time() - qa_start_t

#     print(f"[QA] Done in {qa_time:.1f}s | model={model_used} | "
#           f"grounding={grounding:.1f}% | recall={recall_score:.1f}% | "
#           f"confidence={confidence:.3f} | decision={decision_type}")

#     if not answer.strip():
#         answer = "Could not find a relevant answer in the PDF."
#     else:
#         answer = normalize_answer(answer)

#     state["answer"] = answer
#     _write_metrics(state, model_used, decision_type, grounding,
#                    confidence, retrieval_score, context_precision,
#                    recall_score, llm_calls, retrieved, qa_time)
#     return state
# def node_validate(state: DocState) -> DocState:
#     answer = state["answer"]
#     retry  = state.get("retry_count", 0)

#     # ── Retry for empty/very weak answers ────────────────────
#     if len(answer.strip()) < 3 and retry < 2:
#         state["retry_count"] = retry + 1
#         state["answer"]      = ""
#         return state

#     total_time    = time.time() - state["start_time"]
#     output_words  = len(answer.split())
#     output_tokens = output_words * 1.3

#     extract_time  = state["metrics"].get("extraction_time_sec", 0)
#     llm_time      = max(total_time - extract_time, 1)
#     tps           = round(output_tokens / llm_time, 2) if llm_time > 0 else 0

#     m = state["metrics"]

#     # ── Core metrics ─────────────────────────────────────────
#     m["response_time_sec"]    = round(total_time, 2)
#     m["extraction_time_sec"]  = m.get("extraction_time_sec", 0)
#     m["pages_processed"]      = state.get("page_count", 0)
#     m["characters_processed"] = state.get("char_count", 0)
#     m["words_processed"]      = len(state.get("extracted_text", "").split())

#     # ── Type-specific metrics ────────────────────────────────
#     if m.get("type") == "summary":
#         m["summary_time_sec"]     = m.get("summary_time_sec", 0)
#         m["summary_length_words"] = len(answer.split())

#     if m.get("type") == "qa":
#         m["qa_time_sec"]      = m.get("qa_time_sec", 0)
#         m["confidence_score"] = m.get("confidence_score", 0)

#     # ── Performance ──────────────────────────────────────────
#     m["ttft_sec"]        = round(total_time, 2)
#     m["e2e_latency_sec"] = round(total_time, 2)
#     m["tps"]             = tps

#     # ❌ REMOVED doc_type (no longer used anywhere)
#     # m["doc_type"] = state.get("doc_type", "general")

#     # ── Context info ─────────────────────────────────────────
#     m["query_type"]     = state.get("query_type", "")
#     m["chunks_created"] = m.get("chunks_created", 0)
#     m["retry_count"]    = retry

#     # ── Model + retrieval metrics ────────────────────────────
#     m["model_used"]        = m.get("model_used", "llama_react")
#     m["llm_calls"]         = m.get("llm_calls", 0)
#     m["retrieval_score"]   = m.get("retrieval_score", 0)
#     m["context_precision"] = m.get("context_precision", 0)
#     m["answer_grounding"]  = m.get("answer_grounding", 0)
#     m["recall_at_k"]       = m.get("recall_at_k", 0)

#     # ── Summary-specific ─────────────────────────────────────
#     if m.get("type") == "summary":
#         m["parallel_workers"] = m.get("parallel_workers", 0)
#         m["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
#         m["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)

#     # ── QA-specific ──────────────────────────────────────────
#     if m.get("type") == "qa":
#         m["chunks_retrieved"] = m.get("chunks_retrieved", 0)
#         m["decision_type"]    = m.get("decision_type", "accepted")
#         m["confidence_raw"]   = m.get("confidence_raw", 0.0)

#     state["metrics"] = m
#     return state

# # def node_validate(state: DocState) -> DocState:
# #     answer = state["answer"]
# #     retry  = state.get("retry_count", 0)

# #     if len(answer.strip()) < 3 and retry < 2:
# #         state["retry_count"] = retry + 1
# #         state["answer"]      = ""
# #         return state

# #     total_time    = time.time() - state["start_time"]
# #     output_words  = len(answer.split())
# #     output_tokens = output_words * 1.3
# #     extract_time  = state["metrics"].get("extraction_time_sec", 0)
# #     llm_time      = max(total_time - extract_time, 1)
# #     tps           = round(output_tokens / llm_time, 2) if llm_time > 0 else 0
# #     m             = state["metrics"]

# #     m["response_time_sec"]    = round(total_time, 2)
# #     m["extraction_time_sec"]  = m.get("extraction_time_sec", 0)
# #     m["pages_processed"]      = state.get("page_count", 0)
# #     m["characters_processed"] = state.get("char_count", 0)
# #     m["words_processed"]      = len(state.get("extracted_text", "").split())

# #     if m.get("type") == "summary":
# #         m["summary_time_sec"]     = m.get("summary_time_sec", 0)
# #         m["summary_length_words"] = len(answer.split())
# #     if m.get("type") == "qa":
# #         m["qa_time_sec"]      = m.get("qa_time_sec", 0)
# #         m["confidence_score"] = m.get("confidence_score", 0)

# #     m["ttft_sec"]          = round(total_time, 2)
# #     m["e2e_latency_sec"]   = round(total_time, 2)
# #     m["tps"]               = tps
# #     m["doc_type"]          = state.get("doc_type", "general")
# #     m["query_type"]        = state.get("query_type", "")
# #     m["chunks_created"]    = m.get("chunks_created", 0)
# #     m["retry_count"]       = retry
# #     m["model_used"]        = m.get("model_used", "llama_react")
# #     m["llm_calls"]         = m.get("llm_calls", 0)
# #     m["retrieval_score"]   = m.get("retrieval_score", 0)
# #     m["context_precision"] = m.get("context_precision", 0)
# #     m["answer_grounding"]  = m.get("answer_grounding", 0)
# #     m["recall_at_k"]       = m.get("recall_at_k", 0)

# #     if m.get("type") == "summary":
# #         m["parallel_workers"] = m.get("parallel_workers", 0)
# #         m["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
# #         m["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)
# #     if m.get("type") == "qa":
# #         m["chunks_retrieved"] = m.get("chunks_retrieved", 0)
# #         m["decision_type"]    = m.get("decision_type", "accepted")
# #         m["confidence_raw"]   = m.get("confidence_raw", 0.0)

# #     state["metrics"] = m
# #     return state


# def _write_metrics(state, model_used, decision_type, grounding,
#                    confidence, retrieval_score, context_precision,
#                    recall_score, llm_calls, retrieved, qa_time):
#     m = state.setdefault("metrics", {})
#     m["qa_time_sec"]       = round(qa_time, 2)
#     m["confidence_score"]  = round(confidence * 100, 2)
#     m["retrieval_score"]   = retrieval_score
#     m["context_precision"] = context_precision
#     m["answer_grounding"]  = grounding
#     m["recall_at_k"]       = recall_score
#     m["llm_calls"]         = llm_calls
#     m["model_used"]        = model_used
#     m["chunks_retrieved"]  = len(retrieved)
#     m["type"]              = "qa"
#     m["decision_type"]     = decision_type
#     m["confidence_raw"]    = round(confidence, 4)


































# # chapter 4 works
# import time
# import re
# import spacy
# _nlp = spacy.load("en_core_web_lg")

# from docmind_rag.config.settings import (
#     MAX_WORKERS, FACTUAL_TOP_K, MULTIPART_TOP_K
# )
# from docmind_rag.utils.text import extract_named_entities, expand_answer
# from docmind_rag.core.state import DocState
# from docmind_rag.core.prompts import QA_PROMPT
# from docmind_rag.models.llm import call_llama_streaming
# from docmind_rag.models.embeddings import build_faiss_index
# from docmind_rag.models.reranker import rerank_docs, protect_exact_matches
# from docmind_rag.services.extraction import extract_pdf_parallel, _extraction_cache
# from docmind_rag.services.chunking import semantic_chunk, raptor_summarize, _summary_cache
# from docmind_rag.services.retrieval import multi_query_retrieve
# from docmind_rag.services.agent import react_agent, roberta_qa, clean_reasoning_answer
# from docmind_rag.events.events import emit_event
# from docmind_rag.utils.helpers import get_pdf_hash, clean_artifacts, clean_chunk_text, normalize_answer
# from docmind_rag.utils.text import (
#     classify_from_context,
#     reorder_by_question,
#     normalize_text,
# )
# from docmind_rag.utils.metrics import (
#     compute_answer_grounding,
#     compute_retrieval_score,
#     compute_context_precision,
#     compute_recall_at_k,
#     semantic_similarity,
#     _is_refusal_answer,
# )

# # def trust_gate(answer: str, metrics: dict) -> str:
# #     recall = metrics.get("recall_at_k", 0)
# #     grounding = metrics.get("answer_grounding", 0)
# #     length = len(answer.split())

# #     # 1. No reliable context → reject
# #     if recall < 25:
# #         return "not_found"

# #     # 2. Weak grounding → reject
# #     if grounding < 50:
# #         return "not_found"

# #     # 3. Short answers must be very strong
# #     if length <= 3:
# #         if recall >= 70 and grounding >= 70:
# #             return "accepted"
# #         return "not_found"

# #     # 4. Strong signal → accept
# #     if recall >= 50 and grounding >= 60:
# #         return "accepted"
   
# #     # 5. Default safe behavior
# #     else:
# #         return "not_found"
# # ============================================================
# # LOCAL HELPERS
# # ============================================================

# def clean_context_for_llm(chunks):
#     cleaned_chunks = []
#     for chunk in chunks:
#         lines = chunk.split("\n")
#         filtered_lines = []
#         for line in lines:
#             line_strip = line.strip()
#             is_question_like = (
#                 line_strip.endswith("?") or
#                 (len(line_strip.split()) < 15 and "?" in line_strip)
#             )
#             if is_question_like:
#                 continue
#             filtered_lines.append(line)
#         cleaned_chunks.append("\n".join(filtered_lines))
#     return cleaned_chunks


# def is_grounded(answer, retrieved_texts, threshold=0.6):
#     if not answer:
#         return False
#     answer_words = set(answer.lower().split())
#     if not answer_words:
#         return False
#     best_overlap = 0
#     for chunk in retrieved_texts:
#         chunk_words = set(chunk.lower().split())
#         overlap = len(answer_words & chunk_words) / len(answer_words)
#         best_overlap = max(best_overlap, overlap)
#     return best_overlap >= threshold


# # ============================================================
# # NUMERIC INTENT DETECTION
# # ============================================================

# def _extract_numbers(text: str) -> list:
#     return re.findall(r'\b\d+\b', text)


# def _detect_numeric_intent(question: str, chunks: list) -> str:
#     """
#     Detect NAVIGATIONAL or POSITIONAL from chunk structure.
#     Generic — no hardcoded section words.
#     """
#     numbers = _extract_numbers(question)
#     if not numbers:
#         return "NONE"

#     q_words = set(question.lower().split())

#     for chunk in chunks:
#         lines = [l.strip() for l in chunk.split("\n") if l.strip()]
#         if not lines:
#             continue
#         # chunks may have no newlines — take first 10 tokens as heading proxy
#         first_line = " ".join(lines[0].split()[:10]).lower()
#         line_words = set(first_line.split())
#         overlap    = len(q_words & line_words)
#         num_found  = any(re.search(rf'\b{re.escape(num)}\b', first_line) for num in numbers)

#         if num_found:
#             print(f"[NavDebug] first_line='{first_line}' | overlap={overlap} | words={len(first_line.split())}")

#         if num_found and overlap >= 2:  # ← remove word count check entirely
#             return "NAVIGATIONAL"
#     for chunk in chunks:
#         lines       = [l.strip() for l in chunk.split("\n") if l.strip()]
#         short_lines = [l for l in lines if len(l.split()) <= 12]
#         if len(short_lines) >= 3:
#             first = lines[0].lower() if lines else ""
#             if not any(re.search(rf'\b{re.escape(num)}\b', first) for num in numbers):
#                 return "POSITIONAL"

#     return "NONE"


# def _navigate_full_chunks(question: str, all_raw_chunks: list) -> str:
#     numbers = _extract_numbers(question)
#     if not numbers:
#         return ""

#     q_words    = set(question.lower().split())
#     best_line  = ""
#     best_score = 0

#     for chunk in all_raw_chunks:
#         lines = [l.strip() for l in chunk.split("\n") if l.strip()]
#         if not lines:
#             continue
#         first_line  = " ".join(lines[0].split()[:10])  # ← slice to first 10 tokens
#         first_lower = first_line.lower()

#         for num in numbers:
#             if not re.search(rf'\b{re.escape(num)}\b', first_lower):
#                 continue
#             line_words = set(first_lower.split())
#             overlap    = len(q_words & line_words)
#             if overlap >= 2:  # ← remove word count check, raise overlap threshold
#                 score = overlap
#                 if score > best_score:
#                     best_score = score
#                     best_line  = first_line

#     if not best_line:
#         return ""

#     title = best_line
#     title = re.sub(r'^\s*\d+\s*[-—–]+\s*[—–-]?\s*\d*\s*', '', title)  # strip page prefix like "31 --- - 32"
#     title = re.sub(r'[:\-–—]', ' ', title)
#     title = re.sub(r'\s{2,}', ' ', title)
#     return title.strip()

# def _positional_extract(question: str, retrieved_texts: list) -> str:
#     """Extract Nth item from a list. Generic."""
#     numbers = _extract_numbers(question)
#     if not numbers:
#         return ""
#     try:
#         idx = int(numbers[0]) - 1
#     except ValueError:
#         return ""
#     if idx < 0:
#         return ""

#     best_chunk, best_count = "", 0
#     for chunk in retrieved_texts:
#         lines       = [l.strip() for l in chunk.split("\n") if l.strip()]
#         short_lines = [l for l in lines if len(l.split()) <= 12]
#         if len(short_lines) > best_count:
#             best_count = len(short_lines)
#             best_chunk = chunk

#     if not best_chunk:
#         return ""

#     lines       = [l.strip() for l in best_chunk.split("\n") if l.strip()]
#     short_lines = [l for l in lines if len(l.split()) <= 12]
#     if idx < len(short_lines):
#         return re.sub(r'^\s*\d+[\.\)]\s*', '', short_lines[idx]).strip()
#     return ""


# # ============================================================
# # SIGNAL 3 — Structural consistency for NAVIGATIONAL answers
# # ============================================================

# def _check_identifier_grounded(answer: str, retrieved_texts: list) -> bool:
#     """
#     If answer contains a multi-word identifier (word+number or number+word),
#     verify it appears verbatim in retrieved chunks.
#     Generic — no hardcoded section words.
#     """
#     identifiers = re.findall(
#         r'\b[a-zA-Z]+\s+\d+\b|\b\d+\s+[a-zA-Z]+\b',
#         answer.lower()
#     )
#     if not identifiers:
#         return True

#     chunk_text = " ".join(retrieved_texts).lower()
#     for ident in identifiers:
#         if ident not in chunk_text:
#             print(f"[Signal3] ❌ Identifier '{ident}' not in retrieved chunks")
#             return False
#     return True


# # ============================================================
# # REFUSAL — semantic version (local)
# # ============================================================

# _REFUSAL_ANCHOR = "this information is not available in the provided context"


# def _is_refusal_semantic(text: str) -> bool:
#     if not text or len(text.strip()) < 2:
#         return True
#     sim = semantic_similarity(text, [_REFUSAL_ANCHOR])
#     print(f"[Refusal] sim={sim:.3f} | '{text[:50]}'")
#     return sim > 0.72


# # ============================================================
# # LANGGRAPH NODES
# # ============================================================

# def node_extract(state: DocState) -> DocState:
#     extract_start    = time.time()
#     text, page_count = extract_pdf_parallel(state["pdf_path"])
#     extract_time     = time.time() - extract_start
#     state["extracted_text"] = text
#     state["page_count"]     = page_count
#     state["char_count"]     = len(text)
#     state["metrics"]["extraction_time_sec"]  = round(extract_time, 2)
#     state["metrics"]["pages_processed"]      = page_count
#     state["metrics"]["characters_processed"] = len(text)
#     state["metrics"]["words_processed"]      = len(text.split())
#     return state


# def node_chunk(state: DocState) -> DocState:
#     text = state["extracted_text"]
#     state["query_type"] = "QA"

#     summary_chunks, rag_chunks = semantic_chunk(text)

#     # FIX 4 — normalize all chunks after chunking
#     # Converts unicode symbols so retrieval doesn't fail on symbol mismatches
#     rag_chunks     = [normalize_text(c) for c in rag_chunks]
#     summary_chunks = [normalize_text(c) for c in summary_chunks]

#     state["summary_chunks"] = summary_chunks
#     state["chunks"]         = rag_chunks

#     state["metrics"]["summary_chunks"] = len(summary_chunks)
#     state["metrics"]["chunks_created"] = len(rag_chunks)
#     # state["metrics"]["doc_type"]       = state.get("doc_type", "general")
#     state["metrics"]["query_type"]     = state["query_type"]
#     print(f"[Chunk] {len(summary_chunks)} summary chunks | {len(rag_chunks)} RAG chunks")
#     return state


# # def node_summarize(state: DocState) -> DocState:
# #     pdf_hash  = get_pdf_hash(state["pdf_path"])
# #     cache_key = f"{pdf_hash}_{state.get('doc_type', 'general')}"

# #     if cache_key in _summary_cache:
# #         cached = _summary_cache[cache_key]
# #         print("[Summary] ✅ Cache hit")
# #         emit_event(state.get("request_id", ""), "agent_action",
# #                    "⚡ Summary loaded from cache instantly!")
# #         state["answer"] = cached["summary"]
# #         state["metrics"].update(cached["metrics"])
# #         state["metrics"]["type"] = "summary"
# #         return state

# #     summary_start = time.time()
# #     raptor_summarize._request_id = state.get("request_id", "")
# #     summary, map_time, reduce_time = raptor_summarize(
# #         state["summary_chunks"], state.get("doc_type", "general")
# #     )
# #     summary_time = time.time() - summary_start

# #     state["answer"] = summary
# #     metrics_snapshot = {
# #         "summary_time_sec":     round(summary_time, 2),
# #         "summary_length_words": len(summary.split()),
# #         "parallel_workers":     min(MAX_WORKERS, len(state["summary_chunks"])),
# #         "map_time_sec":         round(map_time, 2),
# #         "reduce_time_sec":      round(reduce_time, 2),
# #         "llm_calls":            3,
# #     }
# #     state["metrics"].update(metrics_snapshot)
# #     state["metrics"]["type"] = "summary"
# #     _summary_cache[cache_key] = {"summary": summary, "metrics": metrics_snapshot}
# #     print(f"[Summary] Done ({len(summary.split())} words)")
# #     return state

# def node_summarize(state: DocState) -> DocState:
#     pdf_hash  = get_pdf_hash(state["pdf_path"])
#     cache_key = pdf_hash  # ❌ removed doc_type dependency

#     # ── Cache check ──────────────────────────────────────────
#     if cache_key in _summary_cache:
#         cached = _summary_cache[cache_key]
#         print("[Summary] ✅ Cache hit")
#         emit_event(
#             state.get("request_id", ""),
#             "agent_action",
#             "⚡ Summary loaded from cache instantly!"
#         )
#         state["answer"] = cached["summary"]
#         state["metrics"].update(cached["metrics"])
#         state["metrics"]["type"] = "summary"
#         return state

#     # ── Generate summary ─────────────────────────────────────
#     summary_start = time.time()
#     raptor_summarize._request_id = state.get("request_id", "")

#     # ❌ removed doc_type argument
#     summary, map_time, reduce_time = raptor_summarize(
#         state["summary_chunks"]
#     )

#     summary_time = time.time() - summary_start

#     # ── Store result ─────────────────────────────────────────
#     state["answer"] = summary

#     metrics_snapshot = {
#         "summary_time_sec":     round(summary_time, 2),
#         "summary_length_words": len(summary.split()),
#         "parallel_workers":     min(MAX_WORKERS, len(state["summary_chunks"])),
#         "map_time_sec":         round(map_time, 2),
#         "reduce_time_sec":      round(reduce_time, 2),
#         "llm_calls":            3,
#     }

#     state["metrics"].update(metrics_snapshot)
#     state["metrics"]["type"] = "summary"

#     # ── Cache result ─────────────────────────────────────────
#     _summary_cache[cache_key] = {
#         "summary": summary,
#         "metrics": metrics_snapshot
#     }

#     print(f"[Summary] Done ({len(summary.split())} words)")
#     return state

# def node_qa(state: DocState) -> DocState:
#     qa_start_t = time.time()
#     question   = state["question"]
#     request_id = state.get("request_id", "")
#     all_chunks = state["chunks"]

#     # FIX 4 — normalize question the same way chunks were normalized
#     question = normalize_text(question)

#     # Raw strings for navigational scanning — must NOT be cleaned
#     all_raw = [
#         d if isinstance(d, str) else d.page_content
#         for d in all_chunks
#     ]

#     pdf_hash    = get_pdf_hash(state["pdf_path"])
#     faiss_index = build_faiss_index(all_chunks, pdf_hash)

#     print(f"[QA] START | {len(all_chunks)} chunks")

#     recall_score      = 0.0
#     retrieval_score   = 0.0
#     context_precision = 0.0
#     grounding         = 0.0
#     llm_calls         = 0
#     model_used        = "llama"
#     confidence        = 0.0
#     decision_type     = "accepted"
#     retrieved         = []
#     retrieved_texts   = []
#     answer            = ""

#     # ── STEP 1: RETRIEVE ─────────────────────────────────────
#     retrieved = multi_query_retrieve(
#         question, faiss_index,
#         k=50,
#         all_chunks=all_chunks,
#         query_type="FACTUAL_QA"
#     )


   
#     # 🔥 STEP 2 — RERANK + SAFE FALLBACK

#     retrieved, reranker_top, _ = rerank_docs(
#         question, retrieved, top_k=8, apply_pruning=True
#     )

#     retrieved = protect_exact_matches(
#         question, retrieved, all_chunks, top_k=8
#     )

#     # 🔥 FIX 1 — fallback must replace retrieved (NOT just texts)
#     if not retrieved or len(retrieved) < 3:
#         print("[QA] ⚠️ Reranker too aggressive → fallback to initial chunks")
#         retrieved = all_chunks[:8]

#     # 🔥 FIX 2 — safe conversion
#     retrieved_texts = [
#         d.page_content if hasattr(d, "page_content") else str(d)
#         for d in retrieved
#     ]
#     if reranker_top < -5:
#         print("[Guard] ⚠️ Weak reranker score — continuing")

#     # ── STEP 3: METRICS ───────────────────────────────────────
#     retrieval_score   = compute_retrieval_score(question, retrieved)
#     context_precision = compute_context_precision(question, retrieved)
#     recall_score      = compute_recall_at_k(
#         question, retrieved, all_chunks, k=len(retrieved)
#     )

#     # ── FIX 6 — k expansion: low recall OR numeric question ───
#     is_numeric_question = len(_extract_numbers(question)) > 0
#     if recall_score < 25 or is_numeric_question:
#         print("[QA] ⚠️ Expanding retrieval (low recall or numeric question)")
#         expanded = multi_query_retrieve(
#             question, faiss_index, k=30,
#             all_chunks=all_chunks, query_type="FACTUAL_QA"
#         )
#         if len(expanded) > len(retrieved):
#             retrieved = expanded
#             retrieved_texts = [
#                 d.page_content if hasattr(d, "page_content") else str(d)
#                 for d in retrieved
#             ]

#     # ── FIX 1 — Numeric intent: retrieved first, fallback to all_raw ──
#     numeric_intent = _detect_numeric_intent(question, retrieved_texts)

#     if numeric_intent == "NONE" and is_numeric_question:
#         print("[Navigate] Retrying intent detection on full document...")
#         numeric_intent = _detect_numeric_intent(question, all_raw)
#         if numeric_intent != "NONE":
#             print(f"[Navigate] Intent found in full document: {numeric_intent}")

#     # ── NAVIGATIONAL ──────────────────────────────────────────
#     if numeric_intent == "NAVIGATIONAL":
#         print(f"[Routing] Numeric intent → NAVIGATIONAL")
#         title = _navigate_full_chunks(question, all_raw)
#         if title:
#             print(f"[Navigate] Extracted title: '{title}'")
#             grounding  = compute_answer_grounding(title, retrieved_texts, question)
#             confidence = round(grounding / 100, 3)
#             qa_time    = time.time() - qa_start_t
#             state["answer"]      = title
#             state["query_type"]  = "NAVIGATIONAL"
#             _write_metrics(state, "navigational", "navigational", grounding,
#                            confidence, retrieval_score, context_precision,
#                            recall_score, llm_calls, retrieved, qa_time)
#             return state
#         print(f"[Navigate] Extraction failed — falling through to LLM")

#     # ── POSITIONAL ────────────────────────────────────────────
#     elif numeric_intent == "POSITIONAL":
#         print(f"[Routing] Numeric intent → POSITIONAL")
#         pos_answer = _positional_extract(question, retrieved_texts)
#         if pos_answer:
#             print(f"[Positional] Extracted: '{pos_answer}'")
#             grounding  = compute_answer_grounding(pos_answer, retrieved_texts, question)
#             confidence = round(grounding / 100, 3)
#             qa_time    = time.time() - qa_start_t
#             state["answer"]      = pos_answer
#             state["query_type"]  = "POSITIONAL"
#             _write_metrics(state, "positional", "positional", grounding,
#                            confidence, retrieval_score, context_precision,
#                            recall_score, llm_calls, retrieved, qa_time)
#             return state
#         print(f"[Positional] Extraction failed — falling through to LLM")

#     # ── Normal routing ────────────────────────────────────────
#     query_type = classify_from_context(question, retrieved_texts)
#     state["query_type"] = query_type
#     print(f"[Routing] Context-based → {query_type}")

#     # ── STEP 4: ANSWER GENERATION ─────────────────────────────

#     if query_type == "FULL_SUMMARY":
#         retrieved_texts = clean_context_for_llm(retrieved_texts)
#         ranked  = reorder_by_question(question, retrieved_texts)
#         context = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:8])
#         structured_context = "\n".join(
#             f"[CONTEXT CHUNK]\n{c}" for c in context.split("\n\n---\n\n")
#         )
#         emit_event(request_id, "stream_start", "✍️ Generating answer...")
#         answer, _ = call_llama_streaming(
#             QA_PROMPT.format(context=structured_context[:2500], question=question),
#             request_id=request_id, temperature=0.0
#         )
#         answer     = clean_artifacts(answer).strip()
#         model_used = "llama_summary"
#         llm_calls  = 1

#     elif query_type == "MULTIPART_QA":
#         retrieved_texts = clean_context_for_llm(retrieved_texts)
#         ranked  = reorder_by_question(question, retrieved_texts)
#         # FIX 3 — 4000 chars for MULTIPART to capture all list items
#         context = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:8])
#         emit_event(request_id, "agent_start",
#                    f"🤖 MULTIPART | {len(retrieved)} chunks")
#         emit_event(request_id, "stream_start", "✍️ Generating answer...")
#         answer, _ = call_llama_streaming(
#             QA_PROMPT.format(context=context[:4000], question=question),
#             request_id=request_id, temperature=0.0
#         )
#         answer     = clean_artifacts(answer).strip()
#         model_used = "llama_multipart"
#         llm_calls  = 1

#     else:
        
#         # FACTUAL_QA / VERIFICATION_QA

#         if not retrieved_texts:
#             print("[QA] ❌ No context → NOT FOUND")
#             state["answer"] = "This information is not present in the document."
#             _write_metrics(state, "not_found", "no_context",
#                         0.0, 0.0, retrieval_score, context_precision,
#                         recall_score, llm_calls, retrieved,
#                         time.time() - qa_start_t)
#             return state

#         # 🔥 FIX 5 — SIMPLE QA ROUTING (ADD HERE)

#         context = "\n".join(retrieved_texts)

#         if (
#             query_type == "FACTUAL_QA"
#             and len(question.split()) <= 10
#             and not is_numeric_question
#             and recall_score >= 60
#         ):
#             print("[QA] ⚡ Direct QA (no ReAct)")

#             react_ans, _ = call_llama_streaming(
#                 f"""
#             Answer the question using ONLY the given context.
#             Return ONLY the exact answer phrase from the context.
#             Do NOT explain. Do NOT say 'not found' unless absolutely missing.

#             Context:
#             {context[:2000]}

#             Question: {question}

#             Answer:
#             """,
#                 request_id=request_id,
#                 temperature=0.0
#             )
#             llm_calls += 1
#             model_used = "llama_direct"

#         else:
#             react_ans, model_used, _, _, _, _, react_calls, _ = react_agent(
#     question,
#     faiss_index,
#     query_type,
#     all_chunks,
#     request_id,
#     recall_score
# )

#         # react_ans, model_used, _, _, _, _, react_calls, retrieved_texts = react_agent(
#         #     question, faiss_index, query_type, all_chunks, request_id
#         # )
 
#         react_ans = clean_artifacts(react_ans).strip().strip('"').strip("'")
#         # react_ans = clean_reasoning_answer(react_ans, question)
#         print(f"[QA] Answer: '{react_ans[:60]}'")

#         # FIX 2 — Short answer bypass: skip for VERIFICATION_QA
#         words         = react_ans.split()
#         content_words = [w for w in words if len(w) > 2 and not w.isdigit()]

#         # if (
#         #     query_type != "VERIFICATION_QA"
#         #     and len(words) <= 3
#         #     and len(content_words) >= 1
#         #     and not _is_refusal_answer(react_ans)
#         # ):
#         #     print("[QA] ⚡ Short answer accepted")
#         #     grounding  = compute_answer_grounding(react_ans, retrieved_texts, question)
#         #     confidence = grounding / 100
#         #     qa_time    = time.time() - qa_start_t
#         #     state["answer"] = react_ans
#         #     _write_metrics(state, model_used, "short_answer",
#         #                    grounding, confidence, retrieval_score,
#         #                    context_precision, recall_score, llm_calls,
#         #                    retrieved, qa_time)
#         #     return state
#         print(f"[DEBUG] react_ans before refusal check: repr={repr(react_ans)}")
#         # Refusal
#         if _is_refusal_answer(react_ans):
#             print("[QA] ⚠️ Refusal detected")
#             qa_time = time.time() - qa_start_t
#             state["answer"] = "This information is not present in the document."
#             _write_metrics(state, "not_found", "not_found",
#                            75.0, 0.75, retrieval_score, context_precision,
#                            recall_score, llm_calls, retrieved, qa_time)
#             return state

#         grounding_score = compute_answer_grounding(react_ans, retrieved_texts, question)
#         # 🔥 STRICT VERIFICATION CHECK
#         # if query_type == "VERIFICATION_QA":
#         #     if grounding_score < 80:
#         #         print("[QA] ❌ Verification failed → NOT FOUND")
#         #         answer = "This information is not present in the document."
#         #         decision_type = "verification_failed"
#         #     else:
#         #         answer = react_ans
#         #         decision_type = "accepted"
#         if query_type == "VERIFICATION_QA":
#             if grounding_score < 80:
#                 print("[QA] ❌ Verification failed → NOT FOUND")
#                 answer        = "This information is not present in the document."
#                 decision_type = "verification_failed"
#             else:
#                 answer        = react_ans
#                 decision_type = "accepted"

#             grounding  = compute_answer_grounding(answer, retrieved_texts, question)
#             confidence = round(grounding / 100, 3)
#             qa_time    = time.time() - qa_start_t
#             state["answer"] = normalize_answer(answer)
#             _write_metrics(state, model_used, decision_type, grounding,
#                         confidence, retrieval_score, context_precision,
#                         recall_score, llm_calls, retrieved, qa_time)
#             return state

#         # 🔥 TRUST GATE (FINAL CLEAN VERSION)

#         recall = recall_score
#         grounding = grounding_score
#         length = len(react_ans.split())

#         # 1. No reliable context → reject
#         if recall < 25:
#             print("[QA] ❌ Low recall → NOT FOUND")
#             answer = "This information is not present in the document."
#             decision_type = "low_recall"

#         # 2. Very weak grounding → reject
#         elif grounding < 40:
#             print("[QA] ❌ Very low grounding → NOT FOUND")
#             answer = "This information is not present in the document."
#             decision_type = "low_grounding"

#         # # 3. Short answers must be VERY strong
#         # elif length <= 3:
#         #     if recall >= 70 and grounding >= 70:
#         #         answer = react_ans
#         #         decision_type = "accepted"
        
#         #     else:
#         #         print("[QA] ❌ Weak short answer → NOT FOUND")
#         #         answer = "This information is not present in the document."
#         #         decision_type = "weak_short"

#         # # 4. Strong signals → accept
#         # elif recall >= 50 and grounding >= 60:
#         #     answer = react_ans
#         #     decision_type = "accepted"

#         # # 5. Otherwise reject safely
#         # else:
#         #     print("[QA] ❌ Uncertain → NOT FOUND")
#         #     answer = "This information is not present in the document."
#         #     decision_type = "uncertain"
#         elif length <= 3:
#             if grounding >= 60:
#                 answer = react_ans
#                 decision_type = "accepted"
#             else:
#                 print("[QA] ❌ Weak short answer → NOT FOUND")
#                 answer = "This information is not present in the document."
#                 decision_type = "weak_short"

#         # 4. Strong signals → accept
#         elif recall >= 50 and grounding >= 60:
#             answer = react_ans
#             decision_type = "accepted"

#         # 5. Otherwise reject safely
#         else:
#             print("[QA] ❌ Uncertain → NOT FOUND")
#             answer = "This information is not present in the document."
#             decision_type = "uncertain"
#     # ── STEP 5: METRICS ───────────────────────────────────────
#     grounding  = compute_answer_grounding(answer, retrieved_texts, question)
#     confidence = round(grounding / 100, 3)
#     qa_time    = time.time() - qa_start_t

#     print(f"[QA] Done in {qa_time:.1f}s | model={model_used} | "
#           f"grounding={grounding:.1f}% | recall={recall_score:.1f}% | "
#           f"confidence={confidence:.3f} | decision={decision_type}")

#     if not answer.strip():
#         answer = "Could not find a relevant answer in the PDF."
#     else:
#         answer = normalize_answer(answer)

#     state["answer"] = answer
#     _write_metrics(state, model_used, decision_type, grounding,
#                    confidence, retrieval_score, context_precision,
#                    recall_score, llm_calls, retrieved, qa_time)
#     return state
# def node_validate(state: DocState) -> DocState:
#     answer = state["answer"]
#     retry  = state.get("retry_count", 0)

#     # ── Retry for empty/very weak answers ────────────────────
#     if len(answer.strip()) < 3 and retry < 2:
#         state["retry_count"] = retry + 1
#         state["answer"]      = ""
#         return state

#     total_time    = time.time() - state["start_time"]
#     output_words  = len(answer.split())
#     output_tokens = output_words * 1.3

#     extract_time  = state["metrics"].get("extraction_time_sec", 0)
#     llm_time      = max(total_time - extract_time, 1)
#     tps           = round(output_tokens / llm_time, 2) if llm_time > 0 else 0

#     m = state["metrics"]

#     # ── Core metrics ─────────────────────────────────────────
#     m["response_time_sec"]    = round(total_time, 2)
#     m["extraction_time_sec"]  = m.get("extraction_time_sec", 0)
#     m["pages_processed"]      = state.get("page_count", 0)
#     m["characters_processed"] = state.get("char_count", 0)
#     m["words_processed"]      = len(state.get("extracted_text", "").split())

#     # ── Type-specific metrics ────────────────────────────────
#     if m.get("type") == "summary":
#         m["summary_time_sec"]     = m.get("summary_time_sec", 0)
#         m["summary_length_words"] = len(answer.split())

#     if m.get("type") == "qa":
#         m["qa_time_sec"]      = m.get("qa_time_sec", 0)
#         m["confidence_score"] = m.get("confidence_score", 0)

#     # ── Performance ──────────────────────────────────────────
#     m["ttft_sec"]        = round(total_time, 2)
#     m["e2e_latency_sec"] = round(total_time, 2)
#     m["tps"]             = tps

#     # ❌ REMOVED doc_type (no longer used anywhere)
#     # m["doc_type"] = state.get("doc_type", "general")

#     # ── Context info ─────────────────────────────────────────
#     m["query_type"]     = state.get("query_type", "")
#     m["chunks_created"] = m.get("chunks_created", 0)
#     m["retry_count"]    = retry

#     # ── Model + retrieval metrics ────────────────────────────
#     m["model_used"]        = m.get("model_used", "llama_react")
#     m["llm_calls"]         = m.get("llm_calls", 0)
#     m["retrieval_score"]   = m.get("retrieval_score", 0)
#     m["context_precision"] = m.get("context_precision", 0)
#     m["answer_grounding"]  = m.get("answer_grounding", 0)
#     m["recall_at_k"]       = m.get("recall_at_k", 0)

#     # ── Summary-specific ─────────────────────────────────────
#     if m.get("type") == "summary":
#         m["parallel_workers"] = m.get("parallel_workers", 0)
#         m["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
#         m["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)

#     # ── QA-specific ──────────────────────────────────────────
#     if m.get("type") == "qa":
#         m["chunks_retrieved"] = m.get("chunks_retrieved", 0)
#         m["decision_type"]    = m.get("decision_type", "accepted")
#         m["confidence_raw"]   = m.get("confidence_raw", 0.0)

#     state["metrics"] = m
#     return state

# # def node_validate(state: DocState) -> DocState:
# #     answer = state["answer"]
# #     retry  = state.get("retry_count", 0)

# #     if len(answer.strip()) < 3 and retry < 2:
# #         state["retry_count"] = retry + 1
# #         state["answer"]      = ""
# #         return state

# #     total_time    = time.time() - state["start_time"]
# #     output_words  = len(answer.split())
# #     output_tokens = output_words * 1.3
# #     extract_time  = state["metrics"].get("extraction_time_sec", 0)
# #     llm_time      = max(total_time - extract_time, 1)
# #     tps           = round(output_tokens / llm_time, 2) if llm_time > 0 else 0
# #     m             = state["metrics"]

# #     m["response_time_sec"]    = round(total_time, 2)
# #     m["extraction_time_sec"]  = m.get("extraction_time_sec", 0)
# #     m["pages_processed"]      = state.get("page_count", 0)
# #     m["characters_processed"] = state.get("char_count", 0)
# #     m["words_processed"]      = len(state.get("extracted_text", "").split())

# #     if m.get("type") == "summary":
# #         m["summary_time_sec"]     = m.get("summary_time_sec", 0)
# #         m["summary_length_words"] = len(answer.split())
# #     if m.get("type") == "qa":
# #         m["qa_time_sec"]      = m.get("qa_time_sec", 0)
# #         m["confidence_score"] = m.get("confidence_score", 0)

# #     m["ttft_sec"]          = round(total_time, 2)
# #     m["e2e_latency_sec"]   = round(total_time, 2)
# #     m["tps"]               = tps
# #     m["doc_type"]          = state.get("doc_type", "general")
# #     m["query_type"]        = state.get("query_type", "")
# #     m["chunks_created"]    = m.get("chunks_created", 0)
# #     m["retry_count"]       = retry
# #     m["model_used"]        = m.get("model_used", "llama_react")
# #     m["llm_calls"]         = m.get("llm_calls", 0)
# #     m["retrieval_score"]   = m.get("retrieval_score", 0)
# #     m["context_precision"] = m.get("context_precision", 0)
# #     m["answer_grounding"]  = m.get("answer_grounding", 0)
# #     m["recall_at_k"]       = m.get("recall_at_k", 0)

# #     if m.get("type") == "summary":
# #         m["parallel_workers"] = m.get("parallel_workers", 0)
# #         m["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
# #         m["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)
# #     if m.get("type") == "qa":
# #         m["chunks_retrieved"] = m.get("chunks_retrieved", 0)
# #         m["decision_type"]    = m.get("decision_type", "accepted")
# #         m["confidence_raw"]   = m.get("confidence_raw", 0.0)

# #     state["metrics"] = m
# #     return state


# def _write_metrics(state, model_used, decision_type, grounding,
#                    confidence, retrieval_score, context_precision,
#                    recall_score, llm_calls, retrieved, qa_time):
#     m = state.setdefault("metrics", {})
#     m["qa_time_sec"]       = round(qa_time, 2)
#     m["confidence_score"]  = round(confidence * 100, 2)
#     m["retrieval_score"]   = retrieval_score
#     m["context_precision"] = context_precision
#     m["answer_grounding"]  = grounding
#     m["recall_at_k"]       = recall_score
#     m["llm_calls"]         = llm_calls
#     m["model_used"]        = model_used
#     m["chunks_retrieved"]  = len(retrieved)
#     m["type"]              = "qa"
#     m["decision_type"]     = decision_type
#     m["confidence_raw"]    = round(confidence, 4)


























# idiot code 2
# import time
# import re
# import spacy
# _nlp = spacy.load("en_core_web_lg")

# from docmind_rag.config.settings import (
#     MAX_WORKERS, FACTUAL_TOP_K, MULTIPART_TOP_K
# )
# from docmind_rag.utils.text import extract_named_entities, expand_answer
# from docmind_rag.core.state import DocState
# from docmind_rag.core.prompts import QA_PROMPT
# from docmind_rag.models.llm import call_llama_streaming
# from docmind_rag.models.embeddings import build_faiss_index
# from docmind_rag.models.reranker import rerank_docs, protect_exact_matches
# from docmind_rag.services.extraction import extract_pdf_parallel, _extraction_cache
# from docmind_rag.services.chunking import semantic_chunk, raptor_summarize, _summary_cache
# from docmind_rag.services.retrieval import multi_query_retrieve
# from docmind_rag.services.agent import react_agent,  clean_reasoning_answer
# from docmind_rag.events.events import emit_event
# from docmind_rag.utils.helpers import get_pdf_hash, clean_artifacts, clean_chunk_text, normalize_answer
# from docmind_rag.utils.text import (
#     classify_from_context,
#     reorder_by_question,
#     normalize_text,
# )
# from docmind_rag.utils.metrics import (
#     compute_answer_grounding,
#     compute_retrieval_score,
#     compute_context_precision,
#     compute_recall_at_k,
#     semantic_similarity,
#     _is_refusal_answer,
# )

# # def trust_gate(answer: str, metrics: dict) -> str:
# #     recall = metrics.get("recall_at_k", 0)
# #     grounding = metrics.get("answer_grounding", 0)
# #     length = len(answer.split())

# #     # 1. No reliable context → reject
# #     if recall < 25:
# #         return "not_found"

# #     # 2. Weak grounding → reject
# #     if grounding < 50:
# #         return "not_found"

# #     # 3. Short answers must be very strong
# #     if length <= 3:
# #         if recall >= 70 and grounding >= 70:
# #             return "accepted"
# #         return "not_found"

# #     # 4. Strong signal → accept
# #     if recall >= 50 and grounding >= 60:
# #         return "accepted"
   
# #     # 5. Default safe behavior
# #     else:
# #         return "not_found"
# # ============================================================
# # LOCAL HELPERS
# # ============================================================

# def clean_context_for_llm(chunks):
#     cleaned_chunks = []
#     for chunk in chunks:
#         lines = chunk.split("\n")
#         filtered_lines = []
#         for line in lines:
#             line_strip = line.strip()
#             is_question_like = (
#                 line_strip.endswith("?") or
#                 (len(line_strip.split()) < 15 and "?" in line_strip)
#             )
#             if is_question_like:
#                 continue
#             filtered_lines.append(line)
#         cleaned_chunks.append("\n".join(filtered_lines))
#     return cleaned_chunks


# def is_grounded(answer, retrieved_texts, threshold=0.6):
#     if not answer:
#         return False
#     answer_words = set(answer.lower().split())
#     if not answer_words:
#         return False
#     best_overlap = 0
#     for chunk in retrieved_texts:
#         chunk_words = set(chunk.lower().split())
#         overlap = len(answer_words & chunk_words) / len(answer_words)
#         best_overlap = max(best_overlap, overlap)
#     return best_overlap >= threshold


# # ============================================================
# # NUMERIC INTENT DETECTION
# # ============================================================

# def _extract_numbers(text: str) -> list:
#     return re.findall(r'\b\d+\b', text)


# def _detect_numeric_intent(question: str, chunks: list) -> str:
#     """
#     Detect NAVIGATIONAL or POSITIONAL from chunk structure.
#     Generic — no hardcoded section words.
#     """
#     numbers = _extract_numbers(question)
#     if not numbers:
#         return "NONE"

#     q_words = set(question.lower().split())

#     for chunk in chunks:
#         lines = [l.strip() for l in chunk.split("\n") if l.strip()]
#         if not lines:
#             continue
#         first_line = lines[0].lower()
#         line_words = set(first_line.split())
#         overlap    = len(q_words & line_words)
#         if any(re.search(rf'\b{re.escape(num)}\b', first_line) for num in numbers):
#             if overlap >= 1 and len(first_line.split()) <= 12:
#                 return "NAVIGATIONAL"

#     for chunk in chunks:
#         lines       = [l.strip() for l in chunk.split("\n") if l.strip()]
#         short_lines = [l for l in lines if len(l.split()) <= 12]
#         if len(short_lines) >= 3:
#             first = lines[0].lower() if lines else ""
#             if not any(re.search(rf'\b{re.escape(num)}\b', first) for num in numbers):
#                 return "POSITIONAL"

#     return "NONE"


# def _navigate_full_chunks(question: str, all_raw_chunks: list) -> str:
#     """
#     Scan full raw chunks to find section title for a numbered section.
#     Generic — works for Chapter N, Lecture N, Section N, etc.
#     """
#     numbers = _extract_numbers(question)
#     if not numbers:
#         return ""

#     q_words    = set(question.lower().split())
#     best_line  = ""
#     best_score = 0

#     for chunk in all_raw_chunks:
#         lines = [l.strip() for l in chunk.split("\n") if l.strip()]
#         if not lines:
#             continue
#         first_line  = lines[0]
#         first_lower = first_line.lower()

#         for num in numbers:
#             if not re.search(rf'\b{re.escape(num)}\b', first_lower):
#                 continue
#             line_words = set(first_lower.split())
#             overlap    = len(q_words & line_words)
#             if overlap >= 1 and len(first_line.split()) <= 14:
#                 score = overlap + (5 if len(first_line.split()) <= 8 else 0)
#                 if score > best_score:
#                     best_score = score
#                     best_line  = first_line

#     if not best_line:
#         return ""

#     title = best_line
#     title = re.sub(r'^\s*\d+\s*[-—–]+\s*[—–-]?\s*\d*\s*', '', title)
#     title = re.sub(r'\b\d+\b', '', title)
#     title = re.sub(r'[:\-–—]', ' ', title)
#     title = re.sub(r'\s{2,}', ' ', title)
#     return title.strip()


# def _positional_extract(question: str, retrieved_texts: list) -> str:
#     """Extract Nth item from a list. Generic."""
#     numbers = _extract_numbers(question)
#     if not numbers:
#         return ""
#     try:
#         idx = int(numbers[0]) - 1
#     except ValueError:
#         return ""
#     if idx < 0:
#         return ""

#     best_chunk, best_count = "", 0
#     for chunk in retrieved_texts:
#         lines       = [l.strip() for l in chunk.split("\n") if l.strip()]
#         short_lines = [l for l in lines if len(l.split()) <= 12]
#         if len(short_lines) > best_count:
#             best_count = len(short_lines)
#             best_chunk = chunk

#     if not best_chunk:
#         return ""

#     lines       = [l.strip() for l in best_chunk.split("\n") if l.strip()]
#     short_lines = [l for l in lines if len(l.split()) <= 12]
#     if idx < len(short_lines):
#         return re.sub(r'^\s*\d+[\.\)]\s*', '', short_lines[idx]).strip()
#     return ""


# # ============================================================
# # SIGNAL 3 — Structural consistency for NAVIGATIONAL answers
# # ============================================================

# def _check_identifier_grounded(answer: str, retrieved_texts: list) -> bool:
#     """
#     If answer contains a multi-word identifier (word+number or number+word),
#     verify it appears verbatim in retrieved chunks.
#     Generic — no hardcoded section words.
#     """
#     identifiers = re.findall(
#         r'\b[a-zA-Z]+\s+\d+\b|\b\d+\s+[a-zA-Z]+\b',
#         answer.lower()
#     )
#     if not identifiers:
#         return True

#     chunk_text = " ".join(retrieved_texts).lower()
#     for ident in identifiers:
#         if ident not in chunk_text:
#             print(f"[Signal3] ❌ Identifier '{ident}' not in retrieved chunks")
#             return False
#     return True


# # ============================================================
# # REFUSAL — semantic version (local)
# # ============================================================

# _REFUSAL_ANCHOR = "this information is not available in the provided context"


# def _is_refusal_semantic(text: str) -> bool:
#     if not text or len(text.strip()) < 2:
#         return True
#     sim = semantic_similarity(text, [_REFUSAL_ANCHOR])
#     print(f"[Refusal] sim={sim:.3f} | '{text[:50]}'")
#     return sim > 0.72


# # ============================================================
# # LANGGRAPH NODES
# # ============================================================

# def node_extract(state: DocState) -> DocState:
#     extract_start    = time.time()
#     text, page_count = extract_pdf_parallel(state["pdf_path"])
#     extract_time     = time.time() - extract_start
#     state["extracted_text"] = text
#     state["page_count"]     = page_count
#     state["char_count"]     = len(text)
#     state["metrics"]["extraction_time_sec"]  = round(extract_time, 2)
#     state["metrics"]["pages_processed"]      = page_count
#     state["metrics"]["characters_processed"] = len(text)
#     state["metrics"]["words_processed"]      = len(text.split())
#     return state


# def node_chunk(state: DocState) -> DocState:
#     text = state["extracted_text"]
#     state["query_type"] = "QA"

#     summary_chunks, rag_chunks = semantic_chunk(text)

#     # FIX 4 — normalize all chunks after chunking
#     # Converts unicode symbols so retrieval doesn't fail on symbol mismatches
#     rag_chunks     = [normalize_text(c) for c in rag_chunks]
#     summary_chunks = [normalize_text(c) for c in summary_chunks]

#     state["summary_chunks"] = summary_chunks
#     state["chunks"]         = rag_chunks

#     state["metrics"]["summary_chunks"] = len(summary_chunks)
#     state["metrics"]["chunks_created"] = len(rag_chunks)
#     # state["metrics"]["doc_type"]       = state.get("doc_type", "general")
#     state["metrics"]["query_type"]     = state["query_type"]
#     print(f"[Chunk] {len(summary_chunks)} summary chunks | {len(rag_chunks)} RAG chunks")
#     return state


# # def node_summarize(state: DocState) -> DocState:
# #     pdf_hash  = get_pdf_hash(state["pdf_path"])
# #     cache_key = f"{pdf_hash}_{state.get('doc_type', 'general')}"

# #     if cache_key in _summary_cache:
# #         cached = _summary_cache[cache_key]
# #         print("[Summary] ✅ Cache hit")
# #         emit_event(state.get("request_id", ""), "agent_action",
# #                    "⚡ Summary loaded from cache instantly!")
# #         state["answer"] = cached["summary"]
# #         state["metrics"].update(cached["metrics"])
# #         state["metrics"]["type"] = "summary"
# #         return state

# #     summary_start = time.time()
# #     raptor_summarize._request_id = state.get("request_id", "")
# #     summary, map_time, reduce_time = raptor_summarize(
# #         state["summary_chunks"], state.get("doc_type", "general")
# #     )
# #     summary_time = time.time() - summary_start

# #     state["answer"] = summary
# #     metrics_snapshot = {
# #         "summary_time_sec":     round(summary_time, 2),
# #         "summary_length_words": len(summary.split()),
# #         "parallel_workers":     min(MAX_WORKERS, len(state["summary_chunks"])),
# #         "map_time_sec":         round(map_time, 2),
# #         "reduce_time_sec":      round(reduce_time, 2),
# #         "llm_calls":            3,
# #     }
# #     state["metrics"].update(metrics_snapshot)
# #     state["metrics"]["type"] = "summary"
# #     _summary_cache[cache_key] = {"summary": summary, "metrics": metrics_snapshot}
# #     print(f"[Summary] Done ({len(summary.split())} words)")
# #     return state

# def node_summarize(state: DocState) -> DocState:
#     pdf_hash  = get_pdf_hash(state["pdf_path"])
#     cache_key = pdf_hash  # ❌ removed doc_type dependency

#     # ── Cache check ──────────────────────────────────────────
#     if cache_key in _summary_cache:
#         cached = _summary_cache[cache_key]
#         print("[Summary] ✅ Cache hit")
#         emit_event(
#             state.get("request_id", ""),
#             "agent_action",
#             "⚡ Summary loaded from cache instantly!"
#         )
#         state["answer"] = cached["summary"]
#         state["metrics"].update(cached["metrics"])
#         state["metrics"]["type"] = "summary"
#         return state

#     # ── Generate summary ─────────────────────────────────────
#     summary_start = time.time()
#     raptor_summarize._request_id = state.get("request_id", "")

#     # ❌ removed doc_type argument
#     summary, map_time, reduce_time = raptor_summarize(
#         state["summary_chunks"]
#     )

#     summary_time = time.time() - summary_start

#     # ── Store result ─────────────────────────────────────────
#     state["answer"] = summary

#     metrics_snapshot = {
#         "summary_time_sec":     round(summary_time, 2),
#         "summary_length_words": len(summary.split()),
#         "parallel_workers":     min(MAX_WORKERS, len(state["summary_chunks"])),
#         "map_time_sec":         round(map_time, 2),
#         "reduce_time_sec":      round(reduce_time, 2),
#         "llm_calls":            3,
#     }

#     state["metrics"].update(metrics_snapshot)
#     state["metrics"]["type"] = "summary"

#     # ── Cache result ─────────────────────────────────────────
#     _summary_cache[cache_key] = {
#         "summary": summary,
#         "metrics": metrics_snapshot
#     }

#     print(f"[Summary] Done ({len(summary.split())} words)")
#     return state

# def node_qa(state: DocState) -> DocState:
#     qa_start_t = time.time()
#     question   = state["question"]
#     request_id = state.get("request_id", "")
#     all_chunks = state["chunks"]

#     # FIX 4 — normalize question the same way chunks were normalized
#     question = normalize_text(question)

#     # Raw strings for navigational scanning — must NOT be cleaned
#     all_raw = [
#         d if isinstance(d, str) else d.page_content
#         for d in all_chunks
#     ]

#     pdf_hash    = get_pdf_hash(state["pdf_path"])
#     faiss_index = build_faiss_index(all_chunks, pdf_hash)

#     print(f"[QA] START | {len(all_chunks)} chunks")

#     recall_score      = 0.0
#     retrieval_score   = 0.0
#     context_precision = 0.0
#     grounding         = 0.0
#     llm_calls         = 0
#     model_used        = "llama"
#     confidence        = 0.0
#     decision_type     = "accepted"
#     retrieved         = []
#     retrieved_texts   = []
#     answer            = ""

#     # ── STEP 1: RETRIEVE ─────────────────────────────────────
#     retrieved = multi_query_retrieve(
#         question, faiss_index,
#         k=20,
#         all_chunks=all_chunks,
#         query_type="FACTUAL_QA"
#     )


   
#     # 🔥 STEP 2 — RERANK + SAFE FALLBACK

#     retrieved, reranker_top, _ = rerank_docs(
#         question, retrieved, top_k=8, apply_pruning=True
#     )

#     retrieved = protect_exact_matches(
#         question, retrieved, all_chunks, top_k=8
#     )

#     # 🔥 FIX 1 — fallback must replace retrieved (NOT just texts)
#     if not retrieved or len(retrieved) < 3:
#         print("[QA] ⚠️ Reranker too aggressive → fallback to initial chunks")
#         print("[QA] ⚠️ Using similarity fallback")

#         retrieved = multi_query_retrieve(
#             question,
#             faiss_index,
#             k=8,
#             all_chunks=all_chunks,
#             query_type="FACTUAL_QA"
#         )

#         retrieved_texts = [
#             d.page_content if hasattr(d, "page_content") else str(d)
#             for d in retrieved
#         ]

#     # 🔥 FIX 2 — safe conversion
#     retrieved_texts = [
#         d.page_content if hasattr(d, "page_content") else str(d)
#         for d in retrieved
#     ]
#     if reranker_top < -5:
#         print("[Guard] ⚠️ Weak reranker score — continuing")

#     # ── STEP 3: METRICS ───────────────────────────────────────
#     retrieval_score   = compute_retrieval_score(question, retrieved)
#     context_precision = compute_context_precision(question, retrieved)
#     recall_score      = compute_recall_at_k(
#         question, retrieved, all_chunks, k=len(retrieved)
#     )

#     # ── FIX 6 — k expansion: low recall OR numeric question ───
#     is_numeric_question = len(_extract_numbers(question)) > 0
#     if recall_score < 60 or is_numeric_question:
#         print("[QA] ⚠️ Expanding retrieval (low recall or numeric question)")
#         expanded = multi_query_retrieve(
#             question, faiss_index, k=30,
#             all_chunks=all_chunks, query_type="FACTUAL_QA"
#         )
#         if len(expanded) > len(retrieved):
#             retrieved = expanded
#             retrieved_texts = [
#                 d.page_content if hasattr(d, "page_content") else str(d)
#                 for d in retrieved
#             ]

#     # ── FIX 1 — Numeric intent: retrieved first, fallback to all_raw ──
#     numeric_intent = _detect_numeric_intent(question, retrieved_texts)

#     if numeric_intent == "NONE" and is_numeric_question:
#         print("[Navigate] Retrying intent detection on full document...")
#         numeric_intent = _detect_numeric_intent(question, all_raw)
#         if numeric_intent != "NONE":
#             print(f"[Navigate] Intent found in full document: {numeric_intent}")

#     # ── NAVIGATIONAL ──────────────────────────────────────────
#     if numeric_intent == "NAVIGATIONAL":
#         print(f"[Routing] Numeric intent → NAVIGATIONAL")
#         title = _navigate_full_chunks(question, all_raw)
#         if title:
#             print(f"[Navigate] Extracted title: '{title}'")
#             grounding  = compute_answer_grounding(title, retrieved_texts, question)
#             confidence = round(grounding / 100, 3)
#             qa_time    = time.time() - qa_start_t
#             state["answer"]      = title
#             state["query_type"]  = "NAVIGATIONAL"
#             _write_metrics(state, "navigational", "navigational", grounding,
#                            confidence, retrieval_score, context_precision,
#                            recall_score, llm_calls, retrieved, qa_time)
#             return state
#         print(f"[Navigate] Extraction failed — falling through to LLM")

#     # ── POSITIONAL ────────────────────────────────────────────
#     elif numeric_intent == "POSITIONAL":
#         print(f"[Routing] Numeric intent → POSITIONAL")
#         pos_answer = _positional_extract(question, retrieved_texts)
#         if pos_answer:
#             print(f"[Positional] Extracted: '{pos_answer}'")
#             grounding  = compute_answer_grounding(pos_answer, retrieved_texts, question)
#             confidence = round(grounding / 100, 3)
#             qa_time    = time.time() - qa_start_t
#             state["answer"]      = pos_answer
#             state["query_type"]  = "POSITIONAL"
#             _write_metrics(state, "positional", "positional", grounding,
#                            confidence, retrieval_score, context_precision,
#                            recall_score, llm_calls, retrieved, qa_time)
#             return state
#         print(f"[Positional] Extraction failed — falling through to LLM")

#     # ── Normal routing ────────────────────────────────────────
#     query_type = classify_from_context(question, retrieved_texts)
#     state["query_type"] = query_type
#     print(f"[Routing] Context-based → {query_type}")

#     # ── STEP 4: ANSWER GENERATION ─────────────────────────────

#     if query_type == "FULL_SUMMARY":
#         retrieved_texts = clean_context_for_llm(retrieved_texts)
#         ranked  = reorder_by_question(question, retrieved_texts)
#         context = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:8])
#         structured_context = "\n".join(
#             f"[CONTEXT CHUNK]\n{c}" for c in context.split("\n\n---\n\n")
#         )
#         emit_event(request_id, "stream_start", "✍️ Generating answer...")
#         answer, _ = call_llama_streaming(
#             QA_PROMPT.format(context=structured_context[:2500], question=question),
#             request_id=request_id, temperature=0.0
#         )
#         answer     = clean_artifacts(answer).strip()
#         model_used = "llama_summary"
#         llm_calls  = 1

#     elif query_type == "MULTIPART_QA":
#         retrieved_texts = clean_context_for_llm(retrieved_texts)
#         ranked  = reorder_by_question(question, retrieved_texts)
#         # FIX 3 — 4000 chars for MULTIPART to capture all list items
#         context = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:8])
#         emit_event(request_id, "agent_start",
#                    f"🤖 MULTIPART | {len(retrieved)} chunks")
#         emit_event(request_id, "stream_start", "✍️ Generating answer...")
#         answer, _ = call_llama_streaming(
#             QA_PROMPT.format(context=context[:4000], question=question),
#             request_id=request_id, temperature=0.0
#         )
#         answer     = clean_artifacts(answer).strip()
#         model_used = "llama_multipart"
#         llm_calls  = 1

#     else:
        
#         # FACTUAL_QA / VERIFICATION_QA

#         if not retrieved_texts:
#             print("[QA] ❌ No context → NOT FOUND")
#             state["answer"] = "This information is not present in the document."
#             _write_metrics(state, "not_found", "no_context",
#                         0.0, 0.0, retrieval_score, context_precision,
#                         recall_score, llm_calls, retrieved,
#                         time.time() - qa_start_t)
#             return state

#         # 🔥 FIX 5 — SIMPLE QA ROUTING (ADD HERE)

#         context = "\n".join(retrieved_texts)

#         # if query_type == "FACTUAL_QA" and len(question.split()) <= 10:
#         #     print("[QA] ⚡ Direct QA (no ReAct)")

#         #     react_ans, _ = call_llama_streaming(
#         #         f"""
#         #     Answer the question using ONLY the given context.
#         #     Return ONLY the exact answer phrase from the context.
#         #     Do NOT explain. Do NOT say 'not found' unless absolutely missing.

#         #     Context:
#         #     {context[:2000]}

#         #     Question: {question}

#         #     Answer:
#         #     """,
#         #         request_id=request_id,
#         #         temperature=0.0
#         #     )
#         #     llm_calls += 1
#         #     model_used = "llama_direct"

#         # else:
#         react_ans, model_used, _, _, _, _, react_calls, _ = react_agent(question, query_type, retrieved, request_id)
#         llm_calls += react_calls

#         # react_ans, model_used, _, _, _, _, react_calls, retrieved_texts = react_agent(
#         #     question, faiss_index, query_type, all_chunks, request_id
#         # )
 
#         react_ans = clean_artifacts(react_ans).strip().strip('"').strip("'")
#         # react_ans = clean_reasoning_answer(react_ans, question)
#         answer = react_ans
#         print(f"[QA] Answer: '{react_ans[:60]}'")

#         # FIX 2 — Short answer bypass: skip for VERIFICATION_QA
#         words         = react_ans.split()
#         content_words = [w for w in words if len(w) > 2 and not w.isdigit()]

#         # if (
#         #     query_type != "VERIFICATION_QA"
#         #     and len(words) <= 3
#         #     and len(content_words) >= 1
#         #     and not _is_refusal_answer(react_ans)
#         # ):
#         #     print("[QA] ⚡ Short answer accepted")
#         #     grounding  = compute_answer_grounding(react_ans, retrieved_texts, question)
#         #     confidence = grounding / 100
#         #     qa_time    = time.time() - qa_start_t
#         #     state["answer"] = react_ans
#         #     _write_metrics(state, model_used, "short_answer",
#         #                    grounding, confidence, retrieval_score,
#         #                    context_precision, recall_score, llm_calls,
#         #                    retrieved, qa_time)
#         #     return state

#         # Refusal
#         if _is_refusal_answer(react_ans):
#             print("[QA] ⚠️ Refusal detected")
#             qa_time = time.time() - qa_start_t
#             state["answer"] = "This information is not present in the document."
#             _write_metrics(state, "not_found", "not_found",
#                            75.0, 0.75, retrieval_score, context_precision,
#                            recall_score, llm_calls, retrieved, qa_time)
#             return state

#         grounding_score = compute_answer_grounding(react_ans, retrieved_texts, question)
    
#         # if query_type == "VERIFICATION_QA":
#         #     if grounding_score < 80:
#         #         print("[QA] ❌ Verification failed → NOT FOUND")
#         #         answer = "This information is not present in the document."
#         #         decision_type = "verification_failed"
#         #         state["answer"] = answer
#         #         return state   # ❌ stop pipeline

#         #     else:
#         #         answer = react_ans
#         #         decision_type = "accepted"
#         #         state["answer"] = answer
#         #         return state   # ✅ ALSO stop pipeline
#         if query_type == "VERIFICATION_QA":
#             if grounding_score < 80:
#                 print("[QA] ❌ Verification failed → NOT FOUND")
#                 answer = "This information is not present in the document."
#                 decision_type = "verification_failed"
#             else:
#                 answer = react_ans
#                 decision_type = "accepted"

#             state["answer"] = answer

#             _write_metrics(
#                 state,
#                 model_used,
#                 decision_type,
#                 grounding_score,
#                 grounding_score / 100,
#                 retrieval_score,
#                 context_precision,
#                 recall_score,
#                 llm_calls,
#                 retrieved,
#                 time.time() - qa_start_t
#             )

#             return state

#         # 🔥 TRUST GATE (FINAL CLEAN VERSION)

#         recall = recall_score
#         grounding = grounding_score
#         # length = len(react_ans.split())
#         length = len(answer.split())

#         # 1. No reliable context → reject
#         if recall < 25:
#             print("[QA] ❌ Low recall → NOT FOUND")
#             answer = "This information is not present in the document."
#             decision_type = "low_recall"

#         # 2. Very weak grounding → reject
#         elif grounding < 40:
#             print("[QA] ❌ Very low grounding → NOT FOUND")
#             answer = "This information is not present in the document."
#             decision_type = "low_grounding"

#         # # 3. Short answers must be VERY strong
#         # elif length <= 3:
#         #     if recall >= 70 and grounding >= 70:
#         #         answer = react_ans
#         #         decision_type = "accepted"
        
#         #     else:
#         #         print("[QA] ❌ Weak short answer → NOT FOUND")
#         #         answer = "This information is not present in the document."
#         #         decision_type = "weak_short"

#         # # 4. Strong signals → accept
#         # elif recall >= 50 and grounding >= 60:
#         #     answer = react_ans
#         #     decision_type = "accepted"

#         # # 5. Otherwise reject safely
#         # else:
#         #     print("[QA] ❌ Uncertain → NOT FOUND")
#         #     answer = "This information is not present in the document."
#         #     decision_type = "uncertain"
#         elif length <= 3:
#             if grounding >= 50:
#                 print("[QA] ⚡ Short but grounded → accept")
#                 answer = react_ans
#                 decision_type = "accepted"
#             else:
#                 print("[QA] ❌ Short and ungrounded → NOT FOUND")
#                 answer = "This information is not present in the document."
#                 decision_type = "weak_short"

#         # 4. Strong signals → accept
#         elif recall >= 50 and grounding >= 60:
#             answer = react_ans
#             decision_type = "accepted"

#         # 5. Otherwise reject safely
#         else:
#             print("[QA] ❌ Uncertain → NOT FOUND")
#             answer = "This information is not present in the document."
#             decision_type = "uncertain"
#     if "direct" in model_used:
#         print("[QA] ❌ Blocking direct QA output")
#         answer = "This information is not present in the document."
#         decision_type = "blocked_direct"
#     # ── STEP 5: METRICS ───────────────────────────────────────
#     grounding  = compute_answer_grounding(answer, retrieved_texts, question)
#     confidence = round(grounding / 100, 3)
#     qa_time    = time.time() - qa_start_t

#     print(f"[QA] Done in {qa_time:.1f}s | model={model_used} | "
#           f"grounding={grounding:.1f}% | recall={recall_score:.1f}% | "
#           f"confidence={confidence:.3f} | decision={decision_type}")

#     if not answer.strip():
#         answer = "Could not find a relevant answer in the PDF."
#     else:
#         answer = normalize_answer(answer)

#     state["answer"] = answer
#     _write_metrics(state, model_used, decision_type, grounding,
#                    confidence, retrieval_score, context_precision,
#                    recall_score, llm_calls, retrieved, qa_time)
#     return state
# def node_validate(state: DocState) -> DocState:
#     answer = state["answer"]
#     retry  = state.get("retry_count", 0)

#     # ── Retry for empty/very weak answers ────────────────────
#     if len(answer.strip()) < 3 and retry < 2:
#         state["retry_count"] = retry + 1
#         state["answer"]      = ""
#         return state

#     total_time    = time.time() - state["start_time"]
#     output_words  = len(answer.split())
#     output_tokens = output_words * 1.3

#     extract_time  = state["metrics"].get("extraction_time_sec", 0)
#     llm_time      = max(total_time - extract_time, 1)
#     tps           = round(output_tokens / llm_time, 2) if llm_time > 0 else 0

#     m = state["metrics"]

#     # ── Core metrics ─────────────────────────────────────────
#     m["response_time_sec"]    = round(total_time, 2)
#     m["extraction_time_sec"]  = m.get("extraction_time_sec", 0)
#     m["pages_processed"]      = state.get("page_count", 0)
#     m["characters_processed"] = state.get("char_count", 0)
#     m["words_processed"]      = len(state.get("extracted_text", "").split())

#     # ── Type-specific metrics ────────────────────────────────
#     if m.get("type") == "summary":
#         m["summary_time_sec"]     = m.get("summary_time_sec", 0)
#         m["summary_length_words"] = len(answer.split())

#     if m.get("type") == "qa":
#         m["qa_time_sec"]      = m.get("qa_time_sec", 0)
#         m["confidence_score"] = m.get("confidence_score", 0)

#     # ── Performance ──────────────────────────────────────────
#     m["ttft_sec"]        = round(total_time, 2)
#     m["e2e_latency_sec"] = round(total_time, 2)
#     m["tps"]             = tps

#     # ❌ REMOVED doc_type (no longer used anywhere)
#     # m["doc_type"] = state.get("doc_type", "general")

#     # ── Context info ─────────────────────────────────────────
#     m["query_type"]     = state.get("query_type", "")
#     m["chunks_created"] = m.get("chunks_created", 0)
#     m["retry_count"]    = retry

#     # ── Model + retrieval metrics ────────────────────────────
#     m["model_used"]        = m.get("model_used", "llama_react")
#     m["llm_calls"]         = m.get("llm_calls", 0)
#     m["retrieval_score"]   = m.get("retrieval_score", 0)
#     m["context_precision"] = m.get("context_precision", 0)
#     m["answer_grounding"]  = m.get("answer_grounding", 0)
#     m["recall_at_k"]       = m.get("recall_at_k", 0)

#     # ── Summary-specific ─────────────────────────────────────
#     if m.get("type") == "summary":
#         m["parallel_workers"] = m.get("parallel_workers", 0)
#         m["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
#         m["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)

#     # ── QA-specific ──────────────────────────────────────────
#     if m.get("type") == "qa":
#         m["chunks_retrieved"] = m.get("chunks_retrieved", 0)
#         m["decision_type"]    = m.get("decision_type", "accepted")
#         m["confidence_raw"]   = m.get("confidence_raw", 0.0)

#     state["metrics"] = m
#     return state

# # def node_validate(state: DocState) -> DocState:
# #     answer = state["answer"]
# #     retry  = state.get("retry_count", 0)

# #     if len(answer.strip()) < 3 and retry < 2:
# #         state["retry_count"] = retry + 1
# #         state["answer"]      = ""
# #         return state

# #     total_time    = time.time() - state["start_time"]
# #     output_words  = len(answer.split())
# #     output_tokens = output_words * 1.3
# #     extract_time  = state["metrics"].get("extraction_time_sec", 0)
# #     llm_time      = max(total_time - extract_time, 1)
# #     tps           = round(output_tokens / llm_time, 2) if llm_time > 0 else 0
# #     m             = state["metrics"]

# #     m["response_time_sec"]    = round(total_time, 2)
# #     m["extraction_time_sec"]  = m.get("extraction_time_sec", 0)
# #     m["pages_processed"]      = state.get("page_count", 0)
# #     m["characters_processed"] = state.get("char_count", 0)
# #     m["words_processed"]      = len(state.get("extracted_text", "").split())

# #     if m.get("type") == "summary":
# #         m["summary_time_sec"]     = m.get("summary_time_sec", 0)
# #         m["summary_length_words"] = len(answer.split())
# #     if m.get("type") == "qa":
# #         m["qa_time_sec"]      = m.get("qa_time_sec", 0)
# #         m["confidence_score"] = m.get("confidence_score", 0)

# #     m["ttft_sec"]          = round(total_time, 2)
# #     m["e2e_latency_sec"]   = round(total_time, 2)
# #     m["tps"]               = tps
# #     m["doc_type"]          = state.get("doc_type", "general")
# #     m["query_type"]        = state.get("query_type", "")
# #     m["chunks_created"]    = m.get("chunks_created", 0)
# #     m["retry_count"]       = retry
# #     m["model_used"]        = m.get("model_used", "llama_react")
# #     m["llm_calls"]         = m.get("llm_calls", 0)
# #     m["retrieval_score"]   = m.get("retrieval_score", 0)
# #     m["context_precision"] = m.get("context_precision", 0)
# #     m["answer_grounding"]  = m.get("answer_grounding", 0)
# #     m["recall_at_k"]       = m.get("recall_at_k", 0)

# #     if m.get("type") == "summary":
# #         m["parallel_workers"] = m.get("parallel_workers", 0)
# #         m["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
# #         m["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)
# #     if m.get("type") == "qa":
# #         m["chunks_retrieved"] = m.get("chunks_retrieved", 0)
# #         m["decision_type"]    = m.get("decision_type", "accepted")
# #         m["confidence_raw"]   = m.get("confidence_raw", 0.0)

# #     state["metrics"] = m
# #     return state


# def _write_metrics(state, model_used, decision_type, grounding,
#                    confidence, retrieval_score, context_precision,
#                    recall_score, llm_calls, retrieved, qa_time):
#     m = state.setdefault("metrics", {})
#     m["qa_time_sec"]       = round(qa_time, 2)
#     m["confidence_score"]  = round(confidence * 100, 2)
#     m["retrieval_score"]   = retrieval_score
#     m["context_precision"] = context_precision
#     m["answer_grounding"]  = grounding
#     m["recall_at_k"]       = recall_score
#     m["llm_calls"]         = llm_calls
#     m["model_used"]        = model_used
#     m["chunks_retrieved"]  = len(retrieved)
#     m["type"]              = "qa"
#     m["decision_type"]     = decision_type
#     m["confidence_raw"]    = round(confidence, 4)
































# idiot code 
# import time
# import re
# import spacy
# _nlp = spacy.load("en_core_web_lg")

# from docmind_rag.config.settings import (
#     MAX_WORKERS, FACTUAL_TOP_K, MULTIPART_TOP_K,embedding_model
# )
# from docmind_rag.utils.text import extract_named_entities, expand_answer
# from docmind_rag.core.state import DocState
# from docmind_rag.core.prompts import QA_PROMPT
# from docmind_rag.models.llm import call_llama_streaming
# from docmind_rag.models.embeddings import build_faiss_index
# from docmind_rag.models.reranker import rerank_docs, protect_exact_matches
# from docmind_rag.services.extraction import extract_pdf_parallel, _extraction_cache
# from docmind_rag.services.chunking import semantic_chunk, raptor_summarize, _summary_cache
# from docmind_rag.services.retrieval import multi_query_retrieve
# from docmind_rag.services.agent import react_agent, roberta_qa, clean_reasoning_answer
# from docmind_rag.events.events import emit_event
# from docmind_rag.utils.helpers import get_pdf_hash, clean_artifacts, clean_chunk_text, normalize_answer
# from docmind_rag.utils.text import (
#     classify_from_context,
#     reorder_by_question,
#     normalize_text,
# )
# from docmind_rag.utils.metrics import (
#     compute_answer_grounding,
#     compute_retrieval_score,
#     compute_context_precision,
#     compute_recall_at_k,
#     semantic_similarity,
#     _is_refusal_answer,
# )


# # ============================================================
# # LOCAL HELPERS
# # ============================================================

# def clean_context_for_llm(chunks):
#     cleaned_chunks = []
#     for chunk in chunks:
#         lines = chunk.split("\n")
#         filtered_lines = []
#         for line in lines:
#             line_strip = line.strip()
#             is_question_like = (
#                 line_strip.endswith("?") or
#                 (len(line_strip.split()) < 15 and "?" in line_strip)
#             )
#             if is_question_like:
#                 continue
#             filtered_lines.append(line)
#         cleaned_chunks.append("\n".join(filtered_lines))
#     return cleaned_chunks


# def is_grounded(answer, retrieved_texts, threshold=0.6):
#     if not answer:
#         return False
#     answer_words = set(answer.lower().split())
#     if not answer_words:
#         return False
#     best_overlap = 0
#     for chunk in retrieved_texts:
#         chunk_words = set(chunk.lower().split())
#         overlap = len(answer_words & chunk_words) / len(answer_words)
#         best_overlap = max(best_overlap, overlap)
#     return best_overlap >= threshold


# # ============================================================
# # NUMERIC INTENT DETECTION
# # ============================================================

# def _extract_numbers(text: str) -> list:
#     return re.findall(r'\b\d+\b', text)


# def _detect_numeric_intent(question: str, chunks: list) -> str:
#     """
#     Detect NAVIGATIONAL or POSITIONAL from chunk structure.
#     Generic — no hardcoded section words.
#     """
#     numbers = _extract_numbers(question)
#     if not numbers:
#         return "NONE"

#     q_words = set(question.lower().split())

#     for chunk in chunks:
#         lines = [l.strip() for l in chunk.split("\n") if l.strip()]
#         if not lines:
#             continue
#         first_line = lines[0].lower()
#         line_words = set(first_line.split())
#         overlap    = len(q_words & line_words)
#         if any(re.search(rf'\b{re.escape(num)}\b', first_line) for num in numbers):
#             if overlap >= 1 and len(first_line.split()) <= 12:
#                 return "NAVIGATIONAL"

#     for chunk in chunks:
#         lines       = [l.strip() for l in chunk.split("\n") if l.strip()]
#         short_lines = [l for l in lines if len(l.split()) <= 12]
#         if len(short_lines) >= 3:
#             first = lines[0].lower() if lines else ""
#             if not any(re.search(rf'\b{re.escape(num)}\b', first) for num in numbers):
#                 return "POSITIONAL"

#     return "NONE"


# def _navigate_full_chunks(question: str, all_raw_chunks: list) -> str:
#     """
#     Scan full raw chunks to find section title for a numbered section.
#     Generic — works for Chapter N, Lecture N, Section N, etc.
#     """
#     numbers = _extract_numbers(question)
#     if not numbers:
#         return ""

#     q_words    = set(question.lower().split())
#     best_line  = ""
#     best_score = 0

#     for chunk in all_raw_chunks:
#         lines = [l.strip() for l in chunk.split("\n") if l.strip()]
#         if not lines:
#             continue
#         first_line  = lines[0]
#         first_lower = first_line.lower()

#         for num in numbers:
#             if not re.search(rf'\b{re.escape(num)}\b', first_lower):
#                 continue
#             line_words = set(first_lower.split())
#             overlap    = len(q_words & line_words)
#             if overlap >= 1 and len(first_line.split()) <= 14:
#                 score = overlap + (5 if len(first_line.split()) <= 8 else 0)
#                 if score > best_score:
#                     best_score = score
#                     best_line  = first_line

#     if not best_line:
#         return ""

#     title = best_line
#     title = re.sub(r'^\s*\d+\s*[-—–]+\s*[—–-]?\s*\d*\s*', '', title)
#     title = re.sub(r'\b\d+\b', '', title)
#     title = re.sub(r'[:\-–—]', ' ', title)
#     title = re.sub(r'\s{2,}', ' ', title)
#     return title.strip()


# def _positional_extract(question: str, retrieved_texts: list) -> str:
#     """Extract Nth item from a list. Generic."""
#     numbers = _extract_numbers(question)
#     if not numbers:
#         return ""
#     try:
#         idx = int(numbers[0]) - 1
#     except ValueError:
#         return ""
#     if idx < 0:
#         return ""

#     best_chunk, best_count = "", 0
#     for chunk in retrieved_texts:
#         lines       = [l.strip() for l in chunk.split("\n") if l.strip()]
#         short_lines = [l for l in lines if len(l.split()) <= 12]
#         if len(short_lines) > best_count:
#             best_count = len(short_lines)
#             best_chunk = chunk

#     if not best_chunk:
#         return ""

#     lines       = [l.strip() for l in best_chunk.split("\n") if l.strip()]
#     short_lines = [l for l in lines if len(l.split()) <= 12]
#     if idx < len(short_lines):
#         return re.sub(r'^\s*\d+[\.\)]\s*', '', short_lines[idx]).strip()
#     return ""


# # ============================================================
# # SIGNAL 3 — Structural consistency for NAVIGATIONAL answers
# # ============================================================

# def _check_identifier_grounded(answer: str, retrieved_texts: list) -> bool:
#     """
#     If answer contains a multi-word identifier (word+number or number+word),
#     verify it appears verbatim in retrieved chunks.
#     Generic — no hardcoded section words.
#     """
#     identifiers = re.findall(
#         r'\b[a-zA-Z]+\s+\d+\b|\b\d+\s+[a-zA-Z]+\b',
#         answer.lower()
#     )
#     if not identifiers:
#         return True

#     chunk_text = " ".join(retrieved_texts).lower()
#     for ident in identifiers:
#         if ident not in chunk_text:
#             print(f"[Signal3] ❌ Identifier '{ident}' not in retrieved chunks")
#             return False
#     return True


# # ============================================================
# # REFUSAL — semantic version (local)
# # ============================================================

# _REFUSAL_ANCHOR = "this information is not available in the provided context"


# def _is_refusal_semantic(text: str) -> bool:
#     if not text or len(text.strip()) < 2:
#         return True
#     sim = semantic_similarity(text, [_REFUSAL_ANCHOR])
#     print(f"[Refusal] sim={sim:.3f} | '{text[:50]}'")
#     return sim > 0.72


# # ============================================================
# # LANGGRAPH NODES
# # ============================================================

# def node_extract(state: DocState) -> DocState:
#     extract_start    = time.time()
#     text, page_count = extract_pdf_parallel(state["pdf_path"])
#     extract_time     = time.time() - extract_start
#     state["extracted_text"] = text
#     state["page_count"]     = page_count
#     state["char_count"]     = len(text)
#     state["metrics"]["extraction_time_sec"]  = round(extract_time, 2)
#     state["metrics"]["pages_processed"]      = page_count
#     state["metrics"]["characters_processed"] = len(text)
#     state["metrics"]["words_processed"]      = len(text.split())
#     return state


# def node_chunk(state: DocState) -> DocState:
#     text = state["extracted_text"]
#     state["query_type"] = "QA"

#     summary_chunks, rag_chunks = semantic_chunk(text)

#     # Normalize unicode symbols so retrieval doesn't fail on mismatches
#     rag_chunks     = [normalize_text(c) for c in rag_chunks]
#     summary_chunks = [normalize_text(c) for c in summary_chunks]

#     state["summary_chunks"] = summary_chunks
#     state["chunks"]         = rag_chunks

#     state["metrics"]["summary_chunks"] = len(summary_chunks)
#     state["metrics"]["chunks_created"] = len(rag_chunks)
#     state["metrics"]["query_type"]     = state["query_type"]
#     print(f"[Chunk] {len(summary_chunks)} summary chunks | {len(rag_chunks)} RAG chunks")
#     return state


# def node_summarize(state: DocState) -> DocState:
#     pdf_hash  = get_pdf_hash(state["pdf_path"])
#     cache_key = pdf_hash

#     if cache_key in _summary_cache:
#         cached = _summary_cache[cache_key]
#         print("[Summary] ✅ Cache hit")
#         emit_event(state.get("request_id", ""), "agent_action",
#                    "⚡ Summary loaded from cache instantly!")
#         state["answer"] = cached["summary"]
#         state["metrics"].update(cached["metrics"])
#         state["metrics"]["type"] = "summary"
#         return state

#     summary_start = time.time()
#     raptor_summarize._request_id = state.get("request_id", "")
#     summary, map_time, reduce_time = raptor_summarize(state["summary_chunks"])
#     summary_time = time.time() - summary_start

#     state["answer"] = summary
#     metrics_snapshot = {
#         "summary_time_sec":     round(summary_time, 2),
#         "summary_length_words": len(summary.split()),
#         "parallel_workers":     min(MAX_WORKERS, len(state["summary_chunks"])),
#         "map_time_sec":         round(map_time, 2),
#         "reduce_time_sec":      round(reduce_time, 2),
#         "llm_calls":            3,
#     }
#     state["metrics"].update(metrics_snapshot)
#     state["metrics"]["type"] = "summary"
#     _summary_cache[cache_key] = {"summary": summary, "metrics": metrics_snapshot}
#     print(f"[Summary] Done ({len(summary.split())} words)")
#     return state


# def node_qa(state: DocState) -> DocState:
#     qa_start_t = time.time()
#     question   = state["question"]
#     request_id = state.get("request_id", "")
#     all_chunks = state["chunks"]

#     # Normalize question same way chunks were normalized
#     question = normalize_text(question)

#     # Raw strings for navigational scanning — must NOT be cleaned
#     all_raw = [
#         d if isinstance(d, str) else d.page_content
#         for d in all_chunks
#     ]

#     pdf_hash    = get_pdf_hash(state["pdf_path"])
#     faiss_index = build_faiss_index(all_chunks, pdf_hash)

#     print(f"[QA] START | {len(all_chunks)} chunks")

#     recall_score      = 0.0
#     retrieval_score   = 0.0
#     context_precision = 0.0
#     grounding         = 0.0
#     llm_calls         = 0
#     model_used        = "llama"
#     confidence        = 0.0
#     decision_type     = "accepted"
#     retrieved         = []
#     retrieved_texts   = []
#     answer            = ""

#     # ── STEP 1: RETRIEVE ─────────────────────────────────────
#     retrieved = multi_query_retrieve(
#         question, faiss_index,
#         k=20,
#         all_chunks=all_chunks,
#         query_type="FACTUAL_QA"
#     )

#     # ── STEP 2: RERANK + SAFE FALLBACK ───────────────────────
#     retrieved, reranker_top, _ = rerank_docs(
#         question, retrieved, top_k=8, apply_pruning=True
#     )
#     retrieved = protect_exact_matches(
#         question, retrieved, all_chunks, top_k=8
#     )

#     # Safe fallback if reranker is too aggressive
#     if not retrieved or len(retrieved) < 3:
#         print("[QA] ⚠️ Reranker too aggressive → fallback to initial chunks")
#         retrieved = all_chunks[:8]

#     retrieved_texts = [
#         d.page_content if hasattr(d, "page_content") else str(d)
#         for d in retrieved
#     ]

#     if reranker_top < -5:
#         print("[Guard] ⚠️ Weak reranker score — continuing")

#     # ── STEP 3: METRICS ───────────────────────────────────────
#     retrieval_score   = compute_retrieval_score(question, retrieved)
#     context_precision = compute_context_precision(question, retrieved)
#     recall_score      = compute_recall_at_k(
#         question, retrieved, all_chunks, k=len(retrieved)
#     )

#     # k expansion: low recall OR numeric question
#     is_numeric_question = len(_extract_numbers(question)) > 0
#     if recall_score < 25 or is_numeric_question:
#         print("[QA] ⚠️ Expanding retrieval (low recall or numeric question)")
#         expanded = multi_query_retrieve(
#             question, faiss_index, k=30,
#             all_chunks=all_chunks, query_type="FACTUAL_QA"
#         )
#         if len(expanded) > len(retrieved):
#             retrieved = expanded
#             retrieved_texts = [
#                 d.page_content if hasattr(d, "page_content") else str(d)
#                 for d in retrieved
#             ]

#     # ── Numeric intent: retrieved first, fallback to all_raw ──
#     numeric_intent = _detect_numeric_intent(question, retrieved_texts)

#     if numeric_intent == "NONE" and is_numeric_question:
#         print("[Navigate] Retrying intent detection on full document...")
#         numeric_intent = _detect_numeric_intent(question, all_raw)
#         if numeric_intent != "NONE":
#             print(f"[Navigate] Intent found in full document: {numeric_intent}")

#     # ── NAVIGATIONAL ──────────────────────────────────────────
#     if numeric_intent == "NAVIGATIONAL":
#         print(f"[Routing] Numeric intent → NAVIGATIONAL")
#         title = _navigate_full_chunks(question, all_raw)
#         if title:
#             print(f"[Navigate] Extracted title: '{title}'")
#             grounding  = compute_answer_grounding(title, retrieved_texts, question)
#             confidence = round(grounding / 100, 3)
#             qa_time    = time.time() - qa_start_t
#             state["answer"]     = title
#             state["query_type"] = "NAVIGATIONAL"
#             _write_metrics(state, "navigational", "navigational", grounding,
#                            confidence, retrieval_score, context_precision,
#                            recall_score, llm_calls, retrieved, qa_time)
#             return state
#         print(f"[Navigate] Extraction failed — falling through to LLM")

#     # ── POSITIONAL ────────────────────────────────────────────
#     elif numeric_intent == "POSITIONAL":
#         print(f"[Routing] Numeric intent → POSITIONAL")
#         pos_answer = _positional_extract(question, retrieved_texts)
#         if pos_answer:
#             print(f"[Positional] Extracted: '{pos_answer}'")
#             grounding  = compute_answer_grounding(pos_answer, retrieved_texts, question)
#             confidence = round(grounding / 100, 3)
#             qa_time    = time.time() - qa_start_t
#             state["answer"]     = pos_answer
#             state["query_type"] = "POSITIONAL"
#             _write_metrics(state, "positional", "positional", grounding,
#                            confidence, retrieval_score, context_precision,
#                            recall_score, llm_calls, retrieved, qa_time)
#             return state
#         print(f"[Positional] Extraction failed — falling through to LLM")

#     # ── Normal routing ────────────────────────────────────────
#     query_type = classify_from_context(question, retrieved_texts)
#     state["query_type"] = query_type
#     print(f"[Routing] Context-based → {query_type}")

#     # ── STEP 4: ANSWER GENERATION ─────────────────────────────

#     if query_type == "FULL_SUMMARY":
#         retrieved_texts = clean_context_for_llm(retrieved_texts)
#         ranked  = reorder_by_question(question, retrieved_texts)
#         context = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:8])
#         structured_context = "\n".join(
#             f"[CONTEXT CHUNK]\n{c}" for c in context.split("\n\n---\n\n")
#         )
#         emit_event(request_id, "stream_start", "✍️ Generating answer...")
#         answer, _ = call_llama_streaming(
#             QA_PROMPT.format(context=structured_context[:2500], question=question),
#             request_id=request_id, temperature=0.0
#         )
#         answer     = clean_artifacts(answer).strip()
#         model_used = "llama_summary"
#         llm_calls  = 1

#     elif query_type == "MULTIPART_QA":
#         retrieved_texts = clean_context_for_llm(retrieved_texts)
#         ranked  = reorder_by_question(question, retrieved_texts)
#         context = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:8])
#         emit_event(request_id, "agent_start",
#                    f"🤖 MULTIPART | {len(retrieved)} chunks")
#         emit_event(request_id, "stream_start", "✍️ Generating answer...")
#         # 4000 chars to capture all list items
#         answer, _ = call_llama_streaming(
#             QA_PROMPT.format(context=context[:4000], question=question),
#             request_id=request_id, temperature=0.0
#         )
#         answer     = clean_artifacts(answer).strip()
#         model_used = "llama_multipart"
#         llm_calls  = 1

#     else:
#         # FACTUAL_QA / VERIFICATION_QA
#         # if not retrieved_texts:
#         #     print("[QA] ❌ No context → NOT FOUND")
#         #     state["answer"] = "This information is not present in the document."
#         #     _write_metrics(...)
#         #     return state
#         # 🔥 FIX — fallback instead of immediate failure
#         if not retrieved_texts or len(retrieved_texts) < 3:
#             print("[QA] ⚠️ No/weak context → forcing fallback")

#             retrieved = all_chunks[:8]
#             retrieved_texts = [
#                 d if isinstance(d, str) else d.page_content
#                 for d in retrieved
#             ]
#         # ── FIX 1: Always use ReAct — no Direct QA bypass ────
#         # Direct QA was causing regressions by skipping the full
#         # # ReAct reasoning chain for short questions.
#         # react_ans, model_used, _, _, _, _, react_calls, retrieved_texts = react_agent(
#         #     question, faiss_index, query_type, all_chunks, request_id
#         # )
#         react_ans, model_used, _, _, _, _, react_calls = react_agent(
#     question, retrieved_texts, query_type, request_id, embedding_model
# )
#         llm_calls += react_calls

#         react_ans = clean_artifacts(react_ans).strip().strip('"').strip("'")
#         # react_ans = clean_reasoning_answer(react_ans, question, retrieved_texts)
#         print(f"[QA] Answer: '{react_ans[:60]}'")

#         # Refusal check
#         if _is_refusal_answer(react_ans):
#             print("[QA] ⚠️ Refusal detected")
#             qa_time = time.time() - qa_start_t
#             state["answer"] = "This information is not present in the document."
#             _write_metrics(state, "not_found", "not_found",
#                            75.0, 0.75, retrieval_score, context_precision,
#                            recall_score, llm_calls, retrieved, qa_time)
#             return state

#         grounding_score = compute_answer_grounding(react_ans, retrieved_texts, question)
#         length          = len(react_ans.split())

#         # ── FIX 2: VERIFICATION_QA — write metrics before returning ──
#         # Short answers (Yes/No) can't score high on word overlap so
#         # they use a lower grounding threshold. All branches write
#         # metrics before returning — previously this was missing.
#         if query_type == "VERIFICATION_QA":
#             if length <= 3 and grounding_score >= 30:
#                 # Short verification answer with minimal grounding — accept
#                 answer        = react_ans
#                 decision_type = "accepted"
#             elif grounding_score < 80:
#                 print("[QA] ❌ Verification failed → NOT FOUND")
#                 answer        = "This information is not present in the document."
#                 decision_type = "verification_failed"
#             else:
#                 answer        = react_ans
#                 decision_type = "accepted"

#             grounding  = compute_answer_grounding(answer, retrieved_texts, question)
#             confidence = round(grounding / 100, 3)
#             qa_time    = time.time() - qa_start_t
#             state["answer"] = normalize_answer(answer)
#             _write_metrics(state, model_used, decision_type, grounding,
#                            confidence, retrieval_score, context_precision,
#                            recall_score, llm_calls, retrieved, qa_time)
#             return state

#         # ── FIX 3: Trust gate — individual signals, no AND condition ──
#         # Previous version required recall >= 50 AND grounding >= 60
#         # simultaneously, which rejected valid answers with moderate
#         # recall but high grounding. Each signal now guards independently.

#         # Signal 1 — no reliable context
#         if recall_score < 25:
#             print("[QA] ❌ Low recall → NOT FOUND")
#             answer        = "This information is not present in the document."
#             decision_type = "low_recall"

#         # Signal 2 — answer not grounded in context
#         elif grounding_score < 40:
#             print("[QA] ❌ Very low grounding → NOT FOUND")
#             answer        = "This information is not present in the document."
#             decision_type = "low_grounding"

#         # Signal 3 — short answer needs stronger grounding signal
#         #elif length <= 3 and grounding_score < 60:
#          #   print("[QA] ❌ Weak short answer → NOT FOUND")
#           #  answer        = "This information is not present in the document."
#            # decision_type = "weak_short"
#         elif length <= 3:
#             # 🔥 TRUST SHORT FACTUAL ANSWERS
#             answer = react_ans
#             decision_type = "accepted"

#         # All signals passed — accept
#         else:
#             answer        = react_ans
#             decision_type = "accepted"

#     # ── STEP 5: METRICS ───────────────────────────────────────
#     grounding  = compute_answer_grounding(answer, retrieved_texts, question)
#     confidence = round(grounding / 100, 3)
#     qa_time    = time.time() - qa_start_t

#     print(f"[QA] Done in {qa_time:.1f}s | model={model_used} | "
#           f"grounding={grounding:.1f}% | recall={recall_score:.1f}% | "
#           f"confidence={confidence:.3f} | decision={decision_type}")

#     if not answer.strip():
#         answer = "Could not find a relevant answer in the PDF."
#     else:
#         answer = normalize_answer(answer)

#     state["answer"] = answer
#     _write_metrics(state, model_used, decision_type, grounding,
#                    confidence, retrieval_score, context_precision,
#                    recall_score, llm_calls, retrieved, qa_time)
#     return state


# def node_validate(state: DocState) -> DocState:
#     answer = state["answer"]
#     retry  = state.get("retry_count", 0)

#     if len(answer.strip()) < 3 and retry < 2:
#         state["retry_count"] = retry + 1
#         state["answer"]      = ""
#         return state

#     total_time    = time.time() - state["start_time"]
#     output_words  = len(answer.split())
#     output_tokens = output_words * 1.3
#     extract_time  = state["metrics"].get("extraction_time_sec", 0)
#     llm_time      = max(total_time - extract_time, 1)
#     tps           = round(output_tokens / llm_time, 2) if llm_time > 0 else 0
#     m             = state["metrics"]

#     m["response_time_sec"]    = round(total_time, 2)
#     m["extraction_time_sec"]  = m.get("extraction_time_sec", 0)
#     m["pages_processed"]      = state.get("page_count", 0)
#     m["characters_processed"] = state.get("char_count", 0)
#     m["words_processed"]      = len(state.get("extracted_text", "").split())

#     if m.get("type") == "summary":
#         m["summary_time_sec"]     = m.get("summary_time_sec", 0)
#         m["summary_length_words"] = len(answer.split())
#     if m.get("type") == "qa":
#         m["qa_time_sec"]      = m.get("qa_time_sec", 0)
#         m["confidence_score"] = m.get("confidence_score", 0)

#     m["ttft_sec"]          = round(total_time, 2)
#     m["e2e_latency_sec"]   = round(total_time, 2)
#     m["tps"]               = tps
#     m["query_type"]        = state.get("query_type", "")
#     m["chunks_created"]    = m.get("chunks_created", 0)
#     m["retry_count"]       = retry
#     m["model_used"]        = m.get("model_used", "llama_react")
#     m["llm_calls"]         = m.get("llm_calls", 0)
#     m["retrieval_score"]   = m.get("retrieval_score", 0)
#     m["context_precision"] = m.get("context_precision", 0)
#     m["answer_grounding"]  = m.get("answer_grounding", 0)
#     m["recall_at_k"]       = m.get("recall_at_k", 0)

#     if m.get("type") == "summary":
#         m["parallel_workers"] = m.get("parallel_workers", 0)
#         m["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
#         m["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)
#     if m.get("type") == "qa":
#         m["chunks_retrieved"] = m.get("chunks_retrieved", 0)
#         m["decision_type"]    = m.get("decision_type", "accepted")
#         m["confidence_raw"]   = m.get("confidence_raw", 0.0)

#     state["metrics"] = m
#     return state


# def _write_metrics(state, model_used, decision_type, grounding,
#                    confidence, retrieval_score, context_precision,
#                    recall_score, llm_calls, retrieved, qa_time):
#     m = state.setdefault("metrics", {})
#     m["qa_time_sec"]       = round(qa_time, 2)
#     m["confidence_score"]  = round(confidence * 100, 2)
#     m["retrieval_score"]   = retrieval_score
#     m["context_precision"] = context_precision
#     m["answer_grounding"]  = grounding
#     m["recall_at_k"]       = recall_score
#     m["llm_calls"]         = llm_calls
#     m["model_used"]        = model_used
#     m["chunks_retrieved"]  = len(retrieved)
#     m["type"]              = "qa"
#     m["decision_type"]     = decision_type
#     m["confidence_raw"]    = round(confidence, 4)








# 80, 80 news 50
# import time
# import re
# import spacy
# _nlp = spacy.load("en_core_web_lg")

# from docmind_rag.config.settings import (
#     MAX_WORKERS, FACTUAL_TOP_K, MULTIPART_TOP_K
# )
# from docmind_rag.utils.text import extract_named_entities, expand_answer
# from docmind_rag.core.state import DocState
# from docmind_rag.core.prompts import QA_PROMPT
# from docmind_rag.models.llm import call_llama_streaming
# from docmind_rag.models.embeddings import build_faiss_index
# from docmind_rag.models.reranker import rerank_docs, protect_exact_matches
# from docmind_rag.services.extraction import extract_pdf_parallel, _extraction_cache
# from docmind_rag.services.chunking import semantic_chunk, raptor_summarize, _summary_cache
# from docmind_rag.services.retrieval import multi_query_retrieve
# from docmind_rag.services.agent import react_agent, roberta_qa, clean_reasoning_answer
# from docmind_rag.events.events import emit_event
# from docmind_rag.utils.helpers import get_pdf_hash, clean_artifacts, clean_chunk_text, normalize_answer
# from docmind_rag.utils.text import (
#     classify_from_context,
#     reorder_by_question,
#     normalize_text,
# )
# from docmind_rag.utils.metrics import (
#     compute_answer_grounding,
#     compute_retrieval_score,
#     compute_context_precision,
#     compute_recall_at_k,
#     semantic_similarity,
#     _is_refusal_answer,
# )


# # ============================================================
# # LOCAL HELPERS
# # ============================================================

# def clean_context_for_llm(chunks):
#     cleaned_chunks = []
#     for chunk in chunks:
#         lines = chunk.split("\n")
#         filtered_lines = []
#         for line in lines:
#             line_strip = line.strip()
#             is_question_like = (
#                 line_strip.endswith("?") or
#                 (len(line_strip.split()) < 15 and "?" in line_strip)
#             )
#             if is_question_like:
#                 continue
#             filtered_lines.append(line)
#         cleaned_chunks.append("\n".join(filtered_lines))
#     return cleaned_chunks


# def is_grounded(answer, retrieved_texts, threshold=0.6):
#     if not answer:
#         return False
#     answer_words = set(answer.lower().split())
#     if not answer_words:
#         return False
#     best_overlap = 0
#     for chunk in retrieved_texts:
#         chunk_words = set(chunk.lower().split())
#         overlap = len(answer_words & chunk_words) / len(answer_words)
#         best_overlap = max(best_overlap, overlap)
#     return best_overlap >= threshold


# # ============================================================
# # NUMERIC INTENT DETECTION
# # ============================================================

# def _extract_numbers(text: str) -> list:
#     return re.findall(r'\b\d+\b', text)


# def _detect_numeric_intent(question: str, chunks: list) -> str:
#     """
#     Detect NAVIGATIONAL or POSITIONAL from chunk structure.
#     Generic — no hardcoded section words.
#     """
#     numbers = _extract_numbers(question)
#     if not numbers:
#         return "NONE"

#     q_words = set(question.lower().split())

#     for chunk in chunks:
#         lines = [l.strip() for l in chunk.split("\n") if l.strip()]
#         if not lines:
#             continue
#         first_line = lines[0].lower()
#         line_words = set(first_line.split())
#         overlap    = len(q_words & line_words)
#         if any(re.search(rf'\b{re.escape(num)}\b', first_line) for num in numbers):
#             if overlap >= 1 and len(first_line.split()) <= 12:
#                 return "NAVIGATIONAL"

#     for chunk in chunks:
#         lines       = [l.strip() for l in chunk.split("\n") if l.strip()]
#         short_lines = [l for l in lines if len(l.split()) <= 12]
#         if len(short_lines) >= 3:
#             first = lines[0].lower() if lines else ""
#             if not any(re.search(rf'\b{re.escape(num)}\b', first) for num in numbers):
#                 return "POSITIONAL"

#     return "NONE"


# def _navigate_full_chunks(question: str, all_raw_chunks: list) -> str:
#     """
#     Scan full raw chunks to find section title for a numbered section.
#     Generic — works for Chapter N, Lecture N, Section N, etc.
#     """
#     numbers = _extract_numbers(question)
#     if not numbers:
#         return ""

#     q_words    = set(question.lower().split())
#     best_line  = ""
#     best_score = 0

#     for chunk in all_raw_chunks:
#         lines = [l.strip() for l in chunk.split("\n") if l.strip()]
#         if not lines:
#             continue
#         first_line  = lines[0]
#         first_lower = first_line.lower()

#         for num in numbers:
#             if not re.search(rf'\b{re.escape(num)}\b', first_lower):
#                 continue
#             line_words = set(first_lower.split())
#             overlap    = len(q_words & line_words)
#             if overlap >= 1 and len(first_line.split()) <= 14:
#                 score = overlap + (5 if len(first_line.split()) <= 8 else 0)
#                 if score > best_score:
#                     best_score = score
#                     best_line  = first_line

#     if not best_line:
#         return ""

#     title = best_line
#     title = re.sub(r'^\s*\d+\s*[-—–]+\s*[—–-]?\s*\d*\s*', '', title)
#     title = re.sub(r'\b\d+\b', '', title)
#     title = re.sub(r'[:\-–—]', ' ', title)
#     title = re.sub(r'\s{2,}', ' ', title)
#     return title.strip()


# def _positional_extract(question: str, retrieved_texts: list) -> str:
#     """Extract Nth item from a list. Generic."""
#     numbers = _extract_numbers(question)
#     if not numbers:
#         return ""
#     try:
#         idx = int(numbers[0]) - 1
#     except ValueError:
#         return ""
#     if idx < 0:
#         return ""

#     best_chunk, best_count = "", 0
#     for chunk in retrieved_texts:
#         lines       = [l.strip() for l in chunk.split("\n") if l.strip()]
#         short_lines = [l for l in lines if len(l.split()) <= 12]
#         if len(short_lines) > best_count:
#             best_count = len(short_lines)
#             best_chunk = chunk

#     if not best_chunk:
#         return ""

#     lines       = [l.strip() for l in best_chunk.split("\n") if l.strip()]
#     short_lines = [l for l in lines if len(l.split()) <= 12]
#     if idx < len(short_lines):
#         return re.sub(r'^\s*\d+[\.\)]\s*', '', short_lines[idx]).strip()
#     return ""


# # ============================================================
# # SIGNAL 3 — Structural consistency for NAVIGATIONAL answers
# # ============================================================

# def _check_identifier_grounded(answer: str, retrieved_texts: list) -> bool:
#     """
#     If answer contains a multi-word identifier (word+number or number+word),
#     verify it appears verbatim in retrieved chunks.
#     Generic — no hardcoded section words.
#     """
#     identifiers = re.findall(
#         r'\b[a-zA-Z]+\s+\d+\b|\b\d+\s+[a-zA-Z]+\b',
#         answer.lower()
#     )
#     if not identifiers:
#         return True

#     chunk_text = " ".join(retrieved_texts).lower()
#     for ident in identifiers:
#         if ident not in chunk_text:
#             print(f"[Signal3] ❌ Identifier '{ident}' not in retrieved chunks")
#             return False
#     return True


# # ============================================================
# # REFUSAL — semantic version (local)
# # ============================================================

# _REFUSAL_ANCHOR = "this information is not available in the provided context"


# def _is_refusal_semantic(text: str) -> bool:
#     if not text or len(text.strip()) < 2:
#         return True
#     sim = semantic_similarity(text, [_REFUSAL_ANCHOR])
#     print(f"[Refusal] sim={sim:.3f} | '{text[:50]}'")
#     return sim > 0.72


# # ============================================================
# # LANGGRAPH NODES
# # ============================================================

# def node_extract(state: DocState) -> DocState:
#     extract_start    = time.time()
#     text, page_count = extract_pdf_parallel(state["pdf_path"])
#     extract_time     = time.time() - extract_start
#     state["extracted_text"] = text
#     state["page_count"]     = page_count
#     state["char_count"]     = len(text)
#     state["metrics"]["extraction_time_sec"]  = round(extract_time, 2)
#     state["metrics"]["pages_processed"]      = page_count
#     state["metrics"]["characters_processed"] = len(text)
#     state["metrics"]["words_processed"]      = len(text.split())
#     return state


# def node_chunk(state: DocState) -> DocState:
#     text = state["extracted_text"]
#     state["query_type"] = "QA"

#     summary_chunks, rag_chunks = semantic_chunk(text)

#     # Normalize unicode symbols so retrieval doesn't fail on mismatches
#     rag_chunks     = [normalize_text(c) for c in rag_chunks]
#     summary_chunks = [normalize_text(c) for c in summary_chunks]

#     state["summary_chunks"] = summary_chunks
#     state["chunks"]         = rag_chunks

#     state["metrics"]["summary_chunks"] = len(summary_chunks)
#     state["metrics"]["chunks_created"] = len(rag_chunks)
#     state["metrics"]["query_type"]     = state["query_type"]
#     print(f"[Chunk] {len(summary_chunks)} summary chunks | {len(rag_chunks)} RAG chunks")
#     return state


# def node_summarize(state: DocState) -> DocState:
#     pdf_hash  = get_pdf_hash(state["pdf_path"])
#     cache_key = pdf_hash

#     if cache_key in _summary_cache:
#         cached = _summary_cache[cache_key]
#         print("[Summary] ✅ Cache hit")
#         emit_event(state.get("request_id", ""), "agent_action",
#                    "⚡ Summary loaded from cache instantly!")
#         state["answer"] = cached["summary"]
#         state["metrics"].update(cached["metrics"])
#         state["metrics"]["type"] = "summary"
#         return state

#     summary_start = time.time()
#     raptor_summarize._request_id = state.get("request_id", "")
#     summary, map_time, reduce_time = raptor_summarize(state["summary_chunks"])
#     summary_time = time.time() - summary_start

#     state["answer"] = summary
#     metrics_snapshot = {
#         "summary_time_sec":     round(summary_time, 2),
#         "summary_length_words": len(summary.split()),
#         "parallel_workers":     min(MAX_WORKERS, len(state["summary_chunks"])),
#         "map_time_sec":         round(map_time, 2),
#         "reduce_time_sec":      round(reduce_time, 2),
#         "llm_calls":            3,
#     }
#     state["metrics"].update(metrics_snapshot)
#     state["metrics"]["type"] = "summary"
#     _summary_cache[cache_key] = {"summary": summary, "metrics": metrics_snapshot}
#     print(f"[Summary] Done ({len(summary.split())} words)")
#     return state


# def node_qa(state: DocState) -> DocState:
#     qa_start_t = time.time()
#     question   = state["question"]
#     request_id = state.get("request_id", "")
#     all_chunks = state["chunks"]

#     # Normalize question same way chunks were normalized
#     question = normalize_text(question)

#     # Raw strings for navigational scanning — must NOT be cleaned
#     all_raw = [
#         d if isinstance(d, str) else d.page_content
#         for d in all_chunks
#     ]

#     pdf_hash    = get_pdf_hash(state["pdf_path"])
#     faiss_index = build_faiss_index(all_chunks, pdf_hash)

#     print(f"[QA] START | {len(all_chunks)} chunks")

#     recall_score      = 0.0
#     retrieval_score   = 0.0
#     context_precision = 0.0
#     grounding         = 0.0
#     llm_calls         = 0
#     model_used        = "llama"
#     confidence        = 0.0
#     decision_type     = "accepted"
#     retrieved         = []
#     retrieved_texts   = []
#     answer            = ""

#     # ── STEP 1: RETRIEVE ─────────────────────────────────────
#     retrieved = multi_query_retrieve(
#         question, faiss_index,
#         k=20,
#         all_chunks=all_chunks,
#         query_type="FACTUAL_QA"
#     )

#     # ── STEP 2: RERANK + SAFE FALLBACK ───────────────────────
#     retrieved, reranker_top, _ = rerank_docs(
#         question, retrieved, top_k=8, apply_pruning=True
#     )
#     retrieved = protect_exact_matches(
#         question, retrieved, all_chunks, top_k=8
#     )

#     # Safe fallback if reranker is too aggressive
#     if not retrieved or len(retrieved) < 3:
#         print("[QA] ⚠️ Reranker too aggressive → fallback to initial chunks")
#         retrieved = all_chunks[:8]

#     retrieved_texts = [
#         d.page_content if hasattr(d, "page_content") else str(d)
#         for d in retrieved
#     ]

#     if reranker_top < -5:
#         print("[Guard] ⚠️ Weak reranker score — continuing")

#     # ── STEP 3: METRICS ───────────────────────────────────────
#     retrieval_score   = compute_retrieval_score(question, retrieved)
#     context_precision = compute_context_precision(question, retrieved)
#     recall_score      = compute_recall_at_k(
#         question, retrieved, all_chunks, k=len(retrieved)
#     )

#     # k expansion: low recall OR numeric question
#     is_numeric_question = len(_extract_numbers(question)) > 0
#     if recall_score < 25 or is_numeric_question:
#         print("[QA] ⚠️ Expanding retrieval (low recall or numeric question)")
#         expanded = multi_query_retrieve(
#             question, faiss_index, k=30,
#             all_chunks=all_chunks, query_type="FACTUAL_QA"
#         )
#         if len(expanded) > len(retrieved):
#             retrieved = expanded
#             retrieved_texts = [
#                 d.page_content if hasattr(d, "page_content") else str(d)
#                 for d in retrieved
#             ]

#     # ── Numeric intent: retrieved first, fallback to all_raw ──
#     numeric_intent = _detect_numeric_intent(question, retrieved_texts)

#     if numeric_intent == "NONE" and is_numeric_question:
#         print("[Navigate] Retrying intent detection on full document...")
#         numeric_intent = _detect_numeric_intent(question, all_raw)
#         if numeric_intent != "NONE":
#             print(f"[Navigate] Intent found in full document: {numeric_intent}")

#     # ── NAVIGATIONAL ──────────────────────────────────────────
#     if numeric_intent == "NAVIGATIONAL":
#         print(f"[Routing] Numeric intent → NAVIGATIONAL")
#         title = _navigate_full_chunks(question, all_raw)
#         if title:
#             print(f"[Navigate] Extracted title: '{title}'")
#             grounding  = compute_answer_grounding(title, retrieved_texts, question)
#             confidence = round(grounding / 100, 3)
#             qa_time    = time.time() - qa_start_t
#             state["answer"]     = title
#             state["query_type"] = "NAVIGATIONAL"
#             _write_metrics(state, "navigational", "navigational", grounding,
#                            confidence, retrieval_score, context_precision,
#                            recall_score, llm_calls, retrieved, qa_time)
#             return state
#         print(f"[Navigate] Extraction failed — falling through to LLM")

#     # ── POSITIONAL ────────────────────────────────────────────
#     elif numeric_intent == "POSITIONAL":
#         print(f"[Routing] Numeric intent → POSITIONAL")
#         pos_answer = _positional_extract(question, retrieved_texts)
#         if pos_answer:
#             print(f"[Positional] Extracted: '{pos_answer}'")
#             grounding  = compute_answer_grounding(pos_answer, retrieved_texts, question)
#             confidence = round(grounding / 100, 3)
#             qa_time    = time.time() - qa_start_t
#             state["answer"]     = pos_answer
#             state["query_type"] = "POSITIONAL"
#             _write_metrics(state, "positional", "positional", grounding,
#                            confidence, retrieval_score, context_precision,
#                            recall_score, llm_calls, retrieved, qa_time)
#             return state
#         print(f"[Positional] Extraction failed — falling through to LLM")

#     # ── Normal routing ────────────────────────────────────────
#     query_type = classify_from_context(question, retrieved_texts)
#     state["query_type"] = query_type
#     print(f"[Routing] Context-based → {query_type}")

#     # ── STEP 4: ANSWER GENERATION ─────────────────────────────

#     if query_type == "FULL_SUMMARY":
#         retrieved_texts = clean_context_for_llm(retrieved_texts)
#         ranked  = reorder_by_question(question, retrieved_texts)
#         context = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:8])
#         structured_context = "\n".join(
#             f"[CONTEXT CHUNK]\n{c}" for c in context.split("\n\n---\n\n")
#         )
#         emit_event(request_id, "stream_start", "✍️ Generating answer...")
#         answer, _ = call_llama_streaming(
#             QA_PROMPT.format(context=structured_context[:2500], question=question),
#             request_id=request_id, temperature=0.0
#         )
#         answer     = clean_artifacts(answer).strip()
#         model_used = "llama_summary"
#         llm_calls  = 1

#     elif query_type == "MULTIPART_QA":
#         retrieved_texts = clean_context_for_llm(retrieved_texts)
#         ranked  = reorder_by_question(question, retrieved_texts)
#         context = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:8])
#         emit_event(request_id, "agent_start",
#                    f"🤖 MULTIPART | {len(retrieved)} chunks")
#         emit_event(request_id, "stream_start", "✍️ Generating answer...")
#         # 4000 chars to capture all list items
#         answer, _ = call_llama_streaming(
#             QA_PROMPT.format(context=context[:4000], question=question),
#             request_id=request_id, temperature=0.0
#         )
#         answer     = clean_artifacts(answer).strip()
#         model_used = "llama_multipart"
#         llm_calls  = 1

#     else:
#         # FACTUAL_QA / VERIFICATION_QA

#         if not retrieved_texts:
#             print("[QA] ❌ No context → NOT FOUND")
#             state["answer"] = "This information is not present in the document."
#             _write_metrics(state, "not_found", "no_context",
#                            0.0, 0.0, retrieval_score, context_precision,
#                            recall_score, llm_calls, retrieved,
#                            time.time() - qa_start_t)
#             return state

#         # ── FIX 1: Always use ReAct — no Direct QA bypass ────
#         # Direct QA was causing regressions by skipping the full
#         # ReAct reasoning chain for short questions.
#         react_ans, model_used, _, _, _, _, react_calls, retrieved_texts = react_agent(
#             question, faiss_index, query_type, all_chunks, request_id
#         )
#         llm_calls += react_calls

#         react_ans = clean_artifacts(react_ans).strip().strip('"').strip("'")
#         react_ans = clean_reasoning_answer(react_ans, question)
#         print(f"[QA] Answer: '{react_ans[:60]}'")

#         # Refusal check
#         if _is_refusal_answer(react_ans):
#             print("[QA] ⚠️ Refusal detected")
#             qa_time = time.time() - qa_start_t
#             state["answer"] = "This information is not present in the document."
#             _write_metrics(state, "not_found", "not_found",
#                            75.0, 0.75, retrieval_score, context_precision,
#                            recall_score, llm_calls, retrieved, qa_time)
#             return state

#         grounding_score = compute_answer_grounding(react_ans, retrieved_texts, question)
#         length          = len(react_ans.split())

#         # ── FIX 2: VERIFICATION_QA — write metrics before returning ──
#         # Short answers (Yes/No) can't score high on word overlap so
#         # they use a lower grounding threshold. All branches write
#         # metrics before returning — previously this was missing.
#         if query_type == "VERIFICATION_QA":
#             if length <= 3 and grounding_score >= 30:
#                 # Short verification answer with minimal grounding — accept
#                 answer        = react_ans
#                 decision_type = "accepted"
#             elif grounding_score < 80:
#                 print("[QA] ❌ Verification failed → NOT FOUND")
#                 answer        = "This information is not present in the document."
#                 decision_type = "verification_failed"
#             else:
#                 answer        = react_ans
#                 decision_type = "accepted"

#             grounding  = compute_answer_grounding(answer, retrieved_texts, question)
#             confidence = round(grounding / 100, 3)
#             qa_time    = time.time() - qa_start_t
#             state["answer"] = normalize_answer(answer)
#             _write_metrics(state, model_used, decision_type, grounding,
#                            confidence, retrieval_score, context_precision,
#                            recall_score, llm_calls, retrieved, qa_time)
#             return state

#         # ── FIX 3: Trust gate — individual signals, no AND condition ──
#         # Previous version required recall >= 50 AND grounding >= 60
#         # simultaneously, which rejected valid answers with moderate
#         # recall but high grounding. Each signal now guards independently.

#         # Signal 1 — no reliable context
#         if recall_score < 25:
#             print("[QA] ❌ Low recall → NOT FOUND")
#             answer        = "This information is not present in the document."
#             decision_type = "low_recall"

#         # Signal 2 — answer not grounded in context
#         elif grounding_score < 40:
#             print("[QA] ❌ Very low grounding → NOT FOUND")
#             answer        = "This information is not present in the document."
#             decision_type = "low_grounding"

#         # Signal 3 — short answer needs stronger grounding signal
#         elif length <= 3 and grounding_score < 60:
#             print("[QA] ❌ Weak short answer → NOT FOUND")
#             answer        = "This information is not present in the document."
#             decision_type = "weak_short"

#         # All signals passed — accept
#         else:
#             answer        = react_ans
#             decision_type = "accepted"

#     # ── STEP 5: METRICS ───────────────────────────────────────
#     grounding  = compute_answer_grounding(answer, retrieved_texts, question)
#     confidence = round(grounding / 100, 3)
#     qa_time    = time.time() - qa_start_t

#     print(f"[QA] Done in {qa_time:.1f}s | model={model_used} | "
#           f"grounding={grounding:.1f}% | recall={recall_score:.1f}% | "
#           f"confidence={confidence:.3f} | decision={decision_type}")

#     if not answer.strip():
#         answer = "Could not find a relevant answer in the PDF."
#     else:
#         answer = normalize_answer(answer)

#     state["answer"] = answer
#     _write_metrics(state, model_used, decision_type, grounding,
#                    confidence, retrieval_score, context_precision,
#                    recall_score, llm_calls, retrieved, qa_time)
#     return state


# def node_validate(state: DocState) -> DocState:
#     answer = state["answer"]
#     retry  = state.get("retry_count", 0)

#     if len(answer.strip()) < 3 and retry < 2:
#         state["retry_count"] = retry + 1
#         state["answer"]      = ""
#         return state

#     total_time    = time.time() - state["start_time"]
#     output_words  = len(answer.split())
#     output_tokens = output_words * 1.3
#     extract_time  = state["metrics"].get("extraction_time_sec", 0)
#     llm_time      = max(total_time - extract_time, 1)
#     tps           = round(output_tokens / llm_time, 2) if llm_time > 0 else 0
#     m             = state["metrics"]

#     m["response_time_sec"]    = round(total_time, 2)
#     m["extraction_time_sec"]  = m.get("extraction_time_sec", 0)
#     m["pages_processed"]      = state.get("page_count", 0)
#     m["characters_processed"] = state.get("char_count", 0)
#     m["words_processed"]      = len(state.get("extracted_text", "").split())

#     if m.get("type") == "summary":
#         m["summary_time_sec"]     = m.get("summary_time_sec", 0)
#         m["summary_length_words"] = len(answer.split())
#     if m.get("type") == "qa":
#         m["qa_time_sec"]      = m.get("qa_time_sec", 0)
#         m["confidence_score"] = m.get("confidence_score", 0)

#     m["ttft_sec"]          = round(total_time, 2)
#     m["e2e_latency_sec"]   = round(total_time, 2)
#     m["tps"]               = tps
#     m["query_type"]        = state.get("query_type", "")
#     m["chunks_created"]    = m.get("chunks_created", 0)
#     m["retry_count"]       = retry
#     m["model_used"]        = m.get("model_used", "llama_react")
#     m["llm_calls"]         = m.get("llm_calls", 0)
#     m["retrieval_score"]   = m.get("retrieval_score", 0)
#     m["context_precision"] = m.get("context_precision", 0)
#     m["answer_grounding"]  = m.get("answer_grounding", 0)
#     m["recall_at_k"]       = m.get("recall_at_k", 0)

#     if m.get("type") == "summary":
#         m["parallel_workers"] = m.get("parallel_workers", 0)
#         m["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
#         m["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)
#     if m.get("type") == "qa":
#         m["chunks_retrieved"] = m.get("chunks_retrieved", 0)
#         m["decision_type"]    = m.get("decision_type", "accepted")
#         m["confidence_raw"]   = m.get("confidence_raw", 0.0)

#     state["metrics"] = m
#     return state


# def _write_metrics(state, model_used, decision_type, grounding,
#                    confidence, retrieval_score, context_precision,
#                    recall_score, llm_calls, retrieved, qa_time):
#     m = state.setdefault("metrics", {})
#     m["qa_time_sec"]       = round(qa_time, 2)
#     m["confidence_score"]  = round(confidence * 100, 2)
#     m["retrieval_score"]   = retrieval_score
#     m["context_precision"] = context_precision
#     m["answer_grounding"]  = grounding
#     m["recall_at_k"]       = recall_score
#     m["llm_calls"]         = llm_calls
#     m["model_used"]        = model_used
#     m["chunks_retrieved"]  = len(retrieved)
#     m["type"]              = "qa"
#     m["decision_type"]     = decision_type
#     m["confidence_raw"]    = round(confidence, 4)









# # 90,80,50
# import time
# import re
# import spacy
# _nlp = spacy.load("en_core_web_lg")

# from docmind_rag.config.settings import (
#     MAX_WORKERS, FACTUAL_TOP_K, MULTIPART_TOP_K
# )
# from docmind_rag.utils.text import extract_named_entities, expand_answer
# from docmind_rag.core.state import DocState
# from docmind_rag.core.prompts import QA_PROMPT
# from docmind_rag.models.llm import call_llama_streaming
# from docmind_rag.models.embeddings import build_faiss_index
# from docmind_rag.models.reranker import rerank_docs, protect_exact_matches
# from docmind_rag.services.extraction import extract_pdf_parallel, _extraction_cache
# from docmind_rag.services.chunking import semantic_chunk, raptor_summarize, _summary_cache
# from docmind_rag.services.retrieval import multi_query_retrieve
# from docmind_rag.services.agent import react_agent, roberta_qa, clean_reasoning_answer
# from docmind_rag.events.events import emit_event
# from docmind_rag.utils.helpers import get_pdf_hash, clean_artifacts, clean_chunk_text, normalize_answer
# from docmind_rag.utils.text import (
#     classify_from_context,
#     reorder_by_question,
#     normalize_text,
# )
# from docmind_rag.utils.metrics import (
#     compute_answer_grounding,
#     compute_retrieval_score,
#     compute_context_precision,
#     compute_recall_at_k,
#     semantic_similarity,
#     _is_refusal_answer,
# )

# # def trust_gate(answer: str, metrics: dict) -> str:
# #     recall = metrics.get("recall_at_k", 0)
# #     grounding = metrics.get("answer_grounding", 0)
# #     length = len(answer.split())

# #     # 1. No reliable context → reject
# #     if recall < 25:
# #         return "not_found"

# #     # 2. Weak grounding → reject
# #     if grounding < 50:
# #         return "not_found"

# #     # 3. Short answers must be very strong
# #     if length <= 3:
# #         if recall >= 70 and grounding >= 70:
# #             return "accepted"
# #         return "not_found"

# #     # 4. Strong signal → accept
# #     if recall >= 50 and grounding >= 60:
# #         return "accepted"
   
# #     # 5. Default safe behavior
# #     else:
# #         return "not_found"
# # ============================================================
# # LOCAL HELPERS
# # ============================================================

# def clean_context_for_llm(chunks):
#     cleaned_chunks = []
#     for chunk in chunks:
#         lines = chunk.split("\n")
#         filtered_lines = []
#         for line in lines:
#             line_strip = line.strip()
#             is_question_like = (
#                 line_strip.endswith("?") or
#                 (len(line_strip.split()) < 15 and "?" in line_strip)
#             )
#             if is_question_like:
#                 continue
#             filtered_lines.append(line)
#         cleaned_chunks.append("\n".join(filtered_lines))
#     return cleaned_chunks


# def is_grounded(answer, retrieved_texts, threshold=0.6):
#     if not answer:
#         return False
#     answer_words = set(answer.lower().split())
#     if not answer_words:
#         return False
#     best_overlap = 0
#     for chunk in retrieved_texts:
#         chunk_words = set(chunk.lower().split())
#         overlap = len(answer_words & chunk_words) / len(answer_words)
#         best_overlap = max(best_overlap, overlap)
#     return best_overlap >= threshold


# # ============================================================
# # NUMERIC INTENT DETECTION
# # ============================================================

# def _extract_numbers(text: str) -> list:
#     return re.findall(r'\b\d+\b', text)


# def _detect_numeric_intent(question: str, chunks: list) -> str:
#     """
#     Detect NAVIGATIONAL or POSITIONAL from chunk structure.
#     Generic — no hardcoded section words.
#     """
#     numbers = _extract_numbers(question)
#     if not numbers:
#         return "NONE"

#     q_words = set(question.lower().split())

#     for chunk in chunks:
#         lines = [l.strip() for l in chunk.split("\n") if l.strip()]
#         if not lines:
#             continue
#         first_line = lines[0].lower()
#         line_words = set(first_line.split())
#         overlap    = len(q_words & line_words)
#         if any(re.search(rf'\b{re.escape(num)}\b', first_line) for num in numbers):
#             if overlap >= 1 and len(first_line.split()) <= 12:
#                 return "NAVIGATIONAL"

#     for chunk in chunks:
#         lines       = [l.strip() for l in chunk.split("\n") if l.strip()]
#         short_lines = [l for l in lines if len(l.split()) <= 12]
#         if len(short_lines) >= 3:
#             first = lines[0].lower() if lines else ""
#             if not any(re.search(rf'\b{re.escape(num)}\b', first) for num in numbers):
#                 return "POSITIONAL"

#     return "NONE"


# def _navigate_full_chunks(question: str, all_raw_chunks: list) -> str:
#     """
#     Scan full raw chunks to find section title for a numbered section.
#     Generic — works for Chapter N, Lecture N, Section N, etc.
#     """
#     numbers = _extract_numbers(question)
#     if not numbers:
#         return ""

#     q_words    = set(question.lower().split())
#     best_line  = ""
#     best_score = 0

#     for chunk in all_raw_chunks:
#         lines = [l.strip() for l in chunk.split("\n") if l.strip()]
#         if not lines:
#             continue
#         first_line  = lines[0]
#         first_lower = first_line.lower()

#         for num in numbers:
#             if not re.search(rf'\b{re.escape(num)}\b', first_lower):
#                 continue
#             line_words = set(first_lower.split())
#             overlap    = len(q_words & line_words)
#             if overlap >= 1 and len(first_line.split()) <= 14:
#                 score = overlap + (5 if len(first_line.split()) <= 8 else 0)
#                 if score > best_score:
#                     best_score = score
#                     best_line  = first_line

#     if not best_line:
#         return ""

#     title = best_line
#     title = re.sub(r'^\s*\d+\s*[-—–]+\s*[—–-]?\s*\d*\s*', '', title)
#     title = re.sub(r'\b\d+\b', '', title)
#     title = re.sub(r'[:\-–—]', ' ', title)
#     title = re.sub(r'\s{2,}', ' ', title)
#     return title.strip()


# def _positional_extract(question: str, retrieved_texts: list) -> str:
#     """Extract Nth item from a list. Generic."""
#     numbers = _extract_numbers(question)
#     if not numbers:
#         return ""
#     try:
#         idx = int(numbers[0]) - 1
#     except ValueError:
#         return ""
#     if idx < 0:
#         return ""

#     best_chunk, best_count = "", 0
#     for chunk in retrieved_texts:
#         lines       = [l.strip() for l in chunk.split("\n") if l.strip()]
#         short_lines = [l for l in lines if len(l.split()) <= 12]
#         if len(short_lines) > best_count:
#             best_count = len(short_lines)
#             best_chunk = chunk

#     if not best_chunk:
#         return ""

#     lines       = [l.strip() for l in best_chunk.split("\n") if l.strip()]
#     short_lines = [l for l in lines if len(l.split()) <= 12]
#     if idx < len(short_lines):
#         return re.sub(r'^\s*\d+[\.\)]\s*', '', short_lines[idx]).strip()
#     return ""


# # ============================================================
# # SIGNAL 3 — Structural consistency for NAVIGATIONAL answers
# # ============================================================

# def _check_identifier_grounded(answer: str, retrieved_texts: list) -> bool:
#     """
#     If answer contains a multi-word identifier (word+number or number+word),
#     verify it appears verbatim in retrieved chunks.
#     Generic — no hardcoded section words.
#     """
#     identifiers = re.findall(
#         r'\b[a-zA-Z]+\s+\d+\b|\b\d+\s+[a-zA-Z]+\b',
#         answer.lower()
#     )
#     if not identifiers:
#         return True

#     chunk_text = " ".join(retrieved_texts).lower()
#     for ident in identifiers:
#         if ident not in chunk_text:
#             print(f"[Signal3] ❌ Identifier '{ident}' not in retrieved chunks")
#             return False
#     return True


# # ============================================================
# # REFUSAL — semantic version (local)
# # ============================================================

# _REFUSAL_ANCHOR = "this information is not available in the provided context"


# def _is_refusal_semantic(text: str) -> bool:
#     if not text or len(text.strip()) < 2:
#         return True
#     sim = semantic_similarity(text, [_REFUSAL_ANCHOR])
#     print(f"[Refusal] sim={sim:.3f} | '{text[:50]}'")
#     return sim > 0.72


# # ============================================================
# # LANGGRAPH NODES
# # ============================================================

# def node_extract(state: DocState) -> DocState:
#     extract_start    = time.time()
#     text, page_count = extract_pdf_parallel(state["pdf_path"])
#     extract_time     = time.time() - extract_start
#     state["extracted_text"] = text
#     state["page_count"]     = page_count
#     state["char_count"]     = len(text)
#     state["metrics"]["extraction_time_sec"]  = round(extract_time, 2)
#     state["metrics"]["pages_processed"]      = page_count
#     state["metrics"]["characters_processed"] = len(text)
#     state["metrics"]["words_processed"]      = len(text.split())
#     return state


# def node_chunk(state: DocState) -> DocState:
#     text = state["extracted_text"]
#     state["query_type"] = "QA"

#     summary_chunks, rag_chunks = semantic_chunk(text)

#     # FIX 4 — normalize all chunks after chunking
#     # Converts unicode symbols so retrieval doesn't fail on symbol mismatches
#     rag_chunks     = [normalize_text(c) for c in rag_chunks]
#     summary_chunks = [normalize_text(c) for c in summary_chunks]

#     state["summary_chunks"] = summary_chunks
#     state["chunks"]         = rag_chunks

#     state["metrics"]["summary_chunks"] = len(summary_chunks)
#     state["metrics"]["chunks_created"] = len(rag_chunks)
#     # state["metrics"]["doc_type"]       = state.get("doc_type", "general")
#     state["metrics"]["query_type"]     = state["query_type"]
#     print(f"[Chunk] {len(summary_chunks)} summary chunks | {len(rag_chunks)} RAG chunks")
#     return state


# # def node_summarize(state: DocState) -> DocState:
# #     pdf_hash  = get_pdf_hash(state["pdf_path"])
# #     cache_key = f"{pdf_hash}_{state.get('doc_type', 'general')}"

# #     if cache_key in _summary_cache:
# #         cached = _summary_cache[cache_key]
# #         print("[Summary] ✅ Cache hit")
# #         emit_event(state.get("request_id", ""), "agent_action",
# #                    "⚡ Summary loaded from cache instantly!")
# #         state["answer"] = cached["summary"]
# #         state["metrics"].update(cached["metrics"])
# #         state["metrics"]["type"] = "summary"
# #         return state

# #     summary_start = time.time()
# #     raptor_summarize._request_id = state.get("request_id", "")
# #     summary, map_time, reduce_time = raptor_summarize(
# #         state["summary_chunks"], state.get("doc_type", "general")
# #     )
# #     summary_time = time.time() - summary_start

# #     state["answer"] = summary
# #     metrics_snapshot = {
# #         "summary_time_sec":     round(summary_time, 2),
# #         "summary_length_words": len(summary.split()),
# #         "parallel_workers":     min(MAX_WORKERS, len(state["summary_chunks"])),
# #         "map_time_sec":         round(map_time, 2),
# #         "reduce_time_sec":      round(reduce_time, 2),
# #         "llm_calls":            3,
# #     }
# #     state["metrics"].update(metrics_snapshot)
# #     state["metrics"]["type"] = "summary"
# #     _summary_cache[cache_key] = {"summary": summary, "metrics": metrics_snapshot}
# #     print(f"[Summary] Done ({len(summary.split())} words)")
# #     return state

# def node_summarize(state: DocState) -> DocState:
#     pdf_hash  = get_pdf_hash(state["pdf_path"])
#     cache_key = pdf_hash  # ❌ removed doc_type dependency

#     # ── Cache check ──────────────────────────────────────────
#     if cache_key in _summary_cache:
#         cached = _summary_cache[cache_key]
#         print("[Summary] ✅ Cache hit")
#         emit_event(
#             state.get("request_id", ""),
#             "agent_action",
#             "⚡ Summary loaded from cache instantly!"
#         )
#         state["answer"] = cached["summary"]
#         state["metrics"].update(cached["metrics"])
#         state["metrics"]["type"] = "summary"
#         return state

#     # ── Generate summary ─────────────────────────────────────
#     summary_start = time.time()
#     raptor_summarize._request_id = state.get("request_id", "")

#     # ❌ removed doc_type argument
#     summary, map_time, reduce_time = raptor_summarize(
#         state["summary_chunks"]
#     )

#     summary_time = time.time() - summary_start

#     # ── Store result ─────────────────────────────────────────
#     state["answer"] = summary

#     metrics_snapshot = {
#         "summary_time_sec":     round(summary_time, 2),
#         "summary_length_words": len(summary.split()),
#         "parallel_workers":     min(MAX_WORKERS, len(state["summary_chunks"])),
#         "map_time_sec":         round(map_time, 2),
#         "reduce_time_sec":      round(reduce_time, 2),
#         "llm_calls":            3,
#     }

#     state["metrics"].update(metrics_snapshot)
#     state["metrics"]["type"] = "summary"

#     # ── Cache result ─────────────────────────────────────────
#     _summary_cache[cache_key] = {
#         "summary": summary,
#         "metrics": metrics_snapshot
#     }

#     print(f"[Summary] Done ({len(summary.split())} words)")
#     return state

# def node_qa(state: DocState) -> DocState:
#     qa_start_t = time.time()
#     question   = state["question"]
#     request_id = state.get("request_id", "")
#     all_chunks = state["chunks"]

#     # FIX 4 — normalize question the same way chunks were normalized
#     question = normalize_text(question)

#     # Raw strings for navigational scanning — must NOT be cleaned
#     all_raw = [
#         d if isinstance(d, str) else d.page_content
#         for d in all_chunks
#     ]

#     pdf_hash    = get_pdf_hash(state["pdf_path"])
#     faiss_index = build_faiss_index(all_chunks, pdf_hash)

#     print(f"[QA] START | {len(all_chunks)} chunks")

#     recall_score      = 0.0
#     retrieval_score   = 0.0
#     context_precision = 0.0
#     grounding         = 0.0
#     llm_calls         = 0
#     model_used        = "llama"
#     confidence        = 0.0
#     decision_type     = "accepted"
#     retrieved         = []
#     retrieved_texts   = []
#     answer            = ""

#     # ── STEP 1: RETRIEVE ─────────────────────────────────────
#     retrieved = multi_query_retrieve(
#         question, faiss_index,
#         k=20,
#         all_chunks=all_chunks,
#         query_type="FACTUAL_QA"
#     )


   
#     # 🔥 STEP 2 — RERANK + SAFE FALLBACK

#     retrieved, reranker_top, _ = rerank_docs(
#         question, retrieved, top_k=8, apply_pruning=True
#     )

#     retrieved = protect_exact_matches(
#         question, retrieved, all_chunks, top_k=8
#     )

#     # 🔥 FIX 1 — fallback must replace retrieved (NOT just texts)
#     if not retrieved or len(retrieved) < 3:
#         print("[QA] ⚠️ Reranker too aggressive → fallback to initial chunks")
#         retrieved = all_chunks[:8]

#     # 🔥 FIX 2 — safe conversion
#     retrieved_texts = [
#         d.page_content if hasattr(d, "page_content") else str(d)
#         for d in retrieved
#     ]
#     if reranker_top < -5:
#         print("[Guard] ⚠️ Weak reranker score — continuing")

#     # ── STEP 3: METRICS ───────────────────────────────────────
#     retrieval_score   = compute_retrieval_score(question, retrieved)
#     context_precision = compute_context_precision(question, retrieved)
#     recall_score      = compute_recall_at_k(
#         question, retrieved, all_chunks, k=len(retrieved)
#     )

#     # ── FIX 6 — k expansion: low recall OR numeric question ───
#     is_numeric_question = len(_extract_numbers(question)) > 0
#     if recall_score < 25 or is_numeric_question:
#         print("[QA] ⚠️ Expanding retrieval (low recall or numeric question)")
#         expanded = multi_query_retrieve(
#             question, faiss_index, k=30,
#             all_chunks=all_chunks, query_type="FACTUAL_QA"
#         )
#         if len(expanded) > len(retrieved):
#             retrieved = expanded
#             retrieved_texts = [
#                 d.page_content if hasattr(d, "page_content") else str(d)
#                 for d in retrieved
#             ]

#     # ── FIX 1 — Numeric intent: retrieved first, fallback to all_raw ──
#     numeric_intent = _detect_numeric_intent(question, retrieved_texts)

#     if numeric_intent == "NONE" and is_numeric_question:
#         print("[Navigate] Retrying intent detection on full document...")
#         numeric_intent = _detect_numeric_intent(question, all_raw)
#         if numeric_intent != "NONE":
#             print(f"[Navigate] Intent found in full document: {numeric_intent}")

#     # ── NAVIGATIONAL ──────────────────────────────────────────
#     if numeric_intent == "NAVIGATIONAL":
#         print(f"[Routing] Numeric intent → NAVIGATIONAL")
#         title = _navigate_full_chunks(question, all_raw)
#         if title:
#             print(f"[Navigate] Extracted title: '{title}'")
#             grounding  = compute_answer_grounding(title, retrieved_texts, question)
#             confidence = round(grounding / 100, 3)
#             qa_time    = time.time() - qa_start_t
#             state["answer"]      = title
#             state["query_type"]  = "NAVIGATIONAL"
#             _write_metrics(state, "navigational", "navigational", grounding,
#                            confidence, retrieval_score, context_precision,
#                            recall_score, llm_calls, retrieved, qa_time)
#             return state
#         print(f"[Navigate] Extraction failed — falling through to LLM")

#     # ── POSITIONAL ────────────────────────────────────────────
#     elif numeric_intent == "POSITIONAL":
#         print(f"[Routing] Numeric intent → POSITIONAL")
#         pos_answer = _positional_extract(question, retrieved_texts)
#         if pos_answer:
#             print(f"[Positional] Extracted: '{pos_answer}'")
#             grounding  = compute_answer_grounding(pos_answer, retrieved_texts, question)
#             confidence = round(grounding / 100, 3)
#             qa_time    = time.time() - qa_start_t
#             state["answer"]      = pos_answer
#             state["query_type"]  = "POSITIONAL"
#             _write_metrics(state, "positional", "positional", grounding,
#                            confidence, retrieval_score, context_precision,
#                            recall_score, llm_calls, retrieved, qa_time)
#             return state
#         print(f"[Positional] Extraction failed — falling through to LLM")

#     # ── Normal routing ────────────────────────────────────────
#     query_type = classify_from_context(question, retrieved_texts)
#     state["query_type"] = query_type
#     print(f"[Routing] Context-based → {query_type}")

#     # ── STEP 4: ANSWER GENERATION ─────────────────────────────

#     if query_type == "FULL_SUMMARY":
#         retrieved_texts = clean_context_for_llm(retrieved_texts)
#         ranked  = reorder_by_question(question, retrieved_texts)
#         context = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:8])
#         structured_context = "\n".join(
#             f"[CONTEXT CHUNK]\n{c}" for c in context.split("\n\n---\n\n")
#         )
#         emit_event(request_id, "stream_start", "✍️ Generating answer...")
#         answer, _ = call_llama_streaming(
#             QA_PROMPT.format(context=structured_context[:2500], question=question),
#             request_id=request_id, temperature=0.0
#         )
#         answer     = clean_artifacts(answer).strip()
#         model_used = "llama_summary"
#         llm_calls  = 1

#     elif query_type == "MULTIPART_QA":
#         retrieved_texts = clean_context_for_llm(retrieved_texts)
#         ranked  = reorder_by_question(question, retrieved_texts)
#         # FIX 3 — 4000 chars for MULTIPART to capture all list items
#         context = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:8])
#         emit_event(request_id, "agent_start",
#                    f"🤖 MULTIPART | {len(retrieved)} chunks")
#         emit_event(request_id, "stream_start", "✍️ Generating answer...")
#         answer, _ = call_llama_streaming(
#             QA_PROMPT.format(context=context[:4000], question=question),
#             request_id=request_id, temperature=0.0
#         )
#         answer     = clean_artifacts(answer).strip()
#         model_used = "llama_multipart"
#         llm_calls  = 1

#     else:
        
#         # FACTUAL_QA / VERIFICATION_QA

#         if not retrieved_texts:
#             print("[QA] ❌ No context → NOT FOUND")
#             state["answer"] = "This information is not present in the document."
#             _write_metrics(state, "not_found", "no_context",
#                         0.0, 0.0, retrieval_score, context_precision,
#                         recall_score, llm_calls, retrieved,
#                         time.time() - qa_start_t)
#             return state

#         # 🔥 FIX 5 — SIMPLE QA ROUTING (ADD HERE)

#         context = "\n".join(retrieved_texts)

#         if query_type == "FACTUAL_QA" and len(question.split()) <= 10:
#             print("[QA] ⚡ Direct QA (no ReAct)")

#             react_ans, _ = call_llama_streaming(
#                 f"""
#             Answer the question using ONLY the given context.
#             Return ONLY the exact answer phrase from the context.
#             Do NOT explain. Do NOT say 'not found' unless absolutely missing.

#             Context:
#             {context[:2000]}

#             Question: {question}

#             Answer:
#             """,
#                 request_id=request_id,
#                 temperature=0.0
#             )
#             llm_calls += 1
#             model_used = "llama_direct"

#         else:
#             react_ans, model_used, _, _, _, _, react_calls, _ = react_agent(
#                 question, faiss_index, query_type, all_chunks, request_id
#             )
#             llm_calls += react_calls

#         # react_ans, model_used, _, _, _, _, react_calls, retrieved_texts = react_agent(
#         #     question, faiss_index, query_type, all_chunks, request_id
#         # )
 
#         react_ans = clean_artifacts(react_ans).strip().strip('"').strip("'")
#         react_ans = clean_reasoning_answer(react_ans, question)
#         print(f"[QA] Answer: '{react_ans[:60]}'")

#         # FIX 2 — Short answer bypass: skip for VERIFICATION_QA
#         words         = react_ans.split()
#         content_words = [w for w in words if len(w) > 2 and not w.isdigit()]

#         # if (
#         #     query_type != "VERIFICATION_QA"
#         #     and len(words) <= 3
#         #     and len(content_words) >= 1
#         #     and not _is_refusal_answer(react_ans)
#         # ):
#         #     print("[QA] ⚡ Short answer accepted")
#         #     grounding  = compute_answer_grounding(react_ans, retrieved_texts, question)
#         #     confidence = grounding / 100
#         #     qa_time    = time.time() - qa_start_t
#         #     state["answer"] = react_ans
#         #     _write_metrics(state, model_used, "short_answer",
#         #                    grounding, confidence, retrieval_score,
#         #                    context_precision, recall_score, llm_calls,
#         #                    retrieved, qa_time)
#         #     return state

#         # Refusal
#         if _is_refusal_answer(react_ans):
#             print("[QA] ⚠️ Refusal detected")
#             qa_time = time.time() - qa_start_t
#             state["answer"] = "This information is not present in the document."
#             _write_metrics(state, "not_found", "not_found",
#                            75.0, 0.75, retrieval_score, context_precision,
#                            recall_score, llm_calls, retrieved, qa_time)
#             return state

#         grounding_score = compute_answer_grounding(react_ans, retrieved_texts, question)
#         # 🔥 STRICT VERIFICATION CHECK
#         # if query_type == "VERIFICATION_QA":
#         #     if grounding_score < 80:
#         #         print("[QA] ❌ Verification failed → NOT FOUND")
#         #         answer = "This information is not present in the document."
#         #         decision_type = "verification_failed"
#         #     else:
#         #         answer = react_ans
#         #         decision_type = "accepted"
#         if query_type == "VERIFICATION_QA":
#             if grounding_score < 80:
#                 print("[QA] ❌ Verification failed → NOT FOUND")
#                 answer = "This information is not present in the document."
#                 decision_type = "verification_failed"
#                 state["answer"] = answer
#                 return state   # ❌ stop pipeline

#             else:
#                 answer = react_ans
#                 decision_type = "accepted"
#                 state["answer"] = answer
#                 return state   # ✅ ALSO stop pipeline

#         # 🔥 TRUST GATE (FINAL CLEAN VERSION)

#         recall = recall_score
#         grounding = grounding_score
#         length = len(react_ans.split())

#         # 1. No reliable context → reject
#         if recall < 25:
#             print("[QA] ❌ Low recall → NOT FOUND")
#             answer = "This information is not present in the document."
#             decision_type = "low_recall"

#         # 2. Very weak grounding → reject
#         elif grounding < 40:
#             print("[QA] ❌ Very low grounding → NOT FOUND")
#             answer = "This information is not present in the document."
#             decision_type = "low_grounding"

#         # # 3. Short answers must be VERY strong
#         # elif length <= 3:
#         #     if recall >= 70 and grounding >= 70:
#         #         answer = react_ans
#         #         decision_type = "accepted"
        
#         #     else:
#         #         print("[QA] ❌ Weak short answer → NOT FOUND")
#         #         answer = "This information is not present in the document."
#         #         decision_type = "weak_short"

#         # # 4. Strong signals → accept
#         # elif recall >= 50 and grounding >= 60:
#         #     answer = react_ans
#         #     decision_type = "accepted"

#         # # 5. Otherwise reject safely
#         # else:
#         #     print("[QA] ❌ Uncertain → NOT FOUND")
#         #     answer = "This information is not present in the document."
#         #     decision_type = "uncertain"
#         elif length <= 3:
#             if grounding >= 60:
#                 answer = react_ans
#                 decision_type = "accepted"
#             else:
#                 print("[QA] ❌ Weak short answer → NOT FOUND")
#                 answer = "This information is not present in the document."
#                 decision_type = "weak_short"

#         # 4. Strong signals → accept
#         elif recall >= 50 and grounding >= 60:
#             answer = react_ans
#             decision_type = "accepted"

#         # 5. Otherwise reject safely
#         else:
#             print("[QA] ❌ Uncertain → NOT FOUND")
#             answer = "This information is not present in the document."
#             decision_type = "uncertain"
#     # ── STEP 5: METRICS ───────────────────────────────────────
#     grounding  = compute_answer_grounding(answer, retrieved_texts, question)
#     confidence = round(grounding / 100, 3)
#     qa_time    = time.time() - qa_start_t

#     print(f"[QA] Done in {qa_time:.1f}s | model={model_used} | "
#           f"grounding={grounding:.1f}% | recall={recall_score:.1f}% | "
#           f"confidence={confidence:.3f} | decision={decision_type}")

#     if not answer.strip():
#         answer = "Could not find a relevant answer in the PDF."
#     else:
#         answer = normalize_answer(answer)

#     state["answer"] = answer
#     _write_metrics(state, model_used, decision_type, grounding,
#                    confidence, retrieval_score, context_precision,
#                    recall_score, llm_calls, retrieved, qa_time)
#     return state
# def node_validate(state: DocState) -> DocState:
#     answer = state["answer"]
#     retry  = state.get("retry_count", 0)

#     # ── Retry for empty/very weak answers ────────────────────
#     if len(answer.strip()) < 3 and retry < 2:
#         state["retry_count"] = retry + 1
#         state["answer"]      = ""
#         return state

#     total_time    = time.time() - state["start_time"]
#     output_words  = len(answer.split())
#     output_tokens = output_words * 1.3

#     extract_time  = state["metrics"].get("extraction_time_sec", 0)
#     llm_time      = max(total_time - extract_time, 1)
#     tps           = round(output_tokens / llm_time, 2) if llm_time > 0 else 0

#     m = state["metrics"]

#     # ── Core metrics ─────────────────────────────────────────
#     m["response_time_sec"]    = round(total_time, 2)
#     m["extraction_time_sec"]  = m.get("extraction_time_sec", 0)
#     m["pages_processed"]      = state.get("page_count", 0)
#     m["characters_processed"] = state.get("char_count", 0)
#     m["words_processed"]      = len(state.get("extracted_text", "").split())

#     # ── Type-specific metrics ────────────────────────────────
#     if m.get("type") == "summary":
#         m["summary_time_sec"]     = m.get("summary_time_sec", 0)
#         m["summary_length_words"] = len(answer.split())

#     if m.get("type") == "qa":
#         m["qa_time_sec"]      = m.get("qa_time_sec", 0)
#         m["confidence_score"] = m.get("confidence_score", 0)

#     # ── Performance ──────────────────────────────────────────
#     m["ttft_sec"]        = round(total_time, 2)
#     m["e2e_latency_sec"] = round(total_time, 2)
#     m["tps"]             = tps

#     # ❌ REMOVED doc_type (no longer used anywhere)
#     # m["doc_type"] = state.get("doc_type", "general")

#     # ── Context info ─────────────────────────────────────────
#     m["query_type"]     = state.get("query_type", "")
#     m["chunks_created"] = m.get("chunks_created", 0)
#     m["retry_count"]    = retry

#     # ── Model + retrieval metrics ────────────────────────────
#     m["model_used"]        = m.get("model_used", "llama_react")
#     m["llm_calls"]         = m.get("llm_calls", 0)
#     m["retrieval_score"]   = m.get("retrieval_score", 0)
#     m["context_precision"] = m.get("context_precision", 0)
#     m["answer_grounding"]  = m.get("answer_grounding", 0)
#     m["recall_at_k"]       = m.get("recall_at_k", 0)

#     # ── Summary-specific ─────────────────────────────────────
#     if m.get("type") == "summary":
#         m["parallel_workers"] = m.get("parallel_workers", 0)
#         m["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
#         m["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)

#     # ── QA-specific ──────────────────────────────────────────
#     if m.get("type") == "qa":
#         m["chunks_retrieved"] = m.get("chunks_retrieved", 0)
#         m["decision_type"]    = m.get("decision_type", "accepted")
#         m["confidence_raw"]   = m.get("confidence_raw", 0.0)

#     state["metrics"] = m
#     return state

# # def node_validate(state: DocState) -> DocState:
# #     answer = state["answer"]
# #     retry  = state.get("retry_count", 0)

# #     if len(answer.strip()) < 3 and retry < 2:
# #         state["retry_count"] = retry + 1
# #         state["answer"]      = ""
# #         return state

# #     total_time    = time.time() - state["start_time"]
# #     output_words  = len(answer.split())
# #     output_tokens = output_words * 1.3
# #     extract_time  = state["metrics"].get("extraction_time_sec", 0)
# #     llm_time      = max(total_time - extract_time, 1)
# #     tps           = round(output_tokens / llm_time, 2) if llm_time > 0 else 0
# #     m             = state["metrics"]

# #     m["response_time_sec"]    = round(total_time, 2)
# #     m["extraction_time_sec"]  = m.get("extraction_time_sec", 0)
# #     m["pages_processed"]      = state.get("page_count", 0)
# #     m["characters_processed"] = state.get("char_count", 0)
# #     m["words_processed"]      = len(state.get("extracted_text", "").split())

# #     if m.get("type") == "summary":
# #         m["summary_time_sec"]     = m.get("summary_time_sec", 0)
# #         m["summary_length_words"] = len(answer.split())
# #     if m.get("type") == "qa":
# #         m["qa_time_sec"]      = m.get("qa_time_sec", 0)
# #         m["confidence_score"] = m.get("confidence_score", 0)

# #     m["ttft_sec"]          = round(total_time, 2)
# #     m["e2e_latency_sec"]   = round(total_time, 2)
# #     m["tps"]               = tps
# #     m["doc_type"]          = state.get("doc_type", "general")
# #     m["query_type"]        = state.get("query_type", "")
# #     m["chunks_created"]    = m.get("chunks_created", 0)
# #     m["retry_count"]       = retry
# #     m["model_used"]        = m.get("model_used", "llama_react")
# #     m["llm_calls"]         = m.get("llm_calls", 0)
# #     m["retrieval_score"]   = m.get("retrieval_score", 0)
# #     m["context_precision"] = m.get("context_precision", 0)
# #     m["answer_grounding"]  = m.get("answer_grounding", 0)
# #     m["recall_at_k"]       = m.get("recall_at_k", 0)

# #     if m.get("type") == "summary":
# #         m["parallel_workers"] = m.get("parallel_workers", 0)
# #         m["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
# #         m["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)
# #     if m.get("type") == "qa":
# #         m["chunks_retrieved"] = m.get("chunks_retrieved", 0)
# #         m["decision_type"]    = m.get("decision_type", "accepted")
# #         m["confidence_raw"]   = m.get("confidence_raw", 0.0)

# #     state["metrics"] = m
# #     return state


# def _write_metrics(state, model_used, decision_type, grounding,
#                    confidence, retrieval_score, context_precision,
#                    recall_score, llm_calls, retrieved, qa_time):
#     m = state.setdefault("metrics", {})
#     m["qa_time_sec"]       = round(qa_time, 2)
#     m["confidence_score"]  = round(confidence * 100, 2)
#     m["retrieval_score"]   = retrieval_score
#     m["context_precision"] = context_precision
#     m["answer_grounding"]  = grounding
#     m["recall_at_k"]       = recall_score
#     m["llm_calls"]         = llm_calls
#     m["model_used"]        = model_used
#     m["chunks_retrieved"]  = len(retrieved)
#     m["type"]              = "qa"
#     m["decision_type"]     = decision_type
#     m["confidence_raw"]    = round(confidence, 4)










# 80,80,50
# import time
# import re
# import spacy
# _nlp = spacy.load("en_core_web_lg")

# from docmind_rag.config.settings import (
#     MAX_WORKERS, FACTUAL_TOP_K, MULTIPART_TOP_K
# )
# from docmind_rag.utils.text import extract_named_entities, expand_answer
# from docmind_rag.core.state import DocState
# from docmind_rag.core.prompts import QA_PROMPT
# from docmind_rag.models.llm import call_llama_streaming
# from docmind_rag.models.embeddings import build_faiss_index
# from docmind_rag.models.reranker import rerank_docs, protect_exact_matches
# from docmind_rag.services.extraction import extract_pdf_parallel, _extraction_cache
# from docmind_rag.services.chunking import semantic_chunk, raptor_summarize, _summary_cache
# from docmind_rag.services.retrieval import multi_query_retrieve
# from docmind_rag.services.agent import react_agent, roberta_qa, clean_reasoning_answer
# from docmind_rag.events.events import emit_event
# from docmind_rag.utils.helpers import get_pdf_hash, clean_artifacts, clean_chunk_text, normalize_answer
# from docmind_rag.utils.text import (
#     classify_from_context,
#     reorder_by_question,
#     normalize_text,
# )
# from docmind_rag.utils.metrics import (
#     compute_answer_grounding,
#     compute_retrieval_score,
#     compute_context_precision,
#     compute_recall_at_k,
#     semantic_similarity,
#     _is_refusal_answer,
# )

# # def trust_gate(answer: str, metrics: dict) -> str:
# #     recall = metrics.get("recall_at_k", 0)
# #     grounding = metrics.get("answer_grounding", 0)
# #     length = len(answer.split())

# #     # 1. No reliable context → reject
# #     if recall < 25:
# #         return "not_found"

# #     # 2. Weak grounding → reject
# #     if grounding < 50:
# #         return "not_found"

# #     # 3. Short answers must be very strong
# #     if length <= 3:
# #         if recall >= 70 and grounding >= 70:
# #             return "accepted"
# #         return "not_found"

# #     # 4. Strong signal → accept
# #     if recall >= 50 and grounding >= 60:
# #         return "accepted"
   
# #     # 5. Default safe behavior
# #     else:
# #         return "not_found"
# # ============================================================
# # LOCAL HELPERS
# # ============================================================

# def clean_context_for_llm(chunks):
#     cleaned_chunks = []
#     for chunk in chunks:
#         lines = chunk.split("\n")
#         filtered_lines = []
#         for line in lines:
#             line_strip = line.strip()
#             is_question_like = (
#                 line_strip.endswith("?") or
#                 (len(line_strip.split()) < 15 and "?" in line_strip)
#             )
#             if is_question_like:
#                 continue
#             filtered_lines.append(line)
#         cleaned_chunks.append("\n".join(filtered_lines))
#     return cleaned_chunks


# def is_grounded(answer, retrieved_texts, threshold=0.6):
#     if not answer:
#         return False
#     answer_words = set(answer.lower().split())
#     if not answer_words:
#         return False
#     best_overlap = 0
#     for chunk in retrieved_texts:
#         chunk_words = set(chunk.lower().split())
#         overlap = len(answer_words & chunk_words) / len(answer_words)
#         best_overlap = max(best_overlap, overlap)
#     return best_overlap >= threshold


# # ============================================================
# # NUMERIC INTENT DETECTION
# # ============================================================

# def _extract_numbers(text: str) -> list:
#     return re.findall(r'\b\d+\b', text)


# def _detect_numeric_intent(question: str, chunks: list) -> str:
#     """
#     Detect NAVIGATIONAL or POSITIONAL from chunk structure.
#     Generic — no hardcoded section words.
#     """
#     numbers = _extract_numbers(question)
#     if not numbers:
#         return "NONE"

#     q_words = set(question.lower().split())

#     for chunk in chunks:
#         lines = [l.strip() for l in chunk.split("\n") if l.strip()]
#         if not lines:
#             continue
#         first_line = lines[0].lower()
#         line_words = set(first_line.split())
#         overlap    = len(q_words & line_words)
#         if any(re.search(rf'\b{re.escape(num)}\b', first_line) for num in numbers):
#             if overlap >= 1 and len(first_line.split()) <= 12:
#                 return "NAVIGATIONAL"

#     for chunk in chunks:
#         lines       = [l.strip() for l in chunk.split("\n") if l.strip()]
#         short_lines = [l for l in lines if len(l.split()) <= 12]
#         if len(short_lines) >= 3:
#             first = lines[0].lower() if lines else ""
#             if not any(re.search(rf'\b{re.escape(num)}\b', first) for num in numbers):
#                 return "POSITIONAL"

#     return "NONE"


# def _navigate_full_chunks(question: str, all_raw_chunks: list) -> str:
#     """
#     Scan full raw chunks to find section title for a numbered section.
#     Generic — works for Chapter N, Lecture N, Section N, etc.
#     """
#     numbers = _extract_numbers(question)
#     if not numbers:
#         return ""

#     q_words    = set(question.lower().split())
#     best_line  = ""
#     best_score = 0

#     for chunk in all_raw_chunks:
#         lines = [l.strip() for l in chunk.split("\n") if l.strip()]
#         if not lines:
#             continue
#         first_line  = lines[0]
#         first_lower = first_line.lower()

#         for num in numbers:
#             if not re.search(rf'\b{re.escape(num)}\b', first_lower):
#                 continue
#             line_words = set(first_lower.split())
#             overlap    = len(q_words & line_words)
#             if overlap >= 1 and len(first_line.split()) <= 14:
#                 score = overlap + (5 if len(first_line.split()) <= 8 else 0)
#                 if score > best_score:
#                     best_score = score
#                     best_line  = first_line

#     if not best_line:
#         return ""

#     title = best_line
#     title = re.sub(r'^\s*\d+\s*[-—–]+\s*[—–-]?\s*\d*\s*', '', title)
#     title = re.sub(r'\b\d+\b', '', title)
#     title = re.sub(r'[:\-–—]', ' ', title)
#     title = re.sub(r'\s{2,}', ' ', title)
#     return title.strip()


# def _positional_extract(question: str, retrieved_texts: list) -> str:
#     """Extract Nth item from a list. Generic."""
#     numbers = _extract_numbers(question)
#     if not numbers:
#         return ""
#     try:
#         idx = int(numbers[0]) - 1
#     except ValueError:
#         return ""
#     if idx < 0:
#         return ""

#     best_chunk, best_count = "", 0
#     for chunk in retrieved_texts:
#         lines       = [l.strip() for l in chunk.split("\n") if l.strip()]
#         short_lines = [l for l in lines if len(l.split()) <= 12]
#         if len(short_lines) > best_count:
#             best_count = len(short_lines)
#             best_chunk = chunk

#     if not best_chunk:
#         return ""

#     lines       = [l.strip() for l in best_chunk.split("\n") if l.strip()]
#     short_lines = [l for l in lines if len(l.split()) <= 12]
#     if idx < len(short_lines):
#         return re.sub(r'^\s*\d+[\.\)]\s*', '', short_lines[idx]).strip()
#     return ""


# # ============================================================
# # SIGNAL 3 — Structural consistency for NAVIGATIONAL answers
# # ============================================================

# def _check_identifier_grounded(answer: str, retrieved_texts: list) -> bool:
#     """
#     If answer contains a multi-word identifier (word+number or number+word),
#     verify it appears verbatim in retrieved chunks.
#     Generic — no hardcoded section words.
#     """
#     identifiers = re.findall(
#         r'\b[a-zA-Z]+\s+\d+\b|\b\d+\s+[a-zA-Z]+\b',
#         answer.lower()
#     )
#     if not identifiers:
#         return True

#     chunk_text = " ".join(retrieved_texts).lower()
#     for ident in identifiers:
#         if ident not in chunk_text:
#             print(f"[Signal3] ❌ Identifier '{ident}' not in retrieved chunks")
#             return False
#     return True


# # ============================================================
# # REFUSAL — semantic version (local)
# # ============================================================

# _REFUSAL_ANCHOR = "this information is not available in the provided context"


# def _is_refusal_semantic(text: str) -> bool:
#     if not text or len(text.strip()) < 2:
#         return True
#     sim = semantic_similarity(text, [_REFUSAL_ANCHOR])
#     print(f"[Refusal] sim={sim:.3f} | '{text[:50]}'")
#     return sim > 0.72


# # ============================================================
# # LANGGRAPH NODES
# # ============================================================

# def node_extract(state: DocState) -> DocState:
#     extract_start    = time.time()
#     text, page_count = extract_pdf_parallel(state["pdf_path"])
#     extract_time     = time.time() - extract_start
#     state["extracted_text"] = text
#     state["page_count"]     = page_count
#     state["char_count"]     = len(text)
#     state["metrics"]["extraction_time_sec"]  = round(extract_time, 2)
#     state["metrics"]["pages_processed"]      = page_count
#     state["metrics"]["characters_processed"] = len(text)
#     state["metrics"]["words_processed"]      = len(text.split())
#     return state


# def node_chunk(state: DocState) -> DocState:
#     text = state["extracted_text"]
#     state["query_type"] = "QA"

#     summary_chunks, rag_chunks = semantic_chunk(text)

#     # FIX 4 — normalize all chunks after chunking
#     # Converts unicode symbols so retrieval doesn't fail on symbol mismatches
#     rag_chunks     = [normalize_text(c) for c in rag_chunks]
#     summary_chunks = [normalize_text(c) for c in summary_chunks]

#     state["summary_chunks"] = summary_chunks
#     state["chunks"]         = rag_chunks

#     state["metrics"]["summary_chunks"] = len(summary_chunks)
#     state["metrics"]["chunks_created"] = len(rag_chunks)
#     # state["metrics"]["doc_type"]       = state.get("doc_type", "general")
#     state["metrics"]["query_type"]     = state["query_type"]
#     print(f"[Chunk] {len(summary_chunks)} summary chunks | {len(rag_chunks)} RAG chunks")
#     return state


# # def node_summarize(state: DocState) -> DocState:
# #     pdf_hash  = get_pdf_hash(state["pdf_path"])
# #     cache_key = f"{pdf_hash}_{state.get('doc_type', 'general')}"

# #     if cache_key in _summary_cache:
# #         cached = _summary_cache[cache_key]
# #         print("[Summary] ✅ Cache hit")
# #         emit_event(state.get("request_id", ""), "agent_action",
# #                    "⚡ Summary loaded from cache instantly!")
# #         state["answer"] = cached["summary"]
# #         state["metrics"].update(cached["metrics"])
# #         state["metrics"]["type"] = "summary"
# #         return state

# #     summary_start = time.time()
# #     raptor_summarize._request_id = state.get("request_id", "")
# #     summary, map_time, reduce_time = raptor_summarize(
# #         state["summary_chunks"], state.get("doc_type", "general")
# #     )
# #     summary_time = time.time() - summary_start

# #     state["answer"] = summary
# #     metrics_snapshot = {
# #         "summary_time_sec":     round(summary_time, 2),
# #         "summary_length_words": len(summary.split()),
# #         "parallel_workers":     min(MAX_WORKERS, len(state["summary_chunks"])),
# #         "map_time_sec":         round(map_time, 2),
# #         "reduce_time_sec":      round(reduce_time, 2),
# #         "llm_calls":            3,
# #     }
# #     state["metrics"].update(metrics_snapshot)
# #     state["metrics"]["type"] = "summary"
# #     _summary_cache[cache_key] = {"summary": summary, "metrics": metrics_snapshot}
# #     print(f"[Summary] Done ({len(summary.split())} words)")
# #     return state

# def node_summarize(state: DocState) -> DocState:
#     pdf_hash  = get_pdf_hash(state["pdf_path"])
#     cache_key = pdf_hash  # ❌ removed doc_type dependency

#     # ── Cache check ──────────────────────────────────────────
#     if cache_key in _summary_cache:
#         cached = _summary_cache[cache_key]
#         print("[Summary] ✅ Cache hit")
#         emit_event(
#             state.get("request_id", ""),
#             "agent_action",
#             "⚡ Summary loaded from cache instantly!"
#         )
#         state["answer"] = cached["summary"]
#         state["metrics"].update(cached["metrics"])
#         state["metrics"]["type"] = "summary"
#         return state

#     # ── Generate summary ─────────────────────────────────────
#     summary_start = time.time()
#     raptor_summarize._request_id = state.get("request_id", "")

#     # ❌ removed doc_type argument
#     summary, map_time, reduce_time = raptor_summarize(
#         state["summary_chunks"]
#     )

#     summary_time = time.time() - summary_start

#     # ── Store result ─────────────────────────────────────────
#     state["answer"] = summary

#     metrics_snapshot = {
#         "summary_time_sec":     round(summary_time, 2),
#         "summary_length_words": len(summary.split()),
#         "parallel_workers":     min(MAX_WORKERS, len(state["summary_chunks"])),
#         "map_time_sec":         round(map_time, 2),
#         "reduce_time_sec":      round(reduce_time, 2),
#         "llm_calls":            3,
#     }

#     state["metrics"].update(metrics_snapshot)
#     state["metrics"]["type"] = "summary"

#     # ── Cache result ─────────────────────────────────────────
#     _summary_cache[cache_key] = {
#         "summary": summary,
#         "metrics": metrics_snapshot
#     }

#     print(f"[Summary] Done ({len(summary.split())} words)")
#     return state

# def node_qa(state: DocState) -> DocState:
#     qa_start_t = time.time()
#     question   = state["question"]
#     request_id = state.get("request_id", "")
#     all_chunks = state["chunks"]

#     # FIX 4 — normalize question the same way chunks were normalized
#     question = normalize_text(question)

#     # Raw strings for navigational scanning — must NOT be cleaned
#     all_raw = [
#         d if isinstance(d, str) else d.page_content
#         for d in all_chunks
#     ]

#     pdf_hash    = get_pdf_hash(state["pdf_path"])
#     faiss_index = build_faiss_index(all_chunks, pdf_hash)

#     print(f"[QA] START | {len(all_chunks)} chunks")

#     recall_score      = 0.0
#     retrieval_score   = 0.0
#     context_precision = 0.0
#     grounding         = 0.0
#     llm_calls         = 0
#     model_used        = "llama"
#     confidence        = 0.0
#     decision_type     = "accepted"
#     retrieved         = []
#     retrieved_texts   = []
#     answer            = ""

#     # ── STEP 1: RETRIEVE ─────────────────────────────────────
#     retrieved = multi_query_retrieve(
#         question, faiss_index,
#         k=20,
#         all_chunks=all_chunks,
#         query_type="FACTUAL_QA"
#     )


   
#     # 🔥 STEP 2 — RERANK + SAFE FALLBACK

#     retrieved, reranker_top, _ = rerank_docs(
#         question, retrieved, top_k=8, apply_pruning=True
#     )

#     retrieved = protect_exact_matches(
#         question, retrieved, all_chunks, top_k=8
#     )

#     # 🔥 FIX 1 — fallback must replace retrieved (NOT just texts)
#     if not retrieved or len(retrieved) < 3:
#         print("[QA] ⚠️ Reranker too aggressive → fallback to initial chunks")
#         retrieved = all_chunks[:8]

#     # 🔥 FIX 2 — safe conversion
#     retrieved_texts = [
#         d.page_content if hasattr(d, "page_content") else str(d)
#         for d in retrieved
#     ]
#     if reranker_top < -5:
#         print("[Guard] ⚠️ Weak reranker score — continuing")

#     # ── STEP 3: METRICS ───────────────────────────────────────
#     retrieval_score   = compute_retrieval_score(question, retrieved)
#     context_precision = compute_context_precision(question, retrieved)
#     recall_score      = compute_recall_at_k(
#         question, retrieved, all_chunks, k=len(retrieved)
#     )

#     # ── FIX 6 — k expansion: low recall OR numeric question ───
#     is_numeric_question = len(_extract_numbers(question)) > 0
#     if recall_score < 25 or is_numeric_question:
#         print("[QA] ⚠️ Expanding retrieval (low recall or numeric question)")
#         expanded = multi_query_retrieve(
#             question, faiss_index, k=30,
#             all_chunks=all_chunks, query_type="FACTUAL_QA"
#         )
#         if len(expanded) > len(retrieved):
#             retrieved = expanded
#             retrieved_texts = [
#                 d.page_content if hasattr(d, "page_content") else str(d)
#                 for d in retrieved
#             ]

#     # ── FIX 1 — Numeric intent: retrieved first, fallback to all_raw ──
#     numeric_intent = _detect_numeric_intent(question, retrieved_texts)

#     if numeric_intent == "NONE" and is_numeric_question:
#         print("[Navigate] Retrying intent detection on full document...")
#         numeric_intent = _detect_numeric_intent(question, all_raw)
#         if numeric_intent != "NONE":
#             print(f"[Navigate] Intent found in full document: {numeric_intent}")

#     # ── NAVIGATIONAL ──────────────────────────────────────────
#     if numeric_intent == "NAVIGATIONAL":
#         print(f"[Routing] Numeric intent → NAVIGATIONAL")
#         title = _navigate_full_chunks(question, all_raw)
#         if title:
#             print(f"[Navigate] Extracted title: '{title}'")
#             grounding  = compute_answer_grounding(title, retrieved_texts, question)
#             confidence = round(grounding / 100, 3)
#             qa_time    = time.time() - qa_start_t
#             state["answer"]      = title
#             state["query_type"]  = "NAVIGATIONAL"
#             _write_metrics(state, "navigational", "navigational", grounding,
#                            confidence, retrieval_score, context_precision,
#                            recall_score, llm_calls, retrieved, qa_time)
#             return state
#         print(f"[Navigate] Extraction failed — falling through to LLM")

#     # ── POSITIONAL ────────────────────────────────────────────
#     elif numeric_intent == "POSITIONAL":
#         print(f"[Routing] Numeric intent → POSITIONAL")
#         pos_answer = _positional_extract(question, retrieved_texts)
#         if pos_answer:
#             print(f"[Positional] Extracted: '{pos_answer}'")
#             grounding  = compute_answer_grounding(pos_answer, retrieved_texts, question)
#             confidence = round(grounding / 100, 3)
#             qa_time    = time.time() - qa_start_t
#             state["answer"]      = pos_answer
#             state["query_type"]  = "POSITIONAL"
#             _write_metrics(state, "positional", "positional", grounding,
#                            confidence, retrieval_score, context_precision,
#                            recall_score, llm_calls, retrieved, qa_time)
#             return state
#         print(f"[Positional] Extraction failed — falling through to LLM")

#     # ── Normal routing ────────────────────────────────────────
#     query_type = classify_from_context(question, retrieved_texts)
#     state["query_type"] = query_type
#     print(f"[Routing] Context-based → {query_type}")

#     # ── STEP 4: ANSWER GENERATION ─────────────────────────────

#     if query_type == "FULL_SUMMARY":
#         retrieved_texts = clean_context_for_llm(retrieved_texts)
#         ranked  = reorder_by_question(question, retrieved_texts)
#         context = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:8])
#         structured_context = "\n".join(
#             f"[CONTEXT CHUNK]\n{c}" for c in context.split("\n\n---\n\n")
#         )
#         emit_event(request_id, "stream_start", "✍️ Generating answer...")
#         answer, _ = call_llama_streaming(
#             QA_PROMPT.format(context=structured_context[:2500], question=question),
#             request_id=request_id, temperature=0.0
#         )
#         answer     = clean_artifacts(answer).strip()
#         model_used = "llama_summary"
#         llm_calls  = 1

#     elif query_type == "MULTIPART_QA":
#         retrieved_texts = clean_context_for_llm(retrieved_texts)
#         ranked  = reorder_by_question(question, retrieved_texts)
#         # FIX 3 — 4000 chars for MULTIPART to capture all list items
#         context = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:8])
#         emit_event(request_id, "agent_start",
#                    f"🤖 MULTIPART | {len(retrieved)} chunks")
#         emit_event(request_id, "stream_start", "✍️ Generating answer...")
#         answer, _ = call_llama_streaming(
#             QA_PROMPT.format(context=context[:4000], question=question),
#             request_id=request_id, temperature=0.0
#         )
#         answer     = clean_artifacts(answer).strip()
#         model_used = "llama_multipart"
#         llm_calls  = 1

#     else:
        
#         # FACTUAL_QA / VERIFICATION_QA

#         if not retrieved_texts:
#             print("[QA] ❌ No context → NOT FOUND")
#             state["answer"] = "This information is not present in the document."
#             _write_metrics(state, "not_found", "no_context",
#                         0.0, 0.0, retrieval_score, context_precision,
#                         recall_score, llm_calls, retrieved,
#                         time.time() - qa_start_t)
#             return state

#         # 🔥 FIX 5 — SIMPLE QA ROUTING (ADD HERE)

#         context = "\n".join(retrieved_texts)

#         if query_type == "FACTUAL_QA" and len(question.split()) <= 10:
#             print("[QA] ⚡ Direct QA (no ReAct)")

#             react_ans, _ = call_llama_streaming(
#                 f"""
#             Answer the question using ONLY the given context.
#             Return ONLY the exact answer phrase from the context.
#             Do NOT explain. Do NOT say 'not found' unless absolutely missing.

#             Context:
#             {context[:2000]}

#             Question: {question}

#             Answer:
#             """,
#                 request_id=request_id,
#                 temperature=0.0
#             )
#             llm_calls += 1
#             model_used = "llama_direct"

#         else:
#             react_ans, model_used, _, _, _, _, react_calls, _ = react_agent(
#                 question, faiss_index, query_type, all_chunks, request_id
#             )
#             llm_calls += react_calls

#         # react_ans, model_used, _, _, _, _, react_calls, retrieved_texts = react_agent(
#         #     question, faiss_index, query_type, all_chunks, request_id
#         # )
 
#         react_ans = clean_artifacts(react_ans).strip().strip('"').strip("'")
#         react_ans = clean_reasoning_answer(react_ans, question)
#         print(f"[QA] Answer: '{react_ans[:60]}'")

#         # FIX 2 — Short answer bypass: skip for VERIFICATION_QA
#         words         = react_ans.split()
#         content_words = [w for w in words if len(w) > 2 and not w.isdigit()]

#         # if (
#         #     query_type != "VERIFICATION_QA"
#         #     and len(words) <= 3
#         #     and len(content_words) >= 1
#         #     and not _is_refusal_answer(react_ans)
#         # ):
#         #     print("[QA] ⚡ Short answer accepted")
#         #     grounding  = compute_answer_grounding(react_ans, retrieved_texts, question)
#         #     confidence = grounding / 100
#         #     qa_time    = time.time() - qa_start_t
#         #     state["answer"] = react_ans
#         #     _write_metrics(state, model_used, "short_answer",
#         #                    grounding, confidence, retrieval_score,
#         #                    context_precision, recall_score, llm_calls,
#         #                    retrieved, qa_time)
#         #     return state

#         # Refusal
#         if _is_refusal_answer(react_ans):
#             print("[QA] ⚠️ Refusal detected")
#             qa_time = time.time() - qa_start_t
#             state["answer"] = "This information is not present in the document."
#             _write_metrics(state, "not_found", "not_found",
#                            75.0, 0.75, retrieval_score, context_precision,
#                            recall_score, llm_calls, retrieved, qa_time)
#             return state

#         grounding_score = compute_answer_grounding(react_ans, retrieved_texts, question)
#         # 🔥 STRICT VERIFICATION CHECK
#         # if query_type == "VERIFICATION_QA":
#         #     if grounding_score < 80:
#         #         print("[QA] ❌ Verification failed → NOT FOUND")
#         #         answer = "This information is not present in the document."
#         #         decision_type = "verification_failed"
#         #     else:
#         #         answer = react_ans
#         #         decision_type = "accepted"
#         if query_type == "VERIFICATION_QA":
#             if grounding_score < 80:
#                 print("[QA] ❌ Verification failed → NOT FOUND")
#                 answer = "This information is not present in the document."
#                 decision_type = "verification_failed"
#                 state["answer"] = answer
#                 return state   # ❌ stop pipeline

#             else:
#                 answer = react_ans
#                 decision_type = "accepted"
#                 state["answer"] = answer
#                 return state   # ✅ ALSO stop pipeline

#         # 🔥 TRUST GATE (FINAL CLEAN VERSION)

#         recall = recall_score
#         grounding = grounding_score
#         length = len(react_ans.split())

#         # 1. No reliable context → reject
#         if recall < 25:
#             print("[QA] ❌ Low recall → NOT FOUND")
#             answer = "This information is not present in the document."
#             decision_type = "low_recall"

#         # 2. Very weak grounding → reject
#         elif grounding < 40:
#             print("[QA] ❌ Very low grounding → NOT FOUND")
#             answer = "This information is not present in the document."
#             decision_type = "low_grounding"

#         # # 3. Short answers must be VERY strong
#         # elif length <= 3:
#         #     if recall >= 70 and grounding >= 70:
#         #         answer = react_ans
#         #         decision_type = "accepted"
        
#         #     else:
#         #         print("[QA] ❌ Weak short answer → NOT FOUND")
#         #         answer = "This information is not present in the document."
#         #         decision_type = "weak_short"

#         # # 4. Strong signals → accept
#         # elif recall >= 50 and grounding >= 60:
#         #     answer = react_ans
#         #     decision_type = "accepted"

#         # # 5. Otherwise reject safely
#         # else:
#         #     print("[QA] ❌ Uncertain → NOT FOUND")
#         #     answer = "This information is not present in the document."
#         #     decision_type = "uncertain"
#         elif length <= 3:
#             if grounding >= 60:
#                 answer = react_ans
#                 decision_type = "accepted"
#             else:
#                 print("[QA] ❌ Weak short answer → NOT FOUND")
#                 answer = "This information is not present in the document."
#                 decision_type = "weak_short"

#         # 4. Strong signals → accept
#         elif recall >= 50 and grounding >= 60:
#             answer = react_ans
#             decision_type = "accepted"

#         # 5. Otherwise reject safely
#         else:
#             print("[QA] ❌ Uncertain → NOT FOUND")
#             answer = "This information is not present in the document."
#             decision_type = "uncertain"
#     # ── STEP 5: METRICS ───────────────────────────────────────
#     grounding  = compute_answer_grounding(answer, retrieved_texts, question)
#     confidence = round(grounding / 100, 3)
#     qa_time    = time.time() - qa_start_t

#     print(f"[QA] Done in {qa_time:.1f}s | model={model_used} | "
#           f"grounding={grounding:.1f}% | recall={recall_score:.1f}% | "
#           f"confidence={confidence:.3f} | decision={decision_type}")

#     if not answer.strip():
#         answer = "Could not find a relevant answer in the PDF."
#     else:
#         answer = normalize_answer(answer)

#     state["answer"] = answer
#     _write_metrics(state, model_used, decision_type, grounding,
#                    confidence, retrieval_score, context_precision,
#                    recall_score, llm_calls, retrieved, qa_time)
#     return state
# def node_validate(state: DocState) -> DocState:
#     answer = state["answer"]
#     retry  = state.get("retry_count", 0)

#     # ── Retry for empty/very weak answers ────────────────────
#     if len(answer.strip()) < 3 and retry < 2:
#         state["retry_count"] = retry + 1
#         state["answer"]      = ""
#         return state

#     total_time    = time.time() - state["start_time"]
#     output_words  = len(answer.split())
#     output_tokens = output_words * 1.3

#     extract_time  = state["metrics"].get("extraction_time_sec", 0)
#     llm_time      = max(total_time - extract_time, 1)
#     tps           = round(output_tokens / llm_time, 2) if llm_time > 0 else 0

#     m = state["metrics"]

#     # ── Core metrics ─────────────────────────────────────────
#     m["response_time_sec"]    = round(total_time, 2)
#     m["extraction_time_sec"]  = m.get("extraction_time_sec", 0)
#     m["pages_processed"]      = state.get("page_count", 0)
#     m["characters_processed"] = state.get("char_count", 0)
#     m["words_processed"]      = len(state.get("extracted_text", "").split())

#     # ── Type-specific metrics ────────────────────────────────
#     if m.get("type") == "summary":
#         m["summary_time_sec"]     = m.get("summary_time_sec", 0)
#         m["summary_length_words"] = len(answer.split())

#     if m.get("type") == "qa":
#         m["qa_time_sec"]      = m.get("qa_time_sec", 0)
#         m["confidence_score"] = m.get("confidence_score", 0)

#     # ── Performance ──────────────────────────────────────────
#     m["ttft_sec"]        = round(total_time, 2)
#     m["e2e_latency_sec"] = round(total_time, 2)
#     m["tps"]             = tps

#     # ❌ REMOVED doc_type (no longer used anywhere)
#     # m["doc_type"] = state.get("doc_type", "general")

#     # ── Context info ─────────────────────────────────────────
#     m["query_type"]     = state.get("query_type", "")
#     m["chunks_created"] = m.get("chunks_created", 0)
#     m["retry_count"]    = retry

#     # ── Model + retrieval metrics ────────────────────────────
#     m["model_used"]        = m.get("model_used", "llama_react")
#     m["llm_calls"]         = m.get("llm_calls", 0)
#     m["retrieval_score"]   = m.get("retrieval_score", 0)
#     m["context_precision"] = m.get("context_precision", 0)
#     m["answer_grounding"]  = m.get("answer_grounding", 0)
#     m["recall_at_k"]       = m.get("recall_at_k", 0)

#     # ── Summary-specific ─────────────────────────────────────
#     if m.get("type") == "summary":
#         m["parallel_workers"] = m.get("parallel_workers", 0)
#         m["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
#         m["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)

#     # ── QA-specific ──────────────────────────────────────────
#     if m.get("type") == "qa":
#         m["chunks_retrieved"] = m.get("chunks_retrieved", 0)
#         m["decision_type"]    = m.get("decision_type", "accepted")
#         m["confidence_raw"]   = m.get("confidence_raw", 0.0)

#     state["metrics"] = m
#     return state

# # def node_validate(state: DocState) -> DocState:
# #     answer = state["answer"]
# #     retry  = state.get("retry_count", 0)

# #     if len(answer.strip()) < 3 and retry < 2:
# #         state["retry_count"] = retry + 1
# #         state["answer"]      = ""
# #         return state

# #     total_time    = time.time() - state["start_time"]
# #     output_words  = len(answer.split())
# #     output_tokens = output_words * 1.3
# #     extract_time  = state["metrics"].get("extraction_time_sec", 0)
# #     llm_time      = max(total_time - extract_time, 1)
# #     tps           = round(output_tokens / llm_time, 2) if llm_time > 0 else 0
# #     m             = state["metrics"]

# #     m["response_time_sec"]    = round(total_time, 2)
# #     m["extraction_time_sec"]  = m.get("extraction_time_sec", 0)
# #     m["pages_processed"]      = state.get("page_count", 0)
# #     m["characters_processed"] = state.get("char_count", 0)
# #     m["words_processed"]      = len(state.get("extracted_text", "").split())

# #     if m.get("type") == "summary":
# #         m["summary_time_sec"]     = m.get("summary_time_sec", 0)
# #         m["summary_length_words"] = len(answer.split())
# #     if m.get("type") == "qa":
# #         m["qa_time_sec"]      = m.get("qa_time_sec", 0)
# #         m["confidence_score"] = m.get("confidence_score", 0)

# #     m["ttft_sec"]          = round(total_time, 2)
# #     m["e2e_latency_sec"]   = round(total_time, 2)
# #     m["tps"]               = tps
# #     m["doc_type"]          = state.get("doc_type", "general")
# #     m["query_type"]        = state.get("query_type", "")
# #     m["chunks_created"]    = m.get("chunks_created", 0)
# #     m["retry_count"]       = retry
# #     m["model_used"]        = m.get("model_used", "llama_react")
# #     m["llm_calls"]         = m.get("llm_calls", 0)
# #     m["retrieval_score"]   = m.get("retrieval_score", 0)
# #     m["context_precision"] = m.get("context_precision", 0)
# #     m["answer_grounding"]  = m.get("answer_grounding", 0)
# #     m["recall_at_k"]       = m.get("recall_at_k", 0)

# #     if m.get("type") == "summary":
# #         m["parallel_workers"] = m.get("parallel_workers", 0)
# #         m["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
# #         m["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)
# #     if m.get("type") == "qa":
# #         m["chunks_retrieved"] = m.get("chunks_retrieved", 0)
# #         m["decision_type"]    = m.get("decision_type", "accepted")
# #         m["confidence_raw"]   = m.get("confidence_raw", 0.0)

# #     state["metrics"] = m
# #     return state


# def _write_metrics(state, model_used, decision_type, grounding,
#                    confidence, retrieval_score, context_precision,
#                    recall_score, llm_calls, retrieved, qa_time):
#     m = state.setdefault("metrics", {})
#     m["qa_time_sec"]       = round(qa_time, 2)
#     m["confidence_score"]  = round(confidence * 100, 2)
#     m["retrieval_score"]   = retrieval_score
#     m["context_precision"] = context_precision
#     m["answer_grounding"]  = grounding
#     m["recall_at_k"]       = recall_score
#     m["llm_calls"]         = llm_calls
#     m["model_used"]        = model_used
#     m["chunks_retrieved"]  = len(retrieved)
#     m["type"]              = "qa"
#     m["decision_type"]     = decision_type
#     m["confidence_raw"]    = round(confidence, 4)






























# got 80 nd 80,44
# import time
# import re
# import spacy
# _nlp = spacy.load("en_core_web_lg")

# from docmind_rag.config.settings import (
#     MAX_WORKERS, FACTUAL_TOP_K, MULTIPART_TOP_K
# )
# from docmind_rag.utils.text import extract_named_entities, expand_answer
# from docmind_rag.core.state import DocState
# from docmind_rag.core.prompts import QA_PROMPT
# from docmind_rag.models.llm import call_llama_streaming
# from docmind_rag.models.embeddings import build_faiss_index
# from docmind_rag.models.reranker import rerank_docs, protect_exact_matches
# from docmind_rag.services.extraction import extract_pdf_parallel, _extraction_cache
# from docmind_rag.services.chunking import semantic_chunk, raptor_summarize, _summary_cache
# from docmind_rag.services.retrieval import multi_query_retrieve
# from docmind_rag.services.agent import react_agent, roberta_qa, clean_reasoning_answer
# from docmind_rag.events.events import emit_event
# from docmind_rag.utils.helpers import get_pdf_hash, clean_artifacts, clean_chunk_text, normalize_answer
# from docmind_rag.utils.text import (
#     classify_from_context,
#     reorder_by_question,
#     normalize_text,
# )
# from docmind_rag.utils.metrics import (
#     compute_answer_grounding,
#     compute_retrieval_score,
#     compute_context_precision,
#     compute_recall_at_k,
#     semantic_similarity,
#     _is_refusal_answer,
# )


# # ============================================================
# # LOCAL HELPERS
# # ============================================================

# def clean_context_for_llm(chunks):
#     cleaned_chunks = []
#     for chunk in chunks:
#         lines = chunk.split("\n")
#         filtered_lines = []
#         for line in lines:
#             line_strip = line.strip()
#             is_question_like = (
#                 line_strip.endswith("?") or
#                 (len(line_strip.split()) < 15 and "?" in line_strip)
#             )
#             if is_question_like:
#                 continue
#             filtered_lines.append(line)
#         cleaned_chunks.append("\n".join(filtered_lines))
#     return cleaned_chunks


# def is_grounded(answer, retrieved_texts, threshold=0.6):
#     if not answer:
#         return False
#     answer_words = set(answer.lower().split())
#     if not answer_words:
#         return False
#     best_overlap = 0
#     for chunk in retrieved_texts:
#         chunk_words = set(chunk.lower().split())
#         overlap = len(answer_words & chunk_words) / len(answer_words)
#         best_overlap = max(best_overlap, overlap)
#     return best_overlap >= threshold


# # ============================================================
# # NUMERIC INTENT DETECTION
# # ============================================================

# def _extract_numbers(text: str) -> list:
#     return re.findall(r'\b\d+\b', text)


# def _detect_numeric_intent(question: str, chunks: list) -> str:
#     """
#     Detect NAVIGATIONAL or POSITIONAL from chunk structure.
#     Generic — no hardcoded section words.
#     """
#     numbers = _extract_numbers(question)
#     if not numbers:
#         return "NONE"

#     q_words = set(question.lower().split())

#     for chunk in chunks:
#         lines = [l.strip() for l in chunk.split("\n") if l.strip()]
#         if not lines:
#             continue
#         first_line = lines[0].lower()
#         line_words = set(first_line.split())
#         overlap    = len(q_words & line_words)
#         if any(re.search(rf'\b{re.escape(num)}\b', first_line) for num in numbers):
#             if overlap >= 1 and len(first_line.split()) <= 12:
#                 return "NAVIGATIONAL"

#     for chunk in chunks:
#         lines       = [l.strip() for l in chunk.split("\n") if l.strip()]
#         short_lines = [l for l in lines if len(l.split()) <= 12]
#         if len(short_lines) >= 3:
#             first = lines[0].lower() if lines else ""
#             if not any(re.search(rf'\b{re.escape(num)}\b', first) for num in numbers):
#                 return "POSITIONAL"

#     return "NONE"


# def _navigate_full_chunks(question: str, all_raw_chunks: list) -> str:
#     """
#     Scan full raw chunks to find section title for a numbered section.
#     Generic — works for Chapter N, Lecture N, Section N, etc.
#     """
#     numbers = _extract_numbers(question)
#     if not numbers:
#         return ""

#     q_words    = set(question.lower().split())
#     best_line  = ""
#     best_score = 0

#     for chunk in all_raw_chunks:
#         lines = [l.strip() for l in chunk.split("\n") if l.strip()]
#         if not lines:
#             continue
#         first_line  = lines[0]
#         first_lower = first_line.lower()

#         for num in numbers:
#             if not re.search(rf'\b{re.escape(num)}\b', first_lower):
#                 continue
#             line_words = set(first_lower.split())
#             overlap    = len(q_words & line_words)
#             if overlap >= 1 and len(first_line.split()) <= 14:
#                 score = overlap + (5 if len(first_line.split()) <= 8 else 0)
#                 if score > best_score:
#                     best_score = score
#                     best_line  = first_line

#     if not best_line:
#         return ""

#     title = best_line
#     title = re.sub(r'^\s*\d+\s*[-—–]+\s*[—–-]?\s*\d*\s*', '', title)
#     title = re.sub(r'\b\d+\b', '', title)
#     title = re.sub(r'[:\-–—]', ' ', title)
#     title = re.sub(r'\s{2,}', ' ', title)
#     return title.strip()


# def _positional_extract(question: str, retrieved_texts: list) -> str:
#     """Extract Nth item from a list. Generic."""
#     numbers = _extract_numbers(question)
#     if not numbers:
#         return ""
#     try:
#         idx = int(numbers[0]) - 1
#     except ValueError:
#         return ""
#     if idx < 0:
#         return ""

#     best_chunk, best_count = "", 0
#     for chunk in retrieved_texts:
#         lines       = [l.strip() for l in chunk.split("\n") if l.strip()]
#         short_lines = [l for l in lines if len(l.split()) <= 12]
#         if len(short_lines) > best_count:
#             best_count = len(short_lines)
#             best_chunk = chunk

#     if not best_chunk:
#         return ""

#     lines       = [l.strip() for l in best_chunk.split("\n") if l.strip()]
#     short_lines = [l for l in lines if len(l.split()) <= 12]
#     if idx < len(short_lines):
#         return re.sub(r'^\s*\d+[\.\)]\s*', '', short_lines[idx]).strip()
#     return ""


# # ============================================================
# # SIGNAL 3 — Structural consistency for NAVIGATIONAL answers
# # ============================================================

# def _check_identifier_grounded(answer: str, retrieved_texts: list) -> bool:
#     """
#     If answer contains a multi-word identifier (word+number or number+word),
#     verify it appears verbatim in retrieved chunks.
#     Generic — no hardcoded section words.
#     """
#     identifiers = re.findall(
#         r'\b[a-zA-Z]+\s+\d+\b|\b\d+\s+[a-zA-Z]+\b',
#         answer.lower()
#     )
#     if not identifiers:
#         return True

#     chunk_text = " ".join(retrieved_texts).lower()
#     for ident in identifiers:
#         if ident not in chunk_text:
#             print(f"[Signal3] ❌ Identifier '{ident}' not in retrieved chunks")
#             return False
#     return True


# # ============================================================
# # REFUSAL — semantic version (local)
# # ============================================================

# _REFUSAL_ANCHOR = "this information is not available in the provided context"


# def _is_refusal_semantic(text: str) -> bool:
#     if not text or len(text.strip()) < 2:
#         return True
#     sim = semantic_similarity(text, [_REFUSAL_ANCHOR])
#     print(f"[Refusal] sim={sim:.3f} | '{text[:50]}'")
#     return sim > 0.72


# # ============================================================
# # LANGGRAPH NODES
# # ============================================================

# def node_extract(state: DocState) -> DocState:
#     extract_start    = time.time()
#     text, page_count = extract_pdf_parallel(state["pdf_path"])
#     extract_time     = time.time() - extract_start
#     state["extracted_text"] = text
#     state["page_count"]     = page_count
#     state["char_count"]     = len(text)
#     state["metrics"]["extraction_time_sec"]  = round(extract_time, 2)
#     state["metrics"]["pages_processed"]      = page_count
#     state["metrics"]["characters_processed"] = len(text)
#     state["metrics"]["words_processed"]      = len(text.split())
#     return state


# def node_chunk(state: DocState) -> DocState:
#     text = state["extracted_text"]
#     state["query_type"] = "QA"

#     summary_chunks, rag_chunks = semantic_chunk(text)

#     # FIX 4 — normalize all chunks after chunking
#     # Converts unicode symbols so retrieval doesn't fail on symbol mismatches
#     rag_chunks     = [normalize_text(c) for c in rag_chunks]
#     summary_chunks = [normalize_text(c) for c in summary_chunks]

#     state["summary_chunks"] = summary_chunks
#     state["chunks"]         = rag_chunks

#     state["metrics"]["summary_chunks"] = len(summary_chunks)
#     state["metrics"]["chunks_created"] = len(rag_chunks)
#     # state["metrics"]["doc_type"]       = state.get("doc_type", "general")
#     state["metrics"]["query_type"]     = state["query_type"]
#     print(f"[Chunk] {len(summary_chunks)} summary chunks | {len(rag_chunks)} RAG chunks")
#     return state


# # def node_summarize(state: DocState) -> DocState:
# #     pdf_hash  = get_pdf_hash(state["pdf_path"])
# #     cache_key = f"{pdf_hash}_{state.get('doc_type', 'general')}"

# #     if cache_key in _summary_cache:
# #         cached = _summary_cache[cache_key]
# #         print("[Summary] ✅ Cache hit")
# #         emit_event(state.get("request_id", ""), "agent_action",
# #                    "⚡ Summary loaded from cache instantly!")
# #         state["answer"] = cached["summary"]
# #         state["metrics"].update(cached["metrics"])
# #         state["metrics"]["type"] = "summary"
# #         return state

# #     summary_start = time.time()
# #     raptor_summarize._request_id = state.get("request_id", "")
# #     summary, map_time, reduce_time = raptor_summarize(
# #         state["summary_chunks"], state.get("doc_type", "general")
# #     )
# #     summary_time = time.time() - summary_start

# #     state["answer"] = summary
# #     metrics_snapshot = {
# #         "summary_time_sec":     round(summary_time, 2),
# #         "summary_length_words": len(summary.split()),
# #         "parallel_workers":     min(MAX_WORKERS, len(state["summary_chunks"])),
# #         "map_time_sec":         round(map_time, 2),
# #         "reduce_time_sec":      round(reduce_time, 2),
# #         "llm_calls":            3,
# #     }
# #     state["metrics"].update(metrics_snapshot)
# #     state["metrics"]["type"] = "summary"
# #     _summary_cache[cache_key] = {"summary": summary, "metrics": metrics_snapshot}
# #     print(f"[Summary] Done ({len(summary.split())} words)")
# #     return state

# def node_summarize(state: DocState) -> DocState:
#     pdf_hash  = get_pdf_hash(state["pdf_path"])
#     cache_key = pdf_hash  # ❌ removed doc_type dependency

#     # ── Cache check ──────────────────────────────────────────
#     if cache_key in _summary_cache:
#         cached = _summary_cache[cache_key]
#         print("[Summary] ✅ Cache hit")
#         emit_event(
#             state.get("request_id", ""),
#             "agent_action",
#             "⚡ Summary loaded from cache instantly!"
#         )
#         state["answer"] = cached["summary"]
#         state["metrics"].update(cached["metrics"])
#         state["metrics"]["type"] = "summary"
#         return state

#     # ── Generate summary ─────────────────────────────────────
#     summary_start = time.time()
#     raptor_summarize._request_id = state.get("request_id", "")

#     # ❌ removed doc_type argument
#     summary, map_time, reduce_time = raptor_summarize(
#         state["summary_chunks"]
#     )

#     summary_time = time.time() - summary_start

#     # ── Store result ─────────────────────────────────────────
#     state["answer"] = summary

#     metrics_snapshot = {
#         "summary_time_sec":     round(summary_time, 2),
#         "summary_length_words": len(summary.split()),
#         "parallel_workers":     min(MAX_WORKERS, len(state["summary_chunks"])),
#         "map_time_sec":         round(map_time, 2),
#         "reduce_time_sec":      round(reduce_time, 2),
#         "llm_calls":            3,
#     }

#     state["metrics"].update(metrics_snapshot)
#     state["metrics"]["type"] = "summary"

#     # ── Cache result ─────────────────────────────────────────
#     _summary_cache[cache_key] = {
#         "summary": summary,
#         "metrics": metrics_snapshot
#     }

#     print(f"[Summary] Done ({len(summary.split())} words)")
#     return state

# def node_qa(state: DocState) -> DocState:
#     qa_start_t = time.time()
#     question   = state["question"]
#     request_id = state.get("request_id", "")
#     all_chunks = state["chunks"]

#     # FIX 4 — normalize question the same way chunks were normalized
#     question = normalize_text(question)

#     # Raw strings for navigational scanning — must NOT be cleaned
#     all_raw = [
#         d if isinstance(d, str) else d.page_content
#         for d in all_chunks
#     ]

#     pdf_hash    = get_pdf_hash(state["pdf_path"])
#     faiss_index = build_faiss_index(all_chunks, pdf_hash)

#     print(f"[QA] START | {len(all_chunks)} chunks")

#     recall_score      = 0.0
#     retrieval_score   = 0.0
#     context_precision = 0.0
#     grounding         = 0.0
#     llm_calls         = 0
#     model_used        = "llama"
#     confidence        = 0.0
#     decision_type     = "accepted"
#     retrieved         = []
#     retrieved_texts   = []
#     answer            = ""

#     # ── STEP 1: RETRIEVE ─────────────────────────────────────
#     retrieved = multi_query_retrieve(
#         question, faiss_index,
#         k=20,
#         all_chunks=all_chunks,
#         query_type="FACTUAL_QA"
#     )


#     # ── STEP 2: RERANK ────────────────────────────────────────
#     retrieved, reranker_top, _ = rerank_docs(
#         question, retrieved, top_k=8, apply_pruning=True
#     )
#     retrieved       = protect_exact_matches(
#         question, retrieved, all_chunks, top_k=8
#     )
#     retrieved_texts = [d.page_content for d in retrieved]

#     if reranker_top < -5:
#         print("[Guard] ⚠️ Weak reranker score — continuing")

#     # ── STEP 3: METRICS ───────────────────────────────────────
#     retrieval_score   = compute_retrieval_score(question, retrieved)
#     context_precision = compute_context_precision(question, retrieved)
#     recall_score      = compute_recall_at_k(
#         question, retrieved, all_chunks, k=len(retrieved)
#     )

#     # ── FIX 6 — k expansion: low recall OR numeric question ───
#     is_numeric_question = len(_extract_numbers(question)) > 0
#     if recall_score < 25 or is_numeric_question:
#         print("[QA] ⚠️ Expanding retrieval (low recall or numeric question)")
#         expanded = multi_query_retrieve(
#             question, faiss_index, k=30,
#             all_chunks=all_chunks, query_type="FACTUAL_QA"
#         )
#         if len(expanded) > len(retrieved):
#             retrieved       = expanded
#             retrieved_texts = [d.page_content for d in retrieved]

#     # ── FIX 1 — Numeric intent: retrieved first, fallback to all_raw ──
#     numeric_intent = _detect_numeric_intent(question, retrieved_texts)

#     if numeric_intent == "NONE" and is_numeric_question:
#         print("[Navigate] Retrying intent detection on full document...")
#         numeric_intent = _detect_numeric_intent(question, all_raw)
#         if numeric_intent != "NONE":
#             print(f"[Navigate] Intent found in full document: {numeric_intent}")

#     # ── NAVIGATIONAL ──────────────────────────────────────────
#     if numeric_intent == "NAVIGATIONAL":
#         print(f"[Routing] Numeric intent → NAVIGATIONAL")
#         title = _navigate_full_chunks(question, all_raw)
#         if title:
#             print(f"[Navigate] Extracted title: '{title}'")
#             grounding  = compute_answer_grounding(title, retrieved_texts, question)
#             confidence = round(grounding / 100, 3)
#             qa_time    = time.time() - qa_start_t
#             state["answer"]      = title
#             state["query_type"]  = "NAVIGATIONAL"
#             _write_metrics(state, "navigational", "navigational", grounding,
#                            confidence, retrieval_score, context_precision,
#                            recall_score, llm_calls, retrieved, qa_time)
#             return state
#         print(f"[Navigate] Extraction failed — falling through to LLM")

#     # ── POSITIONAL ────────────────────────────────────────────
#     elif numeric_intent == "POSITIONAL":
#         print(f"[Routing] Numeric intent → POSITIONAL")
#         pos_answer = _positional_extract(question, retrieved_texts)
#         if pos_answer:
#             print(f"[Positional] Extracted: '{pos_answer}'")
#             grounding  = compute_answer_grounding(pos_answer, retrieved_texts, question)
#             confidence = round(grounding / 100, 3)
#             qa_time    = time.time() - qa_start_t
#             state["answer"]      = pos_answer
#             state["query_type"]  = "POSITIONAL"
#             _write_metrics(state, "positional", "positional", grounding,
#                            confidence, retrieval_score, context_precision,
#                            recall_score, llm_calls, retrieved, qa_time)
#             return state
#         print(f"[Positional] Extraction failed — falling through to LLM")

#     # ── Normal routing ────────────────────────────────────────
#     query_type = classify_from_context(question, retrieved_texts)
#     state["query_type"] = query_type
#     print(f"[Routing] Context-based → {query_type}")

#     # ── STEP 4: ANSWER GENERATION ─────────────────────────────

#     if query_type == "FULL_SUMMARY":
#         retrieved_texts = clean_context_for_llm(retrieved_texts)
#         ranked  = reorder_by_question(question, retrieved_texts)
#         context = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:8])
#         structured_context = "\n".join(
#             f"[CONTEXT CHUNK]\n{c}" for c in context.split("\n\n---\n\n")
#         )
#         emit_event(request_id, "stream_start", "✍️ Generating answer...")
#         answer, _ = call_llama_streaming(
#             QA_PROMPT.format(context=structured_context[:2500], question=question),
#             request_id=request_id, temperature=0.0
#         )
#         answer     = clean_artifacts(answer).strip()
#         model_used = "llama_summary"
#         llm_calls  = 1

#     elif query_type == "MULTIPART_QA":
#         retrieved_texts = clean_context_for_llm(retrieved_texts)
#         ranked  = reorder_by_question(question, retrieved_texts)
#         # FIX 3 — 4000 chars for MULTIPART to capture all list items
#         context = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:8])
#         emit_event(request_id, "agent_start",
#                    f"🤖 MULTIPART | {len(retrieved)} chunks")
#         emit_event(request_id, "stream_start", "✍️ Generating answer...")
#         answer, _ = call_llama_streaming(
#             QA_PROMPT.format(context=context[:4000], question=question),
#             request_id=request_id, temperature=0.0
#         )
#         answer     = clean_artifacts(answer).strip()
#         model_used = "llama_multipart"
#         llm_calls  = 1

#     else:
#         # FACTUAL_QA / VERIFICATION_QA

#         if not retrieved_texts:
#             print("[QA] ❌ No context → NOT FOUND")
#             state["answer"] = "This information is not present in the document."
#             _write_metrics(state, "not_found", "no_context",
#                            0.0, 0.0, retrieval_score, context_precision,
#                            recall_score, llm_calls, retrieved,
#                            time.time() - qa_start_t)
#             return state

#         react_ans, model_used, _, _, _, _, react_calls, retrieved_texts = react_agent(
#             question, faiss_index, query_type, all_chunks, request_id
#         )
#         llm_calls += react_calls

#         react_ans = clean_artifacts(react_ans).strip().strip('"').strip("'")
#         react_ans = clean_reasoning_answer(react_ans, question)
#         print(f"[QA] ReAct: '{react_ans[:60]}'")

#         # FIX 2 — Short answer bypass: skip for VERIFICATION_QA
#         words         = react_ans.split()
#         content_words = [w for w in words if len(w) > 2 and not w.isdigit()]

#         if (
#             query_type != "VERIFICATION_QA"
#             and len(words) <= 3
#             and len(content_words) >= 1
#             and not _is_refusal_answer(react_ans)
#         ):
#             print("[QA] ⚡ Short answer accepted")
#             grounding  = compute_answer_grounding(react_ans, retrieved_texts, question)
#             confidence = grounding / 100
#             qa_time    = time.time() - qa_start_t
#             state["answer"] = react_ans
#             _write_metrics(state, model_used, "short_answer",
#                            grounding, confidence, retrieval_score,
#                            context_precision, recall_score, llm_calls,
#                            retrieved, qa_time)
#             return state

#         # Refusal
#         if _is_refusal_answer(react_ans):
#             print("[QA] ⚠️ Refusal detected")
#             qa_time = time.time() - qa_start_t
#             state["answer"] = "This information is not present in the document."
#             _write_metrics(state, "not_found", "not_found",
#                            75.0, 0.75, retrieval_score, context_precision,
#                            recall_score, llm_calls, retrieved, qa_time)
#             return state

#         grounding_score = compute_answer_grounding(react_ans, retrieved_texts, question)
#         short_answer    = len(react_ans.split()) <= 2

#         # Signal 1 — weak retrieval
#         if recall_score < 25 and grounding_score < 25:
#             print("[QA] ❌ Weak retrieval → NOT FOUND")
#             answer        = "This information is not present in the document."
#             model_used    = "not_found"
#             decision_type = "weak_retrieval"

#         # Signal 2 — low grounding on long answer
#         elif grounding_score < 30 and not short_answer:
#             print("[QA] ❌ Not grounded → reject")
#             answer        = "This information is not present in the document."
#             decision_type = "low_grounding"

#         # FIX 7 — Signal 3: identifier not found in context
#         elif numeric_intent == "NAVIGATIONAL" and not _check_identifier_grounded(
#             react_ans, retrieved_texts
#         ):
#             print("[QA] ❌ Ungrounded identifier → NOT FOUND")
#             answer        = "This information is not present in the document."
#             decision_type = "hallucination_blocked"

#         else:
#             answer        = react_ans
#             decision_type = "accepted"

#     # ── STEP 5: METRICS ───────────────────────────────────────
#     grounding  = compute_answer_grounding(answer, retrieved_texts, question)
#     confidence = round(grounding / 100, 3)
#     qa_time    = time.time() - qa_start_t

#     print(f"[QA] Done in {qa_time:.1f}s | model={model_used} | "
#           f"grounding={grounding:.1f}% | recall={recall_score:.1f}% | "
#           f"confidence={confidence:.3f} | decision={decision_type}")

#     if not answer.strip():
#         answer = "Could not find a relevant answer in the PDF."
#     else:
#         answer = normalize_answer(answer)

#     state["answer"] = answer
#     _write_metrics(state, model_used, decision_type, grounding,
#                    confidence, retrieval_score, context_precision,
#                    recall_score, llm_calls, retrieved, qa_time)
#     return state
# def node_validate(state: DocState) -> DocState:
#     answer = state["answer"]
#     retry  = state.get("retry_count", 0)

#     # ── Retry for empty/very weak answers ────────────────────
#     if len(answer.strip()) < 3 and retry < 2:
#         state["retry_count"] = retry + 1
#         state["answer"]      = ""
#         return state

#     total_time    = time.time() - state["start_time"]
#     output_words  = len(answer.split())
#     output_tokens = output_words * 1.3

#     extract_time  = state["metrics"].get("extraction_time_sec", 0)
#     llm_time      = max(total_time - extract_time, 1)
#     tps           = round(output_tokens / llm_time, 2) if llm_time > 0 else 0

#     m = state["metrics"]

#     # ── Core metrics ─────────────────────────────────────────
#     m["response_time_sec"]    = round(total_time, 2)
#     m["extraction_time_sec"]  = m.get("extraction_time_sec", 0)
#     m["pages_processed"]      = state.get("page_count", 0)
#     m["characters_processed"] = state.get("char_count", 0)
#     m["words_processed"]      = len(state.get("extracted_text", "").split())

#     # ── Type-specific metrics ────────────────────────────────
#     if m.get("type") == "summary":
#         m["summary_time_sec"]     = m.get("summary_time_sec", 0)
#         m["summary_length_words"] = len(answer.split())

#     if m.get("type") == "qa":
#         m["qa_time_sec"]      = m.get("qa_time_sec", 0)
#         m["confidence_score"] = m.get("confidence_score", 0)

#     # ── Performance ──────────────────────────────────────────
#     m["ttft_sec"]        = round(total_time, 2)
#     m["e2e_latency_sec"] = round(total_time, 2)
#     m["tps"]             = tps

#     # ❌ REMOVED doc_type (no longer used anywhere)
#     # m["doc_type"] = state.get("doc_type", "general")

#     # ── Context info ─────────────────────────────────────────
#     m["query_type"]     = state.get("query_type", "")
#     m["chunks_created"] = m.get("chunks_created", 0)
#     m["retry_count"]    = retry

#     # ── Model + retrieval metrics ────────────────────────────
#     m["model_used"]        = m.get("model_used", "llama_react")
#     m["llm_calls"]         = m.get("llm_calls", 0)
#     m["retrieval_score"]   = m.get("retrieval_score", 0)
#     m["context_precision"] = m.get("context_precision", 0)
#     m["answer_grounding"]  = m.get("answer_grounding", 0)
#     m["recall_at_k"]       = m.get("recall_at_k", 0)

#     # ── Summary-specific ─────────────────────────────────────
#     if m.get("type") == "summary":
#         m["parallel_workers"] = m.get("parallel_workers", 0)
#         m["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
#         m["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)

#     # ── QA-specific ──────────────────────────────────────────
#     if m.get("type") == "qa":
#         m["chunks_retrieved"] = m.get("chunks_retrieved", 0)
#         m["decision_type"]    = m.get("decision_type", "accepted")
#         m["confidence_raw"]   = m.get("confidence_raw", 0.0)

#     state["metrics"] = m
#     return state

# # def node_validate(state: DocState) -> DocState:
# #     answer = state["answer"]
# #     retry  = state.get("retry_count", 0)

# #     if len(answer.strip()) < 3 and retry < 2:
# #         state["retry_count"] = retry + 1
# #         state["answer"]      = ""
# #         return state

# #     total_time    = time.time() - state["start_time"]
# #     output_words  = len(answer.split())
# #     output_tokens = output_words * 1.3
# #     extract_time  = state["metrics"].get("extraction_time_sec", 0)
# #     llm_time      = max(total_time - extract_time, 1)
# #     tps           = round(output_tokens / llm_time, 2) if llm_time > 0 else 0
# #     m             = state["metrics"]

# #     m["response_time_sec"]    = round(total_time, 2)
# #     m["extraction_time_sec"]  = m.get("extraction_time_sec", 0)
# #     m["pages_processed"]      = state.get("page_count", 0)
# #     m["characters_processed"] = state.get("char_count", 0)
# #     m["words_processed"]      = len(state.get("extracted_text", "").split())

# #     if m.get("type") == "summary":
# #         m["summary_time_sec"]     = m.get("summary_time_sec", 0)
# #         m["summary_length_words"] = len(answer.split())
# #     if m.get("type") == "qa":
# #         m["qa_time_sec"]      = m.get("qa_time_sec", 0)
# #         m["confidence_score"] = m.get("confidence_score", 0)

# #     m["ttft_sec"]          = round(total_time, 2)
# #     m["e2e_latency_sec"]   = round(total_time, 2)
# #     m["tps"]               = tps
# #     m["doc_type"]          = state.get("doc_type", "general")
# #     m["query_type"]        = state.get("query_type", "")
# #     m["chunks_created"]    = m.get("chunks_created", 0)
# #     m["retry_count"]       = retry
# #     m["model_used"]        = m.get("model_used", "llama_react")
# #     m["llm_calls"]         = m.get("llm_calls", 0)
# #     m["retrieval_score"]   = m.get("retrieval_score", 0)
# #     m["context_precision"] = m.get("context_precision", 0)
# #     m["answer_grounding"]  = m.get("answer_grounding", 0)
# #     m["recall_at_k"]       = m.get("recall_at_k", 0)

# #     if m.get("type") == "summary":
# #         m["parallel_workers"] = m.get("parallel_workers", 0)
# #         m["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
# #         m["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)
# #     if m.get("type") == "qa":
# #         m["chunks_retrieved"] = m.get("chunks_retrieved", 0)
# #         m["decision_type"]    = m.get("decision_type", "accepted")
# #         m["confidence_raw"]   = m.get("confidence_raw", 0.0)

# #     state["metrics"] = m
# #     return state


# def _write_metrics(state, model_used, decision_type, grounding,
#                    confidence, retrieval_score, context_precision,
#                    recall_score, llm_calls, retrieved, qa_time):
#     m = state.setdefault("metrics", {})
#     m["qa_time_sec"]       = round(qa_time, 2)
#     m["confidence_score"]  = round(confidence * 100, 2)
#     m["retrieval_score"]   = retrieval_score
#     m["context_precision"] = context_precision
#     m["answer_grounding"]  = grounding
#     m["recall_at_k"]       = recall_score
#     m["llm_calls"]         = llm_calls
#     m["model_used"]        = model_used
#     m["chunks_retrieved"]  = len(retrieved)
#     m["type"]              = "qa"
#     m["decision_type"]     = decision_type
#     m["confidence_raw"]    = round(confidence, 4)





























# got 80 nd 70 - gonna change some lap claude 
# import time
# import re
# import spacy
# _nlp = spacy.load("en_core_web_lg")


# from docmind_rag.config.settings import (
#     MAX_WORKERS, FACTUAL_TOP_K, MULTIPART_TOP_K
# )

# from docmind_rag.core.state import DocState
# from docmind_rag.core.prompts import QA_PROMPT
# from docmind_rag.models.llm import call_llama_streaming
# from docmind_rag.models.embeddings import build_faiss_index
# from docmind_rag.models.reranker import rerank_docs, protect_exact_matches
# from docmind_rag.services.extraction import extract_pdf_parallel, _extraction_cache
# from docmind_rag.services.chunking import semantic_chunk, raptor_summarize, _summary_cache
# from docmind_rag.services.retrieval import multi_query_retrieve
# from docmind_rag.services.agent import react_agent, roberta_qa,clean_reasoning_answer
# from docmind_rag.events.events import emit_event
# from docmind_rag.utils.helpers import get_pdf_hash, clean_artifacts, clean_chunk_text,normalize_answer
# from docmind_rag.utils.text import (
#     classify_from_context,
#     reorder_by_question
# )
# from docmind_rag.utils.metrics import (
#     compute_answer_grounding,
#     compute_retrieval_score,
#     compute_context_precision,
#     compute_recall_at_k,
#     semantic_similarity,_is_refusal_answer
# )


# # ============================================================
# # LOCAL HELPERS (unchanged from original)
# # ============================================================

# def clean_context_for_llm(chunks):
#     cleaned_chunks = []
#     for chunk in chunks:
#         lines = chunk.split("\n")
#         filtered_lines = []
#         for line in lines:
#             line_strip = line.strip()
#             is_question_like = (
#                 line_strip.endswith("?") or
#                 (len(line_strip.split()) < 15 and "?" in line_strip)
#             )
#             if is_question_like:
#                 continue
#             filtered_lines.append(line)
#         cleaned_chunks.append("\n".join(filtered_lines))
#     return cleaned_chunks


# def is_grounded(answer, retrieved_texts, threshold=0.6):
#     if not answer:
#         return False
#     answer_words = set(answer.lower().split())
#     if not answer_words:
#         return False
#     best_overlap = 0
#     for chunk in retrieved_texts:
#         chunk_words = set(chunk.lower().split())
#         overlap = len(answer_words & chunk_words) / len(answer_words)
#         best_overlap = max(best_overlap, overlap)
#     return best_overlap >= threshold


# # ============================================================
# # CHANGE A — Structural question classifier (no embeddings)
# # ============================================================

# _LOOKUP_STARTS = (
#     "what is", "what are", "what was", "what were",
#     "who is", "who was", "who are",
#     "when did", "when was", "when were",
#     "how many", "how much",
#     "name the", "name of",
# )

# # _REASONING_WORDS = {
# #     "because", "therefore", "however", "although", "since",
# #     "thus", "hence", "consequently", "furthermore", "moreover",
# #     "nevertheless", "regardless", "whereas", "while", "despite",
# # }


# def _is_simple_lookup(question: str) -> bool:
#     """
#     True only for short, structurally simple factual questions.
#     No embeddings — pure string logic. Generic across all PDFs.
#     """
#     q     = question.strip().lower()
#     words = q.split()
#     if len(words) > 12:
#         return False
#     if not any(q.startswith(s) for s in _LOOKUP_STARTS):
#         return False
#     return True


# # def _span_is_clean(span: str) -> bool:
# #     """Span is clean if short and contains no reasoning language."""
# #     words = span.strip().split()
# #     if len(words) > 6:
# #         return False
# #     if any(w.lower() in _REASONING_WORDS for w in words):
# #         return False
# #     return True


# def _select_best_span_structural(question: str, chunks: list) -> str:
#     """
#     Select best short span using word overlap.
#     Only called after _is_simple_lookup passes.
#     """
#     q_words    = set(question.lower().split())
#     best_span  = ""
#     best_score = 0

#     for chunk in chunks:
#         for line in chunk.split("\n"):
#             line = line.strip()
#             if not line:
#                 continue
#             wc = len(line.split())
#             if wc < 2 or wc > 6:
#                 continue
#             s_words = set(line.lower().split())
#             overlap = len(q_words & s_words)
#             if overlap > best_score:
#                 best_score = overlap
#                 best_span  = line
#             elif overlap == best_score and len(line.split()) > len(best_span.split()):
#                 best_span = line

#     return best_span


# # ============================================================
# # CHANGE B — Structure-aware numeric intent detection
# # ============================================================

# def _extract_numbers(text: str) -> list:
#     return re.findall(r'\b\d+\b', text)


# def _detect_numeric_intent(question: str, retrieved_texts: list) -> str:
#     """
#     Detect whether a numeric question is NAVIGATIONAL or POSITIONAL.
#     Uses retrieved_texts as fast signal only.
#     Returns: "NAVIGATIONAL" | "POSITIONAL" | "NONE"
#     """
#     numbers = _extract_numbers(question)
#     if not numbers:
#         return "NONE"

#     q_words = set(question.lower().split())

#     for chunk in retrieved_texts:
#         lines = [l.strip() for l in chunk.split("\n") if l.strip()]
#         if not lines:
#             continue
#         first_line = lines[0].lower()
#         line_words = set(first_line.split())
#         overlap    = len(q_words & line_words)

#         if any(re.search(rf'\b{re.escape(num)}\b', first_line) for num in numbers):
#             if overlap >= 1 and len(first_line.split()) <= 12:
#                 return "NAVIGATIONAL"

#     for chunk in retrieved_texts:
#         lines       = [l.strip() for l in chunk.split("\n") if l.strip()]
#         short_lines = [l for l in lines if len(l.split()) <= 12]
#         if len(short_lines) >= 3:
#             first = lines[0].lower() if lines else ""
#             if not any(
#                 re.search(rf'\b{re.escape(num)}\b', first) for num in numbers
#             ):
#                 return "POSITIONAL"

#     return "NONE"


# def _navigate_full_chunks(question: str, all_raw_chunks: list) -> str:
#     """
#     NAVIGATIONAL extraction scanning the FULL raw chunk set.
#     Must use raw chunks — clean_chunk_text strips the numbering we need.
#     Generic — works for any numbered section in any document.
#     """
#     numbers = _extract_numbers(question)
#     if not numbers:
#         return ""

#     q_words    = set(question.lower().split())
#     best_line  = ""
#     best_score = 0

#     for chunk in all_raw_chunks:
#         lines = [l.strip() for l in chunk.split("\n") if l.strip()]
#         if not lines:
#             continue

#         first_line  = lines[0]
#         first_lower = first_line.lower()

#         for num in numbers:
#             if not re.search(rf'\b{re.escape(num)}\b', first_lower):
#                 continue
#             line_words = set(first_lower.split())
#             overlap    = len(q_words & line_words)
#             if overlap >= 1 and len(first_line.split()) <= 14:
#                 score = overlap + (5 if len(first_line.split()) <= 8 else 0)
#                 if score > best_score:
#                     best_score = score
#                     best_line  = first_line

#     if not best_line:
#         return ""

#     # Extract title: remove page markers, numbers, separators
#     title = best_line
#     title = re.sub(r'^\s*\d+\s*[-—–]+\s*[—–-]?\s*\d*\s*', '', title)
#     title = re.sub(r'\b\d+\b', '', title)
#     title = re.sub(r'[:\-–—]', ' ', title)
#     title = re.sub(r'\s{2,}', ' ', title)
#     title = title.strip()

#     return title if title else ""


# def _positional_extract(question: str, retrieved_texts: list) -> str:
#     """
#     POSITIONAL extraction: find Nth item in a list structure.
#     Generic — works for any numbered list in any document.
#     """
#     numbers = _extract_numbers(question)
#     if not numbers:
#         return ""

#     try:
#         idx = int(numbers[0]) - 1
#     except ValueError:
#         return ""

#     if idx < 0:
#         return ""

#     best_chunk       = ""
#     best_short_count = 0

#     for chunk in retrieved_texts:
#         lines       = [l.strip() for l in chunk.split("\n") if l.strip()]
#         short_lines = [l for l in lines if len(l.split()) <= 12]
#         if len(short_lines) > best_short_count:
#             best_short_count = len(short_lines)
#             best_chunk       = chunk

#     if not best_chunk:
#         return ""

#     lines       = [l.strip() for l in best_chunk.split("\n") if l.strip()]
#     short_lines = [l for l in lines if len(l.split()) <= 12]

#     if idx < len(short_lines):
#         answer = re.sub(r'^\s*\d+[\.\)]\s*', '', short_lines[idx]).strip()
#         return answer

#     return ""


# # ============================================================
# # REFUSAL HELPERS
# # ============================================================

# _REFUSAL_PHRASES = [
#     "not present", "not available", "not found",
#     "not in the document", "not present in the document",
#     "information is not", "cannot find", "no information",
# ]

# _REFUSAL_ANCHOR = "this information is not available in the provided context"


# def _is_refusal_answer(text: str) -> bool:
#     """
#     Lightweight string-matching refusal check. No LLM, no embeddings.
#     Used before grounding gates in node_qa.
#     """
#     if not text or len(text.strip()) < 2:
#         return True
#     t = text.lower().strip()
#     return any(p in t for p in _REFUSAL_PHRASES)


# def _is_refusal_semantic(text: str) -> bool:
#     """Semantic refusal check using embeddings. Used in final decision only."""
#     if not text or len(text.strip()) < 2:
#         return True
#     sim = semantic_similarity(text, [_REFUSAL_ANCHOR])
#     print(f"[Refusal] sim={sim:.3f} | '{text[:50]}'")
#     return sim > 0.72


# # ============================================================
# # LANGGRAPH NODES — extract / chunk / summarize unchanged
# # ============================================================

# def node_extract(state: DocState) -> DocState:
#     extract_start = time.time()
#     text, page_count = extract_pdf_parallel(state["pdf_path"])
#     extract_time  = time.time() - extract_start
#     state["extracted_text"] = text
#     state["page_count"]     = page_count
#     state["char_count"]     = len(text)
#     state["metrics"]["extraction_time_sec"]  = round(extract_time, 2)
#     state["metrics"]["pages_processed"]      = page_count
#     state["metrics"]["characters_processed"] = len(text)
#     state["metrics"]["words_processed"]      = len(text.split())
#     return state


# def node_chunk(state: DocState) -> DocState:
#     text = state["extracted_text"]
    
#     state["query_type"] = "QA"

#     summary_chunks, rag_chunks = semantic_chunk(text)
#     state["summary_chunks"] = summary_chunks
#     state["chunks"]         = rag_chunks

#     state["metrics"]["summary_chunks"] = len(summary_chunks)
#     state["metrics"]["chunks_created"] = len(rag_chunks)
#     state["metrics"]["doc_type"]       = state["doc_type"]
#     state["metrics"]["query_type"]     = state["query_type"]
#     print(f"[Chunk] {len(summary_chunks)} summary chunks | {len(rag_chunks)} RAG chunks")
#     return state


# def node_summarize(state: DocState) -> DocState:
#     pdf_hash  = get_pdf_hash(state["pdf_path"])
#     cache_key = f"{pdf_hash}_{state['doc_type']}"

#     if cache_key in _summary_cache:
#         cached = _summary_cache[cache_key]
#         print("[Summary] ✅ Cache hit")
#         emit_event(state.get("request_id", ""), "agent_action",
#                    "⚡ Summary loaded from cache instantly!")
#         state["answer"] = cached["summary"]
#         state["metrics"].update(cached["metrics"])
#         state["metrics"]["type"] = "summary"
#         return state

#     summary_start = time.time()
#     raptor_summarize._request_id = state.get("request_id", "")
#     summary, map_time, reduce_time = raptor_summarize(
#         state["summary_chunks"], state["doc_type"]
#     )
#     summary_time = time.time() - summary_start

#     state["answer"] = summary
#     metrics_snapshot = {
#         "summary_time_sec":     round(summary_time, 2),
#         "summary_length_words": len(summary.split()),
#         "parallel_workers":     min(MAX_WORKERS, len(state["summary_chunks"])),
#         "map_time_sec":         round(map_time, 2),
#         "reduce_time_sec":      round(reduce_time, 2),
#         "llm_calls":            3,
#     }
#     state["metrics"].update(metrics_snapshot)
#     state["metrics"]["type"] = "summary"
#     _summary_cache[cache_key] = {"summary": summary, "metrics": metrics_snapshot}
#     print(f"[Summary] Done ({len(summary.split())} words)")
#     return state


# # ============================================================
# # node_qa — all fixes applied
# # ============================================================

# def node_qa(state: DocState) -> DocState:
#     qa_start_t = time.time()
#     question   = state["question"]
#     request_id = state.get("request_id", "")
#     all_chunks = state["chunks"]

#     # Raw string list — used for navigational scanning
#     # Must NOT be cleaned: clean_chunk_text strips the numbering we need
#     all_raw = [
#         d if isinstance(d, str) else d.page_content
#         for d in all_chunks
#     ]

#     pdf_hash    = get_pdf_hash(state["pdf_path"])
#     faiss_index = build_faiss_index(all_chunks, pdf_hash)

#     print(f"[QA] START | {len(all_chunks)} chunks")

#     recall_score      = 0.0
#     retrieval_score   = 0.0
#     context_precision = 0.0
#     grounding         = 0.0
#     llm_calls         = 0
#     model_used        = "llama"
#     confidence        = 0.0
#     decision_type     = "accepted"
#     retrieved         = []
#     retrieved_texts   = []
#     answer            = ""

#     # ============================================================
#     # STEP 1: RETRIEVE
#     # ============================================================
#     retrieved = multi_query_retrieve(
#         question, faiss_index,
#         k=20,
#         all_chunks=all_chunks,
#         query_type="FACTUAL_QA"
#     )

#     # ============================================================
#     # STEP 2: RERANK
#     # ============================================================
#     retrieved, reranker_top, _ = rerank_docs(
#         question, retrieved,
#         top_k=8,
#         apply_pruning=True
#     )

#     retrieved = protect_exact_matches(
#         question, retrieved, all_chunks, top_k=8
#     )

#     retrieved_texts = [d.page_content for d in retrieved]

#     if reranker_top < -5:
#         print("[Guard] ⚠️ Weak reranker score — continuing")

#     # ============================================================
#     # STEP 3: METRICS
#     # ============================================================
#     retrieval_score   = compute_retrieval_score(question, retrieved)
#     context_precision = compute_context_precision(question, retrieved)
#     recall_score      = compute_recall_at_k(
#         question, retrieved, all_chunks, k=len(retrieved)
#     )

#     # ============================================================
#     # CHANGE B — NUMERIC INTENT DETECTION
#     # Detection: retrieved_texts (fast signal)
#     # Extraction: all_raw (full document — guarantees title found)
#     # ============================================================
#     numeric_intent = _detect_numeric_intent(question, retrieved_texts)

#     if numeric_intent == "NAVIGATIONAL":
#         print(f"[Routing] Numeric intent → NAVIGATIONAL")
#         title = _navigate_full_chunks(question, all_raw)
#         if title:
#             print(f"[Navigate] Extracted title: '{title}'")
#             answer        = title
#             model_used    = "navigational"
#             decision_type = "navigational"
#             state["query_type"] = "NAVIGATIONAL"
#             grounding  = compute_answer_grounding(answer, retrieved_texts, question)
#             confidence = round(grounding / 100, 3)
#             qa_time    = time.time() - qa_start_t
#             state["answer"] = answer
#             _write_metrics(state, model_used, decision_type, grounding,
#                            confidence, retrieval_score, context_precision,
#                            recall_score, llm_calls, retrieved, qa_time)
#             return state
#         print(f"[Navigate] Extraction failed — falling through to LLM")

#     elif numeric_intent == "POSITIONAL":
#         print(f"[Routing] Numeric intent → POSITIONAL")
#         pos_answer = _positional_extract(question, retrieved_texts)
#         if pos_answer:
#             print(f"[Positional] Extracted: '{pos_answer}'")
#             answer        = pos_answer
#             model_used    = "positional"
#             decision_type = "positional"
#             state["query_type"] = "POSITIONAL"
#             grounding  = compute_answer_grounding(answer, retrieved_texts, question)
#             confidence = round(grounding / 100, 3)
#             qa_time    = time.time() - qa_start_t
#             state["answer"] = answer
#             _write_metrics(state, model_used, decision_type, grounding,
#                            confidence, retrieval_score, context_precision,
#                            recall_score, llm_calls, retrieved, qa_time)
#             return state
#         print(f"[Positional] Extraction failed — falling through to LLM")

#     # ── Normal routing ─────────────────────────────────────────
#     query_type = classify_from_context(question, retrieved_texts)
#     state["query_type"] = query_type
#     print(f"[Routing] Context-based → {query_type}")

#     # ============================================================
#     # STEP 4: ANSWER GENERATION
#     # ============================================================

#     if query_type == "FULL_SUMMARY":
#         retrieved_texts = clean_context_for_llm(retrieved_texts)
#         ranked = reorder_by_question(question, retrieved_texts)
#         context = "\n\n---\n\n".join(
#             clean_chunk_text(c) for c in ranked[:8]
#         )
#         structured_context = "\n".join(
#             f"[CONTEXT CHUNK]\n{c}" for c in context.split("\n\n---\n\n")
#         )
#         emit_event(request_id, "stream_start", "✍️ Generating answer...")
#         answer, _ = call_llama_streaming(
#             QA_PROMPT.format(context=structured_context[:2500], question=question),
#             request_id=request_id,
#             temperature=0.0
#         )
#         answer     = clean_artifacts(answer).strip()
#         model_used = "llama_summary"
#         llm_calls  = 1

#     elif query_type == "MULTIPART_QA":
#         retrieved_texts = clean_context_for_llm(retrieved_texts)
#         ranked = reorder_by_question(question, retrieved_texts)
#         context = "\n\n---\n\n".join(
#             clean_chunk_text(c) for c in ranked[:8]
#         )
#         emit_event(request_id, "agent_start",
#                    f"🤖 MULTIPART | {len(retrieved)} chunks")
#         emit_event(request_id, "stream_start", "✍️ Generating answer...")
#         answer, _ = call_llama_streaming(
#             QA_PROMPT.format(context=context[:2500], question=question),
#             request_id=request_id,
#             temperature=0.0
#         )
#         answer     = clean_artifacts(answer).strip()
#         model_used = "llama_multipart"
#         llm_calls  = 1

#     else:
    
        
#         # ── LOW RECALL → EXPAND RETRIEVAL ────────────────────
#         if recall_score < 25:
#             print("[QA] ⚠️ Low recall → expanding retrieval")
#             retrieved = multi_query_retrieve(
#                 question, faiss_index,
#                 k=30,
#                 all_chunks=all_chunks,
#                 query_type=query_type
#             )
#             retrieved_texts = [d.page_content for d in retrieved]

#         # ── NO CONTEXT GUARD ──────────────────────────────────
#         if not retrieved_texts:
#             print("[QA] ❌ No context → NOT FOUND")
#             state["answer"] = "This information is not present in the document."
#             _write_metrics(state, "not_found", "no_context",
#                         0.0, 0.0, retrieval_score, context_precision,
#                         recall_score, llm_calls, retrieved,
#                         time.time() - qa_start_t)
#             return state

#         # ── ReAct ─────────────────────────────────────────────
#         react_ans, model_used, _, _, _, _, react_calls, retrieved_texts = react_agent(
#             question, faiss_index, query_type, all_chunks, request_id
#         )
#         llm_calls += react_calls

#         # ── Clean answer ──────────────────────────────────────
#         react_ans = clean_artifacts(react_ans).strip().strip('"').strip("'")
#         react_ans = clean_reasoning_answer(react_ans, question)
#         print(f"[QA] ReAct: '{react_ans[:60]}'")

#         # ── Short answer bypass ───────────────────────────────
#         words = react_ans.split()
#         content_words = [w for w in words if len(w) > 2 and not w.isdigit()]

#         if (
#             len(words) <= 3
#             and len(content_words) >= 1
#             and not _is_refusal_answer(react_ans)
#         ):
#             print("[QA] ⚡ Short answer accepted")
#             grounding  = compute_answer_grounding(react_ans, retrieved_texts, question)
#             confidence = grounding / 100
#             qa_time    = time.time() - qa_start_t
#             state["answer"] = react_ans
#             _write_metrics(state, "llama_react", "short_answer",
#                         grounding, confidence, retrieval_score,
#                         context_precision, recall_score, llm_calls,
#                         retrieved, qa_time)
#             return state

#         # ── Refusal handling ──────────────────────────────────
#         if _is_refusal_answer(react_ans):
#             print("[QA] ⚠️ Refusal detected")
#             qa_time = time.time() - qa_start_t
#             state["answer"] = "This information is not present in the document."
#             _write_metrics(state, "not_found", "not_found",
#                         0.0, 0.0, retrieval_score, context_precision,
#                         recall_score, llm_calls, retrieved, qa_time)
#             return state

#         # ── Grounding + validation ────────────────────────────
#         grounding_score = compute_answer_grounding(react_ans, retrieved_texts, question)
#         short_answer    = len(react_ans.split()) <= 2

#         if recall_score < 30 and grounding_score < 30:
#             print("[QA] ❌ Weak retrieval → forcing NOT FOUND")
#             answer        = "This information is not present in the document."
#             model_used    = "not_found"
#             decision_type = "weak_retrieval"

#         elif grounding_score < 30 and not short_answer:
#             print("[QA] ❌ Not grounded → reject")
#             answer        = "This information is not present in the document."
#             decision_type = "low_grounding"

#         else:
#             answer        = react_ans
#             decision_type = "accepted"

#         # ── normalize_answer ──────────────────────────────────
#         answer = normalize_answer(answer)
#     # ============================================================
#     # STEP 5: METRICS
#     # ============================================================
#     grounding  = compute_answer_grounding(answer, retrieved_texts, question)
#     confidence = round(grounding / 100, 3)
#     qa_time    = time.time() - qa_start_t

#     print(f"[QA] Done in {qa_time:.1f}s | model={model_used} | "
#           f"grounding={grounding:.1f}% | recall={recall_score:.1f}% | "
#           f"confidence={confidence:.3f} | decision={decision_type}")

#     if not answer.strip():
#         answer = "Could not find a relevant answer in the PDF."
#     else:
#         answer = answer

#     state["answer"] = answer
#     if answer and not _is_refusal_answer(answer):
#         answer = normalize_answer(answer)
#     state["answer"] = answer
#     _write_metrics(state, model_used, decision_type, grounding,
#                    confidence, retrieval_score, context_precision,
#                    recall_score, llm_calls, retrieved, qa_time)
#     return state


# # ============================================================
# # node_validate — unchanged from original
# # ============================================================

# def node_validate(state: DocState) -> DocState:
#     answer = state["answer"]
#     retry  = state.get("retry_count", 0)

#     if len(answer.strip()) < 3 and retry < 2:
#         state["retry_count"] = retry + 1
#         state["answer"]      = ""
#         return state

#     total_time    = time.time() - state["start_time"]
#     output_words  = len(answer.split())
#     output_tokens = output_words * 1.3
#     extract_time  = state["metrics"].get("extraction_time_sec", 0)
#     llm_time      = max(total_time - extract_time, 1)
#     tps           = round(output_tokens / llm_time, 2) if llm_time > 0 else 0
#     m             = state["metrics"]

#     m["response_time_sec"]    = round(total_time, 2)
#     m["extraction_time_sec"]  = m.get("extraction_time_sec", 0)
#     m["pages_processed"]      = state.get("page_count", 0)
#     m["characters_processed"] = state.get("char_count", 0)
#     m["words_processed"]      = len(state.get("extracted_text", "").split())

#     if m.get("type") == "summary":
#         m["summary_time_sec"]     = m.get("summary_time_sec", 0)
#         m["summary_length_words"] = len(answer.split())
#     if m.get("type") == "qa":
#         m["qa_time_sec"]      = m.get("qa_time_sec", 0)
#         m["confidence_score"] = m.get("confidence_score", 0)

#     m["ttft_sec"]          = round(total_time, 2)
#     m["e2e_latency_sec"]   = round(total_time, 2)
#     m["tps"]               = tps
#     m["doc_type"]          = state.get("doc_type", "general")
#     m["query_type"]        = state.get("query_type", "")
#     m["chunks_created"]    = m.get("chunks_created", 0)
#     m["retry_count"]       = retry
#     m["model_used"]        = m.get("model_used", "llama_react")
#     m["llm_calls"]         = m.get("llm_calls", 0)
#     m["retrieval_score"]   = m.get("retrieval_score", 0)
#     m["context_precision"] = m.get("context_precision", 0)
#     m["answer_grounding"]  = m.get("answer_grounding", 0)
#     m["recall_at_k"]       = m.get("recall_at_k", 0)

#     if m.get("type") == "summary":
#         m["parallel_workers"] = m.get("parallel_workers", 0)
#         m["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
#         m["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)
#     if m.get("type") == "qa":
#         m["chunks_retrieved"] = m.get("chunks_retrieved", 0)
#         m["decision_type"]    = m.get("decision_type", "accepted")
#         m["confidence_raw"]   = m.get("confidence_raw", 0.0)

#     state["metrics"] = m
#     return state


# # ============================================================
# # SHARED METRICS WRITER — called from every exit path in node_qa
# # ============================================================

# def _write_metrics(state, model_used, decision_type, grounding,
#                    confidence, retrieval_score, context_precision,
#                    recall_score, llm_calls, retrieved, qa_time):
#     m = state.setdefault("metrics", {})
#     m["qa_time_sec"]       = round(qa_time, 2)
#     m["confidence_score"]  = round(confidence * 100, 2)
#     m["retrieval_score"]   = retrieval_score
#     m["context_precision"] = context_precision
#     m["answer_grounding"]  = grounding
#     m["recall_at_k"]       = recall_score
#     m["llm_calls"]         = llm_calls
#     m["model_used"]        = model_used
#     m["chunks_retrieved"]  = len(retrieved)
#     m["type"]              = "qa"
#     m["decision_type"]     = decision_type
#     m["confidence_raw"]    = round(confidence, 4)





































# worked for 70 70 but gotta do some changes 
# import time
# import re
# import spacy
# _nlp = spacy.load("en_core_web_lg")


# from docmind_rag.config.settings import (
#     MAX_WORKERS, FACTUAL_TOP_K, MULTIPART_TOP_K
# )

# from docmind_rag.core.state import DocState
# from docmind_rag.core.prompts import QA_PROMPT
# from docmind_rag.models.llm import call_llama_streaming
# from docmind_rag.models.embeddings import build_faiss_index
# from docmind_rag.models.reranker import rerank_docs, protect_exact_matches
# from docmind_rag.services.extraction import extract_pdf_parallel, _extraction_cache
# from docmind_rag.services.chunking import semantic_chunk, raptor_summarize, _summary_cache
# from docmind_rag.services.retrieval import multi_query_retrieve
# from docmind_rag.services.agent import react_agent, roberta_qa
# from docmind_rag.events.events import emit_event
# from docmind_rag.utils.helpers import get_pdf_hash, clean_artifacts, clean_chunk_text
# from docmind_rag.utils.text import (
#     classify_from_context,
#     reorder_by_question
# )
# from docmind_rag.utils.metrics import (
#     compute_answer_grounding,
#     compute_retrieval_score,
#     compute_context_precision,
#     compute_recall_at_k,
#     semantic_similarity,
# )


# # ============================================================
# # LOCAL HELPERS (unchanged from original)
# # ============================================================

# def clean_context_for_llm(chunks):
#     cleaned_chunks = []
#     for chunk in chunks:
#         lines = chunk.split("\n")
#         filtered_lines = []
#         for line in lines:
#             line_strip = line.strip()
#             is_question_like = (
#                 line_strip.endswith("?") or
#                 (len(line_strip.split()) < 15 and "?" in line_strip)
#             )
#             if is_question_like:
#                 continue
#             filtered_lines.append(line)
#         cleaned_chunks.append("\n".join(filtered_lines))
#     return cleaned_chunks


# def is_grounded(answer, retrieved_texts, threshold=0.6):
#     if not answer:
#         return False
#     answer_words = set(answer.lower().split())
#     if not answer_words:
#         return False
#     best_overlap = 0
#     for chunk in retrieved_texts:
#         chunk_words = set(chunk.lower().split())
#         overlap = len(answer_words & chunk_words) / len(answer_words)
#         best_overlap = max(best_overlap, overlap)
#     return best_overlap >= threshold


# # ============================================================
# # CHANGE A — Structural question classifier (no embeddings)
# # ============================================================

# _LOOKUP_STARTS = (
#     "what is", "what are", "what was", "what were",
#     "who is", "who was", "who are",
#     "when did", "when was", "when were",
#     "how many", "how much",
#     "name the", "name of",
# )

# # _REASONING_WORDS = {
# #     "because", "therefore", "however", "although", "since",
# #     "thus", "hence", "consequently", "furthermore", "moreover",
# #     "nevertheless", "regardless", "whereas", "while", "despite",
# # }


# def _is_simple_lookup(question: str) -> bool:
#     """
#     True only for short, structurally simple factual questions.
#     No embeddings — pure string logic. Generic across all PDFs.
#     """
#     q     = question.strip().lower()
#     words = q.split()
#     if len(words) > 12:
#         return False
#     if not any(q.startswith(s) for s in _LOOKUP_STARTS):
#         return False
#     return True


# # def _span_is_clean(span: str) -> bool:
# #     """Span is clean if short and contains no reasoning language."""
# #     words = span.strip().split()
# #     if len(words) > 6:
# #         return False
# #     if any(w.lower() in _REASONING_WORDS for w in words):
# #         return False
# #     return True


# def _select_best_span_structural(question: str, chunks: list) -> str:
#     """
#     Select best short span using word overlap.
#     Only called after _is_simple_lookup passes.
#     """
#     q_words    = set(question.lower().split())
#     best_span  = ""
#     best_score = 0

#     for chunk in chunks:
#         for line in chunk.split("\n"):
#             line = line.strip()
#             if not line:
#                 continue
#             wc = len(line.split())
#             if wc < 2 or wc > 6:
#                 continue
#             s_words = set(line.lower().split())
#             overlap = len(q_words & s_words)
#             if overlap > best_score:
#                 best_score = overlap
#                 best_span  = line
#             elif overlap == best_score and len(line.split()) > len(best_span.split()):
#                 best_span = line

#     return best_span


# # ============================================================
# # CHANGE B — Structure-aware numeric intent detection
# # ============================================================

# def _extract_numbers(text: str) -> list:
#     return re.findall(r'\b\d+\b', text)


# def _detect_numeric_intent(question: str, retrieved_texts: list) -> str:
#     """
#     Detect whether a numeric question is NAVIGATIONAL or POSITIONAL.
#     Uses retrieved_texts as fast signal only.
#     Returns: "NAVIGATIONAL" | "POSITIONAL" | "NONE"
#     """
#     numbers = _extract_numbers(question)
#     if not numbers:
#         return "NONE"

#     q_words = set(question.lower().split())

#     for chunk in retrieved_texts:
#         lines = [l.strip() for l in chunk.split("\n") if l.strip()]
#         if not lines:
#             continue
#         first_line = lines[0].lower()
#         line_words = set(first_line.split())
#         overlap    = len(q_words & line_words)

#         if any(re.search(rf'\b{re.escape(num)}\b', first_line) for num in numbers):
#             if overlap >= 1 and len(first_line.split()) <= 12:
#                 return "NAVIGATIONAL"

#     for chunk in retrieved_texts:
#         lines       = [l.strip() for l in chunk.split("\n") if l.strip()]
#         short_lines = [l for l in lines if len(l.split()) <= 12]
#         if len(short_lines) >= 3:
#             first = lines[0].lower() if lines else ""
#             if not any(
#                 re.search(rf'\b{re.escape(num)}\b', first) for num in numbers
#             ):
#                 return "POSITIONAL"

#     return "NONE"


# def _navigate_full_chunks(question: str, all_raw_chunks: list) -> str:
#     """
#     NAVIGATIONAL extraction scanning the FULL raw chunk set.
#     Must use raw chunks — clean_chunk_text strips the numbering we need.
#     Generic — works for any numbered section in any document.
#     """
#     numbers = _extract_numbers(question)
#     if not numbers:
#         return ""

#     q_words    = set(question.lower().split())
#     best_line  = ""
#     best_score = 0

#     for chunk in all_raw_chunks:
#         lines = [l.strip() for l in chunk.split("\n") if l.strip()]
#         if not lines:
#             continue

#         first_line  = lines[0]
#         first_lower = first_line.lower()

#         for num in numbers:
#             if not re.search(rf'\b{re.escape(num)}\b', first_lower):
#                 continue
#             line_words = set(first_lower.split())
#             overlap    = len(q_words & line_words)
#             if overlap >= 1 and len(first_line.split()) <= 14:
#                 score = overlap + (5 if len(first_line.split()) <= 8 else 0)
#                 if score > best_score:
#                     best_score = score
#                     best_line  = first_line

#     if not best_line:
#         return ""

#     # Extract title: remove page markers, numbers, separators
#     title = best_line
#     title = re.sub(r'^\s*\d+\s*[-—–]+\s*[—–-]?\s*\d*\s*', '', title)
#     title = re.sub(r'\b\d+\b', '', title)
#     title = re.sub(r'[:\-–—]', ' ', title)
#     title = re.sub(r'\s{2,}', ' ', title)
#     title = title.strip()

#     return title if title else ""


# def _positional_extract(question: str, retrieved_texts: list) -> str:
#     """
#     POSITIONAL extraction: find Nth item in a list structure.
#     Generic — works for any numbered list in any document.
#     """
#     numbers = _extract_numbers(question)
#     if not numbers:
#         return ""

#     try:
#         idx = int(numbers[0]) - 1
#     except ValueError:
#         return ""

#     if idx < 0:
#         return ""

#     best_chunk       = ""
#     best_short_count = 0

#     for chunk in retrieved_texts:
#         lines       = [l.strip() for l in chunk.split("\n") if l.strip()]
#         short_lines = [l for l in lines if len(l.split()) <= 12]
#         if len(short_lines) > best_short_count:
#             best_short_count = len(short_lines)
#             best_chunk       = chunk

#     if not best_chunk:
#         return ""

#     lines       = [l.strip() for l in best_chunk.split("\n") if l.strip()]
#     short_lines = [l for l in lines if len(l.split()) <= 12]

#     if idx < len(short_lines):
#         answer = re.sub(r'^\s*\d+[\.\)]\s*', '', short_lines[idx]).strip()
#         return answer

#     return ""


# # ============================================================
# # REFUSAL HELPERS
# # ============================================================

# _REFUSAL_PHRASES = [
#     "not present", "not available", "not found",
#     "not in the document", "not present in the document",
#     "information is not", "cannot find", "no information",
# ]

# _REFUSAL_ANCHOR = "this information is not available in the provided context"


# def _is_refusal_answer(text: str) -> bool:
#     """
#     Lightweight string-matching refusal check. No LLM, no embeddings.
#     Used before grounding gates in node_qa.
#     """
#     if not text or len(text.strip()) < 2:
#         return True
#     t = text.lower().strip()
#     return any(p in t for p in _REFUSAL_PHRASES)


# def _is_refusal_semantic(text: str) -> bool:
#     """Semantic refusal check using embeddings. Used in final decision only."""
#     if not text or len(text.strip()) < 2:
#         return True
#     sim = semantic_similarity(text, [_REFUSAL_ANCHOR])
#     print(f"[Refusal] sim={sim:.3f} | '{text[:50]}'")
#     return sim > 0.72


# # ============================================================
# # LANGGRAPH NODES — extract / chunk / summarize unchanged
# # ============================================================

# def node_extract(state: DocState) -> DocState:
#     extract_start = time.time()
#     text, page_count = extract_pdf_parallel(state["pdf_path"])
#     extract_time  = time.time() - extract_start
#     state["extracted_text"] = text
#     state["page_count"]     = page_count
#     state["char_count"]     = len(text)
#     state["metrics"]["extraction_time_sec"]  = round(extract_time, 2)
#     state["metrics"]["pages_processed"]      = page_count
#     state["metrics"]["characters_processed"] = len(text)
#     state["metrics"]["words_processed"]      = len(text.split())
#     return state


# def node_chunk(state: DocState) -> DocState:
#     text = state["extracted_text"]
    
#     state["query_type"] = "QA"

#     summary_chunks, rag_chunks = semantic_chunk(text)
#     state["summary_chunks"] = summary_chunks
#     state["chunks"]         = rag_chunks

#     state["metrics"]["summary_chunks"] = len(summary_chunks)
#     state["metrics"]["chunks_created"] = len(rag_chunks)
#     state["metrics"]["doc_type"]       = state["doc_type"]
#     state["metrics"]["query_type"]     = state["query_type"]
#     print(f"[Chunk] {len(summary_chunks)} summary chunks | {len(rag_chunks)} RAG chunks")
#     return state


# def node_summarize(state: DocState) -> DocState:
#     pdf_hash  = get_pdf_hash(state["pdf_path"])
#     cache_key = f"{pdf_hash}_{state['doc_type']}"

#     if cache_key in _summary_cache:
#         cached = _summary_cache[cache_key]
#         print("[Summary] ✅ Cache hit")
#         emit_event(state.get("request_id", ""), "agent_action",
#                    "⚡ Summary loaded from cache instantly!")
#         state["answer"] = cached["summary"]
#         state["metrics"].update(cached["metrics"])
#         state["metrics"]["type"] = "summary"
#         return state

#     summary_start = time.time()
#     raptor_summarize._request_id = state.get("request_id", "")
#     summary, map_time, reduce_time = raptor_summarize(
#         state["summary_chunks"], state["doc_type"]
#     )
#     summary_time = time.time() - summary_start

#     state["answer"] = summary
#     metrics_snapshot = {
#         "summary_time_sec":     round(summary_time, 2),
#         "summary_length_words": len(summary.split()),
#         "parallel_workers":     min(MAX_WORKERS, len(state["summary_chunks"])),
#         "map_time_sec":         round(map_time, 2),
#         "reduce_time_sec":      round(reduce_time, 2),
#         "llm_calls":            3,
#     }
#     state["metrics"].update(metrics_snapshot)
#     state["metrics"]["type"] = "summary"
#     _summary_cache[cache_key] = {"summary": summary, "metrics": metrics_snapshot}
#     print(f"[Summary] Done ({len(summary.split())} words)")
#     return state


# # ============================================================
# # node_qa — all fixes applied
# # ============================================================

# def node_qa(state: DocState) -> DocState:
#     qa_start_t = time.time()
#     question   = state["question"]
#     request_id = state.get("request_id", "")
#     all_chunks = state["chunks"]

#     # Raw string list — used for navigational scanning
#     # Must NOT be cleaned: clean_chunk_text strips the numbering we need
#     all_raw = [
#         d if isinstance(d, str) else d.page_content
#         for d in all_chunks
#     ]

#     pdf_hash    = get_pdf_hash(state["pdf_path"])
#     faiss_index = build_faiss_index(all_chunks, pdf_hash)

#     print(f"[QA] START | {len(all_chunks)} chunks")

#     recall_score      = 0.0
#     retrieval_score   = 0.0
#     context_precision = 0.0
#     grounding         = 0.0
#     llm_calls         = 0
#     model_used        = "llama"
#     confidence        = 0.0
#     decision_type     = "accepted"
#     retrieved         = []
#     retrieved_texts   = []
#     answer            = ""

#     # ============================================================
#     # STEP 1: RETRIEVE
#     # ============================================================
#     retrieved = multi_query_retrieve(
#         question, faiss_index,
#         k=20,
#         all_chunks=all_chunks,
#         query_type="FACTUAL_QA"
#     )

#     # ============================================================
#     # STEP 2: RERANK
#     # ============================================================
#     retrieved, reranker_top, _ = rerank_docs(
#         question, retrieved,
#         top_k=8,
#         apply_pruning=True
#     )

#     retrieved = protect_exact_matches(
#         question, retrieved, all_chunks, top_k=8
#     )

#     retrieved_texts = [d.page_content for d in retrieved]

#     if reranker_top < -5:
#         print("[Guard] ⚠️ Weak reranker score — continuing")

#     # ============================================================
#     # STEP 3: METRICS
#     # ============================================================
#     retrieval_score   = compute_retrieval_score(question, retrieved)
#     context_precision = compute_context_precision(question, retrieved)
#     recall_score      = compute_recall_at_k(
#         question, retrieved, all_chunks, k=len(retrieved)
#     )

#     # ============================================================
#     # CHANGE B — NUMERIC INTENT DETECTION
#     # Detection: retrieved_texts (fast signal)
#     # Extraction: all_raw (full document — guarantees title found)
#     # ============================================================
#     numeric_intent = _detect_numeric_intent(question, retrieved_texts)

#     if numeric_intent == "NAVIGATIONAL":
#         print(f"[Routing] Numeric intent → NAVIGATIONAL")
#         title = _navigate_full_chunks(question, all_raw)
#         if title:
#             print(f"[Navigate] Extracted title: '{title}'")
#             answer        = title
#             model_used    = "navigational"
#             decision_type = "navigational"
#             state["query_type"] = "NAVIGATIONAL"
#             grounding  = compute_answer_grounding(answer, retrieved_texts, question)
#             confidence = round(grounding / 100, 3)
#             qa_time    = time.time() - qa_start_t
#             state["answer"] = answer
#             _write_metrics(state, model_used, decision_type, grounding,
#                            confidence, retrieval_score, context_precision,
#                            recall_score, llm_calls, retrieved, qa_time)
#             return state
#         print(f"[Navigate] Extraction failed — falling through to LLM")

#     elif numeric_intent == "POSITIONAL":
#         print(f"[Routing] Numeric intent → POSITIONAL")
#         pos_answer = _positional_extract(question, retrieved_texts)
#         if pos_answer:
#             print(f"[Positional] Extracted: '{pos_answer}'")
#             answer        = pos_answer
#             model_used    = "positional"
#             decision_type = "positional"
#             state["query_type"] = "POSITIONAL"
#             grounding  = compute_answer_grounding(answer, retrieved_texts, question)
#             confidence = round(grounding / 100, 3)
#             qa_time    = time.time() - qa_start_t
#             state["answer"] = answer
#             _write_metrics(state, model_used, decision_type, grounding,
#                            confidence, retrieval_score, context_precision,
#                            recall_score, llm_calls, retrieved, qa_time)
#             return state
#         print(f"[Positional] Extraction failed — falling through to LLM")

#     # ── Normal routing ─────────────────────────────────────────
#     query_type = classify_from_context(question, retrieved_texts)
#     state["query_type"] = query_type
#     print(f"[Routing] Context-based → {query_type}")

#     # ============================================================
#     # STEP 4: ANSWER GENERATION
#     # ============================================================

#     if query_type == "FULL_SUMMARY":
#         retrieved_texts = clean_context_for_llm(retrieved_texts)
#         ranked = reorder_by_question(question, retrieved_texts)
#         context = "\n\n---\n\n".join(
#             clean_chunk_text(c) for c in ranked[:8]
#         )
#         structured_context = "\n".join(
#             f"[CONTEXT CHUNK]\n{c}" for c in context.split("\n\n---\n\n")
#         )
#         emit_event(request_id, "stream_start", "✍️ Generating answer...")
#         answer, _ = call_llama_streaming(
#             QA_PROMPT.format(context=structured_context[:2500], question=question),
#             request_id=request_id,
#             temperature=0.0
#         )
#         answer     = clean_artifacts(answer).strip()
#         model_used = "llama_summary"
#         llm_calls  = 1

#     elif query_type == "MULTIPART_QA":
#         retrieved_texts = clean_context_for_llm(retrieved_texts)
#         ranked = reorder_by_question(question, retrieved_texts)
#         context = "\n\n---\n\n".join(
#             clean_chunk_text(c) for c in ranked[:8]
#         )
#         emit_event(request_id, "agent_start",
#                    f"🤖 MULTIPART | {len(retrieved)} chunks")
#         emit_event(request_id, "stream_start", "✍️ Generating answer...")
#         answer, _ = call_llama_streaming(
#             QA_PROMPT.format(context=context[:2500], question=question),
#             request_id=request_id,
#             temperature=0.0
#         )
#         answer     = clean_artifacts(answer).strip()
#         model_used = "llama_multipart"
#         llm_calls  = 1

#     else:
#         # FACTUAL_QA / VERIFICATION_QA

#         # ── CHANGE A — Structural span gate ──────────────────
#         # Only fires for genuinely simple lookup questions.
#         # No embeddings, pure string logic.
#         # if _is_simple_lookup(question):
#         #     candidate_span = _select_best_span_structural(question, retrieved_texts)
#         #     if candidate_span and _span_is_clean(candidate_span):
#         #         print(f"[QA] ⚡ Span gate passed: '{candidate_span}'")
#         #         answer        = candidate_span
#         #         model_used    = "span_extraction"
#         #         decision_type = "span_direct"
#         #         answer        = answer
#         #         grounding     = compute_answer_grounding(answer, retrieved_texts, question)
#         #         confidence    = round(grounding / 100, 3)
#         #         qa_time       = time.time() - qa_start_t
#         #         state["answer"] = answer
#         #         _write_metrics(state, model_used, decision_type, grounding,
#         #                        confidence, retrieval_score, context_precision,
#         #                        recall_score, llm_calls, retrieved, qa_time)
#         #         return state
#         # ============================================================
#         # ⚡ SAFE SPAN EXTRACTION (GENERIC)
#         # ============================================================

#         # use_span = False

#         # if _is_simple_lookup(question):
#         #     candidate_span = _select_best_span_structural(question, retrieved_texts)

#         #     if candidate_span:
#         #         word_count = len(candidate_span.split())

#         #         # ✅ PURELY STRUCTURAL CONDITIONS
#         #         if (
#         #             1 <= word_count <= 5                      # short answer
#         #             and _span_is_clean(candidate_span)        # no junk chars
#         #             and candidate_span.lower() not in question.lower()  # not echo
#         #             and not candidate_span.endswith(":")      # not heading
#         #             and (candidate_span[0].isupper() or candidate_span.isdigit())
#         #             # and candidate_span[0].isupper() or candidate_span.isdigit()
#         #         ):
#         #             use_span = True

#         # if use_span:
#         #     print(f"[QA] ⚡ Span accepted: '{candidate_span}'")

#         #     answer        = candidate_span
#         #     model_used    = "span_extraction"
#         #     decision_type = "span_direct"

#         #     state["answer"] = answer
#         #     return state
#         # Span extraction disabled — rely on ReAct + grounding
#         # ── LOW RECALL → EXPAND RETRIEVAL ────────────────────
#         if recall_score < 25:
#             print("[QA] ⚠️ Low recall → expanding retrieval")
#             retrieved = multi_query_retrieve(
#                 question, faiss_index,
#                 k=30,
#                 all_chunks=all_chunks,
#                 query_type=query_type
#             )
#             retrieved_texts = [d.page_content for d in retrieved]

#         # ── NO CONTEXT GUARD ──────────────────────────────────
#         if not retrieved_texts:
#             print("[QA] ❌ No context → NOT FOUND")
#             state["answer"] = "This information is not present in the document."
#             state["metrics"]["decision_type"] = "no_context"
#             return state

#         # ── ReAct ─────────────────────────────────────────────
#         react_ans, model_used, _, _, _, _, react_calls, retrieved_texts = react_agent(
#             question, faiss_index, query_type, all_chunks, request_id
#         )
#         llm_calls += react_calls

#         react_ans = clean_artifacts(react_ans).strip().strip('"').strip("'")
#         print(f"[QA] ReAct: '{react_ans[:60]}'")
#         if len(react_ans.split()) <= 2 and not _is_refusal_answer(react_ans):
#             print("[QA] ⚡ Short answer accepted")
#             state["answer"] = react_ans
#             return state
#         # Drift filter
#         bad_patterns = [
#             "since the question",
#             "the question is about",
#             "i will extract",
#             "the answer is based on",
#         ]
#         if any(p in react_ans.lower() for p in bad_patterns):
#             print("[QA] ❌ Drift detected → fallback")
#             react_ans = ""

#         # ── CHANGE C — Refusal bypass BEFORE grounding gate ──
#         # Correctly-detected refusals skip all grounding checks.
#         if _is_refusal_answer(react_ans):
#             print("[QA] ✅ Refusal detected — accepting without grounding gate")
#             answer        = "This information is not present in the document."
#             model_used    = "not_found"
#             decision_type = "not_found"

#         else:
#             grounding_score = compute_answer_grounding(
#                 react_ans, retrieved_texts, question
#             )
#             short_answer = len(react_ans.split()) <= 2

#             # ── RECALL GUARD — hallucination firewall ─────────
#             # Only fires for non-refusal answers with weak retrieval.
#             if recall_score < 30 and grounding_score < 30:
#                 print("[QA] ❌ Weak retrieval → forcing NOT FOUND")
#                 answer        = "This information is not present in the document."
#                 model_used    = "not_found"
#                 decision_type = "weak_retrieval"

#             elif grounding_score < 30 and not short_answer:
#                 print("[QA] ❌ Not grounded → reject")
#                 answer        = "This information is not present in the document."
#                 decision_type = "low_grounding"

#             elif react_ans and not _is_refusal_semantic(react_ans):
#                 answer        = react_ans
#                 decision_type = "accepted"

#             else:
#                 answer        = "This information is not present in the document."
#                 decision_type = "not_found"

#         answer = answer

#     # ============================================================
#     # STEP 5: METRICS
#     # ============================================================
#     grounding  = compute_answer_grounding(answer, retrieved_texts, question)
#     confidence = round(grounding / 100, 3)
#     qa_time    = time.time() - qa_start_t

#     print(f"[QA] Done in {qa_time:.1f}s | model={model_used} | "
#           f"grounding={grounding:.1f}% | recall={recall_score:.1f}% | "
#           f"confidence={confidence:.3f} | decision={decision_type}")

#     if not answer.strip():
#         answer = "Could not find a relevant answer in the PDF."
#     else:
#         answer = answer

#     state["answer"] = answer
#     _write_metrics(state, model_used, decision_type, grounding,
#                    confidence, retrieval_score, context_precision,
#                    recall_score, llm_calls, retrieved, qa_time)
#     return state


# # ============================================================
# # node_validate — unchanged from original
# # ============================================================

# def node_validate(state: DocState) -> DocState:
#     answer = state["answer"]
#     retry  = state.get("retry_count", 0)

#     if len(answer.strip()) < 3 and retry < 2:
#         state["retry_count"] = retry + 1
#         state["answer"]      = ""
#         return state

#     total_time    = time.time() - state["start_time"]
#     output_words  = len(answer.split())
#     output_tokens = output_words * 1.3
#     extract_time  = state["metrics"].get("extraction_time_sec", 0)
#     llm_time      = max(total_time - extract_time, 1)
#     tps           = round(output_tokens / llm_time, 2) if llm_time > 0 else 0
#     m             = state["metrics"]

#     m["response_time_sec"]    = round(total_time, 2)
#     m["extraction_time_sec"]  = m.get("extraction_time_sec", 0)
#     m["pages_processed"]      = state.get("page_count", 0)
#     m["characters_processed"] = state.get("char_count", 0)
#     m["words_processed"]      = len(state.get("extracted_text", "").split())

#     if m.get("type") == "summary":
#         m["summary_time_sec"]     = m.get("summary_time_sec", 0)
#         m["summary_length_words"] = len(answer.split())
#     if m.get("type") == "qa":
#         m["qa_time_sec"]      = m.get("qa_time_sec", 0)
#         m["confidence_score"] = m.get("confidence_score", 0)

#     m["ttft_sec"]          = round(total_time, 2)
#     m["e2e_latency_sec"]   = round(total_time, 2)
#     m["tps"]               = tps
#     m["doc_type"]          = state.get("doc_type", "general")
#     m["query_type"]        = state.get("query_type", "")
#     m["chunks_created"]    = m.get("chunks_created", 0)
#     m["retry_count"]       = retry
#     m["model_used"]        = m.get("model_used", "llama_react")
#     m["llm_calls"]         = m.get("llm_calls", 0)
#     m["retrieval_score"]   = m.get("retrieval_score", 0)
#     m["context_precision"] = m.get("context_precision", 0)
#     m["answer_grounding"]  = m.get("answer_grounding", 0)
#     m["recall_at_k"]       = m.get("recall_at_k", 0)

#     if m.get("type") == "summary":
#         m["parallel_workers"] = m.get("parallel_workers", 0)
#         m["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
#         m["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)
#     if m.get("type") == "qa":
#         m["chunks_retrieved"] = m.get("chunks_retrieved", 0)
#         m["decision_type"]    = m.get("decision_type", "accepted")
#         m["confidence_raw"]   = m.get("confidence_raw", 0.0)

#     state["metrics"] = m
#     return state


# # ============================================================
# # SHARED METRICS WRITER — called from every exit path in node_qa
# # ============================================================

# def _write_metrics(state, model_used, decision_type, grounding,
#                    confidence, retrieval_score, context_precision,
#                    recall_score, llm_calls, retrieved, qa_time):
#     m = state.setdefault("metrics", {})
#     m["qa_time_sec"]       = round(qa_time, 2)
#     m["confidence_score"]  = round(confidence * 100, 2)
#     m["retrieval_score"]   = retrieval_score
#     m["context_precision"] = context_precision
#     m["answer_grounding"]  = grounding
#     m["recall_at_k"]       = recall_score
#     m["llm_calls"]         = llm_calls
#     m["model_used"]        = model_used
#     m["chunks_retrieved"]  = len(retrieved)
#     m["type"]              = "qa"
#     m["decision_type"]     = decision_type
#     m["confidence_raw"]    = round(confidence, 4)

























# not at all workeed 40 %
# import time
# import re
# import spacy
# _nlp = spacy.load("en_core_web_lg")

# from docmind_rag.config.settings import (
#     MAX_WORKERS, FACTUAL_TOP_K, MULTIPART_TOP_K
# )
# from docmind_rag.utils.text import extract_named_entities,expand_answer
# from docmind_rag.core.state import DocState
# from docmind_rag.core.prompts import QA_PROMPT
# from docmind_rag.models.llm import call_llama_streaming
# from docmind_rag.models.embeddings import build_faiss_index
# from docmind_rag.models.reranker import rerank_docs, protect_exact_matches
# from docmind_rag.services.extraction import extract_pdf_parallel, _extraction_cache
# from docmind_rag.services.chunking import semantic_chunk, raptor_summarize, _summary_cache
# from docmind_rag.services.retrieval import multi_query_retrieve
# from docmind_rag.services.agent import react_agent, roberta_qa
# from docmind_rag.events.events import emit_event
# from docmind_rag.utils.helpers import get_pdf_hash, clean_artifacts, clean_chunk_text
# from docmind_rag.utils.text import (
#     detect_doc_type, classify_from_context,
#     reorder_by_question, normalize_answer
# )
# from docmind_rag.utils.metrics import (
#     compute_answer_grounding,
#     compute_retrieval_score,
#     compute_context_precision,
#     compute_recall_at_k,
#     semantic_similarity,
# )
# def clean_context_for_llm(chunks):
#     cleaned_chunks = []

#     for chunk in chunks:
#         lines = chunk.split("\n")

#         filtered_lines = []

#         for line in lines:
#             line_strip = line.strip()

#             # ✅ Generic signal of question-like content
#             is_question_like = (
#                 line_strip.endswith("?") or
#                 (len(line_strip.split()) < 15 and "?" in line_strip)
#             )

#             # ✅ Remove only standalone short question lines
#             if is_question_like:
#                 continue

#             filtered_lines.append(line)

#         cleaned_chunks.append("\n".join(filtered_lines))

#     return cleaned_chunks

# def is_grounded(answer, retrieved_texts, threshold=0.6):
#     if not answer:
#         return False

#     answer_words = set(answer.lower().split())
#     if not answer_words:
#         return False

#     best_overlap = 0

#     for chunk in retrieved_texts:
#         chunk_words = set(chunk.lower().split())
#         overlap = len(answer_words & chunk_words) / len(answer_words)
#         best_overlap = max(best_overlap, overlap)

#     return best_overlap >= threshold
# # ============================================================
# # LANGGRAPH NODES
# # ============================================================

# def node_extract(state: DocState) -> DocState:
#     extract_start = time.time()
#     text, page_count = extract_pdf_parallel(state["pdf_path"])
#     extract_time  = time.time() - extract_start
#     state["extracted_text"] = text
#     state["page_count"]     = page_count
#     state["char_count"]     = len(text)
#     state["metrics"]["extraction_time_sec"]  = round(extract_time, 2)
#     state["metrics"]["pages_processed"]      = page_count
#     state["metrics"]["characters_processed"] = len(text)
#     state["metrics"]["words_processed"]      = len(text.split())
#     return state


# def node_chunk(state: DocState) -> DocState:
#     text = state["extracted_text"]
#     state["doc_type"]   = detect_doc_type(text)
#     state["query_type"] = "QA"   # routing decided post-retrieval in node_qa

#     summary_chunks, rag_chunks = semantic_chunk(text)
#     state["summary_chunks"] = summary_chunks
#     state["chunks"]         = rag_chunks

#     state["metrics"]["summary_chunks"] = len(summary_chunks)
#     state["metrics"]["chunks_created"] = len(rag_chunks)
#     state["metrics"]["doc_type"]       = state["doc_type"]
#     state["metrics"]["query_type"]     = state["query_type"]
#     print(f"[Chunk] {len(summary_chunks)} summary chunks | {len(rag_chunks)} RAG chunks")
#     return state


# def node_summarize(state: DocState) -> DocState:
#     pdf_hash  = get_pdf_hash(state["pdf_path"])
#     cache_key = f"{pdf_hash}_{state['doc_type']}"

#     if cache_key in _summary_cache:
#         cached = _summary_cache[cache_key]
#         print("[Summary] ✅ Cache hit")
#         emit_event(state.get("request_id", ""), "agent_action",
#                    "⚡ Summary loaded from cache instantly!")
#         state["answer"] = cached["summary"]
#         state["metrics"].update(cached["metrics"])
#         state["metrics"]["type"] = "summary"
#         return state

#     summary_start = time.time()
#     raptor_summarize._request_id = state.get("request_id", "")
#     summary, map_time, reduce_time = raptor_summarize(
#         state["summary_chunks"], state["doc_type"]
#     )
#     summary_time = time.time() - summary_start

#     state["answer"] = summary
#     metrics_snapshot = {
#         "summary_time_sec":     round(summary_time, 2),
#         "summary_length_words": len(summary.split()),
#         "parallel_workers":     min(MAX_WORKERS, len(state["summary_chunks"])),
#         "map_time_sec":         round(map_time, 2),
#         "reduce_time_sec":      round(reduce_time, 2),
#         "llm_calls":            3,
#     }
#     state["metrics"].update(metrics_snapshot)
#     state["metrics"]["type"] = "summary"
#     _summary_cache[cache_key] = {"summary": summary, "metrics": metrics_snapshot}
#     print(f"[Summary] Done ({len(summary.split())} words)")
#     return state



# # def node_qa(state: DocState) -> DocState:
# #     qa_start_t = time.time()
# #     question   = state["question"]
# #     request_id = state.get("request_id", "")
# #     all_chunks = state["chunks"]

# #     pdf_hash    = get_pdf_hash(state["pdf_path"])
# #     faiss_index = build_faiss_index(all_chunks, pdf_hash)

# #     print(f"[QA] START | {len(all_chunks)} chunks")

# #     recall_score      = 0.0
# #     retrieval_score   = 0.0
# #     context_precision = 0.0
# #     grounding         = 0.0
# #     llm_calls         = 0
# #     model_used        = "llama"
# #     confidence        = 0.0
# #     decision_type     = "accepted"
# #     retrieved         = []
# #     retrieved_texts   = []
# #     answer            = ""
# #     used_fast_path = False

# #     # ── Refusal detector ─────────────────────────────────────
# #     _REFUSAL_ANCHOR = "this information is not available in the provided context"

# #     def is_refusal(text: str) -> bool:
# #         if not text or len(text.strip()) < 2:
# #             return True
# #         sim = semantic_similarity(text, [_REFUSAL_ANCHOR])
# #         print(f"[Refusal] sim={sim:.3f} | '{text[:50]}'")
# #         return sim > 0.72

# #     # ============================================================
# #     # STEP 1: RETRIEVE FIRST — always
# #     # ============================================================
# #     retrieved = multi_query_retrieve(
# #         question, faiss_index,
# #         k=20,
# #         all_chunks=all_chunks,
# #         query_type="FACTUAL_QA"   # neutral — no assumption yet
# #     )

# #     # ============================================================
# #     # STEP 2: RERANK
# #     # ============================================================
# #     retrieved, reranker_top, all_scores = rerank_docs(
# #         question, retrieved,
# #         top_k=8,
# #         apply_pruning=True
# #     )

   
# #     # # ============================================================
# #     # # STEP 3: CLASSIFY FROM CONTEXT — post-retrieval, no LLM call
# #     # # ============================================================
# #     # query_type = classify_from_context(question, retrieved_texts)
# #     retrieved = protect_exact_matches(
# #         question, retrieved, all_chunks, top_k=8
# #     )
# #     retrieved_texts = [d.page_content for d in retrieved]
# # # ✅ define ONCE (outside everything)
# #     def contains_entity(entity, text):
# #         return re.search(rf'\b{re.escape(entity)}\b', text, re.IGNORECASE)


# #     # ============================================================
# #     # # ✅ ENTITY INJECTION (MUST COME BEFORE GUARD)
# #     # # ============================================================
# #     # named_entities = extract_named_entities(question, all_chunks)

# #     # if named_entities:
# #     #     existing_fps = {d.page_content[:120].strip() for d in retrieved}
# #     #     injected = 0

# #     #     # ✅ limit chunks to avoid noise + slowdown
# #     #     for chunk in all_chunks[:50]:
# #     #         if any(contains_entity(entity, chunk) for entity in named_entities):
# #     #             fp = chunk[:120].strip()

# #     #             if fp not in existing_fps:
# #     #                 from langchain.schema import Document
# #     #                 retrieved.append(Document(
# #     #                     page_content=chunk,
# #     #                     metadata={"chunk_id": -1, "exact_score": 0}
# #     #                 ))
# #     #                 existing_fps.add(fp)
# #     #                 injected += 1

# #     #     if injected:
# #     #         print(f"[EntityInject] Injected {injected} chunks for entities: {named_entities}")


# #     # ✅ ALWAYS define (important fix)
# #     retrieved_texts = [d.page_content for d in retrieved]


# #     # ============================================================
# #     # ✅ GUARD (FIXED)
# #     # ============================================================
# #     # if reranker_top < -5:

# #     #     strong_exact = any(
# #     #         d.metadata.get("exact_score", 0) >= 3
# #     #         for d in retrieved
# #     #     )

# #     #     entity_found = bool(named_entities) and any(
# #     #         any(contains_entity(e, chunk) for e in named_entities)
# #     #         for chunk in retrieved_texts
# #     #     )
# #     if reranker_top < -5:
# #         strong_exact = any(
# #             d.metadata.get("exact_score", 0) >= 3
# #             for d in retrieved
# #         )

# #         if not strong_exact:
# #             print("[Guard] ⚠️ Negative reranker → continuing (no hard reject)")

# #         # if not entity_found and not strong_exact:
# #         #     print("[Guard] ⚠️ Negative reranker + weak signals → continuing")
           

# #         # if entity_found:
# #         #     print(f"[Guard] ⚠️ Negative reranker but entity {named_entities} found → continuing")
# #         elif strong_exact:
# #             print("[Guard] ⚠️ Negative reranker but strong exact match → continuing")
    


# #     # ============================================================
# #     # CONTINUE PIPELINE
# #     # ============================================================
# #     retrieval_score   = compute_retrieval_score(question, retrieved)
# #     context_precision = compute_context_precision(question, retrieved)
# #     recall_score      = compute_recall_at_k(
# #         question, retrieved, all_chunks, k=len(retrieved)
# #     )

# #     query_type = classify_from_context(question, retrieved_texts)
# #     state["query_type"] = query_type
# #     print(f"[Routing] Context-based → {query_type}")

# #     # ============================================================
# #     # STEP 4: ANSWER GENERATION
# #     # ============================================================

# #     if query_type == "FULL_SUMMARY":
# #         # Rare — only if question is a summary request
# #         # ✅ CLEAN FIRST
# #         retrieved_texts = clean_context_for_llm(retrieved_texts)

# #         # ✅ THEN rank
# #         ranked = reorder_by_question(question, retrieved_texts)

# #         # ✅ THEN build context
# #         context = "\n\n---\n\n".join(
# #             clean_chunk_text(c) for c in ranked[:8]
# #         )
# #         # ✅ ADD THIS (structured context)
# #         structured_context = "\n".join(
# #             f"[CONTEXT CHUNK]\n{c}" for c in context.split("\n\n---\n\n")
# #         )
# #         emit_event(request_id, "stream_start", "✍️ Generating answer...")
# #         answer, _ = call_llama_streaming(
# #             QA_PROMPT.format(context=structured_context[:2500], question=question),
# #             request_id=request_id,
# #             temperature=0.0
# #         )
# #         answer     = clean_artifacts(answer).strip()
# #         model_used = "llama_summary"
# #         llm_calls  = 1

# #     elif query_type == "MULTIPART_QA":
# #         retrieved_texts = clean_context_for_llm(retrieved_texts)
# #         ranked     = reorder_by_question(question, retrieved_texts)
# #         context    = "\n\n---\n\n".join(
# #             clean_chunk_text(c) for c in ranked[:8]
# #         )
# #         emit_event(request_id, "agent_start",
# #                    f"🤖 MULTIPART | {len(retrieved)} chunks")
# #         emit_event(request_id, "stream_start", "✍️ Generating answer...")
# #         answer, _ = call_llama_streaming(
# #             QA_PROMPT.format(context=context[:2500], question=question),
# #             request_id=request_id,
# #             temperature=0.0
# #         )
# #         answer     = clean_artifacts(answer).strip()
# #         model_used = "llama_multipart"
# #         llm_calls  = 1

# #     else:
      
# #         # ============================================================
# #         # FAST PATH + REACT (UNIFIED FLOW)
# #         # ============================================================
# #         # ============================================================
# #         # ⚡ FAST PATH — strong exact match bypass
# #         # ============================================================
# #         strong_exact_chunks = [
# #             d for d in retrieved
# #             if d.metadata.get("exact_score", 0) >= 8
# #         ]



# #         if strong_exact_chunks:
# #             numbers_in_question = re.findall(r'\b\d+\b', question)

# #             best_chunk = max(
# #                 strong_exact_chunks,
# #                 key=lambda d: d.metadata.get("exact_score", 0)
# #             )

# #             cleaned = clean_chunk_text(best_chunk.page_content)
# #             first_line = cleaned.split("\n")[0].strip().lower()

# #             # ✅ ONLY trust chunk if number matches
# #             number_confirmed = not numbers_in_question or any(
# #                 num in first_line for num in numbers_in_question
# #             )

# #             if number_confirmed:
# #                 lines = [l.strip() for l in cleaned.split("\n") if l.strip()]

# #                 candidate_lines = [
# #                     l for l in lines
# #                     if 2 <= len(l.split()) <= 12
# #                 ]

# #                 fast_answer = candidate_lines[0] if candidate_lines else lines[0] if lines else ""

# #                 if fast_answer:
# #                     print("[QA] ⚡ Fast path candidate — sending to verification")
# #                     react_ans = fast_answer
# #                     used_fast_path = True
# #                 else:
# #                     used_fast_path = False

# #             else:
# #                 print("[QA] ⚠️ Fast path rejected — number mismatch")
# #                 used_fast_path = False

# #         # fallback to react_agent if not used
# #         if not used_fast_path:
# #             react_ans, model_used, _, _, _, _, react_calls, retrieved_texts = react_agent(
# #                 question, faiss_index, query_type, all_chunks, request_id
# #             )
# #             llm_calls += react_calls
# #         # ============================================================
# #         # COMMON PIPELINE (APPLIES TO BOTH PATHS)
# #         # ============================================================

# #         react_ans = clean_artifacts(react_ans).strip().strip('"').strip("'")
# #         print(f"[QA] ReAct: '{react_ans[:60]}'")

# #         short_answer = len(react_ans.split()) <= 2

# #         answer_words = [
# #             w for w in react_ans.lower().split()
# #             if len(w) > 2
# #         ]

# #         match_count = 0

# #         for chunk in retrieved_texts:
# #             chunk_lower = chunk.lower()

# #             hits = sum(1 for w in answer_words if w in chunk_lower)

# #             if answer_words:
# #                 overlap = hits / len(answer_words)
# #                 if overlap >= 0.6:
# #                     match_count += 1
# #                     break

# #         if short_answer:
# #             is_grounded = True
# #         else:
# #             is_grounded = match_count > 0

# #         # ⚠️ DO NOT REJECT HERE
# #         if not is_grounded:
# #             print("[QA] ⚠️ Weak grounding signal (will verify later)")


# #         # ============================================================
# #         # FINAL DECISION
# #         # ============================================================

# #         grounding_score = compute_answer_grounding(
# #             react_ans, retrieved_texts, question
# #         )

# #         print(f"[QA] Grounding={grounding_score:.1f}% | Recall={recall_score:.1f}%")

# #         # ✅ FIXED CONDITION (correct precedence)
# #         if grounding_score < 40 and recall_score < 40 and not short_answer:
# #             print(f"[QA] ❌ Low grounding → rejecting")

# #             answer        = "This information is not present in the document."
# #             model_used    = "llama_verified_not_found"
# #             decision_type = "rejected_verification"

# #         # # ✅ EXTRA SAFETY FOR SHORT ANSWERS (IMPORTANT)
# #         # elif short_answer and grounding_score < 30:
# #         #     print("[QA] ❌ Short answer not grounded → rejecting")

# #         #     answer        = "This information is not present in the document."
# #         #     model_used    = "not_found"
# #         #     decision_type = "short_not_grounded"

# #         elif react_ans and react_ans != "NOT PRESENT" and not is_refusal(react_ans):
# #             answer        = react_ans
# #             model_used    = "llama_react" if not used_fast_path else "fast_path_verified"
# #             decision_type = "accepted"

# #         else:
# #             rob_ans, rob_score = roberta_qa(question, retrieved_texts)
# #             rob_ans = rob_ans.strip()

# #             print(f"[QA] RoBERTa fallback: '{rob_ans[:40]}' score={rob_score:.3f}")

# #             if rob_ans and rob_score >= 0.25 and not is_refusal(rob_ans):

# #                 rob_grounding = compute_answer_grounding(
# #                     rob_ans, retrieved_texts, question
# #                 )

# #                 if rob_grounding < 40:
# #                     print("[QA] ❌ RoBERTa not grounded → rejecting")

# #                     answer        = "This information is not present in the document."
# #                     model_used    = "not_found"
# #                     decision_type = "not_found"

# #                 else:
# #                     answer        = rob_ans
# #                     model_used    = "roberta_fallback"
# #                     decision_type = "accepted_roberta"

# #             else:
# #                 answer        = "This information is not present in the document."
# #                 model_used    = "not_found"
# #                 decision_type = "not_found"


# #         # ============================================================
# #         # FINAL CLEANUP
# #         # ============================================================

# #         answer = expand_answer(answer, retrieved_texts)
   
# #     # ============================================================
# #     # STEP 5: METRICS
# #     # ============================================================
# #     grounding  = compute_answer_grounding(answer, retrieved_texts, question)
# #     confidence = round(grounding / 100, 3)
# #     qa_time    = time.time() - qa_start_t

# #     print(f"[QA] Done in {qa_time:.1f}s | model={model_used} | "
# #           f"grounding={grounding:.1f}% | recall={recall_score:.1f}% | "
# #           f"confidence={confidence:.3f} | decision={decision_type}")

# #     if not answer.strip():
# #         answer = "Could not find a relevant answer in the PDF."
# #     else:
# #         answer = normalize_answer(answer)

# #     state["answer"]                       = answer
# #     state["metrics"]["qa_time_sec"]       = round(qa_time, 2)
# #     state["metrics"]["confidence_score"]  = round(confidence * 100, 2)
# #     state["metrics"]["retrieval_score"]   = retrieval_score
# #     state["metrics"]["context_precision"] = context_precision
# #     state["metrics"]["answer_grounding"]  = grounding
# #     state["metrics"]["recall_at_k"]       = recall_score
# #     state["metrics"]["llm_calls"]         = llm_calls
# #     state["metrics"]["model_used"]        = model_used
# #     state["metrics"]["chunks_retrieved"]  = len(retrieved)
# #     state["metrics"]["type"]              = "qa"
# #     state["metrics"]["decision_type"]     = decision_type
# #     state["metrics"]["confidence_raw"]    = round(confidence, 4)
# #     return state

# def node_qa(state: DocState) -> DocState:
#     qa_start_t = time.time()
#     question   = state["question"]
#     request_id = state.get("request_id", "")
#     all_chunks = state["chunks"]

#     pdf_hash    = get_pdf_hash(state["pdf_path"])
#     faiss_index = build_faiss_index(all_chunks, pdf_hash)

#     print(f"[QA] START | {len(all_chunks)} chunks")

#     recall_score      = 0.0
#     retrieval_score   = 0.0
#     context_precision = 0.0
#     grounding         = 0.0
#     llm_calls         = 0
#     model_used        = "llama"
#     confidence        = 0.0
#     decision_type     = "accepted"
#     retrieved         = []
#     retrieved_texts   = []
#     answer            = ""
#     used_fast_path    = False

#     # ── Refusal detector ─────────────────────────────────────
#     _REFUSAL_ANCHOR = "this information is not available in the provided context"

#     def is_refusal(text: str) -> bool:
#         if not text or len(text.strip()) < 2:
#             return True
#         sim = semantic_similarity(text, [_REFUSAL_ANCHOR])
#         print(f"[Refusal] sim={sim:.3f} | '{text[:50]}'")
#         return sim > 0.72

#     # ============================================================
#     # STEP 1: RETRIEVE
#     # ============================================================
#     retrieved = multi_query_retrieve(
#         question, faiss_index,
#         k=20,
#         all_chunks=all_chunks,
#         query_type="FACTUAL_QA"
#     )

#     # ============================================================
#     # STEP 2: RERANK
#     # ============================================================
#     retrieved, reranker_top, _ = rerank_docs(
#         question, retrieved,
#         top_k=8,
#         apply_pruning=True
#     )

#     retrieved = protect_exact_matches(
#         question, retrieved, all_chunks, top_k=8
#     )

#     retrieved_texts = [d.page_content for d in retrieved]
#     # ============================================================
#     # 🔥 SPAN EXTRACTION (GENERIC — NO HARDCODE)
#     # ============================================================

#     def extract_candidate_spans(chunks):
#         spans = []
#         for chunk in chunks:
#             for line in chunk.split("\n"):
#                 line = line.strip()
#                 if not line:
#                     continue
#                 wc = len(line.split())
#                 if 2 <= wc <= 20:
#                     spans.append(line)
#         return spans


#     def select_best_span(question, spans):
#         q_words = set(question.lower().split())

#         best_span = ""
#         best_score = 0

#         for span in spans:
#             s_words = set(span.lower().split())
#             overlap = len(q_words & s_words)

#             # if overlap > best_score:
#             #     best_score = overlap
#             #     best_span = span
#             if overlap > best_score:
#                 best_score = overlap
#                 best_span = span

#             elif overlap == best_score:
#                 # ✅ prefer more informative span
#                 if len(span.split()) > len(best_span.split()):
#                     best_span = span

#         return best_span


#     # 🔥 RUN SPAN EXTRACTION
#     spans = extract_candidate_spans(retrieved_texts)
#     candidate_span = select_best_span(question, spans)

#     # Soft reranker warning (NO rejection)
#     if reranker_top < -5:
#         print("[Guard] ⚠️ Weak reranker score — continuing")

#     # ============================================================
#     # STEP 3: METRICS + CLASSIFICATION
#     # ============================================================
#     retrieval_score   = compute_retrieval_score(question, retrieved)
#     context_precision = compute_context_precision(question, retrieved)
#     recall_score      = compute_recall_at_k(
#         question, retrieved, all_chunks, k=len(retrieved)
#     )

#     query_type = classify_from_context(question, retrieved_texts)
#     state["query_type"] = query_type
#     print(f"[Routing] Context-based → {query_type}")

#     # ============================================================
#     # STEP 4: ANSWER GENERATION
#     # ============================================================

#     if query_type == "FULL_SUMMARY":
#         retrieved_texts = clean_context_for_llm(retrieved_texts)
#         ranked = reorder_by_question(question, retrieved_texts)

#         context = "\n\n---\n\n".join(
#             clean_chunk_text(c) for c in ranked[:8]
#         )

#         structured_context = "\n".join(
#             f"[CONTEXT CHUNK]\n{c}" for c in context.split("\n\n---\n\n")
#         )

#         emit_event(request_id, "stream_start", "✍️ Generating answer...")

#         answer, _ = call_llama_streaming(
#             QA_PROMPT.format(context=structured_context[:2500], question=question),
#             request_id=request_id,
#             temperature=0.0
#         )

#         answer     = clean_artifacts(answer).strip()
#         model_used = "llama_summary"
#         llm_calls  = 1

#     elif query_type == "MULTIPART_QA":
#         retrieved_texts = clean_context_for_llm(retrieved_texts)
#         ranked = reorder_by_question(question, retrieved_texts)

#         context = "\n\n---\n\n".join(
#             clean_chunk_text(c) for c in ranked[:8]
#         )

#         emit_event(request_id, "agent_start",
#                    f"🤖 MULTIPART | {len(retrieved)} chunks")

#         emit_event(request_id, "stream_start", "✍️ Generating answer...")

#         answer, _ = call_llama_streaming(
#             QA_PROMPT.format(context=context[:2500], question=question),
#             request_id=request_id,
#             temperature=0.0
#         )

#         answer     = clean_artifacts(answer).strip()
#         model_used = "llama_multipart"
#         llm_calls  = 1

#     else:
    
#         # ============================================================
#         # ⚡ TRY DIRECT SPAN FIRST (NO LLM)
#         # ============================================================

#         if candidate_span and len(candidate_span.split()) <= 15:
#             print("[QA] ⚡ Using span extraction")

#             answer = candidate_span
#             model_used = "span_extraction"
#             decision_type = "span_direct"

#             answer = expand_answer(answer, retrieved_texts)
#             state["answer"] = normalize_answer(answer)
#             return state

#         # ============================================================
#         # ⚠️ FIX D — LOW RECALL → EXPAND RETRIEVAL
#         # ============================================================
#         if recall_score < 25:
#             print("[QA] ⚠️ Low recall → expanding retrieval")

#             retrieved = multi_query_retrieve(
#                 question, faiss_index,
#                 k=30,
#                 all_chunks=all_chunks,
#                 query_type=query_type
#             )

#             retrieved_texts = [d.page_content for d in retrieved]

#         # ============================================================
#         # FAST PATH
#         # ============================================================
#         strong_exact_chunks = [
#             d for d in retrieved
#             if d.metadata.get("exact_score", 0) >= 8
#         ]

#         if strong_exact_chunks:
#             numbers = re.findall(r'\d+', question)

#             best_chunk = max(
#                 strong_exact_chunks,
#                 key=lambda d: d.metadata.get("exact_score", 0)
#             )

#             cleaned = clean_chunk_text(best_chunk.page_content)
#             first_line = cleaned.split("\n")[0].lower()

#             number_match = not numbers or any(n in first_line for n in numbers)

#             if number_match:
#                 lines = [l.strip() for l in cleaned.split("\n") if l.strip()]
#                 candidates = [
#                     l for l in lines if 2 <= len(l.split()) <= 20
#                 ]

#                 if candidates:
#                     react_ans = candidates[0]
#                     used_fast_path = True
#                     print("[QA] ⚡ Fast path candidate")

#         # ============================================================
#         # ❌ FIX C — NO CONTEXT → STOP BEFORE LLM
#         # ============================================================
#         if not retrieved_texts or len(retrieved_texts) == 0:
#             print("[QA] ❌ No context → NOT FOUND")

#             state["answer"] = "This information is not present in the document."
#             state["metrics"]["decision_type"] = "no_context"
#             return state

#         # ============================================================
#         # REACT (fallback)
#         # ============================================================
#         if not used_fast_path:
#             react_ans, model_used, _, _, _, _, react_calls, retrieved_texts = react_agent(
#                 question, faiss_index, query_type, all_chunks, request_id
#             )
#             llm_calls += react_calls

#         # ============================================================
#         # CLEAN ANSWER
#         # ============================================================
#         react_ans = clean_artifacts(react_ans).strip().strip('"').strip("'")
#         print(f"[QA] ReAct: '{react_ans[:60]}'")

#         # ============================================================
#         # ❌ DRIFT FILTER (GENERIC)
#         # ============================================================
#         bad_patterns = [
#             "since the question",
#             "the question is about",
#             "i will extract",
#             "the answer is based on"
#         ]

#         if any(p in react_ans.lower() for p in bad_patterns):
#             print("[QA] ❌ Drift detected → fallback")
#             react_ans = ""

#         # ============================================================
#         # FINAL DECISION
#         # ============================================================
#         grounding_score = compute_answer_grounding(
#             react_ans, retrieved_texts, question
#         )

#         short_answer = len(react_ans.split()) <= 2

#         # ============================================================
#         # ❌ FIX B — BLOCK HALLUCINATION
#         # ============================================================
#         if recall_score < 30 and grounding_score < 30:
#             print("[QA] ❌ Weak retrieval → forcing NOT FOUND")

#             answer = "This information is not present in the document."
#             model_used = "not_found"
#             decision_type = "weak_retrieval"

#         elif grounding_score < 30 and not short_answer:
#             print("[QA] ❌ Not grounded → reject")

#             answer = "This information is not present in the document."
#             decision_type = "low_grounding"

#         elif react_ans and not is_refusal(react_ans):
#             answer = react_ans
#             decision_type = "accepted"

#         else:
#             answer = "This information is not present in the document."
#             decision_type = "not_found"

#         answer = expand_answer(answer, retrieved_texts)
#         # grounding_score = compute_answer_grounding(
#         #     react_ans, retrieved_texts, question
#         # )

#         # print(f"[QA] Grounding={grounding_score:.1f}% | Recall={recall_score:.1f}%")

#         # short_answer = len(react_ans.split()) <= 2

#         # if recall_score < 50:
#         #     print("[QA] ❌ Weak retrieval → reject")
#         #     answer = "This information is not present in the document."
#         #     model_used = "not_found"
#         #     decision_type = "weak_retrieval"

#         # elif grounding_score < 40 and not short_answer:
#         #     print("[QA] ❌ Low grounding → reject")
#         #     answer = "This information is not present in the document."
#         #     model_used = "not_found"
#         #     decision_type = "low_grounding"

#         # elif react_ans and not is_refusal(react_ans):
#         #     answer = react_ans
#         #     model_used = "fast_path" if used_fast_path else "llama_react"
#         #     decision_type = "accepted"

#         # else:
#         #     rob_ans, rob_score = roberta_qa(question, retrieved_texts)
#         #     rob_ans = rob_ans.strip()

#         #     if rob_ans and rob_score >= 0.25:
#         #         answer = rob_ans
#         #         model_used = "roberta_fallback"
#         #         decision_type = "fallback"
#         #     else:
#         #         answer = "This information is not present in the document."
#         #         model_used = "not_found"
#         #         decision_type = "not_found"

#         # answer = expand_answer(answer, retrieved_texts)

#     # ============================================================
#     # STEP 5: METRICS
#     # ============================================================
#     grounding  = compute_answer_grounding(answer, retrieved_texts, question)
#     confidence = round(grounding / 100, 3)
#     qa_time    = time.time() - qa_start_t

#     print(f"[QA] Done in {qa_time:.1f}s | model={model_used} | "
#           f"grounding={grounding:.1f}% | recall={recall_score:.1f}% | "
#           f"confidence={confidence:.3f} | decision={decision_type}")

#     if not answer.strip():
#         answer = "Could not find a relevant answer in the PDF."
#     else:
#         answer = normalize_answer(answer)

#     state["answer"] = answer
#     state["metrics"]["qa_time_sec"] = round(qa_time, 2)
#     state["metrics"]["confidence_score"] = round(confidence * 100, 2)
#     state["metrics"]["retrieval_score"] = retrieval_score
#     state["metrics"]["context_precision"] = context_precision
#     state["metrics"]["answer_grounding"] = grounding
#     state["metrics"]["recall_at_k"] = recall_score
#     state["metrics"]["llm_calls"] = llm_calls
#     state["metrics"]["model_used"] = model_used
#     state["metrics"]["chunks_retrieved"] = len(retrieved)
#     state["metrics"]["type"] = "qa"
#     state["metrics"]["decision_type"] = decision_type
#     state["metrics"]["confidence_raw"] = round(confidence, 4)

#     return state

# def node_validate(state: DocState) -> DocState:
#     answer = state["answer"]
#     retry  = state.get("retry_count", 0)

#     if len(answer.strip()) < 3 and retry < 2:
#         state["retry_count"] = retry + 1
#         state["answer"]      = ""
#         return state

#     total_time    = time.time() - state["start_time"]
#     output_words  = len(answer.split())
#     output_tokens = output_words * 1.3
#     extract_time  = state["metrics"].get("extraction_time_sec", 0)
#     llm_time      = max(total_time - extract_time, 1)
#     tps           = round(output_tokens / llm_time, 2) if llm_time > 0 else 0
#     m             = state["metrics"]

#     m["response_time_sec"]    = round(total_time, 2)
#     m["extraction_time_sec"]  = m.get("extraction_time_sec", 0)
#     m["pages_processed"]      = state.get("page_count", 0)
#     m["characters_processed"] = state.get("char_count", 0)
#     m["words_processed"]      = len(state.get("extracted_text", "").split())

#     if m.get("type") == "summary":
#         m["summary_time_sec"]     = m.get("summary_time_sec", 0)
#         m["summary_length_words"] = len(answer.split())
#     if m.get("type") == "qa":
#         m["qa_time_sec"]      = m.get("qa_time_sec", 0)
#         m["confidence_score"] = m.get("confidence_score", 0)

#     m["ttft_sec"]          = round(total_time, 2)
#     m["e2e_latency_sec"]   = round(total_time, 2)
#     m["tps"]               = tps
#     m["doc_type"]          = state.get("doc_type", "general")
#     m["query_type"]        = state.get("query_type", "")
#     m["chunks_created"]    = m.get("chunks_created", 0)
#     m["retry_count"]       = retry
#     m["model_used"]        = m.get("model_used", "llama_react")
#     m["llm_calls"]         = m.get("llm_calls", 0)
#     m["retrieval_score"]   = m.get("retrieval_score", 0)
#     m["context_precision"] = m.get("context_precision", 0)
#     m["answer_grounding"]  = m.get("answer_grounding", 0)
#     m["recall_at_k"]       = m.get("recall_at_k", 0)

#     if m.get("type") == "summary":
#         m["parallel_workers"] = m.get("parallel_workers", 0)
#         m["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
#         m["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)
#     if m.get("type") == "qa":
#         m["chunks_retrieved"] = m.get("chunks_retrieved", 0)
#         m["decision_type"]    = m.get("decision_type", "accepted")
#         m["confidence_raw"]   = m.get("confidence_raw", 0.0)

#     state["metrics"] = m
#     return state
























# def node_qa(state: DocState) -> DocState:
#     qa_start_t = time.time()
#     question   = state["question"]
#     request_id = state.get("request_id", "")
#     all_chunks = state["chunks"]

#     pdf_hash    = get_pdf_hash(state["pdf_path"])
#     faiss_index = build_faiss_index(all_chunks, pdf_hash)

#     print(f"[QA] START | {len(all_chunks)} chunks")

#     recall_score      = 0.0
#     retrieval_score   = 0.0
#     context_precision = 0.0
#     grounding         = 0.0
#     llm_calls         = 0
#     model_used        = "llama"
#     confidence        = 0.0
#     decision_type     = "accepted"
#     retrieved         = []
#     retrieved_texts   = []
#     answer            = ""

#     # ── Refusal detector ─────────────────────────────────────
#     _REFUSAL_ANCHOR = "this information is not available in the provided context"

#     def is_refusal(text: str) -> bool:
#         if not text or len(text.strip()) < 2:
#             return True
#         sim = semantic_similarity(text, [_REFUSAL_ANCHOR])
#         print(f"[Refusal] sim={sim:.3f} | '{text[:50]}'")
#         return sim > 0.72

#     # ============================================================
#     # STEP 1: RETRIEVE FIRST — always
#     # ============================================================
#     retrieved = multi_query_retrieve(
#         question, faiss_index,
#         k=20,
#         all_chunks=all_chunks,
#         query_type="FACTUAL_QA"   # neutral — no assumption yet
#     )

#     # ============================================================
#     # STEP 2: RERANK
#     # ============================================================
#     retrieved, reranker_top, all_scores = rerank_docs(
#         question, retrieved,
#         top_k=8,
#         apply_pruning=True
#     )

#     # ── Hallucination guard ───────────────────────────────────
#     # Only reject when reranker is strongly negative AND
#     # no chunk has a strong exact-match signal.
#     # Prevents false rejection when phrasing mismatch causes
#     # low reranker scores despite answer existing.
#     # if reranker_top < -5:
#     #     strong_exact = any(
#     #         d.metadata.get("exact_score", 0) >= 5
#     #         for d in retrieved
#     #     )
#     #     if not strong_exact:
#     #         print(f"[Guard] ❌ Negative reranker ({reranker_top:.2f}) "
#     #               f"+ no strong exact match → NOT FOUND")
#     #         state["answer"] = "This information is not present in the document."
#     #         state["metrics"]["decision_type"] = "rejected_negative_reranker"
#     #         state["metrics"]["type"] = "qa"
#     #         return state
#     #     print(f"[Guard] ⚠️ Negative reranker but strong exact match → continuing")
#     # retrieved_texts = [d.page_content for d in retrieved]
#     # if reranker_top < -5:
#     #     # Extract named entities from question
#     #     # Any capitalised word 3+ chars that isn't a question word
#     #     question_words = {"what", "who", "which", "where", "when", 
#     #                     "how", "did", "does", "the", "was", "were",
#     #                     "are", "is", "a", "an", "in", "of", "for"}
#     #     named_entities = [
#     #         w for w in re.findall(r'\b[A-Z][a-z]{2,}\b', question)
#     #         if w.lower() not in question_words
#     #     ]
        
#     #     # If question contains a named entity, check if it appears
#     #     # in any retrieved chunk — binary presence, no threshold
#     #     if named_entities:
#     #         entity_found = any(
#     #             any(entity.lower() in chunk.lower() 
#     #                 for entity in named_entities)
#     #             for chunk in retrieved_texts
#     #         )
#     #         if entity_found:
#     #             print(f"[Guard] ⚠️ Negative reranker but entity "
#     #                 f"{named_entities} found in chunks → continuing")
#     #         else:
#     #             print(f"[Guard] ❌ Negative reranker + entity not found → NOT FOUND")
#     #             state["answer"] = "This information is not present in the document."
#     #             state["metrics"]["decision_type"] = "rejected_negative_reranker"
#     #             state["metrics"]["type"] = "qa"
#     #             return state
#     #     else:
#     #         # No named entity — rely on reranker signal alone
#     #         print(f"[Guard] ❌ Negative reranker + no named entity → NOT FOUND")
#     #         state["answer"] = "This information is not present in the document."
#     #         state["metrics"]["decision_type"] = "rejected_negative_reranker"
#     #         state["metrics"]["type"] = "qa"
#     #         return state
#     # retrieved = protect_exact_matches(
#     #     question, retrieved, all_chunks, top_k=8
#     # )
#     # # retrieved_texts = [d.page_content for d in retrieved]

#     # retrieval_score   = compute_retrieval_score(question, retrieved)
#     # context_precision = compute_context_precision(question, retrieved)
#     # recall_score      = compute_recall_at_k(
#     #     question, retrieved, all_chunks, k=len(retrieved)
#     # )

#     # # ============================================================
#     # # STEP 3: CLASSIFY FROM CONTEXT — post-retrieval, no LLM call
#     # # ============================================================
#     # query_type = classify_from_context(question, retrieved_texts)
#     retrieved = protect_exact_matches(
#         question, retrieved, all_chunks, top_k=8
#     )
#     retrieved_texts = [d.page_content for d in retrieved]
# # ✅ define ONCE (outside everything)
#     def contains_entity(entity, text):
#         return re.search(rf'\b{re.escape(entity)}\b', text, re.IGNORECASE)


#     # ============================================================
#     # ✅ ENTITY INJECTION (MUST COME BEFORE GUARD)
#     # ============================================================
#     named_entities = extract_named_entities(question, all_chunks)

#     if named_entities:
#         existing_fps = {d.page_content[:120].strip() for d in retrieved}
#         injected = 0

#         # ✅ limit chunks to avoid noise + slowdown
#         for chunk in all_chunks[:50]:
#             if any(contains_entity(entity, chunk) for entity in named_entities):
#                 fp = chunk[:120].strip()

#                 if fp not in existing_fps:
#                     from langchain.schema import Document
#                     retrieved.append(Document(
#                         page_content=chunk,
#                         metadata={"chunk_id": -1, "exact_score": 0}
#                     ))
#                     existing_fps.add(fp)
#                     injected += 1

#         if injected:
#             print(f"[EntityInject] Injected {injected} chunks for entities: {named_entities}")


#     # ✅ ALWAYS define (important fix)
#     retrieved_texts = [d.page_content for d in retrieved]


#     # ============================================================
#     # ✅ GUARD (FIXED)
#     # ============================================================
#     if reranker_top < -5:

#         strong_exact = any(
#             d.metadata.get("exact_score", 0) >= 3
#             for d in retrieved
#         )

#         entity_found = bool(named_entities) and any(
#             any(contains_entity(e, chunk) for e in named_entities)
#             for chunk in retrieved_texts
#         )

#         if not entity_found and not strong_exact:
#             print("[Guard] ❌ Negative reranker + no entity + no strong exact → NOT FOUND")
#             state["answer"] = "This information is not present in the document."
#             state["metrics"]["decision_type"] = "rejected_negative_reranker"
#             state["metrics"]["type"] = "qa"
#             return state

#         if entity_found:
#             print(f"[Guard] ⚠️ Negative reranker but entity {named_entities} found → continuing")
#         elif strong_exact:
#             print("[Guard] ⚠️ Negative reranker but strong exact match → continuing")
    
#     # if reranker_top < -5:
#     #     if named_entities:
#     #         entity_found = any(
#     #             any(entity.lower() in chunk.lower() for entity in named_entities)
#     #             for chunk in retrieved_texts
#     #         )

#     #         if entity_found:
#     #             print(f"[Guard] ⚠️ Negative reranker but entity {named_entities} found → continuing")
#     #         else:
#     #             print(f"[Guard] ❌ Negative reranker + entity not found → NOT FOUND")
#     #             state["answer"] = "This information is not present in the document."
#     #             state["metrics"]["decision_type"] = "rejected_negative_reranker"
#     #             state["metrics"]["type"] = "qa"
#     #             return state
#     #     else:
#     #         print(f"[Guard] ❌ Negative reranker + no named entity → NOT FOUND")
#     #         state["answer"] = "This information is not present in the document."
#     #         state["metrics"]["decision_type"] = "rejected_negative_reranker"
#     #         state["metrics"]["type"] = "qa"
#     #         return state

#     # ============================================================
#     # CONTINUE PIPELINE
#     # ============================================================
#     retrieval_score   = compute_retrieval_score(question, retrieved)
#     context_precision = compute_context_precision(question, retrieved)
#     recall_score      = compute_recall_at_k(
#         question, retrieved, all_chunks, k=len(retrieved)
#     )

#     query_type = classify_from_context(question, retrieved_texts)
#     state["query_type"] = query_type
#     print(f"[Routing] Context-based → {query_type}")

#     # ============================================================
#     # STEP 4: ANSWER GENERATION
#     # ============================================================

#     if query_type == "FULL_SUMMARY":
#         # Rare — only if question is a summary request
#         # ✅ CLEAN FIRST
#         retrieved_texts = clean_context_for_llm(retrieved_texts)

#         # ✅ THEN rank
#         ranked = reorder_by_question(question, retrieved_texts)

#         # ✅ THEN build context
#         context = "\n\n---\n\n".join(
#             clean_chunk_text(c) for c in ranked[:8]
#         )
#         # ✅ ADD THIS (structured context)
#         structured_context = "\n".join(
#             f"[CONTEXT CHUNK]\n{c}" for c in context.split("\n\n---\n\n")
#         )
#         emit_event(request_id, "stream_start", "✍️ Generating answer...")
#         answer, _ = call_llama_streaming(
#             QA_PROMPT.format(context=structured_context[:2500], question=question),
#             request_id=request_id,
#             temperature=0.0
#         )
#         answer     = clean_artifacts(answer).strip()
#         model_used = "llama_summary"
#         llm_calls  = 1

#     elif query_type == "MULTIPART_QA":
#         retrieved_texts = clean_context_for_llm(retrieved_texts)
#         ranked     = reorder_by_question(question, retrieved_texts)
#         context    = "\n\n---\n\n".join(
#             clean_chunk_text(c) for c in ranked[:8]
#         )
#         emit_event(request_id, "agent_start",
#                    f"🤖 MULTIPART | {len(retrieved)} chunks")
#         emit_event(request_id, "stream_start", "✍️ Generating answer...")
#         answer, _ = call_llama_streaming(
#             QA_PROMPT.format(context=context[:2500], question=question),
#             request_id=request_id,
#             temperature=0.0
#         )
#         answer     = clean_artifacts(answer).strip()
#         model_used = "llama_multipart"
#         llm_calls  = 1

#     else:
#         # ============================================================
#         # ⚡ FAST PATH — skip LLM if strong exact match
#         # ============================================================

#         strong_exact_chunks = [
#             d for d in retrieved
#             if d.metadata.get("exact_score", 0) >= 8
#         ]

#         # if strong_exact_chunks:
#         #     best_chunk = max(
#         #         strong_exact_chunks,
#         #         key=lambda d: d.metadata.get("exact_score", 0)
#         #     )

#         #     lines = best_chunk.page_content.split("\n")

#         #     candidate_lines = [
#         #         l.strip() for l in lines
#         #         if l.strip() and 2 <= len(l.split()) <= 12
#         #     ]

#         #     # answer = candidate_lines[0] if candidate_lines else lines[0].strip()
#         #     question_words = [w.lower() for w in question.split() if len(w) > 3]

#         #     scored = []

#         #     for line in candidate_lines:
#         #         score = sum(1 for w in question_words if w in line.lower())
#         #         scored.append((score, line))

#         #     scored.sort(reverse=True)

#         #     if scored and scored[0][0] > 0:
#         #         answer = scored[0][1]
#         #     else:
#         #         answer = candidate_lines[0] if candidate_lines else lines[0].strip()

#         #     print("[QA] ⚡ Fast path used (exact match)")

#         #     state["answer"] = answer
#         #     state["metrics"]["model_used"] = "exact_match_fast"
#         #     state["metrics"]["decision_type"] = "fast_path"

#         #     print("[QA] ⚠️ Fast path candidate — sending to verification")

#         #     return state
#         # ============================================================
#         # FAST PATH + REACT (UNIFIED FLOW)
#         # ============================================================

#         if strong_exact_chunks:
#             best_chunk = max(
#                 strong_exact_chunks,
#                 key=lambda d: d.metadata.get("exact_score", 0)
#             )

#             lines = best_chunk.page_content.split("\n")

#             candidate_lines = [
#                 l.strip() for l in lines
#                 if l.strip() and 2 <= len(l.split()) <= 12
#             ]

#             fast_answer = candidate_lines[0] if candidate_lines else lines[0].strip()

#             print("[QA] ⚡ Fast path candidate — sending to verification")

#             react_ans = fast_answer
#             used_fast_path = True

#         else:
#             used_fast_path = False

#             react_ans, model_used, _, _, _, _, react_calls, retrieved_texts = react_agent(
#                 question, faiss_index, query_type, all_chunks, request_id
#             )

#             llm_calls += react_calls


#         # ============================================================
#         # COMMON PIPELINE (APPLIES TO BOTH PATHS)
#         # ============================================================

#         react_ans = clean_artifacts(react_ans).strip().strip('"').strip("'")
#         print(f"[QA] ReAct: '{react_ans[:60]}'")

#         short_answer = len(react_ans.split()) <= 2

#         answer_words = [
#             w for w in react_ans.lower().split()
#             if len(w) > 2
#         ]

#         match_count = 0

#         for chunk in retrieved_texts:
#             chunk_lower = chunk.lower()

#             hits = sum(1 for w in answer_words if w in chunk_lower)

#             if answer_words:
#                 overlap = hits / len(answer_words)
#                 if overlap >= 0.6:
#                     match_count += 1
#                     break

#         if short_answer:
#             is_grounded = True
#         else:
#             is_grounded = match_count > 0

#         # ⚠️ DO NOT REJECT HERE
#         if not is_grounded:
#             print("[QA] ⚠️ Weak grounding signal (will verify later)")


#         # ============================================================
#         # FINAL DECISION
#         # ============================================================

#         grounding_score = compute_answer_grounding(
#             react_ans, retrieved_texts, question
#         )

#         print(f"[QA] Grounding={grounding_score:.1f}% | Recall={recall_score:.1f}%")

#         # ✅ FIXED CONDITION (correct precedence)
#         if ((grounding_score < 40) or (recall_score < 40)) and not short_answer:
#             print(f"[QA] ❌ Low grounding → rejecting")

#             answer        = "This information is not present in the document."
#             model_used    = "llama_verified_not_found"
#             decision_type = "rejected_verification"

#         # ✅ EXTRA SAFETY FOR SHORT ANSWERS (IMPORTANT)
#         elif short_answer and grounding_score < 30:
#             print("[QA] ❌ Short answer not grounded → rejecting")

#             answer        = "This information is not present in the document."
#             model_used    = "not_found"
#             decision_type = "short_not_grounded"

#         elif react_ans and react_ans != "NOT PRESENT" and not is_refusal(react_ans):
#             answer        = react_ans
#             model_used    = "llama_react" if not used_fast_path else "fast_path_verified"
#             decision_type = "accepted"

#         else:
#             rob_ans, rob_score = roberta_qa(question, retrieved_texts)
#             rob_ans = rob_ans.strip()

#             print(f"[QA] RoBERTa fallback: '{rob_ans[:40]}' score={rob_score:.3f}")

#             if rob_ans and rob_score >= 0.25 and not is_refusal(rob_ans):

#                 rob_grounding = compute_answer_grounding(
#                     rob_ans, retrieved_texts, question
#                 )

#                 if rob_grounding < 40:
#                     print("[QA] ❌ RoBERTa not grounded → rejecting")

#                     answer        = "This information is not present in the document."
#                     model_used    = "not_found"
#                     decision_type = "not_found"

#                 else:
#                     answer        = rob_ans
#                     model_used    = "roberta_fallback"
#                     decision_type = "accepted_roberta"

#             else:
#                 answer        = "This information is not present in the document."
#                 model_used    = "not_found"
#                 decision_type = "not_found"


#         # ============================================================
#         # FINAL CLEANUP
#         # ============================================================

#         answer = expand_answer(answer, retrieved_texts)
   
#     # ============================================================
#     # STEP 5: METRICS
#     # ============================================================
#     grounding  = compute_answer_grounding(answer, retrieved_texts, question)
#     confidence = round(grounding / 100, 3)
#     qa_time    = time.time() - qa_start_t

#     print(f"[QA] Done in {qa_time:.1f}s | model={model_used} | "
#           f"grounding={grounding:.1f}% | recall={recall_score:.1f}% | "
#           f"confidence={confidence:.3f} | decision={decision_type}")

#     if not answer.strip():
#         answer = "Could not find a relevant answer in the PDF."
#     else:
#         answer = normalize_answer(answer)

#     state["answer"]                       = answer
#     state["metrics"]["qa_time_sec"]       = round(qa_time, 2)
#     state["metrics"]["confidence_score"]  = round(confidence * 100, 2)
#     state["metrics"]["retrieval_score"]   = retrieval_score
#     state["metrics"]["context_precision"] = context_precision
#     state["metrics"]["answer_grounding"]  = grounding
#     state["metrics"]["recall_at_k"]       = recall_score
#     state["metrics"]["llm_calls"]         = llm_calls
#     state["metrics"]["model_used"]        = model_used
#     state["metrics"]["chunks_retrieved"]  = len(retrieved)
#     state["metrics"]["type"]              = "qa"
#     state["metrics"]["decision_type"]     = decision_type
#     state["metrics"]["confidence_raw"]    = round(confidence, 4)
#     return state












# did not fix so trying to make it more generic nd work using lap claude 
# import time
# import numpy as np
# import re

# from docmind_rag.config.settings import (
#     MAX_WORKERS, FACTUAL_TOP_K, MULTIPART_TOP_K, SMALL_DOC_CHUNK_THRESHOLD
# )
# from docmind_rag.core.state import DocState
# from docmind_rag.core.prompts import MULTIPART_PROMPT
# from docmind_rag.models.llm import call_llama_streaming, call_llama
# from docmind_rag.models.embeddings import build_faiss_index
# from docmind_rag.models.reranker import rerank_docs, protect_exact_matches
# from docmind_rag.services.extraction import extract_pdf_parallel, _extraction_cache
# from docmind_rag.services.chunking import semantic_chunk, raptor_summarize, _summary_cache
# from docmind_rag.services.retrieval import multi_query_retrieve
# from docmind_rag.services.agent import react_agent, roberta_qa
# from docmind_rag.events.events import emit_event
# from docmind_rag.utils.helpers import get_pdf_hash, clean_artifacts,clean_chunk_text
# from docmind_rag.utils.text import detect_doc_type, infer_answer_shape, shape_to_query_type, reorder_by_question, normalize_answer
# from docmind_rag.utils.metrics import (
#     compute_answer_grounding,
#     compute_retrieval_score,
#     compute_context_precision,
#     compute_recall_at_k,
#     semantic_similarity,
#     exact_span_match
# )
# def classify_query_type(question: str) -> str:
#     prompt = f"""
# You are a query understanding system.

# Classify the question into ONE type:

# - factual → single explanation or short answer
# - list → multiple distinct items explicitly requested
# - reasoning → requires combining information across multiple chunks
# - verification → yes/no question

# Rules:
# - Do NOT assume list unless explicitly asked (e.g., "list", "enumerate")
# - "what techniques are used" → factual
# - "how does it work" → reasoning

# Question: {question}

# Answer ONLY one word:
# factual / list / reasoning / verification
# """

#     try:
#         result = call_llama(prompt, temperature=0.0).strip().lower()
#     except:
#         return "FACTUAL_QA"

#     if "list" in result:
#         return "MULTIPART_QA"
#     elif "verification" in result:
#         return "VERIFICATION_QA"
#     elif "reasoning" in result:
#         return "REASONING_QA"
#     else:
#         return "FACTUAL_QA"

# # ============================================================
# # LANGGRAPH NODES
# # ============================================================
# def node_extract(state: DocState) -> DocState:
#     extract_start = time.time()
#     text, page_count = extract_pdf_parallel(state["pdf_path"])
#     extract_time  = time.time() - extract_start
#     state["extracted_text"] = text
#     state["page_count"]     = page_count
#     state["char_count"]     = len(text)
#     state["metrics"]["extraction_time_sec"]  = round(extract_time, 2)
#     state["metrics"]["pages_processed"]      = page_count
#     state["metrics"]["characters_processed"] = len(text)
#     state["metrics"]["words_processed"]      = len(text.split())
#     return state


# def node_chunk(state: DocState) -> DocState:
#     text = state["extracted_text"]
#     state["doc_type"] = detect_doc_type(text)
#     # Keep for logging/debug ONLY — not for routing
#     shape = infer_answer_shape(state["question"])
#     state["metrics"]["answer_shape"] = shape

#     # 🚨 IMPORTANT: DO NOT let this control routing anymore
#     state["query_type"] = "QA"   # force all queries into QA pipeline



#     summary_chunks, rag_chunks = semantic_chunk(text)
#     state["summary_chunks"] = summary_chunks
#     state["chunks"]         = rag_chunks

#     state["metrics"]["summary_chunks"] = len(summary_chunks)
#     state["metrics"]["chunks_created"] = len(rag_chunks)
#     state["metrics"]["doc_type"]       = state["doc_type"]
#     state["metrics"]["query_type"]     = state["query_type"]
#     print(f"[Chunk] {len(summary_chunks)} summary chunks | {len(rag_chunks)} RAG chunks")
#     return state


# def node_summarize(state: DocState) -> DocState:
#     pdf_hash  = get_pdf_hash(state["pdf_path"])
#     cache_key = f"{pdf_hash}_{state['doc_type']}"
#     if cache_key in _summary_cache:
#         cached = _summary_cache[cache_key]
#         print(f"[Summary] ✅ Cache hit — returning saved summary instantly")
#         emit_event(state.get("request_id", ""), "agent_action",
#                    "⚡ Summary loaded from cache instantly!")
#         state["answer"] = cached["summary"]
#         state["metrics"].update(cached["metrics"])
#         state["metrics"]["type"] = "summary"
#         return state

#     summary_start = time.time()
#     raptor_summarize._request_id = state.get("request_id", "")
#     summary, map_time, reduce_time = raptor_summarize(
#         state["summary_chunks"], state["doc_type"])
#     summary_time = time.time() - summary_start

#     state["answer"] = summary
#     metrics_snapshot = {
#         "summary_time_sec":     round(summary_time, 2),
#         "summary_length_words": len(summary.split()),
#         "parallel_workers":     min(MAX_WORKERS, len(state["summary_chunks"])),
#         "map_time_sec":         round(map_time, 2),
#         "reduce_time_sec":      round(reduce_time, 2),
#         "llm_calls":            3,
#     }
#     state["metrics"].update(metrics_snapshot)
#     state["metrics"]["type"] = "summary"
#     _summary_cache[cache_key] = {"summary": summary, "metrics": metrics_snapshot}
#     print(f"[Summary] Done ({len(summary.split())} words)")
#     return state


# def node_qa(state: DocState) -> DocState:
#     qa_start_t = time.time()
#     question   = state["question"]
#     request_id = state.get("request_id", "")
#     all_chunks = state["chunks"]

#     pdf_hash    = get_pdf_hash(state["pdf_path"])
#     faiss_index = build_faiss_index(all_chunks, pdf_hash)

#     print(f"[QA] START | {len(all_chunks)} chunks")

#     recall_score      = 0.0
#     retrieval_score   = 0.0
#     context_precision = 0.0
#     grounding         = 0.0
#     llm_calls         = 0
#     model_used        = "llama"
#     confidence        = 0.0
#     decision_type     = "accepted"
#     retrieved         = []
#     retrieved_texts   = []
#     answer            = ""

#     # ── Refusal detector ─────────────────────────────────────
#     _REFUSAL_ANCHOR = "this information is not available in the provided context"

#     def is_refusal(text: str) -> bool:
#         if not text or len(text.strip()) < 2:
#             return True
#         sim = semantic_similarity(text, [_REFUSAL_ANCHOR])
#         print(f"[Refusal] sim={sim:.3f} | '{text[:50]}'")
#         return sim > 0.72

#     # ============================================================
#     # 🔥 STEP 1: CLASSIFY FIRST
#     # ============================================================
#     query_type = classify_query_type(question)
#     state["query_type"] = query_type
#     print(f"[Routing] LLM-based → {query_type}")

#     # ============================================================
#     # 🔥 STEP 2: ADAPTIVE RETRIEVAL
#     # ============================================================
#     if query_type == "FACTUAL_QA":
#         k = 5
#     elif query_type == "MULTIPART_QA":
#         k = 8
#     elif query_type == "REASONING_QA":
#         k = 12
#     elif query_type == "VERIFICATION_QA":
#         k = 6
#     else:
#         k = 6

#     retrieved = multi_query_retrieve(
#         question,
#         faiss_index,
#         k=k,
#         all_chunks=all_chunks,
#         query_type=query_type
#     )

#     # ============================================================
#     # 🔥 STEP 3: RERANK
#     # ============================================================
#     retrieved, reranker_top, all_scores = rerank_docs(
#         question, retrieved,
#         top_k=max(k, 5),   # ensure enough context
#         apply_pruning=True
#     )

#     if reranker_top < -5:
#         print(f"[Guard] ❌ Negative reranker score ({reranker_top}) → NOT FOUND")
#         state["answer"] = "This information is not present in the document."
#         state["metrics"]["decision_type"] = "rejected_negative_reranker"
#         state["metrics"]["type"] = "qa"
#         return state

#     retrieved = protect_exact_matches(
#         question, retrieved, all_chunks,
#         top_k=max(k, 5)
#     )

#     retrieved_texts = [d.page_content for d in retrieved]

#     retrieval_score   = compute_retrieval_score(question, retrieved)
#     context_precision = compute_context_precision(question, retrieved)
#     recall_score = compute_recall_at_k(
#     question,
#     retrieved,
#     all_chunks,
#     k=len(retrieved)
# )

#     # ============================================================
#     # 🔥 NUMBER GUARD (GENERIC)
#     # ============================================================
#     numbers = re.findall(r'\b\d+\b', question)


#     if numbers:
#         found = any(
#             any(re.search(rf'\b{num}\b', chunk.lower()) for num in numbers)
#             for chunk in retrieved_texts
#         )
#         if not found:
#             print("[Guard] ❌ Number not found → NOT PRESENT")
#             state["answer"] = "This information is not present in the document."
#             state["metrics"]["decision_type"] = "not_found"
#             return state

#     # ============================================================
#     # 🔥 STEP 4: ANSWER GENERATION
#     # ============================================================

#     # ---------- MULTIPART ----------
#     if query_type == "MULTIPART_QA":

#         ranked = reorder_by_question(question, retrieved_texts)
#         top_chunks = ranked[:max(8, k)]

#         context = "\n\n---\n\n".join(
#             clean_chunk_text(c) for c in top_chunks
#         )

#         emit_event(request_id, "agent_start",
#                    f"🤖 MULTIPART | {len(retrieved)} chunks")
#         emit_event(request_id, "stream_start", "✍️ Generating answer...")

#         answer = call_llama_streaming(
#             MULTIPART_PROMPT.format(context=context[:2500], question=question),
#             request_id, temperature=0.0
#         )

#         answer     = clean_artifacts(answer).strip().strip('"').strip("'")
#         model_used = "llama_multipart"
#         llm_calls  = 1

#     # ---------- FACTUAL / REASONING ----------
#     elif query_type in ["FACTUAL_QA", "REASONING_QA"]:
#         react_ans, model_used, _, _, _, _, react_calls, retrieved_texts = react_agent(
#             question, faiss_index, query_type, all_chunks, request_id
#         )

#         llm_calls += react_calls

#         react_ans = clean_artifacts(react_ans).strip().strip('"').strip("'")
#         print(f"[QA] ReAct: '{react_ans[:60]}'")

#         grounding_score = compute_answer_grounding(react_ans, retrieved_texts, question)
#         # recall_score = compute_recall_at_k(
#         #     question,
#         #     retrieved,
#         #     all_chunks,
#         #     k=len(retrieved)
#         # )

#         print(f"[QA] Grounding={grounding_score:.1f}% | Recall={recall_score:.1f}%")

#         if recall_score < 30:
#             react_ans = "NOT PRESENT"
#         elif grounding_score < 40:
#             react_ans = "NOT PRESENT"

#         if react_ans and react_ans != "NOT PRESENT" and not is_refusal(react_ans):
#             answer        = react_ans
#             model_used    = "llama_react"
#             decision_type = "accepted"
#         else:
#             rob_ans, rob_score = roberta_qa(question, retrieved_texts)
#             rob_ans = rob_ans.strip()

#             if rob_ans and rob_score >= 0.25 and not is_refusal(rob_ans):
#                 answer        = rob_ans
#                 model_used    = "roberta_fallback"
#                 decision_type = "accepted_roberta"
#             else:
#                 answer        = "This information is not present in the document."
#                 model_used    = "not_found"
#                 decision_type = "not_found"

#     # ============================================================
#     # 🔥 STEP 5: METRICS
#     # ============================================================
#     grounding = compute_answer_grounding(answer, retrieved_texts, question)

#     confidence = round(grounding / 100, 3)

#     qa_time = time.time() - qa_start_t

#     print(f"[QA] Done in {qa_time:.1f}s | model={model_used} | "
#           f"grounding={grounding:.1f}% | recall={recall_score:.1f}% | "
#           f"confidence={confidence:.3f} | decision={decision_type}")

#     if not answer.strip():
#         answer = "Could not find a relevant answer in the PDF."
#     else:
#         answer = normalize_answer(answer)   # 🔥 apply ONLY to valid answers

#     state["answer"] = answer

#     # state["answer"]                       = answer
#     state["metrics"]["qa_time_sec"]       = round(qa_time, 2)
#     state["metrics"]["confidence_score"]  = round(confidence * 100, 2)
#     state["metrics"]["retrieval_score"]   = retrieval_score
#     state["metrics"]["context_precision"] = context_precision
#     state["metrics"]["answer_grounding"]  = grounding
#     state["metrics"]["recall_at_k"]       = recall_score
#     state["metrics"]["llm_calls"]         = llm_calls
#     state["metrics"]["model_used"]        = model_used
#     state["metrics"]["chunks_retrieved"]  = len(retrieved)
#     state["metrics"]["type"]              = "qa"
#     state["metrics"]["decision_type"]     = decision_type
#     state["metrics"]["confidence_raw"]    = round(confidence, 4)

#     return state

# def node_validate(state: DocState) -> DocState:
#     answer = state["answer"]
#     retry  = state.get("retry_count", 0)
#     if len(answer.strip()) < 3 and retry < 2:
#         state["retry_count"] = retry + 1
#         state["answer"]      = ""
#         return state

#     total_time    = time.time() - state["start_time"]
#     output_words  = len(answer.split())
#     output_tokens = output_words * 1.3
#     extract_time  = state["metrics"].get("extraction_time_sec", 0)
#     llm_time      = max(total_time - extract_time, 1)
#     tps           = round(output_tokens / llm_time, 2) if llm_time > 0 else 0
#     m             = state["metrics"]

#     m["response_time_sec"]    = round(total_time, 2)
#     m["extraction_time_sec"]  = m.get("extraction_time_sec", 0)
#     m["pages_processed"]      = state.get("page_count", 0)
#     m["characters_processed"] = state.get("char_count", 0)
#     m["words_processed"]      = len(state.get("extracted_text", "").split())

#     if m.get("type") == "summary":
#         m["summary_time_sec"]     = m.get("summary_time_sec", 0)
#         m["summary_length_words"] = len(answer.split())
#     if m.get("type") == "qa":
#         m["qa_time_sec"]      = m.get("qa_time_sec", 0)
#         m["confidence_score"] = m.get("confidence_score", 0)

#     m["ttft_sec"]          = round(total_time, 2)
#     m["e2e_latency_sec"]   = round(total_time, 2)
#     m["tps"]               = tps
#     m["doc_type"]          = state.get("doc_type", "general")
#     m["query_type"]        = state.get("query_type", "")
#     m["chunks_created"]    = m.get("chunks_created", 0)
#     m["retry_count"]       = retry
#     m["model_used"]        = m.get("model_used", "llama_react")
#     m["llm_calls"]         = m.get("llm_calls", 0)
#     m["retrieval_score"]   = m.get("retrieval_score", 0)
#     m["context_precision"] = m.get("context_precision", 0)
#     m["answer_grounding"]  = m.get("answer_grounding", 0)
#     m["recall_at_k"]       = m.get("recall_at_k", 0)

#     if m.get("type") == "summary":
#         m["parallel_workers"] = m.get("parallel_workers", 0)
#         m["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
#         m["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)
#     if m.get("type") == "qa":
#         m["chunks_retrieved"] = m.get("chunks_retrieved", 0)
#         m["decision_type"]    = m.get("decision_type", "accepted")
#         m["confidence_raw"]   = m.get("confidence_raw", 0.0)

#     state["metrics"] = m
#     return state


























# fixed the issue of  the techniqeus of system but  have to solve the 🎯 FINAL FIX (GENERIC, NOT HARDCODED)
# Your answer:

# "Convolutional Neural Networks (CNNs)"

# Dataset expected:

# "CNN"

# 👉 Your system is SEMANTICALLY CORRECT
# 👉 Evaluator is STRING MATCHING
# We don’t hack answers.

# We add post-normalization layer.



# import time
# import numpy as np
# import re

# from docmind_rag.config.settings import (
#     MAX_WORKERS, FACTUAL_TOP_K, MULTIPART_TOP_K, SMALL_DOC_CHUNK_THRESHOLD
# )
# from docmind_rag.core.state import DocState
# from docmind_rag.core.prompts import MULTIPART_PROMPT
# from docmind_rag.models.llm import call_llama_streaming, call_llama
# from docmind_rag.models.embeddings import build_faiss_index
# from docmind_rag.models.reranker import rerank_docs, protect_exact_matches
# from docmind_rag.services.extraction import extract_pdf_parallel, _extraction_cache
# from docmind_rag.services.chunking import semantic_chunk, raptor_summarize, _summary_cache
# from docmind_rag.services.retrieval import multi_query_retrieve
# from docmind_rag.services.agent import react_agent, roberta_qa
# from docmind_rag.events.events import emit_event
# from docmind_rag.utils.helpers import get_pdf_hash, clean_artifacts,clean_chunk_text
# from docmind_rag.utils.text import detect_doc_type, infer_answer_shape, shape_to_query_type, reorder_by_question
# from docmind_rag.utils.metrics import (
#     compute_answer_grounding,
#     compute_retrieval_score,
#     compute_context_precision,
#     compute_recall_at_k,
#     semantic_similarity,
#     exact_span_match
# )
# def classify_query_type(question: str) -> str:
#     prompt = f"""
# You are a query understanding system.

# Classify the question into ONE type:

# - factual → single explanation or short answer
# - list → multiple distinct items explicitly requested
# - reasoning → requires combining information across multiple chunks
# - verification → yes/no question

# Rules:
# - Do NOT assume list unless explicitly asked (e.g., "list", "enumerate")
# - "what techniques are used" → factual
# - "how does it work" → reasoning

# Question: {question}

# Answer ONLY one word:
# factual / list / reasoning / verification
# """

#     try:
#         result = call_llama(prompt, temperature=0.0).strip().lower()
#     except:
#         return "FACTUAL_QA"

#     if "list" in result:
#         return "MULTIPART_QA"
#     elif "verification" in result:
#         return "VERIFICATION_QA"
#     elif "reasoning" in result:
#         return "REASONING_QA"
#     else:
#         return "FACTUAL_QA"

# # ============================================================
# # LANGGRAPH NODES
# # ============================================================
# def node_extract(state: DocState) -> DocState:
#     extract_start = time.time()
#     text, page_count = extract_pdf_parallel(state["pdf_path"])
#     extract_time  = time.time() - extract_start
#     state["extracted_text"] = text
#     state["page_count"]     = page_count
#     state["char_count"]     = len(text)
#     state["metrics"]["extraction_time_sec"]  = round(extract_time, 2)
#     state["metrics"]["pages_processed"]      = page_count
#     state["metrics"]["characters_processed"] = len(text)
#     state["metrics"]["words_processed"]      = len(text.split())
#     return state


# def node_chunk(state: DocState) -> DocState:
#     text = state["extracted_text"]
#     state["doc_type"] = detect_doc_type(text)
#     # Keep for logging/debug ONLY — not for routing
#     shape = infer_answer_shape(state["question"])
#     state["metrics"]["answer_shape"] = shape

#     # 🚨 IMPORTANT: DO NOT let this control routing anymore
#     state["query_type"] = "QA"   # force all queries into QA pipeline



#     summary_chunks, rag_chunks = semantic_chunk(text)
#     state["summary_chunks"] = summary_chunks
#     state["chunks"]         = rag_chunks

#     state["metrics"]["summary_chunks"] = len(summary_chunks)
#     state["metrics"]["chunks_created"] = len(rag_chunks)
#     state["metrics"]["doc_type"]       = state["doc_type"]
#     state["metrics"]["query_type"]     = state["query_type"]
#     print(f"[Chunk] {len(summary_chunks)} summary chunks | {len(rag_chunks)} RAG chunks")
#     return state


# def node_summarize(state: DocState) -> DocState:
#     pdf_hash  = get_pdf_hash(state["pdf_path"])
#     cache_key = f"{pdf_hash}_{state['doc_type']}"
#     if cache_key in _summary_cache:
#         cached = _summary_cache[cache_key]
#         print(f"[Summary] ✅ Cache hit — returning saved summary instantly")
#         emit_event(state.get("request_id", ""), "agent_action",
#                    "⚡ Summary loaded from cache instantly!")
#         state["answer"] = cached["summary"]
#         state["metrics"].update(cached["metrics"])
#         state["metrics"]["type"] = "summary"
#         return state

#     summary_start = time.time()
#     raptor_summarize._request_id = state.get("request_id", "")
#     summary, map_time, reduce_time = raptor_summarize(
#         state["summary_chunks"], state["doc_type"])
#     summary_time = time.time() - summary_start

#     state["answer"] = summary
#     metrics_snapshot = {
#         "summary_time_sec":     round(summary_time, 2),
#         "summary_length_words": len(summary.split()),
#         "parallel_workers":     min(MAX_WORKERS, len(state["summary_chunks"])),
#         "map_time_sec":         round(map_time, 2),
#         "reduce_time_sec":      round(reduce_time, 2),
#         "llm_calls":            3,
#     }
#     state["metrics"].update(metrics_snapshot)
#     state["metrics"]["type"] = "summary"
#     _summary_cache[cache_key] = {"summary": summary, "metrics": metrics_snapshot}
#     print(f"[Summary] Done ({len(summary.split())} words)")
#     return state


# def node_qa(state: DocState) -> DocState:
#     qa_start_t = time.time()
#     question   = state["question"]
#     request_id = state.get("request_id", "")
#     all_chunks = state["chunks"]

#     pdf_hash    = get_pdf_hash(state["pdf_path"])
#     faiss_index = build_faiss_index(all_chunks, pdf_hash)

#     print(f"[QA] START | {len(all_chunks)} chunks")

#     recall_score      = 0.0
#     retrieval_score   = 0.0
#     context_precision = 0.0
#     grounding         = 0.0
#     llm_calls         = 0
#     model_used        = "llama"
#     confidence        = 0.0
#     decision_type     = "accepted"
#     retrieved         = []
#     retrieved_texts   = []
#     answer            = ""

#     # ── Refusal detector ─────────────────────────────────────
#     _REFUSAL_ANCHOR = "this information is not available in the provided context"

#     def is_refusal(text: str) -> bool:
#         if not text or len(text.strip()) < 2:
#             return True
#         sim = semantic_similarity(text, [_REFUSAL_ANCHOR])
#         print(f"[Refusal] sim={sim:.3f} | '{text[:50]}'")
#         return sim > 0.72

#     # ============================================================
#     # 🔥 STEP 1: CLASSIFY FIRST
#     # ============================================================
#     query_type = classify_query_type(question)
#     state["query_type"] = query_type
#     print(f"[Routing] LLM-based → {query_type}")

#     # ============================================================
#     # 🔥 STEP 2: ADAPTIVE RETRIEVAL
#     # ============================================================
#     if query_type == "FACTUAL_QA":
#         k = 5
#     elif query_type == "MULTIPART_QA":
#         k = 8
#     elif query_type == "REASONING_QA":
#         k = 12
#     elif query_type == "VERIFICATION_QA":
#         k = 6
#     else:
#         k = 6

#     retrieved = multi_query_retrieve(
#         question,
#         faiss_index,
#         k=k,
#         all_chunks=all_chunks,
#         query_type=query_type
#     )

#     # ============================================================
#     # 🔥 STEP 3: RERANK
#     # ============================================================
#     retrieved, reranker_top, all_scores = rerank_docs(
#         question, retrieved,
#         top_k=max(k, 5),   # ensure enough context
#         apply_pruning=True
#     )

#     if reranker_top < -5:
#         print(f"[Guard] ❌ Negative reranker score ({reranker_top}) → NOT FOUND")
#         state["answer"] = "This information is not present in the document."
#         state["metrics"]["decision_type"] = "rejected_negative_reranker"
#         state["metrics"]["type"] = "qa"
#         return state

#     retrieved = protect_exact_matches(
#         question, retrieved, all_chunks,
#         top_k=max(k, 5)
#     )

#     retrieved_texts = [d.page_content for d in retrieved]

#     retrieval_score   = compute_retrieval_score(question, retrieved)
#     context_precision = compute_context_precision(question, retrieved)
#     recall_score = compute_recall_at_k(
#     question,
#     retrieved,
#     all_chunks,
#     k=len(retrieved)
# )

#     # ============================================================
#     # 🔥 NUMBER GUARD (GENERIC)
#     # ============================================================
#     numbers = re.findall(r'\b\d+\b', question)


#     if numbers:
#         found = any(
#             any(re.search(rf'\b{num}\b', chunk.lower()) for num in numbers)
#             for chunk in retrieved_texts
#         )
#         if not found:
#             print("[Guard] ❌ Number not found → NOT PRESENT")
#             state["answer"] = "This information is not present in the document."
#             state["metrics"]["decision_type"] = "not_found"
#             return state

#     # ============================================================
#     # 🔥 STEP 4: ANSWER GENERATION
#     # ============================================================

#     # ---------- MULTIPART ----------
#     if query_type == "MULTIPART_QA":

#         ranked = reorder_by_question(question, retrieved_texts)
#         top_chunks = ranked[:max(8, k)]

#         context = "\n\n---\n\n".join(
#             clean_chunk_text(c) for c in top_chunks
#         )

#         emit_event(request_id, "agent_start",
#                    f"🤖 MULTIPART | {len(retrieved)} chunks")
#         emit_event(request_id, "stream_start", "✍️ Generating answer...")

#         answer = call_llama_streaming(
#             MULTIPART_PROMPT.format(context=context[:2500], question=question),
#             request_id, temperature=0.0
#         )

#         answer     = clean_artifacts(answer).strip().strip('"').strip("'")
#         model_used = "llama_multipart"
#         llm_calls  = 1

#     # ---------- FACTUAL / REASONING ----------
#     elif query_type in ["FACTUAL_QA", "REASONING_QA"]:
#         react_ans, model_used, _, _, _, _, react_calls, retrieved_texts = react_agent(
#             question, faiss_index, query_type, all_chunks, request_id
#         )

#         llm_calls += react_calls

#         react_ans = clean_artifacts(react_ans).strip().strip('"').strip("'")
#         print(f"[QA] ReAct: '{react_ans[:60]}'")

#         grounding_score = compute_answer_grounding(react_ans, retrieved_texts, question)
#         # recall_score = compute_recall_at_k(
#         #     question,
#         #     retrieved,
#         #     all_chunks,
#         #     k=len(retrieved)
#         # )

#         print(f"[QA] Grounding={grounding_score:.1f}% | Recall={recall_score:.1f}%")

#         if recall_score < 30:
#             react_ans = "NOT PRESENT"
#         elif grounding_score < 40:
#             react_ans = "NOT PRESENT"

#         if react_ans and react_ans != "NOT PRESENT" and not is_refusal(react_ans):
#             answer        = react_ans
#             model_used    = "llama_react"
#             decision_type = "accepted"
#         else:
#             rob_ans, rob_score = roberta_qa(question, retrieved_texts)
#             rob_ans = rob_ans.strip()

#             if rob_ans and rob_score >= 0.25 and not is_refusal(rob_ans):
#                 answer        = rob_ans
#                 model_used    = "roberta_fallback"
#                 decision_type = "accepted_roberta"
#             else:
#                 answer        = "This information is not present in the document."
#                 model_used    = "not_found"
#                 decision_type = "not_found"

#     # ============================================================
#     # 🔥 STEP 5: METRICS
#     # ============================================================
#     grounding = compute_answer_grounding(answer, retrieved_texts, question)

#     confidence = round(grounding / 100, 3)

#     qa_time = time.time() - qa_start_t

#     print(f"[QA] Done in {qa_time:.1f}s | model={model_used} | "
#           f"grounding={grounding:.1f}% | recall={recall_score:.1f}% | "
#           f"confidence={confidence:.3f} | decision={decision_type}")

#     if not answer.strip():
#         answer = "Could not find a relevant answer in the PDF."

#     state["answer"] = answer

#     state["answer"]                       = answer
#     state["metrics"]["qa_time_sec"]       = round(qa_time, 2)
#     state["metrics"]["confidence_score"]  = round(confidence * 100, 2)
#     state["metrics"]["retrieval_score"]   = retrieval_score
#     state["metrics"]["context_precision"] = context_precision
#     state["metrics"]["answer_grounding"]  = grounding
#     state["metrics"]["recall_at_k"]       = recall_score
#     state["metrics"]["llm_calls"]         = llm_calls
#     state["metrics"]["model_used"]        = model_used
#     state["metrics"]["chunks_retrieved"]  = len(retrieved)
#     state["metrics"]["type"]              = "qa"
#     state["metrics"]["decision_type"]     = decision_type
#     state["metrics"]["confidence_raw"]    = round(confidence, 4)

#     return state

# def node_validate(state: DocState) -> DocState:
#     answer = state["answer"]
#     retry  = state.get("retry_count", 0)
#     if len(answer.strip()) < 3 and retry < 2:
#         state["retry_count"] = retry + 1
#         state["answer"]      = ""
#         return state

#     total_time    = time.time() - state["start_time"]
#     output_words  = len(answer.split())
#     output_tokens = output_words * 1.3
#     extract_time  = state["metrics"].get("extraction_time_sec", 0)
#     llm_time      = max(total_time - extract_time, 1)
#     tps           = round(output_tokens / llm_time, 2) if llm_time > 0 else 0
#     m             = state["metrics"]

#     m["response_time_sec"]    = round(total_time, 2)
#     m["extraction_time_sec"]  = m.get("extraction_time_sec", 0)
#     m["pages_processed"]      = state.get("page_count", 0)
#     m["characters_processed"] = state.get("char_count", 0)
#     m["words_processed"]      = len(state.get("extracted_text", "").split())

#     if m.get("type") == "summary":
#         m["summary_time_sec"]     = m.get("summary_time_sec", 0)
#         m["summary_length_words"] = len(answer.split())
#     if m.get("type") == "qa":
#         m["qa_time_sec"]      = m.get("qa_time_sec", 0)
#         m["confidence_score"] = m.get("confidence_score", 0)

#     m["ttft_sec"]          = round(total_time, 2)
#     m["e2e_latency_sec"]   = round(total_time, 2)
#     m["tps"]               = tps
#     m["doc_type"]          = state.get("doc_type", "general")
#     m["query_type"]        = state.get("query_type", "")
#     m["chunks_created"]    = m.get("chunks_created", 0)
#     m["retry_count"]       = retry
#     m["model_used"]        = m.get("model_used", "llama_react")
#     m["llm_calls"]         = m.get("llm_calls", 0)
#     m["retrieval_score"]   = m.get("retrieval_score", 0)
#     m["context_precision"] = m.get("context_precision", 0)
#     m["answer_grounding"]  = m.get("answer_grounding", 0)
#     m["recall_at_k"]       = m.get("recall_at_k", 0)

#     if m.get("type") == "summary":
#         m["parallel_workers"] = m.get("parallel_workers", 0)
#         m["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
#         m["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)
#     if m.get("type") == "qa":
#         m["chunks_retrieved"] = m.get("chunks_retrieved", 0)
#         m["decision_type"]    = m.get("decision_type", "accepted")
#         m["confidence_raw"]   = m.get("confidence_raw", 0.0)

#     state["metrics"] = m
#     return state





















# got 100 nd 57 for overfitting gonna chnage k chunks i think in multi_query_retrieve
# import time
# import numpy as np
# import re

# from docmind_rag.config.settings import (
#     MAX_WORKERS, FACTUAL_TOP_K, MULTIPART_TOP_K, SMALL_DOC_CHUNK_THRESHOLD
# )
# from docmind_rag.core.state import DocState
# from docmind_rag.core.prompts import MULTIPART_PROMPT
# from docmind_rag.models.llm import call_llama_streaming
# from docmind_rag.models.embeddings import build_faiss_index
# from docmind_rag.models.reranker import rerank_docs, protect_exact_matches
# from docmind_rag.services.extraction import extract_pdf_parallel, _extraction_cache
# from docmind_rag.services.chunking import semantic_chunk, raptor_summarize, _summary_cache
# from docmind_rag.services.retrieval import multi_query_retrieve
# from docmind_rag.services.agent import react_agent, roberta_qa
# from docmind_rag.events.events import emit_event
# from docmind_rag.utils.helpers import get_pdf_hash, clean_artifacts,clean_chunk_text
# from docmind_rag.utils.text import detect_doc_type, infer_answer_shape, shape_to_query_type, reorder_by_question
# from docmind_rag.utils.metrics import (
#     compute_answer_grounding,
#     compute_retrieval_score,
#     compute_context_precision,
#     compute_recall_at_k,
#     semantic_similarity,
#     exact_span_match
# )
# def classify_query_type(question: str, llm) -> str:
#     prompt = f"""
# You are a query understanding system.

# Classify the question into ONE type:

# - factual → single explanation or short answer
# - list → multiple distinct items explicitly requested
# - reasoning → requires combining information across multiple chunks
# - verification → yes/no question

# Rules:
# - Do NOT assume list unless explicitly asked (e.g., "list", "enumerate")
# - "what techniques are used" → factual
# - "how does it work" → reasoning

# Question: {question}

# Answer ONLY one word:
# factual / list / reasoning / verification
# """

#     try:
#         result = llm.invoke(prompt).content.strip().lower()
#     except:
#         return "FACTUAL_QA"

#     if "list" in result:
#         return "MULTIPART_QA"
#     elif "verification" in result:
#         return "VERIFICATION_QA"
#     elif "reasoning" in result:
#         return "REASONING_QA"
#     else:
#         return "FACTUAL_QA"

# # ============================================================
# # LANGGRAPH NODES
# # ============================================================
# def node_extract(state: DocState) -> DocState:
#     extract_start = time.time()
#     text, page_count = extract_pdf_parallel(state["pdf_path"])
#     extract_time  = time.time() - extract_start
#     state["extracted_text"] = text
#     state["page_count"]     = page_count
#     state["char_count"]     = len(text)
#     state["metrics"]["extraction_time_sec"]  = round(extract_time, 2)
#     state["metrics"]["pages_processed"]      = page_count
#     state["metrics"]["characters_processed"] = len(text)
#     state["metrics"]["words_processed"]      = len(text.split())
#     return state


# def node_chunk(state: DocState) -> DocState:
#     text = state["extracted_text"]
#     state["doc_type"] = detect_doc_type(text)
#     # Keep for logging/debug ONLY — not for routing
#     shape = infer_answer_shape(state["question"])
#     state["metrics"]["answer_shape"] = shape

#     # 🚨 IMPORTANT: DO NOT let this control routing anymore
#     state["query_type"] = "QA"   # force all queries into QA pipeline



#     summary_chunks, rag_chunks = semantic_chunk(text)
#     state["summary_chunks"] = summary_chunks
#     state["chunks"]         = rag_chunks

#     state["metrics"]["summary_chunks"] = len(summary_chunks)
#     state["metrics"]["chunks_created"] = len(rag_chunks)
#     state["metrics"]["doc_type"]       = state["doc_type"]
#     state["metrics"]["query_type"]     = state["query_type"]
#     print(f"[Chunk] {len(summary_chunks)} summary chunks | {len(rag_chunks)} RAG chunks")
#     return state


# def node_summarize(state: DocState) -> DocState:
#     pdf_hash  = get_pdf_hash(state["pdf_path"])
#     cache_key = f"{pdf_hash}_{state['doc_type']}"
#     if cache_key in _summary_cache:
#         cached = _summary_cache[cache_key]
#         print(f"[Summary] ✅ Cache hit — returning saved summary instantly")
#         emit_event(state.get("request_id", ""), "agent_action",
#                    "⚡ Summary loaded from cache instantly!")
#         state["answer"] = cached["summary"]
#         state["metrics"].update(cached["metrics"])
#         state["metrics"]["type"] = "summary"
#         return state

#     summary_start = time.time()
#     raptor_summarize._request_id = state.get("request_id", "")
#     summary, map_time, reduce_time = raptor_summarize(
#         state["summary_chunks"], state["doc_type"])
#     summary_time = time.time() - summary_start

#     state["answer"] = summary
#     metrics_snapshot = {
#         "summary_time_sec":     round(summary_time, 2),
#         "summary_length_words": len(summary.split()),
#         "parallel_workers":     min(MAX_WORKERS, len(state["summary_chunks"])),
#         "map_time_sec":         round(map_time, 2),
#         "reduce_time_sec":      round(reduce_time, 2),
#         "llm_calls":            3,
#     }
#     state["metrics"].update(metrics_snapshot)
#     state["metrics"]["type"] = "summary"
#     _summary_cache[cache_key] = {"summary": summary, "metrics": metrics_snapshot}
#     print(f"[Summary] Done ({len(summary.split())} words)")
#     return state


# def node_qa(state: DocState) -> DocState:
#     qa_start_t = time.time()
#     question   = state["question"]
#     request_id = state.get("request_id", "")
#     all_chunks = state["chunks"]

#     pdf_hash    = get_pdf_hash(state["pdf_path"])
#     faiss_index = build_faiss_index(all_chunks, pdf_hash)

#     print(f"[QA] START | {len(all_chunks)} chunks")

#     recall_score      = 0.0
#     retrieval_score   = 0.0
#     context_precision = 0.0
#     grounding         = 0.0
#     llm_calls         = 0
#     model_used        = "llama"
#     confidence        = 0.0
#     decision_type     = "accepted"
#     retrieved         = []
#     retrieved_texts   = []
#     answer            = ""

#     # ── Refusal detector ─────────────────────────────────────
#     _REFUSAL_ANCHOR = "this information is not available in the provided context"

#     def is_refusal(text: str) -> bool:
#         if not text or len(text.strip()) < 2:
#             return True
#         sim = semantic_similarity(text, [_REFUSAL_ANCHOR])
#         print(f"[Refusal] sim={sim:.3f} | '{text[:50]}'")
#         return sim > 0.72

#     # ============================================================
#     # STEP 1: ALWAYS RETRIEVE FIRST (no routing yet)
#     # ============================================================
#     retrieved = multi_query_retrieve(
#         question, faiss_index, k=20,
#         all_chunks=all_chunks, query_type="FACTUAL_QA"
#     )

#     retrieved, reranker_top, all_scores = rerank_docs(
#         question, retrieved,
#         top_k=FACTUAL_TOP_K,
#         apply_pruning=True
#     )
#     # ============================================================
#     # 🔥 HARD STOP — NO RELEVANT CHUNK FOUND for lecture 23 remaining hallucinations 
#     # ============================================================

#     # if reranker_top < 0:
#     if reranker_top < -5:
#         print(f"[Guard] ❌ Negative reranker score ({reranker_top}) → NOT FOUND")

#         state["answer"] = "This information is not present in the document."
#         state["metrics"]["decision_type"] = "rejected_negative_reranker"
#         state["metrics"]["type"] = "qa"
#         return state

#     retrieved = protect_exact_matches(
#         question, retrieved, all_chunks,
#         top_k=FACTUAL_TOP_K
#     )

#     retrieved_texts = [d.page_content for d in retrieved]

#     retrieval_score   = compute_retrieval_score(question, retrieved)
#     context_precision = compute_context_precision(question, retrieved)
#     # ============================================================
#     # 🔥 CONSTRAINT GUARD (PREVENT HALLUCINATION)
#     # ============================================================

#     numbers = re.findall(r'\b\d+\b', question)

#     if numbers:
#         found = False

#         for chunk in retrieved_texts:
#             chunk_low = chunk.lower()

#             for num in numbers:
#                 if re.search(rf'\b{num}\b', chunk_low):
#                     found = True
#                     break

#             if found:
#                 break

#         if not found:
#             print("[Guard] ❌ Number constraint not found → NOT PRESENT")

#             state["answer"] = "This information is not present in the document."
#             state["metrics"]["decision_type"] = "not_found"
#             return state

#     # ============================================================
#     # STEP 2: CONTEXT-BASED ROUTING (KEY FIX)
#     # ============================================================


#     def detect_query_type_from_context(question, retrieved_texts):
#         """
#         Generalized routing based on evidence shape in retrieved chunks.
#         """

#         if not retrieved_texts:
#             return "REASONING_QA"

#         # Clean tokens (remove stopword noise)
#         q_words = set(re.findall(r'\b\w+\b', question.lower()))
#         q_words = {w for w in q_words if len(w) > 2}  # remove "is", "the"

#         scores = []

#         for chunk in retrieved_texts:
#             chunk_low = chunk.lower()

#             overlap = sum(1 for w in q_words if w in chunk_low)
#             length  = len(chunk.split())

#             scores.append((overlap, length, chunk))

#         # Sort by strongest overlap
#         scores.sort(reverse=True, key=lambda x: x[0])

#         top_overlap, top_len, _ = scores[0]

#         # 🔥 CASE 1 — strong single chunk → FACTUAL
#         if top_overlap >= 3 and top_len < 60:
#             return "FACTUAL_QA"

#         # 🔥 CASE 2 — multiple strong chunks → MULTIPART
#         strong_chunks = [s for s in scores if s[0] >= 2]

#         if len(strong_chunks) >= 3:
#             return "MULTIPART_QA"

#         # 🔥 CASE 3 — fallback → REASONING
#         return "REASONING_QA"

#     # query_type = detect_query_type_from_context(question, retrieved_texts)
#     # # # 🔥 FINAL OVERRIDE (STRONG FIX)
#     # # numbers = re.findall(r'\b\d+\b', question)

#     # # if numbers and len(question.split()) <= 8:
#     # #     print("[Routing Override] Forcing FACTUAL_QA for numeric lookup")
#     # #     query_type = "FACTUAL_QA"
#     # print(f"[Routing] Context-based → {query_type}")
#     question = state["question"]

#     query_type = classify_query_type(question, llm)

#     state["query_type"] = query_type

#     print(f"[Routing] LLM-based → {query_type}")
# # ============================================================
# # 🔥 FINAL GUARD (AFTER ROUTING — CRITICAL)
# # ============================================================

#     numbers = re.findall(r'\b\d+\b', question)

#     if numbers:
#         found = False

#         for chunk in retrieved_texts:
#             chunk_low = chunk.lower()

#             for num in numbers:
#                 if re.search(rf'\b{num}\b', chunk_low):
#                     found = True
#                     break

#             if found:
#                 break

#         if not found:
#             print("[Guard] ❌ Number not found in ANY retrieved chunk → stopping pipeline")

#             state["answer"] = "This information is not present in the document."
#             state["metrics"]["decision_type"] = "not_found"
#             return state
#     # ============================================================
#     # STEP 3: EXECUTION BASED ON ROUTING
#     # ============================================================

#     # ---------- MULTIPART ----------
#     if query_type == "MULTIPART_QA":

#         ranked = reorder_by_question(question, retrieved_texts)

#         top_k = max(MULTIPART_TOP_K, 8)   # ensures enough coverage
#         top_chunks = ranked[:top_k]
#         print("\n========== MULTIPART SOURCE CHUNKS ==========")
#         for i, chunk in enumerate(top_chunks):
#             preview = chunk[:300].replace("\n", " ")
#             print(f"\n--- Source {i+1} ---")
#             print(preview)
#         print("============================================\n")

#         context = "\n\n---\n\n".join(
#             clean_chunk_text(c) for c in top_chunks
#         )


#         emit_event(request_id, "agent_start",
#                    f"🤖 MULTIPART | {len(retrieved)} chunks")
#         emit_event(request_id, "stream_start", "✍️ Generating answer...")

#         answer = call_llama_streaming(
#             MULTIPART_PROMPT.format(context=context[:2500], question=question),
#             request_id, temperature=0.0
#         )

#         answer     = clean_artifacts(answer).strip().strip('"').strip("'")
#         model_used = "llama_multipart"
#         llm_calls  = 1

#     # ---------- FACTUAL / REASONING ----------
#     else:
#         react_ans, model_used, _, _, _, _, react_calls, retrieved_texts = react_agent(
#             question, faiss_index, query_type, all_chunks, request_id
#         )

#         llm_calls += react_calls
#         # react_ans = clean_artifacts(react_ans).strip().strip('"').strip("'")

#         # print(f"[QA] ReAct: '{react_ans[:60]}'")
#         react_ans = clean_artifacts(react_ans)


#         # react_ans = extract_final_answer(react_ans)
   
#         react_ans = react_ans.strip().strip('"').strip("'")

#         print(f"[QA] ReAct: '{react_ans[:60]}'")
#         # 🔥 NEW: grounding + recall based decision
#         grounding_score = compute_answer_grounding(react_ans, retrieved_texts, question)
#         # recall_score = retrieval_score   # your Recall@K proxy
#         recall_score = compute_recall_at_k(
#     question,
#     retrieved,
#     all_chunks,
#     k=len(retrieved)
# )

#         print(f"[QA] Grounding={grounding_score:.1f}% | Recall={recall_score:.1f}%")

#         # 🚨 Decision logic
#         if recall_score < 30:
#             print("[QA] ❌ Retrieval failed → rejecting")
#             react_ans = "NOT PRESENT"

#         elif grounding_score < 40:
#             print("[QA] ❌ Low grounding → rejecting")
#             react_ans = "NOT PRESENT"

#         elif grounding_score < 70:
#             print("[QA] ⚠️ Medium grounding → accepting with caution")
#             # keep answer (no change)

#         else:
#             print("[QA] ✅ Strong grounding → accepting")
        

#         # # Hallucination guard
#         # if not exact_span_match(react_ans, retrieved_texts):
#         #     print(f"[QA] ⚠️ Not grounded → '{react_ans[:60]}'")
#         #     react_ans = "NOT PRESENT"

#         if react_ans and react_ans != "NOT PRESENT" and not is_refusal(react_ans):
#             answer        = react_ans
#             model_used    = "llama_react"
#             decision_type = "accepted"
#         else:
#             rob_ans, rob_score = roberta_qa(question, retrieved_texts)
#             rob_ans = rob_ans.strip()

#             print(f"[QA] RoBERTa fallback: '{rob_ans[:40]}' score={rob_score:.3f}")

#             if rob_ans and rob_score >= 0.25 and not is_refusal(rob_ans):
#                 answer        = rob_ans
#                 model_used    = "roberta_fallback"
#                 decision_type = "accepted_roberta"
#             else:
#                 answer        = "This information is not present in the document."
#                 model_used    = "not_found"
#                 decision_type = "not_found"

#     # ============================================================
#     # STEP 4: METRICS (FIXED)
#     # ============================================================
#     grounding = compute_answer_grounding(answer, retrieved_texts, question)

#     confidence = round(
#         compute_answer_grounding(answer, retrieved_texts, question) / 100,
#         3
#     )

#     # recall_score = retrieval_score  # FIXED

#     qa_time = time.time() - qa_start_t

#     print(f"[QA] Done in {qa_time:.1f}s | model={model_used} | "
#           f"grounding={grounding:.1f}% | recall={recall_score:.1f}% | "
#           f"confidence={confidence:.3f} | decision={decision_type}")

#     if not answer.strip():
#         answer = "Could not find a relevant answer in the PDF."

#     state["answer"]                       = answer
#     state["metrics"]["qa_time_sec"]       = round(qa_time, 2)
#     state["metrics"]["confidence_score"]  = round(confidence * 100, 2)
#     state["metrics"]["retrieval_score"]   = retrieval_score
#     state["metrics"]["context_precision"] = context_precision
#     state["metrics"]["answer_grounding"]  = grounding
#     state["metrics"]["recall_at_k"]       = recall_score
#     state["metrics"]["llm_calls"]         = llm_calls
#     state["metrics"]["model_used"]        = model_used
#     state["metrics"]["chunks_retrieved"]  = len(retrieved)
#     state["metrics"]["type"]              = "qa"
#     state["metrics"]["decision_type"]     = decision_type
#     state["metrics"]["confidence_raw"]    = round(confidence, 4)

#     return state


# def node_validate(state: DocState) -> DocState:
#     answer = state["answer"]
#     retry  = state.get("retry_count", 0)
#     if len(answer.strip()) < 3 and retry < 2:
#         state["retry_count"] = retry + 1
#         state["answer"]      = ""
#         return state

#     total_time    = time.time() - state["start_time"]
#     output_words  = len(answer.split())
#     output_tokens = output_words * 1.3
#     extract_time  = state["metrics"].get("extraction_time_sec", 0)
#     llm_time      = max(total_time - extract_time, 1)
#     tps           = round(output_tokens / llm_time, 2) if llm_time > 0 else 0
#     m             = state["metrics"]

#     m["response_time_sec"]    = round(total_time, 2)
#     m["extraction_time_sec"]  = m.get("extraction_time_sec", 0)
#     m["pages_processed"]      = state.get("page_count", 0)
#     m["characters_processed"] = state.get("char_count", 0)
#     m["words_processed"]      = len(state.get("extracted_text", "").split())

#     if m.get("type") == "summary":
#         m["summary_time_sec"]     = m.get("summary_time_sec", 0)
#         m["summary_length_words"] = len(answer.split())
#     if m.get("type") == "qa":
#         m["qa_time_sec"]      = m.get("qa_time_sec", 0)
#         m["confidence_score"] = m.get("confidence_score", 0)

#     m["ttft_sec"]          = round(total_time, 2)
#     m["e2e_latency_sec"]   = round(total_time, 2)
#     m["tps"]               = tps
#     m["doc_type"]          = state.get("doc_type", "general")
#     m["query_type"]        = state.get("query_type", "")
#     m["chunks_created"]    = m.get("chunks_created", 0)
#     m["retry_count"]       = retry
#     m["model_used"]        = m.get("model_used", "llama_react")
#     m["llm_calls"]         = m.get("llm_calls", 0)
#     m["retrieval_score"]   = m.get("retrieval_score", 0)
#     m["context_precision"] = m.get("context_precision", 0)
#     m["answer_grounding"]  = m.get("answer_grounding", 0)
#     m["recall_at_k"]       = m.get("recall_at_k", 0)

#     if m.get("type") == "summary":
#         m["parallel_workers"] = m.get("parallel_workers", 0)
#         m["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
#         m["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)
#     if m.get("type") == "qa":
#         m["chunks_retrieved"] = m.get("chunks_retrieved", 0)
#         m["decision_type"]    = m.get("decision_type", "accepted")
#         m["confidence_raw"]   = m.get("confidence_raw", 0.0)

#     state["metrics"] = m
#     return state
























# #all good 100 for ethics  but 57 something for crop becuase it got overfitted towards onne pdf specific  so gonnna change 
# import time
# import numpy as np
# import re

# from docmind_rag.config.settings import (
#     MAX_WORKERS, FACTUAL_TOP_K, MULTIPART_TOP_K, SMALL_DOC_CHUNK_THRESHOLD
# )
# from docmind_rag.core.state import DocState
# from docmind_rag.core.prompts import MULTIPART_PROMPT
# from docmind_rag.models.llm import call_llama_streaming
# from docmind_rag.models.embeddings import build_faiss_index
# from docmind_rag.models.reranker import rerank_docs, protect_exact_matches
# from docmind_rag.services.extraction import extract_pdf_parallel, _extraction_cache
# from docmind_rag.services.chunking import semantic_chunk, raptor_summarize, _summary_cache
# from docmind_rag.services.retrieval import multi_query_retrieve
# from docmind_rag.services.agent import react_agent, roberta_qa
# from docmind_rag.events.events import emit_event
# from docmind_rag.utils.helpers import get_pdf_hash, clean_artifacts,clean_chunk_text
# from docmind_rag.utils.text import detect_doc_type, infer_answer_shape, shape_to_query_type, reorder_by_question
# from docmind_rag.utils.metrics import (
#     compute_answer_grounding,
#     compute_retrieval_score,
#     compute_context_precision,
#     compute_recall_at_k,
#     semantic_similarity,
#     exact_span_match
# )


# # ============================================================
# # LANGGRAPH NODES
# # ============================================================
# def node_extract(state: DocState) -> DocState:
#     extract_start = time.time()
#     text, page_count = extract_pdf_parallel(state["pdf_path"])
#     extract_time  = time.time() - extract_start
#     state["extracted_text"] = text
#     state["page_count"]     = page_count
#     state["char_count"]     = len(text)
#     state["metrics"]["extraction_time_sec"]  = round(extract_time, 2)
#     state["metrics"]["pages_processed"]      = page_count
#     state["metrics"]["characters_processed"] = len(text)
#     state["metrics"]["words_processed"]      = len(text.split())
#     return state


# def node_chunk(state: DocState) -> DocState:
#     text = state["extracted_text"]
#     state["doc_type"] = detect_doc_type(text)
#     # Keep for logging/debug ONLY — not for routing
#     shape = infer_answer_shape(state["question"])
#     state["metrics"]["answer_shape"] = shape

#     # 🚨 IMPORTANT: DO NOT let this control routing anymore
#     state["query_type"] = "QA"   # force all queries into QA pipeline



#     summary_chunks, rag_chunks = semantic_chunk(text)
#     state["summary_chunks"] = summary_chunks
#     state["chunks"]         = rag_chunks

#     state["metrics"]["summary_chunks"] = len(summary_chunks)
#     state["metrics"]["chunks_created"] = len(rag_chunks)
#     state["metrics"]["doc_type"]       = state["doc_type"]
#     state["metrics"]["query_type"]     = state["query_type"]
#     print(f"[Chunk] {len(summary_chunks)} summary chunks | {len(rag_chunks)} RAG chunks")
#     return state


# def node_summarize(state: DocState) -> DocState:
#     pdf_hash  = get_pdf_hash(state["pdf_path"])
#     cache_key = f"{pdf_hash}_{state['doc_type']}"
#     if cache_key in _summary_cache:
#         cached = _summary_cache[cache_key]
#         print(f"[Summary] ✅ Cache hit — returning saved summary instantly")
#         emit_event(state.get("request_id", ""), "agent_action",
#                    "⚡ Summary loaded from cache instantly!")
#         state["answer"] = cached["summary"]
#         state["metrics"].update(cached["metrics"])
#         state["metrics"]["type"] = "summary"
#         return state

#     summary_start = time.time()
#     raptor_summarize._request_id = state.get("request_id", "")
#     summary, map_time, reduce_time = raptor_summarize(
#         state["summary_chunks"], state["doc_type"])
#     summary_time = time.time() - summary_start

#     state["answer"] = summary
#     metrics_snapshot = {
#         "summary_time_sec":     round(summary_time, 2),
#         "summary_length_words": len(summary.split()),
#         "parallel_workers":     min(MAX_WORKERS, len(state["summary_chunks"])),
#         "map_time_sec":         round(map_time, 2),
#         "reduce_time_sec":      round(reduce_time, 2),
#         "llm_calls":            3,
#     }
#     state["metrics"].update(metrics_snapshot)
#     state["metrics"]["type"] = "summary"
#     _summary_cache[cache_key] = {"summary": summary, "metrics": metrics_snapshot}
#     print(f"[Summary] Done ({len(summary.split())} words)")
#     return state


# def node_qa(state: DocState) -> DocState:
#     qa_start_t = time.time()
#     question   = state["question"]
#     request_id = state.get("request_id", "")
#     all_chunks = state["chunks"]

#     pdf_hash    = get_pdf_hash(state["pdf_path"])
#     faiss_index = build_faiss_index(all_chunks, pdf_hash)

#     print(f"[QA] START | {len(all_chunks)} chunks")

#     recall_score      = 0.0
#     retrieval_score   = 0.0
#     context_precision = 0.0
#     grounding         = 0.0
#     llm_calls         = 0
#     model_used        = "llama"
#     confidence        = 0.0
#     decision_type     = "accepted"
#     retrieved         = []
#     retrieved_texts   = []
#     answer            = ""

#     # ── Refusal detector ─────────────────────────────────────
#     _REFUSAL_ANCHOR = "this information is not available in the provided context"

#     def is_refusal(text: str) -> bool:
#         if not text or len(text.strip()) < 2:
#             return True
#         sim = semantic_similarity(text, [_REFUSAL_ANCHOR])
#         print(f"[Refusal] sim={sim:.3f} | '{text[:50]}'")
#         return sim > 0.72

#     # ============================================================
#     # STEP 1: ALWAYS RETRIEVE FIRST (no routing yet)
#     # ============================================================
#     retrieved = multi_query_retrieve(
#         question, faiss_index, k=20,
#         all_chunks=all_chunks, query_type="FACTUAL_QA"
#     )

#     retrieved, reranker_top, all_scores = rerank_docs(
#         question, retrieved,
#         top_k=FACTUAL_TOP_K,
#         apply_pruning=True
#     )
#     # ============================================================
#     # 🔥 HARD STOP — NO RELEVANT CHUNK FOUND for lecture 23 remaining hallucinations 
#     # ============================================================

#     # if reranker_top < 0:
#     if reranker_top < -5:
#         print(f"[Guard] ❌ Negative reranker score ({reranker_top}) → NOT FOUND")

#         state["answer"] = "This information is not present in the document."
#         state["metrics"]["decision_type"] = "rejected_negative_reranker"
#         state["metrics"]["type"] = "qa"
#         return state

#     retrieved = protect_exact_matches(
#         question, retrieved, all_chunks,
#         top_k=FACTUAL_TOP_K
#     )

#     retrieved_texts = [d.page_content for d in retrieved]

#     retrieval_score   = compute_retrieval_score(question, retrieved)
#     context_precision = compute_context_precision(question, retrieved)
#     # ============================================================
#     # 🔥 CONSTRAINT GUARD (PREVENT HALLUCINATION)
#     # ============================================================

#     numbers = re.findall(r'\b\d+\b', question)

#     if numbers:
#         found = False

#         for chunk in retrieved_texts:
#             chunk_low = chunk.lower()

#             for num in numbers:
#                 if re.search(rf'\b{num}\b', chunk_low):
#                     found = True
#                     break

#             if found:
#                 break

#         if not found:
#             print("[Guard] ❌ Number constraint not found → NOT PRESENT")

#             state["answer"] = "This information is not present in the document."
#             state["metrics"]["decision_type"] = "not_found"
#             return state

#     # ============================================================
#     # STEP 2: CONTEXT-BASED ROUTING (KEY FIX)
#     # ============================================================


#     def detect_query_type_from_context(question, retrieved_texts):
#         """
#         Generalized routing based on evidence shape in retrieved chunks.
#         """

#         if not retrieved_texts:
#             return "REASONING_QA"

#         # Clean tokens (remove stopword noise)
#         q_words = set(re.findall(r'\b\w+\b', question.lower()))
#         q_words = {w for w in q_words if len(w) > 2}  # remove "is", "the"

#         scores = []

#         for chunk in retrieved_texts:
#             chunk_low = chunk.lower()

#             overlap = sum(1 for w in q_words if w in chunk_low)
#             length  = len(chunk.split())

#             scores.append((overlap, length, chunk))

#         # Sort by strongest overlap
#         scores.sort(reverse=True, key=lambda x: x[0])

#         top_overlap, top_len, _ = scores[0]

#         # 🔥 CASE 1 — strong single chunk → FACTUAL
#         if top_overlap >= 3 and top_len < 60:
#             return "FACTUAL_QA"

#         # 🔥 CASE 2 — multiple strong chunks → MULTIPART
#         strong_chunks = [s for s in scores if s[0] >= 2]

#         if len(strong_chunks) >= 3:
#             return "MULTIPART_QA"

#         # 🔥 CASE 3 — fallback → REASONING
#         return "REASONING_QA"

#     query_type = detect_query_type_from_context(question, retrieved_texts)
#     # # 🔥 FINAL OVERRIDE (STRONG FIX)
#     # numbers = re.findall(r'\b\d+\b', question)

#     # if numbers and len(question.split()) <= 8:
#     #     print("[Routing Override] Forcing FACTUAL_QA for numeric lookup")
#     #     query_type = "FACTUAL_QA"
#     print(f"[Routing] Context-based → {query_type}")
# # ============================================================
# # 🔥 FINAL GUARD (AFTER ROUTING — CRITICAL)
# # ============================================================

#     numbers = re.findall(r'\b\d+\b', question)

#     if numbers:
#         found = False

#         for chunk in retrieved_texts:
#             chunk_low = chunk.lower()

#             for num in numbers:
#                 if re.search(rf'\b{num}\b', chunk_low):
#                     found = True
#                     break

#             if found:
#                 break

#         if not found:
#             print("[Guard] ❌ Number not found in ANY retrieved chunk → stopping pipeline")

#             state["answer"] = "This information is not present in the document."
#             state["metrics"]["decision_type"] = "not_found"
#             return state
#     # ============================================================
#     # STEP 3: EXECUTION BASED ON ROUTING
#     # ============================================================

#     # ---------- MULTIPART ----------
#     if query_type == "MULTIPART_QA":

#         ranked = reorder_by_question(question, retrieved_texts)

#         top_k = max(MULTIPART_TOP_K, 8)   # ensures enough coverage
#         top_chunks = ranked[:top_k]
#         print("\n========== MULTIPART SOURCE CHUNKS ==========")
#         for i, chunk in enumerate(top_chunks):
#             preview = chunk[:300].replace("\n", " ")
#             print(f"\n--- Source {i+1} ---")
#             print(preview)
#         print("============================================\n")

#         context = "\n\n---\n\n".join(
#             clean_chunk_text(c) for c in top_chunks
#         )


#         emit_event(request_id, "agent_start",
#                    f"🤖 MULTIPART | {len(retrieved)} chunks")
#         emit_event(request_id, "stream_start", "✍️ Generating answer...")

#         answer = call_llama_streaming(
#             MULTIPART_PROMPT.format(context=context[:2500], question=question),
#             request_id, temperature=0.0
#         )

#         answer     = clean_artifacts(answer).strip().strip('"').strip("'")
#         model_used = "llama_multipart"
#         llm_calls  = 1

#     # ---------- FACTUAL / REASONING ----------
#     else:
#         react_ans, model_used, _, _, _, _, react_calls, retrieved_texts = react_agent(
#             question, faiss_index, query_type, all_chunks, request_id
#         )

#         llm_calls += react_calls
#         # react_ans = clean_artifacts(react_ans).strip().strip('"').strip("'")

#         # print(f"[QA] ReAct: '{react_ans[:60]}'")
#         react_ans = clean_artifacts(react_ans)


#         # react_ans = extract_final_answer(react_ans)
   
#         react_ans = react_ans.strip().strip('"').strip("'")

#         print(f"[QA] ReAct: '{react_ans[:60]}'")
#         # 🔥 NEW: grounding + recall based decision
#         grounding_score = compute_answer_grounding(react_ans, retrieved_texts, question)
#         # recall_score = retrieval_score   # your Recall@K proxy
#         recall_score = compute_recall_at_k(
#     question,
#     retrieved,
#     all_chunks,
#     k=len(retrieved)
# )

#         print(f"[QA] Grounding={grounding_score:.1f}% | Recall={recall_score:.1f}%")

#         # 🚨 Decision logic
#         if recall_score < 30:
#             print("[QA] ❌ Retrieval failed → rejecting")
#             react_ans = "NOT PRESENT"

#         elif grounding_score < 40:
#             print("[QA] ❌ Low grounding → rejecting")
#             react_ans = "NOT PRESENT"

#         elif grounding_score < 70:
#             print("[QA] ⚠️ Medium grounding → accepting with caution")
#             # keep answer (no change)

#         else:
#             print("[QA] ✅ Strong grounding → accepting")
        

#         # # Hallucination guard
#         # if not exact_span_match(react_ans, retrieved_texts):
#         #     print(f"[QA] ⚠️ Not grounded → '{react_ans[:60]}'")
#         #     react_ans = "NOT PRESENT"

#         if react_ans and react_ans != "NOT PRESENT" and not is_refusal(react_ans):
#             answer        = react_ans
#             model_used    = "llama_react"
#             decision_type = "accepted"
#         else:
#             rob_ans, rob_score = roberta_qa(question, retrieved_texts)
#             rob_ans = rob_ans.strip()

#             print(f"[QA] RoBERTa fallback: '{rob_ans[:40]}' score={rob_score:.3f}")

#             if rob_ans and rob_score >= 0.25 and not is_refusal(rob_ans):
#                 answer        = rob_ans
#                 model_used    = "roberta_fallback"
#                 decision_type = "accepted_roberta"
#             else:
#                 answer        = "This information is not present in the document."
#                 model_used    = "not_found"
#                 decision_type = "not_found"

#     # ============================================================
#     # STEP 4: METRICS (FIXED)
#     # ============================================================
#     grounding = compute_answer_grounding(answer, retrieved_texts, question)

#     confidence = round(
#         compute_answer_grounding(answer, retrieved_texts, question) / 100,
#         3
#     )

#     # recall_score = retrieval_score  # FIXED

#     qa_time = time.time() - qa_start_t

#     print(f"[QA] Done in {qa_time:.1f}s | model={model_used} | "
#           f"grounding={grounding:.1f}% | recall={recall_score:.1f}% | "
#           f"confidence={confidence:.3f} | decision={decision_type}")

#     if not answer.strip():
#         answer = "Could not find a relevant answer in the PDF."

#     state["answer"]                       = answer
#     state["metrics"]["qa_time_sec"]       = round(qa_time, 2)
#     state["metrics"]["confidence_score"]  = round(confidence * 100, 2)
#     state["metrics"]["retrieval_score"]   = retrieval_score
#     state["metrics"]["context_precision"] = context_precision
#     state["metrics"]["answer_grounding"]  = grounding
#     state["metrics"]["recall_at_k"]       = recall_score
#     state["metrics"]["llm_calls"]         = llm_calls
#     state["metrics"]["model_used"]        = model_used
#     state["metrics"]["chunks_retrieved"]  = len(retrieved)
#     state["metrics"]["type"]              = "qa"
#     state["metrics"]["decision_type"]     = decision_type
#     state["metrics"]["confidence_raw"]    = round(confidence, 4)

#     return state


# def node_validate(state: DocState) -> DocState:
#     answer = state["answer"]
#     retry  = state.get("retry_count", 0)
#     if len(answer.strip()) < 3 and retry < 2:
#         state["retry_count"] = retry + 1
#         state["answer"]      = ""
#         return state

#     total_time    = time.time() - state["start_time"]
#     output_words  = len(answer.split())
#     output_tokens = output_words * 1.3
#     extract_time  = state["metrics"].get("extraction_time_sec", 0)
#     llm_time      = max(total_time - extract_time, 1)
#     tps           = round(output_tokens / llm_time, 2) if llm_time > 0 else 0
#     m             = state["metrics"]

#     m["response_time_sec"]    = round(total_time, 2)
#     m["extraction_time_sec"]  = m.get("extraction_time_sec", 0)
#     m["pages_processed"]      = state.get("page_count", 0)
#     m["characters_processed"] = state.get("char_count", 0)
#     m["words_processed"]      = len(state.get("extracted_text", "").split())

#     if m.get("type") == "summary":
#         m["summary_time_sec"]     = m.get("summary_time_sec", 0)
#         m["summary_length_words"] = len(answer.split())
#     if m.get("type") == "qa":
#         m["qa_time_sec"]      = m.get("qa_time_sec", 0)
#         m["confidence_score"] = m.get("confidence_score", 0)

#     m["ttft_sec"]          = round(total_time, 2)
#     m["e2e_latency_sec"]   = round(total_time, 2)
#     m["tps"]               = tps
#     m["doc_type"]          = state.get("doc_type", "general")
#     m["query_type"]        = state.get("query_type", "")
#     m["chunks_created"]    = m.get("chunks_created", 0)
#     m["retry_count"]       = retry
#     m["model_used"]        = m.get("model_used", "llama_react")
#     m["llm_calls"]         = m.get("llm_calls", 0)
#     m["retrieval_score"]   = m.get("retrieval_score", 0)
#     m["context_precision"] = m.get("context_precision", 0)
#     m["answer_grounding"]  = m.get("answer_grounding", 0)
#     m["recall_at_k"]       = m.get("recall_at_k", 0)

#     if m.get("type") == "summary":
#         m["parallel_workers"] = m.get("parallel_workers", 0)
#         m["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
#         m["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)
#     if m.get("type") == "qa":
#         m["chunks_retrieved"] = m.get("chunks_retrieved", 0)
#         m["decision_type"]    = m.get("decision_type", "accepted")
#         m["confidence_raw"]   = m.get("confidence_raw", 0.0)

#     state["metrics"] = m
#     return state














































#all good but i want to make a change for  hallucinations one while aspects worked but lectures did not 
#  import time
# import numpy as np
# import re

# from docmind_rag.config.settings import (
#     MAX_WORKERS, FACTUAL_TOP_K, MULTIPART_TOP_K, SMALL_DOC_CHUNK_THRESHOLD
# )
# from docmind_rag.core.state import DocState
# from docmind_rag.core.prompts import MULTIPART_PROMPT
# from docmind_rag.models.llm import call_llama_streaming
# from docmind_rag.models.embeddings import build_faiss_index
# from docmind_rag.models.reranker import rerank_docs, protect_exact_matches
# from docmind_rag.services.extraction import extract_pdf_parallel, _extraction_cache
# from docmind_rag.services.chunking import semantic_chunk, raptor_summarize, _summary_cache
# from docmind_rag.services.retrieval import multi_query_retrieve
# from docmind_rag.services.agent import react_agent, roberta_qa
# from docmind_rag.events.events import emit_event
# from docmind_rag.utils.helpers import get_pdf_hash, clean_artifacts,clean_chunk_text
# from docmind_rag.utils.text import detect_doc_type, infer_answer_shape, shape_to_query_type, reorder_by_question
# from docmind_rag.utils.metrics import (
#     compute_answer_grounding,
#     compute_retrieval_score,
#     compute_context_precision,
#     compute_recall_at_k,
#     semantic_similarity,
#     exact_span_match
# )


# # ============================================================
# # LANGGRAPH NODES
# # ============================================================
# def node_extract(state: DocState) -> DocState:
#     extract_start = time.time()
#     text, page_count = extract_pdf_parallel(state["pdf_path"])
#     extract_time  = time.time() - extract_start
#     state["extracted_text"] = text
#     state["page_count"]     = page_count
#     state["char_count"]     = len(text)
#     state["metrics"]["extraction_time_sec"]  = round(extract_time, 2)
#     state["metrics"]["pages_processed"]      = page_count
#     state["metrics"]["characters_processed"] = len(text)
#     state["metrics"]["words_processed"]      = len(text.split())
#     return state


# def node_chunk(state: DocState) -> DocState:
#     text = state["extracted_text"]
#     state["doc_type"] = detect_doc_type(text)
#     # Keep for logging/debug ONLY — not for routing
#     shape = infer_answer_shape(state["question"])
#     state["metrics"]["answer_shape"] = shape

#     # 🚨 IMPORTANT: DO NOT let this control routing anymore
#     state["query_type"] = "QA"   # force all queries into QA pipeline



#     summary_chunks, rag_chunks = semantic_chunk(text)
#     state["summary_chunks"] = summary_chunks
#     state["chunks"]         = rag_chunks

#     state["metrics"]["summary_chunks"] = len(summary_chunks)
#     state["metrics"]["chunks_created"] = len(rag_chunks)
#     state["metrics"]["doc_type"]       = state["doc_type"]
#     state["metrics"]["query_type"]     = state["query_type"]
#     print(f"[Chunk] {len(summary_chunks)} summary chunks | {len(rag_chunks)} RAG chunks")
#     return state


# def node_summarize(state: DocState) -> DocState:
#     pdf_hash  = get_pdf_hash(state["pdf_path"])
#     cache_key = f"{pdf_hash}_{state['doc_type']}"
#     if cache_key in _summary_cache:
#         cached = _summary_cache[cache_key]
#         print(f"[Summary] ✅ Cache hit — returning saved summary instantly")
#         emit_event(state.get("request_id", ""), "agent_action",
#                    "⚡ Summary loaded from cache instantly!")
#         state["answer"] = cached["summary"]
#         state["metrics"].update(cached["metrics"])
#         state["metrics"]["type"] = "summary"
#         return state

#     summary_start = time.time()
#     raptor_summarize._request_id = state.get("request_id", "")
#     summary, map_time, reduce_time = raptor_summarize(
#         state["summary_chunks"], state["doc_type"])
#     summary_time = time.time() - summary_start

#     state["answer"] = summary
#     metrics_snapshot = {
#         "summary_time_sec":     round(summary_time, 2),
#         "summary_length_words": len(summary.split()),
#         "parallel_workers":     min(MAX_WORKERS, len(state["summary_chunks"])),
#         "map_time_sec":         round(map_time, 2),
#         "reduce_time_sec":      round(reduce_time, 2),
#         "llm_calls":            3,
#     }
#     state["metrics"].update(metrics_snapshot)
#     state["metrics"]["type"] = "summary"
#     _summary_cache[cache_key] = {"summary": summary, "metrics": metrics_snapshot}
#     print(f"[Summary] Done ({len(summary.split())} words)")
#     return state


# def node_qa(state: DocState) -> DocState:
#     qa_start_t = time.time()
#     question   = state["question"]
#     request_id = state.get("request_id", "")
#     all_chunks = state["chunks"]

#     pdf_hash    = get_pdf_hash(state["pdf_path"])
#     faiss_index = build_faiss_index(all_chunks, pdf_hash)

#     print(f"[QA] START | {len(all_chunks)} chunks")

#     recall_score      = 0.0
#     retrieval_score   = 0.0
#     context_precision = 0.0
#     grounding         = 0.0
#     llm_calls         = 0
#     model_used        = "llama"
#     confidence        = 0.0
#     decision_type     = "accepted"
#     retrieved         = []
#     retrieved_texts   = []
#     answer            = ""

#     # ── Refusal detector ─────────────────────────────────────
#     _REFUSAL_ANCHOR = "this information is not available in the provided context"

#     def is_refusal(text: str) -> bool:
#         if not text or len(text.strip()) < 2:
#             return True
#         sim = semantic_similarity(text, [_REFUSAL_ANCHOR])
#         print(f"[Refusal] sim={sim:.3f} | '{text[:50]}'")
#         return sim > 0.72

#     # ============================================================
#     # STEP 1: ALWAYS RETRIEVE FIRST (no routing yet)
#     # ============================================================
#     retrieved = multi_query_retrieve(
#         question, faiss_index, k=20,
#         all_chunks=all_chunks, query_type="FACTUAL_QA"
#     )

#     retrieved, reranker_top, all_scores = rerank_docs(
#         question, retrieved,
#         top_k=FACTUAL_TOP_K,
#         apply_pruning=True
#     )

#     retrieved = protect_exact_matches(
#         question, retrieved, all_chunks,
#         top_k=FACTUAL_TOP_K
#     )

#     retrieved_texts = [d.page_content for d in retrieved]

#     retrieval_score   = compute_retrieval_score(question, retrieved)
#     context_precision = compute_context_precision(question, retrieved)

#     # ============================================================
#     # STEP 2: CONTEXT-BASED ROUTING (KEY FIX)
#     # ============================================================


#     def detect_query_type_from_context(question, retrieved_texts):
#         """
#         Generalized routing based on evidence shape in retrieved chunks.
#         """

#         if not retrieved_texts:
#             return "REASONING_QA"

#         # Clean tokens (remove stopword noise)
#         q_words = set(re.findall(r'\b\w+\b', question.lower()))
#         q_words = {w for w in q_words if len(w) > 2}  # remove "is", "the"

#         scores = []

#         for chunk in retrieved_texts:
#             chunk_low = chunk.lower()

#             overlap = sum(1 for w in q_words if w in chunk_low)
#             length  = len(chunk.split())

#             scores.append((overlap, length, chunk))

#         # Sort by strongest overlap
#         scores.sort(reverse=True, key=lambda x: x[0])

#         top_overlap, top_len, _ = scores[0]

#         # 🔥 CASE 1 — strong single chunk → FACTUAL
#         if top_overlap >= 3 and top_len < 60:
#             return "FACTUAL_QA"

#         # 🔥 CASE 2 — multiple strong chunks → MULTIPART
#         strong_chunks = [s for s in scores if s[0] >= 2]

#         if len(strong_chunks) >= 3:
#             return "MULTIPART_QA"

#         # 🔥 CASE 3 — fallback → REASONING
#         return "REASONING_QA"

#     query_type = detect_query_type_from_context(question, retrieved_texts)
#     # # 🔥 FINAL OVERRIDE (STRONG FIX)
#     # numbers = re.findall(r'\b\d+\b', question)

#     # if numbers and len(question.split()) <= 8:
#     #     print("[Routing Override] Forcing FACTUAL_QA for numeric lookup")
#     #     query_type = "FACTUAL_QA"
#     print(f"[Routing] Context-based → {query_type}")

#     # ============================================================
#     # STEP 3: EXECUTION BASED ON ROUTING
#     # ============================================================

#     # ---------- MULTIPART ----------
#     if query_type == "MULTIPART_QA":

#         ranked = reorder_by_question(question, retrieved_texts)

#         top_k = max(MULTIPART_TOP_K, 8)   # ensures enough coverage
#         top_chunks = ranked[:top_k]
#         print("\n========== MULTIPART SOURCE CHUNKS ==========")
#         for i, chunk in enumerate(top_chunks):
#             preview = chunk[:300].replace("\n", " ")
#             print(f"\n--- Source {i+1} ---")
#             print(preview)
#         print("============================================\n")

#         context = "\n\n---\n\n".join(
#             clean_chunk_text(c) for c in top_chunks
#         )


#         emit_event(request_id, "agent_start",
#                    f"🤖 MULTIPART | {len(retrieved)} chunks")
#         emit_event(request_id, "stream_start", "✍️ Generating answer...")

#         answer = call_llama_streaming(
#             MULTIPART_PROMPT.format(context=context[:2500], question=question),
#             request_id, temperature=0.0
#         )

#         answer     = clean_artifacts(answer).strip().strip('"').strip("'")
#         model_used = "llama_multipart"
#         llm_calls  = 1

#     # ---------- FACTUAL / REASONING ----------
#     else:
#         react_ans, model_used, _, _, _, _, react_calls, retrieved_texts = react_agent(
#             question, faiss_index, query_type, all_chunks, request_id
#         )

#         llm_calls += react_calls
#         # react_ans = clean_artifacts(react_ans).strip().strip('"').strip("'")

#         # print(f"[QA] ReAct: '{react_ans[:60]}'")
#         react_ans = clean_artifacts(react_ans)


#         # react_ans = extract_final_answer(react_ans)
   
#         react_ans = react_ans.strip().strip('"').strip("'")

#         print(f"[QA] ReAct: '{react_ans[:60]}'")
#         # 🔥 NEW: grounding + recall based decision
#         grounding_score = compute_answer_grounding(react_ans, retrieved_texts, question)
#         # recall_score = retrieval_score   # your Recall@K proxy
#         recall_score = compute_recall_at_k(
#     question,
#     retrieved,
#     all_chunks,
#     k=len(retrieved)
# )

#         print(f"[QA] Grounding={grounding_score:.1f}% | Recall={recall_score:.1f}%")

#         # 🚨 Decision logic
#         if recall_score < 30:
#             print("[QA] ❌ Retrieval failed → rejecting")
#             react_ans = "NOT PRESENT"

#         elif grounding_score < 40:
#             print("[QA] ❌ Low grounding → rejecting")
#             react_ans = "NOT PRESENT"

#         elif grounding_score < 70:
#             print("[QA] ⚠️ Medium grounding → accepting with caution")
#             # keep answer (no change)

#         else:
#             print("[QA] ✅ Strong grounding → accepting")
        

#         # # Hallucination guard
#         # if not exact_span_match(react_ans, retrieved_texts):
#         #     print(f"[QA] ⚠️ Not grounded → '{react_ans[:60]}'")
#         #     react_ans = "NOT PRESENT"

#         if react_ans and react_ans != "NOT PRESENT" and not is_refusal(react_ans):
#             answer        = react_ans
#             model_used    = "llama_react"
#             decision_type = "accepted"
#         else:
#             rob_ans, rob_score = roberta_qa(question, retrieved_texts)
#             rob_ans = rob_ans.strip()

#             print(f"[QA] RoBERTa fallback: '{rob_ans[:40]}' score={rob_score:.3f}")

#             if rob_ans and rob_score >= 0.25 and not is_refusal(rob_ans):
#                 answer        = rob_ans
#                 model_used    = "roberta_fallback"
#                 decision_type = "accepted_roberta"
#             else:
#                 answer        = "This information is not present in the document."
#                 model_used    = "not_found"
#                 decision_type = "not_found"

#     # ============================================================
#     # STEP 4: METRICS (FIXED)
#     # ============================================================
#     grounding = compute_answer_grounding(answer, retrieved_texts, question)

#     confidence = round(
#         compute_answer_grounding(answer, retrieved_texts, question) / 100,
#         3
#     )

#     # recall_score = retrieval_score  # FIXED

#     qa_time = time.time() - qa_start_t

#     print(f"[QA] Done in {qa_time:.1f}s | model={model_used} | "
#           f"grounding={grounding:.1f}% | recall={recall_score:.1f}% | "
#           f"confidence={confidence:.3f} | decision={decision_type}")

#     if not answer.strip():
#         answer = "Could not find a relevant answer in the PDF."

#     state["answer"]                       = answer
#     state["metrics"]["qa_time_sec"]       = round(qa_time, 2)
#     state["metrics"]["confidence_score"]  = round(confidence * 100, 2)
#     state["metrics"]["retrieval_score"]   = retrieval_score
#     state["metrics"]["context_precision"] = context_precision
#     state["metrics"]["answer_grounding"]  = grounding
#     state["metrics"]["recall_at_k"]       = recall_score
#     state["metrics"]["llm_calls"]         = llm_calls
#     state["metrics"]["model_used"]        = model_used
#     state["metrics"]["chunks_retrieved"]  = len(retrieved)
#     state["metrics"]["type"]              = "qa"
#     state["metrics"]["decision_type"]     = decision_type
#     state["metrics"]["confidence_raw"]    = round(confidence, 4)

#     return state


# def node_validate(state: DocState) -> DocState:
#     answer = state["answer"]
#     retry  = state.get("retry_count", 0)
#     if len(answer.strip()) < 3 and retry < 2:
#         state["retry_count"] = retry + 1
#         state["answer"]      = ""
#         return state

#     total_time    = time.time() - state["start_time"]
#     output_words  = len(answer.split())
#     output_tokens = output_words * 1.3
#     extract_time  = state["metrics"].get("extraction_time_sec", 0)
#     llm_time      = max(total_time - extract_time, 1)
#     tps           = round(output_tokens / llm_time, 2) if llm_time > 0 else 0
#     m             = state["metrics"]

#     m["response_time_sec"]    = round(total_time, 2)
#     m["extraction_time_sec"]  = m.get("extraction_time_sec", 0)
#     m["pages_processed"]      = state.get("page_count", 0)
#     m["characters_processed"] = state.get("char_count", 0)
#     m["words_processed"]      = len(state.get("extracted_text", "").split())

#     if m.get("type") == "summary":
#         m["summary_time_sec"]     = m.get("summary_time_sec", 0)
#         m["summary_length_words"] = len(answer.split())
#     if m.get("type") == "qa":
#         m["qa_time_sec"]      = m.get("qa_time_sec", 0)
#         m["confidence_score"] = m.get("confidence_score", 0)

#     m["ttft_sec"]          = round(total_time, 2)
#     m["e2e_latency_sec"]   = round(total_time, 2)
#     m["tps"]               = tps
#     m["doc_type"]          = state.get("doc_type", "general")
#     m["query_type"]        = state.get("query_type", "")
#     m["chunks_created"]    = m.get("chunks_created", 0)
#     m["retry_count"]       = retry
#     m["model_used"]        = m.get("model_used", "llama_react")
#     m["llm_calls"]         = m.get("llm_calls", 0)
#     m["retrieval_score"]   = m.get("retrieval_score", 0)
#     m["context_precision"] = m.get("context_precision", 0)
#     m["answer_grounding"]  = m.get("answer_grounding", 0)
#     m["recall_at_k"]       = m.get("recall_at_k", 0)

#     if m.get("type") == "summary":
#         m["parallel_workers"] = m.get("parallel_workers", 0)
#         m["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
#         m["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)
#     if m.get("type") == "qa":
#         m["chunks_retrieved"] = m.get("chunks_retrieved", 0)
#         m["decision_type"]    = m.get("decision_type", "accepted")
#         m["confidence_raw"]   = m.get("confidence_raw", 0.0)

#     state["metrics"] = m
#     return state























# # 
# # when i akses name of lecture 23 it gave crt but when the queston is this What is the name or topic of Lecture 23?ist is  giving worng , but doenot who is culprit here  ?

# Now this is a good, precise question 👍 — let’s answer it cleanly.

# 🎯 Your observation
# Question	Result
# “name of lecture 23”	✅ correct
# “name or topic of lecture 23”	❌ wrong
# 🧠 Who is the culprit?

# 👉 NOT retrieval
# 👉 NOT chunking
# 👉 NOT reranker

# 👉 ✅ Culprit = Query Routing (Shape / Intent classification)

# 🔍 Why exactly this happens
# Case 1 (works):
# "What is the name of Lecture 23?"

# 👉 Classified as:

# FACTUAL_QA ✅

# 👉 Pipeline:

# retrieve → extract → correct answer
# Case 2 (fails):
# "What is the name or topic of Lecture 23?"

# 👉 LLM sees:

# "topic" → semantic bias → summary-like

# 👉 Classified as:

# SUMMARY ❌

# 👉 Pipeline:

# RAPTOR → summarize → wrong answer
# 💥 So the root cause

# LLM-based classifier is over-interpreting the word "topic"

# 👉 It thinks:

# "topic" = explain/summarize
# but here it's actually label/title lookup
# 🧠 Important insight

# This is NOT about:

# PDF ❌
# data ❌

# 👉 It is about:

# semantic ambiguity in natural language

# 🔥 Why your system fails here

# Because routing is happening:

# question → LLM → pipeline

# 👉 without grounding in actual document

# 🎯 Correct reasoning (what SHOULD happen)

# This question:

# "What is the name or topic of Lecture 23?"

# 👉 is still:

# asking for a label/title
# NOT asking for explanation

# 👉 So it should go to:

# FACTUAL_QA
# 🧠 So who is guilty?

# 👉 The classifier (Shape / infer_answer_shape)

# ⚠️ NOT these components
# Component	Status
# Retrieval	✅ correct
# Reranker	✅ correct
# Agent	✅ correct
# Prompt	✅ correct
# 🚀 How to fix (clean approach, no hacks)

# You have 2 proper options:

# ✅ Option 1 (recommended)

# 👉 Move routing AFTER retrieval

# As I explained earlier:

# retrieve first
# check if exact match exists
# then decide
# # 
# # 
# 
# 
# 
# 

# import time
# import numpy as np

# from docmind_rag.config.settings import (
#     MAX_WORKERS, FACTUAL_TOP_K, MULTIPART_TOP_K, SMALL_DOC_CHUNK_THRESHOLD
# )
# from docmind_rag.core.state import DocState
# from docmind_rag.core.prompts import MULTIPART_PROMPT
# from docmind_rag.models.llm import call_llama_streaming
# from docmind_rag.models.embeddings import build_faiss_index
# from docmind_rag.models.reranker import rerank_docs, protect_exact_matches
# from docmind_rag.services.extraction import extract_pdf_parallel, _extraction_cache
# from docmind_rag.services.chunking import semantic_chunk, raptor_summarize, _summary_cache
# from docmind_rag.services.retrieval import multi_query_retrieve
# from docmind_rag.services.agent import react_agent, roberta_qa
# from docmind_rag.events.events import emit_event
# from docmind_rag.utils.helpers import get_pdf_hash, clean_artifacts
# from docmind_rag.utils.text import detect_doc_type, infer_answer_shape, shape_to_query_type, reorder_by_question
# from docmind_rag.utils.metrics import (
#     compute_answer_grounding,
#     compute_retrieval_score,
#     compute_context_precision,
#     compute_recall_at_k,
#     semantic_similarity,
#     exact_span_match
# )


# # ============================================================
# # LANGGRAPH NODES
# # ============================================================
# def node_extract(state: DocState) -> DocState:
#     extract_start = time.time()
#     text, page_count = extract_pdf_parallel(state["pdf_path"])
#     extract_time  = time.time() - extract_start
#     state["extracted_text"] = text
#     state["page_count"]     = page_count
#     state["char_count"]     = len(text)
#     state["metrics"]["extraction_time_sec"]  = round(extract_time, 2)
#     state["metrics"]["pages_processed"]      = page_count
#     state["metrics"]["characters_processed"] = len(text)
#     state["metrics"]["words_processed"]      = len(text.split())
#     return state


# def node_chunk(state: DocState) -> DocState:
#     text = state["extracted_text"]
#     state["doc_type"] = detect_doc_type(text)

#     shape = infer_answer_shape(state["question"])
#     state["query_type"] = shape_to_query_type(shape)
#     state["metrics"]["answer_shape"] = shape

#     summary_chunks, rag_chunks = semantic_chunk(text)
#     state["summary_chunks"] = summary_chunks
#     state["chunks"]         = rag_chunks

#     state["metrics"]["summary_chunks"] = len(summary_chunks)
#     state["metrics"]["chunks_created"] = len(rag_chunks)
#     state["metrics"]["doc_type"]       = state["doc_type"]
#     state["metrics"]["query_type"]     = state["query_type"]
#     print(f"[Chunk] {len(summary_chunks)} summary chunks | {len(rag_chunks)} RAG chunks")
#     return state


# def node_summarize(state: DocState) -> DocState:
#     pdf_hash  = get_pdf_hash(state["pdf_path"])
#     cache_key = f"{pdf_hash}_{state['doc_type']}"
#     if cache_key in _summary_cache:
#         cached = _summary_cache[cache_key]
#         print(f"[Summary] ✅ Cache hit — returning saved summary instantly")
#         emit_event(state.get("request_id", ""), "agent_action",
#                    "⚡ Summary loaded from cache instantly!")
#         state["answer"] = cached["summary"]
#         state["metrics"].update(cached["metrics"])
#         state["metrics"]["type"] = "summary"
#         return state

#     summary_start = time.time()
#     raptor_summarize._request_id = state.get("request_id", "")
#     summary, map_time, reduce_time = raptor_summarize(
#         state["summary_chunks"], state["doc_type"])
#     summary_time = time.time() - summary_start

#     state["answer"] = summary
#     metrics_snapshot = {
#         "summary_time_sec":     round(summary_time, 2),
#         "summary_length_words": len(summary.split()),
#         "parallel_workers":     min(MAX_WORKERS, len(state["summary_chunks"])),
#         "map_time_sec":         round(map_time, 2),
#         "reduce_time_sec":      round(reduce_time, 2),
#         "llm_calls":            3,
#     }
#     state["metrics"].update(metrics_snapshot)
#     state["metrics"]["type"] = "summary"
#     _summary_cache[cache_key] = {"summary": summary, "metrics": metrics_snapshot}
#     print(f"[Summary] Done ({len(summary.split())} words)")
#     return state


# def node_qa(state: DocState) -> DocState:
#     qa_start_t = time.time()
#     question   = state["question"]
#     query_type = state["query_type"]
#     request_id = state.get("request_id", "")
#     all_chunks = state["chunks"]

#     pdf_hash    = get_pdf_hash(state["pdf_path"])
#     faiss_index = build_faiss_index(all_chunks, pdf_hash)

#     print(f"[QA] {query_type} | {len(all_chunks)} chunks")

#     recall_score      = 0.0
#     retrieval_score   = 0.0
#     context_precision = 0.0
#     grounding         = 0.0
#     llm_calls         = 0
#     model_used        = "llama"
#     confidence        = 0.0
#     decision_type     = "accepted"
#     retrieved         = []
#     retrieved_texts   = []
#     answer            = ""

#     # ── Refusal detector — semantic, no hardcoded phrases ────────────────────
#     _REFUSAL_ANCHOR = "this information is not available in the provided context"

#     def is_refusal(text: str) -> bool:
#         if not text or len(text.strip()) < 2:
#             return True
#         sim = semantic_similarity(text, [_REFUSAL_ANCHOR])
#         print(f"[Refusal] sim={sim:.3f} | '{text[:50]}'")
#         return sim > 0.72

#     # ============================================================
#     # MULTIPART PATH
#     # ============================================================
#     if query_type == "MULTIPART_QA":
#         retrieved = multi_query_retrieve(question, faiss_index, k=20,
#                                          all_chunks=all_chunks, query_type=query_type)
#         retrieved, reranker_top, all_scores = rerank_docs(question, retrieved,
#                                                            top_k=MULTIPART_TOP_K,
#                                                            apply_pruning=False)
#         retrieved       = protect_exact_matches(question, retrieved, all_chunks,
#                                                 top_k=MULTIPART_TOP_K)
#         retrieved_texts = [d.page_content for d in retrieved]

#         retrieval_score   = compute_retrieval_score(question, retrieved)
#         context_precision = compute_context_precision(question, retrieved)
#         recall_score      = compute_recall_at_k(question, retrieved, all_chunks,
#                                                 k=len(retrieved))
#         context = "\n\n---\n\n".join(
#             reorder_by_question(question, retrieved_texts)[:MULTIPART_TOP_K])

#         emit_event(request_id, "agent_start",
#                    f"🤖 MULTIPART | {len(retrieved)} chunks retrieved")
#         emit_event(request_id, "stream_start", "✍️ Generating answer...")

#         answer = call_llama_streaming(
#             MULTIPART_PROMPT.format(context=context[:2500], question=question),
#             request_id, temperature=0.0)

#         answer     = clean_artifacts(answer).strip().strip('"').strip("'")
#         model_used = "llama_multipart"
#         llm_calls  = 1
#         grounding  = compute_answer_grounding(answer, retrieved_texts, question)
#         # confidence = compute_confidence(reranker_top, recall_score,
#         #                                 answer, retrieved_texts, all_scores)
#         confidence = round(
#      compute_answer_grounding(answer, retrieved_texts, question) / 100,
#      3)
#         decision_type = "accepted"
#         emit_event(request_id, "agent_done", "✅ Done!")
#         print(f"[QA] MULTIPART done | grounding={grounding:.1f}% | recall={recall_score:.1f}%")

#     # ============================================================
#     # FACTUAL / REASONING PATH
#     # ============================================================
#     else:
#         react_ans, model_used, _, retrieval_score, context_precision, grounding, react_calls, retrieved_texts = react_agent(
#             question, faiss_index, query_type, all_chunks, request_id)
#         llm_calls += react_calls

#         react_ans = clean_artifacts(react_ans).strip().strip('"').strip("'")
#         print(f"[QA] ReAct: '{react_ans[:60]}'")

#         # Hallucination check — answer must be grounded in context
#         if not exact_span_match(react_ans, retrieved_texts):
#             print(f"[QA] ⚠️ Not grounded in context — possible hallucination: '{react_ans[:60]}'")
#             react_ans = "NOT PRESENT"

#         if react_ans and react_ans != "NOT PRESENT" and not is_refusal(react_ans):
#             answer        = react_ans
#             model_used    = "llama_react"
#             decision_type = "accepted"
#         else:
#             rob_ans, rob_score = roberta_qa(question, retrieved_texts)
#             rob_ans = rob_ans.strip()
#             print(f"[QA] ReAct refused — RoBERTa: '{rob_ans[:40]}' score={rob_score:.3f}")

#             if rob_ans and rob_score >= 0.25 and not is_refusal(rob_ans):
#                 answer        = rob_ans
#                 model_used    = "roberta_fallback"
#                 decision_type = "accepted_roberta"
#             else:
#                 answer        = "This information is not present in the document."
#                 model_used    = "not_found"
#                 decision_type = "not_found"

#         grounding  = compute_answer_grounding(answer, retrieved_texts, question)
#         confidence = compute_answer_grounding(answer, retrieved_texts, question)

#         # ── Step 2: RoBERTa fallback if ReAct refused ─────────────────────────
#         if not is_refusal(react_ans):
#             answer     = react_ans
#             model_used = "llama_react"
#             decision_type = "accepted"
#         else:
#             rob_ans, rob_score = roberta_qa(question, retrieved_texts)
#             rob_ans = rob_ans.strip()
#             print(f"[QA] ReAct refused — trying RoBERTa: '{rob_ans[:40]}' (score={rob_score:.3f})")

#             if rob_ans and rob_score >= 0.25 and not is_refusal(rob_ans):
#                 answer        = rob_ans
#                 model_used    = "roberta_fallback"
#                 decision_type = "accepted_roberta"
#             else:
#                 answer        = "This information is not present in the document."
#                 model_used    = "not_found"
#                 decision_type = "not_found"

#         grounding  = compute_answer_grounding(answer, retrieved_texts, question)
#         # confidence = compute_confidence(reranker_top, recall_score,
#         #                                 answer, retrieved_texts, all_scores)
#         confidence = round(
#         compute_answer_grounding(answer, retrieved_texts, question) / 100,
#         3)
#         recall_score = retrieval_score
       

#     # ── Final metrics ─────────────────────────────────────────────────────────
#     qa_time = time.time() - qa_start_t
#     print(f"[QA] Done in {qa_time:.1f}s | model={model_used} | "
#           f"grounding={grounding:.1f}% | recall={recall_score:.1f}% | "
#           f"confidence={confidence:.3f} | decision={decision_type}")

#     if not answer.strip():
#         answer = "Could not find a relevant answer in the PDF."

#     state["answer"]                       = answer
#     state["metrics"]["qa_time_sec"]       = round(qa_time, 2)
#     state["metrics"]["confidence_score"]  = round(confidence * 100, 2)
#     state["metrics"]["retrieval_score"]   = retrieval_score
#     state["metrics"]["context_precision"] = context_precision
#     state["metrics"]["answer_grounding"]  = grounding
#     state["metrics"]["recall_at_k"]       = recall_score
#     state["metrics"]["llm_calls"]         = llm_calls
#     state["metrics"]["model_used"]        = model_used
#     state["metrics"]["chunks_retrieved"]  = len(retrieved)
#     state["metrics"]["type"]              = "qa"
#     state["metrics"]["decision_type"]     = decision_type
#     state["metrics"]["confidence_raw"]    = round(confidence, 4)
#     return state


# def node_validate(state: DocState) -> DocState:
#     answer = state["answer"]
#     retry  = state.get("retry_count", 0)
#     if len(answer.strip()) < 3 and retry < 2:
#         state["retry_count"] = retry + 1
#         state["answer"]      = ""
#         return state

#     total_time    = time.time() - state["start_time"]
#     output_words  = len(answer.split())
#     output_tokens = output_words * 1.3
#     extract_time  = state["metrics"].get("extraction_time_sec", 0)
#     llm_time      = max(total_time - extract_time, 1)
#     tps           = round(output_tokens / llm_time, 2) if llm_time > 0 else 0
#     m             = state["metrics"]

#     m["response_time_sec"]    = round(total_time, 2)
#     m["extraction_time_sec"]  = m.get("extraction_time_sec", 0)
#     m["pages_processed"]      = state.get("page_count", 0)
#     m["characters_processed"] = state.get("char_count", 0)
#     m["words_processed"]      = len(state.get("extracted_text", "").split())

#     if m.get("type") == "summary":
#         m["summary_time_sec"]     = m.get("summary_time_sec", 0)
#         m["summary_length_words"] = len(answer.split())
#     if m.get("type") == "qa":
#         m["qa_time_sec"]      = m.get("qa_time_sec", 0)
#         m["confidence_score"] = m.get("confidence_score", 0)

#     m["ttft_sec"]          = round(total_time, 2)
#     m["e2e_latency_sec"]   = round(total_time, 2)
#     m["tps"]               = tps
#     m["doc_type"]          = state.get("doc_type", "general")
#     m["query_type"]        = state.get("query_type", "")
#     m["chunks_created"]    = m.get("chunks_created", 0)
#     m["retry_count"]       = retry
#     m["model_used"]        = m.get("model_used", "llama_react")
#     m["llm_calls"]         = m.get("llm_calls", 0)
#     m["retrieval_score"]   = m.get("retrieval_score", 0)
#     m["context_precision"] = m.get("context_precision", 0)
#     m["answer_grounding"]  = m.get("answer_grounding", 0)
#     m["recall_at_k"]       = m.get("recall_at_k", 0)

#     if m.get("type") == "summary":
#         m["parallel_workers"] = m.get("parallel_workers", 0)
#         m["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
#         m["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)
#     if m.get("type") == "qa":
#         m["chunks_retrieved"] = m.get("chunks_retrieved", 0)
#         m["decision_type"]    = m.get("decision_type", "accepted")
#         m["confidence_raw"]   = m.get("confidence_raw", 0.0)

#     state["metrics"] = m
#     return state
























# import re

# import time

# import numpy as np

# from docmind_rag.config.settings import (
#     MAX_WORKERS, FACTUAL_TOP_K, MULTIPART_TOP_K, SMALL_DOC_CHUNK_THRESHOLD
# )
# from docmind_rag.core.state import DocState
# from docmind_rag.core.prompts import MULTIPART_PROMPT, REASONING_PROMPT, FACTUAL_EXTRACT_PROMPT
# from docmind_rag.models.llm import call_llama_streaming
# from docmind_rag.models.embeddings import build_faiss_index
# from docmind_rag.models.reranker import rerank_docs, protect_exact_matches
# from docmind_rag.services.extraction import extract_pdf_parallel, _extraction_cache
# from docmind_rag.services.chunking import semantic_chunk, raptor_summarize, _summary_cache
# from docmind_rag.services.retrieval import multi_query_retrieve
# from docmind_rag.services.agent import react_agent, roberta_qa
# from docmind_rag.events.events import emit_event
# from docmind_rag.utils.helpers import get_pdf_hash, clean_artifacts, _get_dynamic_stopwords
# from docmind_rag.utils.text import detect_doc_type, infer_answer_shape, shape_to_query_type, reorder_by_question
# from docmind_rag.utils.metrics import (
#     compute_answer_grounding, compute_retrieval_score, compute_context_precision,
#     compute_recall_at_k, compute_confidence, compute_grounding_score,
#     semantic_similarity, keyword_overlap, local_substring_match,
#     score_answer, coverage_score, llm_verify_answer
# )
# def extract_answer_candidates(question, chunks):
#     candidates = []
#     q_tokens = set(re.findall(r'\b\w+\b', question.lower()))

#     for chunk in chunks:
#         sentences = re.split(r'(?<=[.!?\n])', chunk)

#         for sent in sentences:
#             sent_clean = sent.strip()
#             sent_lower = sent_clean.lower()

#             if len(sent_clean.split()) < 3:
#                 continue

#             overlap = sum(1 for w in q_tokens if w in sent_lower)

#             if overlap >= 2:
#                 candidates.append(sent_clean)

#     return list(set(candidates))
# def is_valid_answer(ans, question):
#     if not ans or len(ans.split()) < 2:
#         return False

#     q_tokens = set(re.findall(r'\b\w+\b', question.lower()))
#     a_tokens = set(re.findall(r'\b\w+\b', ans.lower()))

#     # reject echo
#     if a_tokens.issubset(q_tokens):
#         return False

#     # reject refusal
#     bad = ["cannot", "not present", "not available", "insufficient"]
#     if any(b in ans.lower() for b in bad):
#         return False

#     return True
# # ============================================================
# # LANGGRAPH NODES
# # ============================================================
# def node_extract(state: DocState) -> DocState:
#     extract_start = time.time()
#     text, page_count = extract_pdf_parallel(state["pdf_path"])
#     extract_time  = time.time() - extract_start
#     state["extracted_text"] = text
#     state["page_count"]     = page_count
#     state["char_count"]     = len(text)
#     state["metrics"]["extraction_time_sec"]  = round(extract_time, 2)
#     state["metrics"]["pages_processed"]      = page_count
#     state["metrics"]["characters_processed"] = len(text)
#     state["metrics"]["words_processed"]      = len(text.split())
#     return state


# def node_chunk(state: DocState) -> DocState:
#     text = state["extracted_text"]
#     state["doc_type"] = detect_doc_type(text)

#     shape = infer_answer_shape(state["question"])
#     state["query_type"] = shape_to_query_type(shape)
#     state["metrics"]["answer_shape"] = shape

#     summary_chunks, rag_chunks = semantic_chunk(text)
#     state["summary_chunks"] = summary_chunks
#     state["chunks"]         = rag_chunks

#     state["metrics"]["summary_chunks"] = len(summary_chunks)
#     state["metrics"]["chunks_created"] = len(rag_chunks)
#     state["metrics"]["doc_type"]       = state["doc_type"]
#     state["metrics"]["query_type"]     = state["query_type"]
#     print(f"[Chunk] {len(summary_chunks)} summary chunks | {len(rag_chunks)} RAG chunks")
#     return state


# def node_summarize(state: DocState) -> DocState:
#     pdf_hash  = get_pdf_hash(state["pdf_path"])
#     cache_key = f"{pdf_hash}_{state['doc_type']}"
#     if cache_key in _summary_cache:
#         cached = _summary_cache[cache_key]
#         print(f"[Summary] ✅ Cache hit — returning saved summary instantly")
#         emit_event(state.get("request_id", ""), "agent_action",
#                    "⚡ Summary loaded from cache instantly!")
#         state["answer"] = cached["summary"]
#         state["metrics"].update(cached["metrics"])
#         state["metrics"]["type"] = "summary"
#         return state

#     summary_start = time.time()
#     raptor_summarize._request_id = state.get("request_id", "")
#     summary, map_time, reduce_time = raptor_summarize(
#         state["summary_chunks"], state["doc_type"])
#     summary_time = time.time() - summary_start

#     state["answer"] = summary
#     metrics_snapshot = {
#         "summary_time_sec":     round(summary_time, 2),
#         "summary_length_words": len(summary.split()),
#         "parallel_workers":     min(MAX_WORKERS, len(state["summary_chunks"])),
#         "map_time_sec":         round(map_time, 2),
#         "reduce_time_sec":      round(reduce_time, 2),
#         "llm_calls":            3,
#     }
#     state["metrics"].update(metrics_snapshot)
#     state["metrics"]["type"] = "summary"
#     _summary_cache[cache_key] = {"summary": summary, "metrics": metrics_snapshot}
#     print(f"[Summary] Done ({len(summary.split())} words)")
#     return state
# def expand_answer(answer, retrieved_texts):
#     if not answer or len(answer.split()) > 4:
#         return answer

#     answer_lower = answer.lower()

#     for chunk in retrieved_texts:
#         if answer_lower in chunk.lower():

#             # 🔥 Prefer lines with ":" (titles)
#             lines = chunk.split("\n")

#             for line in lines:
#                 if answer_lower in line.lower():
#                     line_clean = line.strip()

#                     # Strong signal: title line
#                     if ":" in line_clean and 3 <= len(line_clean.split()) <= 20:
#                         print(f"[Expand:TITLE] '{answer}' → '{line_clean}'")
#                         return line_clean

#             # 🔥 fallback → sentence
#             sentences = re.split(r'(?<=[.!?])', chunk)

#             for sent in sentences:
#                 if answer_lower in sent.lower():
#                     sent_clean = sent.strip()

#                     if 3 <= len(sent_clean.split()) <= 20:
#                         print(f"[Expand:SENT] '{answer}' → '{sent_clean}'")
#                         return sent_clean

#     return answer
# def build_context(question, retrieved_docs, all_chunks, window=1):
#     expanded = []

#     for doc in retrieved_docs:
#         idx = doc.metadata.get("chunk_id")

#         if idx is None:
#             expanded.append(doc.page_content)
#             continue

#         for i in range(max(0, idx - window), min(len(all_chunks), idx + window + 1)):
#             expanded.append(all_chunks[i])

#     seen = set()
#     final = []
#     for c in expanded:
#         if c not in seen:
#             seen.add(c)
#             final.append(c)

#     return final

# # ============================================================
# # FIX: node_qa — CRITICAL scope fix for retrieved_texts + NameError
# # All paths now set `retrieved_texts` before metrics block.
# # Dynamic SMART context candidate replaces hardcoded token list.
# # Dynamic decision scoring replaces hardcoded thresholds.
# # ============================================================
# # ============================================================
# # NODE QA — FINAL STABLE VERSION
# # ============================================================
# def node_qa(state: DocState) -> DocState:
#     qa_start_t = time.time()

#     question   = state["question"]
#     query_type = state["query_type"]
#     request_id = state.get("request_id", "")
#     all_chunks = state["chunks"]

#     pdf_hash    = get_pdf_hash(state["pdf_path"])
#     faiss_index = build_faiss_index(all_chunks, pdf_hash)

#     print(f"[QA] {query_type} | {len(all_chunks)} chunks")

#     recall_score      = 0.0
#     retrieval_score   = 0.0
#     context_precision = 0.0
#     grounding         = 0.0
#     llm_calls         = 0
#     model_used        = "llama"
#     confidence        = 0.0
#     decision_type     = "accepted"
#     retrieved         = []
#     retrieved_texts   = []
#     answer            = ""

#     # ============================================================
#     # SAFE REFUSAL DETECTOR (FIXED)
#     # ============================================================
#     _REFUSAL_ANCHOR = "this information is not available in the provided context"

#     def is_refusal(text: str) -> bool:
#         if not text or len(text.strip()) < 2:
#             return True

#         text_lower = text.lower().strip()

#         # 🔹 Hard rule (fast + reliable)
#         if "not present" in text_lower or "not available" in text_lower:
#             return True

#         # 🔹 Semantic check ONLY for long answers
#         if len(text.split()) > 5:
#             sim = semantic_similarity(text, [_REFUSAL_ANCHOR])
#             print(f"[Refusal] sim={sim:.3f} | '{text[:50]}'")
#             return sim > 0.75

#         return False

#     # ============================================================
#     # MULTIPART PATH
#     # ============================================================
#     if query_type == "MULTIPART_QA":

#         retrieved = multi_query_retrieve(
#             question, faiss_index, k=20,
#             all_chunks=all_chunks, query_type=query_type
#         )

#         retrieved, reranker_top, all_scores = rerank_docs(
#             question, retrieved,
#             top_k=MULTIPART_TOP_K,
#             apply_pruning=False
#         )

#         retrieved = protect_exact_matches(
#             question, retrieved, all_chunks,
#             top_k=MULTIPART_TOP_K
#         )

#         retrieved_texts = build_context(question, retrieved, all_chunks)

#         retrieval_score   = compute_retrieval_score(question, retrieved)
#         context_precision = compute_context_precision(question, retrieved)
#         recall_score      = compute_recall_at_k(
#             question, retrieved, all_chunks, k=len(retrieved)
#         )

#         context = "\n\n---\n\n".join(
#             reorder_by_question(question, retrieved_texts)[:MULTIPART_TOP_K]
#         )

#         emit_event(request_id, "agent_start",
#                    f"🤖 MULTIPART | {len(retrieved)} chunks retrieved")

#         emit_event(request_id, "stream_start", "✍️ Generating answer...")

#         answer = call_llama_streaming(
#             MULTIPART_PROMPT.format(
#                 context=context[:2500],
#                 question=question
#             ),
#             request_id,
#             temperature=0.0
#         )

#         answer = clean_artifacts(answer).strip().strip('"').strip("'")

#         model_used = "llama_multipart"
#         llm_calls  = 1

#         grounding  = compute_answer_grounding(answer, retrieved_texts, question)
#         confidence = compute_confidence(
#             reranker_top, recall_score,
#             answer, retrieved_texts, all_scores
#         )

#         decision_type = "accepted"

#         emit_event(request_id, "agent_done", "✅ Done!")

#         print(f"[QA] MULTIPART done | grounding={grounding:.1f}% | recall={recall_score:.1f}%")

#     # ============================================================
#     # FACTUAL / REASONING PATH (FIXED)
#     # ============================================================
#     else:

#         retrieved = multi_query_retrieve(
#             question, faiss_index, k=20,
#             all_chunks=all_chunks, query_type=query_type
#         )

#         retrieved, reranker_top, all_scores = rerank_docs(
#             question, retrieved,
#             top_k=FACTUAL_TOP_K,
#             apply_pruning=False
#         )

#         retrieved = protect_exact_matches(
#             question, retrieved, all_chunks,
#             top_k=FACTUAL_TOP_K
#         )

#         retrieved_texts = build_context(question, retrieved, all_chunks)

#         retrieval_score   = compute_retrieval_score(question, retrieved)
#         context_precision = compute_context_precision(question, retrieved)
#         recall_score      = compute_recall_at_k(
#             question, retrieved, all_chunks, k=len(retrieved)
#         )

#         # ===============================
#         # STEP 1: ReAct (PRIMARY)
#         # ===============================
#         # ===============================
#         # STEP 1: EXTRACT FROM CONTEXT (PRIMARY)
#         # ===============================
#         candidates = extract_answer_candidates(question, retrieved_texts)

#         print(f"[QA] Extracted candidates: {len(candidates)}")

#         # Clean + filter
#         candidates = [expand_answer(c, retrieved_texts) for c in candidates]
#         candidates = [c for c in candidates if is_valid_answer(c, question)]

#         print(f"[QA] Valid candidates: {len(candidates)}")

#         # ===============================
#         # STEP 2: SELECT BEST
#         # ===============================
#         if candidates:
#             def question_relevance_score(answer, question):
#                 answer = answer.lower()
#                 question = question.lower()

#                 score = 0

#                 # 🔥 remove weak words
#                 stopwords = _get_dynamic_stopwords()

#                 q_words = set(w for w in re.findall(r'\b\w+\b', question)
#                             if w not in stopwords and len(w) > 2)

#                 a_words = set(re.findall(r'\b\w+\b', answer))

#                 # 🔥 strong keyword match
#                 overlap = len(q_words & a_words)
#                 score += overlap * 2   # weight keywords higher

#                 # 🔥 NUMBER constraint (CRITICAL)
#                 q_nums = re.findall(r'\d+', question)
#                 if q_nums:
#                     if any(num in answer for num in q_nums):
#                         score += 10   # STRONG boost
#                     else:
#                         score -= 5    # penalty if missing

#                 # 🔥 penalize too long noisy answers
#                 if len(answer.split()) > 25:
#                     score -= 2

#                 return score


#             scores = [
#                 score_answer(c, retrieved_texts) + question_relevance_score(c, question)
#                 for c in candidates
#             ]

#             best_idx = int(np.argmax(scores))
#             best_score = scores[best_idx]
#             answer = candidates[best_idx]

#             # 🔥 ADD THIS BLOCK HERE
#             if best_score <= 0:
#                 print("[QA] ❌ Low relevance → rejecting")
#                 answer = "This information is not present in the document."
#                 decision_type = "rejected_low_relevance"
#             else:
#                 decision_type = "accepted_extractive"

#             model_used = "extractive"

#             print(f"[QA] Best extracted: {answer} | score={best_score:.3f}")

#         # ===============================
#         # STEP 3: FALLBACK TO LLM
#         # ===============================
#         else:
#             print("[QA] No extraction → fallback to ReAct")

#             react_ans, model_used, _, retrieval_score, context_precision, grounding, react_calls = react_agent(
#                 question, faiss_index, query_type, all_chunks, request_id
#             )

#             llm_calls += react_calls

#             react_ans = clean_artifacts(react_ans)
#             react_ans = react_ans.strip().strip('"').strip("'")

#             if is_valid_answer(react_ans, question):
#                 answer = react_ans
#             else:
#                 answer = "This information is not present in the document."
#                 decision_type = "not_found"
#     # ============================================================
#     # FINAL METRICS (CRITICAL FIX)
#     # ============================================================
#     qa_time = time.time() - qa_start_t

#     print(f"[QA] Done in {qa_time:.1f}s | model={model_used} | "
#           f"grounding={grounding:.1f}% | recall={recall_score:.1f}% | "
#           f"confidence={confidence:.3f} | decision={decision_type}")

#     if not answer.strip():
#         print("[QA] 🚨 Empty answer fallback")
#         answer = "Could not find a relevant answer in the PDF."

#     # ✅ MOST IMPORTANT LINE (your earlier bug)
#     state["answer"] = answer

#     state["metrics"]["qa_time_sec"]       = round(qa_time, 2)
#     state["metrics"]["confidence_score"]  = round(confidence * 100, 2)
#     state["metrics"]["retrieval_score"]   = retrieval_score
#     state["metrics"]["context_precision"] = context_precision
#     state["metrics"]["answer_grounding"]  = grounding
#     state["metrics"]["recall_at_k"]       = recall_score
#     state["metrics"]["llm_calls"]         = llm_calls
#     state["metrics"]["model_used"]        = model_used
#     state["metrics"]["chunks_retrieved"]  = len(retrieved)
#     state["metrics"]["type"]              = "qa"
#     state["metrics"]["decision_type"]     = decision_type
#     state["metrics"]["confidence_raw"]    = round(confidence, 4)

#     return state


# # ============================================================
# # NODE VALIDATE (SAFE)
# # ============================================================
# def node_validate(state: DocState) -> DocState:

#     answer = state["answer"]
#     retry  = state.get("retry_count", 0)

#     print(f"[VALIDATE] answer='{answer[:50]}' retry={retry}")

#     if len(answer.strip()) < 3 and retry < 2:
#         print("[VALIDATE] 🔁 Retrying QA...")
#         state["retry_count"] = retry + 1
#         state["answer"] = ""
#         return state

#     total_time    = time.time() - state["start_time"]
#     output_words  = len(answer.split())
#     output_tokens = output_words * 1.3

#     extract_time  = state["metrics"].get("extraction_time_sec", 0)
#     llm_time      = max(total_time - extract_time, 1)

#     tps = round(output_tokens / llm_time, 2)

#     m = state["metrics"]

#     m["response_time_sec"]  = round(total_time, 2)
#     m["ttft_sec"]           = round(total_time, 2)
#     m["e2e_latency_sec"]    = round(total_time, 2)
#     m["tps"]                = tps
#     m["retry_count"]        = retry

#     state["metrics"] = m

#     return state





















# # recall 100 but got rejected 
# # Instead of:

# # Retrieve → LLM → Hope answer is correct ❌

# # We move to:

# # Retrieve → Extract candidates → Filter → Score → LLM (only if needed) ✅
# # 🔥 CORE IDEA

# # 👉 Answer should come from context, not model imagination
# # react_answer → accepted

# # 👉 even though it's garbage

# # 💥 THIS IS THE ACTUAL SYSTEM DESIGN BUG
# # You are treating:
# # ReAct output = ground truth ❌

# # Instead of:

# # ReAct = just one noisy candidate




# import re

# import time

# import numpy as np

# from docmind_rag.config.settings import (
#     MAX_WORKERS, FACTUAL_TOP_K, MULTIPART_TOP_K, SMALL_DOC_CHUNK_THRESHOLD
# )
# from docmind_rag.core.state import DocState
# from docmind_rag.core.prompts import MULTIPART_PROMPT, REASONING_PROMPT, FACTUAL_EXTRACT_PROMPT
# from docmind_rag.models.llm import call_llama_streaming
# from docmind_rag.models.embeddings import build_faiss_index
# from docmind_rag.models.reranker import rerank_docs, protect_exact_matches
# from docmind_rag.services.extraction import extract_pdf_parallel, _extraction_cache
# from docmind_rag.services.chunking import semantic_chunk, raptor_summarize, _summary_cache
# from docmind_rag.services.retrieval import multi_query_retrieve
# from docmind_rag.services.agent import react_agent, roberta_qa
# from docmind_rag.events.events import emit_event
# from docmind_rag.utils.helpers import get_pdf_hash, clean_artifacts, _get_dynamic_stopwords
# from docmind_rag.utils.text import detect_doc_type, infer_answer_shape, shape_to_query_type, reorder_by_question
# from docmind_rag.utils.metrics import (
#     compute_answer_grounding, compute_retrieval_score, compute_context_precision,
#     compute_recall_at_k, compute_confidence, compute_grounding_score,
#     semantic_similarity, keyword_overlap, local_substring_match,
#     score_answer, coverage_score, llm_verify_answer
# )

# # ============================================================
# # LANGGRAPH NODES
# # ============================================================
# def node_extract(state: DocState) -> DocState:
#     extract_start = time.time()
#     text, page_count = extract_pdf_parallel(state["pdf_path"])
#     extract_time  = time.time() - extract_start
#     state["extracted_text"] = text
#     state["page_count"]     = page_count
#     state["char_count"]     = len(text)
#     state["metrics"]["extraction_time_sec"]  = round(extract_time, 2)
#     state["metrics"]["pages_processed"]      = page_count
#     state["metrics"]["characters_processed"] = len(text)
#     state["metrics"]["words_processed"]      = len(text.split())
#     return state


# def node_chunk(state: DocState) -> DocState:
#     text = state["extracted_text"]
#     state["doc_type"] = detect_doc_type(text)

#     shape = infer_answer_shape(state["question"])
#     state["query_type"] = shape_to_query_type(shape)
#     state["metrics"]["answer_shape"] = shape

#     summary_chunks, rag_chunks = semantic_chunk(text)
#     state["summary_chunks"] = summary_chunks
#     state["chunks"]         = rag_chunks

#     state["metrics"]["summary_chunks"] = len(summary_chunks)
#     state["metrics"]["chunks_created"] = len(rag_chunks)
#     state["metrics"]["doc_type"]       = state["doc_type"]
#     state["metrics"]["query_type"]     = state["query_type"]
#     print(f"[Chunk] {len(summary_chunks)} summary chunks | {len(rag_chunks)} RAG chunks")
#     return state


# def node_summarize(state: DocState) -> DocState:
#     pdf_hash  = get_pdf_hash(state["pdf_path"])
#     cache_key = f"{pdf_hash}_{state['doc_type']}"
#     if cache_key in _summary_cache:
#         cached = _summary_cache[cache_key]
#         print(f"[Summary] ✅ Cache hit — returning saved summary instantly")
#         emit_event(state.get("request_id", ""), "agent_action",
#                    "⚡ Summary loaded from cache instantly!")
#         state["answer"] = cached["summary"]
#         state["metrics"].update(cached["metrics"])
#         state["metrics"]["type"] = "summary"
#         return state

#     summary_start = time.time()
#     raptor_summarize._request_id = state.get("request_id", "")
#     summary, map_time, reduce_time = raptor_summarize(
#         state["summary_chunks"], state["doc_type"])
#     summary_time = time.time() - summary_start

#     state["answer"] = summary
#     metrics_snapshot = {
#         "summary_time_sec":     round(summary_time, 2),
#         "summary_length_words": len(summary.split()),
#         "parallel_workers":     min(MAX_WORKERS, len(state["summary_chunks"])),
#         "map_time_sec":         round(map_time, 2),
#         "reduce_time_sec":      round(reduce_time, 2),
#         "llm_calls":            3,
#     }
#     state["metrics"].update(metrics_snapshot)
#     state["metrics"]["type"] = "summary"
#     _summary_cache[cache_key] = {"summary": summary, "metrics": metrics_snapshot}
#     print(f"[Summary] Done ({len(summary.split())} words)")
#     return state
# def expand_answer(answer, retrieved_texts):
#     if not answer or len(answer.split()) > 4:
#         return answer

#     answer_lower = answer.lower()

#     for chunk in retrieved_texts:
#         if answer_lower in chunk.lower():

#             # 🔥 Prefer lines with ":" (titles)
#             lines = chunk.split("\n")

#             for line in lines:
#                 if answer_lower in line.lower():
#                     line_clean = line.strip()

#                     # Strong signal: title line
#                     if ":" in line_clean and 3 <= len(line_clean.split()) <= 20:
#                         print(f"[Expand:TITLE] '{answer}' → '{line_clean}'")
#                         return line_clean

#             # 🔥 fallback → sentence
#             sentences = re.split(r'(?<=[.!?])', chunk)

#             for sent in sentences:
#                 if answer_lower in sent.lower():
#                     sent_clean = sent.strip()

#                     if 3 <= len(sent_clean.split()) <= 20:
#                         print(f"[Expand:SENT] '{answer}' → '{sent_clean}'")
#                         return sent_clean

#     return answer
# def build_context(question, retrieved_docs, all_chunks, window=1):
#     expanded = []

#     for doc in retrieved_docs:
#         idx = doc.metadata.get("chunk_id")

#         if idx is None:
#             expanded.append(doc.page_content)
#             continue

#         for i in range(max(0, idx - window), min(len(all_chunks), idx + window + 1)):
#             expanded.append(all_chunks[i])

#     seen = set()
#     final = []
#     for c in expanded:
#         if c not in seen:
#             seen.add(c)
#             final.append(c)

#     return final

# # ============================================================
# # FIX: node_qa — CRITICAL scope fix for retrieved_texts + NameError
# # All paths now set `retrieved_texts` before metrics block.
# # Dynamic SMART context candidate replaces hardcoded token list.
# # Dynamic decision scoring replaces hardcoded thresholds.
# # ============================================================
# # ============================================================
# # NODE QA — FINAL STABLE VERSION
# # ============================================================
# def node_qa(state: DocState) -> DocState:
#     qa_start_t = time.time()

#     question   = state["question"]
#     query_type = state["query_type"]
#     request_id = state.get("request_id", "")
#     all_chunks = state["chunks"]

#     pdf_hash    = get_pdf_hash(state["pdf_path"])
#     faiss_index = build_faiss_index(all_chunks, pdf_hash)

#     print(f"[QA] {query_type} | {len(all_chunks)} chunks")

#     recall_score      = 0.0
#     retrieval_score   = 0.0
#     context_precision = 0.0
#     grounding         = 0.0
#     llm_calls         = 0
#     model_used        = "llama"
#     confidence        = 0.0
#     decision_type     = "accepted"
#     retrieved         = []
#     retrieved_texts   = []
#     answer            = ""

#     # ============================================================
#     # SAFE REFUSAL DETECTOR (FIXED)
#     # ============================================================
#     _REFUSAL_ANCHOR = "this information is not available in the provided context"

#     def is_refusal(text: str) -> bool:
#         if not text or len(text.strip()) < 2:
#             return True

#         text_lower = text.lower().strip()

#         # 🔹 Hard rule (fast + reliable)
#         if "not present" in text_lower or "not available" in text_lower:
#             return True

#         # 🔹 Semantic check ONLY for long answers
#         if len(text.split()) > 5:
#             sim = semantic_similarity(text, [_REFUSAL_ANCHOR])
#             print(f"[Refusal] sim={sim:.3f} | '{text[:50]}'")
#             return sim > 0.75

#         return False

#     # ============================================================
#     # MULTIPART PATH
#     # ============================================================
#     if query_type == "MULTIPART_QA":

#         retrieved = multi_query_retrieve(
#             question, faiss_index, k=20,
#             all_chunks=all_chunks, query_type=query_type
#         )

#         retrieved, reranker_top, all_scores = rerank_docs(
#             question, retrieved,
#             top_k=MULTIPART_TOP_K,
#             apply_pruning=False
#         )

#         retrieved = protect_exact_matches(
#             question, retrieved, all_chunks,
#             top_k=MULTIPART_TOP_K
#         )

#         retrieved_texts = build_context(question, retrieved, all_chunks)

#         retrieval_score   = compute_retrieval_score(question, retrieved)
#         context_precision = compute_context_precision(question, retrieved)
#         recall_score      = compute_recall_at_k(
#             question, retrieved, all_chunks, k=len(retrieved)
#         )

#         context = "\n\n---\n\n".join(
#             reorder_by_question(question, retrieved_texts)[:MULTIPART_TOP_K]
#         )

#         emit_event(request_id, "agent_start",
#                    f"🤖 MULTIPART | {len(retrieved)} chunks retrieved")

#         emit_event(request_id, "stream_start", "✍️ Generating answer...")

#         answer = call_llama_streaming(
#             MULTIPART_PROMPT.format(
#                 context=context[:2500],
#                 question=question
#             ),
#             request_id,
#             temperature=0.0
#         )

#         answer = clean_artifacts(answer).strip().strip('"').strip("'")

#         model_used = "llama_multipart"
#         llm_calls  = 1

#         grounding  = compute_answer_grounding(answer, retrieved_texts, question)
#         confidence = compute_confidence(
#             reranker_top, recall_score,
#             answer, retrieved_texts, all_scores
#         )

#         decision_type = "accepted"

#         emit_event(request_id, "agent_done", "✅ Done!")

#         print(f"[QA] MULTIPART done | grounding={grounding:.1f}% | recall={recall_score:.1f}%")

#     # ============================================================
#     # FACTUAL / REASONING PATH (FIXED)
#     # ============================================================
#     else:

#         retrieved = multi_query_retrieve(
#             question, faiss_index, k=20,
#             all_chunks=all_chunks, query_type=query_type
#         )

#         retrieved, reranker_top, all_scores = rerank_docs(
#             question, retrieved,
#             top_k=FACTUAL_TOP_K,
#             apply_pruning=False
#         )

#         retrieved = protect_exact_matches(
#             question, retrieved, all_chunks,
#             top_k=FACTUAL_TOP_K
#         )

#         retrieved_texts = build_context(question, retrieved, all_chunks)

#         retrieval_score   = compute_retrieval_score(question, retrieved)
#         context_precision = compute_context_precision(question, retrieved)
#         recall_score      = compute_recall_at_k(
#             question, retrieved, all_chunks, k=len(retrieved)
#         )

#         # ===============================
#         # STEP 1: ReAct (PRIMARY)
#         # ===============================
#         react_ans, model_used, _, retrieval_score, context_precision, grounding, react_calls = react_agent(
#             question, faiss_index, query_type, all_chunks, request_id
#         )

#         llm_calls += react_calls

#         react_ans = clean_artifacts(react_ans)
#         react_ans = react_ans.strip().strip('"').strip("'")
#         react_ans = expand_answer(react_ans, retrieved_texts)
#         # ============================================================

#         print(f"[QA] ReAct: '{react_ans[:60]}'")

#         # ===============================
#         # STEP 2: RoBERTa (FALLBACK)
#         # ===============================
#         rob_ans, rob_score = roberta_qa(question, retrieved_texts)
#         rob_ans = rob_ans.strip()

#         print(f"[QA] RoBERTa: '{rob_ans[:40]}' (score={rob_score:.3f})")

#         # ===============================
#         # STEP 3: DECISION (FIXED)
#         # ===============================
#         if react_ans and not is_refusal(react_ans):
#             answer     = react_ans
#             model_used = "llama_react"

#         elif rob_ans and rob_score >= 0.25 and not is_refusal(rob_ans):
#             answer     = rob_ans
#             model_used = "roberta_fallback"
#             print(f"[QA] ReAct refused → using RoBERTa")

#         else:
#             answer        = "This information is not present in the document."
#             model_used    = "not_found"
#             decision_type = "not_found"

#         grounding  = compute_answer_grounding(answer, retrieved_texts, question)

#         confidence = compute_confidence(
#             reranker_top, recall_score,
#             answer, retrieved_texts, all_scores
#         )

#         recall_score = compute_recall_at_k(
#             question, retrieved, all_chunks, k=len(retrieved)
#         )

#     # ============================================================
#     # FINAL METRICS (CRITICAL FIX)
#     # ============================================================
#     qa_time = time.time() - qa_start_t

#     print(f"[QA] Done in {qa_time:.1f}s | model={model_used} | "
#           f"grounding={grounding:.1f}% | recall={recall_score:.1f}% | "
#           f"confidence={confidence:.3f} | decision={decision_type}")

#     if not answer.strip():
#         print("[QA] 🚨 Empty answer fallback")
#         answer = "Could not find a relevant answer in the PDF."

#     # ✅ MOST IMPORTANT LINE (your earlier bug)
#     state["answer"] = answer

#     state["metrics"]["qa_time_sec"]       = round(qa_time, 2)
#     state["metrics"]["confidence_score"]  = round(confidence * 100, 2)
#     state["metrics"]["retrieval_score"]   = retrieval_score
#     state["metrics"]["context_precision"] = context_precision
#     state["metrics"]["answer_grounding"]  = grounding
#     state["metrics"]["recall_at_k"]       = recall_score
#     state["metrics"]["llm_calls"]         = llm_calls
#     state["metrics"]["model_used"]        = model_used
#     state["metrics"]["chunks_retrieved"]  = len(retrieved)
#     state["metrics"]["type"]              = "qa"
#     state["metrics"]["decision_type"]     = decision_type
#     state["metrics"]["confidence_raw"]    = round(confidence, 4)

#     return state


# # ============================================================
# # NODE VALIDATE (SAFE)
# # ============================================================
# def node_validate(state: DocState) -> DocState:

#     answer = state["answer"]
#     retry  = state.get("retry_count", 0)

#     print(f"[VALIDATE] answer='{answer[:50]}' retry={retry}")

#     if len(answer.strip()) < 3 and retry < 2:
#         print("[VALIDATE] 🔁 Retrying QA...")
#         state["retry_count"] = retry + 1
#         state["answer"] = ""
#         return state

#     total_time    = time.time() - state["start_time"]
#     output_words  = len(answer.split())
#     output_tokens = output_words * 1.3

#     extract_time  = state["metrics"].get("extraction_time_sec", 0)
#     llm_time      = max(total_time - extract_time, 1)

#     tps = round(output_tokens / llm_time, 2)

#     m = state["metrics"]

#     m["response_time_sec"]  = round(total_time, 2)
#     m["ttft_sec"]           = round(total_time, 2)
#     m["e2e_latency_sec"]    = round(total_time, 2)
#     m["tps"]                = tps
#     m["retry_count"]        = retry

#     state["metrics"] = m

#     return state



















# #getting lecture 3 issue nd answer it is printing is lecture 3 itself 

# what is the name of lecture 3?
# Run Agent →
# AGENT REASONING TRACE
# 📄
# EXTRACTION
# 📄 Extracting PDF text...
# ⚙️
# WORKFLOW
# ⚙️ LangGraph workflow started...
# 🤖
# AGENT INIT
# 🤖 Agent starting | FACTUAL_QA | 5 chunks retrieved
# ✅
# COMPLETE
# ✅ Answer found at step 1!
# ✅
# COMPLETE
# ✅ Done!
# Answer
# Lecture 3
# Benchmark Metrics
# 📊 Core Performance
# Type
# Q&A
# Total Time
# 1 min 9.01 sec
# Extraction
# 0.17 sec
# QA Time
# 1 min 1.64 sec
# Model
# dynamic_scoring
# LLM Calls
# 1 calls
# Pages
# 363 pages
# Characters
# 222,948 chars
# Words
# 34,642 words
# ⚡ Latency Metrics
# TTFT · Time To First Token
# 1 min 9.01 sec
# = E2E (no streaming yet)
# E2E Latency · End To End
# 1 min 9.01 sec
# total response time
# TPS · Tokens/Second
# 0.0
# ❌ Slow
# 🎯 RAG Quality Metrics
# Retrieval Score · Chunk Relevance
# 24.1%
# ⚠️ Low
# Confidence · Retrieval + Verified
# 89.4%
# ✅ High
# Recall@K · Oracle Chunk Hit
# 64.7%
# ⚠️ Partial
# 🧠 Decision Intelligence
# Decision Type
# accepted_short_fact
# Verification Mode · Strictness
# NONE
# — Not applied
# Keyword Score · Answer Grounding
# 0.0%
# ❌ Low
# Verified · Answer Confirmed
# NO
# ❌ Not verified
# 🔬 Debug Metrics · For Analysis Only
# Context Precision · Relevant/Retrieved
# 60.0%
# Debug only — not used in decisions
# Answer Grounding · Word Overlap
# 100.0%
# Debug only — not used in decisions
# what is the name of lecture 3?
# Run Agent →
# AGENT REASONING TRACE
# 📄
# EXTRACTION
# 📄 Extracting PDF text...
# ⚙️
# WORKFLOW
# ⚙️ LangGraph workflow started...
# 🤖
# AGENT INIT
# 🤖 Agent starting | FACTUAL_QA | 5 chunks retrieved
# ✅
# COMPLETE
# ✅ Answer found at step 1!
# ✅
# COMPLETE
# ✅ Done!
# Answer
# Lecture 3
# Benchmark Metrics
# 📊 Core Performance
# Type
# Q&A
# Total Time
# 1 min 9.01 sec
# Extraction
# 0.17 sec
# QA Time
# 1 min 1.64 sec
# Model
# dynamic_scoring
# LLM Calls
# 1 calls
# Pages
# 363 pages
# Characters
# 222,948 chars
# Words
# 34,642 words
# ⚡ Latency Metrics
# TTFT · Time To First Token
# 1 min 9.01 sec
# = E2E (no streaming yet)
# E2E Latency · End To End
# 1 min 9.01 sec
# total response time
# TPS · Tokens/Second
# 0.0
# ❌ Slow
# 🎯 RAG Quality Metrics
# Retrieval Score · Chunk Relevance
# 24.1%
# ⚠️ Low
# Confidence · Retrieval + Verified
# 89.4%
# ✅ High
# Recall@K · Oracle Chunk Hit
# 64.7%
# ⚠️ Partial
# 🧠 Decision Intelligence
# Decision Type
# accepted_short_fact
# Verification Mode · Strictness
# NONE
# — Not applied
# Keyword Score · Answer Grounding
# 0.0%
# ❌ Low
# Verified · Answer Confirmed
# NO
# ❌ Not verified
# 🔬 Debug Metrics · For Analysis Only
# Context Precision · Relevant/Retrieved
# 60.0%
# Debug only — not used in decisions
# Answer Grounding · Word Overlap
# 100.0%
# Debug only — not used in decisions




# import re
# import time

# import numpy as np

# from docmind_rag.config.settings import (
#     MAX_WORKERS, FACTUAL_TOP_K, MULTIPART_TOP_K, SMALL_DOC_CHUNK_THRESHOLD
# )
# from docmind_rag.core.state import DocState
# from docmind_rag.core.prompts import MULTIPART_PROMPT, REASONING_PROMPT, FACTUAL_EXTRACT_PROMPT
# from docmind_rag.models.llm import call_llama_streaming
# from docmind_rag.models.embeddings import build_faiss_index
# from docmind_rag.models.reranker import rerank_docs, protect_exact_matches
# from docmind_rag.services.extraction import extract_pdf_parallel, _extraction_cache
# from docmind_rag.services.chunking import semantic_chunk, raptor_summarize, _summary_cache
# from docmind_rag.services.retrieval import multi_query_retrieve
# from docmind_rag.services.agent import react_agent, roberta_qa
# from docmind_rag.events.events import emit_event
# from docmind_rag.utils.helpers import get_pdf_hash, clean_artifacts, _get_dynamic_stopwords
# from docmind_rag.utils.text import detect_doc_type, infer_answer_shape, shape_to_query_type, reorder_by_question
# from docmind_rag.utils.metrics import (
#     compute_answer_grounding, compute_retrieval_score, compute_context_precision,
#     compute_recall_at_k, compute_confidence, compute_grounding_score,
#     semantic_similarity, keyword_overlap, local_substring_match,
#     score_answer, coverage_score, llm_verify_answer
# )

# # ============================================================
# # LANGGRAPH NODES
# # ============================================================
# def node_extract(state: DocState) -> DocState:
#     extract_start = time.time()
#     text, page_count = extract_pdf_parallel(state["pdf_path"])
#     extract_time  = time.time() - extract_start
#     state["extracted_text"] = text
#     state["page_count"]     = page_count
#     state["char_count"]     = len(text)
#     state["metrics"]["extraction_time_sec"]  = round(extract_time, 2)
#     state["metrics"]["pages_processed"]      = page_count
#     state["metrics"]["characters_processed"] = len(text)
#     state["metrics"]["words_processed"]      = len(text.split())
#     return state


# def node_chunk(state: DocState) -> DocState:
#     text = state["extracted_text"]
#     state["doc_type"] = detect_doc_type(text)

#     shape = infer_answer_shape(state["question"])
#     state["query_type"] = shape_to_query_type(shape)
#     state["metrics"]["answer_shape"] = shape

#     summary_chunks, rag_chunks = semantic_chunk(text)
#     state["summary_chunks"] = summary_chunks
#     state["chunks"]         = rag_chunks

#     state["metrics"]["summary_chunks"] = len(summary_chunks)
#     state["metrics"]["chunks_created"] = len(rag_chunks)
#     state["metrics"]["doc_type"]       = state["doc_type"]
#     state["metrics"]["query_type"]     = state["query_type"]
#     print(f"[Chunk] {len(summary_chunks)} summary chunks | {len(rag_chunks)} RAG chunks")
#     return state


# def node_summarize(state: DocState) -> DocState:
#     pdf_hash  = get_pdf_hash(state["pdf_path"])
#     cache_key = f"{pdf_hash}_{state['doc_type']}"
#     if cache_key in _summary_cache:
#         cached = _summary_cache[cache_key]
#         print(f"[Summary] ✅ Cache hit — returning saved summary instantly")
#         emit_event(state.get("request_id", ""), "agent_action",
#                    "⚡ Summary loaded from cache instantly!")
#         state["answer"] = cached["summary"]
#         state["metrics"].update(cached["metrics"])
#         state["metrics"]["type"] = "summary"
#         return state

#     summary_start = time.time()
#     raptor_summarize._request_id = state.get("request_id", "")
#     summary, map_time, reduce_time = raptor_summarize(
#         state["summary_chunks"], state["doc_type"])
#     summary_time = time.time() - summary_start

#     state["answer"] = summary
#     metrics_snapshot = {
#         "summary_time_sec":     round(summary_time, 2),
#         "summary_length_words": len(summary.split()),
#         "parallel_workers":     min(MAX_WORKERS, len(state["summary_chunks"])),
#         "map_time_sec":         round(map_time, 2),
#         "reduce_time_sec":      round(reduce_time, 2),
#         "llm_calls":            3,
#     }
#     state["metrics"].update(metrics_snapshot)
#     state["metrics"]["type"] = "summary"
#     _summary_cache[cache_key] = {"summary": summary, "metrics": metrics_snapshot}
#     print(f"[Summary] Done ({len(summary.split())} words)")
#     return state


# # ============================================================
# # FIX: node_qa — CRITICAL scope fix for retrieved_texts + NameError
# # All paths now set `retrieved_texts` before metrics block.
# # Dynamic SMART context candidate replaces hardcoded token list.
# # Dynamic decision scoring replaces hardcoded thresholds.
# # ============================================================
# def node_qa(state: DocState) -> DocState:
#     qa_start   = time.time()
#     question   = state["question"]
#     query_type = state["query_type"]
#     request_id = state.get("request_id", "")
#     all_chunks = state["chunks"]

#     pdf_hash    = get_pdf_hash(state["pdf_path"])
#     faiss_index = build_faiss_index(all_chunks, pdf_hash)

#     print(f"[QA] Query type: {query_type} | {len(all_chunks)} RAG chunks in index")

#     recall_score      = 0.0
#     retrieval_score   = 0.0
#     context_precision = 0.0
#     grounding         = 0.0
#     llm_calls         = 0
#     model_used        = "llama_react"
#     confidence        = 0.0
#     decision_type     = "accepted"
#     retrieved         = []

#     # FIX: initialise retrieved_texts here so it is ALWAYS defined
#     # regardless of which branch (MULTIPART or FACTUAL) is taken
#     retrieved_texts: list = []

#     is_small_doc = (len(all_chunks) <= SMALL_DOC_CHUNK_THRESHOLD)
#     if is_small_doc:
#         print(f"[QA] Small doc detected ({len(all_chunks)} chunks)")

#     if query_type == "MULTIPART_QA":
#         print("[QA] MULTIPART → multi_query_retrieve + rerank (no pruning)...")
#         retrieved = multi_query_retrieve(question, faiss_index, k=20,
#                                          all_chunks=all_chunks, query_type=query_type)
#         retrieved, reranker_top, all_scores = rerank_docs(question, retrieved,
#                                                            top_k=MULTIPART_TOP_K,
#                                                            apply_pruning=False)
#         retrieved = protect_exact_matches(question, retrieved, all_chunks,
#                                           top_k=MULTIPART_TOP_K)

#         retrieved_texts   = [d.page_content for d in retrieved]  # FIX: always set
#         retrieval_score   = compute_retrieval_score(question, retrieved)
#         context_precision = compute_context_precision(question, retrieved)
#         recall_score      = compute_recall_at_k(question, retrieved, all_chunks,
#                                                 k=len(retrieved))
#         context           = "\n\n---\n\n".join(
#             reorder_by_question(question, retrieved_texts)[:MULTIPART_TOP_K])

#         emit_event(request_id, "agent_start",
#                    f"🤖 MULTIPART | {len(retrieved)} chunks retrieved")
#         emit_event(request_id, "stream_start", "✍️ Generating complete list...")

#         answer = call_llama_streaming(
#             MULTIPART_PROMPT.format(context=context[:2500], question=question),
#             request_id, temperature=0.0)

#         answer = clean_artifacts(answer)
#         answer = answer.strip('"').strip("'").strip()

#         # Trivial answer filter
#         if answer.strip().lower() in question.strip().lower():
#             print("[QA] 🚫 Trivial answer — rejecting")
#             state["answer"] = "This information is not present in the document."
#             state["metrics"]["decision_type"] = "rejected_trivial"
#             return state

#         grounding_score = compute_grounding_score(answer, retrieved_texts)

#         # MULTIPART grounding override: distributed lists don't concentrate in one chunk
#         sem  = semantic_similarity(answer, retrieved_texts)
#         kw   = keyword_overlap(answer, retrieved_texts)
#         grounding_score = max(sem, kw)
#         print(f"[Grounding] MULTIPART override: max(sem={sem:.3f}, kw={kw:.3f}) = {grounding_score:.3f}")

#         grounding  = compute_answer_grounding(answer, retrieved_texts, question)
#         model_used = "llama_multipart"
#         llm_calls  = 1

#         loc_mp = local_substring_match(answer, retrieved_texts)

#         if recall_score > 70:
#             decision_type = "accepted_high_recall"
#             print(f"[QA] MULTIPART high recall ({recall_score:.1f}%) — accepted")
#         elif loc_mp == 1.0:
#             decision_type = "accept_strong"
#             print(f"[QA] MULTIPART local match — accept_strong")
#         elif sem > 0.75 and kw > 0.5:
#             decision_type = "accept_semantic"
#             print(f"[QA] MULTIPART strong semantic+keyword — accept_semantic")
#         elif kw > 0.7:
#             decision_type = "accept_keyword"
#             print(f"[QA] MULTIPART keyword override (kw={kw:.3f}) — accept_keyword")
#         elif grounding_score >= 0.45:
#             decision_type = "accepted"
#             print(f"[QA] MULTIPART good grounding ({grounding_score:.3f}) — accepted")
#         elif grounding_score >= 0.30:
#             if llm_verify_answer(question, answer, retrieved_texts):
#                 decision_type = "accept_verified"
#                 print(f"[QA] MULTIPART borderline — LLM verified")
#             else:
#                 answer        = "This information is not present in the document."
#                 decision_type = "rejected_low_grounding"
#                 print(f"[QA] MULTIPART borderline — LLM rejected")
#         else:
#             answer        = "This information is not present in the document."
#             decision_type = "rejected_low_grounding"
#             print(f"[QA] MULTIPART grounding too low ({grounding_score:.3f}) — rejected")

#         confidence = compute_confidence(reranker_top, recall_score,
#                                         answer, retrieved_texts, all_scores)
#         emit_event(request_id, "agent_done", "✅ Complete list generated!")
#         print(f"[QA] MULTIPART done | grounding={grounding:.1f}% | "
#               f"recall={recall_score:.1f}% | confidence={confidence:.3f}")

#     else:
#         # ── FACTUAL / VERIFICATION / REASONING path ──────────────────────────
#         print(f"[QA] {query_type} → retrieve + rerank + LLM always...")
#         retrieved = multi_query_retrieve(question, faiss_index, k=20,
#                                          all_chunks=all_chunks, query_type=query_type)
#         retrieved, reranker_top, all_scores = rerank_docs(question, retrieved,
#                                                            top_k=FACTUAL_TOP_K,
#                                                            apply_pruning=True)
#         retrieved = protect_exact_matches(question, retrieved, all_chunks,
#                                           top_k=FACTUAL_TOP_K)

#         retrieved_texts   = [d.page_content for d in retrieved]  # FIX: always set
#         retrieval_score   = compute_retrieval_score(question, retrieved)
#         context_precision = compute_context_precision(question, retrieved)
#         recall_score      = compute_recall_at_k(question, retrieved, all_chunks,
#                                                 k=len(retrieved))

#         print(f"[QA] FIX 1: Pre-LLM gate removed — allow_llm=True always")

#         # ── Stage 1: Candidates from RoBERTa + ReAct ─────────────────────────
#         candidates = []

#         rob_answer, rob_score = roberta_qa(question, retrieved_texts)
#         print(f"[QA] RoBERTa score: {rob_score:.4f} | answer: '{rob_answer[:60]}'")
#         if rob_answer.strip() and rob_score >= 0.25:
#             candidates.append(rob_answer.strip())

#         print(f"[QA] → ReAct agent ({query_type})...")
#         react_answer, model_used, _, retrieval_score, context_precision, grounding, react_calls = react_agent(
#             question, faiss_index, query_type, all_chunks, request_id)
#         llm_calls += react_calls

#         react_answer = clean_artifacts(react_answer)
#         react_answer = react_answer.strip('"').strip("'").strip()

#         if react_answer.strip():
#             candidates.append(react_answer.strip())

#         print(f"[QA] Candidates: {len(candidates)}")

#         # ── FIX: Dynamic SMART context candidate — no hardcoded token list ────
#         if len(candidates) == 1:
#             context_candidate = None

#             # Extract meaningful tokens from the question dynamically
#             stopwords = _get_dynamic_stopwords()
#             # Include numbers — they are always high-signal for lookup questions
#             number_tokens    = re.findall(r'\b\d+\b', question)
#             content_tokens   = [w.lower() for w in question.split()
#                                  if len(w) > 3 and w.lower() not in stopwords]
#             important_tokens = number_tokens + content_tokens

#             if important_tokens:
#                 for chunk in retrieved_texts:
#                     chunk_lower = chunk.lower()
#                     if all(token in chunk_lower for token in important_tokens):
#                         # Take the most specific sentence from this chunk
#                         sentences = re.split(r'(?<=[.!?\n])', chunk)
#                         for sent in sentences:
#                             sent_lower = sent.lower()
#                             if all(token in sent_lower for token in important_tokens):
#                                 sent_clean = sent.strip()
#                                 word_count = len(sent_clean.split())
#                                 if 2 <= word_count <= 12:
#                                     context_candidate = sent_clean
#                                     break
#                     if context_candidate:
#                         break

#             if context_candidate and context_candidate != candidates[0]:
#                 candidates.append(context_candidate)
#                 print(f"[QA] Added SMART context candidate: {context_candidate[:80]}")

#         # ── Stage 2+3: Score and select best candidate ────────────────────────
#         scores = [score_answer(c, retrieved_texts) for c in candidates]

#         best_idx    = int(np.argmax(scores)) if scores else 0
#         best_answer = candidates[best_idx] if candidates else ""
#         best_score  = scores[best_idx] if scores else 0.0
#         other_scores = [s for i, s in enumerate(scores) if i != best_idx]

#         print(f"[QA] Scores: {scores} | best={best_score:.3f}")

#         # ── CASE 1: Multiple candidates — relative decision ───────────────────
#         if other_scores:
#             avg_other = sum(other_scores) / len(other_scores)
#             gap       = best_score - avg_other
#             print(f"[QA] gap={gap:.3f} | avg_other={avg_other:.3f}")

#             if gap > 0 or abs(gap) < 1e-6:
#                 answer        = best_answer
#                 decision_type = "accepted_relative"
#                 print("[QA] ✅ Accepted (relative dominance)")
#             else:
#                 answer        = "This information is not present in the document."
#                 decision_type = "rejected_no_dominance"
#                 print("[QA] ❌ Rejected (no dominance)")

#         # ── CASE 2: Single candidate — evidence-based dynamic decision ────────
#         else:
#             semantic = semantic_similarity(best_answer, retrieved_texts)
#             keyword  = keyword_overlap(best_answer, retrieved_texts)
#             coverage = coverage_score(best_answer, retrieved_texts)
#             print(f"[QA] semantic={semantic:.3f} | keyword={keyword:.3f} | coverage={coverage:.3f}")

#             answer_len = len(best_answer.split())

#             # ── Dynamic thresholds: scale with answer length ──────────────────
#             # Short answers (1–4 words) need high keyword precision (they're
#             # extracted spans). Longer answers need semantic coherence.
#             # Thresholds are derived from the signal magnitudes, not hardcoded.

#             if answer_len <= 4:
#                 # For short extracted spans: keyword overlap is the primary signal.
#                 # We accept if keyword > median of [0,1] and any other signal > noise.
#                 kw_threshold  = 0.5           # lower bound: span must appear in context
#                 aux_threshold = 0.05          # any other signal confirms it's real
#                 if keyword > kw_threshold and (coverage > aux_threshold or semantic > 0.10):
#                     answer        = best_answer
#                     decision_type = "accepted_short_fact"
#                     print("[QA] ✅ Accepted (short factual answer)")
#                 else:
#                     answer        = "This information is not present in the document."
#                     decision_type = "rejected_short_weak"
#                     print("[QA] ❌ Rejected (short weak answer)")

#             else:
#                 # For longer answers: require at least two signals to agree.
#                 # Thresholds are derived from the signal space midpoint (0.5 for
#                 # semantic, half of keyword space). No domain-specific numbers.
#                 strong = (
#                     (keyword > 0.5 and coverage > 0.05) or
#                     (semantic > 0.40 and coverage > 0.04) or
#                     (semantic > 0.50 and keyword > 0.40)
#                 )
#                 if strong:
#                     answer        = best_answer
#                     decision_type = "accepted_strong_evidence"
#                     print("[QA] ✅ Accepted (strong evidence)")
#                 else:
#                     answer        = "This information is not present in the document."
#                     decision_type = "rejected_weak_evidence"
#                     print("[QA] ❌ Rejected (weak evidence)")

#         # ── Final metrics ─────────────────────────────────────────────────────
#         grounding  = compute_answer_grounding(answer, retrieved_texts, question)
#         confidence = compute_confidence(reranker_top, recall_score,
#                                         answer, retrieved_texts, all_scores)
#         model_used = "dynamic_scoring"

#         emit_event(request_id, "stream_start", "✍️ Answer selected...")
#         for _i in range(0, len(answer), 5):
#             emit_event(request_id, "token", answer[_i:_i+5])
#         emit_event(request_id, "agent_done", "✅ Done!")

#         recall_score = compute_recall_at_k(question, retrieved, all_chunks,
#                                            k=len(retrieved))

#     qa_time = time.time() - qa_start
#     print(f"[QA] Done in {qa_time:.1f}s | model={model_used} | "
#           f"grounding={grounding:.1f}% | recall={recall_score:.1f}% | "
#           f"confidence={confidence:.3f} | decision={decision_type}")

#     if not answer.strip():
#         print("[QA] 🚨 Empty answer safety net")
#         answer = "Could not find a relevant answer in the PDF."

#     state["answer"]                       = answer
#     state["metrics"]["qa_time_sec"]       = round(qa_time, 2)
#     state["metrics"]["confidence_score"]  = round(confidence * 100, 2)
#     state["metrics"]["retrieval_score"]   = retrieval_score
#     state["metrics"]["context_precision"] = context_precision
#     state["metrics"]["answer_grounding"]  = grounding
#     state["metrics"]["recall_at_k"]       = recall_score
#     state["metrics"]["llm_calls"]         = llm_calls
#     state["metrics"]["model_used"]        = model_used
#     state["metrics"]["chunks_retrieved"]  = len(retrieved)
#     state["metrics"]["type"]              = "qa"
#     state["metrics"]["decision_type"]     = decision_type
#     state["metrics"]["confidence_raw"]    = round(confidence, 4)
#     return state


# def node_validate(state: DocState) -> DocState:
#     answer = state["answer"]
#     retry  = state.get("retry_count", 0)
#     if len(answer.strip()) < 3 and retry < 2:
#         state["retry_count"] = retry + 1
#         state["answer"]      = ""
#         return state

#     total_time    = time.time() - state["start_time"]
#     output_words  = len(answer.split())
#     output_tokens = output_words * 1.3
#     extract_time  = state["metrics"].get("extraction_time_sec", 0)
#     llm_time      = max(total_time - extract_time, 1)
#     tps           = round(output_tokens / llm_time, 2) if llm_time > 0 else 0
#     m             = state["metrics"]

#     m["response_time_sec"]    = round(total_time, 2)
#     m["extraction_time_sec"]  = m.get("extraction_time_sec", 0)
#     m["pages_processed"]      = state.get("page_count", 0)
#     m["characters_processed"] = state.get("char_count", 0)
#     m["words_processed"]      = len(state.get("extracted_text", "").split())
#     if m.get("type") == "summary":
#         m["summary_time_sec"]     = m.get("summary_time_sec", 0)
#         m["summary_length_words"] = len(answer.split())
#     if m.get("type") == "qa":
#         m["qa_time_sec"]      = m.get("qa_time_sec", 0)
#         m["confidence_score"] = m.get("confidence_score", 0)

#     m["ttft_sec"]          = round(total_time, 2)
#     m["e2e_latency_sec"]   = round(total_time, 2)
#     m["tps"]               = tps
#     m["doc_type"]          = state.get("doc_type", "general")
#     m["query_type"]        = state.get("query_type", "")
#     m["chunks_created"]    = m.get("chunks_created", 0)
#     m["retry_count"]       = retry
#     m["model_used"]        = m.get("model_used", "roberta")
#     m["llm_calls"]         = m.get("llm_calls", 0)
#     m["retrieval_score"]   = m.get("retrieval_score", 0)
#     m["context_precision"] = m.get("context_precision", 0)
#     m["answer_grounding"]  = m.get("answer_grounding", 0)
#     m["recall_at_k"]       = m.get("recall_at_k", 0)

#     if m.get("type") == "summary":
#         m["parallel_workers"] = m.get("parallel_workers", 0)
#         m["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
#         m["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)
#     if m.get("type") == "qa":
#         m["chunks_retrieved"] = m.get("chunks_retrieved", 0)
#         m["decision_type"]    = m.get("decision_type", "accepted")
#         m["confidence_raw"]   = m.get("confidence_raw", 0.0)

#     state["metrics"] = m
#     return state