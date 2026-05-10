
#come here if something goes wrong above i think
import re
from docmind_rag.config.settings import qa_pipeline, FACTUAL_TOP_K
from docmind_rag.models.llm import call_llama, call_llama_streaming
from docmind_rag.core.prompts import REACT_PROMPT, QA_PROMPT
from docmind_rag.models.reranker import rerank_docs, protect_exact_matches
from docmind_rag.services.retrieval import multi_query_retrieve
from docmind_rag.events.events import emit_event
from docmind_rag.utils.helpers import (
    clean_artifacts, _is_cop_out_answer,
    clean_chunk_text, validate_and_correct_span
)
from docmind_rag.utils.text import reorder_by_question, extract_numeric_answer
from docmind_rag.utils.metrics import (
    compute_answer_grounding,
    compute_retrieval_score,
    compute_context_precision,
)

# ============================================================
# HELPERS
# ============================================================

def should_validate(answer: str, question: str) -> bool:
    answer = answer.strip()
    if len(answer.split()) <= 12:
        return True
    if "\n" in answer or "," in answer:
        return False
    if question.lower().startswith(("what is", "who is", "name", "define")):
        return True
    return False


def _is_echo_answer(answer: str, question: str) -> bool:
    q_tokens = set(re.findall(r'\b\w+\b', question.lower()))
    a_tokens = set(re.findall(r'\b\w+\b', answer.lower()))
    return len(a_tokens - q_tokens) == 0


# ============================================================
# FIX — Generic reasoning/echo filter
# Replaces hardcoded startswith("since", "because", ...) check.
# Uses token overlap: if answer adds fewer than 3 new tokens
# beyond what's in the question, it's likely an echo or
# reasoning preamble rather than a real answer.
# Short answers (<=3 words) are never wiped — they may be
# legitimate single-word or short-phrase answers.
# ============================================================

def clean_reasoning_answer(answer: str, question: str) -> str:
    """
    Remove answers that are pure echoes of the question.
    Generic — no hardcoded trigger words.
    """
    words = answer.strip().split()

    # Never wipe short answers — they may be legitimate facts
    if len(words) <= 3:
        return answer

    q_tokens = set(re.findall(r'\b\w{3,}\b', question.lower()))
    a_tokens = set(re.findall(r'\b\w{3,}\b', answer.lower()))
    new_tokens = a_tokens - q_tokens

    if len(new_tokens) < 3:
        print(f"[ReasoningFilter] ❌ Answer adds only {len(new_tokens)} new tokens → wiping")
        return ""

    return answer


# ============================================================
# RoBERTa QA — kept for import compatibility, not called
# ============================================================

def roberta_qa(question: str, chunks: list):
    """Kept for import compatibility. Not called in current pipeline."""
    return "", 0.0


# ============================================================
# ReAct AGENT
# ============================================================

def react_agent(question, faiss_index, query_type, all_chunks, request_id, recall_score):
    grounding = 0.0
    """
    Returns:
        answer, model_used, steps, retrieval_score,
        context_precision, grounding, llm_calls, context_chunks
    """
    MAX_STEPS  = 3
    scratchpad = ""
    model_used = "llama_react"
    llm_calls  = 0

    # ── Retrieval ─────────────────────────────────────────────
    # ── Retrieval ─────────────────────────────────────────────
    context_chunks = [
        d if isinstance(d, str) else d.page_content
        for d in all_chunks
    ]

    retrieval_score = compute_retrieval_score(question, context_chunks)
    context_precision = compute_context_precision(question, context_chunks)
   

    print(
        f"[ReAct] Starting | {query_type} | {len(context_chunks)} chunks | "
        f"retrieval={retrieval_score:.1f}%"
    )

    emit_event(
        request_id,
        "agent_start",
        f"🤖 Agent starting | {query_type} | {len(context_chunks)} chunks"
    )

    # ── Step loop ─────────────────────────────────────────────
    for step in range(MAX_STEPS):
        ranked_chunks = reorder_by_question(question, context_chunks)
        top_chunks    = ranked_chunks[:7]

        print("\n========== DEBUG: TOP CHUNKS ==========")
        for i, chunk in enumerate(top_chunks):
            print(f"\n--- Chunk {i+1} ---")
            print(chunk[:300].replace("\n", " "))
        print("======================================\n")

        context = "\n\n---\n\n".join(clean_chunk_text(c) for c in top_chunks)

        print("\n[DEBUG] ===== CLEANED CONTEXT PREVIEW =====")
        print(context[:200])
        print("=========================================\n")
        print(f"[DEBUG] Full context sent to LLM:\n{context[:2500]}")
        raw = call_llama(
            REACT_PROMPT.format(
                question=question,
                context=context[:2500],
                scratchpad=scratchpad if scratchpad else "None yet"
            ),
            temperature=0.0
        )
        llm_calls += 1
        print(f"[ReAct] Step {step+1}: {raw[:120].strip()}")

        # ── Parse LLM output ──────────────────────────────────
        action       = ""
        action_input = ""
        lines        = raw.split("\n")

        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("Thought:"):
                pass
            elif line.startswith("Action:"):
                action_raw   = line.replace("Action:", "").strip()
                action_lower = action_raw.lower()
                if "final" in action_lower or "answer" in action_lower:
                    action = "final_answer"
                    # capture inline answer e.g. "final_answer: Yes"
                    if ":" in action_raw:
                        inline = action_raw.split(":", 1)[1].strip()
                        if inline:
                            action_input = inline
                elif "search" in action_lower or "more" in action_lower:
                    action = "search_more"
                else:
                    action       = "final_answer"
                    action_input = action_raw
                    print(f"[ReAct] ⚠️ Answer rescued from Action field: '{action_raw[:60]}'")
            elif line.startswith("Input:"):
                if not action_input:
                    action_input = line.replace("Input:", "").strip()
                    for j in range(i + 1, len(lines)):
                        next_line = lines[j].strip()
                        if not next_line:
                            continue
                        if next_line.startswith(("Thought:", "Action:", "Input:")):
                            break
                        action_input += " " + next_line

        if action_input:
            action_input = clean_artifacts(action_input)
        if not action:
            action = "final_answer"
            action_input = ""
        # 🔥 FIX: recover answer if parsing failed
        action_input = clean_artifacts(action_input) if action_input else ""

        if not action_input or len(action_input.strip()) < 2:
            match = re.search(r'final_answer\s*:\s*(.+)', raw, re.IGNORECASE)
            if match:
                recovered = match.group(1).strip()
                if recovered:
                    print("[ReAct] ⚠️ Recovering answer from raw output")
                    action_input = recovered
       
        if "final_answer" in action:

            print(f"[DEBUG] action_input raw: repr={repr(action_input)}")
            answer = action_input.strip() if action_input else ""
            print(f"[DEBUG] answer after strip: repr={repr(answer)}")

            # 🔥 FIXED fallback
            
            if answer.strip().lower() in [
    "this information is not present in the document.",
    "not present",
    "not found"
]:
                if retrieval_score > 30:  # ← retrieval_score is already computed above, no sentinel problem
                    print("[ReAct] ⚠️ Likely wrong refusal → retrying with QA prompt")
                    ranked = reorder_by_question(question, context_chunks)
                    ctx = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:3])
                    answer = call_llama(
                        QA_PROMPT.format(context=ctx[:2500], question=question),
                        temperature=0.0
                    )
                    llm_calls += 1
            # cleaning
            if not answer:
                answer = "This information is not present in the document."
            elif answer.lower() == question.strip().lower():
                answer = "This information is not present in the document."
            else:
                cleaned = clean_reasoning_answer(answer, question)
                if cleaned:
                    answer = cleaned

            print(f"[DEBUG] answer before grounding: repr={repr(answer)}")
            # ✅ MUST ADD THIS
            grounding = compute_answer_grounding(answer, context_chunks, question)

            return (
                answer,
                "llama_react",
                step + 1,
                retrieval_score,
                context_precision,
                grounding,
                llm_calls,
                context_chunks
            )
           

        elif "search" in action:
            print("[ReAct] ❌ Skipping search → NOT FOUND")

            return (
                "This information is not present in the document.",
                "llama_react_fail",
                step + 1,
                retrieval_score,
                context_precision,
                0.0,
                llm_calls,
                context_chunks
            )

        else:
            if len(raw) > 20:
                grounding = compute_answer_grounding(raw, context_chunks)
                return (
                    raw, "llama_react_direct", step + 1,
                    retrieval_score, context_precision,
                    grounding, llm_calls, context_chunks
                )

    # ── Max steps — final direct answer ───────────────────────
    emit_event(request_id, "agent_action", "⚡ Synthesizing final answer...")
    ranked  = reorder_by_question(question, context_chunks)
    ctx     = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:3])

    emit_event(request_id, "stream_start", "✍️ Generating final answer...")
    final, ttft = call_llama_streaming(
        QA_PROMPT.format(context=ctx[:2500], question=question),
        request_id=request_id,
        temperature=0.0
    )
    llm_calls += 1
    grounding = compute_answer_grounding(final, context_chunks)

    return (
        final, "llama_react_final", MAX_STEPS,
        retrieval_score, context_precision,
        grounding, llm_calls, context_chunks
    )















# # # #come here if something goes wrong above i , chapter 4 works 
# import re
# from docmind_rag.config.settings import qa_pipeline, FACTUAL_TOP_K
# from docmind_rag.models.llm import call_llama, call_llama_streaming
# from docmind_rag.core.prompts import REACT_PROMPT, QA_PROMPT
# from docmind_rag.models.reranker import rerank_docs, protect_exact_matches
# from docmind_rag.services.retrieval import multi_query_retrieve
# from docmind_rag.events.events import emit_event
# from docmind_rag.utils.helpers import (
#     clean_artifacts, _is_cop_out_answer,
#     clean_chunk_text, validate_and_correct_span
# )
# from docmind_rag.utils.text import reorder_by_question, extract_numeric_answer
# from docmind_rag.utils.metrics import (
#     compute_answer_grounding,
#     compute_retrieval_score,
#     compute_context_precision,
# )

# # ============================================================
# # HELPERS
# # ============================================================

# def should_validate(answer: str, question: str) -> bool:
#     answer = answer.strip()
#     if len(answer.split()) <= 12:
#         return True
#     if "\n" in answer or "," in answer:
#         return False
#     if question.lower().startswith(("what is", "who is", "name", "define")):
#         return True
#     return False


# def _is_echo_answer(answer: str, question: str) -> bool:
#     q_tokens = set(re.findall(r'\b\w+\b', question.lower()))
#     a_tokens = set(re.findall(r'\b\w+\b', answer.lower()))
#     return len(a_tokens - q_tokens) == 0


# # ============================================================
# # FIX — Generic reasoning/echo filter
# # Replaces hardcoded startswith("since", "because", ...) check.
# # Uses token overlap: if answer adds fewer than 3 new tokens
# # beyond what's in the question, it's likely an echo or
# # reasoning preamble rather than a real answer.
# # Short answers (<=3 words) are never wiped — they may be
# # legitimate single-word or short-phrase answers.
# # ============================================================

# def clean_reasoning_answer(answer: str, question: str) -> str:
#     """
#     Remove answers that are pure echoes of the question.
#     Generic — no hardcoded trigger words.
#     """
#     words = answer.strip().split()

#     # Never wipe short answers — they may be legitimate facts
#     if len(words) <= 3:
#         return answer

#     q_tokens = set(re.findall(r'\b\w{3,}\b', question.lower()))
#     a_tokens = set(re.findall(r'\b\w{3,}\b', answer.lower()))
#     new_tokens = a_tokens - q_tokens

#     if len(new_tokens) < 3:
#         print(f"[ReasoningFilter] ❌ Answer adds only {len(new_tokens)} new tokens → wiping")
#         return ""

#     return answer


# # ============================================================
# # RoBERTa QA — kept for import compatibility, not called
# # ============================================================

# def roberta_qa(question: str, chunks: list):
#     """Kept for import compatibility. Not called in current pipeline."""
#     return "", 0.0


# # ============================================================
# # ReAct AGENT
# # ============================================================

# def react_agent(question, faiss_index, query_type, all_chunks, request_id, recall_score):
#     grounding = 0.0
#     """
#     Returns:
#         answer, model_used, steps, retrieval_score,
#         context_precision, grounding, llm_calls, context_chunks
#     """
#     MAX_STEPS  = 3
#     scratchpad = ""
#     model_used = "llama_react"
#     llm_calls  = 0

#     # ── Retrieval ─────────────────────────────────────────────
#     # ── Retrieval ─────────────────────────────────────────────
#     context_chunks = [
#         d if isinstance(d, str) else d.page_content
#         for d in all_chunks
#     ]

#     retrieval_score = compute_retrieval_score(question, context_chunks)
#     context_precision = compute_context_precision(question, context_chunks)
   

#     print(
#         f"[ReAct] Starting | {query_type} | {len(context_chunks)} chunks | "
#         f"retrieval={retrieval_score:.1f}%"
#     )

#     emit_event(
#         request_id,
#         "agent_start",
#         f"🤖 Agent starting | {query_type} | {len(context_chunks)} chunks"
#     )

#     # ── Step loop ─────────────────────────────────────────────
#     for step in range(MAX_STEPS):
#         ranked_chunks = reorder_by_question(question, context_chunks)
#         top_chunks    = ranked_chunks[:7]

#         print("\n========== DEBUG: TOP CHUNKS ==========")
#         for i, chunk in enumerate(top_chunks):
#             print(f"\n--- Chunk {i+1} ---")
#             print(chunk[:300].replace("\n", " "))
#         print("======================================\n")

#         context = "\n\n---\n\n".join(clean_chunk_text(c) for c in top_chunks)

#         print("\n[DEBUG] ===== CLEANED CONTEXT PREVIEW =====")
#         print(context[:200])
#         print("=========================================\n")
#         print(f"[DEBUG] Full context sent to LLM:\n{context[:2500]}")
#         raw = call_llama(
#             REACT_PROMPT.format(
#                 question=question,
#                 context=context[:2500],
#                 scratchpad=scratchpad if scratchpad else "None yet"
#             ),
#             temperature=0.0
#         )
#         llm_calls += 1
#         print(f"[ReAct] Step {step+1}: {raw[:120].strip()}")

#         # ── Parse LLM output ──────────────────────────────────
#         action       = ""
#         action_input = ""
#         lines        = raw.split("\n")

#         for i, line in enumerate(lines):
#             line = line.strip()
#             if line.startswith("Thought:"):
#                 pass
#             elif line.startswith("Action:"):
#                 action_raw   = line.replace("Action:", "").strip()
#                 action_lower = action_raw.lower()
#                 if "final" in action_lower or "answer" in action_lower:
#                     action = "final_answer"
#                     # capture inline answer e.g. "final_answer: Yes"
#                     if ":" in action_raw:
#                         inline = action_raw.split(":", 1)[1].strip()
#                         if inline:
#                             action_input = inline
#                 elif "search" in action_lower or "more" in action_lower:
#                     action = "search_more"
#                 else:
#                     action       = "final_answer"
#                     action_input = action_raw
#                     print(f"[ReAct] ⚠️ Answer rescued from Action field: '{action_raw[:60]}'")
#             elif line.startswith("Input:"):
#                 if not action_input:
#                     action_input = line.replace("Input:", "").strip()
#                     for j in range(i + 1, len(lines)):
#                         next_line = lines[j].strip()
#                         if not next_line:
#                             continue
#                         if next_line.startswith(("Thought:", "Action:", "Input:")):
#                             break
#                         action_input += " " + next_line

#         if action_input:
#             action_input = clean_artifacts(action_input)

#         if not action:
#             action       = "final_answer"
#             action_input = clean_artifacts(raw)

        

        
#         if "final_answer" in action:

#             print(f"[DEBUG] action_input raw: repr={repr(action_input)}")
#             answer = action_input.strip() if action_input else ""
#             print(f"[DEBUG] answer after strip: repr={repr(answer)}")

#             # 🔥 FIXED fallback
            
#             if "not present" in answer.lower():
#                 if retrieval_score > 30:  # ← retrieval_score is already computed above, no sentinel problem
#                     print("[ReAct] ⚠️ Likely wrong refusal → retrying with QA prompt")
#                     ranked = reorder_by_question(question, context_chunks)
#                     ctx = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:3])
#                     answer = call_llama(
#                         QA_PROMPT.format(context=ctx[:2500], question=question),
#                         temperature=0.0
#                     )
#                     llm_calls += 1
#             # cleaning
#             if not answer:
#                 answer = "This information is not present in the document."
#             elif answer.lower() == question.strip().lower():
#                 answer = "This information is not present in the document."
#             else:
#                 cleaned = clean_reasoning_answer(answer, question)
#                 if cleaned:
#                     answer = cleaned

#             print(f"[DEBUG] answer before grounding: repr={repr(answer)}")
#             # ✅ MUST ADD THIS
#             grounding = compute_answer_grounding(answer, context_chunks, question)

#             return (
#                 answer,
#                 "llama_react",
#                 step + 1,
#                 retrieval_score,
#                 context_precision,
#                 grounding,
#                 llm_calls,
#                 context_chunks
#             )
          
#         elif "search" in action:
#             print("[ReAct] ❌ Skipping search → NOT FOUND")

#             return (
#                 "This information is not present in the document.",
#                 "llama_react_fail",
#                 step + 1,
#                 retrieval_score,
#                 context_precision,
#                 0.0,
#                 llm_calls,
#                 context_chunks
#             )

#         else:
#             if len(raw) > 20:
#                 grounding = compute_answer_grounding(raw, context_chunks)
#                 return (
#                     raw, "llama_react_direct", step + 1,
#                     retrieval_score, context_precision,
#                     grounding, llm_calls, context_chunks
#                 )

#     # ── Max steps — final direct answer ───────────────────────
#     emit_event(request_id, "agent_action", "⚡ Synthesizing final answer...")
#     ranked  = reorder_by_question(question, context_chunks)
#     ctx     = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:3])

#     emit_event(request_id, "stream_start", "✍️ Generating final answer...")
#     final, ttft = call_llama_streaming(
#         QA_PROMPT.format(context=ctx[:2500], question=question),
#         request_id=request_id,
#         temperature=0.0
#     )
#     llm_calls += 1
#     grounding = compute_answer_grounding(final, context_chunks)

#     return (
#         final, "llama_react_final", MAX_STEPS,
#         retrieval_score, context_precision,
#         grounding, llm_calls, context_chunks
#     )

































# # #come here if something goes wrong above i think
# # import re
# # from docmind_rag.config.settings import qa_pipeline, FACTUAL_TOP_K
# # from docmind_rag.models.llm import call_llama, call_llama_streaming
# # from docmind_rag.core.prompts import REACT_PROMPT, QA_PROMPT
# # from docmind_rag.models.reranker import rerank_docs, protect_exact_matches
# # from docmind_rag.services.retrieval import multi_query_retrieve
# # from docmind_rag.events.events import emit_event
# # from docmind_rag.utils.helpers import (
# #     clean_artifacts, _is_cop_out_answer,
# #     clean_chunk_text, validate_and_correct_span
# # )
# # from docmind_rag.utils.text import reorder_by_question, extract_numeric_answer
# # from docmind_rag.utils.metrics import (
# #     compute_answer_grounding,
# #     compute_retrieval_score,
# #     compute_context_precision,
# # )

# # # ============================================================
# # # HELPERS
# # # ============================================================

# # def should_validate(answer: str, question: str) -> bool:
# #     answer = answer.strip()
# #     if len(answer.split()) <= 12:
# #         return True
# #     if "\n" in answer or "," in answer:
# #         return False
# #     if question.lower().startswith(("what is", "who is", "name", "define")):
# #         return True
# #     return False


# # def _is_echo_answer(answer: str, question: str) -> bool:
# #     q_tokens = set(re.findall(r'\b\w+\b', question.lower()))
# #     a_tokens = set(re.findall(r'\b\w+\b', answer.lower()))
# #     return len(a_tokens - q_tokens) == 0


# # # ============================================================
# # # FIX — Generic reasoning/echo filter
# # # Replaces hardcoded startswith("since", "because", ...) check.
# # # Uses token overlap: if answer adds fewer than 3 new tokens
# # # beyond what's in the question, it's likely an echo or
# # # reasoning preamble rather than a real answer.
# # # Short answers (<=3 words) are never wiped — they may be
# # # legitimate single-word or short-phrase answers.
# # # ============================================================

# # def clean_reasoning_answer(answer: str, question: str) -> str:
# #     """
# #     Remove answers that are pure echoes of the question.
# #     Generic — no hardcoded trigger words.
# #     """
# #     words = answer.strip().split()

# #     # Never wipe short answers — they may be legitimate facts
# #     if len(words) <= 3:
# #         return answer

# #     q_tokens = set(re.findall(r'\b\w{3,}\b', question.lower()))
# #     a_tokens = set(re.findall(r'\b\w{3,}\b', answer.lower()))
# #     new_tokens = a_tokens - q_tokens

# #     if len(new_tokens) < 3:
# #         print(f"[ReasoningFilter] ❌ Answer adds only {len(new_tokens)} new tokens → wiping")
# #         return ""

# #     return answer


# # # ============================================================
# # # RoBERTa QA — kept for import compatibility, not called
# # # ============================================================

# # def roberta_qa(question: str, chunks: list):
# #     """Kept for import compatibility. Not called in current pipeline."""
# #     return "", 0.0


# # # ============================================================
# # # ReAct AGENT
# # # ============================================================

# # def react_agent(question, faiss_index, query_type, all_chunks, request_id, recall_score):
# #     grounding = 0.0
# #     """
# #     Returns:
# #         answer, model_used, steps, retrieval_score,
# #         context_precision, grounding, llm_calls, context_chunks
# #     """
# #     MAX_STEPS  = 3
# #     scratchpad = ""
# #     model_used = "llama_react"
# #     llm_calls  = 0

# #     # ── Retrieval ─────────────────────────────────────────────
# #     # ── Retrieval ─────────────────────────────────────────────
# #     context_chunks = [
# #         d if isinstance(d, str) else d.page_content
# #         for d in all_chunks
# #     ]

# #     retrieval_score = compute_retrieval_score(question, context_chunks)
# #     context_precision = compute_context_precision(question, context_chunks)
   

# #     print(
# #         f"[ReAct] Starting | {query_type} | {len(context_chunks)} chunks | "
# #         f"retrieval={retrieval_score:.1f}%"
# #     )

# #     emit_event(
# #         request_id,
# #         "agent_start",
# #         f"🤖 Agent starting | {query_type} | {len(context_chunks)} chunks"
# #     )

# #     # ── Step loop ─────────────────────────────────────────────
# #     for step in range(MAX_STEPS):
# #         ranked_chunks = reorder_by_question(question, context_chunks)
# #         top_chunks    = ranked_chunks[:7]

# #         print("\n========== DEBUG: TOP CHUNKS ==========")
# #         for i, chunk in enumerate(top_chunks):
# #             print(f"\n--- Chunk {i+1} ---")
# #             print(chunk[:300].replace("\n", " "))
# #         print("======================================\n")

# #         context = "\n\n---\n\n".join(clean_chunk_text(c) for c in top_chunks)

# #         print("\n[DEBUG] ===== CLEANED CONTEXT PREVIEW =====")
# #         print(context[:200])
# #         print("=========================================\n")
# #         print(f"[DEBUG] Full context sent to LLM:\n{context[:2500]}")
# #         raw = call_llama(
# #             REACT_PROMPT.format(
# #                 question=question,
# #                 context=context[:2500],
# #                 scratchpad=scratchpad if scratchpad else "None yet"
# #             ),
# #             temperature=0.0
# #         )
# #         llm_calls += 1
# #         print(f"[ReAct] Step {step+1}: {raw[:120].strip()}")

# #         # ── Parse LLM output ──────────────────────────────────
# #         action       = ""
# #         action_input = ""
# #         lines        = raw.split("\n")

# #         for i, line in enumerate(lines):
# #             line = line.strip()
# #             if line.startswith("Thought:"):
# #                 pass
# #             elif line.startswith("Action:"):
# #                 action_raw   = line.replace("Action:", "").strip()
# #                 action_lower = action_raw.lower()
# #                 if "final" in action_lower or "answer" in action_lower:
# #                     action = "final_answer"
# #                     # capture inline answer e.g. "final_answer: Yes"
# #                     if ":" in action_raw:
# #                         inline = action_raw.split(":", 1)[1].strip()
# #                         if inline:
# #                             action_input = inline
# #                 elif "search" in action_lower or "more" in action_lower:
# #                     action = "search_more"
# #                 else:
# #                     action       = "final_answer"
# #                     action_input = action_raw
# #                     print(f"[ReAct] ⚠️ Answer rescued from Action field: '{action_raw[:60]}'")
# #             elif line.startswith("Input:"):
# #                 if not action_input:
# #                     action_input = line.replace("Input:", "").strip()
# #                     for j in range(i + 1, len(lines)):
# #                         next_line = lines[j].strip()
# #                         if not next_line:
# #                             continue
# #                         if next_line.startswith(("Thought:", "Action:", "Input:")):
# #                             break
# #                         action_input += " " + next_line

# #         if action_input:
# #             action_input = clean_artifacts(action_input)
# #         if not action:
# #             action       = "final_answer"
# #             action_input = clean_artifacts(raw)
# #         # 🔥 FIX: recover answer if parsing failed
# #         if not action_input and "final_answer" in raw.lower():
# #             print("[ReAct] ⚠️ Parser failed → recovering from raw output")

# #             match = re.search(r'final_answer\s*:\s*(.+)', raw, re.IGNORECASE)
# #             if match:
# #                 action_input = match.group(1).strip()

# #         # ── Handle action ─────────────────────────────────────
# #         # if "final_answer" in action:
# #         #     is_bad = (
# #         #         not action_input
# #         #         or _is_cop_out_answer(action_input, question)
# #         #         or _is_echo_answer(action_input, question)
# #         #     )

# #         #     if is_bad:
# #         #         print("[ReAct] Bad input — using direct QA prompt")
# #         #         emit_event(request_id, "agent_action", "⚡ Using direct prompt...")
# #         #         ranked = reorder_by_question(question, context_chunks)
# #         #         ctx    = "\n\n---\n\n".join(
# #         #             clean_chunk_text(c) for c in ranked[:3]
# #         #         )
# #         #         answer     = call_llama(
# #         #             QA_PROMPT.format(context=ctx[:2500], question=question),
# #         #             temperature=0.0
# #         #         )
# #         #         llm_calls += 1
# #         #         model_used = "llama_react_direct"
# #         #     else:
# #         #         answer = action_input
# #         #         # Apply generic reasoning filter
# #         #         answer = clean_reasoning_answer(answer, question)
# #         #         if not answer:
# #         #             print("[ReAct] ❌ Reasoning filter wiped answer → direct QA")
# #         #             ranked = reorder_by_question(question, context_chunks)
# #         #             ctx    = "\n\n---\n\n".join(
# #         #                 clean_chunk_text(c) for c in ranked[:3]
# #         #             )
# #         #             answer     = call_llama(
# #         #                 QA_PROMPT.format(context=ctx[:2500], question=question),
# #         #                 temperature=0.0
# #         #             )
# #         #             llm_calls += 1
# #         #             model_used = "llama_react_direct"
# #         #         else:
# #         #             model_used = "llama_react"

        
# #         if "final_answer" in action:

# #             print(f"[DEBUG] action_input raw: repr={repr(action_input)}")
# #             answer = action_input.strip() if action_input else ""
# #             print(f"[DEBUG] answer after strip: repr={repr(answer)}")

# #             # 🔥 FIXED fallback
            
# #             if "not present" in answer.lower():
# #                 if retrieval_score > 30:  # ← retrieval_score is already computed above, no sentinel problem
# #                     print("[ReAct] ⚠️ Likely wrong refusal → retrying with QA prompt")
# #                     ranked = reorder_by_question(question, context_chunks)
# #                     ctx = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:3])
# #                     answer = call_llama(
# #                         QA_PROMPT.format(context=ctx[:2500], question=question),
# #                         temperature=0.0
# #                     )
# #                     llm_calls += 1
# #             # cleaning
# #             if not answer:
# #                 answer = "This information is not present in the document."
# #             elif answer.lower() == question.strip().lower():
# #                 answer = "This information is not present in the document."
# #             else:
# #                 cleaned = clean_reasoning_answer(answer, question)
# #                 if cleaned:
# #                     answer = cleaned

# #             print(f"[DEBUG] answer before grounding: repr={repr(answer)}")
# #             # ✅ MUST ADD THIS
# #             grounding = compute_answer_grounding(answer, context_chunks, question)

# #             return (
# #                 answer,
# #                 "llama_react",
# #                 step + 1,
# #                 retrieval_score,
# #                 context_precision,
# #                 grounding,
# #                 llm_calls,
# #                 context_chunks
# #             )
# #              # is_bad = (
# #             #     not action_input
# #             #     or _is_cop_out_answer(action_input, question)
# #             # )

# #             # # Allow short answers ONLY if they are not empty
# #             # if action_input and len(action_input.split()) <= 5:
# #             #     print("[ReAct] ⚡ Short answer — allow")
# #             #     is_bad = False

# #             # if is_bad:
# #             #     print("[ReAct] ⚠️ Weak answer → returning anyway")

# #             # # ✅ ALWAYS assign answer
# #             # answer = action_input

# #             # # ✅ ALWAYS clean
# #             # cleaned = clean_reasoning_answer(answer, question)

# #             # # Only replace if cleaning keeps useful content
# #             # if cleaned:
# #             #     answer = cleaned
# #             # else:
# #             #     print("[ReAct] ⚠️ Cleaning too aggressive → keeping original")
# #             #     answer = action_input

# #             # model_used = "llama_react"

# #             # if should_validate(answer, question):
# #             #     validated = validate_and_correct_span(
# #             #         answer, question, context_chunks
# #             #     )
# #             #     if validated and len(validated) >= len(answer) * 0.6:
# #             #         answer = validated

# #             # grounding = compute_answer_grounding(answer, context_chunks)
# #             # print(f"[ReAct] ✅ Answer at step {step+1} | grounding={grounding:.1f}%")
# #             # emit_event(request_id, "agent_done", f"✅ Answer found at step {step+1}!")
# #             # return (
# #             #     answer, model_used, step + 1,
# #             #     retrieval_score, context_precision,
# #             #     grounding, llm_calls, context_chunks
# #             # )

# #         elif "search" in action:
# #             print("[ReAct] ❌ Skipping search → NOT FOUND")

# #             return (
# #                 "This information is not present in the document.",
# #                 "llama_react_fail",
# #                 step + 1,
# #                 retrieval_score,
# #                 context_precision,
# #                 0.0,
# #                 llm_calls,
# #                 context_chunks
# #             )

# #         else:
# #             if len(raw) > 20:
# #                 grounding = compute_answer_grounding(raw, context_chunks)
# #                 return (
# #                     raw, "llama_react_direct", step + 1,
# #                     retrieval_score, context_precision,
# #                     grounding, llm_calls, context_chunks
# #                 )

# #     # ── Max steps — final direct answer ───────────────────────
# #     emit_event(request_id, "agent_action", "⚡ Synthesizing final answer...")
# #     ranked  = reorder_by_question(question, context_chunks)
# #     ctx     = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:3])

# #     emit_event(request_id, "stream_start", "✍️ Generating final answer...")
# #     final, ttft = call_llama_streaming(
# #         QA_PROMPT.format(context=ctx[:2500], question=question),
# #         request_id=request_id,
# #         temperature=0.0
# #     )
# #     llm_calls += 1
# #     grounding = compute_answer_grounding(final, context_chunks)

# #     return (
# #         final, "llama_react_final", MAX_STEPS,
# #         retrieval_score, context_precision,
# #         grounding, llm_calls, context_chunks
# #     )


# # # #come here if something goes wrong above i , chapter 4 works 
# import re
# from docmind_rag.config.settings import qa_pipeline, FACTUAL_TOP_K
# from docmind_rag.models.llm import call_llama, call_llama_streaming
# from docmind_rag.core.prompts import REACT_PROMPT, QA_PROMPT
# from docmind_rag.models.reranker import rerank_docs, protect_exact_matches
# from docmind_rag.services.retrieval import multi_query_retrieve
# from docmind_rag.events.events import emit_event
# from docmind_rag.utils.helpers import (
#     clean_artifacts, _is_cop_out_answer,
#     clean_chunk_text, validate_and_correct_span
# )
# from docmind_rag.utils.text import reorder_by_question, extract_numeric_answer
# from docmind_rag.utils.metrics import (
#     compute_answer_grounding,
#     compute_retrieval_score,
#     compute_context_precision,
# )

# # ============================================================
# # HELPERS
# # ============================================================

# def should_validate(answer: str, question: str) -> bool:
#     answer = answer.strip()
#     if len(answer.split()) <= 12:
#         return True
#     if "\n" in answer or "," in answer:
#         return False
#     if question.lower().startswith(("what is", "who is", "name", "define")):
#         return True
#     return False


# def _is_echo_answer(answer: str, question: str) -> bool:
#     q_tokens = set(re.findall(r'\b\w+\b', question.lower()))
#     a_tokens = set(re.findall(r'\b\w+\b', answer.lower()))
#     return len(a_tokens - q_tokens) == 0


# # ============================================================
# # FIX — Generic reasoning/echo filter
# # Replaces hardcoded startswith("since", "because", ...) check.
# # Uses token overlap: if answer adds fewer than 3 new tokens
# # beyond what's in the question, it's likely an echo or
# # reasoning preamble rather than a real answer.
# # Short answers (<=3 words) are never wiped — they may be
# # legitimate single-word or short-phrase answers.
# # ============================================================

# def clean_reasoning_answer(answer: str, question: str) -> str:
#     """
#     Remove answers that are pure echoes of the question.
#     Generic — no hardcoded trigger words.
#     """
#     words = answer.strip().split()

#     # Never wipe short answers — they may be legitimate facts
#     if len(words) <= 3:
#         return answer

#     q_tokens = set(re.findall(r'\b\w{3,}\b', question.lower()))
#     a_tokens = set(re.findall(r'\b\w{3,}\b', answer.lower()))
#     new_tokens = a_tokens - q_tokens

#     if len(new_tokens) < 3:
#         print(f"[ReasoningFilter] ❌ Answer adds only {len(new_tokens)} new tokens → wiping")
#         return ""

#     return answer


# # ============================================================
# # RoBERTa QA — kept for import compatibility, not called
# # ============================================================

# def roberta_qa(question: str, chunks: list):
#     """Kept for import compatibility. Not called in current pipeline."""
#     return "", 0.0


# # ============================================================
# # ReAct AGENT
# # ============================================================

# def react_agent(question, faiss_index, query_type, all_chunks, request_id, recall_score):
#     grounding = 0.0
#     """
#     Returns:
#         answer, model_used, steps, retrieval_score,
#         context_precision, grounding, llm_calls, context_chunks
#     """
#     MAX_STEPS  = 3
#     scratchpad = ""
#     model_used = "llama_react"
#     llm_calls  = 0

#     # ── Retrieval ─────────────────────────────────────────────
#     # ── Retrieval ─────────────────────────────────────────────
#     context_chunks = [
#         d if isinstance(d, str) else d.page_content
#         for d in all_chunks
#     ]

#     retrieval_score = compute_retrieval_score(question, context_chunks)
#     context_precision = compute_context_precision(question, context_chunks)
   

#     print(
#         f"[ReAct] Starting | {query_type} | {len(context_chunks)} chunks | "
#         f"retrieval={retrieval_score:.1f}%"
#     )

#     emit_event(
#         request_id,
#         "agent_start",
#         f"🤖 Agent starting | {query_type} | {len(context_chunks)} chunks"
#     )

#     # ── Step loop ─────────────────────────────────────────────
#     for step in range(MAX_STEPS):
#         ranked_chunks = reorder_by_question(question, context_chunks)
#         top_chunks    = ranked_chunks[:7]

#         print("\n========== DEBUG: TOP CHUNKS ==========")
#         for i, chunk in enumerate(top_chunks):
#             print(f"\n--- Chunk {i+1} ---")
#             print(chunk[:300].replace("\n", " "))
#         print("======================================\n")

#         context = "\n\n---\n\n".join(clean_chunk_text(c) for c in top_chunks)

#         print("\n[DEBUG] ===== CLEANED CONTEXT PREVIEW =====")
#         print(context[:200])
#         print("=========================================\n")
#         print(f"[DEBUG] Full context sent to LLM:\n{context[:2500]}")
#         raw = call_llama(
#             REACT_PROMPT.format(
#                 question=question,
#                 context=context[:2500],
#                 scratchpad=scratchpad if scratchpad else "None yet"
#             ),
#             temperature=0.0
#         )
#         llm_calls += 1
#         print(f"[ReAct] Step {step+1}: {raw[:120].strip()}")

#         # ── Parse LLM output ──────────────────────────────────
#         action       = ""
#         action_input = ""
#         lines        = raw.split("\n")

#         for i, line in enumerate(lines):
#             line = line.strip()
#             if line.startswith("Thought:"):
#                 pass
#             elif line.startswith("Action:"):
#                 action_raw   = line.replace("Action:", "").strip()
#                 action_lower = action_raw.lower()
#                 if "final" in action_lower or "answer" in action_lower:
#                     action = "final_answer"
#                     # capture inline answer e.g. "final_answer: Yes"
#                     if ":" in action_raw:
#                         inline = action_raw.split(":", 1)[1].strip()
#                         if inline:
#                             action_input = inline
#                 elif "search" in action_lower or "more" in action_lower:
#                     action = "search_more"
#                 else:
#                     action       = "final_answer"
#                     action_input = action_raw
#                     print(f"[ReAct] ⚠️ Answer rescued from Action field: '{action_raw[:60]}'")
#             elif line.startswith("Input:"):
#                 if not action_input:
#                     action_input = line.replace("Input:", "").strip()
#                     for j in range(i + 1, len(lines)):
#                         next_line = lines[j].strip()
#                         if not next_line:
#                             continue
#                         if next_line.startswith(("Thought:", "Action:", "Input:")):
#                             break
#                         action_input += " " + next_line

#         if action_input:
#             action_input = clean_artifacts(action_input)

#         if not action:
#             action       = "final_answer"
#             action_input = clean_artifacts(raw)

#         # ── Handle action ─────────────────────────────────────
#         # if "final_answer" in action:
#         #     is_bad = (
#         #         not action_input
#         #         or _is_cop_out_answer(action_input, question)
#         #         or _is_echo_answer(action_input, question)
#         #     )

#         #     if is_bad:
#         #         print("[ReAct] Bad input — using direct QA prompt")
#         #         emit_event(request_id, "agent_action", "⚡ Using direct prompt...")
#         #         ranked = reorder_by_question(question, context_chunks)
#         #         ctx    = "\n\n---\n\n".join(
#         #             clean_chunk_text(c) for c in ranked[:3]
#         #         )
#         #         answer     = call_llama(
#         #             QA_PROMPT.format(context=ctx[:2500], question=question),
#         #             temperature=0.0
#         #         )
#         #         llm_calls += 1
#         #         model_used = "llama_react_direct"
#         #     else:
#         #         answer = action_input
#         #         # Apply generic reasoning filter
#         #         answer = clean_reasoning_answer(answer, question)
#         #         if not answer:
#         #             print("[ReAct] ❌ Reasoning filter wiped answer → direct QA")
#         #             ranked = reorder_by_question(question, context_chunks)
#         #             ctx    = "\n\n---\n\n".join(
#         #                 clean_chunk_text(c) for c in ranked[:3]
#         #             )
#         #             answer     = call_llama(
#         #                 QA_PROMPT.format(context=ctx[:2500], question=question),
#         #                 temperature=0.0
#         #             )
#         #             llm_calls += 1
#         #             model_used = "llama_react_direct"
#         #         else:
#         #             model_used = "llama_react"

        
#         if "final_answer" in action:

#             print(f"[DEBUG] action_input raw: repr={repr(action_input)}")
#             answer = action_input.strip() if action_input else ""
#             print(f"[DEBUG] answer after strip: repr={repr(answer)}")

#             # 🔥 FIXED fallback
            
#             if "not present" in answer.lower():
#                 if retrieval_score > 30:  # ← retrieval_score is already computed above, no sentinel problem
#                     print("[ReAct] ⚠️ Likely wrong refusal → retrying with QA prompt")
#                     ranked = reorder_by_question(question, context_chunks)
#                     ctx = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:3])
#                     answer = call_llama(
#                         QA_PROMPT.format(context=ctx[:2500], question=question),
#                         temperature=0.0
#                     )
#                     llm_calls += 1
#             # cleaning
#             if not answer:
#                 answer = "This information is not present in the document."
#             elif answer.lower() == question.strip().lower():
#                 answer = "This information is not present in the document."
#             else:
#                 cleaned = clean_reasoning_answer(answer, question)
#                 if cleaned:
#                     answer = cleaned

#             print(f"[DEBUG] answer before grounding: repr={repr(answer)}")
#             # ✅ MUST ADD THIS
#             grounding = compute_answer_grounding(answer, context_chunks, question)

#             return (
#                 answer,
#                 "llama_react",
#                 step + 1,
#                 retrieval_score,
#                 context_precision,
#                 grounding,
#                 llm_calls,
#                 context_chunks
#             )
#              # is_bad = (
#             #     not action_input
#             #     or _is_cop_out_answer(action_input, question)
#             # )

#             # # Allow short answers ONLY if they are not empty
#             # if action_input and len(action_input.split()) <= 5:
#             #     print("[ReAct] ⚡ Short answer — allow")
#             #     is_bad = False

#             # if is_bad:
#             #     print("[ReAct] ⚠️ Weak answer → returning anyway")

#             # # ✅ ALWAYS assign answer
#             # answer = action_input

#             # # ✅ ALWAYS clean
#             # cleaned = clean_reasoning_answer(answer, question)

#             # # Only replace if cleaning keeps useful content
#             # if cleaned:
#             #     answer = cleaned
#             # else:
#             #     print("[ReAct] ⚠️ Cleaning too aggressive → keeping original")
#             #     answer = action_input

#             # model_used = "llama_react"

#             # if should_validate(answer, question):
#             #     validated = validate_and_correct_span(
#             #         answer, question, context_chunks
#             #     )
#             #     if validated and len(validated) >= len(answer) * 0.6:
#             #         answer = validated

#             # grounding = compute_answer_grounding(answer, context_chunks)
#             # print(f"[ReAct] ✅ Answer at step {step+1} | grounding={grounding:.1f}%")
#             # emit_event(request_id, "agent_done", f"✅ Answer found at step {step+1}!")
#             # return (
#             #     answer, model_used, step + 1,
#             #     retrieval_score, context_precision,
#             #     grounding, llm_calls, context_chunks
#             # )

#         elif "search" in action:
#             print("[ReAct] ❌ Skipping search → NOT FOUND")

#             return (
#                 "This information is not present in the document.",
#                 "llama_react_fail",
#                 step + 1,
#                 retrieval_score,
#                 context_precision,
#                 0.0,
#                 llm_calls,
#                 context_chunks
#             )

#         else:
#             if len(raw) > 20:
#                 grounding = compute_answer_grounding(raw, context_chunks)
#                 return (
#                     raw, "llama_react_direct", step + 1,
#                     retrieval_score, context_precision,
#                     grounding, llm_calls, context_chunks
#                 )

#     # ── Max steps — final direct answer ───────────────────────
#     emit_event(request_id, "agent_action", "⚡ Synthesizing final answer...")
#     ranked  = reorder_by_question(question, context_chunks)
#     ctx     = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:3])

#     emit_event(request_id, "stream_start", "✍️ Generating final answer...")
#     final, ttft = call_llama_streaming(
#         QA_PROMPT.format(context=ctx[:2500], question=question),
#         request_id=request_id,
#         temperature=0.0
#     )
#     llm_calls += 1
#     grounding = compute_answer_grounding(final, context_chunks)

#     return (
#         final, "llama_react_final", MAX_STEPS,
#         retrieval_score, context_precision,
#         grounding, llm_calls, context_chunks
#     )



















# import re
# from docmind_rag.config.settings import qa_pipeline, FACTUAL_TOP_K,embedding_model
# from docmind_rag.models.llm import call_llama, call_llama_streaming
# from docmind_rag.core.prompts import REACT_PROMPT, QA_PROMPT
# from docmind_rag.models.reranker import rerank_docs, protect_exact_matches
# from docmind_rag.services.retrieval import multi_query_retrieve
# from docmind_rag.events.events import emit_event
# from docmind_rag.utils.helpers import (
#     clean_artifacts, _is_cop_out_answer,
#     clean_chunk_text, validate_and_correct_span
# )
# from docmind_rag.utils.text import reorder_by_question, extract_numeric_answer, semantic_rank,normalize_text
# from docmind_rag.utils.metrics import (
#     compute_answer_grounding,
#     compute_retrieval_score,
#     compute_context_precision,
# )


# # ============================================================
# # HELPERS
# # ============================================================

# def should_validate(answer: str, question: str) -> bool:
#     answer = answer.strip()
#     if len(answer.split()) <= 12:
#         return True
#     if "\n" in answer or "," in answer:
#         return False
#     if question.lower().startswith(("what is", "who is", "name", "define")):
#         return True
#     return False


# def _is_echo_answer(answer: str, question: str) -> bool:
#     q_tokens = set(re.findall(r'\b\w+\b', question.lower()))
#     a_tokens = set(re.findall(r'\b\w+\b', answer.lower()))
#     return len(a_tokens - q_tokens) == 0


# # ============================================================
# # FIX — Generic reasoning/echo filter
# # Replaces hardcoded startswith("since", "because", ...) check.
# # Uses token overlap: if answer adds fewer than 3 new tokens
# # beyond what's in the question, it's likely an echo or
# # reasoning preamble rather than a real answer.
# # Short answers (<=3 words) are never wiped — they may be
# # legitimate single-word or short-phrase answers.
# # ============================================================

# def clean_reasoning_answer(answer: str, question: str, retrieved_texts: list) -> str:
#     words = answer.strip().split()

#     # ✅ Never wipe short answers
#     if len(words) <= 3:
#         return answer

#     q_tokens = set(re.findall(r'\b\w{3,}\b', question.lower()))
#     a_tokens = set(re.findall(r'\b\w{3,}\b', answer.lower()))

#     # 🔥 NEW: check overlap with retrieved context
#     context_text = " ".join(retrieved_texts).lower()
#     context_tokens = set(re.findall(r'\b\w{3,}\b', context_text))

#     # tokens in answer that are NOT in question
#     new_tokens = a_tokens - q_tokens

#     # ✅ Keep if at least ONE new token exists in context
#     grounded_tokens = [t for t in new_tokens if t in context_tokens]

#     if len(grounded_tokens) == 0:
#         return ""   # hallucination / drift
#     else:
#         return answer

# # ============================================================
# # RoBERTa QA — kept for import compatibility, not called
# # ============================================================

# def roberta_qa(question: str, chunks: list):
#     """Kept for import compatibility. Not called in current pipeline."""
#     return "", 0.0


# # ============================================================
# # ReAct AGENT
# # ============================================================

# # def react_agent(
# #     question:   str,
# #     faiss_index,
# #     query_type: str,
# #     all_chunks: list,
# #     request_id: str = ""
# # ) -> tuple:
# def react_agent(question, context_chunks, query_type, request_id, embedding_model):
#     # 🔥 FIX: normalize OCR text
#     context_chunks = [normalize_text(c) for c in context_chunks]
#     """
#     Returns:
#         answer, model_used, steps, retrieval_score,
#         context_precision, grounding, llm_calls, context_chunks
#     """
#     MAX_STEPS  = 3
#     scratchpad = ""
#     model_used = "llama_react"
#     llm_calls  = 0

#     # ── Retrieval ─────────────────────────────────────────────
#     # ✅ Use pre-retrieved context ONLY
#     context_chunks = context_chunks  # already passed

#     retrieval_score = 100.0   # or keep from node_qa if you want
#     context_precision = 100.0

#     print(f"[ReAct] Starting | {query_type} | {len(context_chunks)} chunks")

#     retrieval_score = 100.0
#     context_precision = 100.0
#     emit_event(request_id, "agent_start",
#                f"🤖 Agent starting | {query_type} | {len(context_chunks)} chunks retrieved")

#     # ── Step loop ─────────────────────────────────────────────
#     for step in range(MAX_STEPS):
#         ranked_chunks = semantic_rank(question, context_chunks, embedding_model)
#         top_chunks    = ranked_chunks[:7]
#         # 🔥 FIX: always keep first chunk (header)
#         # top_chunks = context_chunks[:1] + ranked_chunks[:6]
#         print("\n========== DEBUG: TOP CHUNKS ==========")
#         for i, chunk in enumerate(top_chunks):
#             print(f"\n--- Chunk {i+1} ---")
#             print(chunk[:300].replace("\n", " "))
#         print("======================================\n")

#         context = "\n\n---\n\n".join(clean_chunk_text(c) for c in top_chunks)

#         print("\n[DEBUG] ===== CLEANED CONTEXT PREVIEW =====")
#         print(context[:200])
#         print("=========================================\n")

#         raw = call_llama(
#             REACT_PROMPT.format(
#                 question=question,
#                 context=context[:2500],
#                 scratchpad=scratchpad if scratchpad else "None yet"
#             ),
#             temperature=0.0
#         )
#         llm_calls += 1
#         print(f"[ReAct] Step {step+1}: {raw[:120].strip()}")

#         # ── Parse LLM output ──────────────────────────────────
#         action       = ""
#         action_input = ""
#         lines        = raw.split("\n")

#         for i, line in enumerate(lines):
#             line = line.strip()
#             if line.startswith("Thought:"):
#                 pass
#             elif line.startswith("Action:"):
#                 action_raw   = line.replace("Action:", "").strip()
#                 action_lower = action_raw.lower()
#                 if "final" in action_lower or "answer" in action_lower:
#                     action = "final_answer"
#                 elif "search" in action_lower or "more" in action_lower:
#                     action = "search_more"
#                 else:
#                     action       = "final_answer"
#                     action_input = action_raw
#                     print(f"[ReAct] ⚠️ Answer rescued from Action field: '{action_raw[:60]}'")
#             elif line.startswith("Input:"):
#                 if not action_input:
#                     action_input = line.replace("Input:", "").strip()
#                     for j in range(i + 1, len(lines)):
#                         next_line = lines[j].strip()
#                         if not next_line:
#                             continue
#                         if next_line.startswith(("Thought:", "Action:", "Input:")):
#                             break
#                         action_input += " " + next_line

#         if action_input:
#             action_input = clean_artifacts(action_input)

#         if not action:
#             action       = "final_answer"
#             action_input = clean_artifacts(raw)

#         # ── Handle action ─────────────────────────────────────
#         if "final_answer" in action:
#             # is_bad = (
#             #     not action_input
#             #     or _is_cop_out_answer(action_input, question)
#             #     or _is_echo_answer(action_input, question)
#             # )
#             is_bad = (
#                 not action_input
#                 or _is_cop_out_answer(action_input, question)
#             )

#             if is_bad:
#                 print("[ReAct] Bad input — using direct QA prompt")
#                 emit_event(request_id, "agent_action", "⚡ Using direct prompt...")
#                 ranked = reorder_by_question(question, context_chunks)
#                 ctx    = "\n\n---\n\n".join(
#                     clean_chunk_text(c) for c in ranked[:6]
#                 )
#                 answer     = call_llama(
#                     QA_PROMPT.format(context=ctx[-2500:], question=question),
#                     temperature=0.0
#                 )
               
#                 llm_calls += 1
#                 model_used = "llama_react_direct"
#                 #  # 🔥 HARD VALIDATION — prevents hallucination
#                 # if answer.strip() and answer.lower() not in ctx.lower():
#                 #     print("[VALIDATE] ❌ Answer not in context → rejecting")
#                 #     answer = "This information is not present in the document."
#             else:
#                 answer = action_input
#                 # Apply generic reasoning filter
#                 # answer = clean_reasoning_answer(answer, question, context_chunks)
#                 if not answer:
#                     print("[ReAct] ❌ Reasoning filter wiped answer → direct QA")
#                     ranked = reorder_by_question(question, context_chunks)
#                     ctx    = "\n\n---\n\n".join(
#                         clean_chunk_text(c) for c in ranked[:6]
#                     )
#                     answer     = call_llama(
#                         QA_PROMPT.format(context=ctx[:2500], question=question),
#                         temperature=0.0
#                     )
#                     llm_calls += 1
#                     model_used = "llama_react_direct"
#                 else:
#                     model_used = "llama_react"

#             # if should_validate(answer, question):
#             #     validated = validate_and_correct_span(
#             #         answer, question, context_chunks
#             #     )
#             #     if validated and len(validated) >= len(answer) * 0.6:
#             #         answer = validated

#             grounding = compute_answer_grounding(answer, context_chunks)
#             print(f"[ReAct] ✅ Answer at step {step+1} | grounding={grounding:.1f}%")
#             emit_event(request_id, "agent_done", f"✅ Answer found at step {step+1}!")
#             return (
#     answer, model_used, step + 1,
#     retrieval_score, context_precision,
#     grounding, llm_calls
# )       # return (
#             #     answer, model_used, step + 1,
#             #     retrieval_score, context_precision,
#             #     grounding, llm_calls, context_chunks
#             # )

#         elif "search" in action:
#             print("[ReAct] ⚠️ Skipping search → forcing final answer")
#             ranked = reorder_by_question(question, context_chunks)
#             ctx    = "\n\n---\n\n".join(
#                 clean_chunk_text(c) for c in ranked[:6]
#             )
#             answer    = call_llama(
#                 QA_PROMPT.format(context=ctx[:2500], question=question),
#                 temperature=0.0
#             )
#             llm_calls += 1
#             grounding  = compute_answer_grounding(answer, context_chunks)
#             return (
#     answer, model_used, step + 1,
#     retrieval_score, context_precision,
#     grounding, llm_calls
# )

#         else:
#             if len(raw) > 20:
#                 grounding = compute_answer_grounding(raw, context_chunks)
#                 return (raw, model_used, step + 1,
#         retrieval_score, context_precision,
#         grounding, llm_calls)
#     # ── Max steps — final direct answer ───────────────────────
#     emit_event(request_id, "agent_action", "⚡ Synthesizing final answer...")
#     ranked  = reorder_by_question(question, context_chunks)
#     ctx     = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:6])

#     emit_event(request_id, "stream_start", "✍️ Generating final answer...")
#     final, ttft = call_llama_streaming(
#         QA_PROMPT.format(context=ctx[:2500], question=question),
#         request_id=request_id,
#         temperature=0.0
#     )
#     llm_calls += 1
#     grounding = compute_answer_grounding(final, context_chunks)

#     return (final, model_used, step + 1,
#         retrieval_score, context_precision,
#         grounding, llm_calls)



























# #come here if something goes wrong above i think
# import re
# from docmind_rag.config.settings import qa_pipeline, FACTUAL_TOP_K
# from docmind_rag.models.llm import call_llama, call_llama_streaming
# from docmind_rag.core.prompts import REACT_PROMPT, QA_PROMPT
# from docmind_rag.models.reranker import rerank_docs, protect_exact_matches
# from docmind_rag.services.retrieval import multi_query_retrieve
# from docmind_rag.events.events import emit_event
# from docmind_rag.utils.helpers import (
#     clean_artifacts, _is_cop_out_answer,
#     clean_chunk_text, validate_and_correct_span
# )
# from docmind_rag.utils.text import reorder_by_question, extract_numeric_answer
# from docmind_rag.utils.metrics import (
#     compute_answer_grounding,
#     compute_retrieval_score,
#     compute_context_precision,
# )


# # ============================================================
# # HELPERS
# # ============================================================

# def should_validate(answer: str, question: str) -> bool:
#     answer = answer.strip()
#     if len(answer.split()) <= 12:
#         return True
#     if "\n" in answer or "," in answer:
#         return False
#     if question.lower().startswith(("what is", "who is", "name", "define")):
#         return True
#     return False


# def _is_echo_answer(answer: str, question: str) -> bool:
#     q_tokens = set(re.findall(r'\b\w+\b', question.lower()))
#     a_tokens = set(re.findall(r'\b\w+\b', answer.lower()))
#     return len(a_tokens - q_tokens) == 0


# # ============================================================
# # FIX — Generic reasoning/echo filter
# # Replaces hardcoded startswith("since", "because", ...) check.
# # Uses token overlap: if answer adds fewer than 3 new tokens
# # beyond what's in the question, it's likely an echo or
# # reasoning preamble rather than a real answer.
# # Short answers (<=3 words) are never wiped — they may be
# # legitimate single-word or short-phrase answers.
# # ============================================================

# def clean_reasoning_answer(answer: str, question: str) -> str:
#     """
#     Remove answers that are pure echoes of the question.
#     Generic — no hardcoded trigger words.
#     """
#     words = answer.strip().split()

#     # Never wipe short answers — they may be legitimate facts
#     if len(words) <= 3:
#         return answer

#     q_tokens = set(re.findall(r'\b\w{3,}\b', question.lower()))
#     a_tokens = set(re.findall(r'\b\w{3,}\b', answer.lower()))
#     new_tokens = a_tokens - q_tokens

#     if len(new_tokens) < 3:
#         print(f"[ReasoningFilter] ❌ Answer adds only {len(new_tokens)} new tokens → wiping")
#         return ""

#     return answer


# # ============================================================
# # RoBERTa QA — kept for import compatibility, not called
# # ============================================================

# def roberta_qa(question: str, chunks: list):
#     """Kept for import compatibility. Not called in current pipeline."""
#     return "", 0.0


# # ============================================================
# # ReAct AGENT
# # ============================================================

# def react_agent(
#     question:   str,
#     faiss_index,
#     query_type: str,
#     all_chunks: list,
#     request_id: str = ""
# ) -> tuple:
#     """
#     Returns:
#         answer, model_used, steps, retrieval_score,
#         context_precision, grounding, llm_calls, context_chunks
#     """
#     MAX_STEPS  = 3
#     scratchpad = ""
#     model_used = "llama_react"
#     llm_calls  = 0

#     # ── Retrieval ─────────────────────────────────────────────
#     initial = multi_query_retrieve(
#         question, faiss_index, k=20,
#         all_chunks=all_chunks, query_type=query_type
#     )
#     apply_pruning = (query_type != "MULTIPART_QA")
#     initial, _, _ = rerank_docs(
#         question, initial,
#         top_k=FACTUAL_TOP_K,
#         apply_pruning=apply_pruning
#     )
#     initial = protect_exact_matches(
#         question, initial, all_chunks, top_k=FACTUAL_TOP_K
#     )
#     context_chunks    = [d.page_content for d in initial]
#     retrieval_score   = compute_retrieval_score(question, initial)
#     context_precision = compute_context_precision(question, initial)

#     print(f"[ReAct] Starting | {query_type} | {len(initial)} chunks | "
#           f"retrieval={retrieval_score:.1f}%")
#     emit_event(request_id, "agent_start",
#                f"🤖 Agent starting | {query_type} | {len(initial)} chunks retrieved")

#     # ── Step loop ─────────────────────────────────────────────
#     for step in range(MAX_STEPS):
#         ranked_chunks = reorder_by_question(question, context_chunks)
#         top_chunks    = ranked_chunks[:7]

#         print("\n========== DEBUG: TOP CHUNKS ==========")
#         for i, chunk in enumerate(top_chunks):
#             print(f"\n--- Chunk {i+1} ---")
#             print(chunk[:300].replace("\n", " "))
#         print("======================================\n")

#         context = "\n\n---\n\n".join(clean_chunk_text(c) for c in top_chunks)

#         print("\n[DEBUG] ===== CLEANED CONTEXT PREVIEW =====")
#         print(context[:200])
#         print("=========================================\n")

#         raw = call_llama(
#             REACT_PROMPT.format(
#                 question=question,
#                 context=context[:2500],
#                 scratchpad=scratchpad if scratchpad else "None yet"
#             ),
#             temperature=0.0
#         )
#         llm_calls += 1
#         print(f"[ReAct] Step {step+1}: {raw[:120].strip()}")

#         # ── Parse LLM output ──────────────────────────────────
#         action       = ""
#         action_input = ""
#         lines        = raw.split("\n")

#         for i, line in enumerate(lines):
#             line = line.strip()
#             if line.startswith("Thought:"):
#                 pass
#             elif line.startswith("Action:"):
#                 action_raw   = line.replace("Action:", "").strip()
#                 action_lower = action_raw.lower()
#                 if "final" in action_lower or "answer" in action_lower:
#                     action = "final_answer"
#                 elif "search" in action_lower or "more" in action_lower:
#                     action = "search_more"
#                 else:
#                     action       = "final_answer"
#                     action_input = action_raw
#                     print(f"[ReAct] ⚠️ Answer rescued from Action field: '{action_raw[:60]}'")
#             elif line.startswith("Input:"):
#                 if not action_input:
#                     action_input = line.replace("Input:", "").strip()
#                     for j in range(i + 1, len(lines)):
#                         next_line = lines[j].strip()
#                         if not next_line:
#                             continue
#                         if next_line.startswith(("Thought:", "Action:", "Input:")):
#                             break
#                         action_input += " " + next_line

#         if action_input:
#             action_input = clean_artifacts(action_input)

#         if not action:
#             action       = "final_answer"
#             action_input = clean_artifacts(raw)

#         # ── Handle action ─────────────────────────────────────
#         if "final_answer" in action:
#             is_bad = (
#                 not action_input
#                 or _is_cop_out_answer(action_input, question)
#                 or _is_echo_answer(action_input, question)
#             )

#             if is_bad:
#                 print("[ReAct] Bad input — using direct QA prompt")
#                 emit_event(request_id, "agent_action", "⚡ Using direct prompt...")
#                 ranked = reorder_by_question(question, context_chunks)
#                 ctx    = "\n\n---\n\n".join(
#                     clean_chunk_text(c) for c in ranked[:3]
#                 )
#                 answer     = call_llama(
#                     QA_PROMPT.format(context=ctx[:2500], question=question),
#                     temperature=0.0
#                 )
#                 llm_calls += 1
#                 model_used = "llama_react_direct"
#             else:
#                 answer = action_input
#                 # Apply generic reasoning filter
#                 answer = clean_reasoning_answer(answer, question)
#                 if not answer:
#                     print("[ReAct] ❌ Reasoning filter wiped answer → direct QA")
#                     ranked = reorder_by_question(question, context_chunks)
#                     ctx    = "\n\n---\n\n".join(
#                         clean_chunk_text(c) for c in ranked[:3]
#                     )
#                     answer     = call_llama(
#                         QA_PROMPT.format(context=ctx[:2500], question=question),
#                         temperature=0.0
#                     )
#                     llm_calls += 1
#                     model_used = "llama_react_direct"
#                 else:
#                     model_used = "llama_react"

#             if should_validate(answer, question):
#                 validated = validate_and_correct_span(
#                     answer, question, context_chunks
#                 )
#                 if validated and len(validated) >= len(answer) * 0.6:
#                     answer = validated

#             grounding = compute_answer_grounding(answer, context_chunks)
#             print(f"[ReAct] ✅ Answer at step {step+1} | grounding={grounding:.1f}%")
#             emit_event(request_id, "agent_done", f"✅ Answer found at step {step+1}!")
#             return (
#                 answer, model_used, step + 1,
#                 retrieval_score, context_precision,
#                 grounding, llm_calls, context_chunks
#             )

#         elif "search" in action:
#             print("[ReAct] ⚠️ Skipping search → forcing final answer")
#             ranked = reorder_by_question(question, context_chunks)
#             ctx    = "\n\n---\n\n".join(
#                 clean_chunk_text(c) for c in ranked[:3]
#             )
#             answer    = call_llama(
#                 QA_PROMPT.format(context=ctx[:2500], question=question),
#                 temperature=0.0
#             )
#             llm_calls += 1
#             grounding  = compute_answer_grounding(answer, context_chunks)
#             return (
#                 answer, "llama_react_forced", step + 1,
#                 retrieval_score, context_precision,
#                 grounding, llm_calls, context_chunks
#             )

#         else:
#             if len(raw) > 20:
#                 grounding = compute_answer_grounding(raw, context_chunks)
#                 return (
#                     raw, "llama_react_direct", step + 1,
#                     retrieval_score, context_precision,
#                     grounding, llm_calls, context_chunks
#                 )

#     # ── Max steps — final direct answer ───────────────────────
#     emit_event(request_id, "agent_action", "⚡ Synthesizing final answer...")
#     ranked  = reorder_by_question(question, context_chunks)
#     ctx     = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:3])

#     emit_event(request_id, "stream_start", "✍️ Generating final answer...")
#     final, ttft = call_llama_streaming(
#         QA_PROMPT.format(context=ctx[:2500], question=question),
#         request_id=request_id,
#         temperature=0.0
#     )
#     llm_calls += 1
#     grounding = compute_answer_grounding(final, context_chunks)

#     return (
#         final, "llama_react_final", MAX_STEPS,
#         retrieval_score, context_precision,
#         grounding, llm_calls, context_chunks
#     )



























# #    got 80 nd 70 - lap claude changes 
# import re
# from docmind_rag.config.settings import qa_pipeline, FACTUAL_TOP_K
# from docmind_rag.models.llm import call_llama, call_llama_streaming
# from docmind_rag.core.prompts import REACT_PROMPT, QA_PROMPT
# from docmind_rag.models.reranker import rerank_docs, protect_exact_matches
# from docmind_rag.services.retrieval import multi_query_retrieve
# from docmind_rag.events.events import emit_event
# from docmind_rag.utils.helpers import (
#     clean_artifacts, _is_cop_out_answer,
#     clean_chunk_text
# )
# from docmind_rag.utils.text import reorder_by_question, extract_numeric_answer
# from docmind_rag.utils.metrics import (
#     compute_answer_grounding,
#     compute_retrieval_score,
#     compute_context_precision
# )


# # ============================================================
# # HELPERS
# # ============================================================


# def _is_echo_answer(answer: str, question: str) -> bool:
#     q_tokens = set(re.findall(r'\b\w+\b', question.lower()))
#     a_tokens = set(re.findall(r'\b\w+\b', answer.lower()))
#     return len(a_tokens - q_tokens) == 0


# # ============================================================
# # RoBERTa QA
# # ============================================================

# def roberta_qa(question: str, chunks: list):
#     best_answer      = ""
#     best_score       = 0.0
#     best_empty_score = 0.0

#     for chunk in chunks:
#         try:
#             result = qa_pipeline(question=question, context=chunk[:3000])
#             ans    = (result.get('answer') or "").strip()
#             score  = result.get('score', 0.0)
#             if ans:
#                 if score > best_score:
#                     best_score  = score
#                     best_answer = ans
#             else:
#                 if score > best_empty_score:
#                     best_empty_score = score
#         except Exception:
#             continue

#     if not best_answer and best_empty_score >= 0.4:
#         regex_ans = extract_numeric_answer(question, chunks)
#         if regex_ans:
#             print(f"[RoBERTa] Rescued by regex: '{regex_ans}'")
#             return regex_ans, round(best_empty_score, 4)

#     return best_answer.strip(), round(best_score, 4)

# def clean_reasoning_answer(answer: str, question: str) -> str:
#     words = answer.strip().split()

#     # Do NOT touch short answers
#     if len(words) <= 3:
#         return answer

#     q_tokens = set(re.findall(r'\b\w{3,}\b', question.lower()))
#     a_tokens = set(re.findall(r'\b\w{3,}\b', answer.lower()))

#     new_tokens = a_tokens - q_tokens

#     if len(new_tokens) < 3:
#         return ""

#     return answer
# # ============================================================
# # ReAct AGENT
# # ============================================================

# def react_agent(
#     question:   str,
#     faiss_index,
#     query_type: str,
#     all_chunks: list,
#     request_id: str = ""
# ) -> tuple:
#     """
#     Returns:
#         answer, model_used, steps, retrieval_score,
#         context_precision, grounding, llm_calls, context_chunks
#     """
#     MAX_STEPS  = 3
#     scratchpad = ""
#     model_used = "llama_react"
#     llm_calls  = 0

#     # ── Retrieval ─────────────────────────────────────────────
#     initial = multi_query_retrieve(
#         question, faiss_index, k=20,
#         all_chunks=all_chunks, query_type=query_type
#     )
#     apply_pruning = (query_type != "MULTIPART_QA")
#     initial, _, _ = rerank_docs(
#         question, initial,
#         top_k=FACTUAL_TOP_K,
#         apply_pruning=apply_pruning
#     )
#     initial = protect_exact_matches(
#         question, initial, all_chunks, top_k=FACTUAL_TOP_K
#     )
#     context_chunks    = [d.page_content for d in initial]
#     retrieval_score   = compute_retrieval_score(question, initial)
#     context_precision = compute_context_precision(question, initial)

#     print(f"[ReAct] Starting | {query_type} | {len(initial)} chunks | "
#           f"retrieval={retrieval_score:.1f}%")
#     emit_event(request_id, "agent_start",
#                f"🤖 Agent starting | {query_type} | {len(initial)} chunks retrieved")

#     # ── Step loop ─────────────────────────────────────────────
#     for step in range(MAX_STEPS):
#         ranked_chunks = reorder_by_question(question, context_chunks)
#         top_chunks    = ranked_chunks[:7]

#         # Debug
#         print("\n========== DEBUG: TOP CHUNKS ==========")
#         for i, chunk in enumerate(top_chunks):
#             print(f"\n--- Chunk {i+1} ---")
#             print(chunk[:300].replace("\n", " "))
#         print("======================================\n")

#         context = "\n\n---\n\n".join(clean_chunk_text(c) for c in top_chunks)

#         print("\n[DEBUG] ===== CLEANED CONTEXT PREVIEW =====")
#         print(context[:200])
#         print("=========================================\n")

#         raw = call_llama(
#             REACT_PROMPT.format(
#                 question=question,
#                 context=context[:2500],
#                 scratchpad=scratchpad if scratchpad else "None yet"
#             ),
#             temperature=0.0
#         )
#         llm_calls += 1
#         print(f"[ReAct] Step {step+1}: {raw[:120].strip()}")

#         # ── Parse LLM output ──────────────────────────────────
#         action       = ""
#         action_input = ""
#         lines        = raw.split("\n")

#         for i, line in enumerate(lines):
#             line = line.strip()
#             if line.startswith("Thought:"):
#                 pass  # logged only
#             elif line.startswith("Action:"):
#                 action_raw   = line.replace("Action:", "").strip()
#                 action_lower = action_raw.lower()
#                 if "final" in action_lower or "answer" in action_lower:
#                     action = "final_answer"
#                 elif "search" in action_lower or "more" in action_lower:
#                     action = "search_more"
#                 else:
#                     # LLM put answer directly in Action field — rescue it
#                     action       = "final_answer"
#                     action_input = action_raw
#                     print(f"[ReAct] ⚠️ Answer rescued from Action field: '{action_raw[:60]}'")
#             elif line.startswith("Input:"):
#                 if not action_input:
#                     action_input = line.replace("Input:", "").strip()
#                     for j in range(i + 1, len(lines)):
#                         next_line = lines[j].strip()
#                         if not next_line:
#                             continue
#                         if next_line.startswith(("Thought:", "Action:", "Input:")):
#                             break
#                         action_input += " " + next_line

#         if action_input:
#             action_input = clean_artifacts(action_input)

#         if not action:
#             action       = "final_answer"
#             action_input = clean_artifacts(raw)

#         # ── Handle action ─────────────────────────────────────
#         if "final_answer" in action:
#             is_bad = (
#                 not action_input
#                 or _is_cop_out_answer(action_input, question)
#                 or _is_echo_answer(action_input, question)
#             )

#             if is_bad:
#                 # Fall back to universal QA prompt
#                 print("[ReAct] Bad input — using direct QA prompt")
#                 emit_event(request_id, "agent_action", "⚡ Using direct prompt...")
#                 ranked   = reorder_by_question(question, context_chunks)
#                 ctx      = "\n\n---\n\n".join(
#                     clean_chunk_text(c) for c in ranked[:3]
#                 )
#                 answer     = call_llama(
#                     QA_PROMPT.format(context=ctx[:2500], question=question),
#                     temperature=0.0
#                 )
#                 llm_calls += 1
#                 model_used = "llama_react_direct"
#                 answer = clean_reasoning_answer(answer, question)
#             else:
#                 answer     = action_input
#                 # 🔥 Remove reasoning-style answers (generic pattern)
#                 answer = clean_reasoning_answer(answer, question)
#                 model_used = "llama_react"


#             grounding = compute_answer_grounding(answer, context_chunks)
#             print(f"[ReAct] ✅ Answer at step {step+1} | grounding={grounding:.1f}%")
#             emit_event(request_id, "agent_done", f"✅ Answer found at step {step+1}!")
#             return (
#                 answer, model_used, step + 1,
#                 retrieval_score, context_precision,
#                 grounding, llm_calls, context_chunks
#             )
#         elif "search" in action:
#             print("[ReAct] ⚠️ Skipping search → forcing final answer")

#             ranked = reorder_by_question(question, context_chunks)
#             ctx = "\n\n---\n\n".join(
#                 clean_chunk_text(c) for c in ranked[:3]
#             )

#             answer = call_llama(
#                 QA_PROMPT.format(context=ctx[:2500], question=question),
#                 temperature=0.0
#             )

#             llm_calls += 1
#             grounding = compute_answer_grounding(answer, context_chunks)

#             return (
#                 answer, "llama_react_forced", step + 1,
#                 retrieval_score, context_precision,
#                 grounding, llm_calls, context_chunks
#             )
                
#         # elif "search_more" in action:
#         #     print("[ReAct] ⚠️ Skipping search_more → forcing final answer")

#         #     # fallback to best available context immediately
#         #     ranked = reorder_by_question(question, context_chunks)
#         #     ctx = "\n\n---\n\n".join(
#         #         clean_chunk_text(c) for c in ranked[:3]
#         #     )

#         #     answer = call_llama(
#         #         QA_PROMPT.format(context=ctx[:2500], question=question),
#         #         temperature=0.0
#         #     )

#         #     llm_calls += 1
#         #     grounding = compute_answer_grounding(answer, context_chunks)

#         #     return (
#         #         answer, "llama_react_forced", step + 1,
#         #         retrieval_score, context_precision,
#         #         grounding, llm_calls, context_chunks
#         #     )

#         # elif "search_more" in action:
#         #     # query    = action_input.strip('"\'') if action_input else question
#         #     # 🔒 ALWAYS use original question (prevents drift)
#         #     query = question
            
#         #     if query == getattr(react_agent, '_last_query', None):
#         #         print(f"[ReAct] ⚠️ Duplicate search detected — forcing final answer")
#         #         break  # exit loop, fall through to final answer generation
            
#         #     react_agent._last_query = query
#         #     new_docs = multi_query_retrieve(
#         #         query, faiss_index, k=20,
#         #         all_chunks=all_chunks, query_type=query_type
#         #     )
#         #     new_docs, _, _ = rerank_docs(
#         #         query, new_docs, top_k=FACTUAL_TOP_K, apply_pruning=True
#         #     )
#         #     seen = set(context_chunks)
#         #     for d in new_docs:
#         #         if d.page_content not in seen:
#         #             context_chunks.append(d.page_content)
#         #             seen.add(d.page_content)
#         #     print(f"[ReAct] 🔍 Searched: '{query}' → {len(new_docs)} new chunks")
#         #     emit_event(request_id, "agent_search",
#         #                f"🔍 Searching: '{query[:60]}' → {len(new_docs)} chunks")
#         #     scratchpad += f"  Found {len(new_docs)} additional chunks\n"

#         else:
#             # Unrecognised action — return raw as answer
#             if len(raw) > 20:
#                 grounding = compute_answer_grounding(raw, context_chunks)
#                 return (
#                     raw, "llama_react_direct", step + 1,
#                     retrieval_score, context_precision,
#                     grounding, llm_calls, context_chunks  # ← fixed: was missing context_chunks
#                 )

#     # ── Max steps reached — final direct answer ───────────────
#     emit_event(request_id, "agent_action", "⚡ Synthesizing final answer...")
#     ranked  = reorder_by_question(question, context_chunks)
#     ctx     = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:3])

#     emit_event(request_id, "stream_start", "✍️ Generating final answer...")
#     final, ttft = call_llama_streaming(
#         QA_PROMPT.format(context=ctx[:2500], question=question),
#         request_id=request_id,
#         temperature=0.0
#     )
#     llm_calls += 1
#     grounding = compute_answer_grounding(final, context_chunks)

#     return (
#         final, "llama_react_final", MAX_STEPS,
#         retrieval_score, context_precision,
#         grounding, llm_calls, context_chunks
#     )


























#got 70 nd 70 gotto do some changes now 
#  import re
# from docmind_rag.config.settings import qa_pipeline, FACTUAL_TOP_K
# from docmind_rag.models.llm import call_llama, call_llama_streaming
# from docmind_rag.core.prompts import REACT_PROMPT, QA_PROMPT
# from docmind_rag.models.reranker import rerank_docs, protect_exact_matches
# from docmind_rag.services.retrieval import multi_query_retrieve
# from docmind_rag.events.events import emit_event
# from docmind_rag.utils.helpers import (
#     clean_artifacts, _is_cop_out_answer,
#     clean_chunk_text, validate_and_correct_span
# )
# from docmind_rag.utils.text import reorder_by_question, extract_numeric_answer
# from docmind_rag.utils.metrics import (
#     compute_answer_grounding,
#     compute_retrieval_score,
#     compute_context_precision
# )


# # ============================================================
# # HELPERS
# # ============================================================

# def should_validate(answer: str, question: str) -> bool:
#     answer = answer.strip()
#     if len(answer.split()) <= 12:
#         return True
#     if "\n" in answer or "," in answer:
#         return False
#     if question.lower().startswith(("what is", "who is", "name", "define")):
#         return True
#     return False


# def _is_echo_answer(answer: str, question: str) -> bool:
#     q_tokens = set(re.findall(r'\b\w+\b', question.lower()))
#     a_tokens = set(re.findall(r'\b\w+\b', answer.lower()))
#     return len(a_tokens - q_tokens) == 0


# # ============================================================
# # RoBERTa QA
# # ============================================================

# def roberta_qa(question: str, chunks: list):
#     best_answer      = ""
#     best_score       = 0.0
#     best_empty_score = 0.0

#     for chunk in chunks:
#         try:
#             result = qa_pipeline(question=question, context=chunk[:3000])
#             ans    = (result.get('answer') or "").strip()
#             score  = result.get('score', 0.0)
#             if ans:
#                 if score > best_score:
#                     best_score  = score
#                     best_answer = ans
#             else:
#                 if score > best_empty_score:
#                     best_empty_score = score
#         except Exception:
#             continue

#     if not best_answer and best_empty_score >= 0.4:
#         regex_ans = extract_numeric_answer(question, chunks)
#         if regex_ans:
#             print(f"[RoBERTa] Rescued by regex: '{regex_ans}'")
#             return regex_ans, round(best_empty_score, 4)

#     return best_answer.strip(), round(best_score, 4)


# # ============================================================
# # ReAct AGENT
# # ============================================================

# def react_agent(
#     question:   str,
#     faiss_index,
#     query_type: str,
#     all_chunks: list,
#     request_id: str = ""
# ) -> tuple:
#     """
#     Returns:
#         answer, model_used, steps, retrieval_score,
#         context_precision, grounding, llm_calls, context_chunks
#     """
#     MAX_STEPS  = 3
#     scratchpad = ""
#     model_used = "llama_react"
#     llm_calls  = 0

#     # ── Retrieval ─────────────────────────────────────────────
#     initial = multi_query_retrieve(
#         question, faiss_index, k=20,
#         all_chunks=all_chunks, query_type=query_type
#     )
#     apply_pruning = (query_type != "MULTIPART_QA")
#     initial, _, _ = rerank_docs(
#         question, initial,
#         top_k=FACTUAL_TOP_K,
#         apply_pruning=apply_pruning
#     )
#     initial = protect_exact_matches(
#         question, initial, all_chunks, top_k=FACTUAL_TOP_K
#     )
#     context_chunks    = [d.page_content for d in initial]
#     retrieval_score   = compute_retrieval_score(question, initial)
#     context_precision = compute_context_precision(question, initial)

#     print(f"[ReAct] Starting | {query_type} | {len(initial)} chunks | "
#           f"retrieval={retrieval_score:.1f}%")
#     emit_event(request_id, "agent_start",
#                f"🤖 Agent starting | {query_type} | {len(initial)} chunks retrieved")

#     # ── Step loop ─────────────────────────────────────────────
#     for step in range(MAX_STEPS):
#         ranked_chunks = reorder_by_question(question, context_chunks)
#         top_chunks    = ranked_chunks[:7]

#         # Debug
#         print("\n========== DEBUG: TOP CHUNKS ==========")
#         for i, chunk in enumerate(top_chunks):
#             print(f"\n--- Chunk {i+1} ---")
#             print(chunk[:300].replace("\n", " "))
#         print("======================================\n")

#         context = "\n\n---\n\n".join(clean_chunk_text(c) for c in top_chunks)

#         print("\n[DEBUG] ===== CLEANED CONTEXT PREVIEW =====")
#         print(context[:200])
#         print("=========================================\n")

#         raw = call_llama(
#             REACT_PROMPT.format(
#                 question=question,
#                 context=context[:2500],
#                 scratchpad=scratchpad if scratchpad else "None yet"
#             ),
#             temperature=0.0
#         )
#         llm_calls += 1
#         print(f"[ReAct] Step {step+1}: {raw[:120].strip()}")

#         # ── Parse LLM output ──────────────────────────────────
#         action       = ""
#         action_input = ""
#         lines        = raw.split("\n")

#         for i, line in enumerate(lines):
#             line = line.strip()
#             if line.startswith("Thought:"):
#                 pass  # logged only
#             elif line.startswith("Action:"):
#                 action_raw   = line.replace("Action:", "").strip()
#                 action_lower = action_raw.lower()
#                 if "final" in action_lower or "answer" in action_lower:
#                     action = "final_answer"
#                 elif "search" in action_lower or "more" in action_lower:
#                     action = "search_more"
#                 else:
#                     # LLM put answer directly in Action field — rescue it
#                     action       = "final_answer"
#                     action_input = action_raw
#                     print(f"[ReAct] ⚠️ Answer rescued from Action field: '{action_raw[:60]}'")
#             elif line.startswith("Input:"):
#                 if not action_input:
#                     action_input = line.replace("Input:", "").strip()
#                     for j in range(i + 1, len(lines)):
#                         next_line = lines[j].strip()
#                         if not next_line:
#                             continue
#                         if next_line.startswith(("Thought:", "Action:", "Input:")):
#                             break
#                         action_input += " " + next_line

#         if action_input:
#             action_input = clean_artifacts(action_input)

#         if not action:
#             action       = "final_answer"
#             action_input = clean_artifacts(raw)

#         # ── Handle action ─────────────────────────────────────
#         if "final_answer" in action:
#             is_bad = (
#                 not action_input
#                 or _is_cop_out_answer(action_input, question)
#                 or _is_echo_answer(action_input, question)
#             )

#             if is_bad:
#                 # Fall back to universal QA prompt
#                 print("[ReAct] Bad input — using direct QA prompt")
#                 emit_event(request_id, "agent_action", "⚡ Using direct prompt...")
#                 ranked   = reorder_by_question(question, context_chunks)
#                 ctx      = "\n\n---\n\n".join(
#                     clean_chunk_text(c) for c in ranked[:3]
#                 )
#                 answer     = call_llama(
#                     QA_PROMPT.format(context=ctx[:2500], question=question),
#                     temperature=0.0
#                 )
#                 llm_calls += 1
#                 model_used = "llama_react_direct"
#             else:
#                 answer     = action_input
#                 # 🔥 Remove reasoning-style answers (generic pattern)
#                 if answer.lower().startswith(("since", "because", "the question is")):
#                     print("[ReAct] ❌ Reasoning detected → forcing fallback")
#                     answer = ""
#                 model_used = "llama_react"

#             # Span validation for short answers only
#             if should_validate(answer, question):
#                 validated = validate_and_correct_span(
#                     answer, question, context_chunks
#                 )
#                 if validated and len(validated) >= len(answer) * 0.6:
#                     answer = validated

#             grounding = compute_answer_grounding(answer, context_chunks)
#             print(f"[ReAct] ✅ Answer at step {step+1} | grounding={grounding:.1f}%")
#             emit_event(request_id, "agent_done", f"✅ Answer found at step {step+1}!")
#             return (
#                 answer, model_used, step + 1,
#                 retrieval_score, context_precision,
#                 grounding, llm_calls, context_chunks
#             )
#         elif "search" in action:
#             print("[ReAct] ⚠️ Skipping search → forcing final answer")

#             ranked = reorder_by_question(question, context_chunks)
#             ctx = "\n\n---\n\n".join(
#                 clean_chunk_text(c) for c in ranked[:3]
#             )

#             answer = call_llama(
#                 QA_PROMPT.format(context=ctx[:2500], question=question),
#                 temperature=0.0
#             )

#             llm_calls += 1
#             grounding = compute_answer_grounding(answer, context_chunks)

#             return (
#                 answer, "llama_react_forced", step + 1,
#                 retrieval_score, context_precision,
#                 grounding, llm_calls, context_chunks
#             )
                
#         # elif "search_more" in action:
#         #     print("[ReAct] ⚠️ Skipping search_more → forcing final answer")

#         #     # fallback to best available context immediately
#         #     ranked = reorder_by_question(question, context_chunks)
#         #     ctx = "\n\n---\n\n".join(
#         #         clean_chunk_text(c) for c in ranked[:3]
#         #     )

#         #     answer = call_llama(
#         #         QA_PROMPT.format(context=ctx[:2500], question=question),
#         #         temperature=0.0
#         #     )

#         #     llm_calls += 1
#         #     grounding = compute_answer_grounding(answer, context_chunks)

#         #     return (
#         #         answer, "llama_react_forced", step + 1,
#         #         retrieval_score, context_precision,
#         #         grounding, llm_calls, context_chunks
#         #     )

#         # elif "search_more" in action:
#         #     # query    = action_input.strip('"\'') if action_input else question
#         #     # 🔒 ALWAYS use original question (prevents drift)
#         #     query = question
            
#         #     if query == getattr(react_agent, '_last_query', None):
#         #         print(f"[ReAct] ⚠️ Duplicate search detected — forcing final answer")
#         #         break  # exit loop, fall through to final answer generation
            
#         #     react_agent._last_query = query
#         #     new_docs = multi_query_retrieve(
#         #         query, faiss_index, k=20,
#         #         all_chunks=all_chunks, query_type=query_type
#         #     )
#         #     new_docs, _, _ = rerank_docs(
#         #         query, new_docs, top_k=FACTUAL_TOP_K, apply_pruning=True
#         #     )
#         #     seen = set(context_chunks)
#         #     for d in new_docs:
#         #         if d.page_content not in seen:
#         #             context_chunks.append(d.page_content)
#         #             seen.add(d.page_content)
#         #     print(f"[ReAct] 🔍 Searched: '{query}' → {len(new_docs)} new chunks")
#         #     emit_event(request_id, "agent_search",
#         #                f"🔍 Searching: '{query[:60]}' → {len(new_docs)} chunks")
#         #     scratchpad += f"  Found {len(new_docs)} additional chunks\n"

#         else:
#             # Unrecognised action — return raw as answer
#             if len(raw) > 20:
#                 grounding = compute_answer_grounding(raw, context_chunks)
#                 return (
#                     raw, "llama_react_direct", step + 1,
#                     retrieval_score, context_precision,
#                     grounding, llm_calls, context_chunks  # ← fixed: was missing context_chunks
#                 )

#     # ── Max steps reached — final direct answer ───────────────
#     emit_event(request_id, "agent_action", "⚡ Synthesizing final answer...")
#     ranked  = reorder_by_question(question, context_chunks)
#     ctx     = "\n\n---\n\n".join(clean_chunk_text(c) for c in ranked[:3])

#     emit_event(request_id, "stream_start", "✍️ Generating final answer...")
#     final, ttft = call_llama_streaming(
#         QA_PROMPT.format(context=ctx[:2500], question=question),
#         request_id=request_id,
#         temperature=0.0
#     )
#     llm_calls += 1
#     grounding = compute_answer_grounding(final, context_chunks)

#     return (
#         final, "llama_react_final", MAX_STEPS,
#         retrieval_score, context_precision,
#         grounding, llm_calls, context_chunks
#     )









# tried to fix that crop nd issue could not so making the  code fully generic nd workss uding  lap claude  for this first made changes in agent.py,text.py,prompts.py,qa pipelin
# import re
# from docmind_rag.config.settings import qa_pipeline, FACTUAL_TOP_K
# from docmind_rag.models.llm import call_llama, call_llama_streaming
# from docmind_rag.core.prompts import REACT_PROMPT, MULTIPART_PROMPT, REASONING_PROMPT, FACTUAL_EXTRACT_PROMPT
# from docmind_rag.models.reranker import rerank_docs, protect_exact_matches
# from docmind_rag.services.retrieval import multi_query_retrieve
# from docmind_rag.events.events import emit_event
# from docmind_rag.utils.helpers import clean_artifacts, _is_cop_out_answer,clean_chunk_text,validate_and_correct_span
# from docmind_rag.utils.text import reorder_by_question, extract_numeric_answer
# from docmind_rag.utils.metrics import compute_answer_grounding, compute_retrieval_score, compute_context_precision

# def should_validate(answer: str, question: str) -> bool:
#     answer = answer.strip()
#     question = question.lower()

#     # short answers → validate
#     if len(answer.split()) <= 12:
#         return True

#     # list / multi-line → skip validation
#     if "\n" in answer or "," in answer:
#         return False

#     # definition-type questions → validate
#     if question.startswith(("what is", "who is", "name", "define")):
#         return True

#     return False
# def _is_echo_answer(answer: str, question: str) -> bool:
#     """
#     Returns True if the answer adds zero new information beyond the question.
#     Catches cases like Q: "what is the name of lecture 3?" A: "Lecture 3"
#     """
#     q_tokens = set(re.findall(r'\b\w+\b', question.lower()))
#     a_tokens = set(re.findall(r'\b\w+\b', answer.lower()))
#     new_tokens = a_tokens - q_tokens
#     return len(new_tokens) == 0


# # ============================================================
# # RoBERTa QA
# # ============================================================
# def roberta_qa(question: str, chunks: list):
#     best_answer      = ""
#     best_score       = 0.0
#     best_empty_score = 0.0

#     for chunk in chunks:
#         try:
#             result = qa_pipeline(question=question, context=chunk[:3000])
#             ans    = (result.get('answer') or "").strip()
#             score  = result.get('score', 0.0)
#             if ans:
#                 if score > best_score:
#                     best_score  = score
#                     best_answer = ans
#             else:
#                 if score > best_empty_score:
#                     best_empty_score = score
#         except Exception:
#             continue

#     final_answer = best_answer.strip()
#     if not final_answer and best_empty_score >= 0.4:
#         regex_ans = extract_numeric_answer(question, chunks)
#         if regex_ans:
#             print(f"[RoBERTa] Empty span rescued by regex: '{regex_ans}'")
#             return regex_ans, round(best_empty_score, 4)

#     return final_answer, round(best_score, 4)


# # ============================================================
# # CORE: ReAct AGENT ENGINE
# # ============================================================
# def react_agent(question: str, faiss_index, query_type: str,
#                 all_chunks: list, request_id: str = "") -> tuple:
#     MAX_STEPS      = 3
#     scratchpad     = ""
#     context_chunks = []
#     model_used     = "llama_react"
#     llm_calls      = 0
#     collected_answers = []

#     initial = multi_query_retrieve(question, faiss_index, k=20,
#                                    all_chunks=all_chunks, query_type=query_type)
#     apply_pruning = (query_type != "MULTIPART_QA")
#     initial, reranker_top_score, _ = rerank_docs(question, initial,
#                                                   top_k=FACTUAL_TOP_K,
#                                                   apply_pruning=apply_pruning)
#     initial = protect_exact_matches(question, initial, all_chunks,
#                                     top_k=FACTUAL_TOP_K)

#     # Pass full chunks — no sentence windowing.
#     # _focus_context_on_query was cutting out correct sentences
#     # (e.g. the Hinton attribution sentence) causing wrong answers.
#     # The LLM prompt rules handle attribution correctly when given full context.
#     context_chunks = [d.page_content for d in initial]

#     retrieval_score   = compute_retrieval_score(question, initial)
#     context_precision = compute_context_precision(question, initial)

#     print(f"[ReAct] Starting | {query_type} | {len(initial)} chunks | "
#           f"retrieval={retrieval_score:.1f}%")
#     emit_event(request_id, "agent_start",
#                f"🤖 Agent starting | {query_type} | {len(initial)} chunks retrieved")

#     for step in range(MAX_STEPS):
#         # 🔥 prioritize most relevant chunks BEFORE truncation
#         ranked_chunks = reorder_by_question(question, context_chunks)

#         # keep top 3 instead of all (prevents cutting important lines)
#         top_chunks = ranked_chunks[:7]
#         print("\n========== DEBUG: TOP CHUNKS ==========")
#         for i, chunk in enumerate(top_chunks):
#             preview = chunk[:300].replace("\n", " ")
#             print(f"\n--- Chunk {i+1} ---")
#             print(preview)
#         print("======================================\n")

#         context = "\n\n---\n\n".join(clean_chunk_text(c) for c in top_chunks)
#         # ✅ DEBUG START
#         print("\n[DEBUG] ===== CLEANED CONTEXT PREVIEW =====")
#         print(context[:200])
#         print("=========================================\n")

#         print(f"[DEBUG] Cleaner source: {clean_chunk_text.__module__}")
#         # ✅ DEBUG END
#         raw = call_llama(REACT_PROMPT.format(
#             question=question, context=context[:2500],
#             scratchpad=scratchpad if scratchpad else "None yet"),
#             temperature=0.0)
#         llm_calls += 1
#         print(f"[ReAct] Step {step+1}: {raw[:120].strip()}")

#         action       = ""
#         action_input = ""
#         thought      = raw[:100]
#         lines        = raw.split("\n")
#         for i, line in enumerate(lines):
#             line = line.strip()
#             if line.startswith("Thought:"):
#                 thought = line.replace("Thought:", "").strip()
#             elif line.startswith("Action:"):
#                 action_raw = line.replace("Action:", "").strip()
#                 action_lower = action_raw.lower()
#                 if "final" in action_lower or "answer" in action_lower:
#                     action = "final_answer"
#                 elif "search" in action_lower or "more" in action_lower:
#                     action = "search_more"
#                 else:
#                     # LLM put the answer directly in the Action field
#                     # Rescue it instead of discarding
#                     action       = "final_answer"
#                     action_input = action_raw  # ← rescue the answer
#                     print(f"[ReAct] ⚠️ Answer rescued from Action field: '{action_raw[:60]}'")

#             elif line.startswith("Input:"):
#                 if not action_input:
#                     action_input = line.replace("Input:", "").strip()
#                     for j in range(i+1, len(lines)):
#                         next_line = lines[j].strip()
#                         if not next_line:
#                             continue
#                         if next_line.startswith(("Thought:", "Action:", "Input:")):
#                             break
#                         action_input += " " + next_line

#         if action_input:
#             action_input = clean_artifacts(action_input)

#         if not action:
#             action       = "final_answer"
#             action_input = clean_artifacts(raw)

#         if "final_answer" in action or action == "bad_format":
#             is_bad_input = (
#                 action == "bad_format" or
#                 not action_input or
#                 _is_cop_out_answer(action_input, question) or
#                 _is_echo_answer(action_input, question)
#             )

#             if is_bad_input:
#                 print(f"[ReAct] Bad input detected — using direct prompt")
#                 emit_event(request_id, "agent_action", "⚡ Using direct prompt...")

#                 ranked_chunks = reorder_by_question(question, context_chunks)
#                 top_chunks    = ranked_chunks[:3]
#                 context       = "\n\n---\n\n".join(clean_chunk_text(c) for c in top_chunks)

#                 if query_type == "MULTIPART_QA":
#                     answer = call_llama(MULTIPART_PROMPT.format(
#                         context=context[:2500], question=question),
#                         temperature=0.0)
#                 elif query_type == "REASONING_QA":
#                     answer = call_llama(REASONING_PROMPT.format(
#                         context=context[:2500], question=question),
#                         temperature=0.0)
#                 else:
#                     answer = call_llama(FACTUAL_EXTRACT_PROMPT.format(
#                         context=context[:2500], question=question),
#                         temperature=0.0)

#                 llm_calls += 1
#                 model_used = "llama_react_direct"

#             # else:
#             #     answer     = action_input
#             #     model_used = "llama_react"
#             else:
#                 original_answer = action_input   # ⭐ ADD THIS
#                 answer          = original_answer
#                 model_used      = "llama_react"

#             # Post-generation span validation — FACTUAL_QA only
#             # No else block here — just the validation call
#             # if query_type in ("FACTUAL_QA", "REASONING_QA"):
#             #     answer = validate_and_correct_span(
#             #         answer, question, context_chunks
#             #     )
#             # if query_type in ("FACTUAL_QA", "REASONING_QA"):
#             #     validated = validate_and_correct_span(
#             #         answer, question, context_chunks
#             #     )
#             # ✅ Adaptive validation (SAFE + GENERIC)
#             # ✅ Adaptive validation (SAFE + GENERIC)
#             if query_type in ("FACTUAL_QA", "REASONING_QA"):

#                 validated = None   # ✅ ALWAYS initialize

#                 if should_validate(answer, question):
#                     validated = validate_and_correct_span(
#                         answer, question, context_chunks
#                     )

#                 # ✅ Only replace if safe
#                 if validated:
#                     if len(validated) >= len(answer) * 0.6:
#                         answer = validated
#             grounding = compute_answer_grounding(answer, context_chunks)
#             print(f"[ReAct] ✅ Answer at step {step+1} | grounding={grounding:.1f}%")
#             emit_event(request_id, "agent_done", f"✅ Answer found at step {step+1}!")
#             # return answer, model_used, step+1, retrieval_score, context_precision, grounding, llm_calls, context_chunks
#             if not is_bad_input:
#                 collected_answers.append(answer)
#                 # collected_answers.append(original_answer)
#                 scratchpad += f"  Found answer: {answer}\n"

#             # stop after few answers (safe)
#             if len(collected_answers) >= 3:
#                 break

#             continue

#         elif "search_more" in action:
#             query    = action_input.strip('"\'') if action_input else question
#             new_docs = multi_query_retrieve(query, faiss_index, k=20,
#                                              all_chunks=all_chunks, query_type=query_type)
#             new_docs, _, _ = rerank_docs(query, new_docs, top_k=FACTUAL_TOP_K,
#                                           apply_pruning=True)
#             new_texts = [d.page_content for d in new_docs]
#             seen      = set(context_chunks)
#             for t in new_texts:
#                 if t not in seen:
#                     context_chunks.append(t)
#                     seen.add(t)
#             print(f"[ReAct] 🔍 Searched: '{query}' → {len(new_texts)} chunks | "
#                   f"total ctx: {len(context_chunks)}")
#             emit_event(request_id, "agent_search",
#                        f"🔍 Searching: '{query[:60]}' → {len(new_texts)} chunks")
#             scratchpad += f"  Found {len(new_texts)} additional chunks\n"
#         else:
#             if len(raw) > 20:
#                 grounding = compute_answer_grounding(raw, context_chunks)
#                 return raw, "llama_react_direct", step+1, retrieval_score, context_precision, grounding, llm_calls, context_chunks
    
#     # ============================================================
#     # 🔥 GENERIC FINAL ANSWER SELECTION (NO HARDCODING)
#     # ============================================================

#     if collected_answers:
#         unique_answers = list(dict.fromkeys(collected_answers))

#         # --------------------------------------------------------
#         # SIGNAL 1: number of distinct answers
#         # --------------------------------------------------------
#         num_answers = len(unique_answers)

#         # --------------------------------------------------------
#         # SIGNAL 2: average answer length
#         # --------------------------------------------------------
#         avg_len = sum(len(a.split()) for a in unique_answers) / max(1, num_answers)

#         # --------------------------------------------------------
#         # DECISION LOGIC (DATA-DRIVEN)
#         # --------------------------------------------------------

#         # Case 1: Single strong short answer → use directly
#         # if num_answers == 1 and avg_len < 12:
#         if num_answers == 1:
#             final_answer = unique_answers[0]

#         # Case 2: Multiple short answers → combine (list-like)
#         # elif num_answers > 1 and avg_len < 12:
#         #     final_answer = ", ".join(unique_answers[:5])
#         elif num_answers > 1 and avg_len < 12:
#             final_answer = unique_answers[0]

#         # Case 3: Long / complex answers → synthesize
#         else:
#             context_text = "\n\n".join(context_chunks[:5])

#     #         prompt = f"""
#     # Answer the question using the context.

#     # Rules:
#     # - Combine information if needed
#     # - Be concise
#     # - Do NOT copy long sentences
#     # - Focus only on key information
#             prompt = f"""
# Answer the question using ONLY the provided context.

# Return the MINIMAL COMPLETE answer:
# - Include everything required to fully answer the question
# - Do not include extra or unrelated information

#     Question:
#     {question}

#     Context:
#     {context_text}

#     Answer:
#     """
#             try:
#                 final_answer = call_llama(prompt, temperature=0.0)
#                 llm_calls += 1
#             except:
#                 final_answer = unique_answers[0]

#         grounding = compute_answer_grounding(final_answer, context_chunks)

#         return final_answer, "llama_react_multi", MAX_STEPS, retrieval_score, context_precision, grounding, llm_calls, context_chunks
#     emit_event(request_id, "agent_action", "⚡ Synthesizing final answer...")
#     # 🔥 prioritize most relevant chunks BEFORE truncation
#     ranked_chunks = reorder_by_question(question, context_chunks)

#     # keep top 3 instead of all (prevents cutting important lines)
#     top_chunks = ranked_chunks[:3]

#     context = "\n\n---\n\n".join(clean_chunk_text(c) for c in top_chunks)
#     emit_event(request_id, "stream_start", "✍️ Generating final answer...")
#     if query_type == "MULTIPART_QA":
#         final = call_llama_streaming(MULTIPART_PROMPT.format(
#             context=context[:2500], question=question), request_id, temperature=0.0)
#     elif query_type == "REASONING_QA":
#         final = call_llama_streaming(REASONING_PROMPT.format(
#             context=context[:2500], question=question), request_id, temperature=0.0)
#     else:
#         final = call_llama_streaming(FACTUAL_EXTRACT_PROMPT.format(
#             context=context[:2500], question=question), request_id, temperature=0.0)
#     llm_calls += 1
#     grounding = compute_answer_grounding(final, context_chunks)
#     return final, "llama_react_final", MAX_STEPS, retrieval_score, context_precision, grounding, llm_calls







































# works now chjanging some output like  (CNNs) become cnn like this 
# import re
# from docmind_rag.config.settings import qa_pipeline, FACTUAL_TOP_K
# from docmind_rag.models.llm import call_llama, call_llama_streaming
# from docmind_rag.core.prompts import REACT_PROMPT, MULTIPART_PROMPT, REASONING_PROMPT, FACTUAL_EXTRACT_PROMPT
# from docmind_rag.models.reranker import rerank_docs, protect_exact_matches
# from docmind_rag.services.retrieval import multi_query_retrieve
# from docmind_rag.events.events import emit_event
# from docmind_rag.utils.helpers import clean_artifacts, _is_cop_out_answer,clean_chunk_text,validate_and_correct_span
# from docmind_rag.utils.text import reorder_by_question, extract_numeric_answer
# from docmind_rag.utils.metrics import compute_answer_grounding, compute_retrieval_score, compute_context_precision


# def _is_echo_answer(answer: str, question: str) -> bool:
#     """
#     Returns True if the answer adds zero new information beyond the question.
#     Catches cases like Q: "what is the name of lecture 3?" A: "Lecture 3"
#     """
#     q_tokens = set(re.findall(r'\b\w+\b', question.lower()))
#     a_tokens = set(re.findall(r'\b\w+\b', answer.lower()))
#     new_tokens = a_tokens - q_tokens
#     return len(new_tokens) == 0


# # ============================================================
# # RoBERTa QA
# # ============================================================
# def roberta_qa(question: str, chunks: list):
#     best_answer      = ""
#     best_score       = 0.0
#     best_empty_score = 0.0

#     for chunk in chunks:
#         try:
#             result = qa_pipeline(question=question, context=chunk[:3000])
#             ans    = (result.get('answer') or "").strip()
#             score  = result.get('score', 0.0)
#             if ans:
#                 if score > best_score:
#                     best_score  = score
#                     best_answer = ans
#             else:
#                 if score > best_empty_score:
#                     best_empty_score = score
#         except Exception:
#             continue

#     final_answer = best_answer.strip()
#     if not final_answer and best_empty_score >= 0.4:
#         regex_ans = extract_numeric_answer(question, chunks)
#         if regex_ans:
#             print(f"[RoBERTa] Empty span rescued by regex: '{regex_ans}'")
#             return regex_ans, round(best_empty_score, 4)

#     return final_answer, round(best_score, 4)


# # ============================================================
# # CORE: ReAct AGENT ENGINE
# # ============================================================
# def react_agent(question: str, faiss_index, query_type: str,
#                 all_chunks: list, request_id: str = "") -> tuple:
#     MAX_STEPS      = 3
#     scratchpad     = ""
#     context_chunks = []
#     model_used     = "llama_react"
#     llm_calls      = 0
#     collected_answers = []

#     initial = multi_query_retrieve(question, faiss_index, k=20,
#                                    all_chunks=all_chunks, query_type=query_type)
#     apply_pruning = (query_type != "MULTIPART_QA")
#     initial, reranker_top_score, _ = rerank_docs(question, initial,
#                                                   top_k=FACTUAL_TOP_K,
#                                                   apply_pruning=apply_pruning)
#     initial = protect_exact_matches(question, initial, all_chunks,
#                                     top_k=FACTUAL_TOP_K)

#     # Pass full chunks — no sentence windowing.
#     # _focus_context_on_query was cutting out correct sentences
#     # (e.g. the Hinton attribution sentence) causing wrong answers.
#     # The LLM prompt rules handle attribution correctly when given full context.
#     context_chunks = [d.page_content for d in initial]

#     retrieval_score   = compute_retrieval_score(question, initial)
#     context_precision = compute_context_precision(question, initial)

#     print(f"[ReAct] Starting | {query_type} | {len(initial)} chunks | "
#           f"retrieval={retrieval_score:.1f}%")
#     emit_event(request_id, "agent_start",
#                f"🤖 Agent starting | {query_type} | {len(initial)} chunks retrieved")

#     for step in range(MAX_STEPS):
#         # 🔥 prioritize most relevant chunks BEFORE truncation
#         ranked_chunks = reorder_by_question(question, context_chunks)

#         # keep top 3 instead of all (prevents cutting important lines)
#         top_chunks = ranked_chunks[:7]
#         print("\n========== DEBUG: TOP CHUNKS ==========")
#         for i, chunk in enumerate(top_chunks):
#             preview = chunk[:300].replace("\n", " ")
#             print(f"\n--- Chunk {i+1} ---")
#             print(preview)
#         print("======================================\n")

#         context = "\n\n---\n\n".join(clean_chunk_text(c) for c in top_chunks)
#         # ✅ DEBUG START
#         print("\n[DEBUG] ===== CLEANED CONTEXT PREVIEW =====")
#         print(context[:200])
#         print("=========================================\n")

#         print(f"[DEBUG] Cleaner source: {clean_chunk_text.__module__}")
#         # ✅ DEBUG END
#         raw = call_llama(REACT_PROMPT.format(
#             question=question, context=context[:2500],
#             scratchpad=scratchpad if scratchpad else "None yet"),
#             temperature=0.0)
#         llm_calls += 1
#         print(f"[ReAct] Step {step+1}: {raw[:120].strip()}")

#         action       = ""
#         action_input = ""
#         thought      = raw[:100]
#         lines        = raw.split("\n")
#         for i, line in enumerate(lines):
#             line = line.strip()
#             if line.startswith("Thought:"):
#                 thought = line.replace("Thought:", "").strip()
#             elif line.startswith("Action:"):
#                 action_raw = line.replace("Action:", "").strip()
#                 action_lower = action_raw.lower()
#                 if "final" in action_lower or "answer" in action_lower:
#                     action = "final_answer"
#                 elif "search" in action_lower or "more" in action_lower:
#                     action = "search_more"
#                 else:
#                     # LLM put the answer directly in the Action field
#                     # Rescue it instead of discarding
#                     action       = "final_answer"
#                     action_input = action_raw  # ← rescue the answer
#                     print(f"[ReAct] ⚠️ Answer rescued from Action field: '{action_raw[:60]}'")

#             elif line.startswith("Input:"):
#                 if not action_input:
#                     action_input = line.replace("Input:", "").strip()
#                     for j in range(i+1, len(lines)):
#                         next_line = lines[j].strip()
#                         if not next_line:
#                             continue
#                         if next_line.startswith(("Thought:", "Action:", "Input:")):
#                             break
#                         action_input += " " + next_line

#         if action_input:
#             action_input = clean_artifacts(action_input)

#         if not action:
#             action       = "final_answer"
#             action_input = clean_artifacts(raw)

#         if "final_answer" in action or action == "bad_format":
#             is_bad_input = (
#                 action == "bad_format" or
#                 not action_input or
#                 _is_cop_out_answer(action_input, question) or
#                 _is_echo_answer(action_input, question)
#             )

#             if is_bad_input:
#                 print(f"[ReAct] Bad input detected — using direct prompt")
#                 emit_event(request_id, "agent_action", "⚡ Using direct prompt...")

#                 ranked_chunks = reorder_by_question(question, context_chunks)
#                 top_chunks    = ranked_chunks[:3]
#                 context       = "\n\n---\n\n".join(clean_chunk_text(c) for c in top_chunks)

#                 if query_type == "MULTIPART_QA":
#                     answer = call_llama(MULTIPART_PROMPT.format(
#                         context=context[:2500], question=question),
#                         temperature=0.0)
#                 elif query_type == "REASONING_QA":
#                     answer = call_llama(REASONING_PROMPT.format(
#                         context=context[:2500], question=question),
#                         temperature=0.0)
#                 else:
#                     answer = call_llama(FACTUAL_EXTRACT_PROMPT.format(
#                         context=context[:2500], question=question),
#                         temperature=0.0)

#                 llm_calls += 1
#                 model_used = "llama_react_direct"

#             else:
#                 answer     = action_input
#                 model_used = "llama_react"

#             # Post-generation span validation — FACTUAL_QA only
#             # No else block here — just the validation call
#             if query_type in ("FACTUAL_QA", "REASONING_QA"):
#                 answer = validate_and_correct_span(
#                     answer, question, context_chunks
#                 )

#             grounding = compute_answer_grounding(answer, context_chunks)
#             print(f"[ReAct] ✅ Answer at step {step+1} | grounding={grounding:.1f}%")
#             emit_event(request_id, "agent_done", f"✅ Answer found at step {step+1}!")
#             # return answer, model_used, step+1, retrieval_score, context_precision, grounding, llm_calls, context_chunks
#             if not is_bad_input:
#                 collected_answers.append(answer)
#                 scratchpad += f"  Found answer: {answer}\n"

#             # stop after few answers (safe)
#             if len(collected_answers) >= 3:
#                 break

#             continue

#         elif "search_more" in action:
#             query    = action_input.strip('"\'') if action_input else question
#             new_docs = multi_query_retrieve(query, faiss_index, k=20,
#                                              all_chunks=all_chunks, query_type=query_type)
#             new_docs, _, _ = rerank_docs(query, new_docs, top_k=FACTUAL_TOP_K,
#                                           apply_pruning=True)
#             new_texts = [d.page_content for d in new_docs]
#             seen      = set(context_chunks)
#             for t in new_texts:
#                 if t not in seen:
#                     context_chunks.append(t)
#                     seen.add(t)
#             print(f"[ReAct] 🔍 Searched: '{query}' → {len(new_texts)} chunks | "
#                   f"total ctx: {len(context_chunks)}")
#             emit_event(request_id, "agent_search",
#                        f"🔍 Searching: '{query[:60]}' → {len(new_texts)} chunks")
#             scratchpad += f"  Found {len(new_texts)} additional chunks\n"
#         else:
#             if len(raw) > 20:
#                 grounding = compute_answer_grounding(raw, context_chunks)
#                 return raw, "llama_react_direct", step+1, retrieval_score, context_precision, grounding, llm_calls, context_chunks
    
#     # ============================================================
#     # 🔥 GENERIC FINAL ANSWER SELECTION (NO HARDCODING)
#     # ============================================================

#     if collected_answers:
#         unique_answers = list(dict.fromkeys(collected_answers))

#         # --------------------------------------------------------
#         # SIGNAL 1: number of distinct answers
#         # --------------------------------------------------------
#         num_answers = len(unique_answers)

#         # --------------------------------------------------------
#         # SIGNAL 2: average answer length
#         # --------------------------------------------------------
#         avg_len = sum(len(a.split()) for a in unique_answers) / max(1, num_answers)

#         # --------------------------------------------------------
#         # DECISION LOGIC (DATA-DRIVEN)
#         # --------------------------------------------------------

#         # Case 1: Single strong short answer → use directly
#         if num_answers == 1 and avg_len < 12:
#             final_answer = unique_answers[0]

#         # Case 2: Multiple short answers → combine (list-like)
#         elif num_answers > 1 and avg_len < 12:
#             final_answer = ", ".join(unique_answers[:5])

#         # Case 3: Long / complex answers → synthesize
#         else:
#             context_text = "\n\n".join(context_chunks[:5])

#     #         prompt = f"""
#     # Answer the question using the context.

#     # Rules:
#     # - Combine information if needed
#     # - Be concise
#     # - Do NOT copy long sentences
#     # - Focus only on key information
#             prompt = f"""
# Answer the question using ONLY the provided context.

# Return the MINIMAL COMPLETE answer:
# - Include everything required to fully answer the question
# - Do not include extra or unrelated information

#     Question:
#     {question}

#     Context:
#     {context_text}

#     Answer:
#     """
#             try:
#                 final_answer = call_llama(prompt, temperature=0.0)
#                 llm_calls += 1
#             except:
#                 final_answer = unique_answers[0]

#         grounding = compute_answer_grounding(final_answer, context_chunks)

#         return final_answer, "llama_react_multi", MAX_STEPS, retrieval_score, context_precision, grounding, llm_calls, context_chunks
#     emit_event(request_id, "agent_action", "⚡ Synthesizing final answer...")
#     # 🔥 prioritize most relevant chunks BEFORE truncation
#     ranked_chunks = reorder_by_question(question, context_chunks)

#     # keep top 3 instead of all (prevents cutting important lines)
#     top_chunks = ranked_chunks[:3]

#     context = "\n\n---\n\n".join(clean_chunk_text(c) for c in top_chunks)
#     emit_event(request_id, "stream_start", "✍️ Generating final answer...")
#     if query_type == "MULTIPART_QA":
#         final = call_llama_streaming(MULTIPART_PROMPT.format(
#             context=context[:2500], question=question), request_id, temperature=0.0)
#     elif query_type == "REASONING_QA":
#         final = call_llama_streaming(REASONING_PROMPT.format(
#             context=context[:2500], question=question), request_id, temperature=0.0)
#     else:
#         final = call_llama_streaming(FACTUAL_EXTRACT_PROMPT.format(
#             context=context[:2500], question=question), request_id, temperature=0.0)
#     llm_calls += 1
#     grounding = compute_answer_grounding(final, context_chunks)
#     return final, "llama_react_final", MAX_STEPS, retrieval_score, context_precision, grounding, llm_calls



































# got 100 fro ai but  57 for crop 
# import re
# from docmind_rag.config.settings import qa_pipeline, FACTUAL_TOP_K
# from docmind_rag.models.llm import call_llama, call_llama_streaming
# from docmind_rag.core.prompts import REACT_PROMPT, MULTIPART_PROMPT, REASONING_PROMPT, FACTUAL_EXTRACT_PROMPT
# from docmind_rag.models.reranker import rerank_docs, protect_exact_matches
# from docmind_rag.services.retrieval import multi_query_retrieve
# from docmind_rag.events.events import emit_event
# from docmind_rag.utils.helpers import clean_artifacts, _is_cop_out_answer,clean_chunk_text,validate_and_correct_span
# from docmind_rag.utils.text import reorder_by_question, extract_numeric_answer
# from docmind_rag.utils.metrics import compute_answer_grounding, compute_retrieval_score, compute_context_precision


# def _is_echo_answer(answer: str, question: str) -> bool:
#     """
#     Returns True if the answer adds zero new information beyond the question.
#     Catches cases like Q: "what is the name of lecture 3?" A: "Lecture 3"
#     """
#     q_tokens = set(re.findall(r'\b\w+\b', question.lower()))
#     a_tokens = set(re.findall(r'\b\w+\b', answer.lower()))
#     new_tokens = a_tokens - q_tokens
#     return len(new_tokens) == 0


# # ============================================================
# # RoBERTa QA
# # ============================================================
# def roberta_qa(question: str, chunks: list):
#     best_answer      = ""
#     best_score       = 0.0
#     best_empty_score = 0.0

#     for chunk in chunks:
#         try:
#             result = qa_pipeline(question=question, context=chunk[:3000])
#             ans    = (result.get('answer') or "").strip()
#             score  = result.get('score', 0.0)
#             if ans:
#                 if score > best_score:
#                     best_score  = score
#                     best_answer = ans
#             else:
#                 if score > best_empty_score:
#                     best_empty_score = score
#         except Exception:
#             continue

#     final_answer = best_answer.strip()
#     if not final_answer and best_empty_score >= 0.4:
#         regex_ans = extract_numeric_answer(question, chunks)
#         if regex_ans:
#             print(f"[RoBERTa] Empty span rescued by regex: '{regex_ans}'")
#             return regex_ans, round(best_empty_score, 4)

#     return final_answer, round(best_score, 4)


# # ============================================================
# # CORE: ReAct AGENT ENGINE
# # ============================================================
# def react_agent(question: str, faiss_index, query_type: str,
#                 all_chunks: list, request_id: str = "") -> tuple:
#     MAX_STEPS      = 3
#     scratchpad     = ""
#     context_chunks = []
#     model_used     = "llama_react"
#     llm_calls      = 0

#     initial = multi_query_retrieve(question, faiss_index, k=20,
#                                    all_chunks=all_chunks, query_type=query_type)
#     apply_pruning = (query_type != "MULTIPART_QA")
#     initial, reranker_top_score, _ = rerank_docs(question, initial,
#                                                   top_k=FACTUAL_TOP_K,
#                                                   apply_pruning=apply_pruning)
#     initial = protect_exact_matches(question, initial, all_chunks,
#                                     top_k=FACTUAL_TOP_K)

#     # Pass full chunks — no sentence windowing.
#     # _focus_context_on_query was cutting out correct sentences
#     # (e.g. the Hinton attribution sentence) causing wrong answers.
#     # The LLM prompt rules handle attribution correctly when given full context.
#     context_chunks = [d.page_content for d in initial]

#     retrieval_score   = compute_retrieval_score(question, initial)
#     context_precision = compute_context_precision(question, initial)

#     print(f"[ReAct] Starting | {query_type} | {len(initial)} chunks | "
#           f"retrieval={retrieval_score:.1f}%")
#     emit_event(request_id, "agent_start",
#                f"🤖 Agent starting | {query_type} | {len(initial)} chunks retrieved")

#     for step in range(MAX_STEPS):
#         # 🔥 prioritize most relevant chunks BEFORE truncation
#         ranked_chunks = reorder_by_question(question, context_chunks)

#         # keep top 3 instead of all (prevents cutting important lines)
#         top_chunks = ranked_chunks[:7]
#         print("\n========== DEBUG: TOP CHUNKS ==========")
#         for i, chunk in enumerate(top_chunks):
#             preview = chunk[:300].replace("\n", " ")
#             print(f"\n--- Chunk {i+1} ---")
#             print(preview)
#         print("======================================\n")

#         context = "\n\n---\n\n".join(clean_chunk_text(c) for c in top_chunks)
#         # ✅ DEBUG START
#         print("\n[DEBUG] ===== CLEANED CONTEXT PREVIEW =====")
#         print(context[:200])
#         print("=========================================\n")

#         print(f"[DEBUG] Cleaner source: {clean_chunk_text.__module__}")
#         # ✅ DEBUG END
#         raw = call_llama(REACT_PROMPT.format(
#             question=question, context=context[:2500],
#             scratchpad=scratchpad if scratchpad else "None yet"),
#             temperature=0.0)
#         llm_calls += 1
#         print(f"[ReAct] Step {step+1}: {raw[:120].strip()}")

#         action       = ""
#         action_input = ""
#         thought      = raw[:100]
#         lines        = raw.split("\n")
#         for i, line in enumerate(lines):
#             line = line.strip()
#             if line.startswith("Thought:"):
#                 thought = line.replace("Thought:", "").strip()
#             elif line.startswith("Action:"):
#                 action_raw = line.replace("Action:", "").strip()
#                 action_lower = action_raw.lower()
#                 if "final" in action_lower or "answer" in action_lower:
#                     action = "final_answer"
#                 elif "search" in action_lower or "more" in action_lower:
#                     action = "search_more"
#                 else:
#                     # LLM put the answer directly in the Action field
#                     # Rescue it instead of discarding
#                     action       = "final_answer"
#                     action_input = action_raw  # ← rescue the answer
#                     print(f"[ReAct] ⚠️ Answer rescued from Action field: '{action_raw[:60]}'")

#             elif line.startswith("Input:"):
#                 if not action_input:
#                     action_input = line.replace("Input:", "").strip()
#                     for j in range(i+1, len(lines)):
#                         next_line = lines[j].strip()
#                         if not next_line:
#                             continue
#                         if next_line.startswith(("Thought:", "Action:", "Input:")):
#                             break
#                         action_input += " " + next_line

#         if action_input:
#             action_input = clean_artifacts(action_input)

#         if not action:
#             action       = "final_answer"
#             action_input = clean_artifacts(raw)

#         if "final_answer" in action or action == "bad_format":
#             is_bad_input = (
#                 action == "bad_format" or
#                 not action_input or
#                 _is_cop_out_answer(action_input, question) or
#                 _is_echo_answer(action_input, question)
#             )

#             if is_bad_input:
#                 print(f"[ReAct] Bad input detected — using direct prompt")
#                 emit_event(request_id, "agent_action", "⚡ Using direct prompt...")

#                 ranked_chunks = reorder_by_question(question, context_chunks)
#                 top_chunks    = ranked_chunks[:3]
#                 context       = "\n\n---\n\n".join(clean_chunk_text(c) for c in top_chunks)

#                 if query_type == "MULTIPART_QA":
#                     answer = call_llama(MULTIPART_PROMPT.format(
#                         context=context[:2500], question=question),
#                         temperature=0.0)
#                 elif query_type == "REASONING_QA":
#                     answer = call_llama(REASONING_PROMPT.format(
#                         context=context[:2500], question=question),
#                         temperature=0.0)
#                 else:
#                     answer = call_llama(FACTUAL_EXTRACT_PROMPT.format(
#                         context=context[:2500], question=question),
#                         temperature=0.0)

#                 llm_calls += 1
#                 model_used = "llama_react_direct"

#             else:
#                 answer     = action_input
#                 model_used = "llama_react"

#             # Post-generation span validation — FACTUAL_QA only
#             # No else block here — just the validation call
#             if query_type in ("FACTUAL_QA", "REASONING_QA"):
#                 answer = validate_and_correct_span(
#                     answer, question, context_chunks
#                 )

#             grounding = compute_answer_grounding(answer, context_chunks)
#             print(f"[ReAct] ✅ Answer at step {step+1} | grounding={grounding:.1f}%")
#             emit_event(request_id, "agent_done", f"✅ Answer found at step {step+1}!")
#             return answer, model_used, step+1, retrieval_score, context_precision, grounding, llm_calls, context_chunks

#         elif "search_more" in action:
#             query    = action_input.strip('"\'') if action_input else question
#             new_docs = multi_query_retrieve(query, faiss_index, k=20,
#                                              all_chunks=all_chunks, query_type=query_type)
#             new_docs, _, _ = rerank_docs(query, new_docs, top_k=FACTUAL_TOP_K,
#                                           apply_pruning=True)
#             new_texts = [d.page_content for d in new_docs]
#             seen      = set(context_chunks)
#             for t in new_texts:
#                 if t not in seen:
#                     context_chunks.append(t)
#                     seen.add(t)
#             print(f"[ReAct] 🔍 Searched: '{query}' → {len(new_texts)} chunks | "
#                   f"total ctx: {len(context_chunks)}")
#             emit_event(request_id, "agent_search",
#                        f"🔍 Searching: '{query[:60]}' → {len(new_texts)} chunks")
#             scratchpad += f"  Found {len(new_texts)} additional chunks\n"
#         else:
#             if len(raw) > 20:
#                 grounding = compute_answer_grounding(raw, context_chunks)
#                 return raw, "llama_react_direct", step+1, retrieval_score, context_precision, grounding, llm_calls, context_chunks

#     emit_event(request_id, "agent_action", "⚡ Synthesizing final answer...")
#     # 🔥 prioritize most relevant chunks BEFORE truncation
#     ranked_chunks = reorder_by_question(question, context_chunks)

#     # keep top 3 instead of all (prevents cutting important lines)
#     top_chunks = ranked_chunks[:3]

#     context = "\n\n---\n\n".join(clean_chunk_text(c) for c in top_chunks)
#     emit_event(request_id, "stream_start", "✍️ Generating final answer...")
#     if query_type == "MULTIPART_QA":
#         final = call_llama_streaming(MULTIPART_PROMPT.format(
#             context=context[:2500], question=question), request_id, temperature=0.0)
#     elif query_type == "REASONING_QA":
#         final = call_llama_streaming(REASONING_PROMPT.format(
#             context=context[:2500], question=question), request_id, temperature=0.0)
#     else:
#         final = call_llama_streaming(FACTUAL_EXTRACT_PROMPT.format(
#             context=context[:2500], question=question), request_id, temperature=0.0)
#     llm_calls += 1
#     grounding = compute_answer_grounding(final, context_chunks)
#     return final, "llama_react_final", MAX_STEPS, retrieval_score, context_precision, grounding, llm_calls

















#got issue in react_agent action block-lap claude
#  import re
# from docmind_rag.config.settings import qa_pipeline, FACTUAL_TOP_K
# from docmind_rag.models.llm import call_llama, call_llama_streaming
# from docmind_rag.core.prompts import REACT_PROMPT, MULTIPART_PROMPT, REASONING_PROMPT, FACTUAL_EXTRACT_PROMPT
# from docmind_rag.models.reranker import rerank_docs, protect_exact_matches
# from docmind_rag.services.retrieval import multi_query_retrieve
# from docmind_rag.events.events import emit_event
# from docmind_rag.utils.helpers import clean_artifacts, _is_cop_out_answer
# from docmind_rag.utils.text import reorder_by_question, extract_numeric_answer
# from docmind_rag.utils.metrics import compute_answer_grounding, compute_retrieval_score, compute_context_precision


# def _is_echo_answer(answer: str, question: str) -> bool:
#     """
#     Returns True if the answer adds zero new information beyond the question.
#     Catches cases like Q: "what is the name of lecture 3?" A: "Lecture 3"
#     """
#     q_tokens = set(re.findall(r'\b\w+\b', question.lower()))
#     a_tokens = set(re.findall(r'\b\w+\b', answer.lower()))
#     new_tokens = a_tokens - q_tokens
#     return len(new_tokens) == 0


# # ============================================================
# # RoBERTa QA
# # ============================================================
# def roberta_qa(question: str, chunks: list):
#     best_answer      = ""
#     best_score       = 0.0
#     best_empty_score = 0.0

#     for chunk in chunks:
#         try:
#             result = qa_pipeline(question=question, context=chunk[:3000])
#             ans    = (result.get('answer') or "").strip()
#             score  = result.get('score', 0.0)
#             if ans:
#                 if score > best_score:
#                     best_score  = score
#                     best_answer = ans
#             else:
#                 if score > best_empty_score:
#                     best_empty_score = score
#         except Exception:
#             continue

#     final_answer = best_answer.strip()
#     if not final_answer and best_empty_score >= 0.4:
#         regex_ans = extract_numeric_answer(question, chunks)
#         if regex_ans:
#             print(f"[RoBERTa] Empty span rescued by regex: '{regex_ans}'")
#             return regex_ans, round(best_empty_score, 4)

#     return final_answer, round(best_score, 4)


# # ============================================================
# # CORE: ReAct AGENT ENGINE
# # ============================================================
# def react_agent(question: str, faiss_index, query_type: str,
#                 all_chunks: list, request_id: str = "") -> tuple:
#     MAX_STEPS      = 3
#     scratchpad     = ""
#     context_chunks = []
#     model_used     = "llama_react"
#     llm_calls      = 0

#     initial = multi_query_retrieve(question, faiss_index, k=20,
#                                    all_chunks=all_chunks, query_type=query_type)
#     apply_pruning = (query_type != "MULTIPART_QA")
#     initial, reranker_top_score, _ = rerank_docs(question, initial,
#                                                   top_k=FACTUAL_TOP_K,
#                                                   apply_pruning=apply_pruning)
#     initial = protect_exact_matches(question, initial, all_chunks,
#                                     top_k=FACTUAL_TOP_K)

#     # Pass full chunks — no sentence windowing.
#     # _focus_context_on_query was cutting out correct sentences
#     # (e.g. the Hinton attribution sentence) causing wrong answers.
#     # The LLM prompt rules handle attribution correctly when given full context.
#     context_chunks = [d.page_content for d in initial]

#     retrieval_score   = compute_retrieval_score(question, initial)
#     context_precision = compute_context_precision(question, initial)

#     print(f"[ReAct] Starting | {query_type} | {len(initial)} chunks | "
#           f"retrieval={retrieval_score:.1f}%")
#     emit_event(request_id, "agent_start",
#                f"🤖 Agent starting | {query_type} | {len(initial)} chunks retrieved")

#     for step in range(MAX_STEPS):
#         # 🔥 prioritize most relevant chunks BEFORE truncation
#         ranked_chunks = reorder_by_question(question, context_chunks)

#         # keep top 3 instead of all (prevents cutting important lines)
#         top_chunks = ranked_chunks[:7]
#         print("\n========== DEBUG: TOP CHUNKS ==========")
#         for i, chunk in enumerate(top_chunks):
#             preview = chunk[:300].replace("\n", " ")
#             print(f"\n--- Chunk {i+1} ---")
#             print(preview)
#         print("======================================\n")

#         context = "\n\n---\n\n".join(top_chunks)
#         raw = call_llama(REACT_PROMPT.format(
#             question=question, context=context[:2500],
#             scratchpad=scratchpad if scratchpad else "None yet"),
#             temperature=0.0)
#         llm_calls += 1
#         print(f"[ReAct] Step {step+1}: {raw[:120].strip()}")

#         action       = ""
#         action_input = ""
#         thought      = raw[:100]
#         lines        = raw.split("\n")
#         for i, line in enumerate(lines):
#             line = line.strip()
#             if line.startswith("Thought:"):
#                 thought = line.replace("Thought:", "").strip()
#             elif line.startswith("Action:"):
#                 action_raw = line.replace("Action:", "").strip().lower()
#                 if "final" in action_raw or "answer" in action_raw:
#                     action = "final_answer"
#                 elif "search" in action_raw or "more" in action_raw:
#                     action = "search_more"
#                 else:
#                     action = "final_answer"
#             elif line.startswith("Input:"):
#                 action_input = line.replace("Input:", "").strip()
#                 for j in range(i+1, len(lines)):
#                     next_line = lines[j].strip()
#                     if not next_line:
#                         continue
#                     if next_line.startswith(("Thought:", "Action:", "Input:")):
#                         break
#                     action_input += " " + next_line

#         if action_input:
#             action_input = clean_artifacts(action_input)

#         if not action:
#             action       = "final_answer"
#             action_input = clean_artifacts(raw)

#         if "final_answer" in action:
#             is_bad_input = (
#                 _is_cop_out_answer(action_input, question) or
#                 _is_echo_answer(action_input, question)
#             )

#             if is_bad_input:
#                 print(f"[ReAct] Bad input detected — using direct prompt")
#                 emit_event(request_id, "agent_action", "⚡ Using direct prompt...")
#                 # 🔥 prioritize most relevant chunks BEFORE truncation
#                 ranked_chunks = reorder_by_question(question, context_chunks)

#                 # keep top 3 instead of all (prevents cutting important lines)
#                 top_chunks = ranked_chunks[:3]

#                 context = "\n\n---\n\n".join(top_chunks)
#                 if query_type == "MULTIPART_QA":
#                     answer = call_llama(MULTIPART_PROMPT.format(
#                         context=context[:2500], question=question),
#                         temperature=0.0)
#                 elif query_type == "REASONING_QA":
#                     answer = call_llama(REASONING_PROMPT.format(
#                         context=context[:2500], question=question),
#                         temperature=0.0)
#                 else:
#                     answer = call_llama(FACTUAL_EXTRACT_PROMPT.format(
#                         context=context[:2500], question=question),
#                         temperature=0.0)
#                 llm_calls += 1
#                 model_used = "llama_react_direct"
#             else:
#                 answer     = action_input
#                 model_used = "llama_react"

#             grounding = compute_answer_grounding(answer, context_chunks)
#             print(f"[ReAct] ✅ Answer at step {step+1} | grounding={grounding:.1f}%")
#             emit_event(request_id, "agent_done", f"✅ Answer found at step {step+1}!")
#             # return answer, model_used, step+1, retrieval_score, context_precision, grounding, llm_calls changed this to this
#             return answer, model_used, step+1, retrieval_score, context_precision, grounding, llm_calls, context_chunks

#         elif "search_more" in action:
#             query    = action_input.strip('"\'') if action_input else question
#             new_docs = multi_query_retrieve(query, faiss_index, k=20,
#                                              all_chunks=all_chunks, query_type=query_type)
#             new_docs, _, _ = rerank_docs(query, new_docs, top_k=FACTUAL_TOP_K,
#                                           apply_pruning=True)
#             new_texts = [d.page_content for d in new_docs]
#             seen      = set(context_chunks)
#             for t in new_texts:
#                 if t not in seen:
#                     context_chunks.append(t)
#                     seen.add(t)
#             print(f"[ReAct] 🔍 Searched: '{query}' → {len(new_texts)} chunks | "
#                   f"total ctx: {len(context_chunks)}")
#             emit_event(request_id, "agent_search",
#                        f"🔍 Searching: '{query[:60]}' → {len(new_texts)} chunks")
#             scratchpad += f"  Found {len(new_texts)} additional chunks\n"
#         else:
#             if len(raw) > 20:
#                 grounding = compute_answer_grounding(raw, context_chunks)
#                 return raw, "llama_react_direct", step+1, retrieval_score, context_precision, grounding, llm_calls

#     emit_event(request_id, "agent_action", "⚡ Synthesizing final answer...")
#     # 🔥 prioritize most relevant chunks BEFORE truncation
#     ranked_chunks = reorder_by_question(question, context_chunks)

#     # keep top 3 instead of all (prevents cutting important lines)
#     top_chunks = ranked_chunks[:3]

#     context = "\n\n---\n\n".join(top_chunks)
#     emit_event(request_id, "stream_start", "✍️ Generating final answer...")
#     if query_type == "MULTIPART_QA":
#         final = call_llama_streaming(MULTIPART_PROMPT.format(
#             context=context[:2500], question=question), request_id, temperature=0.0)
#     elif query_type == "REASONING_QA":
#         final = call_llama_streaming(REASONING_PROMPT.format(
#             context=context[:2500], question=question), request_id, temperature=0.0)
#     else:
#         final = call_llama_streaming(FACTUAL_EXTRACT_PROMPT.format(
#             context=context[:2500], question=question), request_id, temperature=0.0)
#     llm_calls += 1
#     grounding = compute_answer_grounding(final, context_chunks)
#     return final, "llama_react_final", MAX_STEPS, retrieval_score, context_precision, grounding, llm_calls





























# from docmind_rag.config.settings import qa_pipeline, FACTUAL_TOP_K
# from docmind_rag.models.llm import call_llama, call_llama_streaming
# from docmind_rag.core.prompts import REACT_PROMPT, MULTIPART_PROMPT, REASONING_PROMPT, FACTUAL_EXTRACT_PROMPT
# from docmind_rag.models.reranker import rerank_docs, protect_exact_matches
# from docmind_rag.services.retrieval import multi_query_retrieve
# from docmind_rag.events.events import emit_event
# from docmind_rag.utils.helpers import clean_artifacts, _is_cop_out_answer
# from docmind_rag.utils.text import reorder_by_question, extract_numeric_answer
# from docmind_rag.utils.metrics import compute_answer_grounding, compute_retrieval_score, compute_context_precision

# # ============================================================
# # RoBERTa QA — candidate extractor
# # ============================================================
# def roberta_qa(question: str, chunks: list):
#     best_answer      = ""
#     best_score       = 0.0
#     best_empty_score = 0.0

#     for chunk in chunks:
#         try:
#             result = qa_pipeline(question=question, context=chunk[:3000])
#             ans    = (result.get('answer') or "").strip()
#             score  = result.get('score', 0.0)

#             if ans:
#                 if score > best_score:
#                     best_score  = score
#                     best_answer = ans
#             else:
#                 if score > best_empty_score:
#                     best_empty_score = score

#         except Exception:
#             continue

#     final_answer = best_answer.strip()

#     if not final_answer and best_empty_score >= 0.4:
#         regex_ans = extract_numeric_answer(question, chunks)
#         if regex_ans:
#             print(f"[RoBERTa] Empty span rescued by regex: '{regex_ans}' "
#                   f"(empty_score={best_empty_score:.4f})")
#             return regex_ans, round(best_empty_score, 4)

#     return final_answer, round(best_score, 4)


# # ============================================================
# # CORE: ReAct AGENT ENGINE
# # ============================================================
# def react_agent(question: str, faiss_index, query_type: str,
#                 all_chunks: list, request_id: str = "") -> tuple:
#     MAX_STEPS      = 3
#     scratchpad     = ""
#     context_chunks = []
#     model_used     = "llama_react"
#     llm_calls      = 0

#     initial = multi_query_retrieve(question, faiss_index, k=20,
#                                    all_chunks=all_chunks, query_type=query_type)
#     apply_pruning = (query_type != "MULTIPART_QA")
#     initial, reranker_top_score, _ = rerank_docs(question, initial,
#                                                   top_k=FACTUAL_TOP_K,
#                                                   apply_pruning=apply_pruning)
#     initial = protect_exact_matches(question, initial, all_chunks,
#                                     top_k=FACTUAL_TOP_K)
#     context_chunks = [d.page_content for d in initial]
#     retrieval_score   = compute_retrieval_score(question, initial)
#     context_precision = compute_context_precision(question, initial)

#     print(f"[ReAct] Starting | {query_type} | {len(initial)} chunks | "
#           f"retrieval={retrieval_score:.1f}%")
#     emit_event(request_id, "agent_start",
#                f"🤖 Agent starting | {query_type} | {len(initial)} chunks retrieved")

#     for step in range(MAX_STEPS):
#         context = "\n\n---\n\n".join(reorder_by_question(question, context_chunks))
#         raw = call_llama(REACT_PROMPT.format(
#             question=question, context=context[:2500],
#             scratchpad=scratchpad if scratchpad else "None yet"),
#             temperature=0.0)
#         llm_calls += 1
#         print(f"[ReAct] Step {step+1}: {raw[:120].strip()}")

#         action       = ""
#         action_input = ""
#         thought      = raw[:100]
#         lines        = raw.split("\n")
#         for i, line in enumerate(lines):
#             line = line.strip()
#             if line.startswith("Thought:"):
#                 thought = line.replace("Thought:", "").strip()
#             elif line.startswith("Action:"):
#                 action_raw = line.replace("Action:", "").strip().lower()
#                 if "final" in action_raw or "answer" in action_raw:
#                     action = "final_answer"
#                 elif "search" in action_raw or "more" in action_raw:
#                     action = "search_more"
#                 else:
#                     action = "final_answer"
#             elif line.startswith("Input:"):
#                 action_input = line.replace("Input:", "").strip()
#                 for j in range(i+1, len(lines)):
#                     next_line = lines[j].strip()
#                     if not next_line:
#                         continue
#                     if next_line.startswith(("Thought:", "Action:", "Input:")):
#                         break
#                     action_input += " " + next_line

#         if action_input:
#             action_input = clean_artifacts(action_input)

#         if not action:
#             action       = "final_answer"
#             action_input = clean_artifacts(raw)

#         if "final_answer" in action:
#             # FIX: use dynamic cop-out detection instead of hardcoded phrase list
#             is_bad_input = _is_cop_out_answer(action_input, question)

#             if is_bad_input:
#                 emit_event(request_id, "agent_action", "⚡ Using direct prompt...")
#                 context = "\n\n---\n\n".join(reorder_by_question(question, context_chunks))
#                 if query_type == "MULTIPART_QA":
#                     answer = call_llama(MULTIPART_PROMPT.format(
#                         context=context[:2500], question=question),
#                         temperature=0.0)
#                 elif query_type == "REASONING_QA":
#                     answer = call_llama(REASONING_PROMPT.format(
#                         context=context[:2500], question=question),
#                         temperature=0.0)
#                 else:
#                     answer = call_llama(FACTUAL_EXTRACT_PROMPT.format(
#                         context=context[:2500], question=question),
#                         temperature=0.0)
#                 llm_calls += 1
#                 model_used = "llama_react_direct"
#             else:
#                 answer     = action_input
#                 model_used = "llama_react"

#             grounding = compute_answer_grounding(answer, context_chunks)
#             print(f"[ReAct] ✅ Answer at step {step+1} | grounding={grounding:.1f}%")
#             emit_event(request_id, "agent_done", f"✅ Answer found at step {step+1}!")
#             return answer, model_used, step+1, retrieval_score, context_precision, grounding, llm_calls

#         elif "search_more" in action:
#             query     = action_input.strip('"\'') if action_input else question
#             new_docs  = multi_query_retrieve(query, faiss_index, k=20,
#                                              all_chunks=all_chunks, query_type=query_type)
#             new_docs, _, _ = rerank_docs(query, new_docs, top_k=FACTUAL_TOP_K,
#                                           apply_pruning=True)
#             new_texts = [d.page_content for d in new_docs]
#             seen      = set(context_chunks)
#             for t in new_texts:
#                 if t not in seen:
#                     context_chunks.append(t)
#                     seen.add(t)
#             print(f"[ReAct] 🔍 Searched: '{query}' → {len(new_texts)} chunks | "
#                   f"total ctx: {len(context_chunks)}")
#             emit_event(request_id, "agent_search",
#                        f"🔍 Searching: '{query[:60]}' → {len(new_texts)} chunks")
#             scratchpad += f"  Found {len(new_texts)} additional chunks\n"
#         else:
#             if len(raw) > 20:
#                 grounding = compute_answer_grounding(raw, context_chunks)
#                 return raw, "llama_react_direct", step+1, retrieval_score, context_precision, grounding, llm_calls

#     emit_event(request_id, "agent_action", "⚡ Synthesizing final answer...")
#     context = "\n\n---\n\n".join(reorder_by_question(question, context_chunks))
#     emit_event(request_id, "stream_start", "✍️ Generating final answer...")
#     if query_type == "MULTIPART_QA":
#         final = call_llama_streaming(MULTIPART_PROMPT.format(
#             context=context[:2500], question=question), request_id, temperature=0.0)
#     elif query_type == "REASONING_QA":
#         final = call_llama_streaming(REASONING_PROMPT.format(
#             context=context[:2500], question=question), request_id, temperature=0.0)
#     else:
#         final = call_llama_streaming(FACTUAL_EXTRACT_PROMPT.format(
#             context=context[:2500], question=question), request_id, temperature=0.0)
#     llm_calls += 1
#     grounding = compute_answer_grounding(final, context_chunks)
#     return final, "llama_react_final", MAX_STEPS, retrieval_score, context_precision, grounding, llm_calls