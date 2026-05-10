import os 
import json
import time

from flask import request, jsonify

from docmind_rag.models.llm import call_llama, _ttft_tracker
from docmind_rag.core.state import DocState
from docmind_rag.services.extraction import _extraction_cache
from docmind_rag.services.chunking import _summary_cache
from docmind_rag.models.embeddings import _faiss_cache
from docmind_rag.events.events import emit_event, cleanup_queue
from docmind_rag.config.settings import OLLAMA_MODEL, device

def register_routes(app, workflow):
    """Register all Flask routes onto the given app instance."""

    SAVED_SUMMARIES = os.path.join(os.path.dirname(__file__), "..", "saved_summaries")
    os.makedirs(SAVED_SUMMARIES, exist_ok=True)

    SUMMARY_TRIGGERS = ["summary", "summarise", "summarize", "tell me about this pdf"]

    def is_summary_question(q: str) -> bool:
        q_lower = q.lower()
        return any(trigger in q_lower for trigger in SUMMARY_TRIGGERS)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "model":  OLLAMA_MODEL,
            "device": "cuda" if device == 0 else "cpu",
            "faiss_cached":     len(_faiss_cache),
            "extracted_pdfs":   len(_extraction_cache),
            "summaries_cached": len(_summary_cache),
        })

    @app.route("/upload", methods=["POST"])
    def upload():
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No file provided"}), 400
        file = request.files["file"]
        if not file.filename.lower().endswith(".pdf"):
            return jsonify({"status": "error", "message": "Only PDF files allowed"}), 400
        filename = file.filename
        save_path = os.path.join(SAVED_SUMMARIES, filename)
        file.save(save_path)
        print(f"[Upload] Saved: {save_path}")
        return jsonify({"status": "success", "filename": filename, "path": save_path})

    @app.route("/save_summary", methods=["POST"])
    def save_summary():
        try:
            data    = request.json or {}
            hash_id = data.get("hash", "").strip()
            summary = data.get("summary", "").strip()
            if not hash_id or not summary:
                return jsonify({"status": "error", "message": "Missing hash or summary"}), 400
            txt_path = os.path.join(SAVED_SUMMARIES, hash_id + ".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(summary)
            print(f"[Summary] Saved: {txt_path}")
            return jsonify({"status": "success"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/delete_summary", methods=["POST"])
    def delete_summary():
        try:
            data    = request.json or {}
            hash_id = data.get("hash", "").strip()
            if not hash_id:
                return jsonify({"status": "error", "message": "Missing hash"}), 400
            pdf_path = os.path.join(SAVED_SUMMARIES, hash_id + ".pdf")
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
                print(f"[Delete] Removed PDF: {pdf_path}")
            txt_path = os.path.join(SAVED_SUMMARIES, hash_id + ".txt")
            if os.path.exists(txt_path):
                os.remove(txt_path)
                print(f"[Delete] Removed summary: {txt_path}")
            return jsonify({"status": "success"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/process", methods=["POST"])
    def process():
        data = request.json
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        pdf_path   = data.get("pdf_path", "").strip()
        question   = data.get("question", "").strip()
        request_id = data.get("request_id", "")
        file_hash  = data.get("file_hash", "").strip()

        if not pdf_path:
            return jsonify({"error": "Missing required field: pdf_path"}), 400
        if not question:
            return jsonify({"error": "Missing required field: question"}), 400
        if len(question) > 2000:
            return jsonify({"error": "Question too long (max 2000 chars)"}), 400
        if not os.path.exists(pdf_path):
            return jsonify({"error": f"PDF not found: {pdf_path}"}), 404
        if not pdf_path.lower().endswith(".pdf"):
            return jsonify({"error": "Only PDF files are supported"}), 400

        # ── Return cached summary if available ──
        if file_hash and is_summary_question(question):
            txt_path = os.path.join(SAVED_SUMMARIES, file_hash + ".txt")
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    cached_summary = f.read().strip()
                if cached_summary:
                    print(f"[Cache] Returning saved summary for hash: {file_hash}")
                    emit_event(request_id, "done", "✅ Loaded from cache!")
                    cleanup_queue(request_id)
                    return jsonify({
                        "answer":     cached_summary,
                        "metrics":    {"type": "summary", "cache_hit": True},
                        "metrics_v1": {"type": "summary", "cache_hit": True},
                        "metrics_v2": {"cache_hit": True},
                    })

        emit_event(request_id, "extract_start", "📄 Extracting PDF text...")

        initial_state: DocState = {
            "pdf_path":       pdf_path,
            "question":       question,
            "extracted_text": "",
            "chunks":         [],
            "summary_chunks": [],
            "faiss_index":    None,
            "answer":         "",
            "metrics":        {},
            "doc_type":       "general",
            "query_type":     "",
            "retry_count":    0,
            "start_time":     time.time(),
            "page_count":     0,
            "char_count":     0,
            "request_id":     request_id
        }

        try:
            request_start = time.time()
            emit_event(request_id, "workflow_start", "⚙️ LangGraph workflow started...")
            result = workflow.invoke(initial_state)
            m      = result["metrics"]

            # ── Always save summary if this was a summary type ──
            if m.get("type") == "summary" and file_hash:
                txt_path = os.path.join(SAVED_SUMMARIES, file_hash + ".txt")
                if not os.path.exists(txt_path):  # don't overwrite existing
                    try:
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(result.get("answer", ""))
                        print(f"[Summary] Auto-saved: {txt_path}")
                    except Exception as e:
                        print(f"[Summary] Warning: could not save txt: {e}")

            total_e2e = round(time.time() - request_start, 2)
            m["e2e_latency_sec"] = total_e2e

            real_ttft = _ttft_tracker.pop(request_id, None)
            if real_ttft is not None:
                m["ttft_sec"] = real_ttft
            else:
                m["ttft_sec"] = m.get("ttft_sec", total_e2e)

            print(f"[LATENCY] E2E={total_e2e}s | TTFT={m['ttft_sec']}s | TPS={m.get('tps',0)}")
            if m.get("type") == "qa":
                print(f"[QUALITY] Grounding={m.get('answer_grounding',0)}% | "
                      f"Retrieval={m.get('retrieval_score',0)}% | "
                      f"Precision={m.get('context_precision',0)}% | "
                      f"Recall={m.get('recall_at_k',0)}% | "
                      f"Confidence={m.get('confidence_score',0)}%")

            emit_event(request_id, "done", "✅ Processing complete!")
            cleanup_queue(request_id)

            v1 = {
                "type":                 m.get("type", ""),
                "response_time_sec":    round(m.get("response_time_sec", 0), 2),
                "extraction_time_sec":  round(m.get("extraction_time_sec", 0), 2),
                "pages_processed":      m.get("pages_processed", 0),
                "characters_processed": m.get("characters_processed", 0),
                "words_processed":      m.get("words_processed", 0),
            }
            if m.get("type") == "summary":
                v1["summary_time_sec"]     = round(m.get("summary_time_sec", 0), 2)
                v1["summary_length_words"] = m.get("summary_length_words", 0)
            if m.get("type") == "qa":
                v1["qa_time_sec"]      = round(m.get("qa_time_sec", 0), 2)
                v1["confidence_score"] = round(m.get("confidence_score", 0), 2)

            v2 = {
                "ttft_sec":          round(m.get("ttft_sec", 0), 2),
                "e2e_latency_sec":   round(m.get("e2e_latency_sec", 0), 2),
                "tps":               m.get("tps", 0),
                "doc_type":          m.get("doc_type", "general"),
                "query_type":        m.get("query_type", ""),
                "model_used":        m.get("model_used", ""),
                "chunks_created":    m.get("chunks_created", 0),
                "retry_count":       m.get("retry_count", 0),
                "llm_calls":         m.get("llm_calls", 0),
                "retrieval_score":   m.get("retrieval_score", 0),
                "context_precision": m.get("context_precision", 0),
                "answer_grounding":  m.get("answer_grounding", 0),
                "recall_at_k":       m.get("recall_at_k", 0),
            }
            if m.get("type") == "summary":
                v2["parallel_workers"] = m.get("parallel_workers", 0)
                v2["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
                v2["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)
            if m.get("type") == "qa":
                v2["chunks_retrieved"] = m.get("chunks_retrieved", 0)
                v2["decision_type"]    = m.get("decision_type", "accepted")
                v2["confidence_raw"]   = m.get("confidence_raw", 0.0)

            return jsonify({
                "answer":     result["answer"],
                "metrics":    {**v1, **v2},
                "metrics_v1": v1,
                "metrics_v2": v2,
            })

        except Exception as e:
            return jsonify({
                "answer":     f"Error: {str(e)}",
                "metrics":    {},
                "metrics_v1": {},
                "metrics_v2": {}
            })

    def is_refusal_answer(answer: str) -> bool:
        if not answer or len(answer.strip()) < 3:
            return True
        prompt = (
            f"Does this answer say that the requested information "
            f"was not found, not present, or not available?\n\n"
            f"Answer: {answer}\n\n"
            f"Reply with YES or NO only."
        )
        try:
            result = call_llama(prompt, num_ctx=256, temperature=0.0).strip().upper()
            is_refusal = "YES" in result
            print(f"[Refusal] '{answer[:60]}' → {'REFUSAL' if is_refusal else 'FACTUAL'}")
            return is_refusal
        except Exception:
            return False

    @app.route("/evaluate", methods=["POST"])
    def evaluate():
        try:
            data     = request.json or {}
            pdf_path = data.get("pdf_path", "").strip()
            run_desc = data.get("run_description", f"run_{int(time.time())}")

            if not pdf_path or not os.path.exists(pdf_path):
                return jsonify({"error": f"PDF not found: {pdf_path}"}), 404

            pdf_name = os.path.basename(pdf_path)

            PDF_DATASET_MAP = {
                "Ethics of AI_new.pdf": r"C:\xampp\htdocs\genai-summary+qa - kowshik\docmind_rag\grounding_dataset_ethics.json",
                "crop report for testin -genai.pdf": r"C:\xampp\htdocs\genai-summary+qa - kowshik\docmind_rag\grounding_dataset_crop.json",
                "news -genai.pdf": r"C:\xampp\htdocs\genai-summary+qa - kowshik\docmind_rag\grounding_dataset_news.json"
            }

            dataset_path = PDF_DATASET_MAP.get(pdf_name)

            if not dataset_path or not os.path.exists(dataset_path):
                return jsonify({"error": f"No dataset mapped for {pdf_name}"}), 404

            with open(dataset_path, "r") as f:
                dataset = json.load(f)

            questions = dataset.get("questions", [])
            if not questions:
                return jsonify({"error": "No questions found in grounding_dataset.json"}), 400

            results       = []
            pass_count    = 0
            partial_count = 0
            fail_count    = 0

            print(f"[Evaluate] Starting evaluation run: {run_desc}")
            print(f"[Evaluate] PDF: {pdf_path}")
            print(f"[Evaluate] Questions: {len(questions)}")

            for q in questions:
                try:
                    q_id       = q["id"]
                    question   = q["question"]
                    query_type = q.get("query_type", "FACTUAL_QA")
                    keywords   = [kw.lower() for kw in q.get("expected_keywords", [])]
                    threshold  = q.get("pass_threshold", 1)
                    max_score  = q.get("max_score", len(keywords))
                    is_halluc  = q.get("hallucination_test", False)
                    is_roberta = q.get("roberta_test", False)

                    if q.get("skip", False):
                        print(f"[Evaluate] Q{q_id}: ⏭️ SKIPPED")
                        continue

                    print(f"[Evaluate] Q{q_id}: {question[:60]}...")

                    initial_state: DocState = {
                        "pdf_path":       pdf_path,
                        "question":       question,
                        "extracted_text": "",
                        "chunks":         [],
                        "summary_chunks": [],
                        "faiss_index":    None,
                        "answer":         "",
                        "metrics":        {},
                        "doc_type":       "general",
                        "query_type":     "",
                        "retry_count":    0,
                        "start_time":     time.time(),
                        "page_count":     0,
                        "char_count":     0,
                        "request_id":     ""
                    }

                    try:
                        result     = workflow.invoke(initial_state)
                        answer     = result.get("answer", "")
                        metrics    = result.get("metrics", {})
                        model_used = metrics.get("model_used", "unknown")
                        recall_k   = metrics.get("recall_at_k", 0)
                        grounding  = metrics.get("answer_grounding", 0)
                        confidence = metrics.get("confidence_score", 0)
                        print(f"[Evaluate] Q{q_id} actual answer: {answer[:300]}")
                    except Exception as invoke_err:
                        print(f"[Evaluate] Q{q_id} workflow error: {invoke_err}")
                        answer     = f"ERROR: {str(invoke_err)}"
                        metrics    = {}
                        model_used = "error"
                        recall_k   = 0
                        grounding  = 0
                        confidence = 0

                    answer_lower = answer.lower()
                    keyword_hits = 0

                    short_answer = len(answer.split()) <= 3

                    hallucination_detected = False

                    if not is_halluc and grounding < 60 and not short_answer:
                        hallucination_detected = True
                        print(f"[Eval] ❌ Low grounding ({grounding:.1f}%) → hallucination detected")

                    if is_halluc:
                        found_safe = is_refusal_answer(answer)
                        if found_safe:
                            verdict = "PASS"
                            keyword_hits = max_score
                            hallucination_detected = False
                        else:
                            verdict = "FAIL"
                            keyword_hits = 0
                            hallucination_detected = True
                            print(f"[Halluc] ❌ Answer is not a refusal — hallucination detected")
                    else:
                        if hallucination_detected:
                            verdict = "FAIL"
                            keyword_hits = 0
                            print(f"[Eval] ❌ Forced FAIL due to low grounding")
                        else:
                            keyword_hits = sum(1 for kw in keywords if kw in answer_lower)
                            if keyword_hits >= threshold:
                                verdict = "PASS"
                            elif keyword_hits >= max(1, threshold // 2):
                                verdict = "PARTIAL"
                            else:
                                verdict = "FAIL"

                    if verdict == "PASS":
                        pass_count += 1
                    elif verdict == "PARTIAL":
                        partial_count += 1
                    else:
                        fail_count += 1

                    result_entry = {
                        "id":               q_id,
                        "question":         question,
                        "query_type":       query_type,
                        "verdict":          verdict,
                        "keyword_hits":     keyword_hits,
                        "max_score":        max_score,
                        "pass_threshold":   threshold,
                        "model_used":       model_used,
                        "recall_at_k":      round(recall_k, 1),
                        "answer_grounding": round(grounding, 1),
                        "confidence":       round(confidence, 1),
                        "hallucination":    hallucination_detected,
                        "roberta_test":     is_roberta,
                        "actual_answer":    answer[:300],
                        "tests":            q.get("tests", "")
                    }
                    results.append(result_entry)

                    verdict_icon = "✅" if verdict == "PASS" else "⚠️" if verdict == "PARTIAL" else "❌"
                    print(f"[Evaluate] Q{q_id} {verdict_icon} {verdict} | "
                          f"hits={keyword_hits}/{max_score} | model={model_used} | "
                          f"recall={recall_k:.1f}% | confidence={confidence:.1f}%")

                except Exception as q_err:
                    print(f"[Evaluate] Q{q.get('id','?')} unexpected error: {q_err}")
                    continue

            total = len(results)
            if total == 0:
                return jsonify({"error": "All questions were skipped or failed"}), 400

            pass_rate = round(pass_count / total * 100, 1)
            run_summary = {
                "run_id":          run_desc,
                "date":            time.strftime("%Y-%m-%d %H:%M:%S"),
                "pdf":             os.path.basename(pdf_path),
                "total_questions": total,
                "pass":            pass_count,
                "partial":         partial_count,
                "fail":            fail_count,
                "pass_rate":       pass_rate,
                "results":         results
            }

            results_path = os.path.join(os.path.dirname(__file__), "..", "grounding_results.json")
            try:
                if os.path.exists(results_path):
                    with open(results_path, "r") as f:
                        existing = json.load(f)
                else:
                    existing = {"runs": []}
                existing["runs"].append(run_summary)
                with open(results_path, "w") as f:
                    json.dump(existing, f, indent=2)
                print(f"[Evaluate] Results saved to grounding_results.json")
            except Exception as save_err:
                print(f"[Evaluate] Warning: could not save results: {save_err}")

            print(f"[Evaluate] ✅ Done | Pass={pass_count} Partial={partial_count} "
                  f"Fail={fail_count} | Pass rate={pass_rate}%")

            return jsonify({
                "run_id":          run_desc,
                "total_questions": total,
                "pass":            pass_count,
                "partial":         partial_count,
                "fail":            fail_count,
                "pass_rate":       f"{pass_rate}%",
                "results":         results
            })

        except Exception as fatal_err:
            print(f"[Evaluate] ❌ Fatal error: {fatal_err}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"Fatal evaluation error: {str(fatal_err)}"}), 500

#changing it to add save yes or no option for saved_summaries

# import os 
# import json
# import time

# from flask import request, jsonify

# from docmind_rag.models.llm import call_llama, _ttft_tracker
# from docmind_rag.core.state import DocState
# from docmind_rag.services.extraction import _extraction_cache
# from docmind_rag.services.chunking import _summary_cache
# from docmind_rag.models.embeddings import _faiss_cache
# from docmind_rag.events.events import emit_event, cleanup_queue
# from docmind_rag.config.settings import OLLAMA_MODEL, device

# def register_routes(app, workflow):
#     """Register all Flask routes onto the given app instance."""

#     # ============================================================
#     # FLASK ROUTES
#     # ============================================================
#     @app.route("/health", methods=["GET"])
#     def health():
#         return jsonify({
#             "status": "ok",
#             "model":  OLLAMA_MODEL,
#             "device": "cuda" if device == 0 else "cpu",
#             "faiss_cached":     len(_faiss_cache),
#             "extracted_pdfs":   len(_extraction_cache),
#             "summaries_cached": len(_summary_cache),
#         })

#     @app.route("/process", methods=["POST"])
#     def process():
#         data = request.json
#         if not data:
#             return jsonify({"error": "Request body must be JSON"}), 400

#         pdf_path   = data.get("pdf_path", "").strip()
#         question   = data.get("question", "").strip()
#         request_id = data.get("request_id", "")

#         if not pdf_path:
#             return jsonify({"error": "Missing required field: pdf_path"}), 400
#         if not question:
#             return jsonify({"error": "Missing required field: question"}), 400
#         if len(question) > 2000:
#             return jsonify({"error": "Question too long (max 2000 chars)"}), 400
#         if not os.path.exists(pdf_path):
#             return jsonify({"error": f"PDF not found: {pdf_path}"}), 404
#         if not pdf_path.lower().endswith(".pdf"):
#             return jsonify({"error": "Only PDF files are supported"}), 400

#         emit_event(request_id, "extract_start", "📄 Extracting PDF text...")

#         initial_state: DocState = {
#             "pdf_path":       pdf_path,
#             "question":       question,
#             "extracted_text": "",
#             "chunks":         [],
#             "summary_chunks": [],
#             "faiss_index":    None,
#             "answer":         "",
#             "metrics":        {},
#             "doc_type":       "general",
#             "query_type":     "",
#             "retry_count":    0,
#             "start_time":     time.time(),
#             "page_count":     0,
#             "char_count":     0,
#             "request_id":     request_id
#         }

#         try:
#             request_start = time.time()
#             emit_event(request_id, "workflow_start", "⚙️ LangGraph workflow started...")
#             result      = workflow.invoke(initial_state)
#             m           = result["metrics"]
#             total_e2e   = round(time.time() - request_start, 2)
#             m["e2e_latency_sec"] = total_e2e

#             real_ttft = _ttft_tracker.pop(request_id, None)
#             if real_ttft is not None:
#                 m["ttft_sec"] = real_ttft
#             else:
#                 m["ttft_sec"] = m.get("ttft_sec", total_e2e)

#             print(f"[LATENCY] E2E={total_e2e}s | TTFT={m['ttft_sec']}s | TPS={m.get('tps',0)}")
#             if m.get("type") == "qa":
#                 print(f"[QUALITY] Grounding={m.get('answer_grounding',0)}% | "
#                       f"Retrieval={m.get('retrieval_score',0)}% | "
#                       f"Precision={m.get('context_precision',0)}% | "
#                       f"Recall={m.get('recall_at_k',0)}% | "
#                       f"Confidence={m.get('confidence_score',0)}%")

#             emit_event(request_id, "done", "✅ Processing complete!")
#             cleanup_queue(request_id)

#             v1 = {
#                 "type":                 m.get("type", ""),
#                 "response_time_sec":    round(m.get("response_time_sec", 0), 2),
#                 "extraction_time_sec":  round(m.get("extraction_time_sec", 0), 2),
#                 "pages_processed":      m.get("pages_processed", 0),
#                 "characters_processed": m.get("characters_processed", 0),
#                 "words_processed":      m.get("words_processed", 0),
#             }
#             if m.get("type") == "summary":
#                 v1["summary_time_sec"]     = round(m.get("summary_time_sec", 0), 2)
#                 v1["summary_length_words"] = m.get("summary_length_words", 0)
#             if m.get("type") == "qa":
#                 v1["qa_time_sec"]      = round(m.get("qa_time_sec", 0), 2)
#                 v1["confidence_score"] = round(m.get("confidence_score", 0), 2)

#             v2 = {
#                 "ttft_sec":          round(m.get("ttft_sec", 0), 2),
#                 "e2e_latency_sec":   round(m.get("e2e_latency_sec", 0), 2),
#                 "tps":               m.get("tps", 0),
#                 "doc_type":          m.get("doc_type", "general"),
#                 "query_type":        m.get("query_type", ""),
#                 "model_used":        m.get("model_used", ""),
#                 "chunks_created":    m.get("chunks_created", 0),
#                 "retry_count":       m.get("retry_count", 0),
#                 "llm_calls":         m.get("llm_calls", 0),
#                 "retrieval_score":   m.get("retrieval_score", 0),
#                 "context_precision": m.get("context_precision", 0),
#                 "answer_grounding":  m.get("answer_grounding", 0),
#                 "recall_at_k":       m.get("recall_at_k", 0),
#             }
#             if m.get("type") == "summary":
#                 v2["parallel_workers"] = m.get("parallel_workers", 0)
#                 v2["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
#                 v2["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)
#             if m.get("type") == "qa":
#                 v2["chunks_retrieved"] = m.get("chunks_retrieved", 0)
#                 v2["decision_type"]    = m.get("decision_type", "accepted")
#                 v2["confidence_raw"]   = m.get("confidence_raw", 0.0)

#             return jsonify({
#                 "answer":     result["answer"],
#                 "metrics":    {**v1, **v2},
#                 "metrics_v1": v1,
#                 "metrics_v2": v2,
#             })

#         except Exception as e:
#             return jsonify({
#                 "answer":     f"Error: {str(e)}",
#                 "metrics":    {},
#                 "metrics_v1": {},
#                 "metrics_v2": {}
#             })

#     def is_refusal_answer(answer: str) -> bool:
#         if not answer or len(answer.strip()) < 3:
#             return True
#         prompt = (
#             f"Does this answer say that the requested information "
#             f"was not found, not present, or not available?\n\n"
#             f"Answer: {answer}\n\n"
#             f"Reply with YES or NO only."
#         )
#         try:
#             result = call_llama(prompt, num_ctx=256, temperature=0.0).strip().upper()
#             is_refusal = "YES" in result
#             print(f"[Refusal] '{answer[:60]}' → {'REFUSAL' if is_refusal else 'FACTUAL'}")
#             return is_refusal
#         except Exception:
#             return False

#     @app.route("/evaluate", methods=["POST"])
#     def evaluate():
#         try:
#             data        = request.json or {}
#             pdf_path    = data.get("pdf_path", "").strip()
#             run_desc    = data.get("run_description", f"run_{int(time.time())}")

#             if not pdf_path or not os.path.exists(pdf_path):
#                 return jsonify({"error": f"PDF not found: {pdf_path}"}), 404

#             pdf_name = os.path.basename(pdf_path)

#             PDF_DATASET_MAP = {
#             "Ethics of AI_new.pdf": r"C:\xampp\htdocs\genai-summary+qa - kowshik\docmind_rag\grounding_dataset_ethics.json",
#             "crop report for testin -genai.pdf": r"C:\xampp\htdocs\genai-summary+qa - kowshik\docmind_rag\grounding_dataset_crop.json",
#             "news -genai.pdf": r"C:\xampp\htdocs\genai-summary+qa - kowshik\docmind_rag\grounding_dataset_news.json"
#             }

#             dataset_path = PDF_DATASET_MAP.get(pdf_name)

#             if not dataset_path or not os.path.exists(dataset_path):
#                 return jsonify({"error": f"No dataset mapped for {pdf_name}"}), 404

#             with open(dataset_path, "r") as f:
#                 dataset = json.load(f)

#             questions = dataset.get("questions", [])
#             if not questions:
#                 return jsonify({"error": "No questions found in grounding_dataset.json"}), 400

#             results       = []
#             pass_count    = 0
#             partial_count = 0
#             fail_count    = 0

#             print(f"[Evaluate] Starting evaluation run: {run_desc}")
#             print(f"[Evaluate] PDF: {pdf_path}")
#             print(f"[Evaluate] Questions: {len(questions)}")

#             for q in questions:
#                 try:
#                     q_id       = q["id"]
#                     question   = q["question"]
#                     query_type = q.get("query_type", "FACTUAL_QA")
#                     keywords   = [kw.lower() for kw in q.get("expected_keywords", [])]
#                     threshold  = q.get("pass_threshold", 1)
#                     max_score  = q.get("max_score", len(keywords))
#                     is_halluc  = q.get("hallucination_test", False)
#                     is_roberta = q.get("roberta_test", False)

#                     if q.get("skip", False):
#                         print(f"[Evaluate] Q{q_id}: ⏭️ SKIPPED")
#                         continue

#                     print(f"[Evaluate] Q{q_id}: {question[:60]}...")

#                     initial_state: DocState = {
#                         "pdf_path":       pdf_path,
#                         "question":       question,
#                         "extracted_text": "",
#                         "chunks":         [],
#                         "summary_chunks": [],
#                         "faiss_index":    None,
#                         "answer":         "",
#                         "metrics":        {},
#                         "doc_type":       "general",
#                         "query_type":     "",
#                         "retry_count":    0,
#                         "start_time":     time.time(),
#                         "page_count":     0,
#                         "char_count":     0,
#                         "request_id":     ""
#                     }

#                     try:
#                         result     = workflow.invoke(initial_state)
#                         answer     = result.get("answer", "")
#                         metrics    = result.get("metrics", {})
#                         model_used = metrics.get("model_used", "unknown")
#                         recall_k   = metrics.get("recall_at_k", 0)
#                         grounding  = metrics.get("answer_grounding", 0)
#                         confidence = metrics.get("confidence_score", 0)
#                         print(f"[Evaluate] Q{q_id} actual answer: {answer[:300]}")
#                     except Exception as invoke_err:
#                         print(f"[Evaluate] Q{q_id} workflow error: {invoke_err}")
#                         answer     = f"ERROR: {str(invoke_err)}"
#                         metrics    = {}
#                         model_used = "error"
#                         recall_k   = 0
#                         grounding  = 0
#                         confidence = 0

#                     answer_lower = answer.lower()
#                     keyword_hits = 0

#                     # ✅ Detect short answers (IMPORTANT FIX)
#                     short_answer = len(answer.split()) <= 3

#                     hallucination_detected = False

#                     # ✅ Apply grounding ONLY for long answers
#                     if not is_halluc and grounding < 60 and not short_answer:
#                         hallucination_detected = True
#                         print(f"[Eval] ❌ Low grounding ({grounding:.1f}%) → hallucination detected")

#                     if is_halluc:
#                         found_safe = is_refusal_answer(answer)

#                         if found_safe:
#                             verdict = "PASS"
#                             keyword_hits = max_score
#                             hallucination_detected = False
#                         else:
#                             verdict = "FAIL"
#                             keyword_hits = 0
#                             hallucination_detected = True
#                             print(f"[Halluc] ❌ Answer is not a refusal — hallucination detected")

#                     else:
#                         # 🔥 FIRST → hallucination override
#                         if hallucination_detected:
#                             verdict = "FAIL"
#                             keyword_hits = 0
#                             print(f"[Eval] ❌ Forced FAIL due to low grounding")

#                         else:
#                             # 🔥 THEN → keyword evaluation
#                             keyword_hits = sum(1 for kw in keywords if kw in answer_lower)

#                             if keyword_hits >= threshold:
#                                 verdict = "PASS"
#                             elif keyword_hits >= max(1, threshold // 2):
#                                 verdict = "PARTIAL"
#                             else:
#                                 verdict = "FAIL"
#                     if verdict == "PASS":
#                         pass_count += 1
#                     elif verdict == "PARTIAL":
#                         partial_count += 1
#                     else:
#                         fail_count += 1

#                     result_entry = {
#                         "id":               q_id,
#                         "question":         question,
#                         "query_type":       query_type,
#                         "verdict":          verdict,
#                         "keyword_hits":     keyword_hits,
#                         "max_score":        max_score,
#                         "pass_threshold":   threshold,
#                         "model_used":       model_used,
#                         "recall_at_k":      round(recall_k, 1),
#                         "answer_grounding": round(grounding, 1),
#                         "confidence":       round(confidence, 1),
#                         "hallucination":    hallucination_detected,
#                         "roberta_test":     is_roberta,
#                         "actual_answer":    answer[:300],
#                         "tests":            q.get("tests", "")
#                     }
#                     results.append(result_entry)

#                     verdict_icon = "✅" if verdict == "PASS" else "⚠️" if verdict == "PARTIAL" else "❌"
#                     print(f"[Evaluate] Q{q_id} {verdict_icon} {verdict} | "
#                           f"hits={keyword_hits}/{max_score} | model={model_used} | "
#                           f"recall={recall_k:.1f}% | confidence={confidence:.1f}%")

#                 except Exception as q_err:
#                     print(f"[Evaluate] Q{q.get('id','?')} unexpected error: {q_err}")
#                     continue

#             total = len(results)
#             if total == 0:
#                 return jsonify({"error": "All questions were skipped or failed"}), 400

#             pass_rate = round(pass_count / total * 100, 1)
#             run_summary = {
#                 "run_id":          run_desc,
#                 "date":            time.strftime("%Y-%m-%d %H:%M:%S"),
#                 "pdf":             os.path.basename(pdf_path),
#                 "total_questions": total,
#                 "pass":            pass_count,
#                 "partial":         partial_count,
#                 "fail":            fail_count,
#                 "pass_rate":       pass_rate,
#                 "results":         results
#             }

#             results_path = os.path.join(os.path.dirname(__file__), "..", "grounding_results.json")
#             try:
#                 if os.path.exists(results_path):
#                     with open(results_path, "r") as f:
#                         existing = json.load(f)
#                 else:
#                     existing = {"runs": []}
#                 existing["runs"].append(run_summary)
#                 with open(results_path, "w") as f:
#                     json.dump(existing, f, indent=2)
#                 print(f"[Evaluate] Results saved to grounding_results.json")
#             except Exception as save_err:
#                 print(f"[Evaluate] Warning: could not save results: {save_err}")

#             print(f"[Evaluate] ✅ Done | Pass={pass_count} Partial={partial_count} "
#                   f"Fail={fail_count} | Pass rate={pass_rate}%")

#             return jsonify({
#                 "run_id":          run_desc,
#                 "total_questions": total,
#                 "pass":            pass_count,
#                 "partial":         partial_count,
#                 "fail":            fail_count,
#                 "pass_rate":       f"{pass_rate}%",
#                 "results":         results
#             })

#         except Exception as fatal_err:
#             print(f"[Evaluate] ❌ Fatal error: {fatal_err}")
#             import traceback
#             traceback.print_exc()
#             return jsonify({"error": f"Fatal evaluation error: {str(fatal_err)}"}), 500







































































# #single pdf 
# import os
# import json
# import time

# from flask import request, jsonify

# from docmind_rag.models.llm import call_llama, _ttft_tracker
# from docmind_rag.core.state import DocState
# from docmind_rag.services.extraction import _extraction_cache
# from docmind_rag.services.chunking import _summary_cache
# from docmind_rag.models.embeddings import _faiss_cache
# from docmind_rag.events.events import emit_event, cleanup_queue
# from docmind_rag.config.settings import OLLAMA_MODEL, device

# def register_routes(app, workflow):
#     """Register all Flask routes onto the given app instance."""

#     # ============================================================
#     # FLASK ROUTES
#     # ============================================================
#     @app.route("/health", methods=["GET"])
#     def health():
#         return jsonify({
#             "status": "ok",
#             "model":  OLLAMA_MODEL,
#             "device": "cuda" if device == 0 else "cpu",
#             "faiss_cached":     len(_faiss_cache),
#             "extracted_pdfs":   len(_extraction_cache),
#             "summaries_cached": len(_summary_cache),
#         })

#     @app.route("/process", methods=["POST"])
#     def process():
#         data = request.json
#         if not data:
#             return jsonify({"error": "Request body must be JSON"}), 400

#         pdf_path   = data.get("pdf_path", "").strip()
#         question   = data.get("question", "").strip()
#         request_id = data.get("request_id", "")

#         if not pdf_path:
#             return jsonify({"error": "Missing required field: pdf_path"}), 400
#         if not question:
#             return jsonify({"error": "Missing required field: question"}), 400
#         if len(question) > 2000:
#             return jsonify({"error": "Question too long (max 2000 chars)"}), 400
#         if not os.path.exists(pdf_path):
#             return jsonify({"error": f"PDF not found: {pdf_path}"}), 404
#         if not pdf_path.lower().endswith(".pdf"):
#             return jsonify({"error": "Only PDF files are supported"}), 400

#         emit_event(request_id, "extract_start", "📄 Extracting PDF text...")

#         initial_state: DocState = {
#             "pdf_path":       pdf_path,
#             "question":       question,
#             "extracted_text": "",
#             "chunks":         [],
#             "summary_chunks": [],
#             "faiss_index":    None,
#             "answer":         "",
#             "metrics":        {},
#             "doc_type":       "general",
#             "query_type":     "",
#             "retry_count":    0,
#             "start_time":     time.time(),
#             "page_count":     0,
#             "char_count":     0,
#             "request_id":     request_id
#         }

#         try:
#             request_start = time.time()
#             emit_event(request_id, "workflow_start", "⚙️ LangGraph workflow started...")
#             result      = workflow.invoke(initial_state)
#             m           = result["metrics"]
#             total_e2e   = round(time.time() - request_start, 2)
#             m["e2e_latency_sec"] = total_e2e

#             real_ttft = _ttft_tracker.pop(request_id, None)
#             if real_ttft is not None:
#                 m["ttft_sec"] = real_ttft
#             else:
#                 m["ttft_sec"] = m.get("ttft_sec", total_e2e)

#             print(f"[LATENCY] E2E={total_e2e}s | TTFT={m['ttft_sec']}s | TPS={m.get('tps',0)}")
#             if m.get("type") == "qa":
#                 print(f"[QUALITY] Grounding={m.get('answer_grounding',0)}% | "
#                       f"Retrieval={m.get('retrieval_score',0)}% | "
#                       f"Precision={m.get('context_precision',0)}% | "
#                       f"Recall={m.get('recall_at_k',0)}% | "
#                       f"Confidence={m.get('confidence_score',0)}%")

#             emit_event(request_id, "done", "✅ Processing complete!")
#             cleanup_queue(request_id)

#             v1 = {
#                 "type":                 m.get("type", ""),
#                 "response_time_sec":    round(m.get("response_time_sec", 0), 2),
#                 "extraction_time_sec":  round(m.get("extraction_time_sec", 0), 2),
#                 "pages_processed":      m.get("pages_processed", 0),
#                 "characters_processed": m.get("characters_processed", 0),
#                 "words_processed":      m.get("words_processed", 0),
#             }
#             if m.get("type") == "summary":
#                 v1["summary_time_sec"]     = round(m.get("summary_time_sec", 0), 2)
#                 v1["summary_length_words"] = m.get("summary_length_words", 0)
#             if m.get("type") == "qa":
#                 v1["qa_time_sec"]      = round(m.get("qa_time_sec", 0), 2)
#                 v1["confidence_score"] = round(m.get("confidence_score", 0), 2)

#             v2 = {
#                 "ttft_sec":          round(m.get("ttft_sec", 0), 2),
#                 "e2e_latency_sec":   round(m.get("e2e_latency_sec", 0), 2),
#                 "tps":               m.get("tps", 0),
#                 "doc_type":          m.get("doc_type", "general"),
#                 "query_type":        m.get("query_type", ""),
#                 "model_used":        m.get("model_used", ""),
#                 "chunks_created":    m.get("chunks_created", 0),
#                 "retry_count":       m.get("retry_count", 0),
#                 "llm_calls":         m.get("llm_calls", 0),
#                 "retrieval_score":   m.get("retrieval_score", 0),
#                 "context_precision": m.get("context_precision", 0),
#                 "answer_grounding":  m.get("answer_grounding", 0),
#                 "recall_at_k":       m.get("recall_at_k", 0),
#             }
#             if m.get("type") == "summary":
#                 v2["parallel_workers"] = m.get("parallel_workers", 0)
#                 v2["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
#                 v2["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)
#             if m.get("type") == "qa":
#                 v2["chunks_retrieved"] = m.get("chunks_retrieved", 0)
#                 v2["decision_type"]    = m.get("decision_type", "accepted")
#                 v2["confidence_raw"]   = m.get("confidence_raw", 0.0)

#             return jsonify({
#                 "answer":     result["answer"],
#                 "metrics":    {**v1, **v2},
#                 "metrics_v1": v1,
#                 "metrics_v2": v2,
#             })

#         except Exception as e:
#             return jsonify({
#                 "answer":     f"Error: {str(e)}",
#                 "metrics":    {},
#                 "metrics_v1": {},
#                 "metrics_v2": {}
#             })

#     def is_refusal_answer(answer: str) -> bool:
#         if not answer or len(answer.strip()) < 3:
#             return True
#         prompt = (
#             f"Does this answer say that the requested information "
#             f"was not found, not present, or not available?\n\n"
#             f"Answer: {answer}\n\n"
#             f"Reply with YES or NO only."
#         )
#         try:
#             result = call_llama(prompt, num_ctx=256, temperature=0.0).strip().upper()
#             is_refusal = "YES" in result
#             print(f"[Refusal] '{answer[:60]}' → {'REFUSAL' if is_refusal else 'FACTUAL'}")
#             return is_refusal
#         except Exception:
#             return False

#     @app.route("/evaluate", methods=["POST"])
#     def evaluate():
#         try:
#             data        = request.json or {}
#             pdf_path    = data.get("pdf_path", "").strip()
#             run_desc    = data.get("run_description", f"run_{int(time.time())}")

#             if not pdf_path or not os.path.exists(pdf_path):
#                 return jsonify({"error": f"PDF not found: {pdf_path}"}), 404

#             dataset_path = os.path.join(os.path.dirname(__file__), "..", "grounding_dataset.json")
#             if not os.path.exists(dataset_path):
#                 return jsonify({"error": "grounding_dataset.json not found next to app.py"}), 404

#             with open(dataset_path, "r") as f:
#                 dataset = json.load(f)

#             questions = dataset.get("questions", [])
#             if not questions:
#                 return jsonify({"error": "No questions found in grounding_dataset.json"}), 400

#             results       = []
#             pass_count    = 0
#             partial_count = 0
#             fail_count    = 0

#             print(f"[Evaluate] Starting evaluation run: {run_desc}")
#             print(f"[Evaluate] PDF: {pdf_path}")
#             print(f"[Evaluate] Questions: {len(questions)}")

#             for q in questions:
#                 try:
#                     q_id       = q["id"]
#                     question   = q["question"]
#                     query_type = q.get("query_type", "FACTUAL_QA")
#                     keywords   = [kw.lower() for kw in q.get("expected_keywords", [])]
#                     threshold  = q.get("pass_threshold", 1)
#                     max_score  = q.get("max_score", len(keywords))
#                     is_halluc  = q.get("hallucination_test", False)
#                     is_roberta = q.get("roberta_test", False)

#                     if q.get("skip", False):
#                         print(f"[Evaluate] Q{q_id}: ⏭️ SKIPPED")
#                         continue

#                     print(f"[Evaluate] Q{q_id}: {question[:60]}...")

#                     initial_state: DocState = {
#                         "pdf_path":       pdf_path,
#                         "question":       question,
#                         "extracted_text": "",
#                         "chunks":         [],
#                         "summary_chunks": [],
#                         "faiss_index":    None,
#                         "answer":         "",
#                         "metrics":        {},
#                         "doc_type":       "general",
#                         "query_type":     "",
#                         "retry_count":    0,
#                         "start_time":     time.time(),
#                         "page_count":     0,
#                         "char_count":     0,
#                         "request_id":     ""
#                     }

#                     try:
#                         result     = workflow.invoke(initial_state)
#                         answer     = result.get("answer", "")
#                         metrics    = result.get("metrics", {})
#                         model_used = metrics.get("model_used", "unknown")
#                         recall_k   = metrics.get("recall_at_k", 0)
#                         grounding  = metrics.get("answer_grounding", 0)
#                         confidence = metrics.get("confidence_score", 0)
#                         print(f"[Evaluate] Q{q_id} actual answer: {answer[:300]}")
#                     except Exception as invoke_err:
#                         print(f"[Evaluate] Q{q_id} workflow error: {invoke_err}")
#                         answer     = f"ERROR: {str(invoke_err)}"
#                         metrics    = {}
#                         model_used = "error"
#                         recall_k   = 0
#                         grounding  = 0
#                         confidence = 0

#                     answer_lower = answer.lower()
#                     keyword_hits = 0

#                     # ✅ Detect short answers (IMPORTANT FIX)
#                     short_answer = len(answer.split()) <= 3

#                     hallucination_detected = False

#                     # ✅ Apply grounding ONLY for long answers
#                     if not is_halluc and grounding < 60 and not short_answer:
#                         hallucination_detected = True
#                         print(f"[Eval] ❌ Low grounding ({grounding:.1f}%) → hallucination detected")

#                     if is_halluc:
#                         found_safe = is_refusal_answer(answer)

#                         if found_safe:
#                             verdict = "PASS"
#                             keyword_hits = max_score
#                             hallucination_detected = False
#                         else:
#                             verdict = "FAIL"
#                             keyword_hits = 0
#                             hallucination_detected = True
#                             print(f"[Halluc] ❌ Answer is not a refusal — hallucination detected")

#                     else:
#                         # 🔥 FIRST → hallucination override
#                         if hallucination_detected:
#                             verdict = "FAIL"
#                             keyword_hits = 0
#                             print(f"[Eval] ❌ Forced FAIL due to low grounding")

#                         else:
#                             # 🔥 THEN → keyword evaluation
#                             keyword_hits = sum(1 for kw in keywords if kw in answer_lower)

#                             if keyword_hits >= threshold:
#                                 verdict = "PASS"
#                             elif keyword_hits >= max(1, threshold // 2):
#                                 verdict = "PARTIAL"
#                             else:
#                                 verdict = "FAIL"
#                     if verdict == "PASS":
#                         pass_count += 1
#                     elif verdict == "PARTIAL":
#                         partial_count += 1
#                     else:
#                         fail_count += 1

#                     result_entry = {
#                         "id":               q_id,
#                         "question":         question,
#                         "query_type":       query_type,
#                         "verdict":          verdict,
#                         "keyword_hits":     keyword_hits,
#                         "max_score":        max_score,
#                         "pass_threshold":   threshold,
#                         "model_used":       model_used,
#                         "recall_at_k":      round(recall_k, 1),
#                         "answer_grounding": round(grounding, 1),
#                         "confidence":       round(confidence, 1),
#                         "hallucination":    hallucination_detected,
#                         "roberta_test":     is_roberta,
#                         "actual_answer":    answer[:300],
#                         "tests":            q.get("tests", "")
#                     }
#                     results.append(result_entry)

#                     verdict_icon = "✅" if verdict == "PASS" else "⚠️" if verdict == "PARTIAL" else "❌"
#                     print(f"[Evaluate] Q{q_id} {verdict_icon} {verdict} | "
#                           f"hits={keyword_hits}/{max_score} | model={model_used} | "
#                           f"recall={recall_k:.1f}% | confidence={confidence:.1f}%")

#                 except Exception as q_err:
#                     print(f"[Evaluate] Q{q.get('id','?')} unexpected error: {q_err}")
#                     continue

#             total = len(results)
#             if total == 0:
#                 return jsonify({"error": "All questions were skipped or failed"}), 400

#             pass_rate = round(pass_count / total * 100, 1)
#             run_summary = {
#                 "run_id":          run_desc,
#                 "date":            time.strftime("%Y-%m-%d %H:%M:%S"),
#                 "pdf":             os.path.basename(pdf_path),
#                 "total_questions": total,
#                 "pass":            pass_count,
#                 "partial":         partial_count,
#                 "fail":            fail_count,
#                 "pass_rate":       pass_rate,
#                 "results":         results
#             }

#             results_path = os.path.join(os.path.dirname(__file__), "..", "grounding_results.json")
#             try:
#                 if os.path.exists(results_path):
#                     with open(results_path, "r") as f:
#                         existing = json.load(f)
#                 else:
#                     existing = {"runs": []}
#                 existing["runs"].append(run_summary)
#                 with open(results_path, "w") as f:
#                     json.dump(existing, f, indent=2)
#                 print(f"[Evaluate] Results saved to grounding_results.json")
#             except Exception as save_err:
#                 print(f"[Evaluate] Warning: could not save results: {save_err}")

#             print(f"[Evaluate] ✅ Done | Pass={pass_count} Partial={partial_count} "
#                   f"Fail={fail_count} | Pass rate={pass_rate}%")

#             return jsonify({
#                 "run_id":          run_desc,
#                 "total_questions": total,
#                 "pass":            pass_count,
#                 "partial":         partial_count,
#                 "fail":            fail_count,
#                 "pass_rate":       f"{pass_rate}%",
#                 "results":         results
#             })

#         except Exception as fatal_err:
#             print(f"[Evaluate] ❌ Fatal error: {fatal_err}")
#             import traceback
#             traceback.print_exc()
#             return jsonify({"error": f"Fatal evaluation error: {str(fatal_err)}"}), 500
































# works perfectly want  to make it  geenric - lap claude it does not have any  changes much just adding print statedment in last line of evaluate for debugging purpose 
# import os
# import json
# import time

# from flask import request, jsonify

# from docmind_rag.models.llm import call_llama, _ttft_tracker
# from docmind_rag.core.state import DocState
# from docmind_rag.services.extraction import _extraction_cache
# from docmind_rag.services.chunking import _summary_cache
# from docmind_rag.models.embeddings import _faiss_cache
# from docmind_rag.events.events import emit_event, cleanup_queue
# from docmind_rag.config.settings import OLLAMA_MODEL, device

# def register_routes(app, workflow):
#     """Register all Flask routes onto the given app instance."""

#     # ============================================================
#     # FLASK ROUTES
#     # ============================================================
#     @app.route("/health", methods=["GET"])
#     def health():
#         return jsonify({
#             "status": "ok",
#             "model":  OLLAMA_MODEL,
#             "device": "cuda" if device == 0 else "cpu",
#             "faiss_cached":     len(_faiss_cache),
#             "extracted_pdfs":   len(_extraction_cache),
#             "summaries_cached": len(_summary_cache),
#         })

#     @app.route("/process", methods=["POST"])
#     def process():
#         data = request.json
#         if not data:
#             return jsonify({"error": "Request body must be JSON"}), 400

#         pdf_path   = data.get("pdf_path", "").strip()
#         question   = data.get("question", "").strip()
#         request_id = data.get("request_id", "")

#         if not pdf_path:
#             return jsonify({"error": "Missing required field: pdf_path"}), 400
#         if not question:
#             return jsonify({"error": "Missing required field: question"}), 400
#         if len(question) > 2000:
#             return jsonify({"error": "Question too long (max 2000 chars)"}), 400
#         if not os.path.exists(pdf_path):
#             return jsonify({"error": f"PDF not found: {pdf_path}"}), 404
#         if not pdf_path.lower().endswith(".pdf"):
#             return jsonify({"error": "Only PDF files are supported"}), 400

#         emit_event(request_id, "extract_start", "📄 Extracting PDF text...")

#         initial_state: DocState = {
#             "pdf_path":       pdf_path,
#             "question":       question,
#             "extracted_text": "",
#             "chunks":         [],
#             "summary_chunks": [],
#             "faiss_index":    None,
#             "answer":         "",
#             "metrics":        {},
#             "doc_type":       "general",
#             "query_type":     "",
#             "retry_count":    0,
#             "start_time":     time.time(),
#             "page_count":     0,
#             "char_count":     0,
#             "request_id":     request_id
#         }

#         try:
#             request_start = time.time()
#             emit_event(request_id, "workflow_start", "⚙️ LangGraph workflow started...")
#             result      = workflow.invoke(initial_state)
#             m           = result["metrics"]
#             total_e2e   = round(time.time() - request_start, 2)
#             m["e2e_latency_sec"] = total_e2e

#             real_ttft = _ttft_tracker.pop(request_id, None)
#             if real_ttft is not None:
#                 m["ttft_sec"] = real_ttft
#             else:
#                 m["ttft_sec"] = m.get("ttft_sec", total_e2e)

#             print(f"[LATENCY] E2E={total_e2e}s | TTFT={m['ttft_sec']}s | TPS={m.get('tps',0)}")
#             if m.get("type") == "qa":
#                 print(f"[QUALITY] Grounding={m.get('answer_grounding',0)}% | "
#                       f"Retrieval={m.get('retrieval_score',0)}% | "
#                       f"Precision={m.get('context_precision',0)}% | "
#                       f"Recall={m.get('recall_at_k',0)}% | "
#                       f"Confidence={m.get('confidence_score',0)}%")

#             emit_event(request_id, "done", "✅ Processing complete!")
#             cleanup_queue(request_id)

#             v1 = {
#                 "type":                 m.get("type", ""),
#                 "response_time_sec":    round(m.get("response_time_sec", 0), 2),
#                 "extraction_time_sec":  round(m.get("extraction_time_sec", 0), 2),
#                 "pages_processed":      m.get("pages_processed", 0),
#                 "characters_processed": m.get("characters_processed", 0),
#                 "words_processed":      m.get("words_processed", 0),
#             }
#             if m.get("type") == "summary":
#                 v1["summary_time_sec"]     = round(m.get("summary_time_sec", 0), 2)
#                 v1["summary_length_words"] = m.get("summary_length_words", 0)
#             if m.get("type") == "qa":
#                 v1["qa_time_sec"]      = round(m.get("qa_time_sec", 0), 2)
#                 v1["confidence_score"] = round(m.get("confidence_score", 0), 2)

#             v2 = {
#                 "ttft_sec":          round(m.get("ttft_sec", 0), 2),
#                 "e2e_latency_sec":   round(m.get("e2e_latency_sec", 0), 2),
#                 "tps":               m.get("tps", 0),
#                 "doc_type":          m.get("doc_type", "general"),
#                 "query_type":        m.get("query_type", ""),
#                 "model_used":        m.get("model_used", ""),
#                 "chunks_created":    m.get("chunks_created", 0),
#                 "retry_count":       m.get("retry_count", 0),
#                 "llm_calls":         m.get("llm_calls", 0),
#                 "retrieval_score":   m.get("retrieval_score", 0),
#                 "context_precision": m.get("context_precision", 0),
#                 "answer_grounding":  m.get("answer_grounding", 0),
#                 "recall_at_k":       m.get("recall_at_k", 0),
#             }
#             if m.get("type") == "summary":
#                 v2["parallel_workers"] = m.get("parallel_workers", 0)
#                 v2["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
#                 v2["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)
#             if m.get("type") == "qa":
#                 v2["chunks_retrieved"] = m.get("chunks_retrieved", 0)
#                 v2["decision_type"]    = m.get("decision_type", "accepted")
#                 v2["confidence_raw"]   = m.get("confidence_raw", 0.0)

#             return jsonify({
#                 "answer":     result["answer"],
#                 "metrics":    {**v1, **v2},
#                 "metrics_v1": v1,
#                 "metrics_v2": v2,
#             })

#         except Exception as e:
#             return jsonify({
#                 "answer":     f"Error: {str(e)}",
#                 "metrics":    {},
#                 "metrics_v1": {},
#                 "metrics_v2": {}
#             })

#     def is_refusal_answer(answer: str) -> bool:
#         if not answer or len(answer.strip()) < 3:
#             return True
#         prompt = (
#             f"Does this answer say that the requested information "
#             f"was not found, not present, or not available?\n\n"
#             f"Answer: {answer}\n\n"
#             f"Reply with YES or NO only."
#         )
#         try:
#             result = call_llama(prompt, num_ctx=256, temperature=0.0).strip().upper()
#             is_refusal = "YES" in result
#             print(f"[Refusal] '{answer[:60]}' → {'REFUSAL' if is_refusal else 'FACTUAL'}")
#             return is_refusal
#         except Exception:
#             return False

#     @app.route("/evaluate", methods=["POST"])
#     def evaluate():
#         try:
#             data        = request.json or {}
#             pdf_path    = data.get("pdf_path", "").strip()
#             run_desc    = data.get("run_description", f"run_{int(time.time())}")

#             if not pdf_path or not os.path.exists(pdf_path):
#                 return jsonify({"error": f"PDF not found: {pdf_path}"}), 404

#             dataset_path = os.path.join(os.path.dirname(__file__), "..", "grounding_dataset.json")
#             if not os.path.exists(dataset_path):
#                 return jsonify({"error": "grounding_dataset.json not found next to app.py"}), 404

#             with open(dataset_path, "r") as f:
#                 dataset = json.load(f)

#             questions = dataset.get("questions", [])
#             if not questions:
#                 return jsonify({"error": "No questions found in grounding_dataset.json"}), 400

#             results       = []
#             pass_count    = 0
#             partial_count = 0
#             fail_count    = 0

#             print(f"[Evaluate] Starting evaluation run: {run_desc}")
#             print(f"[Evaluate] PDF: {pdf_path}")
#             print(f"[Evaluate] Questions: {len(questions)}")

#             for q in questions:
#                 try:
#                     q_id       = q["id"]
#                     question   = q["question"]
#                     query_type = q.get("query_type", "FACTUAL_QA")
#                     keywords   = [kw.lower() for kw in q.get("expected_keywords", [])]
#                     threshold  = q.get("pass_threshold", 1)
#                     max_score  = q.get("max_score", len(keywords))
#                     is_halluc  = q.get("hallucination_test", False)
#                     is_roberta = q.get("roberta_test", False)

#                     if q.get("skip", False):
#                         print(f"[Evaluate] Q{q_id}: ⏭️ SKIPPED")
#                         continue

#                     print(f"[Evaluate] Q{q_id}: {question[:60]}...")

#                     initial_state: DocState = {
#                         "pdf_path":       pdf_path,
#                         "question":       question,
#                         "extracted_text": "",
#                         "chunks":         [],
#                         "summary_chunks": [],
#                         "faiss_index":    None,
#                         "answer":         "",
#                         "metrics":        {},
#                         "doc_type":       "general",
#                         "query_type":     "",
#                         "retry_count":    0,
#                         "start_time":     time.time(),
#                         "page_count":     0,
#                         "char_count":     0,
#                         "request_id":     ""
#                     }

#                     try:
#                         result     = workflow.invoke(initial_state)
#                         answer     = result.get("answer", "")
#                         metrics    = result.get("metrics", {})
#                         model_used = metrics.get("model_used", "unknown")
#                         recall_k   = metrics.get("recall_at_k", 0)
#                         grounding  = metrics.get("answer_grounding", 0)
#                         confidence = metrics.get("confidence_score", 0)
#                     except Exception as invoke_err:
#                         print(f"[Evaluate] Q{q_id} workflow error: {invoke_err}")
#                         answer     = f"ERROR: {str(invoke_err)}"
#                         metrics    = {}
#                         model_used = "error"
#                         recall_k   = 0
#                         grounding  = 0
#                         confidence = 0

#                     answer_lower = answer.lower()

#                     if is_halluc:
#                         found_safe = is_refusal_answer(answer)
#                         if found_safe:
#                             verdict      = "PASS"
#                             keyword_hits = max_score
#                             hallucination_detected = False
#                         else:
#                             invented = any(kw in answer_lower for kw in keywords)
#                             verdict      = "FAIL"
#                             keyword_hits = 0
#                             hallucination_detected = True
#                             print(f"[Halluc] ❌ Answer is not a refusal — hallucination detected")
#                             print(f"[Halluc] Answer: '{answer[:100]}'")
#                     else:
#                         keyword_hits = sum(1 for kw in keywords if kw in answer_lower)
#                         hallucination_detected = False
#                         if keyword_hits >= threshold:
#                             verdict = "PASS"
#                         elif keyword_hits >= max(1, threshold // 2):
#                             verdict = "PARTIAL"
#                         else:
#                             verdict = "FAIL"

#                     if verdict == "PASS":
#                         pass_count += 1
#                     elif verdict == "PARTIAL":
#                         partial_count += 1
#                     else:
#                         fail_count += 1

#                     result_entry = {
#                         "id":               q_id,
#                         "question":         question,
#                         "query_type":       query_type,
#                         "verdict":          verdict,
#                         "keyword_hits":     keyword_hits,
#                         "max_score":        max_score,
#                         "pass_threshold":   threshold,
#                         "model_used":       model_used,
#                         "recall_at_k":      round(recall_k, 1),
#                         "answer_grounding": round(grounding, 1),
#                         "confidence":       round(confidence, 1),
#                         "hallucination":    hallucination_detected,
#                         "roberta_test":     is_roberta,
#                         "actual_answer":    answer[:300],
#                         "tests":            q.get("tests", "")
#                     }
#                     results.append(result_entry)

#                     verdict_icon = "✅" if verdict == "PASS" else "⚠️" if verdict == "PARTIAL" else "❌"
#                     print(f"[Evaluate] Q{q_id} {verdict_icon} {verdict} | "
#                           f"hits={keyword_hits}/{max_score} | model={model_used} | "
#                           f"recall={recall_k:.1f}% | confidence={confidence:.1f}%")

#                 except Exception as q_err:
#                     print(f"[Evaluate] Q{q.get('id','?')} unexpected error: {q_err}")
#                     continue

#             total = len(results)
#             if total == 0:
#                 return jsonify({"error": "All questions were skipped or failed"}), 400

#             pass_rate = round(pass_count / total * 100, 1)
#             run_summary = {
#                 "run_id":          run_desc,
#                 "date":            time.strftime("%Y-%m-%d %H:%M:%S"),
#                 "pdf":             os.path.basename(pdf_path),
#                 "total_questions": total,
#                 "pass":            pass_count,
#                 "partial":         partial_count,
#                 "fail":            fail_count,
#                 "pass_rate":       pass_rate,
#                 "results":         results
#             }

#             results_path = os.path.join(os.path.dirname(__file__), "..", "grounding_results.json")
#             try:
#                 if os.path.exists(results_path):
#                     with open(results_path, "r") as f:
#                         existing = json.load(f)
#                 else:
#                     existing = {"runs": []}
#                 existing["runs"].append(run_summary)
#                 with open(results_path, "w") as f:
#                     json.dump(existing, f, indent=2)
#                 print(f"[Evaluate] Results saved to grounding_results.json")
#             except Exception as save_err:
#                 print(f"[Evaluate] Warning: could not save results: {save_err}")

#             print(f"[Evaluate] ✅ Done | Pass={pass_count} Partial={partial_count} "
#                   f"Fail={fail_count} | Pass rate={pass_rate}%")

#             return jsonify({
#                 "run_id":          run_desc,
#                 "total_questions": total,
#                 "pass":            pass_count,
#                 "partial":         partial_count,
#                 "fail":            fail_count,
#                 "pass_rate":       f"{pass_rate}%",
#                 "results":         results
#             })

#         except Exception as fatal_err:
#             print(f"[Evaluate] ❌ Fatal error: {fatal_err}")
#             import traceback
#             traceback.print_exc()
#             return jsonify({"error": f"Fatal evaluation error: {str(fatal_err)}"}), 500