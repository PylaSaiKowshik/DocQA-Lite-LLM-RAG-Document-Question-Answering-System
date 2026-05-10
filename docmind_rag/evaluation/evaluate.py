"""
DocMind — Multi-Run Stability Evaluation
Runs /evaluate N times and reports:
  - Per-question accuracy across runs
  - Overall accuracy
  - Hallucination rate
  - Instability rate (questions that change verdict across runs)
  - All results saved to stability_results.json (append, never overwrite)
  - Each session named by user (auto-increments if no name given)
"""

import requests
import json
import time
import sys
import os
from collections import defaultdict

# ── CONFIG ──────────────────────────────────────────────────
FLASK_URL        = "http://127.0.0.1:5000/evaluate"

BASE_DIR = os.path.dirname(__file__)
PDF_PATH = None   # ✅ add this line
PDF_PATHS = [
    r"C:\Users\Rhearitu\Downloads\Ethics of AI_new.pdf",
    r"C:\Users\Rhearitu\Downloads\crop report for testin -genai.pdf",
    r"C:\Users\Rhearitu\Downloads\news -genai.pdf"
]
NUM_RUNS         = 3
RESULTS_FILE = os.path.join(BASE_DIR, "results", "stability_results.json")
# DATASET_FILE = os.path.join(BASE_DIR, "grounding_dataset.json")

# ────────────────────────────────────────────────────────────


def load_existing_results() -> dict:
    """Load existing results file or create fresh structure."""
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r") as f:
                data = json.load(f)
            # Ensure sessions key exists
            if "sessions" not in data:
                data["sessions"] = []
            return data
        except Exception as e:
            print(f"[Warning] Could not read {RESULTS_FILE}: {e} — starting fresh")
    return {"sessions": []}


def get_next_session_number(existing: dict) -> int:
    """Auto-increment session number based on existing sessions."""
    return len(existing.get("sessions", [])) + 1


def get_session_name(next_num: int) -> str:
    """Prompt user for session name. Auto-names if empty."""
    auto_name = f"Test {next_num}"
    try:
        user_input = input(
            f"\nEnter session name (press Enter for auto '{auto_name}'): "
        ).strip()
        return user_input if user_input else auto_name
    except (EOFError, KeyboardInterrupt):
        return auto_name
def load_dataset_meta(pdf_path: str) -> dict:
    """
    Load dataset corresponding to the given PDF.
    Returns dict keyed by question id.
    """
    meta = {}

    # 🔥 Map PDF → dataset file
    pdf_name = os.path.basename(pdf_path)
    PDF_DATASET_MAP = {
    "Ethics of AI_new.pdf": r"C:\xampp\htdocs\genai-summary+qa - kowshik\docmind_rag\grounding_dataset_ethics.json",
    "crop report for testin -genai.pdf": r"C:\xampp\htdocs\genai-summary+qa - kowshik\docmind_rag\grounding_dataset_crop.json",
    "news -genai.pdf": r"C:\xampp\htdocs\genai-summary+qa - kowshik\docmind_rag\grounding_dataset_news.json"
    }
    # BASE_DIR = os.path.dirname(os.path.dirname(__file__))

    # PDF_DATASET_MAP = {
    # "Ethics of AI_new.pdf": os.path.join(BASE_DIR, "grounding_dataset_ethics.json"),
    # "crop report for testin -genai.pdf": os.path.join(BASE_DIR, "grounding_dataset_crop.json"),
    # "news -genai.pdf": os.path.join(BASE_DIR, "grounding_dataset_news.json")
    # }

# 
    dataset_file = PDF_DATASET_MAP.get(pdf_name)

    if not dataset_file:
        print(f"[Warning] No dataset mapped for {pdf_name}")
        return meta

    dataset_path = dataset_file

    if not os.path.exists(dataset_path):
        print(f"[Warning] {dataset_path} not found")
        return meta

    try:
        with open(dataset_path, "r") as f:
            dataset = json.load(f)

        for q in dataset.get("questions", []):
            meta[q["id"]] = {
                   "question":           q.get("question", ""),
                   "answer":             q.get("answer", ""),   # 🔥 ADD THIS
                "hallucination_test": q.get("hallucination_test", False),
                "skip":               q.get("skip", False),
            }

    except Exception as e:
        print(f"[Warning] Could not read dataset meta: {e}")

    return meta

def run_evaluate(run_num: int, session_name: str) -> dict:
    """Call Flask /evaluate endpoint for one run."""
    payload = {
        "pdf_path":        PDF_PATH,
        "run_description": f"{session_name}_run_{run_num}"
    }
    print(f"\n{'='*60}")
    print(f"  RUN {run_num} / {NUM_RUNS}")
    print(f"{'='*60}")
    try:
        resp = requests.post(FLASK_URL, json=payload, timeout=7200)
        if resp.status_code != 200:
            print(f"[ERROR] Run {run_num} returned HTTP {resp.status_code}")
            print(f"[ERROR] Response: {resp.text[:500]}")
            return {}
        try:
            data = resp.json()
        except Exception as json_err:
            print(f"[ERROR] Run {run_num} — invalid JSON response: {json_err}")
            print(f"[ERROR] Raw response: {resp.text[:500]}")
            return {}
        return data
    except requests.exceptions.Timeout:
        print(f"[ERROR] Run {run_num} timed out")
        return {}
    except Exception as e:
        print(f"[ERROR] Run {run_num} failed: {e}")
        return {}

def build_run_record(run_num: int, result: dict, elapsed: float) -> dict:
    """
    Build a structured record for a single run including
    all per-question details exactly as received from Flask.
    """
    questions_detail = []
    for r in result.get("results", []):
        questions_detail.append({
            "id":               r.get("id"),
            "question":         r.get("question", ""),
            "verdict":          r.get("verdict", "FAIL"),
            "keyword_hits":     r.get("keyword_hits", 0),
            "max_score":        r.get("max_score", 0),
            "pass_threshold":   r.get("pass_threshold", 1),
            "model_used":       r.get("model_used", "unknown"),
            "recall_at_k":      r.get("recall_at_k", 0.0),
            "answer_grounding": r.get("answer_grounding", 0.0),
            "confidence":       r.get("confidence", 0.0),
            "hallucination":    r.get("hallucination", False),
            "actual_answer":    r.get("actual_answer", ""),
            "tests":            r.get("tests", ""),
        })

    total_q = (
        result.get("pass", 0)
        + result.get("partial", 0)
        + result.get("fail", 0)
    )

    return {
        "run_number": run_num,
        "pdf":        os.path.basename(PDF_PATH),  # ✅ FIXED
        "pass":       result.get("pass", 0),
        "partial":    result.get("partial", 0),
        "fail":       result.get("fail", 0),
        "total":      total_q,
        "pass_rate":  float(str(result.get("pass_rate", "0")).replace("%", "")),
        "time_sec":   elapsed,
        "questions":  questions_detail,
    }

def compute_stability_report(all_run_records: list,
                              dataset_meta: dict,
                              run_times: list,
                              total_runs: int) -> dict:
    """
    Compute full stability report from all run records.
    Returns structured dict matching console output.
    """
    q_verdicts  = defaultdict(list)
    q_questions = {}
    q_halluc    = {}

    for run_record in all_run_records:
        for q in run_record["questions"]:
            qid = q["id"]
            q_verdicts[qid].append(q["verdict"])
            q_questions[qid] = q["question"]
            # Use dataset meta for hallucination_test (authoritative)
            # Fall back to response flag if dataset not available
            if qid in dataset_meta:
                q_halluc[qid] = dataset_meta[qid]["hallucination_test"]
            else:
                q_halluc[qid] = q.get("hallucination", False)

    total_questions = len(q_verdicts)
    total_answers   = total_questions * total_runs
    total_pass      = sum(v.count("PASS")    for v in q_verdicts.values())
    total_partial   = sum(v.count("PARTIAL") for v in q_verdicts.values())
    total_fail      = sum(v.count("FAIL")    for v in q_verdicts.values())

    overall_accuracy = round(total_pass    / total_answers * 100, 1)
    partial_rate     = round(total_partial / total_answers * 100, 1)
    fail_rate        = round(total_fail    / total_answers * 100, 1)

    # Stability: same verdict across ALL runs
    q_stable = {qid: len(set(v)) == 1 for qid, v in q_verdicts.items()}
    unstable_qs      = sorted([qid for qid, s in q_stable.items() if not s])
    instability_rate = round(len(unstable_qs) / total_questions * 100, 1)

    # Hallucination rate: % of hallucination-test runs that FAILED
    halluc_q_ids = [qid for qid, is_h in q_halluc.items() if is_h]
    if halluc_q_ids:
        halluc_fails = sum(q_verdicts[qid].count("FAIL") for qid in halluc_q_ids)
        halluc_total = len(halluc_q_ids) * total_runs
        halluc_rate  = round(halluc_fails / halluc_total * 100, 1)
    else:
        halluc_rate = 0.0

    avg_run_time = round(sum(run_times) / len(run_times), 1)

    # Per-question breakdown
    per_question = []
    for qid in sorted(q_verdicts.keys()):
        verdicts   = q_verdicts[qid]
        pass_count = verdicts.count("PASS")
        part_count = verdicts.count("PARTIAL")
        fail_count = verdicts.count("FAIL")
        pass_pct   = round(pass_count / len(verdicts) * 100, 1)
        per_question.append({
            "id":            qid,
            "question":      q_questions[qid],
            "pass_count":    pass_count,
            "partial_count": part_count,
            "fail_count":    fail_count,
            "pass_pct":      pass_pct,
            "stable":        q_stable[qid],
            "halluc_test":   q_halluc.get(qid, False),
        })

    return {
        "total_questions":   total_questions,
        "total_answers":     total_answers,
        "overall_accuracy":  overall_accuracy,
        "partial_rate":      partial_rate,
        "fail_rate":         fail_rate,
        "instability_rate":  instability_rate,
        "unstable_q_ids":    unstable_qs,
        "hallucination_rate": halluc_rate,
        "avg_run_time_sec":  avg_run_time,
        "per_question":      per_question,
    }


def print_stability_report(report: dict, total_runs: int, session_name: str):
    """Print the stability report to console — exactly matching original format."""
    total_questions = report["total_questions"]
    total_answers   = report["total_answers"]
    per_question    = report["per_question"]

    print(f"\n{'='*60}")
    print(f"  STABILITY REPORT  ({total_runs} runs × {total_questions} questions)")
    print(f"{'='*60}")
    print(f"\n{'Q':>3}  {'Verdict Distribution':<30}  {'Pass%':>6}  {'Stable':>9}  Question")
    print(f"{'─'*3}  {'─'*30}  {'─'*6}  {'─'*9}  {'─'*40}")

    for pq in per_question:
        dist       = f"P={pq['pass_count']} Pr={pq['partial_count']} F={pq['fail_count']}"
        stable_str = "✅ STABLE" if pq["stable"] else "⚠️ UNSTABLE"
        q_short    = pq["question"][:45]
        print(f"{pq['id']:>3}  {dist:<30}  {pq['pass_pct']:>5.0f}%  {stable_str:>11}  {q_short}")

    total_pass    = sum(pq["pass_count"]    for pq in per_question)
    total_partial = sum(pq["partial_count"] for pq in per_question)
    total_fail    = sum(pq["fail_count"]    for pq in per_question)
    unstable_qs   = report["unstable_q_ids"]

    print(f"\n{'='*60}")
    print(f"  SUMMARY METRICS")
    print(f"{'='*60}")
    print(f"  Session              : {session_name}")
    print(f"  Runs completed       : {total_runs} / {total_runs}")
    print(f"  Questions per run    : {total_questions}")
    print(f"  Total answers scored : {total_answers}")
    print(f"")
    print(f"  ✅ Overall Accuracy  : {report['overall_accuracy']}%  ({total_pass}/{total_answers} PASS)")
    print(f"  ⚠️  Partial rate      : {report['partial_rate']}%  ({total_partial}/{total_answers} PARTIAL)")
    print(f"  ❌ Fail rate         : {report['fail_rate']}%  ({total_fail}/{total_answers} FAIL)")
    print(f"")
    print(f"  🔁 Instability rate  : {report['instability_rate']}%  "
          f"({len(unstable_qs)}/{total_questions} unstable questions)")
    if unstable_qs:
        print(f"     Unstable Q IDs   : {unstable_qs}")
    print(f"")
    print(f"  🚨 Hallucination rate: {report['hallucination_rate']}%  "
          f"(hallucination tests only)")
    print(f"")
    print(f"  ⏱️  Avg run time      : {report['avg_run_time_sec']}s")
    print(f"{'='*60}")


def save_session(session: dict, existing: dict):
    """Append session to existing results and save. Never overwrites."""
    existing["sessions"].append(session)
    try:
        with open(RESULTS_FILE, "w") as f:
            json.dump(existing, f, indent=2)
        total_sessions = len(existing["sessions"])
        print(f"\n  Results saved → {RESULTS_FILE}")
        print(f"  Total sessions stored: {total_sessions}")
    except Exception as e:
        print(f"[ERROR] Could not save results: {e}")

def print_final_metrics(report):
    accuracy = report["overall_accuracy"]
    reliability = round(100 - report["instability_rate"], 1)
    hallucination = report["hallucination_rate"]

    print("\n" + "="*60)
    print("🔥 FINAL SYSTEM METRICS (ALL RUNS)")
    print("="*60)
    print(f"✅ Accuracy           : {accuracy}%")
    print(f"🧠 Reliability       : {reliability}%")
    print(f"🚨 Hallucination Rate: {hallucination}%")
    print("="*60)

def main():
    existing     = load_existing_results()
    next_num     = get_next_session_number(existing)
    session_name = get_session_name(next_num)

    print(f"\n{'='*60}")
    print(f"  DocMind Stability Evaluation")
    print(f"  Session : {session_name}")
    print(f"  Runs    : {NUM_RUNS}")
    print(f"{'='*60}")

    all_pdf_reports = []
    overall_pass = 0
    overall_total = 0
    total_instability = 0
    total_hallucination = 0
    pdf_count = 0

    # 🔥 LOOP OVER ALL PDFs
    for pdf_path in PDF_PATHS:
        print(f"\n🚀 Running PDF: {os.path.basename(pdf_path)}")

        global PDF_PATH
        PDF_PATH = pdf_path

        # ✅ Load correct dataset for this PDF
        dataset_meta = load_dataset_meta(pdf_path)

        # ✅ Skip info (NOW SAFE)
        skipped = [qid for qid, m in dataset_meta.items() if m.get("skip")]
        if skipped:
            print(f"  ⏭️  Skipping Q IDs  : {skipped}")

        all_run_records = []
        run_times = []

        # 🔁 Run multiple times
        for i in range(1, NUM_RUNS + 1):
            start = time.time()

            result = run_evaluate(i, session_name)

            elapsed = round(time.time() - start, 1)
            run_times.append(elapsed)

            if result:
                run_record = build_run_record(i, result, elapsed)
                all_run_records.append(run_record)

                # ✅ ADD THIS PART
                print(f"\n[Run {i}] Pass={run_record['pass']} "
                    f"Partial={run_record['partial']} "
                    f"Fail={run_record['fail']} "
                    f"Rate={run_record['pass_rate']}% "
                    f"Time={elapsed}s")
            else:
                print(f"[Run {i}] FAILED")
        # ✅ Stability report per PDF
        report = compute_stability_report(
            all_run_records, dataset_meta, run_times, len(all_run_records)
        )
        total_instability += report["instability_rate"]
        total_hallucination += report["hallucination_rate"]
        pdf_count += 1

        pdf_pass = sum(pq["pass_count"] for pq in report["per_question"])
        pdf_total = report["total_answers"]

        acc = round(pdf_pass / pdf_total * 100, 2)

        print(f"✅ {os.path.basename(pdf_path)} → {acc}%")

        overall_pass += pdf_pass
        overall_total += pdf_total

        all_pdf_reports.append({
            "pdf": os.path.basename(pdf_path),
            "accuracy": acc
        })

    overall_acc = round(overall_pass / overall_total * 100, 2) if overall_total else 0

    avg_instability = total_instability / pdf_count if pdf_count else 0
    reliability = round(100 - avg_instability, 2)

    avg_hallucination = total_hallucination / pdf_count if pdf_count else 0

    print("\n==============================")
    print("📊 FINAL SYSTEM METRICS")
    print("==============================")

    for r in all_pdf_reports:
        print(f"{r['pdf']} → {r['accuracy']}%")

    print("\n🔥 SYSTEM LEVEL PERFORMANCE")
    print(f"✅ Accuracy           : {overall_acc}%")
    print(f"🧠 Reliability       : {reliability}%")
    print(f"🚨 Hallucination Rate: {round(avg_hallucination,2)}%")
    print("==============================")
if __name__ == "__main__":
    main()























# # # single pdf 
# """
# DocMind — Multi-Run Stability Evaluation
# Runs /evaluate N times and reports:
#   - Per-question accuracy across runs
#   - Overall accuracy
#   - Hallucination rate
#   - Instability rate (questions that change verdict across runs)
#   - All results saved to stability_results.json (append, never overwrite)
#   - Each session named by user (auto-increments if no name given)
# """

# import requests
# import json
# import time
# import sys
# import os
# from collections import defaultdict

# # ── CONFIG ──────────────────────────────────────────────────
# FLASK_URL        = "http://127.0.0.1:5000/evaluate"

# BASE_DIR = os.path.dirname(__file__)

# # PDF_PATH = r"C:\Users\Rhearitu\Downloads\Ethics of AI_new.pdf"
# PDF_PATH = r"C:\Users\Rhearitu\Downloads\crop report for testin -genai.pdf"
# # PDF_PATH = r"C:\Users\Rhearitu\Downloads\news -genai.pdf"
# NUM_RUNS         = 3
# RESULTS_FILE = os.path.join(BASE_DIR, "results", "stability_results.json")
# DATASET_FILE = os.path.join(BASE_DIR, "grounding_dataset.json")
# # ────────────────────────────────────────────────────────────


# def load_existing_results() -> dict:
#     """Load existing results file or create fresh structure."""
#     if os.path.exists(RESULTS_FILE):
#         try:
#             with open(RESULTS_FILE, "r") as f:
#                 data = json.load(f)
#             # Ensure sessions key exists
#             if "sessions" not in data:
#                 data["sessions"] = []
#             return data
#         except Exception as e:
#             print(f"[Warning] Could not read {RESULTS_FILE}: {e} — starting fresh")
#     return {"sessions": []}


# def get_next_session_number(existing: dict) -> int:
#     """Auto-increment session number based on existing sessions."""
#     return len(existing.get("sessions", [])) + 1


# def get_session_name(next_num: int) -> str:
#     """Prompt user for session name. Auto-names if empty."""
#     auto_name = f"Test {next_num}"
#     try:
#         user_input = input(
#             f"\nEnter session name (press Enter for auto '{auto_name}'): "
#         ).strip()
#         return user_input if user_input else auto_name
#     except (EOFError, KeyboardInterrupt):
#         return auto_name


# def load_dataset_meta() -> dict:
#     """
#     Load grounding_dataset.json to get hallucination_test flags
#     and skip flags per question. Returns dict keyed by question id.
#     """
#     meta = {}
#     if not os.path.exists(DATASET_FILE):
#         print(f"[Warning] {DATASET_FILE} not found — hallucination flags unavailable")
#         return meta
#     try:
#         with open(DATASET_FILE, "r") as f:
#             dataset = json.load(f)
#         for q in dataset.get("questions", []):
#             meta[q["id"]] = {
#                 "hallucination_test": q.get("hallucination_test", False),
#                 "skip":               q.get("skip", False),
#             }
#     except Exception as e:
#         print(f"[Warning] Could not read dataset meta: {e}")
#     return meta


# def run_evaluate(run_num: int, session_name: str) -> dict:
#     """Call Flask /evaluate endpoint for one run."""
#     payload = {
#         "pdf_path":        PDF_PATH,
#         "run_description": f"{session_name}_run_{run_num}"
#     }
#     print(f"\n{'='*60}")
#     print(f"  RUN {run_num} / {NUM_RUNS}")
#     print(f"{'='*60}")
#     try:
#         resp = requests.post(FLASK_URL, json=payload, timeout=7200)
#         if resp.status_code != 200:
#             print(f"[ERROR] Run {run_num} returned HTTP {resp.status_code}")
#             print(f"[ERROR] Response: {resp.text[:500]}")
#             return {}
#         try:
#             data = resp.json()
#         except Exception as json_err:
#             print(f"[ERROR] Run {run_num} — invalid JSON response: {json_err}")
#             print(f"[ERROR] Raw response: {resp.text[:500]}")
#             return {}
#         return data
#     except requests.exceptions.Timeout:
#         print(f"[ERROR] Run {run_num} timed out")
#         return {}
#     except Exception as e:
#         print(f"[ERROR] Run {run_num} failed: {e}")
#         return {}


# def build_run_record(run_num: int, result: dict, elapsed: float) -> dict:
#     """
#     Build a structured record for a single run including
#     all per-question details exactly as received from Flask.
#     """
#     questions_detail = []
#     for r in result.get("results", []):
#         questions_detail.append({
#             "id":               r.get("id"),
#             "question":         r.get("question", ""),
#             "verdict":          r.get("verdict", "FAIL"),
#             "keyword_hits":     r.get("keyword_hits", 0),
#             "max_score":        r.get("max_score", 0),
#             "pass_threshold":   r.get("pass_threshold", 1),
#             "model_used":       r.get("model_used", "unknown"),
#             "recall_at_k":      r.get("recall_at_k", 0.0),
#             "answer_grounding": r.get("answer_grounding", 0.0),
#             "confidence":       r.get("confidence", 0.0),
#             "hallucination":    r.get("hallucination", False),
#             "actual_answer":    r.get("actual_answer", ""),
#             "tests":            r.get("tests", ""),
#         })

#     return {
#         "run_number": run_num,
#         "pass":       result.get("pass", 0),
#         "partial":    result.get("partial", 0),
#         "fail":       result.get("fail", 0),
#         "pass_rate":  float(str(result.get("pass_rate", "0")).replace("%", "")),
#         "time_sec":   elapsed,
#         "questions":  questions_detail,
#     }


# def compute_stability_report(all_run_records: list,
#                               dataset_meta: dict,
#                               run_times: list,
#                               total_runs: int) -> dict:
#     """
#     Compute full stability report from all run records.
#     Returns structured dict matching console output.
#     """
#     q_verdicts  = defaultdict(list)
#     q_questions = {}
#     q_halluc    = {}

#     for run_record in all_run_records:
#         for q in run_record["questions"]:
#             qid = q["id"]
#             q_verdicts[qid].append(q["verdict"])
#             q_questions[qid] = q["question"]
#             # Use dataset meta for hallucination_test (authoritative)
#             # Fall back to response flag if dataset not available
#             if qid in dataset_meta:
#                 q_halluc[qid] = dataset_meta[qid]["hallucination_test"]
#             else:
#                 q_halluc[qid] = q.get("hallucination", False)

#     total_questions = len(q_verdicts)
#     total_answers   = total_questions * total_runs
#     total_pass      = sum(v.count("PASS")    for v in q_verdicts.values())
#     total_partial   = sum(v.count("PARTIAL") for v in q_verdicts.values())
#     total_fail      = sum(v.count("FAIL")    for v in q_verdicts.values())

#     overall_accuracy = round(total_pass    / total_answers * 100, 1)
#     partial_rate     = round(total_partial / total_answers * 100, 1)
#     fail_rate        = round(total_fail    / total_answers * 100, 1)

#     # Stability: same verdict across ALL runs
#     q_stable = {qid: len(set(v)) == 1 for qid, v in q_verdicts.items()}
#     unstable_qs      = sorted([qid for qid, s in q_stable.items() if not s])
#     instability_rate = round(len(unstable_qs) / total_questions * 100, 1)

#     # Hallucination rate: % of hallucination-test runs that FAILED
#     halluc_q_ids = [qid for qid, is_h in q_halluc.items() if is_h]
#     if halluc_q_ids:
#         halluc_fails = sum(q_verdicts[qid].count("FAIL") for qid in halluc_q_ids)
#         halluc_total = len(halluc_q_ids) * total_runs
#         halluc_rate  = round(halluc_fails / halluc_total * 100, 1)
#     else:
#         halluc_rate = 0.0

#     avg_run_time = round(sum(run_times) / len(run_times), 1)

#     # Per-question breakdown
#     per_question = []
#     for qid in sorted(q_verdicts.keys()):
#         verdicts   = q_verdicts[qid]
#         pass_count = verdicts.count("PASS")
#         part_count = verdicts.count("PARTIAL")
#         fail_count = verdicts.count("FAIL")
#         pass_pct   = round(pass_count / len(verdicts) * 100, 1)
#         per_question.append({
#             "id":            qid,
#             "question":      q_questions[qid],
#             "pass_count":    pass_count,
#             "partial_count": part_count,
#             "fail_count":    fail_count,
#             "pass_pct":      pass_pct,
#             "stable":        q_stable[qid],
#             "halluc_test":   q_halluc.get(qid, False),
#         })

#     return {
#         "total_questions":   total_questions,
#         "total_answers":     total_answers,
#         "overall_accuracy":  overall_accuracy,
#         "partial_rate":      partial_rate,
#         "fail_rate":         fail_rate,
#         "instability_rate":  instability_rate,
#         "unstable_q_ids":    unstable_qs,
#         "hallucination_rate": halluc_rate,
#         "avg_run_time_sec":  avg_run_time,
#         "per_question":      per_question,
#     }


# def print_stability_report(report: dict, total_runs: int, session_name: str):
#     """Print the stability report to console — exactly matching original format."""
#     total_questions = report["total_questions"]
#     total_answers   = report["total_answers"]
#     per_question    = report["per_question"]

#     print(f"\n{'='*60}")
#     print(f"  STABILITY REPORT  ({total_runs} runs × {total_questions} questions)")
#     print(f"{'='*60}")
#     print(f"\n{'Q':>3}  {'Verdict Distribution':<30}  {'Pass%':>6}  {'Stable':>9}  Question")
#     print(f"{'─'*3}  {'─'*30}  {'─'*6}  {'─'*9}  {'─'*40}")

#     for pq in per_question:
#         dist       = f"P={pq['pass_count']} Pr={pq['partial_count']} F={pq['fail_count']}"
#         stable_str = "✅ STABLE" if pq["stable"] else "⚠️ UNSTABLE"
#         q_short    = pq["question"][:45]
#         print(f"{pq['id']:>3}  {dist:<30}  {pq['pass_pct']:>5.0f}%  {stable_str:>11}  {q_short}")

#     total_pass    = sum(pq["pass_count"]    for pq in per_question)
#     total_partial = sum(pq["partial_count"] for pq in per_question)
#     total_fail    = sum(pq["fail_count"]    for pq in per_question)
#     unstable_qs   = report["unstable_q_ids"]

#     print(f"\n{'='*60}")
#     print(f"  SUMMARY METRICS")
#     print(f"{'='*60}")
#     print(f"  Session              : {session_name}")
#     print(f"  Runs completed       : {total_runs} / {total_runs}")
#     print(f"  Questions per run    : {total_questions}")
#     print(f"  Total answers scored : {total_answers}")
#     print(f"")
#     print(f"  ✅ Overall Accuracy  : {report['overall_accuracy']}%  ({total_pass}/{total_answers} PASS)")
#     print(f"  ⚠️  Partial rate      : {report['partial_rate']}%  ({total_partial}/{total_answers} PARTIAL)")
#     print(f"  ❌ Fail rate         : {report['fail_rate']}%  ({total_fail}/{total_answers} FAIL)")
#     print(f"")
#     print(f"  🔁 Instability rate  : {report['instability_rate']}%  "
#           f"({len(unstable_qs)}/{total_questions} unstable questions)")
#     if unstable_qs:
#         print(f"     Unstable Q IDs   : {unstable_qs}")
#     print(f"")
#     print(f"  🚨 Hallucination rate: {report['hallucination_rate']}%  "
#           f"(hallucination tests only)")
#     print(f"")
#     print(f"  ⏱️  Avg run time      : {report['avg_run_time_sec']}s")
#     print(f"{'='*60}")


# def save_session(session: dict, existing: dict):
#     """Append session to existing results and save. Never overwrites."""
#     existing["sessions"].append(session)
#     try:
#         with open(RESULTS_FILE, "w") as f:
#             json.dump(existing, f, indent=2)
#         total_sessions = len(existing["sessions"])
#         print(f"\n  Results saved → {RESULTS_FILE}")
#         print(f"  Total sessions stored: {total_sessions}")
#     except Exception as e:
#         print(f"[ERROR] Could not save results: {e}")
# def print_final_metrics(report):
#     accuracy = report["overall_accuracy"]
#     reliability = round(100 - report["instability_rate"], 1)
#     hallucination = report["hallucination_rate"]

#     print("\n" + "="*60)
#     print("🔥 FINAL SYSTEM METRICS (ALL RUNS)")
#     print("="*60)
#     print(f"✅ Accuracy           : {accuracy}%")
#     print(f"🧠 Reliability       : {reliability}%")
#     print(f"🚨 Hallucination Rate: {hallucination}%")
#     print("="*60)

# def main():
#     # ── Load existing results + dataset meta ─────────────────
#     existing     = load_existing_results()
#     dataset_meta = load_dataset_meta()
#     next_num     = get_next_session_number(existing)
#     session_name = get_session_name(next_num)

#     print(f"\n{'='*60}")
#     print(f"  DocMind Stability Evaluation")
#     print(f"  Session : {session_name}")
#     print(f"  PDF     : {os.path.basename(PDF_PATH)}")
#     print(f"  Runs    : {NUM_RUNS}")
#     print(f"{'='*60}")

#     # ── Check for skipped questions ───────────────────────────
#     skipped = [qid for qid, m in dataset_meta.items() if m.get("skip")]
#     if skipped:
#         print(f"  ⏭️  Skipping Q IDs  : {skipped} (skip=true in dataset)")

#     all_run_records = []
#     run_times       = []

#     # ── Run evaluation N times ────────────────────────────────
#     for i in range(1, NUM_RUNS + 1):
#         start  = time.time()
#         result = run_evaluate(i, session_name)
#         elapsed = round(time.time() - start, 1)
#         run_times.append(elapsed)

#         if result:
#             run_record = build_run_record(i, result, elapsed)
#             all_run_records.append(run_record)
#             print(f"\n[Run {i}] Pass={run_record['pass']} "
#                   f"Partial={run_record['partial']} "
#                   f"Fail={run_record['fail']} "
#                   f"Rate={run_record['pass_rate']}% "
#                   f"Time={elapsed}s")
#         else:
#             print(f"[Run {i}] FAILED — skipping")

#     if not all_run_records:
#         print("No successful runs. Exiting.")
#         sys.exit(1)

#     actual_runs = len(all_run_records)

#     # ── Compute stability report ──────────────────────────────
#     report = compute_stability_report(
#         all_run_records, dataset_meta, run_times, actual_runs
#     )

#     # ── Print to console ──────────────────────────────────────
#     print_stability_report(report, actual_runs, session_name)
#     print_final_metrics(report) 

#     # ── Build session record ──────────────────────────────────
#     session = {
#         "name":              session_name,
#         "date":              time.strftime("%Y-%m-%d %H:%M:%S"),
#         "pdf":               os.path.basename(PDF_PATH),
#         "num_runs":          actual_runs,
#         "runs":              all_run_records,
#         "stability_report":  report,
#     }

#     # ── Save (append) ─────────────────────────────────────────
#     save_session(session, existing)


# if __name__ == "__main__":
#     main()