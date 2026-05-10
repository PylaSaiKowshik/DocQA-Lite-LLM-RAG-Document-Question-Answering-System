# import re
# from docmind_rag.utils.helpers import _get_dynamic_stopwords
# from numpy import dot
# from numpy.linalg import norm

# # ============================================================
# # FIX 4 — Text normalization
# # Converts unicode symbols to ASCII equivalents so retrieval
# # doesn't silently fail on symbol mismatches between question
# # and document text. Applied to both chunks and question.
# # Generic — no hardcoded domain terms.
# # ============================================================

# _UNICODE_MAP = {
#     '\u2265': '>=',   # ≥
#     '\u2264': '<=',   # ≤
#     '\u2260': '!=',   # ≠
#     '\u2248': '~=',   # ≈
#     '\u00b1': '+-',   # ±
#     '\u00d7': 'x',    # ×
#     '\u00f7': '/',    # ÷
#     '\u2192': '->',   # →
#     '\u2190': '<-',   # ←
#     '\u2014': '-',    # —
#     '\u2013': '-',    # –
#     '\u2019': "'",    # '
#     '\u2018': "'",    # '
#     '\u201c': '"',    # "
#     '\u201d': '"',    # "
#     '\u2022': '-',    # •
#     '\u00e9': 'e',    # é
#     '\u00e8': 'e',    # è
#     '\u00ea': 'e',    # ê
#     '\u00e0': 'a',    # à
#     '\u00e2': 'a',    # â
#     '\u00fc': 'u',    # ü
#     '\u00f6': 'o',    # ö
#     '\u00e4': 'a',    # ä
#     '\u00df': 'ss',   # ß
#     '\u00b0': ' deg', # °
#     '\u03b1': 'alpha',
#     '\u03b2': 'beta',
#     '\u03b3': 'gamma',
#     '\u03c3': 'sigma',
#     '\u03bc': 'mu',
#     '\u03c0': 'pi',
# }


# def semantic_rank(question, chunks, embedding_model):
#     q_emb = embedding_model.embed_query(question)

#     scored = []
#     for c in chunks:
#         text = c if isinstance(c, str) else c.page_content
#         c_emb = embedding_model.embed_query(text[:300])

#         sim = dot(q_emb, c_emb) / (norm(q_emb) * norm(c_emb) + 1e-8)
#         scored.append((sim, c))

#         scored.sort(key=lambda x: x[0], reverse=True)
#         return [c for _, c in scored]
# def normalize_text(text: str) -> str:
#     """
#     Normalize unicode symbols to ASCII-friendly equivalents.
#     Applied to both chunks (in node_chunk) and questions (in node_qa).
#     Prevents silent retrieval mismatches when document uses special characters.
#     """
#     if not text:
#         return text
#     for char, replacement in _UNICODE_MAP.items():
#         text = text.replace(char, replacement)
#     # Collapse multiple spaces created by replacements
#     text = re.sub(r'[ \t]{2,}', ' ', text)
#     return text


# # ============================================================
# # TEXT UTILITIES
# # ============================================================

# def _normalize_text(text: str) -> str:
#     """Lowercase, strip punctuation, collapse whitespace."""
#     text = text.lower()
#     text = re.sub(r'[^a-z0-9\s]', ' ', text)
#     text = re.sub(r'\s+', ' ', text)
#     return text.strip()


# def reorder_by_question(question: str, chunks: list) -> list:
#     """
#     Reorder chunks by relevance to question using keyword overlap.
#     Gives bonus when a number from the question appears in the
#     first line of a chunk (structural signal for navigational queries).
#     """
#     if not chunks:
#         return chunks

#     stopwords    = _get_dynamic_stopwords()
#     q_words      = set(w.lower() for w in question.split()
#                        if w.lower() not in stopwords and len(w) > 2)
#     q_numbers    = set(re.findall(r'\b\d+\b', question))

#     def score(chunk: str) -> int:
#         chunk_lower  = chunk.lower()
#         kw_score     = sum(1 for w in q_words if w in chunk_lower)
#         num_score    = sum(3 for n in q_numbers
#                           if re.search(rf'\b{re.escape(n)}\b', chunk_lower))

#         first_line = chunk.strip().split('\n')[0].lower() if chunk.strip() else ""
#         first_line_num_bonus = sum(
#             5 for n in q_numbers
#             if re.search(rf'\b{re.escape(n)}\b', first_line)
#         )

#         total          = kw_score + num_score + first_line_num_bonus
#         repeated_spans = _count_repeated_spans(chunk)
#         return total - repeated_spans

#     scored = sorted(chunks, key=score, reverse=True)

#     top_score = score(scored[0]) if scored else 0
#     if top_score > 0:
#         print(f"[Reorder] top chunk score={top_score} | "
#               f"numbers={list(q_numbers)} | "
#               f"keywords={list(q_words)[:4]}")
#     return scored


# def _count_repeated_spans(chunk: str, min_len: int = 4) -> int:
#     words  = chunk.lower().split()
#     seen   = {}
#     repeat = 0
#     for i in range(len(words) - min_len + 1):
#         span = " ".join(words[i:i + min_len])
#         if span in seen:
#             repeat += 1
#             if repeat == 1:
#                 print(f"[Reorder] Tiebreaker applied — repeated span: '{span}'")
#         seen[span] = i
#     return repeat


# def extract_numeric_answer(question: str, chunks: list) -> str:
#     """
#     Regex-based numeric answer extraction.
#     Finds a number adjacent to question keywords in context.
#     Generic — no hardcoding.
#     """
#     stopwords = _get_dynamic_stopwords()
#     keywords  = [w.lower() for w in question.split()
#                  if len(w) > 3 and w.lower() not in stopwords]

#     for chunk in chunks:
#         sentences = re.split(r'(?<=[.!?\n])', chunk)
#         for sent in sentences:
#             sent_lower = sent.lower()
#             kw_hits    = sum(1 for kw in keywords if kw in sent_lower)
#             if kw_hits >= 2:
#                 numbers = re.findall(
#                     r'\b\d+(?:[.,]\d+)?(?:\s*%|\s*percent)?\b', sent
#                 )
#                 if numbers:
#                     return numbers[0]
#     return ""


# def clean_context_for_llm(chunks: list) -> list:
#     """Remove duplicate chunks before sending to LLM."""
#     seen    = set()
#     cleaned = []
#     for chunk in chunks:
#         fp = chunk.strip()[:80]
#         if fp not in seen:
#             seen.add(fp)
#             cleaned.append(chunk)
#     return cleaned


# def classify_from_context(question: str, chunks: list) -> str:
#     """
#     Determine answer type from retrieved chunks + question structure.
#     NAVIGATIONAL and POSITIONAL are handled upstream in node_qa before
#     this function is called.

#     Returns one of: FULL_SUMMARY | MULTIPART_QA | FACTUAL_QA | VERIFICATION_QA
#     """
#     q_low = question.strip().lower()
#     words = q_low.split()

#     # Verification — polar question structure
#     if re.match(
#         r'^(is|was|were|does|did|can|should|would|could|has|have|had|are|do|will)\b',
#         q_low
#     ):
#         return "VERIFICATION_QA"

#     # Summary
#     if (
#         len(words) > 12 and "?" not in question
#     ) or re.match(
#         r'^(summarize|summarise|give\s+a|provide\s+a|generate\s+a|'
#         r'write\s+a|create\s+a)\b', q_low
#     ):
#         return "FULL_SUMMARY"

#     # List structure in chunks
#     list_chunk_count = 0
#     for chunk in chunks:
#         lines   = chunk.strip().split("\n")
#         numbered = sum(
#             1 for line in lines
#             if re.match(r'^\s*(\d+[\.\)]|[-•*])\s+\w', line.strip())
#         )
#         if numbered >= 2:
#             list_chunk_count += 1

#     stopwords     = _get_dynamic_stopwords()
#     content_words = [w for w in words if len(w) > 3 and w not in stopwords]

#     chunks_with_hits = sum(
#         1 for chunk in chunks
#         if sum(1 for w in content_words if w in chunk.lower()) >= 2
#     )

#     top_chunk_lines = chunks[0].strip().split("\n") if chunks else []
#     short_answer_lines = [
#         line for line in top_chunk_lines
#         if 2 <= len(line.split()) <= 8
#         and any(w in line.lower() for w in content_words)
#     ]
#     has_short_span = len(short_answer_lines) > 0

#     if list_chunk_count >= 2:
#         return "MULTIPART_QA"

#     if chunks_with_hits >= 3 and not has_short_span:
#         return "MULTIPART_QA"

#     if chunks_with_hits <= 2 and has_short_span:
#         return "FACTUAL_QA"

#     return "FACTUAL_QA"


# # ============================================================
# # STUBS — kept for import compatibility
# # ============================================================

# def extract_named_entities(text: str) -> list:
#     """Extract named entities using spacy. Kept for import compatibility."""
#     try:
#         import spacy
#         nlp  = spacy.load("en_core_web_lg")
#         doc  = nlp(text[:5000])
#         return [ent.text for ent in doc.ents]
#     except Exception:
#         return []


# def expand_answer(answer: str, chunks: list) -> str:
#     """
#     Minimal expansion — if answer is a single word and context has
#     a line starting with that word, return the fuller line.
#     Generic, no hardcoding.
#     """
#     if not answer or len(answer.split()) > 3:
#         return answer
#     ans_lower = answer.lower().strip()
#     for chunk in chunks:
#         for line in chunk.split("\n"):
#             line = line.strip()
#             if line.lower().startswith(ans_lower) and 2 <= len(line.split()) <= 8:
#                 return line
#     return answer


# def normalize_answer(answer: str) -> str:
#     """Minimal normalization — strip whitespace and double spaces."""
#     if not answer:
#         return answer
#     answer = answer.strip()
#     answer = re.sub(r'\s{2,}', ' ', answer)
#     return answer


# # def detect_doc_type(text: str) -> str:
# #     """Detect document type from content patterns. Generic."""
# #     text_lower = text.lower()
# #     if any(w in text_lower for w in ['abstract', 'methodology', 'references', 'doi']):
# #         return "research"
# #     if any(w in text_lower for w in ['revenue', 'profit', 'fiscal', 'earnings']):
# #         return "financial"
# #     if any(w in text_lower for w in ['clause', 'agreement', 'liability', 'jurisdiction']):
# #         return "legal"
# #     return "general"
















































































# single pdf 
import re
from docmind_rag.utils.helpers import _get_dynamic_stopwords
from numpy import dot
from numpy.linalg import norm

# ============================================================
# FIX 4 — Text normalization
# Converts unicode symbols to ASCII equivalents so retrieval
# doesn't silently fail on symbol mismatches between question
# and document text. Applied to both chunks and question.
# Generic — no hardcoded domain terms.
# ============================================================

_UNICODE_MAP = {
    '\u2265': '>=',   # ≥
    '\u2264': '<=',   # ≤
    '\u2260': '!=',   # ≠
    '\u2248': '~=',   # ≈
    '\u00b1': '+-',   # ±
    '\u00d7': 'x',    # ×
    '\u00f7': '/',    # ÷
    '\u2192': '->',   # →
    '\u2190': '<-',   # ←
    '\u2014': '-',    # —
    '\u2013': '-',    # –
    '\u2019': "'",    # '
    '\u2018': "'",    # '
    '\u201c': '"',    # "
    '\u201d': '"',    # "
    '\u2022': '-',    # •
    '\u00e9': 'e',    # é
    '\u00e8': 'e',    # è
    '\u00ea': 'e',    # ê
    '\u00e0': 'a',    # à
    '\u00e2': 'a',    # â
    '\u00fc': 'u',    # ü
    '\u00f6': 'o',    # ö
    '\u00e4': 'a',    # ä
    '\u00df': 'ss',   # ß
    '\u00b0': ' deg', # °
    '\u03b1': 'alpha',
    '\u03b2': 'beta',
    '\u03b3': 'gamma',
    '\u03c3': 'sigma',
    '\u03bc': 'mu',
    '\u03c0': 'pi',
}


def semantic_rank(question, chunks, embedding_model):
    q_emb = embedding_model.embed_query(question)

    scored = []
    for c in chunks:
        text = c if isinstance(c, str) else c.page_content
        c_emb = embedding_model.embed_query(text[:300])

        sim = dot(q_emb, c_emb) / (norm(q_emb) * norm(c_emb) + 1e-8)
        scored.append((sim, text))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]

def normalize_text(text: str) -> str:
    """
    Normalize unicode symbols to ASCII-friendly equivalents.
    Applied to both chunks (in node_chunk) and questions (in node_qa).
    Prevents silent retrieval mismatches when document uses special characters.
    """
    if not text:
        return text
    for char, replacement in _UNICODE_MAP.items():
        text = text.replace(char, replacement)
    # Collapse multiple spaces created by replacements
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text


# ============================================================
# TEXT UTILITIES
# ============================================================

def _normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def reorder_by_question(question: str, chunks: list) -> list:
    """
    Reorder chunks by relevance to question using keyword overlap.
    Gives bonus when a number from the question appears in the
    first line of a chunk (structural signal for navigational queries).
    """
    if not chunks:
        return chunks

    stopwords    = _get_dynamic_stopwords()
    q_words      = set(w.lower() for w in question.split()
                       if w.lower() not in stopwords and len(w) > 2)
    q_numbers    = set(re.findall(r'\b\d+\b', question))

    def score(chunk: str) -> int:
        chunk_lower  = chunk.lower()
        kw_score     = sum(1 for w in q_words if w in chunk_lower)
        num_score    = sum(3 for n in q_numbers
                          if re.search(rf'\b{re.escape(n)}\b', chunk_lower))

        first_line = chunk.strip().split('\n')[0].lower() if chunk.strip() else ""
        first_line_num_bonus = sum(
            5 for n in q_numbers
            if re.search(rf'\b{re.escape(n)}\b', first_line)
        )

        total          = kw_score + num_score + first_line_num_bonus
        repeated_spans = _count_repeated_spans(chunk)
        return total - repeated_spans

    scored = sorted(chunks, key=score, reverse=True)

    top_score = score(scored[0]) if scored else 0
    if top_score > 0:
        print(f"[Reorder] top chunk score={top_score} | "
              f"numbers={list(q_numbers)} | "
              f"keywords={list(q_words)[:4]}")
    return scored


def _count_repeated_spans(chunk: str, min_len: int = 4) -> int:
    words  = chunk.lower().split()
    seen   = {}
    repeat = 0
    for i in range(len(words) - min_len + 1):
        span = " ".join(words[i:i + min_len])
        if span in seen:
            repeat += 1
            if repeat == 1:
                print(f"[Reorder] Tiebreaker applied — repeated span: '{span}'")
        seen[span] = i
    return repeat


def extract_numeric_answer(question: str, chunks: list) -> str:
    """
    Regex-based numeric answer extraction.
    Finds a number adjacent to question keywords in context.
    Generic — no hardcoding.
    """
    stopwords = _get_dynamic_stopwords()
    keywords  = [w.lower() for w in question.split()
                 if len(w) > 3 and w.lower() not in stopwords]

    for chunk in chunks:
        sentences = re.split(r'(?<=[.!?\n])', chunk)
        for sent in sentences:
            sent_lower = sent.lower()
            kw_hits    = sum(1 for kw in keywords if kw in sent_lower)
            if kw_hits >= 2:
                numbers = re.findall(
                    r'\b\d+(?:[.,]\d+)?(?:\s*%|\s*percent)?\b', sent
                )
                if numbers:
                    return numbers[0]
    return ""


def clean_context_for_llm(chunks: list) -> list:
    """Remove duplicate chunks before sending to LLM."""
    seen    = set()
    cleaned = []
    for chunk in chunks:
        fp = chunk.strip()[:80]
        if fp not in seen:
            seen.add(fp)
            cleaned.append(chunk)
    return cleaned


def classify_from_context(question: str, chunks: list) -> str:
    """
    Determine answer type from retrieved chunks + question structure.
    NAVIGATIONAL and POSITIONAL are handled upstream in node_qa before
    this function is called.

    Returns one of: FULL_SUMMARY | MULTIPART_QA | FACTUAL_QA | VERIFICATION_QA
    """
    q_low = question.strip().lower()
    words = q_low.split()

    # Verification — polar question structure
    if re.match(
        r'^(is|was|were|does|did|can|should|would|could|has|have|had|are|do|will)\b',
        q_low
    ):
        return "VERIFICATION_QA"

    # Summary
    if (
        len(words) > 12 and "?" not in question
    ) or re.match(
        r'^(summarize|summarise|give\s+a|provide\s+a|generate\s+a|'
        r'write\s+a|create\s+a)\b', q_low
    ):
        return "FULL_SUMMARY"

    # List structure in chunks
    list_chunk_count = 0
    for chunk in chunks:
        lines   = chunk.strip().split("\n")
        numbered = sum(
            1 for line in lines
            if re.match(r'^\s*(\d+[\.\)]|[-•*])\s+\w', line.strip())
        )
        if numbered >= 2:
            list_chunk_count += 1

    stopwords     = _get_dynamic_stopwords()
    content_words = [w for w in words if len(w) > 3 and w not in stopwords]

    chunks_with_hits = sum(
        1 for chunk in chunks
        if sum(1 for w in content_words if w in chunk.lower()) >= 2
    )

    top_chunk_lines = chunks[0].strip().split("\n") if chunks else []
    short_answer_lines = [
        line for line in top_chunk_lines
        if 2 <= len(line.split()) <= 8
        and any(w in line.lower() for w in content_words)
    ]
    has_short_span = len(short_answer_lines) > 0

    if list_chunk_count >= 2:
        return "MULTIPART_QA"

    if chunks_with_hits >= 3 and not has_short_span:
        return "MULTIPART_QA"

    if chunks_with_hits <= 2 and has_short_span:
        return "FACTUAL_QA"

    return "FACTUAL_QA"


# ============================================================
# STUBS — kept for import compatibility
# ============================================================

def extract_named_entities(text: str) -> list:
    """Extract named entities using spacy. Kept for import compatibility."""
    try:
        import spacy
        nlp  = spacy.load("en_core_web_lg")
        doc  = nlp(text[:5000])
        return [ent.text for ent in doc.ents]
    except Exception:
        return []


def expand_answer(answer: str, chunks: list) -> str:
    """
    Minimal expansion — if answer is a single word and context has
    a line starting with that word, return the fuller line.
    Generic, no hardcoding.
    """
    if not answer or len(answer.split()) > 3:
        return answer
    ans_lower = answer.lower().strip()
    for chunk in chunks:
        for line in chunk.split("\n"):
            line = line.strip()
            if line.lower().startswith(ans_lower) and 2 <= len(line.split()) <= 8:
                return line
    return answer


def normalize_answer(answer: str) -> str:
    """Minimal normalization — strip whitespace and double spaces."""
    if not answer:
        return answer
    answer = answer.strip()
    answer = re.sub(r'\s{2,}', ' ', answer)
    return answer


# def detect_doc_type(text: str) -> str:
#     """Detect document type from content patterns. Generic."""
#     text_lower = text.lower()
#     if any(w in text_lower for w in ['abstract', 'methodology', 'references', 'doi']):
#         return "research"
#     if any(w in text_lower for w in ['revenue', 'profit', 'fiscal', 'earnings']):
#         return "financial"
#     if any(w in text_lower for w in ['clause', 'agreement', 'liability', 'jurisdiction']):
#         return "legal"
#     return "general"


































# got 70 nd 80 - lap claud e
# import re
# from docmind_rag.utils.helpers import _get_dynamic_stopwords


# def _normalize_text(text: str) -> str:
#     """Lowercase, strip punctuation, collapse whitespace."""
#     text = text.lower()
#     text = re.sub(r'[^a-z0-9\s]', ' ', text)
#     text = re.sub(r'\s+', ' ', text)
#     return text.strip()


# def reorder_by_question(question: str, chunks: list) -> list:
#     """
#     Reorder chunks by relevance to question using keyword overlap.
#     Tiebreak: prefer chunks with numbers matching question numbers.
#     Generic — no hardcoding.
#     """
#     if not chunks:
#         return chunks

#     stopwords   = _get_dynamic_stopwords()
#     q_words     = set(w.lower() for w in question.split()
#                       if w.lower() not in stopwords and len(w) > 2)
#     q_numbers   = set(re.findall(r'\b\d+\b', question))

#     def score(chunk: str) -> int:
#         chunk_lower = chunk.lower()
#         kw_score    = sum(1 for w in q_words if w in chunk_lower)
#         num_score   = sum(3 for n in q_numbers
#                           if re.search(rf'\b{re.escape(n)}\b', chunk_lower))

#         # Prefer chunk where a number appears in the first line (structural signal)
#         first_line = chunk.strip().split('\n')[0].lower() if chunk.strip() else ""
#         first_line_num_bonus = sum(
#             5 for n in q_numbers
#             if re.search(rf'\b{re.escape(n)}\b', first_line)
#         )

#         total = kw_score + num_score + first_line_num_bonus

#         # Tiebreaker: penalise repeated span
#         repeated_spans = _count_repeated_spans(chunk)
#         return total - repeated_spans

#     scored = sorted(chunks, key=score, reverse=True)

#     top_score = score(scored[0]) if scored else 0
#     if top_score > 0:
#         print(f"[Reorder] top chunk score={top_score} | "
#               f"numbers={list(q_numbers)} | "
#               f"keywords={[w for w in list(q_words)[:4]]}")
#     return scored


# def _count_repeated_spans(chunk: str, min_len: int = 4) -> int:
#     """Count repeated word spans — used as a quality penalty."""
#     words  = chunk.lower().split()
#     seen   = {}
#     repeat = 0
#     for i in range(len(words) - min_len + 1):
#         span = " ".join(words[i:i + min_len])
#         if span in seen:
#             repeat += 1
#             if repeat == 1:
#                 print(f"[Reorder] Tiebreaker applied — repeated span: '{span}'")
#         seen[span] = i
#     return repeat


# def extract_numeric_answer(question: str, chunks: list) -> str:
#     """
#     Regex-based numeric answer extraction.
#     Finds a number adjacent to question keywords in context.
#     Generic — no hardcoding.
#     """
#     stopwords = _get_dynamic_stopwords()
#     keywords  = [w.lower() for w in question.split()
#                  if len(w) > 3 and w.lower() not in stopwords]

#     for chunk in chunks:
#         sentences = re.split(r'(?<=[.!?\n])', chunk)
#         for sent in sentences:
#             sent_lower = sent.lower()
#             kw_hits    = sum(1 for kw in keywords if kw in sent_lower)
#             if kw_hits >= 2:
#                 numbers = re.findall(
#                     r'\b\d+(?:[.,]\d+)?(?:\s*%|\s*percent)?\b', sent
#                 )
#                 if numbers:
#                     return numbers[0]
#     return ""


# def clean_context_for_llm(chunks: list) -> list:
#     """
#     Remove duplicate chunks before sending to LLM.
#     Uses fingerprint (first 80 chars) for dedup.
#     """
#     seen    = set()
#     cleaned = []
#     for chunk in chunks:
#         fp = chunk.strip()[:80]
#         if fp not in seen:
#             seen.add(fp)
#             cleaned.append(chunk)
#     return cleaned


# def classify_from_context(question: str, chunks: list) -> str:
#     """
#     Determine answer type from retrieved chunks + question structure.
#     Called AFTER retrieval — the document tells us the answer shape.

#     NOTE: NAVIGATIONAL and POSITIONAL are handled upstream in node_qa
#     before this function is called. This function handles the remaining
#     routing only.

#     Returns one of: FULL_SUMMARY | MULTIPART_QA | FACTUAL_QA | VERIFICATION_QA
#     """
#     q_low = question.strip().lower()
#     words = q_low.split()

#     # ── Signal 1: Verification — polar question structure ─────
#     if re.match(
#         r'^(is|was|were|does|did|can|should|would|could|has|have|had|are|do|will)\b',
#         q_low
#     ):
#         return "VERIFICATION_QA"

#     # ── Signal 2: Summary ─────────────────────────────────────
#     if (
#         len(words) > 12 and "?" not in question
#     ) or re.match(
#         r'^(summarize|summarise|give\s+a|provide\s+a|generate\s+a|'
#         r'write\s+a|create\s+a)\b', q_low
#     ):
#         return "FULL_SUMMARY"

#     # ── Signal 3: Chunk list structure ────────────────────────
#     list_chunk_count = 0
#     for chunk in chunks:
#         lines   = chunk.strip().split("\n")
#         numbered = sum(
#             1 for line in lines
#             if re.match(r'^\s*(\d+[\.\)]|[-•*])\s+\w', line.strip())
#         )
#         if numbered >= 2:
#             list_chunk_count += 1

#     # ── Signal 4: Information spread across chunks ────────────
#     stopwords     = _get_dynamic_stopwords()
#     content_words = [
#         w for w in words
#         if len(w) > 3 and w not in stopwords
#     ]

#     chunks_with_hits = sum(
#         1 for chunk in chunks
#         if sum(1 for w in content_words if w in chunk.lower()) >= 2
#     )

#     # ── Signal 5: Single-span answer likely ───────────────────
#     top_chunk_lines = chunks[0].strip().split("\n") if chunks else []
#     short_answer_lines = [
#         line for line in top_chunk_lines
#         if 2 <= len(line.split()) <= 8
#         and any(w in line.lower() for w in content_words)
#     ]
#     has_short_span = len(short_answer_lines) > 0

#     # ── Decision ──────────────────────────────────────────────
#     if list_chunk_count >= 2:
#         return "MULTIPART_QA"

#     if chunks_with_hits >= 3 and not has_short_span:
#         return "MULTIPART_QA"

#     if chunks_with_hits <= 2 and has_short_span:
#         return "FACTUAL_QA"

#     return "FACTUAL_QA"


























# same not at wall worjed 40% so laap claaude using now 
# import re
# from docmind_rag.utils.helpers import _get_dynamic_stopwords
# import spacy
# _nlp = spacy.load("en_core_web_lg")

# # ============================================================
# # POST-RETRIEVAL QUERY CLASSIFIER
# # No LLM call. No hardcoded domain words.
# # Classifies based on what the retrieved chunks actually contain.
# # Works for any document, any question type.
# # ============================================================
# def extract_named_entities(question: str, all_chunks: list = None) -> list[str]:

#     _QUESTION_WORDS = {
#         "what", "which", "who", "where", "when", "how",
#         "did", "does", "is", "are", "was", "were",
#         "can", "could", "should", "would"
#     }

#     def contains_entity(entity: str, text: str) -> bool:
#         return bool(re.search(
#             rf'\b{re.escape(entity)}\b', text, re.IGNORECASE
#         ))

#     def is_subpart(e: str, others: list) -> bool:
#         for other in others:
#             if e != other and re.search(
#                 rf'\b{re.escape(e)}\b', other, re.IGNORECASE
#             ):
#                 return True
#         return False

#     doc = _nlp(question)

#     # 1. spaCy named entities — multi-word names preserved
#     entities = [
#         ent.text for ent in doc.ents
#         if ent.label_ in ("PERSON", "ORG", "PRODUCT", "WORK_OF_ART")
#     ]

#     # 2. All-caps + technical model names — CNN, YOLOv5, XAI
#     # Requires 2+ uppercase chars at start to exclude "The", "System"
#     caps_tokens = [
#         t.text for t in doc
#         if re.match(r'^[A-Z]{2,}[a-zA-Z0-9]*$', t.text)
#         and len(t.text) > 1
#         and t.text.lower() not in _QUESTION_WORDS
#     ]

#     # 3. Proper nouns fallback — catches rare names spaCy misses
#     proper_nouns = [
#         t.text for t in doc
#         if t.pos_ == "PROPN"
#         and t.text.lower() not in _QUESTION_WORDS
#     ]

#     # Merge, deduplicate, minimum length filter
#     raw = list(set(entities + caps_tokens + proper_nouns))
#     candidates = [e for e in raw if len(e) > 1]

#     # Remove subparts when full multi-word entity exists
#     # Use word-boundary check to avoid "net" matching "network"
#     candidates = [
#         e for e in candidates
#         if not is_subpart(e, candidates)
#     ]

#     # Frequency filter
#     # Always keep multi-word names
#     # Filter single-word terms that appear in 80%+ of chunks
#     if all_chunks and candidates:
#         total = len(all_chunks)
#         candidates = [
#             e for e in candidates
#             if len(e.split()) > 1
#             or sum(1 for c in all_chunks
#                    if contains_entity(e, c)) < total * 0.8
#         ]

#     print(f"[EntityExtract] '{question[:50]}' → {candidates}")
#     return candidates
# import re

# def expand_answer(answer, retrieved_texts):
#     answer_clean = answer.strip().lower()

#     for chunk in retrieved_texts:
#         lines = chunk.split("\n")

#         for line in lines:
#             line_clean = line.strip().lower()

#             # If answer is part of a longer meaningful line
#             if answer_clean in line_clean and len(line_clean) > len(answer_clean) + 5:

#                 # Prefer title-like lines (all caps or structured)
#                 if line.strip().isupper() or len(line.split()) <= 10:
#                     return line.strip()

#     return answer
# def classify_from_context(question: str, chunks: list) -> str:
#     """
#     Determine answer type from retrieved chunks + question structure.
#     Called AFTER retrieval — the document tells us the answer shape.

#     Returns one of: FULL_SUMMARY | MULTIPART_QA | FACTUAL_QA | VERIFICATION_QA
#     """
#     q_low  = question.strip().lower()
#     words  = q_low.split()

#     # ── Signal 1: Verification — polar question structure ─────────────────
#     if re.match(
#         r'^(is|was|were|does|did|can|should|would|could|has|have|had|are|do|will)\b',
#         q_low
#     ):
#         return "VERIFICATION_QA"

#     # ── Signal 2: Summary — long declarative or explicit summary request ──
#     if (
#         len(words) > 12 and "?" not in question
#     ) or re.match(
#         r'^(summarize|summarise|give\s+a|provide\s+a|generate\s+a|'
#         r'write\s+a|create\s+a)\b', q_low
#     ):
#         return "FULL_SUMMARY"

#     # ── Signal 3: Chunk structure signals ─────────────────────────────────
#     # Count how many chunks contain numbered/bulleted lists
#     list_chunk_count = 0
#     for chunk in chunks:
#         lines = chunk.strip().split("\n")
#         numbered = sum(
#             1 for line in lines
#             if re.match(r'^\s*(\d+[\.\)]|[-•*])\s+\w', line.strip())
#         )
#         if numbered >= 2:
#             list_chunk_count += 1

#     # ── Signal 4: Information spread across chunks ────────────────────────
#     stopwords    = _get_dynamic_stopwords()
#     content_words = [
#         w for w in words
#         if len(w) > 3 and w not in stopwords
#     ]

#     chunks_with_hits = sum(
#         1 for chunk in chunks
#         if sum(1 for w in content_words if w in chunk.lower()) >= 2
#     )

#     # ── Signal 5: Single-span answer likely ───────────────────────────────
#     # Short question + top chunk has a short direct answer line
#     top_chunk_lines = chunks[0].strip().split("\n") if chunks else []
#     short_answer_lines = [
#         line for line in top_chunk_lines
#         if 2 <= len(line.split()) <= 8
#         and any(w in line.lower() for w in content_words)
#     ]
#     has_short_span = len(short_answer_lines) > 0

#     # ── Decision ──────────────────────────────────────────────────────────
#     # Multiple chunks with list structure → MULTIPART
#     if list_chunk_count >= 2:
#         return "MULTIPART_QA"

#     # Information spread across 3+ chunks → likely needs all of them → MULTIPART
#     if chunks_with_hits >= 3 and not has_short_span:
#         return "MULTIPART_QA"

#     # Single chunk dominates + short span exists → FACTUAL
#     if chunks_with_hits <= 2 and has_short_span:
#         return "FACTUAL_QA"

#     # Default → FACTUAL (let the universal prompt handle shape)
#     return "FACTUAL_QA"


# # ============================================================
# # TEXT NORMALIZATION
# # ============================================================

# def _normalize_text(text: str) -> str:
#     return re.sub(r'[^a-z0-9 ]', '', text.lower())


# def normalize_answer(answer: str) -> str:
#     if not answer:
#         return answer
#     ans = answer.lower()
#     ans = re.sub(r"\((.*?)\)", r"\1", ans)
#     ans = ans.replace(";", ",").replace("\n", ", ")
#     ans = re.sub(r"[^a-z0-9,\s]", " ", ans)
#     ans = re.sub(r"\s+", " ", ans).strip()
#     return ans


# # ============================================================
# # DOCUMENT TYPE DETECTION
# # ============================================================

# def detect_doc_type(text: str) -> str:
#     from docmind_rag.models.llm import call_llama
#     sample = text[:500].strip()
#     prompt = (
#         f"Classify this document into exactly one type: "
#         f"academic, legal, technical, or general.\n"
#         f"Reply with only one word.\n\nText:\n{sample}"
#     )
#     try:
#         result = call_llama(prompt, num_ctx=256, temperature=0.0).strip().lower()
#         for doc_type in ["academic", "legal", "technical", "general"]:
#             if doc_type in result:
#                 return doc_type
#     except Exception:
#         pass
#     return "general"


# # ============================================================
# # CONSTRAINT EXTRACTION
# # ============================================================

# def extract_constraints(question: str) -> dict:
#     return {
#         "numbers":  re.findall(r'\b\d+\b', question),
#         "keywords": [w.lower() for w in question.split() if len(w) > 3]
#     }


# def has_constraint_alignment(chunks: list, constraints: dict) -> bool:
#     numbers  = constraints["numbers"]
#     keywords = constraints["keywords"]

#     if not numbers:
#         return True

#     for chunk in chunks:
#         chunk_lower = chunk.lower()
#         for num in numbers:
#             for kw in keywords:
#                 pattern = (
#                     rf'\b{re.escape(kw)}\W+(?:\w+\W+){{0,3}}?{re.escape(num)}\b'
#                     rf'|\b{re.escape(num)}\W+(?:\w+\W+){{0,3}}?{re.escape(kw)}\b'
#                 )
#                 if re.search(pattern, chunk_lower):
#                     return True
#     return False

# def find_most_repeated_span(chunks: list, min_words: int = 3, max_words: int = 8) -> str:
#     from collections import Counter
#     span_counts = Counter()

#     for chunk in chunks:
#         clean = re.sub(r'^\s*\d+\s*[-—]+\s*[—-]?\s*', '', chunk, flags=re.MULTILINE)
#         sentences = re.split(r'[.!?\n]', clean)
#         for sent in sentences:
#             words = sent.strip().split()
#             for n in range(min_words, min(max_words + 1, len(words) + 1)):
#                 for i in range(len(words) - n + 1):
#                     span = " ".join(words[i:i+n]).strip()
#                     if span and len(span) > 10:
#                         span_counts[span.lower()] += 1

#     if not span_counts:
#         return ""

#     for span, count in span_counts.most_common(10):
#         if count >= 2:
#             return span
#     return ""

# # ============================================================
# # REORDER CHUNKS BY QUESTION RELEVANCE
# # ============================================================

# def reorder_by_question(question: str, chunks: list) -> list:
#     numbers   = re.findall(r'\b\d+\b', question)
#     stopwords = _get_dynamic_stopwords()
#     keywords  = [
#         w.lower() for w in question.split()
#         if len(w) > 3 and w.lower() not in stopwords
#     ]

#     if not numbers and not keywords:
#         return chunks

#     scored = []
#     for chunk in chunks:
#         chunk_lower = chunk.lower()
#         score       = 0

#         # Number + keyword proximity match
#         for num in numbers:
#             matched_proximity = False
#             for kw in keywords:
#                 pattern = (
#                     rf'\b{re.escape(kw)}\W+(?:\w+\W+){{0,3}}?{re.escape(num)}\b'
#                     rf'|\b{re.escape(num)}\W+(?:\w+\W+){{0,3}}?{re.escape(kw)}\b'
#                 )
#                 if re.search(pattern, chunk_lower):
#                     score          += 10
#                     matched_proximity = True
#                     break
#             if not matched_proximity:
#                 if re.search(rf'\b{re.escape(num)}\b', chunk_lower):
#                     score += 1
#                 else:
#                     score -= 3

#         # Title line keyword boost
#         first_line  = chunk.strip().split("\n")[0].lower()
#         title_hits  = sum(1 for w in keywords if w in first_line)
#         if title_hits >= max(2, len(keywords) // 2):
#             score += 5

#         # General keyword overlap
#         for kw in keywords:
#             if kw in chunk_lower:
#                 score += 1

#         scored.append((score, chunk))

#     scored.sort(key=lambda x: x[0], reverse=True)
#     # After scored.sort(key=lambda x: x[0], reverse=True)

#     # Tiebreaker: when top chunks have equal scores,
#     # boost chunks containing the most repeated short span
#     top_score = scored[0][0] if scored else 0
#     tied = [s for s in scored if s[0] == top_score]

#     if len(tied) > 1:
#         repeated = find_most_repeated_span([c for _, c in tied])
#         if repeated:
#             scored = [
#                 (s + 3 if repeated in c.lower() else s, c)
#                 for s, c in scored
#             ]
#             scored.sort(key=lambda x: x[0], reverse=True)
#             print(f"[Reorder] Tiebreaker applied — repeated span: '{repeated[:40]}'")
#     if scored and scored[0][0] > 0:
#         print(
#             f"[Reorder] top chunk score={scored[0][0]} | "
#             f"numbers={numbers} | keywords={keywords[:4]}"
#         )

#     return [c for _, c in scored]


# # ============================================================
# # NUMERIC REGEX FALLBACK
# # ============================================================

# def extract_numeric_answer(question: str, chunks: list) -> str:
#     num_pattern = re.compile(
#         r'\b(\d{1,3}(?:[–\-]\d{1,3})?(?:\.\d+)?'
#         r'(?:\s*(?:%|percent(?:age)?)))\b',
#         re.IGNORECASE
#     )
#     stopwords   = _get_dynamic_stopwords() | {
#         "estimate", "percentage", "probability", "much", "many"
#     }
#     q_keywords  = [
#         w.lower() for w in question.split()
#         if len(w) > 3 and w.lower() not in stopwords
#     ]

#     for chunk in chunks:
#         for sent in re.split(r'(?<=[.!?])\s+', chunk):
#             sent_lower  = sent.lower()
#             has_keyword = any(kw in sent_lower for kw in q_keywords)
#             match       = num_pattern.search(sent)
#             if has_keyword and match:
#                 return match.group(0).strip()
#     return ""






















# made somme changes yet did  not work so  leaviing it nd moving to  making fully  generic nd work using lap claude 
# import re
# from docmind_rag.utils.helpers import _get_dynamic_stopwords
# def extract_constraints(question: str) -> dict:
#     return {
#         "numbers": re.findall(r'\b\d+\b', question),
#         "keywords": [w.lower() for w in question.split() if len(w) > 3]
#     }


# def has_constraint_alignment(chunks: list, constraints: dict) -> bool:
#     numbers = constraints["numbers"]
#     keywords = constraints["keywords"]

#     # If no numeric constraint → always valid
#     if not numbers:
#         return True

#     for chunk in chunks:
#         chunk_lower = chunk.lower()

#         for num in numbers:
#             for kw in keywords:
#                 # generic proximity match (NO hardcoding)
#                 pattern = rf'\b{re.escape(kw)}\W+(?:\w+\W+){{0,3}}?{re.escape(num)}\b|\b{re.escape(num)}\W+(?:\w+\W+){{0,3}}?{re.escape(kw)}\b'
#                 if re.search(pattern, chunk_lower):
#                     return True

#     return False
# # ============================================================
# # FIX 6: TEXT NORMALIZATION
# # ============================================================
# def _normalize_text(text: str) -> str:
#     """Strip punctuation/case so 'Hinton.' matches 'hinton' etc."""
#     return re.sub(r'[^a-z0-9 ]', '', text.lower())
# # def is_structured_answer(chunk: str) -> bool: # this whole def is added to solve AI ethical aspects issue
# #     lines = chunk.split("\n")
# #     count = 0

# #     for line in lines:
# #         if re.match(r'^\s*\d+\.', line.strip()):
# #             count += 1

# #     return count >= 3

# # REPLACE the entire function body with:
# def normalize_answer(answer: str) -> str:
#     if not answer:
#         return answer

#     ans = answer.lower()

#     # remove bracket expansions but keep main term
#     ans = re.sub(r"\((.*?)\)", r"\1", ans)

#     # normalize separators (keep structure)
#     ans = ans.replace(";", ",")
#     ans = ans.replace("\n", ", ")

#     # remove extra symbols but KEEP commas
#     ans = re.sub(r"[^a-z0-9,\s]", " ", ans)

#     # normalize spaces
#     ans = re.sub(r"\s+", " ", ans).strip()

#     return ans

# def detect_doc_type(text: str) -> str:
#     # line ~15
#     from docmind_rag.models.llm import call_llama
#     sample = text[:500].strip()
#     prompt = (
#         f"Classify this document into exactly one type: academic, legal, technical, or general.\n"
#         f"Reply with only one word.\n\nText:\n{sample}"
#     )
#     try:
#         result = call_llama(prompt, num_ctx=256, temperature=0.0).strip().lower()
#         for doc_type in ["academic", "legal", "technical", "general"]:
#             if doc_type in result:
#                 return doc_type
#     except Exception:
#         pass
#     return "general"


# # ============================================================
# # FIX: FULLY DYNAMIC answer shape inference — no hardcoded word lists
# # Uses LLM for all decisions, heuristics only as a fast-path pre-filter
# # ============================================================
# def infer_answer_shape(question: str) -> dict:
#     """
#     Dynamically infer answer shape using structural signals + LLM verification.
#     No hardcoded starter/phrase word lists — structure is inferred from
#     grammar patterns and confirmed by LLM when uncertain.
#     """
#     # Import here to avoid circular dependency (llm → text → llm)
#     from docmind_rag.models.llm import call_llama

#     q     = question.strip()
#     q_low = q.lower()
#     words = q_low.split()
#     n     = len(words)

#     # ── Structural grammar signals (language-agnostic patterns) ──────────────
#     # List: plural interrogative + "are" at position 1, or imperative list verbs
#     is_list = (
#         n >= 3
#         and words[0] in {"what", "which"}
#         and words[1] == "are"
#     ) or (
#         n >= 2
#         and re.match(r'^(list|enumerate|name|identify)\b', q_low)
#     )

#     # Summary: long declarative (no ?) OR starts with generative/summary verb phrase
#     is_summary = (
#         n > 12 and "?" not in q
#     ) or (
#         n >= 3
#         and re.match(r'^(summarize|summarise|give\s+a|provide\s+a|generate\s+a|'
#                      r'write\s+a|create\s+a)\b', q_low)
#         and re.search(r'\b(summary|overview|brief|outline|recap)\b', q_low)
#     )

#     # Verification: polar question — starts with auxiliary verb
#     is_verification = bool(
#         re.match(r'^(is|was|were|does|did|can|should|would|could|has|have|had|'
#                  r'are|do|will|has|have)\b', q_low)
#     )

#     is_short = not is_list and not is_summary

#     # ── LLM tie-break for conflicts or low-confidence cases ──────────────────
#     conflict    = is_list and is_summary
#     low_conf    = not is_list and not is_summary and not is_verification and n > 6
#     need_llm    = conflict or low_conf

#     if need_llm:
#         print(f"[Shape] LLM disambiguation for: '{q[:60]}'")
#         prompt = (
#             f"Classify the ideal answer type for this question.\n"
#             f"Question: {question}\n\n"
#             f"A) Full document summary\n"
#             f"B) List of multiple items\n"
#             f"C) Yes/No verification with evidence\n"
#             f"D) Single fact or short answer\n\n"
#             f"Reply with only the letter A, B, C, or D."
#         )
#         try:
#             raw = call_llama(prompt, num_ctx=256, temperature=0.0).strip().upper()
#             letter = re.search(r'\b[ABCD]\b', raw)
#             if letter:
#                 ch = letter.group()
#                 is_summary      = (ch == "A")
#                 is_list         = (ch == "B")
#                 is_verification = (ch == "C")
#                 is_short        = (ch == "D")
#         except Exception as e:
#             print(f"[Shape] LLM failed: {e} — keeping structural result")

#     shape = {
#         "is_summary":      is_summary,
#         "is_list":         is_list,
#         "is_verification": is_verification,
#         "is_short":        is_short,
#     }
#     print(f"[Shape] {shape} for: '{q[:60]}'")
#     return shape


# def shape_to_query_type(shape: dict) -> str:
#     if shape.get("is_summary"):
#         return "FULL_SUMMARY"
#     if shape.get("is_list"):
#         return "MULTIPART_QA"
#     if shape.get("is_verification") or shape.get("is_short"):
#         return "FACTUAL_QA"
#     return "REASONING_QA"


# # ============================================================
# # FIX: FULLY DYNAMIC reorder_by_question — no hardcoded domain words
# # Scores chunks by number co-occurrence and content-word overlap only.
# # ============================================================
# def reorder_by_question(question: str, chunks: list) -> list:
#     """
#     Re-rank chunks so the most question-relevant ones come first.
#     Uses only numbers from the question and content words (no hardcoded domain terms).
#     """
#     numbers  = re.findall(r'\b\d+\b', question)
#     stopwords = _get_dynamic_stopwords()
#     keywords = [w.lower() for w in question.split()
#                 if len(w) > 3 and w.lower() not in stopwords]

#     if not numbers and not keywords:
#         return chunks

#     scored = []
#     for chunk in chunks:
#         chunk_lower = chunk.lower()
#         score = 0
#             # 🔥 FIX 3 — NUMBER DOMINANCE (ADD HERE) for the issue i got after fixing aspects of ai but lecture number s are afiled q6,q11
#         if numbers:
#             strong_match = False

#             for num in numbers:
#                 # 🔥 STRICT proximity match (generic)
#                 # number must appear NEAR a keyword (within 3 words)
#                 for kw in keywords:
#                     pattern = rf'\b{re.escape(kw)}\W+(?:\w+\W+){{0,3}}?{re.escape(num)}\b|\b{re.escape(num)}\W+(?:\w+\W+){{0,3}}?{re.escape(kw)}\b'
#                     if re.search(pattern, chunk_lower):
#                         score += 10   # strong semantic binding
#                         strong_match = True
#                         break
#                 else:
#                     # weak fallback if number exists alone
#                     if re.search(rf'\b{re.escape(num)}\b', chunk_lower):
#                         score += 1

#             if not strong_match:
#                 score -= 3   # penalize misleading matches
#         # this added to solve the ethicaa aspects of AI issue 
#         # 🔥 TITLE MATCH BOOST (NEW)
#         first_line = chunk.strip().split("\n")[0].lower()

#         title_hits = sum(1 for w in keywords if w in first_line)
#         # if title_hits >= 2: this worked fo  aspects fo ai but q6 nd q7 failed whihc both are like lecture 6 ,7 
#         #     score += 5
#         if title_hits >= max(2, len(keywords)//2):
#             score += 5

#         # 🔥 STRUCTURED ANSWER BOOST (NEW)
#         # if is_structured_answer(chunk): same this  also worked fo  apspects but for those failed 
#         #     score += 3
#         # if is_structured_answer(chunk) and len(numbers) == 0:
#         #     score += 3
#         #----end-------

#         # Number matches: score higher when a number appears near other keywords
#         # REPLACE the entire "Number matches" inner loop (lines 141-151) with:

#         for num in numbers:
#             for kw in keywords:
#                 # Match compound phrase like "lecture 3" or "chapter 11"
#                 pattern = rf'\b{re.escape(kw)}\s*{re.escape(num)}\b|\b{re.escape(num)}\s*{re.escape(kw)}\b'
#                 if re.search(pattern, chunk_lower):
#                     score += 5  # strong compound match
#                     break
#             else:
#                 # Number appears but not next to any keyword — weak signal
#                 if re.search(rf'\b{re.escape(num)}\b', chunk_lower):
#                     score += 1

#         for kw in keywords:
#             if kw in chunk_lower:
#                 score += 1

#         scored.append((score, chunk))

#     scored.sort(key=lambda x: x[0], reverse=True)
#     reordered = [c for _, c in scored]

#     if scored and scored[0][0] > 0:
#         print(f"[Reorder] top chunk score={scored[0][0]} | "
#               f"numbers={numbers} | keywords={keywords[:4]}")
#     return reordered


# # ============================================================
# # NUMERIC REGEX FALLBACK
# # ============================================================
# def extract_numeric_answer(question: str, chunks: list) -> str:
#     num_pattern = re.compile(
#         r'\b(\d{1,3}(?:[–\-]\d{1,3})?(?:\.\d+)?(?:\s*(?:%|percent(?:age)?)))\b',
#         re.IGNORECASE
#     )
#     # FIX: dynamic stopwords instead of hardcoded list
#     stopwords = _get_dynamic_stopwords() | {"estimate", "percentage", "probability", "much", "many"}
#     q_keywords = [w.lower() for w in question.split()
#                   if len(w) > 3 and w.lower() not in stopwords]

#     for chunk in chunks:
#         sentences = re.split(r'(?<=[.!?])\s+', chunk)
#         for sent in sentences:
#             sent_lower = sent.lower()
#             has_keyword = any(kw in sent_lower for kw in q_keywords)
#             match = num_pattern.search(sent)
#             if has_keyword and match:
#                 return match.group(0).strip()
#     return ""






















# the one thaat returned techquies of the system correctly but above codes adding validation fix ot make it genereic like (CNNs) become cnns like this 
# import re
# from docmind_rag.utils.helpers import _get_dynamic_stopwords
# def extract_constraints(question: str) -> dict:
#     return {
#         "numbers": re.findall(r'\b\d+\b', question),
#         "keywords": [w.lower() for w in question.split() if len(w) > 3]
#     }


# def has_constraint_alignment(chunks: list, constraints: dict) -> bool:
#     numbers = constraints["numbers"]
#     keywords = constraints["keywords"]

#     # If no numeric constraint → always valid
#     if not numbers:
#         return True

#     for chunk in chunks:
#         chunk_lower = chunk.lower()

#         for num in numbers:
#             for kw in keywords:
#                 # generic proximity match (NO hardcoding)
#                 pattern = rf'\b{re.escape(kw)}\W+(?:\w+\W+){{0,3}}?{re.escape(num)}\b|\b{re.escape(num)}\W+(?:\w+\W+){{0,3}}?{re.escape(kw)}\b'
#                 if re.search(pattern, chunk_lower):
#                     return True

#     return False
# # ============================================================
# # FIX 6: TEXT NORMALIZATION
# # ============================================================
# def _normalize_text(text: str) -> str:
#     """Strip punctuation/case so 'Hinton.' matches 'hinton' etc."""
#     return re.sub(r'[^a-z0-9 ]', '', text.lower())
# # def is_structured_answer(chunk: str) -> bool: # this whole def is added to solve AI ethical aspects issue
# #     lines = chunk.split("\n")
# #     count = 0

# #     for line in lines:
# #         if re.match(r'^\s*\d+\.', line.strip()):
# #             count += 1

# #     return count >= 3



# def detect_doc_type(text: str) -> str:
#     # line ~15
#     from docmind_rag.models.llm import call_llama
#     sample = text[:500].strip()
#     prompt = (
#         f"Classify this document into exactly one type: academic, legal, technical, or general.\n"
#         f"Reply with only one word.\n\nText:\n{sample}"
#     )
#     try:
#         result = call_llama(prompt, num_ctx=256, temperature=0.0).strip().lower()
#         for doc_type in ["academic", "legal", "technical", "general"]:
#             if doc_type in result:
#                 return doc_type
#     except Exception:
#         pass
#     return "general"


# # ============================================================
# # FIX: FULLY DYNAMIC answer shape inference — no hardcoded word lists
# # Uses LLM for all decisions, heuristics only as a fast-path pre-filter
# # ============================================================
# def infer_answer_shape(question: str) -> dict:
#     """
#     Dynamically infer answer shape using structural signals + LLM verification.
#     No hardcoded starter/phrase word lists — structure is inferred from
#     grammar patterns and confirmed by LLM when uncertain.
#     """
#     # Import here to avoid circular dependency (llm → text → llm)
#     from docmind_rag.models.llm import call_llama

#     q     = question.strip()
#     q_low = q.lower()
#     words = q_low.split()
#     n     = len(words)

#     # ── Structural grammar signals (language-agnostic patterns) ──────────────
#     # List: plural interrogative + "are" at position 1, or imperative list verbs
#     is_list = (
#         n >= 3
#         and words[0] in {"what", "which"}
#         and words[1] == "are"
#     ) or (
#         n >= 2
#         and re.match(r'^(list|enumerate|name|identify)\b', q_low)
#     )

#     # Summary: long declarative (no ?) OR starts with generative/summary verb phrase
#     is_summary = (
#         n > 12 and "?" not in q
#     ) or (
#         n >= 3
#         and re.match(r'^(summarize|summarise|give\s+a|provide\s+a|generate\s+a|'
#                      r'write\s+a|create\s+a)\b', q_low)
#         and re.search(r'\b(summary|overview|brief|outline|recap)\b', q_low)
#     )

#     # Verification: polar question — starts with auxiliary verb
#     is_verification = bool(
#         re.match(r'^(is|was|were|does|did|can|should|would|could|has|have|had|'
#                  r'are|do|will|has|have)\b', q_low)
#     )

#     is_short = not is_list and not is_summary

#     # ── LLM tie-break for conflicts or low-confidence cases ──────────────────
#     conflict    = is_list and is_summary
#     low_conf    = not is_list and not is_summary and not is_verification and n > 6
#     need_llm    = conflict or low_conf

#     if need_llm:
#         print(f"[Shape] LLM disambiguation for: '{q[:60]}'")
#         prompt = (
#             f"Classify the ideal answer type for this question.\n"
#             f"Question: {question}\n\n"
#             f"A) Full document summary\n"
#             f"B) List of multiple items\n"
#             f"C) Yes/No verification with evidence\n"
#             f"D) Single fact or short answer\n\n"
#             f"Reply with only the letter A, B, C, or D."
#         )
#         try:
#             raw = call_llama(prompt, num_ctx=256, temperature=0.0).strip().upper()
#             letter = re.search(r'\b[ABCD]\b', raw)
#             if letter:
#                 ch = letter.group()
#                 is_summary      = (ch == "A")
#                 is_list         = (ch == "B")
#                 is_verification = (ch == "C")
#                 is_short        = (ch == "D")
#         except Exception as e:
#             print(f"[Shape] LLM failed: {e} — keeping structural result")

#     shape = {
#         "is_summary":      is_summary,
#         "is_list":         is_list,
#         "is_verification": is_verification,
#         "is_short":        is_short,
#     }
#     print(f"[Shape] {shape} for: '{q[:60]}'")
#     return shape


# def shape_to_query_type(shape: dict) -> str:
#     if shape.get("is_summary"):
#         return "FULL_SUMMARY"
#     if shape.get("is_list"):
#         return "MULTIPART_QA"
#     if shape.get("is_verification") or shape.get("is_short"):
#         return "FACTUAL_QA"
#     return "REASONING_QA"


# # ============================================================
# # FIX: FULLY DYNAMIC reorder_by_question — no hardcoded domain words
# # Scores chunks by number co-occurrence and content-word overlap only.
# # ============================================================
# def reorder_by_question(question: str, chunks: list) -> list:
#     """
#     Re-rank chunks so the most question-relevant ones come first.
#     Uses only numbers from the question and content words (no hardcoded domain terms).
#     """
#     numbers  = re.findall(r'\b\d+\b', question)
#     stopwords = _get_dynamic_stopwords()
#     keywords = [w.lower() for w in question.split()
#                 if len(w) > 3 and w.lower() not in stopwords]

#     if not numbers and not keywords:
#         return chunks

#     scored = []
#     for chunk in chunks:
#         chunk_lower = chunk.lower()
#         score = 0
#             # 🔥 FIX 3 — NUMBER DOMINANCE (ADD HERE) for the issue i got after fixing aspects of ai but lecture number s are afiled q6,q11
#         if numbers:
#             strong_match = False

#             for num in numbers:
#                 # 🔥 STRICT proximity match (generic)
#                 # number must appear NEAR a keyword (within 3 words)
#                 for kw in keywords:
#                     pattern = rf'\b{re.escape(kw)}\W+(?:\w+\W+){{0,3}}?{re.escape(num)}\b|\b{re.escape(num)}\W+(?:\w+\W+){{0,3}}?{re.escape(kw)}\b'
#                     if re.search(pattern, chunk_lower):
#                         score += 10   # strong semantic binding
#                         strong_match = True
#                         break
#                 else:
#                     # weak fallback if number exists alone
#                     if re.search(rf'\b{re.escape(num)}\b', chunk_lower):
#                         score += 1

#             if not strong_match:
#                 score -= 3   # penalize misleading matches
#         # this added to solve the ethicaa aspects of AI issue 
#         # 🔥 TITLE MATCH BOOST (NEW)
#         first_line = chunk.strip().split("\n")[0].lower()

#         title_hits = sum(1 for w in keywords if w in first_line)
#         # if title_hits >= 2: this worked fo  aspects fo ai but q6 nd q7 failed whihc both are like lecture 6 ,7 
#         #     score += 5
#         if title_hits >= max(2, len(keywords)//2):
#             score += 5

#         # 🔥 STRUCTURED ANSWER BOOST (NEW)
#         # if is_structured_answer(chunk): same this  also worked fo  apspects but for those failed 
#         #     score += 3
#         # if is_structured_answer(chunk) and len(numbers) == 0:
#         #     score += 3
#         #----end-------

#         # Number matches: score higher when a number appears near other keywords
#         # REPLACE the entire "Number matches" inner loop (lines 141-151) with:

#         for num in numbers:
#             for kw in keywords:
#                 # Match compound phrase like "lecture 3" or "chapter 11"
#                 pattern = rf'\b{re.escape(kw)}\s*{re.escape(num)}\b|\b{re.escape(num)}\s*{re.escape(kw)}\b'
#                 if re.search(pattern, chunk_lower):
#                     score += 5  # strong compound match
#                     break
#             else:
#                 # Number appears but not next to any keyword — weak signal
#                 if re.search(rf'\b{re.escape(num)}\b', chunk_lower):
#                     score += 1

#         for kw in keywords:
#             if kw in chunk_lower:
#                 score += 1

#         scored.append((score, chunk))

#     scored.sort(key=lambda x: x[0], reverse=True)
#     reordered = [c for _, c in scored]

#     if scored and scored[0][0] > 0:
#         print(f"[Reorder] top chunk score={scored[0][0]} | "
#               f"numbers={numbers} | keywords={keywords[:4]}")
#     return reordered


# # ============================================================
# # NUMERIC REGEX FALLBACK
# # ============================================================
# def extract_numeric_answer(question: str, chunks: list) -> str:
#     num_pattern = re.compile(
#         r'\b(\d{1,3}(?:[–\-]\d{1,3})?(?:\.\d+)?(?:\s*(?:%|percent(?:age)?)))\b',
#         re.IGNORECASE
#     )
#     # FIX: dynamic stopwords instead of hardcoded list
#     stopwords = _get_dynamic_stopwords() | {"estimate", "percentage", "probability", "much", "many"}
#     q_keywords = [w.lower() for w in question.split()
#                   if len(w) > 3 and w.lower() not in stopwords]

#     for chunk in chunks:
#         sentences = re.split(r'(?<=[.!?])\s+', chunk)
#         for sent in sentences:
#             sent_lower = sent.lower()
#             has_keyword = any(kw in sent_lower for kw in q_keywords)
#             match = num_pattern.search(sent)
#             if has_keyword and match:
#                 return match.group(0).strip()
#     return ""























# got 100 for ai  but 57 fro croop -  overfitt
# import re
# from docmind_rag.utils.helpers import _get_dynamic_stopwords
# def extract_constraints(question: str) -> dict:
#     return {
#         "numbers": re.findall(r'\b\d+\b', question),
#         "keywords": [w.lower() for w in question.split() if len(w) > 3]
#     }


# def has_constraint_alignment(chunks: list, constraints: dict) -> bool:
#     numbers = constraints["numbers"]
#     keywords = constraints["keywords"]

#     # If no numeric constraint → always valid
#     if not numbers:
#         return True

#     for chunk in chunks:
#         chunk_lower = chunk.lower()

#         for num in numbers:
#             for kw in keywords:
#                 # generic proximity match (NO hardcoding)
#                 pattern = rf'\b{re.escape(kw)}\W+(?:\w+\W+){{0,3}}?{re.escape(num)}\b|\b{re.escape(num)}\W+(?:\w+\W+){{0,3}}?{re.escape(kw)}\b'
#                 if re.search(pattern, chunk_lower):
#                     return True

#     return False
# # ============================================================
# # FIX 6: TEXT NORMALIZATION
# # ============================================================
# def _normalize_text(text: str) -> str:
#     """Strip punctuation/case so 'Hinton.' matches 'hinton' etc."""
#     return re.sub(r'[^a-z0-9 ]', '', text.lower())
# def is_structured_answer(chunk: str) -> bool: # this whole def is added to solve AI ethical aspects issue
#     lines = chunk.split("\n")
#     count = 0

#     for line in lines:
#         if re.match(r'^\s*\d+\.', line.strip()):
#             count += 1

#     return count >= 3

# # REPLACE the entire function body with:

# def detect_doc_type(text: str) -> str:
#     # line ~15
#     from docmind_rag.models.llm import call_llama
#     sample = text[:500].strip()
#     prompt = (
#         f"Classify this document into exactly one type: academic, legal, technical, or general.\n"
#         f"Reply with only one word.\n\nText:\n{sample}"
#     )
#     try:
#         result = call_llama(prompt, num_ctx=256, temperature=0.0).strip().lower()
#         for doc_type in ["academic", "legal", "technical", "general"]:
#             if doc_type in result:
#                 return doc_type
#     except Exception:
#         pass
#     return "general"


# # ============================================================
# # FIX: FULLY DYNAMIC answer shape inference — no hardcoded word lists
# # Uses LLM for all decisions, heuristics only as a fast-path pre-filter
# # ============================================================
# def infer_answer_shape(question: str) -> dict:
#     """
#     Dynamically infer answer shape using structural signals + LLM verification.
#     No hardcoded starter/phrase word lists — structure is inferred from
#     grammar patterns and confirmed by LLM when uncertain.
#     """
#     # Import here to avoid circular dependency (llm → text → llm)
#     from docmind_rag.models.llm import call_llama

#     q     = question.strip()
#     q_low = q.lower()
#     words = q_low.split()
#     n     = len(words)

#     # ── Structural grammar signals (language-agnostic patterns) ──────────────
#     # List: plural interrogative + "are" at position 1, or imperative list verbs
#     is_list = (
#         n >= 3
#         and words[0] in {"what", "which"}
#         and words[1] == "are"
#     ) or (
#         n >= 2
#         and re.match(r'^(list|enumerate|name|identify)\b', q_low)
#     )

#     # Summary: long declarative (no ?) OR starts with generative/summary verb phrase
#     is_summary = (
#         n > 12 and "?" not in q
#     ) or (
#         n >= 3
#         and re.match(r'^(summarize|summarise|give\s+a|provide\s+a|generate\s+a|'
#                      r'write\s+a|create\s+a)\b', q_low)
#         and re.search(r'\b(summary|overview|brief|outline|recap)\b', q_low)
#     )

#     # Verification: polar question — starts with auxiliary verb
#     is_verification = bool(
#         re.match(r'^(is|was|were|does|did|can|should|would|could|has|have|had|'
#                  r'are|do|will|has|have)\b', q_low)
#     )

#     is_short = not is_list and not is_summary

#     # ── LLM tie-break for conflicts or low-confidence cases ──────────────────
#     conflict    = is_list and is_summary
#     low_conf    = not is_list and not is_summary and not is_verification and n > 6
#     need_llm    = conflict or low_conf

#     if need_llm:
#         print(f"[Shape] LLM disambiguation for: '{q[:60]}'")
#         prompt = (
#             f"Classify the ideal answer type for this question.\n"
#             f"Question: {question}\n\n"
#             f"A) Full document summary\n"
#             f"B) List of multiple items\n"
#             f"C) Yes/No verification with evidence\n"
#             f"D) Single fact or short answer\n\n"
#             f"Reply with only the letter A, B, C, or D."
#         )
#         try:
#             raw = call_llama(prompt, num_ctx=256, temperature=0.0).strip().upper()
#             letter = re.search(r'\b[ABCD]\b', raw)
#             if letter:
#                 ch = letter.group()
#                 is_summary      = (ch == "A")
#                 is_list         = (ch == "B")
#                 is_verification = (ch == "C")
#                 is_short        = (ch == "D")
#         except Exception as e:
#             print(f"[Shape] LLM failed: {e} — keeping structural result")

#     shape = {
#         "is_summary":      is_summary,
#         "is_list":         is_list,
#         "is_verification": is_verification,
#         "is_short":        is_short,
#     }
#     print(f"[Shape] {shape} for: '{q[:60]}'")
#     return shape


# def shape_to_query_type(shape: dict) -> str:
#     if shape.get("is_summary"):
#         return "FULL_SUMMARY"
#     if shape.get("is_list"):
#         return "MULTIPART_QA"
#     if shape.get("is_verification") or shape.get("is_short"):
#         return "FACTUAL_QA"
#     return "REASONING_QA"


# # ============================================================
# # FIX: FULLY DYNAMIC reorder_by_question — no hardcoded domain words
# # Scores chunks by number co-occurrence and content-word overlap only.
# # ============================================================
# def reorder_by_question(question: str, chunks: list) -> list:
#     """
#     Re-rank chunks so the most question-relevant ones come first.
#     Uses only numbers from the question and content words (no hardcoded domain terms).
#     """
#     numbers  = re.findall(r'\b\d+\b', question)
#     stopwords = _get_dynamic_stopwords()
#     keywords = [w.lower() for w in question.split()
#                 if len(w) > 3 and w.lower() not in stopwords]

#     if not numbers and not keywords:
#         return chunks

#     scored = []
#     for chunk in chunks:
#         chunk_lower = chunk.lower()
#         score = 0
#             # 🔥 FIX 3 — NUMBER DOMINANCE (ADD HERE) for the issue i got after fixing aspects of ai but lecture number s are afiled q6,q11
#         if numbers:
#             strong_match = False

#             for num in numbers:
#                 # 🔥 STRICT proximity match (generic)
#                 # number must appear NEAR a keyword (within 3 words)
#                 for kw in keywords:
#                     pattern = rf'\b{re.escape(kw)}\W+(?:\w+\W+){{0,3}}?{re.escape(num)}\b|\b{re.escape(num)}\W+(?:\w+\W+){{0,3}}?{re.escape(kw)}\b'
#                     if re.search(pattern, chunk_lower):
#                         score += 10   # strong semantic binding
#                         strong_match = True
#                         break
#                 else:
#                     # weak fallback if number exists alone
#                     if re.search(rf'\b{re.escape(num)}\b', chunk_lower):
#                         score += 1

#             if not strong_match:
#                 score -= 3   # penalize misleading matches
#         # this added to solve the ethicaa aspects of AI issue 
#         # 🔥 TITLE MATCH BOOST (NEW)
#         first_line = chunk.strip().split("\n")[0].lower()

#         title_hits = sum(1 for w in keywords if w in first_line)
#         # if title_hits >= 2: this worked fo  aspects fo ai but q6 nd q7 failed whihc both are like lecture 6 ,7 
#         #     score += 5
#         if title_hits >= max(2, len(keywords)//2):
#             score += 5

#         # 🔥 STRUCTURED ANSWER BOOST (NEW)
#         # if is_structured_answer(chunk): same this  also worked fo  apspects but for those failed 
#         #     score += 3
#         if is_structured_answer(chunk) and len(numbers) == 0:
#             score += 3
#         #----end-------

#         # Number matches: score higher when a number appears near other keywords
#         # REPLACE the entire "Number matches" inner loop (lines 141-151) with:

#         for num in numbers:
#             for kw in keywords:
#                 # Match compound phrase like "lecture 3" or "chapter 11"
#                 pattern = rf'\b{re.escape(kw)}\s*{re.escape(num)}\b|\b{re.escape(num)}\s*{re.escape(kw)}\b'
#                 if re.search(pattern, chunk_lower):
#                     score += 5  # strong compound match
#                     break
#             else:
#                 # Number appears but not next to any keyword — weak signal
#                 if re.search(rf'\b{re.escape(num)}\b', chunk_lower):
#                     score += 1

#         for kw in keywords:
#             if kw in chunk_lower:
#                 score += 1

#         scored.append((score, chunk))

#     scored.sort(key=lambda x: x[0], reverse=True)
#     reordered = [c for _, c in scored]

#     if scored and scored[0][0] > 0:
#         print(f"[Reorder] top chunk score={scored[0][0]} | "
#               f"numbers={numbers} | keywords={keywords[:4]}")
#     return reordered


# # ============================================================
# # NUMERIC REGEX FALLBACK
# # ============================================================
# def extract_numeric_answer(question: str, chunks: list) -> str:
#     num_pattern = re.compile(
#         r'\b(\d{1,3}(?:[–\-]\d{1,3})?(?:\.\d+)?(?:\s*(?:%|percent(?:age)?)))\b',
#         re.IGNORECASE
#     )
#     # FIX: dynamic stopwords instead of hardcoded list
#     stopwords = _get_dynamic_stopwords() | {"estimate", "percentage", "probability", "much", "many"}
#     q_keywords = [w.lower() for w in question.split()
#                   if len(w) > 3 and w.lower() not in stopwords]

#     for chunk in chunks:
#         sentences = re.split(r'(?<=[.!?])\s+', chunk)
#         for sent in sentences:
#             sent_lower = sent.lower()
#             has_keyword = any(kw in sent_lower for kw in q_keywords)
#             match = num_pattern.search(sent)
#             if has_keyword and match:
#                 return match.group(0).strip()
#     return ""





























# # fixed ai aspects one but failed for that lecture questions 
# import re
# from docmind_rag.utils.helpers import _get_dynamic_stopwords

# # ============================================================
# # FIX 6: TEXT NORMALIZATION
# # ============================================================
# def _normalize_text(text: str) -> str:
#     """Strip punctuation/case so 'Hinton.' matches 'hinton' etc."""
#     return re.sub(r'[^a-z0-9 ]', '', text.lower())
# def is_structured_answer(chunk: str) -> bool: # this whole def is added to solve AI ethical aspects issue
#     lines = chunk.split("\n")
#     count = 0

#     for line in lines:
#         if re.match(r'^\s*\d+\.', line.strip()):
#             count += 1

#     return count >= 3

# # REPLACE the entire function body with:

# def detect_doc_type(text: str) -> str:
#     # line ~15
#     from docmind_rag.models.llm import call_llama
#     sample = text[:500].strip()
#     prompt = (
#         f"Classify this document into exactly one type: academic, legal, technical, or general.\n"
#         f"Reply with only one word.\n\nText:\n{sample}"
#     )
#     try:
#         result = call_llama(prompt, num_ctx=256, temperature=0.0).strip().lower()
#         for doc_type in ["academic", "legal", "technical", "general"]:
#             if doc_type in result:
#                 return doc_type
#     except Exception:
#         pass
#     return "general"


# # ============================================================
# # FIX: FULLY DYNAMIC answer shape inference — no hardcoded word lists
# # Uses LLM for all decisions, heuristics only as a fast-path pre-filter
# # ============================================================
# def infer_answer_shape(question: str) -> dict:
#     """
#     Dynamically infer answer shape using structural signals + LLM verification.
#     No hardcoded starter/phrase word lists — structure is inferred from
#     grammar patterns and confirmed by LLM when uncertain.
#     """
#     # Import here to avoid circular dependency (llm → text → llm)
#     from docmind_rag.models.llm import call_llama

#     q     = question.strip()
#     q_low = q.lower()
#     words = q_low.split()
#     n     = len(words)

#     # ── Structural grammar signals (language-agnostic patterns) ──────────────
#     # List: plural interrogative + "are" at position 1, or imperative list verbs
#     is_list = (
#         n >= 3
#         and words[0] in {"what", "which"}
#         and words[1] == "are"
#     ) or (
#         n >= 2
#         and re.match(r'^(list|enumerate|name|identify)\b', q_low)
#     )

#     # Summary: long declarative (no ?) OR starts with generative/summary verb phrase
#     is_summary = (
#         n > 12 and "?" not in q
#     ) or (
#         n >= 3
#         and re.match(r'^(summarize|summarise|give\s+a|provide\s+a|generate\s+a|'
#                      r'write\s+a|create\s+a)\b', q_low)
#         and re.search(r'\b(summary|overview|brief|outline|recap)\b', q_low)
#     )

#     # Verification: polar question — starts with auxiliary verb
#     is_verification = bool(
#         re.match(r'^(is|was|were|does|did|can|should|would|could|has|have|had|'
#                  r'are|do|will|has|have)\b', q_low)
#     )

#     is_short = not is_list and not is_summary

#     # ── LLM tie-break for conflicts or low-confidence cases ──────────────────
#     conflict    = is_list and is_summary
#     low_conf    = not is_list and not is_summary and not is_verification and n > 6
#     need_llm    = conflict or low_conf

#     if need_llm:
#         print(f"[Shape] LLM disambiguation for: '{q[:60]}'")
#         prompt = (
#             f"Classify the ideal answer type for this question.\n"
#             f"Question: {question}\n\n"
#             f"A) Full document summary\n"
#             f"B) List of multiple items\n"
#             f"C) Yes/No verification with evidence\n"
#             f"D) Single fact or short answer\n\n"
#             f"Reply with only the letter A, B, C, or D."
#         )
#         try:
#             raw = call_llama(prompt, num_ctx=256, temperature=0.0).strip().upper()
#             letter = re.search(r'\b[ABCD]\b', raw)
#             if letter:
#                 ch = letter.group()
#                 is_summary      = (ch == "A")
#                 is_list         = (ch == "B")
#                 is_verification = (ch == "C")
#                 is_short        = (ch == "D")
#         except Exception as e:
#             print(f"[Shape] LLM failed: {e} — keeping structural result")

#     shape = {
#         "is_summary":      is_summary,
#         "is_list":         is_list,
#         "is_verification": is_verification,
#         "is_short":        is_short,
#     }
#     print(f"[Shape] {shape} for: '{q[:60]}'")
#     return shape


# def shape_to_query_type(shape: dict) -> str:
#     if shape.get("is_summary"):
#         return "FULL_SUMMARY"
#     if shape.get("is_list"):
#         return "MULTIPART_QA"
#     if shape.get("is_verification") or shape.get("is_short"):
#         return "FACTUAL_QA"
#     return "REASONING_QA"


# # ============================================================
# # FIX: FULLY DYNAMIC reorder_by_question — no hardcoded domain words
# # Scores chunks by number co-occurrence and content-word overlap only.
# # ============================================================
# def reorder_by_question(question: str, chunks: list) -> list:
#     """
#     Re-rank chunks so the most question-relevant ones come first.
#     Uses only numbers from the question and content words (no hardcoded domain terms).
#     """
#     numbers  = re.findall(r'\b\d+\b', question)
#     stopwords = _get_dynamic_stopwords()
#     keywords = [w.lower() for w in question.split()
#                 if len(w) > 3 and w.lower() not in stopwords]

#     if not numbers and not keywords:
#         return chunks

#     scored = []
#     for chunk in chunks:
#         chunk_lower = chunk.lower()
#         score = 0
#             # 🔥 FIX 3 — NUMBER DOMINANCE (ADD HERE) for the issue i got after fixing aspects of ai but lecture number s are afiled q6,q11
#         if numbers:
#             if any(re.search(rf'\b{num}\b', chunk_lower) for num in numbers):
#                 score += 8   # strong positive signal
#             else:
#                 score -= 4   # penalize wrong-number chunks
#         # this added to solve the ethicaa aspects of AI issue 
#         # 🔥 TITLE MATCH BOOST (NEW)
#         first_line = chunk.strip().split("\n")[0].lower()

#         title_hits = sum(1 for w in keywords if w in first_line)
#         # if title_hits >= 2: this worked fo  aspects fo ai but q6 nd q7 failed whihc both are like lecture 6 ,7 
#         #     score += 5
#         if title_hits >= max(2, len(keywords)//2):
#             score += 5

#         # 🔥 STRUCTURED ANSWER BOOST (NEW)
#         # if is_structured_answer(chunk): same this  also worked fo  apspects but for those failed 
#         #     score += 3
#         if is_structured_answer(chunk) and len(numbers) == 0:
#             score += 3
#         #----end-------

#         # Number matches: score higher when a number appears near other keywords
#         # REPLACE the entire "Number matches" inner loop (lines 141-151) with:

#         for num in numbers:
#             for kw in keywords:
#                 # Match compound phrase like "lecture 3" or "chapter 11"
#                 pattern = rf'\b{re.escape(kw)}\s*{re.escape(num)}\b|\b{re.escape(num)}\s*{re.escape(kw)}\b'
#                 if re.search(pattern, chunk_lower):
#                     score += 5  # strong compound match
#                     break
#             else:
#                 # Number appears but not next to any keyword — weak signal
#                 if re.search(rf'\b{re.escape(num)}\b', chunk_lower):
#                     score += 1

#         for kw in keywords:
#             if kw in chunk_lower:
#                 score += 1

#         scored.append((score, chunk))

#     scored.sort(key=lambda x: x[0], reverse=True)
#     reordered = [c for _, c in scored]

#     if scored and scored[0][0] > 0:
#         print(f"[Reorder] top chunk score={scored[0][0]} | "
#               f"numbers={numbers} | keywords={keywords[:4]}")
#     return reordered


# # ============================================================
# # NUMERIC REGEX FALLBACK
# # ============================================================
# def extract_numeric_answer(question: str, chunks: list) -> str:
#     num_pattern = re.compile(
#         r'\b(\d{1,3}(?:[–\-]\d{1,3})?(?:\.\d+)?(?:\s*(?:%|percent(?:age)?)))\b',
#         re.IGNORECASE
#     )
#     # FIX: dynamic stopwords instead of hardcoded list
#     stopwords = _get_dynamic_stopwords() | {"estimate", "percentage", "probability", "much", "many"}
#     q_keywords = [w.lower() for w in question.split()
#                   if len(w) > 3 and w.lower() not in stopwords]

#     for chunk in chunks:
#         sentences = re.split(r'(?<=[.!?])\s+', chunk)
#         for sent in sentences:
#             sent_lower = sent.lower()
#             has_keyword = any(kw in sent_lower for kw in q_keywords)
#             match = num_pattern.search(sent)
#             if has_keyword and match:
#                 return match.group(0).strip()
#     return ""













# # # got   9 pass one 1 partial whihc is that aspects of ai 
# import re
# from docmind_rag.utils.helpers import _get_dynamic_stopwords

# # ============================================================
# # FIX 6: TEXT NORMALIZATION
# # ============================================================
# def _normalize_text(text: str) -> str:
#     """Strip punctuation/case so 'Hinton.' matches 'hinton' etc."""
#     return re.sub(r'[^a-z0-9 ]', '', text.lower())

# # REPLACE the entire function body with:

# def detect_doc_type(text: str) -> str:
#     # line ~15
#     from docmind_rag.models.llm import call_llama
#     sample = text[:500].strip()
#     prompt = (
#         f"Classify this document into exactly one type: academic, legal, technical, or general.\n"
#         f"Reply with only one word.\n\nText:\n{sample}"
#     )
#     try:
#         result = call_llama(prompt, num_ctx=256, temperature=0.0).strip().lower()
#         for doc_type in ["academic", "legal", "technical", "general"]:
#             if doc_type in result:
#                 return doc_type
#     except Exception:
#         pass
#     return "general"


# # ============================================================
# # FIX: FULLY DYNAMIC answer shape inference — no hardcoded word lists
# # Uses LLM for all decisions, heuristics only as a fast-path pre-filter
# # ============================================================
# def infer_answer_shape(question: str) -> dict:
#     """
#     Dynamically infer answer shape using structural signals + LLM verification.
#     No hardcoded starter/phrase word lists — structure is inferred from
#     grammar patterns and confirmed by LLM when uncertain.
#     """
#     # Import here to avoid circular dependency (llm → text → llm)
#     from docmind_rag.models.llm import call_llama

#     q     = question.strip()
#     q_low = q.lower()
#     words = q_low.split()
#     n     = len(words)

#     # ── Structural grammar signals (language-agnostic patterns) ──────────────
#     # List: plural interrogative + "are" at position 1, or imperative list verbs
#     is_list = (
#         n >= 3
#         and words[0] in {"what", "which"}
#         and words[1] == "are"
#     ) or (
#         n >= 2
#         and re.match(r'^(list|enumerate|name|identify)\b', q_low)
#     )

#     # Summary: long declarative (no ?) OR starts with generative/summary verb phrase
#     is_summary = (
#         n > 12 and "?" not in q
#     ) or (
#         n >= 3
#         and re.match(r'^(summarize|summarise|give\s+a|provide\s+a|generate\s+a|'
#                      r'write\s+a|create\s+a)\b', q_low)
#         and re.search(r'\b(summary|overview|brief|outline|recap)\b', q_low)
#     )

#     # Verification: polar question — starts with auxiliary verb
#     is_verification = bool(
#         re.match(r'^(is|was|were|does|did|can|should|would|could|has|have|had|'
#                  r'are|do|will|has|have)\b', q_low)
#     )

#     is_short = not is_list and not is_summary

#     # ── LLM tie-break for conflicts or low-confidence cases ──────────────────
#     conflict    = is_list and is_summary
#     low_conf    = not is_list and not is_summary and not is_verification and n > 6
#     need_llm    = conflict or low_conf

#     if need_llm:
#         print(f"[Shape] LLM disambiguation for: '{q[:60]}'")
#         prompt = (
#             f"Classify the ideal answer type for this question.\n"
#             f"Question: {question}\n\n"
#             f"A) Full document summary\n"
#             f"B) List of multiple items\n"
#             f"C) Yes/No verification with evidence\n"
#             f"D) Single fact or short answer\n\n"
#             f"Reply with only the letter A, B, C, or D."
#         )
#         try:
#             raw = call_llama(prompt, num_ctx=256, temperature=0.0).strip().upper()
#             letter = re.search(r'\b[ABCD]\b', raw)
#             if letter:
#                 ch = letter.group()
#                 is_summary      = (ch == "A")
#                 is_list         = (ch == "B")
#                 is_verification = (ch == "C")
#                 is_short        = (ch == "D")
#         except Exception as e:
#             print(f"[Shape] LLM failed: {e} — keeping structural result")

#     shape = {
#         "is_summary":      is_summary,
#         "is_list":         is_list,
#         "is_verification": is_verification,
#         "is_short":        is_short,
#     }
#     print(f"[Shape] {shape} for: '{q[:60]}'")
#     return shape


# def shape_to_query_type(shape: dict) -> str:
#     if shape.get("is_summary"):
#         return "FULL_SUMMARY"
#     if shape.get("is_list"):
#         return "MULTIPART_QA"
#     if shape.get("is_verification") or shape.get("is_short"):
#         return "FACTUAL_QA"
#     return "REASONING_QA"


# # ============================================================
# # FIX: FULLY DYNAMIC reorder_by_question — no hardcoded domain words
# # Scores chunks by number co-occurrence and content-word overlap only.
# # ============================================================
# def reorder_by_question(question: str, chunks: list) -> list:
#     """
#     Re-rank chunks so the most question-relevant ones come first.
#     Uses only numbers from the question and content words (no hardcoded domain terms).
#     """
#     numbers  = re.findall(r'\b\d+\b', question)
#     stopwords = _get_dynamic_stopwords()
#     keywords = [w.lower() for w in question.split()
#                 if len(w) > 3 and w.lower() not in stopwords]

#     if not numbers and not keywords:
#         return chunks

#     scored = []
#     for chunk in chunks:
#         chunk_lower = chunk.lower()
#         score = 0

#         # Number matches: score higher when a number appears near other keywords
#         # REPLACE the entire "Number matches" inner loop (lines 141-151) with:

#         for num in numbers:
#             for kw in keywords:
#                 # Match compound phrase like "lecture 3" or "chapter 11"
#                 pattern = rf'\b{re.escape(kw)}\s*{re.escape(num)}\b|\b{re.escape(num)}\s*{re.escape(kw)}\b'
#                 if re.search(pattern, chunk_lower):
#                     score += 5  # strong compound match
#                     break
#             else:
#                 # Number appears but not next to any keyword — weak signal
#                 if re.search(rf'\b{re.escape(num)}\b', chunk_lower):
#                     score += 1

#         for kw in keywords:
#             if kw in chunk_lower:
#                 score += 1

#         scored.append((score, chunk))

#     scored.sort(key=lambda x: x[0], reverse=True)
#     reordered = [c for _, c in scored]

#     if scored and scored[0][0] > 0:
#         print(f"[Reorder] top chunk score={scored[0][0]} | "
#               f"numbers={numbers} | keywords={keywords[:4]}")
#     return reordered


# # ============================================================
# # NUMERIC REGEX FALLBACK
# # ============================================================
# def extract_numeric_answer(question: str, chunks: list) -> str:
#     num_pattern = re.compile(
#         r'\b(\d{1,3}(?:[–\-]\d{1,3})?(?:\.\d+)?(?:\s*(?:%|percent(?:age)?)))\b',
#         re.IGNORECASE
#     )
#     # FIX: dynamic stopwords instead of hardcoded list
#     stopwords = _get_dynamic_stopwords() | {"estimate", "percentage", "probability", "much", "many"}
#     q_keywords = [w.lower() for w in question.split()
#                   if len(w) > 3 and w.lower() not in stopwords]

#     for chunk in chunks:
#         sentences = re.split(r'(?<=[.!?])\s+', chunk)
#         for sent in sentences:
#             sent_lower = sent.lower()
#             has_keyword = any(kw in sent_lower for kw in q_keywords)
#             match = num_pattern.search(sent)
#             if has_keyword and match:
#                 return match.group(0).strip()
#     return ""










# import re
# from docmind_rag.utils.helpers import _get_dynamic_stopwords

# # ============================================================
# # FIX 6: TEXT NORMALIZATION
# # ============================================================
# def _normalize_text(text: str) -> str:
#     """Strip punctuation/case so 'Hinton.' matches 'hinton' etc."""
#     return re.sub(r'[^a-z0-9 ]', '', text.lower())


# def detect_doc_type(text: str) -> str:
#     sample = text[:1000].lower()
#     if any(w in sample for w in ["abstract","methodology","conclusion","research","university","lecture"]):
#         return "academic"
#     if any(w in sample for w in ["whereas","clause","agreement","party","hereby","legal","contract"]):
#         return "legal"
#     if any(w in sample for w in ["api","function","install","configure","technical","specification"]):
#         return "technical"
#     return "general"


# # ============================================================
# # FIX: FULLY DYNAMIC answer shape inference — no hardcoded word lists
# # Uses LLM for all decisions, heuristics only as a fast-path pre-filter
# # ============================================================
# def infer_answer_shape(question: str) -> dict:
#     """
#     Dynamically infer answer shape using structural signals + LLM verification.
#     No hardcoded starter/phrase word lists — structure is inferred from
#     grammar patterns and confirmed by LLM when uncertain.
#     """
#     # Import here to avoid circular dependency (llm → text → llm)
#     from docmind_rag.models.llm import call_llama

#     q     = question.strip()
#     q_low = q.lower()
#     words = q_low.split()
#     n     = len(words)

#     # ── Structural grammar signals (language-agnostic patterns) ──────────────
#     # List: plural interrogative + "are" at position 1, or imperative list verbs
#     is_list = (
#         n >= 3
#         and words[0] in {"what", "which"}
#         and words[1] == "are"
#     ) or (
#         n >= 2
#         and re.match(r'^(list|enumerate|name|identify)\b', q_low)
#     )

#     # Summary: long declarative (no ?) OR starts with generative/summary verb phrase
#     is_summary = (
#         n > 12 and "?" not in q
#     ) or (
#         n >= 3
#         and re.match(r'^(summarize|summarise|give\s+a|provide\s+a|generate\s+a|'
#                      r'write\s+a|create\s+a)\b', q_low)
#         and re.search(r'\b(summary|overview|brief|outline|recap)\b', q_low)
#     )

#     # Verification: polar question — starts with auxiliary verb
#     is_verification = bool(
#         re.match(r'^(is|was|were|does|did|can|should|would|could|has|have|had|'
#                  r'are|do|will|has|have)\b', q_low)
#     )

#     is_short = not is_list and not is_summary

#     # ── LLM tie-break for conflicts or low-confidence cases ──────────────────
#     conflict    = is_list and is_summary
#     low_conf    = not is_list and not is_summary and not is_verification and n > 6
#     need_llm    = conflict or low_conf

#     if need_llm:
#         print(f"[Shape] LLM disambiguation for: '{q[:60]}'")
#         prompt = (
#             f"Classify the ideal answer type for this question.\n"
#             f"Question: {question}\n\n"
#             f"A) Full document summary\n"
#             f"B) List of multiple items\n"
#             f"C) Yes/No verification with evidence\n"
#             f"D) Single fact or short answer\n\n"
#             f"Reply with only the letter A, B, C, or D."
#         )
#         try:
#             raw = call_llama(prompt, num_ctx=256, temperature=0.0).strip().upper()
#             letter = re.search(r'\b[ABCD]\b', raw)
#             if letter:
#                 ch = letter.group()
#                 is_summary      = (ch == "A")
#                 is_list         = (ch == "B")
#                 is_verification = (ch == "C")
#                 is_short        = (ch == "D")
#         except Exception as e:
#             print(f"[Shape] LLM failed: {e} — keeping structural result")

#     shape = {
#         "is_summary":      is_summary,
#         "is_list":         is_list,
#         "is_verification": is_verification,
#         "is_short":        is_short,
#     }
#     print(f"[Shape] {shape} for: '{q[:60]}'")
#     return shape


# def shape_to_query_type(shape: dict) -> str:
#     if shape.get("is_summary"):
#         return "FULL_SUMMARY"
#     if shape.get("is_list"):
#         return "MULTIPART_QA"
#     if shape.get("is_verification") or shape.get("is_short"):
#         return "FACTUAL_QA"
#     return "REASONING_QA"


# # ============================================================
# # FIX: FULLY DYNAMIC reorder_by_question — no hardcoded domain words
# # Scores chunks by number co-occurrence and content-word overlap only.
# # ============================================================
# def reorder_by_question(question: str, chunks: list) -> list:
#     """
#     Re-rank chunks so the most question-relevant ones come first.
#     Uses only numbers from the question and content words (no hardcoded domain terms).
#     """
#     numbers  = re.findall(r'\b\d+\b', question)
#     stopwords = _get_dynamic_stopwords()
#     keywords = [w.lower() for w in question.split()
#                 if len(w) > 3 and w.lower() not in stopwords]

#     if not numbers and not keywords:
#         return chunks

#     scored = []
#     for chunk in chunks:
#         chunk_lower = chunk.lower()
#         score = 0

#         # Number matches: score higher when a number appears near other keywords
#         for num in numbers:
#             if re.search(rf'\b{num}\b', chunk_lower):
#                 # Bonus if a keyword also appears in the same sentence as the number
#                 sentences = re.split(r'(?<=[.!?\n])', chunk_lower)
#                 for sent in sentences:
#                     if re.search(rf'\b{num}\b', sent):
#                         kw_hits = sum(1 for kw in keywords if kw in sent)
#                         score += 3 + kw_hits  # proximity bonus
#                         break
#                 else:
#                     score += 2  # number present but not with keywords

#         for kw in keywords:
#             if kw in chunk_lower:
#                 score += 1

#         scored.append((score, chunk))

#     scored.sort(key=lambda x: x[0], reverse=True)
#     reordered = [c for _, c in scored]

#     if scored and scored[0][0] > 0:
#         print(f"[Reorder] top chunk score={scored[0][0]} | "
#               f"numbers={numbers} | keywords={keywords[:4]}")
#     return reordered


# # ============================================================
# # NUMERIC REGEX FALLBACK
# # ============================================================
# def extract_numeric_answer(question: str, chunks: list) -> str:
#     num_pattern = re.compile(
#         r'\b(\d{1,3}(?:[–\-]\d{1,3})?(?:\.\d+)?(?:\s*(?:%|percent(?:age)?)))\b',
#         re.IGNORECASE
#     )
#     # FIX: dynamic stopwords instead of hardcoded list
#     stopwords = _get_dynamic_stopwords() | {"estimate", "percentage", "probability", "much", "many"}
#     q_keywords = [w.lower() for w in question.split()
#                   if len(w) > 3 and w.lower() not in stopwords]

#     for chunk in chunks:
#         sentences = re.split(r'(?<=[.!?])\s+', chunk)
#         for sent in sentences:
#             sent_lower = sent.lower()
#             has_keyword = any(kw in sent_lower for kw in q_keywords)
#             match = num_pattern.search(sent)
#             if has_keyword and match:
#                 return match.group(0).strip()
#     return ""