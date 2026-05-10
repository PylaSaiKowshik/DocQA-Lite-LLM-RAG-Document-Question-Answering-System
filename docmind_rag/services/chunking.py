
#works for summary but QA breaking
import re
import time
from docmind_rag.core.prompts import SECTION_SUMMARY_PROMPT, MERGE_PROMPT
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain.text_splitter import RecursiveCharacterTextSplitter

from docmind_rag.config.settings import SUMMARY_CHUNK_SIZE, RAG_CHUNK_SIZE, CHUNK_OVERLAP, MAX_WORKERS, RAPTOR_BATCH_CHARS
from docmind_rag.models.llm import call_llama, call_llama_streaming
from docmind_rag.events.events import emit_event

_summary_cache: dict = {}

# ============================================================
# CORE: TF-IDF EXTRACTIVE SUMMARY
# ============================================================
def extractive_summary(chunk: str, top_n: int = 3) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", chunk.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
    sentences = [s for s in sentences
                 if not re.match(r'^\[\d+\]', s)
                 and not re.match(r'^\d+\.\s+[A-Z]', s)
                 and s.count('[') < 3]
    if not sentences:
        return chunk[:1000]
    if len(sentences) <= top_n:
        return " ".join(sentences)
    try:
        vectorizer   = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(sentences)
        scores       = np.array(tfidf_matrix.sum(axis=1)).flatten()
        top_indices  = sorted(np.argsort(scores)[-top_n:].tolist())
        return " ".join(sentences[i] for i in top_indices)
    except Exception:
        return " ".join(sentences[:top_n])


# ============================================================
# DUAL CHUNKING
# ============================================================
def merge_short_lines(text: str) -> str:
    lines = text.split("\n")
    merged = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # detect short header-like lines
        if (
            0 < len(line) < 80
            and not line.endswith(".")
            and i + 1 < len(lines)
        ):
            next_line = lines[i + 1].strip()

            # merge if next line is meaningful
            if next_line and len(next_line) > 20:
                merged.append(f"{line} — {next_line}")
                i += 2
                continue

        merged.append(line)
        i += 1

    return "\n".join(merged)
def semantic_chunk(text: str):
    # Summary chunks — simple version (fast, works for summary)
    clean_text = re.sub(r'--- PAGE \d+ ---\n?', '', text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    summary_splitter = RecursiveCharacterTextSplitter(
        chunk_size=SUMMARY_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    summary_chunks = summary_splitter.split_text(clean_text)

    # RAG chunks — page-aware version (works for QA)
    text_for_rag = merge_short_lines(text)
    rag_splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    raw_pages = [p.strip() for p in text_for_rag.split("--- PAGE") if p.strip()]
    page_texts = []
    for page in raw_pages:
        lines = page.split("\n", 1)
        page_text = lines[1].strip() if len(lines) > 1 else lines[0].strip()
        if page_text:
            page_texts.append(page_text)

    MIN_PAGE_CHARS = 200
    merged_pages = []
    i = 0
    while i < len(page_texts):
        current = page_texts[i]
        if len(current) < MIN_PAGE_CHARS and i + 1 < len(page_texts):
            merged_pages.append(current + "\n\n" + page_texts[i + 1])
            i += 2
        else:
            merged_pages.append(current)
            i += 1

    rag_chunks = []
    for page_text in merged_pages:
        rag_chunks.extend(rag_splitter.split_text(page_text))

    if not rag_chunks:
        rag_chunks = rag_splitter.split_text(clean_text)

    print(f"[Chunk] Summary chunks: {len(summary_chunks)} | RAG chunks: {len(rag_chunks)}")
    return summary_chunks, rag_chunks
# ============================================================
# CORE: RAPTOR HIERARCHICAL SUMMARY
# ============================================================
def raptor_summarize(chunks: list, doc_type: str):
    if not chunks:
        return "No content to summarize.", 0, 0

    map_start = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures        = [ex.submit(extractive_summary, chunk, 3) for chunk in chunks]
        mini_summaries = [f.result() for f in futures if f.result().strip()]
    map_time = time.time() - map_start
    print(f"[RAPTOR] Map (TF-IDF): {len(mini_summaries)} chunks in {map_time:.1f}s")

    if not mini_summaries:
        return "Could not generate summary.", map_time, 0

    reduce_start  = time.time()
    BATCH_CHARS = RAPTOR_BATCH_CHARS
    batches       = []
    current_batch = ""
    for summary in mini_summaries:
        if len(current_batch) + len(summary) > BATCH_CHARS and current_batch:
            batches.append(current_batch.strip())
            current_batch = summary + "\n\n"
        else:
            current_batch += summary + "\n\n"
    if current_batch.strip():
        batches.append(current_batch.strip())

    request_id = getattr(raptor_summarize, '_request_id', "")

    partial_summaries = []
    print(f"[RAPTOR] Reduce: {len(batches)} batches → LLaMA")

    if len(batches) == 1:
        emit_event(request_id, "stream_start", "✍️ Generating summary...")
        final_summary, _ = call_llama_streaming(
            SECTION_SUMMARY_PROMPT.format(text=batches[0]),
            request_id, temperature=0.7)
        partial_summaries = [final_summary]
    else:
        for i, batch in enumerate(batches):
            prompt = SECTION_SUMMARY_PROMPT.format(text=batch)
            result = call_llama(prompt, temperature=0.7)
            partial_summaries.append(result)
            print(f"[RAPTOR] Batch {i+1}/{len(batches)} done")
            emit_event(request_id, "agent_action",
                       f"📝 Processed section {i+1}/{len(batches)}...")

        merged = "\n\n---\n\n".join(partial_summaries)
        emit_event(request_id, "stream_start", "✍️ Generating final summary...")
        final_summary, _ = call_llama_streaming(
            MERGE_PROMPT.format(summaries=merged[:12000]),
            request_id, temperature=0.7)

    reduce_time = time.time() - reduce_start
    total_calls = len(batches) + (1 if len(partial_summaries) > 1 else 0)
    print(f"[RAPTOR] Reduce done: {reduce_time:.1f}s | {total_calls} LLaMA calls | Total: {map_time+reduce_time:.1f}s")
    return final_summary, map_time, reduce_time













































# #works for summary but QA breaking
# import re
# import time
# from docmind_rag.core.prompts import SECTION_SUMMARY_PROMPT, MERGE_PROMPT
# from concurrent.futures import ThreadPoolExecutor

# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# from docmind_rag.config.settings import SUMMARY_CHUNK_SIZE, RAG_CHUNK_SIZE, CHUNK_OVERLAP, MAX_WORKERS, RAPTOR_BATCH_CHARS
# from docmind_rag.models.llm import call_llama, call_llama_streaming
# from docmind_rag.events.events import emit_event

# _summary_cache: dict = {}

# # ============================================================
# # CORE: TF-IDF EXTRACTIVE SUMMARY
# # ============================================================
# def extractive_summary(chunk: str, top_n: int = 3) -> str:
#     sentences = re.split(r"(?<=[.!?])\s+", chunk.strip())
#     sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
#     sentences = [s for s in sentences
#                  if not re.match(r'^\[\d+\]', s)
#                  and not re.match(r'^\d+\.\s+[A-Z]', s)
#                  and s.count('[') < 3]
#     if not sentences:
#         return chunk[:1000]
#     if len(sentences) <= top_n:
#         return " ".join(sentences)
#     try:
#         vectorizer   = TfidfVectorizer(stop_words="english")
#         tfidf_matrix = vectorizer.fit_transform(sentences)
#         scores       = np.array(tfidf_matrix.sum(axis=1)).flatten()
#         top_indices  = sorted(np.argsort(scores)[-top_n:].tolist())
#         return " ".join(sentences[i] for i in top_indices)
#     except Exception:
#         return " ".join(sentences[:top_n])


# # ============================================================
# # DUAL CHUNKING
# # ============================================================
# def merge_short_lines(text: str) -> str:
#     lines = text.split("\n")
#     merged = []
#     i = 0

#     while i < len(lines):
#         line = lines[i].strip()

#         # detect short header-like lines
#         if (
#             0 < len(line) < 80
#             and not line.endswith(".")
#             and i + 1 < len(lines)
#         ):
#             next_line = lines[i + 1].strip()

#             # merge if next line is meaningful
#             if next_line and len(next_line) > 20:
#                 merged.append(f"{line} — {next_line}")
#                 i += 2
#                 continue

#         merged.append(line)
#         i += 1

#     return "\n".join(merged)
# def semantic_chunk(text: str):
#     clean_text = re.sub(r'--- PAGE \d+ ---\n?', '', text)
#     clean_text = re.sub(r'\s+', ' ', clean_text).strip()

#     summary_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=SUMMARY_CHUNK_SIZE,
#         chunk_overlap=CHUNK_OVERLAP,
#         separators=["\n\n", "\n", ". ", " ", ""]
#     )
#     rag_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=RAG_CHUNK_SIZE,
#         chunk_overlap=CHUNK_OVERLAP,
#         separators=["\n\n", "\n", ". ", " ", ""]
#     )

#     summary_chunks = summary_splitter.split_text(clean_text)
#     rag_chunks = rag_splitter.split_text(clean_text)

#     print(f"[Chunk] Summary chunks: {len(summary_chunks)} | RAG chunks: {len(rag_chunks)}")
#     return summary_chunks, rag_chunks

# # ============================================================
# # CORE: RAPTOR HIERARCHICAL SUMMARY
# # ============================================================
# def raptor_summarize(chunks: list, doc_type: str):
#     if not chunks:
#         return "No content to summarize.", 0, 0

#     map_start = time.time()
#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
#         futures        = [ex.submit(extractive_summary, chunk, 3) for chunk in chunks]
#         mini_summaries = [f.result() for f in futures if f.result().strip()]
#     map_time = time.time() - map_start
#     print(f"[RAPTOR] Map (TF-IDF): {len(mini_summaries)} chunks in {map_time:.1f}s")

#     if not mini_summaries:
#         return "Could not generate summary.", map_time, 0

#     reduce_start  = time.time()
#     BATCH_CHARS = RAPTOR_BATCH_CHARS
#     batches       = []
#     current_batch = ""
#     for summary in mini_summaries:
#         if len(current_batch) + len(summary) > BATCH_CHARS and current_batch:
#             batches.append(current_batch.strip())
#             current_batch = summary + "\n\n"
#         else:
#             current_batch += summary + "\n\n"
#     if current_batch.strip():
#         batches.append(current_batch.strip())

#     request_id = getattr(raptor_summarize, '_request_id', "")

#     partial_summaries = []
#     print(f"[RAPTOR] Reduce: {len(batches)} batches → LLaMA")

#     if len(batches) == 1:
#         emit_event(request_id, "stream_start", "✍️ Generating summary...")
#         final_summary, _ = call_llama_streaming(
#             SECTION_SUMMARY_PROMPT.format(text=batches[0]),
#             request_id, temperature=0.7)
#         partial_summaries = [final_summary]
#     else:
#         for i, batch in enumerate(batches):
#             prompt = SECTION_SUMMARY_PROMPT.format(text=batch)
#             result = call_llama(prompt, temperature=0.7)
#             partial_summaries.append(result)
#             print(f"[RAPTOR] Batch {i+1}/{len(batches)} done")
#             emit_event(request_id, "agent_action",
#                        f"📝 Processed section {i+1}/{len(batches)}...")

#         merged = "\n\n---\n\n".join(partial_summaries)
#         emit_event(request_id, "stream_start", "✍️ Generating final summary...")
#         final_summary, _ = call_llama_streaming(
#             MERGE_PROMPT.format(summaries=merged[:12000]),
#             request_id, temperature=0.7)

#     reduce_time = time.time() - reduce_start
#     total_calls = len(batches) + (1 if len(partial_summaries) > 1 else 0)
#     print(f"[RAPTOR] Reduce done: {reduce_time:.1f}s | {total_calls} LLaMA calls | Total: {map_time+reduce_time:.1f}s")
#     return final_summary, map_time, reduce_time










































# #  works 90 90 50 final onew but got  sumamry  issue 


# import re
# import time
# from docmind_rag.core.prompts import SECTION_SUMMARY_PROMPT
# from concurrent.futures import ThreadPoolExecutor

# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# from docmind_rag.config.settings import SUMMARY_CHUNK_SIZE, RAG_CHUNK_SIZE, CHUNK_OVERLAP, MAX_WORKERS, RAPTOR_BATCH_CHARS
# from docmind_rag.models.llm import call_llama, call_llama_streaming
# from docmind_rag.events.events import emit_event

# _summary_cache: dict = {}

# # ============================================================
# # CORE: TF-IDF EXTRACTIVE SUMMARY
# # ============================================================
# def extractive_summary(chunk: str, top_n: int = 3) -> str:
#     sentences = re.split(r"(?<=[.!?])\s+", chunk.strip())
#     sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
#     sentences = [s for s in sentences
#                  if not re.match(r'^\[\d+\]', s)
#                  and not re.match(r'^\d+\.\s+[A-Z]', s)
#                  and s.count('[') < 3]
#     if not sentences:
#         return chunk[:1000]
#     if len(sentences) <= top_n:
#         return " ".join(sentences)
#     try:
#         vectorizer   = TfidfVectorizer(stop_words="english")
#         tfidf_matrix = vectorizer.fit_transform(sentences)
#         scores       = np.array(tfidf_matrix.sum(axis=1)).flatten()
#         top_indices  = sorted(np.argsort(scores)[-top_n:].tolist())
#         return " ".join(sentences[i] for i in top_indices)
#     except Exception:
#         return " ".join(sentences[:top_n])


# # ============================================================
# # DUAL CHUNKING
# # ============================================================
# def merge_short_lines(text: str) -> str:
#     lines = text.split("\n")
#     merged = []
#     i = 0

#     while i < len(lines):
#         line = lines[i].strip()

#         # detect short header-like lines
#         if (
#             0 < len(line) < 80
#             and not line.endswith(".")
#             and i + 1 < len(lines)
#         ):
#             next_line = lines[i + 1].strip()

#             # merge if next line is meaningful
#             if next_line and len(next_line) > 20:
#                 merged.append(f"{line} — {next_line}")
#                 i += 2
#                 continue

#         merged.append(line)
#         i += 1

#     return "\n".join(merged)
# def semantic_chunk(text: str):
#     # 🔥 FIX: preserve structure (titles + content)
#     text = merge_short_lines(text)

#     summary_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=SUMMARY_CHUNK_SIZE,
#         chunk_overlap=400,
#         separators=["\n\n", "\n", ". ", " ", ""]
#     )
#     rag_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=RAG_CHUNK_SIZE,
#         chunk_overlap=CHUNK_OVERLAP,
#         separators=["\n\n", "\n", ". ", " ", ""]
#     )

#     raw_pages = [p.strip() for p in text.split("--- PAGE") if p.strip()]

#     page_texts = []
#     for page in raw_pages:
#         lines = page.split("\n", 1)
#         page_text = lines[1].strip() if len(lines) > 1 else lines[0].strip()
#         if page_text:
#             page_texts.append(page_text)

#     MIN_PAGE_CHARS = 200
#     merged_pages = []
#     i = 0
#     while i < len(page_texts):
#         current = page_texts[i]
#         if len(current) < MIN_PAGE_CHARS and i + 1 < len(page_texts):
#             merged_pages.append(current + "\n\n" + page_texts[i + 1])
#             i += 2
#         else:
#             merged_pages.append(current)
#             i += 1

#     summary_chunks = []
#     rag_chunks = []

#     for page_text in merged_pages:
#         summary_chunks.extend(summary_splitter.split_text(page_text))
#         rag_chunks.extend(rag_splitter.split_text(page_text))

#     if not rag_chunks:
#         summary_chunks = summary_splitter.split_text(text)
#         rag_chunks = rag_splitter.split_text(text)

#     print(f"[Chunk] Page-aware: {len(raw_pages)} pages → {len(merged_pages)} merged blocks")
#     return summary_chunks, rag_chunks

# # ============================================================
# # CORE: RAPTOR HIERARCHICAL SUMMARY
# # ============================================================
# def raptor_summarize(chunks: list, doc_type: str):
#     if not chunks:
#         return "No content to summarize.", 0, 0

#     map_start = time.time()
#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
#         futures        = [ex.submit(extractive_summary, chunk, 3) for chunk in chunks]
#         mini_summaries = [f.result() for f in futures if f.result().strip()]
#     map_time = time.time() - map_start
#     print(f"[RAPTOR] Map (TF-IDF): {len(mini_summaries)} chunks in {map_time:.1f}s")

#     if not mini_summaries:
#         return "Could not generate summary.", map_time, 0

#     reduce_start  = time.time()
#     BATCH_CHARS = RAPTOR_BATCH_CHARS
#     batches       = []
#     current_batch = ""
#     for summary in mini_summaries:
#         if len(current_batch) + len(summary) > BATCH_CHARS and current_batch:
#             batches.append(current_batch.strip())
#             current_batch = summary + "\n\n"
#         else:
#             current_batch += summary + "\n\n"
#     if current_batch.strip():
#         batches.append(current_batch.strip())

#     request_id        = getattr(raptor_summarize, '_request_id', "")
#     partial_summaries = []
#     print(f"[RAPTOR] Reduce: {len(batches)} batches → LLaMA")

#     for i, batch in enumerate(batches):
#         prompt_template = SECTION_SUMMARY_PROMPT.get(doc_type, SECTION_SUMMARY_PROMPT["general"])
#         prompt = prompt_template.format(text=batch)
#         if i == 0 and len(batches) == 1:
#             emit_event(request_id, "stream_start", "✍️ Generating summary...")
#             result = call_llama_streaming(prompt, request_id, temperature=0.7)
#         else:
#             result = call_llama(prompt, temperature=0.7)
#         partial_summaries.append(result)
#         print(f"[RAPTOR] Batch {i+1}/{len(batches)} done")
#         emit_event(request_id, "agent_action",
#                    f"📝 Processed section {i+1}/{len(batches)}...")

#     if len(partial_summaries) == 1:
#         final_summary = partial_summaries[0]
#     else:
#         merged = "\n\n---\n\n".join(partial_summaries)
#         merge_prompt = (
#             "Merge these section summaries into one comprehensive final summary. "
#             "Remove repetition. Keep ALL key points. Use clear organized bullet points.\n\n"
#             f"Section summaries:\n{merged[:12000]}\n\nFinal comprehensive summary:"
#         )
#         emit_event(request_id, "stream_start", "✍️ Generating final summary...")
#         final_summary = call_llama_streaming(merge_prompt, request_id, temperature=0.7)

#     reduce_time = time.time() - reduce_start
#     total_calls = len(batches) + (1 if len(partial_summaries) > 1 else 0)
#     print(f"[RAPTOR] Reduce done: {reduce_time:.1f}s | {total_calls} LLaMA calls | Total: {map_time+reduce_time:.1f}s")
#     return final_summary, map_time, reduce_time






































# # works but trying it change  code  generic  - lap claude here  only import SUMMARY_PROMPTS is changed
# import re
# import time
# from docmind_rag.core.prompts import SUMMARY_PROMPTS
# from concurrent.futures import ThreadPoolExecutor

# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# from docmind_rag.config.settings import SUMMARY_CHUNK_SIZE, RAG_CHUNK_SIZE, CHUNK_OVERLAP, MAX_WORKERS, RAPTOR_BATCH_CHARS
# from docmind_rag.models.llm import call_llama, call_llama_streaming
# from docmind_rag.events.events import emit_event

# _summary_cache: dict = {}

# # ============================================================
# # CORE: TF-IDF EXTRACTIVE SUMMARY
# # ============================================================
# def extractive_summary(chunk: str, top_n: int = 3) -> str:
#     sentences = re.split(r"(?<=[.!?])\s+", chunk.strip())
#     sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
#     sentences = [s for s in sentences
#                  if not re.match(r'^\[\d+\]', s)
#                  and not re.match(r'^\d+\.\s+[A-Z]', s)
#                  and s.count('[') < 3]
#     if not sentences:
#         return chunk[:1000]
#     if len(sentences) <= top_n:
#         return " ".join(sentences)
#     try:
#         vectorizer   = TfidfVectorizer(stop_words="english")
#         tfidf_matrix = vectorizer.fit_transform(sentences)
#         scores       = np.array(tfidf_matrix.sum(axis=1)).flatten()
#         top_indices  = sorted(np.argsort(scores)[-top_n:].tolist())
#         return " ".join(sentences[i] for i in top_indices)
#     except Exception:
#         return " ".join(sentences[:top_n])


# # ============================================================
# # DUAL CHUNKING
# # ============================================================
# def merge_short_lines(text: str) -> str:
#     lines = text.split("\n")
#     merged = []
#     i = 0

#     while i < len(lines):
#         line = lines[i].strip()

#         # detect short header-like lines
#         if (
#             0 < len(line) < 80
#             and not line.endswith(".")
#             and i + 1 < len(lines)
#         ):
#             next_line = lines[i + 1].strip()

#             # merge if next line is meaningful
#             if next_line and len(next_line) > 20:
#                 merged.append(f"{line} — {next_line}")
#                 i += 2
#                 continue

#         merged.append(line)
#         i += 1

#     return "\n".join(merged)
# def semantic_chunk(text: str):
#     # 🔥 FIX: preserve structure (titles + content)
#     text = merge_short_lines(text)

#     summary_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=SUMMARY_CHUNK_SIZE,
#         chunk_overlap=400,
#         separators=["\n\n", "\n", ". ", " ", ""]
#     )
#     rag_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=RAG_CHUNK_SIZE,
#         chunk_overlap=CHUNK_OVERLAP,
#         separators=["\n\n", "\n", ". ", " ", ""]
#     )

#     raw_pages = [p.strip() for p in text.split("--- PAGE") if p.strip()]

#     page_texts = []
#     for page in raw_pages:
#         lines = page.split("\n", 1)
#         page_text = lines[1].strip() if len(lines) > 1 else lines[0].strip()
#         if page_text:
#             page_texts.append(page_text)

#     MIN_PAGE_CHARS = 200
#     merged_pages = []
#     i = 0
#     while i < len(page_texts):
#         current = page_texts[i]
#         if len(current) < MIN_PAGE_CHARS and i + 1 < len(page_texts):
#             merged_pages.append(current + "\n\n" + page_texts[i + 1])
#             i += 2
#         else:
#             merged_pages.append(current)
#             i += 1

#     summary_chunks = []
#     rag_chunks = []

#     for page_text in merged_pages:
#         summary_chunks.extend(summary_splitter.split_text(page_text))
#         rag_chunks.extend(rag_splitter.split_text(page_text))

#     if not rag_chunks:
#         summary_chunks = summary_splitter.split_text(text)
#         rag_chunks = rag_splitter.split_text(text)

#     print(f"[Chunk] Page-aware: {len(raw_pages)} pages → {len(merged_pages)} merged blocks")
#     return summary_chunks, rag_chunks

# # ============================================================
# # CORE: RAPTOR HIERARCHICAL SUMMARY
# # ============================================================
# def raptor_summarize(chunks: list, doc_type: str):
#     if not chunks:
#         return "No content to summarize.", 0, 0

#     map_start = time.time()
#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
#         futures        = [ex.submit(extractive_summary, chunk, 3) for chunk in chunks]
#         mini_summaries = [f.result() for f in futures if f.result().strip()]
#     map_time = time.time() - map_start
#     print(f"[RAPTOR] Map (TF-IDF): {len(mini_summaries)} chunks in {map_time:.1f}s")

#     if not mini_summaries:
#         return "Could not generate summary.", map_time, 0

#     reduce_start  = time.time()
#     BATCH_CHARS = RAPTOR_BATCH_CHARS
#     batches       = []
#     current_batch = ""
#     for summary in mini_summaries:
#         if len(current_batch) + len(summary) > BATCH_CHARS and current_batch:
#             batches.append(current_batch.strip())
#             current_batch = summary + "\n\n"
#         else:
#             current_batch += summary + "\n\n"
#     if current_batch.strip():
#         batches.append(current_batch.strip())

#     request_id        = getattr(raptor_summarize, '_request_id', "")
#     partial_summaries = []
#     print(f"[RAPTOR] Reduce: {len(batches)} batches → LLaMA")

#     for i, batch in enumerate(batches):
#         prompt_template = SUMMARY_PROMPTS.get(doc_type, SUMMARY_PROMPTS["general"])
#         prompt = prompt_template.format(text=batch)
#         if i == 0 and len(batches) == 1:
#             emit_event(request_id, "stream_start", "✍️ Generating summary...")
#             result = call_llama_streaming(prompt, request_id, temperature=0.7)
#         else:
#             result = call_llama(prompt, temperature=0.7)
#         partial_summaries.append(result)
#         print(f"[RAPTOR] Batch {i+1}/{len(batches)} done")
#         emit_event(request_id, "agent_action",
#                    f"📝 Processed section {i+1}/{len(batches)}...")

#     if len(partial_summaries) == 1:
#         final_summary = partial_summaries[0]
#     else:
#         merged = "\n\n---\n\n".join(partial_summaries)
#         merge_prompt = (
#             "Merge these section summaries into one comprehensive final summary. "
#             "Remove repetition. Keep ALL key points. Use clear organized bullet points.\n\n"
#             f"Section summaries:\n{merged[:12000]}\n\nFinal comprehensive summary:"
#         )
#         emit_event(request_id, "stream_start", "✍️ Generating final summary...")
#         final_summary = call_llama_streaming(merge_prompt, request_id, temperature=0.7)

#     reduce_time = time.time() - reduce_start
#     total_calls = len(batches) + (1 if len(partial_summaries) > 1 else 0)
#     print(f"[RAPTOR] Reduce done: {reduce_time:.1f}s | {total_calls} LLaMA calls | Total: {map_time+reduce_time:.1f}s")
#     return final_summary, map_time, reduce_time





















# import re
# import time
# from concurrent.futures import ThreadPoolExecutor

# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# from docmind_rag.config.settings import SUMMARY_CHUNK_SIZE, RAG_CHUNK_SIZE, CHUNK_OVERLAP, MAX_WORKERS
# from docmind_rag.models.llm import call_llama, call_llama_streaming
# from docmind_rag.events.events import emit_event

# _summary_cache: dict = {}

# # ============================================================
# # CORE: TF-IDF EXTRACTIVE SUMMARY
# # ============================================================
# def extractive_summary(chunk: str, top_n: int = 3) -> str:
#     sentences = re.split(r"(?<=[.!?])\s+", chunk.strip())
#     sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
#     sentences = [s for s in sentences
#                  if not re.match(r'^\[\d+\]', s)
#                  and not re.match(r'^\d+\.\s+[A-Z]', s)
#                  and s.count('[') < 3]
#     if not sentences:
#         return chunk[:1000]
#     if len(sentences) <= top_n:
#         return " ".join(sentences)
#     try:
#         vectorizer   = TfidfVectorizer(stop_words="english")
#         tfidf_matrix = vectorizer.fit_transform(sentences)
#         scores       = np.array(tfidf_matrix.sum(axis=1)).flatten()
#         top_indices  = sorted(np.argsort(scores)[-top_n:].tolist())
#         return " ".join(sentences[i] for i in top_indices)
#     except Exception:
#         return " ".join(sentences[:top_n])


# # ============================================================
# # DUAL CHUNKING
# # ============================================================
# def semantic_chunk(text: str):
#     summary_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=SUMMARY_CHUNK_SIZE,
#         chunk_overlap=400,
#         separators=["\n\n", "\n", ". ", " ", ""]
#     )
#     rag_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=RAG_CHUNK_SIZE,
#         chunk_overlap=CHUNK_OVERLAP,
#         separators=["\n\n", "\n", ". ", " ", ""]
#     )

#     raw_pages = [p.strip() for p in text.split("--- PAGE") if p.strip()]

#     page_texts = []
#     for page in raw_pages:
#         lines     = page.split("\n", 1)
#         page_text = lines[1].strip() if len(lines) > 1 else lines[0].strip()
#         if page_text:
#             page_texts.append(page_text)

#     MIN_PAGE_CHARS = 200
#     merged_pages   = []
#     i = 0
#     while i < len(page_texts):
#         current = page_texts[i]
#         if len(current) < MIN_PAGE_CHARS and i + 1 < len(page_texts):
#             merged_pages.append(current + "\n\n" + page_texts[i + 1])
#             i += 2
#         else:
#             merged_pages.append(current)
#             i += 1

#     summary_chunks = []
#     rag_chunks     = []

#     for page_text in merged_pages:
#         summary_chunks.extend(summary_splitter.split_text(page_text))
#         rag_chunks.extend(rag_splitter.split_text(page_text))

#     if not rag_chunks:
#         summary_chunks = summary_splitter.split_text(text)
#         rag_chunks     = rag_splitter.split_text(text)

#     print(f"[Chunk] Page-aware: {len(raw_pages)} pages → {len(merged_pages)} merged blocks")
#     return summary_chunks, rag_chunks


# # ============================================================
# # CORE: RAPTOR HIERARCHICAL SUMMARY
# # ============================================================
# def raptor_summarize(chunks: list, doc_type: str):
#     if not chunks:
#         return "No content to summarize.", 0, 0

#     map_start = time.time()
#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
#         futures        = [ex.submit(extractive_summary, chunk, 3) for chunk in chunks]
#         mini_summaries = [f.result() for f in futures if f.result().strip()]
#     map_time = time.time() - map_start
#     print(f"[RAPTOR] Map (TF-IDF): {len(mini_summaries)} chunks in {map_time:.1f}s")

#     if not mini_summaries:
#         return "Could not generate summary.", map_time, 0

#     reduce_start  = time.time()
#     BATCH_CHARS   = 15000
#     batches       = []
#     current_batch = ""
#     for summary in mini_summaries:
#         if len(current_batch) + len(summary) > BATCH_CHARS and current_batch:
#             batches.append(current_batch.strip())
#             current_batch = summary + "\n\n"
#         else:
#             current_batch += summary + "\n\n"
#     if current_batch.strip():
#         batches.append(current_batch.strip())

#     request_id        = getattr(raptor_summarize, '_request_id', "")
#     partial_summaries = []
#     print(f"[RAPTOR] Reduce: {len(batches)} batches → LLaMA")

#     for i, batch in enumerate(batches):
#         prompt = (f"Extract the key points from this text section "
#                   f"in clear bullet points. Be concise.\n\nText:\n{batch}\n\nKey points:")
#         if i == 0 and len(batches) == 1:
#             emit_event(request_id, "stream_start", "✍️ Generating summary...")
#             result = call_llama_streaming(prompt, request_id, temperature=0.7)
#         else:
#             result = call_llama(prompt, temperature=0.7)
#         partial_summaries.append(result)
#         print(f"[RAPTOR] Batch {i+1}/{len(batches)} done")
#         emit_event(request_id, "agent_action",
#                    f"📝 Processed section {i+1}/{len(batches)}...")

#     if len(partial_summaries) == 1:
#         final_summary = partial_summaries[0]
#     else:
#         merged = "\n\n---\n\n".join(partial_summaries)
#         merge_prompt = (
#             "Merge these section summaries into one comprehensive final summary. "
#             "Remove repetition. Keep ALL key points. Use clear organized bullet points.\n\n"
#             f"Section summaries:\n{merged[:12000]}\n\nFinal comprehensive summary:"
#         )
#         emit_event(request_id, "stream_start", "✍️ Generating final summary...")
#         final_summary = call_llama_streaming(merge_prompt, request_id, temperature=0.7)

#     reduce_time = time.time() - reduce_start
#     total_calls = len(batches) + (1 if len(partial_summaries) > 1 else 0)
#     print(f"[RAPTOR] Reduce done: {reduce_time:.1f}s | {total_calls} LLaMA calls | Total: {map_time+reduce_time:.1f}s")
#     return final_summary, map_time, reduce_time