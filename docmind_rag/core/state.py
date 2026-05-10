from typing import TypedDict, Optional

# ============================================================
# LANGGRAPH STATE
# ============================================================
class DocState(TypedDict):
    pdf_path:         str
    question:         str
    extracted_text:   str
    chunks:           list
    summary_chunks:   list
    faiss_index:      object
    answer:           str
    metrics:          dict
    doc_type:         str
    query_type:       str
    retry_count:      int
    start_time:       float
    page_count:       int
    char_count:       int
    request_id:       str