from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from docmind_rag.config.settings import embedding_model

_faiss_cache: dict = {}

# ============================================================
# CORE: FAISS INDEX
# ============================================================
def build_faiss_index(chunks: list, pdf_hash: str):
    if pdf_hash in _faiss_cache:
        print(f"[FAISS] ✅ Cache hit for {pdf_hash[:8]}...")
        return _faiss_cache[pdf_hash]
    print(f"[FAISS] Building fresh index ({len(chunks)} chunks)...")
    docs  = [Document(page_content=c, metadata={"chunk_id": i})
             for i, c in enumerate(chunks)]
    index = FAISS.from_documents(docs, embedding_model)
    _faiss_cache[pdf_hash] = index
    if len(_faiss_cache) > 3:
        oldest = next(iter(_faiss_cache))
        del _faiss_cache[oldest]
    print(f"[FAISS] Index built | cache size={len(_faiss_cache)}")
    return index