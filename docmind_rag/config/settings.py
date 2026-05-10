import os
import torch
import pytesseract
from transformers import pipeline, logging
from sentence_transformers import CrossEncoder
from langchain_community.embeddings import HuggingFaceEmbeddings

logging.set_verbosity_error()

# ============================================================
# CONFIG
# ============================================================
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

OLLAMA_MODEL       = "llama3.2"
SUMMARY_CHUNK_SIZE = 10000
RAG_CHUNK_SIZE     = 1200
CHUNK_OVERLAP      = 200
MAX_WORKERS        = 6
RAPTOR_BATCH_CHARS = 15000
EMBED_MODEL        = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_CACHE_DIR    = "faiss_cache"
os.makedirs(FAISS_CACHE_DIR, exist_ok=True)

# Reranker pruning: drop chunks more than this below the top score
RERANKER_PRUNE_MARGIN = 3.0

# Max chunks passed to LLM for factual QA (after pruning)
FACTUAL_TOP_K = 8
# Max chunks for multipart QA (needs broader coverage)
MULTIPART_TOP_K = 8

# Confidence scoring weights
CONF_WEIGHT_RERANKER = 0.5
CONF_WEIGHT_RECALL   = 0.3
CONF_WEIGHT_KEYWORD  = 0.2

# FIX 7: Lowered grounding thresholds for final answer decision
GROUNDING_REJECT_BELOW  = 0.3   # below this → "not present"
GROUNDING_BORDERLINE    = 0.5   # below this → borderline, still allow

# C2: Small doc threshold — reranker is unreliable below this
SMALL_DOC_CHUNK_THRESHOLD = 5

device = 0 if torch.cuda.is_available() else -1

print("Loading RoBERTa QA model...")
qa_pipeline = pipeline("question-answering",
    model="deepset/roberta-base-squad2", device=device)
print("RoBERTa loaded!")

print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
print("Embeddings loaded!")

print("Loading cross-encoder reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2",
                        device=device if device == 0 else "cpu")
print("Reranker loaded!")














# import os
# import torch
# import pytesseract
# from transformers import pipeline, logging
# from sentence_transformers import CrossEncoder
# from langchain_community.embeddings import HuggingFaceEmbeddings

# logging.set_verbosity_error()

# # ============================================================
# # CONFIG
# # ============================================================
# TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# OLLAMA_MODEL       = "llama3.2"
# SUMMARY_CHUNK_SIZE = 8000
# RAG_CHUNK_SIZE     = 1200
# CHUNK_OVERLAP      = 200
# MAX_WORKERS        = 6
# EMBED_MODEL        = "sentence-transformers/all-MiniLM-L6-v2"
# FAISS_CACHE_DIR    = "faiss_cache"
# os.makedirs(FAISS_CACHE_DIR, exist_ok=True)

# # Reranker pruning: drop chunks more than this below the top score
# RERANKER_PRUNE_MARGIN = 3.0
# # Max chunks passed to LLM for factual QA (after pruning)
# FACTUAL_TOP_K = 3
# # Max chunks for multipart QA (needs broader coverage)
# MULTIPART_TOP_K = 8

# # Confidence scoring weights
# CONF_WEIGHT_RERANKER = 0.5
# CONF_WEIGHT_RECALL   = 0.3
# CONF_WEIGHT_KEYWORD  = 0.2

# # FIX 7: Lowered grounding thresholds for final answer decision
# GROUNDING_REJECT_BELOW  = 0.3   # below this → "not present"
# GROUNDING_BORDERLINE    = 0.5   # below this → borderline, still allow

# # C2: Small doc threshold — reranker is unreliable below this
# SMALL_DOC_CHUNK_THRESHOLD = 5

# device = 0 if torch.cuda.is_available() else -1

# print("Loading RoBERTa QA model...")
# qa_pipeline = pipeline("question-answering",
#     model="deepset/roberta-base-squad2", device=device)
# print("RoBERTa loaded!")

# print("Loading embedding model...")
# embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
# print("Embeddings loaded!")

# print("Loading cross-encoder reranker...")
# reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2",
#                         device=device if device == 0 else "cpu")
# print("Reranker loaded!")