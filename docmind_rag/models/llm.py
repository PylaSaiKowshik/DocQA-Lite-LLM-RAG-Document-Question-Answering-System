import time
import ollama
from docmind_rag.config.settings import OLLAMA_MODEL
from docmind_rag.events.events import emit_event

_ttft_tracker = {}

# ============================================================
# CORE: LLaMA CALLS
# ============================================================
def call_llama(prompt: str, num_ctx: int = 4096, temperature: float = 0.0) -> str:
    try:
        response = ollama.chat(model=OLLAMA_MODEL,
            options={"num_ctx": num_ctx, "temperature": temperature},
            messages=[{"role": "user", "content": prompt}])
        return response['message']['content'].strip()
    except Exception as e:
        return f"LLaMA error: {str(e)}"


def call_llama_streaming(prompt: str, request_id: str,
                         num_ctx: int = 4096,
                         temperature: float = 0.0) -> str:
    global _ttft_tracker
    full_response = ""
    first_token   = True
    stream_start  = time.time()
    try:
        stream = ollama.chat(model=OLLAMA_MODEL,
            options={"num_ctx": num_ctx, "temperature": temperature},
            messages=[{"role": "user", "content": prompt}],
            stream=True)
        for chunk in stream:
            token = chunk["message"]["content"]
            if token:
                if first_token and request_id:
                    ttft = round(time.time() - stream_start, 2)
                    _ttft_tracker[request_id] = ttft
                    print(f"[STREAM] First token in {ttft}s")
                    first_token = False
                full_response += token
                emit_event(request_id, "token", token)
        # return full_response.strip()
        return full_response.strip(), round(time.time() - stream_start, 2)
    except Exception as e:
        # return f"LLaMA error: {str(e)}"
        return f"LLaMA error: {str(e)}", 0.0




















# good but trying it make it generic -lap claude 
# import time
# import ollama
# from docmind_rag.config.settings import OLLAMA_MODEL
# from docmind_rag.events.events import emit_event

# _ttft_tracker = {}

# # ============================================================
# # CORE: LLaMA CALLS
# # ============================================================
# def call_llama(prompt: str, num_ctx: int = 4096, temperature: float = 0.0) -> str:
#     try:
#         response = ollama.chat(model=OLLAMA_MODEL,
#             options={"num_ctx": num_ctx, "temperature": temperature},
#             messages=[{"role": "user", "content": prompt}])
#         return response['message']['content'].strip()
#     except Exception as e:
#         return f"LLaMA error: {str(e)}"


# def call_llama_streaming(prompt: str, request_id: str,
#                          num_ctx: int = 4096,
#                          temperature: float = 0.0) -> str:
#     global _ttft_tracker
#     full_response = ""
#     first_token   = True
#     stream_start  = time.time()
#     try:
#         stream = ollama.chat(model=OLLAMA_MODEL,
#             options={"num_ctx": num_ctx, "temperature": temperature},
#             messages=[{"role": "user", "content": prompt}],
#             stream=True)
#         for chunk in stream:
#             token = chunk["message"]["content"]
#             if token:
#                 if first_token and request_id:
#                     ttft = round(time.time() - stream_start, 2)
#                     _ttft_tracker[request_id] = ttft
#                     print(f"[STREAM] First token in {ttft}s")
#                     first_token = False
#                 full_response += token
#                 emit_event(request_id, "token", token)
#         return full_response.strip()
#     except Exception as e:
#         return f"LLaMA error: {str(e)}"