import os
import json
import tempfile

# ============================================================
# STREAMING EVENTS
# ============================================================
def get_event_file(request_id: str) -> str:
    return os.path.join(tempfile.gettempdir(), f"docmind_{request_id}.jsonl")

def emit_event(request_id: str, event_type: str, message: str):
    if not request_id:
        return
    try:
        line = json.dumps({"type": event_type, "message": message})
        with open(get_event_file(request_id), "a") as f:
            f.write(line + "\n")
    except Exception:
        pass

def cleanup_queue(request_id: str):
    try:
        line = json.dumps({"type": "done", "message": "complete"})
        with open(get_event_file(request_id), "a") as f:
            f.write(line + "\n")
    except Exception:
        pass