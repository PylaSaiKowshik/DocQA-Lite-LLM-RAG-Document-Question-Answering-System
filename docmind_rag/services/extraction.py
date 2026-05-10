import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz
import pytesseract
from PIL import Image

from docmind_rag.config.settings import MAX_WORKERS
from docmind_rag.utils.helpers import get_pdf_hash
_extraction_cache: dict = {}

# ============================================================
# CORE: EXTRACT SINGLE PAGE
# ============================================================
def extract_page(args):
    page, page_num = args
    try:
        text = page.get_text().strip()
        if len(text) > 10:
            return page_num, text

        if page.rect.width < 10 or page.rect.height < 10:
            return page_num, ""

        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        try:
            t = pytesseract.image_to_string(img).strip()
            real_words = len([w for w in t.split() if len(w) > 2 and w.isalpha()])
            if real_words > 5:
                return page_num, t
        except Exception:
            pass

        best_text = ""
        best_len  = 0
        for angle in [90, 180, 270]:
            rotated = img.rotate(angle, expand=True)
            try:
                t = pytesseract.image_to_string(rotated).strip()
                real_words = len([w for w in t.split() if len(w) > 2 and w.isalpha()])
                if real_words > best_len:
                    best_len  = real_words
                    best_text = t
            except Exception:
                continue

        return page_num, best_text.strip()
    except Exception:
        return page_num, ""


def extract_pdf_parallel(pdf_path: str):
    pdf_hash = get_pdf_hash(pdf_path)
    if pdf_hash in _extraction_cache:
        text, page_count = _extraction_cache[pdf_hash]
        print(f"[Extract] ✅ Cache hit — skipping OCR ({page_count} pages)")
        return text, page_count

    doc        = fitz.open(pdf_path)
    page_count = len(doc)
    pages      = [(doc[i], i) for i in range(page_count)]
    results    = {}
    workers    = min(MAX_WORKERS, page_count)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(extract_page, p): p[1] for p in pages}
        for future in as_completed(futures):
            page_num, text    = future.result()
            results[page_num] = text
    doc.close()

    page_texts = []
    for i in sorted(results.keys()):
        t = results[i].strip()
        if t:
            t = re.sub(r'[^\x00-\x7F]+', ' ', t)
            t = re.sub(r'\s+', ' ', t).strip()
            page_texts.append(f"--- PAGE {i} ---\n{t}")

    full_text = "\n\n".join(page_texts)
    _extraction_cache[pdf_hash] = (full_text, page_count)
    print(f"[Extract] Done — {page_count} pages, {len(full_text)} chars")
    return full_text, page_count
