import sys, os

PDF = r"C:\Users\24048\Documents\xwechat_files\wxid_a2ktmwp277zj22_0df1\msg\file\2026-09\srs.pdf"
OUT = r"C:\Users\24048\WorkBuddy\2026-09-15-21-45-29\_srs.txt"

text = None
err = []

# try pypdf
try:
    from pypdf import PdfReader
    r = PdfReader(PDF)
    parts = []
    for i, pg in enumerate(r.pages):
        parts.append(f"\n===== PAGE {i+1} =====\n")
        parts.append(pg.extract_text() or "")
    text = "".join(parts)
except Exception as e:
    err.append(f"pypdf failed: {e!r}")

if not text:
    try:
        import fitz
        d = fitz.open(PDF)
        parts = []
        for i, pg in enumerate(d):
            parts.append(f"\n===== PAGE {i+1} =====\n")
            parts.append(pg.get_text())
        text = "".join(parts)
    except Exception as e:
        err.append(f"pymupdf failed: {e!r}")

with open(OUT, "w", encoding="utf-8") as f:
    if text:
        f.write(text)
    else:
        f.write("EXTRACTION FAILED\n" + "\n".join(err))

print("done")
