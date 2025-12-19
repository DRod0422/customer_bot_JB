import os
from pypdf import PdfReader

PDF_DIR = "../data/pdfs"

def pdf_text_score(path: str) -> int:
    # returns number of extracted characters from first few pages
    try:
        r = PdfReader(path)
        total = 0
        for i, p in enumerate(r.pages[:3]):  # sample first 3 pages
            t = p.extract_text() or ""
            total += len(t.strip())
        return total
    except Exception:
        return -1

def main():
    pdfs = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    pdfs.sort()

    scanned = []
    errors = []
    ok = []

    for f in pdfs:
        p = os.path.join(PDF_DIR, f)
        score = pdf_text_score(p)
        if score == -1:
            errors.append(f)
        elif score < 50:  # very low text on first pages
            scanned.append((f, score))
        else:
            ok.append((f, score))

    print("Total PDFs:", len(pdfs))
    print("Likely text-based:", len(ok))
    print("Likely scanned/no text:", len(scanned))
    print("Errors reading:", len(errors))

    if scanned:
        print("\nTop 15 likely scanned (lowest extracted chars):")
        for f, s in scanned[:15]:
            print(f"- {f} (chars={s})")

    if errors:
        print("\nErrors (first 10):")
        for f in errors[:10]:
            print(f"- {f}")

if __name__ == "__main__":
    main()
