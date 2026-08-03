import re
import uuid
from bs4 import BeautifulSoup

SECTION_PATTERNS = [
    (r"item\s*1a[\.\s]", "Risk Factors"),
    (r"item\s*1[\.\s]", "Business"),
    (r"item\s*7a[\.\s]", "Market Risk"),
    (r"item\s*7[\.\s]", "MD&A"),
    (r"item\s*8[\.\s]", "Financial Statements"),
]

CHUNK_WORDS = 400
OVERLAP_WORDS = 50


def html_to_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "table"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()


def split_into_sections(text):
    combined = "|".join(f"({p})" for p, _ in SECTION_PATTERNS)
    tokens = re.split(combined, text, flags=re.IGNORECASE)
    sections = []
    current_label = "Preamble"
    current_text = []
    for token in tokens:
        if token is None:
            continue
        matched = next((label for pat, label in SECTION_PATTERNS
                        if re.match(pat, token.strip(), re.IGNORECASE)), None)
        if matched:
            if current_text:
                sections.append({"section": current_label, "text": " ".join(current_text).strip()})
            current_label = matched
            current_text = []
        else:
            current_text.append(token)
    if current_text:
        sections.append({"section": current_label, "text": " ".join(current_text).strip()})
    return [s for s in sections if len(s["text"].split()) > 50]


def chunk_section(text, chunk_words=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_words - overlap
    return chunks


def parse_filing(filing):
    print("[parser] Converting HTML to text...")
    text = html_to_text(filing["html"])
    print(f"[parser] {len(text.split()):,} words.")
    sections = split_into_sections(text)
    print(f"[parser] Sections: {[s['section'] for s in sections]}")
    chunks = []
    for section in sections:
        for idx, chunk_text in enumerate(chunk_section(section["text"])):
            chunks.append({
                "id": str(uuid.uuid4()),
                "ticker": filing["ticker"],
                "year": filing["year"],
                "filing_date": filing["filing_date"],
                "doc_type": "10-K",
                "section": section["section"],
                "chunk_index": idx,
                "text": chunk_text,
            })
    print(f"[parser] {len(chunks)} chunks.")
    return chunks
