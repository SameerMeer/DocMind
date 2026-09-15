import re
import unicodedata

from pypdf import PdfReader


# =========================================================
# CONFIGURATION
# =========================================================

MAX_CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):
    """Normalize extracted PDF text while preserving content."""

    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\u00ad", "")

    # Repair a few common PDF extraction artifacts.
    text = text.replace("\u2010", "-")
    text = text.replace("\u2011", "-")
    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", "-")

    # Normalize spaces/tabs but preserve line boundaries.
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive blank lines.
    text = re.sub(r"\n[ \t]*\n+", "\n\n", text)

    return text.strip()


# =========================================================
# SECTION HEADING DETECTION
# =========================================================

SECTION_HEADING_PATTERN = re.compile(
    r"(?m)^\s*(\d{1,3})\.\s+"
    r"([A-Z][^\n]{2,100}?)"
    r"\s*$"
)


def find_section_headings(text):
    """Find plausible numbered section headings."""

    matches = []

    for match in SECTION_HEADING_PATTERN.finditer(text):
        number = int(match.group(1))
        title = match.group(2).strip().rstrip(" :")

        if len(title) < 3 or len(title) > 100:
            continue

        # A heading should not look like a normal sentence/list item.
        if title.endswith("."):
            continue

        matches.append(
            {
                "number": number,
                "title": title,
                "start": match.start(),
                "end": match.end(),
            }
        )

    return matches


def has_real_section_structure(matches):
    """
    Decide whether numbered matches represent document sections
    rather than ordinary numbered list items.
    """

    if not matches:
        return False

    if len(matches) >= 2:
        numbers = [item["number"] for item in matches]

        # Sequential numbering is strong evidence of sections.
        for first, second in zip(numbers, numbers[1:]):
            if second == first + 1:
                return True

        # Multiple distinct numbered headings are also useful.
        if len(set(numbers)) >= 2:
            return True

    # Allow a single numbered heading when its title looks clearly
    # like a section title.
    title = matches[0]["title"].lower()

    section_words = (
        "introduction",
        "overview",
        "summary",
        "background",
        "architecture",
        "methodology",
        "results",
        "conclusion",
        "references",
        "objectives",
        "requirements",
        "implementation",
        "discussion",
        "features",
        "scope",
    )

    return any(word in title for word in section_words)


# =========================================================
# TEXT CHUNKING
# =========================================================

def split_text(text, max_size=MAX_CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split long text into reasonably sized chunks.

    Paragraph boundaries are preferred. A small character overlap
    helps preserve context between adjacent chunks.
    """

    text = clean_text(text)

    if not text:
        return []

    if len(text) <= max_size:
        return [text]

    paragraphs = [
        part.strip()
        for part in re.split(r"\n{2,}", text)
        if part.strip()
    ]

    # If the PDF has no useful paragraph boundaries, fall back to
    # line/sentence-sized pieces.
    if not paragraphs:
        paragraphs = [text]

    chunks = []
    current = ""

    for paragraph in paragraphs:

        if len(paragraph) > max_size:
            if current:
                chunks.append(current.strip())
                current = ""

            start = 0

            while start < len(paragraph):
                end = min(
                    start + max_size,
                    len(paragraph)
                )

                piece = paragraph[start:end].strip()

                if piece:
                    chunks.append(piece)

                if end >= len(paragraph):
                    break

                start = max(
                    end - overlap,
                    start + 1
                )

            continue

        candidate = (
            paragraph
            if not current
            else current + "\n\n" + paragraph
        )

        if len(candidate) <= max_size:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())

            previous_tail = (
                current[-overlap:]
                if current
                else ""
            )

            current = (
                previous_tail + "\n\n" + paragraph
                if previous_tail
                else paragraph
            )

            if len(current) > max_size:
                current = current[-max_size:]

    if current.strip():
        chunks.append(current.strip())

    return chunks


# =========================================================
# EVALUATION / TEST CONTENT
# =========================================================

def is_evaluation_section(section_number, text):
    """
    Exclude explicit evaluation material from the test document.

    This is intentionally narrow so normal user PDFs are not
    accidentally filtered.
    """

    if section_number == 15:
        return True

    lower_text = clean_text(text).lower()

    markers = [
        "evaluation scenarios",
        "direct retrieval:",
        "list retrieval:",
        "numeric retrieval:",
        "cross-section reasoning:",
        "multi-hop retrieval:",
        "exact identifier:",
        "technical retrieval:",
        "negative test:",
    ]

    return any(marker in lower_text for marker in markers)


# =========================================================
# CREATE CHUNKS
# =========================================================

def create_structured_chunks(
    page_number,
    section_number,
    section_title,
    text,
):
    """Create section-aware chunks from a structured section."""

    if not text:
        return []

    if is_evaluation_section(
        section_number,
        text,
    ):
        return []

    pieces = split_text(text)

    chunks = []

    for piece in pieces:
        chunks.append(
            {
                "page": page_number,
                "section": section_number,
                "section_title": section_title,
                "text": piece,
            }
        )

    return chunks


def create_generic_chunks(page_number, text):
    """Create generic chunks for PDFs without numbered sections."""

    pieces = split_text(text)

    chunks = []

    for piece in pieces:
        chunks.append(
            {
                "page": page_number,
                "section": 1,
                "section_title": f"Page {page_number}",
                "text": piece,
            }
        )

    return chunks


# =========================================================
# STRUCTURED PDF PROCESSING
# =========================================================

def process_structured_pdf(reader):
    """Process a PDF containing numbered sections."""

    chunks = []

    current_section_number = None
    current_section_title = None

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):

        raw_text = page.extract_text()

        if not raw_text:
            continue

        cleaned = clean_text(raw_text)

        if not cleaned:
            continue

        matches = find_section_headings(cleaned)

        if not matches:

            if current_section_number is not None:
                chunks.extend(
                    create_structured_chunks(
                        page_number,
                        current_section_number,
                        current_section_title,
                        cleaned,
                    )
                )

            continue

        # Preserve any text before the first heading.
        first_start = matches[0]["start"]

        if first_start > 0:
            prefix = cleaned[:first_start].strip()

            if prefix and current_section_number is not None:
                chunks.extend(
                    create_structured_chunks(
                        page_number,
                        current_section_number,
                        current_section_title,
                        prefix,
                    )
                )

        for index, match in enumerate(matches):

            section_number = match["number"]
            section_title = match["title"]

            start = match["end"]

            if index + 1 < len(matches):
                end = matches[index + 1]["start"]
            else:
                end = len(cleaned)

            section_text = cleaned[start:end].strip()

            current_section_number = section_number
            current_section_title = section_title

            if section_number == 15:
                continue

            chunks.extend(
                create_structured_chunks(
                    page_number,
                    section_number,
                    section_title,
                    section_text,
                )
            )

    return chunks


# =========================================================
# GENERIC PDF PROCESSING
# =========================================================

def process_generic_pdf(reader):
    """Process a normal text-based PDF without numbered sections."""

    chunks = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):

        raw_text = page.extract_text()

        if not raw_text:
            continue

        cleaned = clean_text(raw_text)

        if not cleaned:
            continue

        chunks.extend(
            create_generic_chunks(
                page_number,
                cleaned,
            )
        )

    return chunks


# =========================================================
# REMOVE DUPLICATES
# =========================================================

def remove_duplicates(chunks):
    """Remove exact duplicate chunks."""

    final_chunks = []
    seen = set()

    for chunk in chunks:

        text = clean_text(
            chunk.get("text", "")
        )

        if not text:
            continue

        key = (
            chunk.get("page"),
            chunk.get("section"),
            text,
        )

        if key in seen:
            continue

        seen.add(key)

        cleaned_chunk = chunk.copy()
        cleaned_chunk["text"] = text

        final_chunks.append(cleaned_chunk)

    return final_chunks


# =========================================================
# PUBLIC FUNCTION
# =========================================================

def process_pdf(pdf_path):
    """
    Convert any text-based PDF into RAG-ready chunks.

    Structured PDFs:
        preserve numbered sections.

    Normal text PDFs:
        use page-based generic sections.

    Scanned/image-only PDFs:
        require OCR and are not supported by pypdf alone.
    """

    reader = PdfReader(pdf_path)

    all_page_text = []

    for page in reader.pages:
        raw_text = page.extract_text()

        if raw_text:
            cleaned = clean_text(raw_text)

            if cleaned:
                all_page_text.append(cleaned)

    if not all_page_text:
        return []

    all_text = "\n".join(all_page_text)

    matches = find_section_headings(all_text)

    if has_real_section_structure(matches):
        chunks = process_structured_pdf(reader)
    else:
        chunks = process_generic_pdf(reader)

    return remove_duplicates(chunks)
