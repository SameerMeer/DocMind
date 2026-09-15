import re
from pypdf import PdfReader


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):
    """Clean extracted PDF text while preserving content."""

    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Normalize spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


# =========================================================
# SECTION HEADING DETECTION
# =========================================================

# Matches headings such as:
#
# 1. Project Overview
# 2. Main Objectives
# 7. Technical Architecture
# 12. Frequently Asked Questions
#
# But NOT ordinary numbered list items such as:
#
# 1. Reduce manual work.
#
# because those are normally followed by sentence-like text.

SECTION_HEADING_PATTERN = re.compile(
    r"(?m)^\s*(\d{1,3})\.\s+"
    r"([A-Z][^\n]{2,100})"
    r"\s*$"
)


def find_section_headings(text):
    """
    Detect numbered section headings.

    Returns:
        [
            {
                "number": 1,
                "title": "Project Overview",
                "start": ...,
                "end": ...
            }
        ]
    """

    matches = []

    for match in SECTION_HEADING_PATTERN.finditer(text):

        number = int(match.group(1))
        title = match.group(2).strip()

        # Remove trailing punctuation
        title = title.rstrip(" :")

        # Avoid treating obvious numbered sentences as headings.
        #
        # A heading normally starts with a capital letter and
        # is relatively short.

        if len(title) > 100:
            continue

        # Skip common list-like sentences.
        if title.endswith("."):
            continue

        matches.append({
            "number": number,
            "title": title,
            "start": match.start(),
            "end": match.end()
        })

    return matches


# =========================================================
# EVALUATION / TEST CONTENT
# =========================================================

def is_evaluation_section(section_number, text):
    """
    Detect evaluation/test sections.

    We exclude explicit evaluation scenarios so that the
    chatbot does not retrieve the test questions themselves.
    """

    if section_number == 15:
        return True

    if not text:
        return False

    lower_text = text.lower()

    markers = [
        "evaluation scenarios",
        "direct retrieval:",
        "list retrieval:",
        "numeric retrieval:",
        "cross-section reasoning:",
        "multi-hop retrieval:",
        "exact identifier:",
        "technical retrieval:",
        "negative test:"
    ]

    return any(
        marker in lower_text
        for marker in markers
    )


# =========================================================
# CREATE CHUNK
# =========================================================

def create_chunk(
    page_number,
    section_number,
    section_title,
    text
):
    """Create a structured chunk."""

    text = clean_text(text)

    if not text:
        return None

    if is_evaluation_section(
        section_number,
        text
    ):
        return None

    return {
        "page": page_number,
        "section": section_number,
        "section_title": section_title,
        "text": text
    }


# =========================================================
# PROCESS PDF
# =========================================================

def process_pdf(pdf_path):
    """
    Convert a PDF into structured section-aware chunks.

    Works with different numbered-section PDFs instead of
    depending on one hardcoded document.
    """

    reader = PdfReader(pdf_path)

    chunks = []

    # Current section continues across pages.
    current_section_number = None
    current_section_title = None

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        raw_text = page.extract_text()

        if not raw_text:
            continue

        cleaned = clean_text(raw_text)

        if not cleaned:
            continue

        matches = find_section_headings(
            cleaned
        )

        # =====================================================
        # NO HEADING ON THIS PAGE
        # =====================================================

        if not matches:

            if current_section_number is not None:

                chunk = create_chunk(
                    page_number,
                    current_section_number,
                    current_section_title,
                    cleaned
                )

                if chunk:
                    chunks.append(chunk)

            continue

        # =====================================================
        # PROCESS SECTIONS ON THIS PAGE
        # =====================================================

        for i, match in enumerate(matches):

            section_number = match["number"]
            section_title = match["title"]

            start = match["end"]

            if i + 1 < len(matches):

                end = matches[i + 1]["start"]

            else:

                end = len(cleaned)

            section_text = cleaned[
                start:end
            ].strip()

            # Update current section BEFORE processing
            # continuation pages.
            current_section_number = (
                section_number
            )

            current_section_title = (
                section_title
            )

            # Evaluation section is never indexed.
            if section_number == 15:
                continue

            if is_evaluation_section(
                section_number,
                section_text
            ):
                continue

            chunk = create_chunk(
                page_number,
                section_number,
                section_title,
                section_text
            )

            if chunk:
                chunks.append(chunk)

    # =========================================================
    # REMOVE DUPLICATES
    # =========================================================

    final_chunks = []

    seen = set()

    for chunk in chunks:

        key = (
            chunk["page"],
            chunk["section"],
            chunk["text"]
        )

        if key in seen:
            continue

        seen.add(key)
        final_chunks.append(chunk)

    return final_chunks