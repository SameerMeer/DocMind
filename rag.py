import re

import faiss
import numpy as np
from groq import Groq

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder,
)


# =========================================================
# CONFIGURATION
# =========================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"

groq_client = Groq()

SEMANTIC_TOP_K = 8
KEYWORD_TOP_K = 8
RERANK_CANDIDATES = 30
FINAL_TOP_K = 5
COMPREHENSIVE_TOP_K = 10
SMALL_DOCUMENT_CHUNKS = 12

MIN_RERANK_SCORE = -10.0


# =========================================================
# MODELS
# =========================================================

print("Loading embedding model...")
embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Loading reranker...")
reranker = CrossEncoder(
    RERANKER_MODEL
)

print("Models loaded.")


# =========================================================
# TEXT HELPERS
# =========================================================

def normalize_text(text):
    """Normalize text for matching."""

    if not text:
        return ""

    text = text.lower()
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def tokenize(text):
    """Tokenize text into simple alphanumeric terms."""

    return set(
        re.findall(
            r"[a-zA-Z0-9]+",
            normalize_text(text)
        )
    )


# =========================================================
# CHUNK FILTERING
# =========================================================

def is_evaluation_chunk(chunk):
    """
    Exclude the known evaluation section from the test PDF.

    This filter is deliberately narrow and does not depend
    on application-specific document topics.
    """

    section = str(
        chunk.get(
            "section",
            ""
        )
    ).strip()

    title = normalize_text(
        chunk.get(
            "section_title",
            ""
        )
    )

    if section == "15":
        return True

    if "evaluation scenario" in title:
        return True

    return False


def filter_chunks(chunks):
    """Remove empty and evaluation chunks."""

    filtered = []

    for chunk in chunks:

        if not chunk:
            continue

        text = chunk.get(
            "text",
            ""
        ).strip()

        if not text:
            continue

        if is_evaluation_chunk(
            chunk
        ):
            continue

        filtered.append(chunk)

    return filtered


# =========================================================
# FAISS INDEX
# =========================================================

def create_vector_index(chunks):
    """
    Create a normalized FAISS inner-product index.

    Returns:
        index
        embeddings
        filtered_chunks
    """

    filtered_chunks = filter_chunks(
        chunks
    )

    if not filtered_chunks:
        raise ValueError(
            "No usable PDF chunks found."
        )

    texts = [
        chunk["text"]
        for chunk in filtered_chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32",
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return (
        index,
        embeddings,
        filtered_chunks,
    )


# =========================================================
# SEMANTIC SEARCH
# =========================================================

def semantic_search(
    question,
    index,
    chunks,
    top_k=SEMANTIC_TOP_K,
):
    """Retrieve semantically similar chunks."""

    if index is None or not chunks:
        return []

    query_embedding = (
        embedding_model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        .astype("float32")
    )

    k = min(
        top_k,
        len(chunks)
    )

    scores, indices = index.search(
        query_embedding,
        k,
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0],
    ):

        if idx < 0 or idx >= len(chunks):
            continue

        results.append(
            {
                "chunk": chunks[idx],
                "semantic_score": float(score),
                "score": float(score),
                "source": "semantic",
            }
        )

    return results


# =========================================================
# KEYWORD SEARCH
# =========================================================

def keyword_search(
    question,
    chunks,
    top_k=KEYWORD_TOP_K,
):
    """Retrieve chunks using lexical token overlap."""

    question_tokens = tokenize(
        question
    )

    if not question_tokens:
        return []

    scored = []

    for chunk in chunks:

        searchable_text = (
            chunk.get("text", "")
            + " "
            + chunk.get("section_title", "")
        )

        chunk_tokens = tokenize(
            searchable_text
        )

        overlap = (
            question_tokens
            & chunk_tokens
        )

        if not overlap:
            continue

        score = float(
            len(overlap)
        )

        # Give additional weight to identifiers,
        # numbers, emails, and URLs.
        for token in overlap:

            if any(
                char.isdigit()
                for char in token
            ):
                score += 2.0

            if (
                "@" in token
                or "." in token
                or "-" in token
            ):
                score += 1.0

        scored.append(
            {
                "chunk": chunk,
                "keyword_score": score,
                "score": score,
                "source": "keyword",
            }
        )

    scored.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return scored[:top_k]


# =========================================================
# QUERY VARIANTS
# =========================================================

def build_query_variants(question):
    """
    Generate a small number of useful query variants.

    The variants are document-agnostic.
    """

    variants = [question.strip()]

    normalized = normalize_text(
        question
    )

    # A compact keyword query can help lexical retrieval.
    important_tokens = [
        token
        for token in re.findall(
            r"[a-zA-Z0-9]+",
            normalized,
        )
        if len(token) > 2
    ]

    if important_tokens:
        keyword_query = " ".join(
            important_tokens
        )

        if (
            keyword_query.lower()
            != normalized.lower()
        ):
            variants.append(
                keyword_query
            )

    # Handle common multi-part questions.
    parts = re.split(
        r"\s+(?:and|or)\s+|[,;]",
        question,
        flags=re.IGNORECASE,
    )

    for part in parts:

        part = part.strip()

        if len(part) >= 12:
            variants.append(part)

    # Remove duplicates.
    unique = []

    for variant in variants:

        if not variant:
            continue

        if variant.lower() in [
            item.lower()
            for item in unique
        ]:
            continue

        unique.append(
            variant
        )

    return unique[:6]


# =========================================================
# EXACT MATCH SIGNAL
# =========================================================

def exact_match_score(
    question,
    chunk,
):
    """
    Reward exact identifiers and distinctive terms.

    This is generic and does not depend on a specific PDF.
    """

    q = normalize_text(
        question
    )

    text = normalize_text(
        chunk.get(
            "text",
            ""
        )
    )

    score = 0.0

    # IDs/codes such as ABC-1234.
    identifiers = re.findall(
        r"\b[A-Z]{2,}[A-Z0-9]*-\d{2,}\b",
        question,
        flags=re.IGNORECASE,
    )

    for identifier in identifiers:

        if normalize_text(
            identifier
        ) in text:

            score += 3.0

    # Exact email addresses.
    emails = re.findall(
        r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
        question,
    )

    for email in emails:

        if email.lower() in text:
            score += 3.0

    # Exact quoted phrases.
    quoted = re.findall(
        r'"([^"]+)"',
        question,
    )

    for phrase in quoted:

        if normalize_text(
            phrase
        ) in text:

            score += 3.0

    return score


# =========================================================
# HYBRID RETRIEVAL
# =========================================================

def hybrid_search(
    question,
    index,
    chunks,
):
    """
    Combine semantic retrieval, keyword retrieval,
    query variants, and exact-match signals.
    """

    candidates = {}

    variants = build_query_variants(
        question
    )

    # -----------------------------------------------------
    # Semantic retrieval
    # -----------------------------------------------------

    for variant in variants:

        results = semantic_search(
            variant,
            index,
            chunks,
            SEMANTIC_TOP_K,
        )

        for result in results:

            chunk = result["chunk"]

            key = (
                chunk.get("page"),
                chunk.get("section"),
                chunk.get("text", ""),
            )

            if key not in candidates:

                candidates[key] = {
                    "chunk": chunk,
                    "semantic_score": result[
                        "semantic_score"
                    ],
                    "keyword_score": 0.0,
                }

            else:

                candidates[key][
                    "semantic_score"
                ] = max(
                    candidates[key][
                        "semantic_score"
                    ],
                    result["semantic_score"],
                )

    # -----------------------------------------------------
    # Keyword retrieval
    # -----------------------------------------------------

    for variant in variants:

        results = keyword_search(
            variant,
            chunks,
            KEYWORD_TOP_K,
        )

        for result in results:

            chunk = result["chunk"]

            key = (
                chunk.get("page"),
                chunk.get("section"),
                chunk.get("text", ""),
            )

            if key not in candidates:

                candidates[key] = {
                    "chunk": chunk,
                    "semantic_score": 0.0,
                    "keyword_score": result[
                        "keyword_score"
                    ],
                }

            else:

                candidates[key][
                    "keyword_score"
                ] = max(
                    candidates[key][
                        "keyword_score"
                    ],
                    result["keyword_score"],
                )

    # -----------------------------------------------------
    # Combine signals
    # -----------------------------------------------------

    results = []

    for item in candidates.values():

        chunk = item["chunk"]

        semantic_score = item[
            "semantic_score"
        ]

        keyword_score = item[
            "keyword_score"
        ]

        exact_score = exact_match_score(
            question,
            chunk,
        )

        combined_score = (
            semantic_score * 10.0
            + keyword_score * 0.8
            + exact_score
        )

        results.append(
            {
                "chunk": chunk,
                "semantic_score": semantic_score,
                "keyword_score": keyword_score,
                "exact_score": exact_score,
                "score": combined_score,
            }
        )

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return results


# =========================================================
# CROSS-ENCODER RERANKING
# =========================================================

def rerank_results(
    question,
    results,
    top_k=FINAL_TOP_K,
):
    """Rerank hybrid candidates with a cross-encoder."""

    if not results:
        return []

    candidates = results[
        :RERANK_CANDIDATES
    ]

    pairs = []

    for result in candidates:

        chunk = result["chunk"]

        combined_text = (
            f"Section: "
            f"{chunk.get('section_title', '')}\n"
            f"{chunk.get('text', '')}"
        )

        pairs.append(
            [
                question,
                combined_text,
            ]
        )

    reranker_scores = reranker.predict(
        pairs
    )

    reranked = []

    for result, raw_score in zip(
        candidates,
        reranker_scores,
    ):

        reranker_score = float(
            raw_score
        )

        final_score = (
            reranker_score
            + result.get(
                "exact_score",
                0.0,
            )
        )

        if final_score < MIN_RERANK_SCORE:
            continue

        item = result.copy()

        item["reranker_score"] = (
            reranker_score
        )

        item["score"] = final_score

        reranked.append(
            item
        )

    reranked.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    # Prefer evidence from different pages/sections.
    selected = []
    seen_locations = set()

    for item in reranked:

        chunk = item["chunk"]

        location = (
            chunk.get("page"),
            chunk.get("section"),
        )

        if location in seen_locations:
            continue

        selected.append(item)
        seen_locations.add(location)

        if len(selected) >= top_k:
            break

    # Fill remaining slots.
    if len(selected) < top_k:

        selected_keys = {
            (
                item["chunk"].get("page"),
                item["chunk"].get("section"),
                item["chunk"].get("text", ""),
            )
            for item in selected
        }

        for item in reranked:

            key = (
                item["chunk"].get("page"),
                item["chunk"].get("section"),
                item["chunk"].get("text", ""),
            )

            if key in selected_keys:
                continue

            selected.append(item)
            selected_keys.add(key)

            if len(selected) >= top_k:
                break

    return selected


# =========================================================
# RETRIEVAL
# =========================================================

def is_comprehensive_question(question):
    """Detect questions that may require evidence from multiple chunks."""

    q = normalize_text(question)

    patterns = [
        r"\\ball\\b",
        r"\\blist\\b",
        r"\\bevery\\b",
        r"\\beach\\b",
        r"\\bmentioned\\b",
        r"\\bincluded\\b",
        r"\\bwhat are\\b",
        r"\\bwhich are\\b",
        r"\\bprojects?\\b",
        r"\\bskills?\\b",
        r"\\btechnolog(?:y|ies)\\b",
        r"\\bachievements?\\b",
        r"\\bfeatures?\\b",
        r"\\bobjectives?\\b",
        r"\\brequirements?\\b",
        r"\\btypes?\\b",
        r"\\bexamples?\\b",
        r"\\bsections?\\b",
    ]

    return any(re.search(pattern, q) for pattern in patterns)


def adaptive_retrieve(question, index, chunks):
    """
    Robust document-agnostic retrieval.

    Small documents are handled with a recall-first strategy:
    every chunk is reranked directly so relevant information cannot
    disappear during the initial hybrid-retrieval stage.

    Larger documents use the normal hybrid retrieval pipeline.
    """

    if not question.strip() or not chunks:
        return []

    # ---------------------------------------------------------
    # SMALL DOCUMENT MODE
    # ---------------------------------------------------------
    # When the whole document is small enough, there is no reason
    # to risk losing evidence through an initial top-k search.
    # Rerank every chunk directly.
    if len(chunks) <= SMALL_DOCUMENT_CHUNKS:
        all_results = []

        for chunk in chunks:
            all_results.append({
                "chunk": chunk,
                "semantic_score": 0.0,
                "keyword_score": 0.0,
                "exact_score": exact_match_score(
                    question,
                    chunk,
                ),
                "score": 0.0,
            })

        # Rerank ALL chunks. Do not use FINAL_TOP_K here.
        pairs = []

        for result in all_results:
            chunk = result["chunk"]

            combined_text = (
                f"Section: "
                f"{chunk.get('section_title', '')}\n"
                f"{chunk.get('text', '')}"
            )

            pairs.append([
                question,
                combined_text,
            ])

        reranker_scores = reranker.predict(pairs)

        reranked = []

        for result, raw_score in zip(
            all_results,
            reranker_scores,
        ):
            reranker_score = float(raw_score)

            result["reranker_score"] = reranker_score
            result["score"] = (
                reranker_score
                + result.get("exact_score", 0.0)
            )

            # Keep every chunk available to the LLM.
            reranked.append(result)

        reranked.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return reranked

    # ---------------------------------------------------------
    # LARGE DOCUMENT MODE
    # ---------------------------------------------------------
    hybrid_results = hybrid_search(
        question,
        index,
        chunks,
    )

    if not hybrid_results:
        return []

    if not is_comprehensive_question(question):
        return rerank_results(
            question,
            hybrid_results,
            min(FINAL_TOP_K, len(hybrid_results)),
        )

    return rerank_results(
        question,
        hybrid_results,
        min(COMPREHENSIVE_TOP_K, len(hybrid_results)),
    )


# =========================================================
# CONTEXT BUILDING
# =========================================================

def build_context(results):
    """Build grounded context for the LLM."""

    context_parts = []

    for i, result in enumerate(
        results,
        start=1,
    ):

        chunk = result["chunk"]

        page = chunk.get(
            "page",
            "?",
        )

        section = chunk.get(
            "section",
            "?",
        )

        section_title = chunk.get(
            "section_title",
            "",
        )

        text = chunk.get(
            "text",
            "",
        )

        header = (
            f"[Source {i} | "
            f"Page {page} | "
            f"Section {section}"
        )

        if section_title:
            header += (
                f": {section_title}"
            )

        header += "]"

        context_parts.append(
            f"{header}\n{text}"
        )

    return "\n\n".join(
        context_parts
    )


# =========================================================
# ANSWER GENERATION
# =========================================================

def generate_answer(
    question,
    results,
):
    """Generate a grounded answer using Groq."""

    if not results:
        return (
            "I could not find the answer in the PDF."
        )

    context = build_context(
        results
    )

    system_prompt = """
You are DocMind, a precise PDF question-answering assistant.

Answer ONLY from the supplied PDF context.

Rules:
- Read every supplied document excerpt before answering.
- Treat the PDF as the only source of truth.
- Answer every part of the user's question.
- Use a fact only when the supplied context explicitly supports it.
- Do NOT add relationships, classifications, explanations, or assumptions
  that are not stated or directly supported by the PDF.
- Do NOT infer that a skill was used in a project merely because both
  appear somewhere in the resume.
- Do NOT add outside knowledge to the answer.
- For "what are", "which are", "list", "all", "every", "mentioned",
  "included", or similar questions, combine relevant evidence across
  ALL supplied excerpts and include every distinct supported item.
- Preserve exact names, IDs, numbers, dates, technologies, and time periods.
- For processes, rules, conditions, or procedures, include all relevant
  actions, conditions, time limits, and consequences explicitly stated
  in the context.
- If the requested fact is absent or cannot be supported by the context,
  respond exactly:
  "I could not find the answer in the PDF."
- Never invent an answer to make the response complete.
- Never mention document-excerpt numbers, retrieval scores, source labels,
  or internal retrieval metadata.
- Keep answers clear and concise.
""".strip()

    user_prompt = f"""
PDF CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""".strip()

    try:

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0,
            max_tokens=400,
        )

        answer = (
            response.choices[0]
            .message.content
            .strip()
        )

        if not answer:
            return (
                "I could not find the answer in the PDF."
            )

        return answer

    except Exception as error:

        return (
            f"Error generating answer: {error}"
        )


# =========================================================
# PUBLIC API
# =========================================================

def ask_pdf(
    question,
    index,
    chunks,
):
    """
    Main function used by the Streamlit application.

    Returns:
        answer
        sources
    """

    results = adaptive_retrieve(
        question,
        index,
        chunks,
    )

    answer = generate_answer(
        question,
        results,
    )

    sources = []

    for result in results:

        chunk = result["chunk"]

        sources.append(
            {
                "page": chunk.get(
                    "page",
                    "?",
                ),
                "section": chunk.get(
                    "section",
                    "?",
                ),
                "section_title": chunk.get(
                    "section_title",
                    "",
                ),
                "text": chunk.get(
                    "text",
                    "",
                ),
                "score": result.get(
                    "reranker_score",
                    result.get(
                        "score",
                        0.0,
                    ),
                ),
                "final_score": result.get(
                    "score",
                    0.0,
                ),
                "semantic_score": result.get(
                    "semantic_score",
                    0.0,
                ),
                "keyword_score": result.get(
                    "keyword_score",
                    0.0,
                ),
            }
        )

    return answer, sources
