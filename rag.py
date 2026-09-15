import re
import numpy as np
import faiss
import ollama

from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OLLAMA_MODEL = "llama3.2:3b"

SEMANTIC_TOP_K = 8
KEYWORD_TOP_K = 8
FINAL_TOP_K = 5

# Cross-encoder scores are logits, NOT probabilities.
# Do not reject results using an arbitrary positive threshold.
MIN_RERANK_SCORE = -10.0

# Boosts used to prefer direct/authoritative sections.
DIRECT_SECTION_BOOST = 1.5
FAQ_SECTION_PENALTY = 1.0
EXACT_MATCH_BOOST = 3.0


# ============================================================
# MODELS
# ============================================================

print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

print("Loading reranker...")
reranker = CrossEncoder(RERANKER_MODEL)

print("Models loaded.")


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_text(text):
    """Normalize text for matching."""
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize(text):
    """Simple tokenization."""
    text = normalize_text(text)
    return set(re.findall(r"[a-zA-Z0-9]+", text))


# ============================================================
# EVALUATION FILTER
# ============================================================

def is_evaluation_chunk(chunk):
    """
    Prevent evaluation/test instructions from becoming
    answer sources.
    """

    section = str(chunk.get("section", "")).strip()
    title = normalize_text(chunk.get("section_title", ""))

    if section == "15":
        return True

    if "evaluation scenario" in title:
        return True

    text = normalize_text(chunk.get("text", ""))

    evaluation_phrases = [
        "evaluation scenarios",
        "test the chatbot",
        "expected answer",
        "deliberately absent information",
    ]

    return any(phrase in text for phrase in evaluation_phrases)


def filter_chunks(chunks):
    """Remove unusable chunks."""

    filtered = []

    for chunk in chunks:
        if not chunk:
            continue

        text = chunk.get("text", "").strip()

        if not text:
            continue

        if is_evaluation_chunk(chunk):
            continue

        filtered.append(chunk)

    return filtered


# ============================================================
# FAISS INDEX
# ============================================================

def create_vector_index(chunks):
    """
    Create FAISS index.

    Returns:
        index
        embeddings
        filtered_chunks

    IMPORTANT:
    filtered_chunks must be used with the index so vector
    positions always match chunk positions.
    """

    filtered_chunks = filter_chunks(chunks)

    if not filtered_chunks:
        raise ValueError("No usable PDF chunks found.")

    texts = [
        chunk["text"]
        for chunk in filtered_chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    # Inner Product + normalized vectors = cosine similarity.
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index, embeddings, filtered_chunks


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(
    question,
    index,
    chunks,
    top_k=SEMANTIC_TOP_K
):
    """FAISS semantic search."""

    if index is None or not chunks:
        return []

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")

    k = min(top_k, len(chunks))

    scores, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0 or idx >= len(chunks):
            continue

        chunk = chunks[idx]

        results.append({
            "chunk": chunk,
            "semantic_score": float(score),
            "score": float(score),
            "source": "semantic"
        })

    return results


# ============================================================
# KEYWORD SEARCH
# ============================================================

def keyword_search(
    question,
    chunks,
    top_k=KEYWORD_TOP_K
):
    """
    Fast lexical search.

    Particularly useful for:
    - IDs
    - numbers
    - technologies
    - names
    - exact terms
    """

    question_tokens = tokenize(question)

    if not question_tokens:
        return []

    scored = []

    for chunk in chunks:

        text = normalize_text(
            chunk.get("text", "")
        )

        title = normalize_text(
            chunk.get("section_title", "")
        )

        chunk_tokens = tokenize(
            text + " " + title
        )

        overlap = question_tokens & chunk_tokens

        if not overlap:
            continue

        # Basic token overlap.
        score = len(overlap)

        # Extra weight for rare/exact identifiers.
        exact_bonus = 0

        for token in overlap:

            # IDs / codes
            if re.search(r"\d", token):
                exact_bonus += 2

            # Technical terms
            if token in {
                "postgresql",
                "react",
                "django",
                "rest",
                "framework",
                "jwt",
                "gunicorn",
                "nginx",
                "api",
                "database",
                "frontend",
                "backend",
            }:
                exact_bonus += 1

        score += exact_bonus

        scored.append({
            "chunk": chunk,
            "keyword_score": float(score),
            "score": float(score),
            "source": "keyword"
        })

    scored.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return scored[:top_k]


# ============================================================
# QUERY EXPANSION
# ============================================================

KEYWORD_MAP = {

    "database": [
        "database",
        "postgresql",
        "postgres",
        "db"
    ],

    "frontend": [
        "frontend",
        "front end",
        "react",
        "react.js"
    ],

    "backend": [
        "backend",
        "back end",
        "django",
        "django rest framework",
        "drf"
    ],

    "student": [
        "student",
        "student role",
        "student capabilities"
    ],

    "faculty": [
        "faculty",
        "faculty member",
        "teacher"
    ],

    "borrow": [
        "borrow",
        "borrowing",
        "loan",
        "keep",
        "days",
        "book"
    ],

    "reserve": [
        "reserve",
        "reservation",
        "reserved",
        "available",
        "48 hours"
    ],

    "member": [
        "member",
        "member id",
        "membership"
    ],

    "book": [
        "book",
        "book id",
        "title"
    ],

    "security": [
        "security",
        "jwt",
        "authentication",
        "authorization"
    ]
}


def expand_query(question):
    """
    Add useful domain terms without making the query huge.
    """

    normalized = normalize_text(question)

    expanded = [question]

    for key, alternatives in KEYWORD_MAP.items():

        if key in normalized:

            for term in alternatives:

                if term not in normalized:
                    expanded.append(term)

    return expanded[:5]


# ============================================================
# QUERY VARIANTS FOR MULTI-PART QUESTIONS
# ============================================================

def build_query_variants(question):
    """
    Generate focused queries.

    Example:

    "What can a student do, and how long can they keep a book?"

    becomes roughly:

    1. original question
    2. student capabilities
    3. student borrowing / loan duration
    """

    normalized = normalize_text(question)

    variants = [question]

    # Student role
    if "student" in normalized:

        variants.append(
            "student user role capabilities actions"
        )

        if any(
            word in normalized
            for word in [
                "borrow",
                "keep",
                "long",
                "days",
                "loan"
            ]
        ):
            variants.append(
                "student borrowing policy loan duration"
            )

    # Faculty
    if "faculty" in normalized:
        variants.append(
            "faculty borrowing limit loan duration"
        )

    # Database / architecture
    if any(
        word in normalized
        for word in [
            "database",
            "frontend",
            "backend",
            "technology",
            "architecture"
        ]
    ):
        variants.append(
            "technical architecture database frontend backend"
        )

    # Reservation
    if "reservation" in normalized or "reserved" in normalized:
        variants.append(
            "reservation rule available book 48 hours"
        )

    # Member ID
    if "member" in normalized and "id" in normalized:
        variants.append(
            "member record member ID"
        )

    # Book ID
    if "book" in normalized and "id" in normalized:
        variants.append(
            "book record book ID"
        )

    # Remove duplicates
    unique = []

    for variant in variants:

        if variant.lower() not in [
            x.lower() for x in unique
        ]:
            unique.append(variant)

    return unique[:6]


# ============================================================
# SECTION PRIORITY
# ============================================================

def section_priority(question, chunk):
    """
    Prefer sections that directly answer the question.

    This helps when FAQ duplicates an answer from a
    more authoritative section.
    """

    q = normalize_text(question)

    title = normalize_text(
        chunk.get("section_title", "")
    )

    section = str(
        chunk.get("section", "")
    )

    boost = 0.0

    # Direct authoritative sections.
    if "database" in q:
        if "technical architecture" in title:
            boost += DIRECT_SECTION_BOOST

    if "frontend" in q or "backend" in q:
        if "technical architecture" in title:
            boost += DIRECT_SECTION_BOOST

    if "student" in q:
        if "user roles" in title:
            boost += DIRECT_SECTION_BOOST

    if "faculty" in q:
        if "user roles" in title:
            boost += DIRECT_SECTION_BOOST

    if (
        "borrow" in q
        or "keep" in q
        or "loan" in q
    ):
        if "borrowing policy" in title:
            boost += DIRECT_SECTION_BOOST

    if (
        "reservation" in q
        or "reserved" in q
    ):
        if "reservation rule" in title:
            boost += DIRECT_SECTION_BOOST

    if "security" in q:
        if "security rules" in title:
            boost += DIRECT_SECTION_BOOST

    if "member id" in q:
        if "example member record" in title:
            boost += DIRECT_SECTION_BOOST

    if "book id" in q:
        if "example book record" in title:
            boost += DIRECT_SECTION_BOOST

    # FAQ is useful, but direct sections are better.
    if "frequently asked questions" in title:
        boost -= FAQ_SECTION_PENALTY

    return boost


# ============================================================
# EXACT MATCH BOOST
# ============================================================

def exact_match_score(question, chunk):
    """
    Strongly reward exact IDs, numbers, names and
    technical terms appearing in the chunk.
    """

    q = normalize_text(question)
    text = normalize_text(
        chunk.get("text", "")
    )

    score = 0.0

    # Exact identifiers
    identifiers = re.findall(
        r"\b[A-Z]{2,}-\d{3,}\b",
        question
    )

    for identifier in identifiers:

        if identifier.lower() in text:
            score += EXACT_MATCH_BOOST

    # Important technical terms.
    exact_terms = [
        "postgresql",
        "react.js",
        "django rest framework",
        "jwt",
        "gunicorn",
        "nginx"
    ]

    for term in exact_terms:

        if term in q and term in text:
            score += EXACT_MATCH_BOOST

    return score


# ============================================================
# HYBRID RETRIEVAL
# ============================================================

def hybrid_search(
    question,
    index,
    chunks
):
    """
    Combine semantic + keyword + query variants.

    Returns a broad candidate pool.
    """

    candidates = {}

    variants = build_query_variants(question)

    # --------------------------------------------------------
    # Semantic retrieval
    # --------------------------------------------------------

    for variant in variants:

        results = semantic_search(
            variant,
            index,
            chunks,
            SEMANTIC_TOP_K
        )

        for result in results:

            chunk = result["chunk"]

            key = (
                chunk.get("page"),
                chunk.get("section"),
                chunk.get("text", "")
            )

            if key not in candidates:

                candidates[key] = {
                    "chunk": chunk,
                    "semantic_score": result[
                        "semantic_score"
                    ],
                    "keyword_score": 0.0
                }

            else:

                candidates[key][
                    "semantic_score"
                ] = max(
                    candidates[key]["semantic_score"],
                    result["semantic_score"]
                )

    # --------------------------------------------------------
    # Keyword retrieval
    # --------------------------------------------------------

    keyword_variants = [
        question
    ]

    if len(variants) > 1:
        keyword_variants.extend(
            variants[1:]
        )

    for variant in keyword_variants:

        results = keyword_search(
            variant,
            chunks,
            KEYWORD_TOP_K
        )

        for result in results:

            chunk = result["chunk"]

            key = (
                chunk.get("page"),
                chunk.get("section"),
                chunk.get("text", "")
            )

            if key not in candidates:

                candidates[key] = {
                    "chunk": chunk,
                    "semantic_score": 0.0,
                    "keyword_score": result[
                        "keyword_score"
                    ]
                }

            else:

                candidates[key][
                    "keyword_score"
                ] = max(
                    candidates[key]["keyword_score"],
                    result["keyword_score"]
                )

    # --------------------------------------------------------
    # Combined pre-reranking score
    # --------------------------------------------------------

    results = []

    for item in candidates.values():

        chunk = item["chunk"]

        semantic_score = item[
            "semantic_score"
        ]

        keyword_score = item[
            "keyword_score"
        ]

        priority = section_priority(
            question,
            chunk
        )

        exact = exact_match_score(
            question,
            chunk
        )

        # Semantic similarity is the main signal.
        combined = (
            semantic_score * 10.0
            + keyword_score * 0.8
            + priority
            + exact
        )

        results.append({
            "chunk": chunk,
            "semantic_score": semantic_score,
            "keyword_score": keyword_score,
            "priority_score": priority,
            "exact_score": exact,
            "score": combined
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results


# ============================================================
# RERANKING
# ============================================================

def rerank_results(
    question,
    results,
    top_k=FINAL_TOP_K
):
    """
    Cross-encoder reranking + section-aware scoring.
    """

    if not results:
        return []

    pairs = []

    for result in results:

        chunk = result["chunk"]

        text = chunk.get("text", "")

        title = chunk.get(
            "section_title",
            ""
        )

        combined_text = (
            f"Section: {title}\n"
            f"{text}"
        )

        pairs.append(
            [question, combined_text]
        )

    reranker_scores = reranker.predict(
        pairs
    )

    reranked = []

    for result, raw_score in zip(
        results,
        reranker_scores
    ):

        score = float(raw_score)

        final_score = (
            score
            + result["priority_score"]
            + result["exact_score"]
        )

        if final_score < MIN_RERANK_SCORE:
            continue

        item = result.copy()

        item["reranker_score"] = score
        item["score"] = final_score

        reranked.append(item)

    reranked.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # --------------------------------------------------------
    # Diversity:
    # Keep useful evidence from multiple sections.
    # --------------------------------------------------------

    selected = []

    seen_sections = set()

    # First pass: one strong result per section.
    for item in reranked:

        chunk = item["chunk"]

        section_key = (
            chunk.get("page"),
            chunk.get("section")
        )

        if section_key not in seen_sections:

            selected.append(item)
            seen_sections.add(section_key)

        if len(selected) >= top_k:
            break

    # Second pass: fill remaining slots.
    if len(selected) < top_k:

        selected_keys = {
            (
                x["chunk"].get("page"),
                x["chunk"].get("section"),
                x["chunk"].get("text", "")
            )
            for x in selected
        }

        for item in reranked:

            key = (
                item["chunk"].get("page"),
                item["chunk"].get("section"),
                item["chunk"].get("text", "")
            )

            if key in selected_keys:
                continue

            selected.append(item)
            selected_keys.add(key)

            if len(selected) >= top_k:
                break

    return selected


# ============================================================
# ADAPTIVE RETRIEVAL
# ============================================================

def adaptive_retrieve(question, index, chunks):
    """
    Coverage-aware retrieval.

    Ensures multi-part questions can retrieve evidence
    from multiple relevant sections instead of allowing
    one high-scoring FAQ chunk to dominate.
    """

    if not question.strip():
        return []

    # --------------------------------------------------------
    # Normal hybrid + reranker retrieval
    # --------------------------------------------------------

    hybrid_results = hybrid_search(
        question,
        index,
        chunks
    )

    if not hybrid_results:
        return []

    reranked = rerank_results(
        question,
        hybrid_results[:20],
        FINAL_TOP_K
    )

    # --------------------------------------------------------
    # Mandatory section retrieval
    #
    # This is the important part.
    # --------------------------------------------------------

    q = normalize_text(question)

    required_sections = []

    # Student questions need User Roles.
    if "student" in q:
        required_sections.append(
            "user roles"
        )

    # Borrowing duration questions need Borrowing Policy.
    if any(
        word in q
        for word in [
            "borrow",
            "borrowing",
            "keep",
            "long",
            "days",
            "loan"
        ]
    ):
        required_sections.append(
            "borrowing policy"
        )

    # Faculty questions.
    if "faculty" in q:
        required_sections.append(
            "user roles"
        )

    # Database / technology questions.
    if any(
        word in q
        for word in [
            "database",
            "frontend",
            "backend",
            "technology",
            "architecture"
        ]
    ):
        required_sections.append(
            "technical architecture"
        )

    # Reservation questions.
    if (
        "reservation" in q
        or "reserved" in q
    ):
        required_sections.append(
            "reservation rule"
        )

    # Security questions.
    if "security" in q:
        required_sections.append(
            "security rules"
        )

    # Remove duplicates.
    required_sections = list(
        dict.fromkeys(required_sections)
    )

    # --------------------------------------------------------
    # Find best chunk from every required section.
    # --------------------------------------------------------

    guaranteed = []

    for required_section in required_sections:

        matching = []

        for chunk in chunks:

            title = normalize_text(
                chunk.get(
                    "section_title",
                    ""
                )
            )

            if required_section in title:

                # Calculate lexical relevance.
                text = normalize_text(
                    chunk.get("text", "")
                )

                q_tokens = tokenize(question)
                text_tokens = tokenize(text)

                overlap = len(
                    q_tokens & text_tokens
                )

                matching.append(
                    (
                        overlap,
                        chunk
                    )
                )

        if matching:

            matching.sort(
                key=lambda x: x[0],
                reverse=True
            )

            best_chunk = matching[0][1]

            guaranteed.append({
                "chunk": best_chunk,
                "score": 100.0,
                "semantic_score": 1.0,
                "keyword_score": 10.0,
                "reranker_score": 10.0,
                "priority_score": 10.0,
                "exact_score": 10.0
            })

    # --------------------------------------------------------
    # Merge guaranteed sources + normal retrieval.
    # --------------------------------------------------------

    combined = []

    seen = set()

    # Guaranteed direct sections first.
    for result in guaranteed:

        chunk = result["chunk"]

        key = (
            chunk.get("page"),
            chunk.get("section"),
            chunk.get("text", "")
        )

        if key not in seen:

            combined.append(result)
            seen.add(key)

    # Then add normal reranked results.
    for result in reranked:

        chunk = result["chunk"]

        key = (
            chunk.get("page"),
            chunk.get("section"),
            chunk.get("text", "")
        )

        if key not in seen:

            combined.append(result)
            seen.add(key)

    return combined[:FINAL_TOP_K]# ============================================================
# CONTEXT BUILDING
# ============================================================

def build_context(results):
    """
    Build clean context for the LLM.
    """

    context_parts = []

    for i, result in enumerate(results, start=1):

        chunk = result["chunk"]

        page = chunk.get(
            "page",
            "?"
        )

        section = chunk.get(
            "section",
            "?"
        )

        section_title = chunk.get(
            "section_title",
            ""
        )

        text = chunk.get(
            "text",
            ""
        )

        header = (
            f"[Source {i} | "
            f"Page {page} | "
            f"Section {section}"
        )

        if section_title:
            header += f": {section_title}"

        header += "]"

        context_parts.append(
            f"{header}\n{text}"
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_answer(
    question,
    results
):
    """
    Generate answer using local Ollama model.
    """

    if not results:
        return (
            "I could not find the answer in the PDF."
        )

    context = build_context(
        results
    )

    prompt = f"""
You are DocMind, an AI assistant that answers questions
ONLY from the supplied PDF context.
RULES:

1. Answer ONLY from the supplied PDF context.

2. Do NOT use outside knowledge.

3. If the answer is not supported by the PDF, respond exactly:
   "I could not find the answer in the PDF."

4. Answer every part of the user's question.

5. For multi-part questions, combine information from all relevant sources.

6. Preserve exact names, IDs, technologies, numbers, dates, and time periods.

7. For questions about a process, event, rule, or condition, include ALL
   relevant actions, conditions, time limits, and consequences explicitly
   stated in the retrieved context.

8. Do not stop after finding the first relevant fact.

9. Do not invent or assume information.

10. Keep the answer clear and concise.

PDF CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    try:

        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """You are DocMind, a precise PDF question-answering assistant.

Answer ONLY from the supplied PDF context.

IMPORTANT:
- Read ALL retrieved sources before answering.
- Answer every part of the user's question completely.
- For processes, rules, events, or conditions, include all relevant conditions,
  actions, time limits, and consequences explicitly stated in the context.
- Do not stop after mentioning only the first relevant fact.
- Preserve exact names, IDs, technologies, numbers, dates, and time periods.
- Never invent or assume information.
- If the answer is not supported by the PDF, respond exactly:
  "I could not find the answer in the PDF."
- Keep the answer clear and concise."""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0,
                "num_predict": 400
            }
        )

        answer = response[
            "message"
        ][
            "content"
        ].strip()

        if not answer:
            return (
                "I could not find the answer in the PDF."
            )

        return answer

    except Exception as e:

        return (
            f"Error generating answer: {str(e)}"
        )


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def ask_pdf(
    question,
    index,
    chunks
):
    """
    Main function used by app.py.

    Returns:
        answer
        sources
    """

    results = adaptive_retrieve(
        question,
        index,
        chunks
    )

    answer = generate_answer(
        question,
        results
    )

    sources = []

    for result in results:

        chunk = result["chunk"]

        sources.append({
            "page": chunk.get(
                "page",
                "?"
            ),

            "section": chunk.get(
                "section",
                "?"
            ),

            "section_title": chunk.get(
                "section_title",
                ""
            ),

            "text": chunk.get(
                "text",
                ""
            ),

            # Keep raw reranker score available.
            "score": result.get(
                "reranker_score",
                result.get(
                    "score",
                    0.0
                )
            ),

            "final_score": result.get(
                "score",
                0.0
            ),

            "semantic_score": result.get(
                "semantic_score",
                0.0
            ),

            "keyword_score": result.get(
                "keyword_score",
                0.0
            )
        })

    return answer, sources