import hashlib
import tempfile
from pathlib import Path

import streamlit as st

from pdf_processor import process_pdf
from rag import create_vector_index, ask_pdf


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="DocMind",
    page_icon="📄",
    layout="centered",
)


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "document_id" not in st.session_state:
    st.session_state.document_id = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "index" not in st.session_state:
    st.session_state.index = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------
# PDF PROCESSING
# ---------------------------------------------------------

def process_uploaded_pdf(file_bytes):
    """
    Save uploaded PDF temporarily, process it,
    and remove the temporary file afterward.
    """

    temp_path = None

    try:
        file_hash = hashlib.md5(file_bytes).hexdigest()

        temp_dir = Path(tempfile.gettempdir())

        temp_path = temp_dir / f"docmind_{file_hash}.pdf"

        temp_path.write_bytes(file_bytes)

        chunks = process_pdf(str(temp_path))

        return chunks

    finally:

        if temp_path and temp_path.exists():

            try:
                temp_path.unlink()

            except OSError:
                pass


# ---------------------------------------------------------
# SOURCE DISPLAY
# ---------------------------------------------------------

def display_sources(sources):
    """Display useful source information for retrieved chunks."""

    if not sources:
        return

    with st.expander("📚 Sources"):

        for i, source in enumerate(sources, start=1):

            page = source.get(
                "page",
                "Unknown"
            )

            section = source.get(
                "section_title",
                ""
            )

            text = source.get(
                "text",
                ""
            ).strip()

            st.markdown(
                f"**Source {i} — Page {page}**"
            )

            if section:
                st.caption(section)

            if text:
                preview = text[:300]

                if len(text) > 300:
                    preview += "..."

                st.write(preview)

            if i < len(sources):
                st.divider()


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("📄 DocMind")

st.caption(
    "Ask questions about any PDF using Retrieval-Augmented Generation."
)


# ---------------------------------------------------------
# PDF UPLOAD
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"],
)


if uploaded_file is None:

    st.info(
        "Upload a PDF to start chatting."
    )

    st.stop()


# ---------------------------------------------------------
# READ UPLOADED FILE
# ---------------------------------------------------------

file_bytes = uploaded_file.getvalue()

document_id = hashlib.md5(
    file_bytes
).hexdigest()


# ---------------------------------------------------------
# PROCESS NEW DOCUMENT
# ---------------------------------------------------------

if st.session_state.document_id != document_id:

    try:

        with st.spinner("Processing PDF..."):

            chunks = process_uploaded_pdf(
                file_bytes
            )

        if not chunks:

            st.error(
                "No readable text was found in this PDF."
            )

            st.info(
                "This PDF may be scanned/image-only, "
                "or its text may not be extractable."
            )

            st.stop()

        with st.spinner(
            "Building document search index..."
        ):

            index, _, filtered_chunks = (
                create_vector_index(chunks)
            )

        # Commit the new document only
        # after successful processing.

        st.session_state.document_id = document_id

        st.session_state.document_name = (
            uploaded_file.name
        )

        st.session_state.chunks = (
            filtered_chunks
        )

        st.session_state.index = index

        st.session_state.messages = []

    except Exception as error:

        st.error(
            f"Failed to process PDF: {error}"
        )

        st.stop()


# ---------------------------------------------------------
# DOCUMENT STATUS
# ---------------------------------------------------------

if (
    st.session_state.chunks is not None
    and st.session_state.index is not None
):

    st.success(
        f"Ready: {st.session_state.document_name}"
    )

    st.caption(
        f"{len(st.session_state.chunks)} document chunks indexed"
    )


# ---------------------------------------------------------
# DISPLAY PREVIOUS CHAT
# ---------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            display_sources(
                message["sources"]
            )


# ---------------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------------

question = st.chat_input(
    "Ask a question about the PDF..."
)


# ---------------------------------------------------------
# QUESTION ANSWERING
# ---------------------------------------------------------

if question:

    # Display user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # Generate answer

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching the document..."
        ):

            try:

                answer, sources = ask_pdf(
                    question,
                    st.session_state.index,
                    st.session_state.chunks,
                )

            except Exception as error:

                answer = (
                    f"Sorry, an error occurred: {error}"
                )

                sources = []

        st.markdown(answer)

        display_sources(sources)

    # Save assistant response

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )
