import hashlib
from pathlib import Path

import streamlit as st

from pdf_processor import process_pdf
from rag import create_vector_index, ask_pdf


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="DocMind",
    page_icon="📄",
    layout="centered",
)


# =========================================================
# CACHED FUNCTIONS
# =========================================================

@st.cache_data
def load_pdf_chunks(pdf_path, file_bytes):
    return process_pdf(pdf_path)


@st.cache_resource
def build_faiss_index(chunks):
    return create_vector_index(chunks)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_id" not in st.session_state:
    st.session_state.document_id = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "index" not in st.session_state:
    st.session_state.index = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None


# =========================================================
# MINIMAL UI
# =========================================================

st.title("📄 DocMind")
st.caption("AI-powered PDF Question Answering")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"],
)

if uploaded_file is None:
    st.info("Upload a PDF to start chatting.")
    st.stop()


# =========================================================
# DOCUMENT CHANGE
# =========================================================

file_bytes = uploaded_file.getvalue()

document_id = hashlib.md5(file_bytes).hexdigest()

if st.session_state.document_id != document_id:
    st.session_state.document_id = document_id
    st.session_state.document_name = uploaded_file.name
    st.session_state.messages = []
    st.session_state.chunks = None
    st.session_state.index = None


# =========================================================
# SAVE PDF
# =========================================================

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

pdf_path = data_dir / "uploaded.pdf"

pdf_path.write_bytes(file_bytes)


# =========================================================
# PROCESS PDF
# =========================================================

try:
    with st.spinner("Processing PDF..."):
        chunks = load_pdf_chunks(
            str(pdf_path),
            file_bytes,
        )

except Exception as error:
    st.error(f"Could not process the PDF: {error}")
    st.stop()


if not chunks:
    st.error("No readable text was found in this PDF.")
    st.stop()


# =========================================================
# BUILD INDEX
# =========================================================

try:
    with st.spinner("Building search index..."):
        index, embeddings, filtered_chunks = build_faiss_index(chunks)

except Exception as error:
    st.error(f"Could not create the search index: {error}")
    st.stop()


st.session_state.chunks = filtered_chunks
st.session_state.index = index


# =========================================================
# DOCUMENT STATUS
# =========================================================

st.caption(
    f"**{uploaded_file.name}** · "
    f"{len(filtered_chunks)} chunks · "
    f"{index.ntotal} vectors"
)

if st.button("Reset chat"):
    st.session_state.messages = []
    st.rerun()


st.divider()


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])

        if message["role"] == "assistant":

            sources = message.get("sources", [])

            if sources:

                with st.expander("Sources"):

                    for source in sources:

                        page = source.get("page", "?")
                        section = source.get("section", "?")
                        section_title = source.get(
                            "section_title",
                            "",
                        )
                        score = source.get("score", 0)
                        text = source.get(
                            "text",
                            "Evidence unavailable.",
                        )

                        label = (
                            f"Page {page} · "
                            f"Section {section}"
                        )

                        if section_title:
                            label += f" — {section_title}"

                        label += f" · {score:.2f}"

                        st.markdown(f"**{label}**")
                        st.caption(text)


# =========================================================
# QUESTION INPUT
# =========================================================

question = st.chat_input(
    "Ask something about your PDF..."
)


if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.write(question)

    try:

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                answer, sources = ask_pdf(
                    question,
                    index,
                    filtered_chunks,
                )

            st.write(answer)

            if sources:

                with st.expander("Sources"):

                    for source in sources:

                        page = source.get("page", "?")
                        section = source.get("section", "?")
                        section_title = source.get(
                            "section_title",
                            "",
                        )
                        score = source.get("score", 0)
                        text = source.get(
                            "text",
                            "Evidence unavailable.",
                        )

                        label = (
                            f"Page {page} · "
                            f"Section {section}"
                        )

                        if section_title:
                            label += f" — {section_title}"

                        label += f" · {score:.2f}"

                        st.markdown(f"**{label}**")
                        st.caption(text)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )

    except Exception as error:

        error_message = (
            f"Error generating answer: {error}"
        )

        with st.chat_message("assistant"):
            st.error(error_message)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": error_message,
                "sources": [],
            }
        )
