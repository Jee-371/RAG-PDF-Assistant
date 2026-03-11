import asyncio
from pathlib import Path
import time
import os

import streamlit as st
import inngest
import requests
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📄",
    layout="centered"
)


# -----------------------------
# Inngest Client
# -----------------------------
@st.cache_resource
def get_inngest_client():
    return inngest.Inngest(
        app_id="rag_app",
        is_production=False
    )


# -----------------------------
# Save Uploaded PDF
# -----------------------------
def save_uploaded_pdf(file):

    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)

    file_path = uploads_dir / file.name
    file_path.write_bytes(file.getbuffer())

    return file_path


# -----------------------------
# Send Ingest Event
# -----------------------------
async def send_ingest_event(pdf_path):

    client = get_inngest_client()

    result = await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name
            }
        )
    )

    return result[0]


# -----------------------------
# Send Query Event
# -----------------------------
async def send_query_event(question, top_k):

    client = get_inngest_client()

    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k
            }
        )
    )

    return result[0]


# -----------------------------
# Inngest Dev API
# -----------------------------
def inngest_api():

    return os.getenv(
        "INNGEST_API_BASE",
        "http://127.0.0.1:8288/v1"
    )


def fetch_runs(event_id):

    url = f"{inngest_api()}/events/{event_id}/runs"

    resp = requests.get(url)
    resp.raise_for_status()

    return resp.json().get("data", [])


def wait_for_output(event_id, timeout=600):

    start = time.time()

    while True:

        runs = fetch_runs(event_id)

        if runs:

            run = runs[0]
            status = run.get("status")

            if status in ["Completed", "Succeeded", "Finished"]:
                return run.get("output") or {}

            if status in ["Failed", "Cancelled"]:
                raise RuntimeError("Workflow failed")

        if time.time() - start > timeout:
            raise TimeoutError("Timeout waiting for result")

        time.sleep(1)


# -----------------------------
# UI
# -----------------------------
st.title("📄 RAG PDF Assistant")


# -------- Upload Section --------
st.header("Upload a PDF")

uploaded = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

if uploaded:

    with st.spinner("Uploading and indexing PDF..."):

        path = save_uploaded_pdf(uploaded)

        event_id = asyncio.run(
            send_ingest_event(path)
        )

        time.sleep(0.3)

    st.success(f"PDF indexed: {path.name}")


# -------- Query Section --------
st.header("Ask a Question")

with st.form("query_form"):

    question = st.text_input("Enter your question")

    top_k = st.slider(
        "Number of chunks to retrieve",
        1,
        10,
        5
    )

    submit = st.form_submit_button("Ask")


if submit and question:

    with st.spinner("Searching documents and generating answer..."):

        event_id = asyncio.run(
            send_query_event(question, top_k)
        )

        output = wait_for_output(event_id)

    answer = output.get("answer", "")
    sources = output.get("sources", [])


    st.subheader("Answer")
    st.write(answer if answer else "No answer generated.")

    if sources:
        st.subheader("Sources")

        for s in sources:
            st.write(f"- {s}")