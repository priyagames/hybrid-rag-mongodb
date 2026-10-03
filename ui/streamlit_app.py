import os
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Hybrid RAG on MongoDB", layout="wide")
st.title("Hybrid RAG on MongoDB: tables + text")

pdf = st.file_uploader("Upload a PDF", type="pdf")
if pdf and st.button("Ingest"):
    with st.spinner("Parsing, embedding and storing in MongoDB..."):
        r = requests.post(f"{API}/ingest", timeout=900,
                          files={"file": (pdf.name, pdf.getvalue(), "application/pdf")})
    if r.status_code != 200:
        st.error(f"Ingest failed ({r.status_code}): {r.text[:500]}")
    else:
        st.session_state["doc"] = r.json()

existing = st.text_input("Or use an already-ingested document ID")
if existing and st.button("Use this document"):
    st.session_state["doc"] = {"doc_id": existing.strip(), "filename": "existing document"}

doc = st.session_state.get("doc")
if doc:
    if "tables" in doc:
        st.success(f"{doc['filename']}: {doc['tables']} tables, {doc['table_rows']} rows, "
                   f"{doc['text_chunks']} text chunks")
    else:
        st.success(f"Using document {doc['doc_id']}")
    question = st.text_input("Ask a question about the document")
    if question:
        body = {"doc_id": doc["doc_id"], "question": question}
        left, right = st.columns(2)
        with left:
            st.subheader("Conventional RAG")
            rb = requests.post(f"{API}/ask/baseline", json=body, timeout=300)
            if rb.status_code != 200:
                st.error(f"Error {rb.status_code}: {rb.text[:300]}")
            else:
                b = rb.json()
                st.write(b["answer"])
                with st.expander("Retrieved chunks"):
                    st.json(b["chunks"])
        with right:
            st.subheader("Hybrid RAG")
            rh = requests.post(f"{API}/ask/hybrid", json=body, timeout=300)
            if rh.status_code != 200:
                st.error(f"Error {rh.status_code}: {rh.text[:300]}")
            else:
                h = rh.json()
                st.write(h["answer"])
                st.caption(f"Route: {h['route']} - {h['route_reason']}")
                if h.get("table_result"):
                    with st.expander("Aggregation pipeline and rows"):
                        st.json(h["table_result"])
                if h.get("text_result"):
                    with st.expander("Text chunks"):
                        st.json(h["text_result"])
