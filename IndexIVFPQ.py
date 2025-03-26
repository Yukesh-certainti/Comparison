import streamlit as st
import os
import google.generativeai as genai
import faiss
from langchain.docstore.document import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import json
import requests
import time
import logging
import numpy as np
import spacy
import sys
import fitz  # PyMuPDF

# Setup logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
INDEX_FILE = "./data/faiss_index"
TEXT_FILE = "retrieved_text.json"

# Load API key for Gemini
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
embedding_endpoint = os.getenv("EMBEDDING_API_ENDPOINT")
if not api_key or not embedding_endpoint:
    st.error("API key or embedding endpoint not configured. Please check your .env file.")
    sys.exit(1)
genai.configure(api_key=api_key)

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Utility Functions
def count_tokens(text):
    """Count the number of tokens in text using SpaCy"""
    doc = nlp(text)
    return len([token for token in doc if not token.is_stop])

def get_embedding_from_api(text, request_id=None, retries=0):
    try:
        if not isinstance(text, str):
            return False, None

        text = preprocess_text(text, request_id)
        response = requests.post(url=embedding_endpoint, json={"text": text})
        response.raise_for_status()
        result = response.json()
        embedding = result["embedding"][0]
        return True, embedding

    except requests.exceptions.RequestException as e:
        error_log(400, f"API request exception: {str(e)}", "get_embedding", request_id)
        return handle_embedding_error(text, request_id, retries)

    except Exception as e:
        error_log(400, f"Other exception: {str(e)}", "get_embedding", request_id)
        return handle_embedding_error(text, request_id, retries)

def handle_embedding_error(text, request_id, retries):
    if retries < 3:
        time.sleep(2 ** retries)  # Exponential backoff
        return get_embedding_from_api(text, request_id, retries + 1)
    return False, None

def error_log(status_code, message, function_name, request_id=None):
    logger.error(f"Request ID: {request_id} - Status Code: {status_code} - Function: {function_name} - {message}")

def preprocess_text(text, request_id=None):
    return text

def remove_stopwords_spacy(text):
    doc = nlp(text)
    return " ".join([token.text for token in doc if not token.is_stop])

# PDF Processing
def get_pdf_text(pdf_docs):
    """Extract text from PDFs and return word-based text chunks."""
    text = ""
    for pdf in pdf_docs:
        try:
            with fitz.open(stream=pdf.read(), filetype="pdf") as doc:
                for page in doc:
                    text += remove_stopwords_spacy(page.get_text())  # Remove stopwords
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            return []
    return get_text_chunks(text)

def get_text_chunks(text, chunk_size=512, chunk_overlap=20):
    """Split text into chunks based on word count."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - chunk_overlap):
        end_idx = min(i + chunk_size, len(words))
        chunk_words = words[i:end_idx]
        chunks.append(" ".join(chunk_words))
        if end_idx == len(words):
            break
    return chunks

# Vector Store with FAISS
def get_vector_store(text_chunks):
    embeddings_list = []
    new_stored_texts = []
    token_counts = []
    
    dimension = 512  # Embedding size
    num_clusters = min(20, max(5, len(text_chunks) // 10))  # Adaptive cluster count
    pq_bits = 8  # Number of bits per sub-vector for PQ

    if os.path.exists(INDEX_FILE) and os.path.exists(TEXT_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(TEXT_FILE, "r") as f:
            stored_texts = json.load(f)
    else:
        coarse_quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFPQ(coarse_quantizer, dimension, num_clusters, pq_bits, 8)
        stored_texts = []
        if not os.path.exists("./data"):
            os.makedirs("./data")

    start_index_build = time.perf_counter()

    # Generate embeddings and collect data
    embedding_times = []
    for chunk in text_chunks:
        token_count = count_tokens(chunk)
        token_counts.append(token_count)
        
        embedding_start = time.perf_counter()
        success, embedding = get_embedding_from_api(chunk)
        embedding_time = time.perf_counter() - embedding_start
        embedding_times.append(embedding_time)
        
        if success:
            embeddings_list.append(embedding)
            new_stored_texts.append(chunk)
        else:
            embeddings_list.append([0.0] * dimension)
            new_stored_texts.append(chunk)

    if not embeddings_list:
        st.error("Failed to generate embeddings.")
        return None, None, 0, 0, 0, []

    embeddings_array = np.array(embeddings_list).astype("float32")

    # Train the index
    start_train = time.perf_counter()
    if not index.is_trained:
        if len(embeddings_array) < num_clusters:
            embeddings_array = np.tile(embeddings_array, (max(1, num_clusters // len(embeddings_array) + 1), 1))[:num_clusters]
        index.train(embeddings_array)
    train_time = time.perf_counter() - start_train

    # Add embeddings to the index
    indexing_start = time.perf_counter()
    index.add(embeddings_array)
    indexing_time = time.perf_counter() - indexing_start

    # Debug output
    print(f"Training Time: {train_time:.6f} seconds")
    print(f"Indexing Time: {indexing_time:.6f} seconds")

    # Cluster assignments
    assignments = index.assign(embeddings_array, k=1).flatten() % num_clusters

    # Aggregate cluster data
    cluster_dict = {i: {"tokens": 0, "embedding_time": 0.0, "text": ""} for i in range(num_clusters)}
    for chunk, token_count, embedding_time, assignment in zip(new_stored_texts, token_counts, embedding_times, assignments):
        cluster_dict[assignment]["tokens"] += token_count
        cluster_dict[assignment]["embedding_time"] += embedding_time
        if not cluster_dict[assignment]["text"]:
            cluster_dict[assignment]["text"] = chunk[:100] + "..." if len(chunk) > 100 else chunk

    # Prepare cluster_info
    cluster_info = []
    per_cluster_indexing_time = indexing_time / num_clusters
    for cluster_id in range(num_clusters):
        if cluster_dict[cluster_id]["tokens"] > 0:
            cluster_info.append({
                "cluster": cluster_id + 1,
                "tokens": cluster_dict[cluster_id]["tokens"],
                "embedding_time": cluster_dict[cluster_id]["embedding_time"],
                "indexing_time": per_cluster_indexing_time,
                "text": cluster_dict[cluster_id]["text"]
            })

    stored_texts.extend(new_stored_texts)
    faiss.write_index(index, INDEX_FILE)
    with open(TEXT_FILE, "w") as f:
        json.dump(stored_texts, f)

    end_index_build = time.perf_counter()
    total_time = end_index_build - start_index_build
    total_tokens = sum(token_counts)

    return index, stored_texts, total_time, total_tokens, num_clusters, cluster_info

# Question Answering
def user_input(user_question):
    try:
        index = faiss.read_index(INDEX_FILE)
        with open(TEXT_FILE, "r") as f:
            retrieved_text = json.load(f)

        success, query_embedding = get_embedding_from_api(user_question)
        if not success:
            st.error("Failed to generate embedding for the question.")
            return

        query_embedding = np.array(query_embedding).reshape(1, -1).astype("float32")

        start_search = time.perf_counter()
        _, I = index.search(query_embedding, k=5)
        end_search = time.perf_counter()

        docs = [Document(page_content=retrieved_text[i]) for i in I[0] if i < len(retrieved_text)]

    except Exception as e:
        st.error(f"FAISS Index Error: {str(e)}")
        docs = []
        end_search = start_search = time.perf_counter()

    document_context = "\n".join([doc.page_content for doc in docs]) if docs else ""

    if document_context:
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        answer = response["output_text"]
    else:
        st.warning("No relevant context found in PDFs. Answering based on general knowledge.")
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
        answer = model.invoke(user_question).content

    st.write("Reply:", answer)
    st.write(f"Search time: {end_search - start_search:.6f} seconds")

def get_conversational_chain():
    prompt_template = """
    Answer based on the provided context. If no answer is found, use general knowledge.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Main Application
def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Interactive RAG-based LLM for Multi-PDF Document Analysis", divider='rainbow')

    user_question = st.text_input("Ask a Question:")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload PDF files before processing.")
            else:
                with st.spinner("Processing..."):
                    text_chunks = get_pdf_text(pdf_docs)
                    if text_chunks:
                        index, stored_texts, index_build_time, total_tokens, num_clusters, cluster_info = get_vector_store(text_chunks)
                        if index:
                            st.success("Processing Complete! You can now ask questions based on all uploaded PDFs.")
                            st.write(f"Index build time: {index_build_time:.4f} seconds")
                            st.write(f"Total tokens in documents: {total_tokens}")
                            st.write(f"Number of clusters: {num_clusters}")
                            
                            st.subheader("Cluster Details:")
                            for info in cluster_info:
                                st.write(f"Cluster {info['cluster']}:")
                                st.write(f"Tokens: {info['tokens']}")
                                st.write(f"Embedding Computation Time: {info['embedding_time']:.4f} seconds")
                                st.write(f"VectorDB Indexing Time: {info['indexing_time']:.6f} seconds")
                                st.write(f"Cluster Text: {info['text']}")
                                st.write("---")
                            
                            index_size = os.path.getsize(INDEX_FILE) / 1024  # KB
                            st.write(f"Index size: {index_size:.2f} KB")
                        else:
                            st.error("Vector store creation/update failed for this batch.")
                    else:
                        st.error("No text extracted from PDFs in this batch.")
        
        if st.button("Reset Index"):
            if os.path.exists(INDEX_FILE):
                os.remove(INDEX_FILE)
            if os.path.exists(TEXT_FILE):
                os.remove(TEXT_FILE)
            st.success("Index and stored texts reset.")

if __name__ == "__main__":
    main()
