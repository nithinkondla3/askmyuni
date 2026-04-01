# embedder.py
# This file creates the FAISS vector store and saves/loads it from disk

import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

def create_and_save_vectorstore(chunks, save_path="faiss_index"):
    """Embeds chunks using OpenAI and saves the FAISS index to disk."""
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)
    print(f"Vector store saved to '{save_path}'")
    return vectorstore

def load_vectorstore(save_path="faiss_index"):
    """Loads an existing FAISS index from disk. Use this to avoid re-embedding."""
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"Vector store loaded from '{save_path}'")
    return vectorstore