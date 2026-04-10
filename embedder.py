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




if __name__ == "__main__":
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    with open("rmit_context.txt", "r", encoding="utf-8") as f:
        raw = f.read()

    start = raw.find("Purpose of the Program")
    text = raw[start:]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=100
    )
    chunks = splitter.create_documents([text])
    print(f"Total chunks: {len(chunks)}")

    vectorstore = create_and_save_vectorstore(chunks, save_path="faiss_index")
    vectorstore.save_local("src/faiss_index")
    print("Done! Both indexes rebuilt.")