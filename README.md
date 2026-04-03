---
title: Askmyuni
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
pinned: false
---

# 🎓 AskMyUni — RAG Chatbot for RMIT Students

> Ask questions about RMIT's course handbook in plain English. Get instant, cited answers.

🔗 **[Live Demo](https://huggingface.co/spaces/nithinkondla3/askmyuni)**

---

## 🔴 The Problem
RMIT's course handbook is 200+ pages. Students waste time searching
for simple answers about credit points, deadlines, and policies.

## ✅ The Solution
AskMyUni uses RAG to retrieve only the relevant parts of the handbook
and answer in plain English — with source citations so you can verify.

## 🏗️ Architecture
PDF → Chunking (500 tokens) → OpenAI Embeddings → FAISS → GPT-4o-mini → Answer + Sources

## 🛠️ Tech Stack
- **LangChain** — pipeline orchestration
- **OpenAI GPT-4o-mini** — language model
- **FAISS** — vector database
- **Streamlit** — chat UI
- **Docker + Hugging Face Spaces** — deployment

## 📸 Screenshots
<!-- Add screenshots here -->

## 🚀 Run Locally
git clone https://github.com/nithinkondla3/askmyuni
cd askmyuni
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py

## 💬 Example Questions
- "How many credit points do I need to graduate?"
- "What is the late submission policy?"
- "How do I apply for special consideration?"