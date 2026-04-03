
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
<img width="1880" height="1064" alt="answer" src="https://github.com/user-attachments/assets/217fd0bb-43f7-43da-9955-1268eebdd8e7" />
 <img width="1903" height="1065" alt="homepage" src="https://github.com/user-attachments/assets/3797fc1b-9f6c-439a-a320-b9216c37b552" />
 <img width="1911" height="1068" alt="sidebar" src="https://github.com/user-attachments/assets/56d79ebd-6d37-4bf4-92b7-a2bdcdabeed4" />



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
