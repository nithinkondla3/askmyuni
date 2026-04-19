# 🎓 AskMyUni — RAG Chatbot for RMIT Students
> Ask questions about RMIT's course handbook in plain English. Get instant, cited answers.

🔗 **[Live Demo](https://huggingface.co/spaces/nithinkondla3/askmyuni)**
🔗 **[Live API](https://askmyuni.onrender.com)**

---

## 🔴 The Problem
RMIT's course handbook is 200+ pages. Students waste time searching
for simple answers about credit points, deadlines, and policies.

## ✅ The Solution
AskMyUni uses RAG to retrieve only the relevant parts of the handbook
and answer in plain English — with source citations so you can verify.

## 🏗️ Architecture
PDF → Chunking (500 tokens) → OpenAI Embeddings → FAISS → FastAPI → Docker → Render

## 📊 Evaluation Results
Evaluated using RAGAS framework:
- **Faithfulness:** measures if the answer is grounded in the retrieved context
- **Answer Relevance:** measures if the answer addresses the question

## 🛠️ Tech Stack
- **LangChain** — pipeline orchestration
- **OpenAI gpt-3.5-turbo** — language model
- **FAISS** — vector database
- **FastAPI** — production API backend
- **Streamlit** — chat UI
- **Docker** — containerisation
- **Render** — cloud deployment
- **GitHub Actions** — CI/CD
- **RAGAS** — evaluation framework

## 🚀 Run Locally
git clone https://github.com/nithinkondla3/askmyuni
cd askmyuni
pip install -r requirements.txt
cp .env.example .env
docker-compose up

## 🔗 API Usage
POST https://askmyuni.onrender.com/query
Content-Type: application/json

{
  "question": "How many credit points do I need to graduate?"
}

## 💬 Example Questions
- "How many credit points do I need to graduate?"
- "What is the late submission policy?"
- "How do I apply for special consideration?"