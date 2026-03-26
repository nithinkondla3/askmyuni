# AskMyUni 🎓

A RAG-powered chatbot that answers RMIT student questions directly 
from the official course handbook — built with LangChain, OpenAI, 
FAISS, and Streamlit.

---

## 🔍 Problem

RMIT students waste hours searching through long PDF handbooks to find 
answers about credit points, course requirements, and enrolment rules. 
AskMyUni gives instant, cited answers in plain English.

---

## 💡 Solution

AskMyUni uses Retrieval-Augmented Generation (RAG) to search the 
handbook semantically and answer questions with source citations — 
so students can verify every answer.

---

## 🏗️ Architecture
```
PDF Handbook → Chunking → Embeddings → FAISS Vector Store
                                              ↓
User Question → Semantic Search → Relevant Chunks → OpenAI LLM → Answer + Sources
```

---

## 🛠️ Tech Stack

- **LangChain** — RAG chain and document loading
- **OpenAI** — Embeddings and answer generation  
- **FAISS** — Local vector store for semantic search
- **Streamlit** — Chat user interface
- **PyPDF** — PDF parsing

---

## 💬 Example

**Q:** How many credit points do I need to graduate?  
**A:** You need 144 credit points to complete the Master of AI program.  
📄 *Source: RMIT Handbook, Page 3*

---

## 🚀 Run Locally
```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/askmyuni.git
cd askmyuni

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
cp .env.example .env
# Open .env and add your OPENAI_API_KEY

# 5. Run the app
streamlit run app.py
```

---

## 📁 Project Structure
```
askmyuni/
├── loader.py          # PDF loading and chunking
├── embedder.py        # Embedding creation and FAISS storage
├── rag_chain.py       # RetrievalQA chain
├── app.py             # Streamlit UI
├── requirements.txt
└── .env.example
```

---

*Built as part of RMIT Master of AI — 2025*