import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings()

# Load your saved index (no need to re-embed!)
db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
print("Index loaded.\n")

# --- TASK 1: Try k=1, k=3, k=5 ---
question = "How many credit points to graduate?"

for k in [1, 3, 5]:
    print(f"--- k={k} ---")
    results = db.similarity_search(question, k=k)
    for i, r in enumerate(results):
        print(f"  Result {i+1}: {r.page_content}")
    print()

# --- TASK 2: Try different queries ---
queries = [
    "Where is the campus?",
    "What is the minimum GPA?",
    "How do I enrol?",
    "Tell me about exams",  # broader/vaguer query
]

print("=== Different query types ===\n")
for q in queries:
    print(f"Query: '{q}'")
    results = db.similarity_search(q, k=2)
    for i, r in enumerate(results):
        print(f"  Result {i+1}: {r.page_content}")
    print()

# --- TASK 3: Note what you observe ---
# After running, ask yourself:
# - Does k=5 always return useful results, or does it return junk?
# - Are any queries returning the WRONG chunk as result 1?
# - Write down 2 things you'd improve