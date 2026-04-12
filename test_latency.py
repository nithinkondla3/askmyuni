from rag_chain import ask

questions = [
    "What is the duration of the Master of AI program at RMIT?",
    "What are the entry requirements for the Master of AI at RMIT?",
    "What core courses are in the Master of AI?"
]

for q in questions:
    print(f"\nQ: {q}")
    answer, pages = ask(q)
    print(f"A: {answer[:80]}...")
    print(f"Pages: {pages}")