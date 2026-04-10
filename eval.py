from ragas.metrics import faithfulness, answer_relevancy, context_recall
from ragas import evaluate
from datasets import Dataset
from rag_chain import retriever, ask
from test_questions import test_pairs

data = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": []
}

print("Running your chatbot on 20 questions... this will take a few minutes")

for i, pair in enumerate(test_pairs):
    question = pair["question"]
    print(f"  Question {i+1}/20: {question[:50]}...")
    answer, pages = ask(question)
    docs = retriever.invoke(question)
    data["question"].append(question)
    data["answer"].append(answer)
    data["contexts"].append([doc.page_content for doc in docs])
    data["ground_truth"].append(pair["ground_truth"])

print("All 20 questions done. Now running RAGAS scoring...")

dataset = Dataset.from_dict(data)

results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall]
)

print("RAGAS Results:")
print(results)