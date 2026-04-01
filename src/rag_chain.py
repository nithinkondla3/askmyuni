from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

embeddings = OpenAIEmbeddings()

db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Updated prompt with guardrail
prompt = PromptTemplate.from_template("""
You are an RMIT student assistant.
Use ONLY the context below to answer the question.
If the question is not related to RMIT or the provided documents,
say: "I can only answer questions about the RMIT program handbook."
If you don't know the answer from the context, say: "I don't have that information."

Context: {context}

Question: {question}

Answer:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ask(question):
    docs = retriever.invoke(question)
    context = format_docs(docs)
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "question": question
    })
    pages = sorted(set(
        doc.metadata.get("page", 0) + 1 for doc in docs
    ))
    return answer, pages

questions = [
    "How meny creddit points do I need too graduate?",
    "Can I defer my enrollment?",
    "What happens if I fail a course?",
    "What is the weather like in Melbourne today?"
]

#Error handling 
for question in questions:
    try:
        answer, pages = ask(question)
        print(f"Q: {question}")
        print(f"A: {answer}")
        print(f"Sources: PDF pages {pages}")
    except Exception as e:
        print(f"Q: {question}")
        print(f"A: Sorry, something went wrong. (Error: {e})")
    print("---")