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
prompt = PromptTemplate.from_template("""
Use the following context to answer the question.
If you don't know the answer, say I don't know.

Context: {context}

Question: {question}

Answer:""")
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
def ask(question):
    # Step 1: retrieve the docs ourselves
    docs = retriever.invoke(question)

    # Step 2: format docs into context text
    context = format_docs(docs)

    # Step 3: run the chain with context + question
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "question": question
    })

    # Step 4: extract page numbers from the docs
    pages = sorted(set(
        doc.metadata.get("page", 0) + 1 for doc in docs
    ))

    return answer, pages

questions = [
    "How many credit points do I need to graduate?",
    "Can I defer my enrollment?",
    "What happens if I fail a course?",
    "What is the weather like in Melbourne?"
]

for question in questions:
    answer, pages = ask(question)
    print(f"Q: {question}")
    print(f"A: {answer}")
    print(f" Sources: PDF pages {pages}")
    print("---")
