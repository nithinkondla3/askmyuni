import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset
import pandas as pd

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

db = FAISS.load_local(
    "src/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": 6})

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

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
    context_chunks = [doc.page_content for doc in docs]
    return answer, context_chunks

test_pairs = [
    {
        "question": "What is the duration of the Master of AI program at RMIT?",
        "ground_truth": "The Master of AI at RMIT is a 2-year full-time program requiring 192 credit points.",
        "ground_truth_contexts": [
            "The program is primarily delivered on campus. Some courses may be delivered online.\nDuration\nStudy Load\tYears\tMonths\tWeeks\nFull-time\n2"
        ]
    },
    {
        "question": "What are the entry requirements for the Master of AI at RMIT?",
        "ground_truth": "Applicants need an Australian bachelor's degree with GPA 2.0 out of 4.0 in computing, science, engineering, health or statistics.",
        "ground_truth_contexts": [
            "Entry Requirements\nEntry requirements and admissions\nProgram Entry Requirements:\nAn Australian bachelor's degree or equivalent with a grade point average (GPA) of at least 2.0 out of 4.0, in one of the following disciplines: computing, science, engineering, health or statistics."
        ]
    },
    {
        "question": "What programming languages are taught in the Master of AI?",
        "ground_truth": "Python is the primary language, along with Java, SQL, and R. Frameworks include TensorFlow, PyTorch, scikit-learn, and LangChain.",
        "ground_truth_contexts": [
            "PROGRAMMING LANGUAGES\nThe Master of AI at RMIT primarily teaches:\n- Python (used across most AI and machine learning courses)\n- Java (used in algorithm and software engineering courses)\n- SQL (used in database and data mining courses)\n- R (used in some statistics and data analysis courses)\nStudents also use tools and frameworks including TensorFlow, PyTorch, scikit-learn, and LangChain."
        ]
    },
    {
        "question": "How many credit points are required to complete the Master of AI?",
        "ground_truth": "Students must complete 192 credit points to finish the Master of AI.",
        "ground_truth_contexts": [
            "Credit Points\n192"
        ]
    },
    {
        "question": "Can I study the Master of AI part-time?",
        "ground_truth": "Yes, the Master of AI can be studied part-time over 4 years. Full-time is 2 years.",
        "ground_truth_contexts": [
            "PART-TIME STUDY\nYes, the Master of AI can be studied part-time. Part-time duration is typically 4 years.\nFull-time duration is 2 years (192 credit points)."
        ]
    },
    {
        "question": "What is the tuition fee for the Master of AI at RMIT?",
        "ground_truth": "Domestic students pay approximately AUD $17,280 per year. International students pay approximately AUD $40,320 per year.",
        "ground_truth_contexts": [
            "TUITION FEES\nDomestic students: Approximately AUD $17,280 per year (full-time).\nInternational students: Approximately AUD $40,320 per year (full-time).\nTotal program cost for international students: approximately AUD $80,640 for 2 years full-time."
        ]
    },
    {
        "question": "Does RMIT offer scholarships for the Master of AI?",
        "ground_truth": "Yes, RMIT offers scholarships including the RMIT University Postgraduate Research Scholarship and the Australian Government Research Training Program Scholarship.",
        "ground_truth_contexts": [
            "SCHOLARSHIPS\nRMIT offers a range of scholarships for postgraduate students including:\n- RMIT University Postgraduate Research Scholarship\n- RMIT International Excellence Scholarship\n- Australian Government Research Training Program (RTP) Scholarship\n- Various industry-sponsored scholarships"
        ]
    },
    {
        "question": "What core courses are in the Master of AI?",
        "ground_truth": "Core courses include Artificial Intelligence, Algorithms and Analysis, Computational Machine Learning, Discrete Mathematics, Deep Learning, and Database Systems.",
        "ground_truth_contexts": [
            "CORE COURSES\nThe Master of AI (MC271) includes the following core courses:\n- COSC3117 Artificial Intelligence\n- COSC3119 Algorithms and Analysis\n- COSC2793 Computational Machine Learning\n- MATH2415 Discrete Mathematics\n- COSC2779 Deep Learning\n- COSC3125 Data Mining\n- COSC3138 Database Systems"
        ]
    },
    {
        "question": "Is there a capstone project in the Master of AI?",
        "ground_truth": "Yes, students complete COSC2777 Artificial Intelligence Postgraduate Project as a capstone, or COSC2179 Minor Thesis for the research stream.",
        "ground_truth_contexts": [
            "You will undertake a capstone course, either from the project or research streams of the program, respectively: COSC2777 Artificial Intelligence Postgraduate Project, or COSC2179 Minor Thesis / Project. The capstone project course provides you with hands on practical experience of an AI development project."
        ]
    },
    {
        "question": "What is the application deadline for the Master of AI?",
        "ground_truth": "Semester 1 intake starts February and applications close November. Semester 2 starts July and applications close April.",
        "ground_truth_contexts": [
            "APPLICATION DEADLINES AND INTAKE DATES\nThe Master of AI at RMIT has two main intakes per year:\n- Semester 1: Starts February/March. Applications typically close November/December prior.\n- Semester 2: Starts July/August. Applications typically close April/May."
        ]
    },
    {
        "question": "Where is the Master of AI taught?",
        "ground_truth": "The Master of AI is taught at RMIT's Melbourne City Campus.",
        "ground_truth_contexts": [
            "Campus\nCity Campus\nLocation\nOnshore, City Campus"
        ]
    },
    {
        "question": "Can international students apply for the Master of AI?",
        "ground_truth": "Yes, international students can apply. The program has CRICOS code 0100716 and is delivered onshore at City Campus Melbourne.",
        "ground_truth_contexts": [
            "INTERNATIONAL STUDENTS\nYes, international students can apply. The program has a CRICOS code of 0100716.\nCampus: City Campus, Melbourne (onshore delivery).\nInternational students must meet English language requirements (IELTS 6.5, no band below 6.0)."
        ]
    },
    {
        "question": "What elective courses are available in the Master of AI?",
        "ground_truth": "Electives include Agent-Oriented Programming, Evolutionary Computing, Mixed Reality, Games and AI Techniques, and Computing Research and Project Preparation.",
        "ground_truth_contexts": [
            "ELECTIVE / PROGRAM OPTION COURSES\nStudents choose electives from the Program Options list including:\n- COSC3123 Agent-Oriented Programming and Design\n- COSC3124 Evolutionary Computing\n- COSC3140 Mixed Reality\n- COSC3144 Games and Artificial Intelligence Techniques\n- COSC3130 Computing Research and Project Preparation\n- Natural Language Processing\n- Computer Vision\n- Robotics"
        ]
    },
    {
        "question": "Does the Master of AI include industry placements?",
        "ground_truth": "Yes, through Work Integrated Learning in COSC2777, students work on industry-sponsored capstone projects.",
        "ground_truth_contexts": [
            "COSC2777 Artificial Intelligence Postgraduate Project is a capstone course designed to provide you with hands-on practical experience. This course includes a Work Integrated Learning experience in which your knowledge and skills will be applied and assessed in a real or simulated workplace context and where feedback from industry and/or community is integral to your experience."
        ]
    },
    {
        "question": "What is the English language requirement for the Master of AI?",
        "ground_truth": "International students need IELTS 6.5 with no band below 6.0, or TOEFL Paper 580+ or TOEFL Computer 237+.",
        "ground_truth_contexts": [
            "English Language Requirements:\nEnglish Language requirement: English IELTS language test score of 6.5 with no band less than 6.0 or equivalent, such as TOEFL (Paper based) = 580+ (TWE 4.5+), or TOEFL (Computer based) = 237+ (TWE 4.5+) or REW English for Academic Purposes Advanced 1&2"
        ]
    },
    {
        "question": "How is the Master of AI assessed?",
        "ground_truth": "Assessment includes timed assessments, assignments, projects, reflective journals, assessed tutorials, presentations, and self and peer assessment.",
        "ground_truth_contexts": [
            "Self-assessment and peer-assessment: for assessment activities such as seminars you may be asked to assess your own work, the work of your group, or the work of other groups. This is part of equipping you to become more independent in your own learning and to develop your assessment skills."
        ]
    },
    {
        "question": "What job titles can graduates of the Master of AI pursue?",
        "ground_truth": "Graduates can work as AI Engineer, Machine Learning Engineer, Data Scientist, Business Intelligence Developer, Research Scientist, and Web Analyst.",
        "ground_truth_contexts": [
            "CAREER OUTCOMES\nGraduates of the Master of AI at RMIT can pursue careers including:\n- AI Engineer\n- Machine Learning Engineer\n- Data Scientist\n- Business Intelligence Developer\n- Research Scientist\n- Web Analyst\n- AI Consultant\n- NLP Engineer"
        ]
    },
    {
        "question": "Is the Master of AI accredited?",
        "ground_truth": "The Master of AI is conditionally provisionally accredited at a professional level by the Australian Computer Society (ACS).",
        "ground_truth_contexts": [
            "External Accreditation\nThe Master of Artificial Intelligence is conditionally provisionally accredited at a professional level by the Australian Computer Society, which accredits Information and Communication Technology related programs that are offered by Australian universities"
        ]
    },
    {
        "question": "What is the CRICOS code for the Master of AI at RMIT?",
        "ground_truth": "The CRICOS code for the Master of AI at RMIT is 0100716.",
        "ground_truth_contexts": [
            "INTERNATIONAL STUDENTS\nYes, international students can apply. The program has a CRICOS code of 0100716.\nCampus: City Campus, Melbourne (onshore delivery)."
        ]
    },
    {
        "question": "When does the Master of AI intake start?",
        "ground_truth": "The program has intakes in Semester 1 starting February and Semester 2 starting July each year.",
        "ground_truth_contexts": [
            "APPLICATION DEADLINES AND INTAKE DATES\nThe Master of AI at RMIT has two main intakes per year:\n- Semester 1: Starts February/March. Applications typically close November/December prior.\n- Semester 2: Starts July/August. Applications typically close April/May."
        ]
    },
]

questions = []
answers = []
contexts = []
ground_truths = []
ground_truth_contexts = []

print("Running RAG chain on 20 questions...\n")

for i, pair in enumerate(test_pairs):
    print(f"  [{i+1}/20] {pair['question'][:60]}...")
    answer, context_chunks = ask(pair["question"])
    questions.append(pair["question"])
    answers.append(answer)
    contexts.append(context_chunks)
    ground_truths.append(pair["ground_truth"])
    ground_truth_contexts.append(pair["ground_truth_contexts"])

print("\nAll 20 questions answered. Running RAGAS evaluation...\n")

dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths,
    "ground_truth_contexts": ground_truth_contexts
})

result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall])

print("\nRAGAS Results:")
print("-" * 33)

df = result.to_pandas()

print(f"  Faithfulness:     {df['faithfulness'].mean():.2f}")
print(f"  Answer Relevancy: {df['answer_relevancy'].mean():.2f}")
print(f"  Context Recall:   {df['context_recall'].mean():.2f}")
print("-" * 33)

print("\nPer-Question Breakdown:")
print("-" * 33)
for i, row in df.iterrows():
    print(f"\nQ{i+1}: {row['user_input'][:60]}...")
    print(f"  Faithfulness:     {row['faithfulness']:.2f}")
    print(f"  Answer Relevancy: {row['answer_relevancy']:.2f}")
    print(f"  Context Recall:   {row['context_recall']:.2f}")

df.to_csv("ragas_results.csv", index=False)
print("\nSaved per-question breakdown to ragas_results.csv")