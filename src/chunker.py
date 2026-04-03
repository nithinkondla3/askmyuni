from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load your text file
with open("rmit_context.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.create_documents([text])

# Print 3 sample chunks
for i, chunk in enumerate(chunks[:13]):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk.page_content)

print(f"\nTotal chunks: {len(chunks)}")
