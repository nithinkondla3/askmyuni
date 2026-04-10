from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load your text file
with open("rmit_context.txt", "r", encoding="utf-8") as f:
    raw = f.read()

# Strip nav noise from the top
start = raw.find("Purpose of the Program")
text = raw[start:]

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=100
)
chunks = splitter.create_documents([text])

# Print 3 sample chunks
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk.page_content)

print(f"\nTotal chunks: {len(chunks)}")