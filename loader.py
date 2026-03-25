from langchain_community.document_loaders import TextLoader

loader = TextLoader("rmit_context.txt")
pages = loader.load()

print(pages[0].page_content[:500])
print(f"Total pages loaded: {len(pages)}")