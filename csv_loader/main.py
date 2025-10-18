from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = CSVLoader(file_path="csv_loader/Pokemon.csv")

data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

splits = text_splitter.split_documents(data)

for split in splits:
    row = split.metadata["row"]
    content = split.page_content
    print(f"Row {row}: {content}")
    print("-" * 50)
