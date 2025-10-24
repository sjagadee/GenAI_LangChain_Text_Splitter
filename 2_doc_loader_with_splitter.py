from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("indian_economy_report_2025.pdf")

docs = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=""
)

result = text_splitter.split_documents(docs)
print(len(result))

print(result[0].page_content)
