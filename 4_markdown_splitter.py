from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

from langchain_community.document_loaders import TextLoader

loader = TextLoader("README.md")

text = loader.load()[0].page_content

text_splitter = RecursiveCharacterTextSplitter.from_language(
    chunk_size=400,
    chunk_overlap=0,
    language=Language.MARKDOWN
)

result = text_splitter.split_text(text)
print(len(result))
print(result[2])
