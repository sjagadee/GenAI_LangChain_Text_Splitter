from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text = """
class A:
    def __init__(self):
        self.a = 1
        self.b = 2
        self.c = 3
        
    def get_a(self):
        return self.a

    def get_b(self):
        return self.b

    def get_c(self):
        return self.c
        
obj = A()
print(obj.get_a())
print(obj.get_b())
print(obj.get_c())
        
"""

text_splitter = RecursiveCharacterTextSplitter.from_language(
    chunk_size=310,
    chunk_overlap=0,
    language=Language.PYTHON
)

result = text_splitter.split_text(text)
print(len(result))
print(result[1])
