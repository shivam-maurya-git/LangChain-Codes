from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

load_dotenv()
model = ChatHuggingFace(llm = HuggingFaceEndpoint(
     repo_id="MiniMaxAI/MiniMax-M2.1",
                          task = "text-generation"))

prompt = PromptTemplate(
    template=  "List ten keywords from given text : - : {text}",
    input_variables=["text"]
)

loader = DirectoryLoader(
    path = "D:\Desktop\LangChainCodes\doc_loaders\\texts",
    # which files want to load
    glob='*.txt',
    loader_cls = TextLoader

)
# docs = loader.load()
# print(len(docs)) # 2
# In case of pdfs, length will be sum of all pages in all files, each page as one document class

parser = StrOutputParser()

doc_loader = RunnableLambda(lambda x : x.load())

data = RunnableLambda(lambda x :[doc.page_content  for doc in x])

chain = doc_loader | data | prompt | model | parser

result = chain.invoke(loader)

print(result)
