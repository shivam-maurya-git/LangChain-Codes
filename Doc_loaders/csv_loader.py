from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

load_dotenv()
model = ChatHuggingFace(llm = HuggingFaceEndpoint(
     repo_id="MiniMaxAI/MiniMax-M2.1",
                          task = "text-generation"))


loader = CSVLoader("doc_loaders\sales_data.csv") # One url gives one document # List of urls = List of document object

prompt = PromptTemplate(
    template=  "Give an overview of the following data : - : {data}",
    input_variables=["data"]
)
# print(loader.load()) # Each row becomes one document object
# Can use lazy loading for large csv data

parser = StrOutputParser()

doc_loader = RunnableLambda(lambda x : x.load())

data = RunnableLambda(lambda x :[doc.page_content  for doc in x])

chain = doc_loader | data | prompt | model | parser

result = chain.invoke(loader)

print(result)


