from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

load_dotenv()
model = ChatHuggingFace(llm = HuggingFaceEndpoint(
     repo_id="MiniMaxAI/MiniMax-M2.1",
                          task = "text-generation"))

url = "https://www.india.gov.in/"

loader = WebBaseLoader(url) # One url gives one document # List of urls = List of document object

prompt = PromptTemplate(
    template=  "Summarize the following text : - : {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

doc_loader = RunnableLambda(lambda x : x.load())

data = RunnableLambda(lambda x :[doc.page_content  for doc in x])

chain = doc_loader | data | prompt | model | parser

result = chain.invoke(loader)

print(result)


