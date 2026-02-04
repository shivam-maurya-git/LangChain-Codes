from langchain_community.document_loaders import TextLoader, PyPDFLoader
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
    template=  "Write an summary of the following text in 5 lines : {text}",
    input_variables=["text"]
)
loader = PyPDFLoader("doc_loaders\Challenges_in_teaching_laws_of_exponents.pdf")

doc_loader = RunnableLambda(lambda x : x.load())

data = RunnableLambda(lambda x : [doc.page_content for doc in x])
parser = StrOutputParser()
chain = doc_loader | data | prompt | model | parser

result = chain.invoke(loader)
print(result)