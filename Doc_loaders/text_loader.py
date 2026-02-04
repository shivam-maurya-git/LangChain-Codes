from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

load_dotenv()

# Creation of document object with given data (not loading)
loader = TextLoader('doc_loaders\poem.txt',encoding='utf-8')

# Loading data
docs = loader.load() # list type

# print(len(docs)) # Chunks of text
# print(docs[0].page_content) # actual data
# print(type(docs[0].page_content))

model = ChatHuggingFace(llm = HuggingFaceEndpoint(
     repo_id="MiniMaxAI/MiniMax-M2.1",
                          task = "text-generation"))

prompt = PromptTemplate(
    template=  "Write an summary of the following peom : {poem}",
    input_variables=["poem"]
)

parser = StrOutputParser()
# text = RunnableLambda(lambda x: x[0].page_content)#text is a Runnable, not actual data # wrong function also
docer = RunnableLambda(lambda x  : x.load())
text = RunnableLambda(lambda x: [doc.page_content for doc in x])

poem = docs[0].page_content

chain = docer|text | prompt | model | parser

# chain.invoke() expects real input values, not pipeline components

#RunnableLambda(lambda x: x[0].page_content) runs only when invoked # Also, function within Lmabda will give error : key error  0, and need to write function correctly

# result = chain.invoke({'poem' : text}) # it will give summary of runnables actually not the poem
# result = chain.invoke({'poem' : poem})

result = chain.invoke(loader)  
# Note we created loader runnable, data accessing runnable and 
# invoking using loader object
# We invoke chain using input of first runnable in chain

print(result)

