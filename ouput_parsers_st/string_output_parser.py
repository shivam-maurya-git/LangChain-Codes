from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct",
                          task = "text-generation")

model = ChatHuggingFace(llm = llm)

template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template='Write 4 line summary on the following text. /n {text}',
    input_variables=['text']
)

# Large code
prompt1 = template1.invoke({'topic':'Exoplanets'})
# result = model.invoke(prompt1)

# prompt2 = template2.invoke({'text':result.content})
# result2 = model.invoke(prompt2)
# print(result2.content)
parser = StrOutputParser()

# Parser will collect only result.content and remove meta data
# Parser here make it easier to integrate in chain
# Q. Why we do need to invoke template here?
chain = template1 | model | parser | template2 | model | parser

results = chain.invoke({'topic':'Exoplanets'})
print(results)