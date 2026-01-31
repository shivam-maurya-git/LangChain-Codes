from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template= 'Write 5 facts about planet {planet_name}',
    input_variables=["planet_name"]
)

llm = HuggingFaceEndpoint(repo_id="DeepHat/DeepHat-V1-7B",
                          task = "text-generation")

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"planet_name":"Mars"})

# print(result)

chain.get_graph().print_ascii() # Viz chain #Return a graph representation of this Runnable.
# Needed grandalf to graph chain